#!/usr/bin/env python
"""运行一局完整游戏并通过 WebSocket 推送事件到可视化面板.

用法:
    python run_live_game.py [--rounds 3] [--delay 0.6] [--ws ws://localhost:8001/ws/game]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import websockets

from game.config import ConfigLoader
from game.debate_engine import DebateEngine
from game.types import CausalScenario

WS_DEFAULT = "ws://localhost:8001/ws/game"


# ── helpers ──────────────────────────────────────────────────────────

def _dag_to_graph(
    scenario: CausalScenario,
    discovered_hidden_vars: list[str] | None = None,
    causal_level: int = 1,
) -> dict[str, Any]:
    """CausalScenario.true_dag → 前端 CausalGraphData 格式.

    节点类型:
      - claimed (红): Agent A 声称的可见因果关系
      - hidden  (紫): 未被发现的隐藏变量
      - verified(绿): Agent B 发现的隐藏变量

    causal_level: 当前回合的 Pearl 因果层级 (1/2/3)，
                  会写入每个节点/边及图整体，供前端着色。
    """
    hidden_set = set(scenario.hidden_variables or [])
    discovered_set = set(discovered_hidden_vars or [])
    nodes: list[dict] = []
    seen: set[str] = set()

    def _node_type(v: str) -> str:
        if v in hidden_set:
            return "verified" if v in discovered_set else "hidden"
        return "claimed"

    for v in scenario.variables or []:
        nodes.append({"id": v, "label": v, "type": _node_type(v), "causal_level": causal_level})
        seen.add(v)

    for src, targets in (scenario.true_dag or {}).items():
        if src not in seen:
            nodes.append({"id": src, "label": src, "type": _node_type(src), "causal_level": causal_level})
            seen.add(src)
        for t in targets:
            if t not in seen:
                nodes.append({"id": t, "label": t, "type": _node_type(t), "causal_level": causal_level})
                seen.add(t)

    # ── 安全网: 确保所有隐藏变量都作为节点出现 ──
    # 噪声变量可能不在 variables 也不在 true_dag 中，
    # 但它们属于 hidden_variables，必须在图中显示为紫色(hidden)或绿色(verified)
    for hv in scenario.hidden_variables or []:
        if hv not in seen:
            nodes.append({"id": hv, "label": hv, "type": _node_type(hv), "causal_level": causal_level})
            seen.add(hv)

    links: list[dict] = []
    for src, targets in (scenario.true_dag or {}).items():
        for t in targets:
            src_hidden = src in hidden_set
            t_hidden = t in hidden_set
            if src_hidden or t_hidden:
                # 涉及隐藏变量的边: 全部已发现→verified, 否则→hidden
                all_discovered = (
                    (not src_hidden or src in discovered_set)
                    and (not t_hidden or t in discovered_set)
                )
                link_type = "verified" if all_discovered else "hidden"
            else:
                link_type = "claimed"
            links.append({"source": src, "target": t, "type": link_type, "causal_level": causal_level})

    return {"nodes": nodes, "links": links, "causal_level": causal_level}


def _strategy_diversity(dist: dict[str, float]) -> float:
    """归一化 Shannon 熵衡量策略多样性.

    用全部 11 种 StrategyType 的最大熵做归一化，
    这样只用了 2/11 种策略时不会虚高到 1.0。
    """
    _TOTAL_STRATEGY_TYPES = 11  # StrategyType 枚举共 11 种
    vals = [v for v in dist.values() if v > 0]
    if not vals or len(vals) == 1:
        return 0.0
    entropy = -sum(v * math.log2(v) for v in vals)
    max_ent = math.log2(_TOTAL_STRATEGY_TYPES)
    return round(entropy / max_ent, 4) if max_ent > 0 else 0.0


def _arms_race(deception: float, detection: float) -> float:
    """军备竞赛指数: 同时考虑攻防均衡度和强度.

    balance: 攻防差距越小越接近 1
    intensity: 攻防平均水平越高越接近 1
    两者相乘 → 只有双方都强且均衡时才接近 1。
    """
    balance = 1.0 - abs(deception - detection)
    intensity = (deception + detection) / 2.0
    return round(balance * intensity, 4)


def _describe_runtime_mode(engine: DebateEngine) -> str:
    """Summarize whether the current run is using real LLM APIs or mock fallbacks."""
    labels: list[str] = []
    for agent_name in ("agent_a", "agent_b", "agent_c"):
        agent = getattr(engine, agent_name, None)
        service = getattr(agent, "llm_service", None)
        if service is None:
            labels.append(f"{agent_name}=mock")
            continue

        backend = getattr(service, "backend", "mock")
        client = getattr(service, "_client", None)
        if backend in {"dashscope", "api"} and client is not None:
            labels.append(f"{agent_name}=dashscope")
        elif backend in {"vllm", "ollama"}:
            labels.append(f"{agent_name}={backend}(fallback)")
        else:
            labels.append(f"{agent_name}=mock")

    unique_labels = {label.split("=", 1)[1] for label in labels}
    if unique_labels == {"mock"}:
        prefix = "mock 模式"
    elif len(unique_labels) == 1:
        prefix = f"{next(iter(unique_labels))} 模式"
    else:
        prefix = "mixed 模式"
    return f"{prefix}; " + ", ".join(labels)


# ── WebSocket 发送 ───────────────────────────────────────────────────

async def _send(ws, event_type: str, round_id: int, data: dict, delay: float):
    payload = json.dumps({
        "event_type": event_type,
        "round_id": round_id,
        "data": data,
        "timestamp": time.time(),
    })
    await ws.send(payload)
    try:
        ack = await asyncio.wait_for(ws.recv(), timeout=3)
        print(f"  [{event_type:12s}] round={round_id}  ack={ack[:60]}")
    except asyncio.TimeoutError:
        print(f"  [{event_type:12s}] round={round_id}  (no ack)")
    await asyncio.sleep(delay)


# ── 主流程 ───────────────────────────────────────────────────────────

async def run(rounds: int, delay: float, ws_url: str):
    cfg = ConfigLoader().load()
    engine = DebateEngine(cfg)
    await engine.initialize()

    print(f"🎮 运行 {rounds} 轮游戏 ({_describe_runtime_mode(engine)})...")
    results: list[dict[str, Any]] = await engine.run_game(num_rounds=rounds)
    print(f"✅ 引擎完成, 共 {len(results)} 轮结果\n")

    print(f"🔗 连接 WebSocket: {ws_url}")
    async with websockets.connect(ws_url) as ws:
        # ── game start ──
        await _send(ws, "system", 0, {
            "role": "system",
            "narrative": "🎮 因果叛徒游戏开始！",
            "game_id": "live",
        }, delay)

        for result in results:
            rn: int = result["round_number"]
            scenario: CausalScenario = result["scenario"]

            # ── round_start + causal graph ──
            await _send(ws, "round_start", rn, {
                "role": "system",
                "narrative": f"📋 第 {rn} 轮 — 因果层级 L{scenario.causal_level}, 难度 {result.get('difficulty', 0.5):.2f}",
                "causal_graph": _dag_to_graph(scenario, result.get("agent_b_analysis", {}).get("discovered_hidden_vars", []), causal_level=scenario.causal_level),
                "causal_level": scenario.causal_level,
                "difficulty": result.get("difficulty", 0.5),
                "game_id": "live",
            }, delay)

            # ── claim (Agent A) ──
            claim = result.get("agent_a_claim", {})
            await _send(ws, "claim", rn, {
                "role": "traitor",
                "claim": claim.get("causal_claim", ""),
                "narrative": claim.get("narrative", ""),
                "strategy": claim.get("strategy", ""),
                "game_id": "live",
            }, delay)

            # ── detection (Agent B) ──
            det = result.get("agent_b_analysis", {})
            # reasoning_chain 是 list[str]，拼接为可读文本
            chain = det.get("reasoning_chain", [])
            reasoning_text = "\n".join(chain) if isinstance(chain, list) else str(chain)
            await _send(ws, "detection", rn, {
                "role": "scientist",
                "narrative": f"🔬 检测分析 (置信度 {det.get('confidence', 0):.0%})\n{reasoning_text}",
                "reasoning": reasoning_text,
                "tool_results": det.get("tools_used", []),
                "confidence": det.get("confidence", 0),
                "detected_fallacies": det.get("detected_fallacies", []),
                "discovered_hidden_vars": det.get("discovered_hidden_vars", []),
                "game_id": "live",
            }, delay)

            # ── rebuttal (Agent A) ──
            reb = result.get("agent_a_rebuttal", {})
            await _send(ws, "claim", rn, {
                "role": "traitor",
                "claim": reb.get("causal_claim", ""),
                "narrative": reb.get("narrative", "（反驳阶段）"),
                "strategy": reb.get("strategy", ""),
                "game_id": "live",
            }, delay)

            # ── jury ──
            jury = result.get("jury_verdict", {})
            # 转换 JuryVote 字段名: model_name→model, winner→vote, 添加 juror_id
            raw_votes = jury.get("votes", [])
            mapped_votes = []
            for idx, v in enumerate(raw_votes):
                mapped_votes.append({
                    "juror_id": f"juror_{idx + 1}",
                    "model": v.get("model_name", "unknown"),
                    "vote": v.get("winner", ""),
                    "confidence": v.get("confidence", 0),
                    "reasoning": v.get("reasoning", ""),
                })
            consensus = jury.get("agreement_rate", 0)
            verdict = jury.get("final_winner", "")
            await _send(ws, "jury", rn, {
                "role": "jury",
                "narrative": f"🗳️ 陪审团裁决: {verdict} (共识度 {consensus:.0%})",
                "votes": mapped_votes,
                "consensus": consensus,
                "verdict": verdict,
                "game_id": "live",
            }, delay)

            # ── audit verdict (Agent C) ──
            audit = result.get("audit_verdict", {})
            await _send(ws, "verdict", rn, {
                "role": "auditor",
                "verdict": audit.get("winner", ""),
                "reasoning": audit.get("reasoning", ""),
                "narrative": audit.get("narrative", ""),
                "causal_validity_score": audit.get("causal_validity_score", 0),
                "game_id": "live",
            }, delay)

            # ── evolution / difficulty data ──
            evo = result.get("evolution_snapshot") or {}
            dist = evo.get("strategy_distribution", {})
            dec_rate = evo.get("avg_deception_rate", 0.0)
            det_rate = evo.get("avg_detection_rate", 0.0)
            diff = evo.get("difficulty_level", result.get("difficulty", 0.5))

            await _send(ws, "system", rn, {
                "role": "system",
                "narrative": f"📊 第 {rn} 轮结束 — 胜者: {result.get('winner', '?')}",
                "round_id": rn,
                "difficulty": diff,
                "dsr": dec_rate,
                "strategy_diversity": _strategy_diversity(dist),
                "arms_race_index": _arms_race(dec_rate, det_rate),
                "game_id": "live",
            }, delay)

        # ── game end ──
        await _send(ws, "game_end", 0, {
            "role": "system",
            "narrative": f"🏁 游戏结束！共 {len(results)} 轮",
            "total_rounds": len(results),
            "game_id": "live",
        }, delay)

    print("\n🏁 所有事件已推送到可视化面板。")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="运行因果叛徒游戏并推送到可视化面板")
    ap.add_argument("--rounds", type=int, default=3, help="游戏轮数 (默认 3)")
    ap.add_argument("--delay", type=float, default=0.6, help="事件间隔秒数 (默认 0.6)")
    ap.add_argument("--ws", default=WS_DEFAULT, help="WebSocket URL")
    args = ap.parse_args()
    asyncio.run(run(args.rounds, args.delay, args.ws))


if __name__ == "__main__":
    main()
