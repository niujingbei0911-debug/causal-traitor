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
import sys
import time
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Windows cmd.exe defaults to GBK — force UTF-8 for emoji output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import websockets

from benchmark.schema import PublicCausalInstance, ensure_public_instance
from game.config import ConfigLoader
from game.debate_engine import DebateEngine
from game.evolution import StrategyRecord
from verifier.claim_parser import parse_claim

WS_DEFAULT = "ws://localhost:8001/ws/game"


# ── helpers ──────────────────────────────────────────────────────────

def _infer_focus_variables(scenario: PublicCausalInstance, claim_text: str) -> tuple[str, str]:
    variables = list(getattr(scenario, "variables", []) or [])
    if not variables:
        return "X", "Y"
    try:
        parsed = parse_claim(claim_text)
    except Exception:
        parsed = None
    if parsed is not None:
        treatment = str(getattr(parsed, "treatment", "") or "").strip()
        outcome = str(getattr(parsed, "outcome", "") or "").strip()
        if treatment in variables and outcome in variables and treatment != outcome:
            return treatment, outcome
    if "X" in variables and "Y" in variables:
        return "X", "Y"
    return variables[0], variables[-1]


def _public_graph(
    scenario: PublicCausalInstance,
    *,
    claim_text: str,
    causal_level: int,
) -> dict[str, Any]:
    """Build a public-evidence graph without leaking gold DAG structure."""

    proxy_variables = set(getattr(scenario, "proxy_variables", []) or [])
    selection_variables = set(getattr(scenario, "selection_variables", []) or [])
    treatment, outcome = _infer_focus_variables(scenario, claim_text)

    nodes: list[dict[str, Any]] = []
    for variable in list(getattr(scenario, "variables", []) or []):
        node_type = "claimed"
        if variable in proxy_variables or variable in selection_variables:
            node_type = "verified"
        nodes.append(
            {
                "id": variable,
                "label": variable,
                "type": node_type,
                "causal_level": causal_level,
            }
        )

    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()

    def add_link(source: str, target: str, link_type: str) -> None:
        if source == target:
            return
        key = (source, target, link_type)
        if key in seen_links:
            return
        seen_links.add(key)
        links.append(
            {
                "source": source,
                "target": target,
                "type": link_type,
                "causal_level": causal_level,
            }
        )

    if treatment in getattr(scenario, "variables", []) and outcome in getattr(scenario, "variables", []):
        add_link(treatment, outcome, "claimed")

    for proxy in sorted(proxy_variables):
        if proxy in getattr(scenario, "variables", []):
            add_link(proxy, treatment, "verified")
            add_link(proxy, outcome, "verified")

    for selection in sorted(selection_variables):
        if selection in getattr(scenario, "variables", []):
            add_link(selection, outcome, "verified")

    return {
        "nodes": nodes,
        "links": links,
        "causal_level": causal_level,
        "schema_view": "public",
    }


def _strategy_diversity(dist: dict[str, float]) -> float:
    """归一化 Shannon 熵衡量策略多样性."""
    _TOTAL_STRATEGY_TYPES = 11
    vals = [v for v in dist.values() if v > 0]
    if not vals or len(vals) == 1:
        return 0.0
    entropy = -sum(v * math.log2(v) for v in vals)
    max_ent = math.log2(_TOTAL_STRATEGY_TYPES)
    return round(entropy / max_ent, 4) if max_ent > 0 else 0.0


def _arms_race(deception: float, detection: float) -> float:
    """军备竞赛指数: balance * intensity."""
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


def _round_backend_tags(engine: DebateEngine) -> str:
    """Return per-agent backend tags like [A:LLM✓ B:LLM✓ C:MOCK]."""
    tags: list[str] = []
    name_map = {"agent_a": "A", "agent_b": "B", "agent_c": "C"}
    for agent_name, short in name_map.items():
        agent = getattr(engine, agent_name, None)
        service = getattr(agent, "llm_service", None)
        if service is None:
            tags.append(f"{short}:MOCK")
            continue
        backend = getattr(service, "backend", "mock")
        client = getattr(service, "_client", None)
        if backend in {"dashscope", "api"} and client is not None:
            tags.append(f"{short}:LLM✓")
        else:
            tags.append(f"{short}:MOCK")
    return "[" + " ".join(tags) + "]"


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


# ── 单轮后处理 (复制自 DebateEngine.run_game 的逐轮逻辑) ─────────────

def _postprocess_round(
    engine: DebateEngine,
    result: dict[str, Any],
    round_index: int,
    scenario,
    *,
    update_difficulty: bool = True,
) -> dict[str, Any]:
    """Replicate per-round post-processing from engine.run_game().

    Handles: difficulty update, continuous scores, evolution tracking.
    Returns the enriched result dict.
    """
    # ── difficulty update ──
    if update_difficulty:
        next_difficulty = engine.difficulty_controller.update(result["deception_success"])
    else:
        next_difficulty = engine.difficulty_controller.get_difficulty()

    # ── continuous scores ──
    jury_v = result["jury_verdict"]
    a_conf = sum(v["confidence"] for v in jury_v["votes"] if v["winner"] == "agent_a")
    b_conf = sum(v["confidence"] for v in jury_v["votes"] if v["winner"] == "agent_b")
    total_conf = sum(v["confidence"] for v in jury_v["votes"]) or 1.0
    jury_a_share = a_conf / total_conf
    jury_b_share = b_conf / total_conf

    audit_v = result["audit_verdict"]
    audit_a = float(audit_v.get("argument_quality_a", 0.0))
    audit_b = float(audit_v.get("argument_quality_b", 0.0))
    audit_total = max(audit_a + audit_b, 1e-6)
    audit_a_share = audit_a / audit_total
    audit_b_share = audit_b / audit_total

    detection_conf = float(result["agent_b_analysis"].get("confidence", 0.0))

    continuous_deception = round(max(0.0, min(1.0, 0.7 * audit_a_share + 0.3 * jury_a_share)), 4)
    continuous_detection = round(max(0.0, min(1.0, 0.45 * audit_b_share + 0.35 * detection_conf + 0.20 * jury_b_share)), 4)

    # ── evolution tracking ──
    record = StrategyRecord(
        round_id=round_index,
        strategy_type=engine._strategy_type_from_result(result),
        success=result["deception_success"],
        deception_score=continuous_deception,
        detection_score=continuous_detection,
        details={
            "difficulty": scenario.difficulty,
            "next_difficulty": next_difficulty,
            "winner": result["winner"],
            "scenario_id": scenario.scenario_id,
        },
    )
    engine.evolution_tracker.record_round(record)
    result["evolution_snapshot"] = asdict(engine.evolution_tracker.take_snapshot(round_index))
    result["next_difficulty"] = next_difficulty
    engine.round_results.append(result)

    return result


# ── 推送单轮 WS 事件 ─────────────────────────────────────────────────

async def _push_round_events(ws, result: dict[str, Any], delay: float):
    """Push all WS events for a single completed round."""
    rn: int = result["round_number"]
    public_scenario = ensure_public_instance(result.get("public_scenario") or result["scenario"])
    agent_a_claim = result.get("agent_a_claim", {})
    public_graph = _public_graph(
        public_scenario,
        claim_text=str(agent_a_claim.get("causal_claim", "")),
        causal_level=public_scenario.causal_level,
    )

    # ── round_start + causal graph ──
    await _send(ws, "round_start", rn, {
        "role": "system",
        "narrative": (
            f"📋 第 {rn} 轮 — Public oversight view | "
            f"因果层级 L{public_scenario.causal_level}, 难度 {result.get('difficulty', 0.5):.2f}"
        ),
        "causal_graph": public_graph,
        "causal_level": public_scenario.causal_level,
        "difficulty": result.get("difficulty", 0.5),
        "scenario_id": public_scenario.scenario_id,
        "variables": list(public_scenario.variables),
        "proxy_variables": list(public_scenario.proxy_variables),
        "selection_variables": list(public_scenario.selection_variables),
        "selection_mechanism": public_scenario.selection_mechanism,
        "game_id": "live",
    }, delay)

    # ── claim (Agent A) ──
    claim = agent_a_claim
    await _send(ws, "claim", rn, {
        "role": "traitor",
        "claim": claim.get("causal_claim", ""),
        "narrative": claim.get("content", claim.get("narrative", "")),
        "strategy": claim.get("deception_strategy", claim.get("strategy", "")),
        "game_id": "live",
    }, delay)

    # ── detection (Agent B) ──
    det = result.get("agent_b_analysis", {})
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
        "narrative": reb.get("content", reb.get("narrative", "（反驳阶段）")),
        "strategy": reb.get("deception_strategy", reb.get("strategy", "")),
        "game_id": "live",
    }, delay)

    # ── jury ──
    jury = result.get("jury_verdict", {})
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
        "verdict_label": audit.get("verdict_label"),
        "verifier_confidence": audit.get("verifier_confidence", 0.0),
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


# ── 逐轮运行核心逻辑 ─────────────────────────────────────────────────

async def _run_rounds(
    engine: DebateEngine,
    num_rounds: int,
    *,
    on_round_done=None,
) -> list[dict[str, Any]]:
    """Run rounds one-by-one, replicating DebateEngine.run_game() logic.

    Calls optional *on_round_done(result)* callback after each round
    so WS events can be pushed immediately.
    """
    levels = engine.config.get("game", {}).get("causal_levels", [1, 2, 3])
    engine.round_results = []

    for round_index in range(1, num_rounds + 1):
        level = levels[(round_index - 1) % len(levels)]
        difficulty = engine.difficulty_controller.get_difficulty()

        print(f"\n{'='*60}")
        print(f"🔄 第 {round_index}/{num_rounds} 轮 | L{level} | 难度 {difficulty:.2f} | {_round_backend_tags(engine)}")
        print(f"{'='*60}")

        scenario = engine.data_generator.generate_scenario(difficulty=difficulty, causal_level=level)
        scenario.difficulty_config = {**scenario.difficulty_config, **engine.difficulty_controller.get_config()}

        evolution_context = engine._build_evolution_context()
        result = await engine.run_round(scenario, round_number=round_index, evolution_context=evolution_context)

        print(f"  ✅ 胜者: {result.get('winner', '?')} {_round_backend_tags(engine)}")

        # post-processing: difficulty, continuous scores, evolution
        _postprocess_round(engine, result, round_index, scenario)

        # callback for WS push
        if on_round_done is not None:
            await on_round_done(result)

    return engine.round_results


# ── 主流程 ───────────────────────────────────────────────────────────

async def run(rounds: int, delay: float, ws_url: str, *, no_ws: bool = False):
    cfg = ConfigLoader().load()
    engine = DebateEngine(cfg)
    await engine.initialize()

    print(f"🎮 运行 {rounds} 轮游戏 ({_describe_runtime_mode(engine)})...")

    # --no-ws 模式: 逐轮运行, 打印摘要, 不连接 WebSocket
    if no_ws:
        results = await _run_rounds(engine, rounds)
        print(f"\n✅ 引擎完成, 共 {len(results)} 轮结果\n")
        for r in results:
            rn = r["round_number"]
            w = r.get("winner", "?")
            d = r.get("difficulty", 0.5)
            print(f"  第 {rn} 轮 — 胜者: {w}, 难度: {d:.2f}")
        print(f"\n🏁 游戏结束！共 {len(results)} 轮 (离线模式, 未推送 WebSocket)")
        return results

    # WS 模式: 先连接 WebSocket, 再逐轮运行, 每轮完成后立即推送
    print(f"🔗 连接 WebSocket: {ws_url}")
    async with websockets.connect(ws_url) as ws:
        # ── game start ──
        await _send(ws, "system", 0, {
            "role": "system",
            "narrative": "🎮 因果叛徒游戏开始！",
            "game_id": "live",
        }, delay)

        async def _on_round_done(result: dict[str, Any]):
            await _push_round_events(ws, result, delay)

        results = await _run_rounds(engine, rounds, on_round_done=_on_round_done)

        # ── game end ──
        await _send(ws, "game_end", 0, {
            "role": "system",
            "narrative": f"🏁 游戏结束！共 {len(results)} 轮",
            "total_rounds": len(results),
            "game_id": "live",
        }, delay)

    print(f"\n✅ 引擎完成, 共 {len(results)} 轮结果")
    print("🏁 所有事件已实时推送到可视化面板。")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="运行因果叛徒游戏并推送到可视化面板")
    ap.add_argument("--rounds", type=int, default=3, help="游戏轮数 (默认 3)")
    ap.add_argument("--delay", type=float, default=0.6, help="事件间隔秒数 (默认 0.6)")
    ap.add_argument("--ws", default=WS_DEFAULT, help="WebSocket URL")
    ap.add_argument("--no-ws", action="store_true", help="仅运行游戏, 不连接 WebSocket")
    args = ap.parse_args()
    asyncio.run(run(args.rounds, args.delay, args.ws, no_ws=args.no_ws))


if __name__ == "__main__":
    main()
