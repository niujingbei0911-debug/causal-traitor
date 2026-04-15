"""
Jury - 陪审团机制
多模型投票，增强裁决鲁棒性
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from game.llm_service import LLMService
from agents.prompts.jury_prompts import JURY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _model_weight(model_name: str) -> float:
    lowered = model_name.lower()
    if "72b" in lowered:
        return 1.6
    if "14b" in lowered:
        return 1.3
    if "7b" in lowered:
        return 1.0
    return 1.0


@dataclass
class JuryVote:
    """单个陪审员投票"""
    model_name: str
    winner: str  # "agent_a" | "agent_b" | "draw"
    confidence: float
    reasoning: str


@dataclass
class JuryVerdict:
    """陪审团汇总裁决"""
    votes: list[JuryVote] = field(default_factory=list)
    final_winner: str = ""
    agreement_rate: float = 0.0
    aggregation_method: str = "weighted"


class JuryAggregator:
    """
    陪审团聚合器

    支持投票方式：
    - majority: 简单多数
    - weighted: 加权投票（按模型能力）
    - bayesian: 贝叶斯聚合
    """

    def __init__(self, config: dict):
        self.config = config
        self.jury_models: list[str] = []
        self._llm_services: list[LLMService] = []
        jury_cfg = self.config.get("models", {}).get("jury", self.config.get("jury", {}))
        self.draw_margin = float(jury_cfg.get("draw_margin", 0.035))
        self.draw_vote_weight = float(jury_cfg.get("draw_vote_weight", 0.2))

    async def initialize(self):
        """初始化所有陪审团模型及其 LLM 服务"""
        jury_cfg = self.config.get("models", {}).get("jury", self.config.get("jury", {}))
        self.jury_models = list(jury_cfg.get("models", []))

        backend = jury_cfg.get("backend", "mock")
        temperature = jury_cfg.get("temperature", 0.2)
        max_tokens = jury_cfg.get("max_tokens", 2048)

        self._llm_services = []
        for model_name in self.jury_models:
            svc = LLMService({
                "name": model_name,
                "backend": backend,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
            await svc.initialize()
            self._llm_services.append(svc)

    # ------------------------------------------------------------------
    # transcript builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transcript(turns: list) -> str:
        """将辩论 turns 拼接为可读文本。"""
        lines: list[str] = []
        for turn in turns:
            speaker = str(turn.get("speaker", "unknown"))
            content = str(turn.get("content", ""))
            lines.append(f"[{speaker}] {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM-based voting
    # ------------------------------------------------------------------

    async def _llm_vote(
        self, svc: LLMService, model_name: str, scenario: Any, transcript: str
    ) -> JuryVote | None:
        """调用单个 LLM 陪审员，返回 JuryVote 或 None（失败时）。"""
        level = int(getattr(scenario, "causal_level", 1))
        user_prompt = (
            f"因果层级: L{level}\n\n"
            f"辩论记录:\n{transcript}\n\n"
            "请根据上述辩论内容，独立评估谁的因果论证更可靠，输出 JSON。"
        )
        try:
            _resp, parsed = await svc.generate_json(
                user_prompt, system_prompt=JURY_SYSTEM_PROMPT
            )
        except Exception as exc:
            logger.warning("Jury LLM call failed for %s: %s", model_name, exc)
            return None

        if parsed is None:
            logger.warning("Jury LLM returned unparseable JSON for %s", model_name)
            return None

        vote_raw = str(parsed.get("vote", "")).lower().strip()
        if vote_raw not in {"agent_a", "agent_b", "draw"}:
            logger.warning("Jury LLM returned invalid vote '%s' for %s", vote_raw, model_name)
            return None

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))

        reasoning = str(parsed.get("reasoning", ""))
        return JuryVote(
            model_name=model_name,
            winner=vote_raw,
            confidence=round(confidence, 3),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # heuristic fallback (original keyword-based logic)
    # ------------------------------------------------------------------

    def _heuristic_vote(
        self,
        index: int,
        model_name: str,
        scenario: Any,
        turns: list,
    ) -> JuryVote:
        """基于关键词的启发式投票（LLM 不可用时的回退）。"""
        transcript = " ".join(str(t.get("content", "")) for t in turns).lower()
        level = int(getattr(scenario, "causal_level", 1))
        agent_a_text = " ".join(
            str(t.get("content", "")) for t in turns if str(t.get("speaker", "")).lower() == "agent_a"
        ).lower()
        agent_b_text = " ".join(
            str(t.get("content", "")) for t in turns if str(t.get("speaker", "")).lower() == "agent_b"
        ).lower()

        base_score_a = {1: 0.47, 2: 0.47, 3: 0.53}.get(level, 0.45)
        base_score_b = 1.0 - base_score_a

        if any(tok in agent_a_text for tok in ["混杂", "confound", "未观测", "latent"]):
            base_score_a += 0.10 if level == 1 else 0.06
        if any(tok in agent_a_text for tok in ["选择偏差", "selection bias", "逆因果", "反向因果"]):
            base_score_a += 0.05 if level == 1 else 0.04
        if level >= 2 and any(tok in agent_a_text for tok in ["工具变量", "iv", "instrument", "排他性"]):
            base_score_a += 0.06
        if level >= 3 and any(tok in agent_a_text for tok in ["反事实", "counterfactual", "scm", "函数形式", "不可识别"]):
            base_score_a += 0.08

        if any(tok in agent_b_text for tok in ["相关系数", "显著", "观测差异", "slope", "correlation"]):
            base_score_b += 0.03 if level == 1 else 0.05
        if level >= 2 and any(tok in agent_b_text for tok in ["后门", "backdoor", "工具变量", "iv", "gamma", "敏感性"]):
            base_score_b += 0.07
        if level >= 3 and any(tok in agent_b_text for tok in ["必要性", "充分性", "反事实", "counterfactual", "ett"]):
            base_score_b += 0.06

        if any(tok in transcript for tok in ["必然", "毫无疑问", "100%", "唯一解释"]):
            if any(tok in agent_a_text for tok in ["必然", "毫无疑问", "100%", "唯一解释"]):
                base_score_b += 0.05
            if any(tok in agent_b_text for tok in ["必然", "毫无疑问", "100%", "唯一解释"]):
                base_score_a += 0.05

        total = max(base_score_a + base_score_b, 1e-6)
        base_score_a = min(0.9, max(0.1, base_score_a / total))
        base_score_b = 1.0 - base_score_a

        num_models = max(len(self.jury_models), 1)
        jitter = (index - (num_models - 1) / 2) * 0.03
        adjusted_a = min(0.95, max(0.05, base_score_a - jitter))
        adjusted_b = 1.0 - adjusted_a
        gap = abs(adjusted_b - adjusted_a)
        if gap < self.draw_margin and max(adjusted_a, adjusted_b) < 0.62:
            winner = "draw"
            confidence = 0.5 - min(0.15, gap)
        elif adjusted_b > adjusted_a:
            winner = "agent_b"
            confidence = adjusted_b
        else:
            winner = "agent_a"
            confidence = adjusted_a

        return JuryVote(
            model_name=model_name,
            winner=winner,
            confidence=float(round(confidence, 3)),
            reasoning="Agent B 的论证更依赖因果工具与混杂检验。"
            if winner == "agent_b"
            else "Agent A 的主张更自洽，未被充分反驳。"
            if winner == "agent_a"
            else "双方证据接近，难以形成清晰多数。",
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def collect_votes(
        self, scenario: "CausalScenario", debate_context: "DebateContext"
    ) -> JuryVerdict:
        """收集所有陪审员投票并聚合"""
        turns = list(getattr(debate_context, "turns", []))
        voting_method = self.config.get("models", {}).get("jury", {}).get("voting", "weighted")
        jury_models = self.jury_models or ["baseline_juror_1", "baseline_juror_2", "baseline_juror_3"]

        votes: list[JuryVote] = []
        transcript = self._build_transcript(turns)

        for index, model_name in enumerate(jury_models):
            svc = self._llm_services[index] if index < len(self._llm_services) else None

            # 尝试 LLM 投票
            if svc is not None and getattr(svc, "backend", "mock") != "mock":
                llm_vote = await self._llm_vote(svc, model_name, scenario, transcript)
                if llm_vote is not None:
                    votes.append(llm_vote)
                    continue

            # 回退到启发式
            votes.append(self._heuristic_vote(index, model_name, scenario, turns))

        final_winner = self.aggregate(votes, method=voting_method)
        agreement_rate = 0.0 if not votes else sum(v.winner == final_winner for v in votes) / len(votes)
        return JuryVerdict(
            votes=votes,
            final_winner=final_winner,
            agreement_rate=float(agreement_rate),
            aggregation_method=voting_method,
        )

    def aggregate(self, votes: list[JuryVote], method: str = "weighted") -> str:
        """聚合投票结果"""
        if not votes:
            return "draw"

        scores = {"agent_a": 0.0, "agent_b": 0.0, "draw": 0.0}
        for vote in votes:
            weight = 1.0
            if method in {"weighted", "bayesian"}:
                weight = _model_weight(vote.model_name)
            confidence = vote.confidence if method != "majority" else 1.0
            if vote.winner == "draw":
                scores["draw"] += weight * confidence * self.draw_vote_weight
                continue
            scores[vote.winner] += weight * confidence

        top_winner = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        if top_winner == "draw":
            non_draw = {key: value for key, value in scores.items() if key != "draw"}
            return max(non_draw.items(), key=lambda item: item[1])[0]
        if len(ordered) >= 2 and abs(ordered[0] - ordered[1]) < self.draw_margin:
            return "draw"
        return top_winner
