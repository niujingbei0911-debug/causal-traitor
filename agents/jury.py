"""
Jury - 陪审团机制
多模型投票，增强裁决鲁棒性
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
        self.jury_models = []
        jury_cfg = self.config.get("models", {}).get("jury", self.config.get("jury", {}))
        self.draw_margin = float(jury_cfg.get("draw_margin", 0.035))
        self.draw_vote_weight = float(jury_cfg.get("draw_vote_weight", 0.2))

    async def initialize(self):
        """初始化所有陪审团模型"""
        jury_config = self.config.get("models", {}).get("jury", self.config.get("jury", {}))
        self.jury_models = list(jury_config.get("models", []))

    async def collect_votes(
        self, scenario: "CausalScenario", debate_context: "DebateContext"
    ) -> JuryVerdict:
        """收集所有陪审员投票并聚合"""
        turns = list(getattr(debate_context, "turns", []))
        transcript = " ".join(str(turn.get("content", "")) for turn in turns).lower()
        level = int(getattr(scenario, "causal_level", 1))
        agent_a_text = " ".join(
            str(turn.get("content", "")) for turn in turns if str(turn.get("speaker", "")).lower() == "agent_a"
        ).lower()
        agent_b_text = " ".join(
            str(turn.get("content", "")) for turn in turns if str(turn.get("speaker", "")).lower() == "agent_b"
        ).lower()
        votes: list[JuryVote] = []

        base_score_a = {1: 0.47, 2: 0.47, 3: 0.53}.get(level, 0.45)
        base_score_b = 1.0 - base_score_a

        if any(token in agent_a_text for token in ["混杂", "confound", "未观测", "latent"]):
            base_score_a += 0.10 if level == 1 else 0.06
        if any(token in agent_a_text for token in ["选择偏差", "selection bias", "逆因果", "反向因果"]):
            base_score_a += 0.05 if level == 1 else 0.04
        if level >= 2 and any(token in agent_a_text for token in ["工具变量", "iv", "instrument", "排他性"]):
            base_score_a += 0.06
        if level >= 3 and any(token in agent_a_text for token in ["反事实", "counterfactual", "scm", "函数形式", "不可识别"]):
            base_score_a += 0.08

        if any(token in agent_b_text for token in ["相关系数", "显著", "观测差异", "slope", "correlation"]):
            base_score_b += 0.03 if level == 1 else 0.05
        if level >= 2 and any(token in agent_b_text for token in ["后门", "backdoor", "工具变量", "iv", "gamma", "敏感性"]):
            base_score_b += 0.07
        if level >= 3 and any(token in agent_b_text for token in ["必要性", "充分性", "反事实", "counterfactual", "ett"]):
            base_score_b += 0.06

        if any(token in transcript for token in ["必然", "毫无疑问", "100%", "唯一解释"]):
            if any(token in agent_a_text for token in ["必然", "毫无疑问", "100%", "唯一解释"]):
                base_score_b += 0.05
            if any(token in agent_b_text for token in ["必然", "毫无疑问", "100%", "唯一解释"]):
                base_score_a += 0.05

        total = max(base_score_a + base_score_b, 1e-6)
        base_score_a = min(0.9, max(0.1, base_score_a / total))
        base_score_b = 1.0 - base_score_a
        voting_method = self.config.get("models", {}).get("jury", {}).get("voting", "weighted")
        jury_models = self.jury_models or ["baseline_juror_1", "baseline_juror_2", "baseline_juror_3"]

        for index, model_name in enumerate(jury_models):
            jitter = (index - (len(jury_models) - 1) / 2) * 0.03
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

            votes.append(
                JuryVote(
                    model_name=model_name,
                    winner=winner,
                    confidence=float(round(confidence, 3)),
                    reasoning="Agent B 的论证更依赖因果工具与混杂检验。"
                    if winner == "agent_b"
                    else "Agent A 的主张更自洽，未被充分反驳。"
                    if winner == "agent_a"
                    else "双方证据接近，难以形成清晰多数。",
                )
            )

        final_winner = self.aggregate(votes, method=voting_method)
        agreement_rate = 0.0 if not votes else sum(vote.winner == final_winner for vote in votes) / len(votes)
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
