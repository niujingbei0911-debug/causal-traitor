"""
Agent C - 审计员（Auditor）
负责评估因果论证质量，裁决胜负
模型：Qwen2.5-72B-Instruct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.jury import JuryAggregator, JuryVerdict
from agents.tool_executor import ToolExecutor
from causal_tools.meta_tools import argument_logic_check
from game.llm_service import LLMService


@dataclass
class AuditVerdict:
    """审计裁决结果"""
    winner: str  # "agent_a" | "agent_b" | "draw"
    causal_validity_score: float  # 0-1
    argument_quality_a: float
    argument_quality_b: float
    reasoning: str
    identified_issues: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    jury_consensus: float = 0.0


class AgentC:
    """
    审计员Agent - 因果论证裁决者
    
    评估维度：
    - 因果有效性（是否符合Pearl框架）
    - 论证质量（逻辑严密性）
    - 工具使用恰当性
    - 证据充分性
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.llm_service: LLMService | None = None
        self.tool_executor = ToolExecutor(config)
        self.jury = JuryAggregator(config)

    async def initialize(self):
        """初始化模型连接"""
        self.model = self.config.get("models", {}).get("agent_c", self.config.get("agent_c", {}))
        if self.llm_service is None:
            self.llm_service = LLMService(self.model if isinstance(self.model, dict) else {})
        await self.llm_service.initialize()
        await self.jury.initialize()

    def attach_llm_service(self, service: LLMService) -> None:
        self.llm_service = service

    async def evaluate_round(
        self,
        scenario: "CausalScenario",
        debate_context: "DebateContext",
        level: int,
    ) -> AuditVerdict:
        """
        评估一轮辩论并给出裁决
        
        Args:
            scenario: 因果场景
            debate_context: 完整辩论上下文
            level: Pearl因果阶梯层级
        """
        turns = list(getattr(debate_context, "turns", []))
        agent_a_text = self._collect_speaker_text(turns, {"agent_a", "a", "agent a"})
        agent_b_text = self._collect_speaker_text(turns, {"agent_b", "b", "agent b"})
        transcript = "\n".join(
            f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}"
            for turn in turns
        )
        jury_verdict = await self._resolve_jury_verdict(scenario, debate_context)
        jury_winner, jury_consensus = self._jury_summary(jury_verdict)

        context_flags = self._build_context_flags(
            level=level,
            agent_a_text=agent_a_text,
            agent_b_text=agent_b_text,
            transcript=transcript,
        )
        tool_report = self.tool_executor.execute_for_claim(
            scenario=scenario,
            claim=agent_a_text or transcript,
            level=level,
            context=context_flags,
        )

        agent_a_logic = argument_logic_check(agent_a_text) if agent_a_text else {"n_fallacies_detected": 0, "detected_fallacies": []}
        agent_b_logic = argument_logic_check(agent_b_text) if agent_b_text else {"n_fallacies_detected": 0, "detected_fallacies": []}

        argument_quality_a = 0.78 - 0.12 * agent_a_logic["n_fallacies_detected"] - 0.08 * len(tool_report["identified_issues"])
        argument_quality_b = 0.62 - 0.08 * agent_b_logic["n_fallacies_detected"] + self._tool_awareness_bonus(agent_b_text)
        argument_quality_b += 0.02 * len(tool_report["successful_tools"])

        if jury_consensus >= 0.8:
            if jury_winner == "agent_b":
                argument_quality_b += 0.12
            elif jury_winner == "agent_a":
                argument_quality_a += 0.12
        elif jury_consensus >= 0.6:
            if jury_winner == "agent_b":
                argument_quality_b += 0.06
            elif jury_winner == "agent_a":
                argument_quality_a += 0.06

        argument_quality_a = self._clamp(argument_quality_a)
        argument_quality_b = self._clamp(argument_quality_b)

        evidence_strength = min(0.25, 0.05 * len(tool_report["identified_issues"]) + 0.02 * len(tool_report["supporting_evidence"]))
        if argument_quality_b - argument_quality_a > 0.05 + evidence_strength / 2:
            winner = "agent_b"
        elif argument_quality_a - argument_quality_b > 0.08 and not tool_report["identified_issues"]:
            winner = "agent_a"
        else:
            winner = "draw"

        causal_validity_score = self._clamp(
            max(argument_quality_a, argument_quality_b)
            if winner != "draw"
            else (argument_quality_a + argument_quality_b) / 2
        )
        identified_issues = self._deduplicate(
            list(agent_a_logic.get("detected_fallacies", []))
            + tool_report["identified_issues"]
        )
        reasoning = self._build_reasoning(
            winner=winner,
            jury_winner=jury_winner,
            jury_consensus=jury_consensus,
            tool_report=tool_report,
            agent_a_logic=agent_a_logic,
            agent_b_logic=agent_b_logic,
        )

        narration = await self._narrate(
            winner=winner,
            jury_winner=jury_winner,
            jury_consensus=jury_consensus,
            tool_report=tool_report,
            agent_a_text=agent_a_text,
            agent_b_text=agent_b_text,
        )
        if narration:
            reasoning = f"{reasoning}\n\n[LLM audit]\n{narration}"

        return AuditVerdict(
            winner=winner,
            causal_validity_score=causal_validity_score,
            argument_quality_a=argument_quality_a,
            argument_quality_b=argument_quality_b,
            reasoning=reasoning,
            identified_issues=identified_issues,
            tools_used=tool_report["successful_tools"],
            jury_consensus=jury_consensus,
        )

    async def _resolve_jury_verdict(self, scenario, debate_context) -> JuryVerdict | dict | None:
        existing = getattr(debate_context, "jury_result", None) or getattr(debate_context, "jury_verdict", None)
        if existing is not None:
            return existing
        turns = list(getattr(debate_context, "turns", []))
        for turn in turns:
            if str(turn.get("speaker", "")).lower() == "jury" and isinstance(turn.get("content"), dict):
                return turn["content"]
        return await self.jury.collect_votes(scenario, debate_context)

    def _jury_summary(self, jury_verdict) -> tuple[str, float]:
        if jury_verdict is None:
            return "draw", 0.0
        if isinstance(jury_verdict, JuryVerdict):
            return jury_verdict.final_winner or "draw", float(jury_verdict.agreement_rate)
        if isinstance(jury_verdict, dict):
            winner = jury_verdict.get("final_winner") or jury_verdict.get("jury_recommendation") or "draw"
            consensus = jury_verdict.get("agreement_rate") or jury_verdict.get("consensus_level") or 0.0
            return winner, float(consensus)
        return "draw", 0.0

    def _collect_speaker_text(self, turns: list[dict], speaker_aliases: set[str]) -> str:
        parts = []
        for turn in turns:
            speaker = str(turn.get("speaker", "")).lower()
            if speaker in speaker_aliases:
                parts.append(str(turn.get("content", "")))
        return "\n".join(parts)

    def _build_context_flags(
        self,
        level: int,
        agent_a_text: str,
        agent_b_text: str,
        transcript: str,
    ) -> dict:
        combined = " ".join([agent_a_text, agent_b_text, transcript]).lower()
        return {
            "has_instrument": any(token in combined for token in ["iv", "instrument", "工具变量", "出生季度"]),
            "has_mediator": any(token in combined for token in ["mediator", "中介", "frontdoor", "前门"]),
            "needs_full_counterfactual": level >= 3 or any(token in combined for token in ["counterfactual", "反事实"]),
            "suspected_confounders": any(token in combined for token in ["confound", "混杂", "遗漏变量"]),
        }

    def _tool_awareness_bonus(self, text: str) -> float:
        if not text:
            return 0.0
        lowered = text.lower()
        keywords = {
            "confound",
            "混杂",
            "后门",
            "backdoor",
            "iv",
            "instrument",
            "工具变量",
            "sensitivity",
            "敏感性",
            "counterfactual",
            "反事实",
            "p=",
            "条件独立",
        }
        matched = sum(1 for token in keywords if token in lowered)
        return min(0.25, matched * 0.04)

    def _build_reasoning(
        self,
        winner: str,
        jury_winner: str,
        jury_consensus: float,
        tool_report: dict,
        agent_a_logic: dict,
        agent_b_logic: dict,
    ) -> str:
        parts = [
            f"陪审团倾向 {jury_winner}，共识度 {jury_consensus:.2f}。",
            f"Agent A 被检测到 {agent_a_logic.get('n_fallacies_detected', 0)} 个明显逻辑问题。",
            f"Agent B 被检测到 {agent_b_logic.get('n_fallacies_detected', 0)} 个明显逻辑问题。",
        ]
        if tool_report["identified_issues"]:
            parts.append("工具验证发现的问题包括：" + "；".join(tool_report["identified_issues"][:3]) + "。")
        if tool_report["supporting_evidence"]:
            parts.append("关键证据：" + "；".join(tool_report["supporting_evidence"][:3]) + "。")
        parts.append(f"综合工具证据与陪审团意见，最终裁定 {winner}。")
        return " ".join(parts)

    def _deduplicate(self, items: list[str]) -> list[str]:
        result = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    async def _narrate(
        self,
        *,
        winner: str,
        jury_winner: str,
        jury_consensus: float,
        tool_report: dict,
        agent_a_text: str,
        agent_b_text: str,
    ) -> str:
        if self.llm_service is None or getattr(self.llm_service, "backend", "mock") == "mock":
            return ""
        prompt = (
            f"你是因果辩论的审计员，已完成工具验证与陪审团听证，请用 4-6 句话给出最终审计陈述。\n"
            f"裁定：{winner}。陪审团倾向：{jury_winner}，共识度 {jury_consensus:.2f}。\n"
            f"工具识别的问题：{'; '.join(tool_report.get('identified_issues', [])) or '无显著问题'}\n"
            f"工具提供的支持证据：{'; '.join(tool_report.get('supporting_evidence', [])) or '较弱'}\n"
            f"Agent A 的论述节选：{agent_a_text[:500]}\n"
            f"Agent B 的论述节选：{agent_b_text[:500]}\n"
            f"请基于以上写一段客观、专业的审计总结。"
        )
        try:
            response = await self.llm_service.generate(
                prompt,
                system_prompt="你是一个冷静、严谨的因果辩论审计员，必须依据工具与陪审团证据给出结论。",
            )
            text = (response.text or "").strip()
            if text.startswith("[mock:"):
                return ""
            return text
        except Exception:
            return ""
