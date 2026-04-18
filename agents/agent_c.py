"""
Agent C - 审计员（Auditor）
负责评估因果论证质量，裁决胜负
模型：Qwen2.5-72B-Instruct
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Optional

from agents.jury import JuryAggregator, JuryVerdict
from agents.prompts.agent_c_prompts import AGENT_C_WITH_JURY_PROMPT
from agents.tool_executor import ToolExecutionResult, ToolExecutor
from benchmark.schema import PublicCausalInstance, require_public_instance
from causal_tools.meta_tools import argument_logic_check
from game.llm_service import LLMService
from verifier.pipeline import VerifierPipeline


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
    verdict_label: str | None = None
    verifier_confidence: float = 0.0
    assumption_ledger: list[dict[str, Any]] = field(default_factory=list)
    witness: dict[str, Any] | None = None
    support_witness: dict[str, Any] | None = None
    countermodel_witness: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    verifier_verdict: dict[str, Any] = field(default_factory=dict)


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
        self.verifier_pipeline = VerifierPipeline()
        # ── 防御升级：跨轮学习状态 ──
        self.detection_history: list[dict] = []
        self.known_patterns: list[str] = []
        self.sensitivity_boost: float = 0.0

    async def initialize(self):
        """初始化模型连接"""
        self.model = self.config.get("models", {}).get("agent_c", self.config.get("agent_c", {}))
        if self.llm_service is None:
            self.llm_service = LLMService(self.model if isinstance(self.model, dict) else {})
        await self.llm_service.initialize()
        await self.jury.initialize()

    def attach_llm_service(self, service: LLMService) -> None:
        self.llm_service = service

    def attach_verifier_pipeline(self, pipeline: VerifierPipeline) -> None:
        self.verifier_pipeline = pipeline

    # ── 防御升级：跨轮学习 ──────────────────────────────
    def upgrade_defense(self, feedback: dict) -> None:
        """根据上一轮检测结果升级防御能力。

        *feedback* 由 debate_engine 在每轮结束后构造，包含:
        - detected (bool): 本轮是否成功识破欺骗
        - strategy_used (str): Agent A 本轮使用的策略类型
        - deception_score (float): Agent A 的欺骗质量分
        - round_id (int): 轮次编号
        """
        self.detection_history.append(feedback)

        # 记录已知策略模式（去重）
        strategy = feedback.get("strategy_used", "")
        if strategy and strategy not in self.known_patterns:
            self.known_patterns.append(strategy)

        # 根据近 N 轮检测成功率动态调整灵敏度
        window = self.detection_history[-10:]
        n_detected = sum(1 for r in window if r.get("detected", False))
        detection_rate = n_detected / len(window)

        if detection_rate >= 0.7:
            # 检测能力强 → 小幅提升，避免过拟合
            self.sensitivity_boost = min(self.sensitivity_boost + 0.02, 0.15)
        elif detection_rate <= 0.3:
            # 检测能力弱 → 大幅提升，加速学习
            self.sensitivity_boost = min(self.sensitivity_boost + 0.05, 0.25)
        else:
            # 中等 → 稳步提升
            self.sensitivity_boost = min(self.sensitivity_boost + 0.03, 0.20)

    async def evaluate_round(
        self,
        scenario: PublicCausalInstance,
        debate_context: "DebateContext",
        level: int,
    ) -> AuditVerdict:
        scenario = require_public_instance(scenario)
        if self._use_verifier_first():
            try:
                return await self._evaluate_round_verifier_first(
                    scenario=scenario,
                    debate_context=debate_context,
                    level=level,
                )
            except Exception as exc:
                raise RuntimeError(
                    "AgentC verifier-first evaluation failed; legacy fallback is disabled for the main benchmark path."
                ) from exc
        return await self._evaluate_round_legacy(
            scenario=scenario,
            debate_context=debate_context,
            level=level,
        )

    async def _evaluate_round_verifier_first(
        self,
        scenario: PublicCausalInstance,
        debate_context: "DebateContext",
        level: int,
    ) -> AuditVerdict:
        turns = list(getattr(debate_context, "turns", []))
        transcript = [
            turn.to_dict() if hasattr(turn, "to_dict") else dict(turn)
            for turn in turns
        ]
        transcript_text = "\n".join(
            f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}"
            for turn in turns
        )
        agent_a_text = self._collect_speaker_text(turns, {"agent_a", "a", "agent a"})
        agent_b_text = self._collect_speaker_text(turns, {"agent_b", "b", "agent b"})
        primary_claim = (
            self._collect_phase_text(turns, {"agent_a", "a", "agent a"}, "challenge")
            or agent_a_text
            or transcript_text
        )
        jury_verdict = await self._resolve_jury_verdict(scenario, debate_context)
        jury_winner, jury_consensus = self._jury_summary(jury_verdict)
        context_flags = self._build_context_flags(
            scenario=scenario,
            level=level,
            agent_a_text=agent_a_text,
            agent_b_text=agent_b_text,
            transcript=transcript_text,
        )
        verifier_verdict = self.verifier_pipeline.run(
            primary_claim,
            scenario=scenario,
            transcript=transcript,
            tool_context=context_flags,
        )
        verdict_payload = verifier_verdict.to_dict()
        tool_trace = list(verifier_verdict.tool_trace)
        winner = self._winner_from_verdict_label(verifier_verdict.label.value)
        argument_quality_a, argument_quality_b, causal_validity_score = self._score_verdict(
            verifier_verdict.label.value,
            verifier_verdict.confidence,
        )
        identified_issues = self._verifier_identified_issues(verifier_verdict)
        reasoning = self._verifier_reasoning_summary(
            verifier_verdict.reasoning_summary,
            verifier_verdict.label.value,
            jury_winner,
            jury_consensus,
        )
        return AuditVerdict(
            winner=winner,
            causal_validity_score=causal_validity_score,
            argument_quality_a=argument_quality_a,
            argument_quality_b=argument_quality_b,
            reasoning=reasoning,
            identified_issues=identified_issues,
            tools_used=self._deduplicate(
                [
                    trace.get("tool_name", "")
                    for trace in tool_trace
                    if trace.get("tool_name")
                ]
            ),
            jury_consensus=jury_consensus,
            verdict_label=verifier_verdict.label.value,
            verifier_confidence=verifier_verdict.confidence,
            assumption_ledger=[entry.to_dict() for entry in verifier_verdict.assumption_ledger.entries],
            witness=verdict_payload.get("witness"),
            support_witness=verdict_payload.get("support_witness"),
            countermodel_witness=verdict_payload.get("countermodel_witness"),
            tool_trace=tool_trace,
            verifier_verdict=verdict_payload,
        )

    async def _evaluate_round_legacy(
        self,
        scenario: PublicCausalInstance,
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
        scenario = require_public_instance(scenario)
        turns = list(getattr(debate_context, "turns", []))
        agent_a_text = self._collect_speaker_text(turns, {"agent_a", "a", "agent a"})
        agent_b_text = self._collect_speaker_text(turns, {"agent_b", "b", "agent b"})
        initial_b_claim = self._collect_phase_text(turns, {"agent_b", "b", "agent b"}, "claim")
        rebuttal_b_text = self._collect_phase_text(turns, {"agent_b", "b", "agent b"}, "rebuttal")
        transcript = "\n".join(
            f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}"
            for turn in turns
        )
        jury_verdict = await self._resolve_jury_verdict(scenario, debate_context)
        jury_winner, jury_consensus = self._jury_summary(jury_verdict)

        context_flags = self._build_context_flags(
            scenario=scenario,
            level=level,
            agent_a_text=agent_a_text,
            agent_b_text=agent_b_text,
            transcript=transcript,
        )
        combined_b_case = "\n".join(
            part for part in [initial_b_claim, rebuttal_b_text] if part
        ).strip() or agent_b_text or transcript
        tool_report_a = self.tool_executor.execute_for_claim(
            scenario=scenario,
            claim=agent_a_text or transcript,
            level=level,
            context={**context_flags, "claim_stance": "anti_causal"},
        )
        tool_report_b = self.tool_executor.execute_for_claim(
            scenario=scenario,
            claim=combined_b_case,
            level=level,
            context={**context_flags, "claim_stance": "pro_causal"},
        )

        agent_a_logic = argument_logic_check(agent_a_text) if agent_a_text else {"n_fallacies_detected": 0, "detected_fallacies": []}
        agent_b_logic = argument_logic_check(agent_b_text) if agent_b_text else {"n_fallacies_detected": 0, "detected_fallacies": []}

        profile = self._level_profile(level)
        rebuttal_confidence = self._extract_rebuttal_confidence(debate_context)
        argument_quality_a = (
            profile["a_base"]
            + profile["a_support_weight"] * tool_report_a["support_score"]
            - 0.05 * agent_a_logic["n_fallacies_detected"]
            + self._anti_causal_bonus(level, agent_a_text, scenario)
        )
        argument_quality_b = (
            profile["b_base"]
            + profile["b_support_weight"] * tool_report_b["support_score"]
            - 0.05 * agent_b_logic["n_fallacies_detected"]
        )
        argument_quality_b += profile["claim_weight"] * self._extract_b_confidence(debate_context)
        argument_quality_b += profile["rebuttal_weight"] * rebuttal_confidence
        argument_quality_b += profile["tool_awareness_weight"] * self._tool_awareness_bonus(agent_b_text)
        argument_quality_a += 0.01 * len(tool_report_a["successful_tools"]) * tool_report_a["support_score"]
        argument_quality_b += 0.01 * len(tool_report_b["successful_tools"]) * tool_report_b["support_score"]

        # ── 防御升级：已知模式惩罚 + 灵敏度提升 ──
        if self.known_patterns and agent_a_text:
            a_lower = agent_a_text.lower()
            pattern_hits = sum(1 for p in self.known_patterns if p.lower() in a_lower)
            if pattern_hits:
                argument_quality_a -= 0.04 * pattern_hits  # 重复策略惩罚
        argument_quality_b += self.sensitivity_boost  # 跨轮学习增益

        if jury_consensus >= 0.8:
            if jury_winner == "agent_b":
                argument_quality_b += profile["jury_high"]
            elif jury_winner == "agent_a":
                argument_quality_a += profile["jury_high"]
        elif jury_consensus >= 0.6:
            if jury_winner == "agent_b":
                argument_quality_b += profile["jury_mid"]
            elif jury_winner == "agent_a":
                argument_quality_a += profile["jury_mid"]

        argument_quality_a = self._clamp(argument_quality_a)
        argument_quality_b = self._clamp(argument_quality_b)

        if argument_quality_b - argument_quality_a > profile["margin"]:
            winner = "agent_b"
        elif argument_quality_a - argument_quality_b > profile["margin"]:
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
            + tool_report_a["identified_issues"]
            + tool_report_b["identified_issues"]
        )
        fallback_reasoning = self._build_reasoning(
            winner=winner,
            jury_winner=jury_winner,
            jury_consensus=jury_consensus,
            tool_report_a=tool_report_a,
            tool_report_b=tool_report_b,
            agent_a_logic=agent_a_logic,
            agent_b_logic=agent_b_logic,
        )

        llm_payload = await self._llm_decide(
            winner=winner,
            jury_winner=jury_winner,
            jury_consensus=jury_consensus,
            tool_report={
                "identified_issues": tool_report_a["identified_issues"] + tool_report_b["identified_issues"],
                "supporting_evidence": tool_report_b["supporting_evidence"] + tool_report_a["supporting_evidence"],
            },
            agent_a_text=agent_a_text,
            agent_b_text=agent_b_text,
            debate_transcript=transcript,
            fallback_reasoning=fallback_reasoning,
        )
        final_winner = self._normalize_winner(llm_payload.get("winner"), fallback=winner)
        final_reasoning = self._clean_text(llm_payload.get("reasoning")) or fallback_reasoning
        final_identified_issues = self._deduplicate(
            identified_issues + self._normalize_list(llm_payload.get("identified_issues"))
        )
        final_quality_a = self._normalize_score(
            llm_payload.get("argument_quality_a"),
            fallback=argument_quality_a,
        )
        final_quality_b = self._normalize_score(
            llm_payload.get("argument_quality_b"),
            fallback=argument_quality_b,
        )
        final_causal_validity = self._normalize_score(
            llm_payload.get("causal_validity_score"),
            fallback=causal_validity_score,
        )

        return AuditVerdict(
            winner=final_winner,
            causal_validity_score=final_causal_validity,
            argument_quality_a=final_quality_a,
            argument_quality_b=final_quality_b,
            reasoning=final_reasoning,
            identified_issues=final_identified_issues,
            tools_used=self._deduplicate(tool_report_a["successful_tools"] + tool_report_b["successful_tools"]),
            jury_consensus=jury_consensus,
        )

    def _use_verifier_first(self) -> bool:
        agent_config = self.config.get("agent_c", {}) or {}
        mode = str(agent_config.get("mode", "verifier")).strip().lower()
        return mode not in {"legacy", "debate", "classic"}

    def _winner_from_verdict_label(self, verdict_label: str) -> str:
        if verdict_label == "valid":
            return "agent_a"
        if verdict_label == "invalid":
            return "agent_b"
        return "draw"

    def _score_verdict(self, verdict_label: str, confidence: float) -> tuple[float, float, float]:
        if verdict_label == "valid":
            argument_quality_a = self._clamp(0.62 + 0.26 * confidence)
            argument_quality_b = self._clamp(0.38 - 0.12 * confidence)
            causal_validity_score = self._clamp(confidence)
        elif verdict_label == "invalid":
            argument_quality_a = self._clamp(0.38 - 0.18 * confidence)
            argument_quality_b = self._clamp(0.62 + 0.22 * confidence)
            causal_validity_score = self._clamp(1.0 - confidence)
        else:
            argument_quality_a = self._clamp(0.5)
            argument_quality_b = self._clamp(0.5)
            causal_validity_score = 0.5
        return argument_quality_a, argument_quality_b, causal_validity_score

    def _verifier_identified_issues(self, verifier_verdict) -> list[str]:
        issues: list[str] = []
        for entry in verifier_verdict.assumption_ledger.entries:
            if entry.status.value != "supported":
                issues.append(f"{entry.name}:{entry.status.value}")
        countermodel_witness = getattr(verifier_verdict, "countermodel_witness", None)
        if countermodel_witness is not None:
            countermodel_type = countermodel_witness.payload.get("countermodel_type")
            if countermodel_type:
                issues.append(str(countermodel_type))
        return self._deduplicate(issues)

    def _verifier_reasoning_summary(
        self,
        reasoning_summary: str,
        verdict_label: str,
        jury_winner: str,
        jury_consensus: float,
    ) -> str:
        verdict_text = f"Verifier label={verdict_label}."
        jury_text = f" Jury signal={jury_winner} ({jury_consensus:.2f})."
        return f"{verdict_text} {reasoning_summary}{jury_text}".strip()

    def _build_pipeline_tool_trace(
        self,
        claim_report: dict[str, Any],
        rebuttal_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tool_trace: list[dict[str, Any]] = []
        tool_trace.extend(
            self._tool_report_to_trace(
                claim_report,
                supports_claim=True,
                claim_stance="pro_causal",
            )
        )
        tool_trace.extend(
            self._tool_report_to_trace(
                rebuttal_report,
                supports_claim=False,
                claim_stance="anti_causal",
            )
        )
        return tool_trace

    def _tool_report_to_trace(
        self,
        report: dict[str, Any],
        *,
        supports_claim: bool,
        claim_stance: str,
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        successful_tools = set(report.get("successful_tools", []))
        for result in report.get("results", []):
            if not isinstance(result, ToolExecutionResult) or not result.success:
                continue
            summary = self._summarize_tool_output(result)
            supports_assumptions, contradicts_assumptions = self._extract_tool_assumptions(result)
            trace.append(
                {
                    "tool_name": result.tool_name,
                    "status": "success",
                    "summary": summary,
                    "supports_assumptions": supports_assumptions,
                    "contradicts_assumptions": contradicts_assumptions,
                    "supports_claim": (
                        supports_claim
                        and result.tool_name in successful_tools
                        and not report.get("identified_issues")
                    ),
                    "supports_primary_claim": (
                        supports_claim
                        and result.tool_name in successful_tools
                        and not report.get("identified_issues")
                    ),
                    "claim_stance": claim_stance,
                    "evidence_direction": "support" if supports_claim else "counter",
                    "error": "",
                    "confidence": report.get("support_score", 0.0),
                }
            )

        if report.get("supporting_evidence"):
            trace.append(
                {
                    "tool_name": "tool_report_summary",
                    "status": "success",
                    "summary": " ".join(report["supporting_evidence"][:3]),
                    "supports_assumptions": [],
                    "contradicts_assumptions": [],
                    "supports_claim": supports_claim and bool(report.get("supporting_evidence")) and not report.get("identified_issues"),
                    "supports_primary_claim": supports_claim and bool(report.get("supporting_evidence")) and not report.get("identified_issues"),
                    "claim_stance": claim_stance,
                    "evidence_direction": "support" if supports_claim else "counter",
                    "error": "",
                    "confidence": report.get("support_score", 0.0),
                }
            )
        return trace

    def _summarize_tool_output(self, result: ToolExecutionResult) -> str:
        output = result.output
        if isinstance(output, dict):
            if "causal_effect" in output:
                return f"{result.tool_name} estimates causal_effect={float(output['causal_effect']):.3f}"
            if "estimated_effect" in output:
                return f"{result.tool_name} estimates effect={float(output['estimated_effect']):.3f}"
            if "frontdoor_effect" in output:
                return f"{result.tool_name} estimates frontdoor_effect={float(output['frontdoor_effect']):.3f}"
            if "counterfactual_outcome" in output:
                return f"{result.tool_name} predicts counterfactual_outcome={float(output['counterfactual_outcome']):.3f}"
            if "is_valid_adjustment" in output:
                return f"{result.tool_name} adjustment_valid={output['is_valid_adjustment']}"
            if "is_strong_instrument" in output:
                return f"{result.tool_name} strong_instrument={output['is_strong_instrument']}"
        if isinstance(output, list) and output:
            return f"{result.tool_name} returned {len(output)} alternative-model diagnostics"
        return f"{result.tool_name} completed"

    def _extract_tool_assumptions(
        self,
        result: ToolExecutionResult,
    ) -> tuple[list[str], list[str]]:
        supports: list[str] = []
        contradicts: list[str] = []
        output = result.output if isinstance(result.output, dict) else None

        if result.tool_name in {"backdoor_adjustment", "backdoor_adjustment_check"} and output is not None:
            if output.get("is_valid_adjustment") is True:
                supports.extend(["valid adjustment set"])
            elif output.get("is_valid_adjustment") is False:
                contradicts.extend(["valid adjustment set"])

        if result.tool_name == "iv_estimation" and output is not None:
            if output.get("is_strong_instrument") is True:
                supports.extend(["instrument relevance"])
            elif output.get("is_strong_instrument") is False:
                contradicts.extend(["instrument relevance"])

        if result.tool_name == "sensitivity_analysis" and output is not None:
            if output.get("is_sensitive") is False:
                supports.extend(["no unobserved confounding"])
            elif output.get("is_sensitive") is True:
                contradicts.extend(["no unobserved confounding"])

        if result.tool_name == "scm_identification_test" and isinstance(result.output, list):
            indistinguishable = [
                item for item in result.output
                if isinstance(item, dict) and not item.get("distinguishable", True)
            ]
            if indistinguishable:
                contradicts.extend(["counterfactual model uniqueness"])
            else:
                supports.extend(["counterfactual model uniqueness"])

        return self._deduplicate(supports), self._deduplicate(contradicts)

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

    def _collect_phase_text(self, turns: list[dict], speaker_aliases: set[str], phase_value: str) -> str:
        parts = []
        for turn in turns:
            speaker = str(turn.get("speaker", "")).lower()
            phase = str(turn.get("phase", "")).lower()
            if speaker in speaker_aliases and phase == phase_value:
                parts.append(str(turn.get("content", "")))
        return "\n".join(parts)

    def _build_context_flags(
        self,
        scenario: PublicCausalInstance,
        level: int,
        agent_a_text: str,
        agent_b_text: str,
        transcript: str,
    ) -> dict:
        combined = " ".join([agent_a_text, agent_b_text, transcript]).lower()
        proxy_variables = list(getattr(scenario, "proxy_variables", []) or [])
        selection_variables = list(getattr(scenario, "selection_variables", []) or [])
        return {
            "has_instrument": any(token in combined for token in ["iv", "instrument", "工具变量", "出生季度"]),
            "has_mediator": any(token in combined for token in ["mediator", "中介", "frontdoor", "前门"]),
            "needs_full_counterfactual": level >= 3 or any(token in combined for token in ["counterfactual", "反事实"]),
            "suspected_confounders": any(token in combined for token in ["confound", "混杂", "遗漏变量"]),
            "proxy_variables": proxy_variables,
            "selection_variables": selection_variables,
            "selection_mechanism": getattr(scenario, "selection_mechanism", None),
            "has_proxy": bool(proxy_variables),
            "has_selection": bool(selection_variables)
            or str(getattr(scenario, "selection_mechanism", "")).strip().lower() not in {"", "none"},
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
        tool_report_a: dict,
        tool_report_b: dict,
        agent_a_logic: dict,
        agent_b_logic: dict,
    ) -> str:
        parts = [
            f"陪审团倾向 {jury_winner}，共识度 {jury_consensus:.2f}。",
            f"Agent A 被检测到 {agent_a_logic.get('n_fallacies_detected', 0)} 个明显逻辑问题。",
            f"Agent B 被检测到 {agent_b_logic.get('n_fallacies_detected', 0)} 个明显逻辑问题。",
        ]
        if tool_report_b["supporting_evidence"]:
            parts.append("支持 B 的关键证据：" + "；".join(tool_report_b["supporting_evidence"][:3]) + "。")
        if tool_report_a["supporting_evidence"]:
            parts.append("支持 A 的关键证据：" + "；".join(tool_report_a["supporting_evidence"][:3]) + "。")
        if tool_report_b["identified_issues"]:
            parts.append("B 主张的薄弱点：" + "；".join(tool_report_b["identified_issues"][:2]) + "。")
        if tool_report_a["counter_evidence"]:
            parts.append("A 质疑面临的反证：" + "；".join(tool_report_a["counter_evidence"][:2]) + "。")
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

    async def _llm_decide(
        self,
        *,
        winner: str,
        jury_winner: str,
        jury_consensus: float,
        tool_report: dict,
        agent_a_text: str,
        agent_b_text: str,
        debate_transcript: str,
        fallback_reasoning: str,
    ) -> dict:
        if self.llm_service is None or getattr(self.llm_service, "backend", "mock") == "mock":
            return {}
        jury_payload = json.dumps(
            {
                "jury_winner": jury_winner,
                "jury_consensus": jury_consensus,
                "tool_issues": tool_report.get("identified_issues", []),
                "tool_evidence": tool_report.get("supporting_evidence", []),
                "fallback_winner": winner,
            },
            ensure_ascii=False,
            indent=2,
        )
        transcript_payload = (
            f"Agent A:\n{agent_a_text[:800]}\n\n"
            f"Agent B:\n{agent_b_text[:800]}\n\n"
            f"Full transcript excerpt:\n{debate_transcript[:1200]}\n\n"
            f"Fallback reasoning:\n{fallback_reasoning}"
        )
        prompt = (
            AGENT_C_WITH_JURY_PROMPT
            .replace("{jury_result}", jury_payload)
            .replace("{debate_transcript}", transcript_payload)
        )
        try:
            _, payload = await self.llm_service.generate_json(
                prompt,
                system_prompt="你是一个冷静、严谨的因果辩论审计员，必须依据工具与陪审团证据给出结论。",
            )
            return payload or {}
        except Exception:
            return {}

    def _normalize_winner(self, value: object, *, fallback: str) -> str:
        if isinstance(value, str) and value in {"agent_a", "agent_b", "draw"}:
            return value
        return fallback

    def _normalize_score(self, value: object, *, fallback: float) -> float:
        try:
            score = float(value)
        except Exception:
            return fallback
        return self._clamp(score)

    def _normalize_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _clean_text(self, value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _extract_b_confidence(self, debate_context) -> float:
        claim = debate_context.metadata.get("agent_b_claim", {})
        if isinstance(claim, dict):
            try:
                return self._clamp(float(claim.get("confidence", 0.0)))
            except Exception:
                return 0.0
        return 0.0

    def _extract_rebuttal_confidence(self, debate_context) -> float:
        for turn in reversed(list(getattr(debate_context, "turns", []))):
            speaker = str(turn.get("speaker", "")).lower()
            phase = str(turn.get("phase", "")).lower()
            if speaker == "agent_b" and phase == "rebuttal":
                try:
                    return self._clamp(float(turn.get("metadata", {}).get("confidence", 0.0)))
                except Exception:
                    return 0.0
        return 0.0

    def _anti_causal_bonus(self, level: int, text: str, scenario=None) -> float:
        if not text:
            return 0.0
        lowered = text.lower()
        if level == 1 and any(token in lowered for token in ["混杂", "未观测", "共同原因", "confound", "latent"]):
            difficulty = float(getattr(scenario, "difficulty", 0.5)) if scenario is not None else 0.5
            scaled = max(0.0, min(1.0, (difficulty - 0.82) / 0.13))
            return 0.04 + 0.10 * scaled
        if level == 2 and any(token in lowered for token in ["工具变量", "iv", "后门", "heterogeneity", "att", "atu"]):
            return 0.08
        if level >= 3 and any(token in lowered for token in ["反事实", "counterfactual", "scm", "结构模型", "不可识别", "函数形式", "替代模型"]):
            return 0.10
        return 0.03 if any(token in lowered for token in ["不稳妥", "需要验证", "尚不足"]) else 0.0

    def _level_profile(self, level: int) -> dict[str, float]:
        return {
            1: {
                "a_base": 0.33,
                "b_base": 0.34,
                "a_support_weight": 0.52,
                "b_support_weight": 0.46,
                "claim_weight": 0.03,
                "rebuttal_weight": 0.04,
                "tool_awareness_weight": 0.45,
                "jury_high": 0.02,
                "jury_mid": 0.01,
                "margin": 0.05,
            },
            2: {
                "a_base": 0.35,
                "b_base": 0.34,
                "a_support_weight": 0.50,
                "b_support_weight": 0.48,
                "claim_weight": 0.08,
                "rebuttal_weight": 0.06,
                "tool_awareness_weight": 0.70,
                "jury_high": 0.03,
                "jury_mid": 0.02,
                "margin": 0.04,
            },
            3: {
                "a_base": 0.38,
                "b_base": 0.32,
                "a_support_weight": 0.50,
                "b_support_weight": 0.48,
                "claim_weight": 0.07,
                "rebuttal_weight": 0.05,
                "tool_awareness_weight": 0.60,
                "jury_high": 0.02,
                "jury_mid": 0.01,
                "margin": 0.04,
            },
        }.get(
            level,
            {
                "a_base": 0.34,
                "b_base": 0.34,
                "a_support_weight": 0.50,
                "b_support_weight": 0.48,
                "claim_weight": 0.06,
                "rebuttal_weight": 0.05,
                "tool_awareness_weight": 0.60,
                "jury_high": 0.04,
                "jury_mid": 0.02,
                "margin": 0.06,
            },
        )
