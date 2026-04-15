"""
Agent B - 科学家（Scientist）
负责识别因果谬误，发现隐变量
模型：Qwen2.5-14B-Instruct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import pandas as pd

from agents.prompts.agent_b_prompts import AGENT_B_SYSTEM_PROMPT
from causal_tools.l1_association import conditional_independence_test, compute_correlation, partial_correlation
from causal_tools.l2_intervention import backdoor_adjustment_check, iv_estimation, sensitivity_analysis
from causal_tools.l3_counterfactual import (
    counterfactual_inference,
    probability_of_necessity,
    probability_of_sufficiency,
)
from causal_tools.meta_tools import ToolSelector, argument_logic_check, causal_graph_validator
from game.llm_service import LLMService


@dataclass
class DetectionResult:
    """隐变量检测结果"""
    detected_fallacies: list[str] = field(default_factory=list)
    discovered_hidden_vars: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class AgentB:
    """
    科学家Agent - 隐变量检测者
    
    使用因果工具集进行：
    - 条件独立性检验
    - 后门/前门准则验证
    - IV有效性检验
    - 敏感性分析
    - 反事实推理验证
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.llm_service: LLMService | None = None
        self.detection_history: list[DetectionResult] = []
        self.tool_selector = ToolSelector()

    async def initialize(self):
        """初始化模型连接"""
        self.model = self.config.get("models", {}).get("agent_b", self.config.get("agent_b", {}))
        if self.llm_service is None:
            self.llm_service = LLMService(self.model if isinstance(self.model, dict) else {})
        await self.llm_service.initialize()

    def attach_llm_service(self, service: LLMService) -> None:
        self.llm_service = service

    async def analyze_claim(
        self,
        claim: str,
        scenario: "CausalScenario",
        level: int,
        context: Optional["DebateContext"] = None,
    ) -> DetectionResult:
        """
        分析因果声明，检测欺骗
        
        Args:
            claim: Agent A的因果声明
            scenario: 因果场景
            level: Pearl因果阶梯层级
            context: 辩论上下文
        """
        tools = await self.select_tools(level, claim)
        data = self._get_data(scenario)
        graph = self._get_graph(scenario)
        variables = self._get_variables(scenario, data)
        treatment, outcome = self._infer_focus_variables(claim, variables, scenario=scenario)
        conditioning = [variable for variable in variables if variable not in {treatment, outcome}][:2]

        reasoning_chain: list[str] = []
        detected_fallacies: list[str] = []
        discovered_hidden_vars: list[str] = []
        tools_used: list[str] = []

        logic_result = argument_logic_check(claim)
        if logic_result["detected_fallacies"]:
            detected_fallacies.extend(logic_result["detected_fallacies"])
            reasoning_chain.append(logic_result["recommendation"])
            tools_used.append("argument_logic_check")

        if graph is not None:
            graph_result = causal_graph_validator(graph)
            tools_used.append("causal_graph_validator")
            if graph_result["confounder_candidates"]:
                reasoning_chain.append("因果图中存在共享父节点，说明需要重点排查混杂路径。")

        if data is not None and treatment in data.columns and outcome in data.columns:
            correlation = compute_correlation(data, treatment, outcome)
            tools_used.append("compute_correlation")
            reasoning_chain.append(
                f"{treatment} 与 {outcome} 的相关系数为 {correlation['pearson_r']:.2f}，显著性 p={correlation['pearson_p']:.4f}。"
            )

            if conditioning:
                ci_result = conditional_independence_test(data, treatment, outcome, conditioning)
                partial = partial_correlation(data, treatment, outcome, conditioning)
                tools_used.extend(["conditional_independence_test", "partial_correlation"])
                reasoning_chain.append(
                    f"控制 {conditioning} 后偏相关为 {partial['partial_correlation']:.2f}，p={partial['p_value']:.4f}。"
                )
                if ci_result["independent"] and correlation["significant"]:
                    detected_fallacies.append("可能存在未控制混杂")
                    hidden_variable_name = self._suggest_hidden_variable(scenario, treatment, outcome)
                    discovered_hidden_vars.append(hidden_variable_name)
                    reasoning_chain.append("总相关显著但控制候选变量后近似独立，符合混杂导致的虚假相关特征。")

            if level >= 2:
                adjustment_set = conditioning[:1]
                if adjustment_set:
                    backdoor = backdoor_adjustment_check(data, treatment, outcome, adjustment_set, graph=graph)
                    tools_used.append("backdoor_adjustment_check")
                    reasoning_chain.append(
                        f"后门调整后效应为 {backdoor['estimated_effect']:.2f}，相对未调整变化 {backdoor['delta_from_naive']:.2f}。"
                    )
                    if abs(backdoor["delta_from_naive"]) > 0.2:
                        detected_fallacies.append("遗漏后门路径")

                sensitivity = sensitivity_analysis(data, treatment, outcome)
                tools_used.append("sensitivity_analysis")
                reasoning_chain.append(
                    f"灵敏度分析显示结论稳健到 Gamma={sensitivity['robust_up_to_gamma']:.2f}。"
                )
                if sensitivity["is_sensitive"]:
                    detected_fallacies.append("结论对未观测混杂敏感")

                instrument = self._find_instrument_variable(claim, variables, treatment, outcome)
                if instrument is not None and instrument in data.columns:
                    iv_result = iv_estimation(data, instrument, treatment, outcome, conditioning[:1])
                    tools_used.append("iv_estimation")
                    reasoning_chain.append(
                        f"工具变量 {instrument} 的第一阶段 F={iv_result['first_stage_f']:.2f}，估计效应 {iv_result['causal_effect']:.2f}。"
                    )
                    if not iv_result["is_strong_instrument"]:
                        detected_fallacies.append("工具变量过弱或识别不足")

            if level >= 3:
                pn = probability_of_necessity(data, treatment, outcome)
                ps = probability_of_sufficiency(data, treatment, outcome)
                tools_used.extend(["probability_of_necessity", "probability_of_sufficiency"])
                reasoning_chain.append(f"必要性概率约为 {pn:.2f}，充分性概率约为 {ps:.2f}。")

                scm = getattr(scenario, "true_scm", None)
                if scm is not None:
                    try:
                        evidence = data.iloc[0].to_dict()
                        intervention = {treatment: 1 - int(round(float(evidence[treatment])))}
                        cf = counterfactual_inference(scm, evidence, intervention, outcome)
                        tools_used.append("counterfactual_inference")
                        reasoning_chain.append(
                            f"在反事实世界中，{outcome} 的预测值变为 {float(cf['counterfactual_outcome']):.2f}。"
                        )
                    except Exception:
                        reasoning_chain.append("反事实推断因场景模型接口不足而跳过。")

        detected_fallacies = self._deduplicate(detected_fallacies)
        discovered_hidden_vars = self._deduplicate(discovered_hidden_vars)
        tools_used = self._deduplicate(tools_used)
        fallback_confidence = min(
            0.95,
            0.35
            + 0.12 * len(detected_fallacies)
            + 0.08 * len(discovered_hidden_vars)
            + 0.03 * len(tools_used),
        )

        llm_payload = await self._llm_decide(
            claim=claim,
            level=level,
            fallacies=detected_fallacies,
            hidden=discovered_hidden_vars,
            reasoning=reasoning_chain,
            tools=tools_used,
        )
        final_fallacies = self._merge_lists(
            detected_fallacies,
            self._normalize_list(llm_payload.get("detected_fallacies")),
        )
        final_hidden = self._merge_lists(
            discovered_hidden_vars,
            self._normalize_list(llm_payload.get("discovered_hidden_vars")),
        )
        final_tools = self._merge_lists(
            tools_used,
            self._normalize_list(llm_payload.get("tools_used")),
        )
        llm_reasoning = self._normalize_list(llm_payload.get("reasoning_chain"))
        final_reasoning = llm_reasoning or reasoning_chain
        final_confidence = self._normalize_confidence(
            llm_payload.get("confidence"),
            fallback=fallback_confidence,
        )

        result = DetectionResult(
            detected_fallacies=final_fallacies,
            discovered_hidden_vars=final_hidden,
            confidence=final_confidence,
            reasoning_chain=final_reasoning,
            tools_used=final_tools,
        )
        self.detection_history.append(result)
        return result

    async def _llm_decide(
        self,
        *,
        claim: str,
        level: int,
        fallacies: list[str],
        hidden: list[str],
        reasoning: list[str],
        tools: list[str],
    ) -> dict:
        if self.llm_service is None or getattr(self.llm_service, "backend", "mock") == "mock":
            return {}
        prompt = (
            f"因果层级：L{level}\n"
            f"对手声明：{claim}\n"
            f"工具初步识别出的谬误：{fallacies or ['无']}\n"
            f"疑似隐变量：{hidden or ['无']}\n"
            f"已调用工具：{tools or ['无']}\n"
            f"工具推理链：{reasoning[:8]}\n"
            "请基于这些证据，直接输出一个 JSON 对象，字段必须包括："
            'detected_fallacies, discovered_hidden_vars, confidence, reasoning_chain, tools_used。'
            "你需要自己做最终判断，而不是复述输入。"
        )
        try:
            _, payload = await self.llm_service.generate_json(
                prompt,
                system_prompt=AGENT_B_SYSTEM_PROMPT,
            )
            return payload or {}
        except Exception:
            return {}

    async def select_tools(self, level: int, claim: str) -> list[str]:
        """根据层级和声明选择合适的因果工具"""
        context = {
            "has_instrument": any(token in claim.lower() for token in ["iv", "instrument", "工具变量", "出生季度"]),
            "has_mediator": any(token in claim.lower() for token in ["mediator", "中介", "frontdoor", "前门"]),
            "needs_full_counterfactual": any(token in claim.lower() for token in ["counterfactual", "反事实", "abduction"]),
        }
        scenario_type = "instrument" if context["has_instrument"] else "mediator" if context["has_mediator"] else "default"
        return self.tool_selector.select(level, scenario_type=scenario_type, claim=claim, context=context)

    def _get_data(self, scenario) -> Optional[pd.DataFrame]:
        if hasattr(scenario, "data") and isinstance(scenario.data, pd.DataFrame):
            return scenario.data
        if hasattr(scenario, "observed_data") and isinstance(scenario.observed_data, pd.DataFrame):
            return scenario.observed_data
        return None

    def _get_graph(self, scenario) -> Optional[nx.DiGraph]:
        graph = getattr(scenario, "true_dag", None)
        if isinstance(graph, nx.DiGraph):
            return graph
        if isinstance(graph, dict):
            dag = nx.DiGraph()
            for source, targets in graph.items():
                for target in targets:
                    dag.add_edge(source, target)
            return dag
        scm = getattr(scenario, "true_scm", None)
        if isinstance(scm, dict) and "graph" in scm:
            return self._get_graph(type("SCMScenario", (), {"true_dag": scm["graph"]})())
        return None

    def _get_variables(self, scenario, data: Optional[pd.DataFrame]) -> list[str]:
        variables = list(getattr(scenario, "variables", []))
        if not variables and data is not None:
            variables = list(data.columns)
        if len(variables) < 2:
            raise ValueError("场景至少需要两个变量")
        return variables

    def _infer_focus_variables(self, claim: str, variables: list[str], scenario=None) -> tuple[str, str]:
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        treatment = ground_truth.get("treatment")
        outcome = ground_truth.get("outcome")
        if treatment in variables and outcome in variables:
            return str(treatment), str(outcome)
        matched = [variable for variable in variables if variable.lower() in claim.lower()]
        if len(matched) >= 2:
            return matched[0], matched[1]
        return variables[0], variables[1]

    def _find_instrument_variable(
        self, claim: str, variables: list[str], treatment: str, outcome: str
    ) -> Optional[str]:
        for variable in variables:
            if variable in {treatment, outcome}:
                continue
            if variable.lower() in claim.lower():
                return variable
        for variable in variables:
            if variable not in {treatment, outcome}:
                return variable
        return None

    def _suggest_hidden_variable(self, scenario, treatment: str, outcome: str) -> str:
        hidden = list(getattr(scenario, "hidden_variables", []))
        if hidden:
            return hidden[0]
        return f"latent_confounder_between_{treatment}_and_{outcome}"

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.append(item)
        return seen

    def _normalize_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _normalize_confidence(self, value: object, *, fallback: float) -> float:
        try:
            number = float(value)
        except Exception:
            return fallback
        return max(0.0, min(1.0, number))

    def _merge_lists(self, left: list[str], right: list[str]) -> list[str]:
        return self._deduplicate([*left, *right])
