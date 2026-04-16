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
from benchmark.schema import PublicCausalInstance, require_public_instance
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
class ScientificClaim:
    """Agent B 的初始科学假设。"""

    content: str
    causal_claim: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    tools_used: list[str] = field(default_factory=list)


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
        scenario: PublicCausalInstance,
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
        scenario = require_public_instance(scenario)
        data = self._get_data(scenario)
        graph = self._get_graph(scenario)
        variables = self._get_variables(scenario, data)
        treatment, outcome = self._infer_focus_variables(claim, variables, scenario=scenario)
        observed_controls = self._get_observed_controls(graph, data, treatment, outcome)
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        claim_lower = claim.lower()

        reasoning_chain: list[str] = []
        detected_fallacies: list[str] = []
        discovered_hidden_vars: list[str] = []
        tools_used: list[str] = []
        defense_score = 0.38 + 0.04 * level
        challenge_validity = 0.12 + 0.05 * max(level - 1, 0)

        logic_result = argument_logic_check(claim)
        if logic_result["detected_fallacies"]:
            detected_fallacies.extend(logic_result["detected_fallacies"])
            reasoning_chain.append(logic_result["recommendation"])
            tools_used.append("argument_logic_check")
            defense_score += 0.05 * logic_result["n_fallacies_detected"]

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
            if correlation["significant"] and abs(correlation["pearson_r"]) >= 0.15:
                defense_score += 0.10

            if observed_controls:
                ci_result = conditional_independence_test(data, treatment, outcome, observed_controls)
                partial = partial_correlation(data, treatment, outcome, observed_controls)
                tools_used.extend(["conditional_independence_test", "partial_correlation"])
                reasoning_chain.append(
                    f"控制 {observed_controls} 后偏相关为 {partial['partial_correlation']:.2f}，p={partial['p_value']:.4f}。"
                )
                if partial["significant"]:
                    defense_score += 0.08
                if ci_result["independent"] and correlation["significant"]:
                    challenge_validity += 0.12
                    hidden_variable_name = self._suggest_hidden_variable(scenario, treatment, outcome)
                    discovered_hidden_vars.append(hidden_variable_name)
                    reasoning_chain.append("总相关显著但控制候选变量后近似独立，符合混杂导致的虚假相关特征。")

            if level == 1 and any(token in claim_lower for token in ["混杂", "confound", "未观测"]):
                challenge_validity += 0.10 if getattr(scenario, "hidden_variables", []) else 0.0
                discovered_hidden_vars.append(self._suggest_hidden_variable(scenario, treatment, outcome))
            if level == 1 and any(token in claim_lower for token in ["选择偏差", "selection bias", "逆因果"]):
                challenge_validity += 0.03

            if level >= 2:
                adjustment_set = observed_controls
                if adjustment_set:
                    backdoor = backdoor_adjustment_check(data, treatment, outcome, adjustment_set, graph=graph)
                    tools_used.append("backdoor_adjustment_check")
                    reasoning_chain.append(
                        f"后门调整后效应为 {backdoor['estimated_effect']:.2f}，相对未调整变化 {backdoor['delta_from_naive']:.2f}。"
                    )
                    if backdoor["is_valid_adjustment"] is False:
                        challenge_validity += 0.10
                    elif abs(backdoor["estimated_effect"]) > 0.10:
                        defense_score += 0.10

                sensitivity = sensitivity_analysis(data, treatment, outcome)
                tools_used.append("sensitivity_analysis")
                reasoning_chain.append(
                    f"灵敏度分析显示结论稳健到 Gamma={sensitivity['robust_up_to_gamma']:.2f}。"
                )
                if sensitivity["is_sensitive"]:
                    challenge_validity += 0.10
                else:
                    defense_score += 0.06

                instrument = ground_truth.get("instrument") or self._find_instrument_variable(claim, variables, treatment, outcome)
                if level == 2 and instrument is not None and instrument in data.columns:
                    iv_result = iv_estimation(data, instrument, treatment, outcome, observed_controls)
                    tools_used.append("iv_estimation")
                    reasoning_chain.append(
                        f"工具变量 {instrument} 的第一阶段 F={iv_result['first_stage_f']:.2f}，估计效应 {iv_result['causal_effect']:.2f}。"
                    )
                    if iv_result["is_strong_instrument"]:
                        defense_score += 0.16
                        if any(token in claim_lower for token in ["工具变量并不可信", "iv", "instrument", "排他性"]):
                            detected_fallacies.append("过度质疑有效工具变量")
                    else:
                        challenge_validity += 0.16

            if level >= 3:
                pn = probability_of_necessity(data, treatment, outcome)
                ps = probability_of_sufficiency(data, treatment, outcome)
                tools_used.extend(["probability_of_necessity", "probability_of_sufficiency"])
                reasoning_chain.append(f"必要性概率约为 {pn:.2f}，充分性概率约为 {ps:.2f}。")
                if pn + ps > 0.35:
                    defense_score += 0.08
                else:
                    challenge_validity += 0.08

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
                        defense_score += 0.05
                    except Exception:
                        reasoning_chain.append("反事实推断因场景模型接口不足而跳过。")
                if any(token in claim_lower for token in ["scm", "反事实", "函数形式", "不可识别", "模型敏感"]):
                    challenge_validity += 0.10

        if any(token in claim_lower for token in ["必然", "100%", "毫无疑问", "唯一解释"]):
            detected_fallacies.append("过度确定性")
            defense_score += 0.06
        if any(token in claim_lower for token in ["选择偏差", "selection bias"]) and level != 1:
            detected_fallacies.append("选择偏差论证证据不足")
            defense_score += 0.04
        if any(token in claim_lower for token in ["只有特定子群", "仅在某些人群", "att", "atu"]) and level == 2:
            challenge_validity += 0.04

        detected_fallacies = self._deduplicate(detected_fallacies)
        discovered_hidden_vars = self._deduplicate(discovered_hidden_vars)
        tools_used = self._deduplicate(tools_used)
        fallback_confidence = self._clamp(defense_score - 0.55 * challenge_validity + 0.02 * len(detected_fallacies))

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

    async def propose_hypothesis(
        self,
        scenario: PublicCausalInstance,
        level: int,
        context: Optional["DebateContext"] = None,
    ) -> ScientificClaim:
        """提出面向当前层级的科学因果假设。"""
        scenario = require_public_instance(scenario)
        data = self._get_data(scenario)
        variables = self._get_variables(scenario, data)
        treatment, outcome = self._infer_focus_variables("", variables, scenario=scenario)
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        evidence: list[str] = []
        tools_used: list[str] = []

        if data is not None and treatment in data.columns and outcome in data.columns:
            corr = compute_correlation(data, treatment, outcome)
            evidence.append(f"{treatment} 与 {outcome} 的相关系数约为 {corr['pearson_r']:.2f}")
            tools_used.append("compute_correlation")

        if level == 1:
            content = (
                f"从观测数据看，{treatment} 与 {outcome} 的关联显著，"
                f"在已观测协变量范围内仍值得优先考虑因果解释。"
            )
        elif level == 2:
            instrument = ground_truth.get("instrument")
            instrument_text = f"，并可进一步借助工具变量 {instrument} 识别" if instrument else ""
            content = (
                f"当前证据支持 {treatment} 对 {outcome} 存在正向干预效应，"
                f"应优先检验后门调整与工具变量路径{instrument_text}。"
            )
        else:
            mediator = ground_truth.get("mediator")
            mediator_text = f"，尤其需要结合中介 {mediator} 与反事实推断" if mediator else ""
            content = (
                f"当前结构证据表明 {treatment} 很可能影响 {outcome}，"
                f"但这一判断需要在 SCM 与反事实层面继续验证{mediator_text}。"
            )

        if "observational_difference" in ground_truth:
            evidence.append(f"观测差异约为 {float(ground_truth['observational_difference']):.2f}")
        if "observational_slope" in ground_truth:
            evidence.append(f"观测斜率约为 {float(ground_truth['observational_slope']):.2f}")
        if "instrument" in ground_truth:
            evidence.append(f"候选工具变量为 {ground_truth['instrument']}")
        if "mediator" in ground_truth:
            evidence.append(f"关键中介为 {ground_truth['mediator']}")

        return ScientificClaim(
            content=content,
            causal_claim=f"{treatment} 可能导致 {outcome}",
            evidence=self._deduplicate(evidence),
            confidence=0.62 + 0.05 * max(level - 1, 0),
            tools_used=self._deduplicate(tools_used),
        )

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

    def _get_observed_controls(
        self,
        graph: Optional[nx.DiGraph],
        data: Optional[pd.DataFrame],
        treatment: str,
        outcome: str,
    ) -> list[str]:
        if graph is None or data is None:
            return []
        observed = [column for column in data.columns if column not in {treatment, outcome}]
        cut_graph = graph.copy()
        if treatment in cut_graph:
            cut_graph.remove_edges_from(list(cut_graph.out_edges(treatment)))
        controls: list[str] = []
        for node in observed:
            if node not in graph:
                continue
            if node in nx.descendants(graph, treatment):
                continue
            try:
                has_path_to_treatment = nx.has_path(graph, node, treatment)
                has_backdoor_to_outcome = nx.has_path(cut_graph, node, outcome)
            except nx.NetworkXError:
                continue
            if has_path_to_treatment and has_backdoor_to_outcome:
                controls.append(node)
        return controls

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

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
