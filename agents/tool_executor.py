"""
Tool executor - Agent C 的工具调用封装与轻量代码沙箱
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from types import CodeType
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from causal_tools.meta_tools import ToolSelector


@dataclass
class ToolExecutionResult:
    """单个工具的执行结果。"""

    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)


class SafeCodeValidator(ast.NodeVisitor):
    """对执行代码做最小必要的语法安全检查。"""

    BLOCKED_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )
    BLOCKED_NAMES = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "help",
        "dir",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
    }

    def generic_visit(self, node):
        if isinstance(node, self.BLOCKED_NODES):
            raise ValueError(f"不允许的语法节点: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.BLOCKED_NAMES:
            raise ValueError(f"不允许访问名称: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__"):
            raise ValueError("不允许访问 dunder 属性")
        if isinstance(node.value, ast.Name) and node.value.id in self.BLOCKED_NAMES:
            raise ValueError(f"不允许访问模块属性: {node.value.id}.{node.attr}")
        self.generic_visit(node)


class SafePythonExecutor:
    """受限 Python 执行器。"""

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "dict": dict,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    def __init__(self):
        self.validator = SafeCodeValidator()

    def compile(self, code: str) -> CodeType:
        tree = ast.parse(code, mode="exec")
        self.validator.visit(tree)
        return compile(tree, "<safe-python>", "exec")

    def execute(self, code: str, extra_context: dict[str, Any] | None = None) -> dict[str, Any]:
        compiled = self.compile(code)
        namespace: dict[str, Any] = {
            "__builtins__": self.SAFE_BUILTINS,
            "math": math,
            "np": np,
            "pd": pd,
            "nx": nx,
        }
        if extra_context:
            namespace.update(extra_context)
        exec(compiled, namespace, namespace)
        return {
            key: value
            for key, value in namespace.items()
            if not key.startswith("_")
            and key not in {"math", "np", "pd", "nx"}
            and key not in self.SAFE_BUILTINS
        }


class ToolExecutor:
    """Agent C 的工具调用编排器。"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.selector = ToolSelector()
        self.python_executor = SafePythonExecutor()

    def select_tools(
        self,
        level: int,
        claim: str,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        context = self._merge_context(context, claim)
        return self.selector.select(
            level,
            scenario_type=context.get("scenario_type", "default"),
            claim=claim,
            context=context,
        )

    def execute_tool(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        try:
            tool = self.selector.get_tool(tool_name)
            output = tool(**kwargs)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                output=output,
                kwargs=kwargs,
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
                kwargs=kwargs,
            )

    def execute_python(self, code: str, extra_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.python_executor.execute(code, extra_context=extra_context)

    def execute_for_claim(
        self,
        scenario,
        claim: str,
        level: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = self._merge_context(context, claim)
        selected_tools = self.select_tools(level, claim, context)
        results: list[ToolExecutionResult] = []

        for tool_name in selected_tools:
            kwargs = self._build_tool_kwargs(tool_name, scenario, claim, context)
            if kwargs is None:
                results.append(
                    ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error="上下文不足，无法构造参数",
                    )
                )
                continue
            results.append(self.execute_tool(tool_name, **kwargs))

        stance = context.get("claim_stance") or self._infer_claim_stance(claim)
        issues, evidence, counter_evidence, support_score = self._summarize_results(
            claim,
            results,
            stance,
            scenario=scenario,
        )
        return {
            "selected_tools": selected_tools,
            "results": results,
            "claim_stance": stance,
            "identified_issues": issues,
            "supporting_evidence": evidence,
            "counter_evidence": counter_evidence,
            "support_score": support_score,
            "successful_tools": [item.tool_name for item in results if item.success],
        }

    def _build_tool_kwargs(
        self,
        tool_name: str,
        scenario,
        claim: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        data = self._get_data(scenario)
        graph = self._get_graph(scenario)
        variables = self._get_variables(scenario, data)
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        treatment, outcome = self._infer_focus_variables(claim, variables, scenario=scenario)
        conditioning = context.get("adjustment_set") or self._get_observed_controls(graph, data, treatment, outcome)
        instrument = context.get("instrument") or ground_truth.get("instrument") or self._guess_instrument(graph, data, variables, treatment, outcome)
        mediator = context.get("mediator") or ground_truth.get("mediator") or self._guess_mediator(graph, data, variables, treatment, outcome)
        evidence = context.get("evidence")
        if evidence is None and data is not None and not data.empty:
            evidence = data.iloc[0].to_dict()
        intervention = context.get("intervention")
        if intervention is None and evidence is not None:
            intervention = {treatment: self._flip_value(evidence.get(treatment, 0))}
        scm = self._get_scm(scenario, graph)

        if tool_name in {"correlation_analysis", "compute_correlation"}:
            return {"data": data, "x": treatment, "y": outcome} if data is not None else None

        if tool_name == "conditional_independence_test":
            return {"data": data, "x": treatment, "y": outcome, "z": conditioning[:1]} if data is not None else None

        if tool_name == "detect_simpson_paradox":
            return {"data": data, "x": treatment, "y": outcome, "z": conditioning[0]} if data is not None and conditioning else None

        if tool_name == "partial_correlation":
            return {"data": data, "x": treatment, "y": outcome, "controls": conditioning[:1]} if data is not None else None

        if tool_name == "backdoor_adjustment":
            return {"data": data, "graph": graph, "treatment": treatment, "outcome": outcome} if data is not None and graph is not None else None

        if tool_name == "backdoor_adjustment_check":
            return {
                "data": data,
                "treatment": treatment,
                "outcome": outcome,
                "adjustment_set": context.get("adjustment_set", conditioning[:1]),
                "graph": graph,
            } if data is not None else None

        if tool_name in {"frontdoor_adjustment", "frontdoor_estimation"}:
            if data is None or mediator is None:
                return None
            if tool_name == "frontdoor_adjustment":
                return {
                    "data": data,
                    "graph": graph,
                    "treatment": treatment,
                    "outcome": outcome,
                    "mediator": mediator,
                } if graph is not None else None
            return {
                "data": data,
                "treatment": treatment,
                "mediator": mediator,
                "outcome": outcome,
            }

        if tool_name == "iv_estimation":
            return {
                "data": data,
                "instrument": instrument,
                "treatment": treatment,
                "outcome": outcome,
                "covariates": conditioning[:1],
            } if data is not None and instrument is not None else None

        if tool_name == "propensity_score_matching":
            return {
                "data": data,
                "treatment": treatment,
                "outcome": outcome,
                "covariates": conditioning[:2] or [column for column in variables if column not in {treatment, outcome}][:2],
            } if data is not None else None

        if tool_name == "sensitivity_analysis":
            return {"data": data, "treatment": treatment, "outcome": outcome} if data is not None else None

        if tool_name == "counterfactual_inference":
            return {
                "model": scm,
                "evidence": evidence,
                "intervention": intervention,
                "target": outcome,
            } if scm is not None and evidence is not None and intervention is not None else None

        if tool_name == "scm_identification_test":
            if data is None or scm is None:
                return None
            alternatives = context.get("alternative_scms") or self._build_alternative_scms(graph)
            return {
                "data": data[[treatment, outcome]].copy(),
                "proposed_scm": scm,
                "alternative_scms": alternatives,
            } if alternatives else None

        if tool_name == "ett_computation":
            return {
                "data": data[[treatment, outcome]].copy(),
                "scm": scm,
                "treatment": treatment,
                "outcome": outcome,
            } if data is not None and scm is not None else None

        if tool_name == "abduction_action_prediction":
            return {
                "scm": scm,
                "factual_world": evidence,
                "hypothetical_action": intervention,
            } if scm is not None and evidence is not None and intervention is not None else None

        if tool_name == "probability_of_necessity":
            return {"data": data[[treatment, outcome]].copy(), "treatment": treatment, "outcome": outcome} if data is not None else None

        if tool_name == "probability_of_sufficiency":
            return {"data": data[[treatment, outcome]].copy(), "treatment": treatment, "outcome": outcome} if data is not None else None

        if tool_name == "argument_logic_check":
            return {
                "argument": claim,
                "claimed_causal_relation": {"treatment": treatment, "outcome": outcome},
            }

        if tool_name == "causal_graph_validator":
            return {"graph": graph} if graph is not None else None

        return None

    def _summarize_results(
        self,
        claim: str,
        results: list[ToolExecutionResult],
        stance: str,
        scenario=None,
    ) -> tuple[list[str], list[str], list[str], float]:
        issues: list[str] = []
        evidence: list[str] = []
        counter_evidence: list[str] = []
        claim_lower = claim.lower()
        level = int(getattr(scenario, "causal_level", 1)) if scenario is not None else 1
        hidden_variables = list(getattr(scenario, "hidden_variables", [])) if scenario is not None else []
        tentative_tokens = [
            "可能",
            "尚不足",
            "不稳妥",
            "需要验证",
            "继续验证",
            "值得优先考虑",
            "优先检验",
            "应优先检验",
            "当前证据支持",
            "很可能",
            "likely",
            "may",
            "might",
            "pending",
            "subject to",
        ]
        hidden_tokens = ["混杂", "未观测", "latent", "hidden", "共同原因", "confound"]
        scm_tokens = ["反事实", "counterfactual", "scm", "结构模型", "函数形式", "不可识别", "拟合观测分布"]
        is_tentative = any(token in claim_lower for token in tentative_tokens)
        mentions_hidden = any(token in claim_lower for token in hidden_tokens)
        mentions_scm_uncertainty = any(token in claim_lower for token in scm_tokens)
        mentions_adjustment = any(token in claim_lower for token in ["后门", "backdoor", "调整", "adjustment", "遗漏"])
        mentions_iv = any(token in claim_lower for token in ["工具变量", "iv", "instrument", "排他性", "第一阶段"])
        negative_multiplier = 0.45 if stance == "pro_causal" and is_tentative else 1.0
        support_score = 0.17 if is_tentative else 0.15

        if stance == "anti_causal" and hidden_variables and mentions_hidden:
            evidence.append("场景中存在未观测变量候选，混杂质疑具可辩性。")
            support_score += 0.10 if level == 1 else 0.06
        if stance == "anti_causal" and level >= 3 and mentions_scm_uncertainty:
            evidence.append("反事实结论依赖结构模型设定，替代 SCM 风险需要审查。")
            support_score += 0.12

        for result in results:
            if not result.success:
                continue
            output = result.output

            if result.tool_name in {"probability_of_necessity", "probability_of_sufficiency"}:
                try:
                    probability = float(output)
                except Exception:
                    continue
                label = "必要性" if result.tool_name == "probability_of_necessity" else "充分性"
                if probability < 0.15:
                    if stance == "anti_causal":
                        evidence.append(f"{label}概率较低，反事实结论对结构设定较敏感。")
                        support_score += 0.08
                    else:
                        issues.append(f"{label}概率偏低，强反事实主张支撑不足")
                        support_score -= 0.08 * negative_multiplier
                elif probability >= 0.30 and stance == "pro_causal":
                    evidence.append(f"{label}概率达到 {probability:.2f}。")
                    support_score += 0.06
                continue

            if not isinstance(output, dict):
                continue

            if result.tool_name in {"correlation_analysis", "compute_correlation"}:
                significant = bool(output.get("significant"))
                effect_size = str(output.get("effect_size", ""))
                if stance == "anti_causal" and not significant:
                    evidence.append("观测相关本身并不显著，支持保持谨慎。")
                    support_score += 0.08
                elif stance == "pro_causal" and significant:
                    evidence.append(f"观测相关显著，效应量级为 {effect_size or '可检测'}。")
                    support_score += 0.04

            if result.tool_name == "argument_logic_check":
                detected = output.get("detected_fallacies", [])
                issues.extend(detected)
                if output.get("recommendation"):
                    if stance == "pro_causal":
                        counter_evidence.append(output["recommendation"])
                    else:
                        evidence.append(output["recommendation"])
                if stance == "pro_causal":
                    support_score -= 0.04 * len(detected) * negative_multiplier
                else:
                    support_score -= 0.02 * len(detected)

            if result.tool_name == "causal_graph_validator":
                confounders = output.get("confounder_candidates", [])
                if stance == "anti_causal" and confounders and mentions_hidden:
                    evidence.append("图结构存在潜在共同原因路径，混杂攻击并非空穴来风。")
                    support_score += 0.08 if level == 1 else 0.05
                elif stance == "anti_causal" and level >= 3 and confounders and mentions_scm_uncertainty:
                    evidence.append("结构中存在多条机制路径，反事实结论依赖模型设定。")
                    support_score += 0.08
                elif stance == "pro_causal" and output.get("is_dag"):
                    support_score += 0.02

            if output.get("simpson_paradox_detected"):
                if stance == "anti_causal":
                    evidence.append("分组趋势与总体趋势不一致。")
                    support_score += 0.18
                else:
                    issues.append("可能存在辛普森悖论")
                    counter_evidence.append("分组趋势与总体趋势不一致。")
                    support_score -= 0.12 * negative_multiplier

            if output.get("independent") and any(token in claim_lower for token in ["导致", "cause", "effect", "prove"]):
                if stance == "pro_causal":
                    issues.append("条件化后相关性消失，因果主张缺乏支撑")
                    support_score -= 0.16 * negative_multiplier
                else:
                    evidence.append("条件化后相关性减弱，支持识别边界存疑。")
                    support_score += 0.14

            if output.get("is_valid_adjustment") is False:
                if stance == "pro_causal":
                    issues.append("给定调整集不满足后门准则")
                    support_score -= 0.16 * negative_multiplier
                else:
                    evidence.append("给定调整集不满足后门准则。")
                    support_score += 0.12 if mentions_adjustment else 0.05

            if output.get("is_sensitive"):
                if stance == "pro_causal":
                    issues.append("结论对未观测混杂敏感")
                    support_score -= 0.14 * negative_multiplier
                else:
                    evidence.append("结论对未观测混杂敏感。")
                    support_score += 0.10 if mentions_hidden else 0.04

            if result.tool_name == "iv_estimation" and output.get("is_strong_instrument") is False:
                if stance == "pro_causal":
                    issues.append("工具变量较弱或第一阶段不足")
                    support_score -= 0.18 * negative_multiplier
                else:
                    evidence.append("工具变量较弱或第一阶段不足。")
                    support_score += 0.14 if mentions_iv else 0.05
            elif result.tool_name == "iv_estimation" and output.get("is_strong_instrument") is True:
                if stance == "pro_causal":
                    evidence.append("工具变量第一阶段足够强。")
                    support_score += 0.16
                else:
                    counter_evidence.append("工具变量第一阶段足够强。")
                    support_score -= 0.12

            if result.tool_name in {"frontdoor_adjustment", "frontdoor_estimation"} and output.get("graph_valid") is False:
                if stance == "pro_causal":
                    issues.append("前门识别条件不足")
                    support_score -= 0.14 * negative_multiplier
                else:
                    evidence.append("前门识别条件不足。")
                    support_score += 0.10

            if result.tool_name == "scm_identification_test":
                indistinguishable = [
                    item for item in output if not item.get("distinguishable", True)
                ]
                if indistinguishable:
                    if stance == "pro_causal":
                        issues.append("当前 SCM 与替代模型难以区分")
                        support_score -= 0.16 * negative_multiplier
                    else:
                        evidence.append("当前 SCM 与替代模型难以区分。")
                        support_score += 0.14

            if "causal_effect" in output:
                target = evidence if stance == "pro_causal" else counter_evidence
                target.append(f"{result.tool_name} 估计效应约为 {output['causal_effect']:.2f}")
                if stance == "pro_causal" and abs(float(output["causal_effect"])) > 0.1:
                    support_score += 0.08
            if "estimated_effect" in output and "causal_effect" not in output:
                target = evidence if stance == "pro_causal" else counter_evidence
                target.append(f"{result.tool_name} 估计效应约为 {output['estimated_effect']:.2f}")
            if "frontdoor_effect" in output:
                target = evidence if stance == "pro_causal" else counter_evidence
                target.append(f"前门通路总效应约为 {output['frontdoor_effect']:.2f}")
            if "counterfactual_outcome" in output:
                target = evidence if stance == "pro_causal" else counter_evidence
                target.append(f"反事实预测结果为 {float(output['counterfactual_outcome']):.2f}")
            if "p_value" in output and output["p_value"] is not None:
                p_value = float(output["p_value"])
                if stance == "pro_causal" and p_value < 0.05:
                    support_score += 0.04
                if stance == "anti_causal" and p_value >= 0.05:
                    support_score += 0.05
            if "robust_up_to_gamma" in output:
                gamma = float(output["robust_up_to_gamma"])
                if stance == "pro_causal" and gamma >= 1.5:
                    support_score += 0.05
                if stance == "anti_causal" and gamma < 1.4:
                    support_score += 0.05

        dedup_issues = []
        for issue in issues:
            if issue not in dedup_issues:
                dedup_issues.append(issue)
        dedup_evidence = []
        for item in evidence:
            if item not in dedup_evidence:
                dedup_evidence.append(item)
        dedup_counter = []
        for item in counter_evidence:
            if item not in dedup_counter:
                dedup_counter.append(item)
        return dedup_issues, dedup_evidence, dedup_counter, max(0.0, min(1.0, support_score))

    def _get_data(self, scenario) -> pd.DataFrame | None:
        if hasattr(scenario, "data") and isinstance(scenario.data, pd.DataFrame):
            return scenario.data
        if hasattr(scenario, "observed_data") and isinstance(scenario.observed_data, pd.DataFrame):
            return scenario.observed_data
        return None

    def _get_graph(self, scenario) -> nx.DiGraph | None:
        graph = getattr(scenario, "true_dag", None)
        if isinstance(graph, nx.DiGraph):
            return graph.copy()
        if isinstance(graph, dict):
            dag = nx.DiGraph()
            for source, targets in graph.items():
                for target in targets:
                    dag.add_edge(source, target)
            return dag
        scm = getattr(scenario, "true_scm", None)
        if isinstance(scm, dict):
            graph_value = scm.get("graph")
            if isinstance(graph_value, nx.DiGraph):
                return graph_value.copy()
            if isinstance(graph_value, dict):
                dag = nx.DiGraph()
                for source, targets in graph_value.items():
                    for target in targets:
                        dag.add_edge(source, target)
                return dag
        return None

    def _get_scm(self, scenario, graph: nx.DiGraph | None):
        scm = getattr(scenario, "true_scm", None)
        if scm is not None:
            return scm
        if graph is not None:
            coefficients: dict[str, dict[str, float]] = {}
            for node in graph.nodes:
                parents = list(graph.predecessors(node))
                coefficients[node] = {"intercept": 0.0}
                for parent in parents:
                    coefficients[node][parent] = 1.0 / max(1, len(parents))
            return {"graph": graph.copy(), "coefficients": coefficients}
        return None

    def _get_variables(self, scenario, data: pd.DataFrame | None) -> list[str]:
        variables = list(getattr(scenario, "variables", []))
        if not variables and data is not None:
            variables = list(data.columns)
        if not variables:
            raise ValueError("场景缺少变量信息")
        return variables

    def _infer_focus_variables(self, claim: str, variables: list[str], scenario=None) -> tuple[str, str]:
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        treatment = ground_truth.get("treatment")
        outcome = ground_truth.get("outcome")
        if treatment in variables and outcome in variables:
            return str(treatment), str(outcome)
        claim_lower = claim.lower()
        matched = [
            (claim_lower.find(variable.lower()), variable)
            for variable in variables
            if variable.lower() in claim_lower
        ]
        matched = [item for item in matched if item[0] >= 0]
        matched.sort(key=lambda item: item[0])
        ordered = [variable for _, variable in matched]
        if len(ordered) >= 2:
            return ordered[0], ordered[1]
        if "X" in variables and "Y" in variables:
            return "X", "Y"
        return variables[0], variables[-1]

    def _guess_instrument(
        self,
        graph: nx.DiGraph | None,
        data: pd.DataFrame | None,
        variables: list[str],
        treatment: str,
        outcome: str,
    ) -> str | None:
        preferred = ["Z", "IV", "instrument", "quarter"]
        observed = set(data.columns) if data is not None else set(variables)
        for candidate in variables:
            if candidate in {treatment, outcome}:
                continue
            if candidate not in observed:
                continue
            if any(token.lower() == candidate.lower() for token in preferred):
                return candidate
        if graph is not None:
            cut_graph = graph.copy()
            if treatment in cut_graph:
                cut_graph.remove_edges_from(list(cut_graph.out_edges(treatment)))
            for candidate in variables:
                if candidate in {treatment, outcome} or candidate not in observed or candidate not in graph:
                    continue
                if candidate in nx.descendants(graph, treatment):
                    continue
                try:
                    has_path_to_treatment = nx.has_path(graph, candidate, treatment)
                    has_backdoor_to_outcome = nx.has_path(cut_graph, candidate, outcome)
                except nx.NetworkXError:
                    continue
                if has_path_to_treatment and not has_backdoor_to_outcome:
                    return candidate
        for candidate in variables:
            if candidate not in {treatment, outcome} and candidate in observed:
                return candidate
        return None

    def _guess_mediator(
        self,
        graph: nx.DiGraph | None,
        data: pd.DataFrame | None,
        variables: list[str],
        treatment: str,
        outcome: str,
    ) -> str | None:
        preferred = ["M", "mediator", "mid"]
        observed = set(data.columns) if data is not None else set(variables)
        for candidate in variables:
            if candidate in {treatment, outcome}:
                continue
            if candidate not in observed:
                continue
            if any(token.lower() == candidate.lower() for token in preferred):
                return candidate
        if graph is not None:
            for candidate in variables:
                if candidate in {treatment, outcome} or candidate not in observed or candidate not in graph:
                    continue
                try:
                    if candidate in nx.descendants(graph, treatment) and nx.has_path(graph, candidate, outcome):
                        return candidate
                except nx.NetworkXError:
                    continue
        return None

    def _build_alternative_scms(self, graph: nx.DiGraph | None) -> list[dict]:
        if graph is None or graph.number_of_edges() == 0:
            return []
        alternative = graph.copy()
        edge_to_remove = next(iter(alternative.edges()))
        alternative.remove_edge(*edge_to_remove)
        coefficients: dict[str, dict[str, float]] = {}
        for node in alternative.nodes:
            parents = list(alternative.predecessors(node))
            coefficients[node] = {"intercept": 0.0}
            for parent in parents:
                coefficients[node][parent] = 1.0 / max(1, len(parents))
        return [{"name": "edge_removed_model", "graph": alternative, "coefficients": coefficients}]

    def _flip_value(self, value: Any) -> Any:
        try:
            numeric = float(value)
        except Exception:
            return value
        if numeric in {0.0, 1.0}:
            return 1 - int(numeric)
        return 0.0 if numeric > 0 else 1.0

    def _get_observed_controls(
        self,
        graph: nx.DiGraph | None,
        data: pd.DataFrame | None,
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

    def _infer_claim_stance(self, claim: str) -> str:
        lowered = claim.lower()
        anti_tokens = [
            "不可信", "不可靠", "不能", "无法", "未必", "混杂", "偏差", "逆因果",
            "selection", "confound", "敏感", "不可识别", "模型假设", "函数形式",
        ]
        pro_tokens = [
            "导致", "会导致", "有效", "提高", "增加", "改善", "证明", "成立",
            "causes", "effect", "works", "robust",
        ]
        anti_score = sum(token in lowered for token in anti_tokens)
        pro_score = sum(token in lowered for token in pro_tokens)
        if anti_score > pro_score:
            return "anti_causal"
        return "pro_causal"

    def _merge_context(self, context: dict[str, Any] | None, claim: str) -> dict[str, Any]:
        merged = dict(context or {})
        lowered = claim.lower()
        merged.setdefault(
            "has_instrument",
            any(token in lowered for token in ["iv", "instrument", "工具变量", "出生季度"]),
        )
        merged.setdefault(
            "has_mediator",
            any(token in lowered for token in ["mediator", "中介", "frontdoor", "前门"]),
        )
        merged.setdefault(
            "needs_full_counterfactual",
            any(token in lowered for token in ["counterfactual", "反事实", "abduction"]),
        )
        if merged["has_instrument"]:
            merged.setdefault("scenario_type", "instrument")
        elif merged["has_mediator"]:
            merged.setdefault("scenario_type", "mediator")
        else:
            merged.setdefault("scenario_type", "default")
        merged.setdefault("claim_stance", self._infer_claim_stance(claim))
        return merged
