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

        issues, evidence = self._summarize_results(claim, results)
        return {
            "selected_tools": selected_tools,
            "results": results,
            "identified_issues": issues,
            "supporting_evidence": evidence,
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
        treatment, outcome = self._infer_focus_variables(claim, variables)
        conditioning = [
            variable for variable in variables if variable not in {treatment, outcome}
        ]
        instrument = context.get("instrument") or self._guess_instrument(variables, treatment, outcome)
        mediator = context.get("mediator") or self._guess_mediator(variables, treatment, outcome)
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
                "covariates": conditioning[:2] or [outcome],
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
    ) -> tuple[list[str], list[str]]:
        issues: list[str] = []
        evidence: list[str] = []
        claim_lower = claim.lower()

        for result in results:
            if not result.success:
                continue
            output = result.output
            if not isinstance(output, dict):
                continue

            if result.tool_name == "argument_logic_check":
                issues.extend(output.get("detected_fallacies", []))
                if output.get("recommendation"):
                    evidence.append(output["recommendation"])

            if output.get("simpson_paradox_detected"):
                issues.append("可能存在辛普森悖论")
                evidence.append("分组趋势与总体趋势不一致。")

            if output.get("independent") and any(token in claim_lower for token in ["导致", "cause", "effect", "prove"]):
                issues.append("条件化后相关性消失，因果主张缺乏支撑")

            if output.get("is_valid_adjustment") is False:
                issues.append("给定调整集不满足后门准则")

            if output.get("is_sensitive"):
                issues.append("结论对未观测混杂敏感")

            if result.tool_name == "iv_estimation" and output.get("is_strong_instrument") is False:
                issues.append("工具变量较弱或第一阶段不足")

            if result.tool_name in {"frontdoor_adjustment", "frontdoor_estimation"} and output.get("graph_valid") is False:
                issues.append("前门识别条件不足")

            if result.tool_name == "scm_identification_test":
                indistinguishable = [
                    item for item in output if not item.get("distinguishable", True)
                ]
                if indistinguishable:
                    issues.append("当前 SCM 与替代模型难以区分")

            if "causal_effect" in output:
                evidence.append(f"{result.tool_name} 估计效应约为 {output['causal_effect']:.2f}")
            if "estimated_effect" in output and "causal_effect" not in output:
                evidence.append(f"{result.tool_name} 估计效应约为 {output['estimated_effect']:.2f}")
            if "frontdoor_effect" in output:
                evidence.append(f"前门通路总效应约为 {output['frontdoor_effect']:.2f}")
            if "counterfactual_outcome" in output:
                evidence.append(f"反事实预测结果为 {float(output['counterfactual_outcome']):.2f}")

        dedup_issues = []
        for issue in issues:
            if issue not in dedup_issues:
                dedup_issues.append(issue)
        dedup_evidence = []
        for item in evidence:
            if item not in dedup_evidence:
                dedup_evidence.append(item)
        return dedup_issues, dedup_evidence

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

    def _infer_focus_variables(self, claim: str, variables: list[str]) -> tuple[str, str]:
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
        self, variables: list[str], treatment: str, outcome: str
    ) -> str | None:
        preferred = ["Z", "IV", "instrument", "quarter"]
        for candidate in variables:
            if candidate in {treatment, outcome}:
                continue
            if any(token.lower() == candidate.lower() for token in preferred):
                return candidate
        for candidate in variables:
            if candidate not in {treatment, outcome}:
                return candidate
        return None

    def _guess_mediator(
        self, variables: list[str], treatment: str, outcome: str
    ) -> str | None:
        preferred = ["M", "mediator", "mid"]
        for candidate in variables:
            if candidate in {treatment, outcome}:
                continue
            if any(token.lower() == candidate.lower() for token in preferred):
                return candidate
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
        return merged
