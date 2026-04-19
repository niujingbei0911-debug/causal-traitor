"""
Tool executor - Agent C 的工具调用封装与轻量代码沙箱
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from types import CodeType
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from benchmark.schema import PublicCausalInstance, require_public_instance
from causal_tools.meta_tools import ToolSelector
from verifier.claim_parser import parse_claim


@dataclass
class ToolExecutionResult:
    """单个工具的执行结果。"""

    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "kwargs": dict(self.kwargs),
        }


@dataclass
class ToolTraceEntry:
    tool_name: str
    status: str
    summary: str
    supports_claim: bool = False
    supports_primary_claim: bool = False
    claim_stance: str = "pro_causal"
    evidence_direction: str = "neutral"
    error: str = ""
    supports_assumptions: list[str] = field(default_factory=list)
    contradicts_assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "summary": self.summary,
            "supports_claim": self.supports_claim,
            "supports_primary_claim": self.supports_primary_claim,
            "claim_stance": self.claim_stance,
            "evidence_direction": self.evidence_direction,
            "error": self.error,
            "supports_assumptions": list(self.supports_assumptions),
            "contradicts_assumptions": list(self.contradicts_assumptions),
        }


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
        *,
        scenario: PublicCausalInstance | None = None,
    ) -> list[str]:
        context = self._merge_context(context, claim, scenario=scenario)
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
        scenario: PublicCausalInstance,
        claim: str,
        level: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scenario = require_public_instance(scenario)
        context = self._merge_context(context, claim, scenario=scenario)
        selected_tools = self.select_tools(level, claim, context, scenario=scenario)
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
        semantic_result = self._build_public_semantics_result(
            scenario,
            claim,
            parsed_claim=context.get("_parsed_claim"),
        )
        if semantic_result is not None:
            results.append(semantic_result)

        stance = context.get("claim_stance") or self._infer_claim_stance(claim)
        issues, evidence, counter_evidence, support_score = self._summarize_results(
            claim,
            results,
            stance,
            scenario=scenario,
        )
        tool_trace = self._build_tool_trace(results, claim=claim, stance=stance)
        return {
            "selected_tools": selected_tools,
            "results": results,
            "tool_trace": tool_trace,
            "claim_stance": stance,
            "identified_issues": issues,
            "supporting_evidence": evidence,
            "counter_evidence": counter_evidence,
            "evidence_component": {
                "heuristic_support": support_score,
                "supporting_evidence": list(evidence),
                "counter_evidence": list(counter_evidence),
                "identified_issues": list(issues),
                "support_score_role": "legacy_evidence_component",
            },
            "support_score": support_score,
            "successful_tools": [item.tool_name for item in results if item.success],
        }

    def _build_tool_trace(
        self,
        results: list[ToolExecutionResult],
        *,
        claim: str,
        stance: str,
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        parsed_claim = self._parse_claim(claim)
        for result in results:
            supports_assumptions, contradicts_assumptions = self._extract_tool_assumptions(
                result,
                claim=claim,
                parsed_claim=parsed_claim,
            )
            status = "success" if result.success else "skipped"
            if result.error and result.kwargs:
                status = "error"

            if result.success:
                if contradicts_assumptions:
                    evidence_direction = "counter"
                elif supports_assumptions:
                    evidence_direction = "support"
                else:
                    evidence_direction = "neutral"
            else:
                evidence_direction = "neutral"

            supports_claim = False
            supports_primary_claim = False
            if result.success:
                if stance == "pro_causal":
                    supports_claim = evidence_direction != "counter"
                else:
                    supports_claim = evidence_direction == "counter"
                supports_primary_claim = evidence_direction == "support"

            trace.append(
                ToolTraceEntry(
                    tool_name=result.tool_name,
                    status=status,
                    summary=self._summarize_tool_output(result),
                    supports_claim=supports_claim,
                    supports_primary_claim=supports_primary_claim,
                    claim_stance=str(stance),
                    evidence_direction=evidence_direction,
                    error=result.error,
                    supports_assumptions=supports_assumptions,
                    contradicts_assumptions=contradicts_assumptions,
                ).to_dict()
            )
        return trace

    def _build_public_semantics_result(
        self,
        scenario: PublicCausalInstance,
        claim: str,
        *,
        parsed_claim: Any | None = None,
    ) -> ToolExecutionResult | None:
        measurement_semantics = self._measurement_semantics(scenario)
        if not measurement_semantics:
            return None

        supports: list[str] = []
        contradicts: list[str] = []
        matched_variables: list[str] = []
        required_assumptions: set[str] = set()
        rhetorical_strategy = ""
        if parsed_claim is not None:
            required_assumptions = set(getattr(parsed_claim, "mentioned_assumptions", [])) | set(
                getattr(parsed_claim, "implied_assumptions", [])
            )
            rhetorical_strategy = str(getattr(parsed_claim, "rhetorical_strategy", "")).strip()

        for raw_variable, raw_semantics in measurement_semantics.items():
            variable = str(raw_variable).strip()
            semantics = raw_semantics if isinstance(raw_semantics, dict) else {}
            if not variable or not re.search(rf"\b{re.escape(variable)}\b", claim, flags=re.IGNORECASE):
                continue
            matched_variables.append(variable)
            supports.extend(
                str(item)
                for item in semantics.get("supports_assumptions", [])
                if str(item).strip()
            )
            contradicts.extend(
                str(item)
                for item in semantics.get("contradicts_assumptions", [])
                if str(item).strip()
            )

        claimed_adjuster = self._extract_named_variable(
            claim,
            list(measurement_semantics),
            patterns=(
                r"\bafter controlling for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bcontrolling for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\badjusting for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bonce (?P<name>[A-Za-z][A-Za-z0-9_]*) is included\b",
                r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*) is the only adjustment needed\b",
                r"\busing (?P<name>[A-Za-z][A-Za-z0-9_]*) as the adjustment set\b",
            ),
        )
        if claimed_adjuster is not None:
            matched_variables.append(claimed_adjuster)
            claimed_view = self._measurement_view(scenario, claimed_adjuster)
            if (
                rhetorical_strategy == "adjustment_sufficiency_assertion"
                and claimed_view is not None
                and claimed_view != "adjustment_covariate"
            ):
                contradicts.append("valid adjustment set")

        if not matched_variables and required_assumptions:
            for assumption in required_assumptions:
                owners = [
                    str(raw_variable).strip()
                    for raw_variable, raw_semantics in measurement_semantics.items()
                    if isinstance(raw_semantics, dict)
                    and assumption in {
                        str(item).strip()
                        for item in raw_semantics.get("supports_assumptions", [])
                        if str(item).strip()
                    }
                ]
                if len(owners) == 1:
                    matched_variables.append(owners[0])
                    supports.append(assumption)

        supports = self._deduplicate(supports)
        contradicts = self._deduplicate(contradicts)
        matched_variables = self._deduplicate(matched_variables)
        if not supports and not contradicts:
            return None
        return ToolExecutionResult(
            tool_name="public_semantics_check",
            success=True,
            output={
                "matched_variables": matched_variables,
                "supports_assumptions": supports,
                "contradicts_assumptions": contradicts,
            },
        )

    def _measurement_semantics(
        self,
        scenario: PublicCausalInstance | None,
    ) -> dict[str, dict[str, Any]]:
        if scenario is None:
            return {}
        metadata = dict(getattr(scenario, "metadata", {}) or {})
        raw_semantics = metadata.get("measurement_semantics")
        if not isinstance(raw_semantics, dict):
            return {}
        semantics: dict[str, dict[str, Any]] = {}
        for raw_variable, raw_value in raw_semantics.items():
            variable = str(raw_variable).strip()
            if not variable or not isinstance(raw_value, dict):
                continue
            semantics[variable] = dict(raw_value)
        return semantics

    def _measurement_view(
        self,
        scenario: PublicCausalInstance | None,
        variable: str | None,
    ) -> str | None:
        if scenario is None or variable is None:
            return None
        semantics = self._measurement_semantics(scenario).get(str(variable).strip(), {})
        measurement_view = semantics.get("measurement_view")
        if measurement_view is None:
            return None
        normalized = str(measurement_view).strip()
        return normalized or None

    def _build_tool_kwargs(
        self,
        tool_name: str,
        scenario,
        claim: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        data = self._get_data(scenario)
        graph = self._get_graph(context, scenario=scenario)
        variables = self._get_variables(scenario, data)
        treatment, outcome = self._infer_focus_variables(claim, variables, scenario=scenario, context=context)
        conditioning = list(context.get("adjustment_set") or self._get_observed_controls(graph, data, treatment, outcome))
        instrument = context.get("instrument") or self._guess_instrument(graph, data, variables, treatment, outcome)
        mediator = context.get("mediator") or self._guess_mediator(graph, data, variables, treatment, outcome)
        proxy = context.get("proxy")
        evidence = context.get("evidence")
        if evidence is None and data is not None and not data.empty:
            evidence = data.iloc[0].to_dict()
        intervention = context.get("intervention")
        if intervention is None and evidence is not None:
            intervention = {treatment: self._flip_value(evidence.get(treatment, 0))}
        scm = self._get_scm(context, scenario=scenario)

        if tool_name in {"correlation_analysis", "compute_correlation"}:
            return {"data": data, "x": treatment, "y": outcome} if data is not None else None

        if tool_name == "conditional_independence_test":
            return {"data": data, "x": treatment, "y": outcome, "z": conditioning[:1]} if data is not None else None

        if tool_name == "detect_simpson_paradox":
            return {"data": data, "x": treatment, "y": outcome, "z": conditioning[0]} if data is not None and conditioning else None

        if tool_name == "partial_correlation":
            return {"data": data, "x": treatment, "y": outcome, "controls": conditioning[:1]} if data is not None else None

        if tool_name == "proxy_support_check":
            return {
                "data": data,
                "treatment": treatment,
                "outcome": outcome,
                "proxy": proxy,
                "controls": conditioning[:1],
            } if data is not None and proxy is not None else None

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
            iv_covariates = (
                [
                    column
                    for column in conditioning
                    if column not in {treatment, outcome, instrument}
                ][:1]
                or [
                    column
                    for column in variables
                    if column not in {treatment, outcome, instrument}
                ][:1]
            )
            return {
                "data": data,
                "instrument": instrument,
                "treatment": treatment,
                "outcome": outcome,
                "covariates": iv_covariates,
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

        if tool_name == "overlap_check":
            return {
                "data": data,
                "treatment": treatment,
                "covariates": conditioning[:2] or [column for column in variables if column not in {treatment, outcome}][:2],
            } if data is not None else None

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

        if tool_name == "counterfactual_bridge_check":
            return {
                "data": data,
                "treatment": treatment,
                "mediator": mediator,
                "outcome": outcome,
                "covariates": conditioning[:2] or [
                    column for column in variables if column not in {treatment, outcome, mediator}
                ][:2],
            } if data is not None and mediator is not None else None

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

    def _summarize_tool_output(self, result: ToolExecutionResult) -> str:
        if not result.success:
            return result.error or f"{result.tool_name} was skipped because the public context was insufficient."

        output = result.output
        if isinstance(output, dict):
            if "matched_variables" in output:
                joined = ", ".join(str(item) for item in output["matched_variables"])
                return f"{result.tool_name} matched public semantics for {joined}"
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
            if "supports_adjustment_set" in output:
                return f"{result.tool_name} adjustment_support={output['supports_adjustment_set']}"
            if "is_strong_instrument" in output:
                return f"{result.tool_name} strong_instrument={output['is_strong_instrument']}"
            if "has_overlap" in output:
                return f"{result.tool_name} overlap={output['has_overlap']}"
            if "supports_proxy_sufficiency" in output:
                return f"{result.tool_name} proxy_support={output['supports_proxy_sufficiency']}"
            if "supports_counterfactual_model_uniqueness" in output:
                return (
                    f"{result.tool_name} counterfactual_bridge="
                    f"{output['supports_counterfactual_model_uniqueness']}"
                )
            if "significant" in output:
                return f"{result.tool_name} significant={output['significant']}"
        if isinstance(output, list):
            return f"{result.tool_name} returned {len(output)} records"
        return f"{result.tool_name} completed"

    def _extract_tool_assumptions(
        self,
        result: ToolExecutionResult,
        *,
        claim: str,
        parsed_claim: Any | None = None,
    ) -> tuple[list[str], list[str]]:
        supports: list[str] = []
        contradicts: list[str] = []
        output = result.output if isinstance(result.output, dict) else None
        claim_lower = claim.lower()
        required_assumptions: set[str] = set()
        rhetorical_strategy = ""
        query_type = ""
        if parsed_claim is not None:
            required_assumptions = set(getattr(parsed_claim, "mentioned_assumptions", [])) | set(
                getattr(parsed_claim, "implied_assumptions", [])
            )
            rhetorical_strategy = str(getattr(parsed_claim, "rhetorical_strategy", ""))
            query_type_value = getattr(parsed_claim, "query_type", None)
            query_type = getattr(query_type_value, "value", str(query_type_value or "")).strip().lower()
        if output is not None:
            supports.extend(
                str(item)
                for item in output.get("supports_assumptions", [])
                if str(item).strip()
            )
            contradicts.extend(
                str(item)
                for item in output.get("contradicts_assumptions", [])
                if str(item).strip()
            )

        conservative_adjustment_claim = any(
            phrase in claim_lower
            for phrase in (
                "after controlling for",
                "controlling for",
                "control for",
                "adjusting for",
                "backdoor",
            )
        )
        aggressive_adjustment_claim = (
            "only adjustment needed" in claim_lower
            or bool(re.search(r"\bonce [A-Za-z][A-Za-z0-9_]* is included\b", claim, flags=re.IGNORECASE))
        )
        is_iv_claim = bool(
            {"instrument relevance", "exclusion restriction", "instrument independence"} & required_assumptions
        ) or rhetorical_strategy == "instrumental_variable_appeal"
        is_proxy_claim = "proxy sufficiency" in required_assumptions
        is_counterfactual_claim = query_type == "counterfactual"
        is_counterfactual_overclaim = rhetorical_strategy in {"false_uniqueness", "counterfactual_certainty"}

        if result.tool_name in {"backdoor_adjustment", "backdoor_adjustment_check"} and output is not None:
            if output.get("is_valid_adjustment") is True:
                supports.append("valid adjustment set")
            elif output.get("is_valid_adjustment") is False:
                contradicts.append("valid adjustment set")
            elif conservative_adjustment_claim and output.get("supports_adjustment_set") is True:
                supports.append("valid adjustment set")
            elif (
                (conservative_adjustment_claim or aggressive_adjustment_claim)
                and output.get("supports_adjustment_set") is False
                and output.get("adjustment_set")
            ):
                contradicts.append("valid adjustment set")

        if result.tool_name == "iv_estimation" and output is not None and is_iv_claim:
            if output.get("is_strong_instrument") is True:
                supports.append("instrument relevance")
            elif output.get("is_strong_instrument") is False:
                contradicts.append("instrument relevance")
            if output.get("supports_exclusion_restriction") is True:
                supports.append("exclusion restriction")
            if output.get("supports_instrument_independence") is True:
                supports.append("instrument independence")

        if result.tool_name == "overlap_check" and output is not None:
            if output.get("has_overlap") is True:
                supports.append("positivity")
            elif output.get("has_overlap") is False:
                contradicts.append("positivity")

        if result.tool_name == "proxy_support_check" and output is not None and is_proxy_claim:
            if output.get("supports_proxy_sufficiency") is True:
                supports.append("proxy sufficiency")
            elif output.get("supports_proxy_sufficiency") is False:
                contradicts.append("proxy sufficiency")

        if (
            result.tool_name == "counterfactual_bridge_check"
            and output is not None
            and is_counterfactual_claim
            and not is_counterfactual_overclaim
        ):
            if output.get("supports_stable_mediation") is True:
                supports.append("stable mediation structure")

            if output.get("supports_cross_world_consistency") is True:
                supports.append("cross-world consistency")

            if output.get("supports_counterfactual_model_uniqueness") is True:
                supports.append("counterfactual model uniqueness")

        if result.tool_name == "scm_identification_test" and isinstance(result.output, list):
            indistinguishable = [
                item for item in result.output
                if isinstance(item, dict) and not item.get("distinguishable", True)
            ]
            if indistinguishable:
                contradicts.append("counterfactual model uniqueness")
            else:
                supports.append("counterfactual model uniqueness")

        return self._deduplicate(supports), self._deduplicate(contradicts)

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

    def _deduplicate(self, items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result

    def _get_data(self, scenario) -> pd.DataFrame | None:
        if hasattr(scenario, "data") and isinstance(scenario.data, pd.DataFrame):
            return scenario.data
        if hasattr(scenario, "observed_data") and isinstance(scenario.observed_data, pd.DataFrame):
            return scenario.observed_data
        return None

    def _fit_public_linear_scm(
        self,
        data: pd.DataFrame | None,
        graph: nx.DiGraph | None,
    ) -> dict[str, Any] | None:
        if data is None or data.empty or graph is None or graph.number_of_edges() == 0:
            return None

        numeric = data.copy(deep=True)
        for column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

        coefficients: dict[str, dict[str, float]] = {}
        for node in graph.nodes:
            if node not in numeric.columns:
                continue
            parents = [parent for parent in graph.predecessors(node) if parent in numeric.columns]
            frame = numeric.loc[:, [*parents, node]].dropna()
            if frame.empty:
                series = numeric[node].dropna()
                coefficients[node] = {
                    "intercept": float(series.mean()) if not series.empty else 0.0,
                }
                continue

            outcome = frame[node].to_numpy(dtype=float)
            if not parents:
                coefficients[node] = {"intercept": float(np.mean(outcome))}
                continue

            design = np.column_stack(
                [
                    np.ones(len(frame), dtype=float),
                    frame.loc[:, parents].to_numpy(dtype=float),
                ]
            )
            try:
                weights, *_ = np.linalg.lstsq(design, outcome, rcond=None)
            except np.linalg.LinAlgError:
                coefficients[node] = {"intercept": float(np.mean(outcome))}
                continue

            spec = {"intercept": float(weights[0])}
            for parent, weight in zip(parents, weights[1:]):
                spec[parent] = float(weight)
            coefficients[node] = spec

        if not coefficients:
            return None

        return {
            "name": "public_linear_scm",
            "schema_view": "public",
            "graph": graph,
            "coefficients": coefficients,
        }

    def _get_graph(
        self,
        context: dict[str, Any],
        *,
        scenario: PublicCausalInstance | None = None,
    ) -> nx.DiGraph | None:
        if scenario is None:
            return None

        public_context = dict(context or {})
        data = self._get_data(scenario)
        variables = self._get_variables(scenario, data)
        treatment = str(public_context.get("treatment", "")).strip()
        outcome = str(public_context.get("outcome", "")).strip()
        claim_text = str(public_context.get("_claim_text", "") or "")
        selection_mechanism = str(public_context.get("selection_mechanism", "") or "").strip().lower()
        scenario_level = str(getattr(scenario, "causal_level", "") or "").strip().upper()
        if scenario_level in {"3", "L3"}:
            scenario_level = "L3"
        allow_public_graph = bool(
            public_context.get("needs_full_counterfactual")
            or scenario_level == "L3"
        )
        if not allow_public_graph:
            return None
        if treatment not in variables or outcome not in variables or treatment == outcome:
            treatment, outcome = self._infer_focus_variables(
                claim_text,
                variables,
                scenario=scenario,
                context=public_context,
            )
        if treatment not in variables or outcome not in variables or treatment == outcome:
            return None

        graph = nx.DiGraph()
        graph.add_nodes_from(variables)
        graph.add_edge(treatment, outcome)

        adjustment_set = [
            column
            for column in list(public_context.get("adjustment_set") or [])
            if column in variables and column not in {treatment, outcome}
        ]
        for covariate in adjustment_set:
            graph.add_edge(covariate, treatment)
            graph.add_edge(covariate, outcome)

        proxy_candidates = [
            column
            for column in list(public_context.get("proxy_variables") or [])
            if column in variables and column not in {treatment, outcome}
        ]
        explicit_proxy = str(public_context.get("proxy", "")).strip()
        if explicit_proxy in variables and explicit_proxy not in {treatment, outcome}:
            proxy_candidates.append(explicit_proxy)
        for proxy in self._deduplicate(proxy_candidates):
            graph.add_edge(proxy, treatment)
            graph.add_edge(proxy, outcome)

        mediator = str(public_context.get("mediator", "")).strip()
        if not mediator and bool(public_context.get("has_mediator")):
            mediator = self._guess_mediator(None, data, variables, treatment, outcome) or ""
        if mediator in variables and mediator not in {treatment, outcome}:
            graph.add_edge(treatment, mediator)
            graph.add_edge(mediator, outcome)

        instrument = str(public_context.get("instrument", "")).strip()
        if instrument in variables and instrument not in {treatment, outcome}:
            graph.add_edge(instrument, treatment)

        selection = str(public_context.get("selection", "")).strip()
        if selection in variables and selection not in {treatment, outcome}:
            graph.add_edge(treatment, selection)
            graph.add_edge(outcome, selection)

        return graph if graph.number_of_edges() > 0 else None

    def _get_scm(
        self,
        context: dict[str, Any],
        *,
        scenario: PublicCausalInstance | None = None,
    ):
        if scenario is None:
            return None
        public_context = dict(context or {})
        scenario_level = str(getattr(scenario, "causal_level", "") or "").strip().upper()
        if scenario_level in {"3", "L3"}:
            scenario_level = "L3"
        if not (public_context.get("needs_full_counterfactual") or scenario_level == "L3"):
            return None

        graph = self._get_graph(public_context, scenario=scenario)
        data = self._get_data(scenario)
        return self._fit_public_linear_scm(data, graph)

    def _get_variables(self, scenario, data: pd.DataFrame | None) -> list[str]:
        variables = list(getattr(scenario, "variables", []))
        if not variables and data is not None:
            variables = list(data.columns)
        if not variables:
            raise ValueError("场景缺少变量信息")
        return variables

    def _infer_focus_variables(
        self,
        claim: str,
        variables: list[str],
        scenario=None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        if context is not None:
            treatment = str(context.get("treatment", "")).strip()
            outcome = str(context.get("outcome", "")).strip()
            if treatment in variables and outcome in variables and treatment != outcome:
                return treatment, outcome

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

    def _parse_claim(self, claim: str):
        try:
            return parse_claim(claim)
        except Exception:
            return None

    def _extract_named_variable(
        self,
        claim: str,
        variables: list[str],
        *,
        patterns: tuple[str, ...],
    ) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, claim, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = str(match.group("name")).strip()
            if candidate in variables:
                return candidate
        return None

    def _role_token_match(
        self,
        variables: list[str],
        claim: str,
        *,
        tokens: tuple[str, ...],
    ) -> str | None:
        claim_lower = claim.lower()
        for variable in variables:
            lowered = variable.lower()
            if lowered not in claim_lower:
                continue
            parts = [part for part in re.split(r"[_\W]+", lowered) if part]
            if any(token == part for token in tokens for part in parts):
                return variable
        return None

    def _infer_claim_hints(
        self,
        claim: str,
        variables: list[str],
        parsed_claim,
        *,
        scenario: PublicCausalInstance | None = None,
    ) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        if parsed_claim is not None:
            if parsed_claim.treatment in variables:
                hints["treatment"] = parsed_claim.treatment
            if parsed_claim.outcome in variables:
                hints["outcome"] = parsed_claim.outcome

        treatment = hints.get("treatment", "")
        outcome = hints.get("outcome", "")
        remaining = [variable for variable in variables if variable not in {treatment, outcome}]

        bridge = self._extract_named_variable(
            claim,
            remaining,
            patterns=(
                r"\bafter controlling for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bcontrolling for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\badjusting for (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bonce (?P<name>[A-Za-z][A-Za-z0-9_]*) is included\b",
                r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*) is the only adjustment needed\b",
                r"\busing (?P<name>[A-Za-z][A-Za-z0-9_]*) as the adjustment set\b",
                r"\busing proxy (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bwith (?P<name>[A-Za-z][A-Za-z0-9_]*) (?:available|observed|measured)\b",
                r"\busing (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
            ),
        )
        explicit_instrument = self._extract_named_variable(
            claim,
            remaining,
            patterns=(
                r"\busing (?P<name>[A-Za-z][A-Za-z0-9_]*) as an instrument\b",
                r"\bwith (?P<name>[A-Za-z][A-Za-z0-9_]*) as an instrument\b",
            ),
        )
        if explicit_instrument is None and re.search(
            r"\binstrument(?:al(?:-?variable)?)?\b|\binstrumental-variable\b|\biv\b",
            claim,
            flags=re.IGNORECASE,
        ):
            explicit_instrument = self._extract_named_variable(
                claim,
                remaining,
                patterns=(r"\busing (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",),
            )
        if explicit_instrument is not None:
            hints["instrument"] = explicit_instrument

        proxy = self._extract_named_variable(
            claim,
            remaining,
            patterns=(r"\bproxy (?P<name>[A-Za-z][A-Za-z0-9_]*)\b",),
        ) or self._role_token_match(
            remaining,
            claim,
            tokens=("proxy", "surrogate", "screening", "sensor", "archive"),
        )
        if proxy is not None:
            hints["proxy"] = proxy
            hints["proxy_variables"] = [proxy]

        mediator = self._role_token_match(
            remaining,
            claim,
            tokens=("mediator", "biomarker", "uptake", "engagement", "intermediate", "dosage", "mechanism"),
        )
        if mediator is not None:
            hints["mediator"] = mediator

        selection = self._role_token_match(
            remaining,
            claim,
            tokens=("selection", "screen", "record", "clinic", "audit", "portal"),
        )
        if selection is not None:
            hints["selection"] = selection
            hints["selection_variables"] = [selection]

        query_type = getattr(parsed_claim, "query_type", None)
        if bridge is not None and bridge not in {treatment, outcome}:
            bridge_view = self._measurement_view(scenario, bridge)
            if explicit_instrument is None and query_type is not None and str(query_type.value) == "counterfactual":
                hints.setdefault("mediator", bridge)
            elif bridge_view == "proxy_measurement":
                hints.setdefault("proxy", bridge)
                hints.setdefault("proxy_variables", [bridge])
            elif bridge_view == "sample_inclusion_indicator":
                hints.setdefault("selection", bridge)
                hints.setdefault("selection_variables", [bridge])
            elif bridge_view == "intermediate_measurement":
                hints.setdefault("mediator", bridge)
            elif bridge_view == "assignment_signal":
                hints.setdefault("instrument", bridge)
            elif bridge_view == "adjustment_covariate" and explicit_instrument is None and proxy is None and selection is None:
                hints.setdefault("adjustment_set", [bridge])
            elif bridge_view is None and explicit_instrument is None and proxy is None and selection is None:
                hints.setdefault("adjustment_set", [bridge])

        if explicit_instrument is not None:
            hints["has_instrument"] = True
        if "mediator" in hints:
            hints["has_mediator"] = True
        if "proxy" in hints:
            hints["has_proxy"] = True
        if "selection" in hints:
            hints["has_selection"] = True

        return hints

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

    def _merge_context(
        self,
        context: dict[str, Any] | None,
        claim: str,
        *,
        scenario: PublicCausalInstance | None = None,
    ) -> dict[str, Any]:
        merged = dict(context or {})
        merged.setdefault("_claim_text", claim)
        lowered = claim.lower()
        data = self._get_data(scenario) if scenario is not None else None
        variables = self._get_variables(scenario, data) if scenario is not None else []
        parsed_claim = merged.get("_parsed_claim") or self._parse_claim(claim)
        if parsed_claim is not None:
            merged["_parsed_claim"] = parsed_claim
        if variables:
            for key, value in self._infer_claim_hints(
                claim,
                variables,
                parsed_claim,
                scenario=scenario,
            ).items():
                merged.setdefault(key, value)
        if parsed_claim is not None:
            if getattr(parsed_claim, "treatment", ""):
                merged.setdefault("treatment", parsed_claim.treatment)
            if getattr(parsed_claim, "outcome", ""):
                merged.setdefault("outcome", parsed_claim.outcome)
        if scenario is not None:
            if getattr(scenario, "proxy_variables", None):
                merged.setdefault("proxy_variables", list(scenario.proxy_variables))
                merged.setdefault("proxy", scenario.proxy_variables[0])
            if getattr(scenario, "selection_mechanism", None):
                merged.setdefault("selection_mechanism", scenario.selection_mechanism)
        has_iv_signal = bool(
            re.search(r"\biv\b|\binstrument(?:al(?:-?variable)?)?\b|\binstrumental-variable\b|\bquarter\b", lowered, flags=re.IGNORECASE)
        ) or any(token in lowered for token in ["工具变量", "出生季度"])
        merged.setdefault(
            "has_instrument",
            bool(merged.get("instrument")) or has_iv_signal,
        )
        merged.setdefault(
            "has_mediator",
            bool(merged.get("mediator")) or any(token in lowered for token in ["mediator", "中介", "frontdoor", "前门"]),
        )
        merged.setdefault(
            "has_proxy",
            bool(merged.get("proxy"))
            or bool(merged.get("proxy_variables"))
            or any(token in lowered for token in ["proxy", "surrogate", "代理"]),
        )
        merged.setdefault(
            "has_selection",
            bool(merged.get("selection"))
            or (
                str(merged.get("selection_mechanism", "") or "").strip().lower()
                not in {"", "none"}
            )
            or any(token in lowered for token in ["selection", "collider", "选择偏差"]),
        )
        merged.setdefault(
            "needs_full_counterfactual",
            any(token in lowered for token in ["counterfactual", "反事实", "abduction"]),
        )
        if merged["has_instrument"]:
            merged.setdefault("scenario_type", "instrument")
        elif merged["has_mediator"]:
            merged.setdefault("scenario_type", "mediator")
        elif merged["has_proxy"]:
            merged.setdefault("scenario_type", "proxy")
        else:
            merged.setdefault("scenario_type", "default")
        merged["has_public_graph"] = bool(self._get_graph(merged, scenario=scenario))
        merged["has_public_scm"] = self._get_scm(merged, scenario=scenario) is not None
        merged.setdefault("claim_stance", self._infer_claim_stance(claim))
        return merged
