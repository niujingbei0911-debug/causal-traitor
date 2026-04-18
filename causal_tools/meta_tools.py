"""Meta causal tools: registry, rule-based checks, and tool selection."""

from __future__ import annotations

import importlib
import re
from typing import Callable

import networkx as nx


TOOL_REGISTRY: dict[str, dict[str, str]] = {
    "L1": {
        "correlation": "causal_tools.l1_association.compute_correlation",
        "correlation_analysis": "causal_tools.l1_association.correlation_analysis",
        "ci_test": "causal_tools.l1_association.conditional_independence_test",
        "conditional_independence_test": "causal_tools.l1_association.conditional_independence_test",
        "simpson": "causal_tools.l1_association.detect_simpson_paradox",
        "detect_simpson_paradox": "causal_tools.l1_association.detect_simpson_paradox",
        "partial_corr": "causal_tools.l1_association.partial_correlation",
        "partial_correlation": "causal_tools.l1_association.partial_correlation",
        "proxy_support_check": "causal_tools.l1_association.proxy_support_check",
    },
    "L2": {
        "backdoor": "causal_tools.l2_intervention.backdoor_adjustment",
        "backdoor_adjustment": "causal_tools.l2_intervention.backdoor_adjustment",
        "backdoor_adjustment_check": "causal_tools.l2_intervention.backdoor_adjustment_check",
        "frontdoor": "causal_tools.l2_intervention.frontdoor_adjustment",
        "frontdoor_adjustment": "causal_tools.l2_intervention.frontdoor_adjustment",
        "frontdoor_estimation": "causal_tools.l2_intervention.frontdoor_estimation",
        "iv": "causal_tools.l2_intervention.iv_estimation",
        "iv_estimation": "causal_tools.l2_intervention.iv_estimation",
        "psm": "causal_tools.l2_intervention.propensity_score_matching",
        "propensity_score_matching": "causal_tools.l2_intervention.propensity_score_matching",
        "sensitivity_analysis": "causal_tools.l2_intervention.sensitivity_analysis",
        "overlap_check": "causal_tools.l2_intervention.overlap_check",
    },
    "L3": {
        "counterfactual": "causal_tools.l3_counterfactual.counterfactual_inference",
        "counterfactual_inference": "causal_tools.l3_counterfactual.counterfactual_inference",
        "sensitivity": "causal_tools.l3_counterfactual.sensitivity_analysis",
        "sensitivity_analysis": "causal_tools.l3_counterfactual.sensitivity_analysis",
        "nde": "causal_tools.l3_counterfactual.natural_direct_effect",
        "natural_direct_effect": "causal_tools.l3_counterfactual.natural_direct_effect",
        "pn": "causal_tools.l3_counterfactual.probability_of_necessity",
        "probability_of_necessity": "causal_tools.l3_counterfactual.probability_of_necessity",
        "ps": "causal_tools.l3_counterfactual.probability_of_sufficiency",
        "probability_of_sufficiency": "causal_tools.l3_counterfactual.probability_of_sufficiency",
        "scm_identification_test": "causal_tools.l3_counterfactual.scm_identification_test",
        "ett_computation": "causal_tools.l3_counterfactual.ett_computation",
        "abduction_action_prediction": "causal_tools.l3_counterfactual.abduction_action_prediction",
        "counterfactual_bridge_check": "causal_tools.l3_counterfactual.counterfactual_bridge_check",
    },
    "META": {
        "argument_logic_check": "causal_tools.meta_tools.argument_logic_check",
        "causal_graph_validator": "causal_tools.meta_tools.causal_graph_validator",
    },
}


def _has_iv_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(r"\biv\b|\binstrument(?:al variable)?\b|\bquarter\b", lowered, flags=re.IGNORECASE)
    ) or any(token in lowered for token in ("工具变量", "出生季度"))


CAUSAL_FALLACIES: dict[str, dict[str, list[str] | str]] = {
    "correlation_implies_causation": {
        "name": "Correlation As Causation",
        "keywords": [
            "prove",
            "proves",
            "correlation means causation",
            "相关就说明因果",
            "一定导致",
            "直接证明",
        ],
    },
    "post_hoc": {
        "name": "Post Hoc",
        "keywords": ["after", "afterward", "因为先发生", "之后所以", "先于"],
    },
    "simpsons_paradox": {
        "name": "Simpson's Paradox",
        "keywords": ["simpson", "辛普森", "subgroup", "子群", "分组后逆转", "aggregat"],
    },
    "berkson_bias": {
        "name": "Berkson Bias",
        "keywords": ["berkson", "伯克森", "selection bias", "选择偏差", "collider conditioning", "条件化对撞"],
    },
    "reverse_causation": {
        "name": "Reverse Causation",
        "keywords": ["reverse caus", "反向因果", "其实是结果导致原因"],
    },
    "ecological_fallacy": {
        "name": "Ecological Fallacy",
        "keywords": ["ecological", "生态谬误", "group level", "群体水平", "aggregate to individual", "群体推个体"],
    },
    "collider_bias": {
        "name": "Collider Bias",
        "keywords": ["collider", "对撞", "conditioning on collider", "对撞因子条件化", "endogenous selection"],
    },
    "ignoring_confounders": {
        "name": "Ignoring Confounders",
        "keywords": ["无需控制", "不需要调整", "without controlling", "ignoring confounders"],
    },
    "overclaiming_certainty": {
        "name": "Overclaiming Certainty",
        "keywords": ["100%", "certainly", "毫无疑问", "必然", "唯一解释"],
    },
}


def _to_level_name(level: int | str) -> str:
    if isinstance(level, int):
        return f"L{level}"
    normalized = str(level).strip().upper()
    return normalized if normalized.startswith("L") else f"L{normalized}"


def argument_logic_check(argument: str, claimed_causal_relation: dict | None = None) -> dict:
    """Rule-based scan for common causal fallacies."""

    text = argument.lower()
    detected: list[dict[str, object]] = []

    for fallacy_id, fallacy_info in CAUSAL_FALLACIES.items():
        keywords = [keyword.lower() for keyword in fallacy_info["keywords"]]
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            detected.append(
                {
                    "fallacy_id": fallacy_id,
                    "fallacy": fallacy_info["name"],
                    "evidence": matched,
                    "severity": "high" if len(matched) >= 2 else "medium",
                }
            )

    if claimed_causal_relation and claimed_causal_relation.get("certainty", 0.0) > 0.9:
        detected.append(
            {
                "fallacy_id": "unsupported_certainty",
                "fallacy": "Unsupported Certainty",
                "evidence": ["claim_confidence>0.9"],
                "severity": "medium",
            }
        )

    overall_logic_score = max(0.0, 10.0 - 2.0 * len(detected))
    return {
        "n_fallacies_detected": len(detected),
        "fallacies": detected,
        "detected_fallacies": [str(item["fallacy"]) for item in detected],
        "overall_logic_score": overall_logic_score,
        "normalized_score": overall_logic_score / 10.0,
        "recommendation": "No obvious causal fallacy detected." if not detected else f"Detected {len(detected)} potential causal fallacies.",
    }


def causal_graph_validator(graph: nx.DiGraph | dict) -> dict:
    """Validate basic structural properties of a causal graph."""

    if isinstance(graph, dict):
        dag = nx.DiGraph()
        for source, targets in graph.items():
            for target in targets:
                dag.add_edge(source, target)
    elif isinstance(graph, nx.DiGraph):
        dag = graph.copy()
    else:
        raise TypeError("graph must be an nx.DiGraph or adjacency dict")

    cycles = [list(cycle) for cycle in nx.simple_cycles(dag)]
    is_dag = len(cycles) == 0
    topological_order = list(nx.topological_sort(dag)) if is_dag else []
    roots = [node for node in dag.nodes if dag.in_degree(node) == 0]
    sinks = [node for node in dag.nodes if dag.out_degree(node) == 0]

    confounder_candidates = []
    for node in dag.nodes:
        children = list(dag.successors(node))
        if len(children) >= 2:
            confounder_candidates.append({"node": node, "children": children})

    return {
        "is_dag": is_dag,
        "has_cycles": not is_dag,
        "cycles": cycles,
        "topological_order": topological_order,
        "roots": roots,
        "sinks": sinks,
        "n_nodes": dag.number_of_nodes(),
        "n_edges": dag.number_of_edges(),
        "confounder_candidates": confounder_candidates,
    }


class ToolSelector:
    """Choose tools based on level, scenario type, and public availability."""

    def select(
        self,
        level: int | str,
        scenario_type: str,
        claim: str | None = None,
        context: dict | None = None,
    ) -> list[str]:
        context = context or {}
        level_name = _to_level_name(level)
        scenario_text = f"{scenario_type} {claim or ''}".lower()
        has_public_graph = bool(context.get("has_public_graph"))
        has_public_scm = bool(context.get("has_public_scm"))
        has_proxy = bool(context.get("has_proxy"))
        has_mediator = bool(context.get("has_mediator"))

        tools = ["correlation_analysis", "argument_logic_check"]
        if has_public_graph:
            tools.append("causal_graph_validator")

        if level_name == "L1":
            tools.extend(["conditional_independence_test", "partial_correlation"])
            if has_proxy:
                tools.append("proxy_support_check")
            if any(token in scenario_text for token in ["group", "subgroup", "strat", "辛普森", "simpson"]):
                tools.append("detect_simpson_paradox")

        elif level_name == "L2":
            tools.extend(["conditional_independence_test", "backdoor_adjustment_check", "sensitivity_analysis", "overlap_check"])
            if has_public_graph:
                tools.append("backdoor_adjustment")
            if _has_iv_signal(scenario_text) or context.get("has_instrument"):
                tools.append("iv_estimation")
            if has_proxy:
                tools.append("proxy_support_check")
            if any(token in scenario_text for token in ["mediator", "中介", "frontdoor", "前门"]) or has_mediator:
                tools.append("frontdoor_adjustment" if has_public_graph else "frontdoor_estimation")
            if context.get("needs_matching"):
                tools.append("propensity_score_matching")

        elif level_name == "L3":
            tools.extend(["backdoor_adjustment_check", "sensitivity_analysis", "overlap_check"])
            if has_public_scm:
                tools.extend(["counterfactual_inference", "scm_identification_test", "ett_computation"])
            if has_mediator:
                tools.append("counterfactual_bridge_check")
            if has_public_graph and has_mediator:
                tools.append("natural_direct_effect")
            if (context.get("needs_full_counterfactual") or any(token in scenario_text for token in ["abduction", "counterfactual", "反事实"])) and has_public_scm:
                tools.append("abduction_action_prediction")

        deduplicated: list[str] = []
        for tool in tools:
            if tool not in deduplicated:
                deduplicated.append(tool)
        return deduplicated

    def get_tool(self, tool_name: str) -> Callable:
        """Return the callable registered for one tool name."""

        tool_path = None
        for section in TOOL_REGISTRY.values():
            if tool_name in section:
                tool_path = section[tool_name]
                break

        if tool_path is None:
            if "." not in tool_name:
                raise KeyError(f"Unknown tool: {tool_name}")
            tool_path = tool_name

        module_path, function_name = tool_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)


def select_tools(causal_level: int | str, debate_context: dict | None = None) -> list[str]:
    """Functional wrapper kept for compatibility with older call sites."""

    debate_context = debate_context or {}
    selector = ToolSelector()
    return selector.select(
        causal_level,
        scenario_type=debate_context.get("scenario_type", ""),
        claim=debate_context.get("claim"),
        context=debate_context,
    )
