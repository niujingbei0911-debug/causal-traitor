"""
元工具 - 工具选择器与工具注册表
根据因果层级和场景自动选择合适的因果工具
"""
from __future__ import annotations

import importlib
from typing import Callable

import networkx as nx

# 工具注册表
TOOL_REGISTRY: dict[str, dict] = {
    "L1": {
        "correlation": "causal_tools.l1_association.compute_correlation",
        "correlation_analysis": "causal_tools.l1_association.correlation_analysis",
        "ci_test": "causal_tools.l1_association.conditional_independence_test",
        "conditional_independence_test": "causal_tools.l1_association.conditional_independence_test",
        "simpson": "causal_tools.l1_association.detect_simpson_paradox",
        "detect_simpson_paradox": "causal_tools.l1_association.detect_simpson_paradox",
        "partial_corr": "causal_tools.l1_association.partial_correlation",
        "partial_correlation": "causal_tools.l1_association.partial_correlation",
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
    },
    "META": {
        "argument_logic_check": "causal_tools.meta_tools.argument_logic_check",
        "causal_graph_validator": "causal_tools.meta_tools.causal_graph_validator",
    },
}


CAUSAL_FALLACIES: dict[str, dict[str, list[str] | str]] = {
    "correlation_implies_causation": {
        "name": "相关即因果",
        "keywords": ["prove", "proves", "correlation means causation", "相关就说明因果", "一定导致", "直接证明"],
    },
    "post_hoc": {
        "name": "后此谬误",
        "keywords": ["after", "afterward", "因为先发生", "之后所以", "先于"],
    },
    "simpsons_paradox": {
        "name": "辛普森悖论",
        "keywords": ["simpson", "辛普森", "subgroup", "子群", "分组后逆转", "aggregat"],
    },
    "berkson_bias": {
        "name": "伯克森偏差",
        "keywords": ["berkson", "伯克森", "selection bias", "选择偏差", "collider conditioning", "条件化对撞"],
    },
    "reverse_causation": {
        "name": "逆因果",
        "keywords": ["reverse caus", "反向因果", "其实是结果导致原因"],
    },
    "ecological_fallacy": {
        "name": "生态谬误",
        "keywords": ["ecological", "生态谬误", "group level", "群体水平", "aggregate to individual", "群体推个体"],
    },
    "collider_bias": {
        "name": "对撞因子偏差",
        "keywords": ["collider", "对撞", "conditioning on collider", "对撞因子条件化", "endogenous selection"],
    },
    "ignoring_confounders": {
        "name": "忽略混杂因子",
        "keywords": ["无需控制", "不需要调整", "without controlling", "ignoring confounders"],
    },
    "overclaiming_certainty": {
        "name": "过度确定性",
        "keywords": ["100%", "certainly", "毫无疑问", "必然", "唯一解释"],
    },
}


def _to_level_name(level: int | str) -> str:
    if isinstance(level, int):
        return f"L{level}"
    normalized = str(level).strip().upper()
    return normalized if normalized.startswith("L") else f"L{normalized}"


def argument_logic_check(argument: str, claimed_causal_relation: dict | None = None) -> dict:
    """基于规则检测常见因果谬误。"""
    text = argument.lower()
    detected = []

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
                "fallacy": "证据不足却过度确信",
                "evidence": ["claim_confidence>0.9"],
                "severity": "medium",
            }
        )

    overall_logic_score = max(0.0, 10.0 - 2.0 * len(detected))
    return {
        "n_fallacies_detected": len(detected),
        "fallacies": detected,
        "detected_fallacies": [item["fallacy"] for item in detected],
        "overall_logic_score": overall_logic_score,
        "normalized_score": overall_logic_score / 10.0,
        "recommendation": "论证基本可靠" if not detected else f"发现 {len(detected)} 个可疑因果谬误",
    }


def causal_graph_validator(graph: nx.DiGraph | dict) -> dict:
    """验证因果图的基本结构合法性。"""
    if isinstance(graph, dict):
        dag = nx.DiGraph()
        for source, targets in graph.items():
            for target in targets:
                dag.add_edge(source, target)
    elif isinstance(graph, nx.DiGraph):
        dag = graph.copy()
    else:
        raise TypeError("graph 必须是 nx.DiGraph 或 adjacency dict")

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
    """根据场景和层级自动选择因果工具"""

    def select(
        self,
        level: int | str,
        scenario_type: str,
        claim: str | None = None,
        context: dict | None = None,
    ) -> list[str]:
        """返回推荐的工具列表"""
        context = context or {}
        level_name = _to_level_name(level)
        scenario_text = f"{scenario_type} {claim or ''}".lower()

        tools = ["correlation_analysis", "argument_logic_check", "causal_graph_validator"]

        if level_name == "L1":
            tools.extend(["conditional_independence_test", "partial_correlation"])
            if any(token in scenario_text for token in ["group", "subgroup", "strat", "辛普森", "simpson"]):
                tools.append("detect_simpson_paradox")

        elif level_name == "L2":
            tools.extend(
                [
                    "conditional_independence_test",
                    "backdoor_adjustment_check",
                    "sensitivity_analysis",
                ]
            )
            if any(token in scenario_text for token in ["iv", "instrument", "工具变量", "quarter", "出生季度"]) or context.get("has_instrument"):
                tools.append("iv_estimation")
            if any(token in scenario_text for token in ["mediator", "中介", "frontdoor", "前门"]) or context.get("has_mediator"):
                tools.append("frontdoor_estimation")
            if context.get("needs_matching"):
                tools.append("propensity_score_matching")

        elif level_name == "L3":
            tools.extend(
                [
                    "backdoor_adjustment_check",
                    "counterfactual_inference",
                    "scm_identification_test",
                    "ett_computation",
                    "probability_of_necessity",
                    "probability_of_sufficiency",
                ]
            )
            if context.get("needs_full_counterfactual") or any(token in scenario_text for token in ["abduction", "counterfactual", "反事实"]):
                tools.append("abduction_action_prediction")

        deduplicated = []
        for tool in tools:
            if tool not in deduplicated:
                deduplicated.append(tool)
        return deduplicated

    def get_tool(self, tool_name: str) -> Callable:
        """根据名称获取工具函数"""
        tool_path = None
        for section in TOOL_REGISTRY.values():
            if tool_name in section:
                tool_path = section[tool_name]
                break

        if tool_path is None:
            if "." not in tool_name:
                raise KeyError(f"未知工具: {tool_name}")
            tool_path = tool_name

        module_path, function_name = tool_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)


def select_tools(causal_level: int | str, debate_context: dict | None = None) -> list[str]:
    """设计文档中的函数式接口。"""
    debate_context = debate_context or {}
    selector = ToolSelector()
    return selector.select(
        causal_level,
        scenario_type=debate_context.get("scenario_type", ""),
        claim=debate_context.get("claim"),
        context=debate_context,
    )
