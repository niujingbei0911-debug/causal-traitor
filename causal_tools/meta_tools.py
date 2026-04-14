"""
元工具 - 工具选择器与工具注册表
根据因果层级和场景自动选择合适的因果工具
"""
from typing import Callable

# 工具注册表
TOOL_REGISTRY: dict[str, dict] = {
    "L1": {
        "correlation": "causal_tools.l1_association.compute_correlation",
        "ci_test": "causal_tools.l1_association.conditional_independence_test",
        "simpson": "causal_tools.l1_association.detect_simpson_paradox",
        "partial_corr": "causal_tools.l1_association.partial_correlation",
    },
    "L2": {
        "backdoor": "causal_tools.l2_intervention.backdoor_adjustment",
        "frontdoor": "causal_tools.l2_intervention.frontdoor_adjustment",
        "iv": "causal_tools.l2_intervention.iv_estimation",
        "psm": "causal_tools.l2_intervention.propensity_score_matching",
    },
    "L3": {
        "counterfactual": "causal_tools.l3_counterfactual.counterfactual_inference",
        "sensitivity": "causal_tools.l3_counterfactual.sensitivity_analysis",
        "nde": "causal_tools.l3_counterfactual.natural_direct_effect",
        "pn": "causal_tools.l3_counterfactual.probability_of_necessity",
        "ps": "causal_tools.l3_counterfactual.probability_of_sufficiency",
    },
}


class ToolSelector:
    """根据场景和层级自动选择因果工具"""

    def select(self, level: int, scenario_type: str) -> list[str]:
        """返回推荐的工具列表"""
        raise NotImplementedError

    def get_tool(self, tool_name: str) -> Callable:
        """根据名称获取工具函数"""
        raise NotImplementedError
