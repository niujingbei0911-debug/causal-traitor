"""
评估指标 - 定义所有14个评估维度的计算方法
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class MetricResult:
    """单个指标的计算结果"""
    name: str
    value: float
    category: str  # "deception", "detection", "game", "causal"
    details: Optional[Dict[str, Any]] = None


class CausalMetrics:
    """
    因果推理评估指标集
    覆盖14个评估维度：
    - 欺骗类: DSR, CSI, HVP, SDS
    - 检测类: DAcc, FPR, TtD, TEff
    - 博弈类: GBI, NE_dist, ECI
    - 因果类: CRA, LTP, IAS
    """

    @staticmethod
    def deception_success_rate(n_success: int, n_total: int) -> MetricResult:
        """DSR: 欺骗成功率"""
        raise NotImplementedError

    @staticmethod
    def causal_sophistication_index(claims: List[Dict]) -> MetricResult:
        """CSI: 因果诡辩复杂度指数"""
        raise NotImplementedError

    @staticmethod
    def hidden_variable_plausibility(scores: List[float]) -> MetricResult:
        """HVP: 隐变量合理性评分"""
        raise NotImplementedError

    @staticmethod
    def strategy_diversity_score(strategies: List[str]) -> MetricResult:
        """SDS: 策略多样性得分"""
        raise NotImplementedError

    @staticmethod
    def detection_accuracy(y_true: List[int], y_pred: List[int]) -> MetricResult:
        """DAcc: 检测准确率"""
        raise NotImplementedError

    @staticmethod
    def false_positive_rate(y_true: List[int], y_pred: List[int]) -> MetricResult:
        """FPR: 误报率"""
        raise NotImplementedError

    @staticmethod
    def time_to_detection(round_detected: int, total_rounds: int) -> MetricResult:
        """TtD: 检测时间"""
        raise NotImplementedError

    @staticmethod
    def tool_efficiency(tools_used: List[str], tools_effective: List[str]) -> MetricResult:
        """TEff: 工具使用效率"""
        raise NotImplementedError

    @staticmethod
    def game_balance_index(deception_rate: float, target: float = 0.4) -> MetricResult:
        """GBI: 博弈平衡指数"""
        raise NotImplementedError

    @staticmethod
    def nash_equilibrium_distance(payoff_matrix: Any) -> MetricResult:
        """NE_dist: 纳什均衡距离"""
        raise NotImplementedError

    @staticmethod
    def evolution_complexity_index(history: List[Dict]) -> MetricResult:
        """ECI: 演化复杂度指数"""
        raise NotImplementedError

    @staticmethod
    def causal_reasoning_accuracy(predictions: List, ground_truths: List) -> MetricResult:
        """CRA: 因果推理准确度"""
        raise NotImplementedError

    @staticmethod
    def ladder_transition_performance(l1: float, l2: float, l3: float) -> MetricResult:
        """LTP: 因果阶梯跨层表现"""
        raise NotImplementedError

    @staticmethod
    def information_asymmetry_score(agent_a_info: Dict, agent_b_info: Dict) -> MetricResult:
        """IAS: 信息不对称利用得分"""
        raise NotImplementedError
