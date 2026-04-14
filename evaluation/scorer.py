"""
评分器 - 综合评分与排名
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from .metrics import MetricResult, CausalMetrics


@dataclass
class RoundScore:
    """单轮评分"""
    round_id: int
    agent_a_score: float
    agent_b_score: float
    agent_c_score: float
    jury_verdict: str
    metric_results: List[MetricResult] = field(default_factory=list)


@dataclass
class GameScore:
    """整局游戏评分"""
    game_id: str
    round_scores: List[RoundScore]
    final_scores: Dict[str, float]
    winner: str
    summary: Dict[str, Any] = field(default_factory=dict)


class Scorer:
    """
    综合评分器
    - 汇总各维度指标
    - 计算加权综合分
    - 生成评分报告
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "deception": 0.25,
            "detection": 0.25,
            "game": 0.25,
            "causal": 0.25,
        }
        self.metrics = CausalMetrics()

    def score_round(self, round_data: Dict[str, Any]) -> RoundScore:
        """对单轮博弈评分"""
        raise NotImplementedError

    def score_game(self, game_data: Dict[str, Any]) -> GameScore:
        """对整局游戏评分"""
        raise NotImplementedError

    def compute_weighted_score(self, metric_results: List[MetricResult]) -> float:
        """计算加权综合分"""
        raise NotImplementedError

    def generate_report(self, game_score: GameScore) -> Dict[str, Any]:
        """生成评分报告"""
        raise NotImplementedError
