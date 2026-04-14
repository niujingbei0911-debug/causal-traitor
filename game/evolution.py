"""
进化追踪器 - 记录和分析多轮博弈中的策略演化
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class StrategyType(Enum):
    """策略类型枚举"""
    CONFOUNDING = "confounding"          # 混杂因子注入
    COLLIDER = "collider"                # 对撞因子伪装
    SELECTION_BIAS = "selection_bias"    # 选择偏差
    MEDIATOR_HIDE = "mediator_hide"     # 中介变量隐藏
    REVERSE_CAUSE = "reverse_cause"     # 因果方向反转


@dataclass
class StrategyRecord:
    """单次策略使用记录"""
    round_id: int
    strategy_type: StrategyType
    success: bool
    deception_score: float
    detection_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionSnapshot:
    """某一时刻的演化快照"""
    round_id: int
    strategy_distribution: Dict[str, float]
    avg_deception_rate: float
    avg_detection_rate: float
    difficulty_level: float


class EvolutionTracker:
    """
    进化追踪器
    - 记录每轮策略使用和结果
    - 分析策略演化趋势
    - 检测策略收敛/振荡
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.records: List[StrategyRecord] = []
        self.snapshots: List[EvolutionSnapshot] = []

    def record_round(self, record: StrategyRecord) -> None:
        """记录一轮博弈结果"""
        raise NotImplementedError

    def take_snapshot(self, round_id: int) -> EvolutionSnapshot:
        """生成当前演化快照"""
        raise NotImplementedError

    def get_strategy_trend(self, window: int = 10) -> Dict[str, List[float]]:
        """获取策略使用趋势（滑动窗口）"""
        raise NotImplementedError

    def detect_convergence(self, threshold: float = 0.05) -> bool:
        """检测策略是否收敛"""
        raise NotImplementedError

    def get_arms_race_index(self) -> float:
        """计算军备竞赛指数（攻防能力同步提升程度）"""
        raise NotImplementedError

    def export_history(self) -> Dict[str, Any]:
        """导出完整演化历史"""
        raise NotImplementedError
