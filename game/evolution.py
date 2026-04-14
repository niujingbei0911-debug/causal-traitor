"""
进化追踪器 - 记录和分析多轮博弈中的策略演化
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from collections import Counter

import numpy as np


class StrategyType(Enum):
    """策略类型枚举 — 与 AgentA._choose_strategy 保持一致"""
    # L1 关联层
    CONFOUNDING = "confounding"                        # 混杂因子注入
    COLLIDER = "collider"                              # 对撞因子伪装
    SELECTION_BIAS = "selection_bias"                   # 选择偏差
    # L2 干预层
    MEDIATOR_HIDE = "mediator_hide"                    # 中介变量隐藏
    INSTRUMENT_MISUSE = "instrument_misuse"            # 工具变量误用
    BACKDOOR_EXPLOIT = "backdoor_exploit"              # 后门路径利用
    FRONTDOOR_BLOCK = "frontdoor_block"                # 前门路径阻断
    # L3 反事实层
    REVERSE_CAUSE = "reverse_cause"                    # 因果方向反转
    SCM_MANIPULATION = "scm_manipulation"              # SCM 结构篡改
    COUNTERFACTUAL_DISTORTION = "counterfactual_distortion"  # 反事实扭曲
    NONCOMPLIANCE_EXPLOIT = "noncompliance_exploit"    # 不依从性利用


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
        self.max_history = int(self.config.get("max_history", 100))

    def record_round(self, record: StrategyRecord) -> None:
        """记录一轮博弈结果"""
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]

    def take_snapshot(self, round_id: int) -> EvolutionSnapshot:
        """生成当前演化快照"""
        if not self.records:
            snapshot = EvolutionSnapshot(
                round_id=round_id,
                strategy_distribution={},
                avg_deception_rate=0.0,
                avg_detection_rate=0.0,
                difficulty_level=0.0,
            )
            self.snapshots.append(snapshot)
            return snapshot

        counts = Counter(record.strategy_type.value for record in self.records)
        total = len(self.records)
        distribution = {name: count / total for name, count in counts.items()}

        # 使用当前轮次的值（而非累积均值），让图表反映逐轮变化
        current = self.records[-1]
        cur_deception = float(current.deception_score)
        cur_detection = float(current.detection_score)
        cur_difficulty = float(current.details.get("difficulty", 0.0))

        snapshot = EvolutionSnapshot(
            round_id=round_id,
            strategy_distribution=distribution,
            avg_deception_rate=cur_deception,
            avg_detection_rate=cur_detection,
            difficulty_level=cur_difficulty,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_strategy_trend(self, window: int = 10) -> Dict[str, List[float]]:
        """获取策略使用趋势（滑动窗口）"""
        if not self.records:
            return {}

        window = max(1, window)
        strategy_names = [strategy.value for strategy in StrategyType]
        trend = {name: [] for name in strategy_names}

        for end in range(1, len(self.records) + 1):
            segment = self.records[max(0, end - window) : end]
            counts = Counter(record.strategy_type.value for record in segment)
            segment_total = len(segment)
            for name in strategy_names:
                trend[name].append(counts.get(name, 0) / segment_total)

        return trend

    def detect_convergence(self, threshold: float = 0.05) -> bool:
        """检测策略是否收敛"""
        if len(self.records) < 8:
            return False

        half = len(self.records) // 2
        first = Counter(record.strategy_type.value for record in self.records[:half])
        second = Counter(record.strategy_type.value for record in self.records[half:])
        first_total = max(1, sum(first.values()))
        second_total = max(1, sum(second.values()))

        distance = 0.0
        for name in {strategy.value for strategy in StrategyType}:
            distance += abs(first.get(name, 0) / first_total - second.get(name, 0) / second_total)

        return distance / 2.0 <= threshold

    def get_arms_race_index(self) -> float:
        """计算军备竞赛指数（攻防能力同步提升程度）"""
        if len(self.records) < 3:
            return 0.0

        deception = np.array([record.deception_score for record in self.records], dtype=float)
        detection = np.array([record.detection_score for record in self.records], dtype=float)
        deception_diff = np.diff(deception)
        detection_diff = np.diff(detection)
        if np.allclose(deception_diff.std(), 0.0) or np.allclose(detection_diff.std(), 0.0):
            return float(np.clip(np.mean(np.sign(deception_diff) == np.sign(detection_diff)), 0.0, 1.0))

        correlation = float(np.corrcoef(deception_diff, detection_diff)[0, 1])
        if np.isnan(correlation):
            return 0.0
        return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

    def get_deception_complexity_trend(self, window: int = 5) -> List[float]:
        """欺骗复杂度趋势 — 设计文档要求的指标。

        使用策略的因果层级作为复杂度代理：L1=1, L2=2, L3=3，
        返回滑动窗口内的平均复杂度序列。
        """
        if not self.records:
            return []

        level_map = {
            "confounding": 1, "collider": 1, "selection_bias": 1,
            "mediator_hide": 2, "instrument_misuse": 2, "backdoor_exploit": 2, "frontdoor_block": 2,
            "reverse_cause": 3, "scm_manipulation": 3, "counterfactual_distortion": 3, "noncompliance_exploit": 3,
        }
        complexities = [level_map.get(r.strategy_type.value, 1) for r in self.records]
        trend = []
        for end in range(1, len(complexities) + 1):
            seg = complexities[max(0, end - window):end]
            trend.append(sum(seg) / len(seg))
        return trend

    def get_detection_sensitivity_trend(self, window: int = 5) -> List[float]:
        """检测灵敏度趋势 — 设计文档要求的指标。

        返回滑动窗口内的平均检测得分序列。
        """
        if not self.records:
            return []

        scores = [r.detection_score for r in self.records]
        trend = []
        for end in range(1, len(scores) + 1):
            seg = scores[max(0, end - window):end]
            trend.append(sum(seg) / len(seg))
        return trend

    def get_nash_convergence(self) -> float:
        """纳什均衡收敛度 — 设计文档要求的指标。

        衡量攻防双方策略是否趋于稳定均衡。
        使用后半段 vs 前半段的策略分布变化量，值越小越接近均衡。
        返回 0~1，1 表示完全收敛。
        """
        if len(self.records) < 6:
            return 0.0

        half = len(self.records) // 2
        first = Counter(r.strategy_type.value for r in self.records[:half])
        second = Counter(r.strategy_type.value for r in self.records[half:])
        first_total = max(1, sum(first.values()))
        second_total = max(1, sum(second.values()))

        all_strategies = {s.value for s in StrategyType}
        distance = sum(
            abs(first.get(s, 0) / first_total - second.get(s, 0) / second_total)
            for s in all_strategies
        )
        # TV distance ∈ [0, 2], 归一化到 [0, 1] 后取反
        return float(np.clip(1.0 - distance / 2.0, 0.0, 1.0))

    def export_history(self) -> Dict[str, Any]:
        """导出完整演化历史"""
        return {
            "records": [
                {
                    "round_id": record.round_id,
                    "strategy_type": record.strategy_type.value,
                    "success": record.success,
                    "deception_score": record.deception_score,
                    "detection_score": record.detection_score,
                    "details": record.details,
                }
                for record in self.records
            ],
            "snapshots": [
                {
                    "round_id": snapshot.round_id,
                    "strategy_distribution": snapshot.strategy_distribution,
                    "avg_deception_rate": snapshot.avg_deception_rate,
                    "avg_detection_rate": snapshot.avg_detection_rate,
                    "difficulty_level": snapshot.difficulty_level,
                }
                for snapshot in self.snapshots
            ],
            "arms_race_index": self.get_arms_race_index(),
            "deception_complexity_trend": self.get_deception_complexity_trend(),
            "detection_sensitivity_trend": self.get_detection_sensitivity_trend(),
            "nash_convergence": self.get_nash_convergence(),
            "converged": self.detect_convergence(),
        }
