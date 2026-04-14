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
        avg_deception = float(np.mean([record.deception_score for record in self.records]))
        avg_detection = float(np.mean([record.detection_score for record in self.records]))
        difficulty_values = [
            float(record.details.get("difficulty", 0.0))
            for record in self.records
            if "difficulty" in record.details
        ]
        snapshot = EvolutionSnapshot(
            round_id=round_id,
            strategy_distribution=distribution,
            avg_deception_rate=avg_deception,
            avg_detection_rate=avg_detection,
            difficulty_level=float(np.mean(difficulty_values)) if difficulty_values else 0.0,
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
            "converged": self.detect_convergence(),
        }
