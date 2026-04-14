"""
实验追踪器 - 记录实验过程，支持W&B集成
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_id: str
    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    实验追踪器
    - 记录实验配置、指标、产物
    - 支持本地JSON日志
    - 可选W&B集成
    """

    def __init__(self, config: ExperimentConfig, log_dir: str = "logs", use_wandb: bool = False):
        self.config = config
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def init(self) -> None:
        """初始化追踪器（创建日志目录，可选初始化W&B）"""
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录一组指标"""
        raise NotImplementedError

    def log_artifact(self, name: str, data: Any, artifact_type: str = "json") -> None:
        """记录产物（模型、数据、图表等）"""
        raise NotImplementedError

    def log_round(self, round_id: int, round_data: Dict[str, Any]) -> None:
        """记录单轮博弈数据"""
        raise NotImplementedError

    def save_checkpoint(self, game_state: Dict[str, Any]) -> str:
        """保存游戏状态检查点，返回路径"""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        raise NotImplementedError

    def finish(self) -> Dict[str, Any]:
        """结束追踪，生成摘要"""
        raise NotImplementedError
