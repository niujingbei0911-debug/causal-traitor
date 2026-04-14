"""
可视化API - FastAPI + WebSocket 实时博弈可视化后端
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class GameEvent:
    """博弈事件（用于WebSocket推送）"""
    event_type: str  # "round_start", "claim", "detection", "verdict", "game_end"
    round_id: int
    data: Dict[str, Any]
    timestamp: Optional[float] = None


class VisualizationAPI:
    """
    可视化后端API
    - REST: 获取历史数据、实验结果
    - WebSocket: 实时推送博弈过程
    - 因果图渲染数据接口
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.app = None  # FastAPI instance, lazy init

    def create_app(self) -> Any:
        """创建FastAPI应用"""
        raise NotImplementedError

    def setup_routes(self) -> None:
        """注册REST路由"""
        raise NotImplementedError

    def setup_websocket(self) -> None:
        """注册WebSocket端点"""
        raise NotImplementedError

    async def broadcast_event(self, event: GameEvent) -> None:
        """向所有连接的客户端广播事件"""
        raise NotImplementedError

    def get_causal_graph_data(self, game_id: str) -> Dict[str, Any]:
        """获取因果图可视化数据（D3.js格式）"""
        raise NotImplementedError

    def get_evolution_chart_data(self, game_id: str) -> Dict[str, Any]:
        """获取演化趋势图数据"""
        raise NotImplementedError

    def get_metrics_dashboard_data(self, experiment_id: str) -> Dict[str, Any]:
        """获取指标仪表盘数据"""
        raise NotImplementedError
