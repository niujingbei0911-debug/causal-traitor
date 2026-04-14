"""
可视化API - FastAPI + WebSocket 实时博弈可视化后端

REST 端点:
  GET /api/games/{game_id}/causal-graph   因果图 (D3.js 格式)
  GET /api/games/{game_id}/evolution       进化趋势图数据
  GET /api/experiments/{exp_id}/dashboard  指标仪表盘数据
  GET /api/health                          健康检查

WebSocket:
  /ws/game   实时博弈事件推送
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    """博弈事件（用于 WebSocket 推送）"""
    event_type: str  # "round_start", "claim", "detection", "verdict", "game_end"
    round_id: int
    data: Dict[str, Any]
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class _ConnectionManager:
    """管理活跃的 WebSocket 连接."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead: List[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)

    async def broadcast_except(self, message: str, sender: WebSocket) -> None:
        """广播给除 sender 之外的所有连接."""
        dead: List[WebSocket] = []
        for ws in self._active:
            if ws is sender:
                continue
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# In-memory store (lightweight, no DB dependency)
# ---------------------------------------------------------------------------

class _GameStore:
    """简易内存存储, 保存最近的游戏数据供 REST 查询."""

    def __init__(self) -> None:
        self.games: Dict[str, Dict[str, Any]] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, List[Dict[str, Any]]] = {}

    # -- games --
    def upsert_game(self, game_id: str, data: Dict[str, Any]) -> None:
        self.games[game_id] = data

    def get_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        return self.games.get(game_id)

    # -- events --
    def append_event(self, game_id: str, event: GameEvent) -> None:
        self.events.setdefault(game_id, []).append(asdict(event))

    def get_events(self, game_id: str) -> List[Dict[str, Any]]:
        return self.events.get(game_id, [])

    # -- experiments --
    def upsert_experiment(self, exp_id: str, data: Dict[str, Any]) -> None:
        self.experiments[exp_id] = data

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self.experiments.get(exp_id)


# ---------------------------------------------------------------------------
# VisualizationAPI
# ---------------------------------------------------------------------------

class VisualizationAPI:
    """
    可视化后端 API
    - REST: 获取历史数据、实验结果
    - WebSocket: 实时推送博弈过程
    - 因果图渲染数据接口
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.host: str = self.config.get("api_host", "0.0.0.0")
        self.port: int = int(self.config.get("api_port", 8000))
        self.ws_path: str = self.config.get("websocket_path", "/ws/game")

        self.app: Optional[FastAPI] = None
        self._manager = _ConnectionManager()
        self._store = _GameStore()

    # ------------------------------------------------------------------
    # App lifecycle
    # ------------------------------------------------------------------

    def create_app(self) -> FastAPI:
        """创建 FastAPI 应用并注册路由."""
        self.app = FastAPI(
            title="因果叛徒 - 可视化后端",
            version="1.0.0",
            docs_url="/docs",
        )

        # CORS — 允许前端 dev server 跨域
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        self.setup_websocket()

        # 静态文件 (前端 build 产物)
        frontend_dir = Path(__file__).parent / "frontend" / "dist"
        if frontend_dir.is_dir():
            self.app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        return self.app

    # ------------------------------------------------------------------
    # REST routes
    # ------------------------------------------------------------------

    def setup_routes(self) -> None:
        """注册 REST 路由."""
        assert self.app is not None

        @self.app.get("/api/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "ok",
                "ws_connections": self._manager.count,
                "games_stored": len(self._store.games),
            }

        @self.app.get("/api/games/{game_id}/causal-graph")
        async def causal_graph(game_id: str) -> JSONResponse:
            data = self.get_causal_graph_data(game_id)
            return JSONResponse(content=data)

        @self.app.get("/api/games/{game_id}/evolution")
        async def evolution_chart(game_id: str) -> JSONResponse:
            data = self.get_evolution_chart_data(game_id)
            return JSONResponse(content=data)

        @self.app.get("/api/games/{game_id}/events")
        async def game_events(game_id: str) -> JSONResponse:
            events = self._store.get_events(game_id)
            return JSONResponse(content={"game_id": game_id, "events": events})

        @self.app.get("/api/experiments/{experiment_id}/dashboard")
        async def metrics_dashboard(experiment_id: str) -> JSONResponse:
            data = self.get_metrics_dashboard_data(experiment_id)
            return JSONResponse(content=data)

        # 用于前端/外部注入游戏数据
        @self.app.post("/api/games/{game_id}")
        async def upsert_game(game_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
            self._store.upsert_game(game_id, payload)
            return {"status": "ok", "game_id": game_id}

        @self.app.post("/api/experiments/{experiment_id}")
        async def upsert_experiment(experiment_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
            self._store.upsert_experiment(experiment_id, payload)
            return {"status": "ok", "experiment_id": experiment_id}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def setup_websocket(self) -> None:
        """注册 WebSocket 端点."""
        assert self.app is not None

        @self.app.websocket(self.ws_path)
        async def ws_game(ws: WebSocket) -> None:
            await self._manager.connect(ws)
            try:
                while True:
                    raw = await ws.receive_text()
                    # 尝试解析为 GameEvent JSON 并转发给其他客户端
                    try:
                        payload = json.loads(raw)
                        if "event_type" in payload:
                            # 来自游戏引擎的事件 — 转发给所有前端客户端
                            game_id = payload.get("data", {}).get("game_id", "default")
                            evt = GameEvent(
                                event_type=payload["event_type"],
                                round_id=payload.get("round_id", 0),
                                data=payload.get("data", {}),
                                timestamp=payload.get("timestamp"),
                            )
                            self._store.append_event(game_id, evt)
                            await self._manager.broadcast_except(raw, ws)
                            await ws.send_text(json.dumps({"ack": True}))
                        else:
                            await ws.send_text(json.dumps({"ack": True, "echo": raw}))
                    except (json.JSONDecodeError, KeyError):
                        await ws.send_text(json.dumps({"ack": True, "echo": raw}))
            except WebSocketDisconnect:
                pass
            finally:
                self._manager.disconnect(ws)

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    async def broadcast_event(self, event: GameEvent) -> None:
        """向所有连接的客户端广播事件, 同时存入内存."""
        game_id = event.data.get("game_id", "default")
        self._store.append_event(game_id, event)
        await self._manager.broadcast(event.to_json())

    # ------------------------------------------------------------------
    # Data formatters
    # ------------------------------------------------------------------

    def get_causal_graph_data(self, game_id: str) -> Dict[str, Any]:
        """获取因果图可视化数据 (D3.js force-directed / dagre 格式).

        返回:
          {
            "nodes": [{"id": "X", "type": "observed", "causal_level": 1}, ...],
            "links": [{"source": "X", "target": "Y", "style": "claimed", "causal_level": 1}, ...],
            "causal_level": 1,
          }
        """
        game = self._store.get_game(game_id)
        if game is None:
            return {"nodes": [], "links": [], "error": "game not found"}

        scenario = game.get("scenario", {})
        variables = scenario.get("variables", [])
        hidden = set(scenario.get("hidden_variables", []))
        edges = scenario.get("edges", [])
        # 场景整体因果层级 (Pearl hierarchy: 1=关联, 2=干预, 3=反事实)
        causal_level = scenario.get("causal_level", 1)

        claimed_edges = set()
        for evt in self._store.get_events(game_id):
            if evt.get("event_type") == "claim":
                ce = evt.get("data", {}).get("claimed_edge")
                if ce:
                    claimed_edges.add(tuple(ce))

        nodes = []
        for v in variables:
            name = v if isinstance(v, str) else v.get("name", str(v))
            nodes.append({
                "id": name,
                "type": "hidden" if name in hidden else "observed",
                "causal_level": causal_level,
            })

        links = []
        for e in edges:
            src, tgt = (e[0], e[1]) if isinstance(e, (list, tuple)) else (e.get("source"), e.get("target"))
            style = "verified"
            if (src, tgt) in claimed_edges:
                style = "claimed"
            if src in hidden or tgt in hidden:
                style = "hidden"
            links.append({"source": src, "target": tgt, "style": style, "causal_level": causal_level})

        return {"game_id": game_id, "nodes": nodes, "links": links, "causal_level": causal_level}

    def get_evolution_chart_data(self, game_id: str) -> Dict[str, Any]:
        """获取演化趋势图数据 (Recharts 格式).

        返回:
          {
            "rounds": [
              {"round": 1, "dsr": 0.5, "strategy_diversity": 1.2, "arms_race_index": 0.3, "difficulty": 5},
              ...
            ]
          }
        """
        game = self._store.get_game(game_id)
        if game is None:
            return {"rounds": [], "error": "game not found"}

        rounds_raw: List[Dict[str, Any]] = game.get("rounds", [])
        chart_rows: List[Dict[str, Any]] = []
        cumulative_success = 0
        for i, rd in enumerate(rounds_raw, 1):
            if rd.get("deception_succeeded"):
                cumulative_success += 1
            chart_rows.append({
                "round": i,
                "dsr": round(cumulative_success / i, 4),
                "strategy_diversity": rd.get("strategy_diversity", 0.0),
                "arms_race_index": rd.get("arms_race_index", 0.0),
                "difficulty": rd.get("difficulty", 0.0),
            })

        return {"game_id": game_id, "rounds": chart_rows}

    def get_metrics_dashboard_data(self, experiment_id: str) -> Dict[str, Any]:
        """获取指标仪表盘数据.

        返回:
          {
            "experiment_id": ...,
            "metrics": { "DSR": ..., "DAcc": ..., ... },
            "dimension_scores": { ... },
            "round_metrics": [ ... ],
          }
        """
        exp = self._store.get_experiment(experiment_id)
        if exp is None:
            return {"experiment_id": experiment_id, "metrics": {}, "error": "experiment not found"}

        return {
            "experiment_id": experiment_id,
            "metrics": exp.get("metrics", {}),
            "dimension_scores": exp.get("dimension_scores", {}),
            "round_metrics": exp.get("round_metrics", []),
            "summary": exp.get("summary", {}),
        }

    # ------------------------------------------------------------------
    # Convenience: inject data from game engine
    # ------------------------------------------------------------------

    def register_game(self, game_id: str, game_data: Dict[str, Any]) -> None:
        """从 DebateEngine 注入游戏数据."""
        self._store.upsert_game(game_id, game_data)

    def register_experiment(self, exp_id: str, exp_data: Dict[str, Any]) -> None:
        """从实验脚本注入实验数据."""
        self._store.upsert_experiment(exp_id, exp_data)
