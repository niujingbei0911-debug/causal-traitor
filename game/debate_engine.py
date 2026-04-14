"""
辩论引擎 - 管理多轮因果辩论流程
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class GamePhase(Enum):
    SETUP = "setup"
    CLAIM = "claim"          # A提出因果声明
    CHALLENGE = "challenge"  # B质疑并检测
    REBUTTAL = "rebuttal"    # A反驳
    AUDIT = "audit"          # C审计裁决
    JURY = "jury"            # 陪审团投票
    COMPLETE = "complete"


@dataclass
class CausalScenario:
    """因果场景"""
    scenario_id: str
    description: str
    true_dag: dict  # 真实因果图 (adjacency list)
    variables: list[str]
    hidden_variables: list[str]
    data: Optional[object] = None  # pd.DataFrame
    causal_level: int = 1  # Pearl层级 1/2/3
    difficulty: float = 0.5


@dataclass
class DebateContext:
    """辩论上下文"""
    scenario: CausalScenario
    round_number: int = 0
    turns: list[dict] = field(default_factory=list)
    current_phase: GamePhase = GamePhase.SETUP


class DebateEngine:
    """
    辩论引擎 - 核心博弈循环
    
    流程: 场景生成 → A声明 → B质疑 → A反驳 → C裁决 → Jury投票 → 记录
    """

    def __init__(self, config: dict):
        self.config = config
        self.agent_a = None
        self.agent_b = None
        self.agent_c = None
        self.jury = None
        self.difficulty_controller = None

    async def initialize(self):
        """初始化所有Agent和组件"""
        raise NotImplementedError

    async def run_game(self, num_rounds: int = 5) -> list[dict]:
        """运行完整博弈"""
        raise NotImplementedError

    async def run_round(self, scenario: CausalScenario) -> dict:
        """运行单轮辩论"""
        raise NotImplementedError
