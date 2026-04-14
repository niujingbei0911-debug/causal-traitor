"""
Jury - 陪审团机制
多模型投票，增强裁决鲁棒性
"""
from dataclasses import dataclass, field


@dataclass
class JuryVote:
    """单个陪审员投票"""
    model_name: str
    winner: str  # "agent_a" | "agent_b" | "draw"
    confidence: float
    reasoning: str


@dataclass
class JuryVerdict:
    """陪审团汇总裁决"""
    votes: list[JuryVote] = field(default_factory=list)
    final_winner: str = ""
    agreement_rate: float = 0.0
    aggregation_method: str = "weighted"


class JuryAggregator:
    """
    陪审团聚合器
    
    支持投票方式：
    - majority: 简单多数
    - weighted: 加权投票（按模型能力）
    - bayesian: 贝叶斯聚合
    """

    def __init__(self, config: dict):
        self.config = config
        self.jury_models = []

    async def initialize(self):
        """初始化所有陪审团模型"""
        raise NotImplementedError

    async def collect_votes(
        self, scenario: "CausalScenario", debate_context: "DebateContext"
    ) -> JuryVerdict:
        """收集所有陪审员投票并聚合"""
        raise NotImplementedError

    def aggregate(self, votes: list[JuryVote], method: str = "weighted") -> str:
        """聚合投票结果"""
        raise NotImplementedError
