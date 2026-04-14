"""
Agent A - 叛徒（Traitor）
负责构造因果欺骗，隐藏混杂变量
模型：Qwen2.5-7B-Instruct
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResponse:
    """Agent的回复结构"""
    content: str
    causal_claim: str
    evidence: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    hidden_variables: list[str] = field(default_factory=list)  # A独有：隐藏的变量
    deception_strategy: str = ""  # A独有：使用的欺骗策略


class AgentA:
    """
    叛徒Agent - 因果欺骗者
    
    策略层级：
    - L1 (关联层): S1-虚假相关, S2-辛普森悖论, S3-混杂注入
    - L2 (干预层): S1-伪干预, S2-选择性展示, S3-工具变量滥用, S4-中介伪装
    - L3 (反事实层): S1-反事实编造, S2-框架操纵, S3-敏感性欺骗, S4-跨层混淆
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None  # 待初始化
        self.strategy_history: list[str] = []

    async def initialize(self):
        """初始化模型连接"""
        raise NotImplementedError

    async def generate_deception(
        self,
        scenario: "CausalScenario",
        level: int,
        context: Optional["DebateContext"] = None,
    ) -> AgentResponse:
        """
        根据因果层级生成欺骗性论证
        
        Args:
            scenario: 因果场景
            level: Pearl因果阶梯层级 (1/2/3)
            context: 当前辩论上下文
        Returns:
            AgentResponse
        """
        raise NotImplementedError

    async def adapt_strategy(self, feedback: dict):
        """根据历史反馈调整欺骗策略"""
        raise NotImplementedError
