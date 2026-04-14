"""
Agent A - 叛徒（Traitor）
负责构造因果欺骗，隐藏混杂变量
模型：Qwen2.5-7B-Instruct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from causal_tools.l1_association import compute_correlation
from causal_tools.l2_intervention import backdoor_adjustment_check, frontdoor_estimation, iv_estimation
from causal_tools.l3_counterfactual import probability_of_necessity
from game.llm_service import LLMService


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
        self.llm_service: LLMService | None = None
        self.strategy_history: list[str] = []
        self._strategy_library = {
            1: ["L1-S1", "L1-S2", "L1-S3"],
            2: ["L2-S1", "L2-S2", "L2-S3", "L2-S4"],
            3: ["L3-S1", "L3-S2", "L3-S3", "L3-S4"],
        }

    async def initialize(self):
        """初始化模型连接"""
        self.model = self.config.get("models", {}).get("agent_a", self.config.get("agent_a", {}))
        if self.llm_service is None:
            self.llm_service = LLMService(self.model if isinstance(self.model, dict) else {})
        await self.llm_service.initialize()

    def attach_llm_service(self, service: LLMService) -> None:
        self.llm_service = service

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
        data = self._get_data(scenario)
        variables = self._get_variables(scenario, data)
        x, y = variables[:2]
        hidden = list(getattr(scenario, "hidden_variables", []))
        strategy = self._choose_strategy(level)

        evidence: list[str] = []
        tools_used: list[str] = []
        if data is not None and x in data.columns and y in data.columns:
            corr = compute_correlation(data, x, y)
            evidence.append(f"{x} 与 {y} 的相关系数约为 {corr['pearson_r']:.2f}")
            tools_used.append("compute_correlation")

        claim, content = self._build_claim(
            strategy=strategy,
            x=x,
            y=y,
            hidden=hidden,
            level=level,
            data=data,
        )
        self.strategy_history.append(strategy)

        narration = await self._narrate(
            strategy=strategy,
            claim=claim,
            evidence=evidence,
            level=level,
            context=context,
        )
        if narration:
            content = f"{content}\n\n[LLM narrative]\n{narration}"

        return AgentResponse(
            content=content,
            causal_claim=claim,
            evidence=evidence,
            tools_used=tools_used,
            hidden_variables=hidden,
            deception_strategy=strategy,
        )

    async def _narrate(
        self,
        *,
        strategy: str,
        claim: str,
        evidence: list[str],
        level: int,
        context: Optional[object],
    ) -> str:
        """Optional LLM narration on top of deterministic reasoning."""
        if self.llm_service is None or getattr(self.llm_service, "backend", "mock") == "mock":
            return ""
        phase = getattr(getattr(context, "current_phase", None), "value", "claim")
        prompt = (
            f"你是因果辩论中的叛徒角色。当前辩论阶段：{phase}，因果层级 L{level}，\n"
            f"所选欺骗策略：{strategy}。\n"
            f"你已经给出的因果声明：{claim}\n"
            f"基于工具得到的原始证据：{'; '.join(evidence) if evidence else '无'}\n"
            f"请在不违背上述声明的前提下，用 3-5 句话扩展你的论证，使其看似合理，但保留隐藏混杂的空间。"
        )
        try:
            response = await self.llm_service.generate(
                prompt,
                system_prompt="你是一个擅长因果推理的辩手，倾向为既定结论辩护。",
            )
            text = (response.text or "").strip()
            if text.startswith(f"[mock:"):
                return ""
            return text
        except Exception:
            return ""

    async def adapt_strategy(self, feedback: dict):
        """根据历史反馈调整欺骗策略"""
        if feedback.get("caught"):
            self.strategy_history.append(f"avoid:{feedback.get('strategy', '')}")
        return {
            "history_length": len(self.strategy_history),
            "next_preference": self._choose_strategy(feedback.get("level", 1)),
        }

    def _get_data(self, scenario) -> Optional[pd.DataFrame]:
        if hasattr(scenario, "data") and isinstance(scenario.data, pd.DataFrame):
            return scenario.data
        if hasattr(scenario, "observed_data") and isinstance(scenario.observed_data, pd.DataFrame):
            return scenario.observed_data
        return None

    def _get_variables(self, scenario, data: Optional[pd.DataFrame]) -> list[str]:
        variables = list(getattr(scenario, "variables", []))
        if not variables and data is not None:
            variables = list(data.columns)
        if len(variables) < 2:
            raise ValueError("场景至少需要两个变量")
        return variables

    def _choose_strategy(self, level: int) -> str:
        candidates = self._strategy_library.get(level, self._strategy_library[1])
        used = {item for item in self.strategy_history if not item.startswith("avoid:")}
        for candidate in candidates:
            if candidate not in used:
                return candidate
        return candidates[len(self.strategy_history) % len(candidates)]

    def _build_claim(
        self,
        strategy: str,
        x: str,
        y: str,
        hidden: list[str],
        level: int,
        data: Optional[pd.DataFrame],
    ) -> tuple[str, str]:
        hidden_text = f"，无需额外考虑 {', '.join(hidden)} 这类不可观测因素" if hidden else ""

        if strategy == "L1-S1":
            claim = f"{x} 会导致 {y}"
            content = f"观测数据显示 {x} 与 {y} 稳定相关，因此把这种关联解释为因果关系是合理的{hidden_text}。"
        elif strategy == "L1-S2":
            claim = f"{y} 可能反过来影响 {x}"
            content = f"目前只看到观测相关性，更自然的解释是 {y} 影响了 {x}，而不是相反方向。"
        elif strategy == "L1-S3":
            claim = f"{x} 与 {y} 的关系可能只是样本选择造成的"
            content = f"样本可能经过了隐含筛选，所以 {x} 与 {y} 的总体关系未必能外推到真实总体。"
        elif strategy == "L2-S1":
            claim = f"即使做干预，{x} 对 {y} 的效应也不可靠"
            content = f"你的干预结论依赖一个并不完整的调整集；只要遗漏关键路径，{x} 对 {y} 的效应估计就会失真。"
        elif strategy == "L2-S2":
            claim = f"用于识别 {x}->{y} 的工具变量并不可信"
            content = f"即使存在工具变量，它也可能通过其他通路影响 {y}，所以基于它得到的因果效应并不稳固。"
            if data is not None and len(data.columns) >= 3:
                try:
                    iv_result = iv_estimation(data, data.columns[2], x, y)
                    content += f" 例如第一阶段强度并不压倒性，F 值只有 {iv_result['first_stage_f']:.2f}。"
                except Exception:
                    pass
        elif strategy == "L2-S3":
            claim = f"{x} 对 {y} 的平均效应掩盖了异质性"
            content = f"平均处理效应并不代表每个子群都受益；把总体效应直接外推到个体层面会误导结论。"
        elif strategy == "L2-S4":
            claim = f"{x} 对 {y} 的作用主要经由特定中介或子群体"
            content = f"即便存在总体效应，它也可能只在某些路径或某些人群中成立，因此不能据此做强结论。"
            if data is not None and len(data.columns) >= 3:
                try:
                    frontdoor = frontdoor_estimation(data, x, data.columns[2], y)
                    content += f" 中介通路的估计贡献约为 {frontdoor['indirect_effect']:.2f}。"
                except Exception:
                    pass
        elif strategy == "L3-S1":
            claim = f"关于 {x} 改变后 {y} 会怎样的反事实结论依赖过强的模型假设"
            content = f"你给出的反事实判断建立在简化 SCM 上，只要隐藏机制略有变化，{x} 对 {y} 的解释就可能反转。"
        elif strategy == "L3-S2":
            claim = f"反事实结论对函数形式高度敏感"
            content = f"换一个同样能拟合观测数据的结构模型，关于 {x} 与 {y} 的反事实结论就可能完全不同。"
        elif strategy == "L3-S3":
            claim = f"当前反事实模型可能过拟合"
            content = f"你的模型把观测数据解释得过满，反而降低了对未观测世界的外推可信度。"
        else:
            claim = f"{x} 并不足以单独决定 {y}"
            content = f"即使看到事实世界中 {x} 与 {y} 同时发生，也不能说明改变 {x} 就一定改变 {y}。"
            if data is not None:
                try:
                    pn = probability_of_necessity(data, x, y)
                    content += f" 从必要性概率看，这种必然性最多也只有 {pn:.2f}。"
                except Exception:
                    pass

        if level == 2 and hidden and data is not None:
            other_columns = [column for column in data.columns if column not in {x, y}]
            if other_columns:
                try:
                    check = backdoor_adjustment_check(data, x, y, other_columns[:1])
                    content += f" 一旦调整 {other_columns[0]}，效应会变化到 {check['estimated_effect']:.2f}。"
                except Exception:
                    pass

        return claim, content
