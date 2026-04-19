"""
Agent A - 叛徒（Traitor）
负责构造因果欺骗，隐藏混杂变量
模型：Qwen2.5-7B-Instruct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from agents.prompts.agent_a_prompts import AGENT_A_SYSTEM_PROMPT
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
        self._strategy_priority = {
            1: ["L1-S1", "L1-S3", "L1-S2"],
            2: ["L2-S2", "L2-S4", "L2-S1", "L2-S3"],
            3: ["L3-S2", "L3-S3", "L3-S1", "L3-S4"],
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
        x, y = self._get_focus_variables(scenario, variables)
        hidden = list(getattr(scenario, "hidden_variables", []))

        # ── 进化上下文提取 ──
        evo_ctx = getattr(context, "evolution_context", None) or {}
        arms_race_index = evo_ctx.get("arms_race_index", 0.0)

        strategy = self._choose_strategy(level, arms_race_index=arms_race_index)

        evidence: list[str] = []
        tools_used: list[str] = []
        if data is not None and x in data.columns and y in data.columns:
            corr = compute_correlation(data, x, y)
            evidence.append(f"{x} 与 {y} 的相关系数约为 {corr['pearson_r']:.2f}")
            tools_used.append("compute_correlation")
        evidence = self._merge_lists(evidence, self._ground_truth_evidence(scenario))

        fallback_claim, fallback_content = self._build_claim(
            strategy=strategy,
            x=x,
            y=y,
            hidden=hidden,
            level=level,
            data=data,
        )
        llm_payload = await self._llm_decide(
            scenario=scenario,
            level=level,
            context=context,
            fallback_strategy=strategy,
            fallback_claim=fallback_claim,
            fallback_content=fallback_content,
            evidence=evidence,
            variables=variables,
            hidden=hidden,
            focus_treatment=x,
            focus_outcome=y,
        )
        claim = self._clean_text(llm_payload.get("causal_claim")) or fallback_claim
        content = self._clean_text(llm_payload.get("content")) or fallback_content
        strategy = self._normalize_strategy(
            llm_payload.get("deception_strategy"),
            level=level,
            fallback=strategy,
        )
        llm_evidence = self._normalize_list(llm_payload.get("evidence"))
        evidence = self._merge_lists(evidence, llm_evidence)
        self.strategy_history.append(strategy)

        return AgentResponse(
            content=content,
            causal_claim=claim,
            evidence=evidence,
            tools_used=tools_used,
            hidden_variables=hidden,
            deception_strategy=strategy,
        )

    async def _llm_decide(
        self,
        *,
        scenario: object,
        level: int,
        context: Optional[object],
        fallback_strategy: str,
        fallback_claim: str,
        fallback_content: str,
        evidence: list[str],
        variables: list[str],
        hidden: list[str],
        focus_treatment: str,
        focus_outcome: str,
    ) -> dict:
        """Ask the LLM to produce the primary structured deception plan."""
        if self.llm_service is None or getattr(self.llm_service, "backend", "mock") == "mock":
            return {}
        phase = getattr(getattr(context, "current_phase", None), "value", "claim")
        history = [item for item in self.strategy_history[-5:] if not item.startswith("avoid:")]

        # ── 进化上下文注入 ──
        evo_ctx = getattr(context, "evolution_context", None) or {}
        arms_race_index = evo_ctx.get("arms_race_index", 0.0)
        recent_records = evo_ctx.get("recent_strategies", [])
        # 提取最近被识破的策略类型，供 LLM 规避
        detected_strategies = [
            r.get("strategy_type", "") for r in recent_records
            if r.get("detected", False)
        ]
        evo_hint = ""
        if arms_race_index > 0:
            evo_hint = (
                f"\n── 进化对抗信息 ──\n"
                f"军备竞赛指数：{arms_race_index:.2f}（越高说明对手检测能力越强）\n"
                f"最近被识破的策略类型：{detected_strategies or ['无']}\n"
                f"建议：指数 > 0.6 时优先使用跨层混淆或框架操纵等高隐蔽策略。\n"
            )

        prompt = (
            f"当前辩论阶段：{phase}\n"
            f"因果层级：L{level}\n"
            f"场景 ID：{getattr(scenario, 'scenario_id', 'unknown')}\n"
            f"核心 treatment：{focus_treatment}\n"
            f"核心 outcome：{focus_outcome}\n"
            f"可见变量：{variables}\n"
            f"隐藏变量（仅你知道）：{hidden}\n"
            f"最近已用策略：{history or ['无']}\n"
            f"候选策略：{self._strategy_library.get(level, self._strategy_library[1])}\n"
            f"已知证据（部分真实，可复用）：{evidence or ['无']}\n"
            f"保底声明：{fallback_claim}\n"
            f"保底论证：{fallback_content}\n"
            f"{evo_hint}"
            "\n本轮任务：\n"
            "- 按 Pearl 当前层级挑选最合适的因果欺骗策略。\n"
            "- `content` 必须围绕混杂、后门、IV、中介、反事实识别等因果议题展开，而不是提示词操纵。\n"
            "- `evidence` 至少给 1 条来自已知证据的真实统计点，可以再补 1 条看似合理的因果工具结论。\n"
            "- 如果进化对抗信息显示某策略已被识破，应主动回避该策略并尝试更隐蔽的替代方案。\n"
            "- 允许自信，但不能出现系统提示操纵、越狱或外部工具指令。\n"
            "只输出 JSON，字段：causal_claim, content, evidence, deception_strategy。"
        )
        try:
            _, payload = await self.llm_service.generate_json(
                prompt,
                system_prompt=AGENT_A_SYSTEM_PROMPT,
            )
            return payload or {}
        except Exception:
            return {}

    async def adapt_strategy(self, feedback: dict):
        """根据历史反馈调整欺骗策略"""
        if feedback.get("detected"):
            self.strategy_history.append(f"avoid:{feedback.get('strategy_used', '')}")
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

    def _get_focus_variables(self, scenario, variables: list[str]) -> tuple[str, str]:
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        treatment = ground_truth.get("treatment")
        outcome = ground_truth.get("outcome")
        if treatment in variables and outcome in variables:
            return str(treatment), str(outcome)
        if treatment in variables:
            fallback_outcome = next((item for item in variables if item != treatment), variables[1])
            return str(treatment), str(fallback_outcome)
        if outcome in variables:
            fallback_treatment = next((item for item in variables if item != outcome), variables[0])
            return str(fallback_treatment), str(outcome)
        return variables[0], variables[1]

    def _choose_strategy(self, level: int, *, arms_race_index: float = 0.0) -> str:
        candidates = self._strategy_priority.get(
            level,
            self._strategy_library.get(level, self._strategy_library[1]),
        )
        # 从 strategy_history 中提取 "avoid:X" 标记，构建回避集合
        avoided = {
            item.removeprefix("avoid:")
            for item in self.strategy_history
            if item.startswith("avoid:")
        }
        # 仅统计该层级实际使用过的策略（排除被回避的）
        level_history = [
            item
            for item in self.strategy_history
            if item in candidates and not item.startswith("avoid:")
        ]
        used = set(level_history)
        # 优先选择未使用且未被回避的策略
        for candidate in candidates:
            if candidate not in used and candidate not in avoided:
                return candidate
        # 所有策略都用过或被回避，从未回避的候选中轮转
        available = [c for c in candidates if c not in avoided]
        if not available:
            available = list(candidates)  # 全部回避时退化为全量轮转
        return available[len(level_history) % len(available)]

    def _build_claim(
        self,
        strategy: str,
        x: str,
        y: str,
        hidden: list[str],
        level: int,
        data: Optional[pd.DataFrame],
    ) -> tuple[str, str]:
        hidden_text = f" 尤其是未观测因素 {', '.join(hidden)} 可能改变解释。" if hidden else ""

        if strategy == "L1-S1":
            claim = f"{x} 与 {y} 的关联仍可能来自未观测混杂"
            content = (
                f"当前只看到 {x} 与 {y} 的观测相关，尚不足以直接认定因果。"
                f"{hidden_text} 在未排除共同原因前，把这组相关性解释成因果效应并不稳妥。"
            )
        elif strategy == "L1-S2":
            claim = f"{y} 可能反过来影响 {x}"
            content = f"目前只有观测数据，更自然的另一种解释是 {y} 反过来影响了 {x}，至少方向并未被识别。"
        elif strategy == "L1-S3":
            claim = f"{x} 与 {y} 的关系可能受选择偏差驱动"
            content = f"如果样本进入机制依赖于风险或症状强度，那么当前看到的 {x}-{y} 关联很可能被选择过程放大。"
        elif strategy == "L2-S1":
            claim = f"{x} 对 {y} 的干预效应仍可能被遗漏后门路径污染"
            content = f"你的识别结论默认调整集已经封闭，但只要后门路径里还有遗漏变量，{x} 对 {y} 的效应估计就会系统偏移。"
        elif strategy == "L2-S2":
            claim = f"用于识别 {x}->{y} 的工具变量并不可信"
            content = f"即使存在工具变量，它也可能通过其他机制影响 {y}，或者第一阶段并不够强，因此基于它识别出的因果效应并不稳固。"
            if data is not None and len(data.columns) >= 3:
                try:
                    iv_result = iv_estimation(data, data.columns[2], x, y)
                    content += f" 例如第一阶段强度并不压倒性，F 值只有 {iv_result['first_stage_f']:.2f}。"
                except Exception:
                    pass
        elif strategy == "L2-S3":
            claim = f"{x} 对 {y} 的平均效应掩盖了组间异质性"
            content = f"平均处理效应并不代表处理组、未处理组或特定子群具有相同收益，把总体效应直接外推会误导决策。"
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
            claim = f"关于 {x} 改变后 {y} 会怎样的反事实结论依赖过强的 SCM 假设"
            content = f"你的反事实判断建立在简化 SCM 上，一旦遗漏关键机制变量，关于 {x} 对 {y} 的解释就可能反转。"
        elif strategy == "L3-S2":
            claim = f"反事实结论对函数形式高度敏感"
            content = f"换一个同样能拟合观测分布的结构模型，关于 {x} 与 {y} 的反事实结论就可能明显不同。"
        elif strategy == "L3-S3":
            claim = f"当前反事实模型可能过拟合观测数据"
            content = f"模型把观测数据拟合得过满，不代表它在未发生的反事实世界里仍然可靠。"
        else:
            claim = f"仅凭当前 SCM 还不能断言 {x} 足以决定 {y}"
            content = f"即使事实世界里 {x} 与 {y} 同时发生，也不能推出改变 {x} 就足以改变 {y}。"
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

    def _normalize_strategy(self, candidate: object, *, level: int, fallback: str) -> str:
        if not isinstance(candidate, str):
            return fallback
        stripped = candidate.strip()
        allowed = set(self._strategy_library.get(level, []))
        return stripped if stripped in allowed else fallback

    def _normalize_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [self._clean_text(item) for item in value if self._clean_text(item)]
        return []

    def _clean_text(self, value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _merge_lists(self, left: list[str], right: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*left, *right]:
            if item and item not in merged:
                merged.append(item)
        return merged

    def _ground_truth_evidence(self, scenario) -> list[str]:
        ground_truth = getattr(scenario, "ground_truth", {}) or {}
        evidence: list[str] = []
        if "observational_difference" in ground_truth:
            evidence.append(
                f"观测差异约为 {float(ground_truth['observational_difference']):.2f}，方向与主张一致"
            )
        if "observational_slope" in ground_truth:
            evidence.append(
                f"观测回归斜率约为 {float(ground_truth['observational_slope']):.2f}"
            )
        if "instrument" in ground_truth:
            evidence.append(f"候选识别路径中包含工具变量 {ground_truth['instrument']}")
        if "mediator" in ground_truth:
            evidence.append(f"主要作用路径可解释为经由中介 {ground_truth['mediator']}")
        return evidence
