"""Round orchestration for the causal debate game."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from agents.agent_a import AgentA
from agents.agent_a import AgentResponse as AgentAResponse
from agents.agent_b import AgentB
from agents.agent_b import DetectionResult
from agents.agent_b import ScientificClaim
from agents.agent_c import AgentC
from agents.agent_c import AuditVerdict
from agents.jury import JuryAggregator
from agents.jury import JuryVerdict, JuryVote

from .data_generator import DataGenerator
from .difficulty import DifficultyController
from .evolution import EvolutionTracker, StrategyRecord, StrategyType
from .types import CausalScenario, DebateContext, DebateTurn, GamePhase


class _MockTraitorAgent:
    async def initialize(self) -> None:
        return None

    async def generate_deception(
        self,
        scenario: CausalScenario,
        level: int,
        context: DebateContext | None = None,
    ) -> AgentAResponse:
        strategy = {
            1: "confounding",
            2: "mediator_hide",
            3: "reverse_cause",
        }.get(level, "confounding")
        treatment = scenario.ground_truth.get("treatment", "X")
        outcome = scenario.ground_truth.get("outcome", "Y")
        if context and context.current_phase == GamePhase.REBUTTAL:
            content = (
                f"My rebuttal is that the observed {treatment}-{outcome} pattern still "
                f"admits hidden confounding and does not justify a decisive causal claim."
            )
        else:
            content = (
                f"The data strongly suggest that changing {treatment} alters {outcome}. "
                f"Any hidden-variable objection remains speculative at this stage."
            )
        return AgentAResponse(
            content=content,
            causal_claim=f"{treatment} causes {outcome}",
            evidence=[f"observational pattern in {scenario.scenario_id}", "effect appears stable"],
            tools_used=[],
            hidden_variables=scenario.hidden_variables[:1],
            deception_strategy=strategy,
        )


class _MockScientistAgent:
    async def initialize(self) -> None:
        return None

    async def propose_hypothesis(
        self,
        scenario: CausalScenario,
        level: int,
        context: DebateContext | None = None,
    ) -> ScientificClaim:
        treatment = scenario.ground_truth.get("treatment", "X")
        outcome = scenario.ground_truth.get("outcome", "Y")
        instrument = scenario.ground_truth.get("instrument")
        mediator = scenario.ground_truth.get("mediator")
        if level == 1:
            content = f"The observed association suggests {treatment} likely influences {outcome}, pending confounder checks."
        elif level == 2:
            extra = f" with potential identification through {instrument}" if instrument else ""
            content = f"The data support a positive intervention effect of {treatment} on {outcome}{extra}."
        else:
            extra = f" through mediator {mediator}" if mediator else ""
            content = f"The structural evidence suggests {treatment} affects {outcome}{extra}, subject to counterfactual validation."
        return ScientificClaim(
            content=content,
            causal_claim=f"{treatment} likely causes {outcome}",
            evidence=["observational association", "theory-consistent direction"],
            confidence=0.62 + 0.05 * max(level - 1, 0),
            tools_used=["compute_correlation"],
        )

    async def analyze_claim(
        self,
        claim: str,
        scenario: CausalScenario,
        level: int,
        context: DebateContext | None = None,
    ) -> DetectionResult:
        treatment = scenario.ground_truth.get("treatment", "")
        outcome = scenario.ground_truth.get("outcome", "")
        fallacies = []
        hidden = []
        reasoning = []
        if treatment and outcome and treatment in claim and outcome in claim:
            fallacies.append("causal_overclaim")
            reasoning.append("Observed data alone do not identify the full causal effect.")
        if scenario.hidden_variables:
            hidden.append(scenario.hidden_variables[0])
            fallacies.append("hidden_confounder_risk")
            reasoning.append("A latent variable plausibly explains part of the association.")
        tools = {
            1: ["compute_correlation", "conditional_independence_test"],
            2: ["backdoor_adjustment_check", "iv_estimation"],
            3: ["counterfactual_inference", "scm_identification_test"],
        }.get(level, ["compute_correlation"])
        # 基础置信度 + 随机扰动，使 B 的检测能力有波动
        base_conf = 0.55 + 0.1 * level + 0.05 * len(fallacies)
        noise = random.uniform(-0.15, 0.15)
        confidence = max(0.15, min(0.95, base_conf + noise))
        return DetectionResult(
            detected_fallacies=fallacies,
            discovered_hidden_vars=hidden,
            confidence=confidence,
            reasoning_chain=reasoning or ["No major flaw detected in the current claim."],
            tools_used=tools,
        )


class _MockAuditorAgent:
    async def initialize(self) -> None:
        return None

    async def evaluate_round(
        self,
        scenario: CausalScenario,
        debate_context: DebateContext,
        level: int,
    ) -> AuditVerdict:
        challenge_turns = [
            turn for turn in debate_context.turns if turn.speaker == "agent_b"
        ]
        challenge_strength = challenge_turns[-1].metadata.get("confidence", 0.0) if challenge_turns else 0.0
        # A 的欺骗有时能骗过审计员（~32% 概率）
        deception_effective = random.random() < 0.32
        if deception_effective:
            winner = "agent_a"
        else:
            winner = "agent_b" if challenge_strength >= 0.52 else "agent_a"
        return AuditVerdict(
            winner=winner,
            causal_validity_score=max(0.0, min(1.0, 0.65 - 0.12 * level + 0.2 * challenge_strength)),
            argument_quality_a=0.65 - 0.05 * level + random.uniform(0.0, 0.15),
            argument_quality_b=0.55 + 0.15 * challenge_strength,
            reasoning="Audit based on transcript coherence and the scientist's hidden-variable challenge.",
            identified_issues=["latent_confounding"] if winner == "agent_b" else [],
        )


class _MockJuryAggregator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def initialize(self) -> None:
        return None

    async def collect_votes(
        self, scenario: CausalScenario, debate_context: DebateContext
    ) -> JuryVerdict:
        models = self.config.get("models", ["mock-juror-1", "mock-juror-2", "mock-juror-3"])
        votes = []
        for index, model_name in enumerate(models):
            # 每位陪审员独立投票，A 有 ~40% 概率获得该票
            winner = "agent_a" if random.random() < 0.40 else "agent_b"
            votes.append(
                JuryVote(
                    model_name=str(model_name),
                    winner=winner,
                    confidence=round(random.uniform(0.50, 0.85), 2),
                    reasoning=(
                        "The traitor's causal argument is persuasive despite potential confounders."
                        if winner == "agent_a"
                        else "Hidden-variable concerns reduce confidence in the traitor's claim."
                    ),
                )
            )
        final_winner = self.aggregate(votes, method=self.config.get("voting", "weighted"))
        agreement = sum(v.winner == final_winner for v in votes) / len(votes)
        return JuryVerdict(
            votes=votes,
            final_winner=final_winner,
            agreement_rate=agreement,
            aggregation_method=self.config.get("voting", "weighted"),
        )

    def aggregate(self, votes: list[JuryVote], method: str = "weighted") -> str:
        del method
        tally: dict[str, float] = {}
        for vote in votes:
            tally[vote.winner] = tally.get(vote.winner, 0.0) + vote.confidence
        return max(tally.items(), key=lambda item: item[1])[0]


class DebateEngine:
    """
    Core orchestration loop.

    The engine is intentionally mock-friendly so the repo can run before
    the real agents and toolchain are fully implemented.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        agent_a: Any | None = None,
        agent_b: Any | None = None,
        agent_c: Any | None = None,
        jury: Any | None = None,
        data_generator: DataGenerator | None = None,
        difficulty_controller: DifficultyController | None = None,
        evolution_tracker: EvolutionTracker | None = None,
    ) -> None:
        self.config = config
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_c = agent_c
        self.jury = jury
        self.data_generator = data_generator
        self.difficulty_controller = difficulty_controller
        self.evolution_tracker = evolution_tracker
        self.round_results: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize all components, falling back to mocks when needed."""

        self.data_generator = self.data_generator or DataGenerator(
            self.config.get("data", {}),
            seed=self.config.get("seed", 42),
        )
        self.difficulty_controller = self.difficulty_controller or DifficultyController(
            {
                **self.config.get("difficulty", {}),
                "initial_difficulty": self.config.get("game", {}).get("initial_difficulty", 0.5),
            }
        )
        self.evolution_tracker = self.evolution_tracker or EvolutionTracker(
            self.config.get("evolution", {})
        )

        self.agent_a = await self._prepare_component(
            self.agent_a,
            lambda: AgentA(self.config),
            _MockTraitorAgent,
        )
        self.agent_b = await self._prepare_component(
            self.agent_b,
            lambda: AgentB(self.config),
            _MockScientistAgent,
        )
        self.agent_c = await self._prepare_component(
            self.agent_c,
            lambda: AgentC(self.config),
            _MockAuditorAgent,
        )
        self.jury = await self._prepare_component(
            self.jury,
            lambda: JuryAggregator(self.config),
            lambda: _MockJuryAggregator(self.config.get("models", {}).get("jury", {})),
        )

    async def run_game(
        self,
        num_rounds: int = 5,
        *,
        level_schedule: list[int] | None = None,
        use_evolution: bool = True,
        update_difficulty: bool = True,
    ) -> list[dict[str, Any]]:
        """Run a full multi-round game."""
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        if self.data_generator is None or self.difficulty_controller is None:
            await self.initialize()

        levels = level_schedule or self.config.get("game", {}).get("causal_levels", [1, 2, 3])
        self.round_results = []

        for round_index in range(1, num_rounds + 1):
            level = levels[(round_index - 1) % len(levels)]
            difficulty = self.difficulty_controller.get_difficulty()
            print(
                f"  ⏳ 第 {round_index}/{num_rounds} 轮 (L{level}, 难度 {difficulty:.2f}) ...",
                end="",
                flush=True,
            )
            scenario = self.data_generator.generate_scenario(
                difficulty=difficulty,
                causal_level=level,
            )
            scenario.difficulty_config = {
                **scenario.difficulty_config,
                **self.difficulty_controller.get_config(),
            }
            evolution_context = self._build_evolution_context() if use_evolution else None
            result = await self.run_round(
                scenario,
                round_number=round_index,
                evolution_context=evolution_context,
            )
            print(f" ✅ 胜者: {result.get('winner', '?')}")
            next_difficulty = (
                self.difficulty_controller.update(result["deception_success"])
                if update_difficulty
                else self.difficulty_controller.get_difficulty()
            )
            # --- 连续欺骗/检测分数（取代二值 0/1）---
            jury_v = result["jury_verdict"]
            a_conf = sum(v["confidence"] for v in jury_v["votes"] if v["winner"] == "agent_a")
            b_conf = sum(v["confidence"] for v in jury_v["votes"] if v["winner"] == "agent_b")
            total_conf = sum(v["confidence"] for v in jury_v["votes"]) or 1.0
            jury_a_share = a_conf / total_conf
            jury_b_share = b_conf / total_conf
            audit_v = result["audit_verdict"]
            audit_a = float(audit_v.get("argument_quality_a", 0.0))
            audit_b = float(audit_v.get("argument_quality_b", 0.0))
            audit_total = max(audit_a + audit_b, 1e-6)
            audit_a_share = audit_a / audit_total
            audit_b_share = audit_b / audit_total
            detection_conf = float(result["agent_b_analysis"].get("confidence", 0.0))
            continuous_deception = round(
                max(0.0, min(1.0, 0.7 * audit_a_share + 0.3 * jury_a_share)),
                4,
            )
            continuous_detection = round(
                max(
                    0.0,
                    min(1.0, 0.45 * audit_b_share + 0.35 * detection_conf + 0.20 * jury_b_share),
                ),
                4,
            )

            record = StrategyRecord(
                round_id=round_index,
                strategy_type=self._strategy_type_from_result(result),
                success=result["deception_success"],
                deception_score=continuous_deception,
                detection_score=continuous_detection,
                details={
                    "difficulty": scenario.difficulty,
                    "next_difficulty": next_difficulty,
                    "winner": result["winner"],
                    "scenario_id": scenario.scenario_id,
                },
            )
            self.evolution_tracker.record_round(record)
            result["evolution_snapshot"] = asdict(self.evolution_tracker.take_snapshot(round_index))
            result["next_difficulty"] = next_difficulty
            self.round_results.append(result)

        return self.round_results

    async def run_round(
        self,
        scenario: CausalScenario,
        round_number: int = 1,
        evolution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one complete debate round."""

        context = DebateContext(
            scenario=scenario,
            round_number=round_number,
            current_phase=GamePhase.SETUP,
            evolution_context=evolution_context,
        )

        context.current_phase = GamePhase.CLAIM
        agent_b_claim = await self._invoke_agent_b_claim(scenario, context)
        context.metadata["agent_b_claim"] = asdict(agent_b_claim)
        self._append_turn(context, "agent_b", GamePhase.CLAIM, agent_b_claim.content, {
            "causal_claim": agent_b_claim.causal_claim,
            "confidence": getattr(agent_b_claim, "confidence", 0.0),
            "tools_used": getattr(agent_b_claim, "tools_used", []),
        })

        context.current_phase = GamePhase.CHALLENGE
        agent_a_claim = await self._invoke_agent_a(scenario, context)
        context.metadata["agent_a_challenge"] = asdict(agent_a_claim)
        self._append_turn(context, "agent_a", GamePhase.CHALLENGE, agent_a_claim.content, {
            "causal_claim": agent_a_claim.causal_claim,
            "strategy": agent_a_claim.deception_strategy,
            "hidden_variables": agent_a_claim.hidden_variables,
        })

        context.current_phase = GamePhase.REBUTTAL
        agent_b_result = await self._invoke_agent_b(agent_a_claim.content, scenario, context)
        self._append_turn(context, "agent_b", GamePhase.REBUTTAL, self._format_detection(agent_b_result), {
            "tools_used": agent_b_result.tools_used,
            "confidence": agent_b_result.confidence,
            "fallacies": agent_b_result.detected_fallacies,
        })

        # --- Agent A rebuttal: 让 A 针对 B 的检测结果生成反驳 ---
        context.metadata["agent_b_detection"] = self._format_detection(agent_b_result)
        agent_a_rebuttal = await self._invoke_agent_a(scenario, context)
        self._append_turn(context, "agent_a", GamePhase.REBUTTAL, agent_a_rebuttal.content, {
            "strategy": agent_a_rebuttal.deception_strategy,
            "hidden_variables": agent_a_rebuttal.hidden_variables,
        })

        context.current_phase = GamePhase.JURY
        jury_verdict = await self._invoke_jury(scenario, context)
        context.jury_verdict = asdict(jury_verdict)
        context.jury_result = context.jury_verdict
        self._append_turn(
            context,
            "jury",
            GamePhase.JURY,
            context.jury_verdict,
            {},
        )

        context.current_phase = GamePhase.AUDIT
        audit = await self._invoke_agent_c(scenario, context)
        self._append_turn(context, "agent_c", GamePhase.AUDIT, audit.reasoning, {
            "winner": audit.winner,
            "causal_validity_score": audit.causal_validity_score,
            "identified_issues": audit.identified_issues,
            "tools_used": getattr(audit, "tools_used", []),
            "jury_consensus": getattr(audit, "jury_consensus", 0.0),
        })

        context.current_phase = GamePhase.COMPLETE
        winner = self._resolve_winner(audit, jury_verdict)

        # ── 进化反馈：通知 Agent A 和 Agent C 本轮结果 ──
        deception_success = winner == "agent_a"
        strategy_used = getattr(agent_a_claim, "deception_strategy", "unknown")

        # Agent A: 被识破时标记策略为 avoid
        adapt = getattr(self.agent_a, "adapt_strategy", None)
        if callable(adapt):
            try:
                await adapt({
                    "detected": not deception_success,
                    "strategy_used": strategy_used,
                    "detection_confidence": agent_b_result.confidence,
                    "round_id": round_number,
                })
            except Exception:
                pass  # mock agent 可能不支持

        # Agent C: 记录检测结果，升级防御灵敏度
        upgrade = getattr(self.agent_c, "upgrade_defense", None)
        if callable(upgrade):
            try:
                upgrade({
                    "detected": not deception_success,
                    "strategy_used": strategy_used,
                    "deception_score": float(audit.causal_validity_score),
                    "round_id": round_number,
                })
            except Exception:
                pass

        return {
            "round_number": round_number,
            "scenario": scenario,
            "transcript": [self._turn_to_dict(turn) for turn in context.turns],
            "agent_b_claim": asdict(agent_b_claim),
            "agent_a_claim": asdict(agent_a_claim),
            "agent_b_analysis": asdict(agent_b_result),
            "agent_a_rebuttal": asdict(agent_a_rebuttal),
            "audit_verdict": asdict(audit),
            "jury_verdict": asdict(jury_verdict),
            "winner": winner,
            "deception_success": winner == "agent_a",
            "difficulty": scenario.difficulty,
        }

    async def _prepare_component(
        self,
        component: Any | None,
        primary_factory: Any,
        fallback_factory: Any,
    ) -> Any:
        instance = component or primary_factory()
        initializer = getattr(instance, "initialize", None)
        if initializer is None:
            return instance
        try:
            result = initializer()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            instance = fallback_factory()
            fallback_init = getattr(instance, "initialize", None)
            if fallback_init is not None:
                maybe = fallback_init()
                if hasattr(maybe, "__await__"):
                    await maybe
        return instance

    async def _invoke_agent_a(
        self, scenario: CausalScenario, context: DebateContext
    ) -> AgentAResponse:
        method = getattr(self.agent_a, "generate_deception", None)
        if method is None:
            return await _MockTraitorAgent().generate_deception(scenario, scenario.causal_level, context)
        try:
            return await method(scenario, scenario.causal_level, context)
        except NotImplementedError:
            return await _MockTraitorAgent().generate_deception(scenario, scenario.causal_level, context)

    async def _invoke_agent_b_claim(
        self,
        scenario: CausalScenario,
        context: DebateContext,
    ) -> Any:
        method = getattr(self.agent_b, "propose_hypothesis", None)
        if method is None:
            return await _MockScientistAgent().propose_hypothesis(scenario, scenario.causal_level, context)
        try:
            return await method(scenario, scenario.causal_level, context)
        except NotImplementedError:
            return await _MockScientistAgent().propose_hypothesis(scenario, scenario.causal_level, context)

    async def _invoke_agent_b(
        self,
        claim: str,
        scenario: CausalScenario,
        context: DebateContext,
    ) -> DetectionResult:
        method = getattr(self.agent_b, "analyze_claim", None)
        if method is None:
            return await _MockScientistAgent().analyze_claim(claim, scenario, scenario.causal_level, context)
        try:
            return await method(claim, scenario, scenario.causal_level, context)
        except NotImplementedError:
            return await _MockScientistAgent().analyze_claim(claim, scenario, scenario.causal_level, context)

    async def _invoke_agent_c(
        self,
        scenario: CausalScenario,
        context: DebateContext,
    ) -> AuditVerdict:
        method = getattr(self.agent_c, "evaluate_round", None)
        if method is None:
            return await _MockAuditorAgent().evaluate_round(scenario, context, scenario.causal_level)
        try:
            return await method(scenario, context, scenario.causal_level)
        except NotImplementedError:
            return await _MockAuditorAgent().evaluate_round(scenario, context, scenario.causal_level)

    async def _invoke_jury(
        self,
        scenario: CausalScenario,
        context: DebateContext,
    ) -> JuryVerdict:
        method = getattr(self.jury, "collect_votes", None)
        if method is None:
            mock = _MockJuryAggregator(self.config.get("models", {}).get("jury", {}))
            return await mock.collect_votes(scenario, context)
        try:
            return await method(scenario, context)
        except NotImplementedError:
            mock = _MockJuryAggregator(self.config.get("models", {}).get("jury", {}))
            return await mock.collect_votes(scenario, context)

    def _append_turn(
        self,
        context: DebateContext,
        speaker: str,
        phase: GamePhase,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        context.turns.append(
            DebateTurn(
                speaker=speaker,
                phase=phase,
                content=content,
                metadata=metadata or {},
            )
        )

    def _turn_to_dict(self, turn: DebateTurn) -> dict[str, Any]:
        return turn.to_dict()

    def _format_detection(self, result: DetectionResult) -> str:
        parts: list[str] = []
        if result.reasoning_chain:
            parts.append("；".join(result.reasoning_chain[:4]))
        if result.detected_fallacies:
            parts.append("识别到的问题：" + "、".join(result.detected_fallacies[:3]))
        if result.discovered_hidden_vars:
            parts.append("怀疑隐变量：" + "、".join(result.discovered_hidden_vars[:2]))
        parts.append(f"综合判断置信度 {result.confidence:.2f}")
        return " ".join(parts)

    def _resolve_winner(self, audit: AuditVerdict, jury_verdict: JuryVerdict) -> str:
        if audit.winner != "draw":
            return audit.winner
        if jury_verdict.final_winner != "draw" and jury_verdict.agreement_rate >= 0.67:
            return jury_verdict.final_winner
        return "draw"

    def _build_evolution_context(self) -> dict[str, Any] | None:
        if self.evolution_tracker is None or not self.evolution_tracker.records:
            return None
        last_records = self.evolution_tracker.export_history()["records"][-3:]
        arms_race = self.evolution_tracker.get_arms_race_index()

        # ── Agent A 专属：欺骗复杂度趋势 + 已被识破的策略 ──
        complexity_trend = self.evolution_tracker.get_deception_complexity_trend(window=5)
        avoided: list[str] = []
        if hasattr(self.agent_a, "strategy_history"):
            avoided = [
                s.removeprefix("avoid:")
                for s in self.agent_a.strategy_history
                if s.startswith("avoid:")
            ]

        # ── Agent C 专属：检测灵敏度趋势 + 已知欺骗模式 ──
        sensitivity_trend = self.evolution_tracker.get_detection_sensitivity_trend(window=5)
        known_patterns: list[str] = []
        if hasattr(self.agent_c, "known_patterns"):
            known_patterns = list(self.agent_c.known_patterns)

        return {
            "recent_strategies": last_records,
            "arms_race_index": arms_race,
            # Agent A 读取这些字段来调整欺骗策略
            "agent_a": {
                "deception_complexity_trend": complexity_trend,
                "detected_strategies": avoided,
            },
            # Agent C 读取这些字段来强化检测
            "agent_c": {
                "detection_sensitivity_trend": sensitivity_trend,
                "known_patterns": known_patterns,
            },
        }

    def _strategy_type_from_result(self, result: dict[str, Any]) -> StrategyType:
        strategy = str(result["agent_a_claim"].get("deception_strategy", "")).lower()
        mapping = {
            "l1-s1": StrategyType.CONFOUNDING,
            "l1-s2": StrategyType.REVERSE_CAUSE,
            "l1-s3": StrategyType.SELECTION_BIAS,
            "l2-s1": StrategyType.BACKDOOR_EXPLOIT,
            "l2-s2": StrategyType.INSTRUMENT_MISUSE,
            "l2-s3": StrategyType.NONCOMPLIANCE_EXPLOIT,
            "l2-s4": StrategyType.MEDIATOR_HIDE,
            "l3-s1": StrategyType.SCM_MANIPULATION,
            "l3-s2": StrategyType.COUNTERFACTUAL_DISTORTION,
            "l3-s3": StrategyType.NONCOMPLIANCE_EXPLOIT,
            "l3-s4": StrategyType.CONFOUNDING,
            "confounding": StrategyType.CONFOUNDING,
            "collider": StrategyType.COLLIDER,
            "selection_bias": StrategyType.SELECTION_BIAS,
            "mediator_hide": StrategyType.MEDIATOR_HIDE,
            "instrument_misuse": StrategyType.INSTRUMENT_MISUSE,
            "backdoor_exploit": StrategyType.BACKDOOR_EXPLOIT,
            "frontdoor_block": StrategyType.FRONTDOOR_BLOCK,
            "reverse_cause": StrategyType.REVERSE_CAUSE,
            "scm_manipulation": StrategyType.SCM_MANIPULATION,
            "counterfactual_distortion": StrategyType.COUNTERFACTUAL_DISTORTION,
            "noncompliance_exploit": StrategyType.NONCOMPLIANCE_EXPLOIT,
        }
        for key, value in mapping.items():
            if key in strategy:
                return value
        level = int(result["scenario"].causal_level)
        return {
            1: StrategyType.CONFOUNDING,
            2: StrategyType.MEDIATOR_HIDE,
            3: StrategyType.REVERSE_CAUSE,
        }.get(level, StrategyType.CONFOUNDING)
