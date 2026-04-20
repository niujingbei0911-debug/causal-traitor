import unittest

import pandas as pd

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC
from agents.jury import JuryAggregator
from agents.tool_executor import ToolExecutionResult
from benchmark.schema import VerdictLabel
from game.debate_engine import CausalScenario, DebateContext
from verifier.assumption_ledger import AssumptionLedger
from verifier.decision import VerifierDecision


class FakeStructuredLLMService:
    def __init__(self, payload):
        self.payload = payload
        self.backend = "dashscope"

    async def initialize(self):
        return None

    async def generate_json(self, *args, **kwargs):
        return None, self.payload


class FakeVerifierPipeline:
    def __init__(self, verdict: VerifierDecision):
        self.verdict = verdict
        self.calls: list[dict] = []

    def run(self, claim_text, *, scenario=None, transcript=None, tool_trace=None, tool_context=None):
        self.calls.append(
            {
                "claim_text": claim_text,
                "scenario": scenario,
                "transcript": transcript,
                "tool_trace": tool_trace,
                "tool_context": tool_context,
            }
        )
        return self.verdict


class ExplodingVerifierPipeline:
    def run(self, *args, **kwargs):
        raise RuntimeError("boom")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "models": {
                "agent_a": {"name": "mock-a"},
                "agent_b": {"name": "mock-b"},
                "jury": {
                    "models": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
                    "voting": "weighted",
                },
            }
        }
        self.data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1, 0, 1],
                "Z": [0, 1, 0, 1, 0, 1],
                "Y": [0.1, 0.5, 1.1, 1.6, 0.0, 1.4],
            }
        )
        self.scenario = CausalScenario(
            scenario_id="education_income",
            description="教育影响收入",
            true_dag={"Z": ["X"], "X": ["Y"]},
            variables=["X", "Z", "Y"],
            hidden_variables=["U"],
            data=self.data,
            causal_level=2,
            difficulty=0.5,
        )
        self.public_scenario = self.scenario.to_public()

    async def test_agent_a_generates_deception(self):
        agent = AgentA(self.config)
        await agent.initialize()
        response = await agent.generate_deception(self.scenario, level=2)
        self.assertTrue(response.content)
        self.assertTrue(response.causal_claim)
        self.assertTrue(response.deception_strategy.startswith("L2"))

    def test_agent_a_rotates_strategies_per_level(self):
        agent = AgentA(self.config)
        sequence = []
        for level in [1, 2, 3, 1, 2, 3, 1, 2, 3]:
            strategy = agent._choose_strategy(level)
            sequence.append(strategy)
            agent.strategy_history.append(strategy)

        self.assertEqual(sequence[0::3], ["L1-S1", "L1-S3", "L1-S2"])
        self.assertEqual(sequence[1::3], ["L2-S2", "L2-S4", "L2-S1"])
        self.assertEqual(sequence[2::3], ["L3-S2", "L3-S3", "L3-S1"])

    async def test_agent_a_adapt_strategy_tracks_detected_strategy_with_current_feedback_keys(self):
        agent = AgentA(self.config)

        result = await agent.adapt_strategy(
            {
                "detected": True,
                "strategy_used": "L2-S4",
                "level": 2,
            }
        )

        self.assertIn("avoid:L2-S4", agent.strategy_history)
        self.assertEqual(result["history_length"], 1)

    async def test_agent_b_analyzes_claim(self):
        agent = AgentB(self.config)
        await agent.initialize()
        result = await agent.analyze_claim(
            claim="X 对 Y 的工具变量效应已经被 Z 证明，无需控制其他因素。",
            scenario=self.public_scenario,
            level=2,
        )
        self.assertGreater(result.confidence, 0.0)
        self.assertTrue(result.tools_used)
        self.assertTrue(result.reasoning_chain)

    async def test_agent_b_does_not_treat_adjustment_variable_as_instrument_without_iv_cues(self):
        agent = AgentB(self.config)
        await agent.initialize()
        result = await agent.analyze_claim(
            claim="After controlling for Z, the causal effect of X on Y is identified.",
            scenario=self.public_scenario,
            level=2,
        )

        self.assertNotIn("iv_estimation", result.tools_used)

    async def test_agent_b_does_not_flag_supportive_iv_claim_as_over_doubting_valid_instrument(self):
        agent = AgentB(self.config)
        await agent.initialize()
        result = await agent.analyze_claim(
            claim="Instrument Z identifies the causal effect of X on Y.",
            scenario=self.public_scenario,
            level=2,
        )

        self.assertNotIn("过度质疑有效工具变量", result.detected_fallacies)

    async def test_agent_b_propose_hypothesis_uses_public_description_focus_pair(self):
        agent = AgentB(self.config)
        self.public_scenario.description = (
            "Observed L2 benchmark case over Z, X, Y. "
            "Evaluate claims about X and Y using only the public evidence in this view."
        )

        claim = await agent.propose_hypothesis(self.public_scenario, level=2)

        self.assertEqual(claim.causal_claim, "X 可能导致 Y")
        self.assertIn("候选工具变量为 Z", claim.evidence)

    def test_agent_b_infer_focus_variables_rejects_empty_variable_list(self):
        agent = AgentB(self.config)
        with self.assertRaises(ValueError):
            agent._infer_focus_variables("X causes Y", [], scenario=None)

    async def test_jury_collects_votes(self):
        jury = JuryAggregator(self.config)
        await jury.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "content": "X 必然导致 Y。"},
                {"speaker": "agent_b", "content": "我检查了混杂和工具变量强度，结论并不稳固。"},
            ],
        )
        verdict = await jury.collect_votes(self.public_scenario, context)
        self.assertEqual(len(verdict.votes), 3)
        self.assertIn(verdict.final_winner, {"agent_a", "agent_b", "draw"})

    async def test_agent_c_evaluates_round(self):
        agent = AgentC(self.config)
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "content": "X 必然导致 Y，无需控制其他因素，相关性已经证明因果。"},
                {"speaker": "agent_b", "content": "这个结论忽略了混杂，需要做后门调整并检查工具变量 Z 的强度。"},
            ],
        )
        verdict = await agent.evaluate_round(self.public_scenario, context, level=2)
        self.assertIn(verdict.winner, {"agent_a", "agent_b", "draw"})
        self.assertTrue(verdict.reasoning)
        self.assertGreaterEqual(verdict.jury_consensus, 0.0)
        self.assertIn(verdict.verdict_label, {"valid", "invalid", "unidentifiable"})
        self.assertTrue(verdict.verifier_verdict)
        self.assertIsInstance(verdict.assumption_ledger, list)
        self.assertIsInstance(verdict.tool_trace, list)
        self.assertIsInstance(verdict.tools_used, list)
        if verdict.tool_trace:
            self.assertTrue(
                {
                    "tool_name",
                    "status",
                    "summary",
                    "supports_claim",
                    "evidence_direction",
                    "error",
                    "supports_assumptions",
                    "contradicts_assumptions",
                }
                <= set(verdict.tool_trace[0])
            )

    def test_agent_c_extract_b_confidence_tolerates_missing_metadata_attribute(self):
        agent = AgentC(self.config)
        self.assertEqual(agent._extract_b_confidence({}), 0.0)

    async def test_agent_c_default_path_calls_verifier_pipeline(self):
        agent = AgentC(self.config)
        fake_pipeline = FakeVerifierPipeline(
            VerifierDecision(
                label=VerdictLabel.VALID,
                confidence=0.83,
                assumption_ledger=AssumptionLedger([]),
                tool_trace=[{"tool_name": "mock_tool", "supports_claim": True}],
                reasoning_summary="Verifier pipeline accepted the claim.",
            )
        )
        agent.attach_verifier_pipeline(fake_pipeline)
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "phase": "challenge", "content": "X causes Y."},
                {"speaker": "agent_b", "phase": "rebuttal", "content": "Check confounding first."},
            ],
        )

        verdict = await agent.evaluate_round(self.public_scenario, context, level=2)

        self.assertEqual(len(fake_pipeline.calls), 1)
        self.assertEqual(verdict.verdict_label, "valid")
        self.assertEqual(verdict.winner, "agent_a")
        self.assertAlmostEqual(verdict.verifier_confidence, 0.83, places=3)
        self.assertTrue(verdict.verifier_verdict)
        self.assertEqual(verdict.verifier_verdict["final_verdict"], "valid")
        self.assertEqual(verdict.verifier_verdict["identification_status"], "identified")
        self.assertIn("missing_information_spec", verdict.verifier_verdict)
        self.assertIs(fake_pipeline.calls[0]["scenario"], self.public_scenario)

    async def test_agent_c_verifier_first_defers_tool_execution_to_pipeline(self):
        agent = AgentC(self.config)
        fake_pipeline = FakeVerifierPipeline(
            VerifierDecision(
                label=VerdictLabel.UNIDENTIFIABLE,
                confidence=0.61,
                assumption_ledger=AssumptionLedger([]),
                tool_trace=[],
                reasoning_summary="Verifier sees mixed evidence.",
            )
        )
        agent.attach_verifier_pipeline(fake_pipeline)
        await agent.initialize()
        agent.tool_executor.execute_for_claim = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("AgentC should not pre-run ToolExecutor before verifier.pipeline.")
        )
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "phase": "challenge", "content": "X causes Y."},
                {"speaker": "agent_b", "phase": "rebuttal", "content": "The evidence undermines that claim."},
            ],
        )

        await agent.evaluate_round(self.public_scenario, context, level=2)

        forwarded_trace = fake_pipeline.calls[0]["tool_trace"]
        self.assertIsNone(forwarded_trace)
        self.assertEqual(fake_pipeline.calls[0]["tool_context"]["proxy_variables"], [])

    async def test_agent_c_maps_unidentifiable_verdict_to_draw(self):
        agent = AgentC(self.config)
        fake_pipeline = FakeVerifierPipeline(
            VerifierDecision(
                label=VerdictLabel.UNIDENTIFIABLE,
                confidence=0.74,
                assumption_ledger=AssumptionLedger([]),
                tool_trace=[],
                reasoning_summary="Verifier abstains because multiple compatible models remain.",
            )
        )
        agent.attach_verifier_pipeline(fake_pipeline)
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "phase": "challenge", "content": "X causes Y for everyone."},
                {"speaker": "agent_b", "phase": "rebuttal", "content": "The data do not uniquely identify that claim."},
            ],
        )

        verdict = await agent.evaluate_round(self.public_scenario, context, level=2)

        self.assertEqual(verdict.verdict_label, "unidentifiable")
        self.assertEqual(verdict.winner, "draw")

    async def test_agent_c_exposes_pipeline_tool_trace_verbatim(self):
        agent = AgentC(self.config)
        fake_pipeline = FakeVerifierPipeline(
            VerifierDecision(
                label=VerdictLabel.UNIDENTIFIABLE,
                confidence=0.61,
                assumption_ledger=AssumptionLedger([]),
                tool_trace=[
                    {
                        "tool_name": "backdoor_adjustment_check",
                        "claim_stance": "pro_causal",
                        "supports_assumptions": [],
                        "contradicts_assumptions": ["valid adjustment set"],
                    }
                ],
                reasoning_summary="Verifier sees mixed evidence.",
            )
        )
        agent.attach_verifier_pipeline(fake_pipeline)
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "phase": "challenge", "content": "X causes Y."},
                {"speaker": "agent_b", "phase": "rebuttal", "content": "The adjustment set is invalid."},
            ],
        )

        await agent.evaluate_round(self.public_scenario, context, level=2)

        exposed_trace = fake_pipeline.verdict.tool_trace
        rebuttal_entry = next(
            entry for entry in exposed_trace
            if entry["tool_name"] == "backdoor_adjustment_check"
        )
        self.assertEqual(rebuttal_entry["claim_stance"], "pro_causal")
        self.assertIn("valid adjustment set", rebuttal_entry["contradicts_assumptions"])

    async def test_agent_c_verifier_first_does_not_silently_fallback_to_legacy(self):
        agent = AgentC(self.config)
        agent.attach_verifier_pipeline(ExplodingVerifierPipeline())
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "phase": "challenge", "content": "X causes Y."},
                {"speaker": "agent_b", "phase": "rebuttal", "content": "Not enough evidence."},
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "legacy fallback is disabled"):
            await agent.evaluate_round(self.public_scenario, context, level=2)

    async def test_agent_a_uses_structured_llm_decision(self):
        agent = AgentA(self.config)
        agent.attach_llm_service(
            FakeStructuredLLMService(
                {
                    "causal_claim": "X 会误导性地影响 Y",
                    "content": "这是 LLM 主导生成的欺骗论证。",
                    "evidence": ["LLM-evidence"],
                    "deception_strategy": "L2-S4",
                }
            )
        )
        response = await agent.generate_deception(self.scenario, level=2)
        self.assertEqual(response.causal_claim, "X 会误导性地影响 Y")
        self.assertEqual(response.deception_strategy, "L2-S4")
        self.assertIn("LLM-evidence", response.evidence)

    async def test_agent_b_uses_structured_llm_decision(self):
        agent = AgentB(self.config)
        agent.attach_llm_service(
            FakeStructuredLLMService(
                {
                    "detected_fallacies": ["llm_fallacy"],
                    "discovered_hidden_vars": ["LLM_U"],
                    "confidence": 0.91,
                    "reasoning_chain": ["LLM says the claim is flawed."],
                    "tools_used": ["llm_tool"],
                }
            )
        )
        result = await agent.analyze_claim(
            claim="X 对 Y 的工具变量效应已经被 Z 证明，无需控制其他因素。",
            scenario=self.public_scenario,
            level=2,
        )
        self.assertIn("llm_fallacy", result.detected_fallacies)
        self.assertIn("LLM_U", result.discovered_hidden_vars)
        self.assertEqual(result.confidence, 0.91)
        self.assertIn("LLM says the claim is flawed.", result.reasoning_chain)

    async def test_agent_c_uses_structured_llm_verdict(self):
        legacy_config = {**self.config, "agent_c": {"mode": "legacy"}}
        agent = AgentC(legacy_config)
        agent.attach_llm_service(
            FakeStructuredLLMService(
                {
                    "winner": "agent_a",
                    "causal_validity_score": 0.73,
                    "argument_quality_a": 0.8,
                    "argument_quality_b": 0.41,
                    "identified_issues": ["llm_issue"],
                    "reasoning": "LLM chooses agent_a after weighing the evidence.",
                }
            )
        )
        await agent.initialize()
        context = DebateContext(
            scenario=self.public_scenario,
            turns=[
                {"speaker": "agent_a", "content": "X 必然导致 Y。"},
                {"speaker": "agent_b", "content": "这个结论忽略了混杂，需要做后门调整。"},
            ],
        )
        verdict = await agent.evaluate_round(self.public_scenario, context, level=2)
        self.assertEqual(verdict.winner, "agent_a")
        self.assertIn("llm_issue", verdict.identified_issues)
        self.assertEqual(verdict.reasoning, "LLM chooses agent_a after weighing the evidence.")
        self.assertIsNone(verdict.verdict_label)


if __name__ == "__main__":
    unittest.main()
