import unittest

import pandas as pd

from agents.tool_executor import ToolExecutionResult, ToolExecutor
from benchmark.generator import BenchmarkGenerator
from game.debate_engine import CausalScenario
from verifier.claim_parser import parse_claim


class ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = ToolExecutor({})
        self.data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1, 0, 1, 0, 1],
                "Z": [0, 1, 0, 1, 0, 1, 0, 1],
                "M": [0.1, 0.5, 0.7, 1.0, 0.2, 0.8, 0.2, 0.9],
                "Y": [0.0, 0.2, 1.0, 1.2, 0.1, 1.1, 0.2, 1.3],
            }
        )
        self.scenario = CausalScenario(
            scenario_id="tool_executor_case",
            description="test tool executor",
            true_dag={"Z": ["X"], "X": ["M", "Y"], "M": ["Y"]},
            variables=["X", "Z", "M", "Y"],
            hidden_variables=["U"],
            data=self.data,
            causal_level=2,
            difficulty=0.4,
            true_scm={"graph": {"Z": ["X"], "X": ["M", "Y"], "M": ["Y"]}},
            ground_truth={"treatment": "X", "outcome": "Y", "instrument": "Z", "mediator": "M"},
        )
        self.public_scenario = self.scenario.to_public()

    def test_execute_for_claim_outputs_standardized_tool_trace_and_evidence_component(self):
        report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="X causes Y and Z is a valid instrument with mediator M.",
            level=2,
            context={"has_instrument": True, "has_mediator": True},
        )

        self.assertIn("iv_estimation", report["selected_tools"])
        self.assertIn("frontdoor_estimation", report["selected_tools"])
        self.assertNotIn("causal_graph_validator", report["selected_tools"])
        self.assertTrue(report["results"])
        self.assertTrue(report["successful_tools"])
        self.assertTrue(report["tool_trace"])
        self.assertIn("evidence_component", report)
        self.assertEqual(
            report["evidence_component"]["heuristic_support"],
            report["support_score"],
        )
        first_entry = report["tool_trace"][0]
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
            <= set(first_entry)
        )

    def test_public_side_does_not_fallback_to_true_graph_or_true_scm(self):
        self.assertIsNone(self.executor._get_graph({}))
        self.assertIsNone(self.executor._get_scm({}))

        l3_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="We need a counterfactual answer for X and Y.",
            level=3,
            context={},
        )

        self.assertIn("counterfactual_inference", l3_report["selected_tools"])
        self.assertIn("scm_identification_test", l3_report["selected_tools"])
        self.assertIn("ett_computation", l3_report["selected_tools"])
        self.assertIn("abduction_action_prediction", l3_report["selected_tools"])

        graph_leak_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Check whether the observed graph is a DAG.",
            level=1,
            context={"graph": self.scenario.true_dag},
        )
        graph_base_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Check whether the observed graph is a DAG.",
            level=1,
            context={},
        )
        scm_leak_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Compute a counterfactual answer for X and Y.",
            level=3,
            context={"scm": self.scenario.true_scm, "scm_model": self.scenario.true_scm},
        )
        scm_base_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Compute a counterfactual answer for X and Y.",
            level=3,
            context={},
        )

        self.assertEqual(graph_leak_report["selected_tools"], graph_base_report["selected_tools"])
        self.assertEqual(scm_leak_report["selected_tools"], scm_base_report["selected_tools"])

    def test_tool_executor_ignores_caller_supplied_public_graph_or_public_scm(self):
        self.assertIsNone(self.executor._get_graph({"public_graph": {"Z": ["X"], "X": ["Y"]}}))
        self.assertIsNone(self.executor._get_scm({"public_scm": {"graph": {"X": ["Y"]}}}))

        graph_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Check whether the observed graph is a DAG.",
            level=1,
            context={"public_graph": {"Z": ["X"], "X": ["Y"]}},
        )
        scm_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Compute a counterfactual answer for X and Y.",
            level=3,
            context={
                "public_scm": {
                    "graph": {"X": ["Y"]},
                    "coefficients": {"X": {"intercept": 0.0}, "Y": {"intercept": 0.0, "X": 1.0}},
                }
            },
        )

        base_graph_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Check whether the observed graph is a DAG.",
            level=1,
            context={},
        )
        base_scm_report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="Compute a counterfactual answer for X and Y.",
            level=3,
            context={},
        )

        self.assertEqual(graph_report["selected_tools"], base_graph_report["selected_tools"])
        self.assertEqual(scm_report["selected_tools"], base_scm_report["selected_tools"])

    def test_iv_family_public_sample_does_not_autoselect_iv_tool_without_claim_signal(self):
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_iv_family",
            difficulty=0.4,
            seed=0,
        )

        report = self.executor.execute_for_claim(
            scenario=sample.public,
            claim="Estimate the causal effect of policy_uptake on yield_score.",
            level=2,
            context={},
        )

        self.assertNotIn("iv_estimation", report["selected_tools"])

    def test_generated_valid_benchmark_claims_receive_real_support_assumptions(self):
        generator = BenchmarkGenerator(seed=17)
        cases = (
            ("l1_proxy_disambiguation_family", 0, {"proxy sufficiency"}),
            ("l2_valid_backdoor_family", 8, {"valid adjustment set", "positivity"}),
            ("l3_mediation_abduction_family", 0, {"cross-world consistency", "counterfactual model uniqueness"}),
        )

        for family_name, seed, expected_assumptions in cases:
            with self.subTest(family_name=family_name, seed=seed):
                sample = generator.generate_benchmark_sample(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                level = int(str(sample.claim.causal_level).replace("L", ""))
                report = self.executor.execute_for_claim(
                    scenario=sample.public,
                    claim=sample.claim.claim_text,
                    level=level,
                    context={"claim_stance": "pro_causal"},
                )

                supported = {
                    assumption
                    for entry in report["tool_trace"]
                    for assumption in entry["supports_assumptions"]
                }

                self.assertTrue(expected_assumptions <= supported)

    def test_invalid_adjustment_claim_does_not_receive_valid_adjustment_support(self):
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_backdoor_family",
            difficulty=0.4,
            seed=17,
        )

        report = self.executor.execute_for_claim(
            scenario=sample.public,
            claim=sample.claim.claim_text,
            level=2,
            context={"claim_stance": "pro_causal"},
        )

        contradicts = {
            assumption
            for entry in report["tool_trace"]
            for assumption in entry["contradicts_assumptions"]
        }
        self.assertIn("valid adjustment set", contradicts)

    def test_backdoor_adjustment_check_without_public_graph_only_promotes_conservative_claim_support(self):
        parsed_claim = parse_claim(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified."
        )
        result = ToolExecutionResult(
            tool_name="backdoor_adjustment_check",
            success=True,
            output={
                "adjustment_set": ["pretest_score"],
                "supports_adjustment_set": True,
                "is_valid_adjustment": None,
            },
        )

        supports, contradicts = self.executor._extract_tool_assumptions(
            result,
            claim="After controlling for pretest_score, the causal effect of exposure on recovery is identified.",
            parsed_claim=parsed_claim,
        )

        self.assertIn("valid adjustment set", supports)
        self.assertNotIn("valid adjustment set", contradicts)

    def test_public_measurement_semantics_are_not_upgraded_into_tool_trace_evidence(self):
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_iv_family",
            difficulty=0.4,
            seed=0,
        )

        report = self.executor.execute_for_claim(
            scenario=sample.public,
            claim=sample.claim.claim_text,
            level=2,
            context={"claim_stance": "pro_causal"},
        )

        self.assertNotIn(
            "public_semantics_check",
            {entry["tool_name"] for entry in report["tool_trace"]},
        )

    def test_role_token_match_does_not_confuse_clinical_stage_with_selection(self):
        scenario = self.public_scenario.__class__(
            scenario_id="clinical-stage-case",
            description="Observed L2 case over treatment_flag, clinical_stage, outcome_score. Evaluate claims about treatment_flag and outcome_score using only the public evidence in this view.",
            variables=["treatment_flag", "clinical_stage", "outcome_score"],
            observed_data=pd.DataFrame(
                {
                    "treatment_flag": [0, 1, 0, 1],
                    "clinical_stage": [0, 1, 1, 0],
                    "outcome_score": [0.2, 0.9, 0.3, 1.0],
                }
            ),
            causal_level=2,
        )

        context = self.executor._merge_context(
            {"claim_stance": "pro_causal"},
            "After controlling for clinical_stage, the causal effect of treatment_flag on outcome_score is identified.",
            scenario=scenario,
        )

        self.assertEqual(context["adjustment_set"], ["clinical_stage"])
        self.assertFalse(context.get("has_selection"))

    def test_sensitivity_analysis_is_not_promoted_to_no_unobserved_confounding_support(self):
        report = self.executor.execute_for_claim(
            scenario=self.public_scenario,
            claim="After controlling for Z, the causal effect of X on Y is identified.",
            level=2,
            context={"claim_stance": "pro_causal"},
        )

        sensitivity_entries = [
            entry
            for entry in report["tool_trace"]
            if entry["tool_name"] == "sensitivity_analysis"
        ]

        self.assertTrue(sensitivity_entries)
        self.assertNotIn(
            "no unobserved confounding",
            {
                assumption
                for entry in sensitivity_entries
                for assumption in entry["supports_assumptions"]
            },
        )

    def test_counterfactual_bridge_support_is_not_promoted_for_false_uniqueness_overclaim(self):
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l3_mediation_abduction_family",
            difficulty=0.4,
            seed=1,
        )

        report = self.executor.execute_for_claim(
            scenario=sample.public,
            claim=sample.claim.claim_text,
            level=3,
            context={"claim_stance": "pro_causal"},
        )

        bridge_entries = [
            entry
            for entry in report["tool_trace"]
            if entry["tool_name"] == "counterfactual_bridge_check"
        ]

        self.assertTrue(bridge_entries)
        for entry in bridge_entries:
            self.assertNotIn("cross-world consistency", entry["supports_assumptions"])
            self.assertNotIn("counterfactual model uniqueness", entry["supports_assumptions"])

    def test_safe_python_executor(self):
        result = self.executor.execute_python(
            "answer = sum(values)\nmean_value = round(answer / len(values), 2)",
            extra_context={"values": [1, 2, 3]},
        )
        self.assertEqual(result["answer"], 6)
        self.assertEqual(result["mean_value"], 2.0)

        with self.assertRaises(ValueError):
            self.executor.execute_python("import os\nx = 1")


if __name__ == "__main__":
    unittest.main()
