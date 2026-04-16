import json
import unittest
from unittest.mock import patch

import pandas as pd

from agents.tool_executor import ToolExecutor
from benchmark.attacks import ATTACK_TEMPLATE_REGISTRY, generate_attack_sample
from benchmark.generator import BenchmarkGenerator
from benchmark.schema import PublicCausalInstance, VerdictLabel, WitnessKind
from benchmark.graph_families import generate_graph_family, list_graph_families
from verifier.assumption_ledger import AssumptionLedger, build_assumption_ledger
from verifier.claim_parser import ClaimParser, parse_claim
from verifier.countermodel_search import CountermodelSearchResult, search_countermodels
from verifier.decision import VerifierDecision, decide_verdict
from verifier.outputs import ClaimPolarity, ClaimStrength, ParsedClaim, QueryType
from verifier.pipeline import VerifierPipeline, run_verifier_pipeline


class FakeToolRunner:
    def __init__(self, tool_trace):
        self.tool_trace = tool_trace
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.tool_trace)


def _build_public_intervention_scenario() -> PublicCausalInstance:
    observed_data = pd.DataFrame(
        {
            "assignment_lottery": [0, 0, 0, 1, 1, 1, 1, 0],
            "exposure": [0, 0, 1, 0, 1, 1, 1, 0],
            "recovery": [0.1, 0.2, 0.6, 0.3, 0.8, 1.0, 0.9, 0.2],
            "proxy_signal": [0.2, 0.1, 0.4, 0.5, 0.8, 0.9, 0.7, 0.3],
        }
    )
    return PublicCausalInstance(
        scenario_id="public_iv_case",
        description="Public intervention case with observed IV-style patterns.",
        variables=list(observed_data.columns),
        observed_data=observed_data,
        causal_level=2,
        metadata={
            "role_bindings": {
                "instrument": "assignment_lottery",
                "treatment": "exposure",
                "outcome": "recovery",
                "proxy": "proxy_signal",
            },
            "proxy_variables": ["proxy_signal"],
        },
    )


def _expected_query_type(query_name: str) -> QueryType:
    association = {"association_strength", "causal_direction", "proxy_adjusted_claim", "selection_bias_check"}
    intervention = {"average_treatment_effect", "interventional_effect", "instrumental_variable_claim"}
    counterfactual = {
        "unit_level_counterfactual",
        "effect_of_treatment_on_treated",
        "abduction_action_prediction",
    }
    if query_name in association:
        return QueryType.ASSOCIATION
    if query_name in intervention:
        return QueryType.INTERVENTION
    if query_name in counterfactual:
        return QueryType.COUNTERFACTUAL
    raise ValueError(f"Unknown benchmark query type: {query_name!r}")


class ClaimParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ClaimParser()

    def test_parser_returns_structured_claim_object(self) -> None:
        parsed = self.parser.parse(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified."
        )

        self.assertIsInstance(parsed, ParsedClaim)
        self.assertEqual(parsed.query_type, QueryType.INTERVENTION)
        self.assertEqual(parsed.treatment, "exposure")
        self.assertEqual(parsed.outcome, "recovery")
        self.assertEqual(parsed.claim_polarity, ClaimPolarity.POSITIVE)
        self.assertEqual(parsed.claim_strength, ClaimStrength.STRONG)

        payload = parsed.to_dict()
        self.assertEqual(payload["query_type"], "intervention")
        self.assertEqual(payload["claim_strength"], "strong")
        json.dumps(payload)

    def test_parser_covers_l1_association_example(self) -> None:
        parsed = parse_claim(
            "There is no serious hidden-variable explanation here, so training_intensity itself drives income_score."
        )

        self.assertEqual(parsed.query_type, QueryType.ASSOCIATION)
        self.assertEqual(parsed.treatment, "training_intensity")
        self.assertEqual(parsed.outcome, "income_score")
        self.assertEqual(parsed.claim_strength, ClaimStrength.STRONG)
        self.assertIn("no unobserved confounding", parsed.mentioned_assumptions)
        self.assertTrue(parsed.needs_abstention_check)
        self.assertEqual(parsed.rhetorical_strategy, "confounder_denial")

    def test_parser_covers_l2_intervention_examples(self) -> None:
        backdoor_claim = self.parser.parse(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified."
        )
        iv_claim = self.parser.parse(
            "Using assignment_lottery as an instrument is enough to recover the causal effect of exposure on recovery."
        )

        self.assertEqual(backdoor_claim.query_type, QueryType.INTERVENTION)
        self.assertIn("valid adjustment set", backdoor_claim.mentioned_assumptions)
        self.assertIn("consistency", backdoor_claim.implied_assumptions)
        self.assertIn("positivity", backdoor_claim.implied_assumptions)
        self.assertIn("no unobserved confounding", backdoor_claim.implied_assumptions)

        self.assertEqual(iv_claim.query_type, QueryType.INTERVENTION)
        self.assertEqual(iv_claim.treatment, "exposure")
        self.assertEqual(iv_claim.outcome, "recovery")
        self.assertIn("instrument relevance", iv_claim.mentioned_assumptions)
        self.assertIn("exclusion restriction", iv_claim.implied_assumptions)
        self.assertIn("instrument independence", iv_claim.implied_assumptions)
        self.assertEqual(iv_claim.rhetorical_strategy, "instrumental_variable_appeal")

    def test_parser_covers_l3_counterfactual_example(self) -> None:
        parsed = self.parser.parse(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
        )

        self.assertEqual(parsed.query_type, QueryType.COUNTERFACTUAL)
        self.assertEqual(parsed.treatment, "therapy_flag")
        self.assertEqual(parsed.outcome, "recovery")
        self.assertEqual(parsed.claim_strength, ClaimStrength.ABSOLUTE)
        self.assertIn("cross-world consistency", parsed.mentioned_assumptions)
        self.assertIn("counterfactual model uniqueness", parsed.mentioned_assumptions)
        self.assertIn("consistency", parsed.implied_assumptions)
        self.assertIn("positivity", parsed.implied_assumptions)
        self.assertTrue(parsed.needs_abstention_check)
        self.assertEqual(parsed.rhetorical_strategy, "false_uniqueness")

    def test_parser_handles_real_benchmark_attack_templates(self) -> None:
        mismatches = []
        abstention_failures = []

        for family_name in list_graph_families():
            blueprint = generate_graph_family(family_name, seed=17)
            for attack_name, template in ATTACK_TEMPLATE_REGISTRY.items():
                for label in template.compatible_labels:
                    if not template.is_compatible(blueprint, label):
                        continue
                    sample = generate_attack_sample(
                        blueprint,
                        gold_label=label,
                        attack_name=attack_name,
                        seed=17,
                    )
                    parsed = parse_claim(sample.claim_text)

                    if (
                        parsed.treatment != sample.target_variables["treatment"]
                        or parsed.outcome != sample.target_variables["outcome"]
                    ):
                        mismatches.append(
                            (
                                family_name,
                                attack_name,
                                label,
                                sample.target_variables,
                                {"treatment": parsed.treatment, "outcome": parsed.outcome},
                                sample.claim_text,
                            )
                        )
                    if not parsed.needs_abstention_check:
                        abstention_failures.append((family_name, attack_name, label, sample.claim_text))

        self.assertFalse(mismatches, msg=json.dumps(mismatches[:5], ensure_ascii=False, indent=2))
        self.assertFalse(
            abstention_failures,
            msg=json.dumps(abstention_failures[:5], ensure_ascii=False, indent=2),
        )

    def test_parser_handles_truthful_valid_benchmark_samples_for_l2_and_l3(self) -> None:
        generator = BenchmarkGenerator(seed=28)
        cases = (
            ("l2_valid_backdoor_family", 28, "average_treatment_effect", QueryType.INTERVENTION),
            ("l2_valid_backdoor_family", 30, "interventional_effect", QueryType.INTERVENTION),
            ("l3_mediation_abduction_family", 28, "unit_level_counterfactual", QueryType.COUNTERFACTUAL),
            ("l3_mediation_abduction_family", 30, "abduction_action_prediction", QueryType.COUNTERFACTUAL),
            ("l3_mediation_abduction_family", 18, "abduction_action_prediction", QueryType.COUNTERFACTUAL),
        )

        for family_name, seed, expected_query_name, expected_query_type in cases:
            with self.subTest(family_name=family_name, seed=seed):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                parsed = parse_claim(claim.claim_text)

                self.assertIs(claim.gold_label, VerdictLabel.VALID)
                self.assertEqual(claim.query_type, expected_query_name)
                self.assertEqual(parsed.query_type, expected_query_type)
                self.assertEqual(parsed.treatment, claim.target_variables["treatment"])
                self.assertEqual(parsed.outcome, claim.target_variables["outcome"])
                self.assertTrue(parsed.needs_abstention_check)

    def test_parser_preserves_treatment_outcome_for_generated_benchmark_samples(self) -> None:
        generator = BenchmarkGenerator(seed=17)
        mismatches = []
        families = (
            "l1_latent_confounding_family",
            "l1_selection_bias_family",
            "l1_proxy_disambiguation_family",
            "l2_valid_backdoor_family",
            "l2_invalid_iv_family",
            "l3_counterfactual_ambiguity_family",
            "l3_mediation_abduction_family",
        )

        for family_name in families:
            for seed in range(24):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                parsed = parse_claim(claim.claim_text)
                if (
                    parsed.treatment != claim.target_variables["treatment"]
                    or parsed.outcome != claim.target_variables["outcome"]
                ):
                    mismatches.append(
                        (
                            family_name,
                            seed,
                            claim.gold_label.value,
                            claim.query_type,
                            claim.meta.get("attack_name"),
                            claim.target_variables,
                            {"treatment": parsed.treatment, "outcome": parsed.outcome},
                            claim.claim_text,
                        )
                    )

        self.assertFalse(mismatches, msg=json.dumps(mismatches[:10], ensure_ascii=False, indent=2))

    def test_parser_preserves_query_layer_for_generated_nonvalid_benchmark_samples(self) -> None:
        generator = BenchmarkGenerator(seed=17)
        mismatches = []
        families = (
            "l1_latent_confounding_family",
            "l1_selection_bias_family",
            "l1_proxy_disambiguation_family",
            "l2_valid_backdoor_family",
            "l2_invalid_iv_family",
            "l3_counterfactual_ambiguity_family",
            "l3_mediation_abduction_family",
        )

        for family_name in families:
            for seed in range(24):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                if claim.gold_label is VerdictLabel.VALID:
                    continue
                parsed = parse_claim(claim.claim_text)
                expected_query_type = _expected_query_type(claim.query_type)
                if parsed.query_type is not expected_query_type:
                    mismatches.append(
                        (
                            family_name,
                            seed,
                            claim.gold_label.value,
                            claim.query_type,
                            expected_query_type.value,
                            parsed.query_type.value,
                            claim.meta.get("attack_name"),
                            claim.claim_text,
                        )
                    )

        self.assertFalse(mismatches, msg=json.dumps(mismatches[:10], ensure_ascii=False, indent=2))


class AssumptionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ClaimParser()

    def test_ledger_is_machine_readable(self) -> None:
        parsed = self.parser.parse(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified."
        )

        ledger = build_assumption_ledger(parsed)
        payload = ledger.to_dict()

        self.assertIsInstance(ledger, AssumptionLedger)
        self.assertIn("assumption_ledger", payload)
        self.assertIn("supported_count", payload)
        self.assertIn("contradicted_count", payload)
        self.assertIn("unresolved_count", payload)
        self.assertTrue(payload["assumption_ledger"])
        self.assertTrue(
            {"name", "source", "category", "status", "note"} <= set(payload["assumption_ledger"][0])
        )
        json.dumps(payload)

    def test_ledger_covers_confounding_assumptions(self) -> None:
        parsed = parse_claim(
            "There is no serious hidden-variable explanation here, so training_intensity itself drives income_score."
        )
        ledger = build_assumption_ledger(parsed).by_name()

        self.assertIn("no unobserved confounding", ledger)
        self.assertEqual(ledger["no unobserved confounding"].category, "confounding")
        self.assertEqual(ledger["no unobserved confounding"].source, "claim explicit")
        self.assertEqual(ledger["no unobserved confounding"].status.value, "unresolved")

    def test_ledger_covers_iv_and_selection_assumptions(self) -> None:
        iv_parsed = self.parser.parse(
            "assignment_lottery affects recovery only through exposure, so using assignment_lottery as an instrument is enough to recover the causal effect of exposure on recovery."
        )
        iv_ledger = build_assumption_ledger(iv_parsed).by_name()

        self.assertIn("instrument relevance", iv_ledger)
        self.assertIn("exclusion restriction", iv_ledger)
        self.assertIn("instrument independence", iv_ledger)
        self.assertEqual(iv_ledger["instrument relevance"].category, "iv")
        self.assertEqual(iv_ledger["instrument relevance"].status.value, "unresolved")
        self.assertEqual(iv_ledger["exclusion restriction"].status.value, "unresolved")
        self.assertEqual(iv_ledger["instrument independence"].status.value, "unresolved")

        selection_parsed = self.parser.parse(
            "Within the selected sample, the link between adoption_level and stability_index is clean enough to read causally."
        )
        selection_ledger = build_assumption_ledger(selection_parsed).by_name()

        self.assertIn("no selection bias", selection_ledger)
        self.assertEqual(selection_ledger["no selection bias"].category, "selection")
        self.assertEqual(selection_ledger["no selection bias"].status.value, "contradicted")

    def test_ledger_covers_counterfactual_assumptions(self) -> None:
        parsed = self.parser.parse(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
        )
        ledger = build_assumption_ledger(parsed)
        ledger_by_name = ledger.by_name()

        self.assertIn("cross-world consistency", ledger_by_name)
        self.assertIn("counterfactual model uniqueness", ledger_by_name)
        self.assertEqual(ledger_by_name["cross-world consistency"].category, "counterfactual")
        self.assertEqual(ledger_by_name["cross-world consistency"].status.value, "unresolved")
        self.assertEqual(
            ledger_by_name["counterfactual model uniqueness"].status.value,
            "contradicted",
        )
        self.assertGreaterEqual(ledger.contradicted_count, 1)

    def test_ledger_handles_real_benchmark_attack_templates(self) -> None:
        empty_ledgers = []

        for family_name in list_graph_families():
            blueprint = generate_graph_family(family_name, seed=17)
            for attack_name, template in ATTACK_TEMPLATE_REGISTRY.items():
                for label in template.compatible_labels:
                    if not template.is_compatible(blueprint, label):
                        continue
                    sample = generate_attack_sample(
                        blueprint,
                        gold_label=label,
                        attack_name=attack_name,
                        seed=17,
                    )
                    ledger = build_assumption_ledger(parse_claim(sample.claim_text))
                    if not ledger.entries:
                        empty_ledgers.append((family_name, attack_name, label, sample.claim_text))

        self.assertFalse(empty_ledgers, msg=json.dumps(empty_ledgers[:5], ensure_ascii=False, indent=2))


class CountermodelSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ClaimParser()

    def test_l1_search_returns_unidentifiable_countermodel_candidate(self) -> None:
        parsed = parse_claim(
            "There is no serious hidden-variable explanation here, so training_intensity itself drives income_score."
        )
        ledger = build_assumption_ledger(parsed)
        result = search_countermodels(parsed, ledger)

        self.assertIsInstance(result, CountermodelSearchResult)
        self.assertTrue(result.found_countermodel)
        self.assertEqual(result.verdict_suggestion, "unidentifiable")
        self.assertIn(
            result.countermodel_type,
            {"latent_confounder_injection", "direction_flip_candidate"},
        )
        self.assertTrue(result.query_disagreement)
        self.assertGreater(result.observational_match_score, 0.8)
        self.assertTrue(result.candidates)
        self.assertTrue(any(candidate.countermodel_type == "latent_confounder_injection" for candidate in result.candidates))

    def test_l1_selection_countermodel_can_suggest_invalid(self) -> None:
        parsed = self.parser.parse(
            "Within the selected sample, the link between adoption_level and stability_index is clean enough to read causally."
        )
        result = search_countermodels(parsed)

        self.assertTrue(result.found_countermodel)
        self.assertEqual(result.countermodel_type, "selection_mechanism_candidate")
        self.assertEqual(result.verdict_suggestion, "invalid")
        self.assertTrue(result.query_disagreement)

    def test_l2_search_returns_iv_or_hidden_confounder_candidate(self) -> None:
        parsed = self.parser.parse(
            "assignment_lottery affects recovery only through exposure, so using assignment_lottery as an instrument is enough to recover the causal effect of exposure on recovery."
        )
        ledger = build_assumption_ledger(parsed)
        result = search_countermodels(
            parsed,
            ledger,
            context={"claim_text": "assignment_lottery affects recovery only through exposure, so using assignment_lottery as an instrument is enough to recover the causal effect of exposure on recovery."},
        )

        self.assertTrue(result.found_countermodel)
        self.assertEqual(result.verdict_suggestion, "unidentifiable")
        self.assertIn(
            result.countermodel_type,
            {"invalid_instrument_alternative", "hidden_confounder_compatible_model"},
        )
        self.assertTrue(any(candidate.countermodel_type == "invalid_instrument_alternative" for candidate in result.candidates))

    def test_l2_search_uses_public_observed_data_when_available(self) -> None:
        scenario = _build_public_intervention_scenario()
        claim_text = (
            "assignment_lottery affects recovery only through exposure, so using assignment_lottery "
            "as an instrument is enough to recover the causal effect of exposure on recovery."
        )
        parsed = self.parser.parse(claim_text)
        ledger = build_assumption_ledger(parsed)

        result = search_countermodels(
            parsed,
            ledger,
            scenario=scenario,
            context={"claim_text": claim_text},
        )

        self.assertTrue(result.found_countermodel)
        self.assertTrue(result.used_observed_data)
        self.assertTrue(any(candidate.observational_evidence.get("used_observed_data") for candidate in result.candidates))
        self.assertIn("first-stage signal", result.countermodel_explanation)

    def test_countermodel_search_ignores_structural_hints_in_context_metadata(self) -> None:
        observed_data = pd.DataFrame(
            {
                "exposure": [0, 0, 1, 1, 0, 1],
                "recovery": [0.1, 0.2, 0.8, 1.0, 0.0, 0.9],
            }
        )
        parsed = ParsedClaim(
            query_type="intervention",
            treatment="exposure",
            outcome="recovery",
            claim_polarity="positive",
            claim_strength="strong",
            mentioned_assumptions=["instrument relevance"],
            implied_assumptions=["exclusion restriction", "instrument independence", "consistency", "positivity"],
            rhetorical_strategy="instrumental_variable_appeal",
            needs_abstention_check=True,
        )
        ledger = build_assumption_ledger(parsed)

        result = search_countermodels(
            parsed,
            ledger,
            observed_data=observed_data,
            context={
                "claim_text": "An instrumental-variable estimate is enough to recover the causal effect of exposure on recovery.",
                "public_metadata": {"role_bindings": {"instrument": "hidden_iv"}},
            },
        )

        self.assertTrue(result.found_countermodel)
        self.assertNotIn("hidden_iv", result.countermodel_explanation)
        for candidate in result.candidates:
            self.assertNotEqual(candidate.observational_evidence.get("instrument"), "hidden_iv")

    def test_l3_search_returns_observationally_equivalent_counterfactual_candidate(self) -> None:
        parsed = self.parser.parse(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
        )
        result = search_countermodels(
            parsed,
            context={
                "claim_text": "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified."
            },
        )
        payload = result.to_dict()

        self.assertTrue(result.found_countermodel)
        self.assertEqual(result.countermodel_type, "observationally_equivalent_countermodel")
        self.assertEqual(result.verdict_suggestion, "invalid")
        self.assertTrue(result.query_disagreement)
        self.assertIn("same-fit structural model", result.countermodel_explanation)
        self.assertTrue(any(candidate.countermodel_type == "observationally_equivalent_countermodel" for candidate in result.candidates))
        self.assertEqual(payload["found_countermodel"], True)
        self.assertEqual(payload["verdict_suggestion"], "invalid")
        json.dumps(payload)

    def test_l2_search_supports_proxy_based_alternative_explanation(self) -> None:
        parsed = ParsedClaim(
            query_type="intervention",
            treatment="exposure",
            outcome="recovery",
            claim_polarity="positive",
            claim_strength="strong",
            mentioned_assumptions=[],
            implied_assumptions=["consistency", "positivity"],
            rhetorical_strategy="plain_causal_assertion",
            needs_abstention_check=True,
        )
        ledger = build_assumption_ledger(parsed)
        result = search_countermodels(
            parsed,
            ledger,
            context={"claim_text": "proxy_signal is enough to support a causal estimate from exposure to recovery."},
        )

        self.assertTrue(result.found_countermodel)
        self.assertTrue(any(candidate.countermodel_type == "proxy_based_alternative_explanation" for candidate in result.candidates))
        self.assertIn(result.verdict_suggestion, {"invalid", "unidentifiable"})

    def test_l3_search_does_not_force_countermodel_for_every_counterfactual_claim(self) -> None:
        parsed = ParsedClaim(
            query_type="counterfactual",
            treatment="therapy_flag",
            outcome="response_quality",
            claim_polarity="positive",
            claim_strength="strong",
            mentioned_assumptions=["stable mediation structure"],
            implied_assumptions=[
                "consistency",
                "positivity",
                "cross-world consistency",
                "counterfactual model uniqueness",
            ],
            rhetorical_strategy="plain_causal_assertion",
            needs_abstention_check=True,
        )
        ledger = build_assumption_ledger(parsed)
        result = search_countermodels(
            parsed,
            ledger,
            context={"claim_text": "The measured mediator intermediate_state supports evaluating the effect of therapy_flag on response_quality for treated units."},
        )

        self.assertFalse(result.found_countermodel)
        self.assertEqual(result.candidates, [])


class DecisionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ClaimParser()

    def test_stage_1_strong_countermodel_returns_invalid_without_support_stage(self) -> None:
        parsed = self.parser.parse(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
        )
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(parsed, ledger)
        decision = decide_verdict(parsed, ledger, countermodel)

        self.assertIsInstance(decision, VerifierDecision)
        self.assertEqual(decision.label, VerdictLabel.INVALID)
        self.assertFalse(decision.metadata["support_stage_entered"])
        self.assertIsNotNone(decision.countermodel_witness)
        self.assertEqual(decision.countermodel_witness.witness_type, WitnessKind.COUNTERMODEL)
        self.assertIs(decision.witness, decision.countermodel_witness)

    def test_stage_2_query_disagreement_returns_unidentifiable_without_support_stage(self) -> None:
        parsed = parse_claim(
            "There is no serious hidden-variable explanation here, so training_intensity itself drives income_score."
        )
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(parsed, ledger)
        decision = decide_verdict(parsed, ledger, countermodel)

        self.assertEqual(decision.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertFalse(decision.metadata["support_stage_entered"])
        self.assertIsNotNone(decision.countermodel_witness)
        self.assertEqual(decision.countermodel_witness.verdict_suggestion, VerdictLabel.UNIDENTIFIABLE)

    def test_stage_3_unresolved_assumptions_return_unidentifiable(self) -> None:
        parsed = parse_claim("Estimate the causal effect of exposure on recovery.")
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(parsed, ledger, context={"claim_text": "Estimate the causal effect of exposure on recovery."})
        decision = decide_verdict(parsed, ledger, countermodel)

        self.assertFalse(countermodel.found_countermodel)
        self.assertTrue(decision.metadata["support_stage_entered"])
        self.assertEqual(decision.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertIsNotNone(decision.witness)
        self.assertEqual(decision.witness.witness_type, WitnessKind.ASSUMPTION)
        self.assertTrue(decision.tool_trace == [])

    def test_stage_4_rejects_bare_l2_claim_with_only_overlap_support(self) -> None:
        parsed = parse_claim("Estimate the causal effect of exposure on recovery.")
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(parsed, ledger, context={"claim_text": "Estimate the causal effect of exposure on recovery."})
        tool_trace = [
            {
                "tool_name": "overlap_check",
                "summary": "Observed overlap supports the required treatment contrast.",
                "supports_assumptions": ["positivity"],
                "supports_claim": True,
                "confidence": 0.86,
            }
        ]
        decision = decide_verdict(parsed, ledger, countermodel, tool_trace=tool_trace)

        self.assertEqual(decision.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertTrue(decision.metadata["support_stage_entered"])
        self.assertEqual(decision.witness.witness_type, WitnessKind.ASSUMPTION)

    def test_stage_4_supportive_tools_return_valid_for_identifiable_l2_claim(self) -> None:
        parsed = parse_claim("After controlling for pretest_score, the causal effect of exposure on recovery is identified.")
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(
            parsed,
            ledger,
            context={"claim_text": "After controlling for pretest_score, the causal effect of exposure on recovery is identified."},
        )
        tool_trace = [
            {
                "tool_name": "backdoor_check",
                "summary": "The observed adjustment set blocks the relevant backdoor paths.",
                "supports_assumptions": ["valid adjustment set", "no unobserved confounding", "positivity"],
                "supports_claim": True,
                "confidence": 0.87,
            }
        ]
        decision = decide_verdict(parsed, ledger, countermodel, tool_trace=tool_trace)
        payload = decision.to_dict()

        self.assertEqual(decision.label, VerdictLabel.VALID)
        self.assertTrue(decision.metadata["support_stage_entered"])
        self.assertIsNotNone(decision.support_witness)
        self.assertEqual(decision.support_witness.witness_type, WitnessKind.SUPPORT)
        self.assertIs(decision.witness, decision.support_witness)
        self.assertEqual(payload["label"], "valid")
        self.assertIn("valid adjustment set", decision.support_witness.assumptions)
        json.dumps(payload)

    def test_stage_4_ignores_stance_relative_rebuttal_trace_when_assessing_primary_claim(self) -> None:
        parsed = ParsedClaim(
            query_type="association",
            treatment="exposure",
            outcome="recovery",
            claim_polarity="positive",
            claim_strength="tentative",
            mentioned_assumptions=[],
            implied_assumptions=[],
            rhetorical_strategy="plain_causal_assertion",
            needs_abstention_check=False,
        )
        ledger = AssumptionLedger([])
        countermodel = CountermodelSearchResult(found_countermodel=False, candidates=[])
        tool_trace = [
            {
                "tool_name": "rebuttal_tool",
                "summary": "This trace supports the anti-causal rebuttal rather than the original claim.",
                "supports_claim": True,
                "claim_stance": "anti_causal",
                "confidence": 0.91,
            }
        ]

        decision = decide_verdict(parsed, ledger, countermodel, tool_trace=tool_trace)

        self.assertEqual(decision.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertTrue(decision.metadata["support_stage_entered"])
        self.assertEqual(decision.metadata["decision_stage"], 3)

    def test_stage_4_supportive_tools_return_valid_for_identifiable_l3_claim(self) -> None:
        parsed = ParsedClaim(
            query_type="counterfactual",
            treatment="therapy_flag",
            outcome="response_quality",
            claim_polarity="positive",
            claim_strength="strong",
            mentioned_assumptions=["stable mediation structure"],
            implied_assumptions=[
                "consistency",
                "positivity",
                "cross-world consistency",
                "counterfactual model uniqueness",
            ],
            rhetorical_strategy="plain_causal_assertion",
            needs_abstention_check=True,
        )
        ledger = build_assumption_ledger(parsed)
        countermodel = search_countermodels(
            parsed,
            ledger,
            context={"claim_text": "The measured mediator intermediate_state supports evaluating the effect of therapy_flag on response_quality for treated units."},
        )
        tool_trace = [
            {
                "tool_name": "scm_identification_test",
                "summary": "No competing SCM with the same observational fit was found.",
                "supports_assumptions": [
                    "counterfactual model uniqueness",
                    "cross-world consistency",
                    "stable mediation structure",
                    "positivity",
                ],
                "supports_claim": True,
                "confidence": 0.84,
            }
        ]
        decision = decide_verdict(parsed, ledger, countermodel, tool_trace=tool_trace)

        self.assertEqual(decision.label, VerdictLabel.VALID)
        self.assertIsNotNone(decision.support_witness)
        self.assertIn("counterfactual model uniqueness", decision.support_witness.assumptions)


class PipelineTests(unittest.TestCase):
    def test_pipeline_does_not_forward_public_metadata_to_countermodel_search(self) -> None:
        scenario = _build_public_intervention_scenario()
        with patch("verifier.pipeline.search_countermodels") as mocked_search:
            mocked_search.return_value = CountermodelSearchResult(found_countermodel=False, candidates=[])
            pipeline = VerifierPipeline()
            pipeline.run(
                "Estimate the causal effect of exposure on recovery.",
                scenario=scenario,
            )

        self.assertEqual(mocked_search.call_count, 1)
        forwarded_context = mocked_search.call_args.kwargs["context"]
        self.assertNotIn("public_metadata", forwarded_context)
        self.assertIn("observed_data", forwarded_context)

    def test_pipeline_can_run_end_to_end_in_one_call(self) -> None:
        tool_runner = FakeToolRunner(
            [
                {
                    "tool_name": "backdoor_check",
                    "summary": "The observed adjustment set blocks the relevant backdoor paths.",
                    "supports_assumptions": ["valid adjustment set", "no unobserved confounding", "positivity"],
                    "supports_claim": True,
                    "confidence": 0.84,
                }
            ]
        )
        pipeline = VerifierPipeline(tool_runner=tool_runner)
        result = pipeline.run(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified.",
        )

        self.assertIsInstance(result, VerifierDecision)
        self.assertEqual(result.label, VerdictLabel.VALID)
        self.assertTrue(result.metadata["support_stage_entered"])
        self.assertIsNotNone(result.support_witness)
        self.assertTrue(result.tool_trace)
        self.assertEqual(len(tool_runner.calls), 1)

    def test_pipeline_handles_truthful_valid_benchmark_samples_with_supporting_tools(self) -> None:
        generator = BenchmarkGenerator(seed=28)
        cases = (
            (
                "l2_valid_backdoor_family",
                28,
                [
                    {
                        "tool_name": "backdoor_check",
                        "summary": "The observed adjustment set blocks the relevant backdoor paths.",
                        "supports_assumptions": ["valid adjustment set", "no unobserved confounding", "positivity"],
                        "supports_claim": True,
                        "confidence": 0.87,
                    }
                ],
            ),
            (
                "l3_mediation_abduction_family",
                30,
                [
                    {
                        "tool_name": "scm_identification_test",
                        "summary": "The measured mediator and observed covariates support the counterfactual query.",
                        "supports_assumptions": [
                            "counterfactual model uniqueness",
                            "cross-world consistency",
                            "stable mediation structure",
                            "positivity",
                        ],
                        "supports_claim": True,
                        "confidence": 0.85,
                    }
                ],
            ),
            (
                "l3_mediation_abduction_family",
                18,
                [
                    {
                        "tool_name": "scm_identification_test",
                        "summary": "The measured mediator and observed covariates support the counterfactual query.",
                        "supports_assumptions": [
                            "counterfactual model uniqueness",
                            "cross-world consistency",
                            "stable mediation structure",
                            "positivity",
                        ],
                        "supports_claim": True,
                        "confidence": 0.85,
                    }
                ],
            ),
        )

        for family_name, seed, tool_trace in cases:
            with self.subTest(family_name=family_name, seed=seed):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                result = run_verifier_pipeline(
                    claim.claim_text,
                    tool_trace=tool_trace,
                )

                self.assertIs(claim.gold_label, VerdictLabel.VALID)
                self.assertEqual(result.label, VerdictLabel.VALID)
                self.assertTrue(result.metadata["support_stage_entered"])
                self.assertIsNotNone(result.support_witness)

    def test_pipeline_handles_truthful_valid_benchmark_samples_with_real_tool_executor(self) -> None:
        generator = BenchmarkGenerator(seed=17)
        executor = ToolExecutor({})
        cases = (
            ("l1_proxy_disambiguation_family", 0),
            ("l2_valid_backdoor_family", 8),
            ("l3_mediation_abduction_family", 0),
        )

        for family_name, seed in cases:
            with self.subTest(family_name=family_name, seed=seed):
                sample = generator.generate_benchmark_sample(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                level = int(str(sample.claim.causal_level).replace("L", ""))
                report = executor.execute_for_claim(
                    scenario=sample.public,
                    claim=sample.claim.claim_text,
                    level=level,
                    context={"claim_stance": "pro_causal"},
                )
                result = run_verifier_pipeline(
                    sample.claim.claim_text,
                    scenario=sample.public,
                    tool_trace=report["tool_trace"],
                )

                self.assertIs(sample.claim.gold_label, VerdictLabel.VALID)
                self.assertEqual(result.label, VerdictLabel.VALID)
                self.assertTrue(result.metadata["support_stage_entered"])

    def test_pipeline_with_real_tool_executor_keeps_invalid_adjustment_claim_off_valid_path(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_backdoor_family",
            difficulty=0.4,
            seed=17,
        )
        executor = ToolExecutor({})
        report = executor.execute_for_claim(
            scenario=sample.public,
            claim=sample.claim.claim_text,
            level=2,
            context={"claim_stance": "pro_causal"},
        )
        result = run_verifier_pipeline(
            sample.claim.claim_text,
            scenario=sample.public,
            tool_trace=report["tool_trace"],
        )

        self.assertIs(sample.claim.gold_label, VerdictLabel.INVALID)
        self.assertNotEqual(result.label, VerdictLabel.VALID)
        self.assertIn("valid adjustment set", {
            assumption
            for entry in report["tool_trace"]
            for assumption in entry["contradicts_assumptions"]
        })

    def test_pipeline_with_real_tool_executor_keeps_selection_biased_sample_off_valid_path(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l1_selection_bias_family",
            difficulty=0.4,
            seed=3,
        )
        executor = ToolExecutor({})
        report = executor.execute_for_claim(
            scenario=sample.public,
            claim=sample.claim.claim_text,
            level=1,
            context={"claim_stance": "pro_causal"},
        )
        result = run_verifier_pipeline(
            sample.claim.claim_text,
            scenario=sample.public,
            tool_trace=report["tool_trace"],
        )

        self.assertIs(sample.claim.gold_label, VerdictLabel.UNIDENTIFIABLE)
        self.assertEqual(result.label, VerdictLabel.UNIDENTIFIABLE)

    def test_pipeline_with_real_tool_executor_matches_representative_benchmark_grid(self) -> None:
        executor = ToolExecutor({})
        generator = BenchmarkGenerator(seed=17)
        cases = (
            ("l1_latent_confounding_family", 0, VerdictLabel.INVALID),
            ("l1_latent_confounding_family", 1, VerdictLabel.UNIDENTIFIABLE),
            ("l1_selection_bias_family", 0, VerdictLabel.INVALID),
            ("l1_selection_bias_family", 3, VerdictLabel.UNIDENTIFIABLE),
            ("l1_proxy_disambiguation_family", 0, VerdictLabel.VALID),
            ("l1_proxy_disambiguation_family", 1, VerdictLabel.INVALID),
            ("l2_valid_backdoor_family", 8, VerdictLabel.VALID),
            ("l2_valid_backdoor_family", 17, VerdictLabel.INVALID),
            ("l2_invalid_iv_family", 0, VerdictLabel.INVALID),
            ("l2_invalid_iv_family", 1, VerdictLabel.UNIDENTIFIABLE),
            ("l3_counterfactual_ambiguity_family", 0, VerdictLabel.UNIDENTIFIABLE),
            ("l3_counterfactual_ambiguity_family", 1, VerdictLabel.INVALID),
            ("l3_mediation_abduction_family", 0, VerdictLabel.VALID),
            ("l3_mediation_abduction_family", 1, VerdictLabel.INVALID),
        )

        for family_name, seed, expected_label in cases:
            with self.subTest(family_name=family_name, seed=seed, expected=expected_label.value):
                sample = generator.generate_benchmark_sample(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                level = int(str(sample.claim.causal_level).replace("L", ""))
                report = executor.execute_for_claim(
                    scenario=sample.public,
                    claim=sample.claim.claim_text,
                    level=level,
                    context={"claim_stance": "pro_causal"},
                )
                result = run_verifier_pipeline(
                    sample.claim.claim_text,
                    scenario=sample.public,
                    tool_trace=report["tool_trace"],
                )

                self.assertIs(sample.claim.gold_label, expected_label)
                self.assertEqual(result.label, expected_label)

    def test_pipeline_keeps_generated_iv_attack_sample_on_intervention_path(self) -> None:
        claim = BenchmarkGenerator(seed=17).generate_claim_instance(
            family_name="l2_invalid_iv_family",
            difficulty=0.4,
            seed=2,
        )
        parsed = parse_claim(claim.claim_text)
        result = run_verifier_pipeline(claim.claim_text)

        self.assertEqual(claim.query_type, "instrumental_variable_claim")
        self.assertIs(claim.gold_label, VerdictLabel.INVALID)
        self.assertEqual(parsed.query_type, QueryType.INTERVENTION)
        self.assertIsNotNone(result.countermodel_witness)
        self.assertIn(
            result.countermodel_witness.payload["countermodel_type"],
            {"invalid_instrument_alternative", "hidden_confounder_compatible_model"},
        )

    def test_pipeline_keeps_generated_l3_attack_sample_on_counterfactual_path(self) -> None:
        claim = BenchmarkGenerator(seed=17).generate_claim_instance(
            family_name="l3_counterfactual_ambiguity_family",
            difficulty=0.4,
            seed=2,
        )
        parsed = parse_claim(claim.claim_text)
        result = run_verifier_pipeline(claim.claim_text)

        self.assertEqual(claim.query_type, "effect_of_treatment_on_treated")
        self.assertIs(claim.gold_label, VerdictLabel.UNIDENTIFIABLE)
        self.assertEqual(parsed.query_type, QueryType.COUNTERFACTUAL)
        self.assertIsNotNone(result.countermodel_witness)
        self.assertEqual(
            result.countermodel_witness.payload["countermodel_type"],
            "observationally_equivalent_countermodel",
        )

    def test_pipeline_preserves_unidentifiable_semantics_for_generated_benchmark_samples(self) -> None:
        generator = BenchmarkGenerator(seed=17)
        mismatches = []
        families = (
            "l1_latent_confounding_family",
            "l1_selection_bias_family",
            "l2_invalid_iv_family",
            "l3_counterfactual_ambiguity_family",
        )

        for family_name in families:
            for seed in range(24):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                if claim.gold_label is not VerdictLabel.UNIDENTIFIABLE:
                    continue
                result = run_verifier_pipeline(claim.claim_text)
                if result.label is VerdictLabel.INVALID:
                    mismatches.append(
                        (
                            family_name,
                            seed,
                            claim.query_type,
                            claim.meta.get("attack_name"),
                            claim.claim_text,
                            result.countermodel_witness.payload["countermodel_type"]
                            if result.countermodel_witness is not None
                            else None,
                        )
                    )

        self.assertFalse(mismatches, msg=json.dumps(mismatches[:10], ensure_ascii=False, indent=2))

    def test_pipeline_regression_for_selection_and_counterfactual_unidentifiable_samples(self) -> None:
        cases = (
            ("l1_selection_bias_family", 9, "selection_bias_obfuscation"),
            ("l3_counterfactual_ambiguity_family", 2, "unidentifiable_disguised_as_valid"),
        )
        generator = BenchmarkGenerator(seed=17)

        for family_name, seed, attack_name in cases:
            with self.subTest(family_name=family_name, seed=seed):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.4,
                    seed=seed,
                )
                result = run_verifier_pipeline(claim.claim_text)

                self.assertIs(claim.gold_label, VerdictLabel.UNIDENTIFIABLE)
                self.assertEqual(claim.meta.get("attack_name"), attack_name)
                self.assertEqual(result.label, VerdictLabel.UNIDENTIFIABLE)

    def test_pipeline_accepts_public_scenario_for_countermodel_grounding(self) -> None:
        scenario = _build_public_intervention_scenario()
        pipeline = VerifierPipeline()
        claim_text = (
            "assignment_lottery affects recovery only through exposure, so using assignment_lottery "
            "as an instrument is enough to recover the causal effect of exposure on recovery."
        )

        result = pipeline.run(
            claim_text,
            scenario=scenario,
        )

        self.assertEqual(result.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertIsNotNone(result.countermodel_witness)
        self.assertTrue(result.countermodel_witness.payload["used_observed_data"])

    def test_pipeline_only_enters_support_stage_when_no_countermodel_exists(self) -> None:
        invalid_result = run_verifier_pipeline(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
            tool_trace=[
                {
                    "tool_name": "mock_support",
                    "summary": "This support trace should never be used once a countermodel wins.",
                    "supports_assumptions": ["counterfactual model uniqueness", "positivity"],
                    "supports_claim": True,
                    "confidence": 0.99,
                }
            ],
        )
        valid_result = run_verifier_pipeline(
            "After controlling for pretest_score, the causal effect of exposure on recovery is identified.",
            tool_trace=[
                {
                    "tool_name": "backdoor_check",
                    "summary": "The observed adjustment set blocks the relevant backdoor paths.",
                    "supports_assumptions": ["valid adjustment set", "no unobserved confounding", "positivity"],
                    "supports_claim": True,
                    "confidence": 0.8,
                }
            ],
        )

        self.assertFalse(invalid_result.metadata["support_stage_entered"])
        self.assertEqual(invalid_result.label, VerdictLabel.INVALID)
        self.assertTrue(valid_result.metadata["support_stage_entered"])
        self.assertEqual(valid_result.label, VerdictLabel.VALID)

    def test_pipeline_skips_tool_stage_when_countermodel_exists(self) -> None:
        tool_runner = FakeToolRunner(
            [
                {
                    "tool_name": "mock_support",
                    "summary": "This tool should not run once countermodel search already defeats the claim.",
                    "supports_assumptions": ["counterfactual model uniqueness"],
                    "supports_claim": True,
                    "confidence": 0.99,
                }
            ]
        )
        pipeline = VerifierPipeline(tool_runner=tool_runner)
        result = pipeline.run(
            "For an individual with the same observed history, switching therapy_flag would definitely change recovery, so the unit-level counterfactual is uniquely identified.",
            transcript=[
                {"speaker": "agent_a", "content": "No extra assumptions are needed."},
                {"speaker": "agent_b", "content": "The answer is uniquely pinned down."},
            ],
        )

        self.assertEqual(result.label, VerdictLabel.INVALID)
        self.assertFalse(result.metadata["support_stage_entered"])
        self.assertEqual(len(tool_runner.calls), 0)
        self.assertEqual(result.tool_trace, [])

    def test_pipeline_preserves_list_transcript_for_countermodel_search_context(self) -> None:
        tool_runner = FakeToolRunner(
            [
                {
                    "tool_name": "mock_support",
                    "summary": "This tool trace should be ignored because countermodel search should trigger first.",
                    "supports_assumptions": ["instrument relevance"],
                    "supports_claim": True,
                    "confidence": 0.9,
                }
            ]
        )
        transcript = [
            {
                "speaker": "agent_a",
                "content": "Using assignment_lottery as an instrument is enough to recover the causal effect of exposure on recovery.",
            }
        ]
        pipeline = VerifierPipeline(tool_runner=tool_runner)
        result = pipeline.run(
            "Estimate the causal effect of exposure on recovery.",
            transcript=transcript,
        )

        self.assertEqual(result.label, VerdictLabel.UNIDENTIFIABLE)
        self.assertFalse(result.metadata["support_stage_entered"])
        self.assertEqual(len(tool_runner.calls), 0)
        self.assertIsNotNone(result.countermodel_witness)
        self.assertEqual(result.countermodel_witness.payload["countermodel_type"], "invalid_instrument_alternative")


if __name__ == "__main__":
    unittest.main()
