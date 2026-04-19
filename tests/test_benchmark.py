import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmark.attacks import (
    build_attack_sample,
    generate_attack_sample,
    get_attack_template,
    list_attack_templates,
)
from benchmark.generator import (
    BenchmarkSample,
    BenchmarkGenerator,
    get_showcase_parent_family,
    list_showcase_families,
    list_supported_benchmark_families,
)
from benchmark.graph_families import (
    IdentifiabilityStatus,
    build_graph_family,
    generate_graph_family,
    get_graph_family_template,
    list_graph_families,
)
from benchmark.loaders import load_split_ids, load_split_instances, load_split_manifest
from benchmark.persuasion_overlays import PERSUASION_STYLE_SPACE
from benchmark.schema import (
    BenchmarkSplitManifest,
    ClaimInstance,
    GoldCausalInstance,
    PublicCausalInstance,
    VerdictLabel,
    Witness,
    WitnessKind,
)
from benchmark.split_builder import build_benchmark_splits, build_split_manifest
from benchmark.witnesses import (
    build_witness_bundle,
    generate_assumption_witness,
    generate_countermodel_witness,
    generate_support_witness,
    generate_witness_bundle,
)
from game.data_generator import DataGenerator


def _build_claim_instance_for_split(
    instance_id: str,
    *,
    graph_family: str,
    language_template_id: str,
    observed_variables: list[str],
    meta: dict[str, object] | None = None,
) -> ClaimInstance:
    treatment = observed_variables[0]
    outcome = observed_variables[-1]
    return ClaimInstance(
        instance_id=instance_id,
        causal_level="L2",
        graph_family=graph_family,
        language_template_id=language_template_id,
        observed_variables=observed_variables,
        claim_text=f"{treatment} identifies {outcome}.",
        query_type="average_treatment_effect",
        target_variables={"treatment": treatment, "outcome": outcome},
        gold_label="valid",
        meta=meta or {},
    )


class BenchmarkSchemaTests(unittest.TestCase):
    def test_claim_instance_round_trip_is_json_serializable(self) -> None:
        support_witness = Witness(
            witness_type="support",
            description="Backdoor adjustment blocks all open backdoor paths.",
            evidence=["Adjusted on Z", "No remaining confounding path"],
            assumptions=["Positivity", "No measurement error"],
            payload={"identified": True, "steps": ("parse", "adjust", "verify")},
            verdict_suggestion="valid",
            metadata={"source": "unit-test"},
        )
        countermodel_witness = Witness(
            witness_type=WitnessKind.COUNTERMODEL,
            description="An alternative SCM preserves observations but flips the query answer.",
            payload={
                "found_countermodel": True,
                "observational_match_score": 0.94,
                "query_disagreement": True,
            },
            verdict_suggestion=VerdictLabel.UNIDENTIFIABLE,
        )
        assumption_witness = Witness(
            witness_type="assumption",
            description="Identification requires exclusion and monotonicity assumptions.",
            assumptions=["Exclusion restriction", "Monotonicity"],
        )

        instance = ClaimInstance(
            instance_id="l2_backdoor_valid_001",
            causal_level=2,
            graph_family="l2_valid_backdoor_family",
            language_template_id="tmpl_07",
            observed_variables=["X", "Y", "Z"],
            proxy_variables=["Z"],
            selection_mechanism="none",
            observed_data_path="benchmark/data/l2_backdoor_valid_001.parquet",
            claim_text="Adjusting for Z identifies the causal effect of X on Y.",
            attacker_rationale="Hide no additional structure; make the claim look straightforward.",
            query_type="average_treatment_effect",
            target_variables={"treatment": "X", "outcome": "Y"},
            gold_label="valid",
            gold_answer="ATE(X->Y) is identifiable after adjusting for Z.",
            gold_assumptions=["Backdoor criterion", "Consistency"],
            support_witness=support_witness,
            countermodel_witness=countermodel_witness,
            assumption_witness=assumption_witness,
            meta={"seed": 7, "tags": ("fresh", "paper-benchmark")},
        )

        payload = instance.to_dict()
        encoded = json.dumps(payload)
        restored = ClaimInstance.from_dict(json.loads(encoded))

        self.assertEqual(instance.causal_level, "L2")
        self.assertIs(instance.gold_label, VerdictLabel.VALID)
        self.assertEqual(payload["support_witness"]["verdict_suggestion"], "valid")
        self.assertEqual(payload["countermodel_witness"]["verdict_suggestion"], "unidentifiable")
        self.assertEqual(payload["meta"]["tags"], ["fresh", "paper-benchmark"])
        self.assertEqual(restored.instance_id, instance.instance_id)
        self.assertEqual(restored.causal_level, "L2")
        self.assertIs(restored.gold_label, VerdictLabel.VALID)
        self.assertIs(restored.support_witness.verdict_suggestion, VerdictLabel.VALID)
        self.assertIs(
            restored.countermodel_witness.verdict_suggestion,
            VerdictLabel.UNIDENTIFIABLE,
        )
        self.assertEqual(restored.assumption_witness.witness_type, WitnessKind.ASSUMPTION)

    def test_witness_schema_supports_three_way_verdict_space(self) -> None:
        expected = {
            "support": VerdictLabel.VALID,
            "countermodel": VerdictLabel.INVALID,
            "assumption": VerdictLabel.UNIDENTIFIABLE,
        }

        for witness_type, label in expected.items():
            with self.subTest(witness_type=witness_type, label=label.value):
                witness = Witness(
                    witness_type=witness_type,
                    description=f"{witness_type} witness",
                    verdict_suggestion=label.value,
                )

                self.assertEqual(witness.verdict_label_space, ("valid", "invalid", "unidentifiable"))
                self.assertIs(witness.verdict_suggestion, label)
                self.assertEqual(witness.to_dict()["verdict_suggestion"], label.value)

    def test_split_manifest_round_trip_is_json_serializable(self) -> None:
        manifest = BenchmarkSplitManifest(
            dataset_name="main_benchmark_v1",
            version="2026-04-16",
            train=["inst_001", "inst_002"],
            dev=["inst_010"],
            test_iid=["inst_020"],
            test_ood=["inst_900", "inst_901"],
            family_holdout=["l3_counterfactual_ambiguity_family"],
            lexical_holdout=["tmpl_holdout_03"],
            variable_renaming_holdout=True,
            metadata={"seed": 13, "builder": "split_builder_v1"},
        )

        payload = manifest.to_dict()
        restored = BenchmarkSplitManifest.from_dict(json.loads(json.dumps(payload)))

        self.assertEqual(manifest.split_name_space, ("train", "dev", "test_iid", "test_ood"))
        self.assertEqual(payload["splits"]["test_ood"], ["inst_900", "inst_901"])
        self.assertEqual(
            payload["holdout_strategy"]["family_holdout"],
            ["l3_counterfactual_ambiguity_family"],
        )
        self.assertTrue(payload["holdout_strategy"]["variable_renaming_holdout"])
        self.assertEqual(restored.dataset_name, "main_benchmark_v1")
        self.assertEqual(restored.test_iid, ["inst_020"])
        self.assertEqual(restored.test_ood, ["inst_900", "inst_901"])

    def test_claim_instance_rejects_missing_gold_label(self) -> None:
        with self.assertRaises(ValueError):
            ClaimInstance(
                instance_id="missing_label",
                causal_level="L1",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_missing",
                observed_variables=["X", "Y"],
                claim_text="Correlation alone proves causation.",
                query_type="causal_direction",
                target_variables={"treatment": "X", "outcome": "Y"},
                gold_label=None,
            )

    def test_claim_instance_from_dict_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValueError) as context:
            ClaimInstance.from_dict(
                {
                    "instance_id": "minimal_case",
                    "causal_level": "L1",
                    "graph_family": "l1_latent_confounding_family",
                    "gold_label": "valid",
                }
            )

        message = str(context.exception)
        self.assertIn("language_template_id", message)
        self.assertIn("claim_text", message)
        self.assertIn("query_type", message)
        self.assertIn("target_variables", message)

    def test_claim_instance_rejects_incomplete_target_variable_schema(self) -> None:
        with self.assertRaises(ValueError) as context:
            ClaimInstance(
                instance_id="missing_outcome",
                causal_level="L2",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_target_error",
                observed_variables=["X", "Y"],
                claim_text="Intervene on X.",
                query_type="average_treatment_effect",
                target_variables={"treatment": "X"},
                gold_label="valid",
            )

        self.assertIn("target_variables", str(context.exception))

    def test_claim_instance_rejects_witness_slot_type_mismatch(self) -> None:
        with self.assertRaises(ValueError) as context:
            ClaimInstance(
                instance_id="wrong_witness_slot",
                causal_level="L1",
                graph_family="l1_proxy_disambiguation_family",
                language_template_id="tmpl_wrong_witness",
                observed_variables=["X", "Y"],
                claim_text="Proxy evidence resolves the ambiguity.",
                query_type="proxy_adjusted_claim",
                target_variables={"treatment": "X", "outcome": "Y"},
                gold_label="valid",
                support_witness={
                    "witness_type": WitnessKind.ASSUMPTION.value,
                    "description": "This witness is in the wrong slot.",
                },
            )

        self.assertIn("support_witness", str(context.exception))
        self.assertIn("support", str(context.exception))

    def test_public_causal_instance_to_dict_is_json_serializable(self) -> None:
        public = PublicCausalInstance(
            scenario_id="serializable_public_case",
            description="Public schema should serialize cleanly.",
            variables=["X", "Y"],
            observed_data=pd.DataFrame({"X": [0, 1], "Y": [1.5, 2.5]}),
        )

        payload = public.to_dict()
        encoded = json.dumps(payload)

        self.assertIsInstance(encoded, str)
        self.assertEqual(payload["observed_data"][0]["X"], 0)
        self.assertEqual(payload["data"][1]["Y"], 2.5)
        self.assertNotIn("verdict", payload)


class GraphFamilyTemplateTests(unittest.TestCase):
    def test_graph_family_registry_has_at_least_two_families_per_causal_level(self) -> None:
        self.assertGreaterEqual(len(list_graph_families(causal_level="L1")), 2)
        self.assertGreaterEqual(len(list_graph_families(causal_level="L2")), 2)
        self.assertGreaterEqual(len(list_graph_families(causal_level="L3")), 2)

    def test_generate_graph_family_by_name_returns_reusable_blueprint(self) -> None:
        required_families = {
            "l1_latent_confounding_family": "L1",
            "l1_selection_bias_family": "L1",
            "l2_valid_backdoor_family": "L2",
            "l2_valid_iv_family": "L2",
            "l2_invalid_iv_family": "L2",
            "l3_counterfactual_ambiguity_family": "L3",
            "l3_mediation_abduction_family": "L3",
        }

        for family_name, causal_level in required_families.items():
            with self.subTest(family_name=family_name):
                template = get_graph_family_template(family_name)
                blueprint = generate_graph_family(family_name, seed=17)
                payload = blueprint.to_dict()

                self.assertEqual(template.family_name, family_name)
                self.assertEqual(template.causal_level, causal_level)
                self.assertEqual(blueprint.family_name, family_name)
                self.assertEqual(blueprint.causal_level, causal_level)
                self.assertEqual(payload["family_name"], family_name)
                self.assertEqual(payload["causal_level"], causal_level)
                self.assertIn("treatment", blueprint.target_variables)
                self.assertIn("outcome", blueprint.target_variables)
                self.assertIn(blueprint.target_variables["treatment"], blueprint.true_dag)
                self.assertIn(blueprint.target_variables["outcome"], blueprint.true_dag)
                self.assertTrue(blueprint.query_types)
                self.assertTrue(blueprint.supported_gold_labels)
                self.assertIn(blueprint.identifiability.value, payload["identifiability"])
                json.dumps(payload)

    def test_graph_family_filters_distinguish_identifiable_and_potentially_unidentifiable(self) -> None:
        identifiable_l2 = list_graph_families(
            causal_level="L2",
            identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        )
        potentially_unidentifiable_l2 = list_graph_families(
            causal_level="L2",
            identifiability="potentially_unidentifiable",
        )

        self.assertIn("l2_valid_backdoor_family", identifiable_l2)
        self.assertIn("l2_valid_iv_family", identifiable_l2)
        self.assertIn("l2_invalid_iv_family", potentially_unidentifiable_l2)
        self.assertNotEqual(identifiable_l2, potentially_unidentifiable_l2)

    def test_graph_family_seed_interface_is_deterministic(self) -> None:
        first = build_graph_family("l3_counterfactual_ambiguity_family", seed=23)
        second = build_graph_family("l3_counterfactual_ambiguity_family", seed=23)
        different = build_graph_family("l3_counterfactual_ambiguity_family", seed=24)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.seed, 23)
        self.assertEqual(second.seed, 23)
        self.assertEqual(different.seed, 24)
        self.assertNotEqual(first.to_dict(), different.to_dict())


class AttackTemplateTests(unittest.TestCase):
    def test_attack_registry_exposes_at_least_five_templates(self) -> None:
        self.assertGreaterEqual(len(list_attack_templates()), 5)

    def test_attack_registry_matches_blueprint_taxonomy_size(self) -> None:
        templates = set(list_attack_templates())

        self.assertEqual(len(templates), 10)
        self.assertIn("heterogeneity_overgeneralization", templates)

    def test_attack_templates_generate_claim_text_and_rationale(self) -> None:
        cases = (
            ("association_overclaim", "l1_latent_confounding_family", "invalid"),
            ("hidden_confounder_denial", "l1_latent_confounding_family", "invalid"),
            ("invalid_adjustment_claim", "l2_valid_backdoor_family", "invalid"),
            ("counterfactual_overclaim", "l3_counterfactual_ambiguity_family", "unidentifiable"),
            ("unidentifiable_disguised_as_valid", "l3_counterfactual_ambiguity_family", "unidentifiable"),
        )

        for attack_name, family_name, gold_label in cases:
            with self.subTest(attack_name=attack_name, family_name=family_name):
                sample = generate_attack_sample(
                    family_name,
                    gold_label=gold_label,
                    attack_name=attack_name,
                    seed=17,
                )

                template = get_attack_template(attack_name)
                treatment = sample.target_variables["treatment"]
                outcome = sample.target_variables["outcome"]

                self.assertEqual(sample.attack_name, attack_name)
                self.assertEqual(sample.requested_label, gold_label)
                self.assertIn(gold_label, sample.compatible_labels)
                self.assertTrue(sample.claim_text)
                self.assertTrue(sample.attacker_rationale)
                self.assertIn(treatment, sample.claim_text)
                self.assertIn(outcome, sample.claim_text)
                self.assertTrue(sample.style_id in sample.style_space)
                self.assertEqual(sample.metadata["template_description"], template.description)
                json.dumps(sample.to_dict())

    def test_attack_generator_is_reproducible_for_same_seed(self) -> None:
        first = build_attack_sample(
            "l2_invalid_iv_family",
            gold_label="invalid",
            attack_name="weak_iv_as_valid_iv",
            seed=23,
        )
        second = build_attack_sample(
            "l2_invalid_iv_family",
            gold_label="invalid",
            attack_name="weak_iv_as_valid_iv",
            seed=23,
        )

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_attack_style_can_randomize_across_seeds(self) -> None:
        variants = {
            generate_attack_sample(
                "l3_counterfactual_ambiguity_family",
                gold_label="unidentifiable",
                attack_name="counterfactual_overclaim",
                seed=seed,
            ).style_id
            for seed in range(6)
        }

        self.assertGreater(len(variants), 1)

    def test_same_structural_attack_can_emit_multiple_persuasion_styles(self) -> None:
        authority = generate_attack_sample(
            "l2_invalid_iv_family",
            gold_label="invalid",
            query_type="instrumental_variable_claim",
            attack_name="weak_iv_as_valid_iv",
            style_id="technical",
            persuasion_style_id="authority_pressure",
            seed=31,
        )
        confidence = generate_attack_sample(
            "l2_invalid_iv_family",
            gold_label="invalid",
            query_type="instrumental_variable_claim",
            attack_name="weak_iv_as_valid_iv",
            style_id="technical",
            persuasion_style_id="confidence_pressure",
            seed=31,
        )

        self.assertEqual(authority.attack_name, confidence.attack_name)
        self.assertEqual(authority.style_id, confidence.style_id)
        self.assertEqual(authority.persuasion_style_id, "authority_pressure")
        self.assertEqual(confidence.persuasion_style_id, "confidence_pressure")
        self.assertNotEqual(authority.claim_text, confidence.claim_text)
        self.assertNotEqual(authority.attacker_rationale, confidence.attacker_rationale)
        self.assertEqual(authority.metadata["pressure_type"], "authority_pressure")
        self.assertEqual(confidence.metadata["pressure_type"], "confidence_pressure")

    def test_attack_generator_uses_family_hints_and_label_compatibility(self) -> None:
        blueprint = generate_graph_family("l2_invalid_iv_family", seed=29)
        sample = generate_attack_sample(
            blueprint,
            gold_label="invalid",
            seed=29,
        )

        self.assertIn(sample.attack_name, blueprint.generator_hints["attack_modes"])
        self.assertIn(sample.requested_label, get_attack_template(sample.attack_name).compatible_labels)

    def test_attack_generator_rejects_label_incompatible_template(self) -> None:
        with self.assertRaises(ValueError):
            generate_attack_sample(
                "l3_counterfactual_ambiguity_family",
                gold_label="valid",
                attack_name="unidentifiable_disguised_as_valid",
                seed=7,
            )

    def test_invalid_adjustment_claim_avoids_true_identifying_adjuster(self) -> None:
        blueprint = generate_graph_family("l2_valid_backdoor_family", seed=11)
        sample = generate_attack_sample(
            blueprint,
            gold_label="invalid",
            attack_name="invalid_adjustment_claim",
            seed=11,
        )
        identifying_sets = blueprint.generator_hints["identifying_set_candidates"]
        flattened_identifying = {variable for group in identifying_sets for variable in group}

        self.assertEqual(sample.attack_name, "invalid_adjustment_claim")
        self.assertIn("adjustment_variable", sample.metadata)
        self.assertNotEqual(
            sample.metadata["adjustment_variable"],
            blueprint.role_bindings["backdoor_adjuster"],
        )
        self.assertNotIn(sample.metadata["adjustment_variable"], flattened_identifying)


class WitnessGenerationTests(unittest.TestCase):
    def test_support_witness_aligns_with_schema(self) -> None:
        witness = generate_support_witness(
            "l2_valid_backdoor_family",
            gold_label="valid",
            seed=31,
        )

        self.assertIsInstance(witness, Witness)
        self.assertEqual(witness.witness_type, WitnessKind.SUPPORT)
        self.assertIs(witness.verdict_suggestion, VerdictLabel.VALID)
        self.assertTrue(witness.description)
        self.assertTrue(witness.evidence)
        self.assertIn("support_strength", witness.payload)
        self.assertTrue(witness.payload["identified"])
        json.dumps(witness.to_dict())

    def test_countermodel_witness_for_unidentifiable_sample_is_schema_aligned(self) -> None:
        witness = generate_countermodel_witness(
            "l3_counterfactual_ambiguity_family",
            gold_label="unidentifiable",
            seed=37,
        )

        self.assertIsInstance(witness, Witness)
        self.assertEqual(witness.witness_type, WitnessKind.COUNTERMODEL)
        self.assertIs(witness.verdict_suggestion, VerdictLabel.UNIDENTIFIABLE)
        self.assertTrue(witness.payload["found_countermodel"])
        self.assertTrue(witness.payload["query_disagreement"])
        self.assertEqual(witness.payload["verdict_suggestion"], "unidentifiable")
        json.dumps(witness.to_dict())

    def test_assumption_witness_contains_structured_ledger(self) -> None:
        witness = generate_assumption_witness(
            "l2_invalid_iv_family",
            gold_label="invalid",
            seed=41,
        )

        self.assertEqual(witness.witness_type, WitnessKind.ASSUMPTION)
        self.assertIs(witness.verdict_suggestion, VerdictLabel.INVALID)
        self.assertTrue(witness.payload["assumption_ledger"])
        ledger_by_name = {
            entry["name"]: entry["status"] for entry in witness.payload["assumption_ledger"]
        }
        self.assertTrue(
            all(
                {"name", "source", "status", "note"} <= set(entry)
                for entry in witness.payload["assumption_ledger"]
            )
        )
        self.assertGreaterEqual(witness.payload["contradicted_count"], 1)
        self.assertEqual(ledger_by_name["exclusion restriction"], "contradicted")
        self.assertEqual(ledger_by_name["instrument relevance"], "supported")

    def test_assumption_witness_invalid_proxy_family_does_not_suggest_valid(self) -> None:
        witness = generate_assumption_witness(
            "l1_proxy_disambiguation_family",
            gold_label="invalid",
            seed=47,
        )

        ledger_by_name = {
            entry["name"]: entry["status"] for entry in witness.payload["assumption_ledger"]
        }

        self.assertIs(witness.verdict_suggestion, VerdictLabel.INVALID)
        self.assertGreaterEqual(witness.payload["contradicted_count"], 1)
        self.assertEqual(ledger_by_name["proxy sufficiency"], "contradicted")

    def test_assumption_witness_verdict_suggestion_aligns_with_supported_family_labels(self) -> None:
        for family_name in list_graph_families():
            blueprint = generate_graph_family(family_name, seed=0)
            for gold_label in blueprint.supported_gold_labels:
                with self.subTest(family_name=family_name, gold_label=gold_label):
                    witness = generate_assumption_witness(
                        family_name,
                        gold_label=gold_label,
                        seed=0,
                    )
                    self.assertEqual(witness.verdict_suggestion.value, gold_label)

    def test_witness_bundle_is_reproducible_and_claim_instance_compatible(self) -> None:
        first = generate_witness_bundle(
            "l3_counterfactual_ambiguity_family",
            gold_label="unidentifiable",
            seed=43,
        )
        second = build_witness_bundle(
            "l3_counterfactual_ambiguity_family",
            gold_label="unidentifiable",
            seed=43,
        )

        self.assertEqual(first.to_dict(), second.to_dict())

        instance = ClaimInstance(
            instance_id="witness_bundle_case",
            causal_level="L3",
            graph_family="l3_counterfactual_ambiguity_family",
            language_template_id="tmpl_witness_bundle",
            observed_variables=["X", "M", "Y"],
            claim_text="The counterfactual effect is uniquely determined.",
            query_type="unit_level_counterfactual",
            target_variables={"treatment": "X", "outcome": "Y"},
            gold_label="unidentifiable",
            support_witness=first.support_witness,
            countermodel_witness=first.countermodel_witness,
            assumption_witness=first.assumption_witness,
        )

        payload = instance.to_dict()
        self.assertEqual(payload["support_witness"]["witness_type"], "support")
        self.assertEqual(payload["countermodel_witness"]["witness_type"], "countermodel")
        self.assertEqual(payload["assumption_witness"]["witness_type"], "assumption")
        self.assertEqual(
            payload["countermodel_witness"]["verdict_suggestion"],
            "unidentifiable",
        )


class ShowcaseMigrationTests(unittest.TestCase):
    def test_legacy_data_generator_keeps_showcase_stories_as_registered_benchmark_subfamilies(self) -> None:
        generator = DataGenerator(seed=11)
        expected = {
            1: ("smoking_cancer", "showcase_smoking_family", "l1_latent_confounding_family"),
            2: ("education_income", "showcase_education_family", "l2_valid_iv_family"),
            3: ("drug_recovery", "showcase_drug_family", "l3_counterfactual_ambiguity_family"),
        }
        registered_families = set(list_graph_families())
        supported_families = set(list_supported_benchmark_families())

        for level, (scenario_id, subfamily, parent_family) in expected.items():
            with self.subTest(level=level, scenario_id=scenario_id):
                scenario = generator.generate_scenario(difficulty=0.45, causal_level=level)
                parent_blueprint = generate_graph_family(parent_family, seed=level)

                self.assertEqual(scenario.scenario_id, scenario_id)
                self.assertEqual(scenario.metadata["scenario_family"], subfamily)
                self.assertEqual(scenario.metadata["benchmark_subfamily"], subfamily)
                self.assertEqual(scenario.metadata["benchmark_family"], parent_family)
                self.assertIn(parent_family, registered_families)
                self.assertIn(parent_family, supported_families)
                self.assertEqual(get_showcase_parent_family(subfamily), parent_family)
                self.assertEqual(parent_blueprint.causal_level, f"L{level}")
                self.assertTrue(scenario.metadata["is_showcase"])

    def test_showcase_alias_and_public_export_remain_available(self) -> None:
        generator = DataGenerator(seed=19)

        gold = generator.generate_scenario(
            difficulty=0.4,
            scenario_id="showcase_smoking_family",
        )
        public = generator.generate_public_scenario(
            difficulty=0.4,
            scenario_id="showcase_smoking_family",
        )
        exported = generator.export_public_instance(gold)

        self.assertEqual(gold.scenario_id, "smoking_cancer")
        self.assertIsInstance(public, PublicCausalInstance)
        self.assertIsInstance(exported, PublicCausalInstance)
        self.assertNotEqual(public.scenario_id, gold.scenario_id)
        self.assertNotEqual(exported.scenario_id, gold.scenario_id)
        self.assertTrue(public.scenario_id.startswith("public_case_"))
        self.assertTrue(exported.scenario_id.startswith("public_case_"))
        self.assertNotEqual(public.description, gold.description)
        self.assertIn("smoking", public.description.lower())
        self.assertIn("cancer", public.description.lower())
        self.assertNotIn("hidden", public.description.lower())
        self.assertNotIn("genetic_risk", public.description)
        self.assertFalse(hasattr(public, "true_dag"))

    def test_benchmark_generator_supports_showcase_and_non_showcase_samples(self) -> None:
        generator = BenchmarkGenerator(seed=23)

        self.assertIn("showcase_smoking_family", list_showcase_families())
        self.assertIn("l1_selection_bias_family", list_supported_benchmark_families())

        showcase = generator.generate_gold_instance(
            family_name="showcase_drug_family",
            difficulty=0.35,
            seed=23,
        )
        programmatic = generator.generate_gold_instance(
            family_name="l1_selection_bias_family",
            difficulty=0.35,
            seed=23,
        )
        public = generator.generate_public_instance(
            family_name="l1_selection_bias_family",
            difficulty=0.35,
            seed=23,
        )

        self.assertEqual(showcase.scenario_id, "drug_recovery")
        self.assertTrue(showcase.metadata["is_showcase"])
        self.assertEqual(programmatic.metadata["scenario_family"], "l1_selection_bias_family")
        self.assertEqual(programmatic.metadata["benchmark_family"], "l1_selection_bias_family")
        self.assertFalse(programmatic.metadata["is_showcase"])
        self.assertFalse(programmatic.observed_data.empty)
        self.assertTrue(set(programmatic.hidden_variables).issubset(set(programmatic.full_data.columns)))
        self.assertIsInstance(public, PublicCausalInstance)
        self.assertNotEqual(public.scenario_id, programmatic.scenario_id)
        self.assertFalse(hasattr(public, "true_dag"))

    def test_programmatic_public_view_drops_structural_benchmark_metadata(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_invalid_iv_family",
            difficulty=0.4,
            seed=17,
        )

        self.assertNotIn("role_bindings", sample.public.metadata)
        self.assertNotIn("identifiability", sample.public.metadata)
        self.assertNotIn("generator_hints", sample.public.metadata)
        self.assertNotIn("benchmark_family", sample.public.metadata)
        self.assertNotIn("scenario_family", sample.public.metadata)
        self.assertNotIn("benchmark_subfamily", sample.public.metadata)

    def test_programmatic_public_view_is_leakage_free_with_respect_to_seed_and_family(self) -> None:
        sample = BenchmarkGenerator(seed=10).generate_benchmark_sample(
            family_name="l2_invalid_iv_family",
            difficulty=0.4,
            seed=10,
        )

        self.assertNotIn("seed", sample.public.metadata)
        self.assertNotIn("winner", sample.public.metadata)
        self.assertNotIn("l2_invalid_iv_family", sample.public.description)
        self.assertNotIn("seed 10", sample.public.description.lower())
        self.assertNotIn("l2_invalid_iv_family", sample.public.scenario_id)
        self.assertTrue(sample.public.scenario_id.startswith("public_case_"))
        self.assertNotIn("generator_seed", sample.public.difficulty_config)
        self.assertNotIn("hidden_strength", sample.public.difficulty_config)
        self.assertNotIn("selection_bias_strength", sample.public.difficulty_config)

    def test_programmatic_public_view_exposes_typed_proxy_and_selection_hints(self) -> None:
        sample = BenchmarkGenerator(seed=41).generate_benchmark_sample(
            family_name="l1_selection_bias_family",
            difficulty=0.4,
            seed=41,
        )

        self.assertTrue(sample.public.proxy_variables == sample.claim.proxy_variables)
        self.assertTrue(sample.public.selection_mechanism)
        self.assertNotIn("proxy_variables", sample.public.metadata)
        self.assertNotIn("selection_variables", sample.public.metadata)
        self.assertNotIn("selection_mechanism", sample.public.metadata)

    def test_programmatic_public_view_avoids_family_and_semantic_leakage(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_iv_family",
            difficulty=0.4,
            seed=0,
        )

        self.assertEqual(sample.public.metadata["task_level"], "L2")
        self.assertEqual(sample.public.difficulty_config["task_level"], "L2")
        self.assertNotIn("difficulty_family", sample.public.metadata)
        self.assertIn("measurement_semantics", sample.public.metadata)
        self.assertIn("variable_descriptions", sample.public.metadata)
        self.assertNotIn("difficulty_family", sample.public.difficulty_config)
        self.assertNotIn("instrument_variables", sample.public.verifier_visible_fields)
        self.assertNotIn("mediator_variables", sample.public.verifier_visible_fields)
        self.assertNotIn("selection_variables", sample.public.verifier_visible_fields)
        self.assertEqual(sample.public.instrument_variables, [])
        self.assertNotIn("instrument_variables", sample.public.to_dict())
        self.assertEqual(sample.public.description, sample.gold.metadata["public_description"])

    def test_programmatic_public_view_routes_benign_metadata_without_structural_role_fields(self) -> None:
        iv_sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_iv_family",
            difficulty=0.4,
            seed=0,
        )
        mediation_sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l3_mediation_abduction_family",
            difficulty=0.4,
            seed=0,
        )

        instrument = iv_sample.blueprint.role_bindings["instrument"]
        mediator = mediation_sample.blueprint.role_bindings["mediator"]

        self.assertEqual(iv_sample.public.instrument_variables, [])
        self.assertEqual(iv_sample.public.mediator_variables, [])
        self.assertEqual(mediation_sample.public.mediator_variables, [])
        self.assertIn(instrument, iv_sample.public.metadata["variable_descriptions"])
        self.assertIn(instrument, iv_sample.public.metadata["measurement_semantics"])
        self.assertIn(mediator, mediation_sample.public.metadata["variable_descriptions"])
        self.assertIn(mediator, mediation_sample.public.metadata["measurement_semantics"])

    def test_benchmark_generator_variable_renaming_covers_valid_iv_family(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l2_valid_iv_family",
            difficulty=0.35,
            seed=3,
        )

        self.assertTrue(sample.claim.meta.get("variable_renaming"))
        self.assertNotIn("rename_map", sample.claim.meta)
        self.assertNotIn("original_variables", sample.claim.meta)
        self.assertTrue(sample.gold.metadata.get("rename_map"))
        self.assertNotEqual(
            sample.gold.metadata["original_variables"],
            list(sample.claim.observed_variables),
        )

    def test_claim_instance_meta_tracks_difficulty_family_and_task_level(self) -> None:
        claim = BenchmarkGenerator(seed=17).generate_claim_instance(
            family_name="l2_valid_backdoor_family",
            difficulty=0.35,
            seed=8,
        )

        self.assertEqual(claim.meta["difficulty_family"], "l2_valid_backdoor_family")
        self.assertEqual(claim.meta["task_level"], "L2")

    def test_claim_instance_meta_tracks_persuasion_style_for_attack_samples(self) -> None:
        claim = BenchmarkGenerator(seed=17).generate_claim_instance(
            family_name="l2_invalid_iv_family",
            difficulty=0.35,
            seed=29,
        )

        self.assertEqual(claim.meta["claim_mode"], "attack")
        self.assertIn(claim.meta["persuasion_style_id"], PERSUASION_STYLE_SPACE)
        self.assertEqual(claim.meta["pressure_type"], claim.meta["persuasion_style_id"])
        self.assertIn(claim.meta["persuasion_style_id"], claim.language_template_id)

    def test_benchmark_generator_produces_claim_instance_level_sample_bundle(self) -> None:
        generator = BenchmarkGenerator(seed=29)

        sample = generator.generate_benchmark_sample(
            family_name="l2_invalid_iv_family",
            difficulty=0.4,
            seed=29,
        )
        valid_claim = generator.generate_claim_instance(
            family_name="l2_valid_backdoor_family",
            difficulty=0.4,
            seed=30,
        )

        self.assertIsInstance(sample, BenchmarkSample)
        self.assertIsInstance(sample.claim, ClaimInstance)
        self.assertIsInstance(sample.gold, GoldCausalInstance)
        self.assertIsInstance(sample.public, PublicCausalInstance)
        self.assertEqual(sample.claim.graph_family, "l2_invalid_iv_family")
        self.assertEqual(sample.claim.gold_label, sample.gold.gold_label)
        self.assertTrue(sample.claim.claim_text)
        self.assertEqual(sample.claim.target_variables["treatment"], sample.gold.ground_truth["treatment"])
        self.assertEqual(sample.claim.support_witness.witness_type, WitnessKind.SUPPORT)
        self.assertEqual(sample.claim.countermodel_witness.witness_type, WitnessKind.COUNTERMODEL)
        self.assertEqual(sample.claim.assumption_witness.witness_type, WitnessKind.ASSUMPTION)
        self.assertEqual(valid_claim.gold_label, VerdictLabel.VALID)
        self.assertTrue(valid_claim.claim_text)
        self.assertIn(valid_claim.meta["claim_mode"], {"truthful", "attack"})

        invalid_sample = generator.generate_benchmark_sample(
            family_name="l2_invalid_iv_family",
            difficulty=0.4,
            seed=28,
        )
        self.assertIs(invalid_sample.claim.gold_label, VerdictLabel.INVALID)

    def test_benchmark_generator_rotates_query_types_within_family(self) -> None:
        generator = BenchmarkGenerator(seed=53)
        families = (
            "l1_proxy_disambiguation_family",
            "l2_valid_backdoor_family",
            "l3_mediation_abduction_family",
        )

        for family_name in families:
            with self.subTest(family_name=family_name):
                query_types = {
                    generator.generate_claim_instance(
                        family_name=family_name,
                        difficulty=0.35,
                        seed=seed,
                    ).query_type
                    for seed in range(8)
                }
                self.assertGreaterEqual(len(query_types), 2)

    def test_benchmark_generator_emits_variable_renaming_samples_for_holdout(self) -> None:
        generator = BenchmarkGenerator(seed=17)
        claims = [
            generator.generate_claim_instance(
                difficulty=0.35,
                seed=17 + offset,
            )
            for offset in range(12)
        ]
        renamed = [claim for claim in claims if claim.meta.get("variable_renaming")]

        self.assertTrue(renamed)
        for claim in renamed:
            self.assertNotIn("rename_map", claim.meta)
            self.assertNotIn("original_variables", claim.meta)
            self.assertTrue(claim.meta["renamed_variables"])

    def test_variable_renaming_metadata_does_not_leak_original_names_to_claim_payload(self) -> None:
        sample = BenchmarkGenerator(seed=17).generate_benchmark_sample(
            family_name="l1_proxy_disambiguation_family",
            difficulty=0.35,
            seed=3,
        )

        if not sample.claim.meta.get("variable_renaming"):
            self.skipTest("seed=3 did not generate a renamed sample")

        original_variables = list(sample.gold.metadata.get("original_variables", []))
        claim_payload = sample.claim.to_dict()
        public_payload = sample.public.to_dict()
        serialized = json.dumps({"claim": claim_payload, "public": public_payload}, ensure_ascii=False)

        for original in original_variables:
            self.assertNotIn(original, serialized)

    def test_showcase_families_can_generate_claim_instances_through_benchmark_chain(self) -> None:
        generator = BenchmarkGenerator(seed=37)
        showcase_expectations = {
            "showcase_smoking_family": {VerdictLabel.INVALID, VerdictLabel.UNIDENTIFIABLE},
            "showcase_education_family": {VerdictLabel.VALID, VerdictLabel.INVALID},
            "showcase_drug_family": {VerdictLabel.INVALID, VerdictLabel.UNIDENTIFIABLE},
        }

        for family_name, expected_labels in showcase_expectations.items():
            with self.subTest(family_name=family_name):
                claim = generator.generate_claim_instance(
                    family_name=family_name,
                    difficulty=0.35,
                    seed=37,
                )
                self.assertIsInstance(claim, ClaimInstance)
                self.assertTrue(claim.meta["is_showcase"])
                self.assertEqual(claim.meta["benchmark_subfamily"], family_name)
                self.assertIn(claim.gold_label, expected_labels)
                self.assertTrue(claim.claim_text)

    def test_education_showcase_inherits_valid_iv_parent_family(self) -> None:
        sample = BenchmarkGenerator(seed=37).generate_benchmark_sample(
            family_name="showcase_education_family",
            difficulty=0.35,
            seed=37,
        )

        self.assertEqual(sample.gold.metadata["benchmark_family"], "l2_valid_iv_family")
        self.assertEqual(sample.claim.graph_family, "l2_valid_iv_family")
        self.assertNotEqual(sample.claim.meta.get("attack_name"), "invalid_iv_exclusion_claim")

    def test_benchmark_sample_to_dict_is_json_serializable(self) -> None:
        sample = BenchmarkGenerator(seed=41).generate_benchmark_sample(
            family_name="l3_counterfactual_ambiguity_family",
            difficulty=0.3,
            seed=41,
        )

        payload = sample.to_dict()
        encoded = json.dumps(payload)

        self.assertIsInstance(encoded, str)
        self.assertIn("claim", payload)
        self.assertIn("gold", payload)
        self.assertIn("public", payload)
        self.assertIn("blueprint", payload)
        self.assertEqual(payload["claim"]["instance_id"], sample.claim.instance_id)
        self.assertEqual(payload["gold"]["scenario_id"], sample.gold.scenario_id)

    def test_generate_benchmark_samples_supports_include_showcase_batch_generation(self) -> None:
        samples = BenchmarkGenerator(seed=43).generate_benchmark_samples(
            num_samples=4,
            include_showcase=True,
            causal_level="L2",
            difficulty=0.35,
            seed=43,
        )

        self.assertEqual(len(samples), 4)
        self.assertTrue(all(isinstance(sample.claim, ClaimInstance) for sample in samples))
        self.assertTrue(any(sample.claim.meta["is_showcase"] for sample in samples))
        self.assertTrue(any(not sample.claim.meta["is_showcase"] for sample in samples))

    def test_benchmark_generator_claim_instances_can_flow_into_split_builder_and_loader(self) -> None:
        generator = BenchmarkGenerator(seed=31)
        claims = generator.generate_claim_instances(
            num_samples=6,
            family_names=[
                "l1_latent_confounding_family",
                "l2_invalid_iv_family",
                "l2_valid_backdoor_family",
                "l3_counterfactual_ambiguity_family",
            ],
            difficulty=0.35,
            seed=31,
        )

        lexical_holdout = [claims[0].language_template_id]
        manifest = build_split_manifest(
            claims,
            dataset_name="generator_chain_benchmark",
            family_holdout=["l3_counterfactual_ambiguity_family"],
            lexical_holdout=lexical_holdout,
            variable_renaming_holdout=False,
            seed=31,
        )
        loaded_instances = load_split_instances(claims, manifest)

        self.assertEqual(len(claims), 6)
        self.assertTrue(all(isinstance(claim, ClaimInstance) for claim in claims))
        self.assertTrue(manifest.test_ood)
        self.assertIn("l3_counterfactual_ambiguity_family", manifest.family_holdout)
        self.assertEqual(
            [instance.instance_id for instance in loaded_instances["test_ood"]],
            manifest.test_ood,
        )
        self.assertTrue(
            all(
                claim.gold_label in {VerdictLabel.VALID, VerdictLabel.INVALID, VerdictLabel.UNIDENTIFIABLE}
                for claim in claims
            )
        )


class SplitBuilderTests(unittest.TestCase):
    def test_split_builder_creates_iid_and_ood_manifest_with_all_holdout_modes(self) -> None:
        instances = [
            _build_claim_instance_for_split(
                "inst_train_1",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["X", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_train_2",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["A", "B", "C"],
            ),
            _build_claim_instance_for_split(
                "inst_dev_candidate",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_beta",
                observed_variables=["T", "M", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_family_holdout",
                graph_family="l3_counterfactual_ambiguity_family",
                language_template_id="tmpl_beta",
                observed_variables=["Q", "R", "S"],
            ),
            _build_claim_instance_for_split(
                "inst_lexical_holdout",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_holdout",
                observed_variables=["L", "M", "N"],
            ),
            _build_claim_instance_for_split(
                "inst_variable_holdout",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_beta",
                observed_variables=["renamed_treatment", "proxy_signal", "renamed_outcome"],
                meta={
                    "variable_renaming": True,
                    "original_variables": ["X", "Z", "Y"],
                    "rename_map": {"X": "renamed_treatment", "Y": "renamed_outcome"},
                },
            ),
        ]

        manifest = build_split_manifest(
            instances,
            dataset_name="unit_split_benchmark",
            version="2026-04-16",
            family_holdout=["l3_counterfactual_ambiguity_family"],
            lexical_holdout=["tmpl_holdout"],
            variable_renaming_holdout=True,
            seed=17,
        )

        self.assertEqual(manifest.dataset_name, "unit_split_benchmark")
        self.assertEqual(manifest.version, "2026-04-16")
        self.assertEqual(manifest.family_holdout, ["l3_counterfactual_ambiguity_family"])
        self.assertEqual(manifest.lexical_holdout, ["tmpl_holdout"])
        self.assertTrue(manifest.variable_renaming_holdout)
        self.assertTrue(manifest.test_ood)
        self.assertIn("inst_family_holdout", manifest.test_ood)
        self.assertIn("inst_lexical_holdout", manifest.test_ood)
        self.assertIn("inst_variable_holdout", manifest.test_ood)
        self.assertTrue(set(manifest.test_ood).isdisjoint(set(manifest.train)))
        self.assertTrue(set(manifest.test_ood).isdisjoint(set(manifest.dev)))
        self.assertTrue(set(manifest.test_ood).isdisjoint(set(manifest.test_iid)))
        self.assertIn("ood_reasons", manifest.metadata)
        self.assertGreaterEqual(manifest.metadata["split_counts"]["test_ood"], 1)

    def test_split_builder_reports_ood_reasons_without_mutating_input_instances(self) -> None:
        instances = [
            _build_claim_instance_for_split(
                "inst_in_domain",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["X", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_in_domain_two",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["A", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_in_domain_three",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["B", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_in_domain_four",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["C", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_ood_family",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_beta",
                observed_variables=["T", "M", "Y"],
            ),
        ]
        original_meta = [dict(instance.meta) for instance in instances]

        manifest = build_split_manifest(
            instances,
            family_holdout=["l2_valid_backdoor_family"],
            lexical_holdout=[],
            variable_renaming_holdout=False,
            seed=11,
        )

        self.assertEqual([instance.meta for instance in instances], original_meta)
        self.assertEqual(
            manifest.metadata["ood_reasons"]["inst_ood_family"],
            ["family_holdout"],
        )

    def test_split_builder_default_holdouts_use_stable_policy_instead_of_last_sorted_values(self) -> None:
        instances = [
            _build_claim_instance_for_split(
                "inst_train_1",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_alpha",
                observed_variables=["X", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_train_2",
                graph_family="l1_proxy_disambiguation_family",
                language_template_id="tmpl_beta",
                observed_variables=["A", "B", "C"],
            ),
            _build_claim_instance_for_split(
                "inst_train_3",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_gamma",
                observed_variables=["T", "M", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_train_4",
                graph_family="l2_valid_iv_family",
                language_template_id="tmpl_delta",
                observed_variables=["I", "J", "K"],
            ),
            _build_claim_instance_for_split(
                "inst_train_5",
                graph_family="l3_counterfactual_ambiguity_family",
                language_template_id="tmpl_epsilon",
                observed_variables=["Q", "R", "S"],
            ),
            _build_claim_instance_for_split(
                "inst_preferred_family",
                graph_family="l3_mediation_abduction_family",
                language_template_id="tmpl_zeta",
                observed_variables=["L", "M", "N"],
            ),
            _build_claim_instance_for_split(
                "inst_preferred_lexical",
                graph_family="l2_invalid_iv_family",
                language_template_id="attack::association_overclaim::plainspoken",
                observed_variables=["P", "Q", "R"],
            ),
            _build_claim_instance_for_split(
                "inst_last_sorted",
                graph_family="zz_new_family",
                language_template_id="zz_template",
                observed_variables=["U", "V", "W"],
            ),
        ]

        manifest = build_split_manifest(
            instances,
            seed=13,
        )

        self.assertEqual(manifest.family_holdout, ["l3_mediation_abduction_family"])
        self.assertEqual(manifest.lexical_holdout, ["attack::association_overclaim::plainspoken"])
        self.assertIn("inst_preferred_family", manifest.test_ood)
        self.assertIn("inst_preferred_lexical", manifest.test_ood)
        self.assertNotIn("inst_last_sorted", manifest.test_ood)

    def test_split_builder_stratifies_in_domain_label_distribution_when_possible(self) -> None:
        instances: list[ClaimInstance] = []
        for label, family in (
            ("valid", "l1_latent_confounding_family"),
            ("invalid", "l1_proxy_disambiguation_family"),
            ("unidentifiable", "l2_valid_backdoor_family"),
        ):
            for index in range(6):
                claim = _build_claim_instance_for_split(
                    f"{label}_{index}",
                    graph_family=family,
                    language_template_id=f"tmpl_{label}_{index % 2}",
                    observed_variables=[f"{label}_T_{index}", f"{label}_Z_{index}", f"{label}_Y_{index}"],
                )
                payload = claim.to_dict()
                payload["gold_label"] = label
                instances.append(ClaimInstance.from_dict(payload))
        instances.append(
            _build_claim_instance_for_split(
                "ood_family_holdout",
                graph_family="l3_counterfactual_ambiguity_family",
                language_template_id="tmpl_ood",
                observed_variables=["Q", "R", "S"],
            )
        )

        manifest = build_split_manifest(
            instances,
            family_holdout=["l3_counterfactual_ambiguity_family"],
            lexical_holdout=[],
            variable_renaming_holdout=False,
            seed=29,
        )

        label_distribution = manifest.metadata["label_distribution"]["test_iid"]
        self.assertGreaterEqual(label_distribution.get("valid", 0), 1)
        self.assertGreaterEqual(label_distribution.get("invalid", 0), 1)
        self.assertGreaterEqual(label_distribution.get("unidentifiable", 0), 1)

    def test_split_builder_rejects_manifests_without_non_empty_four_way_split(self) -> None:
        instances = [
            _build_claim_instance_for_split(
                "inst_train_1",
                graph_family="fam_a",
                language_template_id="tmpl",
                observed_variables=["X", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_train_2",
                graph_family="fam_a",
                language_template_id="tmpl",
                observed_variables=["A", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_ood_1",
                graph_family="fam_a",
                language_template_id="tmpl",
                observed_variables=["renamed_x_1", "Z", "Y"],
                meta={"variable_renaming": True, "rename_map": {"X": "renamed_x_1"}, "original_variables": ["X", "Z", "Y"]},
            ),
            _build_claim_instance_for_split(
                "inst_ood_2",
                graph_family="fam_a",
                language_template_id="tmpl",
                observed_variables=["renamed_x_2", "Z", "Y"],
                meta={"variable_renaming": True, "rename_map": {"X": "renamed_x_2"}, "original_variables": ["X", "Z", "Y"]},
            ),
        ]

        with self.assertRaises(ValueError):
            build_split_manifest(
                instances,
                family_holdout=[],
                lexical_holdout=[],
                variable_renaming_holdout=True,
                seed=0,
            )

    def test_split_builder_alias_and_loader_can_read_manifest(self) -> None:
        instances = [
            _build_claim_instance_for_split(
                "inst_a",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_a",
                observed_variables=["X", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_a2",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_a2",
                observed_variables=["X2", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_a3",
                graph_family="l1_latent_confounding_family",
                language_template_id="tmpl_a3",
                observed_variables=["X3", "Z", "Y"],
            ),
            _build_claim_instance_for_split(
                "inst_b",
                graph_family="l1_selection_bias_family",
                language_template_id="tmpl_b",
                observed_variables=["A", "B", "C"],
            ),
            _build_claim_instance_for_split(
                "inst_c",
                graph_family="l1_selection_bias_family",
                language_template_id="tmpl_b",
                observed_variables=["D", "E", "F"],
                meta={"variable_renaming_holdout": True},
            ),
            _build_claim_instance_for_split(
                "inst_d",
                graph_family="l2_valid_backdoor_family",
                language_template_id="tmpl_c",
                observed_variables=["T", "M", "Y"],
            ),
        ]

        manifest = build_benchmark_splits(
            instances,
            dataset_name="alias_split_benchmark",
            family_holdout=["l2_valid_backdoor_family"],
            lexical_holdout=["tmpl_b"],
            variable_renaming_holdout=True,
            seed=23,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            loaded_manifest = load_split_manifest(manifest_path)
            split_ids = load_split_ids(manifest_path)
            split_instances = load_split_instances(instances, manifest_path)

        self.assertIsInstance(loaded_manifest, BenchmarkSplitManifest)
        self.assertEqual(loaded_manifest.dataset_name, "alias_split_benchmark")
        self.assertEqual(split_ids["test_ood"], loaded_manifest.test_ood)
        self.assertEqual(
            [instance.instance_id for instance in split_instances["test_ood"]],
            loaded_manifest.test_ood,
        )
        self.assertTrue(loaded_manifest.test_ood)
        self.assertEqual(set(split_ids), {"train", "dev", "test_iid", "test_ood"})
        for split_name, resolved_instances in split_instances.items():
            for instance in resolved_instances:
                self.assertEqual(instance.meta.get("ood_split"), split_name)


if __name__ == "__main__":
    unittest.main()
