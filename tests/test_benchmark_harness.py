import unittest
from types import SimpleNamespace

from experiments.benchmark_harness import (
    _apply_no_abstention,
    _apply_family_postprocessing,
    aggregate_seed_metrics,
    build_seed_attack_benchmark_run,
    compare_system_predictions,
    summarize_protocol_compliance,
)
from experiments.exp_leakage_study.run import _mcnemar_significance


def _prediction_record(
    *,
    seed: int,
    split: str,
    instance_id: str,
    gold_label: str,
    predicted_label: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "split": split,
        "instance_id": instance_id,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
    }


class BenchmarkHarnessTests(unittest.TestCase):
    def test_compare_system_predictions_aligns_by_sample_identity(self) -> None:
        baseline_records = [
            _prediction_record(
                seed=0,
                split="test_ood",
                instance_id="inst_a",
                gold_label="valid",
                predicted_label="valid",
            ),
            _prediction_record(
                seed=0,
                split="test_ood",
                instance_id="inst_b",
                gold_label="invalid",
                predicted_label="invalid",
            ),
            _prediction_record(
                seed=1,
                split="test_ood",
                instance_id="inst_a",
                gold_label="unidentifiable",
                predicted_label="unidentifiable",
            ),
        ]
        reordered_records = [
            baseline_records[2],
            baseline_records[0],
            baseline_records[1],
        ]

        report = compare_system_predictions(
            {
                "baseline": baseline_records,
                "reordered": reordered_records,
            },
            baseline="baseline",
        )

        self.assertIsNotNone(report)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["score_a"], 1.0)
        self.assertEqual(comparison["score_b"], 1.0)
        self.assertEqual(comparison["observed_difference"], 0.0)

    def test_compare_system_predictions_rejects_mismatched_sample_sets(self) -> None:
        baseline_records = [
            _prediction_record(
                seed=0,
                split="test_iid",
                instance_id="inst_a",
                gold_label="valid",
                predicted_label="valid",
            ),
            _prediction_record(
                seed=0,
                split="test_iid",
                instance_id="inst_b",
                gold_label="invalid",
                predicted_label="invalid",
            ),
        ]
        mismatched_records = [
            _prediction_record(
                seed=0,
                split="test_iid",
                instance_id="inst_a",
                gold_label="valid",
                predicted_label="valid",
            ),
            _prediction_record(
                seed=0,
                split="test_iid",
                instance_id="inst_c",
                gold_label="invalid",
                predicted_label="invalid",
            ),
        ]

        with self.assertRaises(ValueError):
            compare_system_predictions(
                {
                    "baseline": baseline_records,
                    "candidate": mismatched_records,
                },
                baseline="baseline",
            )

    def test_leakage_mcnemar_significance_rejects_mismatched_sample_sets(self) -> None:
        baseline_records = [
            {
                **_prediction_record(
                    seed=0,
                    split="test_iid",
                    instance_id="inst_a",
                    gold_label="valid",
                    predicted_label="valid",
                ),
                "system_name": "countermodel_grounded",
            },
            {
                **_prediction_record(
                    seed=0,
                    split="test_iid",
                    instance_id="inst_b",
                    gold_label="invalid",
                    predicted_label="invalid",
                ),
                "system_name": "countermodel_grounded",
            },
        ]
        mismatched_records = [
            {
                **_prediction_record(
                    seed=0,
                    split="test_iid",
                    instance_id="inst_a",
                    gold_label="valid",
                    predicted_label="valid",
                ),
                "system_name": "oracle_leaking_partition",
            },
            {
                **_prediction_record(
                    seed=0,
                    split="test_iid",
                    instance_id="inst_c",
                    gold_label="invalid",
                    predicted_label="invalid",
                ),
                "system_name": "oracle_leaking_partition",
            },
        ]

        with self.assertRaises(ValueError):
            _mcnemar_significance(baseline_records + mismatched_records)

    def test_aggregate_seed_metrics_raises_on_missing_primary_metric(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_seed_metrics(
                {
                    0: {
                        "test_iid": {
                            "metrics": {
                                "verdict_accuracy": 1.0,
                            }
                        }
                    }
                },
                split_name="test_iid",
            )

    def test_build_seed_attack_benchmark_run_keeps_attack_only_protocol(self) -> None:
        run = build_seed_attack_benchmark_run(
            seed=0,
            difficulty=0.55,
            samples_per_family=2,
        )

        self.assertTrue(run.split_samples["test_iid"])
        self.assertTrue(run.split_samples["test_ood"])
        for sample in run.samples:
            self.assertIsNotNone(sample.claim.meta.get("attack_name"))
            self.assertEqual(sample.claim.meta.get("claim_mode"), "attack")

    def test_protocol_compliance_tracks_non_seed_requirements(self) -> None:
        protocol = summarize_protocol_compliance(
            [0, 1, 2],
            minimum_count=3,
            minimum_samples_per_family=10,
            observed_samples_per_family=2,
            minimum_audit_subset_size=150,
            observed_audit_subset_size=20,
            allow_protocol_violations=True,
        )

        self.assertFalse(protocol["compliant"])
        self.assertTrue(protocol["override_used"])
        self.assertIn("samples_per_family", protocol["violations"])
        self.assertIn("audit_subset_size", protocol["violations"])

    def test_family_variants_keep_verdict_semantics_consistent(self) -> None:
        skeptical = _apply_family_postprocessing(
            "skeptical_family",
            {
                "predicted_label": "valid",
                "confidence": 0.73,
                "verdict": {
                    "label": "valid",
                    "confidence": 0.73,
                    "probabilities": {
                        "valid": 0.76,
                        "invalid": 0.09,
                        "unidentifiable": 0.15,
                    },
                    "witness": {"witness_type": "support"},
                    "support_witness": {"witness_type": "support"},
                    "countermodel_witness": None,
                    "reasoning_summary": "Stage 4: no effective countermodel remained.",
                    "metadata": {},
                },
                "countermodel_found": False,
                "countermodel_type": None,
                "system_notes": [],
            },
        )
        self.assertEqual(skeptical["predicted_label"], "unidentifiable")
        self.assertIsNone(skeptical["verdict"].get("support_witness"))
        self.assertIsNone(skeptical["verdict"].get("countermodel_witness"))
        self.assertIn("Skeptical family override", skeptical["verdict"].get("reasoning_summary", ""))
        self.assertEqual(
            max(skeptical["verdict"]["probabilities"], key=skeptical["verdict"]["probabilities"].get),
            skeptical["predicted_label"],
        )

        optimistic = _apply_family_postprocessing(
            "optimistic_family",
            {
                "predicted_label": "unidentifiable",
                "confidence": 0.58,
                "verdict": {
                    "label": "unidentifiable",
                    "confidence": 0.58,
                    "probabilities": {
                        "valid": 0.16,
                        "invalid": 0.11,
                        "unidentifiable": 0.73,
                    },
                    "witness": {"witness_type": "assumption"},
                    "support_witness": None,
                    "countermodel_witness": None,
                    "reasoning_summary": "Stage 3: assumptions remain unsupported.",
                    "metadata": {},
                },
                "countermodel_found": False,
                "countermodel_type": None,
                "system_notes": [],
            },
        )
        self.assertEqual(optimistic["predicted_label"], "valid")
        self.assertIsNotNone(optimistic["verdict"].get("support_witness"))
        self.assertIsNone(optimistic["verdict"].get("countermodel_witness"))
        self.assertIn("Optimistic family override", optimistic["verdict"].get("reasoning_summary", ""))
        self.assertEqual(
            max(optimistic["verdict"]["probabilities"], key=optimistic["verdict"]["probabilities"].get),
            optimistic["predicted_label"],
        )

    def test_no_abstention_forces_probabilities_and_reasoning_to_match_forced_label(self) -> None:
        sample = SimpleNamespace(claim=SimpleNamespace(claim_text="X causes Y"))
        adjusted = _apply_no_abstention(
            sample,
            {
                "predicted_label": "valid",
                "confidence": 0.62,
                "verdict": {
                    "label": "unidentifiable",
                    "confidence": 0.62,
                    "probabilities": {
                        "valid": 0.14,
                        "invalid": 0.19,
                        "unidentifiable": 0.67,
                    },
                    "witness": {"witness_type": "countermodel"},
                    "support_witness": None,
                    "countermodel_witness": {"witness_type": "countermodel"},
                    "reasoning_summary": (
                        "Stage 2: the verifier found observationally compatible alternatives "
                        "that disagree on the target query."
                    ),
                    "metadata": {"decision_stage": 2},
                },
                "system_notes": [],
            },
        )

        self.assertEqual(max(adjusted["verdict"]["probabilities"], key=adjusted["verdict"]["probabilities"].get), adjusted["predicted_label"])
        self.assertIn("no-abstention", adjusted["verdict"]["reasoning_summary"].lower())
        self.assertEqual(adjusted["verdict"]["metadata"]["forced_from"], "unidentifiable")


if __name__ == "__main__":
    unittest.main()
