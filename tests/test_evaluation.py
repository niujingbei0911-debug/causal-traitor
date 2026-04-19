import unittest

from evaluation.metrics import CausalMetrics, MetricResult
from evaluation.reporting import (
    compare_predictions,
    compare_prediction_groups,
    summarize_human_audit_agreement,
    summarize_metric,
)
from evaluation.scorer import Scorer, _extract_countermodel_applicable
from evaluation.significance import (
    bootstrap_confidence_interval,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_test,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = Scorer()

    def test_metric_result_supports_partial_construction(self) -> None:
        metric = MetricResult(name="DAcc", value=0.0)

        self.assertEqual(metric.name, "DAcc")
        self.assertEqual(metric.value, 0.0)
        self.assertEqual(metric.category, "core")
        self.assertEqual(metric.details, {})
        self.assertFalse(metric.is_appendix)

    def test_score_round_is_driven_by_verdict_labels_not_winner(self) -> None:
        round_score = self.scorer.score_round(
            {
                "round_id": 7,
                "winner": "agent_a",
                "gold_label": "invalid",
                "verdict_label": "invalid",
                "verifier_confidence": 0.83,
                "countermodel_witness": {"countermodel_type": "hidden_confounder_compatible_model"},
                "jury_verdict": "agent_b",
                "jury_consensus": 0.2,
            }
        )

        self.assertEqual(round_score.round_id, 7)
        self.assertEqual(round_score.gold_label, "invalid")
        self.assertEqual(round_score.predicted_label, "invalid")
        self.assertTrue(round_score.verdict_correct)
        self.assertAlmostEqual(round_score.confidence, 0.83, places=4)
        self.assertIn("jury_consensus", round_score.appendix)

    def test_score_game_ignores_winner_and_appendix_metrics_for_overall_score(self) -> None:
        base_rounds = [
            {
                "round_id": 1,
                "winner": "agent_b",
                "gold_label": "valid",
                "verdict_label": "valid",
                "verifier_confidence": 0.9,
                "deception_succeeded": True,
                "jury_consensus": 0.1,
            },
            {
                "round_id": 2,
                "winner": "agent_a",
                "gold_label": "invalid",
                "verdict_label": "valid",
                "verifier_confidence": 0.8,
                "deception_succeeded": False,
                "jury_consensus": 0.9,
            },
            {
                "round_id": 3,
                "winner": "draw",
                "gold_label": "unidentifiable",
                "verdict_label": "unidentifiable",
                "verifier_confidence": 0.7,
                "countermodel_witness": {"countermodel_type": "observationally_equivalent_countermodel"},
                "deception_succeeded": True,
                "jury_consensus": 0.5,
            },
        ]
        appendix_heavy_rounds = [
            {
                **round_data,
                "winner": "agent_a" if round_data["winner"] != "agent_a" else "agent_b",
                "deception_succeeded": not round_data.get("deception_succeeded", False),
                "jury_consensus": 1.0 - round_data.get("jury_consensus", 0.0),
            }
            for round_data in base_rounds
        ]

        score_a = self.scorer.score_game(
            {
                "game_id": "base",
                "winner": "agent_a",
                "rounds": base_rounds,
                "jury_accuracy": 0.15,
                "evolution_history": [
                    {"strategy": "s1"},
                    {"strategy": "s2"},
                    {"strategy": "s3"},
                ],
            }
        )
        score_b = self.scorer.score_game(
            {
                "game_id": "variant",
                "winner": "agent_b",
                "rounds": appendix_heavy_rounds,
                "jury_accuracy": 0.95,
                "evolution_history": [
                    {"strategy": "s1"},
                    {"strategy": "s1"},
                    {"strategy": "s1"},
                ],
            }
        )

        self.assertEqual(score_a.final_scores, score_b.final_scores)
        self.assertEqual(score_a.final_scores["verdict_accuracy"], 0.6667)
        self.assertEqual(score_a.final_scores["macro_f1"], 0.5556)
        self.assertEqual(score_a.final_scores["invalid_claim_acceptance_rate"], 1.0)
        self.assertEqual(score_a.final_scores["unidentifiable_awareness"], 1.0)
        self.assertEqual(score_a.final_scores["ece"], 0.4)
        self.assertEqual(score_a.final_scores["brier"], 0.1789)
        self.assertEqual(score_a.final_scores["countermodel_coverage"], 0.5)
        self.assertEqual(score_a.final_scores["overall"], 0.585)
        self.assertNotEqual(score_a.summary["appendix_metrics"], score_b.summary["appendix_metrics"])
        self.assertIn("DSR", score_a.summary["appendix_metrics"])
        self.assertIn("jury_accuracy", score_a.summary["appendix_metrics"])
        self.assertIn("jury_consensus", score_a.summary["appendix_metrics"])
        self.assertIn("ECI", score_a.summary["appendix_metrics"])

    def test_compute_weighted_score_ignores_appendix_metrics(self) -> None:
        core_metrics = [
            MetricResult(name="verdict_accuracy", value=0.7, category="verdict", is_primary=True),
            MetricResult(name="macro_f1", value=0.6, category="verdict", is_primary=True),
            MetricResult(
                name="invalid_claim_acceptance_rate",
                value=0.2,
                category="verdict",
                is_primary=True,
                higher_is_better=False,
            ),
            MetricResult(name="unidentifiable_awareness", value=0.8, category="verdict", is_primary=True),
            MetricResult(name="ece", value=0.1, category="verdict", is_primary=True, higher_is_better=False),
            MetricResult(name="brier", value=0.15, category="verdict", is_primary=True, higher_is_better=False),
            MetricResult(name="countermodel_coverage", value=0.5, category="verdict", is_primary=True),
        ]
        appendix_metrics = [
            MetricResult(name="DSR", value=1.0, category="deception", is_appendix=True),
            MetricResult(name="jury_accuracy", value=0.0, category="jury", is_appendix=True),
            MetricResult(name="ECI", value=1.0, category="evolution", is_appendix=True),
        ]

        core_only = self.scorer.compute_weighted_score(core_metrics)
        with_appendix = self.scorer.compute_weighted_score(core_metrics + appendix_metrics)

        self.assertEqual(core_only, 0.7125)
        self.assertEqual(with_appendix, core_only)

    def test_ece_keeps_confidences_aligned_with_filtered_labels(self) -> None:
        result = CausalMetrics.expected_calibration_error(
            [None, "valid"],
            [None, "valid"],
            [0.1, 0.9],
        )

        self.assertEqual(result.details["n"], 1)
        self.assertAlmostEqual(result.value, 0.1, places=4)

    def test_macro_f1_uses_frozen_three_label_space(self) -> None:
        result = CausalMetrics.verdict_macro_f1(
            ["valid", "invalid"],
            ["valid", "invalid"],
        )

        self.assertAlmostEqual(result.value, 2.0 / 3.0, places=4)
        self.assertEqual(
            result.details["labels"],
            ["valid", "invalid", "unidentifiable"],
        )
        self.assertEqual(result.details["per_label_f1"]["unidentifiable"], 0.0)

    def test_brier_score_is_part_of_core_metric_bundle(self) -> None:
        results = CausalMetrics.compute_all(
            {
                "gold_labels": ["valid", "invalid", "unidentifiable"],
                "predicted_labels": ["valid", "valid", "unidentifiable"],
                "confidences": [0.9, 0.8, 0.7],
                "countermodel_hits": [False, False, True],
                "countermodel_applicable": [False, True, True],
            }
        )

        lookup = {result.name: result for result in results}
        self.assertIn("brier", lookup)
        self.assertTrue(lookup["brier"].is_primary)
        self.assertFalse(lookup["brier"].higher_is_better)
        self.assertAlmostEqual(lookup["brier"].value, 0.1789, places=4)

    def test_score_game_uses_explicit_verdict_probabilities_for_calibration_metrics(self) -> None:
        score = self.scorer.score_game(
            {
                "game_id": "calibration-probabilities",
                "rounds": [
                    {
                        "round_id": 1,
                        "gold_label": "valid",
                        "verdict_label": "valid",
                        "verifier_confidence": 0.99,
                        "verdict_probabilities": {
                            "valid": 0.55,
                            "invalid": 0.30,
                            "unidentifiable": 0.15,
                        },
                    }
                ],
            }
        )

        self.assertAlmostEqual(score.final_scores["ece"], 0.45, places=4)
        self.assertAlmostEqual(score.final_scores["brier"], 0.105, places=4)

    def test_countermodel_applicability_is_not_polluted_by_predicted_label(self) -> None:
        self.assertFalse(
            _extract_countermodel_applicable(
                {
                    "gold_label": "valid",
                    "verdict_label": "invalid",
                }
            )
        )

    def test_metrics_compute_all_uses_gold_only_countermodel_applicability(self) -> None:
        results = CausalMetrics.compute_all(
            {
                "rounds": [
                    {
                        "gold_label": "valid",
                        "verdict_label": "invalid",
                        "countermodel_found": True,
                    }
                ]
            }
        )

        lookup = {result.name: result for result in results}
        self.assertEqual(lookup["countermodel_coverage"].value, 0.0)
        self.assertEqual(lookup["countermodel_coverage"].details["applicable"], 0)

    def test_score_game_counts_missing_predictions_as_scored_failures(self) -> None:
        score = self.scorer.score_game(
            {
                "game_id": "missing-verdicts",
                "rounds": [
                    {
                        "round_id": 1,
                        "gold_label": "valid",
                        "verdict_label": "valid",
                        "verifier_confidence": 0.8,
                    },
                    {
                        "round_id": 2,
                        "gold_label": "invalid",
                        "verdict_label": None,
                        "verifier_confidence": 0.0,
                    },
                ],
            }
        )

        self.assertEqual(score.summary["scored_rounds"], 2)
        self.assertAlmostEqual(score.final_scores["verdict_accuracy"], 0.5, places=4)

    def test_score_game_array_mode_counts_missing_predictions_as_failures(self) -> None:
        score = self.scorer.score_game(
            {
                "game_id": "missing-array-verdicts",
                "gold_labels": ["valid", "invalid"],
                "predicted_labels": ["valid"],
            }
        )

        self.assertEqual(score.summary["scored_rounds"], 2)
        self.assertAlmostEqual(score.final_scores["verdict_accuracy"], 0.5, places=4)

    def test_metrics_compute_all_array_mode_counts_missing_predictions_as_failures(self) -> None:
        lookup = {
            result.name: result
            for result in CausalMetrics.compute_all(
                {
                    "gold_labels": ["valid", "invalid"],
                    "predicted_labels": ["valid"],
                }
            )
        }

        self.assertAlmostEqual(lookup["verdict_accuracy"].value, 0.5, places=4)
        self.assertEqual(lookup["verdict_accuracy"].details["n"], 2)


class EvaluationStatisticsTests(unittest.TestCase):
    def test_bootstrap_confidence_interval_is_exact_for_constant_values(self) -> None:
        result = bootstrap_confidence_interval(
            [0.8, 0.8, 0.8],
            n_resamples=200,
            random_state=0,
        )

        self.assertAlmostEqual(result.observed_value, 0.8, places=12)
        self.assertAlmostEqual(result.sample_mean, 0.8, places=12)
        self.assertAlmostEqual(result.sample_std, 0.0, places=12)
        self.assertAlmostEqual(result.ci_lower, 0.8, places=12)
        self.assertAlmostEqual(result.ci_upper, 0.8, places=12)

    def test_reporting_summary_outputs_mean_std_and_ci(self) -> None:
        summary = summarize_metric(
            "verdict_accuracy",
            [0.70, 0.75, 0.80, 0.85],
            n_resamples=2000,
            random_state=0,
        )

        self.assertEqual(summary.metric_name, "verdict_accuracy")
        self.assertAlmostEqual(summary.mean, 0.775, places=6)
        self.assertAlmostEqual(summary.std, 0.0645497, places=6)
        self.assertLess(summary.ci_lower, summary.mean)
        self.assertGreater(summary.ci_upper, summary.mean)
        self.assertIn("0.7750 ± 0.0645", summary.formatted)
        self.assertIn("95% CI", summary.formatted)

    def test_paired_bootstrap_test_detects_significant_prediction_gap(self) -> None:
        y_true = [0, 1] * 10
        pred_a = y_true.copy()
        for index in range(8):
            pred_a[index] = 1 - pred_a[index]
        pred_b = y_true.copy()

        result = paired_bootstrap_test(
            y_true,
            pred_a,
            pred_b,
            n_resamples=4000,
            random_state=0,
        )

        self.assertEqual(result.method, "paired_bootstrap")
        self.assertAlmostEqual(result.score_a, 0.6, places=6)
        self.assertAlmostEqual(result.score_b, 1.0, places=6)
        self.assertAlmostEqual(result.observed_difference, 0.4, places=6)
        self.assertLess(result.p_value, 0.05)
        self.assertTrue(result.significant)
        self.assertIsNotNone(result.ci_lower)
        self.assertIsNotNone(result.ci_upper)
        self.assertGreater(result.ci_lower, 0.0)

    def test_mcnemar_test_detects_significant_prediction_gap(self) -> None:
        y_true = [0, 1] * 10
        pred_a = y_true.copy()
        for index in range(8):
            pred_a[index] = 1 - pred_a[index]
        pred_b = y_true.copy()

        result = mcnemar_test(y_true, pred_a, pred_b, exact=True)

        self.assertEqual(result.method, "mcnemar_exact")
        self.assertAlmostEqual(result.score_a, 0.6, places=6)
        self.assertAlmostEqual(result.score_b, 1.0, places=6)
        self.assertAlmostEqual(result.p_value, 0.0078125, places=8)
        self.assertTrue(result.significant)
        self.assertEqual(result.details["a_correct_b_wrong"], 0)
        self.assertEqual(result.details["a_wrong_b_correct"], 8)

    def test_compare_predictions_preserves_metric_name_for_mcnemar(self) -> None:
        y_true = [0, 1] * 10
        pred_a = y_true.copy()
        pred_b = y_true.copy()
        pred_a[0] = 1 - pred_a[0]
        pred_a[1] = 1 - pred_a[1]

        result = compare_predictions(
            y_true,
            pred_a,
            pred_b,
            method="mcnemar",
            metric_name="verdict_accuracy",
        )

        self.assertEqual(result.metric_name, "verdict_accuracy")

    def test_holm_bonferroni_corrects_multiple_comparisons(self) -> None:
        entries = holm_bonferroni(
            {
                "baseline vs model_b": 0.01,
                "baseline vs model_c": 0.03,
                "baseline vs model_d": 0.04,
            },
            alpha=0.05,
        )

        self.assertEqual(
            [entry.hypothesis for entry in entries],
            [
                "baseline vs model_b",
                "baseline vs model_c",
                "baseline vs model_d",
            ],
        )
        self.assertEqual([entry.reject for entry in entries], [True, False, False])
        self.assertAlmostEqual(entries[0].adjusted_p_value, 0.03, places=8)
        self.assertAlmostEqual(entries[1].adjusted_p_value, 0.06, places=8)
        self.assertAlmostEqual(entries[2].adjusted_p_value, 0.06, places=8)

    def test_compare_prediction_groups_outputs_significance_report(self) -> None:
        y_true = [0, 1] * 10
        baseline = y_true.copy()
        for index in range(8):
            baseline[index] = 1 - baseline[index]
        strong_model = y_true.copy()

        report = compare_prediction_groups(
            y_true,
            {
                "baseline": baseline,
                "strong_model": strong_model,
            },
            baseline="baseline",
            method="mcnemar",
            alpha=0.05,
        )

        self.assertEqual(report.method, "mcnemar")
        self.assertEqual(report.baseline, "baseline")
        self.assertEqual(len(report.comparisons), 1)
        comparison = report.comparisons[0]
        self.assertEqual(comparison.comparison, "baseline vs strong_model")
        self.assertLess(comparison.raw_result.p_value, 0.05)
        self.assertAlmostEqual(
            comparison.adjusted_p_value,
            comparison.raw_result.p_value,
            places=8,
        )
        self.assertTrue(comparison.reject_after_correction)

    def test_human_audit_agreement_summary_reports_percent_and_kappa(self) -> None:
        summary = summarize_human_audit_agreement(
            [
                {
                    "annotator_a_verifier_label_reasonable": "yes",
                    "annotator_b_verifier_label_reasonable": "yes",
                    "annotator_a_witness_quality_reasonable": "no",
                    "annotator_b_witness_quality_reasonable": "yes",
                },
                {
                    "annotator_a_verifier_label_reasonable": "no",
                    "annotator_b_verifier_label_reasonable": "no",
                    "annotator_a_witness_quality_reasonable": "yes",
                    "annotator_b_witness_quality_reasonable": "yes",
                },
            ],
            fields=["verifier_label_reasonable", "witness_quality_reasonable"],
        )

        self.assertEqual(summary["n_records"], 2)
        verifier = summary["fields"]["verifier_label_reasonable"]
        witness = summary["fields"]["witness_quality_reasonable"]
        self.assertEqual(verifier["n_scored"], 2)
        self.assertAlmostEqual(verifier["percent_agreement"], 1.0, places=8)
        self.assertAlmostEqual(verifier["cohen_kappa"], 1.0, places=8)
        self.assertEqual(witness["n_scored"], 2)
        self.assertAlmostEqual(witness["percent_agreement"], 0.5, places=8)
        self.assertLess(witness["cohen_kappa"], 1.0)

    def test_human_audit_agreement_normalizes_binary_aliases_and_drops_na_values(self) -> None:
        summary = summarize_human_audit_agreement(
            [
                {
                    "annotator_a_verifier_label_reasonable": "y",
                    "annotator_b_verifier_label_reasonable": "yes",
                    "annotator_a_witness_quality_reasonable": "n/a",
                    "annotator_b_witness_quality_reasonable": "skip",
                },
                {
                    "annotator_a_verifier_label_reasonable": True,
                    "annotator_b_verifier_label_reasonable": "YES",
                    "annotator_a_witness_quality_reasonable": "no",
                    "annotator_b_witness_quality_reasonable": "n",
                },
            ],
            fields=["verifier_label_reasonable", "witness_quality_reasonable"],
        )

        verifier = summary["fields"]["verifier_label_reasonable"]
        witness = summary["fields"]["witness_quality_reasonable"]
        self.assertEqual(verifier["n_scored"], 2)
        self.assertEqual(verifier["annotator_a_distribution"], {"yes": 2})
        self.assertEqual(verifier["annotator_b_distribution"], {"yes": 2})
        self.assertAlmostEqual(verifier["cohen_kappa"], 1.0, places=8)
        self.assertEqual(witness["n_scored"], 1)
        self.assertEqual(witness["annotator_a_distribution"], {"no": 1})
        self.assertEqual(witness["annotator_b_distribution"], {"no": 1})
        self.assertEqual(witness["invalid_label_count"], 0)

if __name__ == "__main__":
    unittest.main()
