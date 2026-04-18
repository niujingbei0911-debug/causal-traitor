import unittest

from experiments.benchmark_harness import compare_system_predictions


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


if __name__ == "__main__":
    unittest.main()
