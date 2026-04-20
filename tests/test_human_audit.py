import unittest

from experiments.exp_human_audit.run import _build_annotation_record, _round_robin_subset


def _audit_record(
    instance_id: str,
    *,
    seed: int,
    support: bool = False,
    countermodel: bool = False,
) -> dict:
    verdict: dict = {"reasoning_summary": f"summary for {instance_id}"}
    if support:
        verdict["support_witness"] = {
            "witness_type": "support",
            "description": f"support witness for {instance_id}",
        }
    if countermodel:
        verdict["countermodel_witness"] = {
            "witness_type": "countermodel",
            "description": f"countermodel witness for {instance_id}",
        }
    return {
        "split": "dev",
        "instance_id": instance_id,
        "seed": seed,
        "causal_level": "L2",
        "graph_family": "family",
        "query_type": "intervention",
        "attack_name": "truthful",
        "claim_text": f"claim {instance_id}",
        "gold_label": "invalid",
        "predicted_label": "invalid",
        "confidence": 0.8,
        "predicted_probabilities": {},
        "supports_public_only": True,
        "ood_reasons": [],
        "verdict": verdict,
        "public_evidence_summary": {"causal_level": "L2"},
    }


class HumanAuditPackagingTests(unittest.TestCase):
    def test_annotation_record_uses_primary_witness_fields_when_generic_witness_is_absent(self) -> None:
        packaged = _build_annotation_record(
            _audit_record("case_countermodel", seed=0, countermodel=True)
        )

        self.assertEqual(packaged["primary_witness_role"], "countermodel")
        self.assertEqual(
            packaged["witness_description"],
            "countermodel witness for case_countermodel",
        )
        self.assertEqual(packaged["witness_type"], "countermodel")

    def test_round_robin_subset_buckets_by_primary_witness_type(self) -> None:
        records = [
            _audit_record("cm_0", seed=0, countermodel=True),
            _audit_record("cm_1", seed=1, countermodel=True),
            _audit_record("support_2", seed=2, support=True),
        ]

        subset = _round_robin_subset(records, 2)

        self.assertEqual(
            {record["instance_id"] for record in subset},
            {"cm_0", "support_2"},
        )
