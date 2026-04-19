import json
import tempfile
import unittest
from pathlib import Path

from benchmark.loaders import (
    load_real_grounded_cases,
    load_real_grounded_claims,
    load_real_grounded_dataset,
    save_real_grounded_dataset,
)
from benchmark.real_grounded import (
    RealGroundedCase,
    RealGroundedDataset,
    SourceCitation,
)
from benchmark.schema import ClaimInstance


def _build_claim(instance_id: str = "real_grounded_claim_001") -> ClaimInstance:
    return ClaimInstance(
        instance_id=instance_id,
        causal_level="L2",
        graph_family="real_grounded_policy_case",
        language_template_id="real_grounded::literature::policy",
        observed_variables=["program_uptake", "baseline_need", "employment_gain"],
        claim_text="Increasing program_uptake improves employment_gain.",
        query_type="average_treatment_effect",
        target_variables={"treatment": "program_uptake", "outcome": "employment_gain"},
        gold_label="valid",
        gold_answer="The literature-grounded case treats the effect as identified under the stated assumptions.",
        gold_assumptions=["Conditional ignorability", "Positivity"],
    )


class RealGroundedSubsetTests(unittest.TestCase):
    def test_real_grounded_case_round_trip_uses_core_claim_schema_and_keeps_source_citation(self) -> None:
        claim = _build_claim()
        case = RealGroundedCase(
            case_id="rg_policy_001",
            grounding_type="literature_grounded",
            claim=claim,
            source_citation=SourceCitation(
                citation_text="Smith and Lee (2024), Journal of Policy Evaluation.",
                title="Employment effects of targeted training programs",
                url="https://example.com/policy-study",
                year=2024,
            ),
            public_evidence_summary="Public summary of an observational policy evaluation with measured covariates.",
            information_contract={
                "visible_information": [
                    "Observed covariate-adjusted estimates",
                    "Public treatment and outcome definitions",
                ],
                "hidden_information": [
                    "Reviewer-only sensitivity analysis details",
                ],
            },
            identifying_assumptions=["Conditional ignorability", "Stable outcome definition"],
            witness_note="The paper reports robustness checks but still relies on observational assumptions.",
        )

        payload = case.to_dict()
        restored = RealGroundedCase.from_dict(payload)

        self.assertIn("source_citation", payload)
        self.assertEqual(payload["source_citation"]["citation_text"], case.source_citation.citation_text)
        self.assertIsInstance(restored.claim, ClaimInstance)
        self.assertEqual(set(restored.claim.to_dict()), set(claim.to_dict()))
        self.assertEqual(restored.claim.claim_text, claim.claim_text)
        self.assertEqual(restored.claim.meta["data_origin"], "real_grounded")
        self.assertEqual(restored.claim.meta["grounding_type"], "literature_grounded")
        self.assertEqual(restored.claim.meta["real_grounded_case_id"], "rg_policy_001")

    def test_real_grounded_dataset_loader_and_serializer_round_trip(self) -> None:
        dataset = RealGroundedDataset(
            dataset_name="real_grounded_subset",
            version="2026-04-20",
            cases=[
                RealGroundedCase(
                    case_id="rg_policy_001",
                    grounding_type="semi_real",
                    claim=_build_claim("real_grounded_claim_002"),
                    source_citation={
                        "citation_text": "Garcia (2023), Observational Medicine Review.",
                        "title": "Semi-real treatment effect casebook",
                        "year": 2023,
                    },
                    public_evidence_summary="Semi-real observational case with documented public evidence and restricted annotations.",
                    information_contract={
                        "visible_information": ["Public cohort table", "Outcome definition"],
                        "hidden_information": ["Private adjudication memo"],
                    },
                    identifying_assumptions=["No hidden outcome misclassification"],
                    witness_note="The semi-real record includes an auditor note about residual uncertainty.",
                )
            ],
            metadata={"domains": ["policy", "observational_medicine"]},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "real_grounded_subset.json"
            save_real_grounded_dataset(dataset, path)

            loaded_dataset = load_real_grounded_dataset(path)
            loaded_cases = load_real_grounded_cases(path)
            loaded_claims = load_real_grounded_claims(path)

        self.assertEqual(loaded_dataset.dataset_name, "real_grounded_subset")
        self.assertEqual(loaded_dataset.version, "2026-04-20")
        self.assertEqual(len(loaded_cases), 1)
        self.assertEqual(len(loaded_claims), 1)
        self.assertIsInstance(loaded_cases[0], RealGroundedCase)
        self.assertIsInstance(loaded_claims[0], ClaimInstance)
        self.assertEqual(loaded_claims[0].instance_id, "real_grounded_claim_002")
        self.assertEqual(loaded_cases[0].source_citation.citation_text, "Garcia (2023), Observational Medicine Review.")

    def test_real_grounded_case_requires_source_citation(self) -> None:
        with self.assertRaises(ValueError):
            RealGroundedCase.from_dict(
                {
                    "case_id": "rg_missing_citation",
                    "grounding_type": "literature_grounded",
                    "claim": _build_claim("real_grounded_claim_003").to_dict(),
                    "public_evidence_summary": "Summary without citation.",
                    "information_contract": {
                        "visible_information": ["Observed estimate"],
                        "hidden_information": ["Unreleased appendix"],
                    },
                    "identifying_assumptions": ["Ignorability"],
                    "witness_note": "Missing citation should fail.",
                }
            )

    def test_real_grounded_case_requires_information_contract_and_identifying_assumptions(self) -> None:
        claim = _build_claim("real_grounded_claim_004")

        with self.assertRaises(ValueError):
            RealGroundedCase(
                case_id="rg_missing_contract",
                grounding_type="literature_grounded",
                claim=claim,
                source_citation=SourceCitation(citation_text="Nguyen (2025), Policy Studies."),
                public_evidence_summary="Summary without an explicit visible-vs-hidden contract.",
                witness_note="Missing information contract should fail.",
            )

        with self.assertRaises(ValueError):
            RealGroundedCase(
                case_id="rg_missing_assumptions",
                grounding_type="literature_grounded",
                claim=claim,
                source_citation=SourceCitation(citation_text="Nguyen (2025), Policy Studies."),
                public_evidence_summary="Summary with an explicit information contract but no identifying assumptions.",
                information_contract={
                    "visible_information": ["Public estimate"],
                    "hidden_information": ["Reviewer memo"],
                },
                witness_note="Missing identifying assumptions should fail.",
            )

    def test_real_grounded_case_rejects_conflicting_claim_meta_contract_fields(self) -> None:
        claim = _build_claim("real_grounded_claim_005")
        claim.meta = {
            "grounding_type": "semi_real",
            "real_grounded_case_id": "stale_case_id",
            "source_citation": {"citation_text": "Stale citation"},
            "information_contract": {
                "visible_information": ["Wrong public evidence"],
                "hidden_information": ["Wrong hidden evidence"],
            },
        }

        with self.assertRaises(ValueError):
            RealGroundedCase(
                case_id="rg_policy_005",
                grounding_type="literature_grounded",
                claim=claim,
                source_citation=SourceCitation(citation_text="Patel (2024), Causal Policy Review."),
                public_evidence_summary="Updated public evidence summary.",
                information_contract={
                    "visible_information": ["Observed cohort table"],
                    "hidden_information": ["Auditor-only sensitivity notes"],
                },
                identifying_assumptions=["Conditional ignorability"],
                witness_note="Conflicting claim meta should not be silently preserved.",
            )


if __name__ == "__main__":
    unittest.main()
