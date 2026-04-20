import json
import tempfile
import unittest
from pathlib import Path

from benchmark.loaders import (
    load_real_grounded_cases,
    load_real_grounded_claims,
    load_real_grounded_dataset,
    load_real_grounded_samples,
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


def _policy_observed_data_records() -> list[dict[str, float]]:
    return [
        {"program_uptake": 0.0, "baseline_need": 0.90, "employment_gain": 0.10},
        {"program_uptake": 0.0, "baseline_need": 0.75, "employment_gain": 0.15},
        {"program_uptake": 1.0, "baseline_need": 0.80, "employment_gain": 0.55},
        {"program_uptake": 1.0, "baseline_need": 0.65, "employment_gain": 0.60},
    ]


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
        self.assertEqual(
            restored.claim.meta["identifying_assumptions"],
            ["Conditional ignorability", "Stable outcome definition"],
        )

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

    def test_real_grounded_loader_can_emit_benchmark_samples_for_phase4_runners(self) -> None:
        dataset = RealGroundedDataset(
            cases=[
                RealGroundedCase(
                    case_id="rg_eval_001",
                    grounding_type="literature_grounded",
                    claim=_build_claim("real_grounded_claim_eval_001"),
                    source_citation=SourceCitation(
                        citation_text="Rivera (2025), Policy Evaluation Letters.",
                        title="Real-grounded evaluation case",
                        year=2025,
                    ),
                    public_evidence_summary="Evaluation-ready grounded case with public summary and hidden reviewer note.",
                    information_contract={
                        "visible_information": ["Observed estimate"],
                        "hidden_information": ["Private reviewer memo"],
                    },
                    identifying_assumptions=["Ignorability"],
                    witness_note="Evaluation-ready witness note.",
                    metadata={
                        "observed_data_records": _policy_observed_data_records(),
                    },
                )
            ]
        )

        samples = load_real_grounded_samples(dataset)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.claim.meta["dataset_partition"], "real_grounded")
        self.assertEqual(sample.claim.meta["data_origin"], "real_grounded")
        self.assertEqual(sample.public.description, "Evaluation-ready grounded case with public summary and hidden reviewer note.")
        self.assertEqual(sample.gold.gold_label.value, sample.claim.gold_label.value)
        self.assertFalse(sample.public.observed_data.empty)
        self.assertEqual(
            sample.public.observed_data.to_dict(orient="records"),
            _policy_observed_data_records(),
        )
        self.assertEqual(sample.blueprint.true_dag["program_uptake"], [])
        self.assertEqual(sample.blueprint.true_dag["employment_gain"], [])

    def test_real_grounded_loader_rejects_cases_without_explicit_observed_data_source(self) -> None:
        dataset = RealGroundedDataset(
            cases=[
                RealGroundedCase(
                    case_id="rg_missing_data_001",
                    grounding_type="literature_grounded",
                    claim=_build_claim("real_grounded_claim_missing_data"),
                    source_citation=SourceCitation(
                        citation_text="Rivera (2025), Policy Evaluation Letters.",
                        title="Missing data source case",
                        year=2025,
                    ),
                    public_evidence_summary="Grounded case without an explicit observed-data source.",
                    information_contract={
                        "visible_information": ["Observed estimate"],
                        "hidden_information": ["Private reviewer memo"],
                    },
                    identifying_assumptions=["Ignorability"],
                    witness_note="This case should fail until explicit observed data is supplied.",
                )
            ]
        )

        with self.assertRaises(ValueError):
            load_real_grounded_samples(dataset)

    def test_real_grounded_serializer_accepts_single_case_inputs(self) -> None:
        case = RealGroundedCase(
            case_id="rg_policy_single_case",
            grounding_type="literature_grounded",
            claim=_build_claim("real_grounded_claim_single"),
            source_citation=SourceCitation(
                citation_text="Lee (2024), Policy Causality Review.",
                title="Grounded policy case",
                year=2024,
            ),
            public_evidence_summary="Single-case grounded export path.",
            information_contract={
                "visible_information": ["Public policy table"],
                "hidden_information": ["Reviewer memo"],
            },
            identifying_assumptions=["No hidden policy targeting"],
            witness_note="Single-case serialization should follow the dataset contract.",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "real_grounded_single_case.json"
            saved_path = save_real_grounded_dataset(case, path)
            loaded_dataset = load_real_grounded_dataset(path)

        self.assertEqual(saved_path, path)
        self.assertEqual(loaded_dataset.dataset_name, "real_grounded_subset")
        self.assertEqual(len(loaded_dataset.cases), 1)
        self.assertEqual(loaded_dataset.cases[0].case_id, "rg_policy_single_case")

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

    def test_real_grounded_from_dict_normalizes_singleton_strings_without_char_splitting(self) -> None:
        case = RealGroundedCase.from_dict(
            {
                "case_id": "rg_singleton_strings",
                "grounding_type": "literature_grounded",
                "claim": _build_claim("real_grounded_claim_singletons").to_dict(),
                "source_citation": {
                    "citation_text": "Smith (2024), Policy Studies.",
                    "authors": "Smith",
                },
                "public_evidence_summary": "Summary with singleton string fields.",
                "information_contract": {
                    "visible_information": "Observed estimate",
                    "hidden_information": "Private reviewer memo",
                },
                "identifying_assumptions": "Ignorability",
                "witness_note": "Singleton strings should normalize cleanly.",
            }
        )

        self.assertEqual(case.source_citation.authors, ["Smith"])
        self.assertEqual(case.information_contract.visible_information, ["Observed estimate"])
        self.assertEqual(case.information_contract.hidden_information, ["Private reviewer memo"])
        self.assertEqual(case.identifying_assumptions, ["Ignorability"])
        self.assertEqual(case.claim.meta["identifying_assumptions"], ["Ignorability"])

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

    def test_real_grounded_case_rejects_conflicting_identifying_assumptions_in_claim_meta(self) -> None:
        claim = _build_claim("real_grounded_claim_006")
        claim.meta = {"identifying_assumptions": ["STALE"]}

        with self.assertRaisesRegex(ValueError, "identifying_assumptions"):
            RealGroundedCase(
                case_id="rg_policy_006",
                grounding_type="literature_grounded",
                claim=claim,
                source_citation=SourceCitation(citation_text="Patel (2024), Causal Policy Review."),
                public_evidence_summary="Updated public evidence summary.",
                information_contract={
                    "visible_information": ["Observed cohort table"],
                    "hidden_information": ["Auditor-only sensitivity notes"],
                },
                identifying_assumptions=["Conditional ignorability"],
                witness_note="Conflicting identifying assumptions should fail.",
            )


if __name__ == "__main__":
    unittest.main()
