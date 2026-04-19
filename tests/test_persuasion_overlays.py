import unittest

from benchmark.persuasion_overlays import (
    PERSUASION_STYLE_SPACE,
    apply_persuasion_overlay,
    list_persuasion_overlays,
)


class PersuasionOverlayTests(unittest.TestCase):
    def test_overlay_registry_exposes_required_pressure_types(self) -> None:
        self.assertEqual(
            set(list_persuasion_overlays()),
            {
                "authority_pressure",
                "expert_tone_pressure",
                "confidence_pressure",
                "consensus_pressure",
                "concealment_pressure",
            },
        )
        self.assertEqual(
            PERSUASION_STYLE_SPACE,
            (
                "authority_pressure",
                "expert_tone_pressure",
                "confidence_pressure",
                "consensus_pressure",
                "concealment_pressure",
            ),
        )

    def test_overlay_application_is_deterministic_for_same_style_and_seed(self) -> None:
        first = apply_persuasion_overlay(
            claim_text="Adjusting for Z identifies the causal effect of X on Y.",
            attacker_rationale="Present the argument as fully settled.",
            persuasion_style_id="authority_pressure",
            attack_name="invalid_adjustment_claim",
            query_type="average_treatment_effect",
            seed=11,
        )
        second = apply_persuasion_overlay(
            claim_text="Adjusting for Z identifies the causal effect of X on Y.",
            attacker_rationale="Present the argument as fully settled.",
            persuasion_style_id="authority_pressure",
            attack_name="invalid_adjustment_claim",
            query_type="average_treatment_effect",
            seed=11,
        )

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_overlay_application_changes_text_across_pressure_types(self) -> None:
        authority = apply_persuasion_overlay(
            claim_text="Adjusting for Z identifies the causal effect of X on Y.",
            attacker_rationale="Present the argument as fully settled.",
            persuasion_style_id="authority_pressure",
            attack_name="invalid_adjustment_claim",
            query_type="average_treatment_effect",
            seed=23,
        )
        consensus = apply_persuasion_overlay(
            claim_text="Adjusting for Z identifies the causal effect of X on Y.",
            attacker_rationale="Present the argument as fully settled.",
            persuasion_style_id="consensus_pressure",
            attack_name="invalid_adjustment_claim",
            query_type="average_treatment_effect",
            seed=23,
        )

        self.assertNotEqual(authority.claim_text, consensus.claim_text)
        self.assertNotEqual(authority.attacker_rationale, consensus.attacker_rationale)
        self.assertEqual(authority.metadata["pressure_type"], "authority_pressure")
        self.assertEqual(consensus.metadata["pressure_type"], "consensus_pressure")
        self.assertTrue(authority.metadata["pressure_markers"])
        self.assertTrue(consensus.metadata["pressure_markers"])


if __name__ == "__main__":
    unittest.main()
