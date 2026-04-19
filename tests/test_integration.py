from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.tool_executor import ToolExecutor
from benchmark.schema import ensure_public_instance
from evaluation.tracker import ExperimentConfig, ExperimentTracker
from experiments.benchmark_harness import build_seed_benchmark_run
from experiments.exp_adversarial_robustness.run import run_experiment as run_adversarial_robustness
from experiments.exp1_causal_levels.run import run_experiment
from experiments.exp2_jury_ablation.run import run_experiment as run_exp2_jury_ablation
from experiments.exp3_difficulty.run import run_experiment as run_exp3_difficulty
from experiments.exp4_evolution.run import run_experiment as run_exp4_evolution
from experiments.exp_cross_model_transfer.run import run_experiment as run_cross_model_transfer
from experiments.exp_human_audit.run import run_experiment as run_human_audit
from experiments.exp_identifiability_ablation.run import run_experiment as run_identifiability_ablation
from experiments.exp_leakage_study.run import (
    DEFAULT_SAMPLES_PER_FAMILY as LEAKAGE_DEFAULT_SAMPLES_PER_FAMILY,
    _build_oracle_leaking_public_partition,
    _run_oracle_leaking_partition,
    _verifier_tool_context,
    run_experiment as run_leakage_study,
)
from experiments.exp_main_benchmark.run import run_experiment as run_main_benchmark
from experiments.exp_ood_generalization.run import run_experiment as run_ood_generalization
from game.config import ConfigLoader
from game.data_generator import DataGenerator
from game.debate_engine import DebateEngine
from game.difficulty import DifficultyController
from main import run_game
from run_live_game import _public_graph
from verifier.pipeline import VerifierPipeline
from visualization.api import VisualizationAPI


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_config_loader_reads_default_yaml(self) -> None:
        config = ConfigLoader().load()
        self.assertIn("game", config)
        self.assertIn("difficulty", config)
        self.assertGreaterEqual(config["game"]["max_rounds"], 1)

    def test_data_generator_returns_complete_scenario(self) -> None:
        generator = DataGenerator(seed=7)
        scenario = generator.generate_scenario(difficulty=0.55, causal_level=2)
        self.assertEqual(scenario.scenario_id, "education_income")
        self.assertFalse(scenario.observed_data.empty)
        self.assertTrue(set(scenario.hidden_variables).issubset(set(scenario.full_data.columns)))
        self.assertEqual(scenario.ground_truth["treatment"], "education_years")

    def test_difficulty_controller_moves_toward_target_band(self) -> None:
        controller = DifficultyController(
            {
                "target_deception_rate": 0.4,
                "window_size": 4,
                "adjustment_rate": 0.2,
                "min_difficulty": 0.2,
                "max_difficulty": 0.9,
                "initial_difficulty": 0.5,
                "tolerance": 0.05,
            }
        )

        for outcome in [True, True, True, True]:
            controller.update(outcome)
        higher = controller.get_difficulty()

        for outcome in [False, False, False, False]:
            controller.update(outcome)
        lower = controller.get_difficulty()

        self.assertGreater(higher, 0.5)
        self.assertLess(lower, higher)

    async def test_debate_engine_runs_mock_rounds(self) -> None:
        config = ConfigLoader().load()
        engine = DebateEngine(config)
        await engine.initialize()
        results = await engine.run_game(num_rounds=2)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("winner" in result for result in results))
        self.assertTrue(all(result["transcript"] for result in results))
        self.assertTrue(engine.evolution_tracker.records)
        self.assertTrue(
            any(record.deception_score > 0.0 for record in engine.evolution_tracker.records)
        )

    async def test_debate_engine_supports_fixed_level_schedule_without_evolution(self) -> None:
        config = ConfigLoader().load()
        engine = DebateEngine(config)
        await engine.initialize()
        results = await engine.run_game(
            num_rounds=2,
            level_schedule=[2, 2],
            use_evolution=False,
            update_difficulty=False,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["scenario"].causal_level == 2 for result in results))
        self.assertTrue(all(result["difficulty"] == results[0]["difficulty"] for result in results))

    async def test_main_run_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "run.json"
            payload = await run_game(rounds=1, output_path=str(output_path))

            self.assertTrue(output_path.exists())
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["n_rounds"], 1)
            self.assertEqual(payload["summary"]["n_rounds"], 1)
            self.assertEqual(payload["summary"]["primary_metric"], "verdict_accuracy")
            self.assertIn("verdict_metrics", payload["summary"])
            self.assertIn("protocol_metrics", payload["summary"])
            self.assertIn("tracking", payload)

    async def test_exp1_defaults_to_verdict_centric_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "exp1.json"
            payload = await run_experiment(rounds_per_level=1, output_path=str(output_path))

            self.assertTrue(output_path.exists())
            self.assertIn("levels", payload)
            for level_key, level_summary in payload["levels"].items():
                self.assertEqual(level_summary["primary_metric"], "verdict_accuracy")
                self.assertIn("verdict_metrics", level_summary)
                self.assertIn("appendix_metrics", level_summary)
                self.assertIn("verdict_accuracy_ci", level_summary)

    def test_phase4_experiments_emit_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            main_payload = run_main_benchmark(
                seeds=[0],
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_main_benchmark.json"),
            )
            leakage_payload = run_leakage_study(
                seeds=[0],
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_leakage_study.json"),
            )
            ablation_payload = run_identifiability_ablation(
                seeds=[0],
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_identifiability_ablation.json"),
            )
            robustness_payload = run_adversarial_robustness(
                seeds=[0],
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_adversarial_robustness.json"),
            )
            ood_payload = run_ood_generalization(
                seeds=[0],
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_ood_generalization.json"),
            )
            transfer_payload = run_cross_model_transfer(
                seeds=[0],
                samples_per_family=1,
                allow_surrogate_transfer=True,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_cross_model_transfer.json"),
            )
            human_payload = run_human_audit(
                seeds=[0],
                samples_per_family=1,
                audit_subset_size=4,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_human_audit.json"),
            )

            self.assertIn("test_iid", main_payload["aggregated_metrics"]["countermodel_grounded"])
            self.assertIn("test_ood", main_payload["aggregated_metrics"]["countermodel_grounded"])
            self.assertIn("verdict_accuracy", main_payload["aggregated_metrics"]["countermodel_grounded"]["test_iid"])
            self.assertFalse(main_payload["protocol"]["compliant"])
            self.assertTrue(main_payload["protocol"]["override_used"])
            for artifact_key in (
                "config",
                "seed_list",
                "aggregated_metrics",
                "ci",
            ):
                self.assertIn(artifact_key, main_payload["artifacts"])
                self.assertTrue(Path(main_payload["artifacts"][artifact_key]).exists())

            leaking_predictions = [
                record
                for record in leakage_payload["raw_predictions"]
                if record["system_name"] == "oracle_leaking_partition"
            ]
            self.assertTrue(leaking_predictions)
            self.assertTrue(all(not record["supports_public_only"] for record in leaking_predictions))
            self.assertTrue(
                all(
                    not record["verdict"]["metadata"].get("same_verifier_pipeline")
                    for record in leaking_predictions
                )
            )
            self.assertTrue(
                all(
                    record["verdict"]["metadata"].get("leakage_mode")
                    == "oracle_metadata_readout"
                    for record in leaking_predictions
                )
            )
            self.assertEqual(
                leakage_payload["config"]["attack_split_protocol"],
                "dedicated_attack_only_benchmark",
            )
            self.assertEqual(
                leakage_payload["config"]["attack_profile"],
                "hidden_information_aware",
            )
            self.assertIn("inflation", leakage_payload)
            self.assertIn("conclusion", leakage_payload)
            self.assertIn("mcnemar_significance", leakage_payload)
            self.assertFalse(leakage_payload["protocol"]["compliant"])
            self.assertTrue(leakage_payload["protocol"]["override_used"])

            for system_name in ("no_ledger", "no_countermodel", "no_abstention", "no_tools"):
                self.assertIn(system_name, ablation_payload["aggregated_metrics"])
                metrics = ablation_payload["aggregated_metrics"][system_name]["test_iid"]
                self.assertIn("invalid_claim_acceptance_rate", metrics)
                self.assertIn("unidentifiable_awareness", metrics)
            self.assertFalse(ablation_payload["protocol"]["compliant"])
            self.assertTrue(ablation_payload["protocol"]["override_used"])
            self.assertEqual(
                ablation_payload["significance"]["test_iid"]["estimand"],
                "seed_mean_verdict_accuracy",
            )

            self.assertEqual(
                sorted(robustness_payload["aggregated_metrics"].keys()),
                ["hidden_information_aware", "medium", "strong", "weak"],
            )
            self.assertFalse(robustness_payload["protocol"]["compliant"])
            self.assertTrue(robustness_payload["protocol"]["override_used"])
            self.assertTrue(robustness_payload["raw_predictions"])
            self.assertTrue(all(record["attack_name"] is not None for record in robustness_payload["raw_predictions"]))
            attack_texts_by_instance: dict[tuple[str, str], set[str]] = {}
            strengths_by_instance: dict[tuple[str, str], set[str]] = {}
            for record in robustness_payload["raw_predictions"]:
                key = (record["split"], record["instance_id"])
                attack_texts_by_instance.setdefault(key, set()).add(record["claim_text"])
                strengths_by_instance.setdefault(key, set()).add(record["attack_strength"])
            self.assertTrue(
                any(
                    len(strengths_by_instance[key]) >= 2 and len(attack_texts_by_instance[key]) >= 2
                    for key in attack_texts_by_instance
                )
            )
            self.assertIn("## Significance:", robustness_payload["markdown_summary"])

            for bucket_name in ("graph_family_ood", "lexical_ood", "variable_naming_ood"):
                self.assertIn(bucket_name, ood_payload["aggregated_metrics"])
                self.assertIn(bucket_name, ood_payload["ood_gap"])
                self.assertIn(bucket_name, ood_payload["significance"])
            self.assertFalse(ood_payload["protocol"]["compliant"])
            self.assertTrue(ood_payload["protocol"]["override_used"])
            self.assertIn("Sample Count", ood_payload["markdown_summary"])
            self.assertIn("## Significance:", ood_payload["markdown_summary"])

            self.assertGreaterEqual(len(transfer_payload["systems"]), 2)
            self.assertGreaterEqual(len(transfer_payload["attacker_families"]), 2)
            self.assertTrue(transfer_payload["raw_predictions"])
            self.assertIn(
                "attacker_family",
                transfer_payload["raw_predictions"][0],
            )
            self.assertFalse(transfer_payload["protocol"]["compliant"])
            self.assertTrue(transfer_payload["protocol"]["override_used"])

            self.assertIn("aggregated_metrics", human_payload)
            self.assertIn("test_iid", human_payload["aggregated_metrics"])
            self.assertIn("test_ood", human_payload["aggregated_metrics"])
            self.assertTrue(human_payload["annotation_package"])
            self.assertTrue(all(record["supports_public_only"] for record in human_payload["annotation_package"]))
            for field_name in (
                "public_evidence_summary",
                "observed_data",
                "supporting_evidence",
                "counter_evidence",
                "tool_trace",
                "verifier_probabilities",
            ):
                self.assertIn(field_name, human_payload["annotation_package"][0])
            self.assertIn(
                "witness_quality_reasonable",
                human_payload["annotation_package"][0]["annotation_questions"],
            )
            self.assertIn(
                "annotator_a_witness_quality_reasonable",
                human_payload["annotation_package"][0],
            )
            for artifact_path in human_payload["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists())

    def test_phase4_default_main_and_leakage_paths_match_paper_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            main_payload = run_main_benchmark(
                output_path=str(Path(tmp_dir) / "exp_main_benchmark_default.json"),
            )
            leakage_payload = run_leakage_study(
                output_path=str(Path(tmp_dir) / "exp_leakage_study_default.json"),
            )

            self.assertGreaterEqual(len(main_payload["seeds"]), 3)
            self.assertGreaterEqual(len(main_payload["systems"]), 2)
            self.assertNotIn("skeptical_family", main_payload["systems"])
            self.assertNotIn("optimistic_family", main_payload["systems"])
            self.assertIn("no_tools", main_payload["systems"])
            self.assertIn("no_countermodel", main_payload["systems"])
            self.assertIn("ECE", main_payload["markdown_summary"])
            self.assertIn("Brier", main_payload["markdown_summary"])

            main_metrics = main_payload["aggregated_metrics"]["countermodel_grounded"]["test_iid"]
            self.assertIn("ece", main_metrics)
            self.assertIn("brier", main_metrics)
            for field_name in ("mean", "std", "ci_lower", "ci_upper", "formatted"):
                self.assertIn(field_name, main_metrics["verdict_accuracy"])
            for split_name in ("test_iid", "test_ood"):
                self.assertIsNotNone(main_payload["significance"][split_name])
                self.assertTrue(main_payload["significance"][split_name]["comparisons"])
                self.assertEqual(
                    main_payload["significance"][split_name]["estimand"],
                    "seed_mean_verdict_accuracy",
                )
                self.assertEqual(main_payload["significance"][split_name]["method"], "paired_seed_bootstrap")
                self.assertEqual(
                    main_payload["significance"][split_name]["metric_name"],
                    "verdict_accuracy",
                )

            self.assertGreaterEqual(len(leakage_payload["seeds"]), 3)
            self.assertGreaterEqual(
                leakage_payload["config"]["samples_per_family"],
                LEAKAGE_DEFAULT_SAMPLES_PER_FAMILY,
            )
            self.assertIn(
                leakage_payload["conclusion"]["summary_statement"],
                leakage_payload["markdown_summary"],
            )
            self.assertIn("Supports Warning", leakage_payload["markdown_summary"])
            self.assertIn("95% CI", leakage_payload["markdown_summary"])
            self.assertIn("Macro-F1", leakage_payload["markdown_summary"])
            self.assertEqual(
                leakage_payload["global_multiple_comparison_correction"]["family_size"],
                4,
            )
            for split_name in ("test_iid", "test_ood"):
                self.assertTrue(leakage_payload["significance"][split_name]["comparisons"])
                self.assertTrue(leakage_payload["mcnemar_significance"][split_name]["comparisons"])
                self.assertEqual(
                    leakage_payload["significance"][split_name]["estimand"],
                    "seed_mean_verdict_accuracy",
                )
                self.assertEqual(
                    leakage_payload["mcnemar_significance"][split_name]["metric_name"],
                    "verdict_accuracy",
                )
                comparison = leakage_payload["significance"][split_name]["comparisons"][0]
                expected_delta = leakage_payload["inflation"][split_name]["accuracy"]["delta_mean"]
                self.assertAlmostEqual(comparison["observed_difference"], expected_delta)

    def test_phase4_leakage_control_reads_oracle_metadata_hint(self) -> None:
        run = build_seed_benchmark_run(seed=0, difficulty=0.55, samples_per_family=2)
        sample = next(
            candidate
            for candidate in run.split_samples["test_iid"]
            if candidate.claim.graph_family == "l2_invalid_iv_family"
        )
        study = _run_oracle_leaking_partition(sample)

        self.assertEqual(study["predicted_label"], sample.claim.gold_label.value)
        self.assertEqual(study["verdict"]["label"], sample.claim.gold_label.value)
        self.assertEqual(study["verdict"]["metadata"]["leakage_mode"], "oracle_metadata_readout")
        self.assertFalse(study["supports_public_only"])

    def test_phase4_main_benchmark_rejects_oracle_leaking_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_main_benchmark(
                    systems=["countermodel_grounded", "oracle_leaking_partition"],
                    output_path=str(Path(tmp_dir) / "exp_main_benchmark_oracle.json"),
                )

    def test_phase4_main_benchmark_uses_seed_level_significance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = run_main_benchmark(
                output_path=str(Path(tmp_dir) / "exp_main_benchmark_significance.json"),
            )

            for split_name in ("test_iid", "test_ood"):
                report = payload["significance"][split_name]
                self.assertEqual(report["method"], "paired_seed_bootstrap")
                self.assertEqual(report["metric_name"], "verdict_accuracy")
                self.assertEqual(report["estimand"], "seed_mean_verdict_accuracy")

    def test_phase4_main_and_leakage_reject_noncompliant_seed_lists_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_main_benchmark(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_main_benchmark_invalid.json"),
                )
            with self.assertRaises(ValueError):
                run_leakage_study(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_leakage_study_invalid.json"),
                )

    def test_phase4_s3_s5_runners_reject_noncompliant_seed_lists_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_identifiability_ablation(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_identifiability_ablation_invalid.json"),
                )
            with self.assertRaises(ValueError):
                run_adversarial_robustness(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_adversarial_robustness_invalid.json"),
                )
            with self.assertRaises(ValueError):
                run_ood_generalization(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_ood_generalization_invalid.json"),
                )
            with self.assertRaises(ValueError):
                run_cross_model_transfer(
                    seeds=[0],
                    allow_surrogate_transfer=True,
                    output_path=str(Path(tmp_dir) / "exp_cross_model_transfer_invalid.json"),
                )
            with self.assertRaises(ValueError):
                run_human_audit(
                    seeds=[0],
                    output_path=str(Path(tmp_dir) / "exp_human_audit_invalid.json"),
                )

    def test_phase4_payload_records_effective_config_separately_from_requested_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            main_payload = run_main_benchmark(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_main_benchmark_effective.json"),
            )
            leakage_payload = run_leakage_study(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_leakage_study_effective.json"),
            )
            ablation_payload = run_identifiability_ablation(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_identifiability_ablation_effective.json"),
            )
            robustness_payload = run_adversarial_robustness(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_adversarial_robustness_effective.json"),
            )
            ood_payload = run_ood_generalization(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_ood_generalization_effective.json"),
            )
            transfer_payload = run_cross_model_transfer(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_surrogate_transfer=True,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_cross_model_transfer_effective.json"),
            )
            human_payload = run_human_audit(
                seeds=[0],
                samples_per_family=0,
                difficulty=1.5,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_human_audit_effective.json"),
            )

            self.assertEqual(main_payload["config"]["samples_per_family"], 1)
            self.assertEqual(main_payload["config"]["difficulty"], 1.0)
            self.assertEqual(main_payload["requested_config"]["samples_per_family"], 0)
            self.assertEqual(main_payload["requested_config"]["difficulty"], 1.5)

            self.assertEqual(leakage_payload["config"]["samples_per_family"], 2)
            self.assertEqual(leakage_payload["config"]["difficulty"], 1.0)
            self.assertEqual(leakage_payload["requested_config"]["samples_per_family"], 0)
            self.assertEqual(leakage_payload["requested_config"]["difficulty"], 1.5)

            for payload in (
                ablation_payload,
                ood_payload,
                human_payload,
            ):
                self.assertEqual(payload["config"]["samples_per_family"], 1)
                self.assertEqual(payload["config"]["difficulty"], 1.0)
                self.assertEqual(payload["requested_config"]["samples_per_family"], 0)
                self.assertEqual(payload["requested_config"]["difficulty"], 1.5)

            for payload in (robustness_payload, transfer_payload):
                self.assertEqual(payload["config"]["samples_per_family"], 2)
                self.assertEqual(payload["config"]["difficulty"], 1.0)
                self.assertEqual(payload["requested_config"]["samples_per_family"], 0)
                self.assertEqual(payload["requested_config"]["difficulty"], 1.5)

    def test_phase4_protocol_marks_toy_scale_runs_exploratory_even_with_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            robustness_payload = run_adversarial_robustness(
                samples_per_family=1,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_adversarial_robustness_toy.json"),
            )
            transfer_payload = run_cross_model_transfer(
                samples_per_family=1,
                allow_surrogate_transfer=True,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_cross_model_transfer_toy.json"),
            )
            human_payload = run_human_audit(
                samples_per_family=1,
                audit_subset_size=150,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_human_audit_toy.json"),
            )

            self.assertFalse(robustness_payload["protocol"]["compliant"])
            self.assertTrue(robustness_payload["protocol"]["override_used"])
            self.assertIn("samples_per_family", robustness_payload["protocol"]["violations"])

            self.assertFalse(transfer_payload["protocol"]["compliant"])
            self.assertTrue(transfer_payload["protocol"]["override_used"])
            self.assertIn("samples_per_family", transfer_payload["protocol"]["violations"])

            self.assertFalse(human_payload["protocol"]["compliant"])
            self.assertTrue(human_payload["protocol"]["override_used"])
            self.assertIn("samples_per_family", human_payload["protocol"]["violations"])
            self.assertIn("audit_subset_size", human_payload["protocol"]["violations"])

    def test_phase4_main_runner_rejects_duplicate_systems_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_main_benchmark(
                    systems=["countermodel_grounded", "countermodel_grounded"],
                    output_path=str(Path(tmp_dir) / "exp_main_benchmark_duplicate_systems.json"),
                )

    def test_phase4_runners_reject_duplicate_seeds_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_main_benchmark(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_main_benchmark_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_leakage_study(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_leakage_study_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_identifiability_ablation(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_identifiability_ablation_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_adversarial_robustness(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_adversarial_robustness_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_ood_generalization(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_ood_generalization_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_cross_model_transfer(
                    seeds=[0, 0, 1],
                    allow_surrogate_transfer=True,
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_cross_model_transfer_duplicate_seeds.json"),
                )
            with self.assertRaises(ValueError):
                run_human_audit(
                    seeds=[0, 0, 1],
                    allow_protocol_violations=True,
                    output_path=str(Path(tmp_dir) / "exp_human_audit_duplicate_seeds.json"),
                )

    def test_human_audit_round_trips_csv_and_rejects_misaligned_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "exp_human_audit_base.json"
            payload = run_human_audit(
                seeds=[0],
                samples_per_family=1,
                audit_subset_size=4,
                allow_protocol_violations=True,
                output_path=str(base_path),
            )

            csv_payload = run_human_audit(
                seeds=[0],
                samples_per_family=1,
                audit_subset_size=4,
                allow_protocol_violations=True,
                annotations_path=payload["artifacts"]["annotation_package_csv"],
                output_path=str(Path(tmp_dir) / "exp_human_audit_from_csv.json"),
            )
            self.assertIsNotNone(csv_payload["agreement_stats"])
            self.assertIn("causal_level", csv_payload["annotation_package"][0])
            self.assertEqual(
                csv_payload["annotation_package"][0]["causal_level"],
                csv_payload["annotation_package"][0]["public_evidence_summary"]["causal_level"],
            )
            self.assertIn(
                "witness_quality_reasonable",
                csv_payload["annotation_package"][0]["annotation_questions"],
            )
            self.assertIn(
                "annotator_a_witness_quality_reasonable",
                csv_payload["annotation_package"][0],
            )
            self.assertNotIn(
                "annotator_a_countermodel_witness_persuasive",
                csv_payload["annotation_package"][0],
            )

            fake_annotations = [
                {
                    "audit_id": "fake::case",
                    "annotator_a_gold_label_reasonable": "yes",
                    "annotator_b_gold_label_reasonable": "yes",
                    "annotator_a_verifier_label_reasonable": "yes",
                    "annotator_b_verifier_label_reasonable": "yes",
                    "annotator_a_witness_quality_reasonable": "yes",
                    "annotator_b_witness_quality_reasonable": "yes",
                    "annotator_a_explanation_faithful": "yes",
                    "annotator_b_explanation_faithful": "yes",
                }
            ]
            fake_path = Path(tmp_dir) / "fake_annotations.json"
            fake_path.write_text(json.dumps(fake_annotations), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_human_audit(
                    seeds=[0],
                    samples_per_family=1,
                    audit_subset_size=4,
                    allow_protocol_violations=True,
                    annotations_path=str(fake_path),
                    output_path=str(Path(tmp_dir) / "exp_human_audit_fake.json"),
                )

    def test_phase4_ood_runner_keeps_primary_buckets_pure_and_routes_overlap_to_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = run_ood_generalization(
                seeds=[0],
                samples_per_family=10,
                allow_protocol_violations=True,
                output_path=str(Path(tmp_dir) / "exp_ood_generalization_pure.json"),
            )

            seed_bucket_results = payload["per_seed_bucket_results"][0]
            for bucket_name, expected_reason in (
                ("graph_family_ood", ["family_holdout"]),
                ("lexical_ood", ["lexical_holdout"]),
                ("variable_naming_ood", ["variable_renaming_holdout"]),
            ):
                for record in seed_bucket_results[bucket_name]["predictions"]:
                    self.assertEqual(record["ood_reasons"], expected_reason)
            for record in seed_bucket_results["mixed_ood"]["predictions"]:
                self.assertGreater(len(record["ood_reasons"]), 1)

    def test_cross_model_transfer_requires_explicit_surrogate_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_cross_model_transfer(
                    seeds=[0, 1, 2],
                    output_path=str(Path(tmp_dir) / "exp_cross_model_transfer_default_guard.json"),
                )

    def test_human_audit_rejects_duplicate_ids_and_stale_package_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "exp_human_audit_base.json"
            payload = run_human_audit(
                seeds=[0],
                samples_per_family=1,
                audit_subset_size=4,
                allow_protocol_violations=True,
                output_path=str(base_path),
            )
            package = json.loads(Path(payload["artifacts"]["annotation_package_json"]).read_text(encoding="utf-8"))

            duplicate_annotations = [dict(package[0]), dict(package[0]), dict(package[2]), dict(package[3])]
            duplicate_path = Path(tmp_dir) / "duplicate_annotations.json"
            duplicate_path.write_text(json.dumps(duplicate_annotations, ensure_ascii=False), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_human_audit(
                    seeds=[0],
                    samples_per_family=1,
                    audit_subset_size=4,
                    allow_protocol_violations=True,
                    annotations_path=str(duplicate_path),
                    output_path=str(Path(tmp_dir) / "exp_human_audit_duplicate.json"),
                )

            stale_annotations = [dict(row) for row in package]
            stale_annotations[0]["reasoning_summary"] = "stale package content"
            stale_path = Path(tmp_dir) / "stale_annotations.json"
            stale_path.write_text(json.dumps(stale_annotations, ensure_ascii=False), encoding="utf-8")

            with self.assertRaises(ValueError):
                run_human_audit(
                    seeds=[0],
                    samples_per_family=1,
                    audit_subset_size=4,
                    allow_protocol_violations=True,
                    annotations_path=str(stale_path),
                    output_path=str(Path(tmp_dir) / "exp_human_audit_stale.json"),
                )

    async def test_phase5_appendix_and_demo_stay_on_public_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp2_payload = await run_exp2_jury_ablation(
                rounds=1,
                output_path=str(Path(tmp_dir) / "exp2_jury_ablation.json"),
            )
            exp3_payload = await run_exp3_difficulty(
                rounds=1,
                output_path=str(Path(tmp_dir) / "exp3_difficulty.json"),
            )
            exp4_payload = await run_exp4_evolution(
                rounds=1,
                output_path=str(Path(tmp_dir) / "exp4_evolution.json"),
            )

            for condition in exp2_payload["conditions"].values():
                self.assertTrue(condition["appendix_only"])
                self.assertTrue(condition["public_schema_only"])
                self.assertIn("verdict_accuracy", condition["verdict_metrics"])
            for condition in exp3_payload["conditions"].values():
                self.assertTrue(condition["appendix_only"])
                self.assertTrue(condition["public_schema_only"])
                self.assertIn("verdict_accuracy", condition["verdict_metrics"])
            self.assertTrue(exp4_payload["without_evolution"]["appendix_only"])
            self.assertTrue(exp4_payload["without_evolution"]["public_schema_only"])
            self.assertTrue(exp4_payload["with_evolution"]["appendix_only"])
            self.assertTrue(exp4_payload["with_evolution"]["public_schema_only"])

        scenario = DataGenerator(seed=11).generate_scenario(difficulty=0.55, causal_level=2)
        public_scenario = ensure_public_instance(scenario)
        graph = _public_graph(
            public_scenario,
            claim_text=f"{public_scenario.variables[0]} causes {public_scenario.variables[-1]}",
            causal_level=public_scenario.causal_level,
        )

        self.assertEqual(graph["schema_view"], "public")
        self.assertFalse(any(node["id"] in set(scenario.hidden_variables) for node in graph["nodes"]))

        api = VisualizationAPI()
        api.register_game(
            "demo_public",
            {
                "scenario": {
                    "scenario_id": public_scenario.scenario_id,
                    "variables": list(public_scenario.variables),
                    "proxy_variables": list(public_scenario.proxy_variables),
                    "selection_variables": list(public_scenario.selection_variables),
                    "selection_mechanism": public_scenario.selection_mechanism,
                    "causal_level": public_scenario.causal_level,
                    "hidden_variables": list(scenario.hidden_variables),
                    "true_dag": {"leak": True},
                },
                "results": [
                    {
                        "agent_a_claim": {
                            "causal_claim": f"{public_scenario.variables[0]} causes {public_scenario.variables[-1]}",
                        }
                    }
                ],
            },
        )
        graph_payload = api.get_causal_graph_data("demo_public")
        node_ids = {node["id"] for node in graph_payload["nodes"]}

        self.assertEqual(graph_payload["schema_view"], "public")
        self.assertFalse(any(hidden in node_ids for hidden in scenario.hidden_variables))
        self.assertTrue(any(link["type"] == "claimed" for link in graph_payload["links"]))

    def test_experiment_tracker_writes_run_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(
                ExperimentConfig(experiment_id="unit_test", name="Unit Test Run"),
                log_dir=str(tmp_dir),
            )
            tracker.init()
            tracker.log_metrics({"score": 1.0}, step=1)
            tracker.log_round(1, {"winner": "agent_b"})
            checkpoint = tracker.save_checkpoint({"state": "ok"})
            summary = tracker.finish()
            run_dir = Path(summary["run_dir"])

            self.assertTrue((run_dir / "metrics.jsonl").exists())
            self.assertTrue((run_dir / "rounds.jsonl").exists())
            self.assertTrue(checkpoint.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
