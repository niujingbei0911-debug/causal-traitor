from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.schema import ensure_public_instance
from evaluation.tracker import ExperimentConfig, ExperimentTracker
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
                output_path=str(Path(tmp_dir) / "exp_main_benchmark.json"),
            )
            leakage_payload = run_leakage_study(
                seeds=[0],
                samples_per_family=1,
                output_path=str(Path(tmp_dir) / "exp_leakage_study.json"),
            )
            ablation_payload = run_identifiability_ablation(
                seeds=[0],
                samples_per_family=1,
                output_path=str(Path(tmp_dir) / "exp_identifiability_ablation.json"),
            )
            robustness_payload = run_adversarial_robustness(
                seeds=[0],
                samples_per_family=1,
                output_path=str(Path(tmp_dir) / "exp_adversarial_robustness.json"),
            )
            ood_payload = run_ood_generalization(
                seeds=[0],
                samples_per_family=1,
                output_path=str(Path(tmp_dir) / "exp_ood_generalization.json"),
            )
            transfer_payload = run_cross_model_transfer(
                seeds=[0],
                samples_per_family=1,
                output_path=str(Path(tmp_dir) / "exp_cross_model_transfer.json"),
            )
            human_payload = run_human_audit(
                seeds=[0],
                samples_per_family=1,
                audit_subset_size=4,
                output_path=str(Path(tmp_dir) / "exp_human_audit.json"),
            )

            self.assertIn("test_iid", main_payload["aggregated_metrics"]["countermodel_grounded"])
            self.assertIn("test_ood", main_payload["aggregated_metrics"]["countermodel_grounded"])
            self.assertIn("verdict_accuracy", main_payload["aggregated_metrics"]["countermodel_grounded"]["test_iid"])

            leaking_predictions = [
                record
                for record in leakage_payload["raw_predictions"]
                if record["system_name"] == "oracle_leaking_partition"
            ]
            self.assertTrue(leaking_predictions)
            self.assertTrue(all(not record["supports_public_only"] for record in leaking_predictions))
            self.assertIn("inflation", leakage_payload)
            self.assertIn("conclusion", leakage_payload)
            self.assertIn("mcnemar_significance", leakage_payload)

            for system_name in ("no_ledger", "no_countermodel", "no_abstention", "no_tools"):
                self.assertIn(system_name, ablation_payload["aggregated_metrics"])
                metrics = ablation_payload["aggregated_metrics"][system_name]["test_iid"]
                self.assertIn("invalid_claim_acceptance_rate", metrics)
                self.assertIn("unidentifiable_awareness", metrics)

            self.assertEqual(sorted(robustness_payload["aggregated_metrics"].keys()), ["medium", "strong", "weak"])
            self.assertIn("ood_gap", ood_payload)
            self.assertGreaterEqual(len(transfer_payload["systems"]), 2)

            self.assertIn("aggregated_metrics", human_payload)
            self.assertIn("test_iid", human_payload["aggregated_metrics"])
            self.assertIn("test_ood", human_payload["aggregated_metrics"])
            self.assertTrue(human_payload["annotation_package"])
            self.assertTrue(all(record["supports_public_only"] for record in human_payload["annotation_package"]))
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

            self.assertGreaterEqual(len(leakage_payload["seeds"]), 3)
            self.assertGreaterEqual(
                leakage_payload["config"]["samples_per_family"],
                LEAKAGE_DEFAULT_SAMPLES_PER_FAMILY,
            )
            self.assertTrue(leakage_payload["conclusion"]["supported"])
            self.assertTrue(leakage_payload["conclusion"]["robust_support"])
            self.assertIn(
                leakage_payload["conclusion"]["summary_statement"],
                leakage_payload["markdown_summary"],
            )
            self.assertIn("Supports Warning", leakage_payload["markdown_summary"])
            for split_name in ("test_iid", "test_ood"):
                self.assertTrue(leakage_payload["conclusion"]["split_details"][split_name]["supported"])
                self.assertTrue(leakage_payload["significance"][split_name]["comparisons"])
                self.assertTrue(leakage_payload["mcnemar_significance"][split_name]["comparisons"])

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
