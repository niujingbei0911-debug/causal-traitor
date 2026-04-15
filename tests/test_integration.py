from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.data_generator import DataGenerator
from game.debate_engine import DebateEngine
from game.difficulty import DifficultyController
from main import run_game


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
            self.assertIn("tracking", payload)

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
