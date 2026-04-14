from __future__ import annotations

import json

import pytest

from game.config import ConfigLoader
from game.data_generator import DataGenerator
from game.debate_engine import DebateEngine
from game.difficulty import DifficultyController
from main import run_game


def test_config_loader_reads_default_yaml() -> None:
    config = ConfigLoader().load()
    assert "game" in config
    assert "difficulty" in config
    assert config["game"]["max_rounds"] >= 1


def test_data_generator_returns_complete_scenario() -> None:
    generator = DataGenerator(seed=7)
    scenario = generator.generate_scenario(difficulty=0.55, causal_level=2)
    assert scenario.scenario_id == "education_income"
    assert not scenario.observed_data.empty
    assert set(scenario.hidden_variables).issubset(set(scenario.full_data.columns))
    assert scenario.ground_truth["treatment"] == "education_years"


def test_difficulty_controller_moves_toward_target_band() -> None:
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

    assert higher > 0.5
    assert lower < higher


@pytest.mark.asyncio
async def test_debate_engine_runs_mock_rounds() -> None:
    config = ConfigLoader().load()
    engine = DebateEngine(config)
    await engine.initialize()
    results = await engine.run_game(num_rounds=2)

    assert len(results) == 2
    assert all("winner" in result for result in results)
    assert all(result["transcript"] for result in results)
    assert engine.evolution_tracker.records


@pytest.mark.asyncio
async def test_main_run_writes_json_output(tmp_path) -> None:
    output_path = tmp_path / "run.json"
    payload = await run_game(rounds=1, output_path=str(output_path))

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["summary"]["n_rounds"] == 1
    assert payload["summary"]["n_rounds"] == 1
