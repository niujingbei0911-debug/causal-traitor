"""Compare fixed independent rounds against evolution-enabled rounds."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from game.config import ConfigLoader
from game.debate_engine import DebateEngine


async def _run_without_evolution(engine: DebateEngine, rounds: int, level: int) -> list[dict[str, Any]]:
    results = []
    difficulty = engine.config.get("game", {}).get("initial_difficulty", 0.5)
    for round_index in range(1, rounds + 1):
        scenario = engine.data_generator.generate_scenario(difficulty=difficulty, causal_level=level)
        result = await engine.run_round(
            scenario,
            round_number=round_index,
            evolution_context=None,
        )
        results.append(result)
    return results


async def run_experiment(
    rounds: int = 10,
    level: int = 2,
    output_path: str | None = None,
) -> dict[str, Any]:
    config = ConfigLoader().load()

    no_evolution_engine = DebateEngine(config)
    await no_evolution_engine.initialize()
    independent_results = await _run_without_evolution(no_evolution_engine, rounds, level)

    evolution_engine = DebateEngine(config)
    await evolution_engine.initialize()
    evolution_results = await evolution_engine.run_game(num_rounds=rounds)

    payload = {
        "without_evolution": {
            "rounds": rounds,
            "agent_a_win_rate": sum(r["winner"] == "agent_a" for r in independent_results) / rounds,
            "results": [
                {
                    "winner": result["winner"],
                    "deception_success": result["deception_success"],
                }
                for result in independent_results
            ],
        },
        "with_evolution": {
            "rounds": rounds,
            "agent_a_win_rate": sum(r["winner"] == "agent_a" for r in evolution_results) / rounds,
            "arms_race_index": evolution_engine.evolution_tracker.get_arms_race_index(),
            "converged": evolution_engine.evolution_tracker.detect_convergence(),
            "results": [
                {
                    "winner": result["winner"],
                    "deception_success": result["deception_success"],
                    "snapshot": result["evolution_snapshot"],
                }
                for result in evolution_results
            ],
        },
    }

    target = Path(output_path or "outputs/exp4_evolution.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = asyncio.run(run_experiment())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
