"""Run the level-by-level causal benchmark."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from game.config import ConfigLoader
from game.debate_engine import DebateEngine


async def run_experiment(
    rounds_per_level: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    config = ConfigLoader().load()
    engine = DebateEngine(config)
    await engine.initialize()

    summary: dict[str, Any] = {"levels": {}}
    for level in config.get("game", {}).get("causal_levels", [1, 2, 3]):
        level_results = []
        difficulty = config.get("game", {}).get("initial_difficulty", 0.5)
        for round_index in range(1, rounds_per_level + 1):
            scenario = engine.data_generator.generate_scenario(
                difficulty=difficulty,
                causal_level=level,
            )
            result = await engine.run_round(
                scenario,
                round_number=round_index,
                evolution_context=None,
            )
            level_results.append(
                {
                    "winner": result["winner"],
                    "deception_success": result["deception_success"],
                    "causal_validity_score": result["audit_verdict"]["causal_validity_score"],
                    "jury_winner": result["jury_verdict"]["final_winner"],
                }
            )
        summary["levels"][f"L{level}"] = {
            "rounds": rounds_per_level,
            "deception_success_rate": sum(r["deception_success"] for r in level_results) / rounds_per_level,
            "detection_accuracy_proxy": sum(r["winner"] == "agent_b" for r in level_results) / rounds_per_level,
            "causal_validity_mean": sum(r["causal_validity_score"] for r in level_results) / rounds_per_level,
            "results": level_results,
        }

    target = Path(output_path or "outputs/exp1_causal_levels.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    payload = asyncio.run(run_experiment())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
