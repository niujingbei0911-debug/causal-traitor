"""Run the level-by-level causal benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.debate_engine import DebateEngine


async def run_experiment(
    rounds_per_level: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    config = ConfigLoader().load()
    engine = DebateEngine(config)
    await engine.initialize()
    tracker = ExperimentTracker(
        ExperimentConfig(
            experiment_id="exp1_causal_levels",
            name="Pearl Ladder Benchmark",
            params={"rounds_per_level": rounds_per_level},
        ),
        log_dir=config.get("logging", {}).get("save_dir", "logs"),
        use_wandb=bool(config.get("logging", {}).get("use_wandb", False)),
    )
    tracker.init()

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
            tracker.log_round(round_index + (level - 1) * rounds_per_level, result)
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
        tracker.log_metrics(
            {
                f"L{level}_deception_success_rate": summary["levels"][f"L{level}"]["deception_success_rate"],
                f"L{level}_detection_accuracy_proxy": summary["levels"][f"L{level}"]["detection_accuracy_proxy"],
                f"L{level}_causal_validity_mean": summary["levels"][f"L{level}"]["causal_validity_mean"],
            },
            step=level,
        )

    tracker.log_artifact("exp1_summary", summary, artifact_type="json")
    summary["tracking"] = tracker.finish()
    target = Path(output_path or "outputs/exp1_causal_levels.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pearl ladder benchmark experiment.")
    parser.add_argument("--rounds-per-level", type=int, default=20, help="Rounds to run for each causal level.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(
        run_experiment(
            rounds_per_level=args.rounds_per_level,
            output_path=args.output,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
