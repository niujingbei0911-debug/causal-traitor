"""Run the level-by-level causal benchmark."""

from __future__ import annotations

import argparse
import asyncio
import csv
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
    _write_exp1_sidecars(target, summary)
    return summary


def _write_exp1_sidecars(json_path: Path, summary: dict[str, Any]) -> None:
    csv_path = json_path.with_suffix(".csv")
    md_path = json_path.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "rounds",
                "deception_success_rate",
                "detection_accuracy_proxy",
                "causal_validity_mean",
            ]
        )
        for level_key, level_summary in summary.get("levels", {}).items():
            writer.writerow(
                [
                    level_key,
                    level_summary.get("rounds", 0),
                    f"{level_summary.get('deception_success_rate', 0.0):.4f}",
                    f"{level_summary.get('detection_accuracy_proxy', 0.0):.4f}",
                    f"{level_summary.get('causal_validity_mean', 0.0):.4f}",
                ]
            )
    lines = ["# Experiment 1 — Pearl Ladder Benchmark", ""]
    lines.append("| Level | Rounds | DSR | DAcc (proxy) | CLS mean |")
    lines.append("| --- | --- | --- | --- | --- |")
    for level_key, level_summary in summary.get("levels", {}).items():
        lines.append(
            f"| {level_key} | {level_summary.get('rounds', 0)} | "
            f"{level_summary.get('deception_success_rate', 0.0):.3f} | "
            f"{level_summary.get('detection_accuracy_proxy', 0.0):.3f} | "
            f"{level_summary.get('causal_validity_mean', 0.0):.3f} |"
        )
    tracking = summary.get("tracking", {})
    if tracking.get("run_dir"):
        lines.append("")
        lines.append(f"Tracking artifacts: `{tracking['run_dir']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
