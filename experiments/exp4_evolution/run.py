"""Compare fixed independent rounds against evolution-enabled rounds."""

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


async def _run_without_evolution(engine: DebateEngine, rounds: int, level: int) -> list[dict[str, Any]]:
    return await engine.run_game(
        num_rounds=rounds,
        level_schedule=[level] * rounds,
        use_evolution=False,
        update_difficulty=False,
    )


async def run_experiment(
    rounds: int = 10,
    level: int = 2,
    output_path: str | None = None,
) -> dict[str, Any]:
    config = ConfigLoader().load()
    tracker = ExperimentTracker(
        ExperimentConfig(
            experiment_id="exp4_evolution",
            name="Evolution vs No Evolution",
            params={"rounds": rounds, "level": level},
        ),
        log_dir=config.get("logging", {}).get("save_dir", "logs"),
        use_wandb=bool(config.get("logging", {}).get("use_wandb", False)),
    )
    tracker.init()

    no_evolution_engine = DebateEngine(config)
    await no_evolution_engine.initialize()
    independent_results = await _run_without_evolution(no_evolution_engine, rounds, level)
    for result in independent_results:
        tracker.log_round(result["round_number"], {"condition": "without_evolution", **result})

    evolution_engine = DebateEngine(config)
    await evolution_engine.initialize()
    evolution_results = await evolution_engine.run_game(
        num_rounds=rounds,
        level_schedule=[level] * rounds,
        use_evolution=True,
        update_difficulty=False,
    )
    for result in evolution_results:
        tracker.log_round(result["round_number"], {"condition": "with_evolution", **result})

    payload = {
        "without_evolution": {
            "rounds": rounds,
            "agent_a_win_rate": sum(r["winner"] == "agent_a" for r in independent_results) / rounds,
            "results": [
                {
                    "round_number": result.get("round_number"),
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
    tracker.log_metrics(
        {
            "without_evolution_agent_a_win_rate": payload["without_evolution"]["agent_a_win_rate"],
            "with_evolution_agent_a_win_rate": payload["with_evolution"]["agent_a_win_rate"],
            "with_evolution_arms_race_index": payload["with_evolution"]["arms_race_index"],
        },
        step=rounds,
    )

    tracker.log_artifact("exp4_summary", payload, artifact_type="json")
    payload["tracking"] = tracker.finish()
    target = Path(output_path or "outputs/exp4_evolution.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_exp4_sidecars(target, payload)
    return payload


def _write_exp4_sidecars(json_path: Path, payload: dict[str, Any]) -> None:
    csv_path = json_path.with_suffix(".csv")
    md_path = json_path.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "round", "winner", "deception_success"])
        for result in payload.get("without_evolution", {}).get("results", []):
            writer.writerow(
                [
                    "without_evolution",
                    result.get("round_number", ""),
                    result.get("winner", ""),
                    int(bool(result.get("deception_success", False))),
                ]
            )
        for index, result in enumerate(payload.get("with_evolution", {}).get("results", []), start=1):
            writer.writerow(
                [
                    "with_evolution",
                    index,
                    result.get("winner", ""),
                    int(bool(result.get("deception_success", False))),
                ]
            )
    without = payload.get("without_evolution", {})
    with_evo = payload.get("with_evolution", {})
    lines = [
        "# Experiment 4 — Evolution vs No Evolution",
        "",
        f"- Rounds per condition: {without.get('rounds', 0)}",
        f"- Without evolution — Agent A win rate: {without.get('agent_a_win_rate', 0.0):.3f}",
        f"- With evolution — Agent A win rate: {with_evo.get('agent_a_win_rate', 0.0):.3f}",
        f"- Arms race index (with evolution): {with_evo.get('arms_race_index', 0.0):.3f}",
        f"- Strategy converged (with evolution): {with_evo.get('converged', False)}",
    ]
    tracking = payload.get("tracking", {})
    if tracking.get("run_dir"):
        lines.append("")
        lines.append(f"Tracking artifacts: `{tracking['run_dir']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evolution-vs-no-evolution experiment.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds per condition.")
    parser.add_argument("--level", type=int, default=2, help="Pearl level to use for the independent condition.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(
        run_experiment(
            rounds=args.rounds,
            level=args.level,
            output_path=args.output,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
