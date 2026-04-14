"""Difficulty comparison experiment — fixed easy / fixed hard / dynamic."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.debate_engine import DebateEngine


async def _run_condition(
    base_config: dict[str, Any],
    condition: str,
    rounds: int,
) -> list[dict[str, Any]]:
    """Run *rounds* debates under a specific difficulty regime."""
    config = deepcopy(base_config)
    if condition == "fixed_easy":
        config.setdefault("game", {})["initial_difficulty"] = 0.25
        update_difficulty = False
    elif condition == "fixed_hard":
        config.setdefault("game", {})["initial_difficulty"] = 0.85
        update_difficulty = False
    else:  # dynamic
        config.setdefault("game", {})["initial_difficulty"] = 0.5
        update_difficulty = True

    engine = DebateEngine(config)
    await engine.initialize()
    return await engine.run_game(
        num_rounds=rounds,
        use_evolution=False,
        update_difficulty=update_difficulty,
    )


async def run_experiment(
    rounds: int = 30,
    output_path: str | None = None,
) -> dict[str, Any]:
    base_config = ConfigLoader().load()
    tracker = ExperimentTracker(
        ExperimentConfig(
            experiment_id="exp3_difficulty",
            name="Difficulty Comparison (easy / hard / dynamic)",
            params={"rounds": rounds},
        ),
        log_dir=base_config.get("logging", {}).get("save_dir", "logs"),
        use_wandb=bool(base_config.get("logging", {}).get("use_wandb", False)),
    )
    tracker.init()

    conditions = ["fixed_easy", "fixed_hard", "dynamic"]
    payload: dict[str, Any] = {"conditions": {}}

    for condition in conditions:
        results = await _run_condition(base_config, condition, rounds)
        for r in results:
            tracker.log_round(r["round_number"], {"condition": condition, **r})

        wins_a = sum(r["winner"] == "agent_a" for r in results)
        causal_scores = [r["audit_verdict"]["causal_validity_score"] for r in results]
        difficulties = [r["difficulty"] for r in results]
        payload["conditions"][condition] = {
            "rounds": rounds,
            "agent_a_win_rate": wins_a / rounds,
            "causal_validity_mean": sum(causal_scores) / len(causal_scores),
            "difficulty_mean": sum(difficulties) / len(difficulties),
            "difficulty_std": (
                (sum((d - sum(difficulties) / len(difficulties)) ** 2 for d in difficulties) / len(difficulties))
                ** 0.5
            ),
            "results": [
                {
                    "round": r["round_number"],
                    "winner": r["winner"],
                    "deception_success": r["deception_success"],
                    "difficulty": r["difficulty"],
                    "causal_validity": r["audit_verdict"]["causal_validity_score"],
                }
                for r in results
            ],
        }
        tracker.log_metrics(
            {
                f"{condition}_agent_a_win_rate": payload["conditions"][condition]["agent_a_win_rate"],
                f"{condition}_causal_validity_mean": payload["conditions"][condition]["causal_validity_mean"],
                f"{condition}_difficulty_mean": payload["conditions"][condition]["difficulty_mean"],
            },
            step=conditions.index(condition),
        )

    tracker.log_artifact("exp3_summary", payload, artifact_type="json")
    payload["tracking"] = tracker.finish()
    target = Path(output_path or "outputs/exp3_difficulty.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_exp3_sidecars(target, payload)
    return payload


def _write_exp3_sidecars(json_path: Path, payload: dict[str, Any]) -> None:
    csv_path = json_path.with_suffix(".csv")
    md_path = json_path.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "condition", "rounds", "agent_a_win_rate",
            "causal_validity_mean", "difficulty_mean", "difficulty_std",
        ])
        for label, cond in payload.get("conditions", {}).items():
            writer.writerow([
                label,
                cond["rounds"],
                f"{cond['agent_a_win_rate']:.4f}",
                f"{cond['causal_validity_mean']:.4f}",
                f"{cond['difficulty_mean']:.4f}",
                f"{cond['difficulty_std']:.4f}",
            ])

    lines = [
        "# Experiment 3 — Difficulty Comparison",
        "",
        "| Condition | Rounds | A Win Rate | Causal Validity | Diff Mean | Diff Std |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, cond in payload.get("conditions", {}).items():
        lines.append(
            f"| {label} | {cond['rounds']} | "
            f"{cond['agent_a_win_rate']:.3f} | "
            f"{cond['causal_validity_mean']:.3f} | "
            f"{cond['difficulty_mean']:.3f} | "
            f"{cond['difficulty_std']:.3f} |"
        )
    tracking = payload.get("tracking", {})
    if tracking.get("run_dir"):
        lines.append("")
        lines.append(f"Tracking artifacts: `{tracking['run_dir']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the difficulty comparison experiment.")
    parser.add_argument("--rounds", type=int, default=30, help="Rounds per condition.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(
        run_experiment(
            rounds=args.rounds,
            output_path=args.output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
