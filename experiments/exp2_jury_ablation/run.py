"""Jury ablation experiment — compare 0 / 3 / 5 jurors."""

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
    jury_count: int,
    rounds: int,
    level: int,
) -> list[dict[str, Any]]:
    """Run *rounds* debates with a specific jury size."""
    config = deepcopy(base_config)
    # Adjust jury model list to desired count
    jury_cfg = config.setdefault("models", {}).setdefault("jury", {})
    base_models = jury_cfg.get("models", ["qwen2.5-7b-instruct"])
    if jury_count == 0:
        jury_cfg["models"] = []
    else:
        # Cycle through available models to fill the requested count
        jury_cfg["models"] = [base_models[i % len(base_models)] for i in range(jury_count)]

    engine = DebateEngine(config)
    await engine.initialize()
    return await engine.run_game(
        num_rounds=rounds,
        level_schedule=[level] * rounds,
        use_evolution=False,
        update_difficulty=False,
    )


async def run_experiment(
    rounds: int = 30,
    level: int = 2,
    output_path: str | None = None,
) -> dict[str, Any]:
    base_config = ConfigLoader().load()
    tracker = ExperimentTracker(
        ExperimentConfig(
            experiment_id="exp2_jury_ablation",
            name="Jury Ablation (0 / 3 / 5 jurors)",
            params={"rounds": rounds, "level": level},
        ),
        log_dir=base_config.get("logging", {}).get("save_dir", "logs"),
        use_wandb=bool(base_config.get("logging", {}).get("use_wandb", False)),
    )
    tracker.init()

    conditions = [0, 3, 5]
    payload: dict[str, Any] = {"conditions": {}}

    for jury_count in conditions:
        label = f"jury_{jury_count}"
        results = await _run_condition(base_config, jury_count, rounds, level)
        for r in results:
            tracker.log_round(r["round_number"], {"condition": label, **r})

        wins_a = sum(r["winner"] == "agent_a" for r in results)
        causal_scores = [r["audit_verdict"]["causal_validity_score"] for r in results]
        payload["conditions"][label] = {
            "jury_count": jury_count,
            "rounds": rounds,
            "agent_a_win_rate": wins_a / rounds,
            "causal_validity_mean": sum(causal_scores) / len(causal_scores),
            "results": [
                {
                    "round": r["round_number"],
                    "winner": r["winner"],
                    "deception_success": r["deception_success"],
                    "causal_validity": r["audit_verdict"]["causal_validity_score"],
                }
                for r in results
            ],
        }
        tracker.log_metrics(
            {
                f"{label}_agent_a_win_rate": payload["conditions"][label]["agent_a_win_rate"],
                f"{label}_causal_validity_mean": payload["conditions"][label]["causal_validity_mean"],
            },
            step=jury_count,
        )

    tracker.log_artifact("exp2_summary", payload, artifact_type="json")
    payload["tracking"] = tracker.finish()
    target = Path(output_path or "outputs/exp2_jury_ablation.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_exp2_sidecars(target, payload)
    return payload


def _write_exp2_sidecars(json_path: Path, payload: dict[str, Any]) -> None:
    csv_path = json_path.with_suffix(".csv")
    md_path = json_path.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["jury_count", "rounds", "agent_a_win_rate", "causal_validity_mean"])
        for cond in payload.get("conditions", {}).values():
            writer.writerow([
                cond["jury_count"],
                cond["rounds"],
                f"{cond['agent_a_win_rate']:.4f}",
                f"{cond['causal_validity_mean']:.4f}",
            ])

    lines = [
        "# Experiment 2 — Jury Ablation",
        "",
        "| Jury Count | Rounds | Agent A Win Rate | Causal Validity |",
        "| --- | --- | --- | --- |",
    ]
    for cond in payload.get("conditions", {}).values():
        lines.append(
            f"| {cond['jury_count']} | {cond['rounds']} | "
            f"{cond['agent_a_win_rate']:.3f} | "
            f"{cond['causal_validity_mean']:.3f} |"
        )
    tracking = payload.get("tracking", {})
    if tracking.get("run_dir"):
        lines.append("")
        lines.append(f"Tracking artifacts: `{tracking['run_dir']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the jury ablation experiment.")
    parser.add_argument("--rounds", type=int, default=30, help="Rounds per jury condition.")
    parser.add_argument("--level", type=int, default=2, help="Pearl level to use.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(
        run_experiment(
            rounds=args.rounds,
            level=args.level,
            output_path=args.output,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
