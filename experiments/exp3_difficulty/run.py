"""Difficulty comparison experiment — fixed easy / fixed hard / dynamic."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.scorer import Scorer
from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.debate_engine import DebateEngine


def _format_metric(value: object, *, available: bool) -> str:
    if not available or value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _format_metric_short(value: object, *, available: bool) -> str:
    if not available or value is None:
        return "N/A"
    return f"{float(value):.3f}"


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


def _result_gold_label(result: dict[str, Any]) -> str | None:
    scenario = result.get("scenario")
    if scenario is None:
        return None
    gold_label = getattr(scenario, "gold_label", None)
    if gold_label is not None and hasattr(gold_label, "value"):
        return str(gold_label.value)
    if gold_label is not None:
        return str(gold_label)
    ground_truth = getattr(scenario, "ground_truth", None)
    if isinstance(ground_truth, dict) and ground_truth.get("label") is not None:
        return str(ground_truth["label"])
    return None


def _round_for_scoring(result: dict[str, Any]) -> dict[str, Any]:
    audit = result.get("audit_verdict", {}) if isinstance(result.get("audit_verdict"), dict) else {}
    verifier_verdict = audit.get("verifier_verdict", {}) if isinstance(audit.get("verifier_verdict"), dict) else {}
    return {
        "round_id": int(result.get("round_number", 0)),
        "gold_label": _result_gold_label(result),
        "verdict_label": result.get("verdict_label") or audit.get("verdict_label"),
        "verifier_confidence": result.get("verifier_confidence", audit.get("verifier_confidence", 0.0)),
        "predicted_probabilities": verifier_verdict.get("probabilities"),
        "countermodel_witness": audit.get("countermodel_witness"),
        "deception_succeeded": result.get("deception_succeeded", result.get("deception_success")),
    }


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
    scorer = Scorer()

    for condition in conditions:
        results = await _run_condition(base_config, condition, rounds)
        for r in results:
            tracker.log_round(r["round_number"], {"condition": condition, **r})

        score = scorer.score_game(
            {
                "game_id": f"appendix_exp3_{condition}",
                "rounds": [_round_for_scoring(result) for result in results],
            }
        )
        scored_rounds = int(score.summary.get("scored_rounds", 0))
        verdict_metrics_available = scored_rounds > 0
        wins_a = sum(r["winner"] == "agent_a" for r in results)
        causal_scores = [r["audit_verdict"]["causal_validity_score"] for r in results]
        difficulties = [r["difficulty"] for r in results]
        payload["conditions"][condition] = {
            "rounds": rounds,
            "appendix_only": True,
            "public_schema_only": True,
            "scored_rounds": scored_rounds,
            "verdict_metrics_available": verdict_metrics_available,
            "verdict_metrics_unavailable_reason": (
                None
                if verdict_metrics_available
                else "Showcase appendix rounds do not provide frozen gold verdict labels for verdict-centric scoring."
            ),
            "verdict_metrics": dict(score.summary.get("core_metrics", {})) if verdict_metrics_available else {},
            "appendix_metrics": {
                "agent_a_win_rate": wins_a / rounds,
                "causal_validity_mean": sum(causal_scores) / len(causal_scores),
                "difficulty_mean": sum(difficulties) / len(difficulties),
                "difficulty_std": (
                (sum((d - sum(difficulties) / len(difficulties)) ** 2 for d in difficulties) / len(difficulties))
                ** 0.5
                ),
            },
            "results": [
                {
                    "round": r["round_number"],
                    "winner": r["winner"],
                    "deception_success": r["deception_success"],
                    "difficulty": r["difficulty"],
                    "causal_validity": r["audit_verdict"]["causal_validity_score"],
                    "gold_label": _result_gold_label(r),
                    "predicted_label": r.get("verdict_label") or r["audit_verdict"].get("verdict_label"),
                }
                for r in results
            ],
        }
        tracker.log_metrics(
            {
                f"{condition}_agent_a_win_rate": payload["conditions"][condition]["appendix_metrics"]["agent_a_win_rate"],
                f"{condition}_causal_validity_mean": payload["conditions"][condition]["appendix_metrics"]["causal_validity_mean"],
                f"{condition}_difficulty_mean": payload["conditions"][condition]["appendix_metrics"]["difficulty_mean"],
                **(
                    {
                        f"{condition}_verdict_accuracy": payload["conditions"][condition]["verdict_metrics"].get("verdict_accuracy", 0.0),
                    }
                    if payload["conditions"][condition]["verdict_metrics_available"]
                    else {}
                ),
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
            "condition", "rounds", "scored_rounds", "verdict_metrics_available", "verdict_accuracy", "agent_a_win_rate",
            "causal_validity_mean", "difficulty_mean", "difficulty_std",
        ])
        for label, cond in payload.get("conditions", {}).items():
            writer.writerow([
                label,
                cond["rounds"],
                cond.get("scored_rounds", 0),
                cond.get("verdict_metrics_available", False),
                _format_metric(
                    cond["verdict_metrics"].get("verdict_accuracy"),
                    available=bool(cond.get("verdict_metrics_available", False)),
                ),
                f"{cond['appendix_metrics']['agent_a_win_rate']:.4f}",
                f"{cond['appendix_metrics']['causal_validity_mean']:.4f}",
                f"{cond['appendix_metrics']['difficulty_mean']:.4f}",
                f"{cond['appendix_metrics']['difficulty_std']:.4f}",
            ])

    lines = [
        "# Experiment 3 — Difficulty Comparison (Appendix)",
        "",
        "| Condition | Rounds | Verdict Acc. | A Win Rate | Causal Validity | Diff Mean | Diff Std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, cond in payload.get("conditions", {}).items():
        lines.append(
            f"| {label} | {cond['rounds']} | "
            f"{_format_metric_short(cond['verdict_metrics'].get('verdict_accuracy'), available=bool(cond.get('verdict_metrics_available', False)))} | "
            f"{cond['appendix_metrics']['agent_a_win_rate']:.3f} | "
            f"{cond['appendix_metrics']['causal_validity_mean']:.3f} | "
            f"{cond['appendix_metrics']['difficulty_mean']:.3f} | "
            f"{cond['appendix_metrics']['difficulty_std']:.3f} |"
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
