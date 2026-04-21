"""Run the level-by-level causal benchmark."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.reporting import summarize_metric
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
    jury = result.get("jury_verdict", {}) if isinstance(result.get("jury_verdict"), dict) else {}
    return {
        "round_id": int(result.get("round_number", 0)),
        "gold_label": _result_gold_label(result),
        "verdict_label": result.get("verdict_label") or audit.get("verdict_label"),
        "verifier_confidence": result.get("verifier_confidence", audit.get("verifier_confidence", 0.0)),
        "predicted_probabilities": verifier_verdict.get("probabilities"),
        "verifier_verdict": dict(verifier_verdict),
        "countermodel_witness": audit.get("countermodel_witness"),
        "jury_consensus": jury.get("agreement_rate", 0.0),
        "jury_verdict": jury.get("final_winner"),
        "deception_succeeded": result.get("deception_succeeded", result.get("deception_success")),
    }


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
    scorer = Scorer()
    for level in config.get("game", {}).get("causal_levels", [1, 2, 3]):
        raw_rounds: list[dict[str, Any]] = []
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
            raw_rounds.append(result)
        scored_rounds = [_round_for_scoring(result) for result in raw_rounds]
        game_score = scorer.score_game(
            {
                "game_id": f"exp1_L{level}",
                "rounds": scored_rounds,
            }
        )
        scored_round_count = int(game_score.summary.get("scored_rounds", 0))
        verdict_metrics_available = scored_round_count > 0
        verdict_correctness = [
            1.0 if round_score.verdict_correct else 0.0
            for round_score in game_score.round_scores
            if round_score.gold_label is not None and round_score.predicted_label is not None
        ]
        verdict_accuracy_ci = (
            summarize_metric(
                "verdict_accuracy",
                verdict_correctness,
                n_resamples=2000,
                random_state=level,
            ).to_dict()
            if verdict_correctness
            else None
        )
        appendix_metrics = dict(game_score.summary.get("appendix_metrics", {}))
        appendix_metrics["deception_success_rate"] = (
            sum(1 for result in raw_rounds if result.get("deception_success")) / rounds_per_level
        )
        appendix_metrics["causal_validity_mean"] = (
            sum(float(result["audit_verdict"]["causal_validity_score"]) for result in raw_rounds)
            / rounds_per_level
        )
        summary["levels"][f"L{level}"] = {
            "rounds": rounds_per_level,
            "primary_metric": game_score.summary.get("primary_metric", "unsafe_acceptance_rate"),
            "scored_rounds": scored_round_count,
            "verdict_metrics_available": verdict_metrics_available,
            "verdict_metrics_unavailable_reason": (
                None
                if verdict_metrics_available
                else "Showcase debate rounds do not provide frozen gold verdict labels for verdict-centric scoring."
            ),
            "verdict_metrics": dict(game_score.summary.get("core_metrics", {})) if verdict_metrics_available else {},
            "verdict_accuracy_ci": verdict_accuracy_ci,
            "gold_label_distribution": (
                dict(game_score.summary.get("gold_label_distribution", {}))
                if verdict_metrics_available
                else {}
            ),
            "predicted_label_distribution": (
                dict(game_score.summary.get("predicted_label_distribution", {}))
                if verdict_metrics_available
                else {}
            ),
            "appendix_metrics": appendix_metrics,
            "results": [
                {
                    "round_id": round_payload["round_id"],
                    "gold_label": round_payload["gold_label"],
                    "predicted_label": round_payload["verdict_label"],
                    "verifier_confidence": round_payload["verifier_confidence"],
                    "jury_verdict": round_payload["jury_verdict"],
                    "jury_consensus": round_payload["jury_consensus"],
                    "deception_succeeded": round_payload["deception_succeeded"],
                }
                for round_payload in scored_rounds
            ],
        }
        tracker.log_metrics(
            {
                **{
                    f"L{level}_{metric_name}": metric_value
                    for metric_name, metric_value in summary["levels"][f"L{level}"]["verdict_metrics"].items()
                },
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
                "scored_rounds",
                "verdict_metrics_available",
                "verdict_accuracy",
                "macro_f1",
                "unsafe_acceptance_rate",
                "wise_refusal_recall",
                "verdict_accuracy_ci",
            ]
        )
        for level_key, level_summary in summary.get("levels", {}).items():
            verdict_metrics = level_summary.get("verdict_metrics", {})
            accuracy_ci = level_summary.get("verdict_accuracy_ci") or {}
            writer.writerow(
                [
                    level_key,
                    level_summary.get("rounds", 0),
                    level_summary.get("scored_rounds", 0),
                    level_summary.get("verdict_metrics_available", False),
                    _format_metric(
                        verdict_metrics.get("verdict_accuracy"),
                        available=bool(level_summary.get("verdict_metrics_available", False)),
                    ),
                    _format_metric(
                        verdict_metrics.get("macro_f1"),
                        available=bool(level_summary.get("verdict_metrics_available", False)),
                    ),
                    _format_metric(
                        verdict_metrics.get("unsafe_acceptance_rate"),
                        available=bool(level_summary.get("verdict_metrics_available", False)),
                    ),
                    _format_metric(
                        verdict_metrics.get("wise_refusal_recall"),
                        available=bool(level_summary.get("verdict_metrics_available", False)),
                    ),
                    accuracy_ci.get("formatted", ""),
                ]
            )
    lines = ["# Experiment 1 — Pearl Ladder Benchmark", ""]
    lines.append("| Level | Rounds | Verdict Acc. | Macro-F1 | Unsafe Accept | Wise Refusal Recall | 95% CI |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for level_key, level_summary in summary.get("levels", {}).items():
        verdict_metrics = level_summary.get("verdict_metrics", {})
        accuracy_ci = level_summary.get("verdict_accuracy_ci") or {}
        lines.append(
            f"| {level_key} | {level_summary.get('rounds', 0)} | "
            f"{_format_metric_short(verdict_metrics.get('verdict_accuracy'), available=bool(level_summary.get('verdict_metrics_available', False)))} | "
            f"{_format_metric_short(verdict_metrics.get('macro_f1'), available=bool(level_summary.get('verdict_metrics_available', False)))} | "
            f"{_format_metric_short(verdict_metrics.get('unsafe_acceptance_rate'), available=bool(level_summary.get('verdict_metrics_available', False)))} | "
            f"{_format_metric_short(verdict_metrics.get('wise_refusal_recall'), available=bool(level_summary.get('verdict_metrics_available', False)))} | "
            f"{accuracy_ci.get('formatted', '')} |"
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
