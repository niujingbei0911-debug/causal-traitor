"""Compare fixed independent rounds against evolution-enabled rounds."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from evaluation.scorer import Scorer
from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.debate_engine import DebateEngine


def _format_metric_short(value: object, *, available: bool) -> str:
    if not available or value is None:
        return "N/A"
    return f"{float(value):.3f}"


async def _run_without_evolution(engine: DebateEngine, rounds: int, level: int) -> list[dict[str, Any]]:
    return await engine.run_game(
        num_rounds=rounds,
        level_schedule=[level] * rounds,
        use_evolution=False,
        update_difficulty=False,
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
    scorer = Scorer()

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
        tracker.log_round(rounds + result["round_number"], {"condition": "with_evolution", **result})

    no_evolution_score = scorer.score_game(
        {
            "game_id": "appendix_exp4_without_evolution",
            "rounds": [_round_for_scoring(result) for result in independent_results],
        }
    )
    without_scored_rounds = int(no_evolution_score.summary.get("scored_rounds", 0))
    without_verdict_metrics_available = without_scored_rounds > 0
    evolution_score = scorer.score_game(
        {
            "game_id": "appendix_exp4_with_evolution",
            "rounds": [_round_for_scoring(result) for result in evolution_results],
        }
    )
    with_scored_rounds = int(evolution_score.summary.get("scored_rounds", 0))
    with_verdict_metrics_available = with_scored_rounds > 0
    payload = {
        "without_evolution": {
            "rounds": rounds,
            "appendix_only": True,
            "public_schema_only": True,
            "scored_rounds": without_scored_rounds,
            "verdict_metrics_available": without_verdict_metrics_available,
            "verdict_metrics_unavailable_reason": (
                None
                if without_verdict_metrics_available
                else "Showcase appendix rounds do not provide frozen gold verdict labels for verdict-centric scoring."
            ),
            "verdict_metrics": (
                dict(no_evolution_score.summary.get("core_metrics", {}))
                if without_verdict_metrics_available
                else {}
            ),
            "appendix_metrics": {
                "agent_a_win_rate": sum(r["winner"] == "agent_a" for r in independent_results) / rounds,
            },
            "results": [
                {
                    "round_number": result.get("round_number"),
                    "winner": result["winner"],
                    "deception_success": result["deception_success"],
                    "gold_label": _result_gold_label(result),
                    "predicted_label": result.get("verdict_label") or result["audit_verdict"].get("verdict_label"),
                }
                for result in independent_results
            ],
        },
        "with_evolution": {
            "rounds": rounds,
            "appendix_only": True,
            "public_schema_only": True,
            "scored_rounds": with_scored_rounds,
            "verdict_metrics_available": with_verdict_metrics_available,
            "verdict_metrics_unavailable_reason": (
                None
                if with_verdict_metrics_available
                else "Showcase appendix rounds do not provide frozen gold verdict labels for verdict-centric scoring."
            ),
            "verdict_metrics": (
                dict(evolution_score.summary.get("core_metrics", {}))
                if with_verdict_metrics_available
                else {}
            ),
            "appendix_metrics": {
                "agent_a_win_rate": sum(r["winner"] == "agent_a" for r in evolution_results) / rounds,
                "arms_race_index": evolution_engine.evolution_tracker.get_arms_race_index(),
                "converged": evolution_engine.evolution_tracker.detect_convergence(),
            },
            "results": [
                {
                    "winner": result["winner"],
                    "deception_success": result["deception_success"],
                    "snapshot": result["evolution_snapshot"],
                    "gold_label": _result_gold_label(result),
                    "predicted_label": result.get("verdict_label") or result["audit_verdict"].get("verdict_label"),
                }
                for result in evolution_results
            ],
        },
    }
    tracker.log_metrics(
        {
            "without_evolution_agent_a_win_rate": payload["without_evolution"]["appendix_metrics"]["agent_a_win_rate"],
            "with_evolution_agent_a_win_rate": payload["with_evolution"]["appendix_metrics"]["agent_a_win_rate"],
            "with_evolution_arms_race_index": payload["with_evolution"]["appendix_metrics"]["arms_race_index"],
            **(
                {
                    "without_evolution_verdict_accuracy": payload["without_evolution"]["verdict_metrics"].get("verdict_accuracy", 0.0),
                }
                if payload["without_evolution"]["verdict_metrics_available"]
                else {}
            ),
            **(
                {
                    "with_evolution_verdict_accuracy": payload["with_evolution"]["verdict_metrics"].get("verdict_accuracy", 0.0),
                }
                if payload["with_evolution"]["verdict_metrics_available"]
                else {}
            ),
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
        writer.writerow(["condition", "round", "winner", "deception_success", "predicted_label", "gold_label"])
        for result in payload.get("without_evolution", {}).get("results", []):
            writer.writerow(
                [
                    "without_evolution",
                    result.get("round_number", ""),
                    result.get("winner", ""),
                    int(bool(result.get("deception_success", False))),
                    result.get("predicted_label", ""),
                    result.get("gold_label", ""),
                ]
            )
        for index, result in enumerate(payload.get("with_evolution", {}).get("results", []), start=1):
            writer.writerow(
                [
                    "with_evolution",
                    index,
                    result.get("winner", ""),
                    int(bool(result.get("deception_success", False))),
                    result.get("predicted_label", ""),
                    result.get("gold_label", ""),
                ]
            )
    without = payload.get("without_evolution", {})
    with_evo = payload.get("with_evolution", {})
    lines = [
        "# Experiment 4 — Evolution vs No Evolution (Appendix)",
        "",
        f"- Rounds per condition: {without.get('rounds', 0)}",
        f"- Without evolution — Verdict accuracy: {_format_metric_short(without.get('verdict_metrics', {}).get('verdict_accuracy'), available=bool(without.get('verdict_metrics_available', False)))}",
        f"- With evolution — Verdict accuracy: {_format_metric_short(with_evo.get('verdict_metrics', {}).get('verdict_accuracy'), available=bool(with_evo.get('verdict_metrics_available', False)))}",
        f"- Without evolution — Agent A win rate: {without.get('appendix_metrics', {}).get('agent_a_win_rate', 0.0):.3f}",
        f"- With evolution — Agent A win rate: {with_evo.get('appendix_metrics', {}).get('agent_a_win_rate', 0.0):.3f}",
        f"- Arms race index (with evolution): {with_evo.get('appendix_metrics', {}).get('arms_race_index', 0.0):.3f}",
        f"- Strategy converged (with evolution): {with_evo.get('appendix_metrics', {}).get('converged', False)}",
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
