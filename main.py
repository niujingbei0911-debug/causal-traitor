"""CLI entrypoint for running the causal debate game."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from evaluation.reporting import summarize_metric
from evaluation.scorer import Scorer
from evaluation.tracker import ExperimentConfig, ExperimentTracker
from game.config import ConfigLoader
from game.debate_engine import DebateEngine


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _result_gold_label(result: dict[str, Any]) -> str | None:
    scenario = result.get("scenario")
    if scenario is not None:
        gold_label = getattr(scenario, "gold_label", None)
        if isinstance(gold_label, Enum):
            return gold_label.value
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


def _build_summary(results: list[dict[str, Any]], engine: DebateEngine) -> dict[str, Any]:
    scorer = Scorer()
    score = scorer.score_game(
        {
            "game_id": "main_run",
            "rounds": [_round_for_scoring(result) for result in results],
        }
    )
    scored_rounds = int(score.summary.get("scored_rounds", 0))
    verdict_metrics_available = scored_rounds > 0
    correctness = [
        1.0 if round_score.verdict_correct else 0.0
        for round_score in score.round_scores
        if round_score.gold_label is not None and round_score.predicted_label is not None
    ]
    verdict_accuracy_ci = (
        summarize_metric(
            "verdict_accuracy",
            correctness,
            n_resamples=2000,
            random_state=0,
        ).to_dict()
        if correctness
        else None
    )
    verdict_metrics = dict(score.summary.get("core_metrics", {})) if verdict_metrics_available else {}
    core_metric_summaries: dict[str, Any] = {}
    return {
        "n_rounds": len(results),
        "scored_rounds": scored_rounds,
        "primary_metric": (
            score.summary.get("primary_metric", "unsafe_acceptance_rate")
            if verdict_metrics_available
            else None
        ),
        "verdict_metrics_available": verdict_metrics_available,
        "verdict_metrics_unavailable_reason": (
            None
            if verdict_metrics_available
            else "No frozen gold verdict labels were available for this showcase run."
        ),
        "verdict_metrics": verdict_metrics,
        "verdict_metric_summaries": core_metric_summaries,
        "verdict_metric_summaries_unavailable_reason": (
            "Single-run verdict summaries are not bootstrapped here because the available values are already aggregated game-level metrics."
        ),
        "verdict_accuracy_ci": verdict_accuracy_ci,
        "gold_label_distribution": (
            dict(score.summary.get("gold_label_distribution", {}))
            if verdict_metrics_available
            else {}
        ),
        "predicted_label_distribution": (
            dict(score.summary.get("predicted_label_distribution", {}))
            if verdict_metrics_available
            else {}
        ),
        "appendix_metrics": dict(score.summary.get("appendix_metrics", {})),
        "protocol_metrics": {
            "agent_a_wins": sum(result.get("winner") == "agent_a" for result in results),
            "agent_b_wins": sum(result.get("winner") == "agent_b" for result in results),
            "final_difficulty": engine.difficulty_controller.get_difficulty(),
            "arms_race_index": engine.evolution_tracker.get_arms_race_index(),
        },
    }


async def run_game(
    *,
    config_path: str | None = None,
    rounds: int | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    loader = ConfigLoader(config_path=config_path)
    config = loader.load()
    engine = DebateEngine(config)
    await engine.initialize()
    num_rounds = rounds or config.get("game", {}).get("max_rounds", 5)
    tracker = ExperimentTracker(
        ExperimentConfig(
            experiment_id="main_run",
            name="Main Game Run",
            params={
                "rounds": num_rounds,
                "config_path": config_path,
            },
        ),
        log_dir=config.get("logging", {}).get("save_dir", "logs"),
        use_wandb=bool(config.get("logging", {}).get("use_wandb", False)),
    )
    tracker.init()
    results = await engine.run_game(num_rounds=num_rounds)
    for result in results:
        tracker.log_round(result["round_number"], result)
    summary = _build_summary(results, engine)
    tracker.log_metrics(
        {
            **summary.get("verdict_metrics", {}),
            "protocol_final_difficulty": summary["protocol_metrics"]["final_difficulty"],
            "protocol_arms_race_index": summary["protocol_metrics"]["arms_race_index"],
        },
        step=len(results),
    )
    payload = {
        "summary": summary,
        "results": _json_ready(results),
    }
    tracker.log_artifact("run_payload", payload, artifact_type="json")
    payload["tracking"] = tracker.finish()

    target = Path(output_path) if output_path else _default_output_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _default_output_path(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("data", {}).get("output_dir", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"game_run_{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Causal Traitor debate engine.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to YAML config file.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds to run.")
    parser.add_argument("--output", dest="output_path", default=None, help="Path to JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(
        run_game(
            config_path=args.config_path,
            rounds=args.rounds,
            output_path=args.output_path,
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
