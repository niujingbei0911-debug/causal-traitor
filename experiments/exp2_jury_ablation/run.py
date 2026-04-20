"""Jury ablation experiment — compare 0 / 3 / 5 jurors."""

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
        "countermodel_witness": audit.get("countermodel_witness"),
        "jury_consensus": jury.get("agreement_rate", 0.0),
        "jury_verdict": jury.get("final_winner"),
        "deception_succeeded": result.get("deception_succeeded", result.get("deception_success")),
    }


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
    scorer = Scorer()

    for jury_count in conditions:
        label = f"jury_{jury_count}"
        results = await _run_condition(base_config, jury_count, rounds, level)
        for r in results:
            tracker.log_round(r["round_number"], {"condition": label, **r})

        score = scorer.score_game(
            {
                "game_id": f"appendix_exp2_{label}",
                "rounds": [_round_for_scoring(result) for result in results],
            }
        )
        scored_rounds = int(score.summary.get("scored_rounds", 0))
        verdict_metrics_available = scored_rounds > 0
        wins_a = sum(r["winner"] == "agent_a" for r in results)
        causal_scores = [r["audit_verdict"]["causal_validity_score"] for r in results]
        payload["conditions"][label] = {
            "jury_count": jury_count,
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
                "jury_condition": jury_count,
            },
            "results": [
                {
                    "round": r["round_number"],
                    "winner": r["winner"],
                    "deception_success": r["deception_success"],
                    "causal_validity": r["audit_verdict"]["causal_validity_score"],
                    "gold_label": _result_gold_label(r),
                    "predicted_label": r.get("verdict_label") or r["audit_verdict"].get("verdict_label"),
                }
                for r in results
            ],
        }
        tracker.log_metrics(
            {
                f"{label}_agent_a_win_rate": payload["conditions"][label]["appendix_metrics"]["agent_a_win_rate"],
                f"{label}_causal_validity_mean": payload["conditions"][label]["appendix_metrics"]["causal_validity_mean"],
                **(
                    {
                        f"{label}_verdict_accuracy": payload["conditions"][label]["verdict_metrics"].get("verdict_accuracy", 0.0),
                    }
                    if payload["conditions"][label]["verdict_metrics_available"]
                    else {}
                ),
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
        writer.writerow(["jury_count", "rounds", "scored_rounds", "verdict_metrics_available", "verdict_accuracy", "agent_a_win_rate", "causal_validity_mean"])
        for cond in payload.get("conditions", {}).values():
            writer.writerow([
                cond["jury_count"],
                cond["rounds"],
                cond.get("scored_rounds", 0),
                cond.get("verdict_metrics_available", False),
                _format_metric(
                    cond["verdict_metrics"].get("verdict_accuracy"),
                    available=bool(cond.get("verdict_metrics_available", False)),
                ),
                f"{cond['appendix_metrics']['agent_a_win_rate']:.4f}",
                f"{cond['appendix_metrics']['causal_validity_mean']:.4f}",
            ])

    lines = [
        "# Experiment 2 — Jury Ablation (Appendix)",
        "",
        "| Jury Count | Rounds | Verdict Acc. | Agent A Win Rate | Causal Validity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cond in payload.get("conditions", {}).values():
        lines.append(
            f"| {cond['jury_count']} | {cond['rounds']} | "
            f"{_format_metric_short(cond['verdict_metrics'].get('verdict_accuracy'), available=bool(cond.get('verdict_metrics_available', False)))} | "
            f"{cond['appendix_metrics']['agent_a_win_rate']:.3f} | "
            f"{cond['appendix_metrics']['causal_validity_mean']:.3f} |"
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
