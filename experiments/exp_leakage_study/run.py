"""Phase 4 leakage study comparing clean and oracle-leaking partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    DEFAULT_SEEDS,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    compare_system_predictions,
    evaluate_system_on_samples,
    manifest_metadata,
    write_artifacts,
)
from evaluation.reporting import compare_prediction_groups

SYSTEMS: tuple[str, str] = ("countermodel_grounded", "oracle_leaking_partition")
DEFAULT_SAMPLES_PER_FAMILY = 30
LEAKAGE_ALPHA = 0.05


def _normalize_seeds(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    if not seeds:
        return list(DEFAULT_SEEDS)
    return [int(seed) for seed in seeds]


def _inflation_summary(payload: dict[str, Any], split_name: str) -> dict[str, float]:
    clean = payload["aggregated_metrics"]["countermodel_grounded"][split_name]["verdict_accuracy"]["mean"]
    leaking = payload["aggregated_metrics"]["oracle_leaking_partition"][split_name]["verdict_accuracy"]["mean"]
    return {
        "clean_accuracy_mean": float(clean),
        "leaking_accuracy_mean": float(leaking),
        "inflation_delta": float(leaking - clean),
    }


def _aligned_prediction_groups(
    raw_predictions: list[dict[str, Any]],
    *,
    split_name: str,
) -> tuple[list[str], dict[str, list[str]]]:
    truth: list[str] | None = None
    predictions: dict[str, list[str]] = {}

    for system_name in SYSTEMS:
        records = sorted(
            (
                record
                for record in raw_predictions
                if record["system_name"] == system_name and record["split"] == split_name
            ),
            key=lambda record: (int(record["seed"]), str(record["instance_id"])),
        )
        predictions[system_name] = [str(record["predicted_label"]) for record in records]
        system_truth = [str(record["gold_label"]) for record in records]
        if truth is None:
            truth = system_truth
        elif system_truth != truth:
            raise ValueError(
                f"Leakage significance requires identical gold labels across systems for split {split_name!r}."
            )

    if truth is None:
        raise ValueError(f"No predictions found for split {split_name!r}.")
    return truth, predictions


def _mcnemar_significance(raw_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split_name in OOD_SPLITS:
        truth, predictions = _aligned_prediction_groups(raw_predictions, split_name=split_name)
        reports[split_name] = compare_prediction_groups(
            truth,
            predictions,
            baseline=SYSTEMS[0],
            method="mcnemar",
            metric_name="verdict_accuracy",
        ).to_dict()
    return reports


def _comparison_support(
    comparison: dict[str, Any] | None,
    *,
    alpha: float = LEAKAGE_ALPHA,
) -> bool:
    if comparison is None:
        return False
    adjusted_p_value = comparison.get("adjusted_p_value")
    p_value = adjusted_p_value if adjusted_p_value is not None else comparison.get("p_value")
    return p_value is not None and float(p_value) < float(alpha)


def _leakage_conclusion(
    *,
    inflation: dict[str, dict[str, float]],
    significance: dict[str, Any],
    mcnemar_significance: dict[str, Any],
    samples_per_family: int,
) -> dict[str, Any]:
    split_details: dict[str, dict[str, Any]] = {}
    supported_splits: list[str] = []
    unsupported_splits: list[str] = []

    for split_name in OOD_SPLITS:
        bootstrap_report = significance.get(split_name)
        mcnemar_report = mcnemar_significance.get(split_name)
        bootstrap_comparison = (
            bootstrap_report["comparisons"][0]
            if bootstrap_report and bootstrap_report.get("comparisons")
            else None
        )
        mcnemar_comparison = (
            mcnemar_report["comparisons"][0]
            if mcnemar_report and mcnemar_report.get("comparisons")
            else None
        )
        delta = float(inflation[split_name]["inflation_delta"])
        positive_inflation = delta > 0.0
        bootstrap_significant = _comparison_support(bootstrap_comparison)
        mcnemar_significant = _comparison_support(mcnemar_comparison)
        supported = positive_inflation and bootstrap_significant and mcnemar_significant
        detail = {
            "inflation_delta": delta,
            "positive_inflation": positive_inflation,
            "bootstrap_significant": bootstrap_significant,
            "mcnemar_significant": mcnemar_significant,
            "supported": supported,
        }
        split_details[split_name] = detail
        if supported:
            supported_splits.append(split_name)
        else:
            unsupported_splits.append(split_name)

    robust_support = len(supported_splits) == len(OOD_SPLITS)
    supported = bool(supported_splits)
    if robust_support:
        summary_statement = (
            "This run supports the leakage warning on every evaluated split: oracle access "
            "produces statistically supported score inflation."
        )
    elif supported:
        summary_statement = (
            "This run partially supports the leakage warning: statistically supported "
            f"inflation appears on {', '.join(supported_splits)}, but not on "
            f"{', '.join(unsupported_splits)}."
        )
    else:
        summary_statement = (
            "This run does not yet support the leakage warning: no evaluated split shows "
            "statistically supported positive inflation under both paired tests."
        )

    recommendation = None
    if int(samples_per_family) < DEFAULT_SAMPLES_PER_FAMILY:
        recommendation = (
            f"Increase samples_per_family to at least {DEFAULT_SAMPLES_PER_FAMILY} before citing "
            "this experiment as leakage evidence."
        )

    return {
        "alpha": float(LEAKAGE_ALPHA),
        "supported": supported,
        "robust_support": robust_support,
        "supported_splits": supported_splits,
        "unsupported_splits": unsupported_splits,
        "split_details": split_details,
        "summary_statement": summary_statement,
        "recommendation": recommendation,
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Leakage Study",
        "",
        "This experiment contrasts the clean public partition against a deliberately oracle-leaking partition.",
        "",
        "| Split | Clean Accuracy | Oracle-Leaking Accuracy | Inflation | Bootstrap p | McNemar p | Supports Warning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split_name in OOD_SPLITS:
        delta = payload["inflation"][split_name]
        bootstrap_row = payload["significance"][split_name]["comparisons"][0]
        mcnemar_row = payload["mcnemar_significance"][split_name]["comparisons"][0]
        support = payload["conclusion"]["split_details"][split_name]["supported"]
        bootstrap_p = bootstrap_row.get("adjusted_p_value")
        if bootstrap_p is None:
            bootstrap_p = bootstrap_row["p_value"]
        mcnemar_p = mcnemar_row.get("adjusted_p_value")
        if mcnemar_p is None:
            mcnemar_p = mcnemar_row["p_value"]
        lines.append(
            f"| {split_name} | {delta['clean_accuracy_mean']:.4f} | "
            f"{delta['leaking_accuracy_mean']:.4f} | {delta['inflation_delta']:.4f} | "
            f"{float(bootstrap_p):.4f} | {float(mcnemar_p):.4f} | "
            f"{'yes' if support else 'no'} |"
        )
    lines.append("")
    lines.append("## Leakage Warning")
    lines.append("")
    lines.append(payload["conclusion"]["summary_statement"])
    if payload["conclusion"]["recommendation"]:
        lines.append("")
        lines.append(payload["conclusion"]["recommendation"])
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
    difficulty: float = 0.55,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = _normalize_seeds(seeds)
    raw_predictions: list[dict[str, Any]] = []
    per_seed_results: dict[int, dict[str, Any]] = {}
    manifests: dict[int, dict[str, Any]] = {}

    for seed in resolved_seeds:
        run = build_seed_benchmark_run(
            seed=seed,
            difficulty=difficulty,
            samples_per_family=samples_per_family,
        )
        manifests[seed] = manifest_metadata(run)
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_payload: dict[str, Any] = {}
        for system_name in SYSTEMS:
            system_payload: dict[str, Any] = {}
            for split_name in OOD_SPLITS:
                evaluated = evaluate_system_on_samples(
                    run.split_samples[split_name],
                    seed=seed,
                    split_name=split_name,
                    system_name=system_name,
                    ood_reasons=ood_reasons,
                )
                raw_predictions.extend(evaluated["predictions"])
                system_payload[split_name] = evaluated
            seed_payload[system_name] = system_payload
        per_seed_results[seed] = seed_payload

    aggregated_metrics: dict[str, dict[str, Any]] = {}
    for system_name in SYSTEMS:
        aggregated_metrics[system_name] = {}
        for split_name in OOD_SPLITS:
            aggregated_metrics[system_name][split_name] = aggregate_seed_metrics(
                {
                    seed: per_seed_results[seed][system_name]
                    for seed in resolved_seeds
                },
                split_name=split_name,
            )

    significance = {
        split_name: compare_system_predictions(
            {
                system_name: [
                    record
                    for record in raw_predictions
                    if record["system_name"] == system_name and record["split"] == split_name
                ]
                for system_name in SYSTEMS
            },
            baseline=SYSTEMS[0],
        )
        for split_name in OOD_SPLITS
    }
    mcnemar_significance = _mcnemar_significance(raw_predictions)
    inflation = {
        split_name: _inflation_summary(
            {
                "aggregated_metrics": aggregated_metrics,
            },
            split_name,
        )
        for split_name in OOD_SPLITS
    }
    conclusion = _leakage_conclusion(
        inflation=inflation,
        significance=significance,
        mcnemar_significance=mcnemar_significance,
        samples_per_family=int(samples_per_family),
    )

    payload = {
        "experiment_id": "exp_leakage_study",
        "config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
        },
        "systems": list(SYSTEMS),
        "seeds": resolved_seeds,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "inflation": inflation,
        "significance": significance,
        "mcnemar_significance": mcnemar_significance,
        "conclusion": conclusion,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/exp_leakage_study.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 leakage study.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
        help="Samples generated per benchmark family.",
    )
    parser.add_argument("--difficulty", type=float, default=0.55, help="Benchmark generation difficulty.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        seeds=args.seeds,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        output_path=args.output,
    )
    print(json.dumps(payload["inflation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
