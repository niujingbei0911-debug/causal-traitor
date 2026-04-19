"""Phase 4 OOD generalization experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    aggregate_seed_metrics,
    build_seed_metric_significance,
    build_seed_benchmark_run,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    score_prediction_records,
    summarize_protocol_compliance,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
OOD_BUCKETS: tuple[tuple[str, str | None], ...] = (
    ("graph_family_ood", "family_holdout"),
    ("lexical_ood", "lexical_holdout"),
    ("variable_naming_ood", "variable_renaming_holdout"),
    ("mixed_ood", None),
)


def _bucket_sample_count_summary(per_seed_bucket_counts: dict[int, dict[str, int]], bucket_name: str) -> dict[str, Any]:
    per_seed = {
        int(seed): int(bucket_counts.get(bucket_name, 0))
        for seed, bucket_counts in sorted(per_seed_bucket_counts.items())
    }
    counts = list(per_seed.values())
    total = sum(counts)
    return {
        "per_seed": per_seed,
        "total": total,
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": (total / len(counts)) if counts else 0.0,
    }


def _small_bucket_warnings(bucket_sample_counts: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for bucket_name, summary in bucket_sample_counts.items():
        if bucket_name == "test_iid":
            continue
        if int(summary["min"]) < 5:
            warnings.append(
                f"{bucket_name} has a small pure-bucket support: per-seed counts={list(summary['per_seed'].values())}."
            )
    return warnings


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# OOD Generalization",
        "",
        "| Bucket | Sample Count | Verdict Acc. | Macro-F1 | Invalid Accept | Unidentifiable Awareness |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for bucket_name, metrics in payload["aggregated_metrics"].items():
        counts = payload["bucket_sample_counts"][bucket_name]
        lines.append(
            f"| {bucket_name} | {list(counts['per_seed'].values())} | {metrics['verdict_accuracy']['formatted']} | "
            f"{metrics['macro_f1']['formatted']} | {metrics['invalid_claim_acceptance_rate']['formatted']} | "
            f"{metrics['unidentifiable_awareness']['formatted']} |"
        )
    lines.extend(["", "## OOD Gap", ""])
    for bucket_name, gap in payload["ood_gap"].items():
        lines.append(
            f"- {bucket_name}: accuracy gap={gap['verdict_accuracy_gap']:.4f}, "
            f"macro_f1 gap={gap['macro_f1_gap']:.4f}"
        )
    if payload["bucket_warnings"]:
        lines.extend(["", "## Bucket Warnings", ""])
        for warning in payload["bucket_warnings"]:
            lines.append(f"- {warning}")
    for bucket_name in ("graph_family_ood", "lexical_ood", "variable_naming_ood", "mixed_ood"):
        comparison = payload["significance"].get(bucket_name)
        if comparison is None:
            continue
        lines.extend(["", f"## Significance: {bucket_name}", ""])
        for row in comparison["comparisons"]:
            lines.append(
                f"- {row['comparison']}: diff={row['observed_difference']:.4f}, "
                f"p={row['p_value']:.4f}, adjusted={row.get('adjusted_p_value')}"
            )
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    samples_per_family: int = MIN_FORMAL_SAMPLES_PER_FAMILY,
    difficulty: float = 0.55,
    allow_protocol_violations: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = normalize_experiment_seeds(
        seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        minimum_samples_per_family=MIN_FORMAL_SAMPLES_PER_FAMILY,
        observed_samples_per_family=resolved_samples_per_family,
        allow_protocol_violations=allow_protocol_violations,
    )
    raw_predictions: list[dict[str, Any]] = []
    per_seed_results: dict[int, Any] = {}
    per_seed_bucket_results: dict[int, dict[str, Any]] = {}
    per_seed_bucket_counts: dict[int, dict[str, int]] = {}
    manifests: dict[int, dict[str, Any]] = {}

    for seed in resolved_seeds:
        run = build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        manifests[seed] = manifest_metadata(run)
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_payload: dict[str, Any] = {}
        for split_name in ("test_iid", "test_ood"):
            evaluated = evaluate_system_on_samples(
                run.split_samples[split_name],
                seed=seed,
                split_name=split_name,
                system_name=SYSTEM_NAME,
                ood_reasons=ood_reasons,
            )
            raw_predictions.extend(evaluated["predictions"])
            seed_payload[split_name] = evaluated
        per_seed_results[seed] = seed_payload
        bucket_payload: dict[str, Any] = {
            "test_iid": seed_payload["test_iid"],
        }
        bucket_counts: dict[str, int] = {
            "test_iid": len(seed_payload["test_iid"]["predictions"]),
        }
        test_ood_predictions = list(seed_payload["test_ood"]["predictions"])
        for bucket_name, reason_name in OOD_BUCKETS:
            if reason_name is None:
                bucket_predictions = [
                    record
                    for record in test_ood_predictions
                    if len(record.get("ood_reasons", [])) > 1
                ]
            else:
                bucket_predictions = [
                    record
                    for record in test_ood_predictions
                    if record.get("ood_reasons", []) == [reason_name]
                ]
            bucket_counts[bucket_name] = len(bucket_predictions)
            bucket_payload[bucket_name] = score_prediction_records(
                bucket_predictions,
                game_id=f"{SYSTEM_NAME}_{bucket_name}_seed_{seed}",
            )
        per_seed_bucket_results[seed] = bucket_payload
        per_seed_bucket_counts[seed] = bucket_counts

    aggregated_metrics = {
        split_name: aggregate_seed_metrics(
            {
                seed: per_seed_bucket_results[seed]
                for seed in resolved_seeds
            },
            split_name=split_name,
        )
        for split_name in ("test_iid", "graph_family_ood", "lexical_ood", "variable_naming_ood", "mixed_ood")
    }
    bucket_sample_counts = {
        bucket_name: _bucket_sample_count_summary(per_seed_bucket_counts, bucket_name)
        for bucket_name in ("test_iid", "graph_family_ood", "lexical_ood", "variable_naming_ood", "mixed_ood")
    }
    ood_gap = {
        bucket_name: {
            "verdict_accuracy_gap": (
                aggregated_metrics["test_iid"]["verdict_accuracy"]["mean"]
                - aggregated_metrics[bucket_name]["verdict_accuracy"]["mean"]
            ),
            "macro_f1_gap": (
                aggregated_metrics["test_iid"]["macro_f1"]["mean"]
                - aggregated_metrics[bucket_name]["macro_f1"]["mean"]
            ),
        }
        for bucket_name in ("graph_family_ood", "lexical_ood", "variable_naming_ood", "mixed_ood")
    }
    significance, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            bucket_name: {
                "test_iid": [
                    float(per_seed_bucket_results[seed]["test_iid"]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ],
                bucket_name: [
                    float(per_seed_bucket_results[seed][bucket_name]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ],
            }
            for bucket_name in ("graph_family_ood", "lexical_ood", "variable_naming_ood", "mixed_ood")
        },
        baseline="test_iid",
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )

    payload = {
        "experiment_id": "exp_ood_generalization",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "protocol": protocol,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "per_seed_bucket_results": per_seed_bucket_results,
        "per_seed_bucket_counts": per_seed_bucket_counts,
        "aggregated_metrics": aggregated_metrics,
        "bucket_sample_counts": bucket_sample_counts,
        "bucket_warnings": _small_bucket_warnings(bucket_sample_counts),
        "ood_gap": ood_gap,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/exp_ood_generalization.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 OOD generalization experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=MIN_FORMAL_SAMPLES_PER_FAMILY,
        help="Samples generated per benchmark family.",
    )
    parser.add_argument("--difficulty", type=float, default=0.55, help="Benchmark generation difficulty.")
    parser.add_argument(
        "--allow-protocol-violations",
        action="store_true",
        help="Allow exploratory runs that violate the formal >=3-seed Phase 4 protocol.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        seeds=args.seeds,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["ood_gap"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
