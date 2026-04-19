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
    build_seed_benchmark_run,
    build_seed_metric_significance,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
PRIMARY_OOD_BUCKETS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "graph_family_ood",
        "family_holdout",
        {
            "family_holdout": None,
            "lexical_holdout": [],
            "variable_renaming_holdout": False,
        },
    ),
    (
        "lexical_ood",
        "lexical_holdout",
        {
            "family_holdout": [],
            "lexical_holdout": None,
            "variable_renaming_holdout": False,
        },
    ),
    (
        "variable_naming_ood",
        "variable_renaming_holdout",
        {
            "family_holdout": [],
            "lexical_holdout": [],
            "variable_renaming_holdout": True,
        },
    ),
)


def _empty_bucket_result(*, reason: str) -> dict[str, Any]:
    return {
        "predictions": [],
        "metrics": None,
        "appendix_metrics": {},
        "summary": {},
        "available": False,
        "unavailable_reason": reason,
    }


def _bucket_has_predictions(bucket_payload: dict[str, Any]) -> bool:
    return bool(bucket_payload.get("predictions"))


def _aggregate_bucket_metrics(
    per_seed_bucket_results: dict[int, dict[str, Any]],
    *,
    seeds: list[int],
    bucket_name: str,
) -> dict[str, Any] | None:
    available_seed_payloads = {
        seed: per_seed_bucket_results[seed]
        for seed in seeds
        if _bucket_has_predictions(per_seed_bucket_results[seed][bucket_name])
    }
    if not available_seed_payloads:
        return None
    return aggregate_seed_metrics(available_seed_payloads, split_name=bucket_name)


def _format_metric_cell(metrics: dict[str, Any] | None, metric_name: str) -> str:
    if metrics is None:
        return "N/A"
    return str(metrics[metric_name]["formatted"])


def _bucket_sample_count_summary(
    per_seed_bucket_counts: dict[int, dict[str, int]],
    bucket_name: str,
) -> dict[str, Any]:
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


def _small_bucket_warnings(
    bucket_sample_counts: dict[str, dict[str, Any]],
    *,
    threshold: int = 5,
) -> list[str]:
    warnings: list[str] = []
    for bucket_name, summary in bucket_sample_counts.items():
        if int(summary["min"]) < int(threshold):
            warnings.append(
                f"{bucket_name} has a small support set: per-seed counts={list(summary['per_seed'].values())}."
            )
    return warnings


def _augment_protocol_with_bucket_requirement(
    protocol: dict[str, Any],
    *,
    bucket_sample_counts: dict[str, dict[str, Any]],
    allow_protocol_violations: bool,
) -> dict[str, Any]:
    missing_buckets = [
        bucket_name
        for bucket_name, summary in bucket_sample_counts.items()
        if int(summary["min"]) <= 0
    ]
    requirements = dict(protocol.get("requirements", {}))
    requirements["primary_ood_buckets"] = {
        "required": [bucket_name for bucket_name, _, _ in PRIMARY_OOD_BUCKETS],
        "observed_non_empty_per_seed": {
            bucket_name: int(summary["min"]) > 0
            for bucket_name, summary in bucket_sample_counts.items()
        },
        "satisfied": not missing_buckets,
    }
    violations = [str(item) for item in protocol.get("violations", [])]
    if missing_buckets and "primary_ood_buckets" not in violations:
        violations.append("primary_ood_buckets")
    protocol["requirements"] = requirements
    protocol["violations"] = violations
    protocol["compliant"] = len(violations) == 0
    protocol["override_used"] = bool(allow_protocol_violations and bool(violations))
    return protocol


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# OOD Generalization",
        "",
        "## Sample Count",
        "",
        "| Bucket | IID Ref Count | OOD Count | IID Verdict Acc. | OOD Verdict Acc. | Macro-F1 Gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for bucket_name, _, _ in PRIMARY_OOD_BUCKETS:
        reference_counts = payload["reference_sample_counts"][bucket_name]
        bucket_counts = payload["bucket_sample_counts"][bucket_name]
        gap = payload["ood_gap"][bucket_name]
        reference_metrics = payload["reference_metrics"][bucket_name]
        bucket_metrics = payload["aggregated_metrics"][bucket_name]
        lines.append(
            f"| {bucket_name} | {list(reference_counts['per_seed'].values())} | {list(bucket_counts['per_seed'].values())} | "
            f"{_format_metric_cell(reference_metrics, 'verdict_accuracy')} | "
            f"{_format_metric_cell(bucket_metrics, 'verdict_accuracy')} | "
            f"{'N/A' if not gap['available'] else f'{gap['macro_f1_gap']:.4f}'} |"
        )
    lines.extend(["", "## OOD Gap", ""])
    for bucket_name, gap in payload["ood_gap"].items():
        if not gap["available"]:
            lines.append(f"- {bucket_name}: N/A ({gap['reason']})")
            continue
        lines.append(
            f"- {bucket_name}: accuracy gap={gap['verdict_accuracy_gap']:.4f}, "
            f"macro_f1 gap={gap['macro_f1_gap']:.4f}, "
            f"paired seeds={gap['bucket_seed_list']}"
        )
    if payload["bucket_warnings"]:
        lines.extend(["", "## Bucket Warnings", ""])
        for warning in payload["bucket_warnings"]:
            lines.append(f"- {warning}")
    for bucket_name, _, _ in PRIMARY_OOD_BUCKETS:
        comparison = payload["significance"].get(bucket_name)
        if comparison is None:
            continue
        lines.extend(["", f"## Significance: {bucket_name}", ""])
        lines.append(f"- paired seeds: {comparison['paired_seed_list']}")
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
    manifests: dict[str, dict[int, dict[str, Any]]] = {
        bucket_name: {}
        for bucket_name, _, _ in PRIMARY_OOD_BUCKETS
    }
    per_seed_reference_results: dict[int, dict[str, Any]] = {}
    per_seed_bucket_results: dict[int, dict[str, Any]] = {}
    per_seed_reference_counts: dict[int, dict[str, int]] = {}
    per_seed_bucket_counts: dict[int, dict[str, int]] = {}

    for seed in resolved_seeds:
        per_seed_reference_results[seed] = {}
        per_seed_bucket_results[seed] = {}
        per_seed_reference_counts[seed] = {}
        per_seed_bucket_counts[seed] = {}
        for bucket_name, expected_reason, split_kwargs in PRIMARY_OOD_BUCKETS:
            run = build_seed_benchmark_run(
                seed=seed,
                difficulty=resolved_difficulty,
                samples_per_family=resolved_samples_per_family,
                family_holdout=split_kwargs["family_holdout"],
                lexical_holdout=split_kwargs["lexical_holdout"],
                variable_renaming_holdout=split_kwargs["variable_renaming_holdout"],
            )
            manifests[bucket_name][seed] = manifest_metadata(run)
            ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))

            reference_evaluated = evaluate_system_on_samples(
                run.split_samples["test_iid"],
                seed=seed,
                split_name="test_iid",
                system_name=SYSTEM_NAME,
                ood_reasons=ood_reasons,
            )
            for record in reference_evaluated["predictions"]:
                record["bucket_name"] = bucket_name
                record["bucket_role"] = "iid_reference"
            raw_predictions.extend(reference_evaluated["predictions"])
            per_seed_reference_results[seed][bucket_name] = reference_evaluated
            per_seed_reference_counts[seed][bucket_name] = len(reference_evaluated["predictions"])

            bucket_evaluated = evaluate_system_on_samples(
                run.split_samples["test_ood"],
                seed=seed,
                split_name=bucket_name,
                system_name=SYSTEM_NAME,
                ood_reasons=ood_reasons,
            )
            for record in bucket_evaluated["predictions"]:
                record["bucket_name"] = bucket_name
                record["bucket_role"] = "ood_bucket"
                if record.get("ood_reasons") != [expected_reason]:
                    raise ValueError(
                        f"{bucket_name} must contain only pure {expected_reason!r} examples, "
                        f"got {record.get('ood_reasons')!r} for instance {record['instance_id']!r}."
                    )
            raw_predictions.extend(bucket_evaluated["predictions"])
            per_seed_bucket_results[seed][bucket_name] = (
                bucket_evaluated
                if bucket_evaluated["predictions"]
                else _empty_bucket_result(
                    reason=f"No samples matched the pure {expected_reason} bucket."
                )
            )
            per_seed_bucket_counts[seed][bucket_name] = len(bucket_evaluated["predictions"])

    aggregated_metrics = {
        bucket_name: _aggregate_bucket_metrics(
            per_seed_bucket_results,
            seeds=resolved_seeds,
            bucket_name=bucket_name,
        )
        for bucket_name, _, _ in PRIMARY_OOD_BUCKETS
    }
    reference_metrics = {
        bucket_name: aggregate_seed_metrics(
            {
                seed: per_seed_reference_results[seed]
                for seed in resolved_seeds
            },
            split_name=bucket_name,
        )
        for bucket_name, _, _ in PRIMARY_OOD_BUCKETS
    }
    bucket_sample_counts = {
        bucket_name: _bucket_sample_count_summary(per_seed_bucket_counts, bucket_name)
        for bucket_name, _, _ in PRIMARY_OOD_BUCKETS
    }
    reference_sample_counts = {
        bucket_name: _bucket_sample_count_summary(per_seed_reference_counts, bucket_name)
        for bucket_name, _, _ in PRIMARY_OOD_BUCKETS
    }
    protocol = _augment_protocol_with_bucket_requirement(
        protocol,
        bucket_sample_counts=bucket_sample_counts,
        allow_protocol_violations=allow_protocol_violations,
    )
    missing_primary_buckets = [
        bucket_name
        for bucket_name, metrics in aggregated_metrics.items()
        if metrics is None
    ]
    if missing_primary_buckets and not allow_protocol_violations:
        raise ValueError(
            "Formal OOD runs require non-empty pure buckets for every primary OOD axis; "
            f"missing {missing_primary_buckets!r}."
        )

    ood_gap: dict[str, dict[str, Any]] = {}
    significance_inputs: dict[str, dict[str, list[float]]] = {}
    significance_seed_lists: dict[str, list[int]] = {}
    for bucket_name, _, _ in PRIMARY_OOD_BUCKETS:
        bucket_metrics = aggregated_metrics[bucket_name]
        bucket_seed_list = [
            seed
            for seed in resolved_seeds
            if _bucket_has_predictions(per_seed_bucket_results[seed][bucket_name])
        ]
        reference_seed_list = [
            seed
            for seed in resolved_seeds
            if _bucket_has_predictions(per_seed_reference_results[seed][bucket_name])
        ]
        if bucket_metrics is None:
            ood_gap[bucket_name] = {
                "verdict_accuracy_gap": None,
                "macro_f1_gap": None,
                "available": False,
                "reason": "No pure-bucket samples were available for this OOD category.",
                "reference_seed_list": reference_seed_list,
                "bucket_seed_list": bucket_seed_list,
            }
            continue
        ood_gap[bucket_name] = {
            "verdict_accuracy_gap": (
                reference_metrics[bucket_name]["verdict_accuracy"]["mean"]
                - bucket_metrics["verdict_accuracy"]["mean"]
            ),
            "macro_f1_gap": (
                reference_metrics[bucket_name]["macro_f1"]["mean"]
                - bucket_metrics["macro_f1"]["mean"]
            ),
            "available": True,
            "reason": None,
            "reference_seed_list": reference_seed_list,
            "bucket_seed_list": bucket_seed_list,
        }
        significance_inputs[bucket_name] = {
            "iid_reference": [
                float(per_seed_reference_results[seed][bucket_name]["metrics"]["verdict_accuracy"])
                for seed in bucket_seed_list
            ],
            bucket_name: [
                float(per_seed_bucket_results[seed][bucket_name]["metrics"]["verdict_accuracy"])
                for seed in bucket_seed_list
            ],
        }
        significance_seed_lists[bucket_name] = bucket_seed_list

    significance, global_multiple_comparison_correction = build_seed_metric_significance(
        significance_inputs,
        baseline="iid_reference",
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    ) if significance_inputs else ({}, {"family_size": 0, "alpha": 0.05, "correction": "holm-bonferroni", "entries": []})
    for bucket_name, _, _ in PRIMARY_OOD_BUCKETS:
        report = significance.setdefault(bucket_name, None)
        if report is not None:
            report["paired_seed_list"] = list(significance_seed_lists[bucket_name])

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
        "per_seed_reference_results": per_seed_reference_results,
        "per_seed_bucket_results": per_seed_bucket_results,
        "per_seed_reference_counts": per_seed_reference_counts,
        "per_seed_bucket_counts": per_seed_bucket_counts,
        "reference_metrics": reference_metrics,
        "aggregated_metrics": aggregated_metrics,
        "reference_sample_counts": reference_sample_counts,
        "bucket_sample_counts": bucket_sample_counts,
        "bucket_warnings": _small_bucket_warnings(bucket_sample_counts),
        "ood_gap": ood_gap,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_ood_generalization.json",
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
