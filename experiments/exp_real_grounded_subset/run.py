"""Phase 4 real-grounded subset experiment with synthetic vs real-grounded reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.loaders import load_real_grounded_dataset, load_real_grounded_samples
from experiments.benchmark_harness import (
    MAIN_BENCHMARK_SYSTEMS,
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    build_seed_metric_significance,
    evaluate_system_on_samples,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    validate_system_names,
    write_artifacts,
)

PRIMARY_SIGNIFICANCE_METRIC = "unsafe_acceptance_rate"
DATASET_PARTITIONS: tuple[str, ...] = ("synthetic", "real_grounded")
REAL_GROUNDED_SCOPE = "real_grounded"


def _partition_scope_names(dataset_partition: str) -> tuple[str, ...]:
    if dataset_partition == "synthetic":
        return OOD_SPLITS
    if dataset_partition == "real_grounded":
        return (REAL_GROUNDED_SCOPE,)
    raise ValueError(f"Unsupported dataset partition: {dataset_partition!r}.")


def _dataset_request_value(dataset: Any) -> str:
    if dataset is None:
        return ""
    if isinstance(dataset, (str, Path)):
        return str(dataset)
    return "<in-memory dataset>"


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Real-Grounded Subset",
        "",
        "## Setup",
        "",
        f"- Systems: {', '.join(payload['systems'])}",
        f"- Seeds: {payload['seeds']}",
        f"- Samples per family: {payload['config']['samples_per_family']}",
        f"- Difficulty: {payload['config']['difficulty']:.2f}",
        f"- Dataset Partitions: {', '.join(payload['dataset_partitions'])}",
        f"- Dataset Name: {payload['real_grounded_dataset']['dataset_name']}",
        "",
    ]
    for dataset_partition, partition_payload in payload["aggregated_metrics"].items():
        lines.extend(
            [
                f"## {dataset_partition}",
                "",
                "| System | Scope | Verdict Acc. | Unsafe Accept | Wise Refusal Recall | Wise Refusal Precision | Over-Refusal | ECE | Brier |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for system_name, scope_payload in partition_payload.items():
            for scope_name in _partition_scope_names(dataset_partition):
                metrics = scope_payload[scope_name]
                lines.append(
                    f"| {system_name} | {scope_name} | "
                    f"{metrics['verdict_accuracy']['formatted']} | "
                    f"{metrics['unsafe_acceptance_rate']['formatted']} | "
                    f"{metrics['wise_refusal_recall']['formatted']} | "
                    f"{metrics['wise_refusal_precision']['formatted']} | "
                    f"{metrics['over_refusal_rate']['formatted']} | "
                    f"{metrics['ece']['formatted']} | "
                    f"{metrics['brier']['formatted']} |"
                )
        lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    dataset: Any | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
    systems: list[str] | tuple[str, ...] | None = None,
    samples_per_family: int = MIN_FORMAL_SAMPLES_PER_FAMILY,
    difficulty: float = 0.55,
    allow_protocol_violations: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    if dataset is None:
        raise ValueError(
            "exp_real_grounded_subset requires an explicit dataset source. "
            "Pass dataset=<path|payload> or use --dataset <path>."
        )

    resolved_seeds = normalize_experiment_seeds(
        seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    resolved_systems = validate_system_names(
        systems,
        default_systems=MAIN_BENCHMARK_SYSTEMS,
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

    real_grounded_dataset = load_real_grounded_dataset(dataset)
    real_grounded_samples = load_real_grounded_samples(real_grounded_dataset)
    raw_predictions: list[dict[str, Any]] = []
    per_partition_results: dict[str, Any] = {
        "synthetic": {},
        "real_grounded": {},
    }

    for seed in resolved_seeds:
        synthetic_run = build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        synthetic_ood_reasons = dict(synthetic_run.manifest.metadata.get("ood_reasons", {}))
        synthetic_seed_payload: dict[str, Any] = {}
        real_grounded_seed_payload: dict[str, Any] = {}

        for system_name in resolved_systems:
            synthetic_system_payload: dict[str, Any] = {}
            real_grounded_system_payload: dict[str, Any] = {}
            for split_name in OOD_SPLITS:
                synthetic_evaluated = evaluate_system_on_samples(
                    synthetic_run.split_samples[split_name],
                    seed=seed,
                    split_name=split_name,
                    system_name=system_name,
                    ood_reasons=synthetic_ood_reasons,
                )
                for record in synthetic_evaluated["predictions"]:
                    record["dataset_partition"] = "synthetic"
                raw_predictions.extend(synthetic_evaluated["predictions"])
                synthetic_system_payload[split_name] = synthetic_evaluated

            real_grounded_evaluated = evaluate_system_on_samples(
                real_grounded_samples,
                seed=seed,
                split_name=REAL_GROUNDED_SCOPE,
                system_name=system_name,
            )
            for record in real_grounded_evaluated["predictions"]:
                record["dataset_partition"] = "real_grounded"
            raw_predictions.extend(real_grounded_evaluated["predictions"])
            real_grounded_system_payload[REAL_GROUNDED_SCOPE] = real_grounded_evaluated

            synthetic_seed_payload[system_name] = synthetic_system_payload
            real_grounded_seed_payload[system_name] = real_grounded_system_payload

        per_partition_results["synthetic"][seed] = synthetic_seed_payload
        per_partition_results["real_grounded"][seed] = real_grounded_seed_payload

    aggregated_metrics: dict[str, Any] = {}
    for dataset_partition, per_seed_payload in per_partition_results.items():
        aggregated_metrics[dataset_partition] = {}
        for system_name in resolved_systems:
            aggregated_metrics[dataset_partition][system_name] = {}
            for scope_name in _partition_scope_names(dataset_partition):
                aggregated_metrics[dataset_partition][system_name][scope_name] = aggregate_seed_metrics(
                    {
                        seed: per_seed_payload[seed][system_name]
                        for seed in resolved_seeds
                    },
                    split_name=scope_name,
                )

    significance: dict[str, Any] = {}
    global_multiple_comparison_correction: dict[str, Any] = {
        "family_size": 0,
        "alpha": 0.05,
        "correction": "holm-bonferroni",
        "entries": [],
    }
    for dataset_partition, per_seed_payload in per_partition_results.items():
        scope_names = _partition_scope_names(dataset_partition)
        if len(resolved_systems) < 2:
            significance[dataset_partition] = {
                scope_name: None
                for scope_name in scope_names
            }
            continue
        significance[dataset_partition], correction = build_seed_metric_significance(
            {
                scope_name: {
                    system_name: [
                        float(per_seed_payload[seed][system_name][scope_name]["metrics"][PRIMARY_SIGNIFICANCE_METRIC])
                        for seed in sorted(resolved_seeds)
                    ]
                    for system_name in resolved_systems
                }
                for scope_name in scope_names
            },
            baseline=resolved_systems[0],
            metric_name=PRIMARY_SIGNIFICANCE_METRIC,
            estimand=f"seed_mean_{PRIMARY_SIGNIFICANCE_METRIC}",
        )
        if correction["family_size"] >= global_multiple_comparison_correction["family_size"]:
            global_multiple_comparison_correction = correction

    payload = {
        "experiment_id": "exp_real_grounded_subset",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
            "dataset_partitions": list(DATASET_PARTITIONS),
            "partition_scopes": {
                dataset_partition: list(_partition_scope_names(dataset_partition))
                for dataset_partition in DATASET_PARTITIONS
            },
        },
        "requested_config": {
            "dataset": _dataset_request_value(dataset),
            "systems": list(systems) if systems is not None else list(MAIN_BENCHMARK_SYSTEMS),
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "systems": resolved_systems,
        "dataset_partitions": list(DATASET_PARTITIONS),
        "seeds": resolved_seeds,
        "protocol": protocol,
        "real_grounded_dataset": real_grounded_dataset.to_dict(),
        "per_partition_results": per_partition_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_real_grounded_subset.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 real-grounded subset experiment.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the real-grounded dataset JSON file.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument("--systems", nargs="*", default=None, help="Optional system matrix to evaluate.")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=MIN_FORMAL_SAMPLES_PER_FAMILY,
        help="Synthetic samples generated per benchmark family.",
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
        dataset=args.dataset,
        seeds=args.seeds,
        systems=args.systems,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
