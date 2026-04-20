"""Phase 4 persuasion robustness experiment stratified by persuasion pressure type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    DEFAULT_PRESSURE_TYPES,
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    apply_persuasion_pressure_profile,
    build_seed_attack_benchmark_run,
    build_seed_metric_significance,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    validate_pressure_types,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
PRIMARY_SIGNIFICANCE_METRIC = "unsafe_acceptance_rate"


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Persuasion Robustness",
        "",
        "## Setup",
        "",
        f"- System: {payload['system']}",
        f"- Pressure Types: {', '.join(payload['pressure_types'])}",
        f"- Seeds: {payload['seeds']}",
        f"- Samples per family: {payload['config']['samples_per_family']}",
        f"- Difficulty: {payload['config']['difficulty']:.2f}",
        "",
        "| Pressure Type | Split | Verdict Acc. | Unsafe Accept | Wise Refusal Recall | Wise Refusal Precision | Over-Refusal | ECE | Brier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pressure_type, split_payload in payload["aggregated_metrics"].items():
        for split_name in OOD_SPLITS:
            metrics = split_payload[split_name]
            lines.append(
                f"| {pressure_type} | {split_name} | "
                f"{metrics['verdict_accuracy']['formatted']} | "
                f"{metrics['unsafe_acceptance_rate']['formatted']} | "
                f"{metrics['wise_refusal_recall']['formatted']} | "
                f"{metrics['wise_refusal_precision']['formatted']} | "
                f"{metrics['over_refusal_rate']['formatted']} | "
                f"{metrics['ece']['formatted']} | "
                f"{metrics['brier']['formatted']} |"
            )
    for split_name, comparison in payload.get("significance", {}).items():
        if comparison is None:
            continue
        lines.extend(["", f"## Significance: {split_name}", ""])
        for row in comparison["comparisons"]:
            lines.append(
                f"- {row['comparison']}: diff={row['observed_difference']:.4f}, "
                f"p={row['p_value']:.4f}, adjusted={row.get('adjusted_p_value')}"
            )
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    pressure_types: list[str] | tuple[str, ...] | None = None,
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
    resolved_pressure_types = validate_pressure_types(pressure_types)
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    effective_samples_per_family = max(2, resolved_samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        minimum_samples_per_family=MIN_FORMAL_SAMPLES_PER_FAMILY,
        observed_samples_per_family=effective_samples_per_family,
        allow_protocol_violations=allow_protocol_violations,
    )
    raw_predictions: list[dict[str, Any]] = []
    per_pressure_results: dict[str, Any] = {}
    base_runs = {
        seed: build_seed_attack_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=effective_samples_per_family,
        )
        for seed in resolved_seeds
    }

    for pressure_type in resolved_pressure_types:
        pressure_payload: dict[str, Any] = {
            "difficulty": float(resolved_difficulty),
            "manifests": {},
            "per_seed_results": {},
        }
        for seed in resolved_seeds:
            run = base_runs[seed]
            pressure_payload["manifests"][seed] = manifest_metadata(run)
            ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
            seed_payload: dict[str, Any] = {}
            for split_name in OOD_SPLITS:
                profiled_samples = apply_persuasion_pressure_profile(
                    run.split_samples[split_name],
                    pressure_type=pressure_type,
                )
                evaluated = evaluate_system_on_samples(
                    profiled_samples,
                    seed=seed,
                    split_name=split_name,
                    system_name=SYSTEM_NAME,
                    ood_reasons=ood_reasons,
                )
                for record in evaluated["predictions"]:
                    record["pressure_type"] = pressure_type
                raw_predictions.extend(evaluated["predictions"])
                seed_payload[split_name] = evaluated
            pressure_payload["per_seed_results"][seed] = seed_payload
        per_pressure_results[pressure_type] = pressure_payload

    aggregated_metrics = {
        pressure_type: {
            split_name: aggregate_seed_metrics(
                {
                    seed: payload["per_seed_results"][seed]
                    for seed in resolved_seeds
                },
                split_name=split_name,
            )
            for split_name in OOD_SPLITS
        }
        for pressure_type, payload in per_pressure_results.items()
    }

    baseline_pressure_type = "none" if "none" in resolved_pressure_types else resolved_pressure_types[0]
    if len(resolved_pressure_types) < 2:
        significance = {split_name: None for split_name in OOD_SPLITS}
        global_multiple_comparison_correction = {
            "family_size": 0,
            "alpha": 0.05,
            "correction": "holm-bonferroni",
            "entries": [],
        }
    else:
        significance, global_multiple_comparison_correction = build_seed_metric_significance(
            {
                split_name: {
                    pressure_type: [
                        float(
                            per_pressure_results[pressure_type]["per_seed_results"][seed][split_name]["metrics"][
                                PRIMARY_SIGNIFICANCE_METRIC
                            ]
                        )
                        for seed in sorted(resolved_seeds)
                    ]
                    for pressure_type in resolved_pressure_types
                }
                for split_name in OOD_SPLITS
            },
            baseline=baseline_pressure_type,
            metric_name=PRIMARY_SIGNIFICANCE_METRIC,
            estimand=f"seed_mean_{PRIMARY_SIGNIFICANCE_METRIC}",
        )

    payload = {
        "experiment_id": "exp_persuasion_robustness",
        "config": {
            "samples_per_family": int(effective_samples_per_family),
            "difficulty": float(resolved_difficulty),
            "pressure_types": list(resolved_pressure_types),
            "attack_split_protocol": "dedicated_attack_only_benchmark",
            "comparison_axis": "pressure_type",
            "baseline_pressure_type": baseline_pressure_type,
        },
        "requested_config": {
            "pressure_types": list(pressure_types) if pressure_types is not None else list(DEFAULT_PRESSURE_TYPES),
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "system": SYSTEM_NAME,
        "pressure_types": resolved_pressure_types,
        "seeds": resolved_seeds,
        "protocol": protocol,
        "per_pressure_results": per_pressure_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_persuasion_robustness.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 persuasion robustness experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument(
        "--pressure-types",
        nargs="*",
        default=None,
        help="Optional pressure types to evaluate. Defaults to the full persuasion registry plus none.",
    )
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
        pressure_types=args.pressure_types,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
