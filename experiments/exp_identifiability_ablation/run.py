"""Phase 4 identifiability ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_metric_significance,
    build_seed_benchmark_run,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    write_artifacts,
)

SYSTEMS: tuple[str, ...] = (
    "countermodel_grounded",
    "no_ledger",
    "no_countermodel",
    "no_abstention",
    "no_tools",
)

def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Identifiability Ablation",
        "",
        "| System | Split | Verdict Acc. | Invalid Accept | Unidentifiable Awareness | Countermodel Coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for system_name, split_payload in payload["aggregated_metrics"].items():
        for split_name in OOD_SPLITS:
            metrics = split_payload[split_name]
            lines.append(
                f"| {system_name} | {split_name} | "
                f"{metrics['verdict_accuracy']['formatted']} | "
                f"{metrics['invalid_claim_acceptance_rate']['formatted']} | "
                f"{metrics['unidentifiable_awareness']['formatted']} | "
                f"{metrics['countermodel_coverage']['formatted']} |"
            )
    lines.append("")

    for split_name, comparison in payload["significance"].items():
        if comparison is None:
            continue
        lines.append(f"## Significance: {split_name}")
        lines.append("")
        for row in comparison["comparisons"]:
            lines.append(
                f"- {row['comparison']}: diff={row['observed_difference']:.4f}, "
                f"p={row['p_value']:.4f}, adjusted={row.get('adjusted_p_value')}"
            )
        lines.append("")
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
    per_seed_results: dict[int, dict[str, Any]] = {}
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

    significance, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            split_name: {
                system_name: [
                    float(per_seed_results[seed][system_name][split_name]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ]
                for system_name in SYSTEMS
            }
            for split_name in OOD_SPLITS
        },
        baseline=SYSTEMS[0],
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )

    payload = {
        "experiment_id": "exp_identifiability_ablation",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "systems": list(SYSTEMS),
        "seeds": resolved_seeds,
        "protocol": protocol,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_identifiability_ablation.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 identifiability ablation.")
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
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
