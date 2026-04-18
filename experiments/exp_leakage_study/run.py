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

SYSTEMS: tuple[str, str] = ("countermodel_grounded", "oracle_leaking_partition")


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


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Leakage Study",
        "",
        "This experiment contrasts the clean public partition against a deliberately oracle-leaking partition.",
        "",
        "| Split | Clean Accuracy | Oracle-Leaking Accuracy | Inflation |",
        "| --- | --- | --- | --- |",
    ]
    for split_name in OOD_SPLITS:
        delta = payload["inflation"][split_name]
        lines.append(
            f"| {split_name} | {delta['clean_accuracy_mean']:.4f} | "
            f"{delta['leaking_accuracy_mean']:.4f} | {delta['inflation_delta']:.4f} |"
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
    lines.append("## Leakage Warning")
    lines.append("")
    lines.append(
        "Oracle access to gold-only supervision produces an artificially inflated score and must never be mixed into the main benchmark."
    )
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    samples_per_family: int = 2,
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
    inflation = {
        split_name: _inflation_summary(
            {
                "aggregated_metrics": aggregated_metrics,
            },
            split_name,
        )
        for split_name in OOD_SPLITS
    }

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
    parser.add_argument("--samples-per-family", type=int, default=2, help="Samples generated per benchmark family.")
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
