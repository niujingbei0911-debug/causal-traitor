"""Phase 4 cross-model transfer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    DEFAULT_ATTACKER_FAMILIES,
    DEFAULT_MODEL_FAMILIES,
    MIN_FORMAL_SEED_COUNT,
    aggregate_seed_metrics,
    apply_attacker_family_profile,
    attack_only_samples,
    build_seed_metric_significance,
    build_seed_benchmark_run,
    evaluate_system_on_samples,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    validate_attacker_families,
    validate_system_names,
    write_artifacts,
)


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Model Transfer",
        "",
        "| Attacker Family | Verifier Family | Split | Verdict Acc. | Macro-F1 | Unidentifiable Awareness |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for attacker_family, verifier_payload in payload["aggregated_metrics"].items():
        for system_name, split_payload in verifier_payload.items():
            for split_name in ("test_iid", "test_ood"):
                metrics = split_payload[split_name]
                lines.append(
                    f"| {attacker_family} | {system_name} | {split_name} | "
                    f"{metrics['verdict_accuracy']['formatted']} | "
                    f"{metrics['macro_f1']['formatted']} | "
                    f"{metrics['unidentifiable_awareness']['formatted']} |"
                )
    lines.append("")
    for attacker_family, split_reports in payload["significance"].items():
        for split_name, comparison in split_reports.items():
            if comparison is None:
                continue
            lines.append(f"## Significance: {attacker_family} / {split_name}")
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
    systems: list[str] | None = None,
    attacker_families: list[str] | None = None,
    samples_per_family: int = 2,
    difficulty: float = 0.55,
    allow_protocol_violations: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = normalize_experiment_seeds(
        seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    resolved_systems = validate_system_names(
        systems,
        default_systems=DEFAULT_MODEL_FAMILIES,
    )
    resolved_attacker_families = validate_attacker_families(attacker_families or list(DEFAULT_ATTACKER_FAMILIES))
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    raw_predictions: list[dict[str, Any]] = []
    per_seed_results: dict[int, Any] = {}
    manifests: dict[int, dict[str, Any]] = {}
    base_runs = {
        seed: build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        for seed in resolved_seeds
    }

    for seed in resolved_seeds:
        run = base_runs[seed]
        manifests[seed] = manifest_metadata(run)
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_payload: dict[str, Any] = {}
        for attacker_family in resolved_attacker_families:
            attacker_payload: dict[str, Any] = {}
            for system_name in resolved_systems:
                system_payload: dict[str, Any] = {}
                for split_name in ("test_iid", "test_ood"):
                    profiled_samples = apply_attacker_family_profile(
                        run.split_samples[split_name],
                        attacker_family=attacker_family,
                    )
                    evaluated = evaluate_system_on_samples(
                        attack_only_samples(profiled_samples),
                        seed=seed,
                        split_name=split_name,
                        system_name=system_name,
                        ood_reasons=ood_reasons,
                    )
                    for record in evaluated["predictions"]:
                        record["attacker_family"] = attacker_family
                    raw_predictions.extend(evaluated["predictions"])
                    system_payload[split_name] = evaluated
                attacker_payload[system_name] = system_payload
            seed_payload[attacker_family] = attacker_payload
        per_seed_results[seed] = seed_payload

    aggregated_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for attacker_family in resolved_attacker_families:
        aggregated_metrics[attacker_family] = {}
        for system_name in resolved_systems:
            aggregated_metrics[attacker_family][system_name] = {}
            for split_name in ("test_iid", "test_ood"):
                aggregated_metrics[attacker_family][system_name][split_name] = aggregate_seed_metrics(
                    {
                        seed: per_seed_results[seed][attacker_family][system_name]
                        for seed in resolved_seeds
                    },
                    split_name=split_name,
                )

    significance_raw, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            f"{attacker_family}::{split_name}": {
                system_name: [
                    float(per_seed_results[seed][attacker_family][system_name][split_name]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ]
                for system_name in resolved_systems
            }
            for attacker_family in resolved_attacker_families
            for split_name in ("test_iid", "test_ood")
        },
        baseline=resolved_systems[0],
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )
    significance: dict[str, dict[str, Any]] = {attacker_family: {} for attacker_family in resolved_attacker_families}
    for attacker_family in resolved_attacker_families:
        for split_name in ("test_iid", "test_ood"):
            significance[attacker_family][split_name] = significance_raw[f"{attacker_family}::{split_name}"]

    payload = {
        "experiment_id": "exp_cross_model_transfer",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "systems": resolved_systems,
        "attacker_families": resolved_attacker_families,
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
        output_path=output_path or "outputs/exp_cross_model_transfer.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 cross-model transfer experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument("--systems", nargs="*", default=None, help="Predictor family names.")
    parser.add_argument("--attacker-families", nargs="*", default=None, help="Attacker family names.")
    parser.add_argument("--samples-per-family", type=int, default=2, help="Samples generated per benchmark family.")
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
        systems=args.systems,
        attacker_families=args.attacker_families,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
