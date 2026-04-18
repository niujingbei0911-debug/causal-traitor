"""Phase 4 adversarial robustness experiment stratified by attack strength."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    apply_attack_strength_profile,
    attack_only_samples,
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

SYSTEM_NAME = "countermodel_grounded"
STRENGTH_LEVELS: tuple[str, ...] = (
    "weak",
    "medium",
    "strong",
    "hidden_information_aware",
)


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Adversarial Robustness",
        "",
        "| Strength | Split | Verdict Acc. | Invalid Accept | Unidentifiable Awareness |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strength_name, split_payload in payload["aggregated_metrics"].items():
        for split_name in OOD_SPLITS:
            metrics = split_payload[split_name]
            lines.append(
                f"| {strength_name} | {split_name} | "
                f"{metrics['verdict_accuracy']['formatted']} | "
                f"{metrics['invalid_claim_acceptance_rate']['formatted']} | "
                f"{metrics['unidentifiable_awareness']['formatted']} |"
            )
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
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
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    raw_predictions: list[dict[str, Any]] = []
    per_strength_results: dict[str, Any] = {}
    base_runs = {
        seed: build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        for seed in resolved_seeds
    }

    for strength_name in STRENGTH_LEVELS:
        strength_payload: dict[str, Any] = {
            "difficulty": float(resolved_difficulty),
            "manifests": {},
            "per_seed_results": {},
            "excluded_non_attack_predictions": 0,
        }
        for seed in resolved_seeds:
            run = base_runs[seed]
            strength_payload["manifests"][seed] = manifest_metadata(run)
            ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
            seed_payload: dict[str, Any] = {}
            for split_name in OOD_SPLITS:
                profiled_samples = apply_attack_strength_profile(
                    run.split_samples[split_name],
                    strength_name=strength_name,
                )
                attack_samples = attack_only_samples(profiled_samples)
                strength_payload["excluded_non_attack_predictions"] += max(0, len(profiled_samples) - len(attack_samples))
                evaluated = evaluate_system_on_samples(
                    attack_samples,
                    seed=seed,
                    split_name=split_name,
                    system_name=SYSTEM_NAME,
                    ood_reasons=ood_reasons,
                )
                for record in evaluated["predictions"]:
                    record["attack_strength"] = strength_name
                raw_predictions.extend(evaluated["predictions"])
                seed_payload[split_name] = evaluated
            strength_payload["per_seed_results"][seed] = seed_payload
        per_strength_results[strength_name] = strength_payload

    aggregated_metrics = {
        strength_name: {
            split_name: aggregate_seed_metrics(
                {
                    seed: payload["per_seed_results"][seed]
                    for seed in resolved_seeds
                },
                split_name=split_name,
            )
            for split_name in OOD_SPLITS
        }
        for strength_name, payload in per_strength_results.items()
    }
    significance, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            split_name: {
                strength_name: [
                    float(per_strength_results[strength_name]["per_seed_results"][seed][split_name]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ]
                for strength_name in STRENGTH_LEVELS
            }
            for split_name in OOD_SPLITS
        },
        baseline="weak",
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )

    payload = {
        "experiment_id": "exp_adversarial_robustness",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
            "strength_levels": list(STRENGTH_LEVELS),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "protocol": protocol,
        "per_strength_results": per_strength_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/exp_adversarial_robustness.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 adversarial robustness experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
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
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
