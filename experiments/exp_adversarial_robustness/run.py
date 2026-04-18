"""Phase 4 adversarial robustness experiment stratified by attack strength."""

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
    evaluate_system_on_samples,
    manifest_metadata,
    write_artifacts,
)

SYSTEM_NAME = "countermodel_grounded"
STRENGTH_LEVELS: tuple[tuple[str, float], ...] = (
    ("weak", 0.30),
    ("medium", 0.55),
    ("strong", 0.80),
)


def _normalize_seeds(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    if not seeds:
        return list(DEFAULT_SEEDS)
    return [int(seed) for seed in seeds]


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
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = _normalize_seeds(seeds)
    raw_predictions: list[dict[str, Any]] = []
    per_strength_results: dict[str, Any] = {}

    for strength_name, difficulty in STRENGTH_LEVELS:
        strength_payload: dict[str, Any] = {
            "difficulty": float(difficulty),
            "manifests": {},
            "per_seed_results": {},
        }
        for seed in resolved_seeds:
            run = build_seed_benchmark_run(
                seed=seed,
                difficulty=difficulty,
                samples_per_family=samples_per_family,
            )
            strength_payload["manifests"][seed] = manifest_metadata(run)
            ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
            seed_payload: dict[str, Any] = {}
            for split_name in OOD_SPLITS:
                evaluated = evaluate_system_on_samples(
                    run.split_samples[split_name],
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

    payload = {
        "experiment_id": "exp_adversarial_robustness",
        "config": {
            "samples_per_family": int(samples_per_family),
            "strength_levels": {name: diff for name, diff in STRENGTH_LEVELS},
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "per_strength_results": per_strength_results,
        "aggregated_metrics": aggregated_metrics,
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
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_experiment(
        seeds=args.seeds,
        samples_per_family=args.samples_per_family,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
