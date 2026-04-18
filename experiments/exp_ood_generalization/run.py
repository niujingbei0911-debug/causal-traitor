"""Phase 4 OOD generalization experiment."""

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


def _normalize_seeds(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    if not seeds:
        return list(DEFAULT_SEEDS)
    return [int(seed) for seed in seeds]


def _markdown_summary(payload: dict[str, Any]) -> str:
    iid = payload["aggregated_metrics"]["test_iid"]
    ood = payload["aggregated_metrics"]["test_ood"]
    return "\n".join(
        [
            "# OOD Generalization",
            "",
            "| Split | Verdict Acc. | Macro-F1 | Invalid Accept | Unidentifiable Awareness |",
            "| --- | --- | --- | --- | --- |",
            f"| test_iid | {iid['verdict_accuracy']['formatted']} | {iid['macro_f1']['formatted']} | {iid['invalid_claim_acceptance_rate']['formatted']} | {iid['unidentifiable_awareness']['formatted']} |",
            f"| test_ood | {ood['verdict_accuracy']['formatted']} | {ood['macro_f1']['formatted']} | {ood['invalid_claim_acceptance_rate']['formatted']} | {ood['unidentifiable_awareness']['formatted']} |",
            "",
            "## OOD Gap",
            "",
            f"- Verdict accuracy gap (IID - OOD): {payload['ood_gap']['verdict_accuracy_gap']:.4f}",
            f"- Macro-F1 gap (IID - OOD): {payload['ood_gap']['macro_f1_gap']:.4f}",
        ]
    )


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    samples_per_family: int = 2,
    difficulty: float = 0.55,
    output_path: str | None = None,
) -> dict[str, Any]:
    resolved_seeds = _normalize_seeds(seeds)
    raw_predictions: list[dict[str, Any]] = []
    per_seed_results: dict[int, Any] = {}
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
        for split_name in OOD_SPLITS:
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

    aggregated_metrics = {
        split_name: aggregate_seed_metrics(
            {
                seed: per_seed_results[seed]
                for seed in resolved_seeds
            },
            split_name=split_name,
        )
        for split_name in OOD_SPLITS
    }
    ood_gap = {
        "verdict_accuracy_gap": (
            aggregated_metrics["test_iid"]["verdict_accuracy"]["mean"]
            - aggregated_metrics["test_ood"]["verdict_accuracy"]["mean"]
        ),
        "macro_f1_gap": (
            aggregated_metrics["test_iid"]["macro_f1"]["mean"]
            - aggregated_metrics["test_ood"]["macro_f1"]["mean"]
        ),
    }

    payload = {
        "experiment_id": "exp_ood_generalization",
        "config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "ood_gap": ood_gap,
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
    print(json.dumps(payload["ood_gap"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
