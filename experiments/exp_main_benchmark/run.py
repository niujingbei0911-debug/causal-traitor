"""Phase 4 main benchmark experiment."""

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
    validate_system_names,
    write_artifacts,
)
from evaluation.significance import holm_bonferroni

DEFAULT_MAIN_SYSTEMS: tuple[str, ...] = (
    "judge_direct",
    "debate_reduced",
    "tool_only",
    "countermodel_grounded",
)
DEFAULT_SAMPLES_PER_FAMILY = 10
PAIRWISE_ALPHA = 0.05


def _validate_main_benchmark_systems(
    systems: list[str],
    *,
    forbidden: tuple[str, ...] = ("oracle_leaking_partition",),
) -> list[str]:
    disallowed = [system_name for system_name in systems if system_name in set(forbidden)]
    if disallowed:
        raise ValueError(
            "exp_main_benchmark must stay leakage-free. Remove disallowed system(s): "
            f"{disallowed!r}."
        )
    return list(systems)


def _build_formal_paired_significance(
    per_seed_results: dict[int, dict[str, dict[str, Any]]],
    *,
    systems: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_seed_metric_significance(
        {
            split_name: {
                system_name: [
                    float(per_seed_results[seed][system_name][split_name]["metrics"]["verdict_accuracy"])
                    for seed in sorted(per_seed_results)
                ]
                for system_name in systems
            }
            for split_name in OOD_SPLITS
        },
        baseline=systems[0],
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )


def _blueprint_alignment_summary(systems: list[str]) -> dict[str, Any]:
    required = {
        "Judge": "judge_direct",
        "Debate": "debate_reduced",
        "Tool": "tool_only",
        "Ours": "countermodel_grounded",
    }
    missing = [
        category
        for category, system_name in required.items()
        if system_name not in set(systems)
    ]
    connected = not missing
    return {
        "full_baseline_matrix_connected": connected,
        "implemented_systems": list(systems),
        "missing_blueprint_baseline_categories": missing,
        "note": (
            "The runner now evaluates one representative baseline for each required Phase 4 category: "
            "Judge (`judge_direct`), Debate (`debate_reduced`), Tool (`tool_only`), and Ours (`countermodel_grounded`)."
            if connected
            else (
                "The current runner is missing required baseline categories from the formal "
                "Judge / Debate / Tool / Ours matrix."
            )
        ),
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    lines = [
        "# Main Benchmark",
        "",
        "## Setup",
        "",
        f"- Seeds: {payload['seeds']}",
        f"- Samples per family: {payload['config']['samples_per_family']}",
        f"- Difficulty: {payload['config']['difficulty']:.2f}",
        f"- Systems: {', '.join(payload['systems'])}",
        "",
    ]
    if protocol and not protocol.get("compliant", True):
        lines.extend(
            [
                "## Protocol Notice",
                "",
                (
                    "This run is exploratory only because it misses one or more formal protocol "
                    f"requirements: {', '.join(protocol.get('violations', []))}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Blueprint Alignment",
            "",
            payload["blueprint_alignment"]["note"],
            "",
        ]
    )

    for system_name, split_payload in payload["aggregated_metrics"].items():
        lines.extend(
            [
                f"## {system_name}",
                "",
                "| Split | Verdict Acc. | Macro-F1 | Unsafe Accept | Wise Refusal Recall | Wise Refusal Precision | Over-Refusal | ECE | Brier | Countermodel Coverage |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for split_name in OOD_SPLITS:
            metrics = split_payload[split_name]
            lines.append(
                f"| {split_name} | "
                f"{metrics['verdict_accuracy']['formatted']} | "
                f"{metrics['macro_f1']['formatted']} | "
                f"{metrics['unsafe_acceptance_rate']['formatted']} | "
                f"{metrics['wise_refusal_recall']['formatted']} | "
                f"{metrics['wise_refusal_precision']['formatted']} | "
                f"{metrics['over_refusal_rate']['formatted']} | "
                f"{metrics['ece']['formatted']} | "
                f"{metrics['brier']['formatted']} | "
                f"{metrics['countermodel_coverage']['formatted']} |"
            )
        lines.append("")

    for split_name, comparison in payload.get("significance", {}).items():
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
    systems: list[str] | None = None,
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
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
        default_systems=DEFAULT_MAIN_SYSTEMS,
    )
    resolved_systems = _validate_main_benchmark_systems(resolved_systems)
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
        for system_name in resolved_systems:
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
    for system_name in resolved_systems:
        aggregated_metrics[system_name] = {}
        for split_name in OOD_SPLITS:
            aggregated_metrics[system_name][split_name] = aggregate_seed_metrics(
                {
                    seed: per_seed_results[seed][system_name]
                    for seed in resolved_seeds
                },
                split_name=split_name,
            )

    significance: dict[str, Any]
    global_multiple_comparison_correction: dict[str, Any]
    if len(resolved_systems) < 2:
        significance = {split_name: None for split_name in OOD_SPLITS}
        global_multiple_comparison_correction = {
            "family_size": 0,
            "alpha": float(PAIRWISE_ALPHA),
            "correction": "holm-bonferroni",
            "entries": [],
        }
    else:
        significance, global_multiple_comparison_correction = _build_formal_paired_significance(
            per_seed_results,
            systems=resolved_systems,
        )

    payload = {
        "experiment_id": "exp_main_benchmark",
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
        "seeds": resolved_seeds,
        "protocol": protocol,
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "blueprint_alignment": _blueprint_alignment_summary(resolved_systems),
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_main_benchmark.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 main benchmark experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument("--systems", nargs="*", default=None, help="System names to evaluate.")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
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
        systems=args.systems,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
