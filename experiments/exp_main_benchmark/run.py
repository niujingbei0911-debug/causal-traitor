"""Phase 4 main benchmark experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

from experiments.benchmark_harness import (
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    compare_system_predictions,
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
    "countermodel_grounded",
    "no_tools",
    "no_countermodel",
    "claim_only_family",
)
DEFAULT_SAMPLES_PER_FAMILY = 10
PAIRWISE_ALPHA = 0.05
PAIRWISE_RESAMPLES = 2000


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(float(q), 1.0)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _paired_seed_bootstrap_row(
    left: list[float],
    right: list[float],
    *,
    comparison_name: str,
    model_a: str,
    model_b: str,
    random_state: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Paired seed bootstrap requires equal numbers of paired seed metrics.")
    if not left:
        raise ValueError("Paired seed bootstrap requires at least one paired seed metric.")

    score_a = _mean(left)
    score_b = _mean(right)
    differences = [float(r - l) for l, r in zip(left, right)]
    observed = _mean(differences)
    centered = [float(diff - observed) for diff in differences]
    rng = Random(int(random_state))
    bootstrap_differences: list[float] = []
    null_differences: list[float] = []
    for _ in range(PAIRWISE_RESAMPLES):
        bootstrap_sample = rng.choices(differences, k=len(differences))
        null_sample = rng.choices(centered, k=len(centered))
        bootstrap_differences.append(_mean(bootstrap_sample))
        null_differences.append(_mean(null_sample))

    p_value = float(
        (
            sum(abs(sample) >= abs(observed) - 1e-12 for sample in null_differences)
            + 1
        )
        / (len(null_differences) + 1)
    )
    return {
        "comparison": comparison_name,
        "model_a": model_a,
        "model_b": model_b,
        "score_a": score_a,
        "score_b": score_b,
        "observed_difference": observed,
        "p_value": p_value,
        "ci_lower": _quantile(bootstrap_differences, 0.025),
        "ci_upper": _quantile(bootstrap_differences, 0.975),
        "alpha": float(PAIRWISE_ALPHA),
        "n_pairs": len(differences),
        "estimand": "seed_mean_verdict_accuracy",
        "correction": "holm-bonferroni",
    }


def _build_seed_mean_significance(
    per_seed_results: dict[int, dict[str, Any]],
    *,
    systems: list[str],
    baseline: str,
    metric_name: str = "verdict_accuracy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    significance: dict[str, Any] = {}
    comparison_rows: dict[tuple[str, str], dict[str, Any]] = {}
    raw_p_values: dict[str, float] = {}
    seed_order = sorted(per_seed_results)

    for split_name in OOD_SPLITS:
        rows: list[dict[str, Any]] = []
        baseline_values = [
            float(per_seed_results[seed][baseline][split_name]["metrics"][metric_name])
            for seed in seed_order
        ]
        for index, system_name in enumerate(systems):
            if system_name == baseline:
                continue
            system_values = [
                float(per_seed_results[seed][system_name][split_name]["metrics"][metric_name])
                for seed in seed_order
            ]
            comparison_name = f"{baseline} vs {system_name}"
            row = _paired_seed_bootstrap_row(
                baseline_values,
                system_values,
                comparison_name=comparison_name,
                model_a=baseline,
                model_b=system_name,
                random_state=17 + index + len(rows) + (10 * len(significance)),
            )
            hypothesis = f"{split_name}: {comparison_name}"
            comparison_rows[(split_name, system_name)] = row
            raw_p_values[hypothesis] = float(row["p_value"])
            rows.append(row)
        significance[split_name] = {
            "method": "paired_seed_bootstrap",
            "metric_name": metric_name,
            "estimand": "seed_mean_verdict_accuracy",
            "alpha": float(PAIRWISE_ALPHA),
            "baseline": baseline,
            "comparisons": rows,
            "correction": "holm-bonferroni",
        }

    correction_table = holm_bonferroni(raw_p_values, alpha=PAIRWISE_ALPHA) if raw_p_values else []
    correction_lookup = {entry.hypothesis: entry for entry in correction_table}
    for split_name in OOD_SPLITS:
        for row in significance[split_name]["comparisons"]:
            hypothesis = f"{split_name}: {row['comparison']}"
            correction = correction_lookup[hypothesis]
            row["adjusted_p_value"] = float(correction.adjusted_p_value)
            row["reject_after_correction"] = bool(correction.reject)
            row["family_hypothesis"] = hypothesis
        significance[split_name]["correction_scope"] = {
            "family_size": len(raw_p_values),
            "hypotheses": list(raw_p_values),
        }

    return significance, {
        "family_size": len(raw_p_values),
        "alpha": float(PAIRWISE_ALPHA),
        "correction": "holm-bonferroni",
        "entries": [
            {
                "hypothesis": entry.hypothesis,
                "p_value": float(entry.raw_p_value),
                "adjusted_p_value": float(entry.adjusted_p_value),
                "threshold": float(entry.threshold),
                "reject": bool(entry.reject),
            }
            for entry in correction_table
        ],
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
                    f"This run used only {protocol['seed_count']} seed(s), below the formal "
                    f"minimum of {protocol['minimum_seed_count']}. Treat it as exploratory only."
                ),
                "",
            ]
        )

    for system_name, split_payload in payload["aggregated_metrics"].items():
        lines.extend(
            [
                f"## {system_name}",
                "",
                "| Split | Verdict Acc. | Macro-F1 | Invalid Accept | Unidentifiable Awareness | ECE | Brier | Countermodel Coverage |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for split_name in OOD_SPLITS:
            metrics = split_payload[split_name]
            lines.append(
                f"| {split_name} | "
                f"{metrics['verdict_accuracy']['formatted']} | "
                f"{metrics['macro_f1']['formatted']} | "
                f"{metrics['invalid_claim_acceptance_rate']['formatted']} | "
                f"{metrics['unidentifiable_awareness']['formatted']} | "
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
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
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
        significance, global_multiple_comparison_correction = _build_seed_mean_significance(
            per_seed_results,
            systems=resolved_systems,
            baseline=resolved_systems[0],
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
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/exp_main_benchmark.json",
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
