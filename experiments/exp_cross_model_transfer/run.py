"""Phase 4 cross-model transfer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.benchmark_harness import (
    DEFAULT_MODEL_FAMILIES,
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    aggregate_seed_metrics,
    apply_attacker_model_family_profile,
    build_seed_attack_benchmark_run,
    build_seed_metric_significance,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    predict_sample_for_model_family,
    score_prediction_records,
    summarize_protocol_compliance,
    validate_model_families,
    write_artifacts,
)

PRIMARY_SIGNIFICANCE_METRIC = "unsafe_acceptance_rate"

PRIMARY_SIGNIFICANCE_METRIC = "unsafe_acceptance_rate"


def _evaluate_model_family_on_samples(
    samples,
    *,
    seed: int,
    split_name: str,
    attacker_model_family: str,
    verifier_model_family: str,
    ood_reasons: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    manifest_ood_reasons = dict(ood_reasons or {})
    for sample in sorted(samples, key=lambda item: item.claim.instance_id):
        payload = predict_sample_for_model_family(
            sample,
            verifier_model_family=verifier_model_family,
        )
        verdict = dict(payload["verdict"])
        public_payload = sample.public.to_dict()
        record = {
            "seed": int(seed),
            "split": split_name,
            "system_name": verifier_model_family,
            "instance_id": sample.claim.instance_id,
            "scenario_id": sample.gold.scenario_id,
            "causal_level": public_payload["causal_level"],
            "graph_family": sample.claim.graph_family,
            "language_template_id": sample.claim.language_template_id,
            "query_type": sample.claim.query_type,
            "attack_name": sample.claim.meta.get("attack_name"),
            "style_id": sample.claim.meta.get("style_id"),
            "claim_mode": sample.claim.meta.get("claim_mode"),
            "gold_label": sample.claim.gold_label.value,
            "predicted_label": payload["predicted_label"],
            "confidence": float(payload["confidence"]),
            "supports_public_only": bool(payload["supports_public_only"]),
            "ood_reasons": list(manifest_ood_reasons.get(sample.claim.instance_id, [])),
            "claim_text": sample.claim.claim_text,
            "target_variables": dict(sample.claim.target_variables),
            "proxy_variables": list(sample.public.proxy_variables),
            "selection_variables": list(sample.public.selection_variables),
            "selection_mechanism": sample.public.selection_mechanism,
            "tool_report": dict(payload["tool_report"]),
            "tool_trace": list(payload["tool_report"].get("tool_trace", [])),
            "countermodel_found": bool(payload["countermodel_found"]),
            "countermodel_type": payload["countermodel_type"],
            "predicted_probabilities": dict(verdict.get("probabilities", {})),
            "verdict": verdict,
            "attacker_model_family": attacker_model_family,
            "verifier_model_family": verifier_model_family,
            "family_relation": (
                "cross_family"
                if attacker_model_family != verifier_model_family
                else "same_family_reference"
            ),
            "system_notes": list(payload["system_notes"]),
        }
        predictions.append(record)
    return score_prediction_records(
        predictions,
        game_id=f"{attacker_model_family}_{verifier_model_family}_{split_name}_seed_{seed}",
    )


def _family_pairs(
    attacker_model_families: list[str],
    verifier_model_families: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "attacker_model_family": attacker_family,
            "verifier_model_family": verifier_family,
            "cross_family": attacker_family != verifier_family,
        }
        for attacker_family in attacker_model_families
        for verifier_family in verifier_model_families
    ]


def _blueprint_alignment_summary(
    verifier_model_families: list[str],
    attacker_model_families: list[str],
) -> dict[str, Any]:
    return {
        "model_family_transfer_realized": True,
        "transfer_axis": "attacker_model_family_x_verifier_model_family",
        "verifier_model_families": list(verifier_model_families),
        "attacker_model_families": list(attacker_model_families),
        "note": (
            "This runner evaluates the formal Phase 4 transfer matrix by varying attacker and verifier "
            "model-family profiles independently, including both same-family references and cross-family pairs."
        ),
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Model Transfer",
        "",
        payload["blueprint_alignment"]["note"],
        "",
        "| Attacker Family | Verifier Family | Relation | Split | Verdict Acc. | Macro-F1 | Unidentifiable Awareness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for attacker_family, verifier_payload in payload["aggregated_metrics"].items():
        for verifier_family, split_payload in verifier_payload.items():
            relation = "cross" if attacker_family != verifier_family else "same"
            for split_name in ("test_iid", "test_ood"):
                metrics = split_payload[split_name]
                lines.append(
                    f"| {attacker_family} | {verifier_family} | {relation} | {split_name} | "
                    f"{metrics['verdict_accuracy']['formatted']} | "
                    f"{metrics['macro_f1']['formatted']} | "
                    f"{metrics['wise_refusal_recall']['formatted']} |"
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
    verifier_model_families: list[str] | None = None,
    attacker_model_families: list[str] | None = None,
    systems: list[str] | None = None,
    attacker_families: list[str] | None = None,
    samples_per_family: int = MIN_FORMAL_SAMPLES_PER_FAMILY,
    difficulty: float = 0.55,
    allow_surrogate_transfer: bool = False,
    allow_protocol_violations: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    del allow_surrogate_transfer
    resolved_seeds = normalize_experiment_seeds(
        seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        allow_protocol_violations=allow_protocol_violations,
    )
    resolved_verifier_families = validate_model_families(
        verifier_model_families or systems,
        default_families=DEFAULT_MODEL_FAMILIES,
    )
    resolved_attacker_families = validate_model_families(
        attacker_model_families or attacker_families,
        default_families=DEFAULT_MODEL_FAMILIES,
    )
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
    per_seed_results: dict[int, Any] = {}
    manifests: dict[int, dict[str, Any]] = {}
    base_runs = {
        seed: build_seed_attack_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=effective_samples_per_family,
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
            for verifier_family in resolved_verifier_families:
                verifier_payload: dict[str, Any] = {}
                for split_name in ("test_iid", "test_ood"):
                    profiled_samples = apply_attacker_model_family_profile(
                        run.split_samples[split_name],
                        attacker_model_family=attacker_family,
                    )
                    evaluated = _evaluate_model_family_on_samples(
                        profiled_samples,
                        seed=seed,
                        split_name=split_name,
                        attacker_model_family=attacker_family,
                        verifier_model_family=verifier_family,
                        ood_reasons=ood_reasons,
                    )
                    raw_predictions.extend(evaluated["predictions"])
                    verifier_payload[split_name] = evaluated
                attacker_payload[verifier_family] = verifier_payload
            seed_payload[attacker_family] = attacker_payload
        per_seed_results[seed] = seed_payload

    aggregated_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for attacker_family in resolved_attacker_families:
        aggregated_metrics[attacker_family] = {}
        for verifier_family in resolved_verifier_families:
            aggregated_metrics[attacker_family][verifier_family] = {}
            for split_name in ("test_iid", "test_ood"):
                aggregated_metrics[attacker_family][verifier_family][split_name] = aggregate_seed_metrics(
                    {
                        seed: per_seed_results[seed][attacker_family][verifier_family]
                        for seed in resolved_seeds
                    },
                    split_name=split_name,
                )

    significance_raw, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            f"{attacker_family}::{split_name}": {
                verifier_family: [
                    float(per_seed_results[seed][attacker_family][verifier_family][split_name]["metrics"][PRIMARY_SIGNIFICANCE_METRIC])
                    for seed in sorted(resolved_seeds)
                ]
                for verifier_family in resolved_verifier_families
            }
            for attacker_family in resolved_attacker_families
            for split_name in ("test_iid", "test_ood")
        },
        baseline=resolved_verifier_families[0],
        metric_name=PRIMARY_SIGNIFICANCE_METRIC,
        estimand=f"seed_mean_{PRIMARY_SIGNIFICANCE_METRIC}",
    )
    significance: dict[str, dict[str, Any]] = {attacker_family: {} for attacker_family in resolved_attacker_families}
    for attacker_family in resolved_attacker_families:
        for split_name in ("test_iid", "test_ood"):
            report = significance_raw[f"{attacker_family}::{split_name}"]
            report["paired_seed_list"] = list(sorted(resolved_seeds))
            significance[attacker_family][split_name] = report

    payload = {
        "experiment_id": "exp_cross_model_transfer",
        "config": {
            "samples_per_family": int(effective_samples_per_family),
            "difficulty": float(resolved_difficulty),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "verifier_model_families": resolved_verifier_families,
        "attacker_model_families": resolved_attacker_families,
        "systems": resolved_verifier_families,
        "attacker_families": resolved_attacker_families,
        "family_pairs": _family_pairs(
            resolved_attacker_families,
            resolved_verifier_families,
        ),
        "seeds": resolved_seeds,
        "protocol": protocol,
        "blueprint_alignment": _blueprint_alignment_summary(
            resolved_verifier_families,
            resolved_attacker_families,
        ),
        "manifests": manifests,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "raw_predictions": raw_predictions,
    }

    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_cross_model_transfer.json",
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
    parser.add_argument("--verifier-model-families", nargs="*", default=None, help="Verifier model-family names.")
    parser.add_argument("--attacker-model-families", nargs="*", default=None, help="Attacker model-family names.")
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
        verifier_model_families=args.verifier_model_families,
        attacker_model_families=args.attacker_model_families,
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
