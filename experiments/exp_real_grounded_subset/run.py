"""Phase 4 real-grounded subset experiment with synthetic vs real-grounded reporting."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from benchmark.loaders import load_real_grounded_dataset, load_real_grounded_samples, save_real_grounded_dataset
from benchmark.real_grounded import RealGroundedCase, RealGroundedDataset, SourceCitation
from benchmark.schema import ClaimInstance
from experiments.benchmark_harness import (
    MAIN_BENCHMARK_SYSTEMS,
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    build_seed_benchmark_run,
    build_seed_metric_significance,
    evaluate_system_on_samples,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    validate_system_names,
    write_artifacts,
)

PRIMARY_SIGNIFICANCE_METRIC = "unsafe_acceptance_rate"


def _synthetic_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        split_name: payload[split_name]
        for split_name in OOD_SPLITS
    }


def _toy_real_grounded_dataset() -> RealGroundedDataset:
    cases = [
        RealGroundedCase(
            case_id="rg_policy_training_001",
            grounding_type="literature_grounded",
            claim=ClaimInstance(
                instance_id="rg_claim_policy_001",
                causal_level="L2",
                graph_family="real_grounded_policy_case",
                language_template_id="real_grounded::policy::001",
                observed_variables=["program_uptake", "baseline_need", "employment_gain"],
                claim_text="Increasing program_uptake improves employment_gain.",
                query_type="average_treatment_effect",
                target_variables={"treatment": "program_uptake", "outcome": "employment_gain"},
                gold_label="valid",
                gold_answer="The grounded policy case treats the effect as identified under the public assumptions.",
                gold_assumptions=["Conditional ignorability", "Positivity"],
            ),
            source_citation=SourceCitation(
                citation_text="Smith and Lee (2024), Journal of Policy Evaluation.",
                title="Employment effects of targeted training programs",
                year=2024,
            ),
            public_evidence_summary="Public summary of an observational policy evaluation with measured covariates.",
            information_contract={
                "visible_information": ["Observed covariate-adjusted estimates", "Public treatment and outcome definitions"],
                "hidden_information": ["Reviewer-only sensitivity analysis details"],
            },
            identifying_assumptions=["Conditional ignorability", "Stable outcome definition"],
            witness_note="The study reports robustness checks but still relies on observational assumptions.",
        ),
        RealGroundedCase(
            case_id="rg_medicine_case_002",
            grounding_type="semi_real",
            claim=ClaimInstance(
                instance_id="rg_claim_medicine_002",
                causal_level="L2",
                graph_family="real_grounded_medicine_case",
                language_template_id="real_grounded::medicine::002",
                observed_variables=["therapy_flag", "baseline_risk", "recovery_score"],
                claim_text="Therapy_flag improves recovery_score in the observed cohort.",
                query_type="average_treatment_effect",
                target_variables={"treatment": "therapy_flag", "outcome": "recovery_score"},
                gold_label="unidentifiable",
                gold_answer="The semi-real clinical case remains underdetermined under public evidence.",
                gold_assumptions=["No hidden selection effects"],
            ),
            source_citation=SourceCitation(
                citation_text="Garcia (2023), Observational Medicine Review.",
                title="Semi-real treatment effect casebook",
                year=2023,
            ),
            public_evidence_summary="Semi-real observational case with documented public evidence and restricted annotations.",
            information_contract={
                "visible_information": ["Public cohort table", "Outcome definition"],
                "hidden_information": ["Private adjudication memo"],
            },
            identifying_assumptions=["No hidden outcome misclassification"],
            witness_note="The semi-real record includes an auditor note about residual uncertainty.",
        ),
    ]
    return RealGroundedDataset(
        dataset_name="real_grounded_subset",
        version="2026-04-21",
        cases=cases,
        metadata={"domains": ["policy", "observational_medicine"]},
    )


def _build_real_grounded_source() -> tuple[RealGroundedDataset, str]:
    dataset = _toy_real_grounded_dataset()
    temp_dir = tempfile.TemporaryDirectory()
    dataset_path = Path(temp_dir.name) / "real_grounded_subset.json"
    save_real_grounded_dataset(dataset, dataset_path)
    return load_real_grounded_dataset(dataset_path), temp_dir.name


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Real-Grounded Subset",
        "",
        "## Setup",
        "",
        f"- Systems: {', '.join(payload['systems'])}",
        f"- Seeds: {payload['seeds']}",
        f"- Samples per family: {payload['config']['samples_per_family']}",
        f"- Difficulty: {payload['config']['difficulty']:.2f}",
        f"- Dataset Partitions: {', '.join(payload['dataset_partitions'])}",
        "",
    ]
    for dataset_partition, partition_payload in payload["aggregated_metrics"].items():
        lines.extend(
            [
                f"## {dataset_partition}",
                "",
                "| System | Split | Verdict Acc. | Unsafe Accept | Wise Refusal Recall | Wise Refusal Precision | Over-Refusal | ECE | Brier |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for system_name, split_payload in partition_payload.items():
            for split_name in OOD_SPLITS:
                metrics = split_payload[split_name]
                lines.append(
                    f"| {system_name} | {split_name} | "
                    f"{metrics['verdict_accuracy']['formatted']} | "
                    f"{metrics['unsafe_acceptance_rate']['formatted']} | "
                    f"{metrics['wise_refusal_recall']['formatted']} | "
                    f"{metrics['wise_refusal_precision']['formatted']} | "
                    f"{metrics['over_refusal_rate']['formatted']} | "
                    f"{metrics['ece']['formatted']} | "
                    f"{metrics['brier']['formatted']} |"
                )
        lines.append("")
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
    systems: list[str] | tuple[str, ...] | None = None,
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
    resolved_systems = validate_system_names(
        systems,
        default_systems=MAIN_BENCHMARK_SYSTEMS,
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

    real_grounded_dataset, temp_dir_name = _build_real_grounded_source()
    try:
        real_grounded_samples = load_real_grounded_samples(real_grounded_dataset)
        raw_predictions: list[dict[str, Any]] = []
        per_partition_results: dict[str, Any] = {
            "synthetic": {},
            "real_grounded": {},
        }

        for seed in resolved_seeds:
            synthetic_run = build_seed_benchmark_run(
                seed=seed,
                difficulty=resolved_difficulty,
                samples_per_family=resolved_samples_per_family,
            )
            synthetic_ood_reasons = dict(synthetic_run.manifest.metadata.get("ood_reasons", {}))
            synthetic_seed_payload: dict[str, Any] = {}
            real_grounded_seed_payload: dict[str, Any] = {}

            for system_name in resolved_systems:
                synthetic_system_payload: dict[str, Any] = {}
                real_grounded_system_payload: dict[str, Any] = {}
                for split_name in OOD_SPLITS:
                    synthetic_evaluated = evaluate_system_on_samples(
                        synthetic_run.split_samples[split_name],
                        seed=seed,
                        split_name=split_name,
                        system_name=system_name,
                        ood_reasons=synthetic_ood_reasons,
                    )
                    for record in synthetic_evaluated["predictions"]:
                        record["dataset_partition"] = "synthetic"
                    raw_predictions.extend(synthetic_evaluated["predictions"])
                    synthetic_system_payload[split_name] = synthetic_evaluated

                    real_grounded_evaluated = evaluate_system_on_samples(
                        real_grounded_samples,
                        seed=seed,
                        split_name=split_name,
                        system_name=system_name,
                    )
                    for record in real_grounded_evaluated["predictions"]:
                        record["dataset_partition"] = "real_grounded"
                    raw_predictions.extend(real_grounded_evaluated["predictions"])
                    real_grounded_system_payload[split_name] = real_grounded_evaluated

                synthetic_seed_payload[system_name] = synthetic_system_payload
                real_grounded_seed_payload[system_name] = real_grounded_system_payload

            per_partition_results["synthetic"][seed] = synthetic_seed_payload
            per_partition_results["real_grounded"][seed] = real_grounded_seed_payload

        aggregated_metrics: dict[str, Any] = {}
        for dataset_partition, per_seed_payload in per_partition_results.items():
            aggregated_metrics[dataset_partition] = {}
            for system_name in resolved_systems:
                aggregated_metrics[dataset_partition][system_name] = {}
                for split_name in OOD_SPLITS:
                    aggregated_metrics[dataset_partition][system_name][split_name] = aggregate_seed_metrics(
                        {
                            seed: per_seed_payload[seed][system_name]
                            for seed in resolved_seeds
                        },
                        split_name=split_name,
                    )

        significance: dict[str, Any] = {}
        for dataset_partition, per_seed_payload in per_partition_results.items():
            if len(resolved_systems) < 2:
                significance[dataset_partition] = {split_name: None for split_name in OOD_SPLITS}
                continue
            significance[dataset_partition], _ = build_seed_metric_significance(
                {
                    split_name: {
                        system_name: [
                            float(per_seed_payload[seed][system_name][split_name]["metrics"][PRIMARY_SIGNIFICANCE_METRIC])
                            for seed in sorted(resolved_seeds)
                        ]
                        for system_name in resolved_systems
                    }
                    for split_name in OOD_SPLITS
                },
                baseline=resolved_systems[0],
                metric_name=PRIMARY_SIGNIFICANCE_METRIC,
                estimand=f"seed_mean_{PRIMARY_SIGNIFICANCE_METRIC}",
            )

        payload = {
            "experiment_id": "exp_real_grounded_subset",
            "config": {
                "samples_per_family": int(resolved_samples_per_family),
                "difficulty": float(resolved_difficulty),
                "dataset_partitions": ["synthetic", "real_grounded"],
            },
            "requested_config": {
                "systems": list(systems) if systems is not None else list(MAIN_BENCHMARK_SYSTEMS),
                "samples_per_family": int(samples_per_family),
                "difficulty": float(difficulty),
                "allow_protocol_violations": bool(allow_protocol_violations),
            },
            "systems": resolved_systems,
            "dataset_partitions": ["synthetic", "real_grounded"],
            "seeds": resolved_seeds,
            "protocol": protocol,
            "real_grounded_dataset": real_grounded_dataset.to_dict(),
            "per_partition_results": per_partition_results,
            "aggregated_metrics": aggregated_metrics,
            "significance": significance,
            "raw_predictions": raw_predictions,
        }

        summary = _markdown_summary(payload)
        artifacts = write_artifacts(
            output_path=output_path or "outputs/mainline/exp_real_grounded_subset.json",
            payload=payload,
            markdown_summary=summary,
        )
        payload["markdown_summary"] = summary
        payload["artifacts"] = artifacts
        Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        Path(temp_dir_name).unlink(missing_ok=True) if False else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 real-grounded subset experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
    parser.add_argument("--systems", nargs="*", default=None, help="Optional system matrix to evaluate.")
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=MIN_FORMAL_SAMPLES_PER_FAMILY,
        help="Synthetic samples generated per benchmark family.",
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
