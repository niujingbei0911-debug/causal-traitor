"""Phase 4 leakage study comparing clean and oracle-leaking partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

from agents.tool_executor import ToolExecutor
from benchmark.generator import BenchmarkSample
from benchmark.schema import PublicCausalInstance
from experiments.benchmark_harness import (
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    aggregate_seed_metrics,
    align_prediction_records,
    build_seed_benchmark_run,
    manifest_metadata,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    summarize_protocol_compliance,
    write_artifacts,
)
from evaluation.reporting import compare_prediction_groups
from evaluation.scorer import Scorer
from evaluation.significance import holm_bonferroni
from verifier.assumption_ledger import build_assumption_ledger
from verifier.claim_parser import parse_claim
from verifier.decision import VerifierDecision
from verifier.pipeline import VerifierPipeline

SYSTEMS: tuple[str, str] = ("countermodel_grounded", "oracle_leaking_partition")
DEFAULT_SAMPLES_PER_FAMILY = 60
LEAKAGE_ALPHA = 0.05
PAIRWISE_RESAMPLES = 2000
LEAKAGE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "verdict_accuracy", True),
    ("macro_f1", "macro_f1", True),
    ("invalid_claim_acceptance_rate", "invalid_claim_acceptance_rate", False),
    ("unidentifiable_awareness", "unidentifiable_awareness", True),
    ("ece", "ece", False),
    ("brier", "brier", False),
)


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


def _serialize_verifier_decision(decision: VerifierDecision) -> dict[str, Any]:
    payload = decision.to_dict()
    payload["label"] = decision.label.value
    payload["confidence"] = float(decision.confidence)
    return payload


def _verifier_tool_context(
    sample: BenchmarkSample,
    public: PublicCausalInstance,
) -> dict[str, Any]:
    notes = public.metadata.get("notes")
    return {
        "treatment": sample.claim.target_variables["treatment"],
        "outcome": sample.claim.target_variables["outcome"],
        "proxy_variables": list(getattr(public, "proxy_variables", [])),
        "selection_variables": list(getattr(public, "selection_variables", [])),
        "selection_mechanism": getattr(public, "selection_mechanism", None),
        "claim_stance": "pro_causal",
        "notes": json.dumps(notes, ensure_ascii=False, sort_keys=True) if notes else "",
    }


def _run_clean_partition(sample: BenchmarkSample) -> dict[str, Any]:
    public = sample.public
    tool_context = _verifier_tool_context(sample, public)
    tool_executor = ToolExecutor({})
    tool_report = tool_executor.execute_for_claim(
        scenario=public,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    decision = VerifierPipeline(tool_runner=tool_executor).run(
        sample.claim.claim_text,
        scenario=public,
        tool_context=tool_context,
    )
    return {
        "predicted_label": decision.label.value,
        "confidence": float(decision.confidence),
        "verdict": _serialize_verifier_decision(decision),
        "tool_report": {
            "selected_tools": list(tool_report["selected_tools"]),
            "claim_stance": tool_report["claim_stance"],
            "identified_issues": list(tool_report["identified_issues"]),
            "supporting_evidence": list(tool_report["supporting_evidence"]),
            "counter_evidence": list(tool_report["counter_evidence"]),
            "tool_trace": list(tool_report["tool_trace"]),
        },
        "countermodel_found": decision.countermodel_witness is not None,
        "countermodel_type": (
            decision.countermodel_witness.payload.get("countermodel_type")
            if decision.countermodel_witness is not None
            else None
        ),
        "supports_public_only": True,
        "system_notes": ["clean_public_partition"],
        "public_scenario": public,
    }


def _oracle_assumption_hint(sample: BenchmarkSample) -> dict[str, Any]:
    parsed_claim = parse_claim(sample.claim.claim_text)
    ledger = build_assumption_ledger(parsed_claim)
    all_assumptions = [entry.name for entry in ledger.entries]
    explicit_assumptions = [entry.name for entry in ledger.entries if entry.source == "claim explicit"]
    baseline_assumptions = [
        assumption
        for assumption in all_assumptions
        if assumption in {"consistency", "positivity"}
    ]
    gold_verdict = sample.claim.gold_label.value
    if gold_verdict == "valid":
        supported = all_assumptions
        contradicted: list[str] = []
    elif gold_verdict == "invalid":
        supported = []
        contradicted = explicit_assumptions or all_assumptions
    else:
        supported = baseline_assumptions
        contradicted = []
    return {
        "gold_verdict": gold_verdict,
        "supported_assumptions": supported,
        "contradicted_assumptions": contradicted,
    }


def _oracle_measurement_key(sample: BenchmarkSample, public: PublicCausalInstance) -> str:
    for candidate in (
        sample.claim.target_variables.get("treatment"),
        sample.claim.target_variables.get("outcome"),
        *list(public.variables),
    ):
        normalized = str(candidate or "").strip()
        if normalized and normalized in public.variables:
            return normalized
    raise ValueError("Oracle-leaking partition requires at least one public variable.")


def _build_oracle_leaking_public_partition(sample: BenchmarkSample) -> PublicCausalInstance:
    public = sample.public
    hint = _oracle_assumption_hint(sample)
    measurement_semantics = dict(public.metadata.get("measurement_semantics", {}))
    measurement_key = _oracle_measurement_key(sample, public)
    measurement_entry = dict(measurement_semantics.get(measurement_key, {}))
    measurement_entry["supports_assumptions"] = sorted(
        {
            *list(measurement_entry.get("supports_assumptions", [])),
            *list(hint["supported_assumptions"]),
        }
    )
    measurement_entry["contradicts_assumptions"] = sorted(
        {
            *list(measurement_entry.get("contradicts_assumptions", [])),
            *list(hint["contradicted_assumptions"]),
        }
    )
    measurement_semantics[measurement_key] = measurement_entry

    raw_notes = public.metadata.get("notes")
    notes = dict(raw_notes) if isinstance(raw_notes, dict) else {}
    if raw_notes is not None and not isinstance(raw_notes, dict):
        notes["legacy_notes"] = str(raw_notes)
    notes["oracle_partition_hint"] = {
        **hint,
        "leakage_channels": ["notes.oracle_partition_hint", "measurement_semantics"],
        "control_interpretation": "oracle_leaking_public_partition",
    }

    metadata = dict(public.metadata)
    metadata["measurement_semantics"] = measurement_semantics
    metadata["notes"] = notes
    return PublicCausalInstance(
        scenario_id=public.scenario_id,
        description=public.description,
        variables=list(public.variables),
        proxy_variables=list(public.proxy_variables),
        selection_variables=list(public.selection_variables),
        selection_mechanism=public.selection_mechanism,
        observed_data=public.observed_data.copy(deep=True),
        data=public.data.copy(deep=True),
        causal_level=public.causal_level,
        difficulty=public.difficulty,
        difficulty_config=dict(public.difficulty_config),
        metadata=metadata,
    )


def _run_oracle_leaking_partition(sample: BenchmarkSample) -> dict[str, Any]:
    public = _build_oracle_leaking_public_partition(sample)
    tool_context = _verifier_tool_context(sample, public)
    tool_executor = ToolExecutor({})
    tool_report = tool_executor.execute_for_claim(
        scenario=public,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    decision = VerifierPipeline(tool_runner=tool_executor).run(
        sample.claim.claim_text,
        scenario=public,
        tool_context=tool_context,
    )
    verdict = _serialize_verifier_decision(decision)
    verdict["metadata"] = {
        **dict(verdict.get("metadata", {})),
        "leakage_mode": "oracle_public_partition_rerun",
        "oracle_channels": ["notes.oracle_partition_hint", "measurement_semantics"],
        "control_interpretation": "oracle_leaking_public_partition",
        "same_verifier_pipeline": True,
    }
    return {
        "predicted_label": decision.label.value,
        "confidence": float(decision.confidence),
        "verdict": verdict,
        "tool_report": {
            "selected_tools": list(tool_report["selected_tools"]),
            "claim_stance": tool_report["claim_stance"],
            "identified_issues": list(tool_report["identified_issues"]),
            "supporting_evidence": list(tool_report["supporting_evidence"]),
            "counter_evidence": list(tool_report["counter_evidence"]),
            "tool_trace": list(tool_report["tool_trace"]),
        },
        "countermodel_found": decision.countermodel_witness is not None,
        "countermodel_type": (
            decision.countermodel_witness.payload.get("countermodel_type")
            if decision.countermodel_witness is not None
            else None
        ),
        "supports_public_only": True,
        "system_notes": [
            "oracle_leaking_public_partition",
            "same_verifier_pipeline",
        ],
        "public_scenario": public,
    }


def _predict_sample(
    sample: BenchmarkSample,
    *,
    system_name: str,
) -> dict[str, Any]:
    if system_name == "countermodel_grounded":
        return _run_clean_partition(sample)
    if system_name == "oracle_leaking_partition":
        return _run_oracle_leaking_partition(sample)
    raise ValueError(f"Unsupported leakage-study system: {system_name!r}.")


def _evaluate_partition_on_samples(
    samples: list[BenchmarkSample],
    *,
    seed: int,
    split_name: str,
    system_name: str,
    ood_reasons: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    manifest_ood_reasons = dict(ood_reasons or {})

    for index, sample in enumerate(sorted(samples, key=lambda item: item.claim.instance_id), start=1):
        payload = _predict_sample(sample, system_name=system_name)
        verdict = dict(payload["verdict"])
        public = payload["public_scenario"]
        record = {
            "seed": int(seed),
            "split": split_name,
            "system_name": system_name,
            "instance_id": sample.claim.instance_id,
            "scenario_id": public.scenario_id,
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
            "proxy_variables": list(public.proxy_variables),
            "selection_variables": list(public.selection_variables),
            "selection_mechanism": public.selection_mechanism,
            "tool_report": dict(payload["tool_report"]),
            "countermodel_found": bool(payload["countermodel_found"]),
            "countermodel_type": payload["countermodel_type"],
            "verdict": verdict,
            "system_notes": list(payload["system_notes"]),
        }
        predictions.append(record)
        rounds.append(
            {
                "round_id": index,
                "gold_label": record["gold_label"],
                "verdict_label": record["predicted_label"],
                "verifier_confidence": record["confidence"],
                "countermodel_witness": verdict.get("countermodel_witness"),
            }
        )

    score = Scorer().score_game(
        {
            "game_id": f"{system_name}_{split_name}_seed_{seed}",
            "rounds": rounds,
        }
    )
    return {
        "predictions": predictions,
        "metrics": dict(score.summary["core_metrics"]),
        "appendix_metrics": dict(score.summary["appendix_metrics"]),
        "summary": dict(score.summary),
    }


def _paired_seed_bootstrap_row(
    left: list[float],
    right: list[float],
    *,
    comparison_name: str,
    model_a: str,
    model_b: str,
    metric_name: str,
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
        "alpha": float(LEAKAGE_ALPHA),
        "n_pairs": len(differences),
        "estimand": f"seed_mean_{metric_name}",
        "correction": "holm-bonferroni",
    }


def _seed_mean_accuracy_significance(
    per_seed_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seed_order = sorted(per_seed_results)
    reports: dict[str, Any] = {}
    for split_index, split_name in enumerate(OOD_SPLITS):
        clean = [
            float(per_seed_results[seed][SYSTEMS[0]][split_name]["metrics"]["verdict_accuracy"])
            for seed in seed_order
        ]
        leaking = [
            float(per_seed_results[seed][SYSTEMS[1]][split_name]["metrics"]["verdict_accuracy"])
            for seed in seed_order
        ]
        reports[split_name] = {
            "method": "paired_seed_bootstrap",
            "metric_name": "verdict_accuracy",
            "estimand": "seed_mean_verdict_accuracy",
            "alpha": float(LEAKAGE_ALPHA),
            "baseline": SYSTEMS[0],
            "comparisons": [
                _paired_seed_bootstrap_row(
                    clean,
                    leaking,
                    comparison_name=f"{SYSTEMS[0]} vs {SYSTEMS[1]}",
                    model_a=SYSTEMS[0],
                    model_b=SYSTEMS[1],
                    metric_name="verdict_accuracy",
                    random_state=91 + split_index,
                )
            ],
            "correction": "holm-bonferroni",
        }
    return reports


def _aligned_prediction_groups(
    raw_predictions: list[dict[str, Any]],
    *,
    split_name: str,
) -> tuple[list[str], dict[str, list[str]]]:
    return align_prediction_records(
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


def _mcnemar_significance(raw_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split_name in OOD_SPLITS:
        truth, predictions = _aligned_prediction_groups(raw_predictions, split_name=split_name)
        report = compare_prediction_groups(
            truth,
            predictions,
            baseline=SYSTEMS[0],
            method="mcnemar",
            metric_name="verdict_accuracy",
            correction="none",
        ).to_dict()
        report["metric_name"] = "verdict_accuracy"
        for row in report.get("comparisons", []):
            row["metric_name"] = "verdict_accuracy"
            row["correction"] = "holm-bonferroni"
        reports[split_name] = report
    return reports


def _apply_global_multiple_comparison_correction(
    significance: dict[str, Any],
    mcnemar_significance: dict[str, Any],
) -> dict[str, Any]:
    raw_p_values: dict[str, float] = {}
    for split_name in OOD_SPLITS:
        raw_p_values[f"bootstrap:{split_name}"] = float(significance[split_name]["comparisons"][0]["p_value"])
        raw_p_values[f"mcnemar:{split_name}"] = float(
            mcnemar_significance[split_name]["comparisons"][0]["p_value"]
        )

    correction_table = holm_bonferroni(raw_p_values, alpha=LEAKAGE_ALPHA) if raw_p_values else []
    correction_lookup = {entry.hypothesis: entry for entry in correction_table}
    for split_name in OOD_SPLITS:
        bootstrap_key = f"bootstrap:{split_name}"
        bootstrap_entry = correction_lookup[bootstrap_key]
        bootstrap_row = significance[split_name]["comparisons"][0]
        bootstrap_row["adjusted_p_value"] = float(bootstrap_entry.adjusted_p_value)
        bootstrap_row["reject_after_correction"] = bool(bootstrap_entry.reject)
        bootstrap_row["family_hypothesis"] = bootstrap_key

        mcnemar_key = f"mcnemar:{split_name}"
        mcnemar_entry = correction_lookup[mcnemar_key]
        mcnemar_row = mcnemar_significance[split_name]["comparisons"][0]
        mcnemar_row["adjusted_p_value"] = float(mcnemar_entry.adjusted_p_value)
        mcnemar_row["reject_after_correction"] = bool(mcnemar_entry.reject)
        mcnemar_row["family_hypothesis"] = mcnemar_key

        scope = {
            "family_size": len(raw_p_values),
            "hypotheses": list(raw_p_values),
        }
        significance[split_name]["correction_scope"] = dict(scope)
        mcnemar_significance[split_name]["correction_scope"] = dict(scope)

    return {
        "family_size": len(raw_p_values),
        "alpha": float(LEAKAGE_ALPHA),
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


def _inflation_summary(
    aggregated_metrics: dict[str, dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    clean_metrics = aggregated_metrics[SYSTEMS[0]][split_name]
    leaking_metrics = aggregated_metrics[SYSTEMS[1]][split_name]
    summary: dict[str, Any] = {}
    for alias, metric_name, higher_is_better in LEAKAGE_METRICS:
        clean = clean_metrics[metric_name]
        leaking = leaking_metrics[metric_name]
        summary[alias] = {
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "clean_mean": float(clean["mean"]),
            "clean_ci_lower": float(clean["ci_lower"]),
            "clean_ci_upper": float(clean["ci_upper"]),
            "leaking_mean": float(leaking["mean"]),
            "leaking_ci_lower": float(leaking["ci_lower"]),
            "leaking_ci_upper": float(leaking["ci_upper"]),
            "delta_mean": float(leaking["mean"] - clean["mean"]),
        }
    return summary


def _comparison_support(
    comparison: dict[str, Any] | None,
    *,
    alpha: float = LEAKAGE_ALPHA,
) -> bool:
    if comparison is None:
        return False
    adjusted_p_value = comparison.get("adjusted_p_value")
    p_value = adjusted_p_value if adjusted_p_value is not None else comparison.get("p_value")
    return p_value is not None and float(p_value) < float(alpha)


def _leakage_conclusion(
    *,
    inflation: dict[str, dict[str, Any]],
    significance: dict[str, Any],
    mcnemar_significance: dict[str, Any],
    samples_per_family: int,
) -> dict[str, Any]:
    split_details: dict[str, dict[str, Any]] = {}
    supported_splits: list[str] = []
    unsupported_splits: list[str] = []

    for split_name in OOD_SPLITS:
        bootstrap_row = significance[split_name]["comparisons"][0]
        mcnemar_row = mcnemar_significance[split_name]["comparisons"][0]
        accuracy_delta = float(inflation[split_name]["accuracy"]["delta_mean"])
        positive_inflation = accuracy_delta > 0.0
        bootstrap_significant = _comparison_support(bootstrap_row)
        mcnemar_significant = _comparison_support(mcnemar_row)
        supported = positive_inflation and bootstrap_significant and mcnemar_significant
        split_details[split_name] = {
            "accuracy_delta": accuracy_delta,
            "positive_inflation": positive_inflation,
            "bootstrap_significant": bootstrap_significant,
            "mcnemar_significant": mcnemar_significant,
            "supported": supported,
        }
        if supported:
            supported_splits.append(split_name)
        else:
            unsupported_splits.append(split_name)

    robust_support = len(supported_splits) == len(OOD_SPLITS)
    supported = bool(supported_splits)
    if robust_support:
        summary_statement = (
            "This run supports the leakage warning on every evaluated split: rerunning the verifier "
            "on an oracle-leaking public partition inflates accuracy under both paired tests after "
            "global multiple-comparison correction."
        )
    elif supported:
        summary_statement = (
            "This run partially supports the leakage warning: statistically supported accuracy inflation "
            f"appears on {', '.join(supported_splits)}, but not on {', '.join(unsupported_splits)}."
        )
    else:
        summary_statement = (
            "This run does not yet support the leakage warning under the corrected protocol: no split "
            "shows positive accuracy inflation that survives both paired tests after global correction."
        )

    recommendation = None
    if int(samples_per_family) < DEFAULT_SAMPLES_PER_FAMILY:
        recommendation = (
            f"Increase samples_per_family to at least {DEFAULT_SAMPLES_PER_FAMILY} before citing "
            "this experiment as leakage evidence."
        )

    return {
        "alpha": float(LEAKAGE_ALPHA),
        "supported": supported,
        "robust_support": robust_support,
        "supported_splits": supported_splits,
        "unsupported_splits": unsupported_splits,
        "split_details": split_details,
        "summary_statement": summary_statement,
        "recommendation": recommendation,
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    protocol = payload.get("protocol", {})
    lines = [
        "# Leakage Study",
        "",
        "This experiment contrasts the clean public partition against an oracle-leaking public partition rerun.",
        "",
        "Both conditions keep verifier inputs public-only. The leaking control exposes benchmark answers through visible public metadata and reruns the same verifier stages instead of copying `gold_label` into predictions.",
        "",
        "## Accuracy Test",
        "",
        "Paired bootstrap uses the seed-mean accuracy estimand; McNemar uses aligned sample-level disagreements. Holm-Bonferroni is applied jointly across both splits and both tests.",
        "",
        "| Split | Clean Accuracy | Leaking Accuracy | Delta | Bootstrap p | McNemar p | Supports Warning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
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
    for split_name in OOD_SPLITS:
        delta = payload["inflation"][split_name]["accuracy"]
        bootstrap_row = payload["significance"][split_name]["comparisons"][0]
        mcnemar_row = payload["mcnemar_significance"][split_name]["comparisons"][0]
        support = payload["conclusion"]["split_details"][split_name]["supported"]
        lines.append(
            f"| {split_name} | {delta['clean_mean']:.4f} | {delta['leaking_mean']:.4f} | "
            f"{delta['delta_mean']:.4f} | {float(bootstrap_row['adjusted_p_value']):.4f} | "
            f"{float(mcnemar_row['adjusted_p_value']):.4f} | {'yes' if support else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Metric Inflation",
            "",
            "All mean values below report seed-mean estimates with 95% CI.",
            "",
        ]
    )
    for split_name in OOD_SPLITS:
        lines.extend(
            [
                f"### {split_name}",
                "",
                "| Metric | Clean (95% CI) | Leaking (95% CI) | Delta | Favors Leaking |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for alias, _, higher_is_better in LEAKAGE_METRICS:
            label = {
                "accuracy": "Accuracy",
                "macro_f1": "Macro-F1",
                "invalid_claim_acceptance_rate": "Invalid Acceptance",
                "unidentifiable_awareness": "Unidentifiable Awareness",
                "ece": "ECE",
                "brier": "Brier",
            }[alias]
            item = payload["inflation"][split_name][alias]
            delta = float(item["delta_mean"])
            favors_leaking = delta > 0.0 if higher_is_better else delta < 0.0
            lines.append(
                f"| {label} | {item['clean_mean']:.4f} [{item['clean_ci_lower']:.4f}, {item['clean_ci_upper']:.4f}] | "
                f"{item['leaking_mean']:.4f} [{item['leaking_ci_lower']:.4f}, {item['leaking_ci_upper']:.4f}] | "
                f"{delta:.4f} | {'yes' if favors_leaking else 'no'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Leakage Warning",
            "",
            payload["conclusion"]["summary_statement"],
        ]
    )
    if payload["conclusion"]["recommendation"]:
        lines.extend(["", payload["conclusion"]["recommendation"]])
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
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
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    protocol = summarize_protocol_compliance(
        resolved_seeds,
        minimum_count=MIN_FORMAL_SEED_COUNT,
        minimum_samples_per_family=DEFAULT_SAMPLES_PER_FAMILY,
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
                evaluated = _evaluate_partition_on_samples(
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

    significance = _seed_mean_accuracy_significance(per_seed_results)
    mcnemar_significance = _mcnemar_significance(raw_predictions)
    global_multiple_comparison_correction = _apply_global_multiple_comparison_correction(
        significance,
        mcnemar_significance,
    )
    inflation = {
        split_name: _inflation_summary(aggregated_metrics, split_name)
        for split_name in OOD_SPLITS
    }
    conclusion = _leakage_conclusion(
        inflation=inflation,
        significance=significance,
        mcnemar_significance=mcnemar_significance,
        samples_per_family=int(resolved_samples_per_family),
    )

    payload = {
        "experiment_id": "exp_leakage_study",
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
        "inflation": inflation,
        "significance": significance,
        "mcnemar_significance": mcnemar_significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "conclusion": conclusion,
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
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["inflation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
