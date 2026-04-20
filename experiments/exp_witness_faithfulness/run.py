"""Phase 4 witness faithfulness experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any

from agents.tool_executor import ToolExecutor
from benchmark.generator import BenchmarkSample
from evaluation.reporting import summarize_metrics
from experiments.benchmark_harness import (
    MIN_FORMAL_SAMPLES_PER_FAMILY,
    MIN_FORMAL_SEED_COUNT,
    OOD_SPLITS,
    _coerce_public_instance,
    _sample_transcript,
    _verifier_tool_context,
    build_seed_benchmark_run,
    build_seed_metric_significance,
    normalize_benchmark_difficulty,
    normalize_benchmark_samples_per_family,
    normalize_experiment_seeds,
    predict_sample,
    score_prediction_records,
    summarize_protocol_compliance,
    write_artifacts,
)
from verifier.assumption_ledger import AssumptionLedger, build_assumption_ledger
from verifier.countermodel_search import CountermodelSearchResult, search_countermodels
from verifier.decision import decide_verdict

SYSTEM_NAME = "countermodel_grounded"
WITNESS_CONDITIONS: tuple[str, ...] = (
    "original",
    "drop_witness",
    "corrupt_witness",
    "shuffle_witness",
)


@dataclass(slots=True)
class EvidenceBundle:
    sample: BenchmarkSample
    base_prediction: dict[str, Any]
    parsed_claim: Any
    ledger: AssumptionLedger
    countermodel_result: CountermodelSearchResult
    base_tool_trace: list[dict[str, Any]]
    primary_witness_role: str
    primary_witness_type: str


def _primary_witness(verdict: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    for role, key in (
        ("countermodel", "countermodel_witness"),
        ("support", "support_witness"),
        ("generic", "witness"),
    ):
        witness = verdict.get(key)
        if isinstance(witness, dict):
            return dict(witness), role
    return None, "none"


def _witness_text(witness: dict[str, Any] | None) -> str:
    if not witness:
        return ""
    parts: list[str] = [
        str(witness.get("description", "")),
        " ".join(str(item) for item in witness.get("evidence", []) if item is not None),
        " ".join(str(item) for item in witness.get("assumptions", []) if item is not None),
        json.dumps(witness.get("payload", {}), sort_keys=True, default=str),
    ]
    return " ".join(part for part in parts if part).lower()


def _reasoning_text(verdict: dict[str, Any]) -> str:
    return str(verdict.get("reasoning_summary", "")).lower()


def _keyword_overlap_score(witness: dict[str, Any] | None, verdict: dict[str, Any]) -> float:
    witness_tokens = {
        token.strip(".,:;()[]{}")
        for token in _witness_text(witness).split()
        if len(token.strip(".,:;()[]{}")) >= 5
    }
    reasoning_tokens = {
        token.strip(".,:;()[]{}")
        for token in _reasoning_text(verdict).split()
        if len(token.strip(".,:;()[]{}")) >= 5
    }
    if not witness_tokens or not reasoning_tokens:
        return 0.0
    return len(witness_tokens & reasoning_tokens) / max(1, len(witness_tokens))


def _verdict_alignment_score(witness: dict[str, Any] | None, verdict: dict[str, Any]) -> float:
    if not witness:
        return 0.0
    suggestion = str(witness.get("verdict_suggestion", "")).strip().lower()
    label = str(verdict.get("label", verdict.get("final_verdict", ""))).strip().lower()
    if not suggestion:
        return 0.5
    return 1.0 if suggestion == label else 0.0


def _evidence_presence_score(witness: dict[str, Any] | None) -> float:
    if not witness:
        return 0.0
    evidence = witness.get("evidence", [])
    payload = witness.get("payload", {})
    assumptions = witness.get("assumptions", [])
    score = 0.0
    if isinstance(evidence, list) and evidence:
        score += 0.4
    if isinstance(payload, dict) and payload:
        score += 0.4
    if isinstance(assumptions, list) and assumptions:
        score += 0.2
    return min(1.0, score)


def _witness_faithfulness_score(witness: dict[str, Any] | None, verdict: dict[str, Any]) -> float:
    overlap = _keyword_overlap_score(witness, verdict)
    alignment = _verdict_alignment_score(witness, verdict)
    evidence = _evidence_presence_score(witness)
    return round((0.4 * overlap) + (0.35 * alignment) + (0.25 * evidence), 6)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clone_ledger(ledger: AssumptionLedger) -> AssumptionLedger:
    return AssumptionLedger([entry.to_dict() for entry in ledger.entries])


def _countermodel_constructor_payload(value: CountermodelSearchResult | dict[str, Any]) -> dict[str, Any]:
    payload = value.to_dict() if isinstance(value, CountermodelSearchResult) else dict(value)
    return {
        "found_countermodel": bool(payload.get("found_countermodel", False)),
        "countermodel_type": payload.get("countermodel_type"),
        "observational_match_score": float(payload.get("observational_match_score", 0.0)),
        "query_disagreement": bool(payload.get("query_disagreement", False)),
        "countermodel_explanation": str(payload.get("countermodel_explanation", "")),
        "verdict_suggestion": payload.get("verdict_suggestion"),
        "candidates": list(payload.get("candidates", [])),
        "used_observed_data": bool(payload.get("used_observed_data", False)),
    }


def _clone_countermodel_result(result: CountermodelSearchResult) -> CountermodelSearchResult:
    return CountermodelSearchResult(**_countermodel_constructor_payload(result))


def _empty_countermodel_result() -> CountermodelSearchResult:
    return CountermodelSearchResult(found_countermodel=False, candidates=[])


def _tool_report_from_trace(tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    selected_tools = [
        str(record.get("tool_name") or record.get("tool") or "tool")
        for record in tool_trace
    ]
    supporting_evidence = [
        str(record.get("summary"))
        for record in tool_trace
        if record.get("supports_primary_claim") or record.get("supports_claim")
    ]
    counter_evidence = [
        str(record.get("summary"))
        for record in tool_trace
        if record.get("contradicts_assumptions")
        or str(record.get("evidence_direction", "")).strip().lower() == "counter"
    ]
    return {
        "selected_tools": selected_tools,
        "claim_stance": "pro_causal",
        "identified_issues": [],
        "supporting_evidence": supporting_evidence,
        "counter_evidence": counter_evidence,
        "tool_trace": [dict(record) for record in tool_trace],
    }


def _build_evidence_bundle(sample: BenchmarkSample) -> EvidenceBundle:
    base_prediction = predict_sample(sample, system_name=SYSTEM_NAME)
    scenario = _coerce_public_instance(sample)
    transcript = _sample_transcript(sample)
    tool_context = _verifier_tool_context(sample)
    parsed_claim = tool_context["_parsed_claim"]
    ledger = build_assumption_ledger(parsed_claim)
    countermodel_result = search_countermodels(
        parsed_claim,
        ledger,
        scenario=scenario,
        context={
            **tool_context,
            "public_instance": scenario,
            "observed_data": scenario.observed_data.copy(deep=True),
        },
    )
    tool_report = ToolExecutor({}).execute_for_claim(
        scenario=scenario,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    witness, role = _primary_witness(base_prediction["verdict"])
    return EvidenceBundle(
        sample=sample,
        base_prediction=base_prediction,
        parsed_claim=parsed_claim,
        ledger=ledger,
        countermodel_result=countermodel_result,
        base_tool_trace=[dict(record) for record in tool_report["tool_trace"]],
        primary_witness_role=role,
        primary_witness_type=str((witness or {}).get("witness_type", "none")),
    )


def _select_donor_bundle(
    bundle: EvidenceBundle,
    *,
    donor_samples: list[BenchmarkSample],
    bundle_cache: dict[str, EvidenceBundle],
) -> EvidenceBundle | None:
    ordered_ids = [
        sample.claim.instance_id
        for sample in sorted(donor_samples, key=lambda item: item.claim.instance_id)
    ]
    if len(ordered_ids) <= 1:
        return None
    target_id = bundle.sample.claim.instance_id
    target_index = ordered_ids.index(target_id)
    rotated = ordered_ids[target_index + 1:] + ordered_ids[:target_index]
    prioritized = [
        candidate_id
        for candidate_id in rotated
        if bundle_cache[candidate_id].primary_witness_type == bundle.primary_witness_type
    ] or rotated
    return bundle_cache[prioritized[0]]


def _corrupt_countermodel_result(result: CountermodelSearchResult) -> CountermodelSearchResult:
    payload = _countermodel_constructor_payload(result)
    payload["countermodel_type"] = f"corrupted::{payload.get('countermodel_type') or 'countermodel'}"
    payload["countermodel_explanation"] = (
        "Corrupted witness replay injects mismatched countermodel evidence into the decision path."
    )
    if str(payload.get("verdict_suggestion")) == "invalid":
        payload["verdict_suggestion"] = "unidentifiable"
        payload["query_disagreement"] = True
    else:
        payload["verdict_suggestion"] = "invalid"
        payload["query_disagreement"] = False
    for candidate in payload.get("candidates", []):
        candidate["countermodel_type"] = payload["countermodel_type"]
        candidate["countermodel_explanation"] = payload["countermodel_explanation"]
        candidate["verdict_suggestion"] = payload["verdict_suggestion"]
        candidate["query_disagreement"] = payload["query_disagreement"]
    return CountermodelSearchResult(**payload)


def _corrupt_tool_trace(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    assumption_names = [entry.name for entry in bundle.ledger.entries] or ["consistency", "positivity"]
    source_trace = bundle.base_tool_trace or [{}]
    corrupted: list[dict[str, Any]] = []
    for index, record in enumerate(source_trace[:4], start=1):
        mutated = deepcopy(record)
        mutated["supports_claim"] = False
        mutated["supports_primary_claim"] = False
        mutated["claim_stance"] = "anti_causal"
        mutated["evidence_direction"] = "counter"
        mutated["supports_assumptions"] = []
        mutated["contradicts_assumptions"] = assumption_names[:2]
        mutated["summary"] = (
            f"corrupted_witness_trace_{index}: evidence now contradicts {', '.join(assumption_names[:2])}"
        )
        corrupted.append(mutated)
    return corrupted


def _corrupt_supportive_ledger(bundle: EvidenceBundle) -> AssumptionLedger:
    if bundle.ledger.entries:
        return AssumptionLedger(
            [
                {
                    **entry.to_dict(),
                    "status": "supported",
                    "note": "Corrupted replay force-labels this assumption as supported.",
                }
                for entry in bundle.ledger.entries
            ]
        )
    return AssumptionLedger(
        [
            {
                "name": "consistency",
                "source": "corrupted replay",
                "category": "identification",
                "status": "supported",
                "note": "Corrupted replay synthesizes support.",
            },
            {
                "name": "positivity",
                "source": "corrupted replay",
                "category": "identification",
                "status": "supported",
                "note": "Corrupted replay synthesizes support.",
            },
        ]
    )


def _corrupted_support_trace(ledger: AssumptionLedger) -> list[dict[str, Any]]:
    assumptions = [entry.name for entry in ledger.entries] or ["consistency", "positivity"]
    return [
        {
            "tool_name": "corrupted_support_trace",
            "status": "success",
            "summary": "Corrupted replay injects synthetic support for the claim.",
            "supports_claim": True,
            "supports_primary_claim": True,
            "claim_stance": "pro_causal",
            "evidence_direction": "support",
            "error": "",
            "supports_assumptions": assumptions[:4],
            "contradicts_assumptions": [],
            "confidence": 0.9,
        }
    ]


def _condition_components(
    bundle: EvidenceBundle,
    *,
    condition: str,
    donor_bundle: EvidenceBundle | None,
) -> tuple[AssumptionLedger, CountermodelSearchResult, list[dict[str, Any]], dict[str, Any]]:
    if condition == "drop_witness":
        if bundle.primary_witness_type == "countermodel":
            return (
                _clone_ledger(bundle.ledger),
                _empty_countermodel_result(),
                [dict(record) for record in bundle.base_tool_trace],
                {"replay_mode": "drop_countermodel_channel"},
            )
        if bundle.primary_witness_type == "support":
            return (
                _clone_ledger(bundle.ledger),
                _empty_countermodel_result(),
                [],
                {"replay_mode": "drop_tool_support_channel"},
            )
        if bundle.primary_witness_type == "assumption":
            return (
                AssumptionLedger([]),
                _empty_countermodel_result(),
                [dict(record) for record in bundle.base_tool_trace],
                {"replay_mode": "drop_assumption_ledger"},
            )
        return (
            AssumptionLedger([]),
            _empty_countermodel_result(),
            [],
            {"replay_mode": "drop_all_witness_channels"},
        )
    if condition == "corrupt_witness":
        if bundle.primary_witness_type == "countermodel":
            return (
                _clone_ledger(bundle.ledger),
                _corrupt_countermodel_result(bundle.countermodel_result),
                [],
                {"replay_mode": "corrupt_countermodel_channel"},
            )
        if bundle.primary_witness_type == "support":
            return (
                _clone_ledger(bundle.ledger),
                _empty_countermodel_result(),
                _corrupt_tool_trace(bundle),
                {"replay_mode": "corrupt_tool_support_channel"},
            )
        corrupted_ledger = _corrupt_supportive_ledger(bundle)
        return (
            corrupted_ledger,
            _empty_countermodel_result(),
            _corrupted_support_trace(corrupted_ledger),
            {"replay_mode": "corrupt_assumption_channel"},
        )
    if condition == "shuffle_witness":
        donor = donor_bundle or bundle
        return (
            _clone_ledger(donor.ledger),
            _clone_countermodel_result(donor.countermodel_result),
            [] if donor.countermodel_result.found_countermodel else [dict(record) for record in donor.base_tool_trace],
            {
                "replay_mode": "shuffle_evidence_bundle",
                "donor_instance_id": donor.sample.claim.instance_id,
                "donor_witness_type": donor.primary_witness_type,
            },
        )
    raise ValueError(f"Unsupported witness condition: {condition!r}.")


def _decision_payload_from_components(
    bundle: EvidenceBundle,
    *,
    condition: str,
    ledger: AssumptionLedger,
    countermodel_result: CountermodelSearchResult,
    tool_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = decide_verdict(
        bundle.parsed_claim,
        ledger,
        countermodel_result,
        tool_trace=tool_trace,
    )
    verdict = decision.to_dict()
    tool_report = _tool_report_from_trace(tool_trace)
    return {
        "predicted_label": decision.label.value,
        "confidence": float(decision.confidence),
        "verdict": verdict,
        "tool_report": tool_report,
        "countermodel_found": decision.countermodel_witness is not None,
        "countermodel_type": (
            decision.countermodel_witness.payload.get("countermodel_type")
            if decision.countermodel_witness is not None
            else None
        ),
        "supports_public_only": True,
        "system_notes": [f"witness_condition_replay:{condition}"],
    }


def _build_prediction_record(
    sample: BenchmarkSample,
    payload: dict[str, Any],
    *,
    seed: int,
    split_name: str,
    witness_condition: str,
    base_prediction: dict[str, Any],
    donor_bundle: EvidenceBundle | None = None,
    condition_metadata: dict[str, Any] | None = None,
    ood_reasons: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    base_verdict = dict(base_prediction["verdict"])
    conditioned_verdict = dict(payload["verdict"])
    conditioned_witness, conditioned_role = _primary_witness(conditioned_verdict)
    base_witness, base_role = _primary_witness(base_verdict)
    public_payload = sample.public.to_dict()
    tool_report = dict(payload.get("tool_report") or _tool_report_from_trace(payload["verdict"].get("tool_trace", [])))
    conditioned_label = str(payload["predicted_label"])
    base_label = str(base_prediction["predicted_label"])
    return {
        "seed": int(seed),
        "split": split_name,
        "system_name": SYSTEM_NAME,
        "instance_id": sample.claim.instance_id,
        "scenario_id": public_payload["scenario_id"],
        "causal_level": public_payload["causal_level"],
        "graph_family": sample.claim.graph_family,
        "language_template_id": sample.claim.language_template_id,
        "query_type": sample.claim.query_type,
        "attack_name": sample.claim.meta.get("attack_name"),
        "style_id": sample.claim.meta.get("style_id"),
        "lexical_template_id": sample.claim.meta.get(
            "lexical_template_id",
            sample.claim.language_template_id,
        ),
        "persuasion_style_id": sample.claim.meta.get("persuasion_style_id"),
        "pressure_type": sample.claim.meta.get("pressure_type"),
        "mechanism_ood_tag": sample.claim.meta.get("mechanism_ood_tag"),
        "context_shift_group": sample.claim.meta.get("context_shift_group"),
        "paired_flip_id": sample.claim.meta.get("paired_flip_id"),
        "claim_mode": sample.claim.meta.get("claim_mode"),
        "gold_label": sample.claim.gold_label.value,
        "predicted_label": conditioned_label,
        "confidence": float(payload["confidence"]),
        "supports_public_only": bool(payload["supports_public_only"]),
        "ood_reasons": list((ood_reasons or {}).get(sample.claim.instance_id, [])),
        "claim_text": sample.claim.claim_text,
        "target_variables": dict(sample.claim.target_variables),
        "proxy_variables": list(sample.public.proxy_variables),
        "selection_mechanism": sample.public.selection_mechanism,
        "tool_report": tool_report,
        "tool_trace": list(tool_report.get("tool_trace", [])),
        "supporting_evidence": list(tool_report.get("supporting_evidence", [])),
        "counter_evidence": list(tool_report.get("counter_evidence", [])),
        "countermodel_found": bool(payload["countermodel_found"]),
        "countermodel_type": payload["countermodel_type"],
        "predicted_probabilities": dict(conditioned_verdict.get("probabilities", {})),
        "public_evidence_summary": {
            "scenario_id": public_payload["scenario_id"],
            "description": public_payload["description"],
            "variables": list(public_payload["variables"]),
            "proxy_variables": list(public_payload["proxy_variables"]),
            "selection_mechanism": public_payload["selection_mechanism"],
            "causal_level": public_payload["causal_level"],
        },
        "observed_data": public_payload["observed_data"],
        "verdict": conditioned_verdict,
        "conditioned_verdict": conditioned_verdict,
        "base_verdict": base_verdict,
        "system_notes": list(payload.get("system_notes", [])) + [f"witness_condition:{witness_condition}"],
        "witness_condition": witness_condition,
        "base_predicted_label": base_label,
        "base_confidence": float(base_prediction["confidence"]),
        "base_primary_witness_role": base_role,
        "base_witness_type": str((base_witness or {}).get("witness_type", "none")),
        "primary_witness_role": conditioned_role,
        "witness_type": str((conditioned_witness or {}).get("witness_type", "none")),
        "has_witness": conditioned_witness is not None,
        "verdict_changed": conditioned_label != base_label,
        "donor_instance_id": donor_bundle.sample.claim.instance_id if donor_bundle is not None else None,
        "condition_metadata": dict(condition_metadata or {}),
        "witness_faithfulness_score": _witness_faithfulness_score(conditioned_witness, conditioned_verdict),
        "explanation_support_score": _keyword_overlap_score(conditioned_witness, conditioned_verdict),
    }


def build_witness_condition_records(
    *,
    sample: BenchmarkSample,
    seed: int,
    split_name: str,
    donor_samples: list[BenchmarkSample],
    bundle_cache: dict[str, EvidenceBundle] | None = None,
    ood_reasons: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build replayed original/drop/corrupt/shuffle witness rows for one sample."""

    cache = dict(bundle_cache or {})
    for donor_sample in donor_samples:
        cache.setdefault(donor_sample.claim.instance_id, _build_evidence_bundle(donor_sample))
    bundle = cache.setdefault(sample.claim.instance_id, _build_evidence_bundle(sample))
    donor_bundle = _select_donor_bundle(
        bundle,
        donor_samples=donor_samples,
        bundle_cache=cache,
    )

    rows: list[dict[str, Any]] = []
    for condition in WITNESS_CONDITIONS:
        if condition == "original":
            payload = bundle.base_prediction
            condition_metadata = {"replay_mode": "original_system_prediction"}
            donor = None
        else:
            ledger, countermodel_result, tool_trace, condition_metadata = _condition_components(
                bundle,
                condition=condition,
                donor_bundle=donor_bundle,
            )
            payload = _decision_payload_from_components(
                bundle,
                condition=condition,
                ledger=ledger,
                countermodel_result=countermodel_result,
                tool_trace=tool_trace,
            )
            donor = donor_bundle if condition == "shuffle_witness" else None
        rows.append(
            _build_prediction_record(
                sample,
                payload,
                seed=seed,
                split_name=split_name,
                witness_condition=condition,
                base_prediction=bundle.base_prediction,
                donor_bundle=donor,
                condition_metadata=condition_metadata,
                ood_reasons=ood_reasons,
            )
        )
    return rows


def _condition_level_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "verdict_changed_rate": _mean([1.0 if row["verdict_changed"] else 0.0 for row in rows]),
        "witness_present_rate": _mean([1.0 if row["has_witness"] else 0.0 for row in rows]),
        "explanation_support_score": _mean([float(row["explanation_support_score"]) for row in rows]),
        "witness_faithfulness_score": _mean([float(row["witness_faithfulness_score"]) for row in rows]),
    }


def _aggregate_condition_results(per_seed_payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric_name
            for seed_payload in per_seed_payloads.values()
            for metric_name in seed_payload["metrics"]
        }
    )
    metric_values = {
        metric_name: [
            float(per_seed_payloads[seed]["metrics"][metric_name])
            for seed in sorted(per_seed_payloads)
        ]
        for metric_name in metric_names
    }
    summarized = summarize_metrics(
        metric_values,
        n_resamples=2000,
        random_state=0,
    )
    return {
        metric_name: summary.to_dict()
        for metric_name, summary in summarized.items()
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Witness Faithfulness",
        "",
        "## Setup",
        "",
        f"- System: {payload['system']}",
        f"- Seeds: {payload['seeds']}",
        f"- Samples per family: {payload['config']['samples_per_family']}",
        f"- Difficulty: {payload['config']['difficulty']:.2f}",
        "",
        "| Condition | Verdict Acc. | Unsafe Accept | Verdict Changed | Witness Present | Explanation Support | Faithfulness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in WITNESS_CONDITIONS:
        metrics = payload["aggregated_metrics"][condition]
        lines.append(
            f"| {condition} | "
            f"{metrics['verdict_accuracy']['formatted']} | "
            f"{metrics['unsafe_acceptance_rate']['formatted']} | "
            f"{metrics['verdict_changed_rate']['formatted']} | "
            f"{metrics['witness_present_rate']['formatted']} | "
            f"{metrics['explanation_support_score']['formatted']} | "
            f"{metrics['witness_faithfulness_score']['formatted']} |"
        )

    report = payload["significance"].get("verdict_accuracy")
    if report is not None:
        lines.extend(
            [
                "",
                "## Significance",
                "",
            ]
        )
        for row in report["comparisons"]:
            lines.append(
                f"- {row['comparison']}: diff={row['observed_difference']:.4f}, "
                f"p={row['p_value']:.4f}, adjusted={row.get('adjusted_p_value')}"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            payload["conclusion"]["summary"],
        ]
    )
    return "\n".join(lines)


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] | None = None,
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
    condition_rows: list[dict[str, Any]] = []
    per_seed_results: dict[int, dict[str, Any]] = {}

    for seed in resolved_seeds:
        run = build_seed_benchmark_run(
            seed=seed,
            difficulty=resolved_difficulty,
            samples_per_family=resolved_samples_per_family,
        )
        ood_reasons = dict(run.manifest.metadata.get("ood_reasons", {}))
        seed_condition_predictions: dict[str, list[dict[str, Any]]] = {
            condition: []
            for condition in WITNESS_CONDITIONS
        }
        split_cache: dict[str, dict[str, EvidenceBundle]] = {}

        for split_name in OOD_SPLITS:
            split_samples = sorted(run.split_samples[split_name], key=lambda item: item.claim.instance_id)
            split_cache[split_name] = {
                sample.claim.instance_id: _build_evidence_bundle(sample)
                for sample in split_samples
            }
            for sample in split_samples:
                rows = build_witness_condition_records(
                    sample=sample,
                    seed=seed,
                    split_name=split_name,
                    donor_samples=split_samples,
                    bundle_cache=split_cache[split_name],
                    ood_reasons=ood_reasons,
                )
                condition_rows.extend(rows)
                raw_predictions.extend(rows)
                for row in rows:
                    seed_condition_predictions[row["witness_condition"]].append(row)

        seed_results: dict[str, Any] = {}
        for condition, rows in seed_condition_predictions.items():
            scored = score_prediction_records(
                rows,
                game_id=f"witness_faithfulness::{condition}::seed::{seed}",
            )
            scored["metrics"].update(_condition_level_metrics(rows))
            seed_results[condition] = scored
        per_seed_results[seed] = seed_results

    aggregated_metrics = {
        condition: _aggregate_condition_results(
            {
                seed: per_seed_results[seed][condition]
                for seed in resolved_seeds
            }
        )
        for condition in WITNESS_CONDITIONS
    }

    significance, global_multiple_comparison_correction = build_seed_metric_significance(
        {
            "verdict_accuracy": {
                condition: [
                    float(per_seed_results[seed][condition]["metrics"]["verdict_accuracy"])
                    for seed in sorted(resolved_seeds)
                ]
                for condition in WITNESS_CONDITIONS
            }
        },
        baseline="original",
        metric_name="verdict_accuracy",
        estimand="seed_mean_verdict_accuracy",
    )

    original_accuracy = float(aggregated_metrics["original"]["verdict_accuracy"]["mean"])
    original_support = float(aggregated_metrics["original"]["explanation_support_score"]["mean"])
    degraded_conditions = [
        condition
        for condition in ("drop_witness", "corrupt_witness", "shuffle_witness")
        if float(aggregated_metrics[condition]["verdict_accuracy"]["mean"]) < original_accuracy
        or float(aggregated_metrics[condition]["explanation_support_score"]["mean"]) < original_support
        or float(aggregated_metrics[condition]["verdict_changed_rate"]["mean"]) > 0.0
    ]
    conclusion = {
        "degradation_observed": bool(degraded_conditions),
        "degraded_conditions": degraded_conditions,
        "summary": (
            "Replaying witness ablation/corruption/shuffle on the verifier evidence path produces measurable degradation."
            if degraded_conditions
            else "No measurable witness-faithfulness degradation was observed in this run."
        ),
    }
    payload = {
        "experiment_id": "exp_witness_faithfulness",
        "config": {
            "samples_per_family": int(resolved_samples_per_family),
            "difficulty": float(resolved_difficulty),
            "witness_conditions": list(WITNESS_CONDITIONS),
        },
        "requested_config": {
            "samples_per_family": int(samples_per_family),
            "difficulty": float(difficulty),
            "allow_protocol_violations": bool(allow_protocol_violations),
        },
        "system": SYSTEM_NAME,
        "seeds": resolved_seeds,
        "protocol": protocol,
        "raw_predictions": raw_predictions,
        "condition_rows": condition_rows,
        "per_seed_results": per_seed_results,
        "aggregated_metrics": aggregated_metrics,
        "significance": significance,
        "global_multiple_comparison_correction": global_multiple_comparison_correction,
        "conclusion": conclusion,
    }
    summary = _markdown_summary(payload)
    artifacts = write_artifacts(
        output_path=output_path or "outputs/mainline/exp_witness_faithfulness.json",
        payload=payload,
        markdown_summary=summary,
    )
    payload["markdown_summary"] = summary
    payload["artifacts"] = artifacts
    Path(artifacts["json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 4 witness faithfulness experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list.")
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
        samples_per_family=args.samples_per_family,
        difficulty=args.difficulty,
        allow_protocol_violations=args.allow_protocol_violations,
        output_path=args.output,
    )
    print(json.dumps(payload["aggregated_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
