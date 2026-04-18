"""Shared helpers for Phase 4 benchmark experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.tool_executor import ToolExecutor
from benchmark.generator import BenchmarkGenerator, BenchmarkSample, list_supported_benchmark_families
from benchmark.schema import BenchmarkSplitManifest, PublicCausalInstance, VerdictLabel
from benchmark.split_builder import build_benchmark_splits
from evaluation.reporting import compare_prediction_groups, summarize_metrics
from evaluation.scorer import Scorer
from verifier.assumption_ledger import AssumptionLedger, build_assumption_ledger
from verifier.claim_parser import parse_claim
from verifier.countermodel_search import CountermodelSearchResult, search_countermodels
from verifier.decision import VerifierDecision, decide_verdict
from verifier.outputs import ClaimPolarity
from verifier.pipeline import VerifierPipeline

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
PRIMARY_METRICS: tuple[str, ...] = (
    "verdict_accuracy",
    "macro_f1",
    "invalid_claim_acceptance_rate",
    "unidentifiable_awareness",
    "ece",
    "brier",
    "countermodel_coverage",
)
DEFAULT_MODEL_FAMILIES: tuple[str, ...] = (
    "countermodel_grounded",
    "skeptical_family",
    "optimistic_family",
    "claim_only_family",
)
OOD_SPLITS: tuple[str, ...] = ("test_iid", "test_ood")


@dataclass(slots=True)
class SeedBenchmarkRun:
    seed: int
    manifest: BenchmarkSplitManifest
    samples: list[BenchmarkSample]
    split_samples: dict[str, list[BenchmarkSample]]


def default_benchmark_families() -> list[str]:
    return list_supported_benchmark_families(include_showcase=False)


def build_seed_benchmark_run(
    *,
    seed: int,
    difficulty: float,
    samples_per_family: int,
    family_names: list[str] | None = None,
) -> SeedBenchmarkRun:
    families = list(family_names or default_benchmark_families())
    generator = BenchmarkGenerator(seed=seed)
    samples: list[BenchmarkSample] = []
    sample_index = 0
    for family_name in families:
        for _ in range(max(1, int(samples_per_family))):
            sample_seed = int(seed) * 1000 + sample_index
            samples.append(
                generator.generate_benchmark_sample(
                    family_name=family_name,
                    difficulty=difficulty,
                    seed=sample_seed,
                )
            )
            sample_index += 1

    manifest = build_benchmark_splits(
        [sample.claim for sample in samples],
        seed=seed,
    )
    sample_by_id = {sample.claim.instance_id: sample for sample in samples}
    split_samples = {
        split_name: [sample_by_id[instance_id] for instance_id in instance_ids]
        for split_name, instance_ids in manifest.split_map().items()
    }
    return SeedBenchmarkRun(
        seed=seed,
        manifest=manifest,
        samples=samples,
        split_samples=split_samples,
    )


def _verifier_tool_context(sample: BenchmarkSample) -> dict[str, Any]:
    public = sample.public
    return {
        "treatment": sample.claim.target_variables["treatment"],
        "outcome": sample.claim.target_variables["outcome"],
        "proxy_variables": list(getattr(public, "proxy_variables", [])),
        "selection_variables": list(getattr(public, "selection_variables", [])),
        "selection_mechanism": getattr(public, "selection_mechanism", None),
        "claim_stance": "pro_causal",
    }


def _serialize_verifier_decision(decision: VerifierDecision) -> dict[str, Any]:
    payload = decision.to_dict()
    payload["label"] = decision.label.value
    payload["confidence"] = float(decision.confidence)
    return payload


def _coerce_public_instance(sample: BenchmarkSample) -> PublicCausalInstance:
    return sample.public


def _run_main_verifier(sample: BenchmarkSample) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    tool_context = _verifier_tool_context(sample)
    tool_executor = ToolExecutor({})
    tool_report = tool_executor.execute_for_claim(
        scenario=scenario,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    decision = VerifierPipeline().run(
        sample.claim.claim_text,
        scenario=scenario,
        tool_trace=tool_report["tool_trace"],
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
        "system_notes": [],
    }


def _run_no_tools_verifier(sample: BenchmarkSample) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    tool_context = _verifier_tool_context(sample)
    decision = VerifierPipeline().run(
        sample.claim.claim_text,
        scenario=scenario,
        tool_trace=[],
        tool_context=tool_context,
    )
    return {
        "predicted_label": decision.label.value,
        "confidence": float(decision.confidence),
        "verdict": _serialize_verifier_decision(decision),
        "tool_report": {
            "selected_tools": [],
            "claim_stance": "pro_causal",
            "identified_issues": [],
            "supporting_evidence": [],
            "counter_evidence": [],
            "tool_trace": [],
        },
        "countermodel_found": decision.countermodel_witness is not None,
        "countermodel_type": (
            decision.countermodel_witness.payload.get("countermodel_type")
            if decision.countermodel_witness is not None
            else None
        ),
        "supports_public_only": True,
        "system_notes": ["tools_disabled"],
    }


def _run_manual_variant(
    sample: BenchmarkSample,
    *,
    use_ledger: bool,
    use_countermodel: bool,
    use_tools: bool,
) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    tool_context = _verifier_tool_context(sample)
    parsed_claim = parse_claim(sample.claim.claim_text)
    ledger = build_assumption_ledger(parsed_claim) if use_ledger else AssumptionLedger([])
    countermodel = (
        search_countermodels(
            parsed_claim,
            ledger,
            scenario=scenario,
            context={
                **tool_context,
                "public_instance": scenario,
                "observed_data": scenario.observed_data.copy(deep=True),
            },
        )
        if use_countermodel
        else CountermodelSearchResult(found_countermodel=False, candidates=[])
    )

    tool_report: dict[str, Any]
    if use_tools:
        raw_tool_report = ToolExecutor({}).execute_for_claim(
            scenario=scenario,
            claim=sample.claim.claim_text,
            level=int(sample.claim.causal_level[1]),
            context=tool_context,
        )
        tool_trace = list(raw_tool_report["tool_trace"])
        tool_report = {
            "selected_tools": list(raw_tool_report["selected_tools"]),
            "claim_stance": raw_tool_report["claim_stance"],
            "identified_issues": list(raw_tool_report["identified_issues"]),
            "supporting_evidence": list(raw_tool_report["supporting_evidence"]),
            "counter_evidence": list(raw_tool_report["counter_evidence"]),
            "tool_trace": tool_trace,
        }
    else:
        tool_trace = []
        tool_report = {
            "selected_tools": [],
            "claim_stance": "pro_causal",
            "identified_issues": [],
            "supporting_evidence": [],
            "counter_evidence": [],
            "tool_trace": [],
        }

    decision = decide_verdict(
        parsed_claim,
        ledger,
        countermodel,
        tool_trace=tool_trace,
    )
    return {
        "predicted_label": decision.label.value,
        "confidence": float(decision.confidence),
        "verdict": _serialize_verifier_decision(decision),
        "tool_report": tool_report,
        "countermodel_found": decision.countermodel_witness is not None,
        "countermodel_type": (
            decision.countermodel_witness.payload.get("countermodel_type")
            if decision.countermodel_witness is not None
            else None
        ),
        "supports_public_only": True,
        "system_notes": [
            note
            for note, enabled in (
                ("ledger_disabled", use_ledger),
                ("countermodel_disabled", use_countermodel),
                ("tools_disabled", use_tools),
            )
            if not enabled
        ],
    }


def _apply_no_abstention(sample: BenchmarkSample, payload: dict[str, Any]) -> dict[str, Any]:
    adjusted = dict(payload)
    verdict = dict(payload["verdict"])
    predicted_label = str(payload["predicted_label"])
    if predicted_label == VerdictLabel.UNIDENTIFIABLE.value:
        parsed_claim = parse_claim(sample.claim.claim_text)
        forced_label = (
            VerdictLabel.INVALID.value
            if parsed_claim.claim_polarity is ClaimPolarity.NEGATIVE
            else VerdictLabel.VALID.value
        )
        verdict["label"] = forced_label
        verdict["confidence"] = max(float(payload["confidence"]), 0.51)
        verdict["metadata"] = {
            **dict(verdict.get("metadata", {})),
            "ablation": "no_abstention",
            "forced_from": VerdictLabel.UNIDENTIFIABLE.value,
        }
        adjusted["predicted_label"] = forced_label
        adjusted["confidence"] = float(verdict["confidence"])
        adjusted["verdict"] = verdict
        adjusted["system_notes"] = list(payload.get("system_notes", [])) + ["abstention_disabled"]
    return adjusted


def _run_claim_only_family(sample: BenchmarkSample) -> dict[str, Any]:
    parsed_claim = parse_claim(sample.claim.claim_text)
    if parsed_claim.claim_polarity is ClaimPolarity.NEGATIVE:
        label = VerdictLabel.INVALID.value
        confidence = 0.67
    elif parsed_claim.needs_abstention_check:
        label = VerdictLabel.UNIDENTIFIABLE.value
        confidence = 0.58
    else:
        label = VerdictLabel.VALID.value
        confidence = 0.64
    return {
        "predicted_label": label,
        "confidence": confidence,
        "verdict": {
            "label": label,
            "confidence": confidence,
            "assumption_ledger": [],
            "witness": None,
            "support_witness": None,
            "countermodel_witness": None,
            "tool_trace": [],
            "reasoning_summary": "Claim-only family ignores the public benchmark tools and judges only from the surface claim text.",
            "metadata": {"predictor_family": "claim_only_family"},
        },
        "tool_report": {
            "selected_tools": [],
            "claim_stance": "pro_causal",
            "identified_issues": [],
            "supporting_evidence": [],
            "counter_evidence": [],
            "tool_trace": [],
        },
        "countermodel_found": False,
        "countermodel_type": None,
        "supports_public_only": True,
        "system_notes": ["claim_only_family"],
    }


def _apply_family_postprocessing(
    family_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    adjusted = dict(payload)
    verdict = dict(payload["verdict"])
    label = str(payload["predicted_label"])
    confidence = float(payload["confidence"])

    if family_name == "skeptical_family":
        if label in {VerdictLabel.VALID.value, VerdictLabel.INVALID.value} and confidence < 0.82:
            label = VerdictLabel.UNIDENTIFIABLE.value
            confidence = max(0.61, confidence)
    elif family_name == "optimistic_family":
        if label == VerdictLabel.UNIDENTIFIABLE.value:
            label = VerdictLabel.VALID.value
            confidence = max(0.57, confidence)
    elif family_name != "countermodel_grounded":
        raise ValueError(f"Unsupported model family: {family_name!r}.")

    verdict["label"] = label
    verdict["confidence"] = confidence
    verdict["metadata"] = {
        **dict(verdict.get("metadata", {})),
        "predictor_family": family_name,
    }
    adjusted["predicted_label"] = label
    adjusted["confidence"] = confidence
    adjusted["verdict"] = verdict
    adjusted["system_notes"] = list(payload.get("system_notes", [])) + [family_name]
    return adjusted


def predict_sample(
    sample: BenchmarkSample,
    *,
    system_name: str,
) -> dict[str, Any]:
    if system_name == "countermodel_grounded":
        return _run_main_verifier(sample)
    if system_name == "no_tools":
        return _run_no_tools_verifier(sample)
    if system_name == "no_ledger":
        return _run_manual_variant(sample, use_ledger=False, use_countermodel=True, use_tools=True)
    if system_name == "no_countermodel":
        return _run_manual_variant(sample, use_ledger=True, use_countermodel=False, use_tools=True)
    if system_name == "no_abstention":
        return _apply_no_abstention(sample, _run_main_verifier(sample))
    if system_name == "oracle_leaking_partition":
        gold_label = sample.gold.gold_label.value
        return {
            "predicted_label": gold_label,
            "confidence": 0.999,
            "verdict": {
                "label": gold_label,
                "confidence": 0.999,
                "assumption_ledger": [],
                "witness": None,
                "support_witness": None,
                "countermodel_witness": None,
                "tool_trace": [],
                "reasoning_summary": "Oracle-leaking partition accessed gold supervision directly.",
                "metadata": {
                    "leakage_mode": "oracle_leaking_partition",
                    "oracle_fields": ["gold_label", "ground_truth", "true_dag", "true_scm"],
                },
            },
            "tool_report": {
                "selected_tools": [],
                "claim_stance": "pro_causal",
                "identified_issues": [],
                "supporting_evidence": [],
                "counter_evidence": [],
                "tool_trace": [],
            },
            "countermodel_found": gold_label in {VerdictLabel.INVALID.value, VerdictLabel.UNIDENTIFIABLE.value},
            "countermodel_type": "oracle_access",
            "supports_public_only": False,
            "system_notes": ["oracle_leakage"],
        }
    if system_name == "claim_only_family":
        return _run_claim_only_family(sample)
    if system_name in {"skeptical_family", "optimistic_family"}:
        return _apply_family_postprocessing(system_name, _run_main_verifier(sample))
    raise ValueError(f"Unsupported system_name: {system_name!r}.")


def evaluate_system_on_samples(
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
        payload = predict_sample(sample, system_name=system_name)
        verdict = dict(payload["verdict"])
        record = {
            "seed": int(seed),
            "split": split_name,
            "system_name": system_name,
            "instance_id": sample.claim.instance_id,
            "scenario_id": sample.gold.scenario_id,
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


def aggregate_seed_metrics(
    per_seed_payloads: dict[int, dict[str, Any]],
    *,
    split_name: str,
) -> dict[str, Any]:
    metric_values = {
        metric_name: [
            float(seed_payload[split_name]["metrics"].get(metric_name, 0.0))
            for _, seed_payload in sorted(per_seed_payloads.items())
        ]
        for metric_name in PRIMARY_METRICS
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


def compare_system_predictions(
    system_predictions: dict[str, list[dict[str, Any]]],
    *,
    baseline: str,
) -> dict[str, Any] | None:
    if len(system_predictions) < 2:
        return None
    truth = [item["gold_label"] for item in next(iter(system_predictions.values()))]
    predictions = {
        system_name: [item["predicted_label"] for item in records]
        for system_name, records in system_predictions.items()
    }
    report = compare_prediction_groups(
        truth,
        predictions,
        baseline=baseline,
        method="paired_bootstrap",
        metric_name="verdict_accuracy",
        n_resamples=2000,
        random_state=0,
    )
    return report.to_dict()


def write_artifacts(
    *,
    output_path: str | Path,
    payload: dict[str, Any],
    markdown_summary: str,
) -> dict[str, str]:
    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_predictions_path = json_path.with_name(f"{json_path.stem}_raw_predictions.jsonl")
    with raw_predictions_path.open("w", encoding="utf-8") as handle:
        for record in payload.get("raw_predictions", []):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(markdown_summary.rstrip() + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "raw_predictions": str(raw_predictions_path),
        "markdown_summary": str(markdown_path),
    }


def manifest_metadata(run: SeedBenchmarkRun) -> dict[str, Any]:
    return {
        "dataset_name": run.manifest.dataset_name,
        "version": run.manifest.version,
        "holdout_strategy": {
            "family_holdout": list(run.manifest.family_holdout),
            "lexical_holdout": list(run.manifest.lexical_holdout),
            "variable_renaming_holdout": bool(run.manifest.variable_renaming_holdout),
        },
        "metadata": dict(run.manifest.metadata),
    }
