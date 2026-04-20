"""Shared helpers for Phase 4 benchmark experiments."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from agents.tool_executor import ToolExecutor
from benchmark.attacks import generate_attack_sample
from benchmark.generator import BenchmarkGenerator, BenchmarkSample, list_supported_benchmark_families
from benchmark.schema import BenchmarkSplitManifest, ClaimInstance, PublicCausalInstance, VerdictLabel
from benchmark.split_builder import build_benchmark_splits
from evaluation.reporting import compare_prediction_groups, summarize_metrics
from evaluation.scorer import Scorer
from evaluation.significance import holm_bonferroni
from verifier.assumption_ledger import AssumptionLedger, build_assumption_ledger
from verifier.claim_parser import parse_claim
from verifier.countermodel_search import CountermodelSearchResult, search_countermodels
from verifier.decision import VerifierDecision, decide_verdict
from verifier.outputs import ClaimPolarity, ClaimStrength, QueryType, SelectiveVerifierOutput
from verifier.pipeline import VerifierPipeline

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
PRIMARY_METRICS: tuple[str, ...] = (
    "unsafe_acceptance_rate",
    "wise_refusal_recall",
    "wise_refusal_precision",
    "over_refusal_rate",
    "verdict_accuracy",
    "macro_f1",
    "ece",
    "brier",
    "countermodel_coverage",
)
DEFAULT_MODEL_FAMILIES: tuple[str, ...] = (
    "gpt_like",
    "claude_like",
    "qwen_like",
)
SUPPORTED_SYSTEM_NAMES: tuple[str, ...] = (
    "judge_direct",
    "debate_reduced",
    "tool_only",
    "countermodel_grounded",
    "no_tools",
    "no_ledger",
    "no_countermodel",
    "no_abstention",
    "claim_only_family",
    "skeptical_family",
    "optimistic_family",
)
OOD_SPLITS: tuple[str, ...] = ("test_iid", "test_ood")
MIN_FORMAL_SEED_COUNT = 3
MIN_FORMAL_SAMPLES_PER_FAMILY = 10
MIN_HUMAN_AUDIT_SUBSET_SIZE = 150
PAIRWISE_ALPHA = 0.05
PAIRWISE_RESAMPLES = 2000
DEFAULT_ATTACKER_FAMILIES: tuple[str, ...] = (
    "baseline_attacker",
    "formal_attacker",
    "hidden_information_attacker",
)
SUPPORTED_ATTACKER_FAMILIES = frozenset(DEFAULT_ATTACKER_FAMILIES)
SUPPORTED_MODEL_FAMILIES = frozenset(DEFAULT_MODEL_FAMILIES)


@dataclass(slots=True)
class SeedBenchmarkRun:
    seed: int
    manifest: BenchmarkSplitManifest
    samples: list[BenchmarkSample]
    split_samples: dict[str, list[BenchmarkSample]]


def default_benchmark_families() -> list[str]:
    return list_supported_benchmark_families(include_showcase=False)


def normalize_benchmark_samples_per_family(samples_per_family: int) -> int:
    return max(1, int(samples_per_family))


def normalize_benchmark_difficulty(difficulty: float) -> float:
    return float(max(0.0, min(1.0, float(difficulty))))


def normalize_experiment_seeds(
    seeds: list[int] | tuple[int, ...] | None,
    *,
    minimum_count: int | None = None,
    allow_protocol_violations: bool = False,
) -> list[int]:
    resolved = list(DEFAULT_SEEDS) if not seeds else [int(seed) for seed in seeds]
    seen: set[int] = set()
    duplicates: list[int] = []
    for seed in resolved:
        if seed in seen and seed not in duplicates:
            duplicates.append(seed)
        seen.add(seed)
    if duplicates:
        raise ValueError(f"Duplicate seeds are not allowed: {duplicates!r}.")
    if minimum_count is not None and len(resolved) < int(minimum_count) and not allow_protocol_violations:
        raise ValueError(
            f"Formal Phase 4 experiments require at least {int(minimum_count)} seeds; "
            f"received {len(resolved)} seed(s): {resolved!r}."
        )
    return resolved


def validate_system_names(
    systems: list[str] | tuple[str, ...] | None,
    *,
    default_systems: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    resolved = list(default_systems or ()) if not systems else [str(system).strip() for system in systems]
    if not resolved:
        raise ValueError("At least one system_name is required.")
    if any(not system_name for system_name in resolved):
        raise ValueError("System names must be non-empty strings.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for system_name in resolved:
        if system_name in seen and system_name not in duplicates:
            duplicates.append(system_name)
        seen.add(system_name)
    if duplicates:
        raise ValueError(f"Duplicate system names are not allowed: {duplicates!r}.")

    unsupported = sorted(set(resolved) - set(SUPPORTED_SYSTEM_NAMES))
    if unsupported:
        raise ValueError(
            f"Unsupported system_name values: {unsupported!r}. "
            f"Supported values are: {sorted(SUPPORTED_SYSTEM_NAMES)!r}."
        )
    return resolved


def validate_model_families(
    model_families: list[str] | tuple[str, ...] | None,
    *,
    default_families: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    resolved = list(default_families or DEFAULT_MODEL_FAMILIES) if not model_families else [str(name).strip() for name in model_families]
    if not resolved:
        raise ValueError("At least one model_family is required.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for family_name in resolved:
        if not family_name:
            raise ValueError("Model family names must be non-empty strings.")
        if family_name in seen and family_name not in duplicates:
            duplicates.append(family_name)
        seen.add(family_name)
    if duplicates:
        raise ValueError(f"Duplicate model families are not allowed: {duplicates!r}.")
    unsupported = sorted(set(resolved) - SUPPORTED_MODEL_FAMILIES)
    if unsupported:
        raise ValueError(
            f"Unsupported model_family values: {unsupported!r}. "
            f"Supported values are: {sorted(SUPPORTED_MODEL_FAMILIES)!r}."
        )
    return resolved


def summarize_protocol_compliance(
    seeds: list[int] | tuple[int, ...],
    *,
    minimum_count: int = MIN_FORMAL_SEED_COUNT,
    minimum_samples_per_family: int | None = None,
    observed_samples_per_family: int | None = None,
    minimum_audit_subset_size: int | None = None,
    observed_audit_subset_size: int | None = None,
    allow_protocol_violations: bool = False,
) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    requirements: dict[str, dict[str, Any]] = {
        "seeds": {
            "minimum": int(minimum_count),
            "observed": len(seed_list),
            "satisfied": len(seed_list) >= int(minimum_count),
        }
    }
    violations: list[str] = []

    def _register_requirement(name: str, *, minimum: int, observed: int) -> None:
        satisfied = int(observed) >= int(minimum)
        requirements[name] = {
            "minimum": int(minimum),
            "observed": int(observed),
            "satisfied": satisfied,
        }
        if not satisfied:
            violations.append(name)

    if minimum_samples_per_family is not None and observed_samples_per_family is not None:
        _register_requirement(
            "samples_per_family",
            minimum=int(minimum_samples_per_family),
            observed=int(observed_samples_per_family),
        )
    if minimum_audit_subset_size is not None and observed_audit_subset_size is not None:
        _register_requirement(
            "audit_subset_size",
            minimum=int(minimum_audit_subset_size),
            observed=int(observed_audit_subset_size),
        )

    if not requirements["seeds"]["satisfied"]:
        violations.insert(0, "seeds")
    compliant = len(violations) == 0
    return {
        "minimum_seed_count": int(minimum_count),
        "seed_count": len(seed_list),
        "seed_list": seed_list,
        "requirements": requirements,
        "violations": violations,
        "compliant": compliant,
        "override_used": bool(allow_protocol_violations and bool(violations)),
    }


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
    metric_name: str,
    estimand: str,
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
        "metric_name": metric_name,
        "estimand": estimand,
        "correction": "holm-bonferroni",
    }


def build_seed_metric_significance(
    metric_groups_by_scope: dict[str, dict[str, list[float]]],
    *,
    baseline: str,
    metric_name: str = "verdict_accuracy",
    estimand: str = "seed_mean_verdict_accuracy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    significance: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    group_order: dict[str, list[str]] = {}

    for scope_index, (scope_name, group_metrics) in enumerate(metric_groups_by_scope.items()):
        if baseline not in group_metrics:
            raise ValueError(f"Unknown baseline group {baseline!r} for scope {scope_name!r}.")
        ordered_groups = list(group_metrics)
        group_order[scope_name] = ordered_groups
        if len(ordered_groups) < 2:
            significance[scope_name] = None
            continue

        baseline_values = [float(value) for value in group_metrics[baseline]]
        rows: list[dict[str, Any]] = []
        for group_index, group_name in enumerate(ordered_groups):
            if group_name == baseline:
                continue
            row = _paired_seed_bootstrap_row(
                baseline_values,
                [float(value) for value in group_metrics[group_name]],
                comparison_name=f"{baseline} vs {group_name}",
                model_a=baseline,
                model_b=group_name,
                random_state=17 + group_index + (10 * scope_index),
                metric_name=metric_name,
                estimand=estimand,
            )
            hypothesis = f"{scope_name}: {row['comparison']}"
            raw_p_values[hypothesis] = float(row["p_value"])
            rows.append(row)

        significance[scope_name] = {
            "method": "paired_seed_bootstrap",
            "metric_name": metric_name,
            "estimand": estimand,
            "alpha": float(PAIRWISE_ALPHA),
            "baseline": baseline,
            "comparisons": rows,
            "correction": "holm-bonferroni",
        }

    correction_table = holm_bonferroni(raw_p_values, alpha=PAIRWISE_ALPHA) if raw_p_values else []
    correction_lookup = {entry.hypothesis: entry for entry in correction_table}
    for scope_name, report in significance.items():
        if report is None:
            continue
        for row in report["comparisons"]:
            hypothesis = f"{scope_name}: {row['comparison']}"
            correction = correction_lookup[hypothesis]
            row["adjusted_p_value"] = float(correction.adjusted_p_value)
            row["reject_after_correction"] = bool(correction.reject)
            row["family_hypothesis"] = hypothesis
        report["correction_scope"] = {
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


def _stable_sample_seed(seed: int, family_name: str, sample_index: int) -> int:
    material = f"phase4::{int(seed)}::{family_name}::{int(sample_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _apply_split_metadata_to_samples(
    *,
    sample_by_id: dict[str, BenchmarkSample],
    manifest,
) -> None:
    ood_reasons = manifest.metadata.get("ood_reasons", {})
    ood_reasons = ood_reasons if isinstance(ood_reasons, dict) else {}

    for split_name, instance_ids in manifest.split_map().items():
        for instance_id in instance_ids:
            sample = sample_by_id.get(instance_id)
            if sample is None:
                continue
            sample.claim.meta["ood_split"] = split_name
            if instance_id in ood_reasons:
                sample.claim.meta["ood_reasons"] = list(ood_reasons[instance_id])


def build_seed_benchmark_run(
    *,
    seed: int,
    difficulty: float,
    samples_per_family: int,
    family_names: list[str] | None = None,
    family_holdout: list[str] | tuple[str, ...] | None = None,
    lexical_holdout: list[str] | tuple[str, ...] | None = None,
    variable_renaming_holdout: bool | None = None,
    mechanism_holdout: list[str] | tuple[str, ...] | None = None,
    attack_family_holdout: list[str] | tuple[str, ...] | None = None,
    context_shift_holdout: list[str] | tuple[str, ...] | None = None,
    paired_flip_holdout: bool | None = None,
) -> SeedBenchmarkRun:
    families = list(family_names or default_benchmark_families())
    generator = BenchmarkGenerator(seed=seed)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    samples: list[BenchmarkSample] = []
    sample_index = 0
    for family_name in families:
        for sample_slot in range(resolved_samples_per_family):
            sample_seed = _stable_sample_seed(seed, family_name, sample_index)
            if paired_flip_holdout and sample_slot == 0:
                anchor, flipped = generator.generate_paired_flip_samples(
                    family_name=family_name,
                    difficulty=resolved_difficulty,
                    seed=sample_seed,
                )
                samples.extend([anchor, flipped])
                samples.append(
                    generator.generate_benchmark_sample(
                        family_name=family_name,
                        difficulty=resolved_difficulty,
                        seed=sample_seed + 17,
                    )
                )
            else:
                samples.append(
                    generator.generate_benchmark_sample(
                        family_name=family_name,
                        difficulty=resolved_difficulty,
                        seed=sample_seed,
                    )
                )
            sample_index += 1

    manifest = build_benchmark_splits(
        [sample.claim for sample in samples],
        seed=seed,
        family_holdout=family_holdout,
        lexical_holdout=lexical_holdout,
        variable_renaming_holdout=variable_renaming_holdout,
        mechanism_holdout=mechanism_holdout,
        attack_family_holdout=attack_family_holdout,
        context_shift_holdout=context_shift_holdout,
        paired_flip_holdout=paired_flip_holdout,
    )
    sample_by_id = {sample.claim.instance_id: sample for sample in samples}
    _apply_split_metadata_to_samples(sample_by_id=sample_by_id, manifest=manifest)
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


def build_seed_attack_benchmark_run(
    *,
    seed: int,
    difficulty: float,
    samples_per_family: int,
    family_names: list[str] | None = None,
) -> SeedBenchmarkRun:
    families = list(family_names or default_benchmark_families())
    generator = BenchmarkGenerator(seed=seed)
    resolved_difficulty = normalize_benchmark_difficulty(difficulty)
    resolved_samples_per_family = normalize_benchmark_samples_per_family(samples_per_family)
    samples: list[BenchmarkSample] = []
    sample_index = 0

    for family_name in families:
        collected = 0
        attempts = 0
        max_attempts = max(8, resolved_samples_per_family * 24)
        while collected < resolved_samples_per_family:
            sample_seed = _stable_sample_seed(seed, family_name, sample_index)
            candidate = generator.generate_benchmark_sample(
                family_name=family_name,
                difficulty=resolved_difficulty,
                seed=sample_seed,
            )
            sample_index += 1
            attempts += 1
            if candidate.claim.meta.get("attack_name") is None or candidate.claim.meta.get("claim_mode") != "attack":
                if attempts >= max_attempts:
                    raise ValueError(
                        "Unable to build an attack-only benchmark run with enough samples for "
                        f"family={family_name!r}, seed={seed}, samples_per_family={resolved_samples_per_family}."
                    )
                continue
            samples.append(candidate)
            collected += 1

    manifest = build_benchmark_splits([sample.claim for sample in samples], seed=seed)
    sample_by_id = {sample.claim.instance_id: sample for sample in samples}
    _apply_split_metadata_to_samples(sample_by_id=sample_by_id, manifest=manifest)
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


def _rewrite_claim_instance(
    sample: BenchmarkSample,
    *,
    claim_text: str,
    attacker_rationale: str | None = None,
    language_suffix: str | None = None,
    meta_updates: dict[str, Any] | None = None,
) -> BenchmarkSample:
    payload = sample.claim.to_dict()
    payload["claim_text"] = str(claim_text).strip()
    if attacker_rationale is not None:
        payload["attacker_rationale"] = str(attacker_rationale).strip()
    if language_suffix:
        payload["language_template_id"] = f"{payload['language_template_id']}::{language_suffix}"
    payload["meta"] = {
        **dict(payload.get("meta", {})),
        **dict(meta_updates or {}),
    }
    return BenchmarkSample(
        claim=ClaimInstance.from_dict(payload),
        gold=sample.gold,
        public=sample.public,
        blueprint=sample.blueprint,
    )


def _private_anchor(sample: BenchmarkSample) -> str:
    if sample.gold.hidden_variables:
        return str(sample.gold.hidden_variables[0])
    if sample.public.selection_variables:
        return str(sample.public.selection_variables[0])
    proxy_variables = list(sample.public.proxy_variables)
    if proxy_variables:
        return str(proxy_variables[0])
    return str(sample.claim.target_variables["treatment"])


def attack_only_samples(samples: list[BenchmarkSample]) -> list[BenchmarkSample]:
    return [
        sample
        for sample in samples
        if sample.claim.meta.get("attack_name") is not None
        or sample.claim.meta.get("claim_mode") == "attack"
    ]


def _profile_variant_seed(sample: BenchmarkSample, *, profile_name: str) -> int:
    base_seed = int(sample.claim.meta.get("seed", 0))
    material = f"{sample.claim.instance_id}::{profile_name}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return base_seed + offset


def _render_attack_profile_variant(
    sample: BenchmarkSample,
    *,
    profile_name: str,
    style_id: str,
    preferred_attack_names: tuple[str, ...] = (),
    inject_private_hint: bool = False,
    meta_updates: dict[str, Any] | None = None,
) -> BenchmarkSample:
    original_attack_name = str(sample.claim.meta.get("attack_name") or "").strip()
    if not original_attack_name:
        return sample

    resolved_seed = _profile_variant_seed(sample, profile_name=profile_name)
    gold_label = sample.claim.gold_label.value
    query_type = sample.claim.query_type
    variant = None
    for attack_name in (*preferred_attack_names, original_attack_name):
        try:
            variant = generate_attack_sample(
                sample.blueprint,
                gold_label=gold_label,
                query_type=query_type,
                attack_name=attack_name,
                seed=resolved_seed,
                style_id=style_id,
            )
            break
        except ValueError:
            continue
    if variant is None:
        return sample

    claim_text = variant.claim_text.rstrip()
    attacker_rationale = variant.attacker_rationale.rstrip()
    if inject_private_hint:
        anchor = _private_anchor(sample)
        claim_text = (
            f"{claim_text} Private operational details tied to {anchor} close the usual remaining loophole."
        ).strip()
        attacker_rationale = (
            f"{attacker_rationale} Present the argument as if private, non-public evidence closes "
            "the main countermodel escape hatch."
        ).strip()

    return _rewrite_claim_instance(
        sample,
        claim_text=claim_text,
        attacker_rationale=attacker_rationale,
        language_suffix=profile_name,
        meta_updates={
            "attack_name": variant.attack_name,
            "style_id": variant.style_id,
            "attack_profile_source": "structured_attack_regeneration",
            **dict(meta_updates or {}),
        },
    )


def apply_attack_strength_profile(
    samples: list[BenchmarkSample],
    *,
    strength_name: str,
) -> list[BenchmarkSample]:
    rewritten: list[BenchmarkSample] = []
    for sample in samples:
        if sample.claim.meta.get("attack_name") is None:
            rewritten.append(sample)
            continue

        if strength_name == "weak":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=strength_name,
                    style_id="plainspoken",
                    meta_updates={"attack_strength": strength_name},
                )
            )
        elif strength_name == "medium":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=strength_name,
                    style_id="technical",
                    meta_updates={"attack_strength": strength_name},
                )
            )
        elif strength_name == "strong":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=strength_name,
                    style_id="assertive",
                    meta_updates={"attack_strength": strength_name},
                )
            )
        elif strength_name == "hidden_information_aware":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=strength_name,
                    style_id="technical",
                    preferred_attack_names=("hidden_confounder_denial",),
                    inject_private_hint=True,
                    meta_updates={"attack_strength": strength_name},
                )
            )
        else:
            raise ValueError(f"Unsupported attack strength profile: {strength_name!r}.")
    return rewritten


def validate_attacker_families(
    attacker_families: list[str] | tuple[str, ...] | None,
) -> list[str]:
    resolved = list(DEFAULT_ATTACKER_FAMILIES) if not attacker_families else [str(name).strip() for name in attacker_families]
    if not resolved:
        raise ValueError("At least one attacker_family is required.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for family_name in resolved:
        if family_name in seen and family_name not in duplicates:
            duplicates.append(family_name)
        seen.add(family_name)
    if duplicates:
        raise ValueError(f"Duplicate attacker families are not allowed: {duplicates!r}.")
    unsupported = sorted(set(resolved) - SUPPORTED_ATTACKER_FAMILIES)
    if unsupported:
        raise ValueError(
            f"Unsupported attacker_family values: {unsupported!r}. "
            f"Supported values are: {sorted(SUPPORTED_ATTACKER_FAMILIES)!r}."
        )
    return resolved


def apply_attacker_family_profile(
    samples: list[BenchmarkSample],
    *,
    attacker_family: str,
) -> list[BenchmarkSample]:
    rewritten: list[BenchmarkSample] = []
    for sample in samples:
        if sample.claim.meta.get("attack_name") is None:
            rewritten.append(sample)
            continue

        if attacker_family == "baseline_attacker":
            rewritten.append(
                _rewrite_claim_instance(
                    sample,
                    claim_text=sample.claim.claim_text.rstrip(),
                    attacker_rationale=sample.claim.attacker_rationale.rstrip(),
                    language_suffix=attacker_family,
                    meta_updates={"attacker_family": attacker_family},
                )
            )
            continue
        if attacker_family == "formal_attacker":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=attacker_family,
                    style_id="technical",
                    meta_updates={"attacker_family": attacker_family},
                )
            )
        elif attacker_family == "hidden_information_attacker":
            rewritten.append(
                _render_attack_profile_variant(
                    sample,
                    profile_name=attacker_family,
                    style_id="technical",
                    preferred_attack_names=("hidden_confounder_denial",),
                    inject_private_hint=True,
                    meta_updates={"attacker_family": attacker_family},
                )
            )
        else:
            raise ValueError(f"Unsupported attacker_family: {attacker_family!r}.")
    return rewritten


def apply_attacker_model_family_profile(
    samples: list[BenchmarkSample],
    *,
    attacker_model_family: str,
) -> list[BenchmarkSample]:
    family = str(attacker_model_family).strip()
    mapped_family = {
        "gpt_like": "baseline_attacker",
        "claude_like": "formal_attacker",
        "qwen_like": "hidden_information_attacker",
    }.get(family)
    if mapped_family is None:
        raise ValueError(f"Unsupported attacker_model_family: {attacker_model_family!r}.")
    rewritten = apply_attacker_family_profile(samples, attacker_family=mapped_family)
    for sample in rewritten:
        sample.claim.meta["attacker_model_family"] = family
    return rewritten


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
    decision = VerifierPipeline(tool_runner=tool_executor).run(
        sample.claim.claim_text,
        scenario=scenario,
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


def _override_probabilities(
    probabilities: dict[str, Any] | None,
    *,
    forced_label: str,
    minimum_mass: float = 0.56,
) -> dict[str, float]:
    base = {
        VerdictLabel.VALID.value: 1.0 / 3.0,
        VerdictLabel.INVALID.value: 1.0 / 3.0,
        VerdictLabel.UNIDENTIFIABLE.value: 1.0 / 3.0,
    }
    if isinstance(probabilities, dict):
        for label in base:
            try:
                base[label] = max(0.0, float(probabilities.get(label, 0.0)))
            except (TypeError, ValueError):
                base[label] = 0.0

    labels = list(base)
    if forced_label not in labels:
        raise ValueError(f"Unsupported forced label {forced_label!r}.")

    other_labels = [label for label in labels if label != forced_label]
    other_total = sum(base[label] for label in other_labels)
    target_mass = max(float(minimum_mass), max(base.values()) + 1e-6)
    target_mass = min(0.94, target_mass)
    remainder = max(0.0, 1.0 - target_mass)
    adjusted = {label: 0.0 for label in labels}
    adjusted[forced_label] = target_mass
    if other_total <= 0:
        share = remainder / len(other_labels)
        for label in other_labels:
            adjusted[label] = share
        return adjusted
    for label in other_labels:
        adjusted[label] = remainder * (base[label] / other_total)
    return adjusted


def _run_no_tools_verifier(sample: BenchmarkSample) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    tool_context = _verifier_tool_context(sample)
    decision = VerifierPipeline(tool_runner=lambda **_: []).run(
        sample.claim.claim_text,
        scenario=scenario,
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


def _label_probabilities(label: str, confidence: float) -> dict[str, float]:
    other_labels = [
        VerdictLabel.VALID.value,
        VerdictLabel.INVALID.value,
        VerdictLabel.UNIDENTIFIABLE.value,
    ]
    probabilities = {candidate: 0.0 for candidate in other_labels}
    clipped_confidence = max(0.0, min(1.0, float(confidence)))
    remainder = max(0.0, 1.0 - clipped_confidence)
    spill = remainder / 2.0
    for candidate in probabilities:
        probabilities[candidate] = spill
    probabilities[str(label)] = clipped_confidence
    return probabilities


def _canonicalize_verdict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return SelectiveVerifierOutput.from_decision_payload(payload).to_dict()


def _run_tool_only_baseline(sample: BenchmarkSample) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    tool_context = _verifier_tool_context(sample)
    tool_executor = ToolExecutor({})
    tool_report = tool_executor.execute_for_claim(
        scenario=scenario,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    tool_trace = list(tool_report["tool_trace"])
    support_hits = sum(
        1
        for record in tool_trace
        if bool(record.get("supports_primary_claim")) or str(record.get("evidence_direction", "")).strip().lower() == "support"
    )
    contradiction_hits = sum(
        1
        for record in tool_trace
        if record.get("contradicts_assumptions")
        or str(record.get("evidence_direction", "")).strip().lower() == "counter"
    )
    if contradiction_hits > support_hits and contradiction_hits > 0:
        label = VerdictLabel.INVALID.value
        confidence = min(0.82, 0.61 + (0.06 * contradiction_hits))
    elif support_hits > 0 and contradiction_hits == 0:
        label = VerdictLabel.VALID.value
        confidence = min(0.8, 0.62 + (0.05 * support_hits))
    else:
        label = VerdictLabel.UNIDENTIFIABLE.value
        confidence = 0.58
    return {
        "predicted_label": label,
        "confidence": confidence,
        "verdict": _canonicalize_verdict_payload(
            {
                "label": label,
                "confidence": confidence,
                "probabilities": _label_probabilities(label, confidence),
                "assumption_ledger": [],
                "witness": None,
                "support_witness": None,
                "countermodel_witness": None,
                "tool_trace": tool_trace,
                "reasoning_summary": (
                    "Tool-only baseline judges the claim from public tool traces alone, without an "
                    "assumption ledger or countermodel search."
                ),
                "metadata": {
                    "baseline_category": "Tool",
                    "baseline_system": "tool_only",
                },
            }
        ),
        "tool_report": {
            "selected_tools": list(tool_report["selected_tools"]),
            "claim_stance": tool_report["claim_stance"],
            "identified_issues": list(tool_report["identified_issues"]),
            "supporting_evidence": list(tool_report["supporting_evidence"]),
            "counter_evidence": list(tool_report["counter_evidence"]),
            "tool_trace": tool_trace,
        },
        "countermodel_found": False,
        "countermodel_type": None,
        "supports_public_only": True,
        "system_notes": ["baseline_category:Tool", "tool_only"],
    }


def _run_judge_direct_baseline(sample: BenchmarkSample) -> dict[str, Any]:
    payload = _run_claim_only_family(sample)
    payload["verdict"]["metadata"] = {
        **dict(payload["verdict"].get("metadata", {})),
        "baseline_category": "Judge",
        "baseline_system": "judge_direct",
    }
    payload["system_notes"] = ["baseline_category:Judge", "judge_direct"]
    return payload


def _run_debate_reduced_baseline(sample: BenchmarkSample) -> dict[str, Any]:
    scenario = _coerce_public_instance(sample)
    proposer = _run_judge_direct_baseline(sample)
    parsed_claim = parse_claim(sample.claim.claim_text)
    ledger = build_assumption_ledger(parsed_claim)
    tool_context = _verifier_tool_context(sample)
    tool_executor = ToolExecutor({})
    tool_report = tool_executor.execute_for_claim(
        scenario=scenario,
        claim=sample.claim.claim_text,
        level=int(sample.claim.causal_level[1]),
        context=tool_context,
    )
    tool_trace = list(tool_report["tool_trace"])
    contradicted = any(entry.status.value == "contradicted" for entry in ledger.entries) or any(
        record.get("contradicts_assumptions") for record in tool_trace
    )
    unresolved_nonbaseline = [
        entry
        for entry in ledger.entries
        if entry.status.value != "supported" and entry.name not in {"consistency", "positivity"}
    ]
    if contradicted:
        rebuttal_label = VerdictLabel.INVALID.value
        rebuttal_reason = "the rebuttal surfaces an explicit identification conflict"
    elif unresolved_nonbaseline:
        rebuttal_label = VerdictLabel.UNIDENTIFIABLE.value
        rebuttal_reason = "the rebuttal keeps core identifying assumptions unresolved"
    else:
        rebuttal_label = VerdictLabel.VALID.value
        rebuttal_reason = "the rebuttal fails to overturn the proposer's claim"

    proposer_label = str(proposer["predicted_label"])
    if proposer_label == rebuttal_label:
        label = proposer_label
    elif rebuttal_label == VerdictLabel.INVALID.value and proposer_label == VerdictLabel.VALID.value:
        label = VerdictLabel.INVALID.value
    elif VerdictLabel.UNIDENTIFIABLE.value in {proposer_label, rebuttal_label}:
        label = VerdictLabel.UNIDENTIFIABLE.value
    else:
        label = proposer_label
    confidence = {
        VerdictLabel.VALID.value: 0.64,
        VerdictLabel.INVALID.value: 0.67,
        VerdictLabel.UNIDENTIFIABLE.value: 0.6,
    }[label]
    return {
        "predicted_label": label,
        "confidence": confidence,
        "verdict": _canonicalize_verdict_payload(
            {
                "label": label,
                "confidence": confidence,
                "probabilities": _label_probabilities(label, confidence),
                "assumption_ledger": [entry.to_dict() for entry in ledger.entries],
                "witness": None,
                "support_witness": None,
                "countermodel_witness": None,
                "tool_trace": tool_trace,
                "reasoning_summary": (
                    "Reduced debate baseline combines a proposer-style direct judge with a rebuttal pass over "
                    f"the public evidence; here the rebuttal argues that {rebuttal_reason}."
                ),
                "metadata": {
                    "baseline_category": "Debate",
                    "baseline_system": "debate_reduced",
                    "proposer_label": proposer_label,
                    "rebuttal_label": rebuttal_label,
                },
            }
        ),
        "tool_report": {
            "selected_tools": list(tool_report["selected_tools"]),
            "claim_stance": tool_report["claim_stance"],
            "identified_issues": list(tool_report["identified_issues"]),
            "supporting_evidence": list(tool_report["supporting_evidence"]),
            "counter_evidence": list(tool_report["counter_evidence"]),
            "tool_trace": tool_trace,
        },
        "countermodel_found": False,
        "countermodel_type": None,
        "supports_public_only": True,
        "system_notes": ["baseline_category:Debate", "debate_reduced"],
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
    base_label = str(verdict.get("label", payload["predicted_label"]))
    if base_label == VerdictLabel.UNIDENTIFIABLE.value:
        parsed_claim = parse_claim(sample.claim.claim_text)
        forced_label = (
            VerdictLabel.INVALID.value
            if parsed_claim.claim_polarity is ClaimPolarity.NEGATIVE
            else VerdictLabel.VALID.value
        )
        base_probabilities = dict(verdict.get("probabilities", {}))
        base_reasoning_summary = str(verdict.get("reasoning_summary", "")).strip()
        base_witness = deepcopy(verdict.get("witness"))
        base_support_witness = deepcopy(verdict.get("support_witness"))
        base_countermodel_witness = deepcopy(verdict.get("countermodel_witness"))
        verdict["label"] = forced_label
        verdict["confidence"] = max(float(payload["confidence"]), 0.51)
        verdict["probabilities"] = _override_probabilities(
            base_probabilities,
            forced_label=forced_label,
        )
        verdict["witness"] = None
        verdict["support_witness"] = None
        verdict["countermodel_witness"] = None
        verdict["identification_status"] = (
            "contradicted" if forced_label == VerdictLabel.INVALID.value else "identified"
        )
        verdict["refusal_reason"] = None
        verdict["missing_information_spec"] = {}
        verdict["reasoning_summary"] = (
            "No-abstention ablation forced a committed "
            f"{forced_label} verdict after the base verifier abstained. "
            f"Base rationale: {base_reasoning_summary}"
        ).strip()
        verdict["metadata"] = {
            **dict(verdict.get("metadata", {})),
            "ablation": "no_abstention",
            "decision_stage": "no_abstention_override",
            "forced_from": base_label,
            "base_verdict": {
                "label": base_label,
                "probabilities": base_probabilities,
                "reasoning_summary": base_reasoning_summary,
                "witness": base_witness,
                "support_witness": base_support_witness,
                "countermodel_witness": base_countermodel_witness,
            },
        }
        adjusted["predicted_label"] = forced_label
        adjusted["confidence"] = float(verdict["confidence"])
        adjusted["verdict"] = _canonicalize_verdict_payload(verdict)
        adjusted["system_notes"] = list(payload.get("system_notes", [])) + ["abstention_disabled"]
    return adjusted


def _run_claim_only_family(sample: BenchmarkSample) -> dict[str, Any]:
    parsed_claim = parse_claim(sample.claim.claim_text)
    if parsed_claim.claim_polarity is ClaimPolarity.NEGATIVE:
        label = VerdictLabel.INVALID.value
        confidence = 0.67
    elif (
        parsed_claim.claim_strength is ClaimStrength.TENTATIVE
        and parsed_claim.query_type is not QueryType.COUNTERFACTUAL
    ):
        label = VerdictLabel.UNIDENTIFIABLE.value
        confidence = 0.57
    else:
        label = VerdictLabel.VALID.value
        confidence = 0.72 if parsed_claim.claim_strength is ClaimStrength.ABSOLUTE else 0.68
    off_label_mass = max(0.0, 1.0 - float(confidence)) / 2.0
    probabilities = {
        VerdictLabel.VALID.value: off_label_mass,
        VerdictLabel.INVALID.value: off_label_mass,
        VerdictLabel.UNIDENTIFIABLE.value: off_label_mass,
    }
    probabilities[label] = float(confidence)
    return {
        "predicted_label": label,
        "confidence": confidence,
        "verdict": _canonicalize_verdict_payload(
            {
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities,
                "assumption_ledger": [],
                "witness": None,
                "support_witness": None,
                "countermodel_witness": None,
                "tool_trace": [],
                "reasoning_summary": (
                    "Claim-only family ignores the public benchmark tools and defaults to the surface rhetorical force "
                    "of the claim text, so strong positive claims are usually endorsed without an identifiability check."
                ),
                "metadata": {"predictor_family": "claim_only_family"},
            }
        ),
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
    adjusted = deepcopy(payload)
    verdict = dict(adjusted["verdict"])
    label = str(payload["predicted_label"])
    confidence = float(payload["confidence"])
    base_probabilities = dict(verdict.get("probabilities", {}))

    if family_name == "skeptical_family":
        if label == VerdictLabel.VALID.value and confidence < 0.82:
            label = VerdictLabel.UNIDENTIFIABLE.value
            confidence = max(0.61, confidence)
            verdict["witness"] = {
                "witness_type": "assumption",
                "description": "The skeptical family abstains on marginally supported claims when the base verifier lacks a decisive confidence margin.",
                "evidence": [
                    f"base_label={payload['predicted_label']}",
                    f"base_confidence={float(payload['confidence']):.4f}",
                ],
                "assumptions": [],
                "payload": {
                    "base_label": payload["predicted_label"],
                    "base_confidence": float(payload["confidence"]),
                    "family_policy": "skeptical_abstention",
                },
                "verdict_suggestion": VerdictLabel.UNIDENTIFIABLE.value,
                "metadata": {"decision_stage": "skeptical_family_override"},
            }
            verdict["support_witness"] = None
            verdict["countermodel_witness"] = None
            verdict["reasoning_summary"] = (
                "Skeptical family override: the base verifier found no decisive countermodel, "
                "but the remaining support margin is treated as insufficient for a committed valid verdict."
            )
            verdict["probabilities"] = _override_probabilities(
                base_probabilities,
                forced_label=label,
            )
            adjusted["countermodel_found"] = False
            adjusted["countermodel_type"] = None
    elif family_name == "optimistic_family":
        if label == VerdictLabel.UNIDENTIFIABLE.value and verdict.get("countermodel_witness") is None:
            label = VerdictLabel.VALID.value
            confidence = max(0.57, confidence)
            support_witness = {
                "witness_type": "support",
                "description": "The optimistic family resolves residual uncertainty in favor of the claim when no direct countermodel survives.",
                "evidence": [
                    f"base_label={payload['predicted_label']}",
                    f"base_confidence={float(payload['confidence']):.4f}",
                ],
                "assumptions": [],
                "payload": {
                    "base_label": payload["predicted_label"],
                    "base_confidence": float(payload["confidence"]),
                    "family_policy": "optimistic_resolution",
                },
                "verdict_suggestion": VerdictLabel.VALID.value,
                "metadata": {"decision_stage": "optimistic_family_override"},
            }
            verdict["witness"] = support_witness
            verdict["support_witness"] = support_witness
            verdict["countermodel_witness"] = None
            verdict["reasoning_summary"] = (
                "Optimistic family override: no direct countermodel survived, so the remaining "
                "uncertainty is resolved in favor of the claim."
            )
            verdict["probabilities"] = _override_probabilities(
                base_probabilities,
                forced_label=label,
            )
            adjusted["countermodel_found"] = False
            adjusted["countermodel_type"] = None
    elif family_name != "countermodel_grounded":
        raise ValueError(f"Unsupported model family: {family_name!r}.")

    verdict["label"] = label
    verdict["confidence"] = confidence
    if label == VerdictLabel.VALID.value:
        verdict["identification_status"] = "identified"
        verdict["refusal_reason"] = None
        verdict["missing_information_spec"] = {}
    elif label == VerdictLabel.INVALID.value:
        verdict["identification_status"] = "contradicted"
        verdict["refusal_reason"] = None
        verdict["missing_information_spec"] = {}
    else:
        verdict["identification_status"] = "underdetermined"
        verdict.setdefault(
            "missing_information_spec",
            {
                "missing_assumptions": [],
                "required_evidence": [],
                "note": str(verdict.get("reasoning_summary", "")).strip(),
            },
        )
    verdict["metadata"] = {
        **dict(verdict.get("metadata", {})),
        "predictor_family": family_name,
    }
    adjusted["predicted_label"] = label
    adjusted["confidence"] = confidence
    adjusted["verdict"] = _canonicalize_verdict_payload(verdict)
    adjusted["system_notes"] = list(payload.get("system_notes", [])) + [family_name]
    return adjusted


def predict_sample(
    sample: BenchmarkSample,
    *,
    system_name: str,
) -> dict[str, Any]:
    if system_name == "judge_direct":
        return _run_judge_direct_baseline(sample)
    if system_name == "debate_reduced":
        return _run_debate_reduced_baseline(sample)
    if system_name == "tool_only":
        return _run_tool_only_baseline(sample)
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
    if system_name == "claim_only_family":
        return _run_claim_only_family(sample)
    if system_name in {"skeptical_family", "optimistic_family"}:
        return _apply_family_postprocessing(system_name, _run_main_verifier(sample))
    raise ValueError(f"Unsupported system_name: {system_name!r}.")


def predict_sample_for_model_family(
    sample: BenchmarkSample,
    *,
    verifier_model_family: str,
) -> dict[str, Any]:
    family = str(verifier_model_family).strip()
    if family == "gpt_like":
        payload = _run_main_verifier(sample)
    elif family == "claude_like":
        payload = _apply_family_postprocessing("skeptical_family", _run_main_verifier(sample))
    elif family == "qwen_like":
        payload = _apply_family_postprocessing("optimistic_family", _run_main_verifier(sample))
    else:
        raise ValueError(f"Unsupported verifier_model_family: {verifier_model_family!r}.")

    payload["verdict"]["metadata"] = {
        **dict(payload["verdict"].get("metadata", {})),
        "verifier_model_family": family,
    }
    payload["system_notes"] = list(payload.get("system_notes", [])) + [f"verifier_model_family:{family}"]
    return payload


def score_prediction_records(
    predictions: list[dict[str, Any]],
    *,
    game_id: str,
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    for index, record in enumerate(
        sorted(
            predictions,
            key=lambda item: (
                int(item.get("seed", 0)),
                str(item.get("split", "")),
                str(item.get("system_name", "")),
                str(item.get("instance_id", "")),
            ),
        ),
        start=1,
    ):
        verdict = dict(record.get("verdict") or {})
        rounds.append(
            {
                "round_id": index,
                "gold_label": record["gold_label"],
                "predicted_label": record["predicted_label"],
                "verdict_label": record["predicted_label"],
                "verifier_confidence": record["confidence"],
                "predicted_probabilities": verdict.get("probabilities"),
                "countermodel_found": bool(record.get("countermodel_found")),
                "countermodel_witness": verdict.get("countermodel_witness"),
            }
        )

    score = Scorer().score_game(
        {
            "game_id": game_id,
            "rounds": rounds,
        }
    )
    return {
        "predictions": predictions,
        "metrics": dict(score.summary["core_metrics"]),
        "appendix_metrics": dict(score.summary["appendix_metrics"]),
        "summary": dict(score.summary),
    }


def evaluate_system_on_samples(
    samples: list[BenchmarkSample],
    *,
    seed: int,
    split_name: str,
    system_name: str,
    ood_reasons: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    manifest_ood_reasons = dict(ood_reasons or {})

    for index, sample in enumerate(sorted(samples, key=lambda item: item.claim.instance_id), start=1):
        payload = predict_sample(sample, system_name=system_name)
        verdict = dict(payload["verdict"])
        public_payload = sample.public.to_dict()
        record = {
            "seed": int(seed),
            "split": split_name,
            "system_name": system_name,
            "instance_id": sample.claim.instance_id,
            "scenario_id": sample.gold.scenario_id,
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
            "supporting_evidence": list(payload["tool_report"].get("supporting_evidence", [])),
            "counter_evidence": list(payload["tool_report"].get("counter_evidence", [])),
            "countermodel_found": bool(payload["countermodel_found"]),
            "countermodel_type": payload["countermodel_type"],
            "predicted_probabilities": dict(verdict.get("probabilities", {})),
            "public_evidence_summary": {
                "scenario_id": public_payload["scenario_id"],
                "description": public_payload["description"],
                "variables": list(public_payload["variables"]),
                "proxy_variables": list(public_payload["proxy_variables"]),
                "selection_mechanism": public_payload["selection_mechanism"],
                "causal_level": public_payload["causal_level"],
            },
            "observed_data": public_payload["observed_data"],
            "verdict": verdict,
            "system_notes": list(payload["system_notes"]),
        }
        predictions.append(record)
    return score_prediction_records(
        predictions,
        game_id=f"{system_name}_{split_name}_seed_{seed}",
    )


def aggregate_seed_metrics(
    per_seed_payloads: dict[int, dict[str, Any]],
    *,
    split_name: str,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {metric_name: [] for metric_name in PRIMARY_METRICS}
    for seed, seed_payload in sorted(per_seed_payloads.items()):
        metrics = dict(seed_payload[split_name]["metrics"])
        missing_metrics = [
            metric_name
            for metric_name in PRIMARY_METRICS
            if metric_name not in metrics
        ]
        if missing_metrics:
            raise ValueError(
                f"Missing primary metrics for seed={seed}, split={split_name!r}: {missing_metrics!r}."
            )
        for metric_name in PRIMARY_METRICS:
            metric_values[metric_name].append(float(metrics[metric_name]))
    summarized = summarize_metrics(
        metric_values,
        n_resamples=2000,
        random_state=0,
    )
    return {
        metric_name: summary.to_dict()
        for metric_name, summary in summarized.items()
    }


def align_prediction_records(
    system_predictions: dict[str, list[dict[str, Any]]],
    *,
    baseline: str,
) -> tuple[list[Any], dict[str, list[Any]]]:
    if len(system_predictions) < 2:
        raise ValueError("At least two systems are required for aligned paired comparisons.")
    aligned_records: dict[str, dict[tuple[int, str, str], dict[str, Any]]] = {}
    for system_name, records in system_predictions.items():
        indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
        for record in records:
            try:
                key = (
                    int(record["seed"]),
                    str(record["split"]),
                    str(record["instance_id"]),
                )
            except KeyError as exc:
                raise ValueError(
                    "Paired significance requires seed/split/instance_id on every prediction record."
                ) from exc
            if key in indexed:
                raise ValueError(
                    f"Duplicate prediction record for {system_name!r} at sample key {key!r}."
                )
            indexed[key] = record
        aligned_records[system_name] = indexed

    if baseline not in aligned_records:
        raise ValueError(f"Unknown baseline system: {baseline!r}.")

    baseline_keys = set(aligned_records[baseline])
    ordered_keys = sorted(baseline_keys)
    for system_name, indexed in aligned_records.items():
        current_keys = set(indexed)
        if current_keys != baseline_keys:
            missing = sorted(baseline_keys - current_keys)[:3]
            extra = sorted(current_keys - baseline_keys)[:3]
            raise ValueError(
                "Paired significance requires identical sample identities across systems. "
                f"{system_name!r} is missing {missing!r} and has extra {extra!r}."
            )

    truth = [aligned_records[baseline][key]["gold_label"] for key in ordered_keys]
    predictions: dict[str, list[Any]] = {}
    for system_name, indexed in aligned_records.items():
        aligned_system_predictions: list[Any] = []
        for key in ordered_keys:
            record = indexed[key]
            if record["gold_label"] != aligned_records[baseline][key]["gold_label"]:
                raise ValueError(
                    "Paired significance requires identical gold labels for each aligned sample. "
                    f"Mismatch detected for {system_name!r} at sample key {key!r}."
                )
            aligned_system_predictions.append(record["predicted_label"])
        predictions[system_name] = aligned_system_predictions
    return truth, predictions


def compare_system_predictions(
    system_predictions: dict[str, list[dict[str, Any]]],
    *,
    baseline: str,
) -> dict[str, Any] | None:
    if len(system_predictions) < 2:
        return None
    truth, predictions = align_prediction_records(
        system_predictions,
        baseline=baseline,
    )
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


def _extract_confidence_interval_view(node: Any) -> Any:
    if isinstance(node, dict):
        metric_keys = {"mean", "std", "ci_lower", "ci_upper", "formatted"}
        if metric_keys.issubset(set(node)):
            return {
                "mean": node["mean"],
                "std": node["std"],
                "ci_lower": node["ci_lower"],
                "ci_upper": node["ci_upper"],
                "formatted": node["formatted"],
            }
        extracted: dict[str, Any] = {}
        for key, value in node.items():
            child = _extract_confidence_interval_view(value)
            if child is not None:
                extracted[str(key)] = child
        return extracted or None
    return None


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

    config_path = json_path.with_name(f"{json_path.stem}_config.json")
    config_path.write_text(
        json.dumps(
            {
                "effective": payload.get("config"),
                "requested": payload.get("requested_config"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    seed_list_path = json_path.with_name(f"{json_path.stem}_seed_list.json")
    seed_list_path.write_text(
        json.dumps({"seeds": list(payload.get("seeds", []))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aggregated_metrics_path = json_path.with_name(f"{json_path.stem}_aggregated_metrics.json")
    aggregated_metrics_path.write_text(
        json.dumps(payload.get("aggregated_metrics", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ci_path = json_path.with_name(f"{json_path.stem}_ci.json")
    ci_payload = _extract_confidence_interval_view(payload.get("aggregated_metrics", {})) or {}
    ci_path.write_text(
        json.dumps(ci_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "artifact_json": str(json_path),
        "raw_predictions": str(raw_predictions_path),
        "markdown_summary": str(markdown_path),
        "config": str(config_path),
        "seed_list": str(seed_list_path),
        "aggregated_metrics": str(aggregated_metrics_path),
        "ci": str(ci_path),
    }


def manifest_metadata(run: SeedBenchmarkRun) -> dict[str, Any]:
    extended_holdout_strategy = dict(run.manifest.metadata.get("extended_holdout_strategy", {}))
    return {
        "dataset_name": run.manifest.dataset_name,
        "version": run.manifest.version,
        "holdout_strategy": {
            "family_holdout": list(run.manifest.family_holdout),
            "lexical_holdout": list(run.manifest.lexical_holdout),
            "variable_renaming_holdout": bool(run.manifest.variable_renaming_holdout),
            **extended_holdout_strategy,
        },
        "metadata": dict(run.manifest.metadata),
    }
