"""Graph family templates for the benchmark generator.

This module provides deterministic, seed-controlled structural blueprints that
can be reused by future benchmark generators. Each family captures a recurring
causal pattern and explicitly marks whether the family is structurally
identifiable or potentially unidentifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random
from typing import Any, Callable, ClassVar


CAUSAL_LEVEL_SPACE: tuple[str, ...] = ("L1", "L2", "L3")


class IdentifiabilityStatus(str, Enum):
    """Family-level identifiability tag used by the benchmark generator."""

    IDENTIFIABLE = "identifiable"
    POTENTIALLY_UNIDENTIFIABLE = "potentially_unidentifiable"


IDENTIFIABILITY_SPACE: tuple[str, ...] = tuple(status.value for status in IdentifiabilityStatus)

_ROLE_NAME_POOLS: dict[str, tuple[str, ...]] = {
    "treatment": (
        "exposure",
        "training_intensity",
        "dose",
        "policy_uptake",
        "therapy_flag",
        "adoption_level",
    ),
    "outcome": (
        "recovery",
        "income_score",
        "response_quality",
        "completion_rate",
        "yield_score",
        "stability_index",
    ),
    "observed_context": (
        "age_band",
        "baseline_score",
        "site_load",
        "prior_history",
        "ability_index",
        "cohort_signal",
    ),
    "observed_adjuster": (
        "pretest_score",
        "risk_bucket",
        "eligibility_score",
        "market_pressure",
        "clinical_stage",
        "need_index",
    ),
    "latent_confounder": (
        "latent_risk",
        "background_advantage",
        "demand_shock",
        "response_type",
        "genetic_liability",
        "hidden_motivation",
    ),
    "proxy": (
        "proxy_signal",
        "screening_trace",
        "sensor_proxy",
        "triage_note",
        "archive_indicator",
        "surrogate_measure",
    ),
    "selection": (
        "enrollment_gate",
        "screening_pass",
        "recorded_flag",
        "clinic_selection",
        "audit_inclusion",
        "portal_entry",
    ),
    "instrument": (
        "assignment_lottery",
        "calendar_shift",
        "quota_push",
        "distance_offer",
        "encouragement_signal",
        "rollout_wave",
    ),
    "mediator": (
        "biomarker",
        "uptake_level",
        "engagement",
        "intermediate_state",
        "mechanism_trace",
        "dosage_response",
    ),
}


def _stable_rng(family_name: str, seed: int) -> random.Random:
    material = f"{family_name}:{int(seed)}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def _sample_unique_name(
    rng: random.Random,
    role: str,
    used: set[str],
) -> str:
    candidates = list(_ROLE_NAME_POOLS[role])
    rng.shuffle(candidates)
    for candidate in candidates:
        if candidate not in used:
            used.add(candidate)
            return candidate
    fallback = f"{role}_{len(used)}"
    used.add(fallback)
    return fallback


def _build_dag(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    dag = {node: [] for node in nodes}
    for parent, child in edges:
        dag.setdefault(parent, [])
        dag.setdefault(child, [])
        dag[parent].append(child)
    for children in dag.values():
        children.sort()
    return dag


def _normalize_str_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {str(key): item for key, item in value.items()}


def _serialize_json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _serialize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_safe(item) for item in value]
    return value


def _coerce_causal_level(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().upper()
    if normalized in CAUSAL_LEVEL_SPACE:
        return normalized
    raise ValueError(f"{field_name} must be one of {CAUSAL_LEVEL_SPACE}, got {value!r}.")


def _coerce_identifiability(value: IdentifiabilityStatus | str) -> IdentifiabilityStatus:
    if isinstance(value, IdentifiabilityStatus):
        return value
    try:
        return IdentifiabilityStatus(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            "identifiability must be one of "
            f"{IDENTIFIABILITY_SPACE}, got {value!r}."
        ) from exc


@dataclass(slots=True)
class GraphFamilyBlueprint:
    """Reusable structural template emitted by a graph family."""

    family_name: str
    causal_level: str
    identifiability: IdentifiabilityStatus | str
    description: str
    true_dag: dict[str, list[str]]
    observed_variables: list[str]
    hidden_variables: list[str]
    target_variables: dict[str, str]
    seed: int = 0
    proxy_variables: list[str] = field(default_factory=list)
    selection_variables: list[str] = field(default_factory=list)
    role_bindings: dict[str, str] = field(default_factory=dict)
    query_types: list[str] = field(default_factory=list)
    supported_gold_labels: list[str] = field(default_factory=list)
    family_tags: list[str] = field(default_factory=list)
    generator_hints: dict[str, Any] = field(default_factory=dict)

    causal_level_space: ClassVar[tuple[str, ...]] = CAUSAL_LEVEL_SPACE
    identifiability_space: ClassVar[tuple[str, ...]] = IDENTIFIABILITY_SPACE

    def __post_init__(self) -> None:
        self.family_name = str(self.family_name)
        self.causal_level = _coerce_causal_level(self.causal_level, field_name="causal_level")
        self.identifiability = _coerce_identifiability(self.identifiability)
        self.description = str(self.description)
        self.true_dag = {
            str(parent): _normalize_str_list(children)
            for parent, children in dict(self.true_dag).items()
        }
        self.observed_variables = _normalize_str_list(self.observed_variables)
        self.hidden_variables = _normalize_str_list(self.hidden_variables)
        self.target_variables = {
            str(key): str(value)
            for key, value in _normalize_mapping(self.target_variables).items()
        }
        self.seed = int(self.seed)
        self.proxy_variables = _normalize_str_list(self.proxy_variables)
        self.selection_variables = _normalize_str_list(self.selection_variables)
        self.role_bindings = {
            str(key): str(value)
            for key, value in _normalize_mapping(self.role_bindings).items()
        }
        self.query_types = _normalize_str_list(self.query_types)
        self.supported_gold_labels = _normalize_str_list(self.supported_gold_labels)
        self.family_tags = _normalize_str_list(self.family_tags)
        self.generator_hints = dict(self.generator_hints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "causal_level": self.causal_level,
            "identifiability": self.identifiability.value,
            "description": self.description,
            "true_dag": {parent: list(children) for parent, children in self.true_dag.items()},
            "observed_variables": list(self.observed_variables),
            "hidden_variables": list(self.hidden_variables),
            "target_variables": dict(self.target_variables),
            "seed": self.seed,
            "proxy_variables": list(self.proxy_variables),
            "selection_variables": list(self.selection_variables),
            "role_bindings": dict(self.role_bindings),
            "query_types": list(self.query_types),
            "supported_gold_labels": list(self.supported_gold_labels),
            "family_tags": list(self.family_tags),
            "generator_hints": _serialize_json_safe(self.generator_hints),
        }


FamilyBuilder = Callable[[int], GraphFamilyBlueprint]


@dataclass(frozen=True, slots=True)
class GraphFamilyTemplate:
    """Registry entry for a deterministic graph family."""

    family_name: str
    causal_level: str
    identifiability: IdentifiabilityStatus
    description: str
    builder: FamilyBuilder

    def build(self, seed: int = 0) -> GraphFamilyBlueprint:
        blueprint = self.builder(int(seed))
        if blueprint.family_name != self.family_name:
            raise ValueError(
                f"Builder for {self.family_name!r} returned {blueprint.family_name!r}."
            )
        return blueprint


def _make_blueprint(
    *,
    family_name: str,
    causal_level: str,
    identifiability: IdentifiabilityStatus,
    description: str,
    seed: int,
    nodes: list[str],
    edges: list[tuple[str, str]],
    observed_variables: list[str],
    hidden_variables: list[str],
    target_variables: dict[str, str],
    proxy_variables: list[str] | None = None,
    selection_variables: list[str] | None = None,
    role_bindings: dict[str, str] | None = None,
    query_types: list[str] | None = None,
    supported_gold_labels: list[str] | None = None,
    family_tags: list[str] | None = None,
    generator_hints: dict[str, Any] | None = None,
) -> GraphFamilyBlueprint:
    return GraphFamilyBlueprint(
        family_name=family_name,
        causal_level=causal_level,
        identifiability=identifiability,
        description=description,
        true_dag=_build_dag(nodes, edges),
        observed_variables=observed_variables,
        hidden_variables=hidden_variables,
        target_variables=target_variables,
        seed=seed,
        proxy_variables=proxy_variables or [],
        selection_variables=selection_variables or [],
        role_bindings=role_bindings or {},
        query_types=query_types or [],
        supported_gold_labels=supported_gold_labels or [],
        family_tags=family_tags or [],
        generator_hints=generator_hints or {},
    )


def _build_l1_latent_confounding_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l1_latent_confounding_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    context = _sample_unique_name(rng, "observed_context", used)
    latent = _sample_unique_name(rng, "latent_confounder", used)
    proxy = _sample_unique_name(rng, "proxy", used)

    edges = [
        (latent, treatment),
        (latent, outcome),
        (context, treatment),
        (context, outcome),
        (treatment, outcome),
    ]
    if rng.random() < 0.5:
        edges.append((latent, proxy))
        observed_variables = [context, treatment, outcome, proxy]
        proxy_variables = [proxy]
    else:
        observed_variables = [context, treatment, outcome]
        proxy_variables = []

    return _make_blueprint(
        family_name="l1_latent_confounding_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Association-level family with hidden confounding that can mimic a clean causal pattern.",
        seed=seed,
        nodes=observed_variables + [latent],
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[latent],
        target_variables={"treatment": treatment, "outcome": outcome},
        proxy_variables=proxy_variables,
        role_bindings={
            "treatment": treatment,
            "outcome": outcome,
            "observed_context": context,
            "latent_confounder": latent,
            **({"proxy": proxy} if proxy_variables else {}),
        },
        query_types=["association_strength", "causal_direction"],
        supported_gold_labels=["invalid", "unidentifiable"],
        family_tags=["l1", "association", "hidden_confounding"],
        generator_hints={
            "attack_modes": ["association_overclaim", "hidden_confounder_denial"],
            "observational_equivalence_risk": True,
            "preferred_observed_subset": list(observed_variables),
        },
    )


def _build_l1_selection_bias_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l1_selection_bias_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    context = _sample_unique_name(rng, "observed_context", used)
    selection = _sample_unique_name(rng, "selection", used)

    edges = [
        (context, treatment),
        (context, outcome),
        (treatment, selection),
        (outcome, selection),
    ]
    if rng.random() < 0.5:
        adjuster = _sample_unique_name(rng, "observed_adjuster", used)
        edges.append((adjuster, outcome))
        observed_variables = [context, adjuster, treatment, outcome, selection]
        role_bindings = {
            "treatment": treatment,
            "outcome": outcome,
            "observed_context": context,
            "selection": selection,
            "observed_adjuster": adjuster,
        }
    else:
        observed_variables = [context, treatment, outcome, selection]
        role_bindings = {
            "treatment": treatment,
            "outcome": outcome,
            "observed_context": context,
            "selection": selection,
        }

    return _make_blueprint(
        family_name="l1_selection_bias_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Selection/collider family where observed association can be induced by conditioning on sample inclusion.",
        seed=seed,
        nodes=observed_variables,
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[],
        target_variables={"treatment": treatment, "outcome": outcome},
        selection_variables=[selection],
        role_bindings=role_bindings,
        query_types=["association_strength", "selection_bias_check"],
        supported_gold_labels=["invalid", "unidentifiable"],
        family_tags=["l1", "association", "selection_bias", "collider"],
        generator_hints={
            "attack_modes": ["selection_bias_obfuscation", "association_overclaim"],
            "conditioning_variables": [selection],
            "selection_mechanism": "conditioning_on_collider",
        },
    )


def _build_l1_proxy_disambiguation_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l1_proxy_disambiguation_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    adjuster = _sample_unique_name(rng, "observed_adjuster", used)
    proxy = _sample_unique_name(rng, "proxy", used)

    edges = [
        (adjuster, treatment),
        (adjuster, outcome),
        (adjuster, proxy),
        (treatment, outcome),
    ]
    if rng.random() < 0.5:
        context = _sample_unique_name(rng, "observed_context", used)
        edges.append((context, treatment))
        observed_variables = [context, adjuster, proxy, treatment, outcome]
        role_bindings = {
            "treatment": treatment,
            "outcome": outcome,
            "observed_adjuster": adjuster,
            "proxy": proxy,
            "observed_context": context,
        }
    else:
        observed_variables = [adjuster, proxy, treatment, outcome]
        role_bindings = {
            "treatment": treatment,
            "outcome": outcome,
            "observed_adjuster": adjuster,
            "proxy": proxy,
        }

    return _make_blueprint(
        family_name="l1_proxy_disambiguation_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Association-level family with observed proxy structure that supports disambiguation instead of pure correlation chasing.",
        seed=seed,
        nodes=observed_variables,
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[],
        target_variables={"treatment": treatment, "outcome": outcome},
        proxy_variables=[proxy],
        role_bindings=role_bindings,
        query_types=["association_strength", "proxy_adjusted_claim"],
        supported_gold_labels=["valid", "invalid"],
        family_tags=["l1", "association", "proxy_assisted"],
        generator_hints={
            "attack_modes": ["association_overclaim"],
            "identifying_variables": [adjuster, proxy],
            "selection_mechanism": "none",
        },
    )


def _build_l2_valid_backdoor_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l2_valid_backdoor_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    adjuster = _sample_unique_name(rng, "observed_adjuster", used)
    context = _sample_unique_name(rng, "observed_context", used)

    edges = [
        (adjuster, treatment),
        (adjuster, outcome),
        (context, treatment),
        (treatment, outcome),
    ]
    observed_variables = [context, adjuster, treatment, outcome]

    if rng.random() < 0.5:
        proxy = _sample_unique_name(rng, "proxy", used)
        edges.append((adjuster, proxy))
        observed_variables.insert(2, proxy)
        proxy_variables = [proxy]
    else:
        proxy_variables = []

    return _make_blueprint(
        family_name="l2_valid_backdoor_family",
        causal_level="L2",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Intervention-level family where an observed backdoor set identifies P(Y|do(X)).",
        seed=seed,
        nodes=observed_variables,
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[],
        target_variables={"treatment": treatment, "outcome": outcome},
        proxy_variables=proxy_variables,
        role_bindings={
            "treatment": treatment,
            "outcome": outcome,
            "backdoor_adjuster": adjuster,
            "observed_context": context,
            **({"proxy": proxy_variables[0]} if proxy_variables else {}),
        },
        query_types=["average_treatment_effect", "interventional_effect"],
        supported_gold_labels=["valid", "invalid"],
        family_tags=["l2", "intervention", "backdoor"],
        generator_hints={
            "attack_modes": ["invalid_adjustment_claim", "heterogeneity_overgeneralization"],
            "identifying_set_candidates": [[adjuster]],
            "invalid_adjustment_candidates": [context, *proxy_variables],
            "selection_mechanism": "none",
        },
    )


def _build_l2_invalid_iv_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l2_invalid_iv_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    instrument = _sample_unique_name(rng, "instrument", used)
    latent = _sample_unique_name(rng, "latent_confounder", used)
    context = _sample_unique_name(rng, "observed_context", used)

    edges = [
        (instrument, treatment),
        (instrument, outcome),
        (latent, treatment),
        (latent, outcome),
        (context, treatment),
        (treatment, outcome),
    ]
    observed_variables = [context, instrument, treatment, outcome]

    return _make_blueprint(
        family_name="l2_invalid_iv_family",
        causal_level="L2",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Intervention-level family with an invalid instrument that violates exclusion or independence.",
        seed=seed,
        nodes=observed_variables + [latent],
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[latent],
        target_variables={"treatment": treatment, "outcome": outcome},
        role_bindings={
            "treatment": treatment,
            "outcome": outcome,
            "instrument": instrument,
            "latent_confounder": latent,
            "observed_context": context,
        },
        query_types=["average_treatment_effect", "instrumental_variable_claim"],
        supported_gold_labels=["invalid", "unidentifiable"],
        family_tags=["l2", "intervention", "instrument", "invalid_iv"],
        generator_hints={
            "attack_modes": ["weak_iv_as_valid_iv", "invalid_iv_exclusion_claim"],
            "invalidity_reason": "instrument_directly_affects_outcome",
            "selection_mechanism": "none",
        },
    )


def _build_l3_counterfactual_ambiguity_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l3_counterfactual_ambiguity_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    mediator = _sample_unique_name(rng, "mediator", used)
    latent = _sample_unique_name(rng, "latent_confounder", used)
    context = _sample_unique_name(rng, "observed_context", used)

    edges = [
        (latent, mediator),
        (latent, outcome),
        (context, treatment),
        (treatment, mediator),
        (mediator, outcome),
        (treatment, outcome),
    ]
    observed_variables = [context, treatment, mediator, outcome]

    return _make_blueprint(
        family_name="l3_counterfactual_ambiguity_family",
        causal_level="L3",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Counterfactual family where multiple SCMs can match observations but disagree on unit-level counterfactual answers.",
        seed=seed,
        nodes=observed_variables + [latent],
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[latent],
        target_variables={"treatment": treatment, "outcome": outcome},
        role_bindings={
            "treatment": treatment,
            "outcome": outcome,
            "mediator": mediator,
            "latent_response_type": latent,
            "observed_context": context,
        },
        query_types=["unit_level_counterfactual", "effect_of_treatment_on_treated"],
        supported_gold_labels=["unidentifiable", "invalid"],
        family_tags=["l3", "counterfactual", "ambiguity"],
        generator_hints={
            "attack_modes": ["counterfactual_overclaim", "unidentifiable_disguised_as_valid"],
            "requires_countermodel_search": True,
            "selection_mechanism": "none",
        },
    )


def _build_l3_mediation_abduction_family(seed: int) -> GraphFamilyBlueprint:
    rng = _stable_rng("l3_mediation_abduction_family", seed)
    used: set[str] = set()
    treatment = _sample_unique_name(rng, "treatment", used)
    outcome = _sample_unique_name(rng, "outcome", used)
    mediator = _sample_unique_name(rng, "mediator", used)
    adjuster = _sample_unique_name(rng, "observed_adjuster", used)
    context = _sample_unique_name(rng, "observed_context", used)

    edges = [
        (adjuster, treatment),
        (adjuster, outcome),
        (context, treatment),
        (treatment, mediator),
        (mediator, outcome),
        (treatment, outcome),
    ]
    observed_variables = [context, adjuster, treatment, mediator, outcome]

    return _make_blueprint(
        family_name="l3_mediation_abduction_family",
        causal_level="L3",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Counterfactual family with measured mediating state and observed confounding, enabling abduction-action style reasoning.",
        seed=seed,
        nodes=observed_variables,
        edges=edges,
        observed_variables=observed_variables,
        hidden_variables=[],
        target_variables={"treatment": treatment, "outcome": outcome},
        role_bindings={
            "treatment": treatment,
            "outcome": outcome,
            "mediator": mediator,
            "observed_adjuster": adjuster,
            "observed_context": context,
        },
        query_types=["unit_level_counterfactual", "abduction_action_prediction"],
        supported_gold_labels=["valid", "invalid"],
        family_tags=["l3", "counterfactual", "mediation"],
        generator_hints={
            "attack_modes": ["counterfactual_overclaim", "function_form_manipulation"],
            "requires_countermodel_search": False,
            "identifying_assumptions": ["consistency", "measured_confounding", "stable_mediation"],
        },
    )


GRAPH_FAMILY_REGISTRY: dict[str, GraphFamilyTemplate] = {
    "l1_latent_confounding_family": GraphFamilyTemplate(
        family_name="l1_latent_confounding_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Association family with latent confounding.",
        builder=_build_l1_latent_confounding_family,
    ),
    "l1_selection_bias_family": GraphFamilyTemplate(
        family_name="l1_selection_bias_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Association family with selection/collider bias.",
        builder=_build_l1_selection_bias_family,
    ),
    "l1_proxy_disambiguation_family": GraphFamilyTemplate(
        family_name="l1_proxy_disambiguation_family",
        causal_level="L1",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Association family with proxy-assisted disambiguation.",
        builder=_build_l1_proxy_disambiguation_family,
    ),
    "l2_valid_backdoor_family": GraphFamilyTemplate(
        family_name="l2_valid_backdoor_family",
        causal_level="L2",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Intervention family with a valid observed backdoor adjustment set.",
        builder=_build_l2_valid_backdoor_family,
    ),
    "l2_invalid_iv_family": GraphFamilyTemplate(
        family_name="l2_invalid_iv_family",
        causal_level="L2",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Intervention family with an invalid instrument.",
        builder=_build_l2_invalid_iv_family,
    ),
    "l3_counterfactual_ambiguity_family": GraphFamilyTemplate(
        family_name="l3_counterfactual_ambiguity_family",
        causal_level="L3",
        identifiability=IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE,
        description="Counterfactual family with observational equivalence but counterfactual disagreement.",
        builder=_build_l3_counterfactual_ambiguity_family,
    ),
    "l3_mediation_abduction_family": GraphFamilyTemplate(
        family_name="l3_mediation_abduction_family",
        causal_level="L3",
        identifiability=IdentifiabilityStatus.IDENTIFIABLE,
        description="Counterfactual family with observed mediation and abduction-friendly structure.",
        builder=_build_l3_mediation_abduction_family,
    ),
}


def list_graph_families(
    *,
    causal_level: str | None = None,
    identifiability: IdentifiabilityStatus | str | None = None,
) -> list[str]:
    """List registered family names with optional filtering."""

    normalized_level = None
    if causal_level is not None:
        normalized_level = _coerce_causal_level(causal_level, field_name="causal_level")
    normalized_identifiability = None
    if identifiability is not None:
        normalized_identifiability = _coerce_identifiability(identifiability)

    result: list[str] = []
    for family_name, template in GRAPH_FAMILY_REGISTRY.items():
        if normalized_level is not None and template.causal_level != normalized_level:
            continue
        if (
            normalized_identifiability is not None
            and template.identifiability is not normalized_identifiability
        ):
            continue
        result.append(family_name)
    return sorted(result)


def get_graph_family_template(family_name: str) -> GraphFamilyTemplate:
    """Return a registry template by name."""

    try:
        return GRAPH_FAMILY_REGISTRY[str(family_name)]
    except KeyError as exc:
        known = ", ".join(sorted(GRAPH_FAMILY_REGISTRY))
        raise KeyError(f"Unknown graph family {family_name!r}. Known families: {known}.") from exc


def generate_graph_family(family_name: str, seed: int = 0) -> GraphFamilyBlueprint:
    """Generate a deterministic family blueprint by name and seed."""

    return get_graph_family_template(family_name).build(seed=seed)


def build_graph_family(family_name: str, seed: int = 0) -> GraphFamilyBlueprint:
    """Alias kept for readability at generator call sites."""

    return generate_graph_family(family_name, seed=seed)
