"""Witness generation for adversarial causal oversight benchmark samples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

from benchmark.graph_families import GraphFamilyBlueprint, generate_graph_family
from benchmark.schema import VerdictLabel, Witness, WitnessKind


ASSUMPTION_STATUS_SPACE: tuple[str, ...] = ("supported", "contradicted", "unresolved")


def _stable_rng(*parts: Any) -> random.Random:
    material = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def _coerce_label(value: VerdictLabel | str) -> VerdictLabel:
    if isinstance(value, VerdictLabel):
        return value
    return VerdictLabel(str(value).strip().lower())


def _resolve_blueprint(
    family: GraphFamilyBlueprint | str,
    *,
    seed: int,
) -> GraphFamilyBlueprint:
    if isinstance(family, GraphFamilyBlueprint):
        return family
    return generate_graph_family(str(family), seed=seed)


def _non_target_observed_variables(blueprint: GraphFamilyBlueprint) -> list[str]:
    targets = set(blueprint.target_variables.values())
    return [variable for variable in blueprint.observed_variables if variable not in targets]


def _support_anchor_variables(blueprint: GraphFamilyBlueprint) -> list[str]:
    anchors: list[str] = []
    for role in ("backdoor_adjuster", "observed_adjuster", "proxy", "instrument", "mediator", "selection"):
        value = blueprint.role_bindings.get(role)
        if value and value not in anchors:
            anchors.append(value)
    for candidate in _non_target_observed_variables(blueprint):
        if candidate not in anchors:
            anchors.append(candidate)
    return anchors


def _resolve_query_hint(blueprint: GraphFamilyBlueprint, query_type: str | None) -> str:
    if query_type is not None:
        normalized = str(query_type).strip()
        if normalized:
            return normalized
    return blueprint.query_types[0] if blueprint.query_types else "causal claim"


def _select_support_type(blueprint: GraphFamilyBlueprint) -> str:
    if "instrument" in blueprint.role_bindings:
        return "instrument_signal"
    if blueprint.role_bindings.get("backdoor_adjuster") or blueprint.role_bindings.get("observed_adjuster"):
        return "adjustment_support"
    if blueprint.proxy_variables:
        return "proxy_support"
    if "mediator" in blueprint.role_bindings:
        return "mechanism_support"
    if blueprint.selection_variables or blueprint.role_bindings.get("selection"):
        return "selected_sample_pattern"
    return "observed_pattern"


def _countermodel_type(blueprint: GraphFamilyBlueprint) -> str:
    if blueprint.selection_variables or blueprint.role_bindings.get("selection"):
        return "selection_mechanism_candidate"
    if "instrument" in blueprint.role_bindings:
        return "invalid_instrument_alternative"
    if blueprint.hidden_variables and blueprint.causal_level in {"L1", "L2"}:
        return "latent_confounder_injection"
    if blueprint.causal_level == "L3":
        return "observationally_equivalent_countermodel"
    return "alternative_structural_explanation"


def _base_assumption_specs(blueprint: GraphFamilyBlueprint) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []

    def add(name: str, source: str, note: str) -> None:
        if any(existing["name"] == name for existing in specs):
            return
        specs.append({"name": name, "source": source, "note": note})

    add("consistency", "tool requirement", "Potential outcomes and interventions refer to the same treatment variable.")
    if blueprint.causal_level in {"L2", "L3"}:
        add("positivity", "tool requirement", "Relevant treatment contrasts are supported by the observed population.")

    if blueprint.hidden_variables:
        add(
            "no unobserved confounding",
            "implicit",
            "The claim assumes no omitted variable still opens a backdoor explanation.",
        )
    if blueprint.selection_variables or blueprint.role_bindings.get("selection"):
        add(
            "no selection bias",
            "implicit",
            "The observed sample should not create collider-induced dependence.",
        )
    if blueprint.role_bindings.get("backdoor_adjuster") or blueprint.role_bindings.get("observed_adjuster"):
        add(
            "valid adjustment set",
            "tool requirement",
            "The claimed adjustment variables must block the relevant backdoor paths.",
        )
    if blueprint.proxy_variables:
        add(
            "proxy sufficiency",
            "claim implied",
            "The measured proxy variables are assumed to carry enough information to support the claimed causal interpretation.",
        )
    if "instrument" in blueprint.role_bindings:
        add(
            "instrument relevance",
            "tool requirement",
            "The proposed instrument must meaningfully move the treatment.",
        )
        add(
            "exclusion restriction",
            "implicit",
            "The proposed instrument should affect the outcome only through the treatment.",
        )
        add(
            "instrument independence",
            "implicit",
            "The instrument should be independent of unblocked outcome determinants.",
        )
    if "mediator" in blueprint.role_bindings:
        add(
            "stable mediation structure",
            "implicit",
            "The mediating mechanism should remain stable across the modeled interventions.",
        )
    if blueprint.causal_level == "L3":
        add(
            "cross-world consistency",
            "implicit",
            "Counterfactual reasoning relies on compatible cross-world assumptions.",
        )
        add(
            "counterfactual model uniqueness",
            "tool requirement",
            "The observational evidence should rule out alternative SCMs that disagree on the query.",
        )
    return specs


def _derive_assumption_status(
    blueprint: GraphFamilyBlueprint,
    *,
    label: VerdictLabel,
    assumption_name: str,
) -> tuple[str, str]:
    family_name = blueprint.family_name

    if label is VerdictLabel.VALID:
        return "supported", "The benchmark label treats this assumption as adequately backed for the target query."

    if label is VerdictLabel.INVALID:
        if assumption_name == "no unobserved confounding" and blueprint.hidden_variables:
            return "contradicted", "A latent-variable explanation remains observationally plausible."
        if assumption_name == "no selection bias" and (
            blueprint.selection_variables or blueprint.role_bindings.get("selection")
        ):
            return "contradicted", "Conditioning on the selected sample can create a spurious association."
        if assumption_name == "valid adjustment set" and "backdoor" in family_name:
            return "contradicted", "The available adjustment story is not sufficient to justify identification."
        if assumption_name == "proxy sufficiency" and blueprint.proxy_variables:
            return "contradicted", (
                "The measured proxy information is not enough to justify the stronger causal conclusion made by the claim."
            )
        if assumption_name == "exclusion restriction" and "invalid_iv" in family_name:
            return "contradicted", "A direct path or dependence violation can explain the claimed IV effect."
        if assumption_name == "instrument independence" and "invalid_iv" in family_name:
            return "contradicted", "The proposed instrument is not cleanly isolated from outcome determinants."
        if assumption_name == "stable mediation structure" and blueprint.causal_level == "L3":
            return "contradicted", "Alternative mechanism classes can flip the counterfactual answer."
        if assumption_name == "cross-world consistency" and blueprint.causal_level == "L3":
            return "contradicted", "The attack relies on cross-world assumptions that are not warranted."
        if assumption_name == "counterfactual model uniqueness" and blueprint.causal_level == "L3":
            return "contradicted", "A conflicting model can fit the observations while disagreeing on the query."
        return "supported", "This assumption is not the main failure mode for the current invalid sample."

    if label is VerdictLabel.UNIDENTIFIABLE:
        if assumption_name in {
            "no unobserved confounding",
            "no selection bias",
            "valid adjustment set",
            "proxy sufficiency",
            "exclusion restriction",
            "instrument independence",
            "cross-world consistency",
            "counterfactual model uniqueness",
        }:
            return "unresolved", "The available evidence does not settle this identification assumption."
        return "supported", "This assumption is not the main source of remaining ambiguity."

    raise ValueError(f"Unsupported label: {label!r}")


def _build_assumption_ledger(
    blueprint: GraphFamilyBlueprint,
    *,
    gold_label: VerdictLabel,
) -> list[dict[str, str]]:
    ledger: list[dict[str, str]] = []
    for spec in _base_assumption_specs(blueprint):
        status, note = _derive_assumption_status(
            blueprint,
            label=gold_label,
            assumption_name=spec["name"],
        )
        ledger.append(
            {
                "name": spec["name"],
                "source": spec["source"],
                "status": status,
                "note": note,
            }
        )
    _ensure_ledger_alignment(ledger, gold_label=gold_label)
    return ledger


def _ensure_ledger_alignment(
    ledger: list[dict[str, str]],
    *,
    gold_label: VerdictLabel,
) -> None:
    if gold_label is VerdictLabel.INVALID and any(entry["status"] == "contradicted" for entry in ledger):
        return
    if gold_label is VerdictLabel.UNIDENTIFIABLE and any(entry["status"] == "unresolved" for entry in ledger):
        return
    if gold_label is VerdictLabel.VALID:
        return

    fallback_status = "contradicted" if gold_label is VerdictLabel.INVALID else "unresolved"
    fallback_note = (
        "The benchmark label indicates that the specific claim overreaches what the available evidence justifies."
        if gold_label is VerdictLabel.INVALID
        else "The benchmark label indicates that the specific claim is not uniquely determined by the available evidence."
    )
    ledger.append(
        {
            "name": "claim-specific identifying bridge",
            "source": "claim implied",
            "status": fallback_status,
            "note": fallback_note,
        }
    )


def _assumption_summary_counts(ledger: list[dict[str, str]]) -> dict[str, int]:
    return {
        "supported_count": sum(entry["status"] == "supported" for entry in ledger),
        "contradicted_count": sum(entry["status"] == "contradicted" for entry in ledger),
        "unresolved_count": sum(entry["status"] == "unresolved" for entry in ledger),
    }


def _assumption_verdict(ledger: list[dict[str, str]]) -> VerdictLabel:
    if any(entry["status"] == "contradicted" for entry in ledger):
        return VerdictLabel.INVALID
    if any(entry["status"] == "unresolved" for entry in ledger):
        return VerdictLabel.UNIDENTIFIABLE
    return VerdictLabel.VALID


def generate_support_witness(
    family: GraphFamilyBlueprint | str,
    *,
    gold_label: VerdictLabel | str,
    query_type: str | None = None,
    seed: int = 0,
) -> Witness:
    """Generate a support witness aligned with the benchmark schema."""

    blueprint = _resolve_blueprint(family, seed=seed)
    label = _coerce_label(gold_label)
    rng = _stable_rng("support_witness", blueprint.family_name, label.value, seed)
    treatment = blueprint.target_variables["treatment"]
    outcome = blueprint.target_variables["outcome"]
    support_type = _select_support_type(blueprint)
    anchors = _support_anchor_variables(blueprint)
    highlight = anchors[0] if anchors else "the observed evidence"
    query_hint = _resolve_query_hint(blueprint, query_type)

    if label is VerdictLabel.VALID:
        description_options = (
            f"The observed structure around {highlight} supports identification of the {query_hint} from {treatment} to {outcome}.",
            f"Measured evidence centered on {highlight} provides a concrete support witness for the claim about {treatment} and {outcome}.",
            f"The benchmark family contains an identifiable support path, with {highlight} helping justify the claim on {treatment} and {outcome}.",
        )
        support_strength = round(rng.uniform(0.78, 0.95), 3)
        verdict_suggestion: VerdictLabel | None = VerdictLabel.VALID
    else:
        description_options = (
            f"The sample still contains a surface-level pattern around {highlight} that can make the claim about {treatment} and {outcome} sound plausible.",
            f"Observed evidence involving {highlight} gives the attacker a seemingly supportive narrative about {treatment} and {outcome}, even though it is not decisive.",
            f"There is a visible observational pattern around {highlight} that can be used to overstate the link from {treatment} to {outcome}.",
        )
        support_strength = round(rng.uniform(0.38, 0.68), 3)
        verdict_suggestion = None

    evidence = [
        f"Target query: {query_hint}.",
        f"Observed variables directly available to the verifier include {', '.join(blueprint.observed_variables[:4])}.",
        f"Most salient measured support signal: {highlight}.",
    ]
    if anchors[1:]:
        evidence.append(f"Additional supportive anchors: {', '.join(anchors[1:3])}.")

    assumptions = [entry["name"] for entry in _build_assumption_ledger(blueprint, gold_label=label)]

    return Witness(
        witness_type=WitnessKind.SUPPORT,
        description=rng.choice(description_options),
        evidence=evidence,
        assumptions=assumptions,
        payload={
            "support_type": support_type,
            "identified": label is VerdictLabel.VALID,
            "support_strength": support_strength,
            "support_explanation": (
                "The observed structure is sufficient only when the key identification assumptions hold."
            ),
            "query_type": query_hint,
            "identifying_variables": anchors,
        },
        verdict_suggestion=verdict_suggestion,
        metadata={
            "family_name": blueprint.family_name,
            "causal_level": blueprint.causal_level,
            "seed": int(seed),
        },
    )


def generate_countermodel_witness(
    family: GraphFamilyBlueprint | str,
    *,
    gold_label: VerdictLabel | str,
    query_type: str | None = None,
    seed: int = 0,
) -> Witness:
    """Generate a countermodel witness aligned with the benchmark schema."""

    blueprint = _resolve_blueprint(family, seed=seed)
    label = _coerce_label(gold_label)
    rng = _stable_rng("countermodel_witness", blueprint.family_name, label.value, seed)
    treatment = blueprint.target_variables["treatment"]
    outcome = blueprint.target_variables["outcome"]
    ctype = _countermodel_type(blueprint)
    query_hint = _resolve_query_hint(blueprint, query_type)
    found_countermodel = label in {VerdictLabel.INVALID, VerdictLabel.UNIDENTIFIABLE}
    query_disagreement = found_countermodel

    if label is VerdictLabel.INVALID:
        description_options = (
            f"A concrete alternative model challenges the claim that {treatment} identifies or determines {outcome}.",
            f"The benchmark admits a countermodel in which the observed evidence persists but the claim about {treatment} and {outcome} fails.",
            f"A strong counterexample can reproduce the observations while undercutting the claimed effect from {treatment} to {outcome}.",
        )
        explanation = (
            f"An alternative {ctype} keeps the observational pattern plausible while reversing the decisive inference about {treatment} and {outcome}."
        )
        match_score = round(rng.uniform(0.9, 0.98), 3)
        verdict_suggestion: VerdictLabel | None = VerdictLabel.INVALID
    elif label is VerdictLabel.UNIDENTIFIABLE:
        description_options = (
            f"Multiple observationally compatible models remain for {treatment} and {outcome}, so the query is not uniquely pinned down.",
            f"The benchmark family allows a countermodel that matches the measured evidence but gives a different answer about {treatment} and {outcome}.",
            f"A same-fit alternative model leaves the counterfactual or interventional answer about {treatment} and {outcome} unresolved.",
        )
        explanation = (
            f"The family supports at least one {ctype} that preserves the observed data while disagreeing on the target causal query."
        )
        match_score = round(rng.uniform(0.9, 0.99), 3)
        verdict_suggestion = VerdictLabel.UNIDENTIFIABLE
    else:
        description_options = (
            f"No strong countermodel was found that both matches the observations and overturns the claim about {treatment} and {outcome}.",
            f"Countermodel search did not uncover an observationally compatible alternative that defeats the claim on {treatment} and {outcome}.",
            f"The available family structure does not yield a decisive counterexample against the target query from {treatment} to {outcome}.",
        )
        explanation = (
            f"Countermodel search did not surface a compelling {ctype} that disagrees on the target query."
        )
        match_score = round(rng.uniform(0.18, 0.42), 3)
        verdict_suggestion = None

    evidence = [
        f"Countermodel type considered: {ctype}.",
        f"Target variables: treatment={treatment}, outcome={outcome}.",
    ]
    if blueprint.hidden_variables:
        evidence.append(f"Hidden-variable alternative can involve {blueprint.hidden_variables[0]}.")
    if blueprint.role_bindings.get("instrument"):
        evidence.append(f"Instrument stress-test variable: {blueprint.role_bindings['instrument']}.")
    if blueprint.role_bindings.get("mediator"):
        evidence.append(f"Mediator-sensitive path: {blueprint.role_bindings['mediator']}.")
    evidence.append(f"Target query: {query_hint}.")

    assumptions = [
        entry["name"]
        for entry in _build_assumption_ledger(blueprint, gold_label=label)
        if entry["status"] != "supported"
    ]
    triggered_assumptions = assumptions if found_countermodel else []
    candidate_payload = {
        "countermodel_type": ctype,
        "causal_level": blueprint.causal_level,
        "observational_match_score": match_score,
        "query_disagreement": query_disagreement,
        "countermodel_explanation": explanation,
        "verdict_suggestion": verdict_suggestion.value if verdict_suggestion is not None else None,
        "triggered_assumptions": list(triggered_assumptions),
        "observational_evidence": {
            "used_observed_data": False,
            "reference_source": "benchmark_countermodel_witness",
        },
    }
    selected_countermodel = (
        {
            **candidate_payload,
            "type": ctype,
            "match_score": match_score,
            "explanation": explanation,
        }
        if found_countermodel
        else None
    )
    candidate_pool = [selected_countermodel] if selected_countermodel is not None else []

    return Witness(
        witness_type=WitnessKind.COUNTERMODEL,
        description=rng.choice(description_options),
        evidence=evidence,
        assumptions=assumptions,
        payload={
            "found_countermodel": found_countermodel,
            "countermodel_type": ctype,
            "observational_match_score": match_score,
            "query_disagreement": query_disagreement,
            "countermodel_explanation": explanation,
            "triggered_assumptions": list(triggered_assumptions),
            "candidates": [candidate_payload] if found_countermodel else [],
            "used_observed_data": False,
            "type": ctype,
            "match_score": match_score,
            "explanation": explanation,
            "candidate_count": len(candidate_pool),
            "selected_countermodel": selected_countermodel,
            "candidate_pool": candidate_pool,
            "query_type": query_hint,
            "verdict_suggestion": verdict_suggestion.value if verdict_suggestion is not None else None,
        },
        verdict_suggestion=verdict_suggestion,
        metadata={
            "family_name": blueprint.family_name,
            "causal_level": blueprint.causal_level,
            "seed": int(seed),
        },
    )


def generate_assumption_witness(
    family: GraphFamilyBlueprint | str,
    *,
    gold_label: VerdictLabel | str,
    query_type: str | None = None,
    seed: int = 0,
) -> Witness:
    """Generate an assumption witness aligned with the benchmark schema."""

    blueprint = _resolve_blueprint(family, seed=seed)
    label = _coerce_label(gold_label)
    rng = _stable_rng("assumption_witness", blueprint.family_name, label.value, seed)
    query_hint = _resolve_query_hint(blueprint, query_type)
    ledger = _build_assumption_ledger(blueprint, gold_label=label)
    counts = _assumption_summary_counts(ledger)
    verdict_suggestion = _assumption_verdict(ledger)

    if verdict_suggestion is VerdictLabel.VALID:
        description_options = (
            "The key identification assumptions are explicit and supported strongly enough for the target query.",
            "Assumption tracking does not reveal a decisive gap for this sample.",
            "The current ledger keeps the core identification assumptions in a supported state.",
        )
    elif verdict_suggestion is VerdictLabel.INVALID:
        description_options = (
            "At least one core identification assumption is directly contradicted by the benchmark construction.",
            "The assumption ledger exposes a concrete failure point that invalidates the claim.",
            "A contradicted assumption breaks the path from observed evidence to the claimed causal conclusion.",
        )
    else:
        description_options = (
            "The assumption ledger still contains unresolved items, so the query remains under-identified.",
            "Key identification assumptions are explicit but not settled by the available evidence.",
            "The sample leaves at least one central assumption unresolved, blocking a unique answer.",
        )

    evidence = [
        f"{entry['name']}: {entry['status']} ({entry['source']})."
        for entry in ledger[:4]
    ]

    return Witness(
        witness_type=WitnessKind.ASSUMPTION,
        description=rng.choice(description_options),
        evidence=evidence,
        assumptions=[entry["name"] for entry in ledger],
        payload={
            "query_type": query_hint,
            "assumption_ledger": ledger,
            **counts,
        },
        verdict_suggestion=verdict_suggestion,
        metadata={
            "family_name": blueprint.family_name,
            "causal_level": blueprint.causal_level,
            "seed": int(seed),
        },
    )


@dataclass(slots=True)
class WitnessBundle:
    """Convenience bundle for wiring witness generation into benchmark samples."""

    support_witness: Witness
    countermodel_witness: Witness
    assumption_witness: Witness

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_witness": self.support_witness.to_dict(),
            "countermodel_witness": self.countermodel_witness.to_dict(),
            "assumption_witness": self.assumption_witness.to_dict(),
        }


def generate_witness_bundle(
    family: GraphFamilyBlueprint | str,
    *,
    gold_label: VerdictLabel | str,
    query_type: str | None = None,
    seed: int = 0,
) -> WitnessBundle:
    """Generate all benchmark witness types for a family/label pair."""

    blueprint = _resolve_blueprint(family, seed=seed)
    label = _coerce_label(gold_label)
    return WitnessBundle(
        support_witness=generate_support_witness(
            blueprint,
            gold_label=label,
            query_type=query_type,
            seed=seed,
        ),
        countermodel_witness=generate_countermodel_witness(
            blueprint,
            gold_label=label,
            query_type=query_type,
            seed=seed,
        ),
        assumption_witness=generate_assumption_witness(
            blueprint,
            gold_label=label,
            query_type=query_type,
            seed=seed,
        ),
    )


def build_witness_bundle(
    family: GraphFamilyBlueprint | str,
    *,
    gold_label: VerdictLabel | str,
    query_type: str | None = None,
    seed: int = 0,
) -> WitnessBundle:
    """Alias kept for readability in generator call sites."""

    return generate_witness_bundle(
        family,
        gold_label=gold_label,
        query_type=query_type,
        seed=seed,
    )
