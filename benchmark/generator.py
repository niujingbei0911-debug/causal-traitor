"""Benchmark generator for showcase and programmatic causal oversight samples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from benchmark.attacks import AttackSample, generate_attack_sample
from benchmark.graph_families import (
    GraphFamilyBlueprint,
    IdentifiabilityStatus,
    generate_graph_family,
    list_graph_families,
)
from benchmark.schema import ClaimInstance, GoldCausalInstance, PublicCausalInstance, VerdictLabel
from benchmark.witnesses import WitnessBundle, generate_witness_bundle
from game.data_generator import DataGenerator, SHOWCASE_FAMILY_REGISTRY, SHOWCASE_ID_ALIASES


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _stable_rng(*parts: Any) -> np.random.Generator:
    material = "::".join(str(part) for part in parts).encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(digest)


def _coerce_causal_level(level: int | str | None) -> str | None:
    if level is None:
        return None
    if isinstance(level, int):
        return f"L{level}"
    normalized = str(level).strip().upper()
    if normalized in {"1", "2", "3"}:
        return f"L{normalized}"
    return normalized


_CONTEXT_SHIFT_DOMAIN_SPACE: tuple[str, ...] = (
    "policy",
    "clinical",
    "education",
    "market",
)
_ATTACK_FAMILY_BY_ATTACK_NAME: dict[str, str] = {
    "association_overclaim": "observational_shortcut",
    "hidden_confounder_denial": "observational_shortcut",
    "selection_bias_obfuscation": "observational_shortcut",
    "invalid_adjustment_claim": "identification_shortcut",
    "weak_iv_as_valid_iv": "identification_shortcut",
    "invalid_iv_exclusion_claim": "identification_shortcut",
    "heterogeneity_overgeneralization": "transport_shortcut",
    "counterfactual_overclaim": "counterfactual_shortcut",
    "function_form_manipulation": "counterfactual_shortcut",
    "unidentifiable_disguised_as_valid": "counterfactual_shortcut",
}


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = float(array.std())
    if std < 1e-8:
        return array - float(array.mean())
    return (array - float(array.mean())) / std


def _serialize_json_safe(value: Any) -> Any:
    if isinstance(value, VerdictLabel):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_safe(item) for item in value]
    return value


def _coerce_label_override(value: VerdictLabel | str | None) -> VerdictLabel | None:
    if value is None:
        return None
    if isinstance(value, VerdictLabel):
        return value
    return VerdictLabel(str(value).strip().lower())


def _serialize_frame(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    safe = frame.copy(deep=True).where(pd.notna(frame), None)
    return _serialize_json_safe(safe.to_dict(orient="records"))


def _serialize_gold_instance(gold: GoldCausalInstance) -> dict[str, Any]:
    return {
        "scenario_id": gold.scenario_id,
        "description": gold.description,
        "true_dag": {node: list(children) for node, children in gold.true_dag.items()},
        "variables": list(gold.variables),
        "hidden_variables": list(gold.hidden_variables),
        "ground_truth": _serialize_json_safe(dict(gold.ground_truth)),
        "observed_data": _serialize_frame(gold.observed_data),
        "full_data": _serialize_frame(gold.full_data),
        "data": _serialize_frame(gold.data),
        "causal_level": gold.causal_level,
        "difficulty": gold.difficulty,
        "difficulty_config": _serialize_json_safe(dict(gold.difficulty_config)),
        "true_scm": _serialize_json_safe(gold.true_scm),
        "gold_label": gold.gold_label.value if gold.gold_label is not None else None,
        "verdict": gold.verdict.to_dict() if gold.verdict is not None else None,
        "metadata": _serialize_json_safe(dict(gold.metadata)),
    }


def _public_scenario_id(raw_scenario_id: str) -> str:
    digest = hashlib.sha256(str(raw_scenario_id).encode("utf-8")).hexdigest()[:12]
    return f"public_case_{digest}"


def _public_description(
    *,
    observed_variables: list[str],
    treatment: str,
    outcome: str,
    causal_level: str,
    context_shift_group: str | None = None,
) -> str:
    visible = list(observed_variables[:4])
    variable_text = ", ".join(visible)
    if len(observed_variables) > len(visible):
        variable_text = f"{variable_text}, ..."
    prefix = ""
    if context_shift_group:
        prefix = f"{str(context_shift_group).title()} context. "
    return prefix + (
        f"Observed {causal_level} benchmark case over {variable_text}. "
        f"Evaluate claims about {treatment} and {outcome} using only the public evidence in this view."
    )


def _context_shift_group(
    *,
    family_name: str,
    seed: int,
    available_domains: list[str] | tuple[str, ...] | None = None,
) -> str:
    candidates = [str(value).strip() for value in (available_domains or _CONTEXT_SHIFT_DOMAIN_SPACE) if str(value).strip()]
    if not candidates:
        candidates = list(_CONTEXT_SHIFT_DOMAIN_SPACE)
    rng = _stable_rng("context_shift", family_name, seed)
    return str(rng.choice(candidates))


def _context_shift_payload(
    *,
    blueprint: GraphFamilyBlueprint,
    seed: int,
) -> dict[str, Any]:
    available_domains = list(blueprint.generator_hints.get("context_shift_domains", []))
    context_group = _context_shift_group(
        family_name=blueprint.family_name,
        seed=seed,
        available_domains=available_domains,
    )
    return {
        "context_shift_group": context_group,
        "context_shift_id": f"{context_group}::{blueprint.family_name}::seed_{seed}",
        "context_shift_domains": available_domains or list(_CONTEXT_SHIFT_DOMAIN_SPACE),
    }


def _attack_family_for_name(attack_name: str | None) -> str | None:
    if attack_name is None:
        return None
    return _ATTACK_FAMILY_BY_ATTACK_NAME.get(str(attack_name).strip())


def _rename_json_tokens(value: Any, rename_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            str(rename_map.get(str(key), str(key))): _rename_json_tokens(item, rename_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_json_tokens(item, rename_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_rename_json_tokens(item, rename_map) for item in value)
    if isinstance(value, str):
        return rename_map.get(value, value)
    return value


def _parent_map(true_dag: dict[str, list[str]]) -> dict[str, list[str]]:
    parents = {node: [] for node in true_dag}
    for parent, children in true_dag.items():
        for child in children:
            parents.setdefault(child, []).append(parent)
    for items in parents.values():
        items.sort()
    return parents


def _topological_order(true_dag: dict[str, list[str]]) -> list[str]:
    in_degree = {node: 0 for node in true_dag}
    for children in true_dag.values():
        for child in children:
            in_degree[child] = in_degree.get(child, 0) + 1

    queue = sorted([node for node, degree in in_degree.items() if degree == 0])
    ordered: list[str] = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for child in sorted(true_dag.get(node, [])):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()

    if len(ordered) != len(in_degree):
        raise ValueError("Graph family must define a DAG for programmatic generation.")
    return ordered


def _resolve_showcase_scenario_id(identifier: str | None) -> str | None:
    if identifier is None:
        return None
    normalized = str(identifier)
    if normalized in SHOWCASE_ID_ALIASES:
        return SHOWCASE_ID_ALIASES[normalized]
    for scenario_id, info in SHOWCASE_FAMILY_REGISTRY.items():
        if normalized in {info["showcase_name"], info["showcase_family"]}:
            return scenario_id
    return None


def get_showcase_parent_family(showcase_identifier: str) -> str:
    """Resolve a showcase alias/subfamily to its registered benchmark parent family."""

    scenario_id = _resolve_showcase_scenario_id(showcase_identifier)
    if scenario_id is None:
        raise KeyError(f"Unknown showcase identifier: {showcase_identifier!r}.")
    parent_family = str(SHOWCASE_FAMILY_REGISTRY[scenario_id]["benchmark_family"])
    if parent_family not in set(list_graph_families()):
        raise ValueError(
            f"Showcase {showcase_identifier!r} points to unregistered benchmark_family "
            f"{parent_family!r}."
        )
    return parent_family


def list_showcase_families() -> list[str]:
    """List the showcase subfamilies retained for demos and sanity checks."""

    return sorted(info["showcase_family"] for info in SHOWCASE_FAMILY_REGISTRY.values())


def list_supported_benchmark_families(*, include_showcase: bool = True) -> list[str]:
    """List benchmark family names supported by the generator."""

    result = list(list_graph_families())
    if include_showcase:
        result.extend(list_showcase_families())
    return sorted(result)


@dataclass(slots=True)
class BenchmarkSample:
    """Benchmark sample bundle spanning ClaimInstance, gold view, and public view."""

    claim: ClaimInstance
    gold: GoldCausalInstance
    public: PublicCausalInstance
    blueprint: GraphFamilyBlueprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "gold": _serialize_gold_instance(self.gold),
            "public": self.public.to_dict(),
            "blueprint": self.blueprint.to_dict(),
        }


class BenchmarkGenerator:
    """Generate benchmark-ready gold/public instances.

    Showcase stories are preserved for demos, while programmatic graph-family
    samples provide the non-showcase benchmark path used by the paper.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        self.config = config or {}
        self.seed = int(seed)

    def generate_gold_instance(
        self,
        *,
        difficulty: float = 0.5,
        family_name: str | None = None,
        showcase_name: str | None = None,
        causal_level: int | str | None = None,
        seed: int | None = None,
        n_samples: int | None = None,
        gold_label_override: VerdictLabel | str | None = None,
        sample_variant_tag: str | None = None,
    ) -> GoldCausalInstance:
        """Generate one gold instance from either a showcase or a graph family."""

        resolved_seed = self.seed if seed is None else int(seed)
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        showcase_scenario_id = _resolve_showcase_scenario_id(showcase_name or family_name)
        if showcase_scenario_id is not None:
            return self._generate_showcase_instance(
                showcase_scenario_id=showcase_scenario_id,
                difficulty=difficulty,
                seed=resolved_seed,
                n_samples=n_samples,
                gold_label_override=gold_label_override,
                sample_variant_tag=sample_variant_tag,
            )

        normalized_level = _coerce_causal_level(causal_level)
        resolved_family_name = family_name
        if resolved_family_name is None:
            candidates = list_graph_families(causal_level=normalized_level)
            if not candidates:
                raise ValueError(
                    "No programmatic graph families available for the requested causal level."
                )
            resolved_family_name = candidates[resolved_seed % len(candidates)]

        blueprint = generate_graph_family(resolved_family_name, seed=resolved_seed)
        blueprint, renaming_meta = self._maybe_apply_variable_renaming(
            blueprint,
            seed=resolved_seed,
        )
        return self._generate_programmatic_instance(
            blueprint=blueprint,
            difficulty=difficulty,
            n_samples=n_samples or self._resolve_sample_size(difficulty),
            seed=resolved_seed,
            renaming_meta=renaming_meta,
            gold_label_override=gold_label_override,
            sample_variant_tag=sample_variant_tag,
        )

    def generate_public_instance(self, **kwargs: Any) -> PublicCausalInstance:
        """Generate a public-view instance directly from the requested family."""

        return self.generate_gold_instance(**kwargs).to_public()

    def generate_benchmark_sample(
        self,
        **kwargs: Any,
    ) -> BenchmarkSample:
        """Generate the benchmark v1 sample bundle used by downstream builders."""

        gold = self.generate_gold_instance(**kwargs)
        seed = int(kwargs.get("seed", self.seed))
        blueprint = self._build_blueprint_for_gold_instance(gold, seed=seed)
        claim = self._build_claim_instance(gold=gold, blueprint=blueprint, seed=seed)
        public = gold.to_public()
        return BenchmarkSample(
            claim=claim,
            gold=gold,
            public=public,
            blueprint=blueprint,
        )

    def generate_claim_instance(self, **kwargs: Any) -> ClaimInstance:
        """Generate the benchmark sample's main schema object."""

        return self.generate_benchmark_sample(**kwargs).claim

    def generate_benchmark_samples(
        self,
        *,
        num_samples: int,
        family_names: list[str] | None = None,
        include_showcase: bool = False,
        causal_level: int | str | None = None,
        difficulty: float = 0.5,
        seed: int | None = None,
        n_samples: int | None = None,
    ) -> list[BenchmarkSample]:
        """Generate a deterministic batch of benchmark sample bundles."""

        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")

        normalized_level = _coerce_causal_level(causal_level)
        candidate_families = family_names or self._candidate_families(
            include_showcase=include_showcase,
            causal_level=normalized_level,
        )
        if not candidate_families:
            raise ValueError("No benchmark families are available for the requested batch generation.")

        base_seed = self.seed if seed is None else int(seed)
        samples: list[BenchmarkSample] = []
        for index in range(num_samples):
            family_name = candidate_families[index % len(candidate_families)]
            sample_seed = base_seed + index
            sample = self.generate_benchmark_sample(
                family_name=family_name,
                causal_level=causal_level,
                difficulty=difficulty,
                seed=sample_seed,
                n_samples=n_samples,
            )
            samples.append(sample)
        return samples

    def generate_claim_instances(
        self,
        *,
        num_samples: int,
        family_names: list[str] | None = None,
        include_showcase: bool = False,
        causal_level: int | str | None = None,
        difficulty: float = 0.5,
        seed: int | None = None,
        n_samples: int | None = None,
    ) -> list[ClaimInstance]:
        """Generate a deterministic batch of ClaimInstance objects for split building."""

        return [
            sample.claim
            for sample in self.generate_benchmark_samples(
                num_samples=num_samples,
                family_names=family_names,
                include_showcase=include_showcase,
                causal_level=causal_level,
                difficulty=difficulty,
                seed=seed,
                n_samples=n_samples,
            )
        ]

    def generate_paired_flip_samples(
        self,
        *,
        family_name: str,
        difficulty: float = 0.5,
        seed: int | None = None,
        n_samples: int | None = None,
    ) -> tuple[BenchmarkSample, BenchmarkSample]:
        """Generate a deterministic paired-flip sample pair for one family."""

        resolved_seed = self.seed if seed is None else int(seed)
        blueprint = generate_graph_family(family_name, seed=resolved_seed)
        labels = list(blueprint.generator_hints.get("paired_flip_candidates", blueprint.supported_gold_labels))
        if len(labels) < 2:
            raise ValueError(f"Family {family_name!r} does not expose enough labels for paired-flip generation.")

        first_label, second_label = labels[0], labels[1]
        anchor = self.generate_benchmark_sample(
            family_name=family_name,
            difficulty=difficulty,
            seed=resolved_seed,
            n_samples=n_samples,
            gold_label_override=first_label,
            sample_variant_tag=f"paired_flip::{first_label}",
        )
        flipped = self.generate_benchmark_sample(
            family_name=family_name,
            difficulty=difficulty,
            seed=resolved_seed,
            n_samples=n_samples,
            gold_label_override=second_label,
            sample_variant_tag=f"paired_flip::{second_label}",
        )

        pair_id = f"{family_name}::seed_{resolved_seed}::query_{anchor.claim.query_type}"
        pair_meta = (
            (anchor, flipped, "anchor"),
            (flipped, anchor, "flip"),
        )
        for current, partner, role in pair_meta:
            current.claim.meta["paired_flip_id"] = pair_id
            current.claim.meta["paired_flip_role"] = role
            current.claim.meta["paired_flip_partner_id"] = partner.claim.instance_id
            current.claim.meta.setdefault("ood_axes", [])
            if "paired_flip" not in current.claim.meta["ood_axes"]:
                current.claim.meta["ood_axes"] = [
                    *current.claim.meta["ood_axes"],
                    "paired_flip",
                ]
            current.gold.metadata["paired_flip_id"] = pair_id
            current.gold.metadata["paired_flip_role"] = role
            current.gold.metadata["paired_flip_partner_id"] = partner.claim.instance_id
        return anchor, flipped

    def export_public_instance(
        self,
        scenario: GoldCausalInstance | PublicCausalInstance,
    ) -> PublicCausalInstance:
        """Project a gold instance into the verifier-visible public schema."""

        if isinstance(scenario, PublicCausalInstance):
            return scenario
        return scenario.to_public()

    def _candidate_families(
        self,
        *,
        include_showcase: bool,
        causal_level: str | None,
    ) -> list[str]:
        candidates = list_graph_families(causal_level=causal_level)
        if include_showcase:
            for showcase_family in list_showcase_families():
                scenario_id = _resolve_showcase_scenario_id(showcase_family)
                if scenario_id is None:
                    continue
                showcase_level = int(SHOWCASE_FAMILY_REGISTRY[scenario_id]["causal_level"])
                if causal_level is None or causal_level == f"L{showcase_level}":
                    candidates.append(showcase_family)
        return sorted(candidates)

    def _maybe_apply_variable_renaming(
        self,
        blueprint: GraphFamilyBlueprint,
        *,
        seed: int,
    ) -> tuple[GraphFamilyBlueprint, dict[str, Any]]:
        if (
            int(seed) % 4 != 3
            or len(blueprint.observed_variables) < 2
        ):
            return blueprint, {}

        rename_map = self._build_variable_rename_map(blueprint, seed=seed)
        renamed_blueprint = self._apply_variable_renaming(blueprint, rename_map=rename_map)
        renaming_meta = {
            "variable_renaming": True,
            "rename_map": dict(rename_map),
            "original_variables": list(blueprint.observed_variables),
            "renamed_variables": list(renamed_blueprint.observed_variables),
        }
        return renamed_blueprint, renaming_meta

    def _public_variable_descriptions(
        self,
        blueprint: GraphFamilyBlueprint,
    ) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        treatment = blueprint.target_variables.get("treatment")
        outcome = blueprint.target_variables.get("outcome")
        role_text = {
            "backdoor_adjuster": "Observed pre-treatment covariate available for public adjustment checks.",
            "observed_adjuster": "Observed covariate available as part of the public evidence bundle.",
            "observed_context": "Observed contextual covariate available to the verifier.",
            "proxy": "Observed proxy-style measurement available as part of the public evidence.",
            "instrument": "Observed assignment-like or encouragement signal available in the public evidence.",
            "mediator": "Observed intermediate measurement available in the public evidence.",
            "selection": "Observed sample-inclusion indicator visible in the public evidence.",
        }
        for variable in blueprint.observed_variables:
            parts: list[str] = []
            if variable == treatment:
                parts.append("Observed treatment variable used in the query.")
            if variable == outcome:
                parts.append("Observed outcome variable used in the query.")
            for role, bound_variable in blueprint.role_bindings.items():
                if bound_variable != variable or role in {"treatment", "outcome"}:
                    continue
                if role in role_text:
                    parts.append(role_text[role])
            if not parts:
                parts.append("Observed benchmark variable available to the verifier.")
            descriptions[variable] = " ".join(dict.fromkeys(parts))
        return descriptions

    def _public_measurement_semantics(
        self,
        blueprint: GraphFamilyBlueprint,
    ) -> dict[str, dict[str, Any]]:
        semantics: dict[str, dict[str, Any]] = {}
        role_kind = {
            "treatment": ("treatment_measurement", "Directly manipulated or assigned treatment quantity in the public data."),
            "outcome": ("outcome_measurement", "Observed response quantity used as the public outcome."),
            "backdoor_adjuster": ("adjustment_covariate", "Observed pre-treatment covariate used for public adjustment checks."),
            "observed_adjuster": ("adjustment_covariate", "Observed covariate carried into the public evidence bundle."),
            "observed_context": ("context_covariate", "Observed background covariate visible in the public view."),
            "proxy": ("proxy_measurement", "Observed auxiliary measurement that can help disambiguate the public signal."),
            "instrument": ("assignment_signal", "Observed assignment-like or encouragement-style signal in the public data."),
            "mediator": ("intermediate_measurement", "Observed intermediate measurement between treatment and response."),
            "selection": ("sample_inclusion_indicator", "Observed indicator related to which units enter the public sample."),
        }
        for role, variable in blueprint.role_bindings.items():
            if variable not in blueprint.observed_variables:
                continue
            measurement_view, note = role_kind.get(
                role,
                ("observed_measurement", "Observed benchmark measurement available in the public view."),
            )
            entry = semantics.setdefault(
                variable,
                {"measurement_view": measurement_view, "notes": []},
            )
            entry["notes"] = list(dict.fromkeys([*entry.get("notes", []), note]))
        return semantics

    def _public_metadata_payload(
        self,
        blueprint: GraphFamilyBlueprint,
    ) -> dict[str, Any]:
        return {
            "task_level": blueprint.causal_level,
            "variable_descriptions": self._public_variable_descriptions(blueprint),
            "measurement_semantics": self._public_measurement_semantics(blueprint),
        }

    def _build_variable_rename_map(
        self,
        blueprint: GraphFamilyBlueprint,
        *,
        seed: int,
    ) -> dict[str, str]:
        rename_map: dict[str, str] = {}
        for index, variable in enumerate(blueprint.observed_variables, start=1):
            token = hashlib.sha256(
                f"{blueprint.family_name}:{seed}:{variable}".encode("utf-8")
            ).hexdigest()[:6]
            rename_map[str(variable)] = f"feature_{index}_{token}"
        return rename_map

    def _apply_variable_renaming(
        self,
        blueprint: GraphFamilyBlueprint,
        *,
        rename_map: dict[str, str],
    ) -> GraphFamilyBlueprint:
        def rename(value: str) -> str:
            return str(rename_map.get(value, value))

        return GraphFamilyBlueprint(
            family_name=blueprint.family_name,
            causal_level=blueprint.causal_level,
            identifiability=blueprint.identifiability,
            description=blueprint.description,
            true_dag={
                rename(node): [rename(child) for child in children]
                for node, children in blueprint.true_dag.items()
            },
            observed_variables=[rename(variable) for variable in blueprint.observed_variables],
            hidden_variables=[rename(variable) for variable in blueprint.hidden_variables],
            target_variables={
                key: rename(value)
                for key, value in blueprint.target_variables.items()
            },
            seed=blueprint.seed,
            proxy_variables=[rename(variable) for variable in blueprint.proxy_variables],
            selection_variables=[rename(variable) for variable in blueprint.selection_variables],
            role_bindings={
                role: rename(value)
                for role, value in blueprint.role_bindings.items()
            },
            query_types=list(blueprint.query_types),
            supported_gold_labels=list(blueprint.supported_gold_labels),
            family_tags=list(blueprint.family_tags),
            generator_hints=dict(_rename_json_tokens(blueprint.generator_hints, rename_map)),
        )

    def _build_blueprint_for_gold_instance(
        self,
        gold: GoldCausalInstance,
        *,
        seed: int,
    ) -> GraphFamilyBlueprint:
        family_name = str(
            gold.metadata.get("benchmark_family")
            or gold.metadata.get("scenario_family")
            or gold.scenario_id
        )
        base_blueprint = generate_graph_family(family_name, seed=seed)
        rename_map = {
            str(source): str(target)
            for source, target in dict(gold.metadata.get("rename_map", {})).items()
            if str(source).strip() and str(target).strip()
        }
        if gold.metadata.get("variable_renaming") and rename_map:
            base_blueprint = self._apply_variable_renaming(base_blueprint, rename_map=rename_map)
        treatment = str(gold.ground_truth.get("treatment", gold.variables[0]))
        outcome = str(gold.ground_truth.get("outcome", gold.variables[-1]))
        observed_variables = list(gold.variables)
        metadata_role_bindings = {
            str(role): str(variable)
            for role, variable in dict(gold.metadata.get("role_bindings", {})).items()
            if str(role).strip() and str(variable).strip()
        }
        used = {
            value
            for value in metadata_role_bindings.values()
            if value in observed_variables or value in gold.hidden_variables
        }
        used.update({treatment, outcome})
        role_bindings = dict(metadata_role_bindings)
        role_bindings["treatment"] = treatment
        role_bindings["outcome"] = outcome

        instrument = gold.ground_truth.get("instrument")
        if instrument:
            role_bindings["instrument"] = str(instrument)
            used.add(str(instrument))
        mediator = gold.ground_truth.get("mediator")
        if mediator:
            role_bindings["mediator"] = str(mediator)
            used.add(str(mediator))

        for role, variable in base_blueprint.role_bindings.items():
            if role in {"treatment", "outcome"}:
                continue
            if role in role_bindings:
                continue
            normalized = str(variable).strip()
            if not normalized:
                continue
            if normalized in observed_variables or normalized in gold.hidden_variables:
                role_bindings.setdefault(role, normalized)
                if normalized in observed_variables:
                    used.add(normalized)

        remaining_observed = [variable for variable in observed_variables if variable not in used]
        if "backdoor_adjuster" in base_blueprint.role_bindings and remaining_observed:
            role_bindings["backdoor_adjuster"] = remaining_observed[0]
        if "observed_adjuster" in base_blueprint.role_bindings and remaining_observed:
            role_bindings.setdefault("observed_adjuster", remaining_observed[0])
        if "observed_context" in base_blueprint.role_bindings and remaining_observed:
            context = next(
                (variable for variable in remaining_observed if variable not in role_bindings.values()),
                remaining_observed[0],
            )
            role_bindings["observed_context"] = context
        if "selection" in base_blueprint.role_bindings and remaining_observed:
            selection = next(
                (variable for variable in remaining_observed if variable not in role_bindings.values()),
                remaining_observed[-1],
            )
            role_bindings["selection"] = selection

        proxy_variables = list(gold.metadata.get("proxy_variables", base_blueprint.proxy_variables))
        if not proxy_variables and base_blueprint.proxy_variables:
            proxy_candidate = next(
                (
                    variable
                    for variable in remaining_observed
                    if variable not in role_bindings.values() and variable not in proxy_variables
                ),
                None,
            )
            if proxy_candidate is not None:
                proxy_variables = [proxy_candidate]
                role_bindings.setdefault("proxy", proxy_candidate)
        elif proxy_variables:
            role_bindings.setdefault("proxy", proxy_variables[0])

        hidden_variables = list(gold.hidden_variables)
        if hidden_variables:
            latent_role = "latent_response_type" if gold.causal_level == 3 else "latent_confounder"
            role_bindings.setdefault(latent_role, hidden_variables[0])

        selection_variables = list(
            gold.metadata.get("selection_variables", base_blueprint.selection_variables)
        )
        if role_bindings.get("selection") and role_bindings["selection"] not in selection_variables:
            selection_variables.append(role_bindings["selection"])

        identifiability_value = gold.ground_truth.get("identifiability")
        if identifiability_value is None:
            identifiability = (
                IdentifiabilityStatus.IDENTIFIABLE
                if gold.gold_label is VerdictLabel.VALID
                else IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE
            )
        else:
            identifiability = IdentifiabilityStatus(str(identifiability_value).strip().lower())

        return GraphFamilyBlueprint(
            family_name=family_name,
            causal_level=f"L{gold.causal_level}",
            identifiability=identifiability,
            description=gold.description,
            true_dag={node: list(children) for node, children in gold.true_dag.items()},
            observed_variables=observed_variables,
            hidden_variables=hidden_variables,
            target_variables={"treatment": treatment, "outcome": outcome},
            seed=seed,
            proxy_variables=proxy_variables,
            selection_variables=selection_variables,
            role_bindings=role_bindings,
            query_types=list(
                gold.ground_truth.get("query_types", base_blueprint.query_types)
            ),
            supported_gold_labels=list(base_blueprint.supported_gold_labels),
            family_tags=list(gold.metadata.get("family_tags", base_blueprint.family_tags)),
            generator_hints=dict(
                gold.metadata.get("generator_hints", base_blueprint.generator_hints)
            ),
        )

    def _build_claim_instance(
        self,
        *,
        gold: GoldCausalInstance,
        blueprint: GraphFamilyBlueprint,
        seed: int,
    ) -> ClaimInstance:
        if gold.gold_label is None:
            raise ValueError("Gold instance must define gold_label before building ClaimInstance.")

        gold_label = gold.gold_label
        query_type = self._select_query_type(blueprint, seed)
        attack_sample = self._generate_claim_text(
            blueprint=blueprint,
            gold_label=gold_label,
            query_type=query_type,
            seed=seed,
        )
        witness_bundle = generate_witness_bundle(
            blueprint,
            gold_label=gold_label,
            query_type=query_type,
            seed=seed,
        )
        benchmark_family = str(gold.metadata.get("benchmark_family", blueprint.family_name))
        benchmark_subfamily = gold.metadata.get("benchmark_subfamily")
        ood_axes = [
            "mechanism_ood_tag",
            "context_shift_group",
            *(
                ["attack_family"]
                if attack_sample.get("attack_family") is not None
                else []
            ),
            *(
                ["paired_flip"]
                if gold.metadata.get("paired_flip_id")
                else []
            ),
        ]

        return ClaimInstance(
            instance_id=f"{gold.scenario_id}_benchmark_seed_{seed}",
            causal_level=gold.causal_level,
            graph_family=benchmark_family,
            language_template_id=attack_sample["language_template_id"],
            observed_variables=list(gold.variables),
            proxy_variables=list(
                gold.metadata.get("proxy_variables", blueprint.proxy_variables)
            ),
            selection_mechanism=str(
                gold.metadata.get(
                    "selection_mechanism",
                    blueprint.generator_hints.get("selection_mechanism", "none"),
                )
            ),
            observed_data_path=str(gold.metadata.get("observed_data_path", "")) or None,
            claim_text=attack_sample["claim_text"],
            attacker_rationale=attack_sample["attacker_rationale"],
            query_type=query_type,
            target_variables={
                "treatment": str(gold.ground_truth["treatment"]),
                "outcome": str(gold.ground_truth["outcome"]),
            },
            gold_label=gold_label,
            gold_answer=self._build_gold_answer(gold=gold, query_type=query_type),
            gold_assumptions=list(witness_bundle.assumption_witness.assumptions),
            support_witness=witness_bundle.support_witness,
            countermodel_witness=witness_bundle.countermodel_witness,
            assumption_witness=witness_bundle.assumption_witness,
            meta={
                "seed": int(seed),
                "scenario_id": gold.scenario_id,
                "benchmark_family": benchmark_family,
                "benchmark_subfamily": benchmark_subfamily,
                "family_source": gold.metadata.get("family_source", "programmatic"),
                "is_showcase": bool(gold.metadata.get("is_showcase", False)),
                "difficulty_family": gold.metadata.get("difficulty_family", benchmark_family),
                "task_level": gold.metadata.get("task_level", blueprint.causal_level),
                "ood_split": gold.metadata.get("ood_split"),
                "generator_mode": "benchmark_generator_v1",
                "claim_mode": attack_sample["claim_mode"],
                "attack_name": attack_sample.get("attack_name"),
                "attack_family": attack_sample.get("attack_family"),
                "style_id": attack_sample.get("style_id"),
                "persuasion_style_id": attack_sample.get("persuasion_style_id"),
                "pressure_type": attack_sample.get("pressure_type"),
                "pressure_markers": list(attack_sample.get("pressure_markers", [])),
                "conceals_missing_information": bool(
                    attack_sample.get("conceals_missing_information", False)
                ),
                "mechanism_ood_tag": gold.metadata.get(
                    "mechanism_ood_tag",
                    blueprint.generator_hints.get("mechanism_ood_tag"),
                ),
                "context_shift_group": gold.metadata.get("context_shift_group"),
                "context_shift_id": gold.metadata.get("context_shift_id"),
                "ood_axes": ood_axes,
                "paired_flip_id": gold.metadata.get("paired_flip_id"),
                "paired_flip_role": gold.metadata.get("paired_flip_role"),
                "paired_flip_partner_id": gold.metadata.get("paired_flip_partner_id"),
                "selected_query_type": query_type,
                "query_types": list(blueprint.query_types),
                "family_tags": list(blueprint.family_tags),
                **(
                    {
                        "variable_renaming": True,
                        "renamed_variables": list(gold.metadata.get("renamed_variables", [])),
                    }
                    if gold.metadata.get("variable_renaming")
                    else {}
                ),
            },
        )

    def _generate_claim_text(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        gold_label: VerdictLabel,
        query_type: str,
        seed: int,
    ) -> dict[str, Any]:
        if gold_label in {VerdictLabel.INVALID, VerdictLabel.UNIDENTIFIABLE}:
            attack = generate_attack_sample(
                blueprint,
                gold_label=gold_label,
                query_type=query_type,
                seed=seed,
            )
            self._ensure_attack_sample_consistency(blueprint, gold_label=gold_label, attack=attack)
            return {
                "claim_text": attack.claim_text,
                "attacker_rationale": attack.attacker_rationale,
                "language_template_id": (
                    f"attack::{attack.attack_name}::{attack.style_id}"
                    f"::persuasion::{attack.persuasion_style_id}"
                ),
                "claim_mode": "attack",
                "attack_name": attack.attack_name,
                "attack_family": _attack_family_for_name(attack.attack_name),
                "style_id": attack.style_id,
                "persuasion_style_id": attack.persuasion_style_id,
                "pressure_type": attack.metadata.get("pressure_type"),
                "pressure_markers": list(attack.metadata.get("pressure_markers", [])),
                "conceals_missing_information": bool(
                    attack.metadata.get("conceals_missing_information", False)
                ),
            }
        return self._generate_truthful_claim(blueprint=blueprint, query_type=query_type, seed=seed)

    def _generate_truthful_claim(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        query_type: str,
        seed: int,
    ) -> dict[str, Any]:
        rng = _stable_rng("truthful_claim", blueprint.family_name, seed)
        treatment = blueprint.target_variables["treatment"]
        outcome = blueprint.target_variables["outcome"]
        style_id = rng.choice(("direct", "cautious", "formal"))
        bridge_role, bridge = self._resolve_truthful_bridge_spec(
            blueprint=blueprint,
            query_type=query_type,
        )
        claim_options = self._truthful_claim_options(
            query_type=query_type,
            treatment=treatment,
            outcome=outcome,
            bridge=bridge,
            bridge_role=bridge_role,
        )

        rationale_options = (
            "Present the benchmark's supported identification route faithfully instead of overstating it.",
            "Use the benchmark's actual identifying structure and avoid adding unsupported assumptions.",
            "State the claim in a calibrated way that matches the benchmark label and available evidence.",
        )
        return {
            "claim_text": rng.choice(claim_options),
            "attacker_rationale": rng.choice(rationale_options),
            "language_template_id": f"truthful::{style_id}::{query_type}",
            "claim_mode": "truthful",
            "attack_name": None,
            "attack_family": None,
            "style_id": style_id,
            "persuasion_style_id": None,
            "pressure_type": None,
            "pressure_markers": [],
            "conceals_missing_information": False,
        }

    def _build_gold_answer(
        self,
        *,
        gold: GoldCausalInstance,
        query_type: str,
    ) -> str:
        treatment = str(gold.ground_truth["treatment"])
        outcome = str(gold.ground_truth["outcome"])
        label = gold.gold_label.value if gold.gold_label is not None else "unknown"
        if gold.gold_label is VerdictLabel.VALID:
            return (
                f"For query_type={query_type}, the benchmark construction supports the claim "
                f"from {treatment} to {outcome}."
            )
        if gold.gold_label is VerdictLabel.INVALID:
            return (
                f"For query_type={query_type}, the claim about {treatment} and {outcome} "
                "overreaches what the benchmark evidence supports."
            )
        return (
            f"For query_type={query_type}, the benchmark does not uniquely identify the answer "
            f"about {treatment} and {outcome}; label={label}."
        )

    def _generate_showcase_instance(
        self,
        *,
        showcase_scenario_id: str,
        difficulty: float,
        seed: int,
        n_samples: int | None,
        gold_label_override: VerdictLabel | str | None = None,
        sample_variant_tag: str | None = None,
    ) -> GoldCausalInstance:
        legacy_generator = DataGenerator(config=self.config, seed=seed)
        info = SHOWCASE_FAMILY_REGISTRY[showcase_scenario_id]
        scenario = legacy_generator.generate_scenario(
            difficulty=difficulty,
            causal_level=int(info["causal_level"]),
            scenario_id=showcase_scenario_id,
            n_samples=n_samples,
        )
        parent_blueprint = generate_graph_family(str(info["benchmark_family"]), seed=seed)
        resolved_label_override = _coerce_label_override(gold_label_override)
        if resolved_label_override is not None:
            scenario.gold_label = resolved_label_override
        elif scenario.gold_label is None:
            label_value = scenario.ground_truth.get("label") or self._select_gold_label(parent_blueprint, seed)
            scenario.gold_label = VerdictLabel(str(label_value).strip().lower())
        scenario.ground_truth["label"] = scenario.gold_label.value
        scenario.ground_truth.setdefault("identifiability", parent_blueprint.identifiability.value)
        scenario.ground_truth.setdefault("query_types", list(parent_blueprint.query_types))
        context_payload = _context_shift_payload(blueprint=parent_blueprint, seed=seed)
        if sample_variant_tag:
            scenario.scenario_id = f"{scenario.scenario_id}::{sample_variant_tag}"
        scenario.metadata.setdefault("proxy_variables", list(parent_blueprint.proxy_variables))
        scenario.metadata.setdefault("selection_variables", list(parent_blueprint.selection_variables))
        scenario.metadata.setdefault(
            "selection_mechanism",
            parent_blueprint.generator_hints.get("selection_mechanism", "none"),
        )
        scenario.metadata.setdefault(
            "mechanism_ood_tag",
            parent_blueprint.generator_hints.get("mechanism_ood_tag"),
        )
        scenario.metadata.setdefault("family_tags", list(parent_blueprint.family_tags))
        scenario.metadata.setdefault("generator_hints", dict(parent_blueprint.generator_hints))
        scenario.metadata.setdefault("task_level", parent_blueprint.causal_level)
        scenario.difficulty_config.setdefault("task_level", parent_blueprint.causal_level)
        scenario.metadata.setdefault(
            "public_description",
            _public_description(
                observed_variables=list(parent_blueprint.observed_variables),
                treatment=parent_blueprint.target_variables["treatment"],
                outcome=parent_blueprint.target_variables["outcome"],
                causal_level=parent_blueprint.causal_level,
                context_shift_group=context_payload["context_shift_group"],
            ),
        )
        scenario.metadata.setdefault(
            "public_scenario_id",
            _public_scenario_id(f"showcase:{showcase_scenario_id}:{sample_variant_tag or seed}"),
        )
        for key, value in context_payload.items():
            scenario.metadata.setdefault(key, value)
        for key, value in self._public_metadata_payload(parent_blueprint).items():
            scenario.metadata.setdefault(key, value)
        if parent_blueprint.role_bindings.get("instrument"):
            scenario.metadata.setdefault(
                "instrument_variables",
                [str(parent_blueprint.role_bindings["instrument"])],
            )
        if parent_blueprint.role_bindings.get("mediator"):
            scenario.metadata.setdefault(
                "mediator_variables",
                [str(parent_blueprint.role_bindings["mediator"])],
            )
        return scenario

    def _generate_programmatic_instance(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        difficulty: float,
        n_samples: int,
        seed: int,
        renaming_meta: dict[str, Any] | None = None,
        gold_label_override: VerdictLabel | str | None = None,
        sample_variant_tag: str | None = None,
    ) -> GoldCausalInstance:
        resolved_label_override = _coerce_label_override(gold_label_override)
        if (
            resolved_label_override is not None
            and str(resolved_label_override.value) not in set(blueprint.supported_gold_labels)
        ):
            raise ValueError(
                f"gold_label_override={resolved_label_override.value!r} is not supported by "
                f"{blueprint.family_name!r}."
            )
        gold_label = resolved_label_override or VerdictLabel(
            str(self._select_gold_label(blueprint, seed)).strip().lower()
        )
        query_type = self._select_query_type(blueprint, seed)
        sample_seed = seed
        for attempt in range(64):
            full_data, true_scm = self._sample_programmatic_data(
                blueprint=blueprint,
                difficulty=difficulty,
                n_samples=n_samples,
                seed=sample_seed,
            )
            observed_data = full_data.loc[:, blueprint.observed_variables].copy()
            if gold_label != VerdictLabel.VALID or self._supports_public_validity_contract(
                blueprint=blueprint,
                observed_data=observed_data,
                query_type=query_type,
            ):
                break
            sample_seed += 7919
        else:
            raise ValueError(
                f"Unable to generate a public-self-consistent valid sample for {blueprint.family_name} with seed {seed}."
            )

        treatment = blueprint.target_variables["treatment"]
        outcome = blueprint.target_variables["outcome"]
        effect_summary = self._estimate_effect_summary(observed_data, treatment, outcome)
        difficulty_profile = self._difficulty_profile(difficulty)
        renaming_meta = dict(renaming_meta or {})
        public_metadata = self._public_metadata_payload(blueprint)
        context_payload = _context_shift_payload(blueprint=blueprint, seed=seed)
        scenario_suffix = f"::{sample_variant_tag}" if sample_variant_tag else ""

        return GoldCausalInstance(
            scenario_id=f"{blueprint.family_name}_seed_{seed}{scenario_suffix}",
            description=(
                f"{blueprint.description} Programmatic sample generated from "
                f"{blueprint.family_name} with seed {seed}."
            ),
            true_dag={node: list(children) for node, children in blueprint.true_dag.items()},
            variables=list(blueprint.observed_variables),
            hidden_variables=list(blueprint.hidden_variables),
            ground_truth={
                "treatment": treatment,
                "outcome": outcome,
                "label": gold_label,
                "identifiability": blueprint.identifiability.value,
                "query_types": list(blueprint.query_types),
                **effect_summary,
            },
            observed_data=observed_data,
            full_data=full_data,
            data=observed_data.copy(deep=True),
            causal_level=int(blueprint.causal_level[1]),
            difficulty=difficulty,
            difficulty_config={
                **difficulty_profile,
                "task_level": blueprint.causal_level,
                "generator_mode": "programmatic",
                "generator_seed": seed,
            },
            true_scm=true_scm,
            gold_label=gold_label,
            metadata={
                "scenario_family": blueprint.family_name,
                "benchmark_family": blueprint.family_name,
                "benchmark_subfamily": None,
                "family_source": "programmatic",
                "is_showcase": False,
                "seed": seed,
                "sampling_seed": sample_seed,
                "identifiability": blueprint.identifiability.value,
                "proxy_variables": list(blueprint.proxy_variables),
                "instrument_variables": (
                    [str(blueprint.role_bindings["instrument"])]
                    if blueprint.role_bindings.get("instrument")
                    else []
                ),
                "mediator_variables": (
                    [str(blueprint.role_bindings["mediator"])]
                    if blueprint.role_bindings.get("mediator")
                    else []
                ),
                "selection_variables": list(blueprint.selection_variables),
                "selection_mechanism": blueprint.generator_hints.get("selection_mechanism", "none"),
                "mechanism_ood_tag": blueprint.generator_hints.get("mechanism_ood_tag"),
                "role_bindings": dict(blueprint.role_bindings),
                "family_tags": list(blueprint.family_tags),
                "generator_hints": dict(blueprint.generator_hints),
                "public_scenario_id": _public_scenario_id(
                    f"{blueprint.family_name}:{seed}:{sample_variant_tag or 'default'}"
                ),
                "public_description": _public_description(
                    observed_variables=list(blueprint.observed_variables),
                    treatment=treatment,
                    outcome=outcome,
                    causal_level=blueprint.causal_level,
                    context_shift_group=context_payload["context_shift_group"],
                ),
                **context_payload,
                **public_metadata,
                **renaming_meta,
            },
        )

    def _sample_programmatic_data(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        difficulty: float,
        n_samples: int,
        seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        profile = self._difficulty_profile(difficulty)
        true_dag = {node: list(children) for node, children in blueprint.true_dag.items()}
        parents = _parent_map(true_dag)
        ordered_nodes = _topological_order(true_dag)
        role_lookup = {variable: role for role, variable in blueprint.role_bindings.items()}

        edge_weights: dict[tuple[str, str], float] = {}
        for parent, children in true_dag.items():
            for child in children:
                sign = -1.0 if rng.random() < 0.2 and child == blueprint.target_variables["outcome"] else 1.0
                edge_weights[(parent, child)] = float(sign * rng.uniform(0.35, 1.2))

        values: dict[str, np.ndarray] = {}
        for node in ordered_nodes:
            signal = rng.normal(scale=0.15 + 0.2 * profile["nonlinearity"], size=n_samples)
            for parent in parents.get(node, []):
                signal = signal + edge_weights[(parent, node)] * _standardize(values[parent])
            noise = rng.normal(scale=profile["noise_scale"], size=n_samples)
            role = role_lookup.get(node, "")

            if node in blueprint.hidden_variables:
                values[node] = signal + rng.normal(scale=profile["hidden_strength"], size=n_samples)
                continue

            if role in {"treatment", "selection"} or node in blueprint.selection_variables:
                probability = np.clip(_sigmoid(signal + noise), 0.05, 0.95)
                values[node] = rng.binomial(1, probability)
                continue

            if role == "instrument":
                probability = np.clip(_sigmoid(0.75 * signal + noise), 0.1, 0.9)
                values[node] = rng.binomial(1, probability)
                continue

            if role == "outcome" and blueprint.causal_level in {"L1", "L3"}:
                probability = np.clip(_sigmoid(signal + noise), 0.05, 0.95)
                values[node] = rng.binomial(1, probability)
                continue

            nonlinear_signal = np.tanh((1.0 + profile["nonlinearity"]) * signal)
            values[node] = nonlinear_signal + noise

        frame = pd.DataFrame({node: values[node] for node in ordered_nodes})
        frame = self._apply_selection(frame, blueprint)
        true_scm = {
            "graph": true_dag,
            "weights": {
                f"{parent}->{child}": weight for (parent, child), weight in edge_weights.items()
            },
            "seed": seed,
            "generator_mode": "programmatic",
        }
        return frame, true_scm

    def _apply_selection(
        self,
        frame: pd.DataFrame,
        blueprint: GraphFamilyBlueprint,
    ) -> pd.DataFrame:
        if not blueprint.selection_variables:
            return frame

        mask = np.ones(len(frame), dtype=bool)
        for variable in blueprint.selection_variables:
            values = frame[variable].to_numpy()
            current_mask = values >= 1
            ratio = float(current_mask.mean())
            if ratio < 0.15 or ratio > 0.95:
                threshold = float(np.quantile(values, 0.5))
                current_mask = values >= threshold
            mask &= current_mask

        if float(mask.mean()) < 0.1:
            values = frame[blueprint.selection_variables[0]].to_numpy()
            threshold = float(np.quantile(values, 0.5))
            mask = values >= threshold

        selected = frame.loc[mask].reset_index(drop=True)
        if selected.empty:
            return frame
        return selected

    def _estimate_effect_summary(
        self,
        observed_data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> dict[str, float]:
        treatment_values = observed_data[treatment]
        outcome_values = observed_data[outcome]
        if set(pd.unique(treatment_values.dropna())) <= {0, 1}:
            treated = outcome_values[treatment_values == 1]
            control = outcome_values[treatment_values == 0]
            if len(treated) and len(control):
                return {
                    "observational_difference": float(treated.mean() - control.mean()),
                    "approx_effect": float(treated.mean() - control.mean()),
                }
        correlation = treatment_values.corr(outcome_values)
        return {
            "observational_difference": float(0.0 if pd.isna(correlation) else correlation),
            "approx_effect": float(0.0 if pd.isna(correlation) else correlation),
        }

    def _supports_public_validity_contract(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        observed_data: pd.DataFrame,
        query_type: str,
    ) -> bool:
        treatment = blueprint.target_variables["treatment"]
        outcome = blueprint.target_variables["outcome"]
        try:
            if blueprint.family_name == "l1_proxy_disambiguation_family":
                from causal_tools.l1_association import proxy_support_check

                proxy = blueprint.role_bindings.get("proxy") or (
                    blueprint.proxy_variables[0] if blueprint.proxy_variables else None
                )
                if proxy is None:
                    return False
                report = proxy_support_check(observed_data, treatment, outcome, proxy, controls=[])
                return bool(report.get("supports_proxy_sufficiency"))

            if blueprint.family_name == "l2_valid_backdoor_family":
                import networkx as nx
                from causal_tools.l2_intervention import overlap_check, validate_backdoor_criterion

                adjuster = (
                    blueprint.role_bindings.get("backdoor_adjuster")
                    or blueprint.role_bindings.get("observed_adjuster")
                )
                if adjuster is None:
                    return False
                graph = nx.DiGraph()
                for parent, children in blueprint.true_dag.items():
                    for child in children:
                        graph.add_edge(parent, child)
                overlap_report = overlap_check(observed_data, treatment, [adjuster])
                return bool(validate_backdoor_criterion(graph, treatment, outcome, [adjuster])) and bool(
                    overlap_report.get("has_overlap")
                )

            if blueprint.family_name == "l2_valid_iv_family":
                from causal_tools.l2_intervention import iv_estimation

                instrument = blueprint.role_bindings.get("instrument")
                if instrument is None:
                    return False
                covariates = [
                    variable
                    for variable in blueprint.observed_variables
                    if variable not in {treatment, outcome, instrument}
                ][:1]
                report = iv_estimation(observed_data, instrument, treatment, outcome, covariates)
                return bool(report.get("is_strong_instrument")) and bool(
                    report.get("supports_exclusion_restriction")
                ) and bool(report.get("supports_instrument_independence"))

            if blueprint.family_name == "l3_mediation_abduction_family":
                from causal_tools.l3_counterfactual import counterfactual_bridge_check

                mediator = blueprint.role_bindings.get("mediator")
                if mediator is None:
                    return False
                covariates = [
                    variable
                    for variable in blueprint.observed_variables
                    if variable not in {treatment, outcome, mediator}
                ][:2]
                report = counterfactual_bridge_check(
                    observed_data,
                    treatment,
                    mediator,
                    outcome,
                    covariates,
                )
                return bool(report.get("supports_cross_world_consistency")) and bool(
                    report.get("supports_counterfactual_model_uniqueness")
                )
        except Exception:
            return False

        return True

    def _select_gold_label(self, blueprint: GraphFamilyBlueprint, seed: int) -> str:
        labels = list(blueprint.supported_gold_labels)
        if labels:
            return labels[seed % len(labels)]
        if blueprint.identifiability is IdentifiabilityStatus.POTENTIALLY_UNIDENTIFIABLE:
            return "unidentifiable"
        return "valid"

    def _resolve_truthful_bridge_spec(
        self,
        *,
        blueprint: GraphFamilyBlueprint,
        query_type: str,
    ) -> tuple[str | None, str | None]:
        if (
            query_type in {"average_treatment_effect", "interventional_effect"}
            and "valid_iv" in blueprint.family_tags
            and blueprint.role_bindings.get("instrument")
        ):
            return "instrument", str(blueprint.role_bindings["instrument"])
        role_preferences = {
            "association_strength": ("proxy", "observed_adjuster", "observed_context"),
            "causal_direction": ("proxy", "observed_context", "observed_adjuster"),
            "proxy_adjusted_claim": ("proxy", "observed_adjuster", "observed_context"),
            "average_treatment_effect": ("backdoor_adjuster", "observed_adjuster", "proxy"),
            "interventional_effect": ("backdoor_adjuster", "observed_adjuster", "proxy"),
            "instrumental_variable_claim": ("instrument",),
            "unit_level_counterfactual": ("mediator", "observed_adjuster", "observed_context"),
            "effect_of_treatment_on_treated": ("mediator", "observed_adjuster", "observed_context"),
            "abduction_action_prediction": ("mediator", "observed_adjuster", "observed_context"),
        }
        for role in role_preferences.get(query_type, ()):
            value = blueprint.role_bindings.get(role)
            if value:
                return role, str(value)
        if blueprint.proxy_variables:
            return "proxy", blueprint.proxy_variables[0]
        non_target_observed = [
            variable
            for variable in blueprint.observed_variables
            if variable not in blueprint.target_variables.values()
        ]
        if non_target_observed:
            return "observed_context", non_target_observed[0]
        return None, None

    def _truthful_claim_options(
        self,
        *,
        query_type: str,
        treatment: str,
        outcome: str,
        bridge: str | None,
        bridge_role: str | None = None,
    ) -> tuple[str, ...]:
        if query_type == "association_strength":
            if bridge is None:
                return (
                    f"The observed association between {treatment} and {outcome} is supported by the benchmark evidence.",
                    f"The benchmark keeps a stable observed relationship between {treatment} and {outcome}.",
                    f"The available data support the observed association linking {treatment} and {outcome}.",
                )
            if bridge_role == "proxy":
                return (
                    f"With proxy {bridge} available, the observed association between {treatment} and {outcome} is supported by the benchmark evidence.",
                    f"The measured proxy {bridge} supports the observed relationship between {treatment} and {outcome}.",
                    f"Using proxy {bridge}, the benchmark still supports the observed association between {treatment} and {outcome}.",
                )
            return (
                f"Using {bridge}, the observed association between {treatment} and {outcome} is supported by the benchmark evidence.",
                f"With {bridge} available, the benchmark still supports the observed relationship between {treatment} and {outcome}.",
                f"The observed structure around {bridge} supports the association between {treatment} and {outcome}.",
            )

        if query_type == "causal_direction":
            if bridge is None:
                return (
                    f"The observed association supports the direction from {treatment} to {outcome}.",
                    f"The benchmark evidence favors the direction from {treatment} to {outcome} in the observed association.",
                    f"The measured pattern supports reading the observed association in the direction {treatment} to {outcome}.",
                )
            return (
                f"Using {bridge}, the observed association supports the direction from {treatment} to {outcome}.",
                f"With {bridge} available, the benchmark evidence favors the direction from {treatment} to {outcome}.",
                f"The observed association around {bridge} supports the direction from {treatment} to {outcome}.",
            )

        if query_type == "proxy_adjusted_claim":
            if bridge is None:
                return (
                    f"The observed association between {treatment} and {outcome} remains supported after the measured proxy adjustment.",
                    f"The benchmark supports the proxy-adjusted association between {treatment} and {outcome}.",
                    f"The measured evidence supports the proxy-adjusted claim about the association between {treatment} and {outcome}.",
                )
            return (
                f"Using proxy {bridge}, the observed association between {treatment} and {outcome} is supported more cleanly.",
                f"With proxy {bridge} available, the benchmark supports the proxy-adjusted association between {treatment} and {outcome}.",
                f"The measured proxy {bridge} supports the proxy-adjusted claim about the association between {treatment} and {outcome}.",
            )

        if query_type == "average_treatment_effect":
            if bridge is None:
                return (
                    f"The average treatment effect of {treatment} on {outcome} is identified in the benchmark sample.",
                    f"The causal effect of {treatment} on {outcome} is identified by the available benchmark evidence.",
                    f"The treatment effect of {treatment} on {outcome} is identified in this benchmark construction.",
                )
            if bridge_role == "instrument":
                return (
                    f"Using {bridge} as an instrument is enough to recover the causal effect of {treatment} on {outcome}.",
                    f"The benchmark supports an instrumental-variable estimate of the causal effect of {treatment} on {outcome} using {bridge}.",
                    f"With {bridge} as an instrument, the causal effect of {treatment} on {outcome} is identified.",
                )
            return (
                f"After controlling for {bridge}, the average treatment effect of {treatment} on {outcome} is identified.",
                f"After adjusting for {bridge}, the causal effect of {treatment} on {outcome} is identified.",
                f"Controlling for {bridge} identifies the treatment effect of {treatment} on {outcome}.",
            )

        if query_type == "interventional_effect":
            if bridge is None:
                return (
                    f"The effect of intervening on {treatment} on {outcome} is identified by the benchmark evidence.",
                    f"The interventional effect of {treatment} on {outcome} is identified in this benchmark sample.",
                    f"The causal effect of intervening on {treatment} on {outcome} is identified in this benchmark construction.",
                )
            return (
                f"After controlling for {bridge}, the effect of intervening on {treatment} on {outcome} is identified.",
                f"After adjusting for {bridge}, the interventional effect of {treatment} on {outcome} is identified.",
                f"Controlling for {bridge} identifies the effect of intervening on {treatment} on {outcome}.",
            )

        if query_type == "instrumental_variable_claim":
            instrument = bridge or "the proposed instrument"
            return (
                f"Using {instrument} as an instrument is enough to recover the causal effect of {treatment} on {outcome}.",
                f"The benchmark supports an instrumental-variable estimate of the causal effect of {treatment} on {outcome} using {instrument}.",
                f"With {instrument} as an instrument, the causal effect of {treatment} on {outcome} is identified.",
            )

        if query_type == "unit_level_counterfactual":
            if bridge is None:
                return (
                    f"The benchmark supports the unit-level counterfactual response of {outcome} to {treatment}.",
                    f"The counterfactual response of {outcome} to {treatment} can be evaluated in this benchmark sample.",
                    f"The benchmark evidence supports what {outcome} would have been under a different value of {treatment}.",
                )
            return (
                f"With {bridge} measured, the counterfactual response of {outcome} to {treatment} can be evaluated.",
                f"Using {bridge}, the counterfactual response of {outcome} to {treatment} can be evaluated.",
                f"With {bridge} available, the benchmark supports what {outcome} would have been under a different value of {treatment}.",
            )

        if query_type == "effect_of_treatment_on_treated":
            if bridge is None:
                return (
                    f"The effect of treatment on the treated for {treatment} and {outcome} is identified in the benchmark sample.",
                    f"The benchmark supports the effect of treatment on the treated for {treatment} on {outcome}.",
                    f"The measured evidence identifies the effect of treatment on the treated linking {treatment} and {outcome}.",
                )
            return (
                f"With {bridge} available, the effect of treatment on the treated for {treatment} and {outcome} is identified.",
                f"Using {bridge}, the benchmark supports the effect of treatment on the treated for {treatment} on {outcome}.",
                f"The measured evidence through {bridge} identifies the effect of treatment on the treated linking {treatment} and {outcome}.",
            )

        if query_type == "abduction_action_prediction":
            if bridge is None:
                return (
                    f"The benchmark supports the counterfactual question of what {outcome} would have been under a different value of {treatment}.",
                    f"The counterfactual response of {outcome} to {treatment} is supported by the benchmark evidence.",
                    f"The benchmark evidence supports an abduction-action style counterfactual prediction for {outcome} under a different value of {treatment}.",
                )
            return (
                f"With {bridge} observed, the counterfactual question of what {outcome} would have been under a different value of {treatment} can be answered.",
                f"Using {bridge}, the counterfactual response of {outcome} to {treatment} can be evaluated.",
                f"With {bridge} available, the benchmark supports an abduction-action style counterfactual prediction for {outcome} under a different value of {treatment}.",
            )

        if bridge is None:
            return (
                f"The benchmark evidence supports the claim about the effect of {treatment} on {outcome}.",
                f"The relevant causal claim linking {treatment} and {outcome} is supported by the benchmark evidence.",
                f"The benchmark construction supports the claim from {treatment} to {outcome}.",
            )
        return (
            f"With {bridge} available, the benchmark evidence supports the claim about the effect of {treatment} on {outcome}.",
            f"Using {bridge}, the benchmark construction supports the claim from {treatment} to {outcome}.",
            f"The measured evidence through {bridge} supports the claim linking {treatment} and {outcome}.",
        )

    def _select_query_type(self, blueprint: GraphFamilyBlueprint, seed: int) -> str:
        query_types = list(blueprint.query_types)
        if not query_types:
            return "causal_query"
        label_span = max(1, len(blueprint.supported_gold_labels))
        return query_types[(seed // label_span) % len(query_types)]

    def _ensure_attack_sample_consistency(
        self,
        blueprint: GraphFamilyBlueprint,
        *,
        gold_label: VerdictLabel,
        attack: AttackSample,
    ) -> None:
        if gold_label is not VerdictLabel.INVALID:
            return
        if attack.attack_name != "invalid_adjustment_claim":
            return

        claimed_adjuster = str(attack.metadata.get("adjustment_variable", "")).strip()
        if not claimed_adjuster:
            return

        protected: set[str] = set()
        for candidate_set in blueprint.generator_hints.get("identifying_set_candidates", []):
            if isinstance(candidate_set, (list, tuple, set)):
                protected.update(str(item).strip() for item in candidate_set if str(item).strip())
            else:
                normalized = str(candidate_set).strip()
                if normalized:
                    protected.add(normalized)
        canonical_adjuster = blueprint.role_bindings.get("backdoor_adjuster")
        if canonical_adjuster:
            protected.add(str(canonical_adjuster).strip())

        if claimed_adjuster in protected:
            raise ValueError(
                "invalid_adjustment_claim selected a benchmark identifying variable "
                f"{claimed_adjuster!r} for family={blueprint.family_name!r}."
            )

    def _resolve_sample_size(self, difficulty: float) -> int:
        base = int(self.config.get("base_samples", 320))
        spread = int(self.config.get("difficulty_sample_span", 480))
        return max(120, base + int(spread * difficulty))

    def _difficulty_profile(self, difficulty: float) -> dict[str, float]:
        return {
            "noise_scale": 0.12 + 0.25 * difficulty,
            "hidden_strength": 0.45 + 0.55 * difficulty,
            "selection_bias_strength": 0.1 + 0.25 * difficulty,
            "nonlinearity": 0.15 + 0.55 * difficulty,
        }
