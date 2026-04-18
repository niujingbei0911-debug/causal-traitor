"""Benchmark-side schema contract for information-partitioned causal oversight."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import pandas as pd

GOLD_ONLY_FIELDS: tuple[str, ...] = (
    "true_dag",
    "hidden_variables",
    "true_scm",
    "full_data",
    "ground_truth",
    "gold_label",
)

VERIFIER_VISIBLE_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "description",
    "variables",
    "proxy_variables",
    "selection_variables",
    "selection_mechanism",
    "observed_data",
    "data",
    "causal_level",
    "difficulty",
    "difficulty_config",
    "metadata",
)

ATTACKER_VISIBLE_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "description",
    "true_dag",
    "variables",
    "hidden_variables",
    "ground_truth",
    "observed_data",
    "full_data",
    "data",
    "causal_level",
    "difficulty",
    "difficulty_config",
    "true_scm",
    "metadata",
)

EVALUATOR_VISIBLE_FIELDS: tuple[str, ...] = (
    *ATTACKER_VISIBLE_FIELDS,
    "gold_label",
    "verdict",
)

_SANITIZED_METADATA_KEYS = frozenset(
    {
        *GOLD_ONLY_FIELDS,
        "ground_truth",
        "gold_label",
        "gold_answer",
        "gold_assumptions",
        "support_witness",
        "countermodel_witness",
        "assumption_witness",
        "verdict",
        "public_description",
        "public_scenario_id",
        "identifiability",
        "role_bindings",
        "generator_hints",
        "benchmark_family",
        "benchmark_subfamily",
        "scenario_family",
        "family_source",
        "family_tags",
        "proxy_variables",
        "selection_variables",
        "is_showcase",
        "winner",
        "seed",
    }
)
_PUBLIC_METADATA_ALLOWED_KEYS = frozenset(
    {
        "notes",
        "difficulty_family",
        "ood_split",
        "selection_ratio",
        "variable_descriptions",
        "measurement_semantics",
        "task_level",
    }
)
_PUBLIC_DIFFICULTY_CONFIG_ALLOWED_KEYS = frozenset(
    {
        "difficulty_family",
        "task_level",
    }
)


class VerdictLabel(str, Enum):
    """Frozen verdict space for the paper's main causal oversight task."""

    VALID = "valid"
    INVALID = "invalid"
    UNIDENTIFIABLE = "unidentifiable"


VERDICT_LABEL_SPACE: tuple[str, ...] = tuple(label.value for label in VerdictLabel)
CAUSAL_LEVEL_SPACE: tuple[str, ...] = ("L1", "L2", "L3")
CLAIM_INSTANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "instance_id",
    "causal_level",
    "graph_family",
    "language_template_id",
    "observed_variables",
    "claim_text",
    "query_type",
    "target_variables",
    "gold_label",
)
TARGET_VARIABLE_KEYS: tuple[str, ...] = ("treatment", "outcome")


class WitnessKind(str, Enum):
    """Supported witness categories for benchmark supervision and verification."""

    SUPPORT = "support"
    COUNTERMODEL = "countermodel"
    ASSUMPTION = "assumption"


WITNESS_KIND_SPACE: tuple[str, ...] = tuple(kind.value for kind in WitnessKind)
BENCHMARK_SPLIT_SPACE: tuple[str, ...] = ("train", "dev", "test_iid", "test_ood")


def _copy_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.copy(deep=True)


def _coerce_label(value: VerdictLabel | str | None, *, field_name: str) -> VerdictLabel | None:
    if value is None:
        return None
    if isinstance(value, VerdictLabel):
        return value
    try:
        return VerdictLabel(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of {VERDICT_LABEL_SPACE}, got {value!r}."
        ) from exc


def _coerce_causal_level(value: int | str, *, field_name: str) -> str:
    if isinstance(value, int):
        if value in (1, 2, 3):
            return f"L{value}"
    normalized = str(value).strip().upper().replace(" ", "")
    if normalized in {"1", "2", "3"}:
        normalized = f"L{normalized}"
    if normalized in CAUSAL_LEVEL_SPACE:
        return normalized
    raise ValueError(
        f"{field_name} must be one of {CAUSAL_LEVEL_SPACE} or 1/2/3, got {value!r}."
    )


def _coerce_witness_kind(value: WitnessKind | str, *, field_name: str) -> WitnessKind:
    if isinstance(value, WitnessKind):
        return value
    try:
        return WitnessKind(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of {WITNESS_KIND_SPACE}, got {value!r}."
        ) from exc


def _normalize_str_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
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


def _normalize_target_variables(value: dict[str, Any] | None) -> dict[str, str]:
    if value is None:
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _require_mapping_keys(
    value: dict[str, Any],
    *,
    required_keys: tuple[str, ...],
    field_name: str,
) -> None:
    missing = [key for key in required_keys if key not in value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{field_name} is missing required keys: {joined}.")


def _serialize_frame(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    copied = _copy_frame(frame)
    if copied.empty:
        return []
    safe = copied.where(pd.notna(copied), None)
    return _serialize_json_safe(safe.to_dict(orient="records"))


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


def _sanitize_metadata(value: Any, *, root: bool = False) -> Any:
    if isinstance(value, dict):
        allowed_keys = _PUBLIC_METADATA_ALLOWED_KEYS if root else None
        return {
            key: _sanitize_metadata(item)
            for raw_key, item in value.items()
            for key in (str(raw_key),)
            if key not in _SANITIZED_METADATA_KEYS
            and (allowed_keys is None or key in allowed_keys)
        }
    if isinstance(value, list):
        return [_sanitize_metadata(item, root=False) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_metadata(item, root=False) for item in value)
    return value


def _sanitize_difficulty_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _serialize_json_safe(item)
        for raw_key, item in value.items()
        for key in (str(raw_key),)
        if key in _PUBLIC_DIFFICULTY_CONFIG_ALLOWED_KEYS
    }


def _normalize_public_variables(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    observed_variables: list[str],
) -> list[str]:
    normalized = _normalize_str_list(values)
    if not observed_variables:
        return normalized
    observed = set(observed_variables)
    return [value for value in normalized if value in observed]


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


@dataclass(slots=True)
class VerifierVerdict:
    """Structured verdict for evaluation.

    ``winner`` remains a legacy debate-game outcome and is intentionally not the
    paper's supervision target. Evaluators should compare ``label`` against
    ``gold_label``.
    """

    label: VerdictLabel | str | None = None
    confidence: float | None = None
    reasoning_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _coerce_label(self.label, field_name="verdict.label")
        if self.confidence is not None:
            self.confidence = float(self.confidence)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value if self.label is not None else None,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "metadata": dict(self.metadata),
        }


def _normalize_verdict(
    value: VerifierVerdict | dict[str, Any] | None,
) -> VerifierVerdict | None:
    if value is None:
        return None
    if isinstance(value, VerifierVerdict):
        return VerifierVerdict(
            label=value.label,
            confidence=value.confidence,
            reasoning_summary=value.reasoning_summary,
            metadata=dict(value.metadata),
        )
    if isinstance(value, dict):
        return VerifierVerdict(
            label=value.get("label"),
            confidence=value.get("confidence"),
            reasoning_summary=str(value.get("reasoning_summary", "")),
            metadata=dict(value.get("metadata", {})),
        )
    raise TypeError(f"Unsupported verdict type: {type(value)!r}")


@dataclass(slots=True)
class PublicCausalInstance:
    """Verifier-visible scenario view with oracle fields stripped out."""

    scenario_id: str
    description: str
    variables: list[str]
    proxy_variables: list[str] = field(default_factory=list)
    selection_variables: list[str] = field(default_factory=list)
    selection_mechanism: str | None = None
    observed_data: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    causal_level: int = 1
    difficulty: float = 0.5
    difficulty_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    verifier_visible_fields: ClassVar[tuple[str, ...]] = VERIFIER_VISIBLE_FIELDS
    verifier_hidden_fields: ClassVar[tuple[str, ...]] = GOLD_ONLY_FIELDS
    attacker_visible_fields: ClassVar[tuple[str, ...]] = ATTACKER_VISIBLE_FIELDS
    evaluator_visible_fields: ClassVar[tuple[str, ...]] = EVALUATOR_VISIBLE_FIELDS
    verdict_label_space: ClassVar[tuple[str, ...]] = VERDICT_LABEL_SPACE

    def __post_init__(self) -> None:
        observed = _copy_frame(self.observed_data)
        if observed.empty and self.data is not None:
            observed = _copy_frame(self.data)
        if not self.variables and not observed.empty:
            self.variables = list(observed.columns)
        metadata = dict(self.metadata)
        self.proxy_variables = _normalize_public_variables(
            list(self.proxy_variables) + list(metadata.pop("proxy_variables", [])),
            observed_variables=list(self.variables),
        )
        self.selection_variables = _normalize_public_variables(
            list(self.selection_variables) + list(metadata.pop("selection_variables", [])),
            observed_variables=list(self.variables),
        )
        self.selection_mechanism = _coerce_optional_string(
            self.selection_mechanism
            if self.selection_mechanism is not None
            else metadata.pop("selection_mechanism", None)
        )
        self.observed_data = observed
        self.data = _copy_frame(observed)
        self.difficulty_config = _sanitize_difficulty_config(self.difficulty_config)
        self.metadata = dict(_sanitize_metadata(metadata, root=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "variables": list(self.variables),
            "proxy_variables": list(self.proxy_variables),
            "selection_variables": list(self.selection_variables),
            "selection_mechanism": self.selection_mechanism,
            "observed_data": _serialize_frame(self.observed_data),
            "data": _serialize_frame(self.data),
            "causal_level": self.causal_level,
            "difficulty": self.difficulty,
            "difficulty_config": dict(self.difficulty_config),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class GoldCausalInstance:
    """Full-fidelity scenario view reserved for generation and evaluation."""

    scenario_id: str
    description: str
    true_dag: dict[str, list[str]]
    variables: list[str]
    hidden_variables: list[str]
    ground_truth: dict[str, Any] = field(default_factory=dict)
    observed_data: pd.DataFrame | None = None
    full_data: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    causal_level: int = 1
    difficulty: float = 0.5
    difficulty_config: dict[str, Any] = field(default_factory=dict)
    true_scm: dict[str, Any] | None = None
    gold_label: VerdictLabel | str | None = None
    verdict: VerifierVerdict | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    verifier_visible_fields: ClassVar[tuple[str, ...]] = VERIFIER_VISIBLE_FIELDS
    verifier_hidden_fields: ClassVar[tuple[str, ...]] = GOLD_ONLY_FIELDS
    attacker_visible_fields: ClassVar[tuple[str, ...]] = ATTACKER_VISIBLE_FIELDS
    evaluator_visible_fields: ClassVar[tuple[str, ...]] = EVALUATOR_VISIBLE_FIELDS
    verdict_label_space: ClassVar[tuple[str, ...]] = VERDICT_LABEL_SPACE

    def __post_init__(self) -> None:
        observed = _copy_frame(self.observed_data)
        full = _copy_frame(self.full_data)
        data = _copy_frame(self.data)

        if observed.empty and not data.empty:
            observed = data.copy(deep=True)
        if observed.empty and not full.empty:
            observed = full.drop(columns=self.hidden_variables, errors="ignore").copy(deep=True)
        if full.empty:
            full = observed.copy(deep=True)
        if data.empty:
            data = observed.copy(deep=True)

        self.observed_data = observed
        self.full_data = full
        self.data = data
        self.ground_truth = dict(self.ground_truth)
        self.difficulty_config = dict(self.difficulty_config)
        self.gold_label = _coerce_label(
            self.gold_label or self.ground_truth.get("label"),
            field_name="gold_label",
        )
        if self.gold_label is not None:
            self.ground_truth["label"] = self.gold_label.value
        self.verdict = _normalize_verdict(self.verdict)
        self.metadata = dict(self.metadata)
        if self.true_scm is None:
            self.true_scm = {"graph": self.true_dag}

    def to_public(self) -> PublicCausalInstance:
        """Project the gold scenario into the verifier-visible schema."""

        return PublicCausalInstance(
            scenario_id=str(self.metadata.get("public_scenario_id", self.scenario_id)),
            description=str(self.metadata.get("public_description", self.description)),
            variables=list(self.variables),
            proxy_variables=list(self.metadata.get("proxy_variables", [])),
            selection_variables=list(self.metadata.get("selection_variables", [])),
            selection_mechanism=self.metadata.get("selection_mechanism"),
            observed_data=self.observed_data.copy(deep=True),
            data=self.observed_data.copy(deep=True),
            causal_level=self.causal_level,
            difficulty=self.difficulty,
            difficulty_config=_sanitize_difficulty_config(self.difficulty_config),
            metadata=dict(_sanitize_metadata(self.metadata, root=True)),
        )

    def public_view(self) -> PublicCausalInstance:
        """Alias kept for readability in access-control call sites."""

        return self.to_public()


VerifierScenario = PublicCausalInstance


@dataclass(slots=True)
class OversightExample:
    """Bundle the attacker-visible gold view with the verifier-visible public view."""

    gold: GoldCausalInstance
    public: PublicCausalInstance

    attacker_visible_fields: ClassVar[tuple[str, ...]] = ATTACKER_VISIBLE_FIELDS
    verifier_visible_fields: ClassVar[tuple[str, ...]] = VERIFIER_VISIBLE_FIELDS
    evaluator_visible_fields: ClassVar[tuple[str, ...]] = EVALUATOR_VISIBLE_FIELDS

    @classmethod
    def from_gold(cls, gold: GoldCausalInstance) -> "OversightExample":
        return cls(gold=gold, public=gold.to_public())


@dataclass(slots=True)
class Witness:
    """Serializable benchmark witness attached to a causal claim instance."""

    witness_type: WitnessKind | str
    description: str
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    verdict_suggestion: VerdictLabel | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    verdict_label_space: ClassVar[tuple[str, ...]] = VERDICT_LABEL_SPACE
    witness_type_space: ClassVar[tuple[str, ...]] = WITNESS_KIND_SPACE

    def __post_init__(self) -> None:
        self.witness_type = _coerce_witness_kind(
            self.witness_type,
            field_name="witness_type",
        )
        self.description = str(self.description)
        self.evidence = _normalize_str_list(self.evidence)
        self.assumptions = _normalize_str_list(self.assumptions)
        self.payload = dict(self.payload)
        self.verdict_suggestion = _coerce_label(
            self.verdict_suggestion,
            field_name="verdict_suggestion",
        )
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_type": self.witness_type.value,
            "description": self.description,
            "evidence": list(self.evidence),
            "assumptions": list(self.assumptions),
            "payload": _serialize_json_safe(self.payload),
            "verdict_suggestion": (
                self.verdict_suggestion.value if self.verdict_suggestion is not None else None
            ),
            "metadata": _serialize_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Witness":
        return cls(
            witness_type=payload.get("witness_type", payload.get("type")),
            description=str(payload.get("description", "")),
            evidence=list(payload.get("evidence", [])),
            assumptions=list(payload.get("assumptions", [])),
            payload=dict(payload.get("payload", {})),
            verdict_suggestion=payload.get("verdict_suggestion"),
            metadata=dict(payload.get("metadata", {})),
        )


def _normalize_witness(value: Witness | dict[str, Any] | None) -> Witness | None:
    if value is None:
        return None
    if isinstance(value, Witness):
        return Witness.from_dict(value.to_dict())
    if isinstance(value, dict):
        return Witness.from_dict(value)
    raise TypeError(f"Unsupported witness type: {type(value)!r}")


def _normalize_witness_slot(
    value: Witness | dict[str, Any] | None,
    *,
    expected_kind: WitnessKind,
    field_name: str,
) -> Witness | None:
    witness = _normalize_witness(value)
    if witness is None:
        return None
    if witness.witness_type is not expected_kind:
        raise ValueError(
            f"{field_name} must use witness_type={expected_kind.value!r}, "
            f"got {witness.witness_type.value!r}."
        )
    return witness


@dataclass(slots=True)
class ClaimInstance:
    """Main benchmark sample schema used by generators, split builders, and loaders."""

    instance_id: str
    causal_level: int | str
    graph_family: str
    language_template_id: str
    observed_variables: list[str]
    claim_text: str
    query_type: str
    target_variables: dict[str, str]
    gold_label: VerdictLabel | str
    gold_answer: str = ""
    proxy_variables: list[str] = field(default_factory=list)
    selection_mechanism: str | None = None
    observed_data_path: str | None = None
    attacker_rationale: str = ""
    gold_assumptions: list[str] = field(default_factory=list)
    support_witness: Witness | dict[str, Any] | None = None
    countermodel_witness: Witness | dict[str, Any] | None = None
    assumption_witness: Witness | dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    causal_level_space: ClassVar[tuple[str, ...]] = CAUSAL_LEVEL_SPACE
    verdict_label_space: ClassVar[tuple[str, ...]] = VERDICT_LABEL_SPACE

    def __post_init__(self) -> None:
        self.instance_id = _require_non_empty_string(self.instance_id, field_name="instance_id")
        self.causal_level = _coerce_causal_level(self.causal_level, field_name="causal_level")
        self.graph_family = _require_non_empty_string(self.graph_family, field_name="graph_family")
        self.language_template_id = _require_non_empty_string(
            self.language_template_id,
            field_name="language_template_id",
        )
        self.observed_variables = _normalize_str_list(self.observed_variables)
        if not self.observed_variables:
            raise ValueError("observed_variables must contain at least one variable.")
        self.proxy_variables = _normalize_str_list(self.proxy_variables)
        self.claim_text = _require_non_empty_string(self.claim_text, field_name="claim_text")
        self.query_type = _require_non_empty_string(self.query_type, field_name="query_type")
        self.target_variables = _normalize_target_variables(self.target_variables)
        _require_mapping_keys(
            self.target_variables,
            required_keys=TARGET_VARIABLE_KEYS,
            field_name="target_variables",
        )
        for key in TARGET_VARIABLE_KEYS:
            self.target_variables[key] = _require_non_empty_string(
                self.target_variables[key],
                field_name=f"target_variables.{key}",
            )
        self.gold_label = _coerce_label(self.gold_label, field_name="gold_label")
        if self.gold_label is None:
            raise ValueError("gold_label must not be None for ClaimInstance.")
        self.gold_answer = str(self.gold_answer)
        self.selection_mechanism = (
            None if self.selection_mechanism is None else str(self.selection_mechanism)
        )
        self.observed_data_path = (
            None if self.observed_data_path is None else str(self.observed_data_path)
        )
        self.attacker_rationale = str(self.attacker_rationale)
        self.gold_assumptions = _normalize_str_list(self.gold_assumptions)
        self.support_witness = _normalize_witness_slot(
            self.support_witness,
            expected_kind=WitnessKind.SUPPORT,
            field_name="support_witness",
        )
        self.countermodel_witness = _normalize_witness_slot(
            self.countermodel_witness,
            expected_kind=WitnessKind.COUNTERMODEL,
            field_name="countermodel_witness",
        )
        self.assumption_witness = _normalize_witness_slot(
            self.assumption_witness,
            expected_kind=WitnessKind.ASSUMPTION,
            field_name="assumption_witness",
        )
        self.meta = dict(self.meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "causal_level": self.causal_level,
            "graph_family": self.graph_family,
            "language_template_id": self.language_template_id,
            "observed_variables": list(self.observed_variables),
            "proxy_variables": list(self.proxy_variables),
            "selection_mechanism": self.selection_mechanism,
            "observed_data_path": self.observed_data_path,
            "claim_text": self.claim_text,
            "attacker_rationale": self.attacker_rationale,
            "query_type": self.query_type,
            "target_variables": dict(self.target_variables),
            "gold_label": self.gold_label.value,
            "gold_answer": self.gold_answer,
            "gold_assumptions": list(self.gold_assumptions),
            "support_witness": (
                self.support_witness.to_dict() if self.support_witness is not None else None
            ),
            "countermodel_witness": (
                self.countermodel_witness.to_dict()
                if self.countermodel_witness is not None
                else None
            ),
            "assumption_witness": (
                self.assumption_witness.to_dict() if self.assumption_witness is not None else None
            ),
            "meta": _serialize_json_safe(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimInstance":
        _require_mapping_keys(
            payload,
            required_keys=CLAIM_INSTANCE_REQUIRED_FIELDS,
            field_name="ClaimInstance payload",
        )
        return cls(
            instance_id=str(payload["instance_id"]),
            causal_level=payload["causal_level"],
            graph_family=str(payload["graph_family"]),
            language_template_id=str(payload["language_template_id"]),
            observed_variables=list(payload["observed_variables"]),
            proxy_variables=list(payload.get("proxy_variables", [])),
            selection_mechanism=payload.get("selection_mechanism"),
            observed_data_path=payload.get("observed_data_path"),
            claim_text=str(payload["claim_text"]),
            attacker_rationale=str(payload.get("attacker_rationale", "")),
            query_type=str(payload["query_type"]),
            target_variables=dict(payload["target_variables"]),
            gold_label=payload["gold_label"],
            gold_answer=str(payload.get("gold_answer", "")),
            gold_assumptions=list(payload.get("gold_assumptions", [])),
            support_witness=payload.get("support_witness"),
            countermodel_witness=payload.get("countermodel_witness"),
            assumption_witness=payload.get("assumption_witness"),
            meta=dict(payload.get("meta", {})),
        )


@dataclass(slots=True)
class BenchmarkSplitManifest:
    """Serializable split manifest for train/dev/test_iid/test_ood benchmark partitions."""

    dataset_name: str = "causal_oversight_benchmark"
    version: str = "v1"
    train: list[str] = field(default_factory=list)
    dev: list[str] = field(default_factory=list)
    test_iid: list[str] = field(default_factory=list)
    test_ood: list[str] = field(default_factory=list)
    family_holdout: list[str] = field(default_factory=list)
    lexical_holdout: list[str] = field(default_factory=list)
    variable_renaming_holdout: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    split_name_space: ClassVar[tuple[str, ...]] = BENCHMARK_SPLIT_SPACE

    def __post_init__(self) -> None:
        self.dataset_name = str(self.dataset_name)
        self.version = str(self.version)
        self.train = _normalize_str_list(self.train)
        self.dev = _normalize_str_list(self.dev)
        self.test_iid = _normalize_str_list(self.test_iid)
        self.test_ood = _normalize_str_list(self.test_ood)
        self.family_holdout = _normalize_str_list(self.family_holdout)
        self.lexical_holdout = _normalize_str_list(self.lexical_holdout)
        self.variable_renaming_holdout = bool(self.variable_renaming_holdout)
        self.metadata = dict(self.metadata)

    def split_map(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "dev": list(self.dev),
            "test_iid": list(self.test_iid),
            "test_ood": list(self.test_ood),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "splits": self.split_map(),
            "holdout_strategy": {
                "family_holdout": list(self.family_holdout),
                "lexical_holdout": list(self.lexical_holdout),
                "variable_renaming_holdout": self.variable_renaming_holdout,
            },
            "metadata": _serialize_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkSplitManifest":
        splits = dict(payload.get("splits", {}))
        holdout_strategy = dict(payload.get("holdout_strategy", {}))
        return cls(
            dataset_name=str(payload.get("dataset_name", "causal_oversight_benchmark")),
            version=str(payload.get("version", "v1")),
            train=list(splits.get("train", payload.get("train", []))),
            dev=list(splits.get("dev", payload.get("dev", []))),
            test_iid=list(splits.get("test_iid", payload.get("test_iid", []))),
            test_ood=list(splits.get("test_ood", payload.get("test_ood", []))),
            family_holdout=list(
                holdout_strategy.get("family_holdout", payload.get("family_holdout", []))
            ),
            lexical_holdout=list(
                holdout_strategy.get("lexical_holdout", payload.get("lexical_holdout", []))
            ),
            variable_renaming_holdout=holdout_strategy.get(
                "variable_renaming_holdout",
                payload.get("variable_renaming_holdout", False),
            ),
            metadata=dict(payload.get("metadata", {})),
        )


def ensure_public_instance(
    scenario: PublicCausalInstance | GoldCausalInstance | Any,
) -> PublicCausalInstance:
    """Project a scenario-like object into the verifier-safe public view."""

    if isinstance(scenario, PublicCausalInstance):
        return scenario
    if isinstance(scenario, GoldCausalInstance):
        return scenario.to_public()
    to_public = getattr(scenario, "to_public", None)
    if callable(to_public):
        public = to_public()
        if isinstance(public, PublicCausalInstance):
            return public
    raise TypeError(f"Unsupported scenario type for public projection: {type(scenario)!r}")


def require_public_instance(scenario: PublicCausalInstance | Any) -> PublicCausalInstance:
    """Enforce the strict verifier-side contract: only public instances are accepted."""

    if isinstance(scenario, PublicCausalInstance):
        return scenario
    raise TypeError(
        "Verifier-side entrypoints only accept PublicCausalInstance. "
        f"Received {type(scenario)!r}."
    )
