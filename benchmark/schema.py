"""Benchmark-side schema contract for information-partitioned causal oversight."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import re
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
    "selection_mechanism",
    "observed_data",
    "data",
    "causal_level",
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
        "instrument_variables",
        "mediator_variables",
        "selection_variables",
        "is_showcase",
        "winner",
        "seed",
    }
)
_PUBLIC_METADATA_ALLOWED_KEYS = frozenset(
    {
        "variable_descriptions",
        "measurement_semantics",
        "context_shift_group",
        "context_shift_id",
        "context_shift_profile",
    }
)
_PUBLIC_DIFFICULTY_CONFIG_ALLOWED_KEYS = frozenset()


class VerdictLabel(str, Enum):
    """Frozen verdict space for the paper's main causal oversight task."""

    VALID = "valid"
    INVALID = "invalid"
    UNIDENTIFIABLE = "unidentifiable"


VERDICT_LABEL_SPACE: tuple[str, ...] = tuple(label.value for label in VerdictLabel)


class IdentificationStatus(str, Enum):
    """Verifier-side identification state used by selective outputs."""

    IDENTIFIED = "identified"
    CONTRADICTED = "contradicted"
    UNDERDETERMINED = "underdetermined"


IDENTIFICATION_STATUS_SPACE: tuple[str, ...] = tuple(status.value for status in IdentificationStatus)
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


def _coerce_identification_status(
    value: IdentificationStatus | str | None,
    *,
    field_name: str,
) -> IdentificationStatus | None:
    if value is None:
        return None
    if isinstance(value, IdentificationStatus):
        return value
    try:
        return IdentificationStatus(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of {IDENTIFICATION_STATUS_SPACE}, got {value!r}."
        ) from exc


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
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
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


def _normalize_public_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _mentions_forbidden_token(text: str, *, forbidden_tokens: list[str] | tuple[str, ...]) -> bool:
    for raw_token in forbidden_tokens:
        token = str(raw_token).strip()
        if not token:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def _sanitize_public_text(
    value: Any,
    *,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    normalized = _normalize_public_text(value)
    if normalized is None:
        return None
    if forbidden_tokens and _mentions_forbidden_token(normalized, forbidden_tokens=forbidden_tokens):
        return None
    return normalized


def _normalize_public_text_list(
    value: Any,
    *,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if isinstance(value, str):
        candidate = _sanitize_public_text(value, forbidden_tokens=forbidden_tokens)
        return [candidate] if candidate is not None else []
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        candidate = _sanitize_public_text(item, forbidden_tokens=forbidden_tokens)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def sanitize_public_variable_descriptions(
    value: Any,
    *,
    observed_variables: list[str] | tuple[str, ...] | None = None,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    observed_variable_set = set(_normalize_str_list(observed_variables))
    descriptions: dict[str, str] = {}
    for raw_variable, raw_description in value.items():
        variable = _normalize_public_text(raw_variable)
        if variable is None or (observed_variable_set and variable not in observed_variable_set):
            continue
        description: str | None = None
        if isinstance(raw_description, dict):
            for candidate_key in ("public", "description", "text", "summary"):
                description = _sanitize_public_text(
                    raw_description.get(candidate_key),
                    forbidden_tokens=forbidden_tokens,
                )
                if description is not None:
                    break
        else:
            description = _sanitize_public_text(
                raw_description,
                forbidden_tokens=forbidden_tokens,
            )
        if description is not None:
            descriptions[variable] = description
    return descriptions


def sanitize_public_measurement_semantics(
    value: Any,
    *,
    observed_variables: list[str] | tuple[str, ...] | None = None,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    observed_variable_set = set(_normalize_str_list(observed_variables))
    semantics: dict[str, dict[str, Any]] = {}
    for raw_variable, raw_entry in value.items():
        variable = _normalize_public_text(raw_variable)
        if (
            variable is None
            or (observed_variable_set and variable not in observed_variable_set)
            or not isinstance(raw_entry, dict)
        ):
            continue
        entry: dict[str, Any] = {}
        measurement_view = _sanitize_public_text(
            raw_entry.get("measurement_view"),
            forbidden_tokens=forbidden_tokens,
        )
        if measurement_view is not None:
            entry["measurement_view"] = measurement_view
        notes = _normalize_public_text_list(
            raw_entry.get("notes"),
            forbidden_tokens=forbidden_tokens,
        )
        if notes:
            entry["notes"] = notes
        if entry:
            semantics[variable] = entry
    return semantics


def _sanitize_public_metadata_value(
    key: str,
    value: Any,
    *,
    observed_variables: list[str] | tuple[str, ...] | None = None,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> Any:
    if key == "variable_descriptions":
        return sanitize_public_variable_descriptions(
            value,
            observed_variables=observed_variables,
            forbidden_tokens=forbidden_tokens,
        )
    if key == "measurement_semantics":
        return sanitize_public_measurement_semantics(
            value,
            observed_variables=observed_variables,
            forbidden_tokens=forbidden_tokens,
        )
    return _serialize_json_safe(value)


def _sanitize_metadata(
    value: Any,
    *,
    root: bool = False,
    observed_variables: list[str] | tuple[str, ...] | None = None,
    forbidden_tokens: list[str] | tuple[str, ...] | None = None,
) -> Any:
    if isinstance(value, dict):
        if root:
            sanitized: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                if key in _SANITIZED_METADATA_KEYS or key not in _PUBLIC_METADATA_ALLOWED_KEYS:
                    continue
                cleaned = _sanitize_public_metadata_value(
                    key,
                    item,
                    observed_variables=observed_variables,
                    forbidden_tokens=forbidden_tokens,
                )
                if cleaned is None:
                    continue
                sanitized[key] = cleaned
            return sanitized
        return {
            key: _sanitize_metadata(
                item,
                observed_variables=observed_variables,
                forbidden_tokens=forbidden_tokens,
            )
            for raw_key, item in value.items()
            for key in (str(raw_key),)
            if key not in _SANITIZED_METADATA_KEYS
        }
    if isinstance(value, list):
        return [
            _sanitize_metadata(
                item,
                root=False,
                observed_variables=observed_variables,
                forbidden_tokens=forbidden_tokens,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_metadata(
                item,
                root=False,
                observed_variables=observed_variables,
                forbidden_tokens=forbidden_tokens,
            )
            for item in value
        )
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


def _normalize_selective_confidence(value: Any) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not (0.0 <= normalized <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {value!r}.")
    return normalized


def _normalize_selective_probabilities(value: Any) -> dict[str, float]:
    mapping = _normalize_optional_mapping(value, none_as_empty=True) or {}
    unknown_labels = sorted(set(str(key) for key in mapping) - set(VERDICT_LABEL_SPACE))
    if unknown_labels:
        raise ValueError(
            f"probabilities contains labels outside the frozen verdict space: {unknown_labels!r}."
        )

    normalized = {label: 0.0 for label in VERDICT_LABEL_SPACE}
    for raw_key, raw_value in mapping.items():
        weight = float(raw_value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"probabilities must contain finite non-negative weights, got {raw_value!r}.")
        normalized[str(raw_key)] = weight

    total = sum(normalized.values())
    if total == 0.0:
        return normalized
    return {label: round(normalized[label] / total, 12) for label in VERDICT_LABEL_SPACE}


def _public_scenario_id(raw_scenario_id: str) -> str:
    digest = hashlib.sha256(str(raw_scenario_id).encode("utf-8")).hexdigest()[:12]
    return f"public_case_{digest}"


def _public_description(
    *,
    observed_variables: list[str],
    treatment: str | None,
    outcome: str | None,
    causal_level: int | str,
) -> str:
    visible = list(observed_variables[:4])
    variable_text = ", ".join(visible) if visible else "the observed variables"
    if len(observed_variables) > len(visible) and variable_text != "the observed variables":
        variable_text = f"{variable_text}, ..."
    normalized_level = str(causal_level).strip().upper()
    if normalized_level in {"1", "2", "3"}:
        normalized_level = f"L{normalized_level}"
    if treatment and outcome:
        return (
            f"Observed {normalized_level or 'public'} case over {variable_text}. "
            f"Evaluate claims about {treatment} and {outcome} using only the public evidence in this view."
        )
    return (
        f"Observed {normalized_level or 'public'} case over {variable_text}. "
        "Evaluate claims using only the public evidence in this view."
    )


def _frame_variable_names(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return _normalize_str_list(list(frame.columns))


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


def _normalize_optional_mapping(value: Any, *, none_as_empty: bool) -> dict[str, Any] | None:
    if value is None:
        return {} if none_as_empty else None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        serialized = value.to_dict()
        if isinstance(serialized, dict):
            return dict(serialized)
    raise TypeError(f"Expected mapping-compatible payload, got {type(value)!r}.")


def _normalize_mapping_sequence(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list/tuple of mappings, got {type(value)!r}.")

    normalized: list[dict[str, Any]] = []
    for item in value:
        mapping = _normalize_optional_mapping(item, none_as_empty=False)
        if mapping is None:
            raise TypeError(f"{field_name} items must be mapping-compatible, got {type(item)!r}.")
        normalized.append(mapping)
    return normalized


def default_identification_status(
    label: VerdictLabel | str | None,
) -> IdentificationStatus | None:
    normalized_label = _coerce_label(label, field_name="label") if label is not None else None
    if normalized_label is VerdictLabel.VALID:
        return IdentificationStatus.IDENTIFIED
    if normalized_label is VerdictLabel.INVALID:
        return IdentificationStatus.CONTRADICTED
    if normalized_label is VerdictLabel.UNIDENTIFIABLE:
        return IdentificationStatus.UNDERDETERMINED
    return None


@dataclass(slots=True)
class MissingInformationSpec:
    """Structured description of what public evidence is still missing."""

    missing_assumptions: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        self.missing_assumptions = _normalize_str_list(self.missing_assumptions)
        self.required_evidence = _normalize_str_list(self.required_evidence)
        self.note = str(self.note).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_assumptions": list(self.missing_assumptions),
            "required_evidence": list(self.required_evidence),
            "note": self.note,
        }


def _normalize_missing_information_spec(
    value: MissingInformationSpec | dict[str, Any] | None,
) -> MissingInformationSpec:
    if value is None:
        return MissingInformationSpec()
    if isinstance(value, MissingInformationSpec):
        return MissingInformationSpec(
            missing_assumptions=list(value.missing_assumptions),
            required_evidence=list(value.required_evidence),
            note=value.note,
        )
    if isinstance(value, dict):
        raw_note = value.get("note")
        if raw_note is None:
            raw_note = value.get("reasoning_summary", "")
        return MissingInformationSpec(
            missing_assumptions=list(
                value.get("missing_assumptions", value.get("unresolved_assumptions", [])) or []
            ),
            required_evidence=list(
                value.get("required_evidence", value.get("required_observations", [])) or []
            ),
            note=_coerce_optional_string(raw_note) or "",
        )
    raise TypeError(f"Unsupported missing_information_spec type: {type(value)!r}")


def derive_missing_information_spec(
    *,
    value: MissingInformationSpec | dict[str, Any] | None,
    label: VerdictLabel | str | None,
    assumption_ledger: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    reasoning_summary: str = "",
) -> MissingInformationSpec:
    spec = _normalize_missing_information_spec(value)
    normalized_label = _coerce_label(label, field_name="label") if label is not None else None
    if normalized_label is VerdictLabel.UNIDENTIFIABLE and not spec.missing_assumptions:
        spec.missing_assumptions = _normalize_str_list(
            [
                entry.get("name")
                for entry in list(assumption_ledger or [])
                if str(entry.get("status", "")).strip().lower() == "unresolved"
            ]
        )
    if normalized_label is VerdictLabel.UNIDENTIFIABLE and not spec.note and str(reasoning_summary).strip():
        spec.note = str(reasoning_summary).strip()
    return spec


def _has_missing_information(spec: MissingInformationSpec) -> bool:
    return bool(spec.missing_assumptions or spec.required_evidence or spec.note)


def derive_refusal_reason(
    *,
    explicit_reason: str | None,
    label: VerdictLabel | str | None,
    missing_information_spec: MissingInformationSpec,
    countermodel_witness: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    normalized = _coerce_optional_string(explicit_reason)
    if normalized is not None:
        return normalized

    normalized_label = _coerce_label(label, field_name="label") if label is not None else None
    if normalized_label is not VerdictLabel.UNIDENTIFIABLE:
        return None

    metadata_payload = _normalize_optional_mapping(metadata, none_as_empty=True) or {}
    countermodel_payload = _normalize_optional_mapping(countermodel_witness, none_as_empty=False) or {}
    witness_payload = _normalize_optional_mapping(countermodel_payload.get("payload"), none_as_empty=True) or {}

    if bool(witness_payload.get("query_disagreement")):
        return "observational_equivalence"
    if metadata_payload.get("stage_variant") == "abstention_gate":
        return "missing_primary_identifying_support"
    if missing_information_spec.missing_assumptions:
        return "missing_identifying_support"
    return "insufficient_public_information"


def validate_selective_verdict_state(
    *,
    label: VerdictLabel | None,
    identification_status: IdentificationStatus | None,
    refusal_reason: str | None,
    missing_information_spec: MissingInformationSpec,
) -> None:
    if label is None:
        if identification_status is not None or refusal_reason is not None or _has_missing_information(missing_information_spec):
            raise ValueError("Selective verdict fields require a non-null label.")
        return

    expected_status = default_identification_status(label)
    if identification_status is not None and expected_status is not None and identification_status is not expected_status:
        raise ValueError(
            "identification_status must be consistent with label: "
            f"label={label.value!r}, identification_status={identification_status.value!r}."
        )

    if label is not VerdictLabel.UNIDENTIFIABLE:
        if refusal_reason is not None:
            raise ValueError(f"refusal_reason is only valid for label='unidentifiable', got {label.value!r}.")
        if _has_missing_information(missing_information_spec):
            raise ValueError(
                "missing_information_spec must be empty unless label='unidentifiable'."
            )


@dataclass(slots=True)
class VerifierVerdict:
    """Structured verdict for evaluation.

    ``winner`` remains a legacy debate-game outcome and is intentionally not the
    paper's supervision target. Evaluators should compare ``label`` against
    ``gold_label``.
    """

    label: VerdictLabel | str | None = None
    identification_status: IdentificationStatus | str | None = None
    refusal_reason: str | None = None
    missing_information_spec: MissingInformationSpec | dict[str, Any] | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    assumption_ledger: list[dict[str, Any]] = field(default_factory=list)
    witness: dict[str, Any] | None = None
    support_witness: dict[str, Any] | None = None
    countermodel_witness: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    reasoning_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _coerce_label(self.label, field_name="verdict.label")
        self.reasoning_summary = str(self.reasoning_summary).strip()
        self.identification_status = _coerce_identification_status(
            self.identification_status,
            field_name="verdict.identification_status",
        ) or default_identification_status(self.label)
        self.assumption_ledger = _normalize_mapping_sequence(
            self.assumption_ledger,
            field_name="verdict.assumption_ledger",
        )
        self.missing_information_spec = derive_missing_information_spec(
            value=self.missing_information_spec,
            label=self.label,
            assumption_ledger=self.assumption_ledger,
            reasoning_summary=self.reasoning_summary,
        )
        self.probabilities = _normalize_selective_probabilities(self.probabilities)
        self.witness = _normalize_optional_mapping(self.witness, none_as_empty=False)
        self.support_witness = _normalize_optional_mapping(self.support_witness, none_as_empty=False)
        self.countermodel_witness = _normalize_optional_mapping(
            self.countermodel_witness,
            none_as_empty=False,
        )
        self.tool_trace = _normalize_mapping_sequence(
            self.tool_trace,
            field_name="verdict.tool_trace",
        )
        self.metadata = _normalize_optional_mapping(self.metadata, none_as_empty=True) or {}
        self.refusal_reason = derive_refusal_reason(
            explicit_reason=self.refusal_reason,
            label=self.label,
            missing_information_spec=self.missing_information_spec,
            countermodel_witness=self.countermodel_witness,
            metadata=self.metadata,
        )
        validate_selective_verdict_state(
            label=self.label,
            identification_status=self.identification_status,
            refusal_reason=self.refusal_reason,
            missing_information_spec=self.missing_information_spec,
        )
        self.confidence = _normalize_selective_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value if self.label is not None else None,
            "final_verdict": self.label.value if self.label is not None else None,
            "identification_status": (
                self.identification_status.value if self.identification_status is not None else None
            ),
            "refusal_reason": self.refusal_reason,
            "missing_information_spec": self.missing_information_spec.to_dict(),
            "probabilities": dict(self.probabilities),
            "assumption_ledger": [dict(entry) for entry in self.assumption_ledger],
            "witness": dict(self.witness) if self.witness is not None else None,
            "support_witness": (
                dict(self.support_witness) if self.support_witness is not None else None
            ),
            "countermodel_witness": (
                dict(self.countermodel_witness) if self.countermodel_witness is not None else None
            ),
            "tool_trace": [dict(entry) for entry in self.tool_trace],
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
            identification_status=value.identification_status,
            refusal_reason=value.refusal_reason,
            missing_information_spec=value.missing_information_spec,
            probabilities=dict(value.probabilities),
            assumption_ledger=list(value.assumption_ledger),
            witness=value.witness,
            support_witness=value.support_witness,
            countermodel_witness=value.countermodel_witness,
            tool_trace=list(value.tool_trace),
            confidence=value.confidence,
            reasoning_summary=value.reasoning_summary,
            metadata=dict(value.metadata),
        )
    if isinstance(value, dict):
        return VerifierVerdict(
            label=value.get("label", value.get("final_verdict")),
            identification_status=value.get("identification_status"),
            refusal_reason=value.get("refusal_reason"),
            missing_information_spec=value.get("missing_information_spec"),
            probabilities=value.get("probabilities", {}),
            assumption_ledger=value.get("assumption_ledger", []),
            witness=value.get("witness"),
            support_witness=value.get("support_witness"),
            countermodel_witness=value.get("countermodel_witness"),
            tool_trace=value.get("tool_trace", []),
            confidence=value.get("confidence"),
            reasoning_summary=str(value.get("reasoning_summary", "")),
            metadata=value.get("metadata"),
        )
    raise TypeError(f"Unsupported verdict type: {type(value)!r}")


@dataclass(slots=True)
class PublicCausalInstance:
    """Verifier-visible scenario view with oracle fields stripped out."""

    scenario_id: str
    description: str
    variables: list[str]
    proxy_variables: list[str] = field(default_factory=list)
    instrument_variables: list[str] = field(default_factory=list)
    mediator_variables: list[str] = field(default_factory=list)
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
        observed_variables = _frame_variable_names(observed)
        provided_variables = _normalize_str_list(self.variables)
        self.variables = list(observed_variables) if observed_variables else provided_variables
        metadata = dict(self.metadata)
        self.proxy_variables = _normalize_public_variables(
            list(self.proxy_variables) + list(metadata.pop("proxy_variables", [])),
            observed_variables=list(self.variables),
        )
        metadata.pop("instrument_variables", None)
        metadata.pop("mediator_variables", None)
        metadata.pop("selection_variables", None)
        # Legacy structural role fields remain on the dataclass for compatibility,
        # but the public contract no longer carries them to verifier-side callers.
        self.instrument_variables = []
        self.mediator_variables = []
        self.selection_variables = []
        self.selection_mechanism = _coerce_optional_string(
            self.selection_mechanism
            if self.selection_mechanism is not None
            else metadata.pop("selection_mechanism", None)
        )
        self.observed_data = observed
        self.data = _copy_frame(observed)
        self.difficulty = 0.5
        self.difficulty_config = {}
        self.metadata = dict(
            _sanitize_metadata(
                metadata,
                root=True,
                observed_variables=list(self.variables),
                forbidden_tokens=(),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "variables": list(self.variables),
            "proxy_variables": list(self.proxy_variables),
            "selection_mechanism": self.selection_mechanism,
            "observed_data": _serialize_frame(self.observed_data),
            "data": _serialize_frame(self.data),
            "causal_level": self.causal_level,
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
        self.hidden_variables = _normalize_str_list(self.hidden_variables)
        observed = _copy_frame(self.observed_data)
        full = _copy_frame(self.full_data)
        data = _copy_frame(self.data)
        hidden_variable_set = set(self.hidden_variables)

        if observed.empty and not data.empty:
            observed = data.copy(deep=True)
        if observed.empty and not full.empty:
            observed = full.drop(columns=self.hidden_variables, errors="ignore").copy(deep=True)
        if not observed.empty:
            observed = observed.drop(columns=self.hidden_variables, errors="ignore").copy(deep=True)
        if full.empty:
            full = observed.copy(deep=True)
        if data.empty:
            data = observed.copy(deep=True)
        elif not data.empty:
            data = data.drop(columns=self.hidden_variables, errors="ignore").copy(deep=True)

        observed_variables = _frame_variable_names(observed)
        provided_variables = _normalize_str_list(self.variables)
        self.variables = (
            list(observed_variables)
            if observed_variables
            else [value for value in provided_variables if value not in hidden_variable_set]
        )
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

        public_scenario_id = _coerce_optional_string(self.metadata.get("public_scenario_id"))
        forbidden_tokens = list(self.hidden_variables)
        public_description = _sanitize_public_text(
            self.metadata.get("public_description"),
            forbidden_tokens=forbidden_tokens,
        )
        treatment = _coerce_optional_string(self.ground_truth.get("treatment"))
        outcome = _coerce_optional_string(self.ground_truth.get("outcome"))
        selection_mechanism = _sanitize_public_text(
            self.metadata.get("selection_mechanism"),
            forbidden_tokens=forbidden_tokens,
        )
        public_metadata = dict(
            _sanitize_metadata(
                self.metadata,
                root=True,
                observed_variables=list(self.variables),
                forbidden_tokens=forbidden_tokens,
            )
        )
        return PublicCausalInstance(
            scenario_id=public_scenario_id or _public_scenario_id(self.scenario_id),
            description=public_description
            or _public_description(
                observed_variables=list(self.variables),
                treatment=treatment,
                outcome=outcome,
                causal_level=self.causal_level,
            ),
            variables=list(self.variables),
            proxy_variables=list(self.metadata.get("proxy_variables", [])),
            selection_mechanism=selection_mechanism,
            observed_data=self.observed_data.copy(deep=True),
            data=self.observed_data.copy(deep=True),
            causal_level=self.causal_level,
            difficulty=self.difficulty,
            difficulty_config=_sanitize_difficulty_config(self.difficulty_config),
            metadata=public_metadata,
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
    version: str = "v2"
    train: list[str] = field(default_factory=list)
    dev: list[str] = field(default_factory=list)
    test_iid: list[str] = field(default_factory=list)
    test_ood: list[str] = field(default_factory=list)
    family_holdout: list[str] = field(default_factory=list)
    lexical_holdout: list[str] = field(default_factory=list)
    variable_renaming_holdout: bool = False
    mechanism_holdout: list[str] = field(default_factory=list)
    attack_family_holdout: list[str] = field(default_factory=list)
    context_shift_holdout: list[str] = field(default_factory=list)
    paired_flip_holdout: bool = False
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
        self.mechanism_holdout = _normalize_str_list(self.mechanism_holdout)
        self.attack_family_holdout = _normalize_str_list(self.attack_family_holdout)
        self.context_shift_holdout = _normalize_str_list(self.context_shift_holdout)
        self.paired_flip_holdout = bool(self.paired_flip_holdout)
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
                "mechanism_holdout": list(self.mechanism_holdout),
                "attack_family_holdout": list(self.attack_family_holdout),
                "context_shift_holdout": list(self.context_shift_holdout),
                "paired_flip_holdout": self.paired_flip_holdout,
            },
            "metadata": _serialize_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkSplitManifest":
        splits = dict(payload.get("splits", {}))
        holdout_strategy = dict(payload.get("holdout_strategy", {}))
        return cls(
            dataset_name=str(payload.get("dataset_name", "causal_oversight_benchmark")),
            version=str(payload.get("version", "v2")),
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
            mechanism_holdout=list(
                holdout_strategy.get("mechanism_holdout", payload.get("mechanism_holdout", []))
            ),
            attack_family_holdout=list(
                holdout_strategy.get(
                    "attack_family_holdout",
                    payload.get("attack_family_holdout", []),
                )
            ),
            context_shift_holdout=list(
                holdout_strategy.get(
                    "context_shift_holdout",
                    payload.get("context_shift_holdout", []),
                )
            ),
            paired_flip_holdout=holdout_strategy.get(
                "paired_flip_holdout",
                payload.get("paired_flip_holdout", False),
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


def ensure_claim_instance(claim: ClaimInstance | dict[str, Any] | Any) -> ClaimInstance:
    """Normalize a claim-like payload into ClaimInstance."""

    if isinstance(claim, ClaimInstance):
        return ClaimInstance.from_dict(claim.to_dict())
    if isinstance(claim, dict):
        return ClaimInstance.from_dict(dict(claim))
    to_dict = getattr(claim, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return ClaimInstance.from_dict(payload)
    raise TypeError(f"Unsupported claim payload type: {type(claim)!r}")


def require_public_instance(scenario: PublicCausalInstance | Any) -> PublicCausalInstance:
    """Enforce the strict verifier-side contract: only public instances are accepted."""

    if isinstance(scenario, PublicCausalInstance):
        return scenario
    raise TypeError(
        "Verifier-side entrypoints only accept PublicCausalInstance. "
        f"Received {type(scenario)!r}."
    )
