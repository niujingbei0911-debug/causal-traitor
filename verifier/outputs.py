"""Structured verifier-side outputs for claim parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from benchmark.schema import (
    derive_missing_information_spec,
    IdentificationStatus,
    MissingInformationSpec,
    VerdictLabel,
    VERDICT_LABEL_SPACE,
    default_identification_status,
    derive_refusal_reason,
    validate_selective_verdict_state,
)


class QueryType(str, Enum):
    """Coarse causal query categories used by the verifier pipeline."""

    ASSOCIATION = "association"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"


class ClaimPolarity(str, Enum):
    """Directional stance taken by the parsed claim."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"


class ClaimStrength(str, Enum):
    """Rhetorical strength of the claim text."""

    TENTATIVE = "tentative"
    STRONG = "strong"
    ABSOLUTE = "absolute"


def _normalize_unique_strings(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if values is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _coerce_query_type(value: QueryType | str) -> QueryType:
    if isinstance(value, QueryType):
        return value
    return QueryType(str(value).strip().lower())


def _coerce_claim_polarity(value: ClaimPolarity | str) -> ClaimPolarity:
    if isinstance(value, ClaimPolarity):
        return value
    return ClaimPolarity(str(value).strip().lower())


def _coerce_claim_strength(value: ClaimStrength | str) -> ClaimStrength:
    if isinstance(value, ClaimStrength):
        return value
    return ClaimStrength(str(value).strip().lower())


def _coerce_verdict_label(value: VerdictLabel | str) -> VerdictLabel:
    if isinstance(value, VerdictLabel):
        return value
    return VerdictLabel(str(value).strip().lower())


def _coerce_identification_status(value: IdentificationStatus | str | None) -> IdentificationStatus | None:
    if value is None:
        return None
    if isinstance(value, IdentificationStatus):
        return value
    return IdentificationStatus(str(value).strip().lower())


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_payload_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        serialized = value.to_dict()
        if isinstance(serialized, dict):
            return dict(serialized)
    raise TypeError(f"Expected mapping-compatible payload, got {type(value)!r}.")


def _normalize_payload_trace(value: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif hasattr(item, "to_dict") and callable(item.to_dict):
            serialized = item.to_dict()
            if not isinstance(serialized, dict):
                raise TypeError(f"tool_trace items must serialize to dict, got {type(serialized)!r}.")
            normalized.append(dict(serialized))
        else:
            raise TypeError(f"tool_trace items must be mapping-compatible, got {type(item)!r}.")
    return normalized


def _normalize_assumption_ledger(value: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif hasattr(item, "to_dict") and callable(item.to_dict):
            serialized = item.to_dict()
            if isinstance(serialized, dict):
                normalized.append(dict(serialized))
            else:
                raise TypeError(
                    f"assumption_ledger items must serialize to dict, got {type(serialized)!r}."
                )
        else:
            raise TypeError(f"assumption_ledger items must be mapping-compatible, got {type(item)!r}.")
    return normalized


def _normalize_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not (0.0 <= normalized <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {value!r}.")
    return normalized


def _normalize_probabilities(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"probabilities must be a mapping, got {type(value)!r}.")

    unknown_labels = sorted(set(str(key) for key in value) - set(VERDICT_LABEL_SPACE))
    if unknown_labels:
        raise ValueError(
            f"probabilities contains labels outside the frozen verdict space: {unknown_labels!r}."
        )

    weights: dict[str, float] = {label: 0.0 for label in VERDICT_LABEL_SPACE}
    for raw_key, raw_value in value.items():
        weight = float(raw_value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"probabilities must contain finite non-negative weights, got {raw_value!r}.")
        weights[str(raw_key)] = weight

    total = sum(weights.values())
    if total == 0.0:
        return weights

    return {
        label: round(weights[label] / total, 12)
        for label in VERDICT_LABEL_SPACE
    }


@dataclass(slots=True)
class ParsedClaim:
    """Machine-readable representation of a natural-language causal claim."""

    query_type: QueryType | str
    treatment: str
    outcome: str
    claim_polarity: ClaimPolarity | str = ClaimPolarity.NULL
    claim_strength: ClaimStrength | str = ClaimStrength.TENTATIVE
    mentioned_assumptions: list[str] = field(default_factory=list)
    implied_assumptions: list[str] = field(default_factory=list)
    rhetorical_strategy: str = "plain_causal_assertion"
    abstention_risk_cues: list[str] = field(default_factory=list)
    needs_abstention_check: bool = False

    def __post_init__(self) -> None:
        self.query_type = _coerce_query_type(self.query_type)
        self.treatment = str(self.treatment).strip()
        self.outcome = str(self.outcome).strip()
        self.claim_polarity = _coerce_claim_polarity(self.claim_polarity)
        self.claim_strength = _coerce_claim_strength(self.claim_strength)
        self.mentioned_assumptions = _normalize_unique_strings(self.mentioned_assumptions)
        self.implied_assumptions = _normalize_unique_strings(self.implied_assumptions)
        self.rhetorical_strategy = str(self.rhetorical_strategy).strip() or "plain_causal_assertion"
        self.abstention_risk_cues = _normalize_unique_strings(self.abstention_risk_cues)
        self.needs_abstention_check = bool(self.needs_abstention_check or self.abstention_risk_cues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "claim_polarity": self.claim_polarity.value,
            "claim_strength": self.claim_strength.value,
            "explicit_assumptions": list(self.mentioned_assumptions),
            "mentioned_assumptions": list(self.mentioned_assumptions),
            "implied_assumptions": list(self.implied_assumptions),
            "rhetorical_strategy": self.rhetorical_strategy,
            "abstention_risk_cues": list(self.abstention_risk_cues),
            "needs_abstention_check": self.needs_abstention_check,
        }


@dataclass(slots=True)
class SelectiveVerifierOutput:
    """Refusal-aware verifier output contract with explicit identification state."""

    label: VerdictLabel | str
    confidence: float | None = None
    identification_status: IdentificationStatus | str | None = None
    refusal_reason: str | None = None
    missing_information_spec: MissingInformationSpec | dict[str, Any] | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    assumption_ledger: list[dict[str, Any]] = field(default_factory=list)
    witness: dict[str, Any] | None = None
    support_witness: dict[str, Any] | None = None
    countermodel_witness: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _coerce_verdict_label(self.label)
        self.confidence = _normalize_confidence(self.confidence)
        self.reasoning_summary = str(self.reasoning_summary).strip()
        self.assumption_ledger = _normalize_assumption_ledger(self.assumption_ledger)
        self.metadata = dict(self.metadata or {})
        self.witness = _normalize_payload_mapping(self.witness)
        self.support_witness = _normalize_payload_mapping(self.support_witness)
        self.countermodel_witness = _normalize_payload_mapping(self.countermodel_witness)
        self.tool_trace = _normalize_payload_trace(self.tool_trace)
        self.probabilities = _normalize_probabilities(self.probabilities)
        self.identification_status = _coerce_identification_status(
            self.identification_status
        ) or default_identification_status(self.label)
        self.missing_information_spec = derive_missing_information_spec(
            value=self.missing_information_spec,
            label=self.label,
            assumption_ledger=self.assumption_ledger,
            reasoning_summary=self.reasoning_summary,
        )
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "final_verdict": self.label.value,
            "identification_status": (
                self.identification_status.value if self.identification_status is not None else None
            ),
            "refusal_reason": self.refusal_reason,
            "missing_information_spec": self.missing_information_spec.to_dict(),
            "confidence": self.confidence,
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
            "reasoning_summary": self.reasoning_summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_decision_payload(cls, payload: dict[str, Any] | Any) -> "SelectiveVerifierOutput":
        if hasattr(payload, "to_dict") and callable(payload.to_dict):
            payload = payload.to_dict()
        data = dict(payload)
        return cls(
            label=data.get("label", data.get("final_verdict")),
            confidence=data.get("confidence"),
            identification_status=data.get("identification_status"),
            refusal_reason=data.get("refusal_reason"),
            missing_information_spec=data.get("missing_information_spec"),
            probabilities=data.get("probabilities"),
            assumption_ledger=data.get("assumption_ledger"),
            witness=data.get("witness"),
            support_witness=data.get("support_witness"),
            countermodel_witness=data.get("countermodel_witness"),
            tool_trace=data.get("tool_trace"),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            metadata=data.get("metadata"),
        )
