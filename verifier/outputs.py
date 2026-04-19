"""Structured verifier-side outputs for claim parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmark.schema import (
    IdentificationStatus,
    MissingInformationSpec,
    VerdictLabel,
    default_identification_status,
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
    return {"value": value}


def _normalize_payload_trace(value: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
        elif hasattr(item, "to_dict") and callable(item.to_dict):
            serialized = item.to_dict()
            normalized.append(dict(serialized) if isinstance(serialized, dict) else {"value": serialized})
        else:
            normalized.append({"summary": str(item)})
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
    return normalized


def _derive_missing_information_spec(
    *,
    value: MissingInformationSpec | dict[str, Any] | None,
    label: VerdictLabel,
    assumption_ledger: list[dict[str, Any]],
    reasoning_summary: str,
) -> MissingInformationSpec:
    if isinstance(value, MissingInformationSpec):
        spec = MissingInformationSpec(
            missing_assumptions=list(value.missing_assumptions),
            required_evidence=list(value.required_evidence),
            note=value.note,
        )
    elif isinstance(value, dict):
        raw_note = value.get("note")
        if raw_note is None:
            raw_note = value.get("reasoning_summary", "")
        spec = MissingInformationSpec(
            missing_assumptions=list(
                value.get("missing_assumptions", value.get("unresolved_assumptions", [])) or []
            ),
            required_evidence=list(
                value.get("required_evidence", value.get("required_observations", [])) or []
            ),
            note=_coerce_optional_string(raw_note) or "",
        )
    else:
        spec = MissingInformationSpec()

    if label is VerdictLabel.UNIDENTIFIABLE and not spec.missing_assumptions:
        spec.missing_assumptions = _normalize_unique_strings(
            [
                entry.get("name")
                for entry in assumption_ledger
                if str(entry.get("status", "")).strip().lower() == "unresolved"
            ]
        )
    if label is VerdictLabel.UNIDENTIFIABLE and not spec.note and reasoning_summary:
        spec.note = reasoning_summary
    return spec


def _derive_refusal_reason(
    *,
    explicit_reason: str | None,
    label: VerdictLabel,
    missing_information_spec: MissingInformationSpec,
    countermodel_witness: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> str | None:
    normalized = _coerce_optional_string(explicit_reason)
    if normalized is not None:
        return normalized
    if label is not VerdictLabel.UNIDENTIFIABLE:
        return None
    countermodel_payload = dict(countermodel_witness.get("payload", {})) if countermodel_witness else {}
    if bool(countermodel_payload.get("query_disagreement")):
        return "observational_equivalence"
    if metadata.get("stage_variant") == "abstention_gate":
        return "missing_primary_identifying_support"
    if missing_information_spec.missing_assumptions:
        return "missing_identifying_support"
    return "insufficient_public_information"


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
        self.needs_abstention_check = bool(self.needs_abstention_check)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "claim_polarity": self.claim_polarity.value,
            "claim_strength": self.claim_strength.value,
            "mentioned_assumptions": list(self.mentioned_assumptions),
            "implied_assumptions": list(self.implied_assumptions),
            "rhetorical_strategy": self.rhetorical_strategy,
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
        self.confidence = None if self.confidence is None else float(self.confidence)
        self.reasoning_summary = str(self.reasoning_summary).strip()
        self.assumption_ledger = _normalize_assumption_ledger(self.assumption_ledger)
        self.metadata = dict(self.metadata)
        self.witness = _normalize_payload_mapping(self.witness)
        self.support_witness = _normalize_payload_mapping(self.support_witness)
        self.countermodel_witness = _normalize_payload_mapping(self.countermodel_witness)
        self.tool_trace = _normalize_payload_trace(self.tool_trace)
        self.probabilities = {
            str(key): float(value)
            for key, value in dict(self.probabilities).items()
        }
        self.identification_status = _coerce_identification_status(
            self.identification_status
        ) or default_identification_status(self.label)
        self.missing_information_spec = _derive_missing_information_spec(
            value=self.missing_information_spec,
            label=self.label,
            assumption_ledger=self.assumption_ledger,
            reasoning_summary=self.reasoning_summary,
        )
        self.refusal_reason = _derive_refusal_reason(
            explicit_reason=self.refusal_reason,
            label=self.label,
            missing_information_spec=self.missing_information_spec,
            countermodel_witness=self.countermodel_witness,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
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
            probabilities=dict(data.get("probabilities", {})),
            assumption_ledger=list(data.get("assumption_ledger", [])),
            witness=data.get("witness"),
            support_witness=data.get("support_witness"),
            countermodel_witness=data.get("countermodel_witness"),
            tool_trace=list(data.get("tool_trace", [])),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            metadata=dict(data.get("metadata", {})),
        )
