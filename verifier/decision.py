"""Final four-stage decision rule for verifier-side adjudication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.schema import VerdictLabel, Witness, WitnessKind
from verifier.assumption_ledger import (
    AssumptionLedger,
    AssumptionLedgerEntry,
    AssumptionStatus,
)
from verifier.countermodel_search import CountermodelSearchResult
from verifier.outputs import ParsedClaim

_BASELINE_ASSUMPTIONS: frozenset[str] = frozenset({"consistency", "positivity"})


def _coerce_label(value: VerdictLabel | str) -> VerdictLabel:
    if isinstance(value, VerdictLabel):
        return value
    return VerdictLabel(str(value).strip().lower())


def _normalize_tool_trace(tool_trace: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    if tool_trace is None:
        return []

    normalized: list[dict[str, Any]] = []
    for item in tool_trace:
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"summary": str(item)})
    return normalized


def _normalize_assumption_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _tool_supported_assumptions(tool_trace: list[dict[str, Any]]) -> set[str]:
    supported: set[str] = set()
    for record in tool_trace:
        for key in ("supports_assumptions", "supported_assumptions", "supports"):
            supported.update(_normalize_assumption_names(record.get(key)))
    return supported


def _tool_contradicted_assumptions(tool_trace: list[dict[str, Any]]) -> set[str]:
    contradicted: set[str] = set()
    for record in tool_trace:
        for key in ("contradicts_assumptions", "contradicted_assumptions", "contradicts"):
            contradicted.update(_normalize_assumption_names(record.get(key)))
    return contradicted


def _record_supports_primary_claim(record: dict[str, Any]) -> bool:
    if "supports_primary_claim" in record:
        return bool(record.get("supports_primary_claim"))

    claim_stance = str(record.get("claim_stance", "")).strip().lower()
    if claim_stance in {"anti_causal", "anti", "rebuttal"}:
        return False

    if bool(record.get("supports_claim")):
        return True
    if bool(record.get("identified")):
        return True
    verdict_suggestion = record.get("verdict_suggestion")
    return verdict_suggestion is not None and str(verdict_suggestion).strip().lower() == "valid"


def _tool_supports_claim(tool_trace: list[dict[str, Any]]) -> bool:
    for record in tool_trace:
        if _record_supports_primary_claim(record):
            return True
    return False


def _tool_support_confidence(tool_trace: list[dict[str, Any]]) -> float:
    confidences: list[float] = []
    for record in tool_trace:
        if _record_supports_primary_claim(record):
            confidence = record.get("confidence", 0.8)
            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                continue
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _score_ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, float(count) / float(total)))


def _normalize_probabilities(value: Any) -> dict[str, float]:
    normalized = {
        VerdictLabel.VALID.value: 0.0,
        VerdictLabel.INVALID.value: 0.0,
        VerdictLabel.UNIDENTIFIABLE.value: 0.0,
    }
    if isinstance(value, dict):
        for label in normalized:
            try:
                normalized[label] = max(0.0, float(value.get(label, 0.0)))
            except (TypeError, ValueError):
                normalized[label] = 0.0
    total = sum(normalized.values())
    if total <= 0:
        return {
            VerdictLabel.VALID.value: 1.0 / 3.0,
            VerdictLabel.INVALID.value: 1.0 / 3.0,
            VerdictLabel.UNIDENTIFIABLE.value: 1.0 / 3.0,
        }
    return {label: normalized[label] / total for label in normalized}


def _probabilities_from_scores(
    *,
    valid_score: float,
    invalid_score: float,
    unidentifiable_score: float,
) -> dict[str, float]:
    return _normalize_probabilities(
        {
            VerdictLabel.VALID.value: valid_score,
            VerdictLabel.INVALID.value: invalid_score,
            VerdictLabel.UNIDENTIFIABLE.value: unidentifiable_score,
        }
    )


def _ledger_entries_from_tool_support(
    ledger: AssumptionLedger,
    tool_trace: list[dict[str, Any]],
) -> AssumptionLedger:
    supported_by_tools = _tool_supported_assumptions(tool_trace)
    contradicted_by_tools = _tool_contradicted_assumptions(tool_trace)
    entries: list[AssumptionLedgerEntry] = []
    seen_names: set[str] = set()

    for entry in ledger.entries:
        updated_status = entry.status
        updated_note = entry.note

        if entry.name in contradicted_by_tools:
            updated_status = AssumptionStatus.CONTRADICTED
            updated_note = "Tool-backed evidence contradicts this identifying assumption."
        elif entry.name in supported_by_tools and entry.status is AssumptionStatus.UNRESOLVED:
            updated_status = AssumptionStatus.SUPPORTED
            updated_note = "Tool-backed evidence provides direct support for this identifying assumption."

        entries.append(
            AssumptionLedgerEntry(
                name=entry.name,
                source=entry.source,
                category=entry.category,
                status=updated_status,
                note=updated_note,
            )
        )
        seen_names.add(entry.name)

    return AssumptionLedger(entries)


def _unsupported_core_assumptions(ledger: AssumptionLedger) -> list[AssumptionLedgerEntry]:
    supported_names = {
        entry.name
        for entry in ledger.entries
        if entry.status is AssumptionStatus.SUPPORTED
    }
    return [
        entry
        for entry in ledger.entries
        if entry.status is not AssumptionStatus.SUPPORTED
        and not (
            entry.name == "no unobserved confounding"
            and "valid adjustment set" in supported_names
        )
    ]


def _supported_identifying_assumptions(ledger: AssumptionLedger) -> list[AssumptionLedgerEntry]:
    return [
        entry
        for entry in ledger.entries
        if entry.status is AssumptionStatus.SUPPORTED and entry.name not in _BASELINE_ASSUMPTIONS
    ]


def _explicitly_contradicted_assumptions(ledger: AssumptionLedger) -> list[AssumptionLedgerEntry]:
    return [
        entry
        for entry in ledger.entries
        if entry.status is AssumptionStatus.CONTRADICTED and entry.source == "claim explicit"
    ]


def _needs_explicit_identifying_support(
    parsed_claim: ParsedClaim,
    ledger: AssumptionLedger,
) -> bool:
    if parsed_claim.query_type.value in {"intervention", "counterfactual"}:
        return True
    return any(entry.name not in _BASELINE_ASSUMPTIONS for entry in ledger.entries)


def _make_countermodel_witness(
    parsed_claim: ParsedClaim,
    countermodel_result: CountermodelSearchResult,
) -> Witness:
    assumptions: list[str] = []
    for candidate in countermodel_result.candidates:
        if candidate.countermodel_type == countermodel_result.countermodel_type:
            assumptions = list(candidate.triggered_assumptions)
            break

    return Witness(
        witness_type=WitnessKind.COUNTERMODEL,
        description=countermodel_result.countermodel_explanation,
        evidence=[
            f"query_type={parsed_claim.query_type.value}",
            f"countermodel_type={countermodel_result.countermodel_type}",
            f"observational_match_score={countermodel_result.observational_match_score:.2f}",
        ],
        assumptions=assumptions,
        payload=countermodel_result.to_dict(),
        verdict_suggestion=countermodel_result.verdict_suggestion,
        metadata={"decision_stage": "countermodel"},
    )


def _make_assumption_witness(
    ledger: AssumptionLedger,
    *,
    verdict: VerdictLabel = VerdictLabel.UNIDENTIFIABLE,
    description: str | None = None,
    stage: str = "assumption_gate",
) -> Witness:
    unsupported = _unsupported_core_assumptions(ledger)
    resolved_description = description or (
        "Core identification assumptions remain unsupported after countermodel-free review, so the claim stays under-identified."
    )
    evidence = [
        f"{entry.name}: {entry.status.value} ({entry.source})."
        for entry in unsupported[:4]
    ]
    return Witness(
        witness_type=WitnessKind.ASSUMPTION,
        description=resolved_description,
        evidence=evidence,
        assumptions=[entry.name for entry in unsupported],
        payload={
            "assumption_ledger": [entry.to_dict() for entry in ledger.entries],
            "supported_count": ledger.supported_count,
            "contradicted_count": ledger.contradicted_count,
            "unresolved_count": ledger.unresolved_count,
        },
        verdict_suggestion=verdict,
        metadata={"decision_stage": stage},
    )


def _make_support_witness(
    ledger: AssumptionLedger,
    tool_trace: list[dict[str, Any]],
    *,
    confidence: float,
) -> Witness:
    supported = [entry.name for entry in ledger.entries if entry.status is AssumptionStatus.SUPPORTED]
    evidence: list[str] = []
    for record in tool_trace[:4]:
        summary = record.get("summary") or record.get("description") or record.get("tool_name") or record.get("tool")
        if summary:
            evidence.append(str(summary))

    if not evidence:
        evidence.append("Tool-backed adjudication did not expose a surviving alternative explanation.")

    return Witness(
        witness_type=WitnessKind.SUPPORT,
        description="The claim survives the four-stage verifier checks and is jointly supported by the ledger and tool evidence.",
        evidence=evidence,
        assumptions=supported,
        payload={
            "assumption_ledger": [entry.to_dict() for entry in ledger.entries],
            "tool_trace_size": len(tool_trace),
            "support_confidence": confidence,
        },
        verdict_suggestion=VerdictLabel.VALID,
        metadata={"decision_stage": "support_adjudication"},
    )


@dataclass(slots=True)
class VerifierDecision:
    """Serializable output of the verifier's final decision stage."""

    label: VerdictLabel | str
    confidence: float
    assumption_ledger: AssumptionLedger
    probabilities: dict[str, float] = field(default_factory=dict)
    witness: Witness | None = None
    support_witness: Witness | None = None
    countermodel_witness: Witness | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _coerce_label(self.label)
        self.confidence = float(self.confidence)
        self.probabilities = _normalize_probabilities(self.probabilities)
        self.tool_trace = _normalize_tool_trace(self.tool_trace)
        self.reasoning_summary = str(self.reasoning_summary).strip()
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "assumption_ledger": [entry.to_dict() for entry in self.assumption_ledger.entries],
            "witness": self.witness.to_dict() if self.witness is not None else None,
            "support_witness": (
                self.support_witness.to_dict() if self.support_witness is not None else None
            ),
            "countermodel_witness": (
                self.countermodel_witness.to_dict()
                if self.countermodel_witness is not None
                else None
            ),
            "tool_trace": list(self.tool_trace),
            "reasoning_summary": self.reasoning_summary,
            "metadata": dict(self.metadata),
        }


def decide_verdict(
    parsed_claim: ParsedClaim,
    ledger: AssumptionLedger,
    countermodel_result: CountermodelSearchResult,
    *,
    tool_trace: list[Any] | tuple[Any, ...] | None = None,
) -> VerifierDecision:
    """Apply the fixed four-stage decision order to one parsed claim."""

    normalized_tool_trace = _normalize_tool_trace(tool_trace)

    # Stage 1: strong countermodel -> invalid
    if countermodel_result.found_countermodel and countermodel_result.verdict_suggestion == "invalid":
        witness = _make_countermodel_witness(parsed_claim, countermodel_result)
        confidence = max(0.84, min(countermodel_result.observational_match_score, 0.98))
        probabilities = _probabilities_from_scores(
            valid_score=0.03,
            invalid_score=0.72 + (0.35 * confidence),
            unidentifiable_score=0.18 + (0.12 * (1.0 - confidence)),
        )
        return VerifierDecision(
            label=VerdictLabel.INVALID,
            confidence=confidence,
            assumption_ledger=ledger,
            probabilities=probabilities,
            witness=witness,
            countermodel_witness=witness,
            tool_trace=normalized_tool_trace,
            reasoning_summary=(
                "Stage 1: a strong countermodel remains observationally compatible and directly conflicts with the claim."
            ),
            metadata={
                "decision_stage": 1,
                "support_stage_entered": False,
            },
        )

    # Stage 2: multiple compatible models with disagreement -> unidentifiable
    if countermodel_result.found_countermodel and countermodel_result.query_disagreement:
        witness = _make_countermodel_witness(parsed_claim, countermodel_result)
        confidence = max(0.68, min(countermodel_result.observational_match_score - 0.08, 0.9))
        probabilities = _probabilities_from_scores(
            valid_score=0.05,
            invalid_score=0.16 + max(0.0, countermodel_result.observational_match_score - 0.5),
            unidentifiable_score=0.66 + (0.34 * confidence),
        )
        return VerifierDecision(
            label=VerdictLabel.UNIDENTIFIABLE,
            confidence=confidence,
            assumption_ledger=ledger,
            probabilities=probabilities,
            witness=witness,
            countermodel_witness=witness,
            tool_trace=normalized_tool_trace,
            reasoning_summary=(
                "Stage 2: the verifier found observationally compatible alternatives that disagree on the target query."
            ),
            metadata={
                "decision_stage": 2,
                "support_stage_entered": False,
            },
        )

    # Support adjudication is only reached when no countermodel survived.
    adjudicated_ledger = _ledger_entries_from_tool_support(ledger, normalized_tool_trace)
    unsupported = _unsupported_core_assumptions(adjudicated_ledger)
    tools_support_claim = _tool_supports_claim(normalized_tool_trace)
    supported_identifying = _supported_identifying_assumptions(adjudicated_ledger)
    explicitly_contradicted = _explicitly_contradicted_assumptions(adjudicated_ledger)

    # Stage 3: unsupported core assumptions -> unidentifiable
    if unsupported or not tools_support_claim or (
        _needs_explicit_identifying_support(parsed_claim, adjudicated_ledger) and not supported_identifying
    ):
        witness = _make_assumption_witness(adjudicated_ledger)
        confidence = 0.62 if unsupported else 0.58
        unsupported_ratio = _score_ratio(len(unsupported), len(adjudicated_ledger.entries))
        supported_ratio = _score_ratio(len(supported_identifying), len(adjudicated_ledger.entries))
        return VerifierDecision(
            label=VerdictLabel.UNIDENTIFIABLE,
            confidence=confidence,
            assumption_ledger=adjudicated_ledger,
            probabilities=_probabilities_from_scores(
                valid_score=0.08 + (0.2 * supported_ratio),
                invalid_score=0.08 + (0.18 * _score_ratio(len(explicitly_contradicted), len(adjudicated_ledger.entries))),
                unidentifiable_score=0.6 + (0.25 * unsupported_ratio) + (0.15 * (0.0 if tools_support_claim else 1.0)),
            ),
            witness=witness,
            tool_trace=normalized_tool_trace,
            reasoning_summary=(
                "Stage 3: no decisive countermodel survived, but the core identification assumptions are not jointly supported by the available evidence."
            ),
            metadata={
                "decision_stage": 3,
                "support_stage_entered": True,
                "supported_identifying_assumptions": [entry.name for entry in supported_identifying],
            },
        )

    # Stage 4: no countermodel and ledger + tool evidence support -> valid
    support_confidence = max(0.72, min(_tool_support_confidence(normalized_tool_trace), 0.95))
    support_ratio = _score_ratio(len(supported_identifying), len(adjudicated_ledger.entries))
    support_witness = _make_support_witness(
        adjudicated_ledger,
        normalized_tool_trace,
        confidence=support_confidence,
    )
    return VerifierDecision(
        label=VerdictLabel.VALID,
        confidence=support_confidence,
        assumption_ledger=adjudicated_ledger,
        probabilities=_probabilities_from_scores(
            valid_score=0.62 + (0.2 * support_ratio) + (0.2 * support_confidence),
            invalid_score=0.04 + (0.14 * _score_ratio(len(explicitly_contradicted), len(adjudicated_ledger.entries))),
            unidentifiable_score=0.12 + (0.25 * (1.0 - support_ratio)),
        ),
        witness=support_witness,
        support_witness=support_witness,
        tool_trace=normalized_tool_trace,
        reasoning_summary=(
            "Stage 4: no effective countermodel remained, and the ledger plus tool-backed evidence jointly support the claim."
        ),
        metadata={
            "decision_stage": 4,
            "support_stage_entered": True,
        },
    )
