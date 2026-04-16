"""Turn parsed causal claims into an explicit machine-readable assumption ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from verifier.outputs import ClaimStrength, ParsedClaim, QueryType


class AssumptionStatus(str, Enum):
    """Supported status values for assumption-ledger entries."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


ASSUMPTION_STATUS_SPACE: tuple[str, ...] = tuple(status.value for status in AssumptionStatus)

_ASSUMPTION_CATEGORY_MAP: dict[str, str] = {
    "consistency": "identification",
    "positivity": "identification",
    "no unobserved confounding": "confounding",
    "valid adjustment set": "confounding",
    "proxy sufficiency": "proxy",
    "no selection bias": "selection",
    "instrument relevance": "iv",
    "exclusion restriction": "iv",
    "instrument independence": "iv",
    "stable mediation structure": "counterfactual",
    "cross-world consistency": "counterfactual",
    "counterfactual model uniqueness": "counterfactual",
    "correct functional form": "functional_form",
    "monotonicity": "functional_form",
}

_COUNTERFACTUAL_STRATEGIES = {"false_uniqueness", "counterfactual_certainty"}


def _coerce_status(value: AssumptionStatus | str) -> AssumptionStatus:
    if isinstance(value, AssumptionStatus):
        return value
    return AssumptionStatus(str(value).strip().lower())


def _normalize_unique_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
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


def _category_for(name: str) -> str:
    return _ASSUMPTION_CATEGORY_MAP.get(name, "identification")


@dataclass(slots=True)
class AssumptionLedgerEntry:
    """One explicit assumption tracked by the verifier."""

    name: str
    source: str
    status: AssumptionStatus | str
    note: str
    category: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.source = str(self.source).strip()
        self.status = _coerce_status(self.status)
        self.note = str(self.note).strip()
        self.category = str(self.category).strip() or _category_for(self.name)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source": self.source,
            "category": self.category,
            "status": self.status.value,
            "note": self.note,
        }


@dataclass(slots=True)
class AssumptionLedger:
    """Machine-readable ledger emitted from a parsed claim."""

    entries: list[AssumptionLedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        deduped: list[AssumptionLedgerEntry] = []
        seen: set[str] = set()
        for entry in self.entries:
            normalized = entry if isinstance(entry, AssumptionLedgerEntry) else AssumptionLedgerEntry(**entry)
            if normalized.name in seen:
                continue
            seen.add(normalized.name)
            deduped.append(normalized)
        self.entries = deduped

    @property
    def supported_count(self) -> int:
        return sum(entry.status is AssumptionStatus.SUPPORTED for entry in self.entries)

    @property
    def contradicted_count(self) -> int:
        return sum(entry.status is AssumptionStatus.CONTRADICTED for entry in self.entries)

    @property
    def unresolved_count(self) -> int:
        return sum(entry.status is AssumptionStatus.UNRESOLVED for entry in self.entries)

    def by_name(self) -> dict[str, AssumptionLedgerEntry]:
        return {entry.name: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_ledger": [entry.to_dict() for entry in self.entries],
            "supported_count": self.supported_count,
            "contradicted_count": self.contradicted_count,
            "unresolved_count": self.unresolved_count,
        }


def _assumption_source(parsed_claim: ParsedClaim, assumption_name: str) -> str:
    if assumption_name in parsed_claim.mentioned_assumptions:
        return "claim explicit"
    if assumption_name in parsed_claim.implied_assumptions:
        return "claim implied"
    return "tool requirement"


def _add_if_missing(target: list[str], assumption_name: str) -> None:
    if assumption_name not in target:
        target.append(assumption_name)


def _collect_required_assumptions(parsed_claim: ParsedClaim) -> list[str]:
    assumptions = list(parsed_claim.mentioned_assumptions) + list(parsed_claim.implied_assumptions)

    if parsed_claim.query_type in {QueryType.INTERVENTION, QueryType.COUNTERFACTUAL}:
        _add_if_missing(assumptions, "consistency")
        _add_if_missing(assumptions, "positivity")

    if parsed_claim.rhetorical_strategy == "adjustment_sufficiency_assertion":
        _add_if_missing(assumptions, "valid adjustment set")
        _add_if_missing(assumptions, "no unobserved confounding")

    if parsed_claim.rhetorical_strategy == "instrumental_variable_appeal":
        _add_if_missing(assumptions, "instrument relevance")
        _add_if_missing(assumptions, "exclusion restriction")
        _add_if_missing(assumptions, "instrument independence")

    if parsed_claim.rhetorical_strategy == "selection_bias_obfuscation":
        _add_if_missing(assumptions, "no selection bias")

    if parsed_claim.query_type is QueryType.COUNTERFACTUAL:
        _add_if_missing(assumptions, "cross-world consistency")
        _add_if_missing(assumptions, "counterfactual model uniqueness")

    return _normalize_unique_strings(assumptions)


def _status_for_consistency(parsed_claim: ParsedClaim, source: str) -> tuple[AssumptionStatus, str]:
    if parsed_claim.query_type in {QueryType.INTERVENTION, QueryType.COUNTERFACTUAL} and parsed_claim.treatment:
        return (
            AssumptionStatus.SUPPORTED,
            "The parsed claim consistently refers to a concrete treatment variable, so the query is at least well-formed.",
        )
    return AssumptionStatus.UNRESOLVED, "The claim does not say enough to check consistency directly."


def _status_for_positivity(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim asserts overlap or support in the population, but does not verify it."
    return AssumptionStatus.UNRESOLVED, "The claim relies on overlap but does not establish it."


def _status_for_no_unobserved_confounding(
    parsed_claim: ParsedClaim,
    source: str,
) -> tuple[AssumptionStatus, str]:
    if parsed_claim.rhetorical_strategy == "confounder_denial":
        return (
            AssumptionStatus.UNRESOLVED,
            "The claim denies hidden confounding rhetorically, but the text alone does not settle that assumption.",
        )
    if parsed_claim.rhetorical_strategy == "association_overclaim":
        return (
            AssumptionStatus.UNRESOLVED,
            "A causal reading from association still depends on ruling out confounding, which the claim does not do.",
        )
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim explicitly denies confounding, but the denial is not itself evidence."
    return AssumptionStatus.UNRESOLVED, "The claim depends on a no-confounding bridge that remains implicit."


def _status_for_valid_adjustment_set(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim names an adjustment move, but does not verify that it is sufficient."
    return AssumptionStatus.UNRESOLVED, "The claim depends on a valid adjustment set without specifying it clearly."


def _status_for_proxy_sufficiency(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim points to an observed proxy, but does not establish that the proxy carries enough identifying information."
    return AssumptionStatus.UNRESOLVED, "The claim leans on a proxy bridge whose sufficiency has not yet been demonstrated."


def _status_for_no_selection_bias(parsed_claim: ParsedClaim, source: str) -> tuple[AssumptionStatus, str]:
    if parsed_claim.rhetorical_strategy == "selection_bias_obfuscation":
        if parsed_claim.claim_strength is ClaimStrength.TENTATIVE:
            return (
                AssumptionStatus.UNRESOLVED,
                "The claim leans on a selected sample, but the text only makes a tentative causal move and does not settle whether selection bias has been ruled out.",
            )
        return (
            AssumptionStatus.CONTRADICTED,
            "The claim leans on a selected or collider-shaped sample, which raises rather than resolves selection-bias risk.",
        )
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim says selection bias is absent, but does not justify that conclusion."
    return AssumptionStatus.UNRESOLVED, "The claim needs the observed sample to be selection-safe, but that is left implicit."


def _status_for_instrument_relevance(
    parsed_claim: ParsedClaim,
    source: str,
) -> tuple[AssumptionStatus, str]:
    if source in {"claim explicit", "claim implied"} or parsed_claim.rhetorical_strategy == "instrumental_variable_appeal":
        return AssumptionStatus.UNRESOLVED, "The claim presents an IV story, but does not verify that the proposed instrument is relevant."
    return AssumptionStatus.UNRESOLVED, "The claim does not establish that the proposed instrument is relevant."


def _status_for_exclusion_restriction(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim asserts exclusion, but the assertion itself is not verification."
    return AssumptionStatus.UNRESOLVED, "The IV reading needs exclusion, but the claim does not fully establish it."


def _status_for_instrument_independence(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim states an independence condition, but does not support it."
    return AssumptionStatus.UNRESOLVED, "The claim uses an IV argument without settling instrument independence."


def _status_for_counterfactual_uniqueness(parsed_claim: ParsedClaim, source: str) -> tuple[AssumptionStatus, str]:
    if parsed_claim.rhetorical_strategy in _COUNTERFACTUAL_STRATEGIES and (
        parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
    ):
        return (
            AssumptionStatus.CONTRADICTED,
            "The claim overstates unique counterfactual identification instead of justifying it.",
        )
    if parsed_claim.rhetorical_strategy == "false_uniqueness":
        return (
            AssumptionStatus.UNRESOLVED,
            "The claim gestures at uniqueness, but the text does not justify ruling out alternative counterfactual models.",
        )
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim says alternative models are ruled out, but does not justify that step."
    return AssumptionStatus.UNRESOLVED, "The claim needs counterfactual uniqueness, but that bridge is not secured."


def _status_for_cross_world_consistency(source: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, "The claim explicitly invokes same-unit reasoning, but does not validate the cross-world bridge."
    return AssumptionStatus.UNRESOLVED, "The counterfactual reading depends on cross-world consistency that is not established."


def _status_for_stable_mechanism(source: str, assumption_name: str) -> tuple[AssumptionStatus, str]:
    if source == "claim explicit":
        return AssumptionStatus.UNRESOLVED, f"The claim explicitly invokes {assumption_name}, but does not verify it."
    return AssumptionStatus.UNRESOLVED, f"The claim leans on {assumption_name} without fully justifying it."


def _derive_status(
    parsed_claim: ParsedClaim,
    assumption_name: str,
    source: str,
) -> tuple[AssumptionStatus, str]:
    if assumption_name == "consistency":
        return _status_for_consistency(parsed_claim, source)
    if assumption_name == "positivity":
        return _status_for_positivity(source)
    if assumption_name == "no unobserved confounding":
        return _status_for_no_unobserved_confounding(parsed_claim, source)
    if assumption_name == "valid adjustment set":
        return _status_for_valid_adjustment_set(source)
    if assumption_name == "proxy sufficiency":
        return _status_for_proxy_sufficiency(source)
    if assumption_name == "no selection bias":
        return _status_for_no_selection_bias(parsed_claim, source)
    if assumption_name == "instrument relevance":
        return _status_for_instrument_relevance(parsed_claim, source)
    if assumption_name == "exclusion restriction":
        return _status_for_exclusion_restriction(source)
    if assumption_name == "instrument independence":
        return _status_for_instrument_independence(source)
    if assumption_name == "counterfactual model uniqueness":
        return _status_for_counterfactual_uniqueness(parsed_claim, source)
    if assumption_name == "cross-world consistency":
        return _status_for_cross_world_consistency(source)
    if assumption_name in {"stable mediation structure", "correct functional form", "monotonicity"}:
        return _status_for_stable_mechanism(source, assumption_name)
    return AssumptionStatus.UNRESOLVED, "The claim depends on this assumption, but the text does not settle it."


def build_assumption_ledger(parsed_claim: ParsedClaim) -> AssumptionLedger:
    """Expand a parsed claim into an explicit assumption ledger."""

    entries: list[AssumptionLedgerEntry] = []
    for assumption_name in _collect_required_assumptions(parsed_claim):
        source = _assumption_source(parsed_claim, assumption_name)
        status, note = _derive_status(parsed_claim, assumption_name, source)
        entries.append(
            AssumptionLedgerEntry(
                name=assumption_name,
                source=source,
                category=_category_for(assumption_name),
                status=status,
                note=note,
            )
        )
    return AssumptionLedger(entries)


def derive_assumption_ledger(parsed_claim: ParsedClaim) -> AssumptionLedger:
    """Alias kept for readability in verifier call sites."""

    return build_assumption_ledger(parsed_claim)
