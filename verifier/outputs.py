"""Structured verifier-side outputs for claim parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
