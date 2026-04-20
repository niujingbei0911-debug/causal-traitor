"""Rule-based countermodel search for adversarial causal claim verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd

from benchmark.schema import PublicCausalInstance, require_public_instance
from verifier.assumption_ledger import (
    AssumptionLedger,
    AssumptionLedgerEntry,
    AssumptionStatus,
    build_assumption_ledger,
)
from verifier.outputs import ClaimPolarity, ClaimStrength, ParsedClaim, QueryType
from verifier.claim_parser import _flatten_transcript


VERDICT_SUGGESTION_SPACE: tuple[str, ...] = ("invalid", "unidentifiable")
_COUNTERFACTUAL_RISK_STRATEGIES: tuple[str, ...] = (
    "false_uniqueness",
    "counterfactual_certainty",
    "assumption_smuggling",
)
_PROXY_HINT_TOKENS: tuple[str, ...] = (
    "proxy",
    "proxy_signal",
    "screening_trace",
    "sensor_proxy",
    "triage_note",
    "archive_indicator",
    "surrogate",
)
_SELECTION_HINT_TOKENS: tuple[str, ...] = (
    "selection",
    "screen",
    "enroll",
    "record",
    "clinic",
    "audit",
    "portal",
)
_INSTRUMENT_HINT_TOKENS: tuple[str, ...] = (
    "instrument",
    "lottery",
    "encouragement",
    "offer",
    "quota",
    "wave",
    "distance",
    "calendar",
    "assignment",
)
_MEDIATOR_HINT_TOKENS: tuple[str, ...] = (
    "mediator",
    "mechanism",
    "intermediate",
    "uptake",
    "engagement",
    "biomarker",
    "dosage",
)


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


def _normalize_verdict_suggestion(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in VERDICT_SUGGESTION_SPACE:
        raise ValueError(
            f"verdict_suggestion must be one of {VERDICT_SUGGESTION_SPACE}, got {value!r}."
        )
    return normalized


def _severity(verdict_suggestion: str | None) -> int:
    if verdict_suggestion == "invalid":
        return 2
    if verdict_suggestion == "unidentifiable":
        return 1
    return 0


def _context_text(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    parts: list[str] = []
    for key in ("claim_text", "transcript", "notes"):
        value = context.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.append(_flatten_transcript(value))
    return " ".join(parts).lower()


def _has_proxy_hint(context_text: str, context: dict[str, Any] | None) -> bool:
    if any(token in context_text for token in _PROXY_HINT_TOKENS):
        return True
    proxy_variables = context.get("proxy_variables", []) if context else []
    return any(str(proxy).strip().lower() in context_text for proxy in proxy_variables)


def _clamp_score(value: float, *, low: float = 0.0, high: float = 0.99) -> float:
    return max(low, min(high, float(value)))


def _resolve_public_instance(
    scenario: PublicCausalInstance | None,
    context: dict[str, Any] | None,
) -> PublicCausalInstance | None:
    if scenario is not None:
        return require_public_instance(scenario)
    if not context:
        return None
    for key in ("public_instance", "public_scenario", "scenario"):
        value = context.get(key)
        if value is not None:
            return require_public_instance(value)
    return None


def _resolve_countermodel_level(
    parsed_claim: ParsedClaim,
    scenario: PublicCausalInstance | None,
) -> str:
    public_instance = None if scenario is None else require_public_instance(scenario)
    if public_instance is not None:
        try:
            scenario_level = int(getattr(public_instance, "causal_level", 0) or 0)
        except (TypeError, ValueError):
            scenario_level = 0
        if scenario_level >= 3:
            return "L3"
        if scenario_level == 2:
            return "L2"
        if scenario_level == 1:
            return "L1"

    if parsed_claim.query_type is QueryType.COUNTERFACTUAL:
        return "L3"
    if parsed_claim.query_type is QueryType.INTERVENTION:
        return "L2"
    return "L1"


def _resolve_observed_data(
    *,
    scenario: PublicCausalInstance | None,
    observed_data: pd.DataFrame | None,
    context: dict[str, Any] | None,
) -> pd.DataFrame:
    public_instance = _resolve_public_instance(scenario, context)
    if public_instance is not None:
        if isinstance(public_instance.observed_data, pd.DataFrame) and not public_instance.observed_data.empty:
            return public_instance.observed_data.copy(deep=True)
        if isinstance(public_instance.data, pd.DataFrame) and not public_instance.data.empty:
            return public_instance.data.copy(deep=True)
    # Core verifier entrypoints only trust observed data carried by the typed
    # public schema. Raw DataFrame side channels would bypass the public/gold
    # partition contract and can silently reintroduce oracle leakage.
    return pd.DataFrame()


def _observed_columns(
    scenario: PublicCausalInstance | None,
    observed_data: pd.DataFrame,
) -> list[str]:
    if not observed_data.empty:
        return [str(column).strip() for column in observed_data.columns if str(column).strip()]
    public_instance = None if scenario is None else require_public_instance(scenario)
    if public_instance is None:
        return []
    return [str(variable).strip() for variable in public_instance.variables if str(variable).strip()]


def _normalize_public_hint(
    value: Any,
    *,
    observed_columns: list[str],
) -> str | None:
    normalized = str(value).strip()
    if not normalized:
        return None
    if not observed_columns:
        return None
    if normalized in observed_columns:
        return normalized
    return None


def _resolve_named_variables(
    context: dict[str, Any] | None,
    key: str,
) -> list[str]:
    values: list[str] = []
    for source in ((context or {}).get(key, []),):
        if isinstance(source, (list, tuple, set)):
            values.extend(str(item).strip() for item in source if str(item).strip())
    return _normalize_unique_strings(values)


def _find_observed_column(
    context_text: str,
    observed_columns: list[str],
    *,
    tokens: tuple[str, ...] = (),
) -> str | None:
    def has_role_token(column_name: str) -> bool:
        if not tokens:
            return True
        parts = [part for part in re.split(r"[_\W]+", column_name.lower()) if part]
        return any(token in parts for token in tokens)

    for column in observed_columns:
        if re.search(rf"\b{re.escape(column)}\b", context_text, flags=re.IGNORECASE):
            if has_role_token(column):
                return column
    token_matches = [
        column
        for column in observed_columns
        if has_role_token(column)
    ]
    if len(token_matches) == 1:
        return token_matches[0]
    return None


def _resolve_role_hints(
    scenario: PublicCausalInstance | None,
    context: dict[str, Any] | None,
    *,
    observed_data: pd.DataFrame,
    context_text: str,
) -> dict[str, str]:
    observed_columns = _observed_columns(scenario, observed_data)
    hints: dict[str, str] = {}

    explicit_instrument = None if context is None else context.get("instrument")
    instrument = _normalize_public_hint(explicit_instrument, observed_columns=observed_columns)
    if instrument is None:
        explicit_instrument_variables = _resolve_named_variables(context, "instrument_variables")
        if explicit_instrument_variables:
            instrument = _normalize_public_hint(
                explicit_instrument_variables[0],
                observed_columns=observed_columns,
            )
    if instrument is None:
        instrument = _extract_named_variable(
            context_text,
            patterns=(
                r"\busing\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+as an instrument\b",
                r"\bwith\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+as an instrument\b",
            ),
        )
        instrument = _normalize_public_hint(instrument, observed_columns=observed_columns)
    if instrument is None:
        instrument = _find_observed_column(context_text, observed_columns, tokens=_INSTRUMENT_HINT_TOKENS)
    if instrument is not None:
        hints["instrument"] = instrument

    explicit_mediator = None if context is None else context.get("mediator")
    mediator = _normalize_public_hint(explicit_mediator, observed_columns=observed_columns)
    if mediator is None:
        explicit_mediator_variables = _resolve_named_variables(context, "mediator_variables")
        if explicit_mediator_variables:
            mediator = _normalize_public_hint(
                explicit_mediator_variables[0],
                observed_columns=observed_columns,
            )
    if mediator is None:
        mediator = _extract_named_variable(
            context_text,
            patterns=(
                r"\bmediator\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
                r"\bthrough\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\b",
            ),
        )
        mediator = _normalize_public_hint(mediator, observed_columns=observed_columns)
    if mediator is None:
        mediator = _find_observed_column(context_text, observed_columns, tokens=_MEDIATOR_HINT_TOKENS)
    if mediator is not None:
        hints["mediator"] = mediator

    explicit_proxy_variables = _resolve_named_variables(context, "proxy_variables")
    proxy = None
    if explicit_proxy_variables:
        proxy = _normalize_public_hint(explicit_proxy_variables[0], observed_columns=observed_columns)
    if proxy is None:
        proxy = _find_observed_column(context_text, observed_columns, tokens=_PROXY_HINT_TOKENS)
    if proxy is not None:
        hints["proxy"] = proxy

    explicit_selection_variables = _resolve_named_variables(context, "selection_variables")
    selection = None
    if explicit_selection_variables:
        selection = _normalize_public_hint(explicit_selection_variables[0], observed_columns=observed_columns)
    if selection is None:
        selection = _find_observed_column(context_text, observed_columns, tokens=_SELECTION_HINT_TOKENS)
    if selection is not None:
        hints["selection"] = selection

    return hints


def _numeric_pair(data: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if data.empty or left not in data or right not in data:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(data[left], errors="coerce"),
            "right": pd.to_numeric(data[right], errors="coerce"),
        }
    ).dropna()
    return frame


def _is_binary_like(series: pd.Series) -> bool:
    observed = {float(item) for item in series.dropna().unique()}
    return len(observed) <= 2 and observed <= {0.0, 1.0}


def _association_strength(data: pd.DataFrame, left: str, right: str) -> float:
    frame = _numeric_pair(data, left, right)
    if len(frame) < 3:
        return 0.0
    if _is_binary_like(frame["left"]):
        treated = frame.loc[frame["left"] >= 0.5, "right"]
        control = frame.loc[frame["left"] < 0.5, "right"]
        if treated.empty or control.empty:
            return 0.0
        scale = float(frame["right"].std()) or 1.0
        return _clamp_score(abs(float(treated.mean() - control.mean())) / scale, high=1.0)
    correlation = frame["left"].corr(frame["right"])
    if pd.isna(correlation):
        return 0.0
    return _clamp_score(abs(float(correlation)), high=1.0)


def _group_labels(series: pd.Series) -> pd.Series:
    cleaned = series.dropna()
    if cleaned.empty:
        return series.astype(str)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().all() and cleaned.nunique() > 6:
        try:
            bins = pd.qcut(numeric, q=min(4, cleaned.nunique()), duplicates="drop")
            rebased = pd.Series(index=cleaned.index, data=bins.astype(str))
            return series.index.to_series().map(rebased).fillna("nan")
        except ValueError:
            pass
    return series.astype(str).fillna("nan")


def _stratified_effect_instability(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    stratifier: str | None,
) -> float:
    if stratifier is None or data.empty or stratifier not in data or treatment not in data or outcome not in data:
        return 0.0
    frame = pd.DataFrame(
        {
            "treatment": pd.to_numeric(data[treatment], errors="coerce"),
            "outcome": pd.to_numeric(data[outcome], errors="coerce"),
            "stratifier": data[stratifier],
        }
    ).dropna()
    if len(frame) < 6 or frame["stratifier"].nunique() < 2:
        return 0.0
    labels = _group_labels(frame["stratifier"])
    gaps: list[float] = []
    for _, group in frame.assign(group=labels).groupby("group"):
        if group["treatment"].nunique() < 2:
            continue
        treated = group.loc[group["treatment"] >= 0.5, "outcome"]
        control = group.loc[group["treatment"] < 0.5, "outcome"]
        if treated.empty or control.empty:
            continue
        gaps.append(float(treated.mean() - control.mean()))
    if len(gaps) < 2:
        return 0.0
    scale = float(frame["outcome"].std()) or 1.0
    return _clamp_score(pd.Series(gaps).std() / scale, high=1.0)


def _within_treatment_outcome_gap(
    data: pd.DataFrame,
    candidate: str | None,
    treatment: str,
    outcome: str,
) -> float:
    if candidate is None or data.empty or candidate not in data or treatment not in data or outcome not in data:
        return 0.0
    frame = pd.DataFrame(
        {
            "candidate": pd.to_numeric(data[candidate], errors="coerce"),
            "treatment": pd.to_numeric(data[treatment], errors="coerce"),
            "outcome": pd.to_numeric(data[outcome], errors="coerce"),
        }
    ).dropna()
    if len(frame) < 6:
        return 0.0
    gaps: list[float] = []
    for _, group in frame.groupby("treatment"):
        if group["candidate"].nunique() < 2:
            continue
        if _is_binary_like(group["candidate"]):
            high = group.loc[group["candidate"] >= 0.5, "outcome"]
            low = group.loc[group["candidate"] < 0.5, "outcome"]
        else:
            median = float(group["candidate"].median())
            high = group.loc[group["candidate"] >= median, "outcome"]
            low = group.loc[group["candidate"] < median, "outcome"]
        if high.empty or low.empty:
            continue
        gaps.append(float(high.mean() - low.mean()))
    if not gaps:
        return 0.0
    scale = float(frame["outcome"].std()) or 1.0
    return _clamp_score(max(abs(gap) for gap in gaps) / scale, high=1.0)


def _proxy_alignment(
    data: pd.DataFrame,
    proxy: str | None,
    treatment: str,
    outcome: str,
) -> float:
    if proxy is None:
        return 0.0
    return 0.5 * _association_strength(data, proxy, treatment) + 0.5 * _association_strength(
        data,
        proxy,
        outcome,
    )


def _selection_dependence(
    data: pd.DataFrame,
    selection: str | None,
    treatment: str,
    outcome: str,
) -> float:
    if selection is None:
        return 0.0
    return 0.5 * _association_strength(data, selection, treatment) + 0.5 * _association_strength(
        data,
        selection,
        outcome,
    )


def _extract_named_variable(
    context_text: str,
    *,
    patterns: tuple[str, ...],
) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, context_text, flags=re.IGNORECASE)
        if match:
            candidate = str(match.group("name")).strip()
            if candidate:
                return candidate
    return None


def _claimed_adjuster(
    context_text: str,
    context: dict[str, Any] | None,
) -> str | None:
    if context and context.get("adjustment_set"):
        values = _normalize_unique_strings(list(context.get("adjustment_set", [])))
        if values:
            return values[0]
    extracted = _extract_named_variable(
        context_text,
        patterns=(
            r"\b(?:controlling|conditioning)\s+for\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)",
            r"\badjust(?:ing|ment)?\s+(?:on|for)\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)",
            r"\bonce\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+is\s+included\b",
            r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+is\s+the\s+only\s+adjustment\s+needed\b",
            r"\busing\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+as\s+the\s+adjustment\s+set\b",
        ),
    )
    return extracted


def _measurement_view(
    scenario: PublicCausalInstance | None,
    variable: str | None,
) -> str | None:
    if scenario is None or variable is None:
        return None
    metadata = dict(getattr(scenario, "metadata", {}) or {})
    raw_semantics = metadata.get("measurement_semantics")
    if not isinstance(raw_semantics, dict):
        return None
    semantics = raw_semantics.get(str(variable).strip(), {})
    if not isinstance(semantics, dict):
        return None
    measurement_view = semantics.get("measurement_view")
    if measurement_view is None:
        return None
    normalized = str(measurement_view).strip()
    return normalized or None


def _observational_evidence(
    *,
    used_observed_data: bool,
    **details: Any,
) -> dict[str, Any]:
    return {
        "used_observed_data": used_observed_data,
        **details,
    }


@dataclass(slots=True)
class CountermodelCandidate:
    """One candidate countermodel surfaced by the verifier."""

    countermodel_type: str
    causal_level: str
    observational_match_score: float
    query_disagreement: bool
    countermodel_explanation: str
    verdict_suggestion: str | None
    triggered_assumptions: list[str] = field(default_factory=list)
    observational_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.countermodel_type = str(self.countermodel_type).strip()
        self.causal_level = str(self.causal_level).strip()
        self.observational_match_score = float(self.observational_match_score)
        self.query_disagreement = bool(self.query_disagreement)
        self.countermodel_explanation = str(self.countermodel_explanation).strip()
        self.verdict_suggestion = _normalize_verdict_suggestion(self.verdict_suggestion)
        self.triggered_assumptions = _normalize_unique_strings(self.triggered_assumptions)
        self.observational_evidence = dict(self.observational_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "countermodel_type": self.countermodel_type,
            "causal_level": self.causal_level,
            "observational_match_score": self.observational_match_score,
            "query_disagreement": self.query_disagreement,
            "countermodel_explanation": self.countermodel_explanation,
            "verdict_suggestion": self.verdict_suggestion,
            "triggered_assumptions": list(self.triggered_assumptions),
            "observational_evidence": dict(self.observational_evidence),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.update(
            {
                "type": self.countermodel_type,
                "match_score": self.observational_match_score,
                "explanation": self.countermodel_explanation,
            }
        )
        return payload


@dataclass(slots=True)
class CountermodelSearchResult:
    """Top-level countermodel-search output expected by the verifier pipeline."""

    found_countermodel: bool
    countermodel_type: str | None = None
    observational_match_score: float = 0.0
    query_disagreement: bool = False
    countermodel_explanation: str = ""
    verdict_suggestion: str | None = None
    candidates: list[CountermodelCandidate] = field(default_factory=list)
    used_observed_data: bool = False

    def __post_init__(self) -> None:
        self.found_countermodel = bool(self.found_countermodel)
        self.countermodel_type = None if self.countermodel_type is None else str(self.countermodel_type)
        self.observational_match_score = float(self.observational_match_score)
        self.query_disagreement = bool(self.query_disagreement)
        self.countermodel_explanation = str(self.countermodel_explanation)
        self.verdict_suggestion = _normalize_verdict_suggestion(self.verdict_suggestion)
        self.candidates = [
            candidate
            if isinstance(candidate, CountermodelCandidate)
            else CountermodelCandidate(**candidate)
            for candidate in self.candidates
        ]
        self.used_observed_data = bool(
            self.used_observed_data
            or any(candidate.observational_evidence.get("used_observed_data") for candidate in self.candidates)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "found_countermodel": self.found_countermodel,
            "countermodel_type": self.countermodel_type,
            "observational_match_score": self.observational_match_score,
            "query_disagreement": self.query_disagreement,
            "countermodel_explanation": self.countermodel_explanation,
            "verdict_suggestion": self.verdict_suggestion,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "used_observed_data": self.used_observed_data,
        }

    def selected_candidate(self) -> CountermodelCandidate | None:
        if not self.candidates:
            return None
        if self.countermodel_type is not None:
            matching_candidates = [
                candidate
                for candidate in self.candidates
                if candidate.countermodel_type == self.countermodel_type
            ]
            if matching_candidates:
                return max(
                    matching_candidates,
                    key=lambda candidate: (
                        _severity(candidate.verdict_suggestion),
                        candidate.observational_match_score,
                        int(candidate.query_disagreement),
                    ),
                )
        return _choose_best_candidate(self.candidates)

    def to_witness_payload(self) -> dict[str, Any]:
        selected_candidate = self.selected_candidate()
        triggered_assumptions = (
            list(selected_candidate.triggered_assumptions)
            if selected_candidate is not None
            else []
        )
        payload = self.to_dict()
        payload.update(
            {
                "type": self.countermodel_type,
                "match_score": self.observational_match_score,
                "query_disagreement": self.query_disagreement,
                "triggered_assumptions": triggered_assumptions,
                "explanation": self.countermodel_explanation,
                "candidate_count": len(self.candidates),
                "selected_countermodel": (
                    selected_candidate.to_evidence_dict()
                    if selected_candidate is not None
                    else None
                ),
                "candidate_pool": [
                    candidate.to_evidence_dict()
                    for candidate in self.candidates
                ],
            }
        )
        return payload


def _status_lookup(ledger: AssumptionLedger) -> dict[str, AssumptionLedgerEntry]:
    return ledger.by_name()


def _status_of(
    statuses: dict[str, AssumptionLedgerEntry],
    name: str,
) -> AssumptionStatus | None:
    entry = statuses.get(name)
    return None if entry is None else entry.status


def _verdict_from_status(status: AssumptionStatus | None) -> str:
    if status is AssumptionStatus.CONTRADICTED:
        return "invalid"
    return "unidentifiable"


def _choose_best_candidate(candidates: list[CountermodelCandidate]) -> CountermodelCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            _severity(candidate.verdict_suggestion),
            candidate.observational_match_score,
            int(candidate.query_disagreement),
        ),
    )


def _l1_candidates(
    parsed_claim: ParsedClaim,
    statuses: dict[str, AssumptionLedgerEntry],
    *,
    observed_data: pd.DataFrame,
    role_bindings: dict[str, str],
    context: dict[str, Any] | None = None,
    context_text: str = "",
) -> list[CountermodelCandidate]:
    candidates: list[CountermodelCandidate] = []
    confounding_status = _status_of(statuses, "no unobserved confounding")
    selection_status = _status_of(statuses, "no selection bias")
    observed_assoc = _association_strength(observed_data, parsed_claim.treatment, parsed_claim.outcome)
    selection_variable = role_bindings.get("selection")
    selection_fit = _selection_dependence(
        observed_data,
        selection_variable,
        parsed_claim.treatment,
        parsed_claim.outcome,
    )
    used_observed_data = not observed_data.empty

    if confounding_status in {AssumptionStatus.UNRESOLVED, AssumptionStatus.CONTRADICTED}:
        confounding_verdict = (
            "invalid"
            if (
                confounding_status is AssumptionStatus.CONTRADICTED
                or parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
            )
            else "unidentifiable"
        )
        match_score = _clamp_score(
            0.78
            + 0.12 * observed_assoc
            + (0.06 if confounding_verdict == "invalid" else 0.03),
            low=0.7,
            high=0.98,
        )
        if used_observed_data:
            explanation = (
                f"The current observed distribution keeps |assoc({parsed_claim.treatment}, {parsed_claim.outcome})|={observed_assoc:.2f}, "
                "so a latent-variable countermodel can still reproduce the measured pattern while changing the causal interpretation."
            )
        else:
            explanation = (
                f"A latent-variable model can preserve the observed association between "
                f"{parsed_claim.treatment} and {parsed_claim.outcome} while changing the causal interpretation."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="latent_confounder_injection",
                causal_level="L1",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=confounding_verdict,
                triggered_assumptions=["no unobserved confounding"],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                ),
            )
        )

    directional_overclaim = bool(
        re.search(
            r"\b(cause|causes|causal|effect|affect|affects|direction|read causally|causal conclusion)\b",
            context_text,
            flags=re.IGNORECASE,
        )
    )
    if (
        parsed_claim.claim_polarity is ClaimPolarity.POSITIVE
        and parsed_claim.treatment
        and parsed_claim.outcome
        and directional_overclaim
    ):
        match_score = _clamp_score(0.7 + 0.16 * observed_assoc, low=0.62, high=0.92)
        if used_observed_data:
            explanation = (
                f"The same observed association strength of {observed_assoc:.2f} can be fit by a reverse-direction model, "
                f"so the current observations do not uniquely pin down {parsed_claim.treatment}->{parsed_claim.outcome}."
            )
        else:
            explanation = (
                f"A reverse-direction model can explain the same observational link while flipping the direction "
                f"from {parsed_claim.treatment}->{parsed_claim.outcome} to an observationally compatible alternative."
            )
        reverse_verdict = (
            "invalid"
            if (
                parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
                or (confounding_status is None and selection_status is None and selection_variable is None)
            )
            else "unidentifiable"
        )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="direction_flip_candidate",
                causal_level="L1",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=reverse_verdict,
                triggered_assumptions=[],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                ),
            )
        )

    if selection_status in {AssumptionStatus.UNRESOLVED, AssumptionStatus.CONTRADICTED} or selection_variable is not None:
        selection_verdict = (
            "invalid"
            if (
                selection_status is AssumptionStatus.CONTRADICTED
                or (
                    selection_variable is not None
                    and parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
                )
            )
            else "unidentifiable"
        )
        match_score = _clamp_score(
            0.74
            + 0.1 * observed_assoc
            + 0.12 * selection_fit
            + (0.06 if selection_verdict == "invalid" else 0.03),
            low=0.68,
            high=0.99,
        )
        if used_observed_data and selection_variable:
            explanation = (
                f"The current data show selection dependence={selection_fit:.2f} around {selection_variable}, "
                "so a collider or sample-selection countermodel remains consistent with the observed association."
            )
        else:
            explanation = (
                "A collider or sample-selection mechanism can reproduce the observed pattern without preserving the claimed causal reading."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="selection_mechanism_candidate",
                causal_level="L1",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=selection_verdict,
                triggered_assumptions=["no selection bias"],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                    selection_dependence=round(selection_fit, 3),
                    selection_variable=selection_variable,
                ),
            )
        )

    return candidates


def _l2_candidates(
    parsed_claim: ParsedClaim,
    statuses: dict[str, AssumptionLedgerEntry],
    *,
    scenario: PublicCausalInstance | None,
    observed_data: pd.DataFrame,
    role_bindings: dict[str, str],
    context: dict[str, Any] | None = None,
    context_text: str = "",
) -> list[CountermodelCandidate]:
    candidates: list[CountermodelCandidate] = []
    confounding_status = _status_of(statuses, "no unobserved confounding")
    adjustment_status = _status_of(statuses, "valid adjustment set")
    relevance_status = _status_of(statuses, "instrument relevance")
    exclusion_status = _status_of(statuses, "exclusion restriction")
    independence_status = _status_of(statuses, "instrument independence")
    claimed_adjuster = _claimed_adjuster(context_text, context)
    claimed_adjuster_view = _measurement_view(scenario, claimed_adjuster)
    observed_assoc = _association_strength(observed_data, parsed_claim.treatment, parsed_claim.outcome)
    adjustment_instability = _stratified_effect_instability(
        observed_data,
        parsed_claim.treatment,
        parsed_claim.outcome,
        claimed_adjuster,
    )
    claimed_adjustment_report = None
    if claimed_adjuster and not observed_data.empty:
        try:
            from causal_tools.l2_intervention import backdoor_adjustment_check

            claimed_adjustment_report = backdoor_adjustment_check(
                observed_data,
                parsed_claim.treatment,
                parsed_claim.outcome,
                [claimed_adjuster],
                graph=None,
            )
        except Exception:
            claimed_adjustment_report = None
    adjustment_support_basis = (
        ""
        if claimed_adjustment_report is None
        else str(claimed_adjustment_report.get("adjustment_support_basis", "")).strip().lower()
    )
    claimed_adjuster_supported = (
        None
        if claimed_adjustment_report is None
        or adjustment_support_basis != "graph_validation"
        else bool(claimed_adjustment_report.get("supports_adjustment_set"))
    )
    instrument = role_bindings.get("instrument") or _extract_named_variable(
        context_text,
        patterns=(
            r"\busing\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+as an instrument",
            r"\bwith\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+as an instrument",
        ),
    )
    iv_covariates = [
        column
        for column in observed_data.columns
        if column not in {parsed_claim.treatment, parsed_claim.outcome, instrument}
    ][:1]
    iv_diagnostics = None
    if instrument and not observed_data.empty:
        try:
            from causal_tools.l2_intervention import iv_estimation

            iv_diagnostics = iv_estimation(
                observed_data,
                instrument,
                parsed_claim.treatment,
                parsed_claim.outcome,
                iv_covariates,
            )
        except Exception:
            iv_diagnostics = None
    first_stage_strength = _association_strength(observed_data, instrument or "", parsed_claim.treatment)
    within_treatment_gap = _within_treatment_outcome_gap(
        observed_data,
        instrument,
        parsed_claim.treatment,
        parsed_claim.outcome,
    )
    proxy = role_bindings.get("proxy")
    proxy_fit = _proxy_alignment(observed_data, proxy, parsed_claim.treatment, parsed_claim.outcome)
    used_observed_data = not observed_data.empty
    has_explicit_adjustment_overclaim = (
        parsed_claim.rhetorical_strategy == "adjustment_sufficiency_assertion"
        and claimed_adjuster_supported is False
    ) or any(
        phrase in context_text
        for phrase in (
            "only adjustment needed",
            "should be interpreted causally",
            "should be interpreted as identified",
        )
    ) or bool(re.search(r"\bonce [A-Za-z][A-Za-z0-9_]* is included\b", context_text, re.IGNORECASE))
    if (
        parsed_claim.rhetorical_strategy == "adjustment_sufficiency_assertion"
        and claimed_adjuster is not None
        and claimed_adjuster_view is not None
        and claimed_adjuster_view != "adjustment_covariate"
    ):
        has_explicit_adjustment_overclaim = True
    has_population_generalization_overclaim = any(
        phrase in context_text
        for phrase in (
            "subgroup evidence",
            "within ",
            "whole population",
            "for everyone",
            "across the whole population",
            "treated cases marked by",
        )
    )

    if has_population_generalization_overclaim:
        match_score = _clamp_score(
            0.8 + 0.08 * observed_assoc + 0.08 * adjustment_instability,
            low=0.74,
            high=0.97,
        )
        subgroup = claimed_adjuster or role_bindings.get("observed_context") or role_bindings.get("observed_adjuster")
        if used_observed_data and subgroup:
            explanation = (
                f"The observed subgroup pattern indexed by {subgroup} can remain strong while failing to justify a population-wide identification claim for "
                f"{parsed_claim.treatment} on {parsed_claim.outcome}."
            )
        else:
            explanation = (
                f"A subgroup-specific pattern can fit the observations without justifying a population-wide identification claim for "
                f"{parsed_claim.treatment} on {parsed_claim.outcome}."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="subgroup_generalization_candidate",
                causal_level="L2",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion="invalid",
                triggered_assumptions=["positivity"],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                    adjustment_instability=round(adjustment_instability, 3),
                    subgroup=subgroup,
                ),
            )
        )

    if (
        adjustment_status is AssumptionStatus.CONTRADICTED
        or confounding_status is AssumptionStatus.CONTRADICTED
        or has_explicit_adjustment_overclaim
    ):
        strongest = (
            AssumptionStatus.CONTRADICTED
            if adjustment_status is AssumptionStatus.CONTRADICTED
            or confounding_status is AssumptionStatus.CONTRADICTED
            or has_explicit_adjustment_overclaim
            else AssumptionStatus.UNRESOLVED
        )
        match_score = _clamp_score(
            0.72
            + 0.1 * observed_assoc
            + 0.08 * adjustment_instability
            + (0.06 if strongest is AssumptionStatus.CONTRADICTED else 0.03),
            low=0.68,
            high=0.97,
        )
        if used_observed_data:
            explanation = (
                f"The current observed distribution keeps |assoc({parsed_claim.treatment}, {parsed_claim.outcome})|={observed_assoc:.2f}"
                f", and the within-stratum effect instability after adjusting for {claimed_adjuster or 'the claimed covariate'} is {adjustment_instability:.2f}; "
                "a hidden-confounder interventional countermodel therefore remains observationally compatible."
            )
            if claimed_adjuster_view is not None and claimed_adjuster_view != "adjustment_covariate":
                explanation += (
                    f" Public measurement semantics classify {claimed_adjuster} as {claimed_adjuster_view}, "
                    "which is not a verifier-visible adjustment covariate."
                )
        else:
            explanation = (
                f"An interventional model with hidden confounding can remain observationally plausible while changing the estimated effect of "
                f"{parsed_claim.treatment} on {parsed_claim.outcome}."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="hidden_confounder_compatible_model",
                causal_level="L2",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=_verdict_from_status(strongest),
                triggered_assumptions=[
                    name
                    for name, status in (
                        ("valid adjustment set", adjustment_status),
                        ("no unobserved confounding", confounding_status),
                    )
                    if status is not None
                ],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                    adjustment_instability=round(adjustment_instability, 3),
                    claimed_adjuster=claimed_adjuster,
                    claimed_adjuster_measurement_view=claimed_adjuster_view,
                ),
            )
        )

    has_iv_story = any(
        status is not None
        for status in (relevance_status, exclusion_status, independence_status)
    ) or parsed_claim.rhetorical_strategy == "instrumental_variable_appeal"
    has_explicit_iv_overclaim = any(
        phrase in context_text
        for phrase in (
            "only through",
            "except through",
            "only shifts",
            "variation induced by",
            "enough to recover",
            "instrumental-variable estimate",
            "trustworthy",
            "fully valid",
        )
    )
    iv_diagnostics_support = bool(
        iv_diagnostics is not None
        and iv_diagnostics.get("is_strong_instrument") is True
        and iv_diagnostics.get("supports_exclusion_restriction") is True
        and iv_diagnostics.get("supports_instrument_independence") is True
    )
    iv_semantics_resolved = (
        relevance_status is AssumptionStatus.SUPPORTED
        and exclusion_status is AssumptionStatus.SUPPORTED
        and independence_status is AssumptionStatus.SUPPORTED
    ) or iv_diagnostics_support
    if has_iv_story and has_explicit_iv_overclaim and not iv_semantics_resolved:
        iv_statuses = [status for status in (exclusion_status, independence_status) if status is not None]
        strongest = (
            AssumptionStatus.CONTRADICTED
            if (
                any(status is AssumptionStatus.CONTRADICTED for status in iv_statuses)
                or parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
            )
            else AssumptionStatus.UNRESOLVED
        )
        match_score = _clamp_score(
            0.72
            + 0.11 * first_stage_strength
            + 0.11 * within_treatment_gap
            + (0.06 if strongest is AssumptionStatus.CONTRADICTED else 0.03),
            low=0.68,
            high=0.98,
        )
        if used_observed_data and instrument:
            explanation = (
                f"The current observed data retain a first-stage signal of {first_stage_strength:.2f} for {instrument}->{parsed_claim.treatment} "
                f"while the within-treatment outcome gap by {instrument} is {within_treatment_gap:.2f}, "
                "so an exclusion- or independence-violating IV alternative still matches the observations."
            )
        else:
            explanation = (
                f"An alternative IV model can keep the same observed first-stage pattern while violating exclusion or independence, "
                f"changing the causal answer for {parsed_claim.treatment} and {parsed_claim.outcome}."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="invalid_instrument_alternative",
                causal_level="L2",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=_verdict_from_status(strongest),
                triggered_assumptions=[
                    name
                    for name, status in (
                        ("instrument relevance", relevance_status),
                        ("exclusion restriction", exclusion_status),
                        ("instrument independence", independence_status),
                    )
                    if status is not None
                ],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    instrument=instrument,
                    first_stage_strength=round(first_stage_strength, 3),
                    within_treatment_outcome_gap=round(within_treatment_gap, 3),
                ),
            )
        )

    if _has_proxy_hint(context_text, context):
        proxy_verdict = (
            "invalid"
            if parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
            or parsed_claim.rhetorical_strategy == "false_uniqueness"
            else "unidentifiable"
        )
        match_score = _clamp_score(
            0.68 + 0.14 * proxy_fit + (0.06 if proxy_verdict == "invalid" else 0.02),
            low=0.64,
            high=0.95,
        )
        if used_observed_data and proxy:
            explanation = (
                f"The current observed data keep proxy-alignment={proxy_fit:.2f} for {proxy}, "
                f"so a proxy-based alternative can still fit the observations without uniquely fixing the effect of {parsed_claim.treatment} on {parsed_claim.outcome}."
            )
        else:
            explanation = (
                f"A proxy-based alternative explanation can preserve the observed pattern while leaving the effect of "
                f"{parsed_claim.treatment} on {parsed_claim.outcome} non-unique."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="proxy_based_alternative_explanation",
                causal_level="L2",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=proxy_verdict,
                triggered_assumptions=[],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    proxy=proxy,
                    proxy_alignment=round(proxy_fit, 3),
                ),
            )
        )

    return candidates


def _l3_candidates(
    parsed_claim: ParsedClaim,
    statuses: dict[str, AssumptionLedgerEntry],
    *,
    observed_data: pd.DataFrame,
    role_bindings: dict[str, str],
    context_text: str = "",
) -> list[CountermodelCandidate]:
    candidates: list[CountermodelCandidate] = []
    uniqueness_status = _status_of(statuses, "counterfactual model uniqueness")
    cross_world_status = _status_of(statuses, "cross-world consistency")
    mechanism_statuses = {
        name: _status_of(statuses, name)
        for name in ("stable mediation structure", "correct functional form", "monotonicity")
    }
    has_counterfactual_overclaim = (
        parsed_claim.rhetorical_strategy in _COUNTERFACTUAL_RISK_STRATEGIES
        or parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
        or "same observed history" in context_text
        or "would definitely" in context_text
        or "pinned down" in context_text
        or "exactly" in context_text
        or "fully determined" in context_text
    )
    mechanism_issue_names = [
        name
        for name, status in mechanism_statuses.items()
        if status in {AssumptionStatus.UNRESOLVED, AssumptionStatus.CONTRADICTED}
    ]
    mechanism_claimed = any(
        phrase in context_text
        for phrase in ("mediat", "mechanism", "functional form", "monotonic")
    )
    observed_assoc = _association_strength(observed_data, parsed_claim.treatment, parsed_claim.outcome)
    mediator = role_bindings.get("mediator")
    mediator_fit = _proxy_alignment(observed_data, mediator, parsed_claim.treatment, parsed_claim.outcome)
    mechanism_instability = _stratified_effect_instability(
        observed_data,
        parsed_claim.treatment,
        parsed_claim.outcome,
        mediator,
    )
    used_observed_data = not observed_data.empty

    if has_counterfactual_overclaim:
        strongest = (
            AssumptionStatus.CONTRADICTED
            if (
                uniqueness_status is AssumptionStatus.CONTRADICTED
                or parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
            )
            else AssumptionStatus.UNRESOLVED
        )
        match_score = _clamp_score(
            0.8
            + 0.08 * observed_assoc
            + 0.05 * mediator_fit
            + (0.05 if strongest is AssumptionStatus.CONTRADICTED else 0.02),
            low=0.74,
            high=0.99,
        )
        if used_observed_data:
            explanation = (
                f"The current observed data only fix the measured distribution, with |assoc({parsed_claim.treatment}, {parsed_claim.outcome})|={observed_assoc:.2f}"
                f" and mediator-fit={mediator_fit:.2f}; a same-fit structural countermodel can therefore still disagree on the unit-level query."
            )
        else:
            explanation = (
                f"A same-fit structural model can match the observational record for {parsed_claim.treatment} and {parsed_claim.outcome} "
                "while giving a different unit-level counterfactual answer or different PN/PS/ETT values."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="observationally_equivalent_countermodel",
                causal_level="L3",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=_verdict_from_status(strongest),
                triggered_assumptions=[
                    name
                    for name, status in (
                        ("counterfactual model uniqueness", uniqueness_status),
                        ("cross-world consistency", cross_world_status),
                    )
                    if status is not None
                ],
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    association_strength=round(observed_assoc, 3),
                    mediator_fit=round(mediator_fit, 3),
                    mediator=mediator,
                ),
            )
        )

    if mechanism_issue_names and (
        parsed_claim.rhetorical_strategy == "assumption_smuggling"
        or any(
            mechanism_statuses[name] is AssumptionStatus.CONTRADICTED
            for name in mechanism_issue_names
        )
    ):
        strongest_mechanism = (
            AssumptionStatus.CONTRADICTED
            if any(
                mechanism_statuses[name] is AssumptionStatus.CONTRADICTED
                for name in mechanism_issue_names
            )
            or (
                parsed_claim.rhetorical_strategy == "assumption_smuggling"
                and parsed_claim.claim_strength is ClaimStrength.ABSOLUTE
            )
            else AssumptionStatus.UNRESOLVED
        )
        match_score = _clamp_score(
            0.72
            + 0.08 * mediator_fit
            + 0.1 * mechanism_instability
            + (0.06 if strongest_mechanism is AssumptionStatus.CONTRADICTED else 0.03),
            low=0.68,
            high=0.96,
        )
        if used_observed_data and mediator:
            explanation = (
                f"The observed mediator pattern for {mediator} still fits the data, but mechanism-instability={mechanism_instability:.2f} "
                "shows that alternative functional forms can preserve the observations while flipping the counterfactual conclusion."
            )
        else:
            explanation = (
                "A counterfactual SCM with a different mechanism class can preserve the observational fit while flipping the counterfactual conclusion."
            )
        candidates.append(
            CountermodelCandidate(
                countermodel_type="functional_form_flip",
                causal_level="L3",
                observational_match_score=match_score,
                query_disagreement=True,
                countermodel_explanation=explanation,
                verdict_suggestion=_verdict_from_status(strongest_mechanism),
                triggered_assumptions=mechanism_issue_names,
                observational_evidence=_observational_evidence(
                    used_observed_data=used_observed_data,
                    mediator=mediator,
                    mediator_fit=round(mediator_fit, 3),
                    mechanism_instability=round(mechanism_instability, 3),
                ),
            )
        )

    return candidates


def search_countermodels(
    parsed_claim: ParsedClaim,
    ledger: AssumptionLedger | None = None,
    *,
    scenario: PublicCausalInstance | None = None,
    observed_data: pd.DataFrame | None = None,
    context: dict[str, Any] | None = None,
) -> CountermodelSearchResult:
    """Search for countermodel families suggested by the parsed claim and ledger."""

    resolved_ledger = ledger if ledger is not None else build_assumption_ledger(parsed_claim)
    statuses = _status_lookup(resolved_ledger)
    context_text = _context_text(context)
    resolved_data = _resolve_observed_data(
        scenario=scenario,
        observed_data=observed_data,
        context=context,
    )
    role_bindings = _resolve_role_hints(
        scenario,
        context,
        observed_data=resolved_data,
        context_text=context_text,
    )

    resolved_level = _resolve_countermodel_level(parsed_claim, scenario)

    if resolved_level == "L1":
        candidates = _l1_candidates(
            parsed_claim,
            statuses,
            observed_data=resolved_data,
            role_bindings=role_bindings,
            context=context,
            context_text=context_text,
        )
    elif resolved_level == "L2":
        candidates = _l2_candidates(
            parsed_claim,
            statuses,
            scenario=scenario,
            observed_data=resolved_data,
            role_bindings=role_bindings,
            context=context,
            context_text=context_text,
        )
    else:
        candidates = _l3_candidates(
            parsed_claim,
            statuses,
            observed_data=resolved_data,
            role_bindings=role_bindings,
            context_text=context_text,
        )

    best_candidate = _choose_best_candidate(candidates)
    if best_candidate is None:
        return CountermodelSearchResult(
            found_countermodel=False,
            candidates=[],
            used_observed_data=not resolved_data.empty,
        )

    return CountermodelSearchResult(
        found_countermodel=True,
        countermodel_type=best_candidate.countermodel_type,
        observational_match_score=best_candidate.observational_match_score,
        query_disagreement=best_candidate.query_disagreement,
        countermodel_explanation=best_candidate.countermodel_explanation,
        verdict_suggestion=best_candidate.verdict_suggestion,
        candidates=candidates,
        used_observed_data=not resolved_data.empty,
    )


def run_countermodel_search(
    parsed_claim: ParsedClaim,
    ledger: AssumptionLedger | None = None,
    *,
    scenario: PublicCausalInstance | None = None,
    observed_data: pd.DataFrame | None = None,
    context: dict[str, Any] | None = None,
) -> CountermodelSearchResult:
    """Alias kept for readability in future pipeline call sites."""

    return search_countermodels(
        parsed_claim,
        ledger=ledger,
        scenario=scenario,
        observed_data=observed_data,
        context=context,
    )
