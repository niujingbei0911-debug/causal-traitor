"""
Verdict-centric scoring utilities.

The scorer no longer treats a debate-game winner as the main target. Primary
scores are computed from verdict correctness against the frozen three-label
verdict space. Legacy DSR / jury / evolution signals are retained only as
appendix metrics.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import (
    CausalMetrics,
    MetricResult,
    _extract_round_confidence as _metrics_extract_round_confidence,
    _extract_round_confidence_value as _metrics_extract_round_confidence_value,
    _extract_round_gold_identification_status as _metrics_extract_round_gold_identification_status,
    _extract_round_gold_label as _metrics_extract_round_gold_label,
    _extract_round_identification_status as _metrics_extract_round_identification_status,
    _extract_round_predicted_label as _metrics_extract_round_predicted_label,
    _extract_round_probabilities as _metrics_extract_round_probabilities,
    _has_complete_prediction_payloads as _metrics_has_complete_prediction_payloads,
    _legacy_rounds_from_game_data as _metrics_legacy_rounds_from_game_data,
    _nested_value as _metrics_nested_value,
    _prediction_payload as _metrics_prediction_payload,
    validate_round_contracts,
)


def _normalize_verdict_label(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"valid", "invalid", "unidentifiable"}:
        return normalized
    return None


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _nested_value(mapping: Any, key: str) -> Any:
    return _metrics_nested_value(mapping, key)


def _extract_predicted_label(round_data: Dict[str, Any]) -> str | None:
    return _metrics_extract_round_predicted_label(round_data)


def _extract_gold_label(round_data: Dict[str, Any]) -> str | None:
    return _metrics_extract_round_gold_label(round_data)


def _extract_confidence(round_data: Dict[str, Any]) -> float:
    return _metrics_extract_round_confidence(round_data)


def _extract_probability_distribution(round_data: Dict[str, Any]) -> Any:
    return _metrics_extract_round_probabilities(round_data)


def _extract_countermodel_hit(round_data: Dict[str, Any]) -> bool:
    if "countermodel_found" in round_data:
        return bool(round_data.get("countermodel_found"))
    if round_data.get("countermodel_witness"):
        return True
    verifier_verdict = round_data.get("verifier_verdict")
    if _nested_value(verifier_verdict, "countermodel_witness"):
        return True
    verdict = round_data.get("verdict")
    if _nested_value(verdict, "countermodel_witness"):
        return True
    return False


def _extract_countermodel_applicable(round_data: Dict[str, Any]) -> bool:
    if "countermodel_applicable" in round_data:
        return bool(round_data.get("countermodel_applicable"))
    gold_label = _extract_gold_label(round_data)
    return gold_label in {"invalid", "unidentifiable"}


def _label_distribution(labels: List[str]) -> Dict[str, int]:
    counts = Counter(label for label in labels if label)
    return {label: counts.get(label, 0) for label in ("valid", "invalid", "unidentifiable")}


@dataclass
class RoundScore:
    """Per-round verdict evaluation."""

    round_id: int
    predicted_label: str | None
    gold_label: str | None
    verdict_correct: bool
    confidence: float
    probabilities: Any = None
    metric_results: List[MetricResult] = field(default_factory=list)
    appendix: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameScore:
    """Game-level verdict evaluation summary."""

    game_id: str
    round_scores: List[RoundScore]
    final_scores: Dict[str, float]
    winner: str
    metric_results: List[MetricResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class Scorer:
    """
    Verdict-centric scorer.

    Primary score computation is based on verdict correctness, not on the
    debate-game winner. Legacy appendix metrics are surfaced separately and do
    not contribute to the overall score.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "unsafe_acceptance_rate": 0.40,
        "wise_refusal_recall": 0.15,
        "wise_refusal_precision": 0.15,
        "over_refusal_rate": 0.10,
        "ece": 0.05,
        "brier": 0.05,
        "countermodel_coverage": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.metrics = CausalMetrics()

    def _rounds_from_game_data(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        rounds = list(game_data.get("rounds", []))
        if rounds:
            return rounds
        return _metrics_legacy_rounds_from_game_data(game_data)

    def _core_score_value(self, metric: MetricResult | None) -> float:
        if metric is None:
            return 0.0
        if not metric.higher_is_better:
            return max(0.0, min(1.0, 1.0 - metric.value))
        return max(0.0, min(1.0, metric.value))

    def score_round(self, round_data: Dict[str, Any]) -> RoundScore:
        """Score one round against the verdict label supervision target."""

        round_id = int(round_data.get("round_id", 0))
        predicted_label = _extract_predicted_label(round_data)
        gold_label = _extract_gold_label(round_data)
        confidence = _extract_confidence(round_data)
        probabilities = _extract_probability_distribution(round_data)
        verdict_correct = bool(predicted_label and gold_label and predicted_label == gold_label)
        countermodel_hit = _extract_countermodel_hit(round_data)
        countermodel_applicable = _extract_countermodel_applicable(round_data)

        metric_results: List[MetricResult] = [
            MetricResult(
                name="verdict_correctness",
                value=1.0 if verdict_correct else 0.0,
                category="verdict",
                details={"gold_label": gold_label, "predicted_label": predicted_label},
            ),
            MetricResult(
                name="countermodel_hit",
                value=1.0 if countermodel_hit else 0.0,
                category="verdict",
                details={"applicable": countermodel_applicable},
            ),
        ]

        appendix: dict[str, Any] = {}
        if "deception_succeeded" in round_data:
            deception_value = 1.0 if round_data.get("deception_succeeded") else 0.0
            metric_results.append(
                MetricResult(
                    name="DSR_round",
                    value=deception_value,
                    category="deception",
                    is_appendix=True,
                )
            )
            appendix["deception_succeeded"] = bool(round_data.get("deception_succeeded"))

        if "jury_consensus" in round_data or "jury_verdict" in round_data:
            jury_consensus = _clamp_confidence(round_data.get("jury_consensus", 0.0))
            metric_results.append(
                MetricResult(
                    name="jury_consensus_round",
                    value=jury_consensus,
                    category="jury",
                    is_appendix=True,
                )
            )
            appendix["jury_consensus"] = jury_consensus
            appendix["jury_verdict"] = round_data.get("jury_verdict")

        return RoundScore(
            round_id=round_id,
            predicted_label=predicted_label,
            gold_label=gold_label,
            verdict_correct=verdict_correct,
            confidence=confidence,
            probabilities=probabilities,
            metric_results=metric_results,
            appendix=appendix,
        )

    def score_game(self, game_data: Dict[str, Any]) -> GameScore:
        """Aggregate round-level verdict evaluation into a game-level score."""

        game_id = str(game_data.get("game_id", uuid.uuid4().hex[:12]))
        rounds_raw = self._rounds_from_game_data(game_data)
        if list(game_data.get("rounds", [])):
            validate_round_contracts(rounds_raw)
        elif _metrics_has_complete_prediction_payloads(rounds_raw):
            validate_round_contracts(rounds_raw)
        round_scores = [self.score_round(round_data) for round_data in rounds_raw]

        gold_labels = [score.gold_label for score in round_scores if score.gold_label is not None]
        predicted_labels = [score.predicted_label for score in round_scores if score.predicted_label is not None]
        paired_gold_labels: list[str] = []
        paired_predicted_labels: list[str] = []
        confidences: list[float | None] = []
        predicted_probabilities: list[dict[str, float] | None] = []
        countermodel_hits: list[bool] = []
        countermodel_applicable: list[bool] = []
        gold_identification_statuses: list[str | None] = []
        predicted_identification_statuses: list[str | None] = []

        for round_data, round_score in zip(rounds_raw, round_scores):
            if round_score.gold_label is None:
                continue
            paired_gold_labels.append(round_score.gold_label)
            paired_predicted_labels.append(round_score.predicted_label)
            confidences.append(_metrics_extract_round_confidence_value(round_data))
            predicted_probabilities.append(round_score.probabilities)
            countermodel_hits.append(_extract_countermodel_hit(round_data))
            countermodel_applicable.append(_extract_countermodel_applicable(round_data))
            payload, _ = _metrics_prediction_payload(round_data)
            if payload is not None:
                gold_identification_statuses.append(
                    _metrics_extract_round_gold_identification_status(round_data)
                )
                predicted_identification_statuses.append(
                    _metrics_extract_round_identification_status(round_data)
                )

        core_metrics = [
            CausalMetrics.verdict_accuracy(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.verdict_macro_f1(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.unsafe_acceptance_rate(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.wise_refusal_recall(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.wise_refusal_precision(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.over_commitment_rate(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.over_refusal_rate(paired_gold_labels, paired_predicted_labels),
            CausalMetrics.expected_calibration_error(
                paired_gold_labels,
                paired_predicted_labels,
                confidences,
                predicted_probabilities=predicted_probabilities,
            ),
            CausalMetrics.brier_score(
                paired_gold_labels,
                paired_predicted_labels,
                confidences,
                predicted_probabilities=predicted_probabilities,
            ),
            CausalMetrics.countermodel_coverage(countermodel_hits, countermodel_applicable),
        ]
        method_metrics: list[MetricResult] = []
        if gold_identification_statuses:
            method_metrics.append(
                CausalMetrics.identification_stage_accuracy(
                    gold_identification_statuses,
                    predicted_identification_statuses,
                )
            )

        appendix_metrics: list[MetricResult] = []
        if rounds_raw and any("deception_succeeded" in round_data for round_data in rounds_raw):
            deception_successes = sum(1 for round_data in rounds_raw if round_data.get("deception_succeeded"))
            appendix_metrics.append(CausalMetrics.deception_success_rate(deception_successes, len(rounds_raw)))

        jury_consensus_values = [
            _clamp_confidence(round_data.get("jury_consensus"))
            for round_data in rounds_raw
            if "jury_consensus" in round_data
        ]
        if jury_consensus_values:
            appendix_metrics.append(
                CausalMetrics.jury_consensus_metric(sum(jury_consensus_values) / len(jury_consensus_values))
            )
        if "jury_accuracy" in game_data:
            appendix_metrics.append(CausalMetrics.jury_accuracy_metric(game_data.get("jury_accuracy", 0.0)))

        evolution_history = list(game_data.get("evolution_history", []))
        if evolution_history:
            appendix_metrics.append(CausalMetrics.evolution_complexity_index(evolution_history))

        all_metrics = core_metrics + method_metrics + appendix_metrics
        overall = self.compute_weighted_score(all_metrics)

        final_scores = {metric.name: round(metric.value, 4) for metric in core_metrics + method_metrics}
        final_scores["overall"] = overall

        gold_distribution = _label_distribution(paired_gold_labels)
        predicted_distribution = _label_distribution(paired_predicted_labels)

        summary = {
            "primary_metric": "unsafe_acceptance_rate",
            "total_rounds": len(round_scores),
            "scored_rounds": len(paired_gold_labels),
            "core_metrics": {metric.name: round(metric.value, 4) for metric in core_metrics},
            "method_specific_metrics": {
                metric.name: round(metric.value, 4) for metric in method_metrics
            },
            "appendix_metrics": {metric.name: round(metric.value, 4) for metric in appendix_metrics},
            "overall_breakdown": {
                metric.name: round(self._core_score_value(metric), 4)
                for metric in core_metrics
            },
            "gold_label_distribution": gold_distribution,
            "predicted_label_distribution": predicted_distribution,
        }

        return GameScore(
            game_id=game_id,
            round_scores=round_scores,
            final_scores=final_scores,
            winner=str(game_data.get("winner", "n/a")),
            metric_results=all_metrics,
            summary=summary,
        )

    def compute_weighted_score(self, metric_results: List[MetricResult]) -> float:
        """Compute the overall score from core verdict metrics only."""

        core_metrics = {
            metric.name: metric
            for metric in metric_results
            if not metric.is_appendix and metric.name in self.weights
        }

        total = 0.0
        for name, weight in self.weights.items():
            metric = core_metrics.get(name)
            raw_value = self._core_score_value(metric)
            total += weight * raw_value
        return round(total, 4)

    def generate_report(self, game_score: GameScore) -> Dict[str, Any]:
        """Generate a structured verdict-centric report."""

        rounds: list[dict[str, Any]] = []
        for score in game_score.round_scores:
            rounds.append(
                {
                    "round_id": score.round_id,
                    "predicted_label": score.predicted_label,
                    "gold_label": score.gold_label,
                    "verdict_correct": score.verdict_correct,
                    "confidence": round(score.confidence, 4),
                    "metrics": {
                        metric.name: {
                            "value": round(metric.value, 4),
                            "category": metric.category,
                            "is_appendix": metric.is_appendix,
                            "details": metric.details,
                        }
                        for metric in score.metric_results
                    },
                    "appendix": dict(score.appendix),
                }
            )

        return {
            "game_id": game_score.game_id,
            "winner": game_score.winner,
            "primary_metric": game_score.summary.get("primary_metric", "unsafe_acceptance_rate"),
            "overall_score": game_score.final_scores.get("overall", 0.0),
            "final_scores": dict(game_score.final_scores),
            "summary": dict(game_score.summary),
            "metrics": {
                metric.name: {
                    "value": round(metric.value, 4),
                    "category": metric.category,
                    "is_primary": metric.is_primary,
                    "is_appendix": metric.is_appendix,
                    "higher_is_better": metric.higher_is_better,
                    "details": metric.details,
                }
                for metric in game_score.metric_results
            },
            "rounds": rounds,
        }
