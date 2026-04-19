"""Statistical significance utilities for verdict-centric evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

PredictionMetric = Callable[[Sequence[Any], Sequence[Any]], float]


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence of numeric values.")
    if array.size == 0:
        raise ValueError("At least one value is required.")
    return array


def _make_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _sample_std(values: np.ndarray) -> float:
    ddof = 1 if values.size > 1 else 0
    return float(np.std(values, ddof=ddof))


def _validate_ci_level(ci_level: float) -> float:
    ci_level = float(ci_level)
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1.")
    return ci_level


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha


def accuracy_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one prediction is required.")
    correct = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    return correct / len(y_true)


@dataclass(slots=True)
class BootstrapCIResult:
    statistic_name: str
    observed_value: float
    sample_mean: float
    sample_std: float
    ci_level: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    n_resamples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic_name": self.statistic_name,
            "observed_value": self.observed_value,
            "sample_mean": self.sample_mean,
            "sample_std": self.sample_std,
            "ci_level": self.ci_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "n_resamples": self.n_resamples,
        }


@dataclass(slots=True)
class PairedTestResult:
    method: str
    metric_name: str
    score_a: float
    score_b: float
    observed_difference: float
    p_value: float
    alpha: float
    significant: bool
    ci_level: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_samples: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "metric_name": self.metric_name,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "observed_difference": self.observed_difference,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
            "ci_level": self.ci_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class HolmBonferroniEntry:
    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    threshold: float
    reject: bool
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "threshold": self.threshold,
            "reject": self.reject,
            "rank": self.rank,
        }


def bootstrap_confidence_interval(
    values: Sequence[Any],
    *,
    statistic_fn: Callable[[np.ndarray], float] | None = None,
    statistic_name: str = "mean",
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    random_state: int | np.random.Generator | None = 0,
) -> BootstrapCIResult:
    """Estimate a percentile bootstrap confidence interval for one metric."""

    observed = _as_float_array(values)
    ci_level = _validate_ci_level(ci_level)
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive.")
    n_resamples = int(n_resamples)
    rng = _make_rng(random_state)
    statistic_fn = statistic_fn or (lambda sample: float(np.mean(sample)))

    bootstrap_values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample_indices = rng.integers(0, observed.size, size=observed.size)
        sample = observed[sample_indices]
        bootstrap_values[index] = float(statistic_fn(sample))

    alpha_tail = (1.0 - ci_level) / 2.0
    ci_lower = float(np.quantile(bootstrap_values, alpha_tail))
    ci_upper = float(np.quantile(bootstrap_values, 1.0 - alpha_tail))

    return BootstrapCIResult(
        statistic_name=str(statistic_name),
        observed_value=float(statistic_fn(observed)),
        sample_mean=float(np.mean(observed)),
        sample_std=_sample_std(observed),
        ci_level=ci_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=int(observed.size),
        n_resamples=n_resamples,
    )


def _validate_prediction_triplet(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y_true) != len(pred_a) or len(y_true) != len(pred_b):
        raise ValueError("y_true, pred_a, and pred_b must have the same length.")
    if len(y_true) == 0:
        raise ValueError("At least one paired prediction is required.")
    return (
        np.asarray(list(y_true), dtype=object),
        np.asarray(list(pred_a), dtype=object),
        np.asarray(list(pred_b), dtype=object),
    )


def paired_bootstrap_test(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    *,
    metric_fn: PredictionMetric | None = None,
    metric_name: str = "accuracy",
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> PairedTestResult:
    """Paired bootstrap significance test for two prediction sets."""

    truth, first, second = _validate_prediction_triplet(y_true, pred_a, pred_b)
    ci_level = _validate_ci_level(ci_level)
    alpha = _validate_alpha(alpha)
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive.")
    n_resamples = int(n_resamples)
    rng = _make_rng(random_state)
    metric_fn = metric_fn or accuracy_score

    score_a = float(metric_fn(truth.tolist(), first.tolist()))
    score_b = float(metric_fn(truth.tolist(), second.tolist()))
    observed_difference = score_b - score_a

    differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample_indices = rng.integers(0, truth.size, size=truth.size)
        sampled_truth = truth[sample_indices].tolist()
        sampled_a = first[sample_indices].tolist()
        sampled_b = second[sample_indices].tolist()
        differences[index] = float(metric_fn(sampled_truth, sampled_b) - metric_fn(sampled_truth, sampled_a))

    alpha_tail = (1.0 - ci_level) / 2.0
    ci_lower = float(np.quantile(differences, alpha_tail))
    ci_upper = float(np.quantile(differences, 1.0 - alpha_tail))

    non_positive = int(np.sum(differences <= 0.0))
    non_negative = int(np.sum(differences >= 0.0))
    p_value = min(
        1.0,
        2.0 * min(
            (non_positive + 1) / (n_resamples + 1),
            (non_negative + 1) / (n_resamples + 1),
        ),
    )

    return PairedTestResult(
        method="paired_bootstrap",
        metric_name=str(metric_name),
        score_a=score_a,
        score_b=score_b,
        observed_difference=observed_difference,
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
        ci_level=ci_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=int(truth.size),
        details={
            "n_resamples": n_resamples,
            "positive_fraction": float(np.mean(differences > 0.0)),
            "negative_fraction": float(np.mean(differences < 0.0)),
        },
    )


def _exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    if n > 1000:
        raise ValueError("Exact McNemar fallback threshold exceeded.")
    tail = 0.0
    for index in range(0, k + 1):
        tail += math.comb(n, index) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    *,
    metric_name: str = "accuracy",
    exact: bool = True,
    continuity: bool = True,
    alpha: float = 0.05,
) -> PairedTestResult:
    """McNemar test over paired correctness outcomes."""

    truth, first, second = _validate_prediction_triplet(y_true, pred_a, pred_b)
    alpha = _validate_alpha(alpha)

    first_correct = first == truth
    second_correct = second == truth

    a_correct_b_wrong = int(np.sum(first_correct & ~second_correct))
    a_wrong_b_correct = int(np.sum(~first_correct & second_correct))
    discordant = a_correct_b_wrong + a_wrong_b_correct

    score_a = float(np.mean(first_correct))
    score_b = float(np.mean(second_correct))
    observed_difference = score_b - score_a

    method = "mcnemar_exact" if exact else "mcnemar"
    details: Dict[str, Any] = {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
    }

    if discordant == 0:
        p_value = 1.0
        details["test_statistic"] = 0.0
        details["exact_used"] = bool(exact)
    elif exact:
        try:
            p_value = _exact_binomial_two_sided(min(a_correct_b_wrong, a_wrong_b_correct), discordant)
            details["test_statistic"] = None
            details["exact_used"] = True
        except ValueError:
            method = "mcnemar"
            statistic = (
                ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2) / discordant
                if continuity
                else ((a_correct_b_wrong - a_wrong_b_correct) ** 2) / discordant
            )
            p_value = math.erfc(math.sqrt(statistic / 2.0))
            details["test_statistic"] = float(statistic)
            details["exact_used"] = False
    else:
        statistic = (
            ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2) / discordant
            if continuity
            else ((a_correct_b_wrong - a_wrong_b_correct) ** 2) / discordant
        )
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        details["test_statistic"] = float(statistic)
        details["exact_used"] = False

    details["continuity_corrected"] = bool(continuity)

    return PairedTestResult(
        method=method,
        metric_name=str(metric_name),
        score_a=score_a,
        score_b=score_b,
        observed_difference=observed_difference,
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
        ci_level=None,
        ci_lower=None,
        ci_upper=None,
        n_samples=int(truth.size),
        details=details,
    )


def holm_bonferroni(
    p_values: Mapping[str, Any] | Sequence[Any],
    *,
    alpha: float = 0.05,
) -> list[HolmBonferroniEntry]:
    """Apply Holm-Bonferroni correction to one family of p-values."""

    alpha = _validate_alpha(alpha)

    if isinstance(p_values, Mapping):
        items = [(str(label), float(value)) for label, value in p_values.items()]
    else:
        items = [(str(index), float(value)) for index, value in enumerate(p_values)]

    if not items:
        return []

    sorted_items = sorted(items, key=lambda item: item[1])
    m = len(sorted_items)

    adjusted_sorted: list[float] = []
    running_max = 0.0
    for rank, (_, raw_p_value) in enumerate(sorted_items, start=1):
        adjusted = min(1.0, (m - rank + 1) * raw_p_value)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append(running_max)

    entries: list[HolmBonferroniEntry] = []
    still_rejecting = True
    for rank, ((label, raw_p_value), adjusted_p_value) in enumerate(
        zip(sorted_items, adjusted_sorted),
        start=1,
    ):
        threshold = alpha / (m - rank + 1)
        reject = bool(still_rejecting and raw_p_value <= threshold)
        if not reject:
            still_rejecting = False
        entries.append(
            HolmBonferroniEntry(
                hypothesis=label,
                raw_p_value=float(raw_p_value),
                adjusted_p_value=float(adjusted_p_value),
                threshold=float(threshold),
                reject=reject,
                rank=rank,
            )
        )

    return entries
