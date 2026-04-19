"""Reporting helpers for experiment summaries and paired significance outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .significance import (
    HolmBonferroniEntry,
    PairedTestResult,
    accuracy_score,
    bootstrap_confidence_interval,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_test,
)


@dataclass(slots=True)
class MetricSummary:
    metric_name: str
    mean: float
    std: float
    ci_level: float
    ci_lower: float
    ci_upper: float
    n: int
    n_resamples: int
    observed_value: float
    formatted: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "ci_level": self.ci_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "observed_value": self.observed_value,
            "formatted": self.formatted,
        }


@dataclass(slots=True)
class ComparisonSummary:
    comparison: str
    model_a: str
    model_b: str
    raw_result: PairedTestResult
    adjusted_p_value: float | None = None
    reject_after_correction: bool | None = None
    correction: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = self.raw_result.to_dict()
        payload.update(
            {
                "comparison": self.comparison,
                "model_a": self.model_a,
                "model_b": self.model_b,
                "adjusted_p_value": self.adjusted_p_value,
                "reject_after_correction": self.reject_after_correction,
                "correction": self.correction,
            }
        )
        return payload


@dataclass(slots=True)
class PairwiseSignificanceReport:
    method: str
    metric_name: str
    alpha: float
    baseline: str | None
    comparisons: list[ComparisonSummary] = field(default_factory=list)
    correction_table: list[HolmBonferroniEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "metric_name": self.metric_name,
            "alpha": self.alpha,
            "baseline": self.baseline,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "correction_table": [entry.to_dict() for entry in self.correction_table],
        }


def format_mean_std_ci(
    mean: float,
    std: float,
    ci_lower: float,
    ci_upper: float,
    *,
    ci_level: float = 0.95,
    precision: int = 4,
) -> str:
    """Format a metric as mean ± std with its confidence interval."""

    percent = int(round(float(ci_level) * 100))
    plus_minus = "\u00B1"
    return (
        f"{mean:.{precision}f} {plus_minus} {std:.{precision}f} "
        f"({percent}% CI: {ci_lower:.{precision}f}, {ci_upper:.{precision}f})"
    )


def summarize_metric(
    metric_name: str,
    values: Sequence[Any],
    *,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = 0,
    precision: int = 4,
) -> MetricSummary:
    """Summarize one metric across seeds or runs."""

    ci_result = bootstrap_confidence_interval(
        values,
        statistic_name=metric_name,
        n_resamples=n_resamples,
        ci_level=ci_level,
        random_state=random_state,
    )
    return MetricSummary(
        metric_name=str(metric_name),
        mean=ci_result.sample_mean,
        std=ci_result.sample_std,
        ci_level=ci_result.ci_level,
        ci_lower=ci_result.ci_lower,
        ci_upper=ci_result.ci_upper,
        n=ci_result.n_samples,
        n_resamples=ci_result.n_resamples,
        observed_value=ci_result.observed_value,
        formatted=format_mean_std_ci(
            ci_result.sample_mean,
            ci_result.sample_std,
            ci_result.ci_lower,
            ci_result.ci_upper,
            ci_level=ci_result.ci_level,
            precision=precision,
        ),
    )


def summarize_metrics(
    metric_values: Mapping[str, Sequence[Any]],
    *,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = 0,
    precision: int = 4,
) -> Dict[str, MetricSummary]:
    """Summarize multiple metrics at once."""

    summaries: Dict[str, MetricSummary] = {}
    for offset, (metric_name, values) in enumerate(metric_values.items()):
        seed = None if random_state is None else int(random_state) + offset
        summaries[str(metric_name)] = summarize_metric(
            metric_name,
            values,
            n_resamples=n_resamples,
            ci_level=ci_level,
            random_state=seed,
            precision=precision,
        )
    return summaries


def metric_report_rows(metric_values: Mapping[str, Sequence[Any]], **kwargs: Any) -> list[Dict[str, Any]]:
    """Produce serializable rows suitable for tables or artifact JSON."""

    summaries = summarize_metrics(metric_values, **kwargs)
    rows: list[Dict[str, Any]] = []
    for metric_name, summary in summaries.items():
        row = summary.to_dict()
        row["metric"] = metric_name
        rows.append(row)
    return rows


def compare_predictions(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    *,
    method: str = "paired_bootstrap",
    metric_name: str = "accuracy",
    metric_fn=None,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    random_state: int | None = 0,
    exact: bool = True,
    continuity: bool = True,
) -> PairedTestResult:
    """Compare two prediction sets with the requested paired significance test."""

    method = str(method).strip().lower()
    metric_fn = metric_fn or accuracy_score

    if method == "paired_bootstrap":
        return paired_bootstrap_test(
            y_true,
            pred_a,
            pred_b,
            metric_fn=metric_fn,
            metric_name=metric_name,
            n_resamples=n_resamples,
            ci_level=ci_level,
            alpha=alpha,
            random_state=random_state,
        )
    if method == "mcnemar":
        return mcnemar_test(
            y_true,
            pred_a,
            pred_b,
            metric_name=metric_name,
            exact=exact,
            continuity=continuity,
            alpha=alpha,
        )
    raise ValueError(f"Unsupported comparison method: {method!r}")


def compare_prediction_groups(
    y_true: Sequence[Any],
    prediction_groups: Mapping[str, Sequence[Any]],
    *,
    baseline: str | None = None,
    method: str = "paired_bootstrap",
    metric_name: str = "accuracy",
    metric_fn=None,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    random_state: int | None = 0,
    correction: str = "holm-bonferroni",
    exact: bool = True,
    continuity: bool = True,
) -> PairwiseSignificanceReport:
    """Run pairwise significance tests against a baseline and apply correction."""

    if len(prediction_groups) < 2:
        raise ValueError("At least two prediction groups are required.")

    group_items = list(prediction_groups.items())
    if baseline is None:
        baseline = str(group_items[0][0])

    if baseline not in prediction_groups:
        raise ValueError(f"Unknown baseline group: {baseline!r}")

    baseline_predictions = prediction_groups[baseline]
    comparisons: list[ComparisonSummary] = []
    raw_p_values: Dict[str, float] = {}

    comparison_index = 0
    for label, predictions in group_items:
        label = str(label)
        if label == baseline:
            continue

        comparison_name = f"{baseline} vs {label}"
        seed = None if random_state is None else int(random_state) + comparison_index
        result = compare_predictions(
            y_true,
            baseline_predictions,
            predictions,
            method=method,
            metric_name=metric_name,
            metric_fn=metric_fn,
            n_resamples=n_resamples,
            ci_level=ci_level,
            alpha=alpha,
            random_state=seed,
            exact=exact,
            continuity=continuity,
        )
        comparisons.append(
            ComparisonSummary(
                comparison=comparison_name,
                model_a=baseline,
                model_b=label,
                raw_result=result,
            )
        )
        raw_p_values[comparison_name] = result.p_value
        comparison_index += 1

    correction = str(correction).strip().lower()
    correction_table: list[HolmBonferroniEntry] = []
    if correction in {"holm", "holm-bonferroni", "holm_bonferroni"} and raw_p_values:
        correction_table = holm_bonferroni(raw_p_values, alpha=alpha)
        correction_lookup = {entry.hypothesis: entry for entry in correction_table}
        for comparison in comparisons:
            entry = correction_lookup[comparison.comparison]
            comparison.adjusted_p_value = entry.adjusted_p_value
            comparison.reject_after_correction = entry.reject
            comparison.correction = "holm-bonferroni"
    elif correction not in {"none", ""}:
        raise ValueError(f"Unsupported correction method: {correction!r}")

    return PairwiseSignificanceReport(
        method=str(method),
        metric_name=str(metric_name),
        alpha=float(alpha),
        baseline=baseline,
        comparisons=comparisons,
        correction_table=correction_table,
    )


_TRUE_ANNOTATION_LABELS = {"1", "t", "true", "y", "yes"}
_FALSE_ANNOTATION_LABELS = {"0", "f", "false", "n", "no"}
_MISSING_ANNOTATION_LABELS = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "skip",
    "skipped",
    "not applicable",
    "not_applicable",
}


def _normalize_annotation_value_details(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, bool):
        return ("yes" if value else "no"), False
    if isinstance(value, int) and value in {0, 1}:
        return ("yes" if value else "no"), False
    normalized = str(value).strip().lower()
    if normalized in _TRUE_ANNOTATION_LABELS:
        return "yes", False
    if normalized in _FALSE_ANNOTATION_LABELS:
        return "no", False
    if normalized in _MISSING_ANNOTATION_LABELS:
        return None, False
    return None, bool(normalized)


def normalize_human_audit_label(value: Any) -> str | None:
    normalized, _ = _normalize_annotation_value_details(value)
    return normalized


def _cohen_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    paired = [
        (normalize_human_audit_label(left), normalize_human_audit_label(right))
        for left, right in zip(labels_a, labels_b)
    ]
    paired = [(left, right) for left, right in paired if left is not None and right is not None]
    if not paired:
        return 0.0

    categories = sorted({label for pair in paired for label in pair})
    if not categories:
        return 0.0

    observed = sum(1 for left, right in paired if left == right) / len(paired)
    distribution_a = {label: 0 for label in categories}
    distribution_b = {label: 0 for label in categories}
    for left, right in paired:
        distribution_a[left] += 1
        distribution_b[right] += 1

    total = float(len(paired))
    expected = sum(
        (distribution_a[label] / total) * (distribution_b[label] / total)
        for label in categories
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / max(1e-12, 1.0 - expected)


def summarize_human_audit_agreement(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    annotator_a_prefix: str = "annotator_a_",
    annotator_b_prefix: str = "annotator_b_",
) -> Dict[str, Any]:
    """Summarize double-annotation agreement for one human-audit package."""

    summary: Dict[str, Any] = {"n_records": len(records), "fields": {}}
    for field_name in fields:
        invalid_label_count = 0
        labels_a = [
            _normalize_annotation_value_details(record.get(f"{annotator_a_prefix}{field_name}"))[0]
            for record in records
        ]
        labels_b = [
            _normalize_annotation_value_details(record.get(f"{annotator_b_prefix}{field_name}"))[0]
            for record in records
        ]
        for record in records:
            _, left_invalid = _normalize_annotation_value_details(record.get(f"{annotator_a_prefix}{field_name}"))
            _, right_invalid = _normalize_annotation_value_details(record.get(f"{annotator_b_prefix}{field_name}"))
            invalid_label_count += int(left_invalid) + int(right_invalid)
        paired = [
            (left, right)
            for left, right in zip(labels_a, labels_b)
            if left is not None and right is not None
        ]
        n_scored = len(paired)
        percent_agreement = (
            sum(1 for left, right in paired if left == right) / n_scored
            if n_scored
            else 0.0
        )
        distribution_a: Dict[str, int] = {}
        distribution_b: Dict[str, int] = {}
        for left, right in paired:
            distribution_a[left] = distribution_a.get(left, 0) + 1
            distribution_b[right] = distribution_b.get(right, 0) + 1
        summary["fields"][str(field_name)] = {
            "n_scored": n_scored,
            "percent_agreement": percent_agreement,
            "cohen_kappa": _cohen_kappa(labels_a, labels_b),
            "annotator_a_distribution": distribution_a,
            "annotator_b_distribution": distribution_b,
            "invalid_label_count": invalid_label_count,
        }
    return summary
