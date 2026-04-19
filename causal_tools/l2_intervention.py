"""
L2 干预层因果工具
Pearl因果阶梯第二层：P(Y|do(X))
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import networkx as nx
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def _build_design_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=data.index)
    return pd.get_dummies(data.loc[:, columns], drop_first=True, dtype=float)


def _prepare_regression_frame(
    data: pd.DataFrame, target: str, features: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    frame = data.loc[:, [target, *features]].dropna().copy()
    y = pd.to_numeric(frame[target], errors="coerce")
    x = _build_design_matrix(frame, features)
    valid_mask = y.notna()
    if not x.empty:
        valid_mask &= x.notna().all(axis=1)
        x = x.loc[valid_mask]
    y = y.loc[valid_mask]
    if y.empty:
        raise ValueError("回归样本为空")
    return y, x


def _fit_ols(y: pd.Series, x: pd.DataFrame, feature_names: list[str]) -> dict:
    x_values = x.to_numpy(dtype=float) if not x.empty else np.empty((len(y), 0))
    design = np.column_stack([np.ones(len(y)), x_values])
    beta, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    fitted = design @ beta
    residuals = y.to_numpy(dtype=float) - fitted
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = float(np.sum(residuals ** 2) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_stats = np.divide(
        beta,
        standard_errors,
        out=np.zeros_like(beta),
        where=standard_errors > 0,
    )
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    residual_sum_squares = float(np.sum(residuals ** 2))
    r_squared = 0.0 if total_sum_squares == 0 else 1 - residual_sum_squares / total_sum_squares

    names = ["intercept", *feature_names]
    return {
        "coefficients": {name: float(value) for name, value in zip(names, beta)},
        "standard_errors": {name: float(value) for name, value in zip(names, standard_errors)},
        "p_values": {name: float(value) for name, value in zip(names, p_values)},
        "r_squared": float(r_squared),
        "n_samples": int(len(y)),
        "residual_std": float(np.sqrt(sigma2)),
    }


def _linear_effect(
    data: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
) -> dict:
    y, x = _prepare_regression_frame(data, outcome, [treatment, *covariates])
    encoded_treatment = _build_design_matrix(data.loc[y.index], [treatment])
    if encoded_treatment.empty:
        raise ValueError(f"处理变量 {treatment} 无法编码")
    x = pd.concat([encoded_treatment, _build_design_matrix(data.loc[y.index], covariates)], axis=1)
    feature_names = list(x.columns)
    fit = _fit_ols(y, x, feature_names)
    treatment_column = feature_names[0]
    effect = fit["coefficients"][treatment_column]
    error = fit["standard_errors"][treatment_column]
    return {
        "causal_effect": effect,
        "estimated_effect": effect,
        "std_error": error,
        "p_value": fit["p_values"][treatment_column],
        "confidence_interval": [float(effect - 1.96 * error), float(effect + 1.96 * error)],
        "n_samples": fit["n_samples"],
        "r_squared": fit["r_squared"],
        "model_details": fit,
    }


def _candidate_backdoor_set(graph: nx.DiGraph, treatment: str, outcome: str) -> list[str]:
    candidates = []
    for parent in graph.predecessors(treatment):
        if parent != outcome:
            candidates.append(parent)
    return sorted(set(candidates))


def _safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().mean())


def _association_signal(data: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    frame = data.loc[:, [left, right]].dropna().copy()
    if frame.empty or len(frame) < 8:
        return 0.0, 1.0

    left_series = pd.to_numeric(frame[left], errors="coerce")
    right_series = pd.to_numeric(frame[right], errors="coerce")
    valid = left_series.notna() & right_series.notna()
    left_series = left_series.loc[valid]
    right_series = right_series.loc[valid]
    if len(left_series) < 8:
        return 0.0, 1.0

    left_binary = set(left_series.unique()) <= {0, 1}
    right_binary = set(right_series.unique()) <= {0, 1}

    if left_binary and len(left_series.unique()) == 2:
        group_one = right_series.loc[left_series == 1]
        group_zero = right_series.loc[left_series == 0]
        if len(group_one) < 3 or len(group_zero) < 3:
            return 0.0, 1.0
        statistic = float(group_one.mean() - group_zero.mean())
        _, p_value = stats.ttest_ind(group_one, group_zero, equal_var=False, nan_policy="omit")
        return statistic, float(1.0 if np.isnan(p_value) else p_value)

    if right_binary and len(right_series.unique()) == 2:
        group_one = left_series.loc[right_series == 1]
        group_zero = left_series.loc[right_series == 0]
        if len(group_one) < 3 or len(group_zero) < 3:
            return 0.0, 1.0
        statistic = float(group_one.mean() - group_zero.mean())
        _, p_value = stats.ttest_ind(group_one, group_zero, equal_var=False, nan_policy="omit")
        return statistic, float(1.0 if np.isnan(p_value) else p_value)

    if left_series.nunique() < 2 or right_series.nunique() < 2:
        return 0.0, 1.0
    statistic, p_value = stats.pearsonr(left_series, right_series)
    return float(statistic), float(p_value)


def backdoor_adjustment(
    data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str
) -> dict:
    """后门调整估计因果效应"""
    adjustment_set = _candidate_backdoor_set(graph, treatment, outcome)
    validation = validate_backdoor_criterion(graph, treatment, outcome, adjustment_set)
    estimate = _linear_effect(data, treatment, outcome, adjustment_set)
    estimate["adjustment_set"] = adjustment_set
    estimate["is_valid_adjustment"] = validation
    return estimate


def backdoor_adjustment_check(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    graph: nx.DiGraph | None = None,
) -> dict:
    """兼容设计文档中的命名。"""
    estimate = _linear_effect(data, treatment, outcome, adjustment_set)
    naive_effect = _linear_effect(data, treatment, outcome, [])
    estimate["adjustment_set"] = adjustment_set
    estimate["naive_effect"] = naive_effect["causal_effect"]
    estimate["delta_from_naive"] = estimate["causal_effect"] - naive_effect["causal_effect"]
    estimate["is_robust"] = abs(estimate["delta_from_naive"]) <= max(abs(naive_effect["causal_effect"]) * 0.5, 0.1)
    estimate["is_valid_adjustment"] = (
        validate_backdoor_criterion(graph, treatment, outcome, adjustment_set)
        if graph is not None
        else None
    )
    estimate["adjustment_support_basis"] = "graph_validation" if graph is not None else "statistical_heuristic"
    diagnostics = []
    supports_adjustment_set = bool(adjustment_set)
    for candidate in adjustment_set:
        treatment_assoc, treatment_p = _association_signal(data, candidate, treatment)
        outcome_assoc, outcome_p = _association_signal(data, candidate, outcome)
        conditional_effect = _linear_effect(data, candidate, outcome, [treatment])
        candidate_support = treatment_p < 0.05 and conditional_effect["p_value"] < 0.05
        diagnostics.append(
            {
                "variable": candidate,
                "association_with_treatment": float(treatment_assoc),
                "association_with_treatment_p_value": float(treatment_p),
                "association_with_outcome": float(outcome_assoc),
                "association_with_outcome_p_value": float(outcome_p),
                "conditional_outcome_effect": float(conditional_effect["causal_effect"]),
                "conditional_outcome_p_value": float(conditional_effect["p_value"]),
                "acts_like_confounder": bool(candidate_support),
            }
        )
        supports_adjustment_set &= candidate_support
    estimate["adjustment_diagnostics"] = diagnostics
    estimate["supports_adjustment_set"] = bool(adjustment_set) and bool(supports_adjustment_set)
    return estimate


def frontdoor_adjustment(
    data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str, mediator: str
) -> dict:
    """前门调整估计因果效应"""
    mediation = frontdoor_estimation(data, treatment, mediator, outcome)
    mediation["graph_valid"] = graph.has_edge(treatment, mediator) and graph.has_edge(mediator, outcome)
    return mediation


def frontdoor_estimation(
    data: pd.DataFrame, treatment: str, mediator: str, outcome: str
) -> dict:
    """兼容设计文档中的命名。"""
    mediator_fit = _linear_effect(data, treatment, mediator, [])
    outcome_direct = _linear_effect(data, treatment, outcome, [])
    outcome_with_mediator = _linear_effect(data, treatment, outcome, [mediator])

    y, x = _prepare_regression_frame(data, outcome, [treatment, mediator])
    fit = _fit_ols(y, x, list(x.columns))
    mediator_columns = [column for column in x.columns if column.startswith(f"{mediator}_") or column == mediator]
    mediator_effect = float(
        np.mean([fit["coefficients"][column] for column in mediator_columns])
    ) if mediator_columns else 0.0

    indirect = mediator_fit["causal_effect"] * mediator_effect
    total = outcome_with_mediator["causal_effect"] + indirect

    return {
        "frontdoor_effect": float(total),
        "total_effect": float(total),
        "direct_effect": float(outcome_with_mediator["causal_effect"]),
        "indirect_effect": float(indirect),
        "mediator_effect": float(mediator_effect),
        "naive_effect": float(outcome_direct["causal_effect"]),
    }


def iv_estimation(
    data: pd.DataFrame,
    instrument: str,
    treatment: str,
    outcome: str,
    covariates: list[str] | None = None,
) -> dict:
    """工具变量估计（2SLS）"""
    covariates = covariates or []
    frame = data.loc[:, [instrument, treatment, outcome, *covariates]].dropna().copy()
    if frame.empty:
        raise ValueError("没有足够的数据用于IV估计")

    first_stage_y, first_stage_x = _prepare_regression_frame(frame, treatment, [instrument, *covariates])
    first_stage_fit = _fit_ols(first_stage_y, first_stage_x, list(first_stage_x.columns))
    predicted_treatment = pd.Series(
        np.column_stack([np.ones(len(first_stage_y)), first_stage_x.to_numpy()]) @
        np.array([first_stage_fit["coefficients"]["intercept"], *[first_stage_fit["coefficients"][name] for name in first_stage_x.columns]]),
        index=first_stage_y.index,
    )

    reduced_y, reduced_x = _prepare_regression_frame(frame.loc[first_stage_y.index], treatment, covariates)
    reduced_fit = _fit_ols(reduced_y, reduced_x, list(reduced_x.columns)) if covariates else {
        "r_squared": 0.0,
    }
    k = max(1, len(first_stage_x.columns) - len(covariates))
    denom = max(len(first_stage_y) - len(first_stage_x.columns) - 1, 1)
    r2_delta = max(first_stage_fit["r_squared"] - reduced_fit["r_squared"], 0.0)
    first_stage_f = (
        (r2_delta / k) / max((1 - first_stage_fit["r_squared"]) / denom, 1e-8)
    )

    second_stage_features = pd.DataFrame(index=predicted_treatment.index)
    second_stage_features["predicted_treatment"] = predicted_treatment
    covariate_matrix = _build_design_matrix(frame.loc[predicted_treatment.index], covariates)
    if not covariate_matrix.empty:
        second_stage_features = pd.concat([second_stage_features, covariate_matrix], axis=1)

    outcome_y = pd.to_numeric(frame.loc[predicted_treatment.index, outcome], errors="coerce")
    second_stage_fit = _fit_ols(
        outcome_y,
        second_stage_features,
        list(second_stage_features.columns),
    )

    effect = second_stage_fit["coefficients"]["predicted_treatment"]
    error = second_stage_fit["standard_errors"]["predicted_treatment"]
    direct_path_check = _linear_effect(frame, instrument, outcome, [treatment, *covariates])
    covariate_balance: dict[str, dict[str, float]] = {}
    covariate_independence_flags: list[bool] = []
    for covariate in covariates:
        statistic, p_value = _association_signal(frame, instrument, covariate)
        covariate_balance[covariate] = {
            "association": float(statistic),
            "p_value": float(p_value),
        }
        covariate_independence_flags.append(bool(p_value >= 0.1 or abs(statistic) <= 0.1))
    supports_exclusion = bool(
        direct_path_check["p_value"] >= 0.1
        or abs(direct_path_check["causal_effect"]) <= 0.1
    )
    supports_independence = bool(covariate_independence_flags) and all(covariate_independence_flags)
    return {
        "causal_effect": float(effect),
        "std_error": float(error),
        "p_value": float(second_stage_fit["p_values"]["predicted_treatment"]),
        "first_stage_f": float(first_stage_f),
        "first_stage_r2": float(first_stage_fit["r_squared"]),
        "is_strong_instrument": bool(first_stage_f >= 10.0),
        "direct_path_check_effect": float(direct_path_check["causal_effect"]),
        "direct_path_check_p_value": float(direct_path_check["p_value"]),
        "covariate_balance": covariate_balance,
        "supports_exclusion_restriction": supports_exclusion,
        "supports_instrument_independence": supports_independence,
        "n_samples": int(len(frame)),
    }


def propensity_score_matching(
    data: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
) -> dict:
    """倾向得分匹配"""
    frame = data.loc[:, [treatment, outcome, *covariates]].dropna().copy()
    if frame.empty:
        raise ValueError("没有足够的数据用于倾向得分匹配")

    treatment_series = pd.to_numeric(frame[treatment], errors="coerce").astype(int)
    outcome_series = pd.to_numeric(frame[outcome], errors="coerce")
    covariate_matrix = _build_design_matrix(frame, covariates)
    if covariate_matrix.empty:
        raise ValueError("倾向得分匹配至少需要一个协变量")

    model = LogisticRegression(max_iter=1000)
    model.fit(covariate_matrix, treatment_series)
    propensity = model.predict_proba(covariate_matrix)[:, 1]

    treated_mask = treatment_series == 1
    control_mask = treatment_series == 0
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        raise ValueError("倾向得分匹配需要同时存在处理组和对照组")

    matcher = NearestNeighbors(n_neighbors=1)
    matcher.fit(propensity[control_mask].reshape(-1, 1))
    distances, indices = matcher.kneighbors(propensity[treated_mask].reshape(-1, 1))
    matched_outcomes = outcome_series.loc[control_mask].iloc[indices.flatten()].to_numpy()
    treated_outcomes = outcome_series.loc[treated_mask].to_numpy()
    att = float(np.mean(treated_outcomes - matched_outcomes))

    return {
        "att": att,
        "n_treated": int(treated_mask.sum()),
        "n_control": int(control_mask.sum()),
        "avg_match_distance": float(np.mean(distances)),
        "propensity_score_range": [float(propensity.min()), float(propensity.max())],
    }


def sensitivity_analysis(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    gamma_range: tuple[float, float] = (1.0, 3.0),
) -> dict:
    """简化版灵敏度分析。"""
    frame = data.loc[:, [treatment, outcome]].dropna().copy()
    treated = pd.to_numeric(frame.loc[frame[treatment] == 1, outcome], errors="coerce")
    control = pd.to_numeric(frame.loc[frame[treatment] == 0, outcome], errors="coerce")
    effect = float(treated.mean() - control.mean())
    pooled_std = float(
        np.sqrt(
            (
                (treated.var(ddof=1) if len(treated) > 1 else 0.0)
                + (control.var(ddof=1) if len(control) > 1 else 0.0)
            ) / 2
        )
    )
    standardized_effect = 0.0 if pooled_std == 0 else abs(effect) / pooled_std
    robust_gamma = min(gamma_range[1], gamma_range[0] + standardized_effect)
    return {
        "effect": effect,
        "standardized_effect": standardized_effect,
        "robust_up_to_gamma": float(robust_gamma),
        "is_sensitive": robust_gamma < 1.5,
        "interpretation": (
            "结果对未观测混杂较敏感" if robust_gamma < 1.5 else "结果对中等强度未观测混杂较稳健"
        ),
    }


def overlap_check(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str] | None = None,
) -> dict:
    """Simple overlap / positivity diagnostic for binary treatments."""

    covariates = [str(column) for column in (covariates or []) if str(column) != treatment]
    frame = data.loc[:, [treatment, *covariates]].dropna().copy()
    if frame.empty:
        raise ValueError("没有足够的数据用于positivity检查")

    treatment_series = pd.to_numeric(frame[treatment], errors="coerce")
    valid_mask = treatment_series.notna()
    frame = frame.loc[valid_mask]
    treatment_series = treatment_series.loc[valid_mask]
    if frame.empty:
        raise ValueError("处理变量无法数值化")

    observed_values = set(float(value) for value in treatment_series.unique())
    if not observed_values <= {0.0, 1.0}:
        lower = float(treatment_series.quantile(0.1))
        upper = float(treatment_series.quantile(0.9))
        has_overlap = upper > lower
        return {
            "has_overlap": bool(has_overlap),
            "treated_rate": float((treatment_series > treatment_series.median()).mean()),
            "min_propensity": lower,
            "max_propensity": upper,
            "n_samples": int(len(frame)),
            "method": "continuous_quantile_overlap",
        }

    treated_rate = float((treatment_series >= 0.5).mean())
    min_propensity = treated_rate
    max_propensity = treated_rate
    method = "marginal_overlap"

    if covariates:
        covariate_matrix = _build_design_matrix(frame, covariates)
        if not covariate_matrix.empty and treatment_series.nunique() == 2:
            model = LogisticRegression(max_iter=1000)
            model.fit(covariate_matrix, treatment_series.astype(int))
            propensities = model.predict_proba(covariate_matrix)[:, 1]
            min_propensity = float(np.quantile(propensities, 0.05))
            max_propensity = float(np.quantile(propensities, 0.95))
            method = "propensity_model"

    has_overlap = min_propensity >= 0.05 and max_propensity <= 0.95
    return {
        "has_overlap": bool(has_overlap),
        "treated_rate": treated_rate,
        "min_propensity": float(min_propensity),
        "max_propensity": float(max_propensity),
        "n_samples": int(len(frame)),
        "method": method,
    }


def validate_backdoor_criterion(
    graph: nx.DiGraph, treatment: str, outcome: str, adjustment_set: list[str]
) -> bool:
    """验证后门准则是否满足"""
    if treatment not in graph or outcome not in graph:
        raise ValueError("处理变量或结果变量不在因果图中")

    adjustment_nodes = set(adjustment_set)
    if any(node in nx.descendants(graph, treatment) for node in adjustment_nodes):
        return False

    manipulated_graph = graph.copy()
    manipulated_graph.remove_edges_from(list(graph.out_edges(treatment)))
    return nx.algorithms.d_separation.is_d_separator(
        manipulated_graph, {treatment}, {outcome}, adjustment_nodes
    )
