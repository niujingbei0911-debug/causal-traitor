"""
L1 关联层因果工具
Pearl因果阶梯第一层：P(Y|X)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def _build_numeric_subset(
    data: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = data.loc[:, columns].dropna().copy()
    numeric = pd.DataFrame(index=subset.index)

    for column in columns:
        try:
            series = pd.to_numeric(subset[column])
        except Exception:
            series = subset[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric[column] = pd.to_numeric(series, errors="coerce")
        else:
            encoded = pd.get_dummies(series, prefix=column, dtype=float)
            if encoded.shape[1] == 0:
                raise ValueError(f"无法对列 {column} 做数值编码")
            numeric = pd.concat([numeric, encoded], axis=1)

    numeric = numeric.dropna()
    subset = subset.loc[numeric.index]
    if subset.empty:
        raise ValueError("可用于分析的样本为空")
    return subset, numeric


def _safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2 or len(x) < 3:
        return 0.0, 1.0
    statistic, p_value = stats.pearsonr(x, y)
    return float(statistic), float(p_value)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2 or len(x) < 3:
        return 0.0, 1.0
    statistic, p_value = stats.spearmanr(x, y)
    return float(statistic), float(p_value)


def _residualize(
    data: pd.DataFrame, target: str, controls: list[str]
) -> tuple[pd.Series, pd.Index]:
    subset, numeric = _build_numeric_subset(data, [target, *controls])
    y = pd.to_numeric(subset[target], errors="coerce")
    if not controls:
        return y - y.mean(), subset.index

    x_controls = numeric.drop(columns=[target], errors="ignore")
    if x_controls.empty:
        return y - y.mean(), subset.index

    model = LinearRegression()
    model.fit(x_controls, y)
    fitted = pd.Series(model.predict(x_controls), index=subset.index)
    return y - fitted, subset.index


def _effect_size_label(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.1:
        return "negligible"
    if absolute < 0.3:
        return "small"
    if absolute < 0.5:
        return "medium"
    return "large"


def compute_correlation(data: pd.DataFrame, x: str, y: str) -> dict:
    """计算变量间相关系数"""
    subset, numeric = _build_numeric_subset(data, [x, y])
    x_numeric = numeric.iloc[:, 0]
    y_numeric = numeric.iloc[:, 1] if numeric.shape[1] > 1 else numeric.iloc[:, 0]

    pearson_r, pearson_p = _safe_pearson(x_numeric, y_numeric)
    spearman_r, spearman_p = _safe_spearman(x_numeric, y_numeric)

    return {
        "x": x,
        "y": y,
        "n_samples": int(len(subset)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "significant": pearson_p < 0.05,
        "effect_size": _effect_size_label(pearson_r),
    }


def correlation_analysis(data: pd.DataFrame, x: str, y: str) -> dict:
    """兼容设计文档中的命名。"""
    return compute_correlation(data, x, y)


def conditional_independence_test(
    data: pd.DataFrame, x: str, y: str, z: list[str]
) -> dict:
    """条件独立性检验 (Fisher-z / Chi-square)"""
    if not z:
        result = compute_correlation(data, x, y)
        return {
            "x": x,
            "y": y,
            "conditioning_set": [],
            "test": "pearson",
            "statistic": result["pearson_r"],
            "p_value": result["pearson_p"],
            "independent": not result["significant"],
            "n_samples": result["n_samples"],
        }

    residual_x, index_x = _residualize(data, x, z)
    residual_y, index_y = _residualize(data.loc[index_x], y, z)
    aligned_index = index_x.intersection(index_y)
    residual_x = residual_x.loc[aligned_index]
    residual_y = residual_y.loc[aligned_index]
    statistic, p_value = _safe_pearson(residual_x, residual_y)

    return {
        "x": x,
        "y": y,
        "conditioning_set": z,
        "test": "partial_correlation",
        "statistic": statistic,
        "p_value": p_value,
        "independent": p_value >= 0.05,
        "n_samples": int(len(aligned_index)),
    }


def detect_simpson_paradox(
    data: pd.DataFrame, x: str, y: str, z: str
) -> dict:
    """检测辛普森悖论"""
    working = data.loc[:, [x, y, z]].dropna().copy()
    if working.empty:
        raise ValueError("没有足够的数据用于辛普森悖论检测")

    if pd.api.types.is_numeric_dtype(working[z]) and working[z].nunique() > 8:
        working["_group"] = pd.qcut(
            working[z], q=min(4, working[z].nunique()), duplicates="drop"
        )
        group_column = "_group"
    else:
        group_column = z

    overall = compute_correlation(working, x, y)
    group_correlations: list[dict] = []

    for group_value, group_df in working.groupby(group_column):
        if len(group_df) < 3:
            continue
        group_result = compute_correlation(group_df, x, y)
        group_result["group"] = str(group_value)
        group_correlations.append(group_result)

    overall_sign = np.sign(overall["pearson_r"])
    subgroup_signs = [np.sign(item["pearson_r"]) for item in group_correlations if item["significant"]]
    dominant_subgroup_sign = 0
    if subgroup_signs:
        dominant_subgroup_sign = int(np.sign(np.sum(subgroup_signs)))

    paradox = (
        len(group_correlations) >= 2
        and dominant_subgroup_sign != 0
        and overall_sign != 0
        and dominant_subgroup_sign != overall_sign
    )

    return {
        "x": x,
        "y": y,
        "stratifier": z,
        "overall_correlation": overall["pearson_r"],
        "overall_p_value": overall["pearson_p"],
        "group_correlations": group_correlations,
        "simpson_paradox_detected": paradox,
    }


def partial_correlation(
    data: pd.DataFrame, x: str, y: str, controls: list[str]
) -> dict:
    """偏相关分析"""
    result = conditional_independence_test(data, x, y, controls)
    return {
        "x": x,
        "y": y,
        "controls": controls,
        "partial_correlation": result["statistic"],
        "p_value": result["p_value"],
        "significant": result["p_value"] < 0.05,
        "n_samples": result["n_samples"],
    }


def proxy_support_check(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    proxy: str,
    controls: list[str] | None = None,
) -> dict:
    """Check whether an observed proxy carries useful information for the target relation."""

    controls = controls or []
    proxy_treatment = compute_correlation(data, proxy, treatment)
    proxy_outcome = compute_correlation(data, proxy, outcome)
    conditioning_set = [proxy, *controls]
    partial = conditional_independence_test(data, treatment, outcome, conditioning_set)
    proxy_alignment = 0.5 * abs(proxy_treatment["pearson_r"]) + 0.5 * abs(proxy_outcome["pearson_r"])
    supports_proxy = (
        proxy_treatment["significant"]
        and (
            proxy_outcome["significant"]
            or proxy_alignment >= 0.12
            or partial["p_value"] < 0.05
        )
    )

    return {
        "proxy": proxy,
        "treatment": treatment,
        "outcome": outcome,
        "controls": list(controls),
        "proxy_alignment": float(proxy_alignment),
        "proxy_treatment_significant": bool(proxy_treatment["significant"]),
        "proxy_outcome_significant": bool(proxy_outcome["significant"]),
        "conditioned_association": float(partial["statistic"]),
        "conditioned_p_value": float(partial["p_value"]),
        "supports_proxy_sufficiency": bool(supports_proxy),
        "n_samples": int(min(proxy_treatment["n_samples"], proxy_outcome["n_samples"])),
    }
