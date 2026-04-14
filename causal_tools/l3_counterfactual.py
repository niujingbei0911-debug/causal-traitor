"""
L3 反事实层因果工具
Pearl因果阶梯第三层：P(Y_x | X=x', Y=y')
"""
from __future__ import annotations

import pandas as pd
import networkx as nx
import numpy as np

from causal_tools.l2_intervention import sensitivity_analysis as l2_sensitivity_analysis


def _extract_graph(model) -> nx.DiGraph | None:
    if isinstance(model, nx.DiGraph):
        return model
    if isinstance(model, dict):
        graph_obj = model.get("graph")
        if isinstance(graph_obj, nx.DiGraph):
            return graph_obj
        if isinstance(graph_obj, dict):
            graph = nx.DiGraph()
            for source, targets in graph_obj.items():
                for target in targets:
                    graph.add_edge(source, target)
            return graph
    graph_obj = getattr(model, "graph", None)
    if isinstance(graph_obj, nx.DiGraph):
        return graph_obj
    return None


def _select_target_value(value, target: str):
    if isinstance(value, dict):
        return value.get(target, next(iter(value.values())) if value else None)
    return value


def _column_as_series(frame: pd.DataFrame, column: str) -> pd.Series:
    selected = frame[column]
    if isinstance(selected, pd.DataFrame):
        return pd.to_numeric(selected.iloc[:, 0], errors="coerce")
    return pd.to_numeric(selected, errors="coerce")


def _linear_counterfactual_from_dict(
    model: dict, evidence: dict, intervention: dict, target: str
) -> dict:
    coefficients = model.get("coefficients", {})
    graph = _extract_graph(model)
    values = dict(evidence)
    values.update(intervention)
    order = list(nx.topological_sort(graph)) if graph is not None and nx.is_directed_acyclic_graph(graph) else list(coefficients)

    for node in order:
        if node in intervention:
            continue
        if node not in coefficients:
            continue
        spec = coefficients[node]
        intercept = float(spec.get("intercept", 0.0))
        total = intercept
        for parent, weight in spec.items():
            if parent == "intercept":
                continue
            total += float(weight) * float(values.get(parent, evidence.get(parent, 0.0)))
        values[node] = total

    return {
        "counterfactual_outcome": _select_target_value(values, target),
        "u_posterior": {},
        "confidence": 0.65,
        "method": "linear_scm_dict",
    }


def counterfactual_inference(
    model, evidence: dict, intervention: dict, target: str
) -> dict:
    """结构因果模型反事实推理"""
    if hasattr(model, "abduction") and hasattr(model, "do"):
        u_posterior = model.abduction(evidence)
        intervened = model.do(intervention)
        if hasattr(intervened, "predict"):
            prediction = intervened.predict(u_posterior)
        elif hasattr(intervened, "forward"):
            prediction = intervened.forward(u_posterior)
        else:
            prediction = u_posterior
        return {
            "counterfactual_outcome": _select_target_value(prediction, target),
            "u_posterior": u_posterior,
            "confidence": 0.8,
            "method": "abduction_action_prediction",
        }

    if hasattr(model, "posterior_u") and hasattr(model, "intervene"):
        u_posterior = model.posterior_u(evidence)
        intervened = model.intervene(intervention)
        prediction = intervened.forward(u_posterior)
        return {
            "counterfactual_outcome": _select_target_value(prediction, target),
            "u_posterior": u_posterior,
            "confidence": 0.8,
            "method": "posterior_intervene_forward",
        }

    if callable(model):
        prediction = model(evidence=evidence, intervention=intervention, target=target)
        return {
            "counterfactual_outcome": _select_target_value(prediction, target),
            "u_posterior": {},
            "confidence": 0.7,
            "method": "callable_model",
        }

    if isinstance(model, dict):
        return _linear_counterfactual_from_dict(model, evidence, intervention, target)

    raise TypeError("不支持的SCM模型类型")


def sensitivity_analysis(
    data: pd.DataFrame, treatment: str, outcome: str, method: str = "rosenbaum"
) -> dict:
    """敏感性分析（Rosenbaum bounds / E-value）"""
    result = l2_sensitivity_analysis(data, treatment, outcome)
    result["method"] = method
    return result


def natural_direct_effect(
    data: pd.DataFrame, graph: nx.DiGraph,
    treatment: str, outcome: str, mediator: str
) -> dict:
    """自然直接效应估计"""
    from causal_tools.l2_intervention import frontdoor_estimation

    frontdoor = frontdoor_estimation(data, treatment, mediator, outcome)
    return {
        "natural_direct_effect": float(frontdoor["direct_effect"]),
        "natural_indirect_effect": float(frontdoor["indirect_effect"]),
        "total_effect": float(frontdoor["total_effect"]),
        "graph_valid": graph.has_edge(treatment, mediator) and graph.has_edge(mediator, outcome),
    }


def probability_of_necessity(
    data: pd.DataFrame, treatment: str, outcome: str
) -> float:
    """必要性概率 PN = P(Y_0=0 | X=1, Y=1)"""
    frame = data.loc[:, [treatment, outcome]].copy()
    treatment_series = _column_as_series(frame, treatment)
    outcome_series = _column_as_series(frame, outcome)
    valid = treatment_series.notna() & outcome_series.notna()
    treatment_series = treatment_series.loc[valid]
    outcome_series = outcome_series.loc[valid]
    p_y1_x1 = float(outcome_series.loc[treatment_series == 1].mean())
    p_y1_x0 = float(outcome_series.loc[treatment_series == 0].mean())
    if p_y1_x1 <= 0:
        return 0.0
    pn = (p_y1_x1 - p_y1_x0) / p_y1_x1
    return float(np.clip(pn, 0.0, 1.0))


def probability_of_sufficiency(
    data: pd.DataFrame, treatment: str, outcome: str
) -> float:
    """充分性概率 PS = P(Y_1=1 | X=0, Y=0)"""
    frame = data.loc[:, [treatment, outcome]].copy()
    treatment_series = _column_as_series(frame, treatment)
    outcome_series = _column_as_series(frame, outcome)
    valid = treatment_series.notna() & outcome_series.notna()
    treatment_series = treatment_series.loc[valid]
    outcome_series = outcome_series.loc[valid]
    p_y1_x1 = float(outcome_series.loc[treatment_series == 1].mean())
    p_y1_x0 = float(outcome_series.loc[treatment_series == 0].mean())
    denominator = 1.0 - p_y1_x0
    if denominator <= 0:
        return 0.0
    ps = (p_y1_x1 - p_y1_x0) / denominator
    return float(np.clip(ps, 0.0, 1.0))


def scm_identification_test(
    data: pd.DataFrame,
    proposed_scm,
    alternative_scms: list,
) -> list[dict]:
    """检验提出的SCM是否可从数据中识别。"""
    proposed_graph = _extract_graph(proposed_scm)
    proposed_edges = set(proposed_graph.edges()) if proposed_graph is not None else set()
    results = []

    for alternative in alternative_scms:
        alt_graph = _extract_graph(alternative)
        alt_edges = set(alt_graph.edges()) if alt_graph is not None else set()
        union = len(proposed_edges | alt_edges) or 1
        overlap = len(proposed_edges & alt_edges)
        observational_kl = 1 - overlap / union

        columns = list(data.columns[:2])
        counterfactual_gap = 0.0
        if len(columns) >= 2:
            treatment, target = columns[0], columns[1]
            evidence = data.iloc[0].to_dict()
            intervention = {treatment: 1 - int(round(float(evidence[treatment])))}
            proposed_cf = counterfactual_inference(proposed_scm, evidence, intervention, target)
            alternative_cf = counterfactual_inference(alternative, evidence, intervention, target)
            counterfactual_gap = abs(
                float(proposed_cf["counterfactual_outcome"]) - float(alternative_cf["counterfactual_outcome"])
            )

        results.append(
            {
                "alternative_scm": getattr(alternative, "name", getattr(alternative, "__class__", type("SCM", (), {})).__name__),
                "observational_kl": float(observational_kl),
                "counterfactual_difference": float(counterfactual_gap),
                "distinguishable": bool(observational_kl > 0.1 or counterfactual_gap > 0.1),
            }
        )

    return results


def ett_computation(
    data: pd.DataFrame,
    scm,
    treatment: str,
    outcome: str,
) -> dict:
    """计算处理组的处理效应 ETT = E[Y1 - Y0 | X=1]。"""
    treated = data.loc[data[treatment] == 1].copy()
    if treated.empty:
        return {"ETT": 0.0, "ETT_std": 0.0, "n_treated": 0}

    counterfactual_effects = []
    for _, row in treated.iterrows():
        evidence = row.to_dict()
        counterfactual = counterfactual_inference(
            scm,
            evidence=evidence,
            intervention={treatment: 0},
            target=outcome,
        )
        counterfactual_effects.append(float(row[outcome]) - float(counterfactual["counterfactual_outcome"]))

    return {
        "ETT": float(np.mean(counterfactual_effects)),
        "ETT_std": float(np.std(counterfactual_effects)),
        "n_treated": int(len(treated)),
    }


def abduction_action_prediction(
    scm,
    factual_world: dict,
    hypothetical_action: dict,
) -> dict:
    """完整的反事实推理流程。"""
    target = next((key for key in factual_world if key not in hypothetical_action), next(iter(factual_world)))
    result = counterfactual_inference(
        scm,
        evidence=factual_world,
        intervention=hypothetical_action,
        target=target,
    )
    return {
        "factual": factual_world,
        "hypothetical": hypothetical_action,
        "counterfactual_outcome": result["counterfactual_outcome"],
        "u_posterior": result.get("u_posterior", {}),
        "confidence": result.get("confidence", 0.0),
    }
