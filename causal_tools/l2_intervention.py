"""
L2 干预层因果工具
Pearl因果阶梯第二层：P(Y|do(X))
"""
import pandas as pd
import networkx as nx


def backdoor_adjustment(
    data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str
) -> dict:
    """后门调整估计因果效应"""
    raise NotImplementedError


def frontdoor_adjustment(
    data: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str, mediator: str
) -> dict:
    """前门调整估计因果效应"""
    raise NotImplementedError


def iv_estimation(
    data: pd.DataFrame, instrument: str, treatment: str, outcome: str
) -> dict:
    """工具变量估计（2SLS）"""
    raise NotImplementedError


def propensity_score_matching(
    data: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]
) -> dict:
    """倾向得分匹配"""
    raise NotImplementedError


def validate_backdoor_criterion(
    graph: nx.DiGraph, treatment: str, outcome: str, adjustment_set: list[str]
) -> bool:
    """验证后门准则是否满足"""
    raise NotImplementedError
