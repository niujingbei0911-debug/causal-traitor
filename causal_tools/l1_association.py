"""
L1 关联层因果工具
Pearl因果阶梯第一层：P(Y|X)
"""
import numpy as np
import pandas as pd


def compute_correlation(data: pd.DataFrame, x: str, y: str) -> dict:
    """计算变量间相关系数"""
    raise NotImplementedError


def conditional_independence_test(
    data: pd.DataFrame, x: str, y: str, z: list[str]
) -> dict:
    """条件独立性检验 (Fisher-z / Chi-square)"""
    raise NotImplementedError


def detect_simpson_paradox(
    data: pd.DataFrame, x: str, y: str, z: str
) -> dict:
    """检测辛普森悖论"""
    raise NotImplementedError


def partial_correlation(
    data: pd.DataFrame, x: str, y: str, controls: list[str]
) -> dict:
    """偏相关分析"""
    raise NotImplementedError
