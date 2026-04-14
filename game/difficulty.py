"""
动态难度控制器
基于Flow理论，维持目标欺骗成功率在0.4附近
"""

from __future__ import annotations

from typing import Any


class DifficultyController:
    """
    动态难度调整器
    
    原理：监控最近N轮的欺骗成功率，
    若偏离目标值则调整场景难度参数
    """

    def __init__(self, config: dict):
        self.target_rate = config.get("target_deception_rate", 0.4)
        self.window_size = config.get("window_size", 10)
        self.adjustment_rate = config.get("adjustment_rate", 0.1)
        self.min_difficulty = config.get("min_difficulty", 0.2)
        self.max_difficulty = config.get("max_difficulty", 0.95)
        self.current_difficulty = config.get("initial_difficulty", 0.5)
        self.tolerance = config.get("tolerance", 0.1)
        self.history: list[bool] = []  # True=欺骗成功
        self.adjustment_log: list[dict[str, Any]] = []

    def update(self, deception_succeeded: bool) -> float:
        """更新历史并返回新难度"""
        self.history.append(bool(deception_succeeded))
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size :]

        if len(self.history) < self.window_size:
            self.adjustment_log.append(
                {
                    "reason": "warming_up",
                    "deception_rate": None,
                    "new_difficulty": self.current_difficulty,
                }
            )
            return self.current_difficulty

        deception_rate = sum(self.history) / len(self.history)
        delta = max(0.0, abs(deception_rate - self.target_rate))
        scaled_step = self.adjustment_rate * max(0.25, delta)
        reason = "within_tolerance"

        if deception_rate > self.target_rate + self.tolerance:
            self.current_difficulty = min(
                self.max_difficulty,
                self.current_difficulty + scaled_step,
            )
            reason = "increase_difficulty"
        elif deception_rate < self.target_rate - self.tolerance:
            self.current_difficulty = max(
                self.min_difficulty,
                self.current_difficulty - scaled_step,
            )
            reason = "decrease_difficulty"

        self.adjustment_log.append(
            {
                "reason": reason,
                "deception_rate": deception_rate,
                "new_difficulty": self.current_difficulty,
            }
        )
        return self.current_difficulty

    def get_difficulty(self) -> float:
        """获取当前难度"""
        return self.current_difficulty

    def get_config(self) -> dict[str, Any]:
        """Map the scalar difficulty into execution settings."""

        difficulty = self.current_difficulty
        return {
            "difficulty": difficulty,
            "n_hidden_vars": 1 if difficulty < 0.45 else 2 if difficulty < 0.75 else 3,
            "noise_scale": round(0.15 + 0.25 * difficulty, 4),
            "confounding_strength": round(0.4 + 0.5 * difficulty, 4),
            "debate_rounds": 3 if difficulty < 0.6 else 4 if difficulty < 0.8 else 5,
            "n_jurors": 3 if difficulty < 0.8 else 5,
            "agent_c_tool_budget": 2 if difficulty < 0.45 else 4 if difficulty < 0.75 else 6,
        }

    def get_recent_rate(self) -> float | None:
        if not self.history:
            return None
        return sum(self.history) / len(self.history)
