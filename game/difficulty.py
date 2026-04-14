"""
动态难度控制器
基于Flow理论，维持目标欺骗成功率在0.4附近
"""


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
        self.history: list[bool] = []  # True=欺骗成功

    def update(self, deception_succeeded: bool) -> float:
        """更新历史并返回新难度"""
        raise NotImplementedError

    def get_difficulty(self) -> float:
        """获取当前难度"""
        return self.current_difficulty
