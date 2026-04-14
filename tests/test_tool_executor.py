import unittest

import pandas as pd

from agents.tool_executor import ToolExecutor
from game.debate_engine import CausalScenario


class ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = ToolExecutor({})
        self.data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1, 0, 1, 0, 1],
                "Z": [0, 1, 0, 1, 0, 1, 0, 1],
                "M": [0.1, 0.5, 0.7, 1.0, 0.2, 0.8, 0.2, 0.9],
                "Y": [0.0, 0.2, 1.0, 1.2, 0.1, 1.1, 0.2, 1.3],
            }
        )
        self.scenario = CausalScenario(
            scenario_id="tool_executor_case",
            description="测试工具执行器",
            true_dag={"Z": ["X"], "X": ["M", "Y"], "M": ["Y"]},
            variables=["X", "Z", "M", "Y"],
            hidden_variables=["U"],
            data=self.data,
            causal_level=2,
            difficulty=0.4,
        )

    def test_execute_for_claim(self):
        report = self.executor.execute_for_claim(
            scenario=self.scenario,
            claim="X 对 Y 的 IV 估计已经证明因果效应，而且还有中介 M。",
            level=2,
            context={"has_instrument": True, "has_mediator": True},
        )
        self.assertIn("iv_estimation", report["selected_tools"])
        self.assertIn("frontdoor_estimation", report["selected_tools"])
        self.assertTrue(report["results"])
        self.assertTrue(report["successful_tools"])

    def test_safe_python_executor(self):
        result = self.executor.execute_python(
            "answer = sum(values)\nmean_value = round(answer / len(values), 2)",
            extra_context={"values": [1, 2, 3]},
        )
        self.assertEqual(result["answer"], 6)
        self.assertEqual(result["mean_value"], 2.0)

        with self.assertRaises(ValueError):
            self.executor.execute_python("import os\nx = 1")


if __name__ == "__main__":
    unittest.main()
