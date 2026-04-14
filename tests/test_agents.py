import unittest

import pandas as pd

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.jury import JuryAggregator
from game.debate_engine import CausalScenario, DebateContext


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "models": {
                "agent_a": {"name": "mock-a"},
                "agent_b": {"name": "mock-b"},
                "jury": {
                    "models": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
                    "voting": "weighted",
                },
            }
        }
        self.data = pd.DataFrame(
            {
                "X": [0, 0, 1, 1, 0, 1],
                "Z": [0, 1, 0, 1, 0, 1],
                "Y": [0.1, 0.5, 1.1, 1.6, 0.0, 1.4],
            }
        )
        self.scenario = CausalScenario(
            scenario_id="education_income",
            description="教育影响收入",
            true_dag={"Z": ["X"], "X": ["Y"]},
            variables=["X", "Z", "Y"],
            hidden_variables=["U"],
            data=self.data,
            causal_level=2,
            difficulty=0.5,
        )

    async def test_agent_a_generates_deception(self):
        agent = AgentA(self.config)
        await agent.initialize()
        response = await agent.generate_deception(self.scenario, level=2)
        self.assertTrue(response.content)
        self.assertTrue(response.causal_claim)
        self.assertTrue(response.deception_strategy.startswith("L2"))

    async def test_agent_b_analyzes_claim(self):
        agent = AgentB(self.config)
        await agent.initialize()
        result = await agent.analyze_claim(
            claim="X 对 Y 的工具变量效应已经被 Z 证明，无需控制其他因素。",
            scenario=self.scenario,
            level=2,
        )
        self.assertGreater(result.confidence, 0.0)
        self.assertTrue(result.tools_used)
        self.assertTrue(result.reasoning_chain)

    async def test_jury_collects_votes(self):
        jury = JuryAggregator(self.config)
        await jury.initialize()
        context = DebateContext(
            scenario=self.scenario,
            turns=[
                {"speaker": "agent_a", "content": "X 必然导致 Y。"},
                {"speaker": "agent_b", "content": "我检查了混杂和工具变量强度，结论并不稳固。"},
            ],
        )
        verdict = await jury.collect_votes(self.scenario, context)
        self.assertEqual(len(verdict.votes), 3)
        self.assertIn(verdict.final_winner, {"agent_a", "agent_b", "draw"})


if __name__ == "__main__":
    unittest.main()
