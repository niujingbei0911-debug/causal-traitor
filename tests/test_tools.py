import unittest

import networkx as nx
import numpy as np
import pandas as pd

from causal_tools.l1_association import (
    compute_correlation,
    conditional_independence_test,
    correlation_analysis,
    detect_simpson_paradox,
)
from causal_tools.l2_intervention import (
    backdoor_adjustment_check,
    iv_estimation,
    propensity_score_matching,
    validate_backdoor_criterion,
)
from causal_tools.l3_counterfactual import (
    abduction_action_prediction,
    counterfactual_inference,
    ett_computation,
    probability_of_necessity,
    probability_of_sufficiency,
    scm_identification_test,
)
from causal_tools.meta_tools import ToolSelector, argument_logic_check, causal_graph_validator, select_tools


class FakeSCM:
    name = "fake_scm"

    def __init__(self, effect=1.0):
        self.effect = effect
        self.graph = nx.DiGraph([("X", "Y")])

    def abduction(self, evidence):
        return {"baseline": evidence["Y"] - self.effect * evidence["X"]}

    def do(self, intervention):
        return FakeIntervenedModel(self.effect, intervention)


class FakeIntervenedModel:
    def __init__(self, effect, intervention):
        self.effect = effect
        self.intervention = intervention

    def predict(self, u_posterior):
        x_value = self.intervention["X"]
        return {"Y": u_posterior["baseline"] + self.effect * x_value}


class ToolTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 400
        u = rng.normal(size=n)
        z = rng.integers(0, 2, size=n)
        x = 0.9 * z + 0.8 * u + rng.normal(scale=0.3, size=n)
        y = 2.0 * x + 1.0 * u + rng.normal(scale=0.3, size=n)
        mediator = 0.7 * x + rng.normal(scale=0.2, size=n)
        outcome_binary = (y > np.median(y)).astype(int)
        treatment_binary = (x > np.median(x)).astype(int)

        self.df = pd.DataFrame(
            {
                "Z": z,
                "U": u,
                "X": x,
                "M": mediator,
                "Y": y,
                "T": treatment_binary,
                "Y_bin": outcome_binary,
            }
        )
        self.graph = nx.DiGraph([("U", "X"), ("U", "Y"), ("Z", "X"), ("X", "M"), ("M", "Y"), ("X", "Y")])

    def test_correlation_and_conditional_independence(self):
        result = compute_correlation(self.df, "X", "Y")
        self.assertIn("pearson_r", result)
        self.assertTrue(result["significant"])
        self.assertAlmostEqual(result["pearson_r"], correlation_analysis(self.df, "X", "Y")["pearson_r"])

        ci_df = pd.DataFrame(
            {
                "U": self.df["U"],
                "X": self.df["U"] + np.random.default_rng(1).normal(scale=0.1, size=len(self.df)),
                "Y": self.df["U"] + np.random.default_rng(2).normal(scale=0.1, size=len(self.df)),
            }
        )
        ci_result = conditional_independence_test(ci_df, "X", "Y", ["U"])
        self.assertTrue(ci_result["independent"])

    def test_simpson_paradox_detector(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 8, 9, 10],
                "y": [1, 2, 3, -2, -1, 0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        result = detect_simpson_paradox(df, "x", "y", "group")
        self.assertTrue(result["simpson_paradox_detected"])

    def test_backdoor_and_iv_tools(self):
        self.assertTrue(validate_backdoor_criterion(self.graph, "X", "Y", ["U"]))
        self.assertFalse(validate_backdoor_criterion(self.graph, "X", "Y", ["Y"]))

        adjusted = backdoor_adjustment_check(self.df, "X", "Y", ["U"], graph=self.graph)
        self.assertIn("estimated_effect", adjusted)

        iv = iv_estimation(self.df, "Z", "X", "Y", ["U"])
        self.assertIn("causal_effect", iv)
        self.assertGreater(iv["first_stage_f"], 0)

    def test_matching_and_meta_tools(self):
        psm = propensity_score_matching(self.df, "T", "Y", ["U", "Z"])
        self.assertIn("att", psm)

        selector = ToolSelector()
        selected = selector.select(2, scenario_type="instrument", claim="This IV is weak", context={"has_instrument": True})
        self.assertIn("iv_estimation", selected)
        self.assertIn("backdoor_adjustment_check", select_tools(3, {"claim": "反事实", "needs_full_counterfactual": True}))

        logic = argument_logic_check("相关就说明因果，而且这是唯一解释")
        self.assertGreaterEqual(logic["n_fallacies_detected"], 1)
        graph_report = causal_graph_validator(self.graph)
        self.assertTrue(graph_report["is_dag"])

    def test_counterfactual_suite(self):
        model = FakeSCM(effect=1.5)
        evidence = {"X": 1, "Y": 3.0}
        cf = counterfactual_inference(model, evidence, {"X": 0}, "Y")
        self.assertAlmostEqual(cf["counterfactual_outcome"], 1.5, places=3)

        aapp = abduction_action_prediction(model, evidence, {"X": 0})
        self.assertIn("counterfactual_outcome", aapp)

        binary_df = self.df[["T", "Y_bin"]].rename(columns={"T": "X", "Y_bin": "Y"})
        pn = probability_of_necessity(binary_df, "X", "Y")
        ps = probability_of_sufficiency(binary_df, "X", "Y")
        self.assertGreaterEqual(pn, 0.0)
        self.assertLessEqual(ps, 1.0)

        ett = ett_computation(pd.DataFrame({"X": [1, 1, 0], "Y": [3.0, 2.5, 1.0]}), model, "X", "Y")
        self.assertEqual(ett["n_treated"], 2)

        alt = FakeSCM(effect=0.5)
        comparison_df = pd.DataFrame({"X": [0, 1], "Y": [1.0, 2.0]})
        identifiability = scm_identification_test(comparison_df, model, [alt])
        self.assertEqual(len(identifiability), 1)


if __name__ == "__main__":
    unittest.main()
