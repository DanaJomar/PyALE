import unittest
import os
import numpy as np
import pandas as pd

from PyALE._src.lib import cmds, order_groups, quantile_ied, CI_estimate


class TestlibFunctions(unittest.TestCase):
    def setUp(self):
        np.random.seed(876)
        x1 = np.random.uniform(low=0, high=1, size=200)
        x2 = np.random.uniform(low=0, high=1, size=200)
        x3 = np.random.uniform(low=0, high=1, size=200)
        x4 = np.random.choice(range(10), 200)
        x5 = (x1 + x2 + x3) * x4 - 1
        self.X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})
        self.X.loc[:, "x5"] = [
            "a" if x > 5 else "b" if x < 0 else "c" for x in self.X["x5"]
        ]
        np.random.seed(2301)
        self.dist_mat = np.random.rand(10, 10)

    def test_cmds(self):
        with self.assertRaises(Exception) as se:
            cmds(self.X)
        with self.assertRaises(Exception) as ke:
            cmds(self.dist_mat, k=10)
        cmds_res = cmds(self.dist_mat)

        self.assertEqual(se.exception.args[0], "The matrix D should be squared")
        self.assertEqual(
            ke.exception.args[0], "k should be an integer <= D.shape[0] - 1"
        )
        self.assertEqual(cmds_res.shape, (10, 2))
        self.assertCountEqual(np.round(cmds_res[0], 8), [-0.27720001, 0.30273103])
        self.assertCountEqual(
            np.round(cmds_res[:3, 0], 8), [-0.27720001, -0.12125997, 0.23333853],
        )

    def test_quantile_ied(self):
        quantile_res = quantile_ied(self.X["x1"], np.array([0.1, 0.5, 0.9]))
        self.assertCountEqual(
            np.round(quantile_res.values, 8), [0.10025329, 0.46653491, 0.88313677]
        )

    def test_CI_estimate(self):
        self.assertEqual(CI_estimate(self.X["x1"], 0.95), 0.03945173833592088)
        self.assertEqual(CI_estimate(self.X["x2"], 0.95), 0.03905859113262135)
        self.assertEqual(CI_estimate(self.X["x3"], 0.95), 0.04055720685482021)
        self.assertEqual(CI_estimate(self.X["x4"], 0.95), 0.406502818174431)

    def test_order_groups(self):
        order_groups_res = order_groups(self.X, "x5")
        self.assertEqual(order_groups_res["a"], 0)
        self.assertEqual(order_groups_res["c"], 1)
        self.assertEqual(order_groups_res["b"], 2)
