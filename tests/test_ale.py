import os
import numpy as np
import pandas as pd
import pickle
import unittest
from unittest.mock import patch

from PyALE._ALE_generic import ale


class Testale(unittest.TestCase):
    def setUp(self):
        path_to_fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
        np.random.seed(876)
        x1 = np.random.uniform(low=0, high=1, size=200)
        x2 = np.random.uniform(low=0, high=1, size=200)
        x3 = np.random.uniform(low=0, high=1, size=200)
        x4 = np.random.choice(range(10), 200)
        self.y = x1 + 2 * x2 ** 2 + np.log(x4 + 1) + np.random.normal(size=200)
        self.X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        with open(os.path.join(path_to_fixtures, "model.pickle"), "rb") as model_pickle:
            self.model = pickle.load(model_pickle)

    def test_exceptions(self):
        # X should be a data frame
        with self.assertRaises(Exception) as df_ex:
            ale(self.X["x1"], self.model, ["x1"])
        # a python model should be passed (not the predict function)
        with self.assertRaises(Exception) as mod_ex_1:
            ale(self.X, self.model.predict, ["x1"])
        # dataset sould be compatible with the model
        with self.assertRaises(Exception) as mod_ex_2:
            ale(self.X[["x1"]], self.model, ["x1"])
        # features as a list
        with self.assertRaises(Exception) as feat_ex_1:
            ale(self.X, self.model, "x1")
        # at most two features
        with self.assertRaises(Exception) as feat_ex_2:
            ale(self.X, self.model, ["x1", "x2", "x3"])
        # feature names as strings
        with self.assertRaises(Exception) as feat_ex_3:
            ale(self.X, self.model, [1, 2])
        # all feature names should be in X.columns
        with self.assertRaises(Exception) as feat_ex_4:
            ale(self.X, self.model, ["x1", "x6"])
        # feature_type
        with self.assertRaises(Exception) as feattyp_ex:
            ale(self.X, self.model, ["x1"], feature_type="a")
        # C (the level of confidence interval)
        with self.assertRaises(Exception) as c_ex:
            ale(self.X, self.model, ["x1"], include_CI=True, C=95)

        df_ex_msg = "The arguemnt 'X' must be a pandas DataFrame"
        self.assertEqual(df_ex.exception.args[0], df_ex_msg)

        mod_ex_msg = """
        The argument 'model' should be a python model with a predict method 
        that accepts X as input
        """
        self.assertEqual(mod_ex_1.exception.args[0], mod_ex_msg)
        self.assertEqual(mod_ex_2.exception.args[0], mod_ex_msg)

        feat_ex_msg = "The arguemnt 'feature' must be a list of at most two feature names (strings)"
        self.assertEqual(feat_ex_1.exception.args[0], feat_ex_msg)
        self.assertEqual(feat_ex_2.exception.args[0], feat_ex_msg)
        self.assertEqual(feat_ex_3.exception.args[0], feat_ex_msg)

        feat_ex_4_msg = "Feature(s) ['x6'] was(were) not found in the column names of X"
        self.assertEqual(feat_ex_4.exception.args[0], feat_ex_4_msg)

        feattyp_ex_msg = (
            "The argument 'feature_type' should be 'auto', 'continuous', or 'discrete'"
        )
        self.assertEqual(feattyp_ex.exception.args[0], feattyp_ex_msg)

        c_ex_msg = (
            "The argument 'C' (confidence level) should be a value between 0 and 1"
        )
        self.assertEqual(c_ex.exception.args[0], c_ex_msg)

    def test_auto_calls_1D_continuous(self):
        with patch("PyALE._ALE_generic.aleplot_1D_continuous") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x1"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X,
                model=self.model,
                feature="x1",
                grid_size=5,
                include_CI=True,
                C=0.95,
            )

    def test_auto_calls_1D_discrete(self):
        with patch("PyALE._ALE_generic.aleplot_1D_discrete") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x4"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X, model=self.model, feature="x4", include_CI=True, C=0.95
            )

    def test_contin_calls_1D_continuous(self):
        with patch("PyALE._ALE_generic.aleplot_1D_continuous") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x4"],
                feature_type="continuous",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X,
                model=self.model,
                feature="x4",
                grid_size=5,
                include_CI=True,
                C=0.95,
            )

    def test_discr_calls_1D_discrete(self):
        with patch("PyALE._ALE_generic.aleplot_1D_discrete") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x1"],
                feature_type="discrete",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X, model=self.model, feature="x1", include_CI=True, C=0.95
            )

    def test_2D_continuous_called(self):
        with patch("PyALE._ALE_generic.aleplot_2D_continuous") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x1", "x2"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X, model=self.model, features=["x1", "x2"], grid_size=5,
            )

    def test_1D_continuous_plot_called(self):
        with patch("PyALE._ALE_generic.plot_1D_continuous_eff") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x1"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, X=self.X, fig=None, ax=None)

    def test_1D_discrete_plot_called(self):
        with patch("PyALE._ALE_generic.plot_1D_discrete_eff") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x4"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, X=self.X, fig=None, ax=None)

    def test_2D_continuous_plot_called(self):
        with patch("PyALE._ALE_generic.plot_2D_continuous_eff") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x1", "x2"],
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, contour=False, fig=None, ax=None)


if __name__ == "__main__":
    unittest.main()
