import os
import numpy as np
import pandas as pd
import pickle
import unittest
from unittest.mock import patch
from sklearn.preprocessing import OneHotEncoder

from PyALE._ALE_generic import ale

def onehot_encode(feat):
    ohe = OneHotEncoder().fit(feat)
    col_names = ohe.categories_[0]
    feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
    feat_coded.columns = col_names
    return feat_coded
    
def onehot_encode_custom(feat, groups=['A', 'C', 'B']):
    feat_coded = onehot_encode(feat)
    missing_feat = [x for x in groups if x not in feat_coded.columns]
    if missing_feat:
        feat_coded[missing_feat] = 0
    return feat_coded


class Testale(unittest.TestCase):
    def setUp(self):
        path_to_fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
        with open(os.path.join(path_to_fixtures, "X.pickle"), "rb") as model_pickle:
            self.X = pickle.load(model_pickle)
        with open(
            os.path.join(path_to_fixtures, "X_cleaned.pickle"), "rb"
        ) as model_pickle:
            self.X_cleaned = pickle.load(model_pickle)
        with open(os.path.join(path_to_fixtures, "y.npy"), "rb") as y_npy:
            self.y = np.load(y_npy)
        with open(os.path.join(path_to_fixtures, "model.pickle"), "rb") as model_pickle:
            self.model = pickle.load(model_pickle)

    def test_exceptions(self):
        # X should be a data frame
        with self.assertRaises(Exception) as df_ex:
            ale(self.X_cleaned["x1"], self.model, ["x1"])
        # a python model should be passed (not the predict function)
        with self.assertRaises(Exception) as mod_ex_1:
            ale(self.X_cleaned, self.model.predict, ["x1"])
        # features as a list
        with self.assertRaises(Exception) as feat_ex_1:
            ale(self.X_cleaned, self.model, "x1")
        # at most two features
        with self.assertRaises(Exception) as feat_ex_2:
            ale(self.X_cleaned, self.model, ["x1", "x2", "x3"])
        # feature names as strings
        with self.assertRaises(Exception) as feat_ex_3:
            ale(self.X_cleaned, self.model, [1, 2])
        # all feature names should be in X.columns
        with self.assertRaises(Exception) as feat_ex_4:
            ale(self.X_cleaned, self.model, ["x1", "x6"])
        # feature_type
        with self.assertRaises(Exception) as feattyp_ex:
            ale(self.X_cleaned, self.model, ["x1"], feature_type="a")
        # C (the level of confidence interval)
        with self.assertRaises(Exception) as c_ex:
            ale(self.X_cleaned, self.model, ["x1"], include_CI=True, C=95)
        # categorical featrue without predictors
        with self.assertRaises(Exception) as cat_ex_1:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x5"],
                encode_fun=onehot_encode_custom,
            )
        # categorical featrue without encoding fucntion
        with self.assertRaises(Exception) as cat_ex_2:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x5"],
                predictors=self.X_cleaned.columns,
            )

        df_ex_msg = "The arguemnt 'X' must be a pandas DataFrame"
        self.assertEqual(df_ex.exception.args[0], df_ex_msg)

        mod_ex_msg = "The passed model does not seem to have a predict method."
        self.assertEqual(mod_ex_1.exception.args[0], mod_ex_msg)

        feat_ex_msg = (
            "The arguemnt 'feature' must be a list of at most two feature"
            " names (strings)"
        )
        self.assertEqual(feat_ex_1.exception.args[0], feat_ex_msg)
        self.assertEqual(feat_ex_2.exception.args[0], feat_ex_msg)
        self.assertEqual(feat_ex_3.exception.args[0], feat_ex_msg)

        feat_ex_4_msg = "Feature(s) ['x6'] was(were) not found in the column names of X"
        self.assertEqual(feat_ex_4.exception.args[0], feat_ex_4_msg)

        feattyp_ex_msg = (
            "The argument 'feature_type' should be 'auto', 'continuous', "
            "'discrete', or 'categorical'"
        )
        self.assertEqual(feattyp_ex.exception.args[0], feattyp_ex_msg)

        c_ex_msg = (
            "The argument 'C' (confidence level) should be a value between 0 and 1"
        )
        self.assertEqual(c_ex.exception.args[0], c_ex_msg)
        
        cat_ex_1_msg = (
                    "Argument 'predictors' not given. With categorical/string "
                    "features, a list of predictors (column names) should be provided."
                )
        self.assertEqual(cat_ex_1.exception.args[0], cat_ex_1_msg)
        cat_ex_2_msg = (
                    "Argument 'encode_fun' not given. With categorical/string "
                    "features, an encoding function should be provided."
                )
        self.assertEqual(cat_ex_2.exception.args[0], cat_ex_2_msg)

    def test_auto_calls_1D_continuous(self):
        with patch("PyALE._ALE_generic.aleplot_1D_continuous") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x1"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X_cleaned,
                model=self.model,
                feature="x1",
                grid_size=5,
                include_CI=True,
                C=0.95,
            )

    def test_auto_calls_1D_discrete(self):
        with patch("PyALE._ALE_generic.aleplot_1D_discrete") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x4"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X_cleaned,
                model=self.model,
                feature="x4",
                include_CI=True,
                C=0.95,
            )

    def test_contin_calls_1D_continuous(self):
        with patch("PyALE._ALE_generic.aleplot_1D_continuous") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x4"],
                feature_type="continuous",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X_cleaned,
                model=self.model,
                feature="x4",
                grid_size=5,
                include_CI=True,
                C=0.95,
            )

    def test_discr_calls_1D_discrete(self):
        with patch("PyALE._ALE_generic.aleplot_1D_discrete") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x1"],
                feature_type="discrete",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X_cleaned,
                model=self.model,
                feature="x1",
                include_CI=True,
                C=0.95,
            )

    def test_2D_continuous_called(self):
        with patch("PyALE._ALE_generic.aleplot_2D_continuous") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x1", "x2"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=False,
            )
            mock.assert_called_once_with(
                X=self.X_cleaned, model=self.model, features=["x1", "x2"], grid_size=5,
            )

    def test_1D_continuous_plot_called(self):
        with patch("PyALE._ALE_generic.plot_1D_continuous_eff") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x1"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, X=self.X_cleaned, fig=None, ax=None)

    def test_1D_discrete_plot_called(self):
        with patch("PyALE._ALE_generic.plot_1D_discrete_eff") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x4"],
                feature_type="auto",
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, X=self.X_cleaned, fig=None, ax=None)
        
        with patch("PyALE._ALE_generic.plot_1D_discrete_eff") as mock:
            result = ale(
                X=self.X,
                model=self.model,
                feature=["x5"],
                feature_type="auto",
                predictors=self.X_cleaned.columns,
                encode_fun=onehot_encode_custom,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, X=self.X, fig=None, ax=None)

    def test_2D_continuous_plot_called(self):
        with patch("PyALE._ALE_generic.plot_2D_continuous_eff") as mock:
            result = ale(
                X=self.X_cleaned,
                model=self.model,
                feature=["x1", "x2"],
                grid_size=5,
                include_CI=True,
                plot=True,
            )
            mock.assert_called_once_with(result, contour=False, fig=None, ax=None)


if __name__ == "__main__":
    unittest.main()
