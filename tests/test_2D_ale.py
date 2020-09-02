import unittest
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from PyALE.src.ALE_2D import aleplot_2D_continuous


class Test2DFunctions(unittest.TestCase):
    def setUp(self):
        path_to_fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
        with open(os.path.join(path_to_fixtures, "X.pickle"), "rb") as X_pickle:
            self.X = pickle.load(X_pickle)
        with open(os.path.join(path_to_fixtures, "y.npy"), "rb") as y_npy:
            self.y = np.load(y_npy)
        with open(os.path.join(path_to_fixtures, "model.pickle"), "rb") as model_pickle:
            self.model = pickle.load(model_pickle)
        self.ale_eff = aleplot_2D_continuous(
            X=self.X, model=self.model, features=["x1", "x2"], grid_size=5
        )

    def test_indexnames(self):
        self.assertEqual(self.ale_eff.index.name, "x1")
        self.assertEqual(self.ale_eff.columns.name, "x2")

    def test_outputshape(self):
        self.assertEqual(self.ale_eff.shape, (6, 6))

    def test_bins(self):
        self.assertCountEqual(
            self.ale_eff.index,
            [
                0.0013107121819164735,
                0.21205399821897986,
                0.3905585553320686,
                0.5561380185409515,
                0.7797798975036754,
                0.9986526271693825,
            ],
        )
        self.assertCountEqual(
            self.ale_eff.columns,
            [
                0.0031787396802746004,
                0.1947247502687668,
                0.3338438691890313,
                0.5475686771925931,
                0.7514716438352422,
                0.9856548283501907,
            ],
        )

    def test_effvalues(self):
        self.assertCountEqual(
            np.round(self.ale_eff.iloc[0, :], 8),
            [-0.15302919, -0.12544567,  0.07381934,  0.09041414,  0.17723169, -0.19949057])
        self.assertCountEqual(
            np.round(self.ale_eff.iloc[:, 0], 8),
            [-0.15302919, -0.22718564,  0.17506918, -0.00255802,  0.15903598, -0.06443989])