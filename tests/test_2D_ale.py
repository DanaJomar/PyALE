import unittest
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from PyALE._src.ALE_2D import aleplot_2D_continuous, plot_2D_continuous_eff


class Test2DFunctions(unittest.TestCase):
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
        with open(os.path.join(path_to_fixtures, "model.pickle"), "rb") as model_pickle:
            self.model = pickle.load(model_pickle)

        self.ale_eff = aleplot_2D_continuous(
            X=self.X_cleaned, model=self.model, features=["x1", "x2"], grid_size=5
        )
        self.ale_eff_g50 = aleplot_2D_continuous(
            X=self.X_cleaned, model=self.model, features=["x1", "x2"], grid_size=50
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
            [0.18511021, -0.12188315, -0.1852733, -0.1888011, 0.2319353, 0.24136658],
        )
        self.assertCountEqual(
            np.round(self.ale_eff.iloc[:, 0], 8),
            [0.18511021, -0.0496302, 0.03282839, 0.01555219, -0.10290468, -0.12306892],
        )

    def test_effvalues_g50(self):
        self.assertCountEqual(
            np.round(self.ale_eff_g50.iloc[0, 10:20], 8),
            [
                -0.51806894,
                -0.51176885,
                -0.4948861,
                -0.4788409,
                -0.43635714,
                -0.41986254,
                -0.37063614,
                -0.30517262,
                -0.28153319,
                -0.25889381,
            ],
        )
        self.assertCountEqual(
            np.round(self.ale_eff_g50.iloc[10:20, 0], 8),
            [
                -0.36506129,
                -0.39531017,
                -0.38535947,
                -0.38730497,
                -0.32949893,
                -0.24607249,
                -0.21400247,
                -0.20564522,
                -0.15082074,
                -0.14580436,
            ],
        )

    def test_2D_continuous_grid_plot(self):
        eff_grid = aleplot_2D_continuous(
            self.X_cleaned, self.model, ["x1", "x2"], grid_size=10
        )
        fig, ax = plot_2D_continuous_eff(eff_grid, contour=False)
        # values should equal the returned effects
        self.assertTrue(np.all(np.array(eff_grid) == ax.images[0].get_array()))
