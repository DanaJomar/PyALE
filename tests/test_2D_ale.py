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
        np.random.seed(876)
        x1 = np.random.uniform(low=0, high=1, size=200)
        x2 = np.random.uniform(low=0, high=1, size=200)
        x3 = np.random.uniform(low=0, high=1, size=200)
        x4 = np.random.choice(range(10), 200)
        self.y = x1 + 2 * x2 ** 2 + np.log(x4 + 1) + np.random.normal(size=200)
        self.X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

        with open(os.path.join(path_to_fixtures, "model.pickle"), "rb") as model_pickle:
            self.model = pickle.load(model_pickle)
        self.ale_eff = aleplot_2D_continuous(
            X=self.X, model=self.model, features=["x1", "x2"], grid_size=5
        )
        self.ale_eff_g50 = aleplot_2D_continuous(
            X=self.X, model=self.model, features=["x1", "x2"], grid_size=50
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
            [-0.15302919, -0.12544567, 0.07381934, 0.09041414, 0.17723169, -0.19949057],
        )
        self.assertCountEqual(
            np.round(self.ale_eff.iloc[:, 0], 8),
            [
                -0.15302919,
                -0.22718564,
                0.17506918,
                -0.00255802,
                0.15903598,
                -0.06443989,
            ],
        )

    def test_effvalues_g50(self):
        self.assertCountEqual(
            np.round(self.ale_eff_g50.iloc[0, 10:20], 8),
            [
                -0.79777388,
                -0.72708313,
                -0.63036386,
                -0.59708803,
                -0.46666037,
                -0.31568099,
                -0.17738274,
                0.02443929,
                0.12606745,
                0.1965112,
            ],
        )
        self.assertCountEqual(
            np.round(self.ale_eff_g50.iloc[10:20, 0], 8),
            [
                -0.46167797,
                -0.65590743,
                -0.72626874,
                -0.73384303,
                -0.76019481,
                -0.72057576,
                -0.60993731,
                -0.54754028,
                -0.43078769,
                -0.39158206,
            ],
        )

    def test_2D_continuous_grid_plot(self):
        eff_grid = aleplot_2D_continuous(self.X, self.model, ['x1', 'x2'], grid_size=10)
        fig, ax = plot_2D_continuous_eff(eff_grid, contour=False)
        # values should equal the returned effects
        self.assertTrue(np.all(np.array(eff_grid)==ax.images[0].get_array()))