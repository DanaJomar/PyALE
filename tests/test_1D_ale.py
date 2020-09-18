import unittest
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from PyALE._src.ALE_1D import (
    aleplot_1D_continuous,
    plot_1D_continuous_eff,
    aleplot_1D_discrete,
    plot_1D_discrete_eff,
)


class Test1DFunctions(unittest.TestCase):
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


class Test1DContinuous(Test1DFunctions):
    def test_indexname(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=False
        )
        self.assertEqual(ale_eff.index.name, "x1")

    def test_outputshape_noCI(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=False
        )
        self.assertEqual(ale_eff.shape, (6, 2))
        self.assertCountEqual(ale_eff.columns, ["eff", "size"])

    def test_outputshape_withCI(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X,
            model=self.model,
            feature="x1",
            grid_size=5,
            include_CI=True,
            C=0.9,
        )
        self.assertEqual(ale_eff.shape, (6, 4))
        self.assertCountEqual(
            ale_eff.columns, ["eff", "size", "lowerCI_90%", "upperCI_90%"]
        )

    def test_bins(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=False
        )
        self.assertCountEqual(
            ale_eff.index,
            [
                0.0013107121819164735,
                0.21205399821897986,
                0.3905585553320686,
                0.5561380185409515,
                0.7797798975036754,
                0.9986526271693825,
            ],
        )

    def test_effvalues(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=False
        )
        self.assertCountEqual(
            np.round(ale_eff.loc[:, "eff"], 8),
            [-0.35570033, -0.16996644, -0.19291121, 0.10414799, 0.24730329, 0.37855307],
        )

    def test_binsizes(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=False
        )
        self.assertCountEqual(
            ale_eff.loc[:, "size"], [0.0, 40.0, 40.0, 40.0, 40.0, 40.0]
        )

    def test_CIvalues(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X,
            model=self.model,
            feature="x1",
            grid_size=5,
            include_CI=True,
            C=0.9,
        )
        # assert that the first bin do not have a CI
        self.assertTrue(np.isnan(ale_eff.loc[ale_eff.index[0], "lowerCI_90%"]))
        self.assertTrue(np.isnan(ale_eff.loc[ale_eff.index[0], "upperCI_90%"]))
        # check the values of the CI
        self.assertCountEqual(
            np.round(ale_eff.loc[ale_eff.index[1] :, "lowerCI_90%"], 8),
            [-0.21966029, -0.27471201, -0.01534647, 0.20038572, 0.30378132],
        )
        self.assertCountEqual(
            np.round(ale_eff.loc[ale_eff.index[1] :, "upperCI_90%"], 8),
            [-0.12027259, -0.11111041, 0.22364245, 0.29422086, 0.45332483],
        )


class Test1Ddiscrete(Test1DFunctions):
    def test_indexname(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=False
        )
        self.assertEqual(ale_eff.index.name, "x4")

    def test_outputshape_noCI(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=False
        )
        self.assertEqual(ale_eff.shape, (10, 2))
        self.assertCountEqual(ale_eff.columns, ["eff", "size"])

    def test_outputshape_withCI(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=True, C=0.9
        )
        self.assertEqual(ale_eff.shape, (10, 4))
        self.assertCountEqual(
            ale_eff.columns, ["eff", "size", "lowerCI_90%", "upperCI_90%"]
        )

    def test_bins(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=False
        )
        self.assertCountEqual(
            ale_eff.index, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

    def test_effvalues(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=False
        )
        self.assertCountEqual(
            np.round(ale_eff.loc[:, "eff"], 8),
            [
                -1.20935606,
                -0.82901158,
                -0.42415507,
                -0.24192617,
                0.04098572,
                0.32370623,
                0.56468117,
                0.61378063,
                0.6786663,
                0.69330051,
            ],
        )

    def test_binsizes(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=False
        )
        self.assertCountEqual(
            ale_eff.loc[:, "size"], [27, 14, 16, 26, 20, 17, 18, 23, 21, 18]
        )

    def test_CIvalues(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=True, C=0.9
        )
        # assert that the first bin do not have a CI
        self.assertTrue(np.isnan(ale_eff.loc[ale_eff.index[0], "lowerCI_90%"]))
        self.assertTrue(np.isnan(ale_eff.loc[ale_eff.index[0], "upperCI_90%"]))
        # check the values of the CI
        self.assertCountEqual(
            np.round(ale_eff.loc[ale_eff.index[1] :, "lowerCI_90%"], 8),
            [
                -0.91916875,
                -0.54755861,
                -0.29067159,
                -0.04528913,
                0.2512252,
                0.48115024,
                0.58654446,
                0.65270079,
                0.66807024,
            ],
        )
        self.assertCountEqual(
            np.round(ale_eff.loc[ale_eff.index[1] :, "upperCI_90%"], 8),
            [
                -0.73885441,
                -0.30075154,
                -0.19318075,
                0.12726056,
                0.39618726,
                0.6482121,
                0.64101679,
                0.70463181,
                0.71853079,
            ],
        )


class TestContPlottingFun(Test1DFunctions):
    def test_1D_continuous_line_plot(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=True
        )
        fig, ax = plot_1D_continuous_eff(ale_eff, self.X)
        ## effect line
        eff_plt_data = ax.lines[0].get_xydata()
        # the x values should be the bins
        self.assertCountEqual(eff_plt_data[:, 0], ale_eff.index)
        # the y values should be the effect
        self.assertCountEqual(eff_plt_data[:, 1], ale_eff.eff)

    def test_1D_continuous_rug_plot(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=True
        )
        fig, ax = plot_1D_continuous_eff(ale_eff, self.X)        
        ## the rug
        rug_plot_data = ax.lines[1].get_xydata()
        # a line for each data point in X
        self.assertEqual(rug_plot_data.shape[0], self.X.shape[0])
        # y position is always at the lower eff value
        self.assertCountEqual(np.unique(rug_plot_data[:, 1]), [ale_eff.eff.min()])
        # x position should always be plotted within the bin it belongs to
        # (less than the upper bin limit and more than the lower bin limit)
        self.assertTrue(
            np.all(
                ale_eff.index[
                    pd.cut(self.X["x1"], ale_eff.index, include_lowest=True).cat.codes + 1
                ]
                > rug_plot_data[:, 0]
            )
            and np.all(
                ale_eff.index[
                    pd.cut(self.X["x1"], ale_eff.index, include_lowest=True).cat.codes
                ]
                < rug_plot_data[:, 0]
            )
        )
    def test_1D_continuous_ci_plot(self):
        ale_eff = aleplot_1D_continuous(
            X=self.X, model=self.model, feature="x1", grid_size=5, include_CI=True
        )
        fig, ax = plot_1D_continuous_eff(ale_eff, self.X)
        ci_plot_data = pd.DataFrame(ax.collections[0].get_paths()[0].vertices).drop_duplicates().groupby(0).agg(['min', 'max'])
        ci_plot_data.index.name = 'x1'
        ci_plot_data.columns = ['lowerCI_95%', 'upperCI_95%']
        self.assertTrue(np.all(ale_eff.loc[ale_eff.index[1]:, ['lowerCI_95%', 'upperCI_95%']] == ci_plot_data))

class TestDiscPlottingFun(Test1DFunctions):
    def test_1D_continuous_line_plot(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=True
        )
        fig, ax, ax2 = plot_1D_discrete_eff(ale_eff, self.X)
        self.assertCountEqual(ax.lines[0].get_xydata()[:, 0], ale_eff.index)
        self.assertCountEqual(ax.lines[0].get_xydata()[:, 1], ale_eff.eff)

    def test_1D_continuous_ci_plot(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=True
        )
        fig, ax, ax2 = plot_1D_discrete_eff(ale_eff, self.X)
        self.assertCountEqual(np.round(ax.lines[1].get_xydata()[1:, 1], 8) , np.round(ale_eff['lowerCI_95%'][1:], 8))
        self.assertCountEqual(np.round(ax.lines[2].get_xydata()[1:, 1], 8) , np.round(ale_eff['upperCI_95%'][1:], 8))

    def test_1D_continuous_bar_plot(self):
        ale_eff = aleplot_1D_discrete(
            X=self.X, model=self.model, feature="x4", include_CI=True
        )
        fig, ax, ax2 = plot_1D_discrete_eff(ale_eff, self.X)
        self.assertCountEqual(ale_eff['size'], [bar.get_height() for bar in ax2.patches])


if __name__ == "__main__":
    unittest.main()
