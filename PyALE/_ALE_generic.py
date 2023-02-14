import numpy as np
import pandas as pd
import logging

from ._src.ALE_1D import (
    aleplot_1D_continuous,
    plot_1D_continuous_eff,
    aleplot_1D_discrete,
    aleplot_1D_categorical,
    plot_1D_discrete_eff,
)
from ._src.ALE_2D import aleplot_2D_continuous, plot_2D_continuous_eff

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logging.basicConfig(format="%(name)s:%(levelname)s: %(message)s")


def ale(
    X,
    model,
    feature,
    feature_type="auto",
    grid_size=20,
    include_CI=True,
    C=0.95,
    encode_fun=None,
    predictors=None,
    plot=True,
    contour=False,
    fig=None,
    ax=None,
):
    """Compute the accumulated local effect (ALE) of a feature on a model.

    This function computes the effect of one (continuous, discrete, or categorical)
    feature, or two features on a given model.
    Some arguments in the function are relevant for a specific type of effect only,
    while others are relevant for all types. Irrelevant arguments will be ignored.
    The table bellow shows which arguments are for which type of effect relevant,
    and what is (if any) the default value for each.

    |   Argument   | 1D continuous | 1D discrete | 1D categorical | 2D (continuous) |  Default |
    | ------------ | ------------  | ----------- | -------------- | --------------- | -------- |
    | X            |       x       |      x      |        x       |        x        |          |
    | model        |       x       |      x      |        x       |        x        |          |
    | feature      |       x       |      x      |        x       |        x        |          |
    | feature_type |       x       |      x      |        x       |                 |  'auto'  |
    | grid_size    |       x       |             |                |        x        |    20    |
    | include_CI   |       x       |      x      |        x       |                 |   True   |
    | C            |       x       |      x      |        x       |                 |   0.95   |
    | encode_fun   |               |             |        x       |                 |   None   |
    | predictors   |               |             |        x       |                 |   None   |
    | plot         |       x       |      x      |        x       |        x        |   True   |
    | contour      |               |             |                |        x        |   False  |
    | fig          |       x       |      x      |        x       |        x        |   None   |
    | ax           |       x       |      x      |        x       |        x        |   None   |

    Arguments:
    X
    ----
        A pandas DataFrame to pass to the model for prediction.
    model
    ----
        Any python model with a predict method that accepts X as input,
        and return numeric predictions (the predictions for regression tasks and
        the probability for two-class classification tasks).
    feature
    ----
        List of strings, the name of the column (or columns) holding the feature(s)
        to analyze, accepts at most two features.
    feature_type
    ----
        String, one of 'auto', 'discrete', 'continuous', or 'categorical' specifying
        the type of values the feature has. Default is 'auto', in this case:
            * any non-numeric feature is considered categorical
            * for numeric features the number of unique values is tested, if the
            the feature has less than 11 unique values or at most 0.2% of sample
            size (i.e. #uniqueValues/#samples <= 0.002, that is 20 values for a
            sample of 10,000) then the feature is considered discrete, otherwise
            it's continuous.
    grid_size
    ----
        An integer indicating the number of intervals into which the feature range
        is divided.
    include_CI
    ----
        A boolean, if True the confidence interval of the effect is returned with
        the results.
    C
    ----
        A float, the confidence level for which to compute the confidence interval.
    encode_fun
    ----
        Function, used to encode the categorical feature, usually a  one-hot encoder.
        The function's input and output are as follows
            * input: a DataFrame with one column (the feature)
            * output: a DataFrame with the new column(s) encoding the feature.
        It is also important that this function should be able to handle missing
        categories (for example, a one-hot-encoder applied to a column, in which not
        all categories occur, should add a column of zeros for each missing category).
        Examples of use of this function could be found in the README file in github
        or in the description of the package in PyPI https://pypi.org/project/PyALE/.
    predictors
    ----
        List or array of strings containing the names of features used in the model,
        and in the right order.
    plot
    ----
        A boolean indicating whether to plot the effects or not.
    contour
    ----
        A boolean indicating if the heatmap for 2D effects should have labeled
        contours over it.
    fig, ax
    ----
        matplotlib figure and axis.

    Return:
        For 1D effects: A pandas DataFrame containing for each bin or value: the
        size of the sample in it, the accumulated centered effect, and the
        confidence interval of the effect if include_CI is True.
        For 2D effects: A grid of effects as a pandas DataFrame containing for
        each bin in the grid the accumulated centered effect of this bin.
    """
    # general checks
    if not isinstance(X, pd.DataFrame):
        raise Exception("The arguemnt 'X' must be a pandas DataFrame")
    if not hasattr(model, "predict"):
        raise Exception("The passed model does not seem to have a predict method.")
    if (not isinstance(feature, list)) | (
        np.any([not isinstance(x, str) for x in feature])
    ) | len(feature) > 2:
        raise Exception(
            "The arguemnt 'feature' must be a list of at most two feature"
            " names (strings)"
        )
    if np.any([not x in X.columns for x in feature]):
        raise Exception(
            "Feature(s) {} was(were) not found in the column names of X".format(
                [x for x in feature if not x in X.columns]
            )
        )

    if feature_type not in ["auto", "continuous", "discrete", "categorical"]:
        raise Exception(
            "The argument 'feature_type' should be 'auto', 'continuous', "
            "'discrete', or 'categorical'"
        )

    # if one feature is given
    if len(feature) == 1:
        feature = feature[0]
        feat_values_unique = len(X.loc[:, feature].squeeze().unique())
        # check that C has a value between 0 and 1 (only if include_CI is True)
        if include_CI and (not (0 <= C <= 1)):
            raise Exception(
                "The argument 'C' (confidence level) should be a value between 0 and 1"
            )
        # check feature type
        # assign feature type if not given
        if feature_type == "auto":
            logger.debug("Detecteing feature type ....")
            if X.loc[:, feature].dtype.kind in "iuf":
                # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
                if (feat_values_unique <= 10) | (
                    feat_values_unique / X.shape[0] <= 0.002
                ):
                    feature_type = "discrete"
                    logger.info("Discrete feature detected.")
                else:
                    feature_type = "continuous"
                    logger.info("Continuous feature detected.")
            else:
                feature_type = "categorical"
                logger.info("categorical feature detected.")
        # if the feature is continuous
        if feature_type == "continuous":
            arg_eff = {
                "X": X,
                "model": model,
                "feature": feature,
                "grid_size": grid_size,
                "include_CI": include_CI,
                "C": C,
            }
            arg_plot = {
                "X": X,
                "fig": fig,
                "ax": ax,
            }
            alefeat_fun = aleplot_1D_continuous
            plot_fun = plot_1D_continuous_eff
        # if the feature is discrete
        elif feature_type == "discrete":
            arg_eff = {
                "X": X,
                "model": model,
                "feature": feature,
                "include_CI": include_CI,
                "C": C,
            }
            arg_plot = {
                "X": X,
                "fig": fig,
                "ax": ax,
            }
            alefeat_fun = aleplot_1D_discrete
            plot_fun = plot_1D_discrete_eff
        # if the feature is categorical
        elif feature_type == "categorical":
            # check if the special parameters are given
            if encode_fun is None:
                raise Exception(
                    "Argument 'encode_fun' not given. With categorical/string "
                    "features, an encoding function should be provided."
                )
            if predictors is None:
                raise Exception(
                    "Argument 'predictors' not given. With categorical/string "
                    "features, a list of predictors (column names) should be provided."
                )
            arg_eff = {
                "X": X,
                "model": model,
                "feature": feature,
                "encode_fun": encode_fun,
                "predictors": predictors,
                "include_CI": include_CI,
                "C": C,
            }
            arg_plot = {
                "X": X,
                "fig": fig,
                "ax": ax,
            }
            alefeat_fun = aleplot_1D_categorical
            plot_fun = plot_1D_discrete_eff
    # if two features are given
    elif len(feature) == 2:
        arg_eff = {
            "X": X,
            "model": model,
            "features": feature,
            "grid_size": grid_size,
        }
        arg_plot = {
            "contour": contour,
            "fig": fig,
            "ax": ax,
        }
        alefeat_fun = aleplot_2D_continuous
        plot_fun = plot_2D_continuous_eff
    # compute the effects
    eff_res = alefeat_fun(**arg_eff)
    # plot them if wanted
    if plot:
        plot_fun(eff_res, **arg_plot)
    return eff_res
