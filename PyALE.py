from src.ALE_1D import (
    aleplot_1D_continuous,
    plot_1D_continuous_eff,
    aleplot_1D_discrete,
    plot_1D_discrete_eff,
)
from src.ALE_2D import aleplot_2D_continuous, plot_2D_continuous_eff


def ale(
    X,
    model,
    feature,
    feature_type="continuous",
    grid_size=40,
    include_CI=True,
    C=0.95,
    plot=True,
    contour=False, 
    fig=None, 
    ax=None,
):
    if isinstance(feature, str):
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
        else:
            raise Exception("feature_type should be either continuous or discrete")
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
    else:
        raise Exception("feature should have eithe one or two feature names")
    eff_res = alefeat_fun(**arg_eff)
    if plot:
        plot_fun(eff_res, **arg_plot)
    return eff_res
