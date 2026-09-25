"""Tests for the plotting theme."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn

from mmmjax import theme_mmmjax


def test_theme_mmmjax_draws_twelve_by_seven_figures_without_grid():
    frame = pd.DataFrame({"week": [1, 2, 3], "sales": [120.0, 150.0, 130.0]})

    figure = (pn.ggplot(frame, pn.aes("week", "sales")) + pn.geom_line() + theme_mmmjax()).draw()

    axis = figure.axes[0]
    np.testing.assert_array_equal(figure.get_size_inches(), [12.0, 7.0])
    assert figure.dpi == 100
    assert not any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines())
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    plt.close(figure)


def test_theme_mmmjax_settings_can_be_overridden():
    frame = pd.DataFrame({"week": [1, 2, 3], "sales": [120.0, 150.0, 130.0]})

    figure = (
        pn.ggplot(frame, pn.aes("week", "sales")) + pn.geom_line() + theme_mmmjax() + pn.theme(figure_size=(6, 4))
    ).draw()

    np.testing.assert_array_equal(figure.get_size_inches(), [6.0, 4.0])
    plt.close(figure)
