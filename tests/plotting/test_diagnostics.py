"""Tests for plots that check a model's fit and its sampler's convergence."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    Data,
    Model,
    Prior,
    Real,
    fit_data_scaling,
    normal,
    plot_fit,
    plot_ppc_dist,
    plot_ppc_tstat,
    plot_prior_posterior,
    plot_psense,
    plot_rank,
    plot_residuals,
    plot_rhat,
    plot_trace_dist,
    prepare_data,
)
from mmmjax.data._results import _collect_results
from mmmjax.plotting import diagnostics


def _fitted():
    source = pl.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "sales": [100.0, 120.0, 110.0, 130.0],
            "tv": [1.0, 0.0, 2.0, 1.0],
        }
    )
    data = prepare_data(source, time="week", outcome="sales", media=["tv"])
    scaling = fit_data_scaling(data, scale_outcome=True)
    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        data=Data(data, scaling=scaling),
    )
    scaled = scaling.transform(data)
    # Draws sit 0.1, 0.2, and 0.3 above the scaled observations, so their mean is 0.2 above.
    shifts = np.array([0.1, 0.2, 0.3])[None, :, None]
    predictive = scaled.arrays["outcome"] + shifts + np.zeros((2, 1, 1))
    results = _collect_results(
        {"level": np.zeros((2, 3), dtype=np.float32)},
        data=scaled,
        posterior_predictive={"outcome": predictive},
        generated_dims={"outcome": ("time",)},
    )
    results.attrs["data_scale"] = "model"
    outcome_scale = np.asarray(scaling.transformations["outcome"].scale).item()
    return model, results, outcome_scale


def _posterior():
    rng = np.random.default_rng(0)
    results = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "coefficient": (("chain", "draw", "channel"), rng.normal(size=(2, 50, 2))),
                    "sigma": (("chain", "draw"), rng.gamma(2.0, size=(2, 50))),
                },
                coords={"chain": [0, 1], "draw": np.arange(50), "channel": ["TV", "Search"]},
            )
        }
    )
    return results


def _prior():
    rng = np.random.default_rng(1)
    prior = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {
                    "coefficient": (("chain", "draw", "channel"), rng.normal(size=(1, 100, 2))),
                    "sigma": (("chain", "draw"), rng.gamma(2.0, size=(1, 100))),
                },
                coords={"chain": [0], "draw": np.arange(100), "channel": ["TV", "Search"]},
            )
        }
    )
    return prior


def test_plot_fit_restores_outcome_units_and_labels_axes_by_source_columns():
    model, results, outcome_scale = _fitted()
    observed = np.array([100.0, 120.0, 110.0, 130.0])
    expected_prediction = observed + 0.2 * outcome_scale

    plot = plot_fit(model, results)

    predicted = plot.data[plot.data["series"] != "Observed"]["estimate"]
    np.testing.assert_allclose(plot.data[plot.data["series"] == "Observed"]["estimate"], observed, rtol=3e-6, atol=0)
    np.testing.assert_allclose(predicted, expected_prediction, rtol=3e-6, atol=0)
    assert plot.labels.x == "Week"
    assert plot.labels.y == "Sales"


def test_plot_fit_leaves_results_in_original_units_unchanged():
    model, results, _ = _fitted()
    results.attrs["data_scale"] = "original"
    expected = results["observed_data"].to_dataset()["outcome"].values

    plot = plot_fit(model, results)

    np.testing.assert_allclose(plot.data[plot.data["series"] == "Observed"]["estimate"], expected, rtol=1e-12, atol=0)


def test_plot_residuals_subtract_predictions_from_observations():
    model, results, outcome_scale = _fitted()
    expected = np.full(4, -0.2 * outcome_scale)

    plot = plot_residuals(model, results)

    np.testing.assert_allclose(plot.data["estimate"], expected, rtol=2e-5, atol=1e-4)
    assert plot.labels.y.startswith("Sales residual, ")


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"model": object()}, TypeError, "model must be a Model"),
        ({"results": xr.Dataset()}, TypeError, "results must be an xarray DataTree"),
        ({"group": "both"}, ValueError, "group must be 'prior' or 'posterior'"),
        ({"group": "prior"}, ValueError, "results has no prior_predictive group"),
        ({"var_name": "revenue"}, ValueError, "var_name must name a variable in both"),
        ({"var_name": 1}, TypeError, "var_name must be a string"),
        ({"ci_prob": 0.0}, ValueError, "ci_prob must be between 0 and 1"),
    ],
)
@pytest.mark.parametrize("draw", [plot_fit, plot_residuals])
def test_fit_plots_require_predictive_draws_and_observations(draw, options, error, message):
    model, results, _ = _fitted()
    arguments = {"model": model, "results": results} | options

    with pytest.raises(error, match=message):
        draw(arguments.pop("model"), arguments.pop("results"), **arguments)


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(lambda: plot_rank(_posterior()), id="rank"),
        pytest.param(lambda: plot_trace_dist(_posterior()), id="trace"),
        pytest.param(lambda: plot_prior_posterior(_posterior(), _prior()), id="prior and posterior"),
        pytest.param(lambda: plot_ppc_dist(*_fitted()[:2]), id="predictive check"),
    ],
)
def test_arviz_diagnostics_default_to_the_plotnine_figure_size(draw):
    collection = draw()

    figure = collection.viz["figure"].item()
    np.testing.assert_array_equal(figure.get_size_inches(), [12.0, 7.0])
    plt.close("all")


def test_arviz_diagnostics_pass_other_keywords_to_arviz():
    collection = plot_rank(_posterior(), var_names=["sigma"], figure_kwargs={"figsize": (6, 4)})

    figure = collection.viz["figure"].item()
    np.testing.assert_array_equal(figure.get_size_inches(), [6.0, 4.0])
    assert figure.legends
    plt.close("all")


@pytest.mark.parametrize(
    ("var_names", "error", "message"),
    [
        (["gamma"], ValueError, "var_names has no posterior variable 'gamma'"),
        ("sigma", TypeError, "var_names must be a sequence of names"),
    ],
)
@pytest.mark.parametrize("draw", [plot_rank, plot_trace_dist])
def test_convergence_plots_reject_unknown_variables(draw, var_names, error, message):
    with pytest.raises(error, match=message):
        draw(_posterior(), var_names=var_names)


def test_plot_prior_posterior_requires_a_prior_group():
    with pytest.raises(ValueError, match="prior has no prior group"):
        plot_prior_posterior(_posterior(), _posterior())


def _grouped(*, population=True):
    rng = np.random.default_rng(2)
    labels = ["a", "b", "c", "d"]
    coords = {"chain": [0], "draw": np.arange(200), "group": labels}
    groups = {
        "posterior": xr.Dataset(
            {"intercept": (("chain", "draw", "group"), rng.normal(size=(1, 200, 4)))}, coords=coords
        )
    }
    if population:
        groups["constant_data"] = xr.Dataset(
            {"population": (("group",), [10.0, 40.0, 30.0, 20.0])}, coords={"group": labels}
        )
    prior = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {"intercept": (("chain", "draw", "group"), rng.standard_cauchy(size=(1, 200, 4)))}, coords=coords
            )
        }
    )
    return xr.DataTree.from_dict(groups), prior


def test_plot_prior_posterior_shows_the_most_populous_groups():
    results, prior = _grouped()

    collection = plot_prior_posterior(results, prior)

    assert collection.data["series"].values.tolist() == ["b", "c", "d"]
    plt.close("all")


def test_plot_prior_posterior_shows_the_first_groups_without_population():
    results, prior = _grouped(population=False)

    collection = plot_prior_posterior(results, prior, n_groups=2)

    assert collection.data["series"].values.tolist() == ["a", "b"]
    plt.close("all")


def test_plot_prior_posterior_uses_requested_group_labels():
    results, prior = _grouped()

    collection = plot_prior_posterior(results, prior, coords={"group": ["a"]})

    assert collection.data["series"].values.tolist() == ["a"]
    plt.close("all")


def test_plot_prior_posterior_shows_the_first_periods():
    rng = np.random.default_rng(3)
    periods = pd.date_range("2024-01-01", periods=5, freq="W-MON")
    coords = {"chain": [0], "draw": np.arange(100), "time": periods}
    draws = {"trend": (("chain", "draw", "time"), rng.normal(size=(1, 100, 5)))}
    results = xr.DataTree.from_dict({"posterior": xr.Dataset(draws, coords=coords)})
    prior = xr.DataTree.from_dict({"prior": xr.Dataset(draws, coords=coords)})

    collection = plot_prior_posterior(results, prior)

    np.testing.assert_array_equal(collection.data["time"].values, periods[:3].values)
    plt.close("all")


def test_plot_prior_posterior_pulls_in_extreme_prior_draws():
    results, prior = _grouped()
    posterior_draws = results["posterior"].to_dataset()["intercept"].values[..., 1]
    prior_draws = prior["prior"].to_dataset()["intercept"].values[..., 1]
    low = min(np.quantile(posterior_draws, 0.01), np.quantile(prior_draws, 0.01))
    high = max(np.quantile(posterior_draws, 0.99), np.quantile(prior_draws, 0.99))
    margin = 0.2 * (high - low)

    clipped = plot_prior_posterior(results, prior, n_groups=None)
    unclipped = plot_prior_posterior(results, prior, n_groups=None, clip_tails=False)

    values = clipped.data["intercept"].sel(series="b")
    assert float(values.max()) <= high + margin + 1e-9
    assert float(values.min()) >= low - margin - 1e-9
    assert float(unclipped.data["intercept"].sel(series="b").max()) > high + margin
    plt.close("all")


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"n_groups": 0}, ValueError, "n_groups must be at least 1"),
        ({"n_groups": True}, TypeError, "n_groups must be an integer or None"),
        ({"n_periods": "3"}, TypeError, "n_periods must be an integer or None"),
        ({"clip_tails": "yes"}, TypeError, "clip_tails must be a bool"),
    ],
)
def test_plot_prior_posterior_rejects_invalid_panel_settings(options, error, message):
    results, prior = _grouped()

    with pytest.raises(error, match=message):
        plot_prior_posterior(results, prior, **options)


def test_plot_fit_reports_fit_statistics_in_its_subtitle():
    model, results, outcome_scale = _fitted()
    observed = np.array([100.0, 120.0, 110.0, 130.0])
    residual = 0.2 * outcome_scale
    r_squared = 1.0 - 4 * residual**2 / np.sum((observed - observed.mean()) ** 2)
    weighted_error = 4 * residual / observed.sum()
    expected = f"R² {r_squared:.2f}, wMAPE {weighted_error:.1%}, 0% of periods inside the 90% interval"

    plot = plot_fit(model, results, ci_prob=0.9)

    assert plot.labels.subtitle == expected


def _grouped_fit(*, population=True):
    model, _, _ = _fitted()
    labels = ["north", "south"]
    periods = pd.date_range("2024-01-01", periods=4, freq="W-MON")
    observed = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]])
    predictive = np.broadcast_to(observed, (2, 3, 4, 2)) + np.array([0.0, 1.0, 2.0])[None, :, None, None]
    groups = {
        "posterior_predictive": xr.Dataset(
            {"outcome": (("chain", "draw", "time", "group"), predictive)},
            coords={"chain": [0, 1], "draw": np.arange(3), "time": periods, "group": labels},
        ),
        "observed_data": xr.Dataset(
            {"outcome": (("time", "group"), observed)}, coords={"time": periods, "group": labels}
        ),
    }
    if population:
        groups["constant_data"] = xr.Dataset({"population": (("group",), [5.0, 50.0])}, coords={"group": labels})
    results = xr.DataTree.from_dict(groups)
    results.attrs["data_scale"] = "original"
    return model, results


def test_plot_fit_sums_groups_into_one_total_by_default():
    model, results = _grouped_fit()

    plot = plot_fit(model, results)

    observed = plot.data[plot.data["series"] == "Observed"]["estimate"]
    np.testing.assert_allclose(observed, [11.0, 22.0, 33.0, 44.0], rtol=1e-12, atol=0)
    assert isinstance(plot.facet, pn.facet_null)


def test_plot_fit_keeps_the_most_populous_groups_as_panels():
    model, results = _grouped_fit()

    plot = plot_fit(model, results, by="group", n_groups=1)

    panels = list(plot.data["panel"].cat.categories)
    assert len(panels) == 1
    assert panels[0].startswith("south   R² ")


def _effects(baseline, *, labels=None, drawn="posterior"):
    periods = pd.date_range("2024-01-01", periods=4, freq="W-MON")
    axes = ("time",) if labels is None else ("time", "group")
    coords = {"chain": [0, 1], "draw": np.arange(3), "time": periods} | ({} if labels is None else {"group": labels})
    # Draws sit one below, at, and one above each value, so their mean is the value.
    shifts = np.array([-1.0, 0.0, 1.0]).reshape((1, 3) + (1,) * len(axes))
    draws = np.asarray(baseline) + shifts + np.zeros((2,) + (1,) * (len(axes) + 1))
    effects = xr.Dataset({"baseline_response": (("chain", "draw", *axes), draws)}, coords=coords)
    effects.attrs["group"] = drawn
    return effects


def _stub_contributions(monkeypatch, effects):
    calls = []

    def contributions(model, results, **options):
        calls.append(options)
        return effects

    monkeypatch.setattr(diagnostics, "contributions", contributions)
    return calls


def test_plot_fit_draws_the_baseline_from_contributions(monkeypatch):
    model, results, _ = _fitted()
    expected = np.array([90.0, 95.0, 92.0, 97.0])
    calls = _stub_contributions(monkeypatch, _effects(expected))

    plot = plot_fit(model, results, show_baseline=True, quantity="mu", ci_prob=0.9)

    baseline = plot.data[plot.data["series"] == "Baseline"]
    np.testing.assert_allclose(baseline["estimate"], expected, rtol=1e-12, atol=0)
    assert list(plot.data["series"].cat.categories) == ["Observed", "Predicted, 90% interval", "Baseline"]
    assert calls == [{"quantity": "mu", "group": "posterior", "by": ["time"]}]


def test_plot_fit_sums_the_baseline_like_the_fit(monkeypatch):
    model, results = _grouped_fit()
    effects = _effects([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], labels=["north", "south"])
    calls = _stub_contributions(monkeypatch, effects)

    total = plot_fit(model, results, show_baseline=True, quantity="mu")
    panel = plot_fit(model, results, show_baseline=True, quantity="mu", by="group", n_groups=1)
    chosen = plot_fit(model, results, show_baseline=True, quantity="mu", coords={"group": ["north"]})

    summed = total.data[total.data["series"] == "Baseline"]["estimate"]
    kept = panel.data[panel.data["series"] == "Baseline"]
    north = chosen.data[chosen.data["series"] == "Baseline"]["estimate"]
    np.testing.assert_allclose(summed, [11.0, 22.0, 33.0, 44.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(kept["estimate"], [10.0, 20.0, 30.0, 40.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(north, [1.0, 2.0, 3.0, 4.0], rtol=1e-12, atol=0)
    assert kept["panel"].astype(str).str.startswith("south").all()
    # Every observation axis reaches contributions, so coords can pick groups before the baseline is summed.
    assert [call["by"] for call in calls] == [["time", "group"]] * 3


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"show_baseline": True}, ValueError, "quantity must name the expected outcome"),
        ({"quantity": "mu"}, ValueError, "quantity is only used with show_baseline=True"),
        ({"show_baseline": 1, "quantity": "mu"}, TypeError, "show_baseline must be a bool"),
        ({"show_baseline": True, "quantity": 3}, TypeError, "quantity must be a string"),
    ],
)
def test_plot_fit_requires_a_quantity_for_the_baseline(options, error, message):
    model, results, _ = _fitted()

    with pytest.raises(error, match=message):
        plot_fit(model, results, **options)


def test_plot_fit_draws_with_the_baseline(monkeypatch):
    model, results, _ = _fitted()
    _stub_contributions(monkeypatch, _effects(np.full(4, 90.0)))

    figure = plot_fit(model, results, show_baseline=True, quantity="mu").draw()

    assert figure.axes
    plt.close(figure)


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(lambda model, results: plot_fit(model, results), id="fit"),
        pytest.param(lambda model, results: plot_residuals(model, results), id="residuals"),
    ],
)
def test_fit_plots_draw(draw):
    model, results, _ = _fitted()

    figure = draw(model, results).draw()

    assert figure.axes
    plt.close(figure)


@pytest.mark.parametrize(
    ("by", "error", "message"),
    [
        ("time", ValueError, "by must name observation axes other than time"),
        ("channel", ValueError, "by must name observation axes other than time"),
        (3, TypeError, "by must be an axis name or a sequence of them"),
    ],
)
def test_plot_fit_rejects_axes_it_cannot_keep(by, error, message):
    model, results = _grouped_fit()

    with pytest.raises(error, match=message):
        plot_fit(model, results, by=by)


def _large(count=60, *, drifting=(0, 7)):
    rng = np.random.default_rng(4)
    labels = [f"Channel {index:02d}" for index in range(count)]
    draws = rng.normal(size=(4, 100, count))
    for index in drifting:
        # Two chains sit apart from the other two, which R-hat flags.
        draws[1::2, :, index] += 3.0
    results = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"coefficient": (("chain", "draw", "channel"), draws)},
                coords={"chain": [0, 1, 2, 3], "draw": np.arange(100), "channel": labels},
            )
        }
    )
    return results


def test_plot_rank_keeps_the_worst_mixing_parameters_of_large_models():
    collection = plot_rank(_large())

    names = list(collection.data.data_vars)
    figure = collection.viz["figure"].item()
    assert len(names) == 12
    assert {"coefficient[Channel 00]", "coefficient[Channel 07]"} <= set(names)
    assert figure.get_suptitle() == "The 12 of 60 parameters with the highest R-hat"
    plt.close("all")


def test_plot_trace_dist_keeps_six_rows_for_large_models():
    collection = plot_trace_dist(_large())

    assert len(collection.data.data_vars) == 6
    plt.close("all")


def test_plot_trace_dist_grows_taller_with_its_rows_unless_sized():
    grown = plot_trace_dist(_large()).viz["figure"].item()
    sized = plot_trace_dist(_large(), figure_kwargs={"figsize": (12, 5)}).viz["figure"].item()

    # Six rows of 1.8 inches each.
    np.testing.assert_allclose(grown.get_size_inches(), [12.0, 10.8], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sized.get_size_inches(), [12.0, 5.0], rtol=1e-12, atol=0)
    plt.close("all")


def test_plot_trace_dist_names_each_row_once_under_its_density():
    collection = plot_trace_dist(_large())

    plots = collection.viz["plot"].to_dataset()
    traces = [plots[name].sel(column="trace").item() for name in plots.data_vars]
    densities = [plots[name].sel(column="dist").item() for name in plots.data_vars]
    assert {axis.get_ylabel() for axis in traces} == {""}
    assert "coefficient\n[Channel 00]" in {axis.get_xlabel() for axis in densities}
    plt.close("all")


def test_grid_diagnostics_grow_taller_with_their_rows_of_panels():
    rank = plot_rank(_large()).viz["figure"].item()
    wrapped = plot_rank(_large(), col_wrap=2).viz["figure"].item()
    chosen = plot_rank(_large(), coords={"channel": ["Channel 01", "Channel 02"]}).viz["figure"].item()

    # Twelve panels make three rows of four, or six rows of two, at three inches a row.
    np.testing.assert_allclose(rank.get_size_inches(), [12.0, 9.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(wrapped.get_size_inches(), [12.0, 18.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(chosen.get_size_inches(), [12.0, 7.0], rtol=1e-12, atol=0)
    plt.close("all")


def test_convergence_plots_put_long_element_labels_on_their_own_line():
    figure = plot_rank(_large()).viz["figure"].item()

    titles = {axis.get_title() for axis in figure.axes}
    assert "coefficient\n[Channel 00]" in titles
    plt.close("all")


def test_convergence_plots_leave_explicit_coords_alone():
    collection = plot_rank(_large(), coords={"channel": ["Channel 01", "Channel 02"]})

    assert list(collection.data.data_vars) == ["coefficient"]
    plt.close("all")


def test_plot_prior_posterior_keeps_the_parameters_the_data_narrowed_least_in_large_models():
    results = _large(drifting=())
    draws = results["posterior"].to_dataset()["coefficient"].values.copy()
    # Most posteriors are a tenth as wide as the prior, and a shifted narrow one must not count as unchanged.
    narrowed = 0.1 * draws
    narrowed[..., 20] += 2.0
    wide = [3, 11, 17, 29, 30, 33, 41, 44, 48, 52, 57, 59]
    # Exponentiating keeps each draw's place among the prior draws, so the measure must still see no change.
    narrowed[..., wide[0]] = np.exp(draws[..., wide[0]])
    narrowed[..., wide[1:]] = draws[..., wide[1:]]
    results = xr.DataTree.from_dict(
        {"posterior": results["posterior"].to_dataset().assign(coefficient=(("chain", "draw", "channel"), narrowed))}
    )
    unchanged = np.random.default_rng(5).normal(size=(4, 100, 60))
    unchanged[..., wide[0]] = np.exp(unchanged[..., wide[0]])
    prior = xr.DataTree.from_dict(
        {
            "prior": xr.Dataset(
                {"coefficient": (("chain", "draw", "channel"), unchanged)},
                coords=results["posterior"].to_dataset().coords,
            )
        }
    )
    expected = {f"coefficient[Channel {index:02d}]" for index in wide}

    collection = plot_prior_posterior(results, prior)

    names = {str(name) for name in collection.data.data_vars}
    assert names == expected
    assert collection.viz["figure"].item().get_suptitle() == "The 12 of 60 parameters the data narrowed least"
    plt.close("all")


def test_plot_trace_dist_keeps_the_divergence_marks_of_large_models():
    results = _large()
    diverging = np.zeros((4, 100), dtype=bool)
    diverging[1, 10:20] = True
    stats = xr.Dataset(
        {"diverging": (("chain", "draw"), diverging)}, coords={"chain": [0, 1, 2, 3], "draw": np.arange(100)}
    )
    results = xr.DataTree.from_dict({"posterior": results["posterior"].to_dataset(), "sample_stats": stats})

    collection = plot_trace_dist(results)

    assert len(collection.data.data_vars) == 6
    assert {"divergence_trace", "divergence_dist"} <= set(collection.viz.children)
    plt.close("all")


def test_plot_rhat_shows_every_element_by_parameter():
    results = _large(12, drifting=(0,))
    sigma = np.random.default_rng(6).gamma(2.0, size=(4, 100))
    posterior = results["posterior"].to_dataset().assign(sigma=(("chain", "draw"), sigma))
    results = xr.DataTree.from_dict({"posterior": posterior})
    from arviz_stats.sampling_diagnostics import rhat

    expected = rhat(posterior)
    above = int((expected["coefficient"] > 1.01).sum()) + int(expected["sigma"] > 1.01)

    plot = plot_rhat(results)

    values = plot.data.groupby("parameter", observed=True)["rhat"].apply(list)
    np.testing.assert_allclose(values["coefficient"], expected["coefficient"].values, rtol=1e-12, atol=0)
    np.testing.assert_allclose(values["sigma"], [float(expected["sigma"])], rtol=1e-12, atol=0)
    assert list(plot.data["parameter"].cat.categories) == ["coefficient", "sigma"]
    assert above >= 1
    assert plot.labels.subtitle == f"{above} of 13 R-hat values are above 1.01"


def test_plot_rhat_flags_values_past_the_limit_with_parameters_down_the_side():
    plot = plot_rhat(_large(12, drifting=(0,)))

    past = plot.data["rhat"] > 1.01
    assert past.any()
    assert set(plot.data.loc[past, "status"]) == {"Above 1.01"}
    assert set(plot.data.loc[~past, "status"]) == {"At or below 1.01"}
    assert isinstance(plot.coordinates, pn.coord_flip)


def test_plot_rhat_reports_when_every_value_is_below_the_limit():
    plot = plot_rhat(_large(6, drifting=()), var_names=["coefficient"])

    assert plot.labels.subtitle == "All 6 R-hat values are at or below 1.01"


def test_plot_rhat_leaves_out_parameters_without_a_finite_value():
    results = _large(4, drifting=())
    posterior = results["posterior"].to_dataset().assign(fixed=(("chain", "draw"), np.ones((4, 100))))

    plot = plot_rhat(xr.DataTree.from_dict({"posterior": posterior}))

    assert set(plot.data["parameter"]) == {"coefficient"}


@pytest.mark.parametrize(
    ("results", "options", "error", "message"),
    [
        (xr.Dataset(), {}, TypeError, "results must be an xarray DataTree"),
        (xr.DataTree(), {}, ValueError, "results has no posterior group"),
        (_large(4, drifting=()), {"var_names": ["sigma"]}, ValueError, "var_names has no posterior variable 'sigma'"),
        (_large(4, drifting=()), {"var_names": "coefficient"}, TypeError, "var_names must be a sequence of names"),
        (
            xr.DataTree.from_dict(
                {"posterior": xr.Dataset({"fixed": (("chain", "draw"), np.ones((2, 5)))}, coords={"chain": [0, 1]})}
            ),
            {},
            ValueError,
            "results has no parameter with a finite R-hat",
        ),
    ],
)
def test_plot_rhat_rejects_invalid_arguments(results, options, error, message):
    with pytest.raises(error, match=message):
        plot_rhat(results, **options)


def test_plot_rhat_draws():
    figure = plot_rhat(_large(12, drifting=(0,))).draw()

    assert figure.axes
    plt.close(figure)


def test_arviz_diagnostics_wrap_long_names_in_panel_titles():
    results = _large(2, drifting=())
    posterior = (
        results["posterior"].to_dataset().assign_coords(channel=["social_media_meta_dynamic_brand_world_cup", "search"])
    )

    collection = plot_rank(xr.DataTree.from_dict({"posterior": posterior}))

    figure = collection.viz["figure"].item()
    titles = [axes.get_title() for axes in figure.axes if axes.get_title()]
    assert "coefficient\nsocial_media_meta_dynamic_\nbrand_world_cup" in titles
    plt.close("all")


def _series_fit():
    model, _, _ = _fitted()
    rng = np.random.default_rng(6)
    periods = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    observed = np.cumsum(rng.normal(size=30)) + 50.0
    predictive = observed + rng.normal(size=(2, 40, 30))
    results = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {"outcome": (("chain", "draw", "time"), predictive)},
                coords={"chain": [0, 1], "draw": np.arange(40), "time": periods},
            ),
            "observed_data": xr.Dataset({"outcome": (("time",), observed)}, coords={"time": periods}),
        }
    )
    results.attrs["data_scale"] = "original"
    return model, results, observed, predictive


def _lag_one(values):
    centered = values - values.mean()
    # The full correlation holds lag zero at the series length minus one and lag one right after it.
    return np.correlate(centered, centered, "full")[values.size] / np.dot(centered, centered)


def _lag_one_rows(values):
    return np.array([[_lag_one(row) for row in chain] for chain in values])


def _titles(collection):
    return [axis.get_title() for axis in collection.viz["figure"].item().axes if axis.get_title()]


def test_plot_ppc_tstat_compares_each_statistic_of_the_draws_with_the_observations():
    model, results, observed, predictive = _series_fit()
    persistence = np.array([[_lag_one(draw) for draw in chain] for chain in predictive])
    spread = predictive.std(axis=-1)
    largest = predictive.max(axis=-1)
    expected = [
        f"Autocorrelation, p = {np.mean(persistence >= _lag_one(observed)):.2f}",
        f"Standard deviation, p = {np.mean(spread >= observed.std()):.2f}",
        f"Maximum, p = {np.mean(largest >= observed.max()):.2f}",
    ]

    collection = plot_ppc_tstat(model, results)

    # ArviZ stacks chain and draw into one sample axis in that order.
    np.testing.assert_allclose(collection.data["Autocorrelation"].values, persistence.ravel(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(collection.data["Maximum"].values, largest.ravel(), rtol=1e-12, atol=0)
    assert _titles(collection) == expected
    plt.close("all")


def test_plot_ppc_tstat_compares_residuals_of_each_draw_with_its_own_expected_outcome(monkeypatch):
    model, results, observed, predictive = _series_fit()
    rng = np.random.default_rng(8)
    expected_outcome = observed + rng.normal(scale=0.5, size=(2, 40, 30))
    periods = results["observed_data"].to_dataset()["time"].values
    effects = xr.Dataset(
        {"reference_response": (("chain", "draw", "time"), expected_outcome)},
        coords={"chain": [0, 1], "draw": np.arange(40), "time": periods},
    )
    calls = _stub_contributions(monkeypatch, effects)
    replicated = _lag_one_rows(predictive - expected_outcome)
    actual = _lag_one_rows(observed - expected_outcome)
    expected = f"Residual autocorrelation, p = {np.mean(replicated >= actual):.2f}"

    collection = plot_ppc_tstat(model, results, quantity="mu")

    axis = collection.viz["plot"].to_dataset()["Residual autocorrelation"].item()
    np.testing.assert_allclose(
        collection.data["Residual autocorrelation"].values, replicated.ravel(), rtol=1e-10, atol=1e-12
    )
    assert _titles(collection)[0] == expected
    # The observed values vary by draw, so a second curve takes the place of the dot.
    assert len(axis.lines) == 2
    assert not axis.collections
    assert calls == [{"quantity": "mu", "group": "posterior", "by": ["time"]}]
    plt.close("all")


def test_plot_ppc_tstat_names_quantiles_and_functions():
    model, results, observed, predictive = _series_fit()

    def holiday_share(series):
        return series[-4:].sum() / series.sum()

    shares = predictive[..., -4:].sum(axis=-1) / predictive.sum(axis=-1)
    expected = [
        f"90th percentile, p = {np.mean(np.quantile(predictive, 0.9, axis=-1) >= np.quantile(observed, 0.9)):.2f}",
        f"2.5th percentile, p = {np.mean(np.quantile(predictive, 0.025, axis=-1) >= np.quantile(observed, 0.025)):.2f}",
        f"Holiday share, p = {np.mean(shares >= holiday_share(observed)):.2f}",
    ]

    collection = plot_ppc_tstat(model, results, statistics=[0.9, 0.025, holiday_share])

    assert _titles(collection) == expected
    np.testing.assert_allclose(collection.data["Holiday share"].values, shares.ravel(), rtol=1e-12, atol=0)
    plt.close("all")


def test_plot_ppc_tstat_gives_each_group_a_row_of_panels():
    model, results = _grouped_fit()

    summed = plot_ppc_tstat(model, results, statistics=["mean", "min"])
    grouped = plot_ppc_tstat(model, results, statistics=["mean", "min"], by="group", n_groups=None)
    largest = plot_ppc_tstat(model, results, statistics=["mean"], by="group", n_groups=1)
    south = results["posterior_predictive"].to_dataset()["outcome"].sel(group="south").mean("time")

    np.testing.assert_allclose(grouped.data["south   Mean"].values, south.values.ravel(), rtol=1e-12, atol=0)
    assert largest.viz["figure"].item().get_suptitle() == "Showing 1 of 2 groups. Pass coords to choose others."
    assert _titles(summed) == ["Mean, p = 1.00", "Minimum, p = 1.00"]
    assert _titles(grouped) == [
        "north   Mean, p = 1.00",
        "north   Minimum, p = 1.00",
        "south   Mean, p = 1.00",
        "south   Minimum, p = 1.00",
    ]
    assert summed.viz["figure"].item().get_size_inches()[1] == pytest.approx(4.5)
    assert grouped.viz["figure"].item().get_size_inches()[1] == pytest.approx(7.0)
    plt.close("all")


@pytest.mark.parametrize(
    ("statistics", "error", "message"),
    [
        ("mean", TypeError, "statistics must be a sequence"),
        ([], ValueError, "statistics must hold at least one statistic"),
        (["skew"], ValueError, "statistics must name 'mean'"),
        ([1.5], ValueError, "statistics quantiles must lie between 0 and 1"),
        ([object()], TypeError, "statistics must hold names, quantiles, or functions"),
        (["mean", "mean"], ValueError, "statistics must be distinct"),
        ([np.cumsum], ValueError, "statistics functions must return one number per series, got shape"),
        (["residual_autocorrelation"], ValueError, "quantity must name the expected outcome"),
    ],
)
def test_plot_ppc_tstat_rejects_invalid_statistics(statistics, error, message):
    model, results, _, _ = _series_fit()

    with pytest.raises(error, match=message):
        plot_ppc_tstat(model, results, statistics=statistics)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"quantity": 3}, TypeError, "quantity must be a string, got int"),
        ({"quantity": "mu", "statistics": ["mean"]}, ValueError, "quantity is only used with residual autocorrelation"),
    ],
)
def test_plot_ppc_tstat_rejects_a_quantity_it_cannot_use(options, error, message):
    model, results, _, _ = _series_fit()

    with pytest.raises(error, match=message):
        plot_ppc_tstat(model, results, **options)


def test_plot_ppc_tstat_numbers_unnamed_functions_and_takes_numpy_quantiles():
    model, results, _, _ = _series_fit()
    statistics = [lambda series: series[0], np.float32(0.5), lambda series: series[-1]]

    collection = plot_ppc_tstat(model, results, statistics=statistics)

    assert [title.split(",")[0] for title in _titles(collection)] == ["Statistic 1", "50th percentile", "Statistic 3"]
    plt.close("all")


def _sensitive(count=30, *, log_prior=False):
    rng = np.random.default_rng(7)
    labels = [f"Channel {index:02d}" for index in range(count)]
    coefficient = rng.normal(size=(2, 200, count))
    sigma = rng.gamma(4.0, 0.25, size=(2, 200))
    sample = {"chain": [0, 1], "draw": np.arange(200)}
    groups = {
        "posterior": xr.Dataset(
            {"coefficient": (("chain", "draw", "channel"), coefficient), "sigma": (("chain", "draw"), sigma)},
            coords=sample | {"channel": labels},
        ),
        "log_likelihood": xr.Dataset(
            {"outcome": (("chain", "draw", "time"), rng.normal(size=(2, 200, 5)))}, coords=sample | {"time": range(5)}
        ),
    }
    # Two channels get a tight prior, so scaling the prior moves them and hardly moves the rest.
    scale = np.full(count, 10.0)
    scale[[index for index in (4, 9) if index < count]] = 0.3
    priors = {
        "coefficient": Prior(normal, location=0.0, scale=scale),
        "sigma": Prior(normal, location=0.0, scale=10.0),
    }
    if log_prior:
        terms = {
            "coefficient": (("chain", "draw"), (-0.5 * (coefficient / scale) ** 2 - np.log(scale)).sum(axis=-1)),
            "sigma": (("chain", "draw"), -0.5 * (sigma / 10.0) ** 2),
        }
        groups["log_prior"] = xr.Dataset(terms, coords=sample)
    return xr.DataTree.from_dict(groups), priors


def test_plot_psense_keeps_the_parameters_most_sensitive_to_the_priors():
    results, priors = _sensitive()

    collection = plot_psense(results, priors=priors)

    names = [str(name) for name in collection.data.data_vars]
    assert len(names) == 6
    assert {"coefficient[Channel 04]", "coefficient[Channel 09]"} <= set(names[:2])
    assert collection.viz["figure"].item().get_suptitle() == "The 6 of 31 parameters most sensitive to the priors"
    assert collection.viz["figure"].item().get_size_inches()[1] == pytest.approx(1.8 * 6)
    plt.close("all")


def test_plot_psense_counts_one_quantity_given_as_a_string_as_one_column():
    results, priors = _sensitive(count=15)

    named = plot_psense(results, priors=priors, kind="quantities", quantities="mean")
    listed = plot_psense(results, priors=priors, kind="quantities", quantities=["mean"])

    assert list(named.data.data_vars) == list(listed.data.data_vars) == ["coefficient", "sigma"]
    plt.close("all")


def test_plot_psense_limits_the_panels_of_a_single_chosen_label():
    results, priors = _sensitive(count=3)
    values = results["posterior"].to_dataset()["coefficient"].sel(channel="Channel 01").values
    low, high = np.quantile(values, [0.01, 0.99])
    expected = (max(low - 0.2 * (high - low), values.min()), min(high + 0.2 * (high - low), values.max()))

    collection = plot_psense(results, priors=priors, var_names=["coefficient"], coords={"channel": "Channel 01"})

    for axis in np.ravel(collection.viz["plot"].to_dataset()["coefficient"].values):
        np.testing.assert_allclose(axis.get_xlim(), expected, rtol=1e-6, atol=0)
    plt.close("all")


def test_plot_psense_reads_the_log_prior_of_results_without_priors():
    results, _ = _sensitive(log_prior=True)

    collection = plot_psense(results, var_names=["sigma"], kind="quantities")

    assert "sigma" in collection.data.data_vars
    plt.close("all")


def test_plot_psense_cuts_long_tails_from_view():
    results, priors = _sensitive(count=3)
    heavy = np.exp(2.0 * results["posterior"].to_dataset()["sigma"].values)
    results = xr.DataTree.from_dict(
        {
            "posterior": results["posterior"].to_dataset().assign(sigma=(("chain", "draw"), heavy)),
            "log_likelihood": results["log_likelihood"].to_dataset(),
        }
    )
    low, high = np.quantile(heavy, [0.01, 0.99])
    expected = (max(low - 0.2 * (high - low), heavy.min()), min(high + 0.2 * (high - low), heavy.max()))

    collection = plot_psense(results, priors=priors, var_names=["sigma"])

    for axis in np.ravel(collection.viz["plot"].to_dataset()["sigma"].values):
        np.testing.assert_allclose(axis.get_xlim(), expected, rtol=1e-6, atol=0)
    assert expected[1] < heavy.max()
    plt.close("all")


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"kind": "bars"}, ValueError, "kind must be 'dist' or 'quantities'"),
        ({"priors": None}, ValueError, "results has no log_prior group"),
        ({"var_names": ["slope"]}, ValueError, "var_names has no posterior variable 'slope'"),
    ],
)
def test_plot_psense_rejects_invalid_arguments(options, error, message):
    results, priors = _sensitive(count=3)

    with pytest.raises(error, match=message):
        plot_psense(results, **({"priors": priors} | options))
