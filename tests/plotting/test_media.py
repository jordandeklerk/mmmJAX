"""Tests for plots of media effects, returns, and response curves."""

import matplotlib.pyplot as plt
import numpy as np
import plotnine as pn
import pytest
import xarray as xr

from mmmjax import (
    plot_frequency_curves,
    plot_media_metrics,
    plot_response_curves,
    plot_roi_bubbles,
    plot_spend_vs_contribution,
)
from mmmjax.plotting._display import _ScrollingPlot


def _draws(*shape, seed):
    return np.random.default_rng(seed).normal(size=shape)


def _metrics(seed=0):
    metrics = xr.Dataset(
        {
            "roi": (("chain", "draw", "channel"), 2.0 + 0.1 * _draws(2, 40, 2, seed=seed)),
            "marginal_roi": (("chain", "draw", "channel"), 1.2 + 0.1 * _draws(2, 40, 2, seed=seed + 10)),
            "spend_share": (("channel",), [0.6, 0.4]),
        },
        coords={"chain": [0, 1], "draw": np.arange(40), "channel": ["TV", "Search"]},
    )
    return metrics


def _curves():
    multipliers = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    reference_spend = np.array([100.0, 50.0])
    spend = reference_spend[:, None] * multipliers
    increments = (1.0 + 0.05 * _draws(2, 40, 1, 1, seed=4)) * 1_000.0 * (1.0 - np.exp(-spend / 80.0))
    curves = xr.Dataset(
        {
            "incremental_response": (("chain", "draw", "channel", "multiplier"), increments),
            "spend": (("channel", "multiplier"), spend),
            "reference_spend": (("channel",), reference_spend),
        },
        coords={"chain": [0, 1], "draw": np.arange(40), "channel": ["TV", "Search"], "multiplier": multipliers},
    )
    return curves


def _plan(optimized, *, reference=(100.0, 50.0), lower=(60.0, 25.0), upper=(140.0, 75.0), channels=("TV", "Search")):
    plan = xr.Dataset(
        {
            "spend": (("allocation", "channel"), [list(reference), list(optimized)]),
            "response_change": (("chain", "draw"), np.ones((2, 40))),
            "lower_bound": (("channel",), list(lower)),
            "upper_bound": (("channel",), list(upper)),
        },
        coords={
            "chain": [0, 1],
            "draw": np.arange(40),
            "allocation": ["reference", "optimized"],
            "channel": list(channels),
        },
    )
    return plan


def _frequency_curves():
    changes = np.array([[1.0, 3.0, 2.0]]) + 0.01 * _draws(2, 40, 1, 3, seed=5)
    curves = xr.Dataset(
        {
            "response_change": (("chain", "draw", "channel", "frequency"), changes),
            "best_frequency": (("channel",), [2.0]),
        },
        coords={"chain": [0, 1], "draw": np.arange(40), "channel": ["YouTube"], "frequency": [1.0, 2.0, 3.0]},
    )
    return curves


def _layer_data(plot, geom):
    data = next(layer.geom.data for layer in plot.layers if isinstance(layer.geom, geom))
    return data


def _break_even(plot):
    positions = [
        float(value)
        for layer in plot.layers
        if isinstance(layer.geom, pn.geom_hline)
        for value in layer.geom.data["yintercept"]
    ]
    return positions


def test_plot_media_metrics_draws_the_requested_metric_as_bars():
    metrics = _metrics()
    expected = metrics["marginal_roi"].mean(("chain", "draw"))

    plot = plot_media_metrics(metrics, metric="marginal_roi")

    bars = plot.data.set_index("channel")
    np.testing.assert_allclose(
        bars.loc[["TV", "Search"], "estimate"], expected.sel(channel=["TV", "Search"]), rtol=1e-12, atol=0
    )
    assert list(plot.data["channel"].cat.categories) == ["TV", "Search"]
    assert bars.loc["Search", "value"] == f"{float(expected.sel(channel='Search')):.2f}"
    assert plot.labels.y.startswith("Marginal ROI, ")
    assert _break_even(plot) == [1.0]


def test_plot_media_metrics_labels_small_values_with_three_significant_digits():
    metrics = _returns()
    expected = metrics["effectiveness"].mean(("chain", "draw"))

    plot = plot_media_metrics(metrics, metric="effectiveness")

    labels = plot.data.set_index("channel")["value"]
    assert labels.loc["TV"] == f"{float(expected.sel(channel='TV')):.3g}"
    assert len(labels.loc["TV"].replace("0.", "").lstrip("0")) == 3
    # Only returns have a break-even point.
    assert _break_even(plot) == []


def test_plot_media_metrics_compares_labeled_results_in_order():
    labeled = {"Prior": _metrics(seed=1), "Posterior": _metrics(seed=2)}

    plot = plot_media_metrics(labeled, metric="marginal_roi")

    assert list(plot.data["result"].cat.categories) == ["Prior", "Posterior"]
    assert len(plot.data) == 4


def test_plot_media_metrics_uses_the_requested_interval_probability():
    metrics = _metrics()
    draws = metrics["marginal_roi"].transpose("channel", "chain", "draw").values.reshape(2, -1)
    expected = np.quantile(draws, [0.25, 0.75], axis=1).T

    plot = plot_media_metrics(metrics, metric="marginal_roi", ci_prob=0.5)

    frame = plot.data.set_index("channel").loc[["TV", "Search"]]
    np.testing.assert_allclose(frame[["lower", "upper"]].to_numpy(), expected, rtol=1e-12, atol=0)
    assert plot.labels.y == "Marginal ROI, 50% interval"


def _many_metrics(count):
    labels = [f"Channel {index:02d}" for index in range(count)]
    spend = np.arange(1.0, count + 1.0)
    increments = spend * (2.0 + 0.1 * _draws(2, 5, count, seed=11))
    marginal = 0.01 * spend * (1.5 + 0.1 * _draws(2, 5, count, seed=12))
    metrics = xr.Dataset(
        {
            "incremental_response": (("chain", "draw", "channel"), increments),
            "marginal_response": (("chain", "draw", "channel"), marginal),
            "reference_spend": (("channel",), spend),
            "incremental_spend": (("channel",), 0.01 * spend),
            "marginal_roi": (("chain", "draw", "channel"), marginal / (0.01 * spend)),
            "cost_per_incremental_response": (("chain", "draw", "channel"), spend / increments),
        },
        coords={"chain": [0, 1], "draw": np.arange(5), "channel": labels},
    )
    return metrics


@pytest.mark.parametrize(
    ("metric", "numerator", "denominator"),
    [
        ("incremental_response", "incremental_response", None),
        ("marginal_roi", "marginal_response", "incremental_spend"),
        ("cost_per_incremental_response", "reference_spend", "incremental_response"),
    ],
)
def test_plot_media_metrics_pools_the_channels_left_out_from_their_parts(metric, numerator, denominator):
    metrics = _many_metrics(30)
    hidden = [f"Channel {index:02d}" for index in range(5)]
    total = metrics[numerator].sel(channel=hidden).sum("channel")
    pooled = total if denominator is None else total / metrics[denominator].sel(channel=hidden).sum("channel")
    # Cost per incremental response is drawn at its median and every other metric at its mean.
    expected = float(pooled.median() if metric == "cost_per_incremental_response" else pooled.mean())

    plot = plot_media_metrics(
        metrics, metric=metric, channels=[label for label in metrics["channel"].values if label not in hidden]
    )

    bars = plot.data.set_index("channel")
    np.testing.assert_allclose(bars.loc["Other channels", "estimate"], expected, rtol=1e-12, atol=0)
    assert list(plot.data["channel"].cat.categories)[-1] == "Other channels"


def test_plot_media_metrics_draws_cost_per_incremental_response_at_its_median():
    metrics = _many_metrics(3)
    expected = metrics["cost_per_incremental_response"].median(("chain", "draw"))

    plot = plot_media_metrics(metrics, metric="cost_per_incremental_response")

    bars = plot.data.set_index("channel")["estimate"]
    np.testing.assert_allclose(bars.loc[list(expected["channel"].values)], expected.values, rtol=1e-12, atol=0)


def test_plot_media_metrics_leaves_out_the_pooled_bar_for_other_metrics():
    metrics = _many_metrics(30).assign(effectiveness=lambda dataset: dataset["marginal_roi"] * 2.0)

    plot = plot_media_metrics(metrics, metric="effectiveness", channels=["Channel 29", "Channel 28"])

    assert "Other channels" not in set(plot.data["channel"])


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"metric": "lift"}, ValueError, "metrics is missing 'lift'"),
        ({"metric": 3}, TypeError, "metric must be a string"),
        ({"metric": "spend_share"}, ValueError, "must have chain and draw axes"),
        ({"metrics": {}}, ValueError, "metrics must hold at least one result"),
        ({"metrics": [1.0]}, TypeError, "metrics must be an xarray Dataset or a mapping"),
        (
            {"metrics": xr.Dataset({"marginal_roi": (("chain", "draw"), np.ones((1, 2)))})},
            ValueError,
            "must have a channel axis",
        ),
        ({"ci_prob": 1.5}, ValueError, "ci_prob must be between 0 and 1"),
    ],
)
def test_plot_media_metrics_rejects_invalid_arguments(options, error, message):
    arguments = {"metrics": _metrics(), "metric": "marginal_roi"} | options

    with pytest.raises(error, match=message):
        plot_media_metrics(arguments.pop("metrics"), **arguments)


def _bars(plot):
    frame = plot.data.set_index(["result", "channel"])
    return frame


def test_plot_media_metrics_draws_roi_by_default_from_the_most_spending_down():
    metrics = _metrics()
    expected = metrics["roi"].mean(("chain", "draw"))

    plot = plot_media_metrics(metrics)

    bars = _bars(plot).loc[""]
    np.testing.assert_allclose(
        bars.loc[["TV", "Search"], "estimate"], expected.sel(channel=["TV", "Search"]), rtol=1e-12, atol=0
    )
    assert (bars["lower"] < bars["estimate"]).all()
    assert (bars["estimate"] < bars["upper"]).all()
    assert list(plot.data["channel"].cat.categories) == ["TV", "Search"]
    assert bars.loc["TV", "value"] == f"{float(expected.sel(channel='TV')):.2f}"
    assert _break_even(plot) == [1.0]


def test_plot_media_metrics_draws_pale_bars_inside_solid_outlines():
    figure = plot_media_metrics(_metrics()).draw()

    bars = next(collection for collection in figure.axes[0].collections if len(collection.get_paths()) == 2)
    # plotnine stores colors in eight bits, so the fill's opacity comes back to within one step of 255.
    np.testing.assert_allclose(bars.get_facecolor()[:, 3], [0.22, 0.22], rtol=0, atol=1 / 255)
    np.testing.assert_allclose(bars.get_edgecolor()[:, 3], [1.0, 1.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(bars.get_facecolor()[:, :3], bars.get_edgecolor()[:, :3], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_media_metrics_moves_or_leaves_out_the_break_even_line():
    moved = plot_media_metrics(_metrics(), break_even=2.5)
    left_out = plot_media_metrics(_metrics(), break_even=None)

    assert _break_even(moved) == [2.5]
    assert _break_even(left_out) == []


def test_plot_media_metrics_compares_labeled_roi_side_by_side():
    prior = _metrics(1).isel(chain=[0], draw=slice(0, 25))
    posterior = _metrics(2)
    expected = [float(prior["roi"].sel(channel="TV").mean()), float(posterior["roi"].sel(channel="TV").mean())]

    plot = plot_media_metrics({"Prior": prior, "Posterior": posterior})

    bars = _bars(plot)
    np.testing.assert_allclose(
        [bars.loc[("Prior", "TV"), "estimate"], bars.loc[("Posterior", "TV"), "estimate"]], expected, rtol=1e-12, atol=0
    )
    assert list(plot.data["result"].cat.categories) == ["Prior", "Posterior"]


def test_plot_media_metrics_gives_other_axes_panels_on_one_scale():
    returns = np.broadcast_to(np.array([[3.0, 1.0], [2.0, 2.0]]), (2, 4, 2, 2))
    plan = xr.Dataset(
        {
            "roi": (("chain", "draw", "allocation", "channel"), returns),
            "spend": (("allocation", "channel"), [[100.0, 50.0], [140.0, 10.0]]),
        },
        coords={
            "chain": [0, 1],
            "draw": np.arange(4),
            "allocation": ["reference", "optimized"],
            "channel": ["TV", "Search"],
        },
    )

    plot = plot_media_metrics(plan)

    estimates = plot.data.set_index(["allocation", "channel"])["estimate"]
    assert estimates.loc[("optimized", "Search")] == 2.0
    assert estimates.loc[("reference", "TV")] == 3.0
    assert isinstance(plot.facet, pn.facet_wrap)
    assert plot.facet.free == {"x": False, "y": False}


def _many_returns(count, *, increments=True):
    labels = [f"Channel {index:02d}" for index in range(count)]
    spend = np.arange(1.0, count + 1.0)
    returns = 2.0 + 0.1 * _draws(2, 5, count, seed=9)
    variables = {"roi": (("chain", "draw", "channel"), returns), "reference_spend": (("channel",), spend)}
    if increments:
        variables["incremental_response"] = (("chain", "draw", "channel"), returns * spend)
    metrics = xr.Dataset(variables, coords={"chain": [0, 1], "draw": np.arange(5), "channel": labels})
    return metrics


def test_plot_media_metrics_shows_every_channel_and_widens_past_the_default_figure():
    wide = plot_media_metrics(_many_returns(30))
    narrow = plot_media_metrics(_metrics())

    figure = wide.draw()
    assert set(wide.data["channel"]) == {f"Channel {index:02d}" for index in range(30)}
    assert isinstance(wide, _ScrollingPlot)
    np.testing.assert_allclose(figure.get_size_inches(), [1.5 + 30 * 0.8, 7.0], rtol=1e-12, atol=0)
    assert type(narrow) is pn.ggplot
    plt.close(figure)


def _long_returns(count):
    names = [f"social_media_meta_dynamic_brand_world_cup_{index:03d}" for index in range(count)]
    metrics = _many_returns(count).assign_coords(channel=names)
    return metrics


def _ticks(plot):
    figure = plot.draw()
    labels = [(label.get_text(), label.get_rotation()) for label in figure.axes[0].get_xticklabels()]
    return figure, labels


def test_plot_media_metrics_tilts_and_shortens_long_names_like_meridian():
    figure, labels = _ticks(plot_media_metrics(_long_returns(4)))

    assert labels[0] == ("social_media_\u2026world_cup_003", 45.0)
    assert len({text for text, _ in labels}) == 4
    assert all(len(text) == 27 and rotation == 45.0 for text, rotation in labels)
    np.testing.assert_allclose(figure.get_size_inches(), [12.0, 7.0], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_media_metrics_keeps_the_default_height_for_many_long_names():
    figure, _ = _ticks(plot_media_metrics(_long_returns(40)))

    np.testing.assert_allclose(figure.get_size_inches(), [1.5 + 40 * 0.8, 7.0], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_media_metrics_lengthens_shortened_names_until_they_differ():
    names = ["social_media_meta_dynamic_brand_a_world_cup_2026", "social_media_meta_dynamic_brand_b_world_cup_2026"]
    metrics = _metrics().assign_coords(channel=names)

    figure, labels = _ticks(plot_media_metrics(metrics))

    texts = [text for text, _ in labels]
    assert len(set(texts)) == 2
    assert all("\u2026" in text for text in texts)
    plt.close(figure)


def test_plot_response_curves_wrap_long_names_in_panel_titles_and_legends():
    curves = _curves().assign_coords(channel=["tv_linear_national_prime_time_sports", "search_google_non_brand"])
    plan = _plan([150.0, 25.0], channels=("tv_linear_national_prime_time_sports", "search_google_non_brand"))

    paneled = plot_response_curves(curves, plan=plan)
    legended = plot_response_curves(curves, combine=True)

    assert list(paneled.data["panel"].cat.categories) == [
        "tv_linear_national_prime_\ntime_sports",
        "search_google_non_brand",
    ]
    colors = next(scale for scale in legended.scales if "color" in scale.aesthetics)
    assert colors.labels == ["tv_linear_national_prime_\ntime_sports", "search_google_non_brand"]


def test_plot_media_metrics_pools_left_out_roi_into_a_spend_weighted_bar():
    metrics = _many_returns(30)
    hidden = [f"Channel {index:02d}" for index in range(5)]
    pooled = metrics["incremental_response"].sel(channel=hidden).sum("channel") / metrics["reference_spend"].sel(
        channel=hidden
    ).sum("channel")
    expected = float(pooled.mean())

    plot = plot_media_metrics(metrics, channels=[label for label in metrics["channel"].values if label not in hidden])

    bars = _bars(plot).loc[""]
    assert list(plot.data["channel"].cat.categories)[:3] == ["Channel 29", "Channel 28", "Channel 27"]
    assert list(plot.data["channel"].cat.categories)[-1] == "Other channels"
    np.testing.assert_allclose(bars.loc["Other channels", "estimate"], expected, rtol=1e-12, atol=0)


def test_plot_media_metrics_leaves_out_pooled_roi_without_incremental_responses():
    plot = plot_media_metrics(_many_returns(30, increments=False), channels=["Channel 29", "Channel 28"])

    assert set(plot.data["channel"]) == {"Channel 29", "Channel 28"}


@pytest.mark.parametrize(
    ("metrics", "options", "error", "message"),
    [
        (xr.DataArray([1.0]), {}, TypeError, "metrics must be an xarray Dataset or a mapping"),
        (xr.Dataset({"spend": (("channel",), [1.0])}), {}, ValueError, "metrics is missing 'roi'"),
        ({}, {}, ValueError, "metrics must hold at least one result"),
        (_metrics(), {"break_even": "1"}, TypeError, "break_even must be a number or None, got str"),
        (_metrics(), {"break_even": True}, TypeError, "break_even must be a number or None, got bool"),
        (_metrics(), {"break_even": float("inf")}, ValueError, "break_even must be finite"),
    ],
)
def test_plot_media_metrics_rejects_invalid_roi_arguments(metrics, options, error, message):
    with pytest.raises(error, match=message):
        plot_media_metrics(metrics, **options)


def test_plot_response_curves_mark_reference_spending_and_dash_beyond_it():
    curves = _curves()
    at_reference = curves["incremental_response"].sel(multiplier=1.0).mean(("chain", "draw")).values

    plot = plot_response_curves(curves)

    points = _layer_data(plot, pn.geom_point).set_index("channel").loc[["TV", "Search"]]
    lines = _layer_data(plot, pn.geom_line)
    beyond = lines[(lines["style"] == "Above reference spend") & (lines["channel"] == "TV")]
    np.testing.assert_allclose(points["spend"], [100.0, 50.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(points["estimate"], at_reference, rtol=1e-12, atol=0)
    assert beyond["spend"].min() == 100.0
    assert beyond["spend"].max() == 200.0


def test_plot_response_curves_mark_the_plan_spending_on_each_channel_panel():
    curves = _curves()
    means = curves["incremental_response"].mean(("chain", "draw"))
    expected = [
        float(means.sel(channel="TV", multiplier=1.0)),
        float(means.sel(channel="TV", multiplier=1.5)),
        float(means.sel(channel="Search", multiplier=1.0)),
        float(means.sel(channel="Search", multiplier=0.5)),
    ]

    plot = plot_response_curves(curves, plan=_plan([150.0, 25.0]))

    points = _layer_data(plot, pn.geom_point).sort_values(["channel", "level"])
    np.testing.assert_allclose(points["spend"], [100.0, 150.0, 50.0, 25.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(points["estimate"], expected, rtol=1e-12, atol=0)
    assert list(points["level"]) == ["Reference spend", "Optimized spend"] * 2
    assert isinstance(plot.facet, pn.facet_wrap)


def test_plot_response_curves_dash_the_curve_outside_the_plan_bounds():
    curves = _curves()
    means = curves["incremental_response"].sel(channel="TV").mean(("chain", "draw"))
    # The lower bound of 60 lies between the grid points at 50 and 100.
    expected = float(means.sel(multiplier=0.5)) + 0.2 * float(means.sel(multiplier=1.0) - means.sel(multiplier=0.5))

    plot = plot_response_curves(curves, plan=_plan([150.0, 25.0]))

    lines = _layer_data(plot, pn.geom_line)
    television = lines[lines["channel"] == "TV"]
    within = television[television["style"] == "Within bounds"]
    outside = television[television["style"] == "Outside bounds"]
    assert (within["spend"].min(), within["spend"].max()) == (60.0, 140.0)
    assert set(outside["spend"]) == {0.0, 50.0, 60.0, 140.0, 150.0, 200.0}
    np.testing.assert_allclose(within["estimate"].iloc[0], expected, rtol=1e-12, atol=0)


def test_plot_response_curves_name_plan_spending_beyond_the_curves():
    plot = plot_response_curves(_curves(), plan=_plan([250.0, 25.0], upper=(300.0, 75.0)))

    points = _layer_data(plot, pn.geom_point)
    assert len(points) == 3
    assert "Spending beyond the curves is not marked for TV." in plot.labels.caption


def test_plot_response_curves_keep_every_other_spending_break_on_panels():
    combined = plot_response_curves(_curves(), combine=True)
    planned = plot_response_curves(_curves(), plan=_plan([150.0, 25.0]))

    shared = next(scale for scale in combined.scales if "x" in scale.aesthetics)
    panels = next(scale for scale in planned.scales if "x" in scale.aesthetics)
    assert shared.breaks is True
    assert panels.breaks((0.0, 800.0)) == [0.0, 400.0, 800.0]
    assert panels.breaks((0.0, 300.0)) == [0.0, 100.0, 200.0, 300.0]


def test_plot_response_curves_give_each_channel_a_panel_unless_combined():
    paneled = plot_response_curves(_curves())
    combined = plot_response_curves(_curves(), combine=True)

    assert list(paneled.facet.vars) == ["panel"]
    assert paneled.facet.free == {"x": True, "y": True}
    assert (paneled.guides.color, paneled.guides.fill) == ("none", "none")
    assert isinstance(combined.facet, pn.facet_null)
    assert combined.guides.color is None


def test_plot_response_curves_reject_a_combine_that_is_not_a_bool():
    with pytest.raises(TypeError, match="combine must be a bool, got str"):
        plot_response_curves(_curves(), combine="yes")


def test_plot_response_curves_keep_each_channel_color_for_a_subset():
    plot = plot_response_curves(_curves(), channels=["Search"])

    assert _fills(plot)[1]["Search"] == "#fa7c17"


def test_plot_response_curves_show_only_the_channels_the_plan_holds():
    plan = _plan([150.0], reference=(100.0,), lower=(60.0,), upper=(140.0,), channels=("TV",))

    plot = plot_response_curves(_curves(), plan=plan)

    assert set(plot.data["channel"]) == {"TV"}


@pytest.mark.parametrize(
    ("plan", "error", "message"),
    [
        (_plan([150.0, 25.0], reference=(120.0, 50.0)), ValueError, "plan must start from the reference spending"),
        (_plan([150.0, 25.0], channels=("Radio", "Print")), ValueError, "plan must share channels with curves"),
        (_plan([150.0, 25.0]).drop_vars("upper_bound"), ValueError, "plan is missing 'upper_bound'"),
        (_plan([150.0, 25.0]).drop_vars("spend"), ValueError, "plan is missing 'spend'"),
        (xr.DataTree(), TypeError, "plan must be an xarray Dataset"),
    ],
)
def test_plot_response_curves_reject_plans_that_do_not_match_the_curves(plan, error, message):
    with pytest.raises(error, match=message):
        plot_response_curves(_curves(), plan=plan)


def test_plot_frequency_curves_mark_the_best_frequency():
    plot = plot_frequency_curves(_frequency_curves())

    assert _layer_data(plot, pn.geom_point)["frequency"].tolist() == [2.0]


def _returns(*, outcome=None):
    spend = np.array([100.0, 300.0, 50.0])
    increments = spend * np.array([4.0, 2.0, 1.0]) * (1.0 + 0.05 * _draws(2, 40, 3, seed=13))
    marginal = 0.01 * spend * np.array([3.0, 1.0, 0.5]) * (1.0 + 0.05 * _draws(2, 40, 3, seed=14))
    metrics = xr.Dataset(
        {
            "incremental_response": (("chain", "draw", "channel"), increments),
            "roi": (("chain", "draw", "channel"), increments / spend),
            "marginal_response": (("chain", "draw", "channel"), marginal),
            "marginal_roi": (("chain", "draw", "channel"), marginal / (0.01 * spend)),
            "reference_spend": (("channel",), spend),
            "incremental_spend": (("channel",), 0.01 * spend),
            "exposure": (("channel",), 20.0 * spend),
            "effectiveness": (("chain", "draw", "channel"), increments / (20.0 * spend)),
        },
        coords={"chain": [0, 1], "draw": np.arange(40), "channel": ["TV", "Search", "Radio"]},
        attrs={} if outcome is None else {"outcome": outcome},
    )
    return metrics


def _plan_returns():
    metrics = _returns()
    spend = xr.DataArray(
        [[100.0, 300.0, 50.0], [200.0, 200.0, 50.0]],
        dims=("allocation", "channel"),
        coords={"allocation": ["reference", "optimized"], "channel": ["TV", "Search", "Radio"]},
    )
    ratios = spend / spend.sel(allocation="reference")
    plan = xr.Dataset(
        {
            "incremental_response": metrics["incremental_response"] * ratios,
            "roi": metrics["roi"] * xr.ones_like(ratios),
            "marginal_roi": metrics["marginal_roi"] / ratios,
            "spend": spend,
        }
    )
    return plan


def _shares(plot):
    shares = plot.data.pivot(index="channel", columns="measure", values="share")
    return shares


def test_plot_spend_vs_contribution_pairs_each_share_of_spending_with_its_share_of_response():
    metrics = _returns(outcome="revenue")
    spend = metrics["reference_spend"]
    responses = metrics["incremental_response"].mean(("chain", "draw"))
    expected_spend = (spend / spend.sum()).sel(channel=["Search", "TV", "Radio"]).values
    expected_response = (responses / responses.sum()).sel(channel=["Search", "TV", "Radio"]).values
    expected_roi = (responses / spend).sel(channel=["Search", "TV", "Radio"]).values

    plot = plot_spend_vs_contribution(metrics)

    shares = _shares(plot).loc[["Search", "TV", "Radio"]]
    np.testing.assert_allclose(shares["Share of spend"], expected_spend, rtol=1e-12, atol=0)
    np.testing.assert_allclose(shares["Share of incremental revenue"], expected_response, rtol=1e-12, atol=0)
    assert list(plot.data["channel"].cat.categories) == ["Search", "TV", "Radio"]
    labels = _layer_data(plot, pn.geom_text).set_index("channel")
    assert list(labels.loc[["Search", "TV", "Radio"], "text"]) == [f"ROI {value:.2f}" for value in expected_roi]


def test_plot_spend_vs_contribution_hatches_the_spending_bars():
    figure = plot_spend_vs_contribution(_returns()).draw()

    hatches = [collection.get_hatch() for collection in figure.axes[0].collections]
    assert hatches.count("///") == 1
    assert None in hatches
    plt.close(figure)


def test_plot_spend_vs_contribution_puts_each_pair_of_bars_side_by_side():
    figure = plot_spend_vs_contribution(_returns()).draw()

    spans = [
        sorted((path.vertices[:, 0].min(), path.vertices[:, 0].max()) for path in collection.get_paths())
        for collection in figure.axes[0].collections
    ]
    for spend, response in zip(*spans, strict=True):
        assert spend[1] <= response[0] + 1e-9 or response[1] <= spend[0] + 1e-9
    plt.close(figure)


def test_plot_spend_vs_contribution_sums_groups_and_pools_the_channels_left_out():
    metrics = _returns()
    groups = xr.DataArray([0.25, 0.75], dims="group", coords={"group": ["north", "south"]})
    grouped = metrics.assign(incremental_response=metrics["incremental_response"] * groups)
    responses = metrics["incremental_response"].mean(("chain", "draw"))
    spend = metrics["reference_spend"]
    hidden = ["Search", "Radio"]
    expected = [
        float(spend.sel(channel=hidden).sum() / spend.sum()),
        float(responses.sel(channel=hidden).sum() / responses.sum()),
    ]

    plot = plot_spend_vs_contribution(grouped, channels=["TV"])

    shares = _shares(plot)
    np.testing.assert_allclose(
        shares.loc["Other channels", ["Share of spend", "Share of incremental response"]], expected, rtol=1e-6, atol=0
    )
    assert list(plot.data["channel"].cat.categories) == ["TV", "Other channels"]


def test_plot_spend_vs_contribution_gives_each_allocation_a_panel():
    plan = _plan_returns()
    optimized = plan["spend"].sel(allocation="optimized")
    expected = (optimized / optimized.sum()).sel(channel="TV").item()

    plot = plot_spend_vs_contribution(plan)

    shares = plot.data.set_index(["allocation", "channel", "measure"])["share"]
    np.testing.assert_allclose(shares.loc[("optimized", "TV", "Share of spend")], expected, rtol=1e-12, atol=0)
    assert list(plot.data["allocation"].cat.categories) == ["reference", "optimized"]
    assert isinstance(plot.facet, pn.facet_wrap)


def test_plot_roi_bubbles_places_each_channel_at_its_point_estimates():
    metrics = _returns()
    expected = metrics[["roi", "marginal_roi"]].mean(("chain", "draw"))

    plot = plot_roi_bubbles(metrics)

    frame = plot.data.set_index("channel").loc[["TV", "Search", "Radio"]]
    np.testing.assert_allclose(frame["roi"], expected["roi"].values, rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame["marginal_roi"], expected["marginal_roi"].values, rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame["spend"], metrics["reference_spend"].values, rtol=1e-12, atol=0)
    assert _break_even(plot) == [1.0]
    assert (plot.labels.x, plot.labels.y) == ("ROI", "Marginal ROI")


def test_plot_roi_bubbles_sizes_bubbles_by_area_from_zero_spending():
    spend = np.sort(_returns()["reference_spend"].values)
    expected = 24.0 * np.sqrt(spend / spend.max())

    plot = plot_roi_bubbles(_returns())

    figure = plot.draw()
    bubbles = next(layer for layer in plot.layers if isinstance(layer.geom, pn.geom_point))
    np.testing.assert_allclose(np.sort(bubbles.data["size"].to_numpy()), expected, rtol=1e-12, atol=0)
    plt.close(figure)


def _fills(plot):
    scale = next(scale for scale in plot.scales if "fill" in scale.aesthetics)
    colors = scale.palette(len(scale.breaks))
    return scale.breaks, colors


def test_plot_roi_bubbles_colors_each_channel_in_a_legend_without_sizes():
    plot = plot_roi_bubbles(_returns())

    breaks, colors = _fills(plot)
    assert breaks == ["Search", "TV", "Radio"]
    # Colors follow the results' channel order so a channel keeps its color across plots.
    assert [colors[channel] for channel in ["TV", "Search", "Radio"]] == ["#2a2eec", "#fa7c17", "#328c06"]
    assert plot.labels.fill == "Channel"
    assert plot.guides.size == "none"


def test_plot_roi_bubbles_keeps_the_ten_channels_with_the_most_spending():
    labels = [f"Channel {index:02d}" for index in range(30)]
    metrics = _many_metrics(30).assign(roi=lambda dataset: dataset["incremental_response"] / dataset["reference_spend"])

    plot = plot_roi_bubbles(metrics)

    assert _fills(plot)[0] == list(reversed(labels[20:]))
    assert set(plot.data["channel"]) == set(labels[20:])
    assert plot.labels.caption == (
        "Showing the 10 of 30 channels with the largest spending. Pass channels to choose others."
    )


def test_plot_roi_bubbles_draws_effectiveness_without_a_horizontal_line():
    metrics = _returns(outcome="revenue")
    expected = metrics["effectiveness"].mean(("chain", "draw"))

    plot = plot_roi_bubbles(metrics, metric="effectiveness", channels=["TV", "Search"])

    frame = plot.data.set_index("channel")
    np.testing.assert_allclose(
        frame.loc[["TV", "Search"], "effectiveness"], expected.sel(channel=["TV", "Search"]), rtol=1e-12, atol=0
    )
    assert set(plot.data["channel"]) == {"TV", "Search"}
    assert _break_even(plot) == []
    assert plot.labels.y == "Effectiveness"


def test_plot_roi_bubbles_shows_only_the_requested_channels():
    plot = plot_roi_bubbles(_returns(), channels=["TV"])

    assert set(plot.data["channel"]) == {"TV"}
    assert plot.labels.caption == ""


def test_plot_roi_bubbles_keeps_each_channel_color_for_a_subset():
    plot = plot_roi_bubbles(_returns(), channels=["Radio"])

    assert _fills(plot)[1]["Radio"] == "#328c06"


@pytest.mark.parametrize(
    ("draw", "metrics", "options", "error", "message"),
    [
        (plot_spend_vs_contribution, _metrics(), {}, ValueError, "metrics is missing 'incremental_response'"),
        (
            plot_spend_vs_contribution,
            _returns().drop_vars("reference_spend"),
            {},
            ValueError,
            "metrics must record 'reference_spend' or 'spend'",
        ),
        (plot_roi_bubbles, _returns().drop_vars("marginal_roi"), {}, ValueError, "missing 'marginal_roi'"),
        (plot_roi_bubbles, _returns(), {"break_even": "1"}, TypeError, "break_even must be a number"),
        (plot_roi_bubbles, _returns(), {"channels": ["Print"]}, ValueError, "channels has no 'Print'"),
        (plot_roi_bubbles, _returns(), {"metric": "roi"}, ValueError, "metric must differ from 'roi'"),
        (plot_roi_bubbles, _returns(), {"metric": 3}, TypeError, "metric must be a string"),
        (plot_roi_bubbles, _metrics(), {"metric": "effectiveness"}, ValueError, "missing 'effectiveness'"),
        (
            plot_roi_bubbles,
            _returns().drop_vars("reference_spend"),
            {},
            ValueError,
            "metrics must record 'reference_spend' or 'spend'",
        ),
    ],
)
def test_share_and_bubble_plots_reject_invalid_arguments(draw, metrics, options, error, message):
    with pytest.raises(error, match=message):
        draw(metrics, **options)


def _grouped_increments(count):
    labels = [f"g{index}" for index in range(count)]
    scales = np.arange(1.0, count + 1.0)[:, None]
    increments = scales * np.array([2.0, 1.0]) * (1.0 + 0.05 * _draws(2, 40, count, 2, seed=15))
    metrics = xr.Dataset(
        {"incremental_response": (("chain", "draw", "group", "channel"), increments)},
        coords={"chain": [0, 1], "draw": np.arange(40), "group": labels, "channel": ["TV", "Search"]},
        attrs={"outcome": "revenue"},
    )
    return metrics


def test_plot_media_metrics_gives_the_largest_groups_panels():
    metrics = _grouped_increments(5)

    default = plot_media_metrics(metrics, metric="incremental_response", ci_prob=0.9)
    chosen = plot_media_metrics(metrics, metric="incremental_response", coords={"group": ["g0", "g1"]})
    every = plot_media_metrics(metrics, metric="incremental_response", n_groups=None)

    assert set(default.data["group"]) == {"g2", "g3", "g4"}
    assert default.labels.caption == "Showing the 3 of 5 groups with the largest values. Pass coords to choose others."
    assert default.labels.y == "Incremental revenue, 90% interval"
    assert set(chosen.data["group"]) == {"g0", "g1"}
    assert set(every.data["group"]) == {f"g{index}" for index in range(5)}


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: plot_media_metrics({"Prior": _metrics(1), "Posterior": _metrics(2)}, metric="marginal_roi"),
            id="metrics",
        ),
        pytest.param(lambda: plot_media_metrics(_metrics()), id="roi"),
        pytest.param(lambda: plot_response_curves(_curves()), id="response curves"),
        pytest.param(lambda: plot_response_curves(_curves(), plan=_plan([150.0, 25.0])), id="response curves plan"),
        pytest.param(lambda: plot_frequency_curves(_frequency_curves()), id="frequency curves"),
        pytest.param(lambda: plot_spend_vs_contribution(_returns()), id="spend vs contribution"),
        pytest.param(lambda: plot_roi_bubbles(_returns()), id="roi bubbles"),
        pytest.param(lambda: plot_roi_bubbles(_returns(), metric="effectiveness"), id="roi bubbles effectiveness"),
        pytest.param(lambda: plot_roi_bubbles(_plan_returns()), id="roi bubbles plan"),
    ],
)
def test_media_plots_draw(build):
    figure = build().draw()

    assert figure.axes
    plt.close(figure)


def test_plot_response_curves_show_the_channels_with_the_most_spending():
    labels = [f"Channel {index:02d}" for index in range(12)]
    multipliers = np.array([0.0, 1.0, 2.0])
    reference_spend = np.arange(1, 13, dtype=float)
    curves = xr.Dataset(
        {
            "incremental_response": (
                ("chain", "draw", "channel", "multiplier"),
                np.broadcast_to(multipliers, (2, 40, 12, 3)) * reference_spend[:, None],
            ),
            "spend": (("channel", "multiplier"), reference_spend[:, None] * multipliers),
            "reference_spend": (("channel",), reference_spend),
        },
        coords={"chain": [0, 1], "draw": np.arange(40), "channel": labels, "multiplier": multipliers},
    )

    paneled = plot_response_curves(curves)
    combined = plot_response_curves(curves, combine=True)

    assert set(paneled.data["channel"]) == set(labels[2:])
    assert paneled.labels.caption.startswith("Showing the 10 of 12 channels with the largest spending")
    assert set(combined.data["channel"]) == set(labels[7:])
    assert combined.labels.caption.startswith("Showing the 5 of 12 channels with the largest spending")


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(
            lambda channels: plot_media_metrics(_metrics(), metric="marginal_roi", channels=channels), id="metrics"
        ),
        pytest.param(lambda channels: plot_response_curves(_curves(), channels=channels), id="response curves"),
        pytest.param(lambda channels: plot_media_metrics(_metrics(), channels=channels), id="roi"),
        pytest.param(lambda channels: plot_spend_vs_contribution(_returns(), channels=channels), id="spend"),
        pytest.param(lambda channels: plot_roi_bubbles(_returns(), channels=channels), id="bubbles"),
    ],
)
@pytest.mark.parametrize(
    ("channels", "error", "message"),
    [
        (["Print"], ValueError, "channels has no 'Print'"),
        ("TV", TypeError, "channels must be a sequence of names"),
    ],
)
def test_channel_plots_reject_unknown_channels(draw, channels, error, message):
    with pytest.raises(error, match=message):
        draw(channels)
