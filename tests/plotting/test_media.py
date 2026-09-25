"""Tests for plots of media effects, returns, and response curves."""

import matplotlib.pyplot as plt
import numpy as np
import plotnine as pn
import pytest
import xarray as xr

from mmmjax import (
    geometric_adstock,
    plot_adstock,
    plot_frequency_curves,
    plot_media_metrics,
    plot_response_curves,
    plot_roi,
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


def _retention(values):
    draws = np.broadcast_to(np.asarray(values, dtype=np.float32), (2, 3, len(values)))
    results = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"retention": (("chain", "draw", "channel"), draws)},
                coords={"chain": [0, 1], "draw": np.arange(3), "channel": ["TV", "Search"]},
            )
        }
    )
    return results


def _layer_data(plot, geom):
    data = next(layer.geom.data for layer in plot.layers if isinstance(layer.geom, geom))
    return data


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
    assert not [layer for layer in plot.layers if isinstance(layer.geom, pn.geom_hline)]


def test_plot_media_metrics_requires_a_metric():
    with pytest.raises(TypeError, match="metric"):
        plot_media_metrics(_metrics())


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
    expected = float(pooled.mean())

    plot = plot_media_metrics(
        metrics, metric=metric, channels=[label for label in metrics["channel"].values if label not in hidden]
    )

    bars = plot.data.set_index("channel")
    np.testing.assert_allclose(bars.loc["Other channels", "estimate"], expected, rtol=1e-12, atol=0)
    assert list(plot.data["channel"].cat.categories)[-1] == "Other channels"


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


def _break_even(plot):
    positions = [
        float(value)
        for layer in plot.layers
        if isinstance(layer.geom, pn.geom_hline)
        for value in layer.geom.data["yintercept"]
    ]
    return positions


def test_plot_roi_draws_each_channel_estimate_from_the_most_spending_down():
    metrics = _metrics()
    expected = metrics["roi"].mean(("chain", "draw"))

    plot = plot_roi(metrics)

    bars = _bars(plot).loc[""]
    np.testing.assert_allclose(
        bars.loc[["TV", "Search"], "estimate"], expected.sel(channel=["TV", "Search"]), rtol=1e-12, atol=0
    )
    assert (bars["lower"] < bars["estimate"]).all()
    assert (bars["estimate"] < bars["upper"]).all()
    assert list(plot.data["channel"].cat.categories) == ["TV", "Search"]
    assert bars.loc["TV", "value"] == f"{float(expected.sel(channel='TV')):.2f}"
    assert _break_even(plot) == [1.0]


def test_plot_roi_moves_or_leaves_out_the_break_even_line():
    moved = plot_roi(_metrics(), break_even=2.5)
    left_out = plot_roi(_metrics(), break_even=None)

    assert _break_even(moved) == [2.5]
    assert _break_even(left_out) == []


def test_plot_roi_compares_labeled_results_side_by_side():
    prior = _metrics(1).isel(chain=[0], draw=slice(0, 25))
    posterior = _metrics(2)
    expected = [float(prior["roi"].sel(channel="TV").mean()), float(posterior["roi"].sel(channel="TV").mean())]

    plot = plot_roi({"Prior": prior, "Posterior": posterior})

    bars = _bars(plot)
    np.testing.assert_allclose(
        [bars.loc[("Prior", "TV"), "estimate"], bars.loc[("Posterior", "TV"), "estimate"]], expected, rtol=1e-12, atol=0
    )
    assert list(plot.data["result"].cat.categories) == ["Prior", "Posterior"]


def test_plot_roi_gives_other_axes_panels_on_one_scale():
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

    plot = plot_roi(plan)

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


def test_plot_roi_shows_every_channel_and_widens_past_the_default_figure():
    wide = plot_roi(_many_returns(30))
    narrow = plot_roi(_metrics())

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


def test_plot_roi_tilts_and_shortens_long_names_like_meridian():
    figure, labels = _ticks(plot_roi(_long_returns(4)))

    assert labels[0] == ("social_media_\u2026world_cup_003", 45.0)
    assert len({text for text, _ in labels}) == 4
    assert all(len(text) == 27 and rotation == 45.0 for text, rotation in labels)
    np.testing.assert_allclose(figure.get_size_inches(), [12.0, 7.0], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_roi_keeps_the_default_height_for_many_long_names():
    figure, _ = _ticks(plot_roi(_long_returns(40)))

    np.testing.assert_allclose(figure.get_size_inches(), [1.5 + 40 * 0.8, 7.0], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_roi_lengthens_shortened_names_until_they_differ():
    names = ["social_media_meta_dynamic_brand_a_world_cup_2026", "social_media_meta_dynamic_brand_b_world_cup_2026"]
    metrics = _metrics().assign_coords(channel=names)

    figure, labels = _ticks(plot_roi(metrics))

    texts = [text for text, _ in labels]
    assert len(set(texts)) == 2
    assert all("\u2026" in text for text in texts)
    plt.close(figure)


def test_plot_response_curves_wrap_long_names_in_panel_titles_and_legends():
    curves = _curves().assign_coords(channel=["tv_linear_national_prime_time_sports", "search_google_non_brand"])
    plan = _plan([150.0, 25.0], channels=("tv_linear_national_prime_time_sports", "search_google_non_brand"))

    paneled = plot_response_curves(curves, plan=plan)
    legended = plot_response_curves(curves)

    assert list(paneled.data["panel"].cat.categories) == [
        "tv_linear_national_prime_\ntime_sports",
        "search_google_non_brand",
    ]
    colors = next(scale for scale in legended.scales if "color" in scale.aesthetics)
    assert colors.labels == ["tv_linear_national_prime_\ntime_sports", "search_google_non_brand"]


def test_plot_roi_pools_the_channels_left_out_into_a_spend_weighted_bar():
    metrics = _many_returns(30)
    hidden = [f"Channel {index:02d}" for index in range(5)]
    pooled = metrics["incremental_response"].sel(channel=hidden).sum("channel") / metrics["reference_spend"].sel(
        channel=hidden
    ).sum("channel")
    expected = float(pooled.mean())

    plot = plot_roi(metrics, channels=[label for label in metrics["channel"].values if label not in hidden])

    bars = _bars(plot).loc[""]
    assert list(plot.data["channel"].cat.categories)[:3] == ["Channel 29", "Channel 28", "Channel 27"]
    assert list(plot.data["channel"].cat.categories)[-1] == "Other channels"
    np.testing.assert_allclose(bars.loc["Other channels", "estimate"], expected, rtol=1e-12, atol=0)


def test_plot_roi_leaves_out_the_pooled_bar_without_incremental_responses():
    plot = plot_roi(_many_returns(30, increments=False), channels=["Channel 29", "Channel 28"])

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
def test_plot_roi_rejects_invalid_arguments(metrics, options, error, message):
    with pytest.raises(error, match=message):
        plot_roi(metrics, **options)


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


def test_plot_adstock_matches_normalized_geometric_weights():
    retention = np.array([0.5, 0.2])
    kernel = retention[None, :] ** np.arange(4)[:, None]
    expected = kernel / kernel.sum(axis=0)

    plot = plot_adstock(_retention(retention), geometric_adstock, parameters={"alpha": "retention"}, max_lag=3)

    weights = plot.data.pivot(index="lag", columns="channel", values="estimate")[["TV", "Search"]]
    np.testing.assert_allclose(weights.to_numpy(), expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"results": xr.Dataset()}, TypeError, "results must be an xarray DataTree"),
        ({"adstock": "geometric"}, TypeError, "adstock must be callable"),
        ({"parameters": {}}, ValueError, "parameters must map at least one adstock argument"),
        ({"max_lag": -1}, ValueError, "max_lag must be nonnegative"),
        ({"max_lag": True}, TypeError, "max_lag must be an integer"),
        ({"group": "both"}, ValueError, "group must be 'prior' or 'posterior'"),
        ({"group": "prior"}, ValueError, "results has no 'prior' group"),
        ({"parameters": {"alpha": "decay"}}, ValueError, "results has no posterior variable 'decay'"),
        (
            {"adstock": lambda media, alpha, max_lag: media[1:]},
            ValueError,
            "adstock must return an array shaped like its media input",
        ),
    ],
)
def test_plot_adstock_rejects_invalid_arguments(options, error, message):
    arguments = {
        "results": _retention([0.5, 0.2]),
        "adstock": geometric_adstock,
        "parameters": {"alpha": "retention"},
        "max_lag": 3,
    } | options

    with pytest.raises(error, match=message):
        plot_adstock(arguments.pop("results"), arguments.pop("adstock"), **arguments)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: plot_media_metrics({"Prior": _metrics(1), "Posterior": _metrics(2)}, metric="marginal_roi"),
            id="metrics",
        ),
        pytest.param(lambda: plot_roi(_metrics()), id="roi"),
        pytest.param(lambda: plot_response_curves(_curves()), id="response curves"),
        pytest.param(lambda: plot_response_curves(_curves(), plan=_plan([150.0, 25.0])), id="response curves plan"),
        pytest.param(lambda: plot_frequency_curves(_frequency_curves()), id="frequency curves"),
        pytest.param(
            lambda: plot_adstock(
                _retention([0.5, 0.2]), geometric_adstock, parameters={"alpha": "retention"}, max_lag=3
            ),
            id="adstock",
        ),
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

    plot = plot_response_curves(curves)

    assert set(plot.data["channel"]) == set(labels[3:])
    assert plot.labels.caption.startswith("Showing the 9 of 12 channels with the largest spending")


def test_plot_adstock_shows_the_first_channels_of_many():
    labels = [f"Channel {index:02d}" for index in range(12)]
    draws = np.full((2, 3, 12), 0.5, dtype=np.float32)
    results = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"retention": (("chain", "draw", "channel"), draws)},
                coords={"chain": [0, 1], "draw": np.arange(3), "channel": labels},
            )
        }
    )

    plot = plot_adstock(results, geometric_adstock, parameters={"alpha": "retention"}, max_lag=2)

    assert set(plot.data["channel"]) == set(labels[:10])
    assert plot.labels.caption.startswith("Showing the first 10 of 12 channels")


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(
            lambda channels: plot_media_metrics(_metrics(), metric="marginal_roi", channels=channels), id="metrics"
        ),
        pytest.param(lambda channels: plot_response_curves(_curves(), channels=channels), id="response curves"),
        pytest.param(lambda channels: plot_roi(_metrics(), channels=channels), id="roi"),
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
