"""Tests for plots that break the response into its baseline and channel contributions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import pytest
import xarray as xr

from mmmjax import plot_contributions
from mmmjax.plotting._display import _ScrollingPlot


def _effects(channels=("TV", "Search", "Radio"), *, periods=0, groups=(), interactions=0.0, outcome=None):
    extra = {"time": pd.date_range("2024-01-01", periods=periods, freq="W-MON")} if periods else {}
    extra |= {"group": list(groups)} if groups else {}
    shape = (2, 20, *(len(labels) for labels in extra.values()))
    noise = 1.0 + 0.05 * np.random.default_rng(0).normal(size=(*shape, len(channels)))
    increments = np.arange(len(channels), 0, -1) * 10.0 * noise
    baseline = 100.0 * (1.0 + 0.05 * np.random.default_rng(1).normal(size=shape))
    reference = baseline + increments.sum(axis=-1) * (1.0 + interactions)
    dims = ("chain", "draw", *extra)
    effects = xr.Dataset(
        {
            "incremental_response": ((*dims, "channel"), increments),
            "baseline_response": (dims, baseline),
            "reference_response": (dims, reference),
        },
        coords={"chain": [0, 1], "draw": np.arange(20), **extra, "channel": list(channels)},
        attrs={} if outcome is None else {"outcome": outcome},
    )
    return effects


def _grouped(count, *, periods=0):
    effects = _effects(periods=periods, groups=[f"g{index}" for index in range(count)])
    scales = xr.DataArray(np.arange(1.0, count + 1.0), dims="group", coords={"group": effects["group"].values})
    scaled = effects.assign({name: effects[name] * scales for name in effects.data_vars})
    return scaled


def _rows(plot):
    rows = plot.data.set_index("name")
    return rows


def _layer_data(plot, geom):
    data = next(layer.geom.data for layer in plot.layers if isinstance(layer.geom, geom))
    return data


def test_plot_contributions_steps_from_the_baseline_through_each_channel_share():
    effects = _effects()
    reference = float(effects["reference_response"].mean())
    parts = {"Baseline": effects["baseline_response"]} | {
        channel: effects["incremental_response"].sel(channel=channel) for channel in ["TV", "Search", "Radio"]
    }
    shares = [float(values.mean()) / reference for values in parts.values()]
    expected_end = np.cumsum(shares)

    plot = plot_contributions(effects)

    rows = _rows(plot)
    assert list(rows.index) == ["Baseline", "TV", "Search", "Radio"]
    np.testing.assert_allclose(rows["share"], shares, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rows["end"], expected_end, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rows["start"], [0.0, *expected_end[:-1]], rtol=1e-12, atol=1e-15)
    assert list(rows["kind"]) == ["Baseline", "Increase", "Increase", "Increase"]
    assert rows.loc["TV", "text"] == f"{shares[1]:.1%} ({float(parts['TV'].mean()):.3g})"
    assert list(rows["position"]) == [4, 3, 2, 1]


def test_plot_contributions_splits_the_joint_increment_without_the_baseline():
    effects = _effects(outcome="revenue")
    joint = float((effects["reference_response"] - effects["baseline_response"]).mean())
    expected = float(effects["incremental_response"].sel(channel="TV").mean()) / joint

    plot = plot_contributions(effects, include_baseline=False)

    rows = _rows(plot)
    assert list(rows.index) == ["TV", "Search", "Radio"]
    np.testing.assert_allclose(rows.loc["TV", "share"], expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rows["end"].iloc[-1], 1.0, rtol=1e-12, atol=0)
    assert plot.labels.x == "Share of incremental revenue"


def test_plot_contributions_adds_a_row_for_interactions_that_move_the_total():
    small = plot_contributions(_effects(interactions=0.01))
    large = plot_contributions(_effects(interactions=0.2))

    effects = _effects(interactions=0.2)
    leftover = effects["reference_response"] - effects["baseline_response"]
    leftover = leftover - effects["incremental_response"].sum("channel")
    expected = float(leftover.mean() / effects["reference_response"].mean())
    assert "Channel interactions" not in set(small.data["name"])
    rows = _rows(large)
    assert list(rows.index)[-1] == "Channel interactions"
    np.testing.assert_allclose(rows.loc["Channel interactions", "share"], expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rows["end"].iloc[-1], 1.0, rtol=1e-12, atol=0)


def test_plot_contributions_sums_the_channels_left_out_into_one_row():
    effects = _effects()
    hidden = effects["incremental_response"].sel(channel=["Search", "Radio"]).sum("channel")
    expected = float(hidden.mean() / effects["reference_response"].mean())

    plot = plot_contributions(effects, channels=["TV"])

    rows = _rows(plot)
    assert list(rows.index) == ["Baseline", "TV", "Other channels"]
    np.testing.assert_allclose(rows.loc["Other channels", "share"], expected, rtol=1e-12, atol=0)
    assert rows.loc["Other channels", "kind"] == "Other"


def test_plot_contributions_sums_the_axes_that_by_leaves_out():
    effects = _grouped(3, periods=4)
    totals = effects.sum(("time", "group"))
    expected = float(totals["baseline_response"].mean() / totals["reference_response"].mean())

    plot = plot_contributions(effects)

    np.testing.assert_allclose(_rows(plot).loc["Baseline", "share"], expected, rtol=1e-12, atol=0)
    assert "group" not in plot.data.columns


def test_plot_contributions_names_the_outcome_on_its_axis():
    plot = plot_contributions(_effects(outcome="revenue"))

    assert plot.labels.x == "Share of revenue"


def test_plot_contributions_gives_the_largest_groups_panels_by_group():
    effects = _grouped(5)
    north = effects.sel(group="g0")
    expected = float(north["baseline_response"].mean() / north["reference_response"].mean())

    default = plot_contributions(effects, by="group")
    chosen = plot_contributions(effects, by="group", coords={"group": ["g0"]})
    every = plot_contributions(effects, by="group", n_groups=None)
    summed = plot_contributions(effects, coords={"group": ["g0"]})

    assert set(default.data["group"]) == {"g2", "g3", "g4"}
    assert default.labels.caption == (
        "Showing the 3 of 5 groups with the largest responses. Pass coords to choose others."
    )
    assert set(chosen.data["group"]) == {"g0"}
    assert chosen.labels.caption == ""
    assert set(every.data["group"]) == {f"g{index}" for index in range(5)}
    np.testing.assert_allclose(_rows(summed).loc["Baseline", "share"], expected, rtol=1e-12, atol=0)


def test_plot_contributions_grows_taller_for_many_channels():
    effects = _effects([f"Channel {index:02d}" for index in range(40)])

    plot = plot_contributions(effects)

    figure = plot.draw()
    assert isinstance(plot, _ScrollingPlot)
    np.testing.assert_allclose(figure.get_size_inches(), [12.0, 1.4 + 0.32 * 41], rtol=1e-12, atol=0)
    plt.close(figure)


def test_plot_contributions_by_time_stacks_channels_under_the_joint_increment():
    effects = _effects(periods=4)
    expected = effects["incremental_response"].mean(("chain", "draw")).transpose("time", "channel")
    joint = (effects["reference_response"] - effects["baseline_response"]).mean(("chain", "draw"))

    plot = plot_contributions(effects, by="time", include_baseline=False)

    stacked = plot.data.pivot(index="time", columns="channel", values="estimate")[["TV", "Search", "Radio"]]
    np.testing.assert_allclose(stacked.to_numpy(), expected.values, rtol=1e-12, atol=0)
    line = _layer_data(plot, pn.geom_line)
    np.testing.assert_allclose(line["estimate"], joint.values, rtol=1e-12, atol=0)
    assert set(line["series"]) == {"Joint incremental response"}
    assert "Baseline" not in set(plot.data["channel"])
    assert plot.labels.y == "Incremental response"
    assert plot.coordinates.limits.y is None


def test_plot_contributions_by_time_stacks_on_the_baseline_by_default():
    effects = _effects(periods=4, outcome="revenue")
    total = effects["reference_response"].mean(("chain", "draw"))

    plot = plot_contributions(effects, by="time")

    assert list(plot.data["channel"].cat.categories) == ["TV", "Search", "Radio", "Baseline"]
    line = _layer_data(plot, pn.geom_line)
    np.testing.assert_allclose(line["estimate"], total.values, rtol=1e-12, atol=0)
    assert set(line["series"]) == {"Total revenue"}
    assert plot.labels.y == "Revenue"
    low = 0.95 * float(effects["baseline_response"].mean(("chain", "draw")).min())
    np.testing.assert_allclose(plot.coordinates.limits.y[0], low, rtol=1e-12, atol=0)


def test_plot_contributions_by_time_sums_channels_past_the_tenth():
    effects = _effects([f"Channel {index:02d}" for index in range(12)], periods=3)
    hidden = [f"Channel {index:02d}" for index in range(10, 12)]
    expected = effects["incremental_response"].sel(channel=hidden).sum("channel").mean(("chain", "draw"))

    plot = plot_contributions(effects, by="time")

    frame = plot.data.set_index(["channel", "time"])["estimate"]
    np.testing.assert_allclose(frame.loc["Other channels"].to_numpy(), expected.values, rtol=1e-12, atol=0)
    assert plot.labels.caption == (
        "Showing the 10 of 12 channels with the largest total increments. Pass channels to choose others."
    )


def test_plot_contributions_by_time_keeps_each_channel_color_for_a_subset():
    plot = plot_contributions(_effects(periods=4), by="time", channels=["Radio"])

    scale = next(scale for scale in plot.scales if "fill" in scale.aesthetics)
    assert scale.palette(len(scale.breaks))["Radio"] == "#328c06"


def test_plot_contributions_by_time_and_group_stacks_group_panels():
    effects = _grouped(2, periods=3)
    expected = effects["incremental_response"].sum("group").mean(("chain", "draw"))

    national = plot_contributions(effects, by="time")
    regional = plot_contributions(effects, by=("time", "group"))

    frame = national.data.pivot(index="time", columns="channel", values="estimate")[["TV", "Search", "Radio"]]
    np.testing.assert_allclose(frame.to_numpy(), expected.transpose("time", "channel").values, rtol=1e-12, atol=0)
    assert "group" not in national.data.columns
    assert set(regional.data["group"]) == {"g0", "g1"}
    assert isinstance(regional.facet, pn.facet_wrap)


@pytest.mark.parametrize(
    ("effects", "options", "error", "message"),
    [
        (xr.DataArray([1.0]), {}, TypeError, "effects must be an xarray Dataset"),
        (xr.Dataset(), {}, ValueError, "effects is missing 'incremental_response'"),
        (_effects().isel(channel=0, drop=True), {}, ValueError, "must have a channel axis"),
        (_effects(), {"channels": ["Print"]}, ValueError, "channels has no 'Print'"),
        (_effects(), {"by": 3}, TypeError, "by must be an axis name or a sequence of them"),
        (_effects(), {"by": "region"}, ValueError, "by must name 'time' or 'group', got 'region'"),
        (_effects(), {"by": "time"}, ValueError, "by must name axes that effects keep, got 'time'"),
        (_effects(periods=2), {"include_baseline": 1}, TypeError, "include_baseline must be a bool, got int"),
        (_grouped(2), {"coords": {"region": ["g0"]}}, ValueError, "coords has no axis 'region'"),
        (_grouped(2), {"coords": {"group": ["g9"]}}, ValueError, "coords asks for labels"),
        (_grouped(2), {"n_groups": 0}, ValueError, "n_groups must be a positive integer"),
        (
            _effects().assign(reference_response=lambda dataset: -dataset["reference_response"]),
            {},
            ValueError,
            "effects must have a positive response to split into shares",
        ),
    ],
)
def test_plot_contributions_rejects_invalid_arguments(effects, options, error, message):
    with pytest.raises(error, match=message):
        plot_contributions(effects, **options)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: plot_contributions(_effects()), id="waterfall"),
        pytest.param(lambda: plot_contributions(_effects(), include_baseline=False), id="waterfall increments"),
        pytest.param(lambda: plot_contributions(_grouped(2), by="group"), id="waterfall groups"),
        pytest.param(lambda: plot_contributions(_effects(periods=4), by="time"), id="time"),
        pytest.param(
            lambda: plot_contributions(_effects(periods=4), by="time", include_baseline=True), id="time baseline"
        ),
        pytest.param(lambda: plot_contributions(_grouped(2, periods=4), by=("time", "group")), id="time groups"),
    ],
)
def test_plot_contributions_draws(build):
    figure = build().draw()

    assert figure.axes
    plt.close(figure)
