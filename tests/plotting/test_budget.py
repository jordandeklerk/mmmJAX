"""Tests for plots of optimized budget allocations."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from mmmjax import plot_budget_response, plot_budget_spend
from mmmjax.plotting._display import _ScrollingPlot


def _plan(response_change=None, *, spend=None, increments=None, axes=("chain", "draw"), coords=None, channels=None):
    labels = channels or ["TV", "Search", "Radio"]
    spend = spend if spend is not None else [[100.0, 50.0, 30.0], [140.0, 20.0, 30.0]]
    change = response_change if response_change is not None else np.ones((2, 4))
    variables = {
        "spend": (("allocation", "channel"), spend),
        "lower_bound": (("channel",), np.zeros(len(labels))),
        "upper_bound": (("channel",), np.full(len(labels), 180.0)),
        "response_change": (axes, change),
    }
    if increments is not None:
        variables["incremental_response"] = (("chain", "draw", "allocation", "channel"), increments)
    plan = xr.Dataset(
        variables,
        coords={"chain": [0, 1], "draw": np.arange(4), "allocation": ["reference", "optimized"], "channel": labels}
        | (coords or {}),
    )
    return plan


def _increments(reference, optimized):
    # Draws spread symmetrically around each value so their mean is the value itself.
    spread = np.array([-1.0, 1.0, -2.0, 2.0])[None, :, None, None]
    values = np.stack([reference, optimized])[None, None] + spread + np.zeros((2, 1, 1, 1))
    return values


def test_plot_budget_response_steps_from_the_reference_to_the_optimized_response():
    increments = _increments([300.0, 200.0, 100.0], [360.0, 150.0, 100.0])

    plot = plot_budget_response(_plan(increments=increments))

    bars = plot.data.set_index("name")
    assert list(bars.index) == ["Reference", "Search", "TV", "Radio", "Optimized"]
    np.testing.assert_allclose(bars["bottom"], [0.0, 550.0, 550.0, 610.0, 0.0], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(bars["top"], [600.0, 600.0, 610.0, 610.0, 610.0], rtol=1e-12, atol=1e-9)
    assert list(bars["kind"]) == ["Total", "Decrease", "Increase", "Increase", "Total"]
    assert list(bars["label"]) == ["600", "-50", "+60", "+0", "610"]


def test_plot_budget_response_reports_the_response_change_and_the_share_of_draws_that_gain():
    change = np.array([[1_000.0, 2_000.0, -500.0, 1_500.0], [1_200.0, 800.0, 900.0, 1_100.0]])
    increments = _increments([300.0, 200.0, 100.0], [360.0, 150.0, 100.0])

    plot = plot_budget_response(_plan(change, increments=increments), ci_prob=0.9)

    assert plot.labels.subtitle.startswith("Response change 1K, 90% interval")
    assert plot.labels.subtitle.endswith("The plan gains in 88% of draws.")


def test_plot_budget_response_sums_response_changes_kept_by_group():
    increments = _increments([300.0, 200.0, 100.0], [360.0, 150.0, 100.0])
    plan = _plan(
        np.full((2, 4, 2), 1_000.0),
        increments=increments,
        axes=("chain", "draw", "group"),
        coords={"group": ["north", "south"]},
    )

    plot = plot_budget_response(plan, ci_prob=0.9)

    assert plot.labels.subtitle.startswith("Response change 2K, 90% interval 2K to 2K.")


def test_plot_budget_response_sums_the_channels_left_out_into_the_last_step():
    labels = [f"Channel {index:02d}" for index in range(20)]
    reference = np.full(20, 100.0)
    # The first five channels move least, and they add 1.5 together.
    large = np.arange(5.0, 20.0) * np.where(np.arange(15) % 2, 1.0, -1.0)
    optimized = reference + np.concatenate([[0.1, 0.2, 0.3, 0.4, 0.5], large])

    plot = plot_budget_response(
        _plan(spend=[reference, optimized], increments=_increments(reference, optimized), channels=labels),
        channels=labels[5:],
    )

    names = list(plot.data["name"])
    other = plot.data.set_index("name").loc["Other channels"]
    assert names[-2:] == ["Other channels", "Optimized"]
    assert set(names[1:-2]) == set(labels[5:])
    np.testing.assert_allclose(other["top"] - other["bottom"], 1.5, rtol=1e-9, atol=0)
    assert other["kind"] == "Increase"


def test_plot_budget_response_requires_incremental_responses():
    with pytest.raises(ValueError, match="plan is missing 'incremental_response'"):
        plot_budget_response(_plan())


def test_plot_budget_spend_orders_cuts_before_increases():
    spend = [[100.0, 50.0, 30.0, 60.0], [140.0, 20.0, 30.0, 50.0]]

    plot = plot_budget_spend(_plan(spend=spend, channels=["TV", "Search", "Radio", "Print"]))

    bars = plot.data.set_index("channel")
    assert list(plot.data["channel"].cat.categories) == ["Search", "Print", "TV", "Radio"]
    np.testing.assert_allclose(bars["change"], [-30.0, -10.0, 40.0, 0.0], rtol=1e-12, atol=0)
    assert list(bars["kind"]) == ["Decrease", "Decrease", "Increase", "Increase"]
    assert list(bars["label"]) == ["-30", "-10", "+40", "+0"]


def test_plot_budget_spend_shows_every_channel_in_a_wider_figure():
    labels = [f"Channel {index:02d}" for index in range(30)]
    reference = np.full(30, 100.0)
    optimized = reference + np.arange(30)

    plot = plot_budget_spend(_plan(spend=[reference, optimized], channels=labels))

    assert set(plot.data["channel"]) == set(labels)
    assert isinstance(plot, _ScrollingPlot)


@pytest.mark.parametrize("draw", [plot_budget_response, plot_budget_spend])
@pytest.mark.parametrize(
    ("plan", "options", "error", "message"),
    [
        (xr.Dataset(), {}, ValueError, "plan is missing 'spend', 'response_change'"),
        (
            _plan(increments=np.zeros((2, 4, 2, 3))).assign_coords(allocation=["current", "proposed"]),
            {},
            ValueError,
            "plan must hold 'reference' and 'optimized' allocations",
        ),
        (_plan(increments=np.zeros((2, 4, 2, 3))), {"channels": ["Print"]}, ValueError, "channels has no 'Print'"),
        (_plan(increments=np.zeros((2, 4, 2, 3))), {"channels": "TV"}, TypeError, "channels must be a sequence"),
        ({"spend": 1.0}, {}, TypeError, "plan must be an xarray Dataset"),
    ],
)
def test_budget_plots_reject_invalid_arguments(draw, plan, options, error, message):
    with pytest.raises(error, match=message):
        draw(plan, **options)


@pytest.mark.parametrize("draw", [plot_budget_response, plot_budget_spend])
def test_budget_plots_draw(draw):
    increments = _increments([300.0, 200.0, 100.0], [360.0, 150.0, 100.0])

    figure = draw(_plan(increments=increments)).draw()

    assert figure.axes
    plt.close(figure)
