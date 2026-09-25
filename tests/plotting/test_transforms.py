"""Tests for plots of the adstock and saturation curves that parameter draws imply."""

import functools

import matplotlib.pyplot as plt
import numpy as np
import plotnine as pn
import pytest
import xarray as xr

from mmmjax import geometric_adstock, hill_saturation, plot_adstock, plot_saturation


def _retention(values, *, group="posterior"):
    draws = np.broadcast_to(np.asarray(values, dtype=np.float32), (2, 3, len(values)))
    results = xr.DataTree.from_dict(
        {
            group: xr.Dataset(
                {"retention": (("chain", "draw", "channel"), draws)},
                coords={"chain": [0, 1], "draw": np.arange(3), "channel": ["TV", "Search"]},
            )
        }
    )
    return results


def _half_saturation(values, *, population=None):
    values = np.asarray(values, dtype=np.float32)
    dims = ("chain", "draw", "group", "channel") if values.ndim == 2 else ("chain", "draw", "channel")
    coords = {"chain": [0, 1], "draw": np.arange(3), "channel": ["TV", "Search"]}
    if values.ndim == 2:
        coords["group"] = [f"g{index}" for index in range(values.shape[0])]
    groups = {
        "posterior": xr.Dataset(
            {"half_saturation": (dims, np.broadcast_to(values, (2, 3, *values.shape)))}, coords=coords
        )
    }
    if population is not None:
        groups["constant_data"] = xr.Dataset({"population": ("group", population)}, coords={"group": coords["group"]})
    results = xr.DataTree.from_dict(groups)
    return results


def _hill(half_saturation, media):
    curve = media / (media + half_saturation)
    return curve


def test_plot_adstock_matches_normalized_geometric_weights():
    retention = np.array([0.5, 0.2])
    kernel = retention[None, :] ** np.arange(4)[:, None]
    expected = kernel / kernel.sum(axis=0)

    plot = plot_adstock(_retention(retention), geometric_adstock, parameters={"alpha": "retention"}, max_lag=3)

    weights = plot.data.pivot(index="lag", columns="channel", values="estimate")[["TV", "Search"]]
    np.testing.assert_allclose(weights.to_numpy(), expected, rtol=3e-6, atol=0)


def test_plot_adstock_draws_the_prior_beside_the_posterior_in_each_channel_panel():
    kernel = 0.8 ** np.arange(4)
    expected = kernel / kernel.sum()

    plot = plot_adstock(
        _retention([0.5, 0.2]),
        geometric_adstock,
        parameters={"alpha": "retention"},
        max_lag=3,
        prior=_retention([0.8, 0.8], group="prior"),
    )

    frame = plot.data.set_index(["distribution", "channel", "lag"])["estimate"]
    np.testing.assert_allclose(frame.loc[("Prior", "Search")].to_numpy(), expected, rtol=3e-6, atol=0)
    assert list(plot.data["distribution"].cat.categories) == ["Posterior", "Prior"]
    assert list(plot.facet.vars) == ["panel"]
    assert list(plot.data["panel"].cat.categories) == ["TV", "Search"]


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
        ({"prior": xr.Dataset()}, TypeError, "prior must be an xarray DataTree"),
        ({"prior": _retention([0.5, 0.2])}, ValueError, "prior has no 'prior' group"),
        (
            {
                "prior": _retention([0.5, 0.2], group="prior"),
                "results": _retention([0.5, 0.2], group="prior"),
                "group": "prior",
            },
            ValueError,
            "group must be 'posterior' when prior is given",
        ),
        ({"n_groups": 0}, ValueError, "n_groups must be at least 1"),
        ({"coords": {"group": ["g0"]}}, ValueError, "coords has no axis 'group'"),
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


def test_plot_saturation_matches_hill_curves_from_zero_to_the_largest_input():
    inputs = np.linspace(0.0, 4.0, 101)
    expected = np.stack([_hill(1.0, inputs), _hill(2.0, inputs)], axis=1)

    plot = plot_saturation(
        _half_saturation([1.0, 2.0]),
        functools.partial(hill_saturation, slope=1.0),
        parameters={"half_saturation": "half_saturation"},
        max_input=4.0,
    )

    curves = plot.data.pivot(index="media", columns="channel", values="estimate")[["TV", "Search"]]
    np.testing.assert_allclose(curves.index.to_numpy(), inputs, rtol=1e-12, atol=0)
    np.testing.assert_allclose(curves.to_numpy(), expected, rtol=3e-6, atol=1e-7)
    assert (plot.labels.x, plot.labels.y) == ("Media", "Saturated media")


def test_transform_plots_give_the_most_populous_groups_panels():
    results = _half_saturation([[1.0, 2.0]] * 5, population=[10, 50, 20, 40, 30])
    hill = functools.partial(hill_saturation, slope=1.0)
    arguments = {"parameters": {"half_saturation": "half_saturation"}, "max_input": 2.0}

    default = plot_saturation(results, hill, **arguments)
    chosen = plot_saturation(results, hill, coords={"group": ["g0"]}, **arguments)
    every = plot_saturation(results, hill, n_groups=None, **arguments)

    assert set(default.data["group"]) == {"g1", "g3", "g4"}
    assert default.labels.caption == "Showing 3 of 5 groups. Pass coords to choose others."
    assert set(chosen.data["group"]) == {"g0"}
    assert chosen.labels.caption == ""
    assert set(every.data["group"]) == {f"g{index}" for index in range(5)}


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"saturation": "hill"}, TypeError, "saturation must be callable"),
        ({"max_input": "5"}, TypeError, "max_input must be a number, got str"),
        ({"max_input": True}, TypeError, "max_input must be a number, got bool"),
        ({"max_input": 0.0}, ValueError, "max_input must be positive and finite"),
        ({"max_input": float("inf")}, ValueError, "max_input must be positive and finite"),
        ({"parameters": {}}, ValueError, "parameters must map at least one saturation argument"),
        (
            {"saturation": lambda media, half_saturation: media[1:]},
            ValueError,
            "saturation must return an array shaped like its media input",
        ),
        ({"channels": ["Print"]}, ValueError, "channels has no 'Print'"),
    ],
)
def test_plot_saturation_rejects_invalid_arguments(options, error, message):
    arguments = {
        "results": _half_saturation([1.0, 2.0]),
        "saturation": functools.partial(hill_saturation, slope=1.0),
        "parameters": {"half_saturation": "half_saturation"},
        "max_input": 4.0,
    } | options

    with pytest.raises(error, match=message):
        plot_saturation(arguments.pop("results"), arguments.pop("saturation"), **arguments)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: plot_adstock(
                _retention([0.5, 0.2]), geometric_adstock, parameters={"alpha": "retention"}, max_lag=3
            ),
            id="adstock",
        ),
        pytest.param(
            lambda: plot_adstock(
                _retention([0.5, 0.2]),
                geometric_adstock,
                parameters={"alpha": "retention"},
                max_lag=3,
                prior=_retention([0.8, 0.8], group="prior"),
            ),
            id="adstock prior",
        ),
        pytest.param(
            lambda: plot_saturation(
                _half_saturation([1.0, 2.0]),
                functools.partial(hill_saturation, slope=1.0),
                parameters={"half_saturation": "half_saturation"},
                max_input=4.0,
            ),
            id="saturation",
        ),
    ],
)
def test_transform_plots_draw(build):
    figure = build().draw()

    assert figure.axes
    plt.close(figure)


def test_plot_saturation_draws_one_panel_per_channel_with_a_prior():
    prior = _half_saturation([1.0, 1.0])
    prior = xr.DataTree.from_dict({"prior": prior["posterior"].to_dataset()})

    plot = plot_saturation(
        _half_saturation([1.0, 2.0]),
        functools.partial(hill_saturation, slope=1.0),
        parameters={"half_saturation": "half_saturation"},
        max_input=4.0,
        prior=prior,
    )

    assert isinstance(plot.facet, pn.facet_wrap)
    assert set(plot.data["distribution"]) == {"Posterior", "Prior"}
