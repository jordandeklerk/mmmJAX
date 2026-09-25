"""Plots of the adstock and saturation curves that parameter draws imply."""

import functools
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr
from numpy.typing import NDArray

from mmmjax.plotting._layers import _bands, _facet, _scales
from mmmjax.plotting._summary import (
    _ci_prob,
    _ordered,
    _pick_channels,
    _require_draws,
    _restrict,
    _select_panels,
    _summarize,
    _wrap,
)
from mmmjax.plotting.diagnostics import _largest_groups, _require_count
from mmmjax.plotting.theme import _colors, theme_mmmjax

__all__ = ["plot_adstock", "plot_saturation"]


def plot_adstock(
    results: xr.DataTree,
    adstock: Callable[..., jax.Array],
    *,
    parameters: Mapping[str, str],
    max_lag: int,
    prior: xr.DataTree | None = None,
    channels: Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    group: Literal["prior", "posterior"] = "posterior",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot the share of an exposure's effect that reaches each later period.

    Each parameter draw is passed to the adstock function along with one unit
    of exposure in the first period, so the output traces the weight of every
    lag. The line follows the point estimate across draws and the band its
    credible interval.

    The last axis of the parameters, usually ``channel``, sets the colors, and
    any other axis becomes a panel. With ``prior``, each channel gets its own
    panel where the prior weights sit beside the posterior ones. A posterior
    that looks like its prior shows that the data said little about the
    carryover.

    Parameters
    ----------
    results : xarray.DataTree
        Results containing the adstock parameters in the selected group.
    adstock : callable
        Adstock function called as ``adstock(media, **values, max_lag=max_lag)``,
        such as ``geometric_adstock``. Fix other settings the model used, such
        as ``normalize=False``, with ``functools.partial``.
    parameters : mapping of str to str
        Result variable for each adstock argument, as in
        ``{"alpha": "retention"}``.
    max_lag : int
        Longest lag the model's adstock uses. Nonnegative.
    prior : xarray.DataTree, optional
        Output of ``sample_prior`` to draw beside the posterior.
    channels : sequence of str, optional
        Labels of the color axis to show. Defaults to every label, or to the
        first ten when there are more.
    coords : mapping of str to sequence, optional
        Labels to keep on other axes, as in ``{"group": ["north", "south"]}``.
    n_groups : int or None, default 3
        Number of groups to give panels when the parameters have a group axis,
        chosen by population when the results record it. None shows every
        group.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Must be ``"posterior"`` with ``prior``.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Weights by lag, with one line and band per channel or distribution.
    """
    if not callable(adstock):
        raise TypeError(f"adstock must be callable, got {type(adstock).__name__}")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError(f"max_lag must be an integer, got {type(max_lag).__name__}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be nonnegative, got {max_lag}")
    probability = _ci_prob(ci_prob)
    sources, note = _parameter_panels(results, prior, parameters, group, channels, coords, n_groups, "adstock")
    impulse = np.zeros(max_lag + 1)
    impulse[0] = 1.0
    respond = functools.partial(adstock, max_lag=max_lag)
    lags = np.arange(max_lag + 1)
    curves = {
        label: _transform_draws(respond, arrays, impulse, "lag", lags, "adstock") for label, arrays in sources.items()
    }
    frame, color, levels, facets = _curve_frame(curves, "lag", probability)
    point = pn.geom_point(pn.aes(color=color), size=2) if color else pn.geom_point(color=_colors(1)[0], size=2)

    plot = (
        _bands(frame, x="lag", color=color, labels=levels, probability=probability)
        + point
        + pn.labs(x="Lag", y="Weight", caption=note)
        + _scales(frame, "lag")
        + _facet(facets)
        + theme_mmmjax()
    )
    return plot


def plot_saturation(
    results: xr.DataTree,
    saturation: Callable[..., jax.Array],
    *,
    parameters: Mapping[str, str],
    max_input: float,
    prior: xr.DataTree | None = None,
    channels: Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    group: Literal["prior", "posterior"] = "posterior",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot the saturation curve that each parameter draw implies.

    Each parameter draw is passed to the saturation function along with 101
    evenly spaced media inputs from zero to ``max_input``, so the output traces
    how quickly each channel's effect levels off. The line follows the point
    estimate across draws and the band its credible interval.

    The last axis of the parameters, usually ``channel``, sets the colors, and
    any other axis becomes a panel. With ``prior``, each channel gets its own
    panel where the prior curve sits beside the posterior one.

    Parameters
    ----------
    results : xarray.DataTree
        Results containing the saturation parameters in the selected group.
    saturation : callable
        Saturation function called as ``saturation(media, **values)``, such as
        ``hill_saturation``. Fix other arguments the model used, such as a
        known ``slope``, with ``functools.partial``.
    parameters : mapping of str to str
        Result variable for each saturation argument, as in
        ``{"half_saturation": "half_saturation", "slope": "slope"}``.
    max_input : float
        Largest media input to evaluate, in the units the saturation function
        receives. Positive.
    prior : xarray.DataTree, optional
        Output of ``sample_prior`` to draw beside the posterior.
    channels : sequence of str, optional
        Labels of the color axis to show. Defaults to every label, or to the
        first ten when there are more.
    coords : mapping of str to sequence, optional
        Labels to keep on other axes, as in ``{"group": ["north", "south"]}``.
    n_groups : int or None, default 3
        Number of groups to give panels when the parameters have a group axis,
        chosen by population when the results record it. None shows every
        group.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Must be ``"posterior"`` with ``prior``.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Saturated media by media input, with one line and band per channel or
        distribution.
    """
    if not callable(saturation):
        raise TypeError(f"saturation must be callable, got {type(saturation).__name__}")
    if isinstance(max_input, bool) or not isinstance(max_input, numbers.Real):
        raise TypeError(f"max_input must be a number, got {type(max_input).__name__}")
    if not math.isfinite(max_input) or max_input <= 0:
        raise ValueError(f"max_input must be positive and finite, got {max_input!r}")
    probability = _ci_prob(ci_prob)
    sources, note = _parameter_panels(results, prior, parameters, group, channels, coords, n_groups, "saturation")
    inputs = np.linspace(0.0, float(max_input), 101)
    curves = {
        label: _transform_draws(saturation, arrays, inputs, "media", inputs, "saturation")
        for label, arrays in sources.items()
    }
    frame, color, levels, facets = _curve_frame(curves, "media", probability)

    plot = (
        _bands(frame, x="media", color=color, labels=levels, probability=probability)
        + pn.labs(x="Media", y="Saturated media", caption=note)
        + _scales(frame, "media")
        + _facet(facets)
        + theme_mmmjax()
    )
    return plot


def _parameter_panels(
    results: object,
    prior: object,
    parameters: object,
    group: str,
    channels: Sequence[str] | None,
    coords: Mapping[str, object] | None,
    n_groups: int | None,
    name: str,
) -> tuple[dict[str, dict[str, xr.DataArray]], str]:
    """Read the parameter draws of each distribution and keep the channels and groups the plot shows."""
    _require_count(n_groups, "n_groups")
    sources = _parameter_draws(results, prior, parameters, group, name)
    first = next(iter(next(iter(sources.values())).values()))
    selection, _ = _select_panels(first, coords, None, "")
    chosen = _restrict(first, selection)
    event_dims = [str(dim) for dim in chosen.dims if dim not in ("chain", "draw")]
    color = event_dims[-1] if event_dims else None
    notes = []
    if color is not None:
        shown, channel_note = _pick_channels(chosen[color].values, None, channels, 10, None)
        selection[color] = [label for label in chosen[color].values if str(label) in shown]
        notes.append(channel_note)
    elif channels is not None:
        raise ValueError("channels needs parameters with a channel axis")
    crowded = "group" in event_dims and color != "group" and "group" not in selection
    if crowded and n_groups is not None and chosen.sizes["group"] > n_groups:
        trees = [tree for tree in (results, prior) if isinstance(tree, xr.DataTree)]
        selection["group"] = _largest_groups(trees, list(chosen["group"].values), n_groups)
        notes.append(f"Showing {n_groups} of {chosen.sizes['group']} groups. Pass coords to choose others.")
    kept = {
        label: {argument: _restrict(values, selection) for argument, values in arrays.items()}
        for label, arrays in sources.items()
    }
    note = " ".join(text for text in notes if text)
    return kept, note


def _parameter_draws(
    results: object, prior: object, parameters: object, group: str, name: str
) -> dict[str, dict[str, xr.DataArray]]:
    """Read each parameter's draws from every distribution the plot compares and align their axes."""
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray DataTree, got {type(results).__name__}")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(f"parameters must map at least one {name} argument to a result variable")
    if group not in ("prior", "posterior"):
        raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}")
    trees = {group.title(): (results, group, "results")}
    if prior is not None:
        if not isinstance(prior, xr.DataTree):
            raise TypeError(f"prior must be an xarray DataTree, got {type(prior).__name__}")
        if group != "posterior":
            raise ValueError(f"group must be 'posterior' when prior is given, got {group!r}")
        trees["Prior"] = (prior, "prior", "prior")
    sources = {}
    for label, (tree, node, argument) in trees.items():
        if node not in tree.children:
            raise ValueError(f"{argument} has no {node!r} group")
        draws = tree[node].to_dataset()
        missing = [variable for variable in parameters.values() if variable not in draws.data_vars]
        if missing:
            raise ValueError(f"{argument} has no {node} variable {', '.join(repr(variable) for variable in missing)}")
        arrays = {key: draws[variable] for key, variable in parameters.items()}
        for key, values in arrays.items():
            _require_draws(values, f"parameters[{key!r}]")
        sources[label] = dict(zip(arrays, xr.broadcast(*arrays.values()), strict=True))
    return sources


def _transform_draws(
    transform: Callable[..., jax.Array],
    arrays: dict[str, xr.DataArray],
    column: NDArray[np.float64],
    dim: str,
    labels: NDArray[np.generic],
    name: str,
) -> xr.DataArray:
    """Pass one column of media per series through the transformation once per draw and label its rows."""
    first = next(iter(arrays.values()))
    event_dims = [axis for axis in first.dims if axis not in ("chain", "draw")]
    ordered = {argument: values.transpose("chain", "draw", *event_dims) for argument, values in arrays.items()}
    chains, draws = first.sizes["chain"], first.sizes["draw"]
    event_shape = tuple(first.sizes[axis] for axis in event_dims)
    flattened = {
        argument: jnp.asarray(values.values.reshape((chains * draws, *event_shape)))
        for argument, values in ordered.items()
    }
    dtype = jnp.result_type(float, *flattened.values())
    rows = jnp.asarray(column, dtype=dtype).reshape((-1,) + (1,) * len(event_shape))
    media = jnp.broadcast_to(rows, (len(column), *event_shape))

    def respond(values: dict[str, jax.Array]) -> jax.Array:
        return transform(media, **values)

    output = np.asarray(jax.vmap(respond)(flattened))
    if output.shape[1:] != media.shape:
        raise ValueError(
            f"{name} must return an array shaped like its media input, got shape {output.shape[1:]} "
            f"for media of shape {media.shape}"
        )
    coords = {axis: first[axis].values for axis in event_dims if axis in first.coords}
    coords[dim] = labels
    curves = xr.DataArray(
        output.reshape((chains, draws, len(column), *event_shape)),
        dims=("chain", "draw", dim, *event_dims),
        coords=coords,
    )
    return curves


def _curve_frame(
    curves: dict[str, xr.DataArray], dim: str, probability: float
) -> tuple[pd.DataFrame, str | None, list[str], list[str]]:
    """Summarize each distribution's curves and choose the columns that color them and give them panels."""
    first = next(iter(curves.values()))
    event_dims = [str(axis) for axis in first.dims if axis not in ("chain", "draw", dim)]
    color = event_dims[-1] if event_dims else None
    frames = [_summarize(values, probability).assign(distribution=label) for label, values in curves.items()]
    frame = _ordered(pd.concat(frames, ignore_index=True), "distribution", list(curves))
    if len(curves) == 1:
        levels = [] if color is None else [str(label) for label in first[color].values]
        facets = event_dims[:-1]
        return frame, color, levels, facets
    distributions = list(curves)
    if color is None:
        return frame, "distribution", distributions, []
    # Panel titles wrap long channel names, since plotnine's strips never break a line on their own.
    titles = {str(label): _wrap(str(label), 28) for label in first[color].values}
    panels = _ordered(frame.assign(panel=frame[color].astype(str).map(titles)), "panel", list(titles.values()))
    facets = ["panel", *event_dims[:-1]]
    return panels, "distribution", distributions, facets
