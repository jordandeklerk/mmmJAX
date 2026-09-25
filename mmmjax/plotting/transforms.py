"""Plots of the adstock and saturation curves that parameter draws imply."""

import functools
import inspect
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from plotnine.ggplot import PlotAddable

__all__ = ["plot_adstock", "plot_saturation"]


def plot_adstock(
    results: xr.DataTree,
    adstock: Callable[..., jax.Array],
    *,
    max_lag: int,
    parameters: Mapping[str, str | float] | None = None,
    prior: xr.DataTree | None = None,
    combine: bool = False,
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

    The function takes the media first and its other arguments by name, as the
    model's blocks do. An argument receives the draws of the result variable
    with the same name unless ``parameters`` maps it to another variable or
    fixes it at a number, and ``max_lag`` reaches a function that asks for it.

    Each label of the last axis of the parameters, usually ``channel``, gets
    its own panel on shared axes, and any other axis adds panels. With
    ``combine=True``, the channels share one panel in their own colors
    instead. With ``prior``, each panel gets its own vertical axis and shows
    the prior weights beside the posterior ones. A posterior that looks like
    its prior shows that the data said little about the carryover.

    Parameters
    ----------
    results : xarray.DataTree
        Results containing the adstock parameters in the selected group.
    adstock : callable
        Adstock function that takes the media first, such as
        ``geometric_adstock``.
    max_lag : int
        Longest lag the model's adstock uses. Nonnegative.
    parameters : mapping of str to str or float, optional
        Result variable or fixed value for adstock arguments, as in
        ``{"alpha": "retention"}``. Omit when every argument shares its name
        with a result variable or has a default.
    prior : xarray.DataTree, optional
        Output of ``sample_prior`` to draw beside the posterior.
    combine : bool, default False
        Draw the channels in one panel instead of one panel each. Not
        available with ``prior``.
    channels : sequence of str, optional
        Labels of the last axis to show. Defaults to every label, or to the
        first ten when there are more and the first five when ``combine`` is
        set.
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
    _require_combine(combine, prior)
    variables, fixed = _bind_arguments(adstock, parameters, results, group, {"max_lag": max_lag}, "adstock")
    sources, note, order = _parameter_panels(results, prior, variables, group, channels, coords, n_groups, combine)
    impulse = np.zeros(max_lag + 1)
    impulse[0] = 1.0
    respond = functools.partial(adstock, **fixed)
    lags = np.arange(max_lag + 1)
    curves = {
        label: _transform_draws(respond, arrays, impulse, "lag", lags, "adstock") for label, arrays in sources.items()
    }
    frame, color, levels, facets = _curve_frame(curves, "lag", probability, combine)
    # Prior and posterior take the first two colors, and channels keep the colors of their places among all.
    ordering = None if color == "distribution" else order
    point = pn.geom_point(pn.aes(color=color), size=2) if color else pn.geom_point(color=_colors(1)[0], size=2)

    plot = (
        _bands(frame, x="lag", color=color, labels=levels, probability=probability, order=ordering)
        + point
        + pn.labs(x="Lag", y="Weight", caption=note)
        + _scales(frame, "lag", thin="panel" in facets)
        + _curve_layout(facets, color)
        + theme_mmmjax()
    )
    return plot


def plot_saturation(
    results: xr.DataTree,
    saturation: Callable[..., jax.Array],
    *,
    max_input: float,
    parameters: Mapping[str, str | float] | None = None,
    prior: xr.DataTree | None = None,
    combine: bool = False,
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

    The function takes the media first and its other arguments by name, as the
    model's blocks do. An argument receives the draws of the result variable
    with the same name unless ``parameters`` maps it to another variable or
    fixes it at a number, as a model with a known ``slope`` would.

    Each label of the last axis of the parameters, usually ``channel``, gets
    its own panel on shared axes, and any other axis adds panels. With
    ``combine=True``, the channels share one panel in their own colors
    instead. With ``prior``, each panel gets its own vertical axis and shows
    the prior curve beside the posterior one.

    Parameters
    ----------
    results : xarray.DataTree
        Results containing the saturation parameters in the selected group.
    saturation : callable
        Saturation function that takes the media first, such as
        ``hill_saturation``.
    max_input : float
        Largest media input to evaluate, in the units the saturation function
        receives. Positive.
    parameters : mapping of str to str or float, optional
        Result variable or fixed value for saturation arguments, as in
        ``{"slope": 1.0}``. Omit when every argument shares its name with a
        result variable or has a default.
    prior : xarray.DataTree, optional
        Output of ``sample_prior`` to draw beside the posterior.
    combine : bool, default False
        Draw the channels in one panel instead of one panel each. Not
        available with ``prior``.
    channels : sequence of str, optional
        Labels of the last axis to show. Defaults to every label, or to the
        first ten when there are more and the first five when ``combine`` is
        set.
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
    _require_combine(combine, prior)
    variables, fixed = _bind_arguments(saturation, parameters, results, group, {}, "saturation")
    sources, note, order = _parameter_panels(results, prior, variables, group, channels, coords, n_groups, combine)
    inputs = np.linspace(0.0, float(max_input), 101)
    respond = functools.partial(saturation, **fixed)
    curves = {
        label: _transform_draws(respond, arrays, inputs, "media", inputs, "saturation")
        for label, arrays in sources.items()
    }
    frame, color, levels, facets = _curve_frame(curves, "media", probability, combine)
    # Prior and posterior take the first two colors, and channels keep the colors of their places among all.
    ordering = None if color == "distribution" else order

    plot = (
        _bands(frame, x="media", color=color, labels=levels, probability=probability, order=ordering)
        + pn.labs(x="Media", y="Saturated media", caption=note)
        + _scales(frame, "media", thin="panel" in facets)
        + _curve_layout(facets, color)
        + theme_mmmjax()
    )
    return plot


def _require_combine(combine: object, prior: object) -> None:
    """Check that channels share a panel only when no prior needs a panel beside each posterior."""
    if not isinstance(combine, bool):
        raise TypeError(f"combine must be a bool, got {type(combine).__name__}")
    if combine and prior is not None:
        raise ValueError("combine needs a plot without prior, which gives each channel a panel")


def _bind_arguments(
    transform: Callable[..., jax.Array],
    parameters: object,
    results: object,
    group: str,
    supplied: dict[str, object],
    name: str,
) -> tuple[dict[str, str], dict[str, object]]:
    """Give each argument after the media a result variable, a fixed value, or its own default."""
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray DataTree, got {type(results).__name__}")
    if group not in ("prior", "posterior"):
        raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}")
    if group not in results.children:
        raise ValueError(f"results has no {group!r} group")
    given = _given_values(parameters, supplied)
    variables = {argument: value for argument, value in given.items() if isinstance(value, str)}
    fixed: dict[str, object] = {argument: value for argument, value in given.items() if not isinstance(value, str)}
    try:
        arguments = list(inspect.signature(transform).parameters.values())[1:]
    except (TypeError, ValueError):
        # Without a signature the plot passes its own settings and only the arguments parameters names.
        arguments = None
    if arguments is None:
        fixed = supplied | fixed
    else:
        named = [
            argument for argument in arguments if argument.kind not in (argument.VAR_POSITIONAL, argument.VAR_KEYWORD)
        ]
        takes_any = any(argument.kind is argument.VAR_KEYWORD for argument in arguments)
        unknown = [argument for argument in given if argument not in {parameter.name for parameter in named}]
        if unknown and not takes_any:
            raise ValueError(f"parameters sets {unknown[0]!r}, which {name} does not take")
        available = set(results[group].to_dataset().data_vars)
        for argument in named:
            if argument.name in given:
                continue
            if argument.name in supplied:
                fixed[argument.name] = supplied[argument.name]
            elif argument.name in available:
                variables[argument.name] = argument.name
            elif argument.default is argument.empty:
                raise ValueError(
                    f"{name} argument {argument.name!r} matches no {group} variable. "
                    f"Map it in parameters, as in {{{argument.name!r}: 'variable'}} or {{{argument.name!r}: 1.0}}"
                )
    if not variables:
        raise ValueError(f"{name} must take at least one argument from the {group} draws")
    return variables, fixed


def _given_values(parameters: object, supplied: dict[str, object]) -> dict[str, str | float]:
    """Check the result variables and fixed values that parameters assigns to arguments."""
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"parameters must be a mapping, got {type(parameters).__name__}")
    for argument, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
            raise TypeError(
                f"parameters[{argument!r}] must name a result variable or be a number, got {type(value).__name__}"
            )
        if argument in supplied:
            raise ValueError(f"parameters cannot set {argument!r}. Pass {argument} to the plot instead")
    given = dict(parameters)
    return given


def _parameter_panels(
    results: xr.DataTree,
    prior: object,
    variables: dict[str, str],
    group: str,
    channels: Sequence[str] | None,
    coords: Mapping[str, object] | None,
    n_groups: int | None,
    combine: bool,
) -> tuple[dict[str, dict[str, xr.DataArray]], str, list[str]]:
    """Read the parameter draws of each distribution and keep the channels and groups the plot shows."""
    _require_count(n_groups, "n_groups")
    sources = _parameter_draws(results, prior, variables, group)
    first = next(iter(next(iter(sources.values())).values()))
    selection, _ = _select_panels(first, coords, None, "")
    chosen = _restrict(first, selection)
    event_dims = [str(dim) for dim in chosen.dims if dim not in ("chain", "draw")]
    color = event_dims[-1] if event_dims else None
    # Every label of the colored axis fixes the colors, so a channel keeps its color whichever channels are shown.
    order = [] if color is None else [str(label) for label in first[color].values]
    notes = []
    if color is not None:
        # Five bands are about as many as one panel keeps apart.
        shown, channel_note = _pick_channels(chosen[color].values, None, channels, 5 if combine else 10, None)
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
    return kept, note, order


def _parameter_draws(
    results: xr.DataTree, prior: object, variables: dict[str, str], group: str
) -> dict[str, dict[str, xr.DataArray]]:
    """Read each parameter's draws from every distribution the plot compares and align their axes."""
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
        missing = [variable for variable in variables.values() if variable not in draws.data_vars]
        if missing:
            raise ValueError(f"{argument} has no {node} variable {', '.join(repr(variable) for variable in missing)}")
        arrays = {key: draws[variable] for key, variable in variables.items()}
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
    curves: dict[str, xr.DataArray], dim: str, probability: float, combine: bool
) -> tuple[pd.DataFrame, str | None, list[str], list[str]]:
    """Summarize each distribution's curves and choose the columns that color them and give them panels."""
    first = next(iter(curves.values()))
    event_dims = [str(axis) for axis in first.dims if axis not in ("chain", "draw", dim)]
    color = event_dims[-1] if event_dims else None
    frames = [_summarize(values, probability).assign(distribution=label) for label, values in curves.items()]
    frame = _ordered(pd.concat(frames, ignore_index=True), "distribution", list(curves))
    single = len(curves) == 1
    levels = [] if color is None else [str(label) for label in first[color].values]
    colored = color if single else "distribution"
    shown = levels if single else list(curves)
    others = event_dims[:-1]
    if color is None or (single and combine):
        return frame, colored, shown, others
    # Panel titles wrap long channel names, since plotnine's strips never break a line on their own.
    titles = {label: _wrap(label, 28) for label in levels}
    panels = _ordered(frame.assign(panel=frame[color].astype(str).map(titles)), "panel", list(titles.values()))
    facets = ["panel", *others]
    return panels, colored, shown, facets


def _curve_layout(facets: list[str], color: str | None) -> list["PlotAddable"]:
    """Give each channel a panel on shared axes whose strips replace the legend."""
    if "panel" not in facets or color == "distribution":
        return [_facet(facets)]
    layout: list[PlotAddable] = [pn.facet_wrap(facets), pn.guides(color="none", fill="none")]
    return layout
