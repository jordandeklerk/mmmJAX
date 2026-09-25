"""Plots of media effects and their returns."""

import math
import numbers
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr

from mmmjax.plotting._layers import _bands, _bar_layout, _compact, _facet, _scales
from mmmjax.plotting._summary import (
    _ci_prob,
    _facets,
    _label,
    _ordered,
    _percent,
    _pick_channels,
    _plan_spend,
    _require_dataset,
    _require_draws,
    _summarize,
    _wrap,
)
from mmmjax.plotting.theme import _colors, theme_mmmjax

if TYPE_CHECKING:
    from plotnine.ggplot import PlotAddable

__all__ = ["plot_adstock", "plot_frequency_curves", "plot_media_metrics", "plot_response_curves", "plot_roi"]


def plot_adstock(
    results: xr.DataTree,
    adstock: Callable[..., jax.Array],
    *,
    parameters: Mapping[str, str],
    max_lag: int,
    channels: Sequence[str] | None = None,
    group: Literal["prior", "posterior"] = "posterior",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot the share of an exposure's effect that reaches each later period.

    Each parameter draw is passed to the adstock function along with one unit
    of exposure in the first period, so the output traces the weight of every
    lag. The line follows the point estimate across draws and the band its
    credible interval. The last axis of the parameters, usually ``channel``,
    sets the colors, and any other axis becomes a panel.

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
    channels : sequence of str, optional
        Labels of the color axis to show. Defaults to every label, or to the
        first ten when there are more.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Weights by lag, with one line and band per channel.
    """
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray DataTree, got {type(results).__name__}")
    if not callable(adstock):
        raise TypeError(f"adstock must be callable, got {type(adstock).__name__}")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must map at least one adstock argument to a result variable")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError(f"max_lag must be an integer, got {type(max_lag).__name__}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be nonnegative, got {max_lag}")
    if group not in ("prior", "posterior"):
        raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}")
    if group not in results.children:
        raise ValueError(f"results has no {group!r} group")
    probability = _ci_prob(ci_prob)
    draws = results[group].to_dataset()
    missing = [name for name in parameters.values() if name not in draws.data_vars]
    if missing:
        raise ValueError(f"results has no {group} variable {', '.join(repr(name) for name in missing)}")

    arrays = {argument: draws[name] for argument, name in parameters.items()}
    for argument, values in arrays.items():
        _require_draws(values, f"parameters[{argument!r}]")
    weights = _adstock_weights(adstock, arrays, max_lag)
    event_dims = [str(dim) for dim in weights.dims if dim not in ("chain", "draw", "lag")]
    color = event_dims[-1] if event_dims else None
    note = ""
    if color is not None:
        shown, note = _pick_channels(weights[color].values, None, channels, 10, None)
        weights = weights.sel({color: [label for label in weights[color].values if str(label) in shown]})
    elif channels is not None:
        raise ValueError("channels needs parameters with a channel axis")
    frame = _summarize(weights, probability)
    labels = [] if color is None else list(weights[color].values)
    point = pn.geom_point(pn.aes(color=color), size=2) if color else pn.geom_point(color=_colors(1)[0], size=2)

    plot = (
        _bands(frame, x="lag", color=color, labels=labels, probability=probability)
        + point
        + pn.labs(x="Lag", y="Weight", caption=note)
        + _scales(frame, "lag")
        + _facet(event_dims[:-1])
        + theme_mmmjax()
    )
    return plot


def plot_frequency_curves(
    curves: xr.Dataset,
    *,
    channels: Sequence[str] | None = None,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot how each channel's response changes with its average frequency.

    Spending stays fixed along each curve, so a higher frequency reaches fewer
    people more often. The line follows the point estimate of the change from
    the reference response and the band its credible interval. A point marks
    the frequency with the largest mean change.

    Parameters
    ----------
    curves : xarray.Dataset
        Output of ``frequency_curves``.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel, or to the ten with the
        most spending when there are more.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Response changes by frequency, with one line and band per channel.
    """
    curves = _require_dataset(curves, "curves", ["response_change", "best_frequency"])
    changes = curves["response_change"]
    _require_draws(changes, "curves['response_change']")
    probability = _ci_prob(ci_prob)
    labels = curves["channel"].values
    sizes = [float(value) for value in curves["reference_spend"].values] if "reference_spend" in curves else None
    shown, note = _pick_channels(labels, sizes, channels, 10, "spending")
    changes = changes.sel(channel=shown)
    frame = _ordered(_summarize(changes, probability), "channel", shown)
    best = curves["best_frequency"].sel(channel=shown).to_dataframe().reset_index()[["channel", "best_frequency"]]
    marked = frame.merge(best, on="channel")
    marked = marked[np.isclose(marked["frequency"], marked["best_frequency"])]

    plot = (
        _bands(frame, x="frequency", color="channel", labels=shown, probability=probability)
        + pn.geom_hline(yintercept=0, linetype="dashed", color="#8c8c8c", size=0.6)
        + pn.geom_point(pn.aes(color="channel"), data=marked, size=3)
        + pn.labs(x="Average frequency", y="Response change", caption=note)
        + _scales(frame, "frequency")
        + _facet(_facets(changes.dims, ("channel", "frequency")))
        + theme_mmmjax()
    )
    return plot


def plot_media_metrics(
    metrics: xr.Dataset | Mapping[str, xr.Dataset],
    *,
    metric: str,
    channels: Sequence[str] | None = None,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot one channel metric with its credible interval.

    Draws the bars of ``plot_roi`` for another metric with draws, such as
    marginal ROI, cost per incremental response, or incremental response.
    Each bar is a channel's point estimate, labeled with its value, and the
    error bar is its credible interval. Channels run from the most spending
    to the least. Pass several results in a mapping to compare them, as in
    ``{"Prior": prior_metrics, "Posterior": metrics}``, and each label gets
    its own color. Other axes, such as ``allocation`` in the output of
    ``optimize_budget``, become panels on one scale. With many channels the
    figure widens so each bar keeps its width, and a notebook shows it at
    full size in a box that scrolls sideways. When ``channels`` leaves some
    out and the results record what the metric is built from, as
    ``media_metrics`` does for its responses and ratios, one more bar pools
    them.

    Parameters
    ----------
    metrics : xarray.Dataset or mapping of str to xarray.Dataset
        Output of ``media_metrics``, or of ``optimize_budget`` with
        ``include_metrics=True``. A mapping labels each result.
    metric : str
        Variable to plot, such as ``"marginal_roi"``,
        ``"cost_per_incremental_response"``, or ``"incremental_response"``.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel.
    ci_prob : float, optional
        Probability of the credible intervals. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Metric bars with credible intervals by channel.
    """
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a string, got {type(metric).__name__}")
    plot = _metric_bars(metrics, metric, channels, ci_prob, None)
    return plot


def plot_response_curves(
    curves: xr.Dataset,
    *,
    plan: xr.Dataset | None = None,
    channels: Sequence[str] | None = None,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot each channel's incremental response against its spending.

    The line follows the point estimate across draws and the band its
    credible interval. A point marks each channel's reference spending, and
    the line turns dashed beyond it, where the curve extrapolates past the
    spending the data observed. Channels share one panel so their slopes can
    be compared, and adding ``plotnine.facet_wrap("~channel", scales="free")``
    gives each its own.

    With ``plan``, each channel gets its own panel with points at its
    reference and optimized spending, and the line turns dashed outside the
    plan's bounds. The plan must start from the reference spending of the
    curves. A point beyond the evaluated spending is left out and named in the
    caption.

    Parameters
    ----------
    curves : xarray.Dataset
        Output of ``response_curves``.
    plan : xarray.Dataset, optional
        Output of ``optimize_budget`` for the same spending periods. Omit to
        mark only the reference spending.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel, or to the nine with the
        most spending when there are more.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Incremental responses by spending, with one line and band per channel.
    """
    curves = _require_dataset(curves, "curves", ["incremental_response", "spend", "reference_spend"])
    increments = curves["incremental_response"]
    _require_draws(increments, "curves['incremental_response']")
    probability = _ci_prob(ci_prob)
    marks, limits = _spend_marks(curves, plan)
    reference = marks[marks["level"] == "Reference spend"]
    shown, note = _pick_channels(reference["channel"], reference["spend"], channels, 9, "spending")
    increments = increments.sel(channel=shown)
    spend = curves["spend"].sel(channel=shown).to_dataframe().reset_index()[["channel", "multiplier", "spend"]]
    frame = _summarize(increments, probability).merge(spend.astype({"channel": str}), on=["channel", "multiplier"])
    frame = _ordered(frame, "channel", shown)
    facets = _facets(increments.dims, ("channel", "multiplier"))
    # Panel titles wrap long channel names, since plotnine's strips never break a line on their own.
    titles = {label: _wrap(label, 28) for label in shown}
    frame = _ordered(frame.assign(panel=frame["channel"].astype(str).map(titles)), "panel", list(titles.values()))
    labelled = ["panel", *facets]
    points = _ordered(_curve_points(frame, marks[marks["channel"].isin(shown)], labelled), "panel", titles.values())
    styles = ["Up to reference spend", "Above reference spend"] if plan is None else ["Within bounds", "Outside bounds"]
    segments = _ordered(_segments(frame, limits, labelled, styles), "panel", titles.values())
    caption = " ".join(text for text in (note, _unmarked_note(marks[marks["channel"].isin(shown)], points)) if text)
    colors = dict(zip(shown, _colors(len(shown)), strict=True))
    title = f"Channel, {_percent(probability)} interval"
    # Without a plan the line styles already say where the reference spending sits. With one, every channel
    # gets its own panel and the strips name the channels in place of a legend.
    layout: list[PlotAddable] = (
        [_facet(facets), pn.guides(shape="none")]
        if plan is None
        else [pn.facet_wrap(["panel", *facets], ncol=3, scales="free"), pn.guides(color="none", fill="none")]
    )

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes("spend", "estimate"))
        + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper", fill="channel"), alpha=0.2)
        + pn.geom_line(pn.aes(color="channel", linetype="style", group="piece"), data=segments, size=1)
        + pn.geom_point(pn.aes(color="channel", shape="level"), data=points, size=3)
        + pn.scale_color_manual(values=colors, breaks=shown, labels=[_wrap(label, 28) for label in shown])
        + pn.scale_fill_manual(values=colors, breaks=shown, labels=[_wrap(label, 28) for label in shown])
        + pn.scale_linetype_manual(values=dict(zip(styles, ["solid", "dashed"], strict=True)))
        + pn.scale_shape_manual(values={"Reference spend": "o", "Optimized spend": "^"})
        + pn.labs(
            x="Spend",
            y="Incremental response",
            color=title,
            fill=title,
            linetype="",
            shape="",
            caption=caption,
        )
        + _scales(frame, "spend")
        + layout
        + theme_mmmjax()
    )
    return plot


def plot_roi(
    metrics: xr.Dataset | Mapping[str, xr.Dataset],
    *,
    channels: Sequence[str] | None = None,
    break_even: float | None = 1.0,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot each channel's return on investment with its credible interval.

    Each bar is a channel's point estimate of ROI, labeled with its value,
    and the error bar is its credible interval. Channels run from the most
    spending to the least, and a dashed line marks ``break_even``. Pass
    several results in a mapping to compare them, as in
    ``{"Prior": prior_metrics, "Posterior": metrics}``, and each label gets
    its own color. Other axes, such as ``allocation`` in the output of
    ``optimize_budget``, become panels. With many channels the figure widens
    so each bar keeps its width, and a notebook shows it at full size in a
    box that scrolls sideways. When ``channels`` leaves some out and the
    results record spending and incremental response, one more bar gives
    their spend-weighted ROI.

    Parameters
    ----------
    metrics : xarray.Dataset or mapping of str to xarray.Dataset
        Output of ``media_metrics``, or of ``optimize_budget`` with
        ``include_metrics=True``. A mapping labels each result.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel.
    break_even : float or None, default 1.0
        ROI at which a channel returns what it costs. The default suits a
        revenue outcome, and None leaves the line out.
    ci_prob : float, optional
        Probability of the credible intervals. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        ROI bars with credible intervals by channel.
    """
    if break_even is not None and (isinstance(break_even, bool) or not isinstance(break_even, numbers.Real)):
        raise TypeError(f"break_even must be a number or None, got {type(break_even).__name__}")
    if break_even is not None and not math.isfinite(break_even):
        raise ValueError(f"break_even must be finite, got {break_even!r}")
    plot = _metric_bars(metrics, "roi", channels, ci_prob, break_even)
    return plot


def _adstock_weights(adstock: Callable[..., jax.Array], arrays: dict[str, xr.DataArray], max_lag: int) -> xr.DataArray:
    """Pass a unit impulse through the adstock function once per draw to read its lag weights."""
    aligned = dict(zip(arrays, xr.broadcast(*arrays.values()), strict=True))
    first = next(iter(aligned.values()))
    event_dims = [dim for dim in first.dims if dim not in ("chain", "draw")]
    ordered = {argument: values.transpose("chain", "draw", *event_dims) for argument, values in aligned.items()}
    chains, draws = first.sizes["chain"], first.sizes["draw"]
    event_shape = tuple(first.sizes[dim] for dim in event_dims)
    flattened = {
        argument: jnp.asarray(values.values.reshape((chains * draws, *event_shape)))
        for argument, values in ordered.items()
    }
    dtype = jnp.result_type(float, *flattened.values())
    impulse = jnp.zeros((max_lag + 1, *event_shape), dtype=dtype).at[0].set(1)

    def respond(values: dict[str, jax.Array]) -> jax.Array:
        return adstock(impulse, **values, max_lag=max_lag)

    carried = np.asarray(jax.vmap(respond)(flattened))
    if carried.shape[1:] != impulse.shape:
        raise ValueError(
            f"adstock must return an array shaped like its media input, got shape {carried.shape[1:]} "
            f"for media of shape {impulse.shape}"
        )
    coords: dict[Hashable, object] = {dim: first[dim].values for dim in event_dims if dim in first.coords}
    coords["lag"] = np.arange(max_lag + 1)
    weights = xr.DataArray(
        carried.reshape((chains, draws, max_lag + 1, *event_shape)),
        dims=("chain", "draw", "lag", *event_dims),
        coords=coords,
    )
    return weights


def _metric_bars(
    metrics: object,
    metric: str,
    channels: Sequence[str] | None,
    ci_prob: float | None,
    reference: float | None,
) -> pn.ggplot:
    """Draw one channel metric as bars with credible intervals in the layout the metric plots share."""
    datasets = _labeled_metrics(metrics, metric)
    probability = _ci_prob(ci_prob)
    first = next(iter(datasets.values()))
    labels = list(dict.fromkeys(str(channel) for dataset in datasets.values() for channel in dataset["channel"].values))
    spending = _spending(first, labels)
    shown, _ = _pick_channels(labels, spending, channels, len(labels), "spending")
    ordered = shown if spending is None else sorted(shown, key=lambda label: -spending[labels.index(label)])
    hidden = [label for label in labels if label not in shown]
    frames = []
    facets: list[str] = []
    for label, dataset in datasets.items():
        values = dataset[metric].reset_coords(drop=True)
        kept = values.sel(channel=[channel for channel in values["channel"].values if str(channel) in shown])
        pooled = _pooled_metric(dataset, metric, hidden)
        parts = [kept] if pooled is None else [kept, pooled.expand_dims(channel=["Other channels"])]
        stacked = xr.concat(parts, dim="channel")
        facets.extend(dim for dim in _facets(stacked.dims, ("channel",)) if dim not in facets)
        frames.append(_summarize(stacked, probability).assign(result=label))
    joined = pd.concat(frames, ignore_index=True)
    bars = [*ordered, *(["Other channels"] if (joined["channel"] == "Other channels").any() else [])]
    frame = _ordered(_ordered(joined, "channel", bars), "result", list(datasets))
    compared = len(datasets) > 1
    # Values sit just above each interval so they never collide with the error bars.
    offset = 0.015 * (max(frame["upper"].max(), 0.0) - min(frame["lower"].min(), 0.0))
    frame = frame.assign(
        value=[f"{value:.2f}" if abs(value) < 1000 else _compact([value])[0] for value in frame["estimate"]],
        value_position=frame["upper"] + offset,
        kind=np.where(frame["channel"] == "Other channels", "Other channels", "Channel"),
    )
    dodge = pn.position_dodge(width=0.8)
    fill = "result" if compared else "kind"
    colors = (
        dict(zip(datasets, _colors(len(datasets)), strict=True))
        if compared
        else {"Channel": _colors(1)[0], "Other channels": "#a6a6a6"}
    )
    lines: list[PlotAddable] = []
    if reference is not None:
        lines.append(pn.geom_hline(yintercept=reference, linetype="dashed", color="#8c8c8c"))
    # Each channel gets the 80 pixels Meridian gives it, and compared results widen the slot they share.
    figure, texts, layout = _bar_layout(bars, 0.4 + 0.4 * len(datasets))

    plot: pn.ggplot = (
        figure(frame, pn.aes("channel", "estimate", fill=fill))
        + lines
        + pn.geom_col(position=dodge, width=0.6, alpha=0.85)
        + pn.geom_errorbar(pn.aes(ymin="lower", ymax="upper"), position=dodge, width=0.25, color="#262626", size=0.6)
        + pn.geom_text(
            pn.aes(y="value_position", label="value"), position=dodge, va="bottom", size=8 if compared else 9
        )
        + pn.scale_fill_manual(values=colors, breaks=list(datasets) if compared else None)
        + pn.scale_x_discrete(labels=dict(zip(bars, texts, strict=True)))
        + (pn.guides() if compared else pn.guides(fill="none"))
        + pn.labs(x="", y=f"{_label(metric)}, {_percent(probability)} interval", fill="")
        + _scales(frame)
        # Panels share one axis so allocations and groups compare at a glance.
        + (pn.facet_wrap(facets) if facets else pn.facet_null())
        + theme_mmmjax()
        + layout
    )
    return plot


def _labeled_metrics(metrics: object, metric: str) -> dict[str, xr.Dataset]:
    """Check one result or a mapping of labeled results for a metric with draws by channel."""
    # A Dataset is itself a Mapping, so it has to be recognized before the labeled form.
    if isinstance(metrics, xr.Dataset):
        labeled: dict[str, object] = {"": metrics}
    elif isinstance(metrics, Mapping):
        labeled = {str(label): result for label, result in metrics.items()}
    else:
        raise TypeError(
            f"metrics must be an xarray Dataset or a mapping of labels to Datasets, got {type(metrics).__name__}"
        )
    if not labeled:
        raise ValueError("metrics must hold at least one result")
    datasets: dict[str, xr.Dataset] = {}
    for label, result in labeled.items():
        name = "metrics" if label == "" else f"metrics[{label!r}]"
        dataset = _require_dataset(result, name, [metric])
        _require_draws(dataset[metric], f"{name}[{metric!r}]")
        if "channel" not in dataset[metric].dims:
            raise ValueError(f"{name}[{metric!r}] must have a channel axis, got dims {dataset[metric].dims}")
        datasets[label] = dataset
    return datasets


def _spending(metrics: xr.Dataset, labels: list[str]) -> list[float] | None:
    """Read each channel's spending when the metrics record it."""
    for name in ("reference_spend", "spend_share", "spend"):
        if name not in metrics:
            continue
        # Budget plans record spending per allocation, and the first allocation is the reference.
        values = metrics[name].isel(allocation=0) if "allocation" in metrics[name].dims else metrics[name]
        if values.dims == ("channel",):
            by_channel = {
                str(label): float(value) for label, value in zip(values["channel"].values, values.values, strict=True)
            }
            spending = [by_channel.get(label, 0.0) for label in labels]
            return spending
    return None


def _pooled_metric(dataset: xr.Dataset, metric: str, hidden: list[str]) -> xr.DataArray | None:
    """Pool the channels left out into one value of the metric from the parts the result records."""
    spend = "reference_spend" if "reference_spend" in dataset.data_vars else "spend"
    # Responses add up across channels, and each ratio divides one total by another.
    parts = {
        "incremental_response": ("incremental_response", None),
        "marginal_response": ("marginal_response", None),
        "roi": ("incremental_response", spend),
        "marginal_roi": ("marginal_response", "incremental_spend"),
        "cost_per_incremental_response": (spend, "incremental_response"),
    }
    present = [label for label in hidden if label in {str(channel) for channel in dataset["channel"].values}]
    if metric not in parts or not present:
        return None
    numerator_name, denominator_name = parts[metric]
    if any(name is not None and name not in dataset.data_vars for name in parts[metric]):
        return None
    numerator = _channel_total(dataset, numerator_name, present, dataset[metric].dims)
    if denominator_name is None:
        return numerator
    denominator = _channel_total(dataset, denominator_name, present, dataset[metric].dims)
    if not bool((denominator > 0).all()):
        return None
    pooled = numerator / denominator
    return pooled


def _channel_total(dataset: xr.Dataset, name: str, labels: list[str], dims: tuple[Hashable, ...]) -> xr.DataArray:
    """Sum a variable over the chosen channels and over any axes the metric totals over."""
    values = dataset[name].reset_coords(drop=True).sel(channel=labels)
    extra = [dim for dim in values.dims if dim not in dims]
    total = values.sum(["channel", *extra])
    return total


def _spend_marks(curves: xr.Dataset, plan: xr.Dataset | None) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Read the spending to mark on each curve and the range its solid line covers."""
    if plan is None:
        labels = [str(label) for label in curves["channel"].values]
        values = [float(value) for value in curves["reference_spend"].values]
        marks = pd.DataFrame({"channel": labels, "level": "Reference spend", "spend": values})
        limits = {label: (-np.inf, value) for label, value in zip(labels, values, strict=True)}
        return marks, limits
    spend = _plan_spend(plan)
    plan = _require_dataset(plan, "plan", ["lower_bound", "upper_bound"])
    planned = {str(label) for label in spend["channel"].values}
    labels = [str(label) for label in curves["channel"].values if str(label) in planned]
    if not labels:
        raise ValueError("plan must share channels with curves")
    start = spend.sel(allocation="reference", channel=labels).values
    if not np.allclose(start, curves["reference_spend"].sel(channel=labels).values, rtol=1e-5):
        raise ValueError(
            "plan must start from the reference spending of curves. "
            "Pass the same new_data and spend_periods to optimize_budget and response_curves"
        )
    levels = {"reference": "Reference spend", "optimized": "Optimized spend"}
    rows = [
        {"channel": label, "level": level, "spend": float(spend.sel(allocation=allocation, channel=label))}
        for allocation, level in levels.items()
        for label in labels
    ]
    marks = pd.DataFrame(rows)
    lower, upper = plan["lower_bound"], plan["upper_bound"]
    limits = {label: (float(lower.sel(channel=label)), float(upper.sel(channel=label))) for label in labels}
    return marks, limits


def _curve_points(frame: pd.DataFrame, marks: pd.DataFrame, facets: Sequence[str]) -> pd.DataFrame:
    """Place each marked spending level on its channel's point-estimate curve."""
    keys = ["channel", *facets]
    rows = []
    for labels, curve in frame.groupby(keys, observed=True, sort=False):
        values = labels if isinstance(labels, tuple) else (labels,)
        ordered = curve.sort_values("spend")
        grid, estimate = ordered["spend"].to_numpy(), ordered["estimate"].to_numpy()
        # The tolerance keeps a mark at the end of a float32 grid. Marks past the grid would need extrapolation.
        tolerance = 1e-6 * max(abs(grid.min()), abs(grid.max()), 1.0)
        chosen = marks[marks["channel"] == str(values[0])]
        for level, target in zip(chosen["level"], chosen["spend"], strict=True):
            if grid.min() - tolerance <= target <= grid.max() + tolerance:
                height = float(np.interp(target, grid, estimate))
                rows.append(
                    dict(zip(keys, values, strict=True)) | {"level": level, "spend": target, "estimate": height}
                )
    points = pd.DataFrame(rows, columns=[*keys, "level", "spend", "estimate"])
    points = _ordered(
        _ordered(points, "channel", frame["channel"].cat.categories), "level", ["Reference spend", "Optimized spend"]
    )
    return points


def _segments(
    frame: pd.DataFrame, limits: dict[str, tuple[float, float]], facets: Sequence[str], styles: Sequence[str]
) -> pd.DataFrame:
    """Cut each curve at its channel's limits so the parts outside them can be dashed."""
    keys = ["channel", *facets]
    inside, outside = styles
    pieces = []
    for labels, curve in frame.groupby(keys, observed=True, sort=False):
        values = labels if isinstance(labels, tuple) else (labels,)
        lower, upper = limits[str(values[0])]
        ordered = curve.sort_values("spend")
        grid, estimate = ordered["spend"].to_numpy(), ordered["estimate"].to_numpy()
        # The limits join the grid so the solid and dashed pieces meet without a gap.
        spend = np.union1d(grid, [limit for limit in (lower, upper) if grid.min() < limit < grid.max()])
        heights = np.interp(spend, grid, estimate)
        parts = [
            ("below", spend <= lower, outside),
            ("within", (spend >= lower) & (spend <= upper), inside),
            ("above", spend >= upper, outside),
        ]
        for part, mask, style in parts:
            # A piece needs two points to draw a line.
            if mask.sum() >= 2:
                piece = pd.DataFrame(dict(zip(keys, values, strict=True)), index=range(int(mask.sum())))
                piece = piece.assign(spend=spend[mask], estimate=heights[mask], style=style, piece=f"{labels}/{part}")
                pieces.append(piece)
    columns = [*keys, "spend", "estimate", "style", "piece"]
    joined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    segments = _ordered(_ordered(joined, "channel", frame["channel"].cat.categories), "style", styles)
    return segments


def _unmarked_note(marks: pd.DataFrame, points: pd.DataFrame) -> str:
    """Name the channels whose marked spending falls outside their curves."""
    placed = set(zip(points["channel"].astype(str), points["level"].astype(str), strict=True))
    pairs = zip(marks["channel"], marks["level"], strict=True)
    names = list(dict.fromkeys(channel for channel, level in pairs if (channel, level) not in placed))
    note = (
        f"Spending beyond the curves is not marked for {', '.join(names)}. "
        "Widen the multipliers of response_curves to mark it."
        if names
        else ""
    )
    return note
