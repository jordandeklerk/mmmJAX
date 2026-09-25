"""Plots of media effects and their returns."""

import math
import numbers
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from mmmjax.plotting._layers import _bands, _bar_layout, _compact, _facet, _HatchedCol, _scales
from mmmjax.plotting._summary import (
    _ci_prob,
    _facets,
    _label,
    _ordered,
    _outcome_words,
    _percent,
    _pick_channels,
    _plan_spend,
    _require_dataset,
    _require_draws,
    _restrict,
    _select_panels,
    _summarize,
    _wrap,
)
from mmmjax.plotting.theme import _channel_colors, _colors, theme_mmmjax

if TYPE_CHECKING:
    from plotnine.ggplot import PlotAddable

__all__ = [
    "plot_frequency_curves",
    "plot_media_metrics",
    "plot_response_curves",
    "plot_roi_bubbles",
    "plot_spend_vs_contribution",
]


def plot_frequency_curves(
    curves: xr.Dataset,
    *,
    channels: Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
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
    coords : mapping of str to sequence, optional
        Labels to keep on other axes, as in ``{"group": ["north", "south"]}``.
    n_groups : int or None, default 3
        Number of groups to give panels when the results keep a group axis,
        chosen by the size of their values. None shows every group.
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
    selection, group_note = _select_panels(changes, coords, n_groups, "responses")
    changes = _restrict(changes.sel(channel=shown), selection)
    frame = _ordered(_summarize(changes, probability), "channel", shown)
    best = curves["best_frequency"].sel(channel=shown).to_dataframe().reset_index()[["channel", "best_frequency"]]
    marked = frame.merge(best, on="channel")
    marked = marked[np.isclose(marked["frequency"], marked["best_frequency"])]

    plot = (
        _bands(frame, x="frequency", color="channel", labels=shown, probability=probability, order=labels)
        + pn.geom_hline(yintercept=0, linetype="dashed", color="#8c8c8c", size=0.6)
        + pn.geom_point(pn.aes(color="channel"), data=marked, size=3)
        + pn.labs(
            x="Average frequency",
            y=_outcome_words("Response change", curves),
            caption=" ".join(text for text in (note, group_note) if text),
        )
        + _scales(frame, "frequency")
        + _facet(_facets(changes.dims, ("channel", "frequency")))
        + theme_mmmjax()
    )
    return plot


def plot_media_metrics(
    metrics: xr.Dataset | Mapping[str, xr.Dataset],
    *,
    metric: str = "roi",
    channels: Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    break_even: float | None = 1.0,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot a channel metric with its credible interval.

    Each bar is a channel's point estimate of ``metric``, labeled with its
    value, and the error bar is its credible interval. Channels run from the
    most spending to the least. For ROI and marginal ROI, a dashed line marks
    ``break_even``, where a channel returns what it costs.

    Cost per incremental response is drawn at its posterior median, because a
    ratio's mean is unstable when increments come near zero.

    Pass several results in a mapping to compare them, as in
    ``{"Prior": prior_metrics, "Posterior": metrics}``, and each label gets its
    own color. Other axes, such as ``allocation`` in the output of
    ``optimize_budget``, become panels on one scale.

    With many channels the figure widens so each bar keeps its width, and a
    notebook shows it at full size in a box that scrolls sideways. When
    ``channels`` leaves some out and the results record what the metric is
    built from, as ``media_metrics`` does for its responses and ratios, one
    more bar pools them.

    Parameters
    ----------
    metrics : xarray.Dataset or mapping of str to xarray.Dataset
        Output of ``media_metrics``, or of ``optimize_budget`` with
        ``include_metrics=True``. A mapping labels each result.
    metric : str, default "roi"
        Variable to plot, such as ``"roi"``, ``"marginal_roi"``,
        ``"effectiveness"``, or ``"cost_per_incremental_response"``.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel.
    coords : mapping of str to sequence, optional
        Labels to keep on other axes, as in ``{"group": ["north", "south"]}``.
    n_groups : int or None, default 3
        Number of groups to give panels when the results keep a group axis,
        chosen by the size of their values. None shows every group.
    break_even : float or None, default 1.0
        Return at which a channel pays for itself, drawn for ROI and marginal
        ROI. The default suits a revenue outcome, and None leaves the line out.
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
    _validate_break_even(break_even)
    # Only a return has a point where spending pays for itself.
    reference = break_even if metric in ("roi", "marginal_roi") else None
    plot = _metric_bars(metrics, metric, channels, coords, n_groups, ci_prob, reference)
    return plot


def plot_response_curves(
    curves: xr.Dataset,
    *,
    plan: xr.Dataset | None = None,
    combine: bool = False,
    channels: Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot each channel's incremental response against its spending.

    The line follows the point estimate across draws and the band its
    credible interval. A point marks each channel's reference spending, and
    the line turns dashed beyond it, where the curve extrapolates past the
    spending the data observed.

    Each channel gets its own panel with its own axes, and the ten channels
    with the most spending are shown by default. With ``combine=True``, the
    five with the most spending share one panel instead, so their slopes
    compare on common axes.

    With ``plan``, points mark each channel's reference and optimized
    spending, and the line turns dashed outside the plan's bounds. The plan
    must start from the reference spending of the curves. A point beyond the
    evaluated spending is left out and named in the caption.

    Parameters
    ----------
    curves : xarray.Dataset
        Output of ``response_curves``.
    plan : xarray.Dataset, optional
        Output of ``optimize_budget`` for the same spending periods. Omit to
        mark only the reference spending.
    combine : bool, default False
        Draw the channels in one panel instead of one panel each.
    channels : sequence of str, optional
        Channels to show. Defaults to the ten with the most spending, or the
        five with the most when ``combine`` is set.
    coords : mapping of str to sequence, optional
        Labels to keep on other axes, as in ``{"group": ["north", "south"]}``.
    n_groups : int or None, default 3
        Number of groups to give panels when the results keep a group axis,
        chosen by the size of their values. None shows every group.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Incremental responses by spending, with one line and band per channel.
    """
    if not isinstance(combine, bool):
        raise TypeError(f"combine must be a bool, got {type(combine).__name__}")
    curves = _require_dataset(curves, "curves", ["incremental_response", "spend", "reference_spend"])
    increments = curves["incremental_response"]
    _require_draws(increments, "curves['incremental_response']")
    probability = _ci_prob(ci_prob)
    marks, limits = _spend_marks(curves, plan)
    reference = marks[marks["level"] == "Reference spend"]
    # Five bands are about as many as one panel keeps apart.
    shown, note = _pick_channels(reference["channel"], reference["spend"], channels, 5 if combine else 10, "spending")
    selection, group_note = _select_panels(increments, coords, n_groups, "responses")
    increments = _restrict(increments.sel(channel=shown), selection)
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
    unmarked = _unmarked_note(marks[marks["channel"].isin(shown)], points)
    caption = " ".join(text for text in (note, group_note, unmarked) if text)
    colors = _channel_colors(reference["channel"], shown)
    title = f"Channel, {_percent(probability)} interval"
    # Without a plan the line styles already say where the reference spending sits.
    shapes: Literal["none"] | None = "none" if plan is None else None
    # In panels the strips name the channels in place of a legend.
    layout: list[PlotAddable] = (
        [_facet(facets), pn.guides(shape=shapes)]
        if combine
        else [pn.facet_wrap(["panel", *facets], scales="free"), pn.guides(color="none", fill="none", shape=shapes)]
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
            y=_outcome_words("Incremental response", curves),
            color=title,
            fill=title,
            linetype="",
            shape="",
            caption=caption,
        )
        # Panels sit several to a row, too narrow for plotnine's default spending breaks.
        + _scales(frame, "spend", thin=not combine)
        + layout
        + theme_mmmjax()
    )
    return plot


def plot_roi_bubbles(
    metrics: xr.Dataset,
    *,
    metric: str = "marginal_roi",
    channels: Sequence[str] | None = None,
    break_even: float | None = 1.0,
) -> pn.ggplot:
    """Plot a channel metric against ROI with bubbles sized by spending.

    Each bubble places a channel at the point estimates of its ROI and of
    ``metric``, and its area is proportional to the channel's spending. Every
    channel gets its own color, named in the legend.

    Dashed lines at ``break_even`` mark where a channel returns what it costs.
    With marginal ROI, a channel right of the vertical line has returned more
    than it cost, and a channel above the horizontal line would still return
    more than the next unit of spending costs. With effectiveness, the
    incremental response per unit of exposure, a low ROI beside a high
    effectiveness points to costly media rather than weak media.

    The ten channels with the most spending are shown by default, and the
    caption says how many were left out. Other axes, such as ``allocation`` in
    the output of ``optimize_budget``, become panels.

    Parameters
    ----------
    metrics : xarray.Dataset
        Output of ``media_metrics``, or of ``optimize_budget`` with
        ``include_metrics=True``.
    metric : str, default "marginal_roi"
        Variable for the vertical axis, such as ``"marginal_roi"`` or
        ``"effectiveness"``.
    channels : sequence of str, optional
        Channels to show. Defaults to the ten with the most spending.
    break_even : float or None, default 1.0
        ROI at which a channel returns what it costs. The default suits a
        revenue outcome, and None leaves the lines out.

    Returns
    -------
    plotnine.ggplot
        The metric against ROI by channel, sized by spending.
    """
    _validate_break_even(break_even)
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a string, got {type(metric).__name__}")
    if metric == "roi":
        raise ValueError("metric must differ from 'roi', which the horizontal axis shows")
    dataset = _require_dataset(metrics, "metrics", ["roi", metric])
    for name in ("roi", metric):
        _require_draws(dataset[name], f"metrics[{name!r}]")
        if "channel" not in dataset[name].dims:
            raise ValueError(f"metrics[{name!r}] must have a channel axis, got dims {dataset[name].dims}")
    spend = _spend_variable(dataset)
    labels = [str(label) for label in dataset["channel"].values]
    spending = _spending(dataset, labels)
    # Ten colors are all the palette tells apart, so the legend stops at ten channels.
    shown, note = _pick_channels(labels, spending, channels, 10, "spending")
    ordered = shown if spending is None else sorted(shown, key=lambda label: -spending[labels.index(label)])
    frame = _bubbles(dataset, spend, ordered, metric)
    panels = [dim for dim in frame.columns if dim not in ("channel", "roi", metric, "spend")]
    # Colors follow the results' channel order, as in the other plots, while the legend runs by spending.
    colors = _channel_colors(labels, shown)
    lines: list[PlotAddable] = []
    if break_even is not None:
        lines += [
            pn.geom_vline(xintercept=break_even, linetype="dashed", color="#8c8c8c"),
            pn.expand_limits(x=break_even),
        ]
    # Marginal ROI shares the break-even value with ROI, which no other metric does.
    if break_even is not None and metric == "marginal_roi":
        lines += [
            pn.geom_hline(yintercept=break_even, linetype="dashed", color="#8c8c8c"),
            pn.expand_limits(y=break_even),
        ]

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes("roi", metric))
        + lines
        + pn.geom_point(pn.aes(size="spend", fill="channel"), color="#262626", stroke=0.3, alpha=0.75)
        # plotnine 0.15 rescales sizes from the smallest spending, which would shrink that bubble to nothing.
        + pn.scale_size_area(max_size=24, rescaler=_from_zero)
        + pn.scale_fill_manual(values=colors, breaks=ordered, labels=[_wrap(label, 28) for label in ordered])
        # Wider margins keep the largest bubbles inside the panel.
        + pn.scale_x_continuous(expand=(0.1, 0))
        + pn.scale_y_continuous(expand=(0.1, 0))
        + pn.guides(size="none", fill=pn.guide_legend(override_aes={"size": 6}))
        + pn.labs(x="ROI", y=_outcome_words(_label(metric), dataset), fill="Channel", caption=note)
        + (pn.facet_wrap(panels, nrow=1) if panels else pn.facet_null())
        + theme_mmmjax()
    )
    return plot


def plot_spend_vs_contribution(
    metrics: xr.Dataset,
    *,
    channels: Sequence[str] | None = None,
) -> pn.ggplot:
    """Compare each channel's share of spending with its share of the response.

    Each channel gets a wide hatched bar for its share of the spending in the
    results and a narrow solid bar in front of it for its share of their
    incremental response, and the number above is its ROI. A channel whose
    solid bar rises above its hatched frame has an ROI above that of all the
    channels together.

    Responses are posterior means, so each set of shares adds up to one, and
    time and group axes are summed first. Channels run from the most spending
    to the least, and ``allocation`` in the output of ``optimize_budget`` gives
    each allocation a panel.

    With many channels the figure widens so each pair keeps its width, and a
    notebook shows it at full size in a box that scrolls sideways. When
    ``channels`` leaves some out, one more pair pools them.

    Parameters
    ----------
    metrics : xarray.Dataset
        Output of ``media_metrics``, or of ``optimize_budget`` with
        ``include_metrics=True``.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel.

    Returns
    -------
    plotnine.ggplot
        Paired share bars by channel labeled with ROI.
    """
    dataset = _require_dataset(metrics, "metrics", ["incremental_response"])
    increments = dataset["incremental_response"].reset_coords(drop=True)
    _require_draws(increments, "metrics['incremental_response']")
    if "channel" not in increments.dims:
        raise ValueError(f"metrics['incremental_response'] must have a channel axis, got dims {increments.dims}")
    spend = _spend_variable(dataset)
    panels = [str(dim) for dim in spend.dims if dim != "channel"]
    summed = [dim for dim in increments.dims if dim not in ("chain", "draw", "channel", *panels)]
    # Means add up across channels, so the response shares of every channel sum to one.
    responses = increments.sum(summed).mean(("chain", "draw"))
    labels = [str(label) for label in dataset["channel"].values]
    spending = _spending(dataset, labels)
    shown, _ = _pick_channels(labels, spending, channels, len(labels), "spending")
    ordered = shown if spending is None else sorted(shown, key=lambda label: -spending[labels.index(label)])
    hidden = [label for label in labels if label not in shown]
    shares = _share_pairs(responses, spend, ordered, hidden)
    bars = [*ordered, *(["Other channels"] if hidden else [])]
    spend_label = "Share of spend"
    response_label = _outcome_words("Share of incremental response", dataset)
    long = pd.concat(
        [
            shares.assign(measure=spend_label, share=shares["spend_share"]),
            shares.assign(measure=response_label, share=shares["response_share"]),
        ],
        ignore_index=True,
    )
    long = _ordered(_ordered(long, "channel", bars), "measure", [spend_label, response_label])
    # The ROI sits just above the taller bar of each pair.
    offset = 0.015 * (max(float(long["share"].max()), 0.0) - min(float(long["share"].min()), 0.0))
    tops = shares[["spend_share", "response_share"]].max(axis=1).clip(lower=0) + offset
    texts = [
        "" if np.isnan(roi) else f"ROI {roi:.2f}" if abs(roi) < 1000 else f"ROI {_compact([roi])[0]}"
        for roi in shares["roi"]
    ]
    marks = _ordered(shares.assign(position=tops, text=texts), "channel", bars)
    colors = {spend_label: _colors(2)[1], response_label: _colors(1)[0]}
    measures = [spend_label, response_label]
    figure, labels_shown, layout = _bar_layout(bars, 0.9)

    plot: pn.ggplot = (
        figure(long, pn.aes("channel", "share", fill="measure", color="measure", alpha="measure"))
        # The spending frames the response it bought, so a solid bar that rises above its frame beats its share.
        + _HatchedCol(data=long[long["measure"] == spend_label], width=0.8, size=0.6, hatched=colors[spend_label])
        # The frame's layer already draws both legend keys, the hatched one and the solid one.
        + pn.geom_col(data=long[long["measure"] == response_label], width=0.4, size=0.6, show_legend=False)
        + pn.geom_text(pn.aes("channel", "position", label="text"), data=marks, inherit_aes=False, va="bottom", size=9)
        + pn.scale_fill_manual(values=colors, breaks=measures)
        + pn.scale_color_manual(values=colors, breaks=measures)
        + pn.scale_alpha_manual(values={spend_label: 0.2, response_label: 1.0}, breaks=measures)
        + pn.scale_x_discrete(labels=dict(zip(bars, labels_shown, strict=True)))
        + pn.scale_y_continuous(labels=lambda values: [f"{value:.0%}" for value in values])
        + pn.labs(x="", y="Share of all channels", fill="", color="", alpha="")
        + (pn.facet_wrap(panels) if panels else pn.facet_null())
        + theme_mmmjax()
        + layout
    )
    return plot


def _metric_bars(
    metrics: object,
    metric: str,
    channels: Sequence[str] | None,
    coords: Mapping[str, object] | None,
    n_groups: int | None,
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
    selection, note = _select_panels(first[metric], coords, n_groups, "values")
    # A ratio's mean is unstable when increments come near zero, so Meridian reads its median instead.
    point = "median" if metric == "cost_per_incremental_response" else None
    frames = []
    facets: list[str] = []
    for label, result in datasets.items():
        dataset = _restrict(result, selection)
        values = dataset[metric].reset_coords(drop=True)
        kept = values.sel(channel=[channel for channel in values["channel"].values if str(channel) in shown])
        pooled = _pooled_metric(dataset, metric, hidden)
        parts = [kept] if pooled is None else [kept, pooled.expand_dims(channel=["Other channels"])]
        stacked = xr.concat(parts, dim="channel")
        facets.extend(dim for dim in _facets(stacked.dims, ("channel",)) if dim not in facets)
        frames.append(_summarize(stacked, probability, point).assign(result=label))
    joined = pd.concat(frames, ignore_index=True)
    bars = [*ordered, *(["Other channels"] if (joined["channel"] == "Other channels").any() else [])]
    frame = _ordered(_ordered(joined, "channel", bars), "result", list(datasets))
    compared = len(datasets) > 1
    # Values sit just above each interval so they never collide with the error bars.
    offset = 0.015 * (max(frame["upper"].max(), 0.0) - min(frame["lower"].min(), 0.0))
    frame = frame.assign(
        value=[_bar_value(value) for value in frame["estimate"]],
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
        # A pale fill inside a solid outline keeps the bars light enough for the intervals to read over them.
        + pn.geom_col(pn.aes(color=fill), position=dodge, width=0.6, alpha=0.22, size=0.9)
        + pn.geom_errorbar(pn.aes(ymin="lower", ymax="upper"), position=dodge, width=0.25, color="#262626", size=0.6)
        + pn.geom_text(
            pn.aes(y="value_position", label="value"), position=dodge, va="bottom", size=8 if compared else 9
        )
        + pn.scale_fill_manual(values=colors, breaks=list(datasets) if compared else None)
        + pn.scale_color_manual(values=colors, breaks=list(datasets) if compared else None)
        + pn.scale_x_discrete(labels=dict(zip(bars, texts, strict=True)))
        + (pn.guides() if compared else pn.guides(fill="none", color="none"))
        + pn.labs(
            x="",
            y=f"{_outcome_words(_label(metric), first)}, {_percent(probability)} interval",
            fill="",
            color="",
            caption=note,
        )
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


def _bar_value(value: float) -> str:
    """Write a bar's value with two decimals or three significant digits when it is small or large."""
    # Two decimals would round small rates such as effectiveness to the same text.
    if abs(value) >= 1000:
        text = _compact([value])[0]
    elif abs(value) < 1:
        text = f"{value:.3g}"
    else:
        text = f"{value:.2f}"
    return text


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


def _validate_break_even(break_even: object) -> None:
    """Check that the break-even ROI is a finite number or None."""
    if break_even is None:
        return
    if isinstance(break_even, bool) or not isinstance(break_even, numbers.Real):
        raise TypeError(f"break_even must be a number or None, got {type(break_even).__name__}")
    if not math.isfinite(break_even):
        raise ValueError(f"break_even must be finite, got {break_even!r}")


def _spend_variable(metrics: xr.Dataset) -> xr.DataArray:
    """Read the spending that the returns in the metrics divide by."""
    for name in ("reference_spend", "spend"):
        if name in metrics.data_vars:
            spend = metrics[name].reset_coords(drop=True)
            return spend
    raise ValueError("metrics must record 'reference_spend' or 'spend'")


def _bubbles(dataset: xr.Dataset, spend: xr.DataArray, shown: list[str], metric: str) -> pd.DataFrame:
    """Summarize ROI and the metric for the shown channels beside their spending."""
    columns = []
    for name in ("roi", metric):
        values = dataset[name].reset_coords(drop=True).sel(channel=shown)
        # A ratio's mean is unstable when increments come near zero, so this one uses its median.
        point = "median" if name == "cost_per_incremental_response" else None
        # Only the point estimate is drawn, so the interval probability does not matter.
        summary = _summarize(values, 0.5, point)
        columns.append(summary.drop(columns=["lower", "upper"]).rename(columns={"estimate": name}))
    spending = spend.sel(channel=shown).rename("spend").to_dataframe().reset_index()
    keys = [column for column in columns[0].columns if column != "roi"]
    joined = columns[0].astype({"channel": str}).merge(columns[1].astype({"channel": str}), on=keys)
    merged = joined.merge(spending.astype({"channel": str}), on=[key for key in keys if key in spending.columns])
    bubbles = _ordered(merged, "channel", shown)
    # Merging drops the category order, so allocation panels would otherwise sort by name.
    for dim in keys:
        if dim != "channel":
            bubbles = _ordered(bubbles, dim, dataset[dim].values)
    return bubbles


def _from_zero(
    x: ArrayLike, to: tuple[float, float] = (0, 1), _from: tuple[float, float] | None = None
) -> NDArray[np.floating]:
    """Divide sizes by the largest so each bubble's area stays proportional to its spending."""
    # The argument names copy mizani's rescalers, which plotnine calls in their place.
    array = np.asarray(x, dtype=float)
    largest = float(np.max(np.abs(array))) if _from is None else max(abs(_from[0]), abs(_from[1]))
    scaled = array / largest * to[1]
    return scaled


def _share_pairs(responses: xr.DataArray, spend: xr.DataArray, shown: list[str], hidden: list[str]) -> pd.DataFrame:
    """Give each shown channel and the pooled rest their shares of all spending and response with their ROI."""
    totals = {}
    for name, values in (("response", responses), ("spend", spend)):
        pieces = [values.sel(channel=shown)]
        if hidden:
            pieces.append(values.sel(channel=hidden).sum("channel").expand_dims(channel=["Other channels"]))
        totals[name] = xr.concat(pieces, "channel")
    # A channel without spending has no ROI, so its label stays empty.
    paid = totals["spend"].where(totals["spend"] > 0)
    shares = xr.Dataset(
        {
            "spend_share": totals["spend"] / spend.sum("channel"),
            "response_share": totals["response"] / responses.sum("channel"),
            "roi": totals["response"] / paid,
        }
    )
    frame = shares.to_dataframe().reset_index()
    for dim in spend.dims:
        if dim != "channel":
            frame = _ordered(frame, str(dim), spend[dim].values)
    return frame
