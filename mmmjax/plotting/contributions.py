"""Plots that break the response into its baseline and channel contributions."""

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr

from mmmjax.plotting._display import _ScrollingPlot
from mmmjax.plotting._layers import _compact, _facet, _figure_margins, _scales
from mmmjax.plotting._summary import (
    _distinct_shortened,
    _facets,
    _ordered,
    _outcome_words,
    _pick_channels,
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

__all__ = ["plot_contributions"]


def plot_contributions(
    effects: xr.Dataset,
    *,
    channels: Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    include_baseline: bool = True,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
) -> pn.ggplot:
    """Break the response down into the baseline and each channel's contribution.

    By default a horizontal waterfall starts from the baseline and adds each
    channel's share of the whole response, from the largest to the smallest,
    until the bars reach all of it. Each bar is labeled with its share and its
    total in the outcome's units. Shares are ratios of posterior means, so they
    add up.

    When channel effects interact, the separate shares need not add up to
    removing every channel at once, and a last bar carries that difference when
    it reaches half a percent. Without the baseline, the bars split the joint
    increment of every channel instead.

    With ``by="time"``, each channel's contribution stacks by period on the
    baseline under a line that traces the whole response, and the axis starts
    near the lowest baseline unless groups give panels. Without the baseline,
    the channels stack under a line that traces the joint increment of removing
    every channel at once. The ten channels with the largest total increments
    stack separately and the rest share one area.

    Axes that ``by`` leaves out are summed within each draw, and ``by="group"``
    gives each group a panel. Many channels make the waterfall taller, and a
    notebook shows it at full size.

    Parameters
    ----------
    effects : xarray.Dataset
        Output of ``contributions``.
    channels : sequence of str, optional
        Channels to show separately. Defaults to every channel in the
        waterfall and to the ten with the largest total increments by time.
        The rest are summed into one bar or area.
    by : str or sequence of str, optional
        Axes of ``effects`` to keep, ``"time"``, ``"group"``, or both. Omit
        to sum them.
    include_baseline : bool, default True
        Draw the baseline. False shows only what the channels add.
    coords : mapping of str to sequence, optional
        Labels to keep before anything is summed, as in
        ``{"group": ["north", "south"]}``. Groups chosen here replace the
        ``n_groups`` choice.
    n_groups : int or None, default 3
        Number of groups to show when ``by`` keeps ``group``, chosen by the
        size of their responses. None shows every group.

    Returns
    -------
    plotnine.ggplot
        Waterfall of shares, or contributions stacked by period.
    """
    effects = _require_contributions(effects)
    if not isinstance(include_baseline, bool):
        raise TypeError(f"include_baseline must be a bool, got {type(include_baseline).__name__}")
    kept = _kept_axes(effects, by)
    selection, _ = _select_panels(effects["reference_response"], coords, None, "")
    totals = _sum_axes(_restrict(effects, selection), kept)
    chosen = {dim: labels for dim, labels in selection.items() if dim in totals.dims}
    panels, note = _select_panels(totals["reference_response"], chosen, n_groups, "responses")
    totals = _restrict(totals, panels)
    if "time" not in kept:
        plot = _waterfall_plot(totals, channels, include_baseline, note)
    else:
        plot = _stacked_plot(totals, channels, include_baseline, note)
    return plot


def _require_contributions(effects: object) -> xr.Dataset:
    """Check that the results come from contributions and keep draws by channel."""
    dataset = _require_dataset(effects, "effects", ["incremental_response", "baseline_response", "reference_response"])
    increments = dataset["incremental_response"]
    _require_draws(increments, "effects['incremental_response']")
    if "channel" not in increments.dims:
        raise ValueError(f"effects['incremental_response'] must have a channel axis, got dims {increments.dims}")
    return dataset


def _kept_axes(effects: xr.Dataset, by: object) -> list[str]:
    """Check that by names time or group axes the contributions kept."""
    if by is not None and not isinstance(by, (str, Sequence)):
        raise TypeError(f"by must be an axis name or a sequence of them, got {type(by).__name__}")
    kept = [] if by is None else [by] if isinstance(by, str) else [str(dim) for dim in by]
    unknown = [dim for dim in kept if dim not in ("time", "group")]
    if unknown:
        raise ValueError(f"by must name 'time' or 'group', got {unknown[0]!r}")
    missing = [dim for dim in kept if dim not in effects["incremental_response"].dims]
    if missing:
        raise ValueError(
            f"by must name axes that effects keep, got {missing[0]!r}. Pass by={missing[0]!r} to contributions"
        )
    return kept


def _sum_axes(effects: xr.Dataset, kept: list[str]) -> xr.Dataset:
    """Sum each draw's responses over the axes that by leaves out."""
    parts = effects[["incremental_response", "baseline_response", "reference_response"]].reset_coords(drop=True)
    summed = [dim for dim in ("time", "group") if dim in parts.dims and dim not in kept]
    totals = parts.sum(summed, keep_attrs=True) if summed else parts
    return totals


def _waterfall_plot(totals: xr.Dataset, channels: Sequence[str] | None, include_baseline: bool, note: str) -> pn.ggplot:
    """Lay the baseline and the channel shares end to end as a horizontal waterfall."""
    increments = totals["incremental_response"]
    labels = [str(label) for label in increments["channel"].values]
    averaged = abs(increments.mean(("chain", "draw")))
    sizes = averaged.sum([dim for dim in averaged.dims if dim != "channel"]).values
    shown, _ = _pick_channels(labels, sizes, channels, len(labels), "contributions")
    shares, values = _shares(totals, shown, include_baseline)
    panels = [str(dim) for dim in shares.dims if dim != "part"]
    frame = _waterfall_rows(shares, values, panels)
    rows = list(dict.fromkeys(frame["name"]))
    names = dict(zip(rows, _distinct_shortened(rows, 40), strict=True))
    frame = frame.assign(label=frame["name"].map(names))
    panel_count = max(1, int(np.prod([totals.sizes[dim] for dim in panels])))
    # Each row keeps a third of an inch, and panels past the third widen the figure by their own width.
    height = max(7.0, 1.4 + 0.32 * len(rows))
    width = max(12.0, 2.0 + 3.3 * panel_count)
    tall = height > 7 or width > 12
    figure = _ScrollingPlot if tall else pn.ggplot
    sizing: list[PlotAddable] = [_figure_margins(width, height)] if tall else []
    colors = {"Baseline": "#8c8c8c", "Increase": _colors(1)[0], "Decrease": _colors(2)[1], "Other": "#c9c9c9"}
    frame, limits = _place_labels(frame, width, panel_count, max(len(name) for name in names.values()))
    breaks = _share_breaks(frame, limits)
    # A bar that runs below zero needs a line to show where the response starts.
    origin: list[PlotAddable] = [pn.geom_vline(xintercept=0, color="#8c8c8c", size=0.6)] if limits[0] < 0 else []
    title = "Share of response" if include_baseline else "Share of incremental response"

    plot: pn.ggplot = (
        figure(frame)
        + origin
        + pn.geom_rect(pn.aes(xmin="start", xmax="end", ymin="position - 0.35", ymax="position + 0.35", fill="kind"))
        + pn.geom_text(pn.aes(x="text_position", y="position", label="text"), ha="left", size=9)
        + pn.scale_fill_manual(values=colors)
        + pn.scale_y_continuous(
            breaks=list(range(1, len(rows) + 1)), labels=[names[row] for row in reversed(rows)], minor_breaks=[]
        )
        + pn.scale_x_continuous(
            breaks=breaks,
            minor_breaks=[(left + right) / 2 for left, right in itertools.pairwise(breaks)],
            labels=lambda values: [f"{value:.0%}" for value in values],
            limits=limits,
            expand=(0, 0),
        )
        + pn.guides(fill="none")
        + pn.labs(x=_outcome_words(title, totals), y="", caption=note)
        + (pn.facet_wrap(panels, nrow=1) if panels else pn.facet_null())
        + theme_mmmjax()
        + sizing
    )
    return plot


def _shares(totals: xr.Dataset, shown: list[str], include_baseline: bool) -> tuple[xr.DataArray, xr.DataArray]:
    """Average each part of the response and divide it by the average whole it belongs to."""
    increments = totals["incremental_response"]
    baseline = totals["baseline_response"]
    joint = totals["reference_response"] - baseline
    hidden = [label for label in increments["channel"].values if str(label) not in shown]
    parts = {"Baseline": baseline} if include_baseline else {}
    parts |= {label: increments.sel(channel=label, drop=True) for label in shown}
    if hidden:
        parts["Other channels"] = increments.sel(channel=hidden).sum("channel")
    parts["Channel interactions"] = joint - increments.sum("channel")
    values = xr.concat(list(parts.values()), dim="part").assign_coords(part=list(parts)).mean(("chain", "draw"))
    whole = (totals["reference_response"] if include_baseline else joint).mean(("chain", "draw"))
    # Shares of a whole at or below zero have no meaning, which a response in model units can produce.
    if bool((whole <= 0).any()):
        name = "response" if include_baseline else "joint incremental response"
        raise ValueError(
            f"effects must have a positive {name} to split into shares, got a mean of {float(whole.min()):.3g}"
        )
    shares = values / whole
    return shares, values


def _waterfall_rows(shares: xr.DataArray, values: xr.DataArray, panels: list[str]) -> pd.DataFrame:
    """Lay out the floating bars of the waterfall with the channels ordered by their average share."""
    names = [str(name) for name in shares["part"].values]
    overall = shares.mean(panels) if panels else shares
    channels = [name for name in names if name not in ("Baseline", "Other channels", "Channel interactions")]
    ranked = sorted(channels, key=lambda name: -float(overall.sel(part=name)))
    # The interaction bar only earns a row when it moves the total by half a percent somewhere.
    interacting = float(abs(shares.sel(part="Channel interactions")).max()) >= 0.005
    order = [name for name in ("Baseline",) if name in names]
    order += [*ranked, *[name for name in ("Other channels",) if name in names]]
    order += ["Channel interactions"] if interacting else []
    records = []
    share_frame = shares.sel(part=order).to_dataframe(name="share").reset_index()
    value_frame = values.sel(part=order).to_dataframe(name="value").reset_index()
    merged = share_frame.merge(value_frame, on=["part", *panels])
    for _, panel in merged.groupby(panels, sort=False) if panels else [((), merged)]:
        running = 0.0
        panel_shares = dict(zip(panel["part"], panel["share"].to_numpy(dtype=float), strict=True))
        panel_values = dict(zip(panel["part"], panel["value"].to_numpy(dtype=float), strict=True))
        first = panel.iloc[0]
        for index, name in enumerate(order):
            share = float(panel_shares[name])
            value = float(panel_values[name])
            kind = (
                "Baseline"
                if name == "Baseline"
                else "Other"
                if name in ("Other channels", "Channel interactions")
                else "Increase"
                if share >= 0
                else "Decrease"
            )
            record = {dim: first[dim] for dim in panels} | {
                "name": name,
                "start": running,
                "end": running + share,
                "share": share,
                "value": value,
                "kind": kind,
                "position": len(order) - index,
            }
            records.append(record)
            running += share
    frame = pd.DataFrame(records)
    frame = frame.assign(
        text=[
            f"{share:.1%} ({_compact([value])[0]})" for share, value in zip(frame["share"], frame["value"], strict=True)
        ]
    )
    for dim in panels:
        frame = _ordered(frame, dim, shares[dim].values)
    return frame


def _place_labels(
    frame: pd.DataFrame, width: float, panel_count: int, label_chars: int
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Put each bar's text just past its end and stretch the axis so the longest text stays inside its panel."""
    # Row labels take about 0.09 inch a character and bar text about 0.07 on the default theme.
    panel = (width - 0.8 - 0.09 * label_chars) / panel_count
    fraction = min((0.07 * int(frame["text"].str.len().max()) + 0.25) / panel, 0.6)
    ends = frame[["start", "end"]].to_numpy(dtype=float)
    low = min(0.0, 1.05 * float(ends.min()))
    high = (float(ends.max()) - low * fraction) / (1 - fraction)
    # A tenth of an inch separates each text from its bar at any panel width.
    gap = 0.1 * (high - low) / panel
    placed = frame.assign(text_position=ends.max(axis=1) + gap)
    limits = (low, high)
    return placed, limits


def _share_breaks(frame: pd.DataFrame, limits: tuple[float, float]) -> list[float]:
    """Mark the share axis every quarter or every half up to the end of the longest bar."""
    low, high = limits
    # Past the bars the axis only makes room for text, and marks there would suggest shares above one.
    top = min(high, max(1.0, float(frame[["start", "end"]].to_numpy(dtype=float).max())))
    step = 0.25 if top - low <= 1.6 else 0.5
    first = math.ceil(low / step) * step
    breaks = [float(value) for value in np.arange(first, top + 1e-9, step)]
    return breaks


def _stacked_plot(totals: xr.Dataset, channels: Sequence[str] | None, include_baseline: bool, note: str) -> pn.ggplot:
    """Stack each channel's contribution by period under the joint increment or the whole response."""
    increments = totals["incremental_response"]
    labels = [str(label) for label in increments["channel"].values]
    averaged = abs(increments.mean(("chain", "draw")))
    sizes = averaged.sum([dim for dim in averaged.dims if dim != "channel"]).values
    shown, channel_note = _pick_channels(labels, sizes, channels, 10, "total increments")
    facets = _facets(increments.dims, ("time", "channel"))
    frame, components = _stacked_areas(totals, shown, facets, include_baseline)
    series = "Total response" if include_baseline else "Joint incremental response"
    line = (
        totals["reference_response"] if include_baseline else totals["reference_response"] - totals["baseline_response"]
    )
    total = _summarize(line, 0.5).assign(series=_outcome_words(series, totals))
    colors = _channel_colors(labels, shown) | {"Other channels": "#a6a6a6", "Baseline": "#d9d9d9"}
    zoom: list[PlotAddable] = []
    if include_baseline and not facets:
        # Starting the axis near the lowest baseline keeps the channels visible above a much larger baseline.
        low = 0.95 * float(frame.loc[frame["channel"] == "Baseline", "estimate"].min())
        high = 1.02 * max(
            float(total["estimate"].max()), float(frame.groupby("time", observed=True)["estimate"].sum().max())
        )
        zoom.append(pn.coord_cartesian(ylim=(low, high)))
    caption = " ".join(text for text in (channel_note, note) if text)

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes("time", "estimate"))
        + pn.geom_area(pn.aes(fill="channel"), position="stack", alpha=0.85)
        + pn.geom_line(pn.aes(color="series"), data=total, size=0.8)
        + pn.scale_fill_manual(
            values={key: colors[key] for key in components},
            breaks=components,
            labels=[_wrap(key, 28) for key in components],
        )
        + pn.scale_color_manual(values={_outcome_words(series, totals): "#262626"})
        + pn.labs(
            x="Time",
            y=_outcome_words("Response" if include_baseline else "Incremental response", totals),
            fill="",
            color="",
            caption=caption,
        )
        + _scales(frame, "time")
        + _facet(facets, stacked=True)
        + zoom
        + theme_mmmjax()
    )
    return plot


def _stacked_areas(
    effects: xr.Dataset, shown: list[str], facets: list[str], include_baseline: bool
) -> tuple[pd.DataFrame, list[str]]:
    """Summarize each shown channel and the rest by period in the order the stack draws them."""
    increments = effects["incremental_response"]
    labels = [label for label in increments["channel"].values if str(label) in shown]
    hidden = [label for label in increments["channel"].values if str(label) not in shown]
    parts = [increments.sel(channel=labels)]
    if hidden:
        # Summing hidden channels per draw keeps the stack as tall as the full decomposition.
        parts.append(increments.sel(channel=hidden).sum("channel").expand_dims(channel=["Other channels"]))
    stacked = xr.concat(parts, dim="channel")
    columns = ["time", "channel", *facets, "estimate"]
    summary = _summarize(stacked, 0.5)[columns]
    # Round-off can leave a zero increment slightly negative, and plotnine stacks negatives apart from the rest.
    noise = 1e-5 * summary["estimate"].abs().max()
    frame = summary.assign(estimate=summary["estimate"].mask(summary["estimate"].abs() < noise, 0.0))
    components = [*shown, *(["Other channels"] if hidden else [])]
    if include_baseline:
        baseline = _summarize(effects["baseline_response"], 0.5).assign(channel="Baseline")
        frame = pd.concat([frame, baseline[columns]], ignore_index=True)
        # The last category sits at the bottom of a plotnine stack, so the baseline goes last.
        components.append("Baseline")
    layers = _ordered(frame, "channel", components)
    return layers, components
