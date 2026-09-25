"""Plots of optimized budget allocations."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr

from mmmjax.plotting._layers import _bar_layout, _compact
from mmmjax.plotting._summary import (
    _ci_prob,
    _ordered,
    _percent,
    _pick_channels,
    _plan_spend,
    _require_draws,
    _summarize,
)
from mmmjax.plotting.theme import _colors, theme_mmmjax

__all__ = ["plot_budget_response", "plot_budget_spend"]


def plot_budget_response(
    plan: xr.Dataset,
    *,
    channels: Sequence[str] | None = None,
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Break the plan's change in incremental response down by channel.

    The first bar is the incremental response of the reference plan and the
    last is that of the optimized plan. Each bar between them adds one
    channel's change, cuts first and then gains, each from largest to
    smallest. The bars use posterior means so they add up, and the axis
    starts near the smallest running total so the changes stay visible. The
    subtitle gives the change in total response with its credible interval
    and the share of draws in which the plan gains. With many channels the
    figure widens so each bar keeps its width, and a notebook shows it at
    full size in a box that scrolls sideways. The plan needs
    ``include_metrics=True`` to record each channel's incremental response.

    Parameters
    ----------
    plan : xarray.Dataset
        Output of ``optimize_budget`` with ``include_metrics=True``.
    channels : sequence of str, optional
        Channels to show separately. Defaults to every channel. Channels left
        out are summed into one bar before the optimized total.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Waterfall from the reference to the optimized incremental response.
    """
    _plan_spend(plan)
    if "incremental_response" not in plan.data_vars:
        raise ValueError(
            "plan is missing 'incremental_response'. Run optimize_budget with include_metrics=True to record it"
        )
    increments = plan["incremental_response"]
    _require_draws(increments, "plan['incremental_response']")
    probability = _ci_prob(ci_prob)
    extra = [dim for dim in increments.dims if dim not in ("chain", "draw", "allocation", "channel")]
    means = (increments.sum(extra) if extra else increments).mean(("chain", "draw"))
    start = float(means.sel(allocation="reference").sum())
    changes = (means.sel(allocation="optimized") - means.sel(allocation="reference")).to_series()
    changes.index = changes.index.astype(str)
    shown, _ = _pick_channels(changes.index, changes.to_numpy(), channels, len(changes), "changes")
    steps = changes[shown]
    # Cuts come first and gains after, each from largest to smallest, and the channels left out close the run.
    ordered = pd.concat([steps[steps < 0].sort_values(), steps[steps >= 0].sort_values(ascending=False)])
    if len(shown) < len(changes):
        ordered = pd.concat([ordered, pd.Series({"Other channels": changes.drop(shown).sum()})])
    frame = _waterfall(start, ordered)
    low, high = _waterfall_range(frame)
    increase, decrease = _colors(2)
    figure, texts, layout = _bar_layout(list(frame["name"]), 0.8)

    plot: pn.ggplot = (
        figure(frame)
        + pn.geom_rect(pn.aes(xmin="position - 0.3", xmax="position + 0.3", ymin="bottom", ymax="top", fill="kind"))
        + pn.geom_text(pn.aes(x="position", y="top", label="label"), va="bottom", size=10, nudge_y=0.01 * (high - low))
        + pn.scale_fill_manual(values={"Total": "#8c8c8c", "Increase": increase, "Decrease": decrease})
        + pn.scale_x_continuous(breaks=list(frame["position"]), labels=texts)
        + pn.scale_y_continuous(labels=_compact)
        + pn.coord_cartesian(ylim=(low, high))
        + pn.guides(fill="none")
        + pn.labs(x="", y="Incremental response", subtitle=_response_note(plan, probability))
        + theme_mmmjax()
        + layout
    )
    return plot


def plot_budget_spend(plan: xr.Dataset, *, channels: Sequence[str] | None = None) -> pn.ggplot:
    """Plot how the optimized plan changes each channel's spending.

    Each bar is a channel's optimized spending minus its reference spending,
    labeled with the change. Cuts come first and increases after, each from
    largest to smallest. With many channels the figure widens so each bar
    keeps its width, and a notebook shows it at full size in a box that
    scrolls sideways.

    Parameters
    ----------
    plan : xarray.Dataset
        Output of ``optimize_budget``.
    channels : sequence of str, optional
        Channels to show. Defaults to every channel.

    Returns
    -------
    plotnine.ggplot
        Changes in spending by channel.
    """
    spend = _plan_spend(plan)
    changes = (spend.sel(allocation="optimized") - spend.sel(allocation="reference")).to_series()
    changes.index = changes.index.astype(str)
    shown, _ = _pick_channels(changes.index, changes.to_numpy(), channels, len(changes), "changes")
    steps = changes[shown]
    ordered = pd.concat([steps[steps < 0].sort_values(), steps[steps >= 0].sort_values(ascending=False)])
    values = ordered.to_numpy()
    span = float(np.ptp(np.append(values, 0.0))) or 1.0
    frame = pd.DataFrame(
        {
            "channel": ordered.index,
            "change": values,
            "kind": np.where(values < 0, "Decrease", "Increase"),
            "label": [_signed(value) for value in values],
            # Labels sit just past the end of each bar, above gains and below cuts.
            "label_position": values + np.where(values < 0, -0.04, 0.04) * span,
        }
    )
    frame = _ordered(frame, "channel", list(ordered.index))
    increase, decrease = _colors(2)
    names = [str(name) for name in frame["channel"]]
    figure, texts, layout = _bar_layout(names, 0.8)

    plot: pn.ggplot = (
        figure(frame, pn.aes("channel", "change", fill="kind"))
        + pn.geom_hline(yintercept=0, color="#8c8c8c", size=0.6)
        + pn.geom_col(width=0.6)
        + pn.geom_text(pn.aes(y="label_position", label="label"), size=10)
        + pn.scale_fill_manual(values={"Increase": increase, "Decrease": decrease})
        + pn.scale_x_discrete(labels=dict(zip(names, texts, strict=True)))
        + pn.scale_y_continuous(labels=_compact)
        + pn.guides(fill="none")
        + pn.labs(x="", y="Change in spend")
        + theme_mmmjax()
        + layout
    )
    return plot


def _waterfall(start: float, changes: pd.Series) -> pd.DataFrame:
    """Lay out the bars that run from the reference total through each change to the optimized total."""
    running = start + np.cumsum(changes.to_numpy())
    before = np.concatenate([[start], running[:-1]])
    end = float(running[-1]) if len(running) else start
    rows = [{"name": "Reference", "bottom": 0.0, "top": start, "kind": "Total", "label": _compact([start])[0]}]
    for name, value, low, high in zip(changes.index, changes.to_numpy(), before, running, strict=True):
        rows.append(
            {
                "name": str(name),
                "bottom": min(low, high),
                "top": max(low, high),
                "kind": "Decrease" if value < 0 else "Increase",
                "label": _signed(value),
            }
        )
    rows.append({"name": "Optimized", "bottom": 0.0, "top": end, "kind": "Total", "label": _compact([end])[0]})
    frame = pd.DataFrame(rows).assign(position=lambda table: np.arange(1, len(table) + 1))
    return frame


def _waterfall_range(frame: pd.DataFrame) -> tuple[float, float]:
    """Zoom the value axis onto the running totals so small changes stay visible."""
    path = frame.loc[frame["kind"] != "Total", ["bottom", "top"]].to_numpy().ravel()
    totals = frame.loc[frame["kind"] == "Total", "top"].to_numpy()
    values = np.concatenate([path, totals])
    low, high = float(values.min()), float(values.max())
    margin = 0.25 * (high - low) if high > low else max(abs(high), 1.0) * 0.05
    padded = (low - margin, high + margin)
    return padded


def _response_note(plan: xr.Dataset, probability: float) -> str:
    """Summarize the total response change with its interval and the share of draws that gain."""
    change = plan["response_change"]
    _require_draws(change, "plan['response_change']")
    extra = [dim for dim in change.dims if dim not in ("chain", "draw")]
    joint = change.sum(extra) if extra else change
    total = _summarize(joint, probability).iloc[0]
    gain = float((joint > 0).mean())
    estimate, lower, upper = _compact([total["estimate"], total["lower"], total["upper"]])
    note = (
        f"Response change {estimate}, {_percent(probability)} interval {lower} to {upper}. "
        f"The plan gains in {gain:.0%} of draws."
    )
    return note


def _signed(value: float) -> str:
    """Write a change with an explicit sign."""
    text = _compact([value])[0]
    signed = text if text.startswith("-") else f"+{text}"
    return signed
