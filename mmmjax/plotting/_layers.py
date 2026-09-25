"""Plot layers and scales shared by the plotnine plots."""

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import plotnine as pn
from plotnine.options import get_option

from mmmjax.plotting._display import _ScrollingPlot
from mmmjax.plotting._summary import _distinct_shortened, _label, _percent, _wrap
from mmmjax.plotting.theme import _colors

if TYPE_CHECKING:
    from plotnine.ggplot import PlotAddable


type _Side = Literal["t", "b", "l", "r", "unit"]


def _bands(
    frame: pd.DataFrame,
    *,
    x: str,
    color: str | None,
    labels: Iterable[Hashable],
    probability: float,
) -> pn.ggplot:
    """Draw point estimates as lines over their credible bands with one color per level."""
    if color is None:
        single = _colors(1)[0]
        plot = (
            pn.ggplot(frame, pn.aes(x, "estimate"))
            + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), fill=single, alpha=0.2)
            + pn.geom_line(color=single, size=1)
        )
        return plot
    levels = list(dict.fromkeys(str(label) for label in labels))
    colors = dict(zip(levels, _colors(len(levels)), strict=True))
    title = f"{_label(color)}, {_percent(probability)} interval"
    plot = (
        pn.ggplot(frame, pn.aes(x, "estimate"))
        + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper", fill=color), alpha=0.2)
        + pn.geom_line(pn.aes(color=color), size=1)
        + pn.scale_color_manual(values=colors, breaks=levels, labels=[_wrap(level, 28) for level in levels])
        + pn.scale_fill_manual(values=colors, breaks=levels, labels=[_wrap(level, 28) for level in levels])
        + pn.labs(color=title, fill=title)
    )
    return plot


def _facet(dims: Sequence[str], *, stacked: bool = False) -> pn.facet_wrap | pn.facet_null:
    """Give each combination of the leftover axes its own panel and stack time series vertically."""
    facet = pn.facet_wrap(list(dims), ncol=1 if stacked else None, scales="free_y") if dims else pn.facet_null()
    return facet


def _scales(frame: pd.DataFrame, x: str | None = None, *, y: bool = True) -> list["PlotAddable"]:
    """Label dates by month and large numbers compactly so crowded axes stay readable."""
    scales: list[PlotAddable] = [pn.scale_y_continuous(labels=_compact)] if y else []
    if x is not None and pd.api.types.is_datetime64_any_dtype(frame[x]):
        scales.append(pn.scale_x_datetime(date_labels="%b %Y"))
    elif x is not None and pd.api.types.is_numeric_dtype(frame[x]):
        scales.append(pn.scale_x_continuous(labels=_compact))
    return scales


def _bar_layout(labels: Sequence[str], slot: float) -> tuple[type[pn.ggplot], list[str], list["PlotAddable"]]:
    """Size a bar chart so each bar keeps its slot and tilt and shorten its labels."""
    # The y axis, its labels, and its title take about an inch and a half beside the bars.
    width = max(12.0, 1.5 + len(labels) * slot)
    texts, axis = _channel_axis(labels)
    layout: list[PlotAddable] = [axis]
    if width > 12:
        layout.append(_figure_margins(width, 7))
    figure = _ScrollingPlot if width > 12 else pn.ggplot
    return figure, texts, layout


def _channel_axis(labels: Sequence[str]) -> tuple[list[str], pn.theme]:
    """Tilt category labels at 45 degrees and shorten long ones so every label takes the same room."""
    # Meridian's charts stop labels at 180 pixels, about 27 characters of axis text.
    shortened = _distinct_shortened([str(label) for label in labels], 27)
    tilted = pn.theme(axis_text_x=pn.element_text(rotation=45, ha="right"))
    return shortened, tilted


def _figure_margins(width: float, height: float) -> pn.theme:
    """Keep the default figure's spacing in inches on a larger figure."""
    # plotnine sizes spacing as fractions of the figure's width or height, so each fraction shrinks to keep its inches.
    across, down = 12 / width, 7 / height
    base = float(get_option("base_margin"))
    legend_text: dict[_Side, Any] = {
        "t": base / 1.5 * down,
        "b": base / 1.5 * down,
        "l": base / 1.5 * across,
        "r": base / 1.5 * across,
        "unit": "fig",
    }
    legend_title: dict[_Side, Any] = {
        "t": base * down,
        "b": base / 2 * down,
        "l": 2 * base * across,
        "r": 2 * base * across,
        "unit": "fig",
    }
    margins = pn.theme(
        figure_size=(width, height),
        plot_margin=base * across,
        panel_spacing_x=base * across,
        panel_spacing_y=base * down,
        axis_title_x=pn.element_text(margin={"t": base * down, "unit": "fig"}),
        axis_title_y=pn.element_text(margin={"r": base * across, "unit": "fig"}),
        plot_subtitle=pn.element_text(margin={"b": base * down, "unit": "fig"}),
        plot_caption=pn.element_text(margin={"t": base * down, "unit": "fig"}),
        legend_box_spacing=3 * base * across,
        legend_text=pn.element_text(margin=legend_text),
        legend_title=pn.element_text(margin=legend_title),
    )
    return margins


def _compact(breaks: Sequence[float] | Sequence[str]) -> list[str]:
    """Write numbers to three significant digits with a magnitude suffix."""
    labels = []
    for brk in breaks:
        value = float(brk)
        magnitude = abs(value)
        if magnitude >= 1e9:
            text = f"{value / 1e9:.3g}B"
        elif magnitude >= 1e6:
            text = f"{value / 1e6:.3g}M"
        elif magnitude >= 1e3:
            text = f"{value / 1e3:.3g}K"
        else:
            text = f"{value:.3g}"
        labels.append(text)
    return labels
