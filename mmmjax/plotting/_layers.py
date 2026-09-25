"""Plot layers and scales shared by the plotnine plots."""

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import plotnine as pn
from plotnine.options import get_option

from mmmjax.plotting._display import _ScrollingPlot
from mmmjax.plotting._summary import _label, _percent, _shorten, _wrap
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
    """Size a bar chart so each bar keeps its slot and label the bars the way Meridian's charts do."""
    # The y axis, its labels, and its title take about an inch and a half beside the bars.
    width = max(12.0, 1.5 + len(labels) * slot)
    texts, axis = _channel_axis(labels)
    layout: list[PlotAddable] = [axis]
    if width > 12:
        layout.extend([_wide_margins(width), pn.theme(figure_size=(width, 7))])
    figure = _ScrollingPlot if width > 12 else pn.ggplot
    return figure, texts, layout


def _channel_axis(labels: Sequence[str]) -> tuple[list[str], pn.theme]:
    """Tilt category labels at 45 degrees and shorten long ones so every label takes the same room."""
    names = [str(label) for label in labels]
    # Meridian's charts stop labels at 180 pixels, about 27 characters of axis text.
    limit = 27
    longest = max((len(name) for name in names), default=0)
    shortened = [_shorten(name, limit) for name in names]
    # Shortening must keep the labels distinct, so the limit grows until no two collide.
    while len(set(shortened)) < len(set(names)) and limit < longest:
        limit += 4
        shortened = [_shorten(name, limit) for name in names]
    tilted = pn.theme(axis_text_x=pn.element_text(rotation=45, ha="right"))
    return shortened, tilted


def _wide_margins(width: float) -> pn.theme:
    """Keep the default figure's horizontal spacing in inches on a wider figure."""
    # plotnine sizes horizontal spacing as fractions of the figure width, so each fraction shrinks to keep its inches.
    shrink = 12 / width
    base = float(get_option("base_margin"))
    # Vertical fractions follow the height, so only the left and right margins shrink.
    legend_text: dict[_Side, Any] = {
        "t": base / 1.5,
        "b": base / 1.5,
        "l": base / 1.5 * shrink,
        "r": base / 1.5 * shrink,
        "unit": "fig",
    }
    legend_title: dict[_Side, Any] = {
        "t": base,
        "b": base / 2,
        "l": 2 * base * shrink,
        "r": 2 * base * shrink,
        "unit": "fig",
    }
    margins = pn.theme(
        plot_margin=base * shrink,
        panel_spacing_x=base * shrink,
        axis_title_y=pn.element_text(margin={"r": base * shrink, "unit": "fig"}),
        legend_box_spacing=3 * base * shrink,
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
