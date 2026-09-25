"""House style shared by the mmmjax plots."""

from typing import Any

import plotnine as pn
from matplotlib.typing import RcKeyType

__all__ = ["theme_mmmjax"]


def theme_mmmjax() -> pn.theme:
    """Return the plotnine theme of the mmmjax plots.

    White panels carry left and bottom axis lines with outward ticks and no
    grid. Text follows the sizes of the documentation figures, and the figure
    measures 12 by 7 inches at 100 dots per inch. Every plotnine plot in
    mmmjax already ends with this theme. Add it to your own plots to match
    them, and add ``plotnine.theme`` settings after it to change any element.

    Returns
    -------
    plotnine.theme
        Complete theme built on ``plotnine.theme_classic``.
    """
    base = pn.theme_classic(base_size=14)
    styled = base + pn.theme(
        figure_size=(12, 7),
        dpi=100,
        text=pn.element_text(color="#262626"),
        axis_title=pn.element_text(size=15),
        plot_title=pn.element_text(size=16),
        axis_line=pn.element_line(color="#545454", size=0.8),
        axis_ticks=pn.element_line(color="#545454", size=0.8),
        axis_ticks_length=3.5,
        legend_key=pn.element_blank(),
        legend_text=pn.element_text(ma="left"),
        strip_background=pn.element_blank(),
        strip_text=pn.element_text(size=14),
    )
    return styled


def _colors(count: int) -> list[str]:
    """Cycle the ArviZ dark grid palette so each of ``count`` levels gets a color."""
    palette = [
        "#2a2eec",
        "#fa7c17",
        "#328c06",
        "#c10c90",
        "#933708",
        "#65e5f3",
        "#e6e135",
        "#1ccd6a",
        "#bd8ad5",
        "#b16b57",
    ]
    colors = [palette[index % len(palette)] for index in range(count)]
    return colors


def _matplotlib_style() -> dict[RcKeyType, Any]:
    """Give the ArviZ diagnostics the look of the plotnine theme."""
    style: dict[RcKeyType, Any] = {
        "axes.prop_cycle": f"cycler(color={_colors(10)!r})",
        "axes.facecolor": "white",
        "axes.edgecolor": "#545454",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#262626",
        "text.color": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,
        "legend.frameon": False,
        "date.converter": "concise",
    }
    return style
