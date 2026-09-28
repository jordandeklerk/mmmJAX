"""Notebook display for plots wider than a notebook cell."""

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotnine as pn

if TYPE_CHECKING:
    from plotnine.typing import MimeBundle


class _ScrollingPlot(pn.ggplot):
    """Fit a wide plot to its notebook cell, and scroll it once shrinking would make it hard to read."""

    def save(
        self,
        filename: str | Path | BytesIO | None = None,
        format: str | None = None,
        path: str = "",
        width: float | None = None,
        height: float | None = None,
        units: str = "in",
        dpi: int | None = None,
        limitsize: bool | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the plot past plotnine's size guard because these plots are wide on purpose."""
        # plotnine refuses figures over 25 inches to catch sizes given in the wrong units.
        guarded = False if limitsize is None else limitsize
        super().save(filename, format, path, width, height, units, dpi, guarded, verbose, **kwargs)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> "MimeBundle":
        """Shrink the plot to fit the cell down to a readable width, and scroll it past that."""
        buffer = BytesIO()
        self.save(buffer, "svg", verbose=False)
        width, height = self.theme.getp("figure_size")
        dpi = self.theme.getp("dpi")
        full_width = round(width * dpi)
        full_height = round(height * dpi)
        # Below about 60 percent of full size the axis labels get too small, so the box scrolls from there.
        smallest_width = round(0.6 * full_width)
        # A vector image stays sharp at any width, where a bitmap of hundreds of bars would be enormous.
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        markup = (
            '<div style="overflow-x: auto; max-width: 100%;">'
            f'<img src="data:image/svg+xml;base64,{encoded}" width="{full_width}" height="{full_height}" '
            f'style="width: 100%; max-width: {full_width}px; min-width: {smallest_width}px; height: auto;" '
            'alt="Plot">'
            "</div>"
        )
        bundle: MimeBundle = ({"text/html": markup, "image/svg+xml": buffer.getvalue().decode("utf-8")}, {})
        return bundle
