"""Notebook display for plots wider than a notebook cell."""

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotnine as pn

if TYPE_CHECKING:
    from plotnine.typing import MimeBundle


class _ScrollingPlot(pn.ggplot):
    """Show a plot wider than a notebook cell at full size in a box that scrolls sideways."""

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
        """Draw the plot at full size in a box that scrolls instead of shrinking it to fit the cell."""
        buffer = BytesIO()
        self.save(buffer, "svg", verbose=False)
        width, height = self.theme.getp("figure_size")
        dpi = self.theme.getp("dpi")
        # A vector image stays sharp at any width, where a bitmap of hundreds of bars would be enormous.
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        markup = (
            '<div style="overflow-x: auto; max-width: 100%;">'
            f'<img src="data:image/svg+xml;base64,{encoded}" width="{round(width * dpi)}" '
            f'height="{round(height * dpi)}" style="max-width: none;" alt="Plot">'
            "</div>"
        )
        bundle: MimeBundle = ({"text/html": markup, "image/svg+xml": buffer.getvalue().decode("utf-8")}, {})
        return bundle
