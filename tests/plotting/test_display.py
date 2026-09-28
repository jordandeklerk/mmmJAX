"""Tests for the notebook display of plots wider than a notebook cell."""

from io import BytesIO

import numpy as np
import pytest
import xarray as xr
from plotnine.exceptions import PlotnineError

from mmmjax import plot_media_metrics


def _returns(count):
    labels = [f"Channel {index:02d}" for index in range(count)]
    metrics = xr.Dataset(
        {
            "roi": (("chain", "draw", "channel"), 2.0 + 0.1 * np.random.default_rng(3).normal(size=(2, 5, count))),
            "reference_spend": (("channel",), np.arange(1.0, count + 1.0)),
        },
        coords={"chain": [0, 1], "draw": np.arange(5), "channel": labels},
    )
    return metrics


def test_scrolling_plots_shrink_to_a_readable_width_before_they_scroll():
    full_width = round((1.5 + 30 * 0.8) * 100)
    smallest_width = round(0.6 * full_width)

    data, metadata = plot_media_metrics(_returns(30))._repr_mimebundle_()

    assert set(data) == {"text/html", "image/svg+xml"}
    assert data["text/html"].startswith('<div style="overflow-x: auto; max-width: 100%;">')
    assert f'width="{full_width}" height="500"' in data["text/html"]
    assert f"max-width: {full_width}px; min-width: {smallest_width}px; height: auto;" in data["text/html"]
    assert data["image/svg+xml"].lstrip().startswith("<?xml")
    assert metadata == {}


def test_scrolling_plots_save_past_the_plotnine_size_guard():
    plot = plot_media_metrics(_returns(60))
    buffer = BytesIO()

    plot.save(buffer, "png", verbose=False)

    assert buffer.tell() > 0
    with pytest.raises(PlotnineError, match="exceed 25 inches"):
        plot.save(BytesIO(), "png", verbose=False, limitsize=True)
