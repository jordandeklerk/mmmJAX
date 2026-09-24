"""Fitted results for the User Guide."""

from collections.abc import Callable, Sequence
from pathlib import Path

import xarray as xr

import mmmjax as mj


def stored(name: str, fit: Callable[[], xr.DataTree], groups: Sequence[str] | None = None) -> xr.DataTree:
    """Load the stored results called ``name``.

    A missing file is fitted with ``fit`` and saved first. ``groups`` keeps only the groups a page
    reads so the file stays small.
    """
    path = Path(__file__).with_name(f"{name}.nc")
    if not path.exists():
        results = fit()
        if groups is not None:
            kept = xr.DataTree.from_dict({group: results[group].to_dataset() for group in groups})
            kept.attrs = dict(results.attrs)
            results = kept
        results.to_netcdf(path, engine="h5netcdf")
    # Loading into memory lets the pages display values the way a fresh fit does.
    loaded = xr.load_datatree(path, engine="h5netcdf")
    return loaded


def first_model_results(model: mj.Model) -> xr.DataTree:
    """Load the stored fit of the model from A first model."""
    results = stored(
        "first_model",
        lambda: mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7),
    )
    return results
