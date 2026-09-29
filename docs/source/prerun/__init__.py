"""Fits and analysis results that the executed docs pages load instead of computing."""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import xarray as xr

import mmmjax as mj


def stored(
    name: str,
    compute: Callable[[], xr.DataTree | xr.Dataset],
    groups: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
) -> xr.DataTree | xr.Dataset:
    """Load the stored result called ``name``.

    A missing file is computed with ``compute`` and saved first. ``groups`` keeps only the groups a
    page reads, and ``variables`` only the data variables, so the file stays small. A result saved
    without groups, such as the Dataset an analysis function returns, loads as a Dataset.
    """
    path = Path(__file__).with_name(f"{name}.nc")
    if not path.exists():
        # Read the Docs stops a build at fifteen minutes, so it never samples or optimizes.
        if os.environ.get("READTHEDOCS") == "True":
            raise FileNotFoundError(f"{path.name} is missing. Build the docs locally to save it, then commit it")
        result = compute()
        if groups is not None:
            kept = xr.DataTree.from_dict({group: result[group].to_dataset() for group in groups})
            kept.attrs = dict(result.attrs)
            result = kept
        if variables is not None:
            result = result[list(variables)]
        # Analysis results record boolean settings, which netCDF only allows as an HDF5 extension.
        result.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)
    # Loading into memory lets the pages display values the way a fresh computation does.
    loaded = xr.load_datatree(path, engine="h5netcdf")
    if not loaded.children:
        dataset = loaded.to_dataset()
        return dataset
    return loaded


def first_model_results(model: mj.Model) -> xr.DataTree:
    """Load the stored fit of the model from A first model with every group."""
    results = stored(
        "first_model",
        lambda: mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7),
    )
    return results


def first_model_prior_results(model: mj.Model, priors: Mapping[str, mj.Prior]) -> xr.DataTree:
    """Load the stored prior draws of the model from A first model."""
    results = stored(
        "first_model_prior",
        lambda: mj.sample_prior(model, priors, draws=500, seed=0),
        groups=["prior", "prior_predictive", "observed_data"],
    )
    return results


def first_model_curves(model: mj.Model, results: xr.DataTree) -> xr.Dataset:
    """Load the stored response curves of the model from A first model."""
    curves = stored("first_model_curves", lambda: mj.response_curves(model, results, quantity="mu"))
    return curves


def first_model_plan(model: mj.Model, results: xr.DataTree) -> xr.Dataset:
    """Load the stored budget plan that moves the current budget freely among the channels."""
    plan = stored("first_model_plan", lambda: mj.optimize_budget(model, results, quantity="mu", include_metrics=True))
    return plan


def first_model_limited_plan(model: mj.Model, results: xr.DataTree) -> xr.Dataset:
    """Load the stored budget plan that keeps each channel within 30 percent of its spending."""
    plan = stored(
        "first_model_limited_plan",
        lambda: mj.optimize_budget(
            model,
            results,
            quantity="mu",
            spend_constraint_lower=0.3,
            spend_constraint_upper=0.3,
            include_metrics=True,
        ),
    )
    return plan
