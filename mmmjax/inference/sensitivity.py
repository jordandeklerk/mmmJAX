"""Check how much posterior draws depend on the priors and the likelihood."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from mmmjax.inference.priors import Prior

__all__ = ["psense_summary"]


def psense_summary(
    results: xr.DataTree,
    *,
    priors: Mapping[str, Prior] | None = None,
    metrics: xr.Dataset | None = None,
    var_names: Sequence[str] | None = None,
    per_prior: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Measure how much each posterior quantity depends on the priors and the likelihood.

    Computes ArviZ's power-scaling sensitivity for every element of every
    parameter. Raising the prior or the likelihood to a power slightly above
    or below one reweights the posterior draws without refitting, and each
    value measures how far that moves an element's distribution. A diagnosis
    column flags elements whose values pass ArviZ's threshold of 0.05.

    A high prior value with a low likelihood value means the prior decides
    the estimate and the data adds little. High values in both columns point
    to a prior that disagrees with the data. Values near the threshold can be
    noise, especially for long-tailed quantities such as returns.

    The log prior comes from ``priors`` when it is given, evaluated at every
    posterior draw, and otherwise from the ``log_prior`` group that
    ``generated_quantities`` returned. Scaling every prior together can
    credit one prior with another's effect when parameters are correlated,
    so each prior also gets a column of its own.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with ``posterior`` and ``log_likelihood`` groups.
    priors : mapping of str to Prior, optional
        The ``Prior`` objects that ``log_density`` uses, keyed by parameter
        name. Every posterior parameter needs one. Omit to read the
        ``log_prior`` group of ``results``.
    metrics : xarray.Dataset, optional
        Output of ``media_metrics`` or another dataset of values for the
        posterior draws of ``results``. Omit to measure the parameters.
    var_names : sequence of str, optional
        Parameters, or variables of ``metrics``, to measure. Defaults to every
        one with chain and draw axes.
    per_prior : bool, default True
        Add one column for each prior, scaled on its own.
    **kwargs
        Further keywords for ``arviz_stats.psense_summary``, such as
        ``threshold``, ``alphas``, or ``round_to``.

    Returns
    -------
    pandas.DataFrame
        One row per element with ``prior``, ``likelihood``, and ``diagnosis``
        columns. With two or more priors, a ``<name> prior`` column follows
        for each.
    """
    if not isinstance(per_prior, bool):
        raise TypeError(f"per_prior must be a bool, got {type(per_prior).__name__}")
    tree, names = _sensitivity_tree(results, priors, metrics, var_names)
    from arviz_stats.psense import psense
    from arviz_stats.psense import psense_summary as arviz_psense_summary

    summary = cast(pd.DataFrame, arviz_psense_summary(tree, var_names=names, **kwargs))
    priors_scaled = [str(name) for name in tree["log_prior"].to_dataset().data_vars]
    if not per_prior or len(priors_scaled) < 2:
        return summary
    from arviz_base import dataset_to_dataframe

    shared = {key: kwargs[key] for key in ("alphas", "coords", "filter_vars", "sample_dims") if key in kwargs}
    separate = [
        psense(tree, var_names=names, group="prior", group_var_names=[name], **shared) for name in priors_scaled
    ]
    stacked = xr.concat(separate, dim="component").assign_coords(component=[f"{name} prior" for name in priors_scaled])
    columns = dataset_to_dataframe(stacked, sample_dims=["component"]).T
    combined = pd.concat([summary, columns.round(kwargs.get("round_to", 3))], axis=1)
    return combined


def _sensitivity_tree(
    results: object,
    priors: Mapping[str, Prior] | None,
    metrics: xr.Dataset | None,
    var_names: Sequence[str] | None,
) -> tuple[xr.DataTree, list[str]]:
    """Gather the draws to measure with the log prior and log likelihood that reweight them."""
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray DataTree, got {type(results).__name__}")
    for group in ("posterior", "log_likelihood"):
        if group not in results.children:
            raise ValueError(f"results has no {group} group")
    posterior = results["posterior"].to_dataset()
    if priors is not None:
        _require_priors(posterior, priors)
    elif "log_prior" not in results.children:
        raise ValueError(
            "results has no log_prior group. Pass the priors that log_density uses, "
            "or return 'log_prior' from generated_quantities"
        )
    draws = posterior if metrics is None else _metric_draws(metrics, posterior)
    names = _measured_names(draws, var_names, "posterior" if metrics is None else "metrics")
    log_prior = results["log_prior"].to_dataset() if priors is None else _log_prior(posterior, priors)
    # A metric computed on a subset of draws is reweighted with the log terms of those draws.
    selection = {"chain": draws["chain"].values, "draw": draws["draw"].values}
    log_likelihood = results["log_likelihood"].to_dataset().sel(selection)
    tree = xr.DataTree.from_dict(
        {"posterior": draws, "log_prior": log_prior.sel(selection), "log_likelihood": log_likelihood}
    )
    return tree, names


def _require_priors(posterior: xr.Dataset, priors: object) -> None:
    """Check that the priors name every posterior parameter and nothing else."""
    if not isinstance(priors, Mapping):
        raise TypeError(f"priors must be a mapping of parameter names to Prior objects, got {type(priors).__name__}")
    names = [str(name) for name in posterior.data_vars]
    unknown = [str(name) for name in priors if name not in names]
    if unknown:
        raise ValueError(f"priors must name posterior parameters, got {unknown[0]!r}")
    missing = [name for name in names if name not in priors]
    if missing:
        raise ValueError(f"priors must give every posterior parameter a prior, missing {missing[0]!r}")
    for name in names:
        if not isinstance(priors[name], Prior):
            raise TypeError(f"priors[{name!r}] must be a Prior, got {type(priors[name]).__name__}")


def _metric_draws(metrics: object, posterior: xr.Dataset) -> xr.Dataset:
    """Keep the metrics that vary by draw and check that they come from the fit's posterior draws."""
    if not isinstance(metrics, xr.Dataset):
        raise TypeError(f"metrics must be an xarray Dataset, got {type(metrics).__name__}")
    drawn = metrics.attrs.get("group", "posterior")
    if drawn != "posterior":
        raise ValueError(f"metrics must come from posterior draws, got {drawn!r}")
    kept = [name for name, values in metrics.data_vars.items() if {"chain", "draw"} <= set(values.dims)]
    if not kept:
        raise ValueError("metrics has no variable with chain and draw axes")
    # Labels such as channel types sit on no axis, and ArviZ cannot line them up across variables.
    draws = metrics[kept].reset_coords(drop=True)
    for dim in ("chain", "draw"):
        if not np.isin(draws[dim].values, posterior[dim].values).all():
            raise ValueError(f"metrics must use draws of results, got {dim} labels that results lacks")
    return draws


def _measured_names(draws: xr.Dataset, var_names: Sequence[str] | None, source: str) -> list[str]:
    """Resolve the variables to measure and reject names the draws lack."""
    available = [str(name) for name in draws.data_vars]
    if var_names is None:
        return available
    if isinstance(var_names, str):
        raise TypeError("var_names must be a sequence of names, not a single string")
    missing = [name for name in var_names if name not in available]
    if missing:
        raise ValueError(f"var_names has no {source} variable {', '.join(repr(name) for name in missing)}")
    names = list(var_names)
    return names


def _log_prior(posterior: xr.Dataset, priors: Mapping[str, Prior]) -> xr.Dataset:
    """Evaluate each parameter's prior at every posterior draw and sum it within the draw."""
    terms = {}
    for name in [str(name) for name in posterior.data_vars]:
        prior = priors[name]
        draws = posterior[name].transpose("chain", "draw", ...)
        values = np.asarray(prior.logpdf(jnp.asarray(draws.values)), dtype=np.float64)
        # Power scaling needs one prior term per draw, and each element adds to it.
        summed = values.reshape(values.shape[0], values.shape[1], -1).sum(axis=-1)
        terms[name] = (("chain", "draw"), summed)
    log_prior = xr.Dataset(terms, coords={"chain": posterior["chain"].values, "draw": posterior["draw"].values})
    return log_prior
