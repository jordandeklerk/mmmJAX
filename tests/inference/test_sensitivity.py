"""Tests for the power-scaling sensitivity of posterior draws to the priors and the likelihood."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from arviz_stats import psense
from arviz_stats import psense_summary as arviz_psense_summary
from scipy import stats

from mmmjax import Prior, normal, psense_summary


def _fit():
    rng = np.random.default_rng(3)
    sample = {"chain": [0, 1], "draw": np.arange(300)}
    coefficient = rng.normal(0.5, 0.4, size=(2, 300, 3))
    sigma = rng.gamma(9.0, 0.05, size=(2, 300))
    posterior = xr.Dataset(
        {"coefficient": (("chain", "draw", "channel"), coefficient), "sigma": (("chain", "draw"), sigma)},
        coords=sample | {"channel": ["TV", "Search", "Radio"]},
    )
    log_likelihood = xr.Dataset(
        {
            "outcome": (
                ("chain", "draw", "time"),
                -0.5 * (coefficient.sum(axis=-1, keepdims=True) - 1.0) ** 2 / sigma[..., None] ** 2
                + rng.normal(size=(2, 300, 4)),
            )
        },
        coords=sample | {"time": range(4)},
    )
    results = xr.DataTree.from_dict({"posterior": posterior, "log_likelihood": log_likelihood})
    scale = np.array([0.3, 1.0, 3.0])
    priors = {
        "coefficient": Prior(normal, location=0.0, scale=scale),
        "sigma": Prior(normal, location=0.0, scale=0.5),
    }
    terms = {
        "coefficient": (("chain", "draw"), stats.norm.logpdf(coefficient, 0.0, scale).sum(axis=-1)),
        "sigma": (("chain", "draw"), stats.norm.logpdf(sigma, 0.0, 0.5)),
    }
    log_prior = xr.Dataset(terms, coords=sample)
    return results, priors, log_prior


def _with(results, **groups):
    tree = xr.DataTree.from_dict({name: results[name].to_dataset() for name in results.children} | groups)
    return tree


def test_psense_summary_evaluates_the_priors_at_every_posterior_draw():
    results, priors, log_prior = _fit()
    expected = arviz_psense_summary(_with(results, log_prior=log_prior))

    summary = psense_summary(results, priors=priors, per_prior=False)

    pd.testing.assert_frame_equal(summary, expected, rtol=1e-6, atol=1e-3)


def test_psense_summary_reads_the_log_prior_of_results_without_priors():
    results, _, log_prior = _fit()
    tree = _with(results, log_prior=log_prior)
    expected = arviz_psense_summary(tree)

    summary = psense_summary(tree, per_prior=False)

    pd.testing.assert_frame_equal(summary, expected, rtol=0, atol=0)


def test_psense_summary_scales_each_prior_on_its_own():
    results, priors, log_prior = _fit()
    alone = {
        name: psense(_with(results, log_prior=log_prior[[name]]), group="prior") for name in ("coefficient", "sigma")
    }
    expected = {
        name: np.round([*values["coefficient"].values, float(values["sigma"])], 3) for name, values in alone.items()
    }

    summary = psense_summary(results, priors=priors)

    assert list(summary.columns) == ["prior", "likelihood", "diagnosis", "coefficient prior", "sigma prior"]
    np.testing.assert_allclose(summary["coefficient prior"], expected["coefficient"], rtol=0, atol=1.5e-3)
    np.testing.assert_allclose(summary["sigma prior"], expected["sigma"], rtol=0, atol=1.5e-3)


def test_psense_summary_measures_metrics_of_the_same_draws():
    results, priors, log_prior = _fit()
    coefficient = results["posterior"].to_dataset()["coefficient"]
    # Channel types sit on the channel axis without indexing it, as media_metrics records them.
    roi = coefficient.assign_coords(channel_type=("channel", ["paid"] * 3)).rename("roi")
    metrics = roi.to_dataset().assign(reference_spend=("channel", [1.0, 2.0, 3.0])).isel(draw=slice(0, 200))
    metrics.attrs["group"] = "posterior"
    subset = _with(results, log_prior=log_prior).isel(draw=slice(0, 200))
    expected = arviz_psense_summary(subset, var_names=["coefficient"]).to_numpy()

    summary = psense_summary(results, priors=priors, metrics=metrics, per_prior=False)

    assert list(summary.index) == ["roi[TV]", "roi[Search]", "roi[Radio]"]
    np.testing.assert_array_equal(summary.to_numpy(), expected)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"results": xr.Dataset()}, TypeError, "results must be an xarray DataTree"),
        ({"priors": None}, ValueError, "results has no log_prior group. Pass the priors"),
        ({"priors": [Prior(normal, location=0.0, scale=1.0)]}, TypeError, "priors must be a mapping"),
        (
            {"priors": {"slope": Prior(normal, location=0.0, scale=1.0)}},
            ValueError,
            "priors must name posterior parameters, got 'slope'",
        ),
        ({"priors": {"sigma": Prior(normal, location=0.0, scale=1.0)}}, ValueError, "missing 'coefficient'"),
        (
            {"priors": {"sigma": Prior(normal, location=0.0, scale=1.0), "coefficient": 0.3}},
            TypeError,
            r"priors\['coefficient'\] must be a Prior",
        ),
        ({"var_names": "sigma"}, TypeError, "var_names must be a sequence of names"),
        ({"var_names": ["slope"]}, ValueError, "var_names has no posterior variable 'slope'"),
        ({"per_prior": 1}, TypeError, "per_prior must be a bool"),
        ({"metrics": {}}, TypeError, "metrics must be an xarray Dataset"),
        (
            {"metrics": xr.Dataset({"roi": ("channel", [1.0])})},
            ValueError,
            "metrics has no variable with chain and draw axes",
        ),
    ],
)
def test_psense_summary_rejects_invalid_arguments(options, error, message):
    results, priors, _ = _fit()
    arguments = {"results": results, "priors": priors} | options

    with pytest.raises(error, match=message):
        psense_summary(arguments.pop("results"), **arguments)


def test_psense_summary_requires_metrics_of_posterior_draws_from_results():
    results, priors, _ = _fit()
    metrics = results["posterior"].to_dataset()[["sigma"]]
    from_prior = metrics.assign_attrs(group="prior")
    elsewhere = metrics.assign_coords(draw=metrics["draw"] + 1000)

    with pytest.raises(ValueError, match="metrics must come from posterior draws, got 'prior'"):
        psense_summary(results, priors=priors, metrics=from_prior)
    with pytest.raises(ValueError, match="metrics must use draws of results"):
        psense_summary(results, priors=priors, metrics=elsewhere)


def test_psense_summary_requires_a_log_likelihood():
    results, priors, _ = _fit()
    posterior_only = xr.DataTree.from_dict({"posterior": results["posterior"].to_dataset()})

    with pytest.raises(ValueError, match="results has no log_likelihood group"):
        psense_summary(posterior_only, priors=priors)
