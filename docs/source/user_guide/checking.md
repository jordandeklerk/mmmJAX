---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Fitting and checking

Fitting a model takes one call to {func}`~mmmjax.sample`, and deciding whether
to trust the result takes a few more. This page fits the model from [A first
model](first_model), reads its diagnostics, and compares its predictions with
the data.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
```

## Divergences

{func}`~mmmjax.sample` runs the No-U-Turn sampler with window adaptation. The
first thing to check after a fit is whether any transition diverged.

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import first_model_results, stored

results = first_model_results(model)
```

```{code-cell} ipython3
int(results["sample_stats"]["diverging"].sum())
```

None did. A divergent transition is a step where the sampler's simulated
trajectory broke down, usually in a part of the posterior that curves too
sharply for the step size. Draws near those regions can't be trusted, so
{func}`~mmmjax.sample` counts them in `sample_stats` and warns when any
appear. The first remedy is a higher `target_accept`, which shrinks the step
size. When divergences persist even at a high target, the model usually needs
a change, often a reparameterization that gives the sampler an easier shape
to explore.

## What the results hold

```{code-cell} ipython3
sorted(results.children)
```

{func}`~mmmjax.sample` returns an xarray
[DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
whose groups follow ArviZ's conventions. `posterior` holds the draws with
their labels, `sample_stats` holds the sampler's diagnostics, and the
predictive and log likelihood groups hold what `generated_quantities`
returned under those names. A `log_prior` key fills a group of its own, and
any other output lands in `generated_quantities`. The data the model saw is kept in `observed_data`
and `constant_data`, and `sampling_state` records where each chain stopped.

## Convergence

[ArviZ](https://python.arviz.org/) reads the results directly. It is not
installed with mmmJAX, so add it with `pip install arviz`.

```{code-cell} ipython3
import arviz as az

summary = az.summary(
    results, var_names=["coefficient", "retention", "half_saturation", "sigma"]
)
summary[["mean", "sd", "ess_bulk", "r_hat"]]
```

`r_hat` compares the spread between chains with the spread within them, and
values of 1.00 mean the four chains agree. `ess_bulk` estimates how many
independent draws the 4,000 correlated ones are worth, and a few hundred is
usually enough for posterior means and intervals, so these leave a wide
margin. TV's retention is pinned down to 0.70 give or take 0.02, right where
the simulation set it, because each flight's rise and fade shows how long the
effect lasts. Search runs every week, so its retention and half-saturation
point are much less certain. The draws also show whether the model can tell
the two channels apart.

```{code-cell} ipython3
import xarray as xr

coefficient = results["posterior"]["coefficient"]
tv, search = coefficient.sel(channel="TV"), coefficient.sel(channel="Search")
round(float(xr.corr(tv, search)), 2)
```

The two coefficients are almost uncorrelated across draws, so the model
credits each channel without trading one against the other. That follows from
the low overlap between the channels that {func}`~mmmjax.check_data` found in
the data.

## Predictions against the data

A posterior predictive check asks whether data simulated from the fitted
model looks like the data it was fitted to. A simple version counts how often
the observed revenue falls inside the model's 90 percent predictive interval.

```{code-cell} ipython3
prediction = results["posterior_predictive"]["prediction"]
observed = results["observed_data"]["outcome"]
lower, upper = prediction.quantile([0.05, 0.95], dim=("chain", "draw"))
round(float(((observed >= lower) & (observed <= upper)).mean()), 2)
```

About 91 percent of the weeks fall inside, close to the 90 percent a
well-calibrated model would give. Far fewer would mean the model misses
patterns in revenue that its noise term cannot absorb. The pointwise log
likelihood supports a sharper comparison through leave-one-out
cross-validation with `az.loo`, which [Changing the model](changing) uses to
compare versions of this model.

## More draws

{func}`~mmmjax.continue_sampling` picks up each chain where it stopped, with
the step size and mass matrix from warmup, so it adds draws without adapting
again.

```{code-cell} ipython3
:tags: [skip-execution]

more = mj.continue_sampling(model, results, draws=1000)
```

```{code-cell} ipython3
:tags: [remove-cell]

more = stored(
    "first_model_continued",
    lambda: mj.continue_sampling(model, results, draws=1000),
    groups=["posterior", "sample_stats"],
)
```

```{code-cell} ipython3
more["posterior"].sizes["draw"]
```

Each chain now holds 2,000 draws, the second thousand continuing the first
without a break.

## Sampler settings

Besides `target_accept`, {func}`~mmmjax.sample` sets the number of chains and
how they run, as [Installation](../getting_started/installation) describes,
along with the warmup length and the maximum tree depth. A `"dense"` mass
matrix adapts to correlations between parameters, which helps when they are
strongly correlated, at the cost of more work per step. {func}`~mmmjax.sample`
is not the only way to fit the model, and [Inference](inference) runs other
samplers on it.
