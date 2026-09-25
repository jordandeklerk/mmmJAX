---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sampling and diagnostics

Sampling a model takes one call to {func}`~mmmjax.sample`, and deciding whether
to trust the draws takes a few more. This page samples the model from [A first
model](first_model), reads its diagnostics, and compares its predictions with
the data.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
```

```{code-cell} ipython3
:tags: [remove-cell]

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = ".33"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
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
results
```

{func}`~mmmjax.sample` returns an xarray
[DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
whose groups follow ArviZ's conventions. Each folder opens to show a group's
variables and labels, and the attributes at the root record how the sampler
was set up. `posterior` holds the draws, `sample_stats` the sampler's
diagnostics, and `observed_data` and `constant_data` the data the model saw.
The predictive and log likelihood groups hold what `generated_quantities`
returned, and both entries are named `outcome` like the observed revenue,
because ArviZ pairs groups by variable name. A `log_prior` key fills a group
of its own, any other output lands in `generated_quantities`, and
`sampling_state` records where each chain stopped.

## Convergence

[ArviZ](https://python.arviz.org/) reads the results directly, and each of
its functions on this page takes `results` as it is and finds the groups it
needs by name. mmmJAX's diagnostic plots draw ArviZ's with settings that suit
these models.

```{code-cell} ipython3
import arviz as az

print(az.summary(results))
```

Each row gives a parameter's mean, standard deviation, and 89 percent
interval before the diagnostics. `r_hat` compares the spread between chains
with the spread within them, and 1.00 means the four chains agree. `ess_bulk`
and `ess_tail` estimate how many independent draws the 4,000 correlated ones
are worth in the middle and the tails, and a few hundred is usually enough.
The `mcse` columns give the Monte Carlo error in the mean and the standard
deviation. {func}`~mmmjax.plot_rhat` draws every `r_hat` at once.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Each parameter gets a box of its elements' values with a point for each, and
the subtitle counts the values past ArviZ's limit of 1.01, none here. A model
with hundreds of channels still fits one such plot. A trace plot shows the
chains draw by draw.

```{code-cell} ipython3
pc = mj.plot_trace_dist(
    results,
    var_names=["coefficient", "retention", "half_saturation"],
    aes={"color": ["channel"]},
)
pc.add_legend("channel")
plt.show()
```

Each row shows a parameter's density on the left and its draws on the right.
`aes` colors each channel, and each chain gets its own line style. The chains'
densities overlap, and every trace is a flat band with no drift. TV's
retention is pinned down to 0.69 give or take 0.02, right where the simulation
set it, because each flight's rise and fade shows how long the effect lasts.
Search runs every week, so its retention and half-saturation point are much
less certain. A rank plot checks the chains more strictly.

```{code-cell} ipython3
mj.plot_rank(
    results,
    var_names=["coefficient", "half_saturation"],
    col_wrap=2,
    figure_kwargs={"figsize": (12, 9)},
)
plt.show()
```

Each line follows one chain's fractional ranks, the share of all draws below
each of its draws, as the gap between their cumulative distribution and a
uniform one. Chains exploring the same distribution stay near zero, and
{func}`~mmmjax.plot_rank` spaces the draws out first so that autocorrelation
alone can't fail the test. TV's coefficient and half-saturation both fail it at the 1
percent level, and the black dots mark the stretches behind it. Chain 1 holds
too many low draws of both, even though `r_hat` reads 1.00. The draws also
show whether the model can tell the two channels apart.

```{code-cell} ipython3
import xarray as xr

coefficient = results["posterior"]["coefficient"]
tv = coefficient.sel(channel="TV")
search = coefficient.sel(channel="Search")
round(float(xr.corr(tv, search)), 2)
```

The two coefficients are almost uncorrelated across draws, so the model
credits each channel without trading one against the other. That follows from
the low overlap between the channels that {func}`~mmmjax.check_data` found on
[Data and scaling](data.md). A pair plot shows every such relationship at
once.

```{code-cell} ipython3
az.plot_pair(
    results,
    var_names=["intercept", "coefficient", "half_saturation"],
    visuals={"scatter": {"alpha": 0.2, "size": 4}},
    figure_kwargs={"figsize": (12, 10)},
)
plt.show()
```

Panels pairing TV with Search show no trend, but two shapes stand out. TV's
coefficient and half-saturation lie along a narrow ridge, and chains cross a
ridge slowly, which is what the rank plot caught. TV's carried exposure peaks
near 1, below its half-saturation of about 1.2, so the data shows mostly the
lower part of the curve, where the response depends on the ratio of the two
more than on either one. Search's coefficient falls as the intercept rises,
and its half-saturation climbs along a curve, since part of an always-on
channel's effect looks like a higher baseline.

## Predictions against the data

A posterior predictive check asks whether data simulated from the fitted
model looks like the data it was fitted to. A simple version counts how often
the observed revenue falls inside the model's 90 percent predictive interval.

```{code-cell} ipython3
prediction = results["posterior_predictive"]["outcome"]
observed = results["observed_data"]["outcome"]
lower = prediction.quantile(0.05, dim=("chain", "draw"))
upper = prediction.quantile(0.95, dim=("chain", "draw"))
inside = (observed >= lower) & (observed <= upper)
round(float(inside.mean()), 2)
```

About 92 percent of the weeks fall inside, close to the 90 percent a
well-calibrated model would give. Far fewer would mean the model misses
patterns in revenue that its noise term cannot absorb.
{func}`~mmmjax.plot_fit` draws the same interval week by week.

```{code-cell} ipython3
mj.plot_fit(model, results, ci_prob=0.9)
```

The band shows each week's 90 percent predictive interval, the blue line its
mean, and the black line the observed revenue, all in dollars because the plot
undoes the outcome scaling first. The subtitle repeats the 92 percent coverage
next to an $R^2$ of 0.86 and a weighted mean absolute percentage error of 1.6
percent. The twelve weeks outside the band split evenly, six above and six
below. A run of weeks on one side would point to a pattern the
model misses, and the first seven weeks form one, each above its predicted
mean. The simulation's media began before the data, and this model misses the
carryover from those weeks, as the media history note on [A first
model](first_model) explains.

A predictive check can also target any feature the model ought to reproduce.
Revenue in one week resembles the week before, and because the model's noise
is independent from week to week, that persistence has to come from the media,
their carryover, and price. The lag-one autocorrelation measures it.

```{code-cell} ipython3
import numpy as np


def lag_one_autocorrelation(series):
    centered = series - series.mean()
    correlation = np.sum(centered[1:] * centered[:-1]) / np.sum(centered**2)
    return correlation


az.plot_ppc_tstat(results, t_stat=lag_one_autocorrelation, figure_kwargs={"figsize": (12, 7)})
plt.show()
```

ArviZ applies the function to every predictive draw and to the observed
revenue. The curve shows the draws' autocorrelations, which peak near 0.70,
and the black dot marks the observed 0.72, with about one draw in seven above
it, so the model reproduces how closely each week's revenue follows the last.
Any function from one series to a number works the same way, such as the
largest week or the share of weeks above a target.

The pointwise log likelihood supports a sharper check. Leave-one-out
cross-validation asks how well the model predicts each week when that week is
left out of the fit, and ArviZ estimates it from the draws already in hand
instead of refitting the model 156 times.

```{code-cell} ipython3
az.loo(results)
```

`elpd_loo` adds up the log predictive density of each week left out, and it
means something only next to another version's, which is how [Changing the
model](changing) uses it. `p_loo` estimates the effective number of
parameters. At about 7 it sits a little under the 9 the model declares, while
a value well above that count would point to a misspecified model. The Pareto
$k$ values check the shortcut behind the estimate, and all 156 weeks fall in
the good range. The same shortcut gives a calibration check.

```{code-cell} ipython3
az.plot_loo_pit(results, figure_kwargs={"figsize": (12, 7)})
plt.show()
```

LOO-PIT, the leave-one-out probability integral transform, is the probability
that a prediction made without a given week falls below that week's observed
revenue. In a calibrated model these values are uniform, and the line shows
how far their cumulative distribution strays from the uniform one. It stays
within about 0.05 of zero, and the test in the corner gives p = 0.46, so the
wiggles are no larger than uniform values would show by chance. ArviZ took the
log likelihood, the predictive draws, and the observed data for this check
from the same `results`.

Every check so far compares the model with the data. A model can pass them all
while crediting revenue to the wrong cause, and [Recovering the
truth](recovery) checks the causes against the simulation.

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

## New outputs from the same draws

A fit can also gain outputs without sampling again. The analysis functions
convert their results back to revenue on their own, but anything a block
returns stays in the units it was computed in, so a block that should report
dollars asks for `outcome_scaling` and converts.

```{code-cell} ipython3
def revenue_generated_quantities(key, outcome, mu, sigma, outcome_scaling):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    expected_revenue = outcome_scaling.inverse_transform(mu)
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
        "expected_revenue": expected_revenue,
    }


revenue_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=revenue_generated_quantities,
    generated_dims={"expected_revenue": "time"},
)
revenue = mj.generate_quantities(revenue_model, results)
expected = revenue["generated_quantities"]["expected_revenue"].sum("time").mean()
round(float(expected)), round(float(data.arrays["outcome"].sum()))
```

The generative model is still the one from [A first model](first_model), and
the new output is expected revenue in dollars, $\bar{r} + s_r\, \mu_t$.
{func}`~mmmjax.generate_quantities` runs the new block on the draws already in
`results`, which works because the parameters have not changed. The expected
revenue over the three years comes back in dollars, within about \$1,800 of
the 31.4 million observed. JAX arithmetic doesn't carry axis names, so
`generated_dims` labels the new output by week.

## Sampler settings

Besides `target_accept`, {func}`~mmmjax.sample` sets the number of chains and
how they run, as [Installation](../getting_started/installation) describes,
along with the warmup length, the maximum tree depth, and the mass matrix. The
default diagonal mass matrix scales each parameter on its own, while a
`"dense"` one also adapts to correlations between them, such as the ridge
between TV's coefficient and half-saturation, at the cost of more work per
step.

```{code-cell} ipython3
:tags: [skip-execution]

dense = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7, mass_matrix="dense")
```

```{code-cell} ipython3
:tags: [remove-cell]

dense = stored(
    "first_model_dense",
    lambda: mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7, mass_matrix="dense"),
    groups=["posterior", "sample_stats"],
)
```

```{code-cell} ipython3
mj.plot_rank(
    dense,
    var_names=["coefficient", "half_saturation"],
    col_wrap=2,
    figure_kwargs={"figsize": (12, 9)},
)
plt.show()
```

Every chain now stays near zero, and TV's coefficient and half-saturation pass
with p-values of 0.61 and 0.51.

```{code-cell} ipython3
print(az.summary(dense))
```

The estimates barely move, while the bulk effective sample size for TV's
coefficient rises from 2,458 to 3,411, because each step can now travel along
the ridge. When chains disagree more than this, the model usually needs a
change instead, such as a reparameterization. {func}`~mmmjax.sample` is not the
only way to fit the model, and [Inference](inference) runs other samplers on
it.
