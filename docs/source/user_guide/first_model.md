---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A first model

This page builds a small marketing mix model of weekly revenue with two media
channels and a price control, fits it, and asks how much of the revenue each
channel produced. Every later page in the guide starts from this model and
examines or changes one part of it, so it is worth running once from top to
bottom.

## The data

The data comes from the simple setting of {func}`~mmmjax.simulate_data`,
which [The example data](example_data) describes. {func}`~mmmjax.prepare_data`
selects the columns the model uses and gives each one a role, and
{func}`~mmmjax.fit_data_scaling` fits the scaling the model works on.

```{code-cell} ipython3
import mmmjax as mj

example = mj.simulate_data(seed=7, groups=None, complexity="simple", noise_scale=0.02)
data = mj.prepare_data(
    example.frame,
    time="week",
    outcome="revenue",
    media=["linear_tv_impressions", "generic_search_impressions"],
    spend=["linear_tv_spend", "generic_search_spend"],
    channels=["TV", "Search"],
    controls=["price"],
    media_history=example.media_history,
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)
```

The media history adds the eight weeks of exposure before the first modeled
week, so carryover in January starts from what aired in December instead of
from nothing. The scaling divides each channel's exposure by its median
nonzero week and standardizes revenue and price, which puts everything the
model sees on a scale of about one. [Data and scaling](data.md) looks at both
steps in detail.

## Parameters

Each parameter is declared with its support and its axes.

```{code-cell} ipython3
parameters = {
    "intercept": mj.Real(),
    "coefficient": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "control_coefficient": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}
```

{class}`~mmmjax.Positive` and {class}`~mmmjax.Interval` keep a parameter
inside its support, and `dims="channel"` gives it one value per channel,
labeled TV and Search from the data. {class}`~mmmjax.LowerBound`,
{class}`~mmmjax.UpperBound`, {class}`~mmmjax.Simplex` for shares that sum to
one, and {class}`~mmmjax.CorrelationCholesky` for correlation matrices cover
other supports. None of the declarations carries a prior. The priors come
later, in the density.

## Expected revenue

`transformed_parameters` turns exposure into expected revenue.

```{code-cell} ipython3
def transformed_parameters(
    media,
    controls,
    n_periods,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    recent = saturated[-n_periods:]
    mu = intercept + recent @ coefficient + controls @ control_coefficient
    return {"mu": mu}
```

mmmJAX fills each argument by name. `media`, `controls`, and `n_periods` come
from the data, and the rest are the current parameter values. Each channel's
exposure carries forward for up to eight weeks at its retention rate and then
passes through a saturation curve that reaches half its maximum at
`half_saturation`, so every extra impression is worth a little less than the
one before. The slice drops the history weeks once carryover has used them,
which lines `mu` up with the 156 weeks of revenue. Returning `mu` by name is
how the density and the analysis functions find it.

In symbols, with $x_{tc}$ the scaled impressions of channel $c$ in week $t$
and $p_t$ the scaled price, the block computes

$$
\bar{x}_{tc} = \frac{\sum_{l=0}^{8} \rho_c^{\,l}\, x_{t-l,c}}{\sum_{l=0}^{8} \rho_c^{\,l}},
\qquad
\mu_t = \alpha + \sum_{c} \beta_c \, \frac{\bar{x}_{tc}}{\bar{x}_{tc} + \kappa_c} + \gamma\, p_t,
$$

where $\rho_c$ is the retention rate, $\kappa_c$ the half-saturation point,
$\beta_c$ the coefficient, $\alpha$ the intercept, and $\gamma$ the price
coefficient.

## The density

`log_density` adds up every prior and the likelihood, one term at a time.

```{code-cell} ipython3
def log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.half_normal(coefficient, 2.0)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target
```

These lines hold every assumption the model makes, and mmmJAX adds nothing to
them except the Jacobian adjustments its declarations make for their
constraints. The last line is the likelihood, $y_t \sim \mathcal{N}(\mu_t,
\sigma^2)$ for standardized revenue $y_t$. The priors describe the scaled data
too, so a normal prior with a scale of one on the intercept already spans
standardized revenue. [Priors](priors) covers what these choices mean and how
to check them.

## Generated quantities

`generated_quantities` runs once for each saved draw.

```{code-cell} ipython3
def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    return {
        "predictive": {"prediction": prediction},
        "log_likelihood": {"pointwise": pointwise},
    }
```

Posterior predictive checks compare the simulated revenue under `predictive`
with the data, and ArviZ compares models using each week's log likelihood
under `log_likelihood`.

## Fitting

{class}`~mmmjax.Model` binds the blocks to the data and checks every argument
name before anything runs. {func}`~mmmjax.sample` then fits the model with
the No-U-Turn sampler.

```{code-cell} ipython3
model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import first_model_results

results = first_model_results(model)
```

Each of the four chains runs 1,000 warmup steps, which tune the sampler, and
then keeps 1,000 draws. [Fitting and checking](checking) shows what to look
at before trusting a fit.

## Asking a question

With the draws in hand, {func}`~mmmjax.contributions` asks how much of the
revenue each channel produced.

```{code-cell} ipython3
shares = mj.contributions(model, results, quantity="mu")
shares["contribution_share"].mean(("chain", "draw")).to_series().round(3)
```

It removes one channel at a time, runs the blocks again with the same draws,
and compares the result with the fitted revenue. On average, TV produced about
4.1 percent of revenue over the three years and search about 5.7 percent,
close to the true 4.3 and 6.1 percent from [The example data](example_data).
Each draw gives its own answer, so the estimate comes with its uncertainty.
[Media effects and budgets](media_effects) reads it in more detail and goes on
to returns, response curves, and budget allocation.
