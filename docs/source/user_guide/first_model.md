---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A first model

This page builds a small marketing mix model of weekly revenue with two media
channels and a price control, fits it, and asks how much of the revenue each
channel produced. Most later pages in the guide start from this model and
examine or change one part of it, so it is worth running once from top to
bottom.

## The data

The data comes from the simple setting of {func}`~mmmjax.simulate_data`,
which [The example data](example_data) describes, prepared and scaled as in
[Data and scaling](data.md). {func}`~mmmjax.prepare_data` gives each column
the model uses a role, and {func}`~mmmjax.fit_data_scaling` fits the scaling
the model works on.

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
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)
```

The scaling divides each channel's exposure by its median nonzero week and
standardizes revenue and price, which puts everything the model sees on a
scale of about one.

## The model

The model explains standardized revenue as an intercept, plus what each
channel adds once its exposure is carried forward and saturated, plus a price
effect, with normal noise around the sum. Week $t$ runs over the 156 modeled
weeks and $c$ over the two channels. The specification below covers the data
transformations, the model equation, the media transformation, and the priors,
in that order.

### Data transformations

The model works on scaled data. Each channel's impressions $z_{tc}$ are
divided by $m_c$, the channel's median over the weeks with any exposure.
Revenue $r_t$ and price $q_t$ are standardized with their mean and standard
deviation over the modeled weeks,

$$
x_{tc} = \frac{z_{tc}}{m_c}, \qquad
y_t = \frac{r_t - \bar{r}}{s_r}, \qquad
p_t = \frac{q_t - \bar{q}}{s_q}.
$$

### Model equation

$$
y_t = \alpha + \sum_{c} \beta_c \operatorname{HillAdstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big)
+ \gamma\, p_t + \varepsilon_t,
\qquad
\varepsilon_t \sim \operatorname{Normal}(0, \sigma)
$$

Everything except the noise $\varepsilon_t$ is the mean $\mu_t$, which the code
returns as `mu`, so the likelihood is $y_t \sim \operatorname{Normal}(\mu_t,
\sigma)$. The coefficient $\beta_c$ is the most channel $c$ can add in a week,
$\rho_c$ is its retention rate, $\kappa_c$ is its half-saturation point,
$\alpha$ is the intercept, $\gamma$ is the price coefficient, and $\sigma$ is
the noise scale.

### Media transformation

Adstock averages the current week and the eight before it with geometric
weights, and the Hill curve reaches half its maximum at $\kappa$,

$$
\operatorname{Adstock}\big(\{x_{t-\ell}\}_{\ell=0}^{L};\, \rho\big)
= \frac{\sum_{\ell=0}^{L} \rho^{\ell}\, x_{t-\ell}}{\sum_{\ell=0}^{L} \rho^{\ell}},
\qquad
\operatorname{Hill}(u;\, \kappa, s) = \frac{u^{s}}{u^{s} + \kappa^{s}}.
$$

Weeks before the first one in the data count as zero exposure. HillAdstock
applies Adstock first and then Hill, with the slope $s$ fixed at one in this
model.

### Priors

Each parameter has its own fixed prior, so the model has no hyperpriors. A
regional model would add them by drawing each region's coefficients from a
shared distribution with its own priors.

$$
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 1) \\
\beta_c &\sim \operatorname{HalfNormal}(2) \\
\rho_c &\sim \operatorname{Beta}(2, 2) \\
\kappa_c &\sim \operatorname{LogNormal}(0, 0.5) \\
\gamma &\sim \operatorname{Normal}(0, 1) \\
\sigma &\sim \operatorname{HalfNormal}(1)
\end{aligned}
$$

Each distribution takes the same parameters as its mmmJAX function, so
$\operatorname{HalfNormal}(2)$ has scale two, $\operatorname{Beta}(2, 2)$ has
both shape parameters at two, and
$\operatorname{LogNormal}(0, 0.5)$ is the distribution of a variable whose
logarithm has mean zero and standard deviation one half. The sections below
write each part of this specification as code.

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
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated @ coefficient + controls @ control_coefficient
    return {"mu": mu}
```

mmmJAX fills each argument by name. `media` and `controls` come from the
data, and the rest are the current parameter values. Each channel's exposure
carries forward for up to eight weeks at its retention rate and then passes
through a saturation curve that reaches half its maximum at
`half_saturation`, so every extra impression is worth a little less than the
one before. Together these lines compute HillAdstock and the mean $\mu_t$ of
the model equation, and returning `mu` by name is how the density and the
analysis functions find it.

:::{admonition} Media history
:class: important

This model treats the weeks before the data as having no exposure. When you
know what aired before your first week, pass it to
{func}`~mmmjax.prepare_data` as `media_history`, and carryover in the first
weeks starts from it instead of from zero. The media array then has more rows
than the outcome, eight more for eight weeks of history, so the block asks for
`n_periods` by name and keeps only the last `n_periods` rows once carryover
has used the rest.

```python
mu = intercept + saturated[-n_periods:] @ coefficient + controls @ control_coefficient
```

{func}`~mmmjax.media_response` runs the carryover, the saturation, and that
slice in one call.
:::

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
constraints. The first six are the priors of the model above, and the last is
its likelihood, $y_t \sim \operatorname{Normal}(\mu_t, \sigma)$. The priors
describe the scaled data too, so a normal prior with a scale of one on the
intercept already spans standardized revenue. [Priors](priors) covers what
these choices mean and how to check them.

## Generated quantities

`generated_quantities` runs once for each saved draw. Its first argument is
the one exception to requests by name. mmmJAX passes a JAX random key there
for each draw, and the `_rng` functions draw with it.

```{code-cell} ipython3
def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
    }
```

For each draw of the parameters, `prediction` redraws every week's
standardized revenue $\tilde{y}_t \sim \operatorname{Normal}(\mu_t, \sigma)$ from the
posterior predictive distribution, and `pointwise` is the log likelihood
$\log p(y_t \mid \mu_t, \sigma)$ of each observed week. Posterior predictive
checks compare the simulated revenue under `predictive` with the data, and
ArviZ compares models using each week's log likelihood under
`log_likelihood`. Both entries are named `outcome`, like the observed revenue
they describe, because ArviZ pairs groups by variable name. The pointwise and `_rng`
versions sit next to the summed one in `log_density`, as
[Distributions](distributions) describes.

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
then keeps 1,000 draws. {func}`~mmmjax.sample` returns them in an xarray
DataTree, whose `posterior` group labels every parameter by chain and draw and
by any axis its declaration named.

```{code-cell} ipython3
results["posterior"]
```

The channel parameters carry the channel names from the data, and
`control_coefficient` carries the name of the price column. [Fitting and
checking](checking) covers the other groups and what to look at before
trusting a fit.

## Asking a question

With the draws in hand, {func}`~mmmjax.contributions` asks how much of the
revenue each channel produced.

```{code-cell} ipython3
shares = mj.contributions(model, results, quantity="mu")
shares["contribution_share"].mean(("chain", "draw")).to_series().round(3)
```

It removes one channel at a time, runs the blocks again with the same draws,
and compares the result with the fitted revenue. On average, TV produced about
4.2 percent of revenue over the three years and search about 5.6 percent,
close to the true 4.3 and 6.1 percent from [The example data](example_data).
Each draw gives its own answer, so the estimate comes with its uncertainty.
[Media effects](media_effects) reads it in more detail and goes on to returns
and response curves, [Recovering the truth](recovery) checks those answers
against the simulation, and [Budget optimization](budgets) turns them into a
spending plan.
