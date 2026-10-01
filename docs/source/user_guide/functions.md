---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# User-defined functions

Stan programs have a `functions` block for code a model reuses, but mmmJAX
doesn't need one, because the blocks are ordinary Python functions and can call
any function you write. The model from [A first model](first_model) already
calls one, a `hill_adstock` helper. Here you'll write your own response curve
and distribution, swap both into that model, and learn the rules JAX sets for
code that runs inside a block.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
%xmode minimal
```

## A response curve of your own

The curve you'll write here, exponential saturation, levels off faster than the
Hill curve. With a rate $\nu > 0$ it takes exposure $u$ to

$$
1 - e^{-\nu u}.
$$

Write it with `jax.numpy` and it's an ordinary function that JAX can
differentiate and compile.

```{code-cell} ipython3
import jax
import jax.numpy as jnp


def exponential_saturation(exposure, rate):
    saturated = 1.0 - jnp.exp(-rate * exposure)
    return saturated


print(exponential_saturation(jnp.array([0.0, 1.0, 2.0]), rate=1.0))
print(jax.grad(exponential_saturation)(1.0, 1.0))
```

## A distribution of your own

{func}`~mmmjax.custom_distribution` turns a pointwise log density and a draw
function into a family that works like the built-in ones described in
[Distributions](distributions). A half-Cauchy prior, a common choice for scale
parameters, has the density

$$
p(x \mid s) = \frac{2}{\pi s \big(1 + (x / s)^2\big)}, \qquad x \geq 0,
$$

where $s$ is the scale. To draw from it, take $s$ times the absolute value of a
standard Cauchy draw. You can write the log of that density and the draw as two
short functions.

```{code-cell} ipython3
def half_cauchy_logpdf(value, scale):
    standardized = value / scale
    density = jnp.log(2.0 / jnp.pi) - jnp.log(scale) - jnp.log1p(standardized**2)
    return jnp.where(value < 0, -jnp.inf, density)


def half_cauchy_rng(key, scale, *, sample_shape=()):
    shape = sample_shape + jnp.shape(jnp.asarray(scale))
    return jnp.abs(scale * jax.random.cauchy(key, shape))


half_cauchy = mj.custom_distribution(half_cauchy_logpdf, half_cauchy_rng, name="half_cauchy")
print(half_cauchy(jnp.array([0.5, 2.0]), 1.0))
print(mj.Prior(half_cauchy, scale=1.0).sample(jax.random.key(0), sample_shape=(3,)))
```

The first line of output is one number for both values, because the new family
sums the log density the way `mj.normal` does. A {class}`~mmmjax.Prior` built
from it draws and scores like any other.

:::{admonition} Check the draws against the density
:class: warning

Nothing checks that your draw function and your log density describe the same
distribution. If they disagree, you get a prior that draws from one
distribution and scores another, and the model still runs.
:::

To check them, lay a histogram of many draws over the density and see whether
the two agree.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import plotnine as pn

draws = np.asarray(half_cauchy_rng(jax.random.key(0), 1.0, sample_shape=(10_000,)))
edges = np.linspace(0.0, 10.0, 51)
counts, _ = np.histogram(draws, bins=edges)
# Dividing by every draw, not just those below 10, keeps the bars on the density's scale.
heights = counts / (draws.size * np.diff(edges))
bars = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "density": heights})
grid = jnp.linspace(0.0, 10.0, 200)
curve = jnp.exp(half_cauchy_logpdf(grid, 1.0))
density = pd.DataFrame({"value": np.asarray(grid), "density": np.asarray(curve)})
(
    pn.ggplot(bars)
    + pn.geom_rect(
        pn.aes(xmin="left", xmax="right", ymin=0, ymax="density"),
        fill="#d2d3fb",
        color="#2a2eec",
        size=0.3,
    )
    + pn.geom_line(pn.aes("value", "density"), data=density, color="#262626", size=0.9)
    + pn.labs(x="Value", y="Density")
    + mj.theme_mmmjax()
)
```

The bars follow the curve, so the draw function samples the distribution the
density describes. The draws past the right edge make up the heavy tail that
sets the half-Cauchy apart.

```{code-cell} ipython3
draws = half_cauchy_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
half_normal_draws = mj.half_normal_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
print(round(float(jnp.mean(draws > 10.0)), 3))
print(round(float(jnp.mean(half_normal_draws > 10.0)), 3))
```

Of the half-Cauchy draws, 6.7 percent are above 10, while none of 10,000
half-normal draws with the same scale are. As a prior, the half-Cauchy leaves
far more room for large values. On the noise scale, that room barely matters
once the model sees the data, because 156 weeks of residuals pin $\sigma$ down
far more tightly than either prior does, as its narrow interval on
[Sampling and diagnostics](sampling.md#convergence) shows.

## Both in a model

With both pieces written, you can swap them into the model from
[A first model](first_model). Each paid channel's exposure still carries
forward with the same adstock, but the exponential curve now saturates it. A
rate $\nu_c$ takes the place of each channel's half-saturation point $\kappa_c$,
and $\nu_o$ takes the place of email's $\kappa_o$. The pieces of the model that
change are

$$
\begin{aligned}
h_{tc} &= 1 - \exp\Big(-\nu_c \operatorname{Adstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c\big)\Big), \\
h_{to} &= 1 - \exp\Big(-\nu_o \operatorname{Adstock}\big(\{x_{t-\ell,o}\}_{\ell=0}^{8};\, \rho_o\big)\Big), \\
\nu_c, \nu_o &\sim \operatorname{LogNormal}(0, 0.5), \qquad \sigma \sim \operatorname{HalfCauchy}(1),
\end{aligned}
$$

and everything else stays as in [A first model](first_model). The formulas
that turn each channel's ROI $r_c$ into $\beta_c$ and email's share $\phi_o$
into $\lambda_o$ don't change either, though they now sum the new curves over
the training weeks.

The rate takes the half-saturation point's $\operatorname{LogNormal}(0, 0.5)$
prior, and because that prior is symmetric about zero on the log scale,
$1/\nu$, where the curve reaches about 63 percent of its ceiling, gets exactly
the prior $\kappa$ had.

The new symbols and functions take the names below in the code, and the rest
keep their names from the first model.

| Symbol | Name in the code | Where the name comes from |
| --- | --- | --- |
| $\nu_c$ | `rate` | Declared in `parameters` as positive, with the `channel` axis |
| $\nu_o$ | `organic_rate` | Declared in `parameters` as positive, with the `organic_channel` axis |
| $1 - e^{-\nu u}$ | `exponential_saturation` | The function from the first section |
| $\operatorname{HalfCauchy}(1)$ | `half_cauchy(sigma, 1.0)` | A term of `log_density`, with the family from the previous section |

A new plain helper, `exponential_adstock`, applies `exponential_saturation`
after the adstock in place of `hill_adstock`.

```{code-cell} ipython3
# A plain function rather than a block, so mmmJAX never fills its arguments.
def exponential_adstock(media, retention, rate):
    # The same carryover as hill_adstock.
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)

    # The exponential curve takes the Hill curve's place. A larger rate makes it level off sooner.
    saturated = exponential_saturation(carried, rate)
    return saturated


exponential_parameters = {
    # The baseline's level, the trend's growth and curvature, and the season.
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    "annual_coefficients": mj.Real(4),
    # Each paid channel's return, carryover, and saturation rate.
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "rate": mj.Positive(dims="channel"),
    # Email's share of revenue, carryover, and saturation rate.
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_rate": mj.Positive(dims="organic_channel"),
    # One coefficient for each control and each treatment, and the noise scale.
    "control_coefficient": mj.Real(dims="control"),
    "treatment_coefficient": mj.Real(dims="treatment"),
    "sigma": mj.Positive(),
}
```

All four calls in `transformed_parameters` switch to the new helper, the two on
`reference` and the two on `media` and `organic_media`.

:::{admonition} Swap all four calls
:class: important

If the call on `reference.media` kept the Hill curve while the call on `media`
used the exponential one, $\beta_c$ would no longer make the channel's
contribution over the training weeks $r_c S_c$ dollars, and $r_c$ would stop
being its return.
:::

```{code-cell} ipython3
def exponential_transformed_parameters(
    media,
    organic_media,
    controls,
    treatments,
    reference,
    outcome_scaling,
    annual,
    trend,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    rate,
    organic_share,
    organic_retention,
    organic_rate,
    control_coefficient,
    treatment_coefficient,
):
    # Paid media's coefficients come from the returns through the exponential curve over the
    # training weeks. The media effect below needs the same curve, or roi would stop being each
    # channel's return.
    trained = exponential_adstock(reference.media, retention, rate)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)

    # Email's coefficient comes from its share of revenue through its own exponential curve.
    organic_trained = exponential_adstock(reference.organic_media, organic_retention, organic_rate)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )

    # The baseline follows the trend and the season.
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients

    # Each effect reads the inputs the model is given, so a scenario that changes media,
    # sends, or prices changes it.
    media_effect = exponential_adstock(media, retention, rate) @ coefficient
    organic_saturated = exponential_adstock(organic_media, organic_retention, organic_rate)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient

    # The likelihood and every analysis read the expected revenue in each week.
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}
```

In the density, the lines for `rate`, `organic_rate`, and `sigma` change to
match the priors above. The model reuses the first model's `transformed_data`
and `generated_quantities` as they are, because neither block calls the
saturation curve or states a prior.

```{code-cell} ipython3
def exponential_log_density(
    outcome,
    mu,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    rate,
    organic_share,
    organic_retention,
    organic_rate,
    control_coefficient,
    treatment_coefficient,
    sigma,
):
    # The baseline's priors describe standardized revenue.
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)

    # Paid media's returns, carryover, and saturation rates.
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(rate, 0.0, 0.5)

    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_rate, 0.0, 0.5)

    # The treatments get a tighter prior than the controls.
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)

    # The noise scale's prior, now the half-Cauchy from the previous section, and the likelihood,
    # normal noise around the expected revenue.
    target += half_cauchy(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


exponential_model = mj.Model(
    parameters=exponential_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=exponential_transformed_parameters,
    log_density=exponential_log_density,
    generated_quantities=generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

exponential = mj.sample(exponential_model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import stored

exponential = stored(
    "exponential",
    lambda: mj.sample(exponential_model, draws=1000, warmup=1000, chains=4, seed=7),
    groups=["posterior"],
)
```

## The fitted curves

{func}`~mmmjax.plot_saturation` draws the new curve the same way it draws the
Hill curve. It passes each posterior draw of `rate` to the curve's argument of
the same name.

To put prior draws beside the posterior, {func}`~mmmjax.sample_prior` needs the
new density's priors as {class}`~mmmjax.Prior` objects. `exponential_priors`
states them again, and `sample_prior` draws `sigma` with the half-Cauchy
family's own draw function.

```{code-cell} ipython3
exponential_priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "growth": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "curvature": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "annual_coefficients": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "roi": mj.Prior(mj.lognormal, location=1.0, scale=0.6),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "rate": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "organic_share": mj.Prior(mj.beta, alpha=2.0, beta=98.0),
    "organic_retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "organic_rate": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "treatment_coefficient": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "sigma": mj.Prior(half_cauchy, scale=1.0),
}
```

```{code-cell} ipython3
:tags: [skip-execution]

exponential_prior = mj.sample_prior(exponential_model, exponential_priors)
```

```{code-cell} ipython3
:tags: [remove-cell]

exponential_prior = stored(
    "exponential_prior",
    lambda: mj.sample_prior(exponential_model, exponential_priors),
    groups=["prior"],
)
```

```{code-cell} ipython3
mj.plot_saturation(exponential, exponential_saturation, max_input=3.0, prior=exponential_prior)
```

On the horizontal axis, a media input of one is a typical on-air week, because
each channel's impressions are divided by their median over the weeks it ran.
Most of the blue posterior curves lie on their orange priors. The data adds
little about how fast those channels saturate, so the rate's prior carries
most of that answer. Linear TV's curve climbs furthest above its prior, and
Streaming's and YouTube's come next, so for those three the data suggests
faster saturation than the prior expects.

:::{admonition} Email's curve
:class: tip

The panels cover the ten paid channels because `rate` has the `channel` axis.
To see email's curve drawn the same way, pass
`parameters={"rate": "organic_rate"}` to `plot_saturation`.
:::

## What JAX needs from a function

mmmJAX compiles every block with JAX, and to compile a block, JAX first traces
it once with placeholder values and only then runs the compiled version. That's
why the code inside a block, and every function it calls, has to work with
those placeholders. A Python `if` on a parameter fails, for example, because
the value doesn't exist while the function is traced.

```{code-cell} ipython3
:tags: [raises-exception]

def branching_response(exposure, threshold):
    if threshold > 1.0:
        return exposure - threshold
    return exposure


jax.jit(branching_response)(jnp.array([0.5, 1.5]), 2.0)
```

`jnp.where` makes the same kind of choice element by element, and it traces
without trouble.

```{code-cell} ipython3
def threshold_response(exposure, threshold):
    return jnp.where(exposure > threshold, exposure - threshold, 0.0)


jax.jit(threshold_response)(jnp.array([0.5, 1.5, 2.5]), 1.0)
```

NumPy functions fail on placeholders in the same way, so write your helpers with
`jax.numpy`. Anything that sets an array's shape, such as `max_lag=8` in
`exponential_adstock` or `order=2` in the model's `transformed_data`, has to be
a Python integer. Either write it in the code or pass it through the
`constants` of {class}`~mmmjax.Data`.

Tracing also explains why `print` is the wrong tool for looking inside a block.

```{code-cell} ipython3
@jax.jit
def doubled(values):
    print("print sees", values)
    jax.debug.print("jax.debug.print sees {}", values)
    return 2 * values


first = doubled(jnp.array([1.0, 2.0]))
second = doubled(jnp.array([3.0, 4.0]))
```

In the output, `print` runs once, while JAX traces the function, so all it sees
is the placeholder, `JitTracer(float32[2])`. `jax.debug.print` instead runs on
every call and prints the real values each time. Tracing is also why the first
evaluation of a model takes longer than every call after it, since that first
call is when the blocks are compiled.
