---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# User-defined functions

Stan programs have a `functions` block for code a model reuses. mmmJAX doesn't
need one, because the blocks are ordinary Python functions and can call any
function you write. The model from [A first model](first_model) already calls
one, a `hill_adstock` helper that its `transformed_parameters` applies to the
paid channels and to email alike. This page writes its own response curve and
distribution, swaps both into that model, and covers the rules JAX sets for
code that runs inside a block.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
%xmode minimal

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
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["figure.dpi"] = 100
```

## A response curve of your own

An exponential saturation curve levels off faster than the Hill curve, which
keeps climbing slowly for a long time. With a rate $\nu > 0$ it takes exposure
$u$ to

$$
1 - e^{-\nu u},
$$

which starts at zero and approaches one. Written with `jax.numpy`, it's an
ordinary function that JAX can differentiate and compile.

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

with scale $s$, and it puts no mass below zero. A draw is $s$ times the absolute
value of a standard Cauchy draw. The log of that density and the draw take two
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

The new family sums the log density the way `mj.normal` does, and a
{class}`~mmmjax.Prior` built from it draws and scores like any other, so it
works with {func}`~mmmjax.sample_prior` as well.

:::{admonition} Check the draws against the density
:class: warning

Nothing checks that the two functions describe the same distribution. A draw
function that disagrees with its density gives a prior that draws from one
distribution and scores another, and the model still runs.
:::

A histogram of many draws laid over the density shows whether the two agree.

```{code-cell} ipython3
:tags: [hide-input]

draws = half_cauchy_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
edges = jnp.linspace(0.0, 10.0, 51)
counts, _ = jnp.histogram(draws, bins=edges)
width = edges[1] - edges[0]
grid = jnp.linspace(0.0, 10.0, 200)

fig, axis = plt.subplots(layout="constrained")
axis.stairs(counts / (draws.size * width), edges, fill=True, alpha=0.4, label="10,000 draws")
axis.plot(grid, jnp.exp(half_cauchy_logpdf(grid, 1.0)), color="black", label="Density")
axis.set_title("Draws against the density")
axis.legend(frameon=False)
plt.show()
```

The bars follow the curve, so the draw function samples the distribution the
density describes. Dividing each count by all 10,000 draws, not just the ones
below 10, keeps the bars on the density's scale even though some draws land
past the right edge. Those draws are the heavy tail that sets the half-Cauchy
apart.

```{code-cell} ipython3
draws = half_cauchy_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
half_normal_draws = mj.half_normal_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
print(round(float(jnp.mean(draws > 10.0)), 3))
print(round(float(jnp.mean(half_normal_draws > 10.0)), 3))
```

Of the half-Cauchy draws, 6.7 percent are above 10, and none of 10,000
half-normal draws with the same scale are.

## Both in a model

The model from [A first model](first_model) can use both. Each paid channel's
exposure is carried forward with the same adstock and then saturated by the
exponential curve, with its own rate $\nu_c$ in place of the half-saturation
point $\kappa_c$. Email's sends take the same path with a rate $\nu_o$ in place
of $\kappa_o$, and the half-Cauchy with scale one becomes the prior on the
noise scale $\sigma$. The pieces of the model that change are

$$
\begin{aligned}
h_{tc} &= 1 - \exp\Big(-\nu_c \operatorname{Adstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c\big)\Big), \\
h_{to} &= 1 - \exp\Big(-\nu_o \operatorname{Adstock}\big(\{x_{t-\ell,o}\}_{\ell=0}^{8};\, \rho_o\big)\Big), \\
\nu_c, \nu_o &\sim \operatorname{LogNormal}(0, 0.5), \qquad \sigma \sim \operatorname{HalfCauchy}(1),
\end{aligned}
$$

with everything else as in [A first model](first_model). The formulas that
turn each channel's ROI $r_c$ into $\beta_c$ and email's share $\phi_o$ into
$\lambda_o$ stay the same, and they now sum the new curves over the training
weeks.

The new symbols and functions take these names in the code.

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
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = exponential_saturation(carried, rate)
    return saturated


exponential_parameters = {
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    "annual_coefficients": mj.Real(4),
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "rate": mj.Positive(dims="channel"),
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_rate": mj.Positive(dims="organic_channel"),
    "control_coefficient": mj.Real(dims="control"),
    "treatment_coefficient": mj.Real(dims="treatment"),
    "sigma": mj.Positive(),
}
```

All four calls in `transformed_parameters` switch to the new helper. The call
on `reference.media` feeds {func}`~mmmjax.roi_coefficient`, which turns each
channel's ROI into its coefficient through the curve over the training weeks.
The call on `reference.organic_media` does the same for email's share through
{func}`~mmmjax.contribution_coefficient`.

:::{admonition} Swap all four calls
:class: warning

If the call on `reference.media` kept the Hill curve while the call on `media`
used the exponential one, $\beta_c$ would no longer make the channel's
contribution over the training weeks $r_c S_c$ dollars, and $r_c$ would stop
being its return.
:::

In the density, the lines for `rate`, `organic_rate`, and `sigma` change to
match the priors above. The model reuses the first model's
`transformed_data` and `generated_quantities`.

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
    trained = exponential_adstock(reference.media, retention, rate)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = exponential_adstock(reference.organic_media, organic_retention, organic_rate)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = exponential_adstock(media, retention, rate) @ coefficient
    organic_saturated = exponential_adstock(organic_media, organic_retention, organic_rate)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}


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
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(rate, 0.0, 0.5)
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_rate, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)
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

:::{admonition} Only your names change
:class: important

`rate` and `organic_rate` take the place of `half_saturation` and
`organic_half_saturation` among the declared parameters. They're your names,
so they only have to match across `exponential_parameters`, the blocks, and
the priors below. The supplied names stay exactly as mmmJAX spells them.
:::

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

{func}`~mmmjax.plot_saturation` draws the new curve the same way it draws the
Hill curve. It passes each posterior draw of `rate` to the argument of the same
name. To put prior draws beside the posterior, {func}`~mmmjax.sample_prior`
needs the new density's priors as {class}`~mmmjax.Prior` objects.
`exponential_priors` states them again, and `sample_prior` draws `sigma` with
the half-Cauchy family's own draw function.

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

A media input of one is a typical on-air week, because each channel's
impressions are divided by their median over the weeks it ran. For most
channels the blue posterior curve sits on the orange prior, so the data say
little about how fast they saturate and the rate's prior carries most of that
answer. Linear TV's curve climbs furthest above its prior, with Streaming's and
YouTube's next.

:::{admonition} Email's curve
:class: tip

The panels cover the ten paid channels because `rate` has the `channel` axis.
Passing `parameters={"rate": "organic_rate"}` to `plot_saturation` draws
email's curve the same way.
:::

Neither function is special to mmmJAX. They run inside the blocks because
they're written with JAX, the one requirement the next section spells out.

## What JAX needs from a function

mmmJAX compiles every block with JAX, which traces the function once with
placeholder values and then runs the compiled version. Code inside a block,
and every function it calls, has to work with those placeholders. A Python
`if` on a parameter fails, because the value doesn't exist while the function
is traced.

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

NumPy functions fail on placeholders in the same way, which is why helpers use
`jax.numpy`. Anything that sets an array's shape, such as `max_lag=8` in
`exponential_adstock` or `order=2` in the model's `transformed_data`, has to be
a Python integer. Write it in the code or pass it as a constant through
{class}`~mmmjax.Data`, as [Scenarios](scenarios) shows. Tracing also explains
why `print` is the wrong tool for looking inside a block.

```{code-cell} ipython3
@jax.jit
def doubled(values):
    print("print sees", values)
    jax.debug.print("jax.debug.print sees {}", values)
    return 2 * values


first = doubled(jnp.array([1.0, 2.0]))
second = doubled(jnp.array([3.0, 4.0]))
```

`print` runs once, while JAX traces the function, and sees only the
placeholder. `jax.debug.print` runs on every call with the real values. The
same tracing is why the first evaluation of a model takes longer than every
call after it, since that's when the blocks are compiled.
