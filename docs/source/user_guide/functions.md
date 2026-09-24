---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# User-defined functions

Stan programs have a `functions` block for code a model reuses. mmmJAX doesn't
need one, because the blocks are ordinary Python functions and can call any
function you write. This page writes a response curve and a distribution of
its own, puts both into a version of the model from [A first
model](first_model), and covers the rules JAX sets for code that runs inside a
block.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
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
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
```

## A response curve of your own

An exponential saturation curve levels off faster than the Hill curve, which
keeps climbing slowly for a long time. Written with `jax.numpy`, it is an
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
[Distributions](distributions). A half-Cauchy
prior, a common choice for scale parameters, takes two short functions.

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
works with {func}`~mmmjax.sample_prior` as well. Nothing checks that the two
functions describe the same distribution. A draw function that disagrees with
its density gives a prior that draws from one distribution and scores another,
and the model still runs. A histogram of many draws laid over the density shows whether
the two agree.

```{code-cell} ipython3
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
density describes. Each count is divided by all 10,000 draws rather than by the
ones below 10, which keeps the bars on the density's scale when some draws land
past the right edge. Those draws are the heavy tail that sets the half-Cauchy
apart.

```{code-cell} ipython3
half_normal_draws = mj.half_normal_rng(jax.random.key(0), 1.0, sample_shape=(10_000,))
print(round(float(jnp.mean(draws > 10.0)), 3))
print(round(float(jnp.mean(half_normal_draws > 10.0)), 3))
```

Of the half-Cauchy draws, 6.7 percent are above 10, and none of 10,000
half-normal draws with the same scale are.

## Both in a model

The helper replaces the Hill curve in `transformed_parameters`, with a `rate`
parameter $\lambda_c$ in place of `half_saturation`, and the new distribution
becomes the prior on `sigma`. The model equation and the priors that change
are

$$
\begin{aligned}
y_t &= \alpha + \sum_{c} \beta_c \Big(1 - \exp\big(-\lambda_c \operatorname{Adstock}(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c)\big)\Big) + \gamma\, p_t + \varepsilon_t, \\
\lambda_c &\sim \operatorname{LogNormal}(0, 0.5), \qquad \sigma \sim \operatorname{HalfCauchy}(1),
\end{aligned}
$$

with everything else as in [A first model](first_model).

```{code-cell} ipython3
exponential_parameters = {
    "intercept": mj.Real(),
    "coefficient": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "rate": mj.Positive(dims="channel"),
    "control_coefficient": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}


def exponential_transformed_parameters(
    media,
    controls,
    intercept,
    coefficient,
    retention,
    rate,
    control_coefficient,
):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = exponential_saturation(carried, rate)
    mu = intercept + saturated @ coefficient + controls @ control_coefficient
    return {"mu": mu}


def exponential_log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    rate,
    control_coefficient,
    sigma,
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.half_normal(coefficient, 2.0)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(rate, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += half_cauchy(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


exponential_model = mj.Model(
    parameters=exponential_parameters,
    data=mj.Data(data, scaling=scaling),
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

```{code-cell} ipython3
exponential["posterior"]["rate"].mean(("chain", "draw")).to_series().round(2)
```

A typical week of TV, which sits at one on the scaled axis, reaches about 63
percent of the channel's maximum effect under this curve, and a typical week
of search about 53 percent. Nothing about either function is special to
mmmJAX. They run inside the blocks because they are written with JAX, which is
the one requirement the next section spells out.

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
`jax.numpy`. Anything that sets an array's shape, such as the carryover
length or the number of Fourier terms, has to be a Python integer, written in
the block or passed as a constant through {class}`~mmmjax.Data`, as
[Scenarios](scenarios) shows. Tracing also explains why `print` is the wrong
tool for looking inside a block.

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
call after it, since that is when the blocks are compiled.
