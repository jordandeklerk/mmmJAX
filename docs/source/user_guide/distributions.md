---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Distributions

Every prior and likelihood term in a model comes from mmmJAX's distribution
library. Each family is a set of plain JAX functions, named after Stan's where
they can be. Because they're plain functions, the terms compose with the rest
of a block and work under `jax.jit`, `jax.grad`, and `jax.vmap`. This page
covers how the functions are named, how their shapes behave, and what happens
at the edges of a family's support.

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
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["figure.dpi"] = 100
```

## One name, several functions

Each distribution comes as a family of functions that share its name.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import mmmjax as mj

values = jnp.array([-1.0, 0.0, 1.0])
print(mj.normal(values, 0.0, 1.0))
print(mj.normal_logpdf(values, 0.0, 1.0))
print(mj.normal_rng(jax.random.key(0), 0.0, 1.0, sample_shape=(3,)))
print(mj.normal_logcdf(0.0, 0.0, 1.0))
print(mj.normal_logsf(1.0, 0.0, 1.0))
```

The five calls above each do a different job.

- `mj.normal` sums the log density over every value, which is what a line in
  `log_density` needs.
- `normal_logpdf` keeps one value per element for pointwise terms such as the
  log likelihood that `generated_quantities` returns.
- `normal_rng` draws from the distribution.
- `normal_logcdf` and `normal_logsf` give the log probability below and above
  a value, which is how you write truncation and censoring.

:::{admonition} The log likelihood needs the pointwise form
:class: important

`generated_quantities` returns the log likelihood under the
[supplied key](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
`"log_likelihood"`, and it needs one term per observation from the pointwise
form. The summed form is stored without an error, but LOO then scores the
whole series as a single observation.
:::

Discrete families use `_logpmf` in place of `_logpdf`. Every family in the
[distribution reference](../api/distributions) has the summed, pointwise, and
`_rng` forms, and families with a closed-form distribution function also have
`_logcdf` and `_logsf`.

The functions of one family agree with each other, so draws from `normal_rng`
settle on the curve that `normal_logpdf` describes.

```{code-cell} ipython3
:tags: [hide-input]

grid = jnp.linspace(-4.0, 4.0, 200)
draws = mj.normal_rng(jax.random.key(1), 0.0, 1.0, sample_shape=(10_000,))

fig, axis = plt.subplots(layout="constrained")
axis.hist(draws, bins=60, density=True, color="0.85", label="normal_rng draws")
axis.plot(grid, jnp.exp(mj.normal_logpdf(grid, 0.0, 1.0)), label="exp(normal_logpdf)")
axis.set_title("Draws and density from one family")
axis.legend(frameon=False)
plt.show()
```

The names follow Stan's where they can, and the summed form keeps every
constant, as a `target +=` statement in Stan does. A few names differ.

- {func}`~mmmjax.laplace` is `double_exponential`.
- {func}`~mmmjax.inverse_gamma` is `inv_gamma`.
- {func}`~mmmjax.negative_binomial` is `neg_binomial_2`.
- {func}`~mmmjax.multivariate_normal` takes a Cholesky factor, as
  `multi_normal_cholesky` does.
- {func}`~mmmjax.lkj_cholesky` is `lkj_corr_cholesky`.
- {func}`~mmmjax.half_normal` and {func}`~mmmjax.truncated_normal` stand in
  for Stan's bounds and truncation.

## Shapes

Arguments broadcast the way NumPy arrays do, and each form treats the
broadcast shape differently.

```{code-cell} ipython3
values = jnp.zeros((5, 3))
print(mj.normal(values, jnp.zeros(3), 1.0).shape)
print(mj.normal_logpdf(values, jnp.zeros(3), 1.0).shape)
print(mj.multivariate_normal_logpdf(values, jnp.zeros(3), jnp.eye(3)).shape)
print(mj.normal_rng(jax.random.key(0), jnp.zeros(3), 1.0, sample_shape=(4,)).shape)
```

The summed form reduces everything to one number, and the pointwise form keeps
the broadcast shape. A multivariate family treats its last axis as a single
event, so five three-dimensional values give five pointwise terms. Draws put
`sample_shape` in front of the shape the parameters broadcast to.

## Log and logit parameterizations

Following Stan, several families have a second form that takes its parameter
on an unconstrained scale. `poisson_log` and `negative_binomial_log`
take the logarithm of the mean, and `bernoulli_logit`, `binomial_logit`,
`categorical_logit`, and `multinomial_logit` take logits.

```{code-cell} ipython3
counts = jnp.array([1, 4])
print(mj.poisson(counts, 3.0))
print(mj.poisson_log(counts, jnp.log(3.0)))
```

Both forms give the same density. `poisson_log` takes $\eta = \log\lambda$ and
evaluates $\log p(y \mid \eta) = y\,\eta - e^{\eta} - \log y!$ directly, so
the only difference is the scale your model works on. A model of counts, such
as weekly conversions, can build its linear predictor on the real line in
`transformed_parameters` and pass it straight to `negative_binomial_log`, with
no exponential in the block and no constraint on the predictor.

## The edges of the support

A value outside a family's support has a log density of minus infinity, the
log of zero probability.

```{code-cell} ipython3
print(mj.half_normal(-1.0, 1.0))
print(mj.beta(1.5, 2.0, 2.0))
print(mj.poisson(2.5, 3.0))
```

A parameter declared with {class}`~mmmjax.Positive` or
{class}`~mmmjax.Interval` stays inside its support, so this mostly shows up
through the data.

:::{admonition} Check the data against the likelihood
:class: warning

A negative week under a lognormal likelihood or a fractional count under a
Poisson one gives the whole model a log density of minus infinity. A sampler
can't start from a point where the density isn't finite.
:::

The results follow JAX's precision setting, which is 32-bit unless you enable
64-bit as [Installation](../getting_started/installation) describes.

## Built on TensorFlow Probability

Each family wraps the matching distribution from the JAX backend of
TensorFlow Probability, which supplies the numerics and the random draws.

```{code-cell} ipython3
from tensorflow_probability.substrates.jax import distributions as tfd

values = jnp.array([-1.0, 0.0, 1.0])
print(tfd.Normal(0.0, 1.0).log_prob(values).sum())
print(jax.grad(mj.normal, argnums=1)(values, 0.5, 1.0))
```

The summed normal is the same number TensorFlow Probability gives, and its
gradient with respect to the location comes from `jax.grad` like any other JAX
function. mmmJAX adds names close to Stan's and the summed, pointwise, and
draw forms on top. A {class}`~mmmjax.Prior` binds a family to fixed settings,
which [Priors](priors) uses to check a model's priors. If you need a family the
library lacks, you can write it with {func}`~mmmjax.custom_distribution`, as
[User-defined functions](functions) shows.
