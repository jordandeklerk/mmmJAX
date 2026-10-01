---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Distributions

Every prior and likelihood term in a model comes from mmmJAX's distribution
library. Each family is a set of plain JAX functions, so the terms compose with
the rest of a block and work under `jax.jit`, `jax.grad`, and `jax.vmap`. The
sections below go through each family's forms, shapes, parameterizations, and
support in turn.

## One name, several functions

Each distribution comes as a family of functions that share its name, as the
five normal ones below show.

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

The five calls above use the same normal distribution for five different jobs.

- `mj.normal` gives the summed log density that a line in `log_density` adds.
- `normal_logpdf` keeps one value per element for pointwise terms such as the
  log likelihood that `generated_quantities` returns.
- `normal_rng` draws from the distribution.
- `normal_logcdf` and `normal_logsf` give the log probability below and above
  a value for truncation and censoring.

A discrete family's pointwise form ends in `_logpmf` rather than `_logpdf`, as
`poisson_logpmf` does. Every family in the
[distribution reference](../api/distributions) has the summed, pointwise, and
`_rng` forms, and families with a closed-form distribution function also have
`_logcdf` and `_logsf`.

:::{admonition} The log likelihood needs the pointwise form
:class: important

`generated_quantities` returns the log likelihood under the
[supplied key](../getting_started/what_is_mmmjax.md#a-model-is-a-program)
`"log_likelihood"`, and it needs one term per observation from the pointwise
form. If you return the summed form instead, it's stored without an error,
but a check that scores the weeks one at a time then sees the whole series as a
single week.
:::

The functions of one family agree, so draws from `normal_rng` settle on the
curve that `normal_logpdf` describes.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import plotnine as pn

draws = np.asarray(mj.normal_rng(jax.random.key(1), 0.0, 1.0, sample_shape=(10_000,)))
grid = jnp.linspace(draws.min(), draws.max(), 200)
curve = jnp.exp(mj.normal_logpdf(grid, 0.0, 1.0))
density = pd.DataFrame({"value": np.asarray(grid), "density": np.asarray(curve)})
(
    pn.ggplot(pd.DataFrame({"value": draws}), pn.aes("value"))
    + pn.geom_histogram(
        pn.aes(y=pn.after_stat("density")), bins=60, fill="#d2d3fb", color="#2a2eec", size=0.3
    )
    + pn.geom_line(pn.aes("value", "density"), data=density, color="#262626", size=0.9)
    + pn.labs(x="Value", y="Density")
    + mj.theme_mmmjax()
)
```

Each bar counts the draws that land in its bin, scaled to the density's units,
and the dark curve is the density that `normal_logpdf` gives. The bars follow
the curve out into the tails, so the two functions agree.

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

The first line is `()` because the summed form reduces everything to one
number, while the pointwise form keeps the broadcast shape `(5, 3)`. On the
third line, a multivariate family treats its last axis as a single event, so
five three-dimensional values give five pointwise terms. Draws put
`sample_shape` in front of the shape the parameters broadcast to, so four draws
of three means come out as `(4, 3)`.

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

The two calls print the same number because both forms give the same density.
`poisson_log` takes $\eta = \log\lambda$ and evaluates

$$
\log p(y \mid \eta) = y\,\eta - e^{\eta} - \log y!
$$

directly, so the only difference is the scale your model works on. A model of
counts, such as weekly conversions, can build its linear predictor on the real
line in `transformed_parameters` and pass it straight to
`negative_binomial_log`.

## The edges of the support

A value outside a family's support has a log density of minus infinity, the
log of zero probability.

```{code-cell} ipython3
print(mj.half_normal(-1.0, 1.0))
print(mj.beta(1.5, 2.0, 2.0))
print(mj.poisson(2.5, 3.0))
```

A half-normal value can't be negative, a beta value can't pass one, and a
Poisson count must be whole, so all three print `-inf`. Since a parameter
declared with {class}`~mmmjax.Positive` or {class}`~mmmjax.Interval` stays
inside its support, you'll mostly run into this through the data.

:::{admonition} Check the data against the likelihood
:class: warning

A negative week under a lognormal likelihood or a fractional count under a
Poisson one gives the whole model a log density of minus infinity, and a
sampler can't start from a point where the density isn't finite.
:::

## If you know Stan

The summed form keeps every constant, as a `target +=` statement in Stan does,
and the names follow Stan's where they can, so only a handful differ.

- {func}`~mmmjax.laplace` is `double_exponential`.
- {func}`~mmmjax.inverse_gamma` is `inv_gamma`.
- {func}`~mmmjax.negative_binomial` is `neg_binomial_2`.
- {func}`~mmmjax.multivariate_normal` takes a Cholesky factor, as
  `multi_normal_cholesky` does.
- {func}`~mmmjax.lkj_cholesky` is `lkj_corr_cholesky`.
- {func}`~mmmjax.half_normal` and {func}`~mmmjax.truncated_normal` stand in
  for Stan's bounds and truncation.

## Built on TensorFlow Probability

Each family wraps the matching distribution from the JAX backend of
TensorFlow Probability and adds the summed, pointwise, and draw forms on top.

```{code-cell} ipython3
from tensorflow_probability.substrates.jax import distributions as tfd

values = jnp.array([-1.0, 0.0, 1.0])
print(tfd.Normal(0.0, 1.0).log_prob(values).sum())
print(jax.grad(mj.normal, argnums=1)(values, 0.5, 1.0))
```

TensorFlow Probability's summed log density on the first line matches what
`mj.normal` printed at the top of the page. On the second line, `jax.grad`
takes the gradient of `mj.normal` with respect to the location as it would for
any other JAX function.

A {class}`~mmmjax.Prior` binds a family to fixed settings for the checks on
[Priors](priors). If you need a family the library lacks, write it with
{func}`~mmmjax.custom_distribution`, as [User-defined functions](functions)
shows.
