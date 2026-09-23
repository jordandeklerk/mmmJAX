---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Overview

mmmJAX is a library for writing Bayesian marketing mix models in Python. You
write the model yourself as a short program in the shape of a
[Stan](https://mc-stan.org/) program, a few named blocks that state the data,
the parameters, how they combine, the log density, and what to compute from
each posterior draw. The library supplies everything else a marketing model
needs, from data preparation and media transformations to sampling and the
tools that turn a posterior into channel returns and budgets.

Nothing is added on your behalf. The log density is exactly the terms you
write, no prior is chosen for you, and every transformation of the data is a
function call you can see. That is what lets you change the response curve,
the likelihood, or the hierarchy without leaving the package. As the economist
Milton Friedman put it, "There's no such thing as a free lunch," and the same
freedom is why the package asks more of you than a tool with a fixed model
would. It expects you to be comfortable with Bayesian modeling and JAX arrays,
and it leaves the assumptions, and whether the data can support them, in your
hands.

```{code-cell} ipython3
:tags: [remove-cell]

%xmode minimal
```

## The pieces are plain functions

The marketing pieces mmmJAX supplies are ordinary JAX functions of arrays.
Carryover spreads each week's exposure over the weeks that follow, and
saturation makes each additional unit of exposure worth less than the last.

```{code-cell} ipython3
import jax.numpy as jnp
import mmmjax as mj

exposure = jnp.array([0.0, 15.0, 0.0, 0.0, 0.0])
print(mj.geometric_adstock(exposure, alpha=0.5, max_lag=3))
print(mj.hill_saturation(jnp.array([0.5, 1.0, 2.0]), half_saturation=1.0, slope=1.0))
```

The burst of fifteen units in the second week spreads over that week and the
three after it with its total intact, and the saturation curve reaches half
its maximum at its half-saturation point. Because these are plain functions,
you can differentiate them, combine them, or replace them with your own.
Seasonal features, Gaussian processes, and the
[distributions](../user_guide/distributions) work the same way.

## A model is a program you write

A model has up to six blocks, and only the parameters and the log density are
required. The smallest program mmmJAX accepts has just those two.

```{code-cell} ipython3
example = mj.simulate_data(seed=7, n_periods=104, groups=None)
data = mj.prepare_data(example.frame, time="week", outcome="revenue")
scaling = mj.fit_data_scaling(data, scale_outcome=True)


def log_density(outcome, level, noise):
    target = mj.normal(level, 0.0, 1.0) + mj.half_normal(noise, 1.0)
    target += mj.normal(outcome, level, noise)
    return target


model = mj.Model(
    parameters={"level": mj.Real(), "noise": mj.Positive()},
    data=mj.Data(data, scaling=scaling),
    log_density=log_density,
)
```

The program is the model

$$
y_t = \alpha + \varepsilon_t, \qquad
\varepsilon_t \sim \operatorname{Normal}(0, \sigma), \qquad
\alpha \sim \operatorname{Normal}(0, 1), \qquad
\sigma \sim \operatorname{HalfNormal}(1),
$$

where $y_t$ is standardized revenue in week $t$, the level $\alpha$ is
`level`, and the noise scale $\sigma$ is `noise`. Every line of the density
is one line of the model.

A block never receives its inputs by position. Each argument name is a
request, and mmmJAX fills it from the prepared data, from the declared
parameters, or from what an earlier block returned. A name it can't fill
fails when the model is built, and the error lists the names that would have
worked.

```{code-cell} ipython3
:tags: [raises-exception]

def misspelled_log_density(revenue, level, noise):
    return mj.normal(revenue, level, noise)


mj.Model(
    parameters={"level": mj.Real(), "noise": mj.Positive()},
    data=mj.Data(data, scaling=scaling),
    log_density=misspelled_log_density,
)
```

The other blocks run at different moments. `transformed_data` runs once for
each dataset, `transformed_parameters` and `log_density` run at every step of
the sampler, and `generated_quantities` runs once for each saved draw, so
where a calculation goes decides how often it is repeated.

## Your blocks run again for every question

When you ask a fitted model what a channel contributed or how revenue would
respond to a new budget, mmmJAX changes the data, runs your blocks again with
the same posterior draws, and compares the results. That puts two
requirements on the blocks. Anything a prediction depends on has to arrive as
a block argument, so that it follows the change. Anything that defines a
quantity from the training data, such as a normalization, has to read the
training arrays from `reference`, so that it doesn't.
[Scenarios](../user_guide/scenarios) shows what goes wrong otherwise.

The blocks are also JAX code, which JAX traces and compiles before running.
That rules out NumPy functions, Python `if` statements on parameter values,
and array shapes that depend on parameters, as
[User-defined functions](../user_guide/functions) explains.

## What the library handles and what you decide

mmmJAX handles the work that is the same in every marketing mix model. It
takes data from pandas, polars, PyArrow, or any other eager frame that
[narwhals](https://narwhals-dev.github.io/narwhals/) supports, validates and
labels it, fits scaling once and reuses it, provides
carryover, saturation, seasonality, Gaussian process, and distribution
functions, samples the posterior, and evaluates what the fitted model implies
for contributions, returns, response curves, and budgets. Results come back as
labeled xarray objects with the chain and draw axes intact, so every summary
carries the model's uncertainty.

You decide everything that makes a model yours, which means the equations,
the priors and likelihood, the effects the data can actually distinguish, and
whether a contribution the model reports reflects cause or only association.
The analysis tools calculate what your model implies under your assumptions
and add no causal evidence of their own.

## Where to go next

The [User Guide](../user_guide/index) builds a complete model and then
examines and changes each part of it. The [API Reference](../api/index)
documents every function with worked examples.
