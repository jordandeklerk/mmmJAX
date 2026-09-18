<p align="center">
  <img src="docs/source/_static/mmmjax-logo.png" alt="mmmJAX" width="480">
  <!-- Switch to the absolute URL before publishing so PyPI renders it:
  <img src="https://raw.githubusercontent.com/jordandeklerk/mmmJAX/main/docs/source/_static/mmmjax-logo.png" alt="mmmJAX" width="480">
  -->
</p>

# Stan-style Bayesian marketing mix modeling in JAX

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/mmmJAX/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code coverage](https://codecov.io/gh/jordandeklerk/mmmJAX/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/mmmJAX)
[![Build status](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/mmmjax/badge/?version=latest)](https://mmmjax.readthedocs.io/en/latest/)

[**Distributions**](#distributions) | [**Features**](#features) | [**Inference**](#inference) | [**Documentation**](https://mmmjax.readthedocs.io/en/latest/)

## What is mmmJAX?

**mmmJAX** is a Python library for Bayesian marketing mix modeling in JAX, built around Stan’s explicit modeling style.

You write models as a sequence of program blocks, with explicit parameter declarations, constraints, transformations, log-density terms, and generated quantities. Priors and structural assumptions remain visible in the model definition, where you can inspect, test, and revise them as you work. The model you read is the model you fit.

The statistical model stays separate from inference. You declare parameters on their natural scale, and mmmJAX handles the transformations to unconstrained space and the required Jacobian adjustments. The resulting log density can be differentiated, compiled, and vectorized with JAX and used with different inference algorithms.

This structure is intended to make models easier to adapt as assumptions and business questions change, from small national specifications to larger hierarchical models.

```python
import numpy as np
import pandas as pd

import mmmjax as mj

rng = np.random.default_rng(0)
tv = rng.gamma(2.0, 1.0, size=52)
price = rng.normal(size=52)
y = 1.0 + 0.8 * tv - 0.5 * price + 0.3 * rng.normal(size=52)
frame = pd.DataFrame({"week": np.arange(52), "y": y, "tv": tv, "price": price})
data = mj.prepare_data(frame, time="week", outcome="y", media=["tv"], controls=["price"])


def transformed_data(controls):
    centered = controls - controls.mean(axis=0)
    return {"centered": centered}


parameters = {
    "intercept": mj.Real(),
    "coefficient": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "slope": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}


def transformed_parameters(media, centered, intercept, coefficient, retention, slope):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    mu = intercept + carried @ coefficient + centered @ slope
    return {"mu": mu}


def log_density(outcome, mu, intercept, coefficient, retention, slope, sigma):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.half_normal(coefficient, 1.0)
    target += mj.beta(retention, 2.0, 4.0)
    target += mj.normal(slope, 0.0, 1.0)
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    return {
        "predictive": {"prediction": prediction},
        "log_likelihood": {"pointwise": pointwise}
        }


model = mj.Model(
    data,
    transformed_data,
    parameters,
    transformed_parameters,
    log_density,
    generated_quantities,
)

results = mj.sample(model, draws=1000, warmup=1000, chains=4)
```

## Features

The program blocks support different likelihoods and hierarchical structures for an outcome observed over time. mmmJAX adds tools for preparing marketing data, defining media effects, and interpreting fitted models. You choose which tools to use and how to combine them.

* **Data preparation.** Select outcome, media, spending, and control columns and fit their scaling, with support for paid media, organic media, and reach and frequency channels.
* **Adstock and saturation.** Model media effects with geometric, delayed, and Weibull adstock and Hill, logistic, log, and root response curves.
* **Seasonality and baselines.** Add calendar seasonality with Fourier features and smooth trends with a Hilbert space Gaussian process.
* **Channel ROI, response curves, and budget allocation.** Use posterior draws from the fitted model to estimate channel returns, construct response curves, and optimize budgets.
* **Prior predictive checks.** Simulate outcomes from the declared priors to assess their implications before fitting.
* **Scenarios.** Re-evaluate the model on new data using the scaling and reference values established from the training data.

## Distributions

mmmJAX provides Stan-style continuous, discrete, and multivariate distributions built on TensorFlow Probability’s JAX backend, with summed and pointwise log densities that support JAX compilation and automatic differentiation. The suite also includes random draws, log cumulative distribution and log survival functions where applicable, and log or logit parameterizations for supported discrete families.

```python
import jax
import mmmjax as mj

draws = mj.normal_rng(jax.random.key(0), 0.0, 1.0, sample_shape=(3,))
term = mj.normal(draws, 0.0, 1.0)              # summed log density for a model block
pointwise = mj.normal_logpdf(draws, 0.0, 1.0)  # one log density per draw
tail = mj.normal_logcdf(draws, 0.0, 1.0)       # log cumulative probability
```

## Inference

By default, `sample` runs [BlackJAX](https://blackjax-devs.github.io/blackjax/)’s NUTS implementation with window adaptation and returns posterior draws, sampler statistics, predictive draws, and model data in an xarray [DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html) labeled with the prepared data’s coordinates. The model exposes its unconstrained log density, random initialization, and parameter transformations for other compatible samplers. The example below uses [NumPyro](https://num.pyro.ai/en/stable/)’s NUTS implementation, with the negative log density as its potential function.

```python
import jax
from numpyro.infer import MCMC, NUTS

potential = lambda position: -model.log_density(position, model.data)
mcmc = MCMC(NUTS(potential_fn=potential), num_warmup=1000, num_samples=1000)
mcmc.run(jax.random.key(0), init_params=model.initialize_random(jax.random.key(1)))
draws = jax.vmap(model.constrain)(mcmc.get_samples())
```

## Documentation

For details about the API, see the [reference documentation](https://mmmjax.readthedocs.io/en/latest/).
