<div align="center">
<img src="docs/source/_static/mmmjax-logo.png" alt="mmmJAX" width="375">
<!-- Switch to the absolute URL before publishing so PyPI renders it:
<img src="https://raw.githubusercontent.com/jordandeklerk/mmmJAX/main/docs/source/_static/mmmjax-logo.png" alt="mmmJAX" width="350">
-->

## Stan-style Bayesian marketing mix modeling in JAX

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/mmmJAX/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code coverage](https://codecov.io/gh/jordandeklerk/mmmJAX/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/mmmJAX)
[![Build status](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/mmmjax/badge/?version=latest)](https://mmmjax.readthedocs.io/en/latest/)

[Why mmmJAX](#why-mmmjax) | [Installation](#installation) | [Program blocks](#program-blocks) | [Distributions](#distributions) | [Inference](#inference-is-separate-from-the-model) | [Documentation](https://mmmjax.readthedocs.io/en/latest/)

</div>

## What is mmmJAX?

mmmJAX is a library for Bayesian marketing mix modeling in [JAX](https://docs.jax.dev/). It borrows the shape of a [Stan](https://mc-stan.org/) program, so a model is a sequence of named blocks, each a plain Python function or dictionary, and every prior and structural assumption is written where you would change it.

Because the blocks are ordinary JAX code, the finished model can be differentiated, compiled, and vectorized. mmmJAX supplies the marketing pieces that go inside them, such as adstock and saturation functions, seasonal features, and a NUTS sampler, and none of them are required.

## Why mmmJAX

We believe packaged marketing mix modeling APIs are a great way to get started. They fix the structure of the model in advance and choose most of the assumptions for you, and for a first model that is exactly what you want.

Once MMM scales, the business starts asking more specific questions than a first model was built to answer. Answering those questions means changing the model itself, and even a flexible packaged API only lets you do that where and how its authors chose.

mmmJAX intends to fill that gap. The model is a program you write, so extending it means editing a function instead of waiting for a new option. Everything else the library offers, from data preparation to budget optimization, keeps working as the model grows.

## Installation

mmmJAX is in alpha and requires Python 3.12 or later. It is not yet on PyPI, so install the development version from GitHub.

```bash
pip install "git+https://github.com/jordandeklerk/mmmJAX.git"
```

JAX runs on the CPU by default. For GPU sampling, install the JAX wheel for your accelerator first by following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Program blocks

A model is built from up to six blocks, given to `Model` in the order they run. This skeleton shows all of them with their bodies left as comments. Each function asks for what it needs by argument name, and mmmJAX supplies it from the prepared data, the outputs of earlier blocks, or the parameter draws.

```python
import mmmjax as mj

data = mj.prepare_data(frame, time="week", outcome="sales", media=channels)


def transformed_data(media, controls):
    # fixed calculations on the data, evaluated once
    return {...}


parameters = {
    # each name paired with a declaration such as mj.Positive(dims="channel")
}


def transformed_parameters(media, centered, coefficient, retention):
    # derived quantities shared by the two blocks below
    return {...}


def log_density(outcome, mu, coefficient, retention, sigma):
    # priors and likelihood, added one term at a time
    return target


def generated_quantities(key, outcome, mu, sigma):
    # predictions and pointwise terms computed from each draw
    return {...}


model = mj.Model(
    data,
    transformed_data,
    parameters,
    transformed_parameters,
    log_density,
    generated_quantities,
)
```

Only `parameters` and `log_density` are required. Data-only work runs once, the density runs on every evaluation, and generated quantities run once per retained draw. Parameters are declared with their constraints and named axes, and each declaration owns the transform to the unconstrained scale and its Jacobian adjustment, so every block sees parameters on their natural scale and the log density contains only the terms you wrote.

## Distributions

Every prior and likelihood term comes from a Stan-style distribution library built on TensorFlow Probability. Each family has a summed log density for use in `log_density`, a pointwise version for likelihood diagnostics, a random draw function, and log cumulative and log survival functions where they exist. They are plain JAX functions, so they broadcast, differentiate, and compile like any other.

```python
import jax
import mmmjax as mj

term = mj.gamma(values, shape=2.0, rate=0.5)
gradient = jax.grad(mj.gamma)(values, shape=2.0, rate=0.5)
draws = mj.gamma_rng(jax.random.key(0), shape=2.0, rate=0.5, sample_shape=(1000,))
```

## Inference is separate from the model

A `Model` does not know how it will be fit. It exposes `log_density` for an unconstrained position, `initialize_random` for a starting point, and `constrain` to map draws back to the parameter scale, and that is the whole interface a sampler needs. The log density is an ordinary JAX function, so its gradient is one transformation away.

```python
position = model.initialize_random(jax.random.key(0))
value, gradient = jax.value_and_grad(model.log_density)(position, model.data)
parameters = model.constrain(position)
```

The built-in `sample` uses this interface to run NUTS with window adaptation and returns an xarray [DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html) labeled with your data's coordinates, with the sampler state stored alongside so `continue_sampling` can add draws later. Any other sampler that accepts a log density and its gradient, in NumPyro, BlackJAX, or your own code, works with the same object.

## Documentation

For details about the API, see the [reference documentation](https://mmmjax.readthedocs.io/en/latest/).
