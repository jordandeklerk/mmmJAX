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

[What is mmmJAX](#what-is-mmmjax) | [Installation](#installation) | [Program blocks](#program-blocks) | [Distributions](#distributions) | [Inference](#inference-is-separate-from-the-model) | [Documentation](https://mmmjax.readthedocs.io/en/latest/)

</div>

## What is mmmJAX?

mmmJAX is an open-source Python library for Bayesian marketing mix modeling in [JAX](https://docs.jax.dev/), built for experienced developers and practitioners. We believe packaged MMM APIs give users a great first model but often fall short as business demands grow. In mmmJAX you write the model yourself as a [Stan](https://mc-stan.org/)-style program of plain Python blocks, and the library makes none of the modeling decisions for you. If you're new to marketing mix models or to Bayesian modeling, a packaged tool with sensible defaults is a more forgiving place to start.

Everything around the model comes packaged as convenient shortcuts, from data preparation and media transformations to sampling, budget optimization, and plots, so that your time goes into building the model. The blocks are ordinary JAX code, so the model can be differentiated, compiled, and vectorized.

## Installation

mmmJAX is in alpha and requires Python 3.12 or later. It is not yet on PyPI, so install the development version from GitHub.

```bash
pip install "git+https://github.com/jordandeklerk/mmmJAX.git"
```

JAX runs on the CPU by default. On an NVIDIA GPU with CUDA 12, the `gpu` extra installs the CUDA build of JAX with mmmJAX, and the [installation guide](https://mmmjax.readthedocs.io/en/latest/getting_started/installation.html) covers other accelerators and 64-bit precision.

## Program blocks

A model is built from up to six blocks, given to `Model` in the order they run. Each function asks for what it needs by argument name, and mmmJAX supplies it from the prepared data, the outputs of earlier blocks, or the parameter draws.

```python
import mmmjax as mj

data = mj.prepare_data(frame, time="week", outcome="sales", media=channels, controls=controls)


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

## Distributions

Every prior and likelihood term comes from a Stan-style distribution library built on TensorFlow Probability. They are plain JAX functions, so they broadcast, differentiate, and compile like any other.

```python
import jax
import mmmjax as mj

term = mj.gamma(values, shape=2.0, rate=0.5)
gradient = jax.grad(mj.gamma)(values, shape=2.0, rate=0.5)
draws = mj.gamma_rng(jax.random.key(0), shape=2.0, rate=0.5, sample_shape=(1000,))
```

## Inference is separate from the model

A `Model` does not know how it will be fit. It exposes `log_density` for an unconstrained position, `initialize_random` for a starting point, and `constrain` to map draws back to the parameter scale, and that is the whole interface a sampler needs.

```python
position = model.initialize_random(jax.random.key(0))
value, gradient = jax.value_and_grad(model.log_density)(position, model.data)
parameters = model.constrain(position)
```

## Documentation

For details about the API, see the [reference documentation](https://mmmjax.readthedocs.io/en/latest/).
