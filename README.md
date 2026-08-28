# Stan-inspired Bayesian marketing mix modeling in JAX

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/jordandeklerk/mmmJAX/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code coverage](https://codecov.io/gh/jordandeklerk/mmmJAX/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/mmmJAX)
[![Build status](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/mmmJAX/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/mmmjax/badge/?version=latest)](https://mmmjax.readthedocs.io/en/latest/)

[**Documentation**](https://mmmjax.readthedocs.io/en/latest/)

## What is mmmJAX?

**mmmJAX** is a Python library for expressing Bayesian marketing mix models entirely in JAX using
a Stan-inspired modeling style.

Bayesian modeling is an iterative process of specifying, fitting, checking, and revising a model.
mmmJAX keeps named parameters, constraints, log-density terms, and generated quantities visible
and testable in the model definition.

The statistical model remains separate from inference. You define parameters on their natural
scale, while mmmJAX maps them to the unconstrained space required by gradient-based algorithms
and applies the corresponding density adjustments. The resulting log density can be differentiated,
compiled, and vectorized with JAX without tying the model to a particular sampler.

For marketing mix modeling, this architecture is intended to support models that evolve as
assumptions and business questions change, from practical defaults to custom specifications and
large hierarchical settings.
