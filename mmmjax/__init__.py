"""Bayesian marketing mix modeling expressed entirely in JAX."""

from mmmjax.distributions import (
    exponential,
    exponential_logpdf,
    exponential_rng,
    normal,
    normal_logpdf,
    normal_rng,
)
from mmmjax.model import Model
from mmmjax.parameters import Interval, LowerBound, Parameterization, Positive, Real, UpperBound

__version__ = "0.0.1"

__all__ = [
    "Interval",
    "LowerBound",
    "Model",
    "Parameterization",
    "Positive",
    "Real",
    "UpperBound",
    "exponential",
    "exponential_logpdf",
    "exponential_rng",
    "normal",
    "normal_logpdf",
    "normal_rng",
]
