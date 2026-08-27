"""Stan-inspired Bayesian marketing mix models expressed entirely in JAX."""

from mmmjax.distributions import (
    exponential,
    exponential_logpdf,
    exponential_rng,
    normal,
    normal_logpdf,
    normal_rng,
)
from mmmjax.parameters import Parameterization, Positive, Real

__version__ = "0.0.1"

__all__ = [
    "Parameterization",
    "Positive",
    "Real",
    "exponential",
    "exponential_logpdf",
    "exponential_rng",
    "normal",
    "normal_logpdf",
    "normal_rng",
]
