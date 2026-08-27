"""Tests for the public distribution API."""

import mmmjax
import mmmjax.distributions as distributions

DISTRIBUTION_EXPORTS = [
    "beta",
    "beta_logpdf",
    "beta_rng",
    "exponential",
    "exponential_logpdf",
    "exponential_rng",
    "gamma",
    "gamma_logpdf",
    "gamma_rng",
    "half_normal",
    "half_normal_logpdf",
    "half_normal_rng",
    "lognormal",
    "lognormal_logpdf",
    "lognormal_rng",
    "normal",
    "normal_logpdf",
    "normal_rng",
    "student_t",
    "student_t_logpdf",
    "student_t_rng",
]


def test_distribution_exports_match_root_package() -> None:
    assert distributions.__all__ == DISTRIBUTION_EXPORTS
    for name in DISTRIBUTION_EXPORTS:
        assert getattr(distributions, name) is getattr(mmmjax, name)
