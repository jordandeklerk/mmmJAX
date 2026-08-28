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
    "inverse_gamma",
    "inverse_gamma_logpdf",
    "inverse_gamma_rng",
    "laplace",
    "laplace_logpdf",
    "laplace_rng",
    "lognormal",
    "lognormal_logcdf",
    "lognormal_logpdf",
    "lognormal_logsf",
    "lognormal_rng",
    "normal",
    "normal_logcdf",
    "normal_logpdf",
    "normal_logsf",
    "normal_rng",
    "student_t",
    "student_t_logpdf",
    "student_t_rng",
    "uniform",
    "uniform_logpdf",
    "uniform_rng",
]


def test_distribution_exports_match_root_package() -> None:
    assert distributions.__all__ == DISTRIBUTION_EXPORTS
    assert set(DISTRIBUTION_EXPORTS).issubset(mmmjax.__all__)
    for name in DISTRIBUTION_EXPORTS:
        assert getattr(distributions, name) is getattr(mmmjax, name)
