"""Tests for Negative Binomial distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import nbinom as jax_negative_binomial_distribution
from scipy import special, stats

from mmmjax import (
    negative_binomial,
    negative_binomial_log,
    negative_binomial_log_logpmf,
    negative_binomial_log_rng,
    negative_binomial_logpmf,
    negative_binomial_rng,
    poisson_log_logpmf,
)


def test_negative_binomial_logpmf_matches_scipy_across_support_and_broadcasting() -> None:
    values = np.array([[-1.0], [0.0], [1.0], [5.0], [20.0], [0.5], [np.nan]], dtype=np.float32)
    means = np.array([0.2, 3.0, 20.0], dtype=np.float32)
    concentrations = np.array([0.5, 2.5, 10.0], dtype=np.float32)
    probabilities = concentrations.astype(np.float64) / (concentrations.astype(np.float64) + means.astype(np.float64))
    expected = stats.nbinom.logpmf(values.astype(np.float64), concentrations.astype(np.float64), probabilities)

    result = negative_binomial_logpmf(values, means, concentrations)

    assert result.shape == (7, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6, equal_nan=True)


def test_negative_binomial_logpmf_matches_jax_on_ordinary_inputs() -> None:
    values = jnp.array([[0], [1], [4], [20]])
    means = jnp.array([0.2, 1.5, 5.0])
    concentrations = jnp.array([0.7, 2.0, 10.0])
    probabilities = concentrations / (concentrations + means)
    expected = jax_negative_binomial_distribution.logpmf(values, concentrations, probabilities)

    result = negative_binomial_logpmf(values, means, concentrations)
    compiled = jax.jit(negative_binomial_logpmf)(values, means, concentrations)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(compiled, result, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("value", [0, 1, 2, 5])
def test_negative_binomial_logpmf_matches_geometric_special_case(value: int) -> None:
    expected = -(value + 1) * jnp.log(2)

    result = negative_binomial_logpmf(value, mean=1, concentration=1)

    assert jnp.allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("mean", "concentration", "maximum_count"),
    [(0.2, 0.5, 100), (3.0, 2.0, 200), (20.0, 5.0, 500)],
)
def test_negative_binomial_probability_mass_normalizes(
    mean: float,
    concentration: float,
    maximum_count: int,
) -> None:
    values = jnp.arange(maximum_count + 1)
    probabilities = jnp.exp(negative_binomial_logpmf(values, mean, concentration))

    assert jnp.allclose(jnp.sum(probabilities), 1, rtol=0, atol=5e-6)


@pytest.mark.parametrize(
    ("mean", "concentration", "maximum_count"),
    [(0.2, 0.5, 100), (3.0, 2.0, 200), (20.0, 5.0, 500)],
)
def test_negative_binomial_probability_mass_has_expected_moments(
    mean: float,
    concentration: float,
    maximum_count: int,
) -> None:
    values = jnp.arange(maximum_count + 1, dtype=jnp.float32)
    probabilities = jnp.exp(negative_binomial_logpmf(values, mean, concentration))
    calculated_mean = jnp.sum(values * probabilities)
    calculated_variance = jnp.sum(jnp.square(values - calculated_mean) * probabilities)
    expected_variance = mean + mean**2 / concentration

    assert jnp.allclose(calculated_mean, mean, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(calculated_variance, expected_variance, rtol=3e-5, atol=3e-5)


def test_negative_binomial_logpmf_requires_exact_nonnegative_integer_values() -> None:
    values = jnp.array(
        [
            -jnp.inf,
            -1.0,
            -jnp.finfo(jnp.float32).tiny,
            -0.0,
            0.5,
            1.0,
            jnp.nextafter(jnp.float32(1), jnp.inf),
            jnp.inf,
            jnp.nan,
        ]
    )

    result = negative_binomial_logpmf(values, mean=2, concentration=1.5)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 2, 4, 6, 7])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([3, 5])]))
    assert jnp.isnan(result[8])


def test_negative_binomial_logpmf_rejects_invalid_parameters_before_support() -> None:
    invalid_means = jnp.array([-jnp.inf, -1.0, 0.0, jnp.inf, jnp.nan])
    invalid_concentrations = jnp.array([-jnp.inf, -1.0, 0.0, jnp.inf, jnp.nan])

    mean_results = negative_binomial_logpmf(-1, invalid_means, concentration=1)
    concentration_results = negative_binomial_logpmf(-1, mean=1, concentration=invalid_concentrations)

    assert jnp.all(jnp.isnan(mean_results))
    assert jnp.all(jnp.isnan(concentration_results))


def test_negative_binomial_sums_broadcast_log_masses() -> None:
    values = jnp.array([[0], [1], [4]])
    means = jnp.array([0.5, 2.0])
    concentrations = jnp.array([1.5, 5.0])

    pointwise = negative_binomial_logpmf(values, means, concentrations)
    result = negative_binomial(values, means, concentrations)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(pointwise))


def test_negative_binomial_empty_batch_returns_scalar_zero() -> None:
    result = negative_binomial(jnp.empty((0,), dtype=jnp.int32), mean=-1, concentration=-1)

    assert result.shape == ()
    assert result == 0


def test_negative_binomial_logpmf_avoids_large_count_cancellation() -> None:
    value = 10_000_000
    mean = 10_000_000.0
    concentration = 5.0
    probability = concentration / (concentration + mean)
    # SciPy's PMF uses Boost and stays accurate where its direct logpmf cancels
    expected = np.log(stats.nbinom.pmf(value, concentration, probability))

    result = negative_binomial_logpmf(value, mean, concentration)

    assert jnp.allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_negative_binomial_log_parameterization_avoids_large_count_cancellation() -> None:
    value = 10_000_000
    mean = 10_000_000.0
    concentration = 5.0
    probability = concentration / (concentration + mean)
    # SciPy's PMF uses Boost and stays accurate where its direct logpmf cancels
    expected = np.log(stats.nbinom.pmf(value, concentration, probability))

    result = negative_binomial_log_logpmf(value, np.log(mean), concentration)

    assert jnp.allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_negative_binomial_logpmf_avoids_float64_cancellation() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")

    value = np.int64(1_000_000_000_000_000)
    mean = np.float64(1_000_000_000_000_000)
    concentrations = np.array([5.0, 1_000_000_000_000_000], dtype=np.float64)
    probabilities = concentrations / (concentrations + mean)
    # SciPy's PMF uses Boost and stays accurate where its direct logpmf cancels
    expected = np.log(stats.nbinom.pmf(value, concentrations, probabilities))

    result = negative_binomial_logpmf(value, mean, concentrations)

    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)


def test_negative_binomial_logpmf_approaches_poisson_at_large_concentration() -> None:
    values = np.array([0, 1, 4, 20, 40], dtype=np.int32)
    means = np.array([0.2, 1.0, 4.5, 20.0, 30.0], dtype=np.float32)
    expected = stats.poisson.logpmf(values, means.astype(np.float64))

    result = negative_binomial_logpmf(values, means, concentration=jnp.float32(1e20))

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=5e-6)


def test_negative_binomial_parameter_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0, 4.0])
    means = jnp.array([0.2, 1.5, 5.0])
    concentrations = jnp.array([0.7, 2.0, 10.0])

    def evaluate(current_means, current_concentrations):
        return negative_binomial_logpmf(values, current_means, current_concentrations)

    mean_derivative = concentrations * (values - means) / (means * (means + concentrations))
    concentration_derivative = (
        jax.scipy.special.digamma(values + concentrations)
        - jax.scipy.special.digamma(concentrations)
        + jnp.log(concentrations / (means + concentrations))
        + (means - values) / (means + concentrations)
    )
    expected_mean_jacobian = jnp.diag(mean_derivative)
    expected_concentration_jacobian = jnp.diag(concentration_derivative)

    forward = jax.jit(jax.jacfwd(evaluate, argnums=(0, 1)))(means, concentrations)
    reverse = jax.jit(jax.jacrev(evaluate, argnums=(0, 1)))(means, concentrations)

    assert jnp.allclose(forward[0], expected_mean_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(reverse[0], expected_mean_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(forward[1], expected_concentration_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(reverse[1], expected_concentration_jacobian, rtol=3e-6, atol=3e-6)


def test_negative_binomial_parameter_hessian_matches_closed_form() -> None:
    value = jnp.asarray(4.0)
    mean = jnp.asarray(5.0)
    concentration = jnp.asarray(2.3)
    parameter_sum = mean + concentration

    def evaluate(parameters):
        current_mean, current_concentration = parameters
        return negative_binomial_logpmf(value, current_mean, current_concentration)

    mean_curvature = -value / jnp.square(mean) + (value + concentration) / jnp.square(parameter_sum)
    mixed_curvature = (value - mean) / jnp.square(parameter_sum)
    concentration_curvature = (
        jax.scipy.special.polygamma(1, value + concentration)
        - jax.scipy.special.polygamma(1, concentration)
        + 1 / concentration
        - 1 / parameter_sum
        - (mean - value) / jnp.square(parameter_sum)
    )
    expected = jnp.array(
        [
            [mean_curvature, mixed_curvature],
            [mixed_curvature, concentration_curvature],
        ]
    )

    result = jax.jit(jax.hessian(evaluate))(jnp.array([mean, concentration]))

    assert jnp.allclose(result, expected, rtol=5e-6, atol=5e-6)


def test_negative_binomial_large_concentration_gradients_preserve_poisson_limit() -> None:
    value = jnp.float32(10)
    mean = jnp.float32(10)
    concentration = jnp.float32(1e20)

    def evaluate(current_mean, current_concentration):
        return negative_binomial_logpmf(value, current_mean, current_concentration)

    forward = jax.jacfwd(evaluate, argnums=(0, 1))(mean, concentration)
    reverse = jax.jacrev(evaluate, argnums=(0, 1))(mean, concentration)

    assert forward[0] == 0
    assert reverse[0] == 0
    assert jnp.allclose(forward[1], reverse[1], rtol=2e-6, atol=1e-25)
    assert jnp.abs(forward[1]) < 1e-20


def test_negative_binomial_unsupported_values_have_zero_parameter_derivatives() -> None:
    def evaluate(current_mean, current_concentration):
        return negative_binomial_logpmf(-1, current_mean, current_concentration)

    forward = jax.jacfwd(evaluate, argnums=(0, 1))(2.0, 1.5)
    reverse = jax.jacrev(evaluate, argnums=(0, 1))(2.0, 1.5)

    assert forward == (0, 0)
    assert reverse == (0, 0)


def test_negative_binomial_counts_do_not_control_parameter_dtype() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")

    result = negative_binomial_logpmf(
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.float64(2),
        jnp.float64(1.5),
    )

    assert result.dtype == jnp.dtype(jnp.float64)


def test_negative_binomial_log_parameterization_matches_mean_parameterization() -> None:
    values = jnp.array([[0], [1], [4], [20]])
    means = jnp.array([0.2, 1.5, 5.0, 20.0])
    concentrations = jnp.array([0.7, 2.0, 10.0, 50.0])

    result = negative_binomial_log_logpmf(values, jnp.log(means), concentrations)
    compiled = jax.jit(negative_binomial_log_logpmf)(values, jnp.log(means), concentrations)
    expected = negative_binomial_logpmf(values, means, concentrations)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(compiled, expected, rtol=3e-6, atol=3e-6)


def test_negative_binomial_log_parameterization_matches_scipy_and_jax() -> None:
    values = np.array([[0], [1], [4], [12]], dtype=np.float32)
    log_means = np.array([-3.0, -0.5, 1.0, 3.0], dtype=np.float32)
    concentrations = np.array([0.7, 2.0, 5.0, 20.0], dtype=np.float32)
    probabilities = special.expit(np.log(concentrations.astype(np.float64)) - log_means.astype(np.float64))
    scipy_expected = stats.nbinom.logpmf(
        values.astype(np.float64),
        concentrations.astype(np.float64),
        probabilities,
    )
    jax_expected = jax_negative_binomial_distribution.logpmf(
        values,
        concentrations,
        probabilities.astype(np.float32),
    )

    result = negative_binomial_log_logpmf(values, log_means, concentrations)

    np.testing.assert_allclose(result, scipy_expected, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(result, jax_expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("log_mean", [-1000.0, -100.0, 100.0, 1000.0])
def test_negative_binomial_log_parameterization_handles_extreme_finite_means(log_mean: float) -> None:
    values = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    concentrations = np.array([0.7, 2.0, 10.0], dtype=np.float64)
    log_parameter_sum = np.logaddexp(log_mean, np.log(concentrations))
    expected = (
        special.gammaln(values + concentrations)
        - special.gammaln(concentrations)
        - special.gammaln(values + 1)
        + values * log_mean
        + concentrations * np.log(concentrations)
        - (values + concentrations) * log_parameter_sum
    )

    result = negative_binomial_log_logpmf(
        values.astype(np.float32),
        jnp.float32(log_mean),
        concentrations.astype(np.float32),
    )
    compiled = jax.jit(negative_binomial_log_logpmf)(
        values.astype(np.float32),
        jnp.float32(log_mean),
        concentrations.astype(np.float32),
    )

    assert jnp.all(jnp.isfinite(result))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-5)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-5)


@pytest.mark.parametrize(
    ("log_mean", "concentration"),
    [(90.0, jnp.finfo(jnp.float32).max), (100.0, jnp.float32(1e38))],
)
def test_negative_binomial_log_parameterization_returns_negative_infinity_when_mass_overflows(
    log_mean,
    concentration,
) -> None:
    result = negative_binomial_log_logpmf(jnp.array([1, 2]), log_mean, concentration)

    assert jnp.all(jnp.isneginf(result))


def test_negative_binomial_log_parameterization_requires_finite_parameters_before_support() -> None:
    invalid_log_means = jnp.array([-jnp.inf, jnp.inf, jnp.nan])
    invalid_concentrations = jnp.array([-jnp.inf, -1.0, 0.0, jnp.inf, jnp.nan])

    log_mean_results = negative_binomial_log_logpmf(-1, invalid_log_means, concentration=1)
    concentration_results = negative_binomial_log_logpmf(-1, log_mean=0, concentration=invalid_concentrations)

    assert jnp.all(jnp.isnan(log_mean_results))
    assert jnp.all(jnp.isnan(concentration_results))


def test_negative_binomial_log_parameterization_checks_count_support_before_conversion() -> None:
    values = jnp.array(
        [
            -jnp.inf,
            -1.0,
            -jnp.finfo(jnp.float32).tiny,
            -0.0,
            0.5,
            1.0,
            jnp.nextafter(jnp.float32(1), jnp.inf),
            jnp.inf,
            jnp.nan,
        ]
    )

    result = negative_binomial_log_logpmf(values, log_mean=0.5, concentration=1.5)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 2, 4, 6, 7])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([3, 5])]))
    assert jnp.isnan(result[8])


def test_negative_binomial_log_parameter_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0, 4.0])
    log_means = jnp.log(jnp.array([0.2, 1.5, 5.0]))
    concentrations = jnp.array([0.7, 2.0, 10.0])
    probabilities = jax.nn.sigmoid(log_means - jnp.log(concentrations))
    expected_log_mean_jacobian = jnp.diag(values - (values + concentrations) * probabilities)
    expected_concentration_jacobian = jnp.diag(
        jax.scipy.special.digamma(values + concentrations)
        - jax.scipy.special.digamma(concentrations)
        + jax.nn.log_sigmoid(jnp.log(concentrations) - log_means)
        + probabilities
        - values * jnp.exp(-jnp.logaddexp(log_means, jnp.log(concentrations)))
    )

    def evaluate(current_log_means, current_concentrations):
        return negative_binomial_log_logpmf(values, current_log_means, current_concentrations)

    forward = jax.jit(jax.jacfwd(evaluate, argnums=(0, 1)))(log_means, concentrations)
    reverse = jax.jit(jax.jacrev(evaluate, argnums=(0, 1)))(log_means, concentrations)

    assert jnp.allclose(forward[0], expected_log_mean_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(reverse[0], expected_log_mean_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(forward[1], expected_concentration_jacobian, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(reverse[1], expected_concentration_jacobian, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("log_mean", [-100.0, 100.0])
def test_negative_binomial_log_parameter_derivatives_remain_finite_at_extreme_means(log_mean: float) -> None:
    values = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    concentrations = np.array([0.7, 2.0, 10.0], dtype=np.float64)
    log_concentrations = np.log(concentrations)
    count_probabilities = special.expit(log_mean - log_concentrations)
    expected_log_mean_derivative = values - (values + concentrations) * count_probabilities
    expected_concentration_derivative = (
        special.digamma(values + concentrations)
        - special.digamma(concentrations)
        + special.log_expit(log_concentrations - log_mean)
        + count_probabilities
        - values * np.exp(-np.logaddexp(log_mean, log_concentrations))
    )
    values_array = jnp.asarray(values, dtype=jnp.float32)
    concentrations_array = jnp.asarray(concentrations, dtype=jnp.float32)

    def evaluate(current_log_mean, current_concentrations):
        return negative_binomial_log_logpmf(values_array, current_log_mean, current_concentrations)

    forward = jax.jacfwd(evaluate, argnums=(0, 1))(jnp.float32(log_mean), concentrations_array)
    reverse = jax.jacrev(evaluate, argnums=(0, 1))(jnp.float32(log_mean), concentrations_array)

    np.testing.assert_allclose(forward[0], expected_log_mean_derivative, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(reverse[0], expected_log_mean_derivative, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(jnp.diag(forward[1]), expected_concentration_derivative, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(jnp.diag(reverse[1]), expected_concentration_derivative, rtol=3e-6, atol=1e-6)


def test_negative_binomial_log_parameter_hessian_matches_closed_form() -> None:
    value = 4.0
    mean = 5.0
    log_mean = np.log(mean)
    concentration = 2.3
    parameter_sum = mean + concentration
    count_probability = mean / parameter_sum
    expected = np.array(
        [
            [
                -(value + concentration) * count_probability * (1 - count_probability),
                mean * (value - mean) / parameter_sum**2,
            ],
            [
                mean * (value - mean) / parameter_sum**2,
                special.polygamma(1, value + concentration)
                - special.polygamma(1, concentration)
                + 1 / concentration
                - 1 / parameter_sum
                - (mean - value) / parameter_sum**2,
            ],
        ],
        dtype=np.float32,
    )

    def evaluate(parameters):
        current_log_mean, current_concentration = parameters
        return negative_binomial_log_logpmf(value, current_log_mean, current_concentration)

    result = jax.jit(jax.hessian(evaluate))(jnp.array([log_mean, concentration], dtype=jnp.float32))

    np.testing.assert_allclose(result, expected, rtol=5e-6, atol=5e-6)
    assert jnp.allclose(result, result.T, rtol=0, atol=1e-7)


def test_negative_binomial_small_concentration_derivatives_preserve_finite_terms() -> None:
    concentration = jnp.float32(1e-10)
    expected = np.log(1e-10 / (1 + 1e-10)) + 1 / (1 + 1e-10)

    def evaluate_mean(current_concentration):
        return negative_binomial_logpmf(0, mean=1, concentration=current_concentration)

    def evaluate_log_mean(current_concentration):
        return negative_binomial_log_logpmf(0, log_mean=0, concentration=current_concentration)

    mean_forward = jax.jacfwd(evaluate_mean)(concentration)
    mean_reverse = jax.jacrev(evaluate_mean)(concentration)
    log_mean_forward = jax.jacfwd(evaluate_log_mean)(concentration)
    log_mean_reverse = jax.jacrev(evaluate_log_mean)(concentration)

    np.testing.assert_allclose(mean_forward, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(mean_reverse, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(log_mean_forward, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(log_mean_reverse, expected, rtol=2e-6, atol=2e-6)


def test_negative_binomial_tail_concentration_derivatives_retain_log_probability() -> None:
    log_mean = jnp.float32(10)
    mean = jnp.exp(log_mean)
    concentration = jnp.float32(0.01)
    expected_mean = np.log(float(concentration) / (float(mean) + float(concentration))) + float(mean) / (
        float(mean) + float(concentration)
    )
    expected_log_mean = np.log(float(concentration) / (np.exp(10) + float(concentration))) + np.exp(10) / (
        np.exp(10) + float(concentration)
    )

    def evaluate_mean(current_concentration):
        return negative_binomial_logpmf(0, mean=mean, concentration=current_concentration)

    def evaluate_log_mean(current_concentration):
        return negative_binomial_log_logpmf(0, log_mean=log_mean, concentration=current_concentration)

    mean_forward = jax.jacfwd(evaluate_mean)(concentration)
    mean_reverse = jax.jacrev(evaluate_mean)(concentration)
    log_mean_forward = jax.jacfwd(evaluate_log_mean)(concentration)
    log_mean_reverse = jax.jacrev(evaluate_log_mean)(concentration)

    np.testing.assert_allclose(mean_forward, expected_mean, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(mean_reverse, expected_mean, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(log_mean_forward, expected_log_mean, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(log_mean_reverse, expected_log_mean, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("concentration", [1e-10, 1e-6])
def test_negative_binomial_log_small_shape_derivative_avoids_pole_cancellation(
    concentration: float,
) -> None:
    log_mean = -100.0
    log_concentration = np.log(concentration)
    log_concentration_probability = special.log_expit(log_concentration - log_mean)
    count_probability = special.expit(log_mean - log_concentration)
    expected = (
        log_concentration_probability + count_probability - np.expm1(log_concentration_probability) / concentration
    )

    def evaluate(current_concentration):
        return negative_binomial_log_logpmf(1, log_mean=log_mean, concentration=current_concentration)

    forward = jax.jacfwd(evaluate)(jnp.float32(concentration))
    reverse = jax.jacrev(evaluate)(jnp.float32(concentration))

    np.testing.assert_allclose(forward, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(reverse, expected, rtol=2e-5, atol=2e-6)


def test_negative_binomial_mean_small_shape_derivative_avoids_pole_cancellation() -> None:
    mean = jnp.float32(1e-30)
    concentration = jnp.float32(1e-10)
    log_concentration_probability = np.log(float(concentration) / (float(mean) + float(concentration)))
    count_probability = float(mean) / (float(mean) + float(concentration))
    expected = (
        log_concentration_probability
        + count_probability
        - np.expm1(log_concentration_probability) / float(concentration)
    )

    def evaluate(current_concentration):
        return negative_binomial_logpmf(1, mean=mean, concentration=current_concentration)

    forward = jax.jacfwd(evaluate)(concentration)
    reverse = jax.jacrev(evaluate)(concentration)

    np.testing.assert_allclose(forward, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(reverse, expected, rtol=2e-5, atol=2e-6)


def test_negative_binomial_zero_count_preserves_mass_at_maximum_concentration() -> None:
    concentration = jnp.finfo(jnp.float32).max

    def evaluate_mean(mean):
        return negative_binomial_logpmf(0, mean, concentration)

    def evaluate_log_mean(log_mean):
        return negative_binomial_log_logpmf(0, log_mean, concentration)

    mean_result = evaluate_mean(jnp.float32(1))
    log_mean_result = evaluate_log_mean(jnp.float32(0))

    assert jnp.allclose(mean_result, -1, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(log_mean_result, -1, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jax.jacfwd(evaluate_mean)(jnp.float32(1)), -1, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jax.jacrev(evaluate_mean)(jnp.float32(1)), -1, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jax.jacfwd(evaluate_log_mean)(jnp.float32(0)), -1, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jax.jacrev(evaluate_log_mean)(jnp.float32(0)), -1, rtol=2e-6, atol=2e-6)


def test_negative_binomial_log_parameterization_preserves_poisson_limit() -> None:
    values = jnp.array([0, 1, 4, 20])
    log_means = jnp.log(jnp.array([0.2, 1.0, 4.5, 20.0]))

    result = negative_binomial_log_logpmf(values, log_means, concentration=jnp.float32(1e20))
    expected = poisson_log_logpmf(values, log_means)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=5e-6)


def test_negative_binomial_log_sums_log_masses_and_handles_empty_batches() -> None:
    values = jnp.array([[0], [1], [4]])
    log_means = jnp.array([-1.0, 0.5])
    concentrations = jnp.array([1.5, 5.0])

    pointwise = negative_binomial_log_logpmf(values, log_means, concentrations)
    result = negative_binomial_log(values, log_means, concentrations)
    empty = negative_binomial_log(jnp.empty((0,), dtype=jnp.int32), log_mean=jnp.nan, concentration=-1)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(pointwise))
    assert empty.shape == ()
    assert empty == 0


def test_negative_binomial_log_unsupported_values_have_zero_parameter_derivatives() -> None:
    def evaluate(current_log_mean, current_concentration):
        return negative_binomial_log_logpmf(-1, current_log_mean, current_concentration)

    forward = jax.jacfwd(evaluate, argnums=(0, 1))(0.5, 1.5)
    reverse = jax.jacrev(evaluate, argnums=(0, 1))(0.5, 1.5)

    assert forward == (0, 0)
    assert reverse == (0, 0)


@pytest.mark.parametrize(
    ("arguments", "name"),
    [
        ((1 + 1j, 2.0, 1.5), "value"),
        ((1, 2 + 1j, 1.5), "mean"),
        ((1, 2.0, 1.5 + 1j), "concentration"),
    ],
)
def test_negative_binomial_functions_reject_complex_arguments(arguments, name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{name}'.*real numeric dtype"):
        negative_binomial_logpmf(*arguments)


@pytest.mark.parametrize(
    ("arguments", "name"),
    [
        ((1 + 1j, 0.5, 1.5), "value"),
        ((1, 0.5 + 1j, 1.5), "log_mean"),
        ((1, 0.5, 1.5 + 1j), "concentration"),
    ],
)
def test_negative_binomial_log_functions_reject_complex_arguments(arguments, name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{name}'.*real numeric dtype"):
        negative_binomial_log_logpmf(*arguments)


def test_negative_binomial_rng_matches_gamma_poisson_mixture() -> None:
    key = jax.random.key(42)
    means = jnp.array([0.5, 3.0, 20.0], dtype=jnp.float32)
    concentrations = jnp.array([0.7, 2.5, 10.0], dtype=jnp.float32)
    gamma_key, poisson_key = jax.random.split(key)
    log_unit_rate = jax.random.loggamma(
        gamma_key,
        concentrations,
        shape=(4, 3),
        dtype=jnp.float32,
    )
    latent_rate = jnp.exp(log_unit_rate + jnp.log(means) - jnp.log(concentrations))
    expected = jax.random.poisson(
        poisson_key,
        latent_rate,
        shape=(4, 3),
        dtype=jnp.int32,
    )

    result = negative_binomial_rng(key, means, concentrations, sample_shape=(4,))

    assert result.shape == (4, 3)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)


def test_negative_binomial_log_rng_matches_mean_rng() -> None:
    key = jax.random.key(7)
    means = jnp.array([0.2, 2.0, 10.0])
    concentrations = jnp.array([0.5, 3.0, 20.0])

    result = negative_binomial_log_rng(
        key,
        jnp.log(means),
        concentrations,
        sample_shape=(8,),
    )
    expected = negative_binomial_rng(
        key,
        means,
        concentrations,
        sample_shape=(8,),
    )

    assert jnp.array_equal(result, expected)


def test_negative_binomial_rng_uses_broadcast_parameter_shape() -> None:
    means = jnp.ones((2, 1))
    concentrations = jnp.ones(3)

    result = negative_binomial_rng(
        jax.random.key(0),
        means,
        concentrations,
        sample_shape=(4, 5),
    )

    assert result.shape == (4, 5, 2, 3)


def test_negative_binomial_rng_matches_distribution_moments() -> None:
    means = jnp.array([0.5, 4.0, 20.0])
    concentrations = jnp.array([0.25, 3.0, 1_000_000.0])
    expected_variances = means + jnp.square(means) / concentrations

    samples = negative_binomial_rng(
        jax.random.key(11),
        means,
        concentrations,
        sample_shape=(50_000,),
    )

    assert jnp.allclose(jnp.mean(samples, axis=0), means, rtol=0, atol=0.08)
    assert jnp.allclose(jnp.var(samples, axis=0), expected_variances, rtol=0.03, atol=0.03)


def test_negative_binomial_rng_avoids_underflowing_gamma_scale() -> None:
    concentration = jnp.asarray(jnp.finfo(jnp.float32).max)

    samples = negative_binomial_rng(
        jax.random.key(3),
        mean=jnp.float32(1),
        concentration=concentration,
        sample_shape=(128,),
    )

    assert jnp.float32(1) / concentration == 0
    assert jnp.any(samples > 0)


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (negative_binomial_rng, 2.5),
        (negative_binomial_log_rng, jnp.log(2.5)),
    ],
)
def test_negative_binomial_rngs_can_be_jitted_and_vectorized(function, parameter) -> None:
    key = jax.random.key(21)
    concentration = 1.5
    compiled = jax.jit(
        lambda current_key, value, current_concentration: function(
            current_key,
            value,
            current_concentration,
            sample_shape=(4,),
        )
    )(key, parameter, concentration)
    eager = function(key, parameter, concentration, sample_shape=(4,))
    keys = jax.random.split(key, 3)
    vectorized = jax.vmap(lambda current_key: function(current_key, parameter, concentration))(keys)
    expected = jnp.stack([function(current_key, parameter, concentration) for current_key in keys])

    assert jnp.array_equal(compiled, eager)
    assert jnp.array_equal(vectorized, expected)


def test_negative_binomial_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        negative_binomial_rng(
            jax.random.key(0),
            jnp.ones(2),
            jnp.ones(3),
        )


@pytest.mark.parametrize(
    ("function", "arguments", "name"),
    [
        (negative_binomial_rng, (2.0 + 0.0j, 1.5), "mean"),
        (negative_binomial_rng, (2.0, 1.5 + 0.0j), "concentration"),
        (negative_binomial_log_rng, (0.5 + 0.0j, 1.5), "log_mean"),
        (negative_binomial_log_rng, (0.5, 1.5 + 0.0j), "concentration"),
    ],
)
def test_negative_binomial_rngs_reject_complex_arguments(function, arguments, name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{name}'.*real numeric dtype"):
        function(jax.random.key(0), *arguments)
