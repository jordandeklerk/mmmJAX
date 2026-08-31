"""Tests for Negative Binomial distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import nbinom as jax_negative_binomial_distribution
from scipy import stats

from mmmjax.distributions._negative_binomial import (
    negative_binomial,
    negative_binomial_logpmf,
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
