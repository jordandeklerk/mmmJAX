"""Tests for Exponential distribution functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmmjax import (
    exponential,
    exponential_logcdf,
    exponential_logpdf,
    exponential_logsf,
    exponential_rng,
)


def test_exponential_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [0.6931471805599453, -0.3068528194400547, -3.3068528194400546],
        dtype=jnp.float32,
    )

    result = exponential_logpdf(values, 2.0)

    assert jnp.allclose(result, expected)


def test_exponential_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 0.5, 2.0])

    result = exponential(values, 2.0)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(exponential_logpdf(values, 2.0)))


def test_exponential_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    rates = jnp.array([0.5, 1.0, 2.0])

    result = exponential_logpdf(values, rates)

    assert result.shape == (2, 3)
    assert jnp.allclose(exponential(values, rates), jnp.sum(result))


def test_exponential_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-1.0, 0.0, jnp.inf, jnp.nan])

    result = exponential_logpdf(values, 2.0)

    assert jnp.isneginf(result[0])
    assert jnp.allclose(result[1], jnp.log(2.0))
    assert jnp.isneginf(result[2])
    assert jnp.isnan(result[3])


def test_exponential_logpdf_rejects_invalid_rate_before_support_check() -> None:
    rates = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    result = exponential_logpdf(-1.0, rates)

    assert jnp.all(jnp.isnan(result))


def test_exponential_log_probabilities_match_known_values() -> None:
    rate = 2.0
    median = math.log(2) / rate
    values = jnp.array([median, 1.0])
    expected_log_cdf = jnp.array([-math.log(2), math.log1p(-math.exp(-2))])
    expected_log_survival = jnp.array([-math.log(2), -2.0])

    assert jnp.allclose(exponential_logcdf(values, rate), expected_log_cdf)
    assert jnp.allclose(exponential_logsf(values, rate), expected_log_survival)


def test_exponential_logcdf_remains_finite_near_zero() -> None:
    value = jnp.asarray(1e-10, dtype=jnp.float32)
    rate = jnp.asarray(2.3, dtype=jnp.float32)
    expected = np.log(-np.expm1(-float(rate) * float(value)))

    result = exponential_logcdf(value, rate)

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=3e-6)


def test_exponential_log_probabilities_are_complements() -> None:
    values = jnp.array([1e-10, 0.1, 1.0, 20.0, 80.0])
    rates = jnp.array([2.3, 0.5, 1.0, 2.0, 1.0])

    log_cdf = exponential_logcdf(values, rates)
    log_survival = exponential_logsf(values, rates)

    assert jnp.allclose(jnp.logaddexp(log_cdf, log_survival), 0.0, atol=1e-6)


def test_exponential_log_probabilities_enforce_support_and_endpoints() -> None:
    values = jnp.array([-jnp.inf, -1.0, -0.0, 0.0, jnp.inf, jnp.nan])

    log_cdf = exponential_logcdf(values, 2.0)
    log_survival = exponential_logsf(values, 2.0)

    assert jnp.all(jnp.isneginf(log_cdf[:4]))
    assert log_cdf[4] == 0
    assert jnp.isnan(log_cdf[5])
    assert jnp.all(log_survival[:4] == 0)
    assert jnp.isneginf(log_survival[4])
    assert jnp.isnan(log_survival[5])


@pytest.mark.parametrize("function", [exponential_logcdf, exponential_logsf])
def test_exponential_log_probabilities_reject_invalid_rate_before_support(function) -> None:
    rates = jnp.array([0.0, -1.0, jnp.inf, -jnp.inf, jnp.nan])

    result = function(-1.0, rates)

    assert jnp.all(jnp.isnan(result))


def test_exponential_log_probabilities_broadcast_arguments() -> None:
    values = jnp.array([[0.1], [1.0]])
    rates = jnp.array([0.5, 1.0, 2.0])

    log_cdf = exponential_logcdf(values, rates)
    log_survival = exponential_logsf(values, rates)

    assert log_cdf.shape == (2, 3)
    assert log_survival.shape == (2, 3)
    assert jnp.allclose(jnp.logaddexp(log_cdf, log_survival), 0.0, atol=1e-6)


def test_exponential_logcdf_derivatives_match_closed_form() -> None:
    arguments = jnp.array([0.75, 1.3])
    value, rate = map(float, arguments)
    scaled_value = rate * value
    denominator = math.expm1(scaled_value)
    second_derivative = -math.exp(scaled_value) / denominator**2
    expected_gradient = jnp.array([rate / denominator, value / denominator])
    expected_hessian = jnp.array(
        [
            [rate**2 * second_derivative, 1 / denominator + scaled_value * second_derivative],
            [1 / denominator + scaled_value * second_derivative, value**2 * second_derivative],
        ]
    )

    def evaluate(current):
        return exponential_logcdf(current[0], current[1])

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)
    hessian = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.allclose(forward, expected_gradient)
    assert jnp.allclose(reverse, expected_gradient)
    assert jnp.allclose(hessian, expected_hessian)


def test_exponential_logsf_derivatives_match_closed_form() -> None:
    arguments = jnp.array([0.75, 1.3])
    expected_gradient = jnp.array([-1.3, -0.75])
    expected_hessian = jnp.array([[0.0, -1.0], [-1.0, 0.0]])

    def evaluate(current):
        return exponential_logsf(current[0], current[1])

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)
    hessian = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.array_equal(forward, expected_gradient)
    assert jnp.array_equal(reverse, expected_gradient)
    assert jnp.array_equal(hessian, expected_hessian)


def test_exponential_logcdf_preserves_far_tail_curvature() -> None:
    value = 80.0
    expected = -math.exp(-value) / (1 - math.exp(-value)) ** 2

    result = jax.grad(jax.grad(lambda current: exponential_logcdf(current, 1.0)))(value)

    assert result != 0
    np.testing.assert_allclose(result, expected, rtol=3e-6)


@pytest.mark.parametrize("function", [exponential_logcdf, exponential_logsf])
@pytest.mark.parametrize("value", [-1.0, 0.0, jnp.inf])
def test_exponential_log_probabilities_have_zero_rate_gradients_at_boundaries(function, value) -> None:
    differentiate = jax.grad(lambda current: function(value, current))

    result = differentiate(2.0)
    compiled_result = jax.jit(differentiate)(2.0)

    assert result == 0
    assert compiled_result == 0


def test_exponential_is_differentiable_with_respect_to_rate() -> None:
    values = jnp.array([0.25, 1.5])
    rate = 2.0
    expected = values.size / rate - jnp.sum(values)

    result = jax.grad(lambda current_rate: exponential(values, current_rate))(rate)

    assert jnp.allclose(result, expected)


def test_exponential_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [0.5, 2.0]])
    rates = jnp.array([1.0, 2.0])

    result = jax.vmap(exponential)(values, rates)
    expected = jnp.stack([exponential(value, rate) for value, rate in zip(values, rates, strict=True)])

    assert jnp.allclose(result, expected)


def test_exponential_rng_matches_rate_scaled_standard_draws() -> None:
    key = jax.random.key(42)
    rate = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = jax.random.exponential(key, shape=(3, 2), dtype=jnp.float32) / rate

    result = exponential_rng(key, rate, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
