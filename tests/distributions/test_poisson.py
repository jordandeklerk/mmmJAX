"""Tests for Poisson distribution functions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import poisson as jax_poisson_distribution
from scipy import stats

from mmmjax import (
    poisson,
    poisson_log,
    poisson_log_logpmf,
    poisson_log_rng,
    poisson_logpmf,
    poisson_rng,
)


def test_poisson_logpmf_matches_scipy_across_support_and_broadcasting() -> None:
    values = np.array([[-1.0], [0.0], [1.0], [5.0], [0.5], [np.nan]], dtype=np.float32)
    rates = np.array([0.0, 0.2, 3.0, 10.0], dtype=np.float32)
    expected = stats.poisson.logpmf(values.astype(np.float64), rates.astype(np.float64))

    result = poisson_logpmf(values, rates)

    assert result.shape == (6, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-6, equal_nan=True)


def test_poisson_logpmf_matches_jax_on_ordinary_inputs() -> None:
    values = jnp.array([[0], [1], [3], [10]])
    rates = jnp.array([0.2, 1.0, 4.5, 20.0])
    expected = jax_poisson_distribution.logpmf(values, rates)

    result = poisson_logpmf(values, rates)
    compiled = jax.jit(poisson_logpmf)(values, rates)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=2e-6)
    assert jnp.allclose(compiled, result, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("value", "rate", "expected"),
    [
        (0, 0.0, 0.0),
        (1, 0.0, -jnp.inf),
        (0, 2.0, -2.0),
        (1, 2.0, -1.3068528194400546),
        (2, 2.0, -1.3068528194400546),
    ],
)
def test_poisson_logpmf_matches_known_values(value: int, rate: float, expected: float) -> None:
    assert jnp.allclose(poisson_logpmf(value, rate), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("rate", "maximum_count"), [(0.0, 1), (0.2, 12), (3.0, 30), (20.0, 100)])
def test_poisson_probability_mass_normalizes(rate: float, maximum_count: int) -> None:
    values = jnp.arange(maximum_count + 1)
    total_probability = jnp.sum(jnp.exp(poisson_logpmf(values, rate)))

    assert jnp.allclose(total_probability, 1, rtol=0, atol=2e-6)


def test_poisson_logpmf_requires_exact_nonnegative_integer_values() -> None:
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

    result = poisson_logpmf(values, 2.0)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 2, 4, 6, 7])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([3, 5])]))
    assert jnp.isnan(result[8])


def test_poisson_logpmf_rejects_invalid_rates_before_support() -> None:
    rates = jnp.array([-jnp.inf, -1.0, jnp.inf, jnp.nan])

    result = poisson_logpmf(-1, rates)

    assert jnp.all(jnp.isnan(result))


def test_poisson_sums_broadcast_log_masses() -> None:
    values = jnp.array([0, 1, 3, 5])
    rate = jnp.asarray(2.5)
    expected = jnp.sum(poisson_logpmf(values, rate))

    result = poisson(values, rate)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_poisson_logpmf_avoids_large_count_cancellation_near_the_mode() -> None:
    values = np.array([10_000_000, 1_000_000_000, 1_000_000_000_000_000], dtype=np.float32)
    values64 = values.astype(np.float64)
    expected = -0.5 * np.log(2 * np.pi * values64) - 1 / (12 * values64) + 1 / (360 * values64**3)

    log_rates = jnp.log(values)
    log_rates_as_rates = np.asarray(jnp.exp(log_rates), dtype=np.float64)
    relative_rate_difference = (log_rates_as_rates - values64) / values64
    log_rate_expected = expected + values64 * (np.log1p(relative_rate_difference) - relative_rate_difference)

    result = poisson_logpmf(values, values)
    log_rate_result = poisson_log_logpmf(values, log_rates)

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(log_rate_result, log_rate_expected, rtol=2e-6, atol=2e-6)


def test_poisson_rate_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0, 4.0])
    rates = jnp.array([0.2, 1.5, 5.0])
    expected_gradient = jnp.diag(values / rates - 1)
    expected_hessian = jnp.diag(-values / jnp.square(rates))

    def evaluate(current_rates):
        return poisson_logpmf(values, current_rates)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(rates)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(rates)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(rates)

    assert jnp.allclose(forward_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(reverse_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(hessian, expected_hessian, rtol=2e-6, atol=2e-6)


def test_poisson_zero_rate_gradients_match_limiting_values() -> None:
    def zero_count(rate):
        return poisson_logpmf(0, rate)

    def positive_count(rate):
        return poisson_logpmf(2, rate)

    assert jax.jacfwd(zero_count)(0.0) == -1
    assert jax.jacrev(zero_count)(0.0) == -1
    assert jnp.isposinf(jax.jacfwd(positive_count)(0.0))
    assert jnp.isposinf(jax.jacrev(positive_count)(0.0))


@pytest.mark.parametrize(
    ("value", "rate", "expected"),
    [(jnp.float32(1e37), jnp.float32(1e38), -0.9), (jnp.float32(1e38), jnp.float32(1e38), 0.0)],
)
def test_poisson_large_rate_forward_and_reverse_gradients_agree(value, rate, expected: float) -> None:
    def evaluate(current_rate):
        return poisson_logpmf(value, current_rate)

    assert jnp.allclose(jax.jacfwd(evaluate)(rate), expected, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(jax.jacrev(evaluate)(rate), expected, rtol=2e-6, atol=2e-6)


def test_poisson_log_parameterization_matches_rate_parameterization() -> None:
    values = jnp.array([[0], [1], [3], [10]])
    rates = jnp.array([0.1, 1.0, 4.5, 20.0])

    result = poisson_log_logpmf(values, jnp.log(rates))
    expected = poisson_logpmf(values, rates)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=2e-6)


def test_poisson_log_parameterization_matches_scipy() -> None:
    values = np.array([[0], [1], [4], [12]], dtype=np.float32)
    log_rates = np.array([-20.0, -2.0, 0.5, 3.0], dtype=np.float32)
    expected = stats.poisson.logpmf(
        values.astype(np.float64),
        np.exp(log_rates.astype(np.float64)),
    )

    result = poisson_log_logpmf(values, log_rates)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-6)


def test_poisson_log_parameterization_handles_infinite_limits() -> None:
    values = jnp.array([[0], [1], [2]])
    log_rates = jnp.array([-jnp.inf, 0.0, jnp.inf])

    result = poisson_log_logpmf(values, log_rates)

    assert result[0, 0] == 0
    assert jnp.all(jnp.isneginf(result[1:, 0]))
    assert jnp.allclose(result[:, 1], poisson_logpmf(values[:, 0], 1.0), rtol=2e-6, atol=2e-6)
    assert jnp.all(jnp.isneginf(result[:, 2]))
    assert jnp.isnan(poisson_log_logpmf(0, jnp.nan))


def test_poisson_log_parameterization_retains_finite_underflowed_rates() -> None:
    log_rate = jnp.asarray(-100.0, dtype=jnp.float32)
    expected = log_rate - jnp.exp(log_rate)

    result = poisson_log_logpmf(1, log_rate)

    assert jnp.allclose(result, expected, rtol=0, atol=1e-6)


def test_poisson_log_parameterization_handles_overflowed_rates_without_nan() -> None:
    values = jnp.array([1e37, 1e38, 2e38], dtype=jnp.float32)
    log_rate = jnp.float32(89)

    result = poisson_log_logpmf(values, log_rate)
    expected_gradient = -values[2] * jnp.expm1(log_rate - jnp.log(values[2]))

    def evaluate(current_log_rate):
        return poisson_log_logpmf(values[2], current_log_rate)

    assert jnp.isneginf(result[0])
    assert jnp.all(jnp.isfinite(result[1:]))
    assert jnp.allclose(jax.jacfwd(evaluate)(log_rate), expected_gradient, rtol=2e-6)
    assert jnp.allclose(jax.jacrev(evaluate)(log_rate), expected_gradient, rtol=2e-6)


def test_poisson_log_rate_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0, 4.0])
    log_rates = jnp.array([-2.0, 0.5, 2.0])
    rates = jnp.exp(log_rates)
    expected_gradient = jnp.diag(values - rates)
    expected_hessian = jnp.diag(-rates)

    def evaluate(current_log_rates):
        return poisson_log_logpmf(values, current_log_rates)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(log_rates)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(log_rates)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(log_rates)

    assert jnp.allclose(forward_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(reverse_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(hessian, expected_hessian, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (2, 2.0)])
def test_poisson_log_negative_infinity_gradient_matches_limit(value: int, expected: float) -> None:
    def evaluate(log_rate):
        return poisson_log_logpmf(value, log_rate)

    assert jax.jacfwd(evaluate)(-jnp.inf) == expected
    assert jax.jacrev(evaluate)(-jnp.inf) == expected


def test_poisson_log_sums_log_masses() -> None:
    values = jnp.array([0, 1, 3, 5])
    log_rate = jnp.asarray(0.4)

    result = poisson_log(values, log_rate)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(poisson_log_logpmf(values, log_rate)))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_poisson_observations_do_not_control_parameter_dtype() -> None:
    values = jnp.array([0, 1], dtype=jnp.int64)

    assert poisson_logpmf(values, jnp.float32(2.0)).dtype == jnp.dtype(jnp.float32)
    assert poisson_log_logpmf(values, jnp.float32(0.5)).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (poisson_logpmf, (0.0 + 0.0j, 2.0), "value"),
        (poisson_logpmf, (0, 2.0 + 0.0j), "rate"),
        (poisson_log_logpmf, (0, 0.5 + 0.0j), "log_rate"),
    ],
)
def test_poisson_functions_reject_complex_arguments(function, arguments, argument_name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{argument_name}' must have a real numeric dtype, got complex"):
        function(*arguments)


def test_poisson_rng_matches_jax_and_uses_integer_output() -> None:
    key = jax.random.key(42)
    rates = jnp.array([0.0, 0.5, 8.0], dtype=jnp.float32)
    expected = jax.random.poisson(key, rates, shape=(4, 3), dtype=jnp.int32)

    result = poisson_rng(key, rates, sample_shape=(4,))

    assert result.shape == (4, 3)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)


def test_poisson_log_rng_matches_rate_rng() -> None:
    key = jax.random.key(7)
    rates = jnp.array([0.2, 2.0, 10.0])

    assert jnp.array_equal(
        poisson_log_rng(key, jnp.log(rates), sample_shape=(8,)),
        poisson_rng(key, rates, sample_shape=(8,)),
    )


def test_poisson_rng_matches_expected_moments() -> None:
    samples = poisson_rng(jax.random.key(11), 2.5, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 2.5, rtol=0, atol=0.04)
    assert jnp.allclose(jnp.var(samples), 2.5, rtol=0, atol=0.08)


@pytest.mark.parametrize(
    ("function", "parameter"),
    [(poisson_rng, 2.5), (poisson_log_rng, jnp.log(2.5))],
)
def test_poisson_rngs_can_be_jitted_and_vectorized(function, parameter) -> None:
    key = jax.random.key(0)
    compiled = jax.jit(partial(function, sample_shape=(4,)))(key, parameter)
    eager = function(key, parameter, sample_shape=(4,))
    keys = jax.random.split(key, 3)
    vectorized = jax.vmap(lambda current_key: function(current_key, parameter))(keys)

    assert jnp.array_equal(compiled, eager)
    assert vectorized.shape == (3,)
