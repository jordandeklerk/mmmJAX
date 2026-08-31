"""Tests for Bernoulli distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import bernoulli as jax_bernoulli_distribution
from scipy import special, stats

from mmmjax import (
    bernoulli,
    bernoulli_logit,
    bernoulli_logit_logpmf,
    bernoulli_logit_rng,
    bernoulli_logpmf,
    bernoulli_rng,
)


def test_bernoulli_logpmf_matches_scipy_across_support_and_broadcasting() -> None:
    values = np.array([[-1.0], [0.0], [1.0], [2.0], [0.5], [np.nan]], dtype=np.float32)
    probabilities = np.array([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
    expected = stats.bernoulli.logpmf(values.astype(np.float64), probabilities.astype(np.float64))

    result = bernoulli_logpmf(values, probabilities)

    assert result.shape == (6, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0, equal_nan=True)


def test_bernoulli_logpmf_matches_jax_on_binary_support() -> None:
    values = jnp.array([[0], [1]])
    probabilities = jnp.array([0.0, 0.2, 0.8, 1.0])
    expected = jax_bernoulli_distribution.logpmf(values, probabilities)

    result = bernoulli_logpmf(values, probabilities)
    compiled = jax.jit(bernoulli_logpmf)(values, probabilities)

    assert jnp.allclose(result, expected)
    assert jnp.array_equal(compiled, result)


@pytest.mark.parametrize(
    ("value", "probability", "expected"),
    [
        (0, 0.0, 0.0),
        (1, 0.0, -jnp.inf),
        (0, 1.0, -jnp.inf),
        (1, 1.0, 0.0),
        (False, 0.2, -0.22314355131420976),
        (True, 0.2, -1.6094379124341003),
    ],
)
def test_bernoulli_logpmf_matches_known_values(value, probability: float, expected: float) -> None:
    result = bernoulli_logpmf(value, probability)

    assert jnp.allclose(result, expected)


def test_bernoulli_logpmf_requires_exact_binary_values() -> None:
    values = jnp.array(
        [
            -jnp.inf,
            -1.0,
            -jnp.finfo(jnp.float32).tiny,
            -0.0,
            0.5,
            1.0,
            jnp.nextafter(jnp.float32(1), jnp.inf),
            2.0,
            jnp.inf,
            jnp.nan,
        ]
    )

    result = bernoulli_logpmf(values, 0.4)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 2, 4, 6, 7, 8])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([3, 5])]))
    assert jnp.isnan(result[9])


def test_bernoulli_logpmf_rejects_invalid_probability_before_support() -> None:
    probabilities = jnp.array([-jnp.inf, -0.1, 1.1, jnp.inf, jnp.nan])

    result = bernoulli_logpmf(2, probabilities)

    assert jnp.all(jnp.isnan(result))


def test_bernoulli_sums_broadcast_log_masses() -> None:
    values = jnp.array([1, 0, 1, 1, 0])
    probability = jnp.asarray(0.3)
    expected = 3 * jnp.log(probability) + 2 * jnp.log1p(-probability)

    result = bernoulli(values, probability)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_bernoulli_empty_batch_returns_scalar_zero() -> None:
    values = jnp.empty((0,), dtype=jnp.int32)

    assert bernoulli(values, 0.4) == 0
    assert jax.jit(bernoulli)(values, -1.0) == 0


def test_bernoulli_probability_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0])
    probabilities = jnp.array([0.2, 0.8])
    expected_gradient = jnp.diag(values / probabilities - (1 - values) / (1 - probabilities))
    expected_hessian = jnp.diag(-values / jnp.square(probabilities) - (1 - values) / jnp.square(1 - probabilities))

    def evaluate(current_probabilities):
        return bernoulli_logpmf(values, current_probabilities)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(probabilities)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(probabilities)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(probabilities)

    assert jnp.allclose(forward_gradient, expected_gradient)
    assert jnp.allclose(reverse_gradient, expected_gradient)
    assert jnp.allclose(hessian, expected_hessian)


def test_bernoulli_endpoint_gradients_do_not_use_impossible_log_branches() -> None:
    values = jnp.array([0.0, 1.0])
    probabilities = jnp.array([0.0, 1.0])
    expected = jnp.diag(jnp.array([-1.0, 1.0]))

    def evaluate(current_probabilities):
        return bernoulli_logpmf(values, current_probabilities)

    assert jnp.array_equal(jax.jit(jax.jacfwd(evaluate))(probabilities), expected)
    assert jnp.array_equal(jax.jit(jax.jacrev(evaluate))(probabilities), expected)


def test_bernoulli_logit_logpmf_matches_scipy_log_expit() -> None:
    values = jnp.array([[0], [1]])
    logits = jnp.array([-jnp.inf, -1000.0, -5.0, 0.0, 5.0, 1000.0, jnp.inf])
    signed_logits = np.where(np.asarray(values) == 1, np.asarray(logits), -np.asarray(logits))
    expected = special.log_expit(signed_logits.astype(np.float64))

    result = bernoulli_logit_logpmf(values, logits)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_bernoulli_logit_logpmf_handles_invalid_values_and_logits() -> None:
    values = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0, jnp.nan])

    result = bernoulli_logit_logpmf(values, 0.5)
    invalid_logits = bernoulli_logit_logpmf(0, jnp.nan)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 2, 4])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([1, 3])]))
    assert jnp.isnan(result[5])
    assert jnp.isnan(invalid_logits)


def test_bernoulli_logit_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 1.0])
    logits = jnp.array([-3.0, 2.0])
    probabilities = jax.nn.sigmoid(logits)
    expected_gradient = jnp.diag(values - probabilities)
    expected_hessian = jnp.diag(-probabilities * (1 - probabilities))

    def evaluate(current_logits):
        return bernoulli_logit_logpmf(values, current_logits)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(logits)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(logits)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(logits)

    assert jnp.allclose(forward_gradient, expected_gradient)
    assert jnp.allclose(reverse_gradient, expected_gradient)
    assert jnp.allclose(hessian, expected_hessian)


def test_bernoulli_logit_tail_gradients_match_limiting_values() -> None:
    logits = jnp.array([-jnp.inf, jnp.inf])

    def evaluate_failure(current_logits):
        return bernoulli_logit_logpmf(0, current_logits)

    def evaluate_success(current_logits):
        return bernoulli_logit_logpmf(1, current_logits)

    expected_failure = jnp.diag(jnp.array([0.0, -1.0]))
    expected_success = jnp.diag(jnp.array([1.0, 0.0]))

    assert jnp.array_equal(jax.jacfwd(evaluate_failure)(logits), expected_failure)
    assert jnp.array_equal(jax.jacrev(evaluate_failure)(logits), expected_failure)
    assert jnp.array_equal(jax.jacfwd(evaluate_success)(logits), expected_success)
    assert jnp.array_equal(jax.jacrev(evaluate_success)(logits), expected_success)


def test_bernoulli_logit_sums_log_masses() -> None:
    values = jnp.array([0, 1, 1])
    logits = jnp.array([-2.0, 0.5, 3.0])
    signed_logits = np.where(np.asarray(values) == 1, np.asarray(logits), -np.asarray(logits))
    expected = np.sum(special.log_expit(signed_logits.astype(np.float64)))

    result = bernoulli_logit(values, logits)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_bernoulli_observations_do_not_control_parameter_dtype() -> None:
    values = jnp.array([0, 1], dtype=jnp.int64)

    assert bernoulli_logpmf(values, jnp.float32(0.4)).dtype == jnp.dtype(jnp.float32)
    assert bernoulli_logit_logpmf(values, jnp.float32(0.2)).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (bernoulli_logpmf, (0.0 + 0.0j, 0.5), "value"),
        (bernoulli_logpmf, (0, 0.5 + 0.0j), "probability"),
        (bernoulli_logit_logpmf, (0, 0.5 + 0.0j), "logits"),
    ],
)
def test_bernoulli_functions_reject_complex_arguments(function, arguments, argument_name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{argument_name}' must have a real numeric dtype, got complex"):
        function(*arguments)


def test_bernoulli_rng_matches_jax_high_mode_and_shape() -> None:
    key = jax.random.key(42)
    probabilities = jnp.array([0.2, 0.8], dtype=jnp.float32)
    expected = jax.random.bernoulli(key, probabilities, shape=(4, 2), mode="high").astype(jnp.int32)

    result = bernoulli_rng(key, probabilities, sample_shape=(4,))

    assert result.shape == (4, 2)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)


def test_bernoulli_rng_handles_deterministic_probabilities() -> None:
    probabilities = jnp.array([0.0, 1.0])

    result = bernoulli_rng(jax.random.key(0), probabilities, sample_shape=(32,))

    assert jnp.all(result[:, 0] == 0)
    assert jnp.all(result[:, 1] == 1)


def test_bernoulli_rng_matches_expected_means() -> None:
    probabilities = jnp.array([0.2, 0.8])

    samples = bernoulli_rng(jax.random.key(7), probabilities, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples, axis=0), probabilities, rtol=0, atol=0.01)


def test_bernoulli_logit_rng_matches_jax_categorical() -> None:
    key = jax.random.key(5)
    logits = jnp.array([-17.0, 0.0, 17.0])
    categorical_logits = jnp.stack((jnp.zeros_like(logits), logits), axis=-1)

    result = bernoulli_logit_rng(key, logits, sample_shape=(8,))
    expected = jax.random.categorical(
        key,
        categorical_logits,
        shape=(8, 3),
        mode="high",
    ).astype(jnp.int32)

    assert jnp.array_equal(result, expected)


def test_bernoulli_logit_rng_handles_deterministic_logits() -> None:
    logits = jnp.array([-jnp.inf, jnp.inf])

    result = bernoulli_logit_rng(jax.random.key(0), logits, sample_shape=(32,))

    assert jnp.all(result[:, 0] == 0)
    assert jnp.all(result[:, 1] == 1)
