"""Tests for Exponential distribution functions."""

import jax
import jax.numpy as jnp

from mmmjax import (
    exponential,
    exponential_logpdf,
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
