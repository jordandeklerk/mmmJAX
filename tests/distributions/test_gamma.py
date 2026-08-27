"""Tests for Gamma distribution functions."""

import jax
import jax.numpy as jnp
import pytest

from mmmjax import (
    exponential_logpdf,
    gamma,
    gamma_logpdf,
    gamma_rng,
)


def test_gamma_logpdf_matches_known_values() -> None:
    values = jnp.array([0.25, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-1.4625537844973295, -0.6581122428174937, -2.410193809815329],
        dtype=jnp.float32,
    )

    result = gamma_logpdf(values, 2.5, 1.7)

    assert jnp.allclose(result, expected)


def test_gamma_returns_scalar_sum() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = gamma(values, 2.5, 1.7)

    assert result.shape == ()
    assert jnp.allclose(result, -4.530859837130152)


def test_gamma_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.25], [2.0]])
    shapes = jnp.array([0.5, 1.0, 3.0])
    rates = jnp.array([0.5, 1.0, 2.0])
    expected = jnp.array(
        [
            [-0.3507913526447274, -0.25, -1.8862943611198904],
            [-2.2655121234846454, -2.0, -1.2274112777602189],
        ]
    )

    result = gamma_logpdf(values, shapes, rates)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected)
    assert jnp.allclose(gamma(values, shapes, rates), jnp.sum(expected))


def test_gamma_logpdf_uses_zero_boundary_limits() -> None:
    values = jnp.array([[0.0], [-0.0]])
    shapes = jnp.array([0.5, 1.0, 2.0])
    rate = 1.7

    result = gamma_logpdf(values, shapes, rate)

    assert result.shape == (2, 3)
    assert jnp.all(jnp.isposinf(result[:, 0]))
    assert jnp.allclose(result[:, 1], jnp.log(rate))
    assert jnp.all(jnp.isneginf(result[:, 2]))


def test_gamma_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-jnp.inf, -1.0, jnp.inf, jnp.nan])

    result = gamma_logpdf(values, 2.5, 1.7)

    assert jnp.all(jnp.isneginf(result[:3]))
    assert jnp.isnan(result[3])


def test_gamma_logpdf_rejects_invalid_parameters_before_support_check() -> None:
    invalid_parameters = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_shapes = gamma_logpdf(-1.0, invalid_parameters, 1.0)
    invalid_rates = gamma_logpdf(-1.0, 1.0, invalid_parameters)

    assert jnp.all(jnp.isnan(invalid_shapes))
    assert jnp.all(jnp.isnan(invalid_rates))


def test_gamma_logpdf_handles_extreme_valid_parameters() -> None:
    rates = jnp.array([1e-30, jnp.finfo(jnp.float32).max], dtype=jnp.float32)
    tiny_shape = jnp.asarray(jnp.finfo(jnp.float32).tiny)

    rate_result = gamma_logpdf(jnp.zeros(2, dtype=jnp.float32), 1.0, rates)
    shape_result = gamma_logpdf(0.0, tiny_shape, 1.0)
    interior_result = gamma_logpdf(jnp.float32(1e-30), 2.0, jnp.float32(1e-30))

    assert jnp.all(jnp.isfinite(rate_result))
    assert jnp.allclose(rate_result, jnp.log(rates))
    assert jnp.isposinf(shape_result)
    assert jnp.isfinite(interior_result)
    assert jnp.allclose(interior_result, -207.2326583694641)


def test_gamma_with_unit_shape_matches_exponential() -> None:
    values = jnp.array([0.0, 0.25, 1.5, jnp.inf])
    rate = 1.7

    result = gamma_logpdf(values, 1.0, rate)

    assert jnp.allclose(result, exponential_logpdf(values, rate))


def test_gamma_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.25, 1.0, 3.0])
    shape = 2.5
    rate = 1.7
    expected = jnp.array([4.3, -0.2, -1.2])

    result = jax.grad(lambda current_values: gamma(current_values, shape, rate))(values)

    assert jnp.allclose(result, expected)


def test_gamma_is_differentiable_with_respect_to_shape() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = jax.grad(lambda current_shape: gamma(values, current_shape, 1.7))(2.5)

    assert jnp.allclose(result, -0.8052672412009992)


def test_gamma_is_differentiable_with_respect_to_rate() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = jax.grad(lambda current_rate: gamma(values, 2.5, current_rate))(1.7)

    assert jnp.allclose(result, 0.16176470588235325)


def test_gamma_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.25, 1.0], [0.5, 2.0]])
    shapes = jnp.array([1.5, 3.0])
    rates = jnp.array([0.5, 2.0])

    result = jax.vmap(gamma)(values, shapes, rates)
    expected = jnp.stack([gamma(value, shape, rate) for value, shape, rate in zip(values, shapes, rates, strict=True)])

    assert jnp.allclose(result, expected)


def test_gamma_rng_scales_log_space_unit_rate_draws() -> None:
    key = jax.random.key(42)
    shapes = jnp.array([0.5, 2.5], dtype=jnp.float32)
    rates = jnp.array([1.7, 0.8], dtype=jnp.float32)
    expected = jnp.exp(jax.random.loggamma(key, shapes, shape=(3, 2), dtype=jnp.float32) - jnp.log(rates))

    result = gamma_rng(key, shapes, rates, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)


def test_gamma_rng_uses_broadcast_parameter_shape() -> None:
    shapes = jnp.ones((2, 1))
    rates = jnp.ones(3)

    result = gamma_rng(jax.random.key(0), shapes, rates, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_gamma_rng_matches_distribution_moments() -> None:
    samples = gamma_rng(jax.random.key(7), 4.0, 2.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 2.0, rtol=0, atol=0.03)
    assert jnp.allclose(jnp.var(samples), 1.0, rtol=0, atol=0.06)


def test_gamma_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        gamma_rng(jax.random.key(0), jnp.ones(2), jnp.ones(3))
