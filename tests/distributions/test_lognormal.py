"""Tests for LogNormal distribution functions."""

import jax
import jax.numpy as jnp

from mmmjax import (
    lognormal,
    lognormal_logpdf,
    lognormal_rng,
)


def test_lognormal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.25, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-2.431936110359028, -0.7255288953883893, -2.15889539821782],
        dtype=jnp.float32,
    )

    result = lognormal_logpdf(values, 0.4, 0.7)

    assert jnp.allclose(result, expected)


def test_lognormal_returns_scalar_sum() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = lognormal(values, 0.4, 0.7)

    assert result.shape == ()
    assert jnp.allclose(result, -5.316360403965237)


def test_lognormal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.5], [2.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    result = lognormal_logpdf(values, locations, 2.0)

    assert result.shape == (2, 3)
    assert jnp.allclose(lognormal(values, locations, 2.0), jnp.sum(result))


def test_lognormal_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-1.0, 0.0, jnp.inf, jnp.nan])

    result = lognormal_logpdf(values, 0.0, 1.0)

    assert jnp.all(jnp.isneginf(result[:3]))
    assert jnp.isnan(result[3])


def test_lognormal_logpdf_rejects_invalid_parameters_before_support_check() -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_scales = lognormal_logpdf(-1.0, 0.0, scales)
    invalid_locations = lognormal_logpdf(-1.0, jnp.array([jnp.inf, -jnp.inf, jnp.nan]), 1.0)

    assert jnp.all(jnp.isnan(invalid_scales))
    assert jnp.all(jnp.isnan(invalid_locations))


def test_lognormal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    half_log_two_pi = jnp.asarray(0.9189385332046727, dtype=jnp.float32)
    expected = -jnp.log(scales) - half_log_two_pi

    result = lognormal_logpdf(jnp.ones(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_lognormal_is_differentiable_with_respect_to_location() -> None:
    values = jnp.array([0.25, 1.0, 3.0])
    location = 0.4
    scale = 0.7
    expected = jnp.sum(jnp.log(values) - location) / scale**2

    result = jax.grad(lambda current_location: lognormal(values, current_location, scale))(location)

    assert jnp.allclose(result, expected)


def test_lognormal_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.25, 1.0, 3.0])
    location = 0.4
    scale = 0.7
    expected = -(1 + (jnp.log(values) - location) / scale**2) / values

    result = jax.grad(lambda current_values: lognormal(current_values, location, scale))(values)

    assert jnp.allclose(result, expected)


def test_lognormal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.5, 1.0], [1.5, 3.0]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(lognormal)(values, locations, scales)
    expected = jnp.stack(
        [lognormal(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_lognormal_rng_matches_transformed_standard_draws() -> None:
    key = jax.random.key(42)
    location = jnp.array([0.5, -1.0], dtype=jnp.float32)
    scale = jnp.array([0.25, 1.5], dtype=jnp.float32)
    expected = jnp.exp(location + scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32))

    result = lognormal_rng(key, location, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_lognormal_rng_uses_broadcast_parameter_shape() -> None:
    location = jnp.zeros((2, 1))
    scale = jnp.ones(3)

    result = lognormal_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert result.shape == (4, 2, 3)
