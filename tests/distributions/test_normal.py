"""Tests for Normal distribution functions."""

import jax
import jax.numpy as jnp
import pytest

from mmmjax import (
    normal,
    normal_logpdf,
    normal_rng,
)


def test_normal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 1.0, -2.0], dtype=jnp.float32)
    expected = jnp.array(
        [-0.9189385332046727, -1.4189385332046727, -2.9189385332046727],
        dtype=jnp.float32,
    )

    result = normal_logpdf(values, 0.0, 1.0)

    assert jnp.allclose(result, expected)


def test_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 1.0, -2.0])

    result = normal(values, 0.0, 1.0)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(normal_logpdf(values, 0.0, 1.0)))


def test_normal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    result = normal_logpdf(values, locations, 2.0)

    assert result.shape == (2, 3)
    assert jnp.allclose(normal(values, locations, 2.0), jnp.sum(result))


def test_normal_logpdf_rejects_invalid_parameters_without_repairing_them() -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_scales = normal_logpdf(0.0, 0.0, scales)
    invalid_locations = normal_logpdf(0.0, jnp.array([jnp.inf, -jnp.inf, jnp.nan]), 1.0)

    assert jnp.all(jnp.isnan(invalid_scales))
    assert jnp.all(jnp.isnan(invalid_locations))


def test_normal_logpdf_handles_nonfinite_values() -> None:
    values = jnp.array([jnp.inf, -jnp.inf, jnp.nan])

    result = normal_logpdf(values, 0.0, 1.0)

    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.isnan(result[2])


def test_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    half_log_two_pi = jnp.asarray(0.9189385332046727, dtype=jnp.float32)
    expected = -jnp.log(scales) - half_log_two_pi

    result = normal_logpdf(jnp.zeros(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_normal_is_differentiable_with_respect_to_location() -> None:
    values = jnp.array([0.0, 1.0, 2.0])
    location = 0.5
    scale = 2.0
    expected = jnp.sum(values - location) / scale**2

    result = jax.grad(lambda current_location: normal(values, current_location, scale))(location)

    assert jnp.allclose(result, expected)


def test_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(normal)(values, locations, scales)
    expected = jnp.stack(
        [normal(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_normal_rng_matches_transformed_standard_draws() -> None:
    key = jax.random.key(42)
    location = jnp.array([1.0, -2.0], dtype=jnp.float32)
    scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = location + scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32)

    result = normal_rng(key, location, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_normal_rng_uses_broadcast_parameter_shape() -> None:
    location = jnp.zeros((2, 1))
    scale = jnp.ones(3)

    result = normal_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_normal_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        normal_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))
