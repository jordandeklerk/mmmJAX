"""Tests for HalfNormal distribution functions."""

import jax
import jax.numpy as jnp

from mmmjax import (
    half_normal,
    half_normal_logpdf,
    half_normal_rng,
)


def test_half_normal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [-0.6312564607528918, -0.6868120163084473, -1.5201453496417807],
        dtype=jnp.float32,
    )

    result = half_normal_logpdf(values, 1.5)

    assert jnp.allclose(result, expected)


def test_half_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 0.5, 2.0])

    result = half_normal(values, 1.5)

    assert result.shape == ()
    assert jnp.allclose(result, -2.83821382670312)


def test_half_normal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    scales = jnp.array([0.5, 1.0, 2.0])
    expected = jnp.array(
        [
            [0.4673558279152179, -0.2257913526447274, -0.9189385332046727],
            [-1.5326441720847819, -0.7257913526447274, -1.0439385332046727],
        ]
    )

    result = half_normal_logpdf(values, scales)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected)
    assert jnp.allclose(half_normal(values, scales), jnp.sum(expected))


def test_half_normal_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-jnp.inf, -1.0, -0.0, 0.0, jnp.inf, jnp.nan])

    result = half_normal_logpdf(values, 1.0)

    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.allclose(result[2:4], -0.2257913526447274)
    assert jnp.isneginf(result[4])
    assert jnp.isnan(result[5])


def test_half_normal_logpdf_rejects_invalid_scale_before_support_check() -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    result = half_normal_logpdf(-1.0, scales)

    assert jnp.all(jnp.isnan(result))


def test_half_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    expected = jnp.asarray(-0.2257913526447274, dtype=jnp.float32) - jnp.log(scales)

    result = half_normal_logpdf(jnp.zeros(2, dtype=jnp.float32), scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_half_normal_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.25, 1.0, 2.0])
    scale = 1.5
    expected = -values / scale**2

    result = jax.grad(lambda current_values: half_normal(current_values, scale))(values)

    assert jnp.allclose(result, expected)


def test_half_normal_is_differentiable_with_respect_to_scale() -> None:
    values = jnp.array([0.25, 1.0, 2.0])
    scale = 1.5
    expected = -values.size / scale + jnp.sum(jnp.square(values)) / scale**3

    result = jax.grad(lambda current_scale: half_normal(values, current_scale))(scale)

    assert jnp.allclose(result, expected)


def test_half_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [0.5, 2.0]])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(half_normal)(values, scales)
    expected = jnp.stack([half_normal(value, scale) for value, scale in zip(values, scales, strict=True)])

    assert jnp.allclose(result, expected)


def test_half_normal_rng_folds_standard_normal_draws() -> None:
    key = jax.random.key(42)
    scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = jnp.abs(scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32))

    result = half_normal_rng(key, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)
