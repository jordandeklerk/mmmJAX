"""Tests for Laplace distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import laplace, laplace_logpdf, laplace_rng


@pytest.mark.parametrize(
    ("value", "location", "scale", "expected"),
    [
        (1.0, 1.0, 1.0, -0.6931471805599453),
        (2.0, 1.0, 1.0, -1.6931471805599453),
        (-3.0, 2.0, 1.0, -5.6931471805599453),
        (1.0, 0.0, 2.0, -1.8862943611198906),
        (1.9, 2.3, 0.5, -0.8),
        (1.9, 2.3, 0.25, -0.9068528194400547),
    ],
)
def test_laplace_logpdf_matches_stan_reference_values(
    value: float,
    location: float,
    scale: float,
    expected: float,
) -> None:
    result = laplace_logpdf(value, location, scale)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_laplace_logpdf_matches_scipy_reference_grid() -> None:
    values = np.array([[-5.0], [0.0], [4.0]], dtype=np.float32)
    locations = np.array([-2.0, 0.5, 2.3], dtype=np.float32)
    scales = np.array([0.25, 0.5, 3.0], dtype=np.float32)
    expected = stats.laplace.logpdf(
        values.astype(np.float64),
        loc=locations.astype(np.float64),
        scale=scales.astype(np.float64),
    )

    result = laplace_logpdf(values, locations, scales)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    assert jnp.allclose(laplace(values, locations, scales), jnp.sum(result))


def test_laplace_logpdf_rejects_invalid_parameters() -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, -jnp.inf, jnp.nan])
    locations = jnp.array([jnp.inf, -jnp.inf, jnp.nan])

    invalid_scales = laplace_logpdf(0.0, 0.0, scales)
    invalid_locations = laplace_logpdf(0.0, locations, 1.0)

    assert jnp.all(jnp.isnan(invalid_scales))
    assert jnp.all(jnp.isnan(invalid_locations))


def test_laplace_logpdf_handles_nonfinite_values() -> None:
    values = jnp.array([jnp.inf, -jnp.inf, jnp.nan])

    result = laplace_logpdf(values, 0.0, 1.0)

    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.isnan(result[2])


def test_laplace_logpdf_remains_finite_at_extreme_valid_scales() -> None:
    dtype_info = jnp.finfo(jnp.float32)
    scales = jnp.array([dtype_info.tiny, dtype_info.max], dtype=jnp.float32)
    expected = -jnp.log(scales) - jnp.asarray(np.log(2), dtype=jnp.float32)

    result = laplace_logpdf(jnp.zeros(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_laplace_logpdf_handles_opposite_values_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    expected = -jnp.log(maximum) - jnp.asarray(np.log(2), dtype=jnp.float32) - 2

    result = jax.jit(laplace_logpdf)(maximum, -maximum, maximum)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, expected)


def test_laplace_logpdf_keeps_far_tail_in_log_space() -> None:
    result = laplace_logpdf(jnp.float32(1_000), 0.0, 1.0)

    assert jnp.allclose(result, -1000.6931471805599)


def test_laplace_logpdf_gradients_match_analytic_values() -> None:
    values = jnp.array([-3.0, 0.0, 4.0])
    locations = jnp.array([-1.0, 1.0, 2.0])
    scales = jnp.array([0.5, 2.0, 4.0])
    residuals = values - locations
    expected = jnp.stack(
        [
            -jnp.sign(residuals) / scales,
            jnp.sign(residuals) / scales,
            jnp.abs(residuals) / jnp.square(scales) - 1 / scales,
        ],
        axis=-1,
    )

    gradients = jax.vmap(jax.grad(laplace_logpdf, argnums=(0, 1, 2)))(values, locations, scales)
    result = jnp.stack(gradients, axis=-1)

    assert jnp.allclose(result, expected)


def test_laplace_logpdf_uses_stan_subgradient_at_location() -> None:
    compiled_gradient = jax.jit(jax.grad(laplace_logpdf, argnums=(0, 1, 2)))

    reverse_gradients = compiled_gradient(1.5, 1.5, 2.0)
    _, forward_tangent = jax.jvp(
        laplace_logpdf,
        (1.5, 1.5, 2.0),
        (1.0, -1.0, 0.25),
    )

    assert jnp.array_equal(jnp.asarray(reverse_gradients), jnp.array([0.0, 0.0, -0.5]))
    assert jnp.allclose(forward_tangent, -0.125)


def test_laplace_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-2.0, 0.0], [1.0, 4.0]])
    locations = jnp.array([-0.5, 2.0])
    scales = jnp.array([0.75, 1.5])

    result = jax.vmap(laplace)(values, locations, scales)
    expected = jnp.stack(
        [laplace(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_laplace_rng_matches_transformed_jax_draws() -> None:
    key = jax.random.key(42)
    locations = jnp.array([[1.0], [-2.0]], dtype=jnp.float32)
    scales = jnp.array([0.5, 2.0, 1.5], dtype=jnp.float32)
    expected = locations + scales * jax.random.laplace(key, shape=(4, 2, 3), dtype=jnp.float32)

    result = laplace_rng(key, locations, scales, sample_shape=(4,))

    assert result.shape == (4, 2, 3)
    assert jnp.array_equal(result, expected)


def test_laplace_rng_matches_distribution_moments() -> None:
    location = 0.5
    scale = 1.25

    samples = laplace_rng(jax.random.key(7), location, scale, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), location, rtol=0, atol=0.03)
    assert jnp.allclose(jnp.var(samples), 2 * scale**2, rtol=0, atol=0.08)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_laplace_logpdf_handles_float64_finite_limits() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float64).max)
    log_two = jnp.asarray(np.log(2), dtype=jnp.float64)

    at_location = laplace_logpdf(jnp.float64(0), jnp.float64(0), maximum)
    across_zero = laplace_logpdf(maximum, -maximum, maximum)

    assert jnp.allclose(at_location, -jnp.log(maximum) - log_two, rtol=1e-14, atol=0)
    assert jnp.allclose(across_zero, -jnp.log(maximum) - log_two - 2, rtol=1e-14, atol=0)


def test_laplace_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        laplace_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))
