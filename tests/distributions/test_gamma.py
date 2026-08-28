"""Tests for Gamma distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

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


def test_gamma_logpdf_matches_scipy_reference_grid() -> None:
    values = np.array([1e-30, 0.01, 0.1, 0.75, 1.0, 4.0, 20.0, 1e20], dtype=np.float32)
    shapes = np.array([0.1, 0.5, 1.0, 2.5, 7.999, 8.0, 50.0, 3.0], dtype=np.float32)
    rates = np.array([1e-10, 0.1, 1.0, 1.7, 8.0, 25.0, 0.001, 1e-10], dtype=np.float32)
    expected = stats.gamma.logpdf(
        values.astype(np.float64),
        a=shapes.astype(np.float64),
        scale=1 / rates.astype(np.float64),
    )

    result = gamma_logpdf(values, shapes, rates)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_gamma_logpdf_gradients_match_analytic_reference_grid() -> None:
    values = np.array([1e-4, 0.2, 1.0, 3.0, 100.0], dtype=np.float32)
    shapes = np.array([0.2, 1.0, 2.5, 8.0, 50.0], dtype=np.float32)
    rates = np.array([1e-3, 0.7, 2.0, 8.0, 1.0], dtype=np.float32)
    expected = np.stack(
        [
            (shapes - 1) / values - rates,
            np.log(rates) + np.log(values) - special.digamma(shapes),
            shapes / rates - values,
        ],
        axis=-1,
    )

    gradients = jax.vmap(jax.grad(gamma_logpdf, argnums=(0, 1, 2)))(values, shapes, rates)
    result = np.stack(gradients, axis=-1)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


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


def test_gamma_homogeneous_ordinary_batch_matches_independent_references() -> None:
    values = jnp.linspace(0.05, 2.0, 12, dtype=jnp.float32).reshape(4, 3)
    shapes = jnp.array([0.2, 2.5, 7.5], dtype=jnp.float32)
    rates = jnp.array([0.3, 1.7, 5.0], dtype=jnp.float32)
    value_tangents = jnp.linspace(-0.2, 0.2, 12, dtype=jnp.float32).reshape(4, 3)
    shape_tangents = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    rate_tangents = jnp.array([-0.3, 0.2, 0.1], dtype=jnp.float32)

    values_reference = np.asarray(values, dtype=np.float64)
    shapes_reference = np.asarray(shapes, dtype=np.float64)
    rates_reference = np.asarray(rates, dtype=np.float64)
    expected_logpdf = stats.gamma.logpdf(
        values_reference,
        a=shapes_reference,
        scale=1 / rates_reference,
    )
    value_derivatives = (shapes_reference - 1) / values_reference - rates_reference
    shape_derivatives = np.log(rates_reference) + np.log(values_reference) - special.digamma(shapes_reference)
    rate_derivatives = shapes_reference / rates_reference - values_reference
    expected_parameter_gradients = (
        np.sum(shape_derivatives, axis=0),
        np.sum(rate_derivatives, axis=0),
    )
    expected_tangent = (
        value_derivatives * np.asarray(value_tangents)
        + shape_derivatives * np.asarray(shape_tangents)
        + rate_derivatives * np.asarray(rate_tangents)
    )

    result = jax.jit(gamma_logpdf)(values, shapes, rates)
    density, parameter_gradients = jax.jit(jax.value_and_grad(gamma, argnums=(1, 2)))(values, shapes, rates)
    _, tangent = jax.jvp(
        gamma_logpdf,
        (values, shapes, rates),
        (value_tangents, shape_tangents, rate_tangents),
    )

    np.testing.assert_allclose(result, expected_logpdf, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(density, np.sum(expected_logpdf), rtol=3e-6, atol=3e-6)
    for gradient, expected_gradient in zip(parameter_gradients, expected_parameter_gradients, strict=True):
        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-6, atol=3e-6)


def test_gamma_ordinary_shapes_handle_extreme_value_rate_pairs() -> None:
    smallest_normal = np.finfo(np.float32).tiny
    largest = np.finfo(np.float32).max
    values = np.array([smallest_normal, largest, smallest_normal, largest], dtype=np.float32)
    rates = np.array([largest, smallest_normal, smallest_normal, largest], dtype=np.float32)
    shape = np.float32(2.5)
    expected = stats.gamma.logpdf(
        values.astype(np.float64),
        a=np.float64(shape),
        scale=1 / rates.astype(np.float64),
    )

    result = gamma_logpdf(values, shape, rates)

    np.testing.assert_allclose(result[:3], expected[:3], rtol=3e-6, atol=3e-6)
    assert jnp.isneginf(result[3])


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


def test_gamma_logpdf_remains_accurate_for_concentrated_shape() -> None:
    value = jnp.float32(1.0)
    shape = jnp.float32(1e8)
    rate = jnp.float32(1e8)

    result = gamma_logpdf(value, shape, rate)
    gradients = jax.grad(gamma, argnums=(0, 1, 2))(value, shape, rate)

    assert jnp.allclose(result, 8.291401863098145, rtol=2e-7, atol=0)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([-1.0, 5.000000413701855e-9, 0.0]),
        rtol=2e-7,
        atol=0,
    )


def test_gamma_logpdf_preserves_displacement_near_large_shape_mode() -> None:
    value = jnp.float32(1.000100016593933)
    shape = jnp.float32(1e8)
    rate = jnp.float32(1e8)

    result = gamma_logpdf(value, shape, rate)
    gradients = jax.grad(gamma_logpdf, argnums=(0, 1, 2))(value, shape, rate)

    assert jnp.allclose(result, 7.791169166564941, rtol=3e-7, atol=0)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([-10001.6591796875, 0.00010001659393310547, -0.00010001659393310547]),
        rtol=3e-6,
        atol=0,
    )


def test_gamma_logpdf_handles_maximum_finite_concentrated_shape() -> None:
    shape = jnp.asarray(jnp.finfo(jnp.float32).max)

    result = gamma_logpdf(jnp.float32(1.0), shape, shape)
    gradients = jax.grad(gamma_logpdf, argnums=(0, 1, 2))(jnp.float32(1.0), shape, shape)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, 43.442481994628906, rtol=2e-7, atol=0)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_gamma_logpdf_remains_accurate_at_extreme_float64_shape() -> None:
    shape = jnp.float64(1e20)

    result = gamma_logpdf(jnp.float64(1.0), shape, shape)

    assert jnp.allclose(result, 22.106912396735783, rtol=1e-14, atol=0)


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


def test_gamma_logpdf_supports_higher_order_differentiation() -> None:
    parameters = jnp.array([1.0, 2.5, 1.7], dtype=jnp.float32)
    expected = jnp.array(
        [
            [-1.5, 1.0, -1.0],
            [1.0, -0.49035776, 0.5882353],
            [-1.0, 0.5882353, -0.8650519],
        ]
    )

    result = jax.hessian(lambda arguments: gamma_logpdf(*arguments))(parameters)

    assert jnp.allclose(result, expected, rtol=2e-6, atol=1e-7)


def test_gamma_logpdf_supports_forward_mode_differentiation() -> None:
    primals = (jnp.float32(1.0), jnp.float32(2.5), jnp.float32(1.7))
    tangents = (jnp.float32(0.2), jnp.float32(-0.3), jnp.float32(0.4))

    _, result = jax.jvp(gamma_logpdf, primals, tangents)

    assert jnp.allclose(result, 0.19999381099)


@pytest.mark.parametrize(
    "arguments",
    [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (jnp.nan, 1.0, 1.0),
    ],
)
def test_gamma_logpdf_gradients_propagate_invalid_inputs(arguments) -> None:
    gradients = jax.grad(gamma_logpdf, argnums=(0, 1, 2))(*arguments)

    assert jnp.all(jnp.isnan(jnp.asarray(gradients)))


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
