"""Tests for probability density and random sampling functions."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mmmjax import (
    beta,
    beta_logpdf,
    beta_rng,
    exponential,
    exponential_logpdf,
    exponential_rng,
    gamma,
    gamma_logpdf,
    gamma_rng,
    half_normal,
    half_normal_logpdf,
    half_normal_rng,
    lognormal,
    lognormal_logpdf,
    lognormal_rng,
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


def test_half_normal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [-0.6312564607528918, -0.6868120163084473, -1.5201453496417807],
        dtype=jnp.float32,
    )

    result = half_normal_logpdf(values, 1.5)

    assert jnp.allclose(result, expected)


def test_lognormal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.25, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-2.431936110359028, -0.7255288953883893, -2.15889539821782],
        dtype=jnp.float32,
    )

    result = lognormal_logpdf(values, 0.4, 0.7)

    assert jnp.allclose(result, expected)


def test_exponential_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [0.6931471805599453, -0.3068528194400547, -3.3068528194400546],
        dtype=jnp.float32,
    )

    result = exponential_logpdf(values, 2.0)

    assert jnp.allclose(result, expected)


def test_gamma_logpdf_matches_known_values() -> None:
    values = jnp.array([0.25, 1.0, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-1.4625537844973295, -0.6581122428174937, -2.410193809815329],
        dtype=jnp.float32,
    )

    result = gamma_logpdf(values, 2.5, 1.7)

    assert jnp.allclose(result, expected)


def test_beta_logpdf_matches_known_values() -> None:
    values = jnp.array([0.1, 0.4, 0.8], dtype=jnp.float32)
    expected = jnp.array(
        [0.30546208190868773, 0.6074238513643362, -2.556350281979741],
        dtype=jnp.float32,
    )

    result = beta_logpdf(values, 2.3, 4.7)

    assert jnp.allclose(result, expected)


def test_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 1.0, -2.0])

    result = normal(values, 0.0, 1.0)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(normal_logpdf(values, 0.0, 1.0)))


def test_half_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 0.5, 2.0])

    result = half_normal(values, 1.5)

    assert result.shape == ()
    assert jnp.allclose(result, -2.83821382670312)


def test_lognormal_returns_scalar_sum() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = lognormal(values, 0.4, 0.7)

    assert result.shape == ()
    assert jnp.allclose(result, -5.316360403965237)


def test_exponential_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 0.5, 2.0])

    result = exponential(values, 2.0)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(exponential_logpdf(values, 2.0)))


def test_gamma_returns_scalar_sum() -> None:
    values = jnp.array([0.25, 1.0, 3.0])

    result = gamma(values, 2.5, 1.7)

    assert result.shape == ()
    assert jnp.allclose(result, -4.530859837130152)


def test_beta_returns_scalar_sum() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = beta(values, 2.3, 4.7)

    assert result.shape == ()
    assert jnp.allclose(result, -1.6434643487067171)


@pytest.mark.parametrize(
    ("density", "arguments"),
    [
        (normal, (jnp.empty((0,)), 0.0, 1.0)),
        (half_normal, (jnp.empty((0,)), 1.0)),
        (lognormal, (jnp.empty((0,)), 0.0, 1.0)),
        (exponential, (jnp.empty((0,)), 1.0)),
        (gamma, (jnp.empty((0,)), 1.0, 1.0)),
        (beta, (jnp.empty((0,)), 1.0, 1.0)),
    ],
)
def test_density_of_empty_batch_is_scalar_zero(density, arguments) -> None:
    result = density(*arguments)

    assert result.shape == ()
    assert result == 0


@pytest.mark.parametrize(
    ("density", "arguments"),
    [
        (normal, (jnp.empty((0,)), 0.0, 0.0)),
        (normal, (jnp.empty((0,)), jnp.inf, 1.0)),
        (half_normal, (jnp.empty((0,)), 0.0)),
        (half_normal, (jnp.empty((0,)), jnp.inf)),
        (lognormal, (jnp.empty((0,)), 0.0, 0.0)),
        (lognormal, (jnp.empty((0,)), jnp.inf, 1.0)),
        (exponential, (jnp.empty((0,)), 0.0)),
        (exponential, (jnp.empty((0,)), jnp.inf)),
        (gamma, (jnp.empty((0,)), 0.0, 1.0)),
        (gamma, (jnp.empty((0,)), 1.0, jnp.inf)),
        (beta, (jnp.empty((0,)), 0.0, 1.0)),
        (beta, (jnp.empty((0,)), 1.0, jnp.inf)),
    ],
)
def test_invalid_parameters_remain_nan_for_empty_batch(density, arguments) -> None:
    eager = density(*arguments)
    compiled = jax.jit(density)(*arguments)

    assert jnp.isnan(eager)
    assert jnp.isnan(compiled)


def test_normal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    result = normal_logpdf(values, locations, 2.0)

    assert result.shape == (2, 3)
    assert jnp.allclose(normal(values, locations, 2.0), jnp.sum(result))


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


def test_lognormal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.5], [2.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    result = lognormal_logpdf(values, locations, 2.0)

    assert result.shape == (2, 3)
    assert jnp.allclose(lognormal(values, locations, 2.0), jnp.sum(result))


def test_exponential_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    rates = jnp.array([0.5, 1.0, 2.0])

    result = exponential_logpdf(values, rates)

    assert result.shape == (2, 3)
    assert jnp.allclose(exponential(values, rates), jnp.sum(result))


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


def test_beta_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.2], [0.75]])
    alphas = jnp.array([0.5, 1.0, 3.0])
    betas = jnp.array([2.0, 1.0, 0.5])
    expected = jnp.array(
        [
            [0.29389333245105947, 0.0, -3.1718425703486668],
            [-1.530135397345781, 0.0, 0.05324451451881232],
        ]
    )

    result = beta_logpdf(values, alphas, betas)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected, atol=1e-6)
    assert jnp.allclose(beta(values, alphas, betas), jnp.sum(expected))


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


def test_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    half_log_two_pi = jnp.asarray(0.9189385332046727, dtype=jnp.float32)
    expected = -jnp.log(scales) - half_log_two_pi

    result = normal_logpdf(jnp.zeros(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_half_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    expected = jnp.asarray(-0.2257913526447274, dtype=jnp.float32) - jnp.log(scales)

    result = half_normal_logpdf(jnp.zeros(2, dtype=jnp.float32), scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_lognormal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    half_log_two_pi = jnp.asarray(0.9189385332046727, dtype=jnp.float32)
    expected = -jnp.log(scales) - half_log_two_pi

    result = lognormal_logpdf(jnp.ones(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


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


def test_beta_logpdf_uses_boundary_limits() -> None:
    alphas = jnp.array([0.5, 1.0, 2.0])
    betas = jnp.array([0.5, 1.0, 2.0])
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)

    lower_result = beta_logpdf(jnp.array([[0.0], [-0.0]]), alphas, 4.7)
    upper_result = beta_logpdf(1.0, 2.3, betas)
    extreme_equality_result = beta_logpdf(jnp.array([0.0, 1.0]), jnp.array([1.0, maximum]), jnp.array([maximum, 1.0]))
    extreme_concentrated_result = beta_logpdf(jnp.array([0.0, 1.0]), maximum, maximum)

    assert lower_result.shape == (2, 3)
    assert jnp.all(jnp.isposinf(lower_result[:, 0]))
    assert jnp.allclose(lower_result[:, 1], jnp.log(4.7))
    assert jnp.all(jnp.isneginf(lower_result[:, 2]))
    assert jnp.isposinf(upper_result[0])
    assert jnp.allclose(upper_result[1], jnp.log(2.3))
    assert jnp.isneginf(upper_result[2])
    assert jnp.allclose(extreme_equality_result, jnp.log(maximum))
    assert jnp.all(jnp.isneginf(extreme_concentrated_result))


def test_beta_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-jnp.inf, -0.1, 1.1, jnp.inf, jnp.nan])

    result = beta_logpdf(values, 2.3, 4.7)

    assert jnp.all(jnp.isneginf(result[:4]))
    assert jnp.isnan(result[4])


def test_beta_logpdf_rejects_invalid_parameters_before_support_check() -> None:
    invalid_parameters = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_alphas = beta_logpdf(-1.0, invalid_parameters, 1.0)
    invalid_betas = beta_logpdf(-1.0, 1.0, invalid_parameters)

    assert jnp.all(jnp.isnan(invalid_alphas))
    assert jnp.all(jnp.isnan(invalid_betas))


def test_beta_logpdf_preserves_small_tail_terms() -> None:
    result = beta_logpdf(jnp.float32(1e-20), 1.0, jnp.float32(1e20))

    assert jnp.isfinite(result)
    assert jnp.allclose(result, 45.051701859880914)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [(jnp.float16, jnp.float32), (jnp.bfloat16, jnp.float32), (jnp.float32, jnp.float32)],
)
def test_densities_use_at_least_float32(dtype, expected_dtype) -> None:
    values = jnp.array([0.0, 1.0], dtype=dtype)

    assert normal_logpdf(values, dtype(0.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert half_normal_logpdf(values, dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert lognormal_logpdf(values + 1, dtype(0.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert exponential_logpdf(values, dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert gamma_logpdf(values + 1, dtype(1.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert beta_logpdf(values, dtype(1.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)


def test_densities_promote_integer_inputs_to_float32() -> None:
    values = jnp.array([0, 1], dtype=jnp.int32)

    assert normal_logpdf(values, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert half_normal_logpdf(values, 1).dtype == jnp.dtype(jnp.float32)
    assert lognormal_logpdf(values + 1, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert exponential_logpdf(values, 1).dtype == jnp.dtype(jnp.float32)
    assert gamma_logpdf(values + 1, 1, 1).dtype == jnp.dtype(jnp.float32)
    assert beta_logpdf(values, 1, 1).dtype == jnp.dtype(jnp.float32)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_distribution_functions_support_float64() -> None:
    values = jnp.array([0.0, 1.0], dtype=jnp.float64)
    key = jax.random.key(0)

    assert normal_logpdf(values, 0.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert half_normal_logpdf(values, 1.0).dtype == jnp.dtype(jnp.float64)
    assert lognormal_logpdf(values + 1, 0.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert exponential_logpdf(values, 1.0).dtype == jnp.dtype(jnp.float64)
    assert gamma_logpdf(values + 1, 1.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert beta_logpdf(values, 1.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert normal_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert half_normal_rng(key, jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert lognormal_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert exponential_rng(key, jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert gamma_rng(key, jnp.float64(1.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert beta_rng(key, jnp.float64(1.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (normal_logpdf, (jnp.array([0.0, 1.0]), 0.5, 2.0)),
        (normal, (jnp.array([0.0, 1.0]), 0.5, 2.0)),
        (half_normal_logpdf, (jnp.array([0.0, 1.0]), 2.0)),
        (half_normal, (jnp.array([0.0, 1.0]), 2.0)),
        (lognormal_logpdf, (jnp.array([0.5, 1.0]), 0.5, 2.0)),
        (lognormal, (jnp.array([0.5, 1.0]), 0.5, 2.0)),
        (exponential_logpdf, (jnp.array([0.0, 1.0]), 2.0)),
        (exponential, (jnp.array([0.0, 1.0]), 2.0)),
        (gamma_logpdf, (jnp.array([0.5, 1.0]), 2.0, 1.5)),
        (gamma, (jnp.array([0.5, 1.0]), 2.0, 1.5)),
        (beta_logpdf, (jnp.array([0.25, 0.75]), 2.0, 1.5)),
        (beta, (jnp.array([0.25, 0.75]), 2.0, 1.5)),
    ],
)
def test_density_functions_can_be_jitted(function, arguments) -> None:
    assert jnp.allclose(jax.jit(function)(*arguments), function(*arguments))


def test_normal_is_differentiable_with_respect_to_location() -> None:
    values = jnp.array([0.0, 1.0, 2.0])
    location = 0.5
    scale = 2.0
    expected = jnp.sum(values - location) / scale**2

    result = jax.grad(lambda current_location: normal(values, current_location, scale))(location)

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


def test_exponential_is_differentiable_with_respect_to_rate() -> None:
    values = jnp.array([0.25, 1.5])
    rate = 2.0
    expected = values.size / rate - jnp.sum(values)

    result = jax.grad(lambda current_rate: exponential(values, current_rate))(rate)

    assert jnp.allclose(result, expected)


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


def test_beta_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.1, 0.4, 0.8])
    expected = jnp.array([8.888888888888886, -2.9166666666666674, -16.875000000000004])

    result = jax.grad(lambda current_values: beta(current_values, 2.3, 4.7))(values)

    assert jnp.allclose(result, expected)


def test_beta_is_differentiable_with_respect_to_alpha() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = jax.grad(lambda current_alpha: beta(values, current_alpha, 4.7))(2.3)

    assert jnp.allclose(result, 0.37621398802108175)


def test_beta_is_differentiable_with_respect_to_beta() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = jax.grad(lambda current_beta: beta(values, 2.3, current_beta))(4.7)

    assert jnp.allclose(result, -0.91954247545786227)


def test_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(normal)(values, locations, scales)
    expected = jnp.stack(
        [normal(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_half_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [0.5, 2.0]])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(half_normal)(values, scales)
    expected = jnp.stack([half_normal(value, scale) for value, scale in zip(values, scales, strict=True)])

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


def test_exponential_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [0.5, 2.0]])
    rates = jnp.array([1.0, 2.0])

    result = jax.vmap(exponential)(values, rates)
    expected = jnp.stack([exponential(value, rate) for value, rate in zip(values, rates, strict=True)])

    assert jnp.allclose(result, expected)


def test_gamma_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.25, 1.0], [0.5, 2.0]])
    shapes = jnp.array([1.5, 3.0])
    rates = jnp.array([0.5, 2.0])

    result = jax.vmap(gamma)(values, shapes, rates)
    expected = jnp.stack([gamma(value, shape, rate) for value, shape, rate in zip(values, shapes, rates, strict=True)])

    assert jnp.allclose(result, expected)


def test_beta_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.1, 0.4], [0.5, 0.8]])
    alphas = jnp.array([1.5, 3.0])
    betas = jnp.array([0.5, 2.0])

    result = jax.vmap(beta)(values, alphas, betas)
    expected = jnp.stack(
        [beta(value, alpha, beta_value) for value, alpha, beta_value in zip(values, alphas, betas, strict=True)]
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


def test_half_normal_rng_folds_standard_normal_draws() -> None:
    key = jax.random.key(42)
    scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = jnp.abs(scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32))

    result = half_normal_rng(key, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)


def test_lognormal_rng_matches_transformed_standard_draws() -> None:
    key = jax.random.key(42)
    location = jnp.array([0.5, -1.0], dtype=jnp.float32)
    scale = jnp.array([0.25, 1.5], dtype=jnp.float32)
    expected = jnp.exp(location + scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32))

    result = lognormal_rng(key, location, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_exponential_rng_matches_rate_scaled_standard_draws() -> None:
    key = jax.random.key(42)
    rate = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = jax.random.exponential(key, shape=(3, 2), dtype=jnp.float32) / rate

    result = exponential_rng(key, rate, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_gamma_rng_scales_log_space_unit_rate_draws() -> None:
    key = jax.random.key(42)
    shapes = jnp.array([0.5, 2.5], dtype=jnp.float32)
    rates = jnp.array([1.7, 0.8], dtype=jnp.float32)
    expected = jnp.exp(jax.random.loggamma(key, shapes, shape=(3, 2), dtype=jnp.float32) - jnp.log(rates))

    result = gamma_rng(key, shapes, rates, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)


def test_beta_rng_wraps_jax_sampler() -> None:
    key = jax.random.key(42)
    alphas = jnp.array([0.5, 2.5], dtype=jnp.float32)
    betas = jnp.array([1.7, 0.8], dtype=jnp.float32)
    expected = jax.random.beta(key, alphas, betas, shape=(3, 2), dtype=jnp.float32)

    result = beta_rng(key, alphas, betas, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all((result >= 0) & (result <= 1))


def test_normal_rng_uses_broadcast_parameter_shape() -> None:
    location = jnp.zeros((2, 1))
    scale = jnp.ones(3)

    result = normal_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_lognormal_rng_uses_broadcast_parameter_shape() -> None:
    location = jnp.zeros((2, 1))
    scale = jnp.ones(3)

    result = lognormal_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_gamma_rng_uses_broadcast_parameter_shape() -> None:
    shapes = jnp.ones((2, 1))
    rates = jnp.ones(3)

    result = gamma_rng(jax.random.key(0), shapes, rates, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_beta_rng_uses_broadcast_parameter_shape() -> None:
    alphas = jnp.ones((2, 1))
    betas = jnp.ones(3)

    result = beta_rng(jax.random.key(0), alphas, betas, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_rngs_return_scalar_for_scalar_parameters() -> None:
    key = jax.random.key(0)

    assert normal_rng(key, 0.0, 1.0).shape == ()
    assert half_normal_rng(key, 1.0).shape == ()
    assert lognormal_rng(key, 0.0, 1.0).shape == ()
    assert exponential_rng(key, 1.0).shape == ()
    assert gamma_rng(key, 1.0, 1.0).shape == ()
    assert beta_rng(key, 1.0, 1.0).shape == ()


def test_rngs_return_nan_for_invalid_parameters() -> None:
    key = jax.random.key(0)

    normal_result = normal_rng(key, jnp.array([0.0, 0.0, jnp.inf]), jnp.array([1.0, 0.0, 1.0]))
    half_normal_result = half_normal_rng(key, jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]))
    lognormal_result = lognormal_rng(key, jnp.array([0.0, 0.0, jnp.inf]), jnp.array([1.0, 0.0, 1.0]))
    exponential_result = exponential_rng(key, jnp.array([1.0, 0.0, jnp.inf]))
    gamma_shape_result = gamma_rng(key, jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]), 1.0)
    gamma_rate_result = gamma_rng(key, 1.0, jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]))
    beta_alpha_result = beta_rng(key, jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]), 1.0)
    beta_beta_result = beta_rng(key, 1.0, jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]))

    assert jnp.isfinite(normal_result[0])
    assert jnp.all(jnp.isnan(normal_result[1:]))
    assert jnp.isfinite(half_normal_result[0])
    assert jnp.all(jnp.isnan(half_normal_result[1:]))
    assert jnp.isfinite(lognormal_result[0])
    assert jnp.all(jnp.isnan(lognormal_result[1:]))
    assert jnp.isfinite(exponential_result[0])
    assert jnp.all(jnp.isnan(exponential_result[1:]))
    assert jnp.isfinite(gamma_shape_result[0])
    assert jnp.all(jnp.isnan(gamma_shape_result[1:]))
    assert jnp.isfinite(gamma_rate_result[0])
    assert jnp.all(jnp.isnan(gamma_rate_result[1:]))
    assert jnp.isfinite(beta_alpha_result[0])
    assert jnp.all(jnp.isnan(beta_alpha_result[1:]))
    assert jnp.isfinite(beta_beta_result[0])
    assert jnp.all(jnp.isnan(beta_beta_result[1:]))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_rngs_compute_with_float32_for_low_precision_parameters(dtype) -> None:
    key = jax.random.key(0)
    location = jnp.array([0.0, 1.0], dtype=dtype)
    scale = jnp.array([1.0, 2.0], dtype=dtype)
    rate = jnp.array([0.5, 2.0], dtype=dtype)
    expected_normal = location.astype(jnp.float32) + scale.astype(jnp.float32) * jax.random.normal(
        key, shape=(2,), dtype=jnp.float32
    )
    expected_half_normal = jnp.abs(scale.astype(jnp.float32) * jax.random.normal(key, shape=(2,), dtype=jnp.float32))
    expected_lognormal = jnp.exp(expected_normal)
    expected_exponential = jax.random.exponential(key, shape=(2,), dtype=jnp.float32) / rate.astype(jnp.float32)
    expected_gamma = jnp.exp(
        jax.random.loggamma(key, scale.astype(jnp.float32), shape=(2,), dtype=jnp.float32)
        - jnp.log(rate.astype(jnp.float32))
    )
    expected_beta = jax.random.beta(
        key,
        scale.astype(jnp.float32),
        rate.astype(jnp.float32),
        shape=(2,),
        dtype=jnp.float32,
    )

    normal_result = normal_rng(key, location, scale)
    half_normal_result = half_normal_rng(key, scale)
    lognormal_result = lognormal_rng(key, location, scale)
    exponential_result = exponential_rng(key, rate)
    gamma_result = gamma_rng(key, scale, rate)
    beta_result = beta_rng(key, scale, rate)

    assert normal_result.dtype == jnp.dtype(jnp.float32)
    assert half_normal_result.dtype == jnp.dtype(jnp.float32)
    assert lognormal_result.dtype == jnp.dtype(jnp.float32)
    assert exponential_result.dtype == jnp.dtype(jnp.float32)
    assert gamma_result.dtype == jnp.dtype(jnp.float32)
    assert beta_result.dtype == jnp.dtype(jnp.float32)
    assert jnp.array_equal(normal_result, expected_normal)
    assert jnp.array_equal(half_normal_result, expected_half_normal)
    assert jnp.array_equal(lognormal_result, expected_lognormal)
    assert jnp.array_equal(exponential_result, expected_exponential)
    assert jnp.array_equal(gamma_result, expected_gamma)
    assert jnp.array_equal(beta_result, expected_beta)


def test_rngs_can_be_jitted() -> None:
    key = jax.random.key(0)
    compiled_normal = jax.jit(partial(normal_rng, location=0.0, scale=1.0, sample_shape=(2,)))
    compiled_half_normal = jax.jit(partial(half_normal_rng, scale=1.0, sample_shape=(2,)))
    compiled_lognormal = jax.jit(partial(lognormal_rng, location=0.0, scale=1.0, sample_shape=(2,)))
    compiled_exponential = jax.jit(partial(exponential_rng, rate=1.0, sample_shape=(2,)))
    compiled_gamma = jax.jit(partial(gamma_rng, shape=2.0, rate=1.0, sample_shape=(2,)))
    compiled_beta = jax.jit(partial(beta_rng, alpha=2.0, beta=1.0, sample_shape=(2,)))

    assert jnp.array_equal(compiled_normal(key), normal_rng(key, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_half_normal(key), half_normal_rng(key, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_lognormal(key), lognormal_rng(key, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_exponential(key), exponential_rng(key, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_gamma(key), gamma_rng(key, 2.0, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_beta(key), beta_rng(key, 2.0, 1.0, sample_shape=(2,)))


def test_rngs_can_be_vectorized_over_keys() -> None:
    keys = jax.random.split(jax.random.key(0), 3)

    normal_result = jax.vmap(normal_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    half_normal_result = jax.vmap(half_normal_rng, in_axes=(0, None))(keys, 1.0)
    lognormal_result = jax.vmap(lognormal_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    exponential_result = jax.vmap(exponential_rng, in_axes=(0, None))(keys, 1.0)
    gamma_result = jax.vmap(gamma_rng, in_axes=(0, None, None))(keys, 2.0, 1.0)
    beta_result = jax.vmap(beta_rng, in_axes=(0, None, None))(keys, 2.0, 1.0)
    expected_normal = jnp.stack([normal_rng(key, 0.0, 1.0) for key in keys])
    expected_half_normal = jnp.stack([half_normal_rng(key, 1.0) for key in keys])
    expected_lognormal = jnp.stack([lognormal_rng(key, 0.0, 1.0) for key in keys])
    expected_exponential = jnp.stack([exponential_rng(key, 1.0) for key in keys])
    expected_gamma = jnp.stack([gamma_rng(key, 2.0, 1.0) for key in keys])
    expected_beta = jnp.stack([beta_rng(key, 2.0, 1.0) for key in keys])

    assert jnp.array_equal(normal_result, expected_normal)
    assert jnp.array_equal(half_normal_result, expected_half_normal)
    assert jnp.allclose(lognormal_result, expected_lognormal)
    assert jnp.array_equal(exponential_result, expected_exponential)
    assert jnp.allclose(gamma_result, expected_gamma)
    assert jnp.array_equal(beta_result, expected_beta)


def test_rngs_are_deterministic_for_a_given_key() -> None:
    key, different_key = jax.random.split(jax.random.key(0))

    first = normal_rng(key, 0.0, 1.0, sample_shape=(8,))
    repeated = normal_rng(key, 0.0, 1.0, sample_shape=(8,))
    different = normal_rng(different_key, 0.0, 1.0, sample_shape=(8,))

    assert jnp.array_equal(first, repeated)
    assert not jnp.array_equal(first, different)


def test_gamma_rng_matches_distribution_moments() -> None:
    samples = gamma_rng(jax.random.key(7), 4.0, 2.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 2.0, rtol=0, atol=0.03)
    assert jnp.allclose(jnp.var(samples), 1.0, rtol=0, atol=0.06)


def test_beta_rng_matches_distribution_moments() -> None:
    samples = beta_rng(jax.random.key(7), 2.0, 5.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 2 / 7, rtol=0, atol=0.007)
    assert jnp.allclose(jnp.var(samples), 10 / 392, rtol=0, atol=0.002)


def test_rng_supports_empty_sample_dimension() -> None:
    result = exponential_rng(jax.random.key(0), jnp.ones(2), sample_shape=(0, 3))

    assert result.shape == (0, 3, 2)


@pytest.mark.parametrize("sample_shape", [[2], (True,), (1.5,)])
def test_rng_rejects_invalid_sample_shape_type(sample_shape) -> None:
    with pytest.raises(
        TypeError,
        match=r"sample_shape(\[0\])? must be a (tuple of )?nonnegative integer",
    ):
        normal_rng(jax.random.key(0), 0.0, 1.0, sample_shape=sample_shape)


def test_rng_rejects_negative_sample_shape() -> None:
    with pytest.raises(ValueError, match=r"sample_shape\[0\] must be nonnegative, got -1"):
        exponential_rng(jax.random.key(0), 1.0, sample_shape=(-1,))


def test_normal_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        normal_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))


def test_gamma_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        gamma_rng(jax.random.key(0), jnp.ones(2), jnp.ones(3))


def test_beta_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        beta_rng(jax.random.key(0), jnp.ones(2), jnp.ones(3))


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (normal_logpdf, (0.0, 0.0, 1.0 + 0.0j), "scale"),
        (half_normal_logpdf, (0.0, 1.0 + 0.0j), "scale"),
        (lognormal_logpdf, (1.0, 0.0, 1.0 + 0.0j), "scale"),
        (exponential_logpdf, (0.0, 1.0 + 0.0j), "rate"),
        (gamma_logpdf, (1.0, 1.0 + 0.0j, 1.0), "shape"),
        (beta_logpdf, (0.5, 1.0 + 0.0j, 1.0), "alpha"),
        (normal_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "scale"),
        (half_normal_rng, (jax.random.key(0), 1.0 + 0.0j), "scale"),
        (lognormal_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "scale"),
        (exponential_rng, (jax.random.key(0), 1.0 + 0.0j), "rate"),
        (gamma_rng, (jax.random.key(0), 1.0, 1.0 + 0.0j), "rate"),
        (beta_rng, (jax.random.key(0), 1.0, 1.0 + 0.0j), "beta"),
    ],
)
def test_distribution_functions_reject_complex_arguments(function, arguments, argument_name: str) -> None:
    with pytest.raises(
        TypeError,
        match=rf"argument '{argument_name}' must have a real numeric dtype, got complex",
    ):
        function(*arguments)


def test_distribution_functions_identify_non_array_like_argument() -> None:
    with pytest.raises(
        TypeError,
        match="argument 'scale' must be real numeric and array-like, got object",
    ):
        normal_logpdf(0.0, 0.0, object())
