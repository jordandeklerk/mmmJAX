"""Tests for Truncated Normal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf
from jax.scipy.stats import truncnorm as jax_truncnorm
from scipy import stats

from mmmjax import normal_logpdf, truncated_normal, truncated_normal_logpdf


def test_truncated_normal_logpdf_matches_scipy_across_broadcast_batches() -> None:
    values = np.array([[-1.25], [0.25], [2.5]], dtype=np.float32)
    locations = np.array([-0.5, 0.5, 1.0], dtype=np.float32)
    scales = np.array([0.75, 1.25, 2.0], dtype=np.float32)
    lowers = np.array([-2.0, -0.75, -1.0], dtype=np.float32)
    uppers = np.array([0.8, 2.0, 4.0], dtype=np.float32)
    expected = _scipy_logpdf(values, locations, scales, lowers, uppers)
    expected_jax = jax_truncnorm.logpdf(
        values,
        (lowers - locations) / scales,
        (uppers - locations) / scales,
        loc=locations,
        scale=scales,
    )

    result = truncated_normal_logpdf(values, locations, scales, lowers, uppers)
    compiled = jax.jit(truncated_normal_logpdf)(values, locations, scales, lowers, uppers)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(result, expected_jax, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_logpdf_remains_accurate_in_narrow_tail_intervals() -> None:
    values = np.array([-39.5, -20.005, 20.005, 39.5], dtype=np.float32)
    lowers = np.array([-40.0, -20.01, 20.0, 39.0], dtype=np.float32)
    uppers = np.array([-39.0, -20.0, 20.01, 40.0], dtype=np.float32)
    expected = _scipy_logpdf(values, 0.0, 1.0, lowers, uppers)

    result = truncated_normal_logpdf(values, 0.0, 1.0, lowers, uppers)
    value_gradient = jax.jit(
        jax.jacrev(lambda current: jnp.sum(truncated_normal_logpdf(current, 0, 1, lowers, uppers)))
    )(jnp.asarray(values))

    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(jnp.isfinite(value_gradient))
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(value_gradient, -values, rtol=3e-6, atol=3e-6)


def test_truncated_normal_logpdf_remains_finite_for_a_narrow_central_interval() -> None:
    half_width = jnp.asarray(1e-10, dtype=jnp.float32)
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=jnp.float32))
    expected = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(erf(half_width / sqrt_two))

    result = truncated_normal_logpdf(0.0, 0.0, 1.0, -half_width, half_width)
    reverse = jax.jit(jax.jacrev(lambda bounds: truncated_normal_logpdf(0.0, 0.0, 1.0, bounds[0], bounds[1])))(
        jnp.array([-half_width, half_width])
    )

    assert jnp.isfinite(result)
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_logpdf_handles_adjacent_same_side_bounds() -> None:
    lower = jnp.asarray(1.0, dtype=jnp.float32)
    upper = jnp.nextafter(lower, jnp.asarray(jnp.inf, dtype=lower.dtype))
    expected = _scipy_logpdf(lower, 0.0, 1.0, lower, upper)

    result = truncated_normal_logpdf(lower, 0.0, 1.0, lower, upper)
    reverse = jax.jit(jax.jacrev(lambda bounds: truncated_normal_logpdf(bounds[0], 0.0, 1.0, bounds[0], bounds[1])))(
        jnp.array([lower, upper])
    )

    assert jnp.isfinite(result)
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_returns_scalar_sum() -> None:
    values = jnp.array([[-0.5], [0.25], [1.5]])
    locations = jnp.array([-0.25, 0.75])
    scales = jnp.array([0.8, 1.5])

    result = truncated_normal(values, locations, scales, -1.0, 2.0)
    expected = jnp.sum(truncated_normal_logpdf(values, locations, scales, -1.0, 2.0))

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_truncated_normal_logpdf_handles_support_boundaries_and_nonfinite_values() -> None:
    values = jnp.array([-jnp.inf, -1.1, -1.0, 0.25, 2.0, 2.1, jnp.inf, jnp.nan])

    result = truncated_normal_logpdf(values, 0.0, 1.0, -1.0, 2.0)

    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.all(jnp.isfinite(result[2:5]))
    assert jnp.all(jnp.isneginf(result[5:7]))
    assert jnp.isnan(result[7])


def test_truncated_normal_logpdf_rejects_invalid_locations_and_scales() -> None:
    locations = jnp.array([jnp.inf, -jnp.inf, jnp.nan])
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_locations = truncated_normal_logpdf(100.0, locations, 1.0, -1.0, 1.0)
    invalid_scales = truncated_normal_logpdf(100.0, 0.0, scales, -1.0, 1.0)

    assert jnp.all(jnp.isnan(invalid_locations))
    assert jnp.all(jnp.isnan(invalid_scales))


def test_truncated_normal_logpdf_rejects_invalid_bounds_before_support() -> None:
    lowers = jnp.array([0.0, 1.0, jnp.nan, jnp.inf, -jnp.inf])
    uppers = jnp.array([0.0, 0.0, 1.0, jnp.inf, -jnp.inf])

    result = truncated_normal_logpdf(100.0, 0.0, 1.0, lowers, uppers)

    assert jnp.all(jnp.isnan(result))


def test_truncated_normal_logpdf_supports_infinite_bounds() -> None:
    values = jnp.array([-0.5, 0.25, 1.5])
    lower = jnp.array([-jnp.inf, -1.0, -jnp.inf])
    upper = jnp.array([1.0, jnp.inf, jnp.inf])
    expected = _scipy_logpdf(values, 0.0, 1.0, lower, upper)

    result = truncated_normal_logpdf(values, 0.0, 1.0, lower, upper)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(result[2], normal_logpdf(values[2], 0.0, 1.0), rtol=3e-6, atol=3e-6)


def test_truncated_normal_logpdf_derivatives_match_closed_form() -> None:
    arguments = jnp.array([0.4, -0.3, 1.7, -1.2, 2.4])
    expected = np.array(
        [
            -0.2422145328719723,
            0.02923751578460876,
            -0.1576836727027529,
            0.3159515359807837,
            -0.1029745188934201,
        ]
    )

    def evaluate(current):
        return truncated_normal_logpdf(*current)

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)

    np.testing.assert_allclose(forward, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(reverse, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-jnp.inf, 1.0, [0.4875999709, -0.6724000291]),
        (-1.0, jnp.inf, [-0.0875999709, -0.6724000291]),
        (-jnp.inf, jnp.inf, [0.2, -0.96]),
    ],
)
def test_truncated_normal_logpdf_has_finite_gradients_with_infinite_bounds(lower, upper, expected) -> None:
    def evaluate(current):
        return truncated_normal_logpdf(0.2, current[0], current[1], lower, upper)

    result = jax.jit(jax.jacrev(evaluate))(jnp.array([0.0, 1.0]))

    assert jnp.all(jnp.isfinite(result))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_logpdf_derivatives_remain_finite_in_the_far_tail() -> None:
    arguments = jnp.array([20.5, 0.0, 1.0, 20.0, 21.0])
    standardized_value, _, scale, standardized_lower, standardized_upper = np.asarray(
        arguments,
        dtype=np.float64,
    )
    lower_weight = stats.truncnorm.pdf(
        standardized_lower,
        standardized_lower,
        standardized_upper,
    )
    upper_weight = stats.truncnorm.pdf(
        standardized_upper,
        standardized_lower,
        standardized_upper,
    )
    expected = np.array(
        [
            -standardized_value / scale,
            (standardized_value - lower_weight + upper_weight) / scale,
            (np.square(standardized_value) - 1 + standardized_upper * upper_weight - standardized_lower * lower_weight)
            / scale,
            lower_weight / scale,
            -upper_weight / scale,
        ]
    )

    def evaluate(current):
        return truncated_normal_logpdf(*current)

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)

    assert jnp.all(jnp.isfinite(forward))
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(forward, expected, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(reverse, expected, rtol=2e-4, atol=1e-3)


def test_truncated_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-0.5, 0.25], [0.75, 1.5]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 1.5])
    lowers = jnp.array([-1.0, -0.5])
    uppers = jnp.array([2.0, 2.5])

    result = jax.jit(jax.vmap(truncated_normal))(values, locations, scales, lowers, uppers)
    expected = jnp.stack(
        [
            truncated_normal(value, location, scale, lower, upper)
            for value, location, scale, lower, upper in zip(
                values,
                locations,
                scales,
                lowers,
                uppers,
                strict=True,
            )
        ]
    )

    assert jnp.allclose(result, expected)


def _scipy_logpdf(value, location, scale, lower, upper) -> np.ndarray:
    value_array = np.asarray(value, dtype=np.float64)
    location_array = np.asarray(location, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    standardized_lower = (lower_array - location_array) / scale_array
    standardized_upper = (upper_array - location_array) / scale_array
    return stats.truncnorm.logpdf(
        value_array,
        standardized_lower,
        standardized_upper,
        loc=location_array,
        scale=scale_array,
    )
