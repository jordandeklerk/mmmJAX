"""Tests for Truncated Normal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    ("value", "location", "scale", "lower", "upper"),
    [
        pytest.param(0.0, 0.0, 1.0, -0.01, 0.01, id="narrow-central"),
        pytest.param(6.05, 0.0, 1.0, 6.0, 6.1, id="same-side-tail"),
        pytest.param(2.0, 0.0, 1.0, 1.0, np.inf, id="lower-bound"),
        pytest.param(-2.0, 0.0, 1.0, -np.inf, -1.0, id="upper-bound"),
    ],
)
def test_truncated_normal_logpdf_matches_scipy_for_representative_bounds(
    value,
    location,
    scale,
    lower,
    upper,
) -> None:
    expected = _scipy_logpdf(value, location, scale, lower, upper)

    result = truncated_normal_logpdf(value, location, scale, lower, upper)

    assert jnp.isfinite(result)
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


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(jnp.array([0.4, -0.3, 1.7, -1.2, 2.4]), id="two-sided"),
        pytest.param(jnp.array([2.0, 0.0, 1.0, 1.0, 3.0]), id="finite-tail"),
        pytest.param(jnp.array([2.0, 0.0, 1.0, 1.0, jnp.inf]), id="lower-bound"),
        pytest.param(jnp.array([-2.0, 0.0, 1.0, -jnp.inf, -1.0]), id="upper-bound"),
    ],
)
def test_truncated_normal_logpdf_derivatives_match_closed_form(arguments) -> None:
    expected = _scipy_logpdf_gradient(arguments)

    def evaluate(current):
        return truncated_normal_logpdf(*current)

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)

    np.testing.assert_allclose(forward, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(reverse, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-0.5, 0.25], [0.75, 1.5]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 1.5])
    lowers = jnp.array([-1.0, -0.5])
    uppers = jnp.array([2.0, 2.5])
    expected = np.asarray(
        [
            np.sum(_scipy_logpdf(value, location, scale, lower, upper))
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

    result = jax.jit(jax.vmap(truncated_normal))(values, locations, scales, lowers, uppers)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


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


def _scipy_logpdf_gradient(arguments) -> np.ndarray:
    value, location, scale, lower, upper = np.asarray(arguments, dtype=np.float64)
    standardized_value = (value - location) / scale
    standardized_lower = (lower - location) / scale
    standardized_upper = (upper - location) / scale
    lower_weight = (
        0.0
        if np.isneginf(standardized_lower)
        else stats.truncnorm.pdf(
            standardized_lower,
            standardized_lower,
            standardized_upper,
        )
    )
    upper_weight = (
        0.0
        if np.isposinf(standardized_upper)
        else stats.truncnorm.pdf(
            standardized_upper,
            standardized_lower,
            standardized_upper,
        )
    )
    lower_moment = 0.0 if np.isneginf(standardized_lower) else standardized_lower * lower_weight
    upper_moment = 0.0 if np.isposinf(standardized_upper) else standardized_upper * upper_weight
    return np.array(
        [
            -standardized_value / scale,
            (standardized_value - lower_weight + upper_weight) / scale,
            (np.square(standardized_value) - 1 + upper_moment - lower_moment) / scale,
            lower_weight / scale,
            -upper_weight / scale,
        ]
    )
