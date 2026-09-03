"""Tests for Truncated Normal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import truncnorm as jax_truncnorm
from scipy import special, stats

from mmmjax import (
    normal_logcdf,
    normal_logpdf,
    normal_logsf,
    truncated_normal,
    truncated_normal_logcdf,
    truncated_normal_logpdf,
    truncated_normal_logsf,
    truncated_normal_rng,
)
from mmmjax.distributions._truncated_normal import _inverse_normal_logcdf


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


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        pytest.param(truncated_normal_logcdf, stats.truncnorm.logcdf, id="logcdf"),
        pytest.param(truncated_normal_logsf, stats.truncnorm.logsf, id="logsf"),
    ],
)
def test_truncated_normal_log_probabilities_match_scipy_across_broadcast_batches(function, reference) -> None:
    values = np.array([[-1.25], [0.25], [2.5]], dtype=np.float32)
    locations = np.array([-0.5, 0.5, 1.0], dtype=np.float32)
    scales = np.array([0.75, 1.25, 2.0], dtype=np.float32)
    lowers = np.array([-2.0, -0.75, -1.0], dtype=np.float32)
    uppers = np.array([0.8, 2.0, 4.0], dtype=np.float32)
    standardized_lowers = (lowers.astype(np.float64) - locations) / scales
    standardized_uppers = (uppers.astype(np.float64) - locations) / scales
    expected = reference(
        values.astype(np.float64),
        standardized_lowers,
        standardized_uppers,
        loc=locations.astype(np.float64),
        scale=scales.astype(np.float64),
    )

    result = function(values, locations, scales, lowers, uppers)
    compiled = jax.jit(function)(values, locations, scales, lowers, uppers)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize(
    ("value", "location", "scale", "lower", "upper"),
    [
        pytest.param(0.0, 0.0, 1.0, -0.01, 0.01, id="narrow-central"),
        pytest.param(6.05, 0.0, 1.0, 6.0, 6.1, id="right-tail"),
        pytest.param(-6.05, 0.0, 1.0, -6.1, -6.0, id="left-tail"),
        pytest.param(2.0, 0.0, 1.0, 1.0, np.inf, id="lower-bound"),
        pytest.param(-2.0, 0.0, 1.0, -np.inf, -1.0, id="upper-bound"),
        pytest.param(0.25, 0.0, 1.0, -np.inf, np.inf, id="untruncated"),
    ],
)
@pytest.mark.parametrize(
    ("function", "upper_tail"),
    [
        pytest.param(truncated_normal_logcdf, False, id="logcdf"),
        pytest.param(truncated_normal_logsf, True, id="logsf"),
    ],
)
def test_truncated_normal_log_probabilities_match_scipy_for_representative_bounds(
    function,
    upper_tail: bool,
    value: float,
    location: float,
    scale: float,
    lower: float,
    upper: float,
) -> None:
    expected = _scipy_log_probability(value, location, scale, lower, upper, upper_tail=upper_tail)

    result = function(value, location, scale, lower, upper)

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_truncated_normal_log_probabilities_are_complements() -> None:
    values = jnp.linspace(6.001, 6.099, 21)

    log_cdf = truncated_normal_logcdf(values, 0.0, 1.0, 6.0, 6.1)
    log_survival = truncated_normal_logsf(values, 0.0, 1.0, 6.0, 6.1)

    assert jnp.allclose(jnp.logaddexp(log_cdf, log_survival), 0, atol=1e-6)


def test_truncated_normal_log_probabilities_handle_boundaries_and_nonfinite_values() -> None:
    values = jnp.array([-jnp.inf, -1.1, -1.0, 0.25, 2.0, 2.1, jnp.inf, jnp.nan])

    log_cdf = truncated_normal_logcdf(values, 0.0, 1.0, -1.0, 2.0)
    log_survival = truncated_normal_logsf(values, 0.0, 1.0, -1.0, 2.0)

    assert jnp.all(jnp.isneginf(log_cdf[:3]))
    assert jnp.isfinite(log_cdf[3])
    assert jnp.all(log_cdf[4:7] == 0)
    assert jnp.isnan(log_cdf[7])
    assert jnp.all(log_survival[:3] == 0)
    assert jnp.isfinite(log_survival[3])
    assert jnp.all(jnp.isneginf(log_survival[4:7]))
    assert jnp.isnan(log_survival[7])


@pytest.mark.parametrize("function", [truncated_normal_logcdf, truncated_normal_logsf])
def test_truncated_normal_log_probabilities_reject_invalid_parameters(function) -> None:
    locations = jnp.array([jnp.inf, -jnp.inf, jnp.nan])
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])
    lowers = jnp.array([0.0, 1.0, jnp.nan, jnp.inf, -jnp.inf])
    uppers = jnp.array([0.0, 0.0, 1.0, jnp.inf, -jnp.inf])

    invalid_locations = function(100.0, locations, 1.0, -1.0, 1.0)
    invalid_scales = function(100.0, 0.0, scales, -1.0, 1.0)
    invalid_bounds = function(100.0, 0.0, 1.0, lowers, uppers)

    assert jnp.all(jnp.isnan(invalid_locations))
    assert jnp.all(jnp.isnan(invalid_scales))
    assert jnp.all(jnp.isnan(invalid_bounds))


def test_untruncated_log_probabilities_reduce_to_normal() -> None:
    values = jnp.array([-6.0, -0.5, 0.0, 1.5, 8.0])

    log_cdf = truncated_normal_logcdf(values, 0.5, 1.7, -jnp.inf, jnp.inf)
    log_survival = truncated_normal_logsf(values, 0.5, 1.7, -jnp.inf, jnp.inf)

    assert jnp.allclose(log_cdf, normal_logcdf(values, 0.5, 1.7), rtol=3e-6, atol=0)
    assert jnp.allclose(log_survival, normal_logsf(values, 0.5, 1.7), rtol=3e-6, atol=0)


@pytest.mark.parametrize(
    ("function", "upper_tail"),
    [
        pytest.param(truncated_normal_logcdf, False, id="logcdf"),
        pytest.param(truncated_normal_logsf, True, id="logsf"),
    ],
)
@pytest.mark.parametrize(
    ("arguments", "rtol"),
    [
        pytest.param(jnp.array([0.4, -0.3, 1.7, -1.2, 2.4]), 3e-5, id="two-sided"),
        pytest.param(jnp.array([6.05, 0.0, 1.0, 6.0, 6.1]), 1e-3, id="same-side-tail"),
        pytest.param(jnp.array([2.0, 0.0, 1.0, 1.0, jnp.inf]), 3e-5, id="lower-bound"),
        pytest.param(jnp.array([-2.0, 0.0, 1.0, -jnp.inf, -1.0]), 3e-5, id="upper-bound"),
        pytest.param(jnp.array([0.25, 0.0, 1.0, -jnp.inf, jnp.inf]), 3e-5, id="untruncated"),
    ],
)
def test_truncated_normal_log_probability_derivatives_match_closed_form(
    function,
    upper_tail: bool,
    arguments: jax.Array,
    rtol: float,
) -> None:
    expected = _scipy_log_probability_gradient(arguments, upper_tail=upper_tail)

    def evaluate(current):
        return function(*current)

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)

    np.testing.assert_allclose(forward, expected, rtol=rtol, atol=1e-4)
    np.testing.assert_allclose(reverse, expected, rtol=rtol, atol=1e-4)


@pytest.mark.parametrize("function", [truncated_normal_logcdf, truncated_normal_logsf])
def test_truncated_normal_log_probabilities_support_higher_order_tail_derivatives(function) -> None:
    arguments = jnp.array([6.05, 0.0, 1.0, 6.0, 6.1])

    def evaluate(current):
        return function(*current)

    reverse_over_reverse = jax.jit(jax.jacrev(jax.grad(evaluate)))(arguments)
    forward_over_reverse = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.all(jnp.isfinite(reverse_over_reverse))
    np.testing.assert_allclose(reverse_over_reverse, forward_over_reverse, rtol=3e-5, atol=3e-4)


def test_truncated_normal_logcdf_preserves_higher_derivatives_near_the_upper_bound() -> None:
    arguments = jnp.array([19.999998, 0.0, 1.0, -1.0, 20.0])

    def evaluate(current):
        return truncated_normal_logcdf(*current)

    result = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    ("function", "upper_tail"),
    [
        pytest.param(truncated_normal_logcdf, False, id="logcdf"),
        pytest.param(truncated_normal_logsf, True, id="logsf"),
    ],
)
def test_truncated_normal_log_probabilities_vectorize_gradients_over_mixed_bounds(
    function,
    upper_tail: bool,
) -> None:
    arguments = jnp.array(
        [
            [0.4, -0.3, 1.7, -1.2, 2.4],
            [2.0, 0.0, 1.0, 1.0, jnp.inf],
            [0.25, 0.0, 1.0, -jnp.inf, jnp.inf],
        ]
    )
    expected = np.stack(
        [_scipy_log_probability_gradient(current, upper_tail=upper_tail) for current in np.asarray(arguments)]
    )

    def evaluate(current):
        return function(*current)

    result = jax.jit(jax.vmap(jax.grad(evaluate)))(arguments)

    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=1e-4)


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


def test_truncated_normal_logpdf_derivatives_are_finite_in_same_side_tail() -> None:
    arguments = jnp.array([6.05, 0.0, 1.0, 6.0, 6.1])
    expected = _scipy_logpdf_gradient(arguments)

    def evaluate(current):
        return truncated_normal_logpdf(*current)

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)

    np.testing.assert_allclose(forward, expected, rtol=3e-6, atol=1e-4)
    np.testing.assert_allclose(reverse, expected, rtol=3e-6, atol=1e-4)


def test_truncated_normal_logpdf_supports_higher_order_tail_derivatives() -> None:
    arguments = jnp.array([6.05, 0.0, 1.0, 6.0, 6.1])

    def evaluate(current):
        return truncated_normal_logpdf(*current)

    reverse_over_reverse = jax.jit(jax.jacrev(jax.grad(evaluate)))(arguments)
    forward_over_reverse = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.all(jnp.isfinite(reverse_over_reverse))
    np.testing.assert_allclose(reverse_over_reverse, forward_over_reverse, rtol=3e-5, atol=3e-4)


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


def test_truncated_normal_rng_matches_scipy_quantiles_including_extreme_tails() -> None:
    locations = jnp.array([0.5, 1.0, -0.3, 0.3, 0.0, 0.0, 0.0, 0.0])
    scales = jnp.array([1.2, 0.5, 0.8, 1.5, 1.0, 1.0, 1.0, 1.0])
    lowers = jnp.array([-1.0, 0.99, 0.0, -jnp.inf, -jnp.inf, 10.0, -11.0, 39.0])
    uppers = jnp.array([2.0, 1.01, jnp.inf, -1.0, jnp.inf, 11.0, -10.0, 40.0])
    key = jax.random.key(77)
    sample_size = 4_096
    minimum_probability = jnp.asarray(np.finfo(locations.dtype).tiny, dtype=locations.dtype)

    unit_samples = np.asarray(
        jax.random.uniform(
            key,
            shape=(sample_size, 8),
            dtype=locations.dtype,
            minval=minimum_probability,
        ),
        dtype=np.float64,
    )
    locations_numpy = np.asarray(locations, dtype=np.float64)
    scales_numpy = np.asarray(scales, dtype=np.float64)
    lowers_numpy = np.asarray(lowers, dtype=np.float64)
    uppers_numpy = np.asarray(uppers, dtype=np.float64)
    standardized_lowers = (lowers_numpy - locations_numpy) / scales_numpy
    standardized_uppers = (uppers_numpy - locations_numpy) / scales_numpy
    expected = stats.truncnorm.ppf(
        unit_samples,
        standardized_lowers,
        standardized_uppers,
        loc=locations_numpy,
        scale=scales_numpy,
    )

    result = truncated_normal_rng(
        key,
        locations,
        scales,
        lowers,
        uppers,
        sample_shape=(sample_size,),
    )

    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(result[:, jnp.isfinite(lowers)] > lowers[jnp.isfinite(lowers)])
    assert jnp.all(result[:, jnp.isfinite(uppers)] < uppers[jnp.isfinite(uppers)])


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_inverse_normal_logcdf_matches_scipy_around_tail_approximation_boundary() -> None:
    log_probabilities = jnp.array([-31.9999, -32.0, -32.0001], dtype=jnp.float64)
    expected = special.ndtri_exp(np.asarray(log_probabilities))

    result = _inverse_normal_logcdf(log_probabilities)

    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=2e-14)


def test_truncated_normal_rng_stays_strictly_inside_narrow_bounds_around_zero() -> None:
    lowers = jnp.array([0.0, -1e-5])
    uppers = jnp.array([1e-5, 0.0])

    result = truncated_normal_rng(
        jax.random.key(13),
        0.0,
        1.0,
        lowers,
        uppers,
        sample_shape=(4_096,),
    )

    # NumPy preserves subnormal endpoint guards that XLA comparisons can flush to zero
    result_numpy = np.asarray(result)
    assert np.all(result_numpy > np.asarray(lowers))
    assert np.all(result_numpy < np.asarray(uppers))


def test_truncated_normal_rng_does_not_map_zero_probability_to_an_unbounded_tail() -> None:
    key = jax.random.key(4)
    sample_shape = (300_000,)
    default_uniforms = jax.random.uniform(key, shape=sample_shape, dtype=jnp.float32)

    result = truncated_normal_rng(
        key,
        0.0,
        1.0,
        -jnp.inf,
        jnp.inf,
        sample_shape=sample_shape,
    )

    assert jnp.any(default_uniforms == 0)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.min(result) > -20


def test_truncated_normal_rng_uses_broadcast_parameter_shape() -> None:
    locations = jnp.array([[0.0], [1.0]])
    scales = jnp.array([0.5, 1.0, 2.0])
    lowers = jnp.array([[-1.0], [0.0]])
    uppers = jnp.array([0.5, 2.0, jnp.inf])

    result = truncated_normal_rng(
        jax.random.key(0),
        locations,
        scales,
        lowers,
        uppers,
        sample_shape=(4, 5),
    )

    assert result.shape == (4, 5, 2, 3)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(result > lowers)
    assert jnp.all(result[..., :2] < uppers[:2])


def test_truncated_normal_rng_can_be_jitted_with_dynamic_parameters() -> None:
    locations = jnp.array([0.0, 0.0, 0.0, 0.0])
    scales = jnp.array([1.0, 1.0, 1.0, 2.0])
    lowers = jnp.array([-1.0, 10.0, -jnp.inf, -jnp.inf])
    uppers = jnp.array([2.0, 11.0, -8.0, jnp.inf])
    key = jax.random.key(5)
    compiled = jax.jit(
        lambda current_key, current_location, current_scale, current_lower, current_upper: truncated_normal_rng(
            current_key,
            current_location,
            current_scale,
            current_lower,
            current_upper,
            sample_shape=(32,),
        )
    )

    result = compiled(key, locations, scales, lowers, uppers)
    expected = truncated_normal_rng(key, locations, scales, lowers, uppers, sample_shape=(32,))

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_truncated_normal_rng_has_finite_pathwise_tail_gradients() -> None:
    key = jax.random.key(5)

    def sample_sum(arguments):
        return jnp.sum(truncated_normal_rng(key, *arguments, sample_shape=(16,)))

    result = jax.jit(jax.grad(sample_sum))(jnp.array([0.0, 1.0, 6.0, 6.1]))

    assert jnp.all(jnp.isfinite(result))


def test_truncated_normal_rng_is_deterministic_for_a_given_key() -> None:
    key, different_key = jax.random.split(jax.random.key(0))

    first = truncated_normal_rng(key, 0.0, 1.0, 6.0, 6.1, sample_shape=(128,))
    repeated = truncated_normal_rng(key, 0.0, 1.0, 6.0, 6.1, sample_shape=(128,))
    different = truncated_normal_rng(different_key, 0.0, 1.0, 6.0, 6.1, sample_shape=(128,))

    assert jnp.array_equal(first, repeated)
    assert not jnp.array_equal(first, different)


def test_truncated_normal_rng_supports_empty_sample_dimension() -> None:
    result = truncated_normal_rng(
        jax.random.key(0),
        jnp.zeros(2),
        1.0,
        -1.0,
        2.0,
        sample_shape=(0, 3),
    )

    assert result.shape == (0, 3, 2)


def test_truncated_normal_rng_rejects_invalid_parameters() -> None:
    locations = jnp.array([0.0, jnp.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
    scales = jnp.array([1.0, 1.0, 0.0, -1.0, jnp.inf, 1.0, 1.0])
    lowers = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, jnp.nan])
    uppers = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])

    result = truncated_normal_rng(
        jax.random.key(0),
        locations,
        scales,
        lowers,
        uppers,
        sample_shape=(4,),
    )

    assert jnp.all(jnp.isfinite(result[:, 0]))
    assert jnp.all(jnp.isnan(result[:, 1:]))


def test_truncated_normal_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\), \(\), \(\)\)",
    ):
        truncated_normal_rng(
            jax.random.key(0),
            jnp.zeros(2),
            jnp.ones(3),
            -1.0,
            1.0,
        )


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


def _scipy_log_probability(value, location, scale, lower, upper, *, upper_tail: bool) -> np.ndarray:
    value_array = np.asarray(value, dtype=np.float64)
    location_array = np.asarray(location, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    standardized_lower = (lower_array - location_array) / scale_array
    standardized_upper = (upper_array - location_array) / scale_array
    function = stats.truncnorm.logsf if upper_tail else stats.truncnorm.logcdf
    return function(
        value_array,
        standardized_lower,
        standardized_upper,
        loc=location_array,
        scale=scale_array,
    )


def _scipy_log_probability_gradient(arguments, *, upper_tail: bool) -> np.ndarray:
    value, location, scale, lower, upper = np.asarray(arguments, dtype=np.float64)
    standardized_value = (value - location) / scale
    standardized_lower = (lower - location) / scale
    standardized_upper = (upper - location) / scale
    log_density = stats.truncnorm.logpdf(
        standardized_value,
        standardized_lower,
        standardized_upper,
    )
    log_cdf = stats.truncnorm.logcdf(
        standardized_value,
        standardized_lower,
        standardized_upper,
    )
    log_survival = stats.truncnorm.logsf(
        standardized_value,
        standardized_lower,
        standardized_upper,
    )
    lower_log_weight = stats.truncnorm.logpdf(
        standardized_lower,
        standardized_lower,
        standardized_upper,
    )
    upper_log_weight = stats.truncnorm.logpdf(
        standardized_upper,
        standardized_lower,
        standardized_upper,
    )

    if upper_tail:
        value_derivative = -np.exp(log_density - log_survival)
        lower_derivative = np.exp(lower_log_weight)
        upper_derivative = np.exp(upper_log_weight + log_cdf - log_survival)
    else:
        value_derivative = np.exp(log_density - log_cdf)
        lower_derivative = -np.exp(lower_log_weight + log_survival - log_cdf)
        upper_derivative = -np.exp(upper_log_weight)

    lower_scale_derivative = 0.0 if np.isneginf(standardized_lower) else standardized_lower * lower_derivative
    upper_scale_derivative = 0.0 if np.isposinf(standardized_upper) else standardized_upper * upper_derivative
    return np.array(
        [
            value_derivative / scale,
            -(value_derivative + lower_derivative + upper_derivative) / scale,
            -(standardized_value * value_derivative + lower_scale_derivative + upper_scale_derivative) / scale,
            lower_derivative / scale,
            upper_derivative / scale,
        ]
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
