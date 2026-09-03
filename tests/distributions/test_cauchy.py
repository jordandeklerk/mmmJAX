"""Tests for Cauchy distribution functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import cauchy, cauchy_logpdf, cauchy_rng
from mmmjax.distributions._cauchy import cauchy_logcdf, cauchy_logsf


@pytest.mark.parametrize(
    ("value", "location", "scale", "expected"),
    [
        (1.0, 0.0, 1.0, -1.8378770664093453),
        (-1.5, 0.0, 1.0, -2.3233848821910463),
        (-1.5, -1.0, 1.0, -1.3678734371636099),
    ],
)
def test_cauchy_logpdf_matches_stan_reference_values(
    value: float,
    location: float,
    scale: float,
    expected: float,
) -> None:
    result = cauchy_logpdf(value, location, scale)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_cauchy_logpdf_matches_scipy_reference_grid() -> None:
    values = np.array([[-5.0], [0.0], [4.0]], dtype=np.float32)
    locations = np.array([-2.0, 0.5, 2.3], dtype=np.float32)
    scales = np.array([0.25, 0.5, 3.0], dtype=np.float32)
    expected = stats.cauchy.logpdf(
        values.astype(np.float64),
        loc=locations.astype(np.float64),
        scale=scales.astype(np.float64),
    )

    result = cauchy_logpdf(values, locations, scales)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=3e-6, atol=3e-6)


def test_cauchy_logpdf_handles_float32_finite_limits() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    minimum_normal = jnp.asarray(jnp.finfo(jnp.float32).tiny)
    log_maximum = jnp.log(maximum)
    log_pi = jnp.asarray(math.log(math.pi), dtype=jnp.float32)

    results = jnp.stack(
        (
            cauchy_logpdf(maximum, 0.0, 1.0),
            cauchy_logpdf(1.0, 0.0, minimum_normal),
            cauchy_logpdf(maximum, -maximum, maximum),
            cauchy_logpdf(maximum, -maximum, 1.0),
        )
    )
    compiled_results = jax.jit(
        lambda: jnp.stack(
            (
                cauchy_logpdf(maximum, 0.0, 1.0),
                cauchy_logpdf(1.0, 0.0, minimum_normal),
                cauchy_logpdf(maximum, -maximum, maximum),
                cauchy_logpdf(maximum, -maximum, 1.0),
            )
        )
    )()
    expected = jnp.stack(
        (
            -log_pi - 2 * log_maximum,
            jnp.log(minimum_normal) - log_pi,
            -log_pi - log_maximum - jnp.log1p(jnp.float32(4)),
            -log_pi - 2 * (jnp.log(jnp.float32(2)) + log_maximum),
        )
    )

    assert jnp.all(jnp.isfinite(results))
    assert jnp.allclose(results, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(compiled_results, expected, rtol=3e-6, atol=0)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_cauchy_logpdf_handles_float64_finite_limits() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float64).max)
    minimum_normal = jnp.asarray(jnp.finfo(jnp.float64).tiny)
    log_maximum = jnp.log(maximum)
    log_pi = jnp.asarray(math.log(math.pi), dtype=jnp.float64)

    results = jnp.stack(
        (
            cauchy_logpdf(maximum, 0.0, 1.0),
            cauchy_logpdf(1.0, 0.0, minimum_normal),
            cauchy_logpdf(maximum, -maximum, maximum),
            cauchy_logpdf(maximum, -maximum, 1.0),
        )
    )
    compiled_results = jax.jit(
        lambda: jnp.stack(
            (
                cauchy_logpdf(maximum, 0.0, 1.0),
                cauchy_logpdf(1.0, 0.0, minimum_normal),
                cauchy_logpdf(maximum, -maximum, maximum),
                cauchy_logpdf(maximum, -maximum, 1.0),
            )
        )
    )()
    expected = jnp.stack(
        (
            -log_pi - 2 * log_maximum,
            jnp.log(minimum_normal) - log_pi,
            -log_pi - log_maximum - jnp.log1p(jnp.float64(4)),
            -log_pi - 2 * (jnp.log(jnp.float64(2)) + log_maximum),
        )
    )

    assert jnp.all(jnp.isfinite(results))
    assert jnp.allclose(results, expected, rtol=1e-14, atol=0)
    assert jnp.allclose(compiled_results, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (jnp.inf, -jnp.inf),
        (-jnp.inf, -jnp.inf),
        (jnp.nan, jnp.nan),
    ],
)
def test_cauchy_logpdf_handles_nonfinite_values(value, expected) -> None:
    result = cauchy_logpdf(value, 0.0, 1.0)

    if jnp.isnan(expected):
        assert jnp.isnan(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("location", "scale"),
    [
        (jnp.inf, 1.0),
        (-jnp.inf, 1.0),
        (jnp.nan, 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, jnp.inf),
        (0.0, jnp.nan),
    ],
)
def test_cauchy_logpdf_rejects_invalid_parameters(location, scale) -> None:
    result = cauchy_logpdf(0.0, location, scale)

    assert jnp.isnan(result)


def test_cauchy_sums_broadcast_log_densities() -> None:
    values = jnp.array([[-2.0], [0.0], [3.0]])
    locations = jnp.array([-0.5, 1.0])
    scales = jnp.array([0.75, 2.0])

    result = cauchy(values, locations, scales)
    expected = jnp.sum(cauchy_logpdf(values, locations, scales))

    assert result.shape == ()
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        pytest.param(cauchy_logcdf, stats.cauchy.logcdf, id="logcdf"),
        pytest.param(cauchy_logsf, stats.cauchy.logsf, id="logsf"),
    ],
)
def test_cauchy_log_probabilities_match_scipy(function, reference) -> None:
    values = np.array([[-np.inf], [-5.0], [0.5], [4.0], [np.inf]], dtype=np.float32)
    locations = np.array([-2.0, 0.5, 2.3], dtype=np.float32)
    scales = np.array([0.25, 0.5, 3.0], dtype=np.float32)
    expected = reference(
        values.astype(np.float64),
        loc=locations.astype(np.float64),
        scale=scales.astype(np.float64),
    )

    result = function(values, locations, scales)
    compiled = jax.jit(function)(values, locations, scales)

    assert result.shape == (5, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-7)


def test_cauchy_log_probabilities_are_complements() -> None:
    values = jnp.array([-jnp.inf, -10.0, -1.0, 0.5, 3.0, 12.0, jnp.inf])

    log_cdf = cauchy_logcdf(values, 0.5, 1.75)
    log_survival = cauchy_logsf(values, 0.5, 1.75)

    assert jnp.allclose(
        jnp.logaddexp(log_cdf, log_survival),
        0,
        atol=jnp.finfo(log_cdf.dtype).eps,
    )
    assert jnp.allclose(log_cdf[3], -jnp.log(2))
    assert jnp.allclose(log_survival[3], -jnp.log(2))


@pytest.mark.parametrize(
    ("function", "value", "reference"),
    [
        pytest.param(cauchy_logcdf, -1e8, stats.cauchy.logcdf, id="logcdf-lower-tail"),
        pytest.param(cauchy_logcdf, 1e8, stats.cauchy.logcdf, id="logcdf-upper-tail"),
        pytest.param(
            cauchy_logcdf,
            -np.finfo(np.float32).max,
            stats.cauchy.logcdf,
            id="logcdf-finite-limit",
        ),
        pytest.param(cauchy_logsf, -1e8, stats.cauchy.logsf, id="logsf-lower-tail"),
        pytest.param(cauchy_logsf, 1e8, stats.cauchy.logsf, id="logsf-upper-tail"),
        pytest.param(
            cauchy_logsf,
            np.finfo(np.float32).max,
            stats.cauchy.logsf,
            id="logsf-finite-limit",
        ),
    ],
)
def test_cauchy_log_probabilities_remain_accurate_in_deep_float32_tails(
    function,
    value: float,
    reference,
) -> None:
    expected = reference(value)

    result = function(jnp.float32(value), jnp.float32(0), jnp.float32(1))
    compiled = jax.jit(function)(jnp.float32(value), jnp.float32(0), jnp.float32(1))

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=0)


def test_cauchy_log_probabilities_handle_opposite_values_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    values = jnp.array([maximum, -maximum])
    locations = jnp.array([-maximum, maximum])
    expected_log_cdf = stats.cauchy.logcdf([2.0, -2.0])
    expected_log_survival = stats.cauchy.logsf([2.0, -2.0])

    log_cdf = jax.jit(cauchy_logcdf)(values, locations, maximum)
    log_survival = jax.jit(cauchy_logsf)(values, locations, maximum)

    np.testing.assert_allclose(log_cdf, expected_log_cdf, rtol=3e-6, atol=0)
    np.testing.assert_allclose(log_survival, expected_log_survival, rtol=3e-6, atol=0)


@pytest.mark.parametrize("function", [cauchy_logcdf, cauchy_logsf])
def test_cauchy_log_probabilities_reject_invalid_parameters(function) -> None:
    locations = jnp.array([jnp.inf, -jnp.inf, jnp.nan])
    scales = jnp.array([0.0, -1.0, jnp.inf, -jnp.inf, jnp.nan])

    invalid_locations = function(0.0, locations, 1.0)
    invalid_scales = function(0.0, 0.0, scales)

    assert jnp.all(jnp.isnan(invalid_locations))
    assert jnp.all(jnp.isnan(invalid_scales))


def test_cauchy_log_probabilities_handle_nonfinite_values() -> None:
    values = jnp.array([-jnp.inf, jnp.inf, jnp.nan])

    log_cdf = cauchy_logcdf(values, 0.0, 1.0)
    log_survival = cauchy_logsf(values, 0.0, 1.0)

    assert jnp.isneginf(log_cdf[0])
    assert log_cdf[1] == 0
    assert jnp.isnan(log_cdf[2])
    assert log_survival[0] == 0
    assert jnp.isneginf(log_survival[1])
    assert jnp.isnan(log_survival[2])


@pytest.mark.parametrize(
    ("function", "reference", "direction"),
    [
        pytest.param(cauchy_logcdf, stats.cauchy.logcdf, 1, id="logcdf"),
        pytest.param(cauchy_logsf, stats.cauchy.logsf, -1, id="logsf"),
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(jnp.array([0.0, 0.0, 1.0]), id="center"),
        pytest.param(jnp.array([0.4, -0.3, 1.7]), id="central"),
        pytest.param(jnp.array([5e19, 0.0, 1e20]), id="large-scale"),
        pytest.param(jnp.array([5e-21, 0.0, 1e-20]), id="small-scale"),
        pytest.param(jnp.array([-1e8, 0.0, 1.0]), id="lower-tail"),
        pytest.param(jnp.array([1e8, 0.0, 1.0]), id="upper-tail"),
        pytest.param(jnp.array([-1e20, 0.0, 1.0]), id="deep-lower-tail"),
        pytest.param(jnp.array([1e20, 0.0, 1.0]), id="deep-upper-tail"),
    ],
)
def test_cauchy_log_probability_derivatives_match_closed_form(
    function,
    reference,
    direction: int,
    arguments: jax.Array,
) -> None:
    value, location, scale = np.asarray(arguments, dtype=np.float64)
    standardized = (value - location) / scale
    log_density = stats.cauchy.logpdf(value, loc=location, scale=scale)
    log_probability = reference(value, loc=location, scale=scale)
    density_ratio = np.exp(log_density - log_probability)
    expected = jnp.array(
        [
            direction * density_ratio,
            -direction * density_ratio,
            -direction * standardized * density_ratio,
        ]
    )

    def evaluate(current):
        return function(current[0], current[1], current[2])

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)
    absolute_tolerance = np.finfo(arguments.dtype).tiny

    np.testing.assert_allclose(forward, expected, rtol=3e-6, atol=absolute_tolerance)
    np.testing.assert_allclose(reverse, expected, rtol=3e-6, atol=absolute_tolerance)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(jnp.array([0.4, 0.4, 0.8]), id="center"),
        pytest.param(jnp.array([0.6, 0.4, 0.8]), id="central"),
        pytest.param(jnp.array([1.25, 0.25, 1.0]), id="boundary"),
        pytest.param(jnp.array([2.0, 0.25, 0.5]), id="same-sign-tail"),
        pytest.param(jnp.array([1.25, -0.4, 0.8]), id="positive-tail"),
        pytest.param(jnp.array([-1.25, 0.4, 0.8]), id="negative-tail"),
    ],
)
def test_cauchy_logpdf_derivatives_match_stan(arguments: jax.Array) -> None:
    value, location, scale = arguments
    residual = value - location
    denominator = jnp.square(scale) + jnp.square(residual)
    expected = jnp.array(
        [
            -2 * residual / denominator,
            2 * residual / denominator,
            (jnp.square(residual) - jnp.square(scale)) / (scale * denominator),
        ]
    )

    def evaluate(current):
        return cauchy_logpdf(current[0], current[1], current[2])

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(arguments)

    assert jnp.allclose(forward_gradient, expected)
    assert jnp.allclose(reverse_gradient, expected)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(jnp.array([0.0, 0.0, 1e-20]), id="small-center"),
        pytest.param(jnp.array([0.5e-20, 0.0, 1e-20]), id="small-central"),
        pytest.param(jnp.array([0.5e20, 0.0, 1e20]), id="large-central"),
    ],
)
def test_cauchy_logpdf_derivatives_remain_finite_across_float32_scales(arguments: jax.Array) -> None:
    value, location, scale = arguments
    standardized = (value - location) / scale
    expected = jnp.array(
        [
            -2 * standardized / (scale * (1 + jnp.square(standardized))),
            2 * standardized / (scale * (1 + jnp.square(standardized))),
            (jnp.square(standardized) - 1) / (scale * (1 + jnp.square(standardized))),
        ]
    )

    def evaluate(current):
        return cauchy_logpdf(current[0], current[1], current[2])

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(arguments)

    assert jnp.all(jnp.isfinite(forward_gradient))
    assert jnp.all(jnp.isfinite(reverse_gradient))
    assert jnp.allclose(forward_gradient, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(reverse_gradient, expected, rtol=3e-6, atol=0)


def test_cauchy_logpdf_hessian_ignores_inactive_tail_work() -> None:
    arguments = jnp.array([1e-19, 0.0, 1e-16], dtype=jnp.float32)
    value, location, scale = arguments
    standardized = (value - location) / scale
    inverse_scale_squared = jnp.square(1 / scale)
    squared_standardized = jnp.square(standardized)
    denominator = jnp.square(1 + squared_standardized)
    residual_curvature = 2 * (squared_standardized - 1) * inverse_scale_squared / denominator
    cross_curvature = 4 * standardized * inverse_scale_squared / denominator
    scale_curvature = (
        (1 - 4 * squared_standardized - jnp.square(squared_standardized)) * inverse_scale_squared / denominator
    )
    expected = jnp.array(
        [
            [residual_curvature, -residual_curvature, cross_curvature],
            [-residual_curvature, residual_curvature, -cross_curvature],
            [cross_curvature, -cross_curvature, scale_curvature],
        ]
    )

    def evaluate(current):
        return cauchy_logpdf(current[0], current[1], current[2])

    forward_reverse = jax.jit(jax.jacfwd(jax.jacrev(evaluate)))(arguments)
    reverse_reverse = jax.jit(jax.jacrev(jax.jacrev(evaluate)))(arguments)

    assert jnp.all(jnp.isfinite(forward_reverse))
    assert jnp.all(jnp.isfinite(reverse_reverse))
    assert jnp.allclose(forward_reverse, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(reverse_reverse, expected, rtol=3e-6, atol=0)


def test_cauchy_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-2.0, 0.0], [1.0, 4.0]])
    locations = jnp.array([-0.5, 2.0])
    scales = jnp.array([0.75, 1.5])

    result = jax.vmap(cauchy)(values, locations, scales)
    expected = jnp.stack(
        [cauchy(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_cauchy_rng_matches_transformed_jax_draws() -> None:
    key = jax.random.key(42)
    locations = jnp.array([[1.0], [-2.0]], dtype=jnp.float32)
    scales = jnp.array([0.5, 2.0, 1.5], dtype=jnp.float32)
    expected = locations + scales * jax.random.cauchy(key, shape=(4, 2, 3), dtype=jnp.float32)

    result = cauchy_rng(key, locations, scales, sample_shape=(4,))

    assert result.shape == (4, 2, 3)
    assert jnp.array_equal(result, expected)


def test_cauchy_rng_matches_central_probability() -> None:
    samples = cauchy_rng(jax.random.key(7), 0.0, 1.0, sample_shape=(50_000,))
    central_proportion = jnp.mean(jnp.abs(samples) <= 1)

    assert jnp.allclose(central_proportion, 0.5, rtol=0, atol=0.01)


@pytest.mark.parametrize(
    ("location", "scale"),
    [
        (jnp.inf, 1.0),
        (-jnp.inf, 1.0),
        (jnp.nan, 1.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, jnp.inf),
        (0.0, -jnp.inf),
        (0.0, jnp.nan),
    ],
)
def test_cauchy_rng_rejects_invalid_parameters(location, scale) -> None:
    result = cauchy_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert jnp.all(jnp.isnan(result))


def test_cauchy_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        cauchy_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))
