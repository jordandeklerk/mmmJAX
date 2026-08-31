"""Tests for Cauchy distribution functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax.distributions._cauchy import cauchy, cauchy_logpdf, cauchy_rng


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


def test_cauchy_empty_batch_preserves_parameter_validation() -> None:
    values = jnp.empty((0,))

    assert cauchy(values, 0.0, 1.0) == 0
    assert jnp.isnan(cauchy(values, 0.0, 0.0))
    assert jnp.isnan(jax.jit(cauchy)(values, jnp.inf, 1.0))


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
