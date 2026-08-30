"""Tests for Uniform distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import uniform, uniform_logpdf, uniform_rng
from mmmjax.distributions._uniform import uniform_logcdf, uniform_logsf


def test_uniform_logpdf_matches_known_values_and_support() -> None:
    values = jnp.array([-3.0, -2.0, 0.0, 3.0, 4.0, jnp.nan], dtype=jnp.float32)
    expected = jnp.array(
        [-jnp.inf, -1.6094379124341003, -1.6094379124341003, -1.6094379124341003, -jnp.inf, jnp.nan],
        dtype=jnp.float32,
    )

    result = uniform_logpdf(values, -2.0, 3.0)

    assert jnp.allclose(result, expected, equal_nan=True)


def test_uniform_returns_scalar_sum() -> None:
    result = uniform(jnp.array([0.0, 1.0]), -2.0, 3.0)

    assert result.shape == ()
    assert jnp.allclose(result, -3.2188758248682006)


def test_uniform_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [3.0]])
    lowers = jnp.array([-1.0, 1.0, 2.0])
    uppers = jnp.array([1.0, 5.0, 10.0])
    expected = jnp.array(
        [
            [-0.6931471805599453, -jnp.inf, -jnp.inf],
            [-jnp.inf, -1.3862943611198906, -2.0794415416798357],
        ]
    )

    result = uniform_logpdf(values, lowers, uppers)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected)
    assert jnp.allclose(uniform(values, lowers, uppers), jnp.sum(expected))


def test_uniform_logpdf_returns_negative_infinity_for_infinite_values() -> None:
    result = uniform_logpdf(jnp.array([-jnp.inf, jnp.inf]), -2.0, 3.0)

    assert jnp.all(jnp.isneginf(result))


def test_uniform_logpdf_rejects_invalid_bounds_before_support_check() -> None:
    lowers = jnp.array([0.0, 1.0, -jnp.inf, 0.0, jnp.nan])
    uppers = jnp.array([0.0, 0.0, 1.0, jnp.inf, 1.0])

    result = uniform_logpdf(100.0, lowers, uppers)

    assert jnp.all(jnp.isnan(result))


def test_uniform_rejects_bounds_that_collapse_after_dtype_promotion() -> None:
    lower = jnp.int32(16_777_216)
    upper = jnp.int32(16_777_217)

    pointwise_result = uniform_logpdf(lower, lower, upper)
    empty_batch_result = uniform(jnp.empty((0,)), lower, upper)

    assert jnp.isnan(pointwise_result)
    assert jnp.isnan(empty_batch_result)


def test_uniform_logpdf_handles_opposite_sign_bounds_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)

    result = uniform_logpdf(0.0, -maximum, maximum)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, -89.4159862326283)


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        pytest.param(uniform_logcdf, stats.uniform.logcdf, id="logcdf"),
        pytest.param(uniform_logsf, stats.uniform.logsf, id="logsf"),
    ],
)
def test_uniform_log_probabilities_match_scipy(function, reference) -> None:
    values = np.array([-np.inf, -3.0, -2.0, 0.0, 3.0, 4.0, np.inf], dtype=np.float32)
    expected = reference(values.astype(np.float64), loc=-2.0, scale=5.0)

    result = function(values, -2.0, 3.0)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_uniform_log_probabilities_are_complements() -> None:
    values = jnp.array([-jnp.inf, -2.0, -1.0, 0.5, 2.0, 3.0, jnp.inf])

    log_cdf = uniform_logcdf(values, -2.0, 3.0)
    log_survival = uniform_logsf(values, -2.0, 3.0)

    assert jnp.allclose(
        jnp.logaddexp(log_cdf, log_survival),
        0,
        atol=jnp.finfo(log_cdf.dtype).eps,
    )


def test_uniform_log_probabilities_broadcast_arguments() -> None:
    values = jnp.array([[0.0], [3.0]])
    lowers = jnp.array([-1.0, 1.0, 2.0])
    uppers = jnp.array([1.0, 5.0, 10.0])

    log_cdf = uniform_logcdf(values, lowers, uppers)
    log_survival = uniform_logsf(values, lowers, uppers)

    assert log_cdf.shape == (2, 3)
    assert log_survival.shape == (2, 3)
    assert jnp.allclose(
        jnp.logaddexp(log_cdf, log_survival),
        0,
        atol=jnp.finfo(log_cdf.dtype).eps,
    )


def test_uniform_log_probabilities_propagate_nan_and_handle_infinite_values() -> None:
    values = jnp.array([-jnp.inf, -2.0, 3.0, jnp.inf, jnp.nan])

    log_cdf = uniform_logcdf(values, -2.0, 3.0)
    log_survival = uniform_logsf(values, -2.0, 3.0)

    assert jnp.all(jnp.isneginf(log_cdf[:2]))
    assert jnp.all(log_cdf[2:4] == 0)
    assert jnp.isnan(log_cdf[4])
    assert jnp.all(log_survival[:2] == 0)
    assert jnp.all(jnp.isneginf(log_survival[2:4]))
    assert jnp.isnan(log_survival[4])


@pytest.mark.parametrize("function", [uniform_logcdf, uniform_logsf])
def test_uniform_log_probabilities_reject_invalid_bounds_before_support(function) -> None:
    lowers = jnp.array([0.0, 1.0, -jnp.inf, 0.0, jnp.nan])
    uppers = jnp.array([0.0, 0.0, 1.0, jnp.inf, 1.0])

    result = function(100.0, lowers, uppers)

    assert jnp.all(jnp.isnan(result))


@pytest.mark.parametrize("function", [uniform_logcdf, uniform_logsf])
def test_uniform_log_probabilities_reject_bounds_that_collapse_after_dtype_promotion(function) -> None:
    lower = jnp.int32(16_777_216)
    upper = jnp.int32(16_777_217)

    result = function(lower, lower, upper)

    assert jnp.isnan(result)


def test_uniform_log_probabilities_handle_opposite_sign_bounds_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    values = maximum * jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected_log_cdf = jnp.log(jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    expected_log_survival = jnp.log(jnp.array([1.0, 0.75, 0.5, 0.25, 0.0]))

    log_cdf = uniform_logcdf(values, -maximum, maximum)
    log_survival = uniform_logsf(values, -maximum, maximum)

    assert jnp.all(jnp.isfinite(log_cdf[1:]))
    assert jnp.all(jnp.isfinite(log_survival[:-1]))
    assert jnp.allclose(log_cdf, expected_log_cdf)
    assert jnp.allclose(log_survival, expected_log_survival)


def test_uniform_log_probabilities_preserve_values_next_to_extreme_bounds() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    values = jnp.array(
        [
            jnp.nextafter(-maximum, maximum),
            jnp.nextafter(maximum, -maximum),
        ]
    )
    maximum_float64 = float(maximum)
    values_float64 = np.asarray(values, dtype=np.float64)
    expected_log_cdf = np.log((values_float64 + maximum_float64) / (2 * maximum_float64))
    expected_log_survival = np.log((maximum_float64 - values_float64) / (2 * maximum_float64))

    log_cdf = uniform_logcdf(values, -maximum, maximum)
    log_survival = uniform_logsf(values, -maximum, maximum)

    np.testing.assert_allclose(log_cdf, expected_log_cdf, rtol=3e-6, atol=0)
    np.testing.assert_allclose(log_survival, expected_log_survival, rtol=3e-6, atol=0)


@pytest.mark.parametrize(
    ("function", "expected_gradient", "expected_hessian"),
    [
        pytest.param(
            uniform_logcdf,
            [0.4, -0.2, -0.2],
            [[-0.16, 0.16, 0.0], [0.16, -0.12, -0.04], [0.0, -0.04, 0.04]],
            id="logcdf",
        ),
        pytest.param(
            uniform_logsf,
            [-0.4, 0.2, 0.2],
            [[-0.16, 0.0, 0.16], [0.0, 0.04, -0.04], [0.16, -0.04, -0.12]],
            id="logsf",
        ),
    ],
)
def test_uniform_log_probability_derivatives_match_closed_form(
    function,
    expected_gradient,
    expected_hessian,
) -> None:
    arguments = jnp.array([0.5, -2.0, 3.0])

    def evaluate(current):
        return function(current[0], current[1], current[2])

    forward = jax.jit(jax.jacfwd(evaluate))(arguments)
    reverse = jax.jit(jax.jacrev(evaluate))(arguments)
    hessian = jax.jit(jax.hessian(evaluate))(arguments)

    assert jnp.allclose(forward, jnp.asarray(expected_gradient))
    assert jnp.allclose(reverse, jnp.asarray(expected_gradient))
    assert jnp.allclose(hessian, jnp.asarray(expected_hessian))


@pytest.mark.parametrize(("lower", "upper"), [(-1.0, 2.0), (2.0, 6.0)])
def test_uniform_is_differentiable_with_respect_to_value_and_bounds(lower: float, upper: float) -> None:
    values = lower + (upper - lower) * jnp.array([0.25, 0.5, 0.75])
    expected_bound_gradient = values.size / (upper - lower)

    value_gradient = jax.grad(lambda current_values: uniform(current_values, lower, upper))(values)
    lower_gradient = jax.grad(lambda current_lower: uniform(values, current_lower, upper))(lower)
    upper_gradient = jax.grad(lambda current_upper: uniform(values, lower, current_upper))(upper)

    assert jnp.array_equal(value_gradient, jnp.zeros_like(values))
    assert jnp.allclose(lower_gradient, expected_bound_gradient)
    assert jnp.allclose(upper_gradient, -expected_bound_gradient)


def test_uniform_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-0.5, 0.0], [2.0, 3.0]])
    lowers = jnp.array([-1.0, 1.0])
    uppers = jnp.array([1.0, 5.0])

    result = jax.vmap(uniform)(values, lowers, uppers)
    expected = jnp.stack(
        [uniform(value, lower, upper) for value, lower, upper in zip(values, lowers, uppers, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_uniform_rng_matches_jax_sampler_for_moderate_bounds() -> None:
    key = jax.random.key(42)
    lowers = jnp.array([2.0, -5.0], dtype=jnp.float32)
    uppers = jnp.array([5.0, -1.0], dtype=jnp.float32)
    expected = jax.random.uniform(
        key,
        shape=(3, 2),
        dtype=jnp.float32,
        minval=lowers,
        maxval=uppers,
    )

    result = uniform_rng(key, lowers, uppers, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_uniform_rng_uses_broadcast_parameter_shape() -> None:
    lowers = jnp.array([[-2.0], [10.0]])
    uppers = jnp.array([11.0, 12.0, 20.0])

    result = uniform_rng(jax.random.key(0), lowers, uppers, sample_shape=(4,))

    assert result.shape == (4, 2, 3)
    assert jnp.all(result >= lowers)
    assert jnp.all(result < uppers)


def test_uniform_rng_matches_distribution_moments() -> None:
    samples = uniform_rng(jax.random.key(7), -2.0, 3.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 0.5, rtol=0, atol=0.025)
    assert jnp.allclose(jnp.var(samples), 25 / 12, rtol=0, atol=0.035)


def test_uniform_rng_handles_opposite_sign_bounds_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)

    samples = uniform_rng(jax.random.key(0), -maximum, maximum, sample_shape=(2_000,))

    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= -maximum)
    assert jnp.all(samples < maximum)
    assert jnp.any(samples < 0)
    assert jnp.any(samples > 0)


@pytest.mark.parametrize(("lower", "upper"), [(-2.0, 3.0), (2.0, 5.0)])
def test_uniform_rng_is_differentiable_with_respect_to_bounds(lower: float, upper: float) -> None:
    key = jax.random.key(5)
    unit_samples = jax.random.uniform(key, shape=(16,))

    lower_gradient = jax.grad(
        lambda current_lower: jnp.sum(uniform_rng(key, current_lower, upper, sample_shape=(16,)))
    )(lower)
    upper_gradient = jax.grad(
        lambda current_upper: jnp.sum(uniform_rng(key, lower, current_upper, sample_shape=(16,)))
    )(upper)

    assert jnp.allclose(lower_gradient, jnp.sum(1 - unit_samples))
    assert jnp.allclose(upper_gradient, jnp.sum(unit_samples))


def test_uniform_rng_can_be_jitted_with_dynamic_bounds() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)
    lowers = jnp.array([-maximum, 2.0, -5.0])
    uppers = jnp.array([maximum, 5.0, -1.0])
    compiled = jax.jit(lambda key, lower, upper: uniform_rng(key, lower, upper, sample_shape=(32,)))

    samples = compiled(jax.random.key(0), lowers, uppers)

    assert samples.shape == (32, 3)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= lowers)
    assert jnp.all(samples < uppers)


def test_uniform_rng_excludes_an_adjacent_upper_bound() -> None:
    lower = jnp.float32(1.0)
    upper = jnp.nextafter(lower, jnp.float32(jnp.inf))

    samples = uniform_rng(jax.random.key(0), lower, upper, sample_shape=(100,))

    assert jnp.all(samples == lower)
    assert jnp.all(samples < upper)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_uniform_handles_opposite_sign_float64_bounds_at_finite_maximum() -> None:
    maximum = jnp.asarray(jnp.finfo(jnp.float64).max)

    log_density = jax.jit(uniform_logpdf)(jnp.float64(0.0), -maximum, maximum)
    samples = uniform_rng(jax.random.key(0), -maximum, maximum, sample_shape=(500,))

    assert jnp.allclose(log_density, -710.4758600739439)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= -maximum)
    assert jnp.all(samples < maximum)


def test_uniform_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        uniform_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))
