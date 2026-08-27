"""Tests for shared distribution contracts."""

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
    inverse_gamma,
    inverse_gamma_logpdf,
    inverse_gamma_rng,
    laplace,
    laplace_logpdf,
    laplace_rng,
    lognormal,
    lognormal_logpdf,
    lognormal_rng,
    normal,
    normal_logpdf,
    normal_rng,
    student_t,
    student_t_logpdf,
    student_t_rng,
    uniform,
    uniform_logpdf,
    uniform_rng,
)


@pytest.mark.parametrize(
    ("density", "arguments"),
    [
        (normal, (jnp.empty((0,)), 0.0, 1.0)),
        (half_normal, (jnp.empty((0,)), 1.0)),
        (lognormal, (jnp.empty((0,)), 0.0, 1.0)),
        (exponential, (jnp.empty((0,)), 1.0)),
        (gamma, (jnp.empty((0,)), 1.0, 1.0)),
        (beta, (jnp.empty((0,)), 1.0, 1.0)),
        (inverse_gamma, (jnp.empty((0,)), 1.0, 1.0)),
        (laplace, (jnp.empty((0,)), 0.0, 1.0)),
        (student_t, (jnp.empty((0,)), 5.0, 0.0, 1.0)),
        (uniform, (jnp.empty((0,)), 0.0, 1.0)),
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
        (inverse_gamma, (jnp.empty((0,)), 0.0, 1.0)),
        (inverse_gamma, (jnp.empty((0,)), 1.0, jnp.inf)),
        (inverse_gamma, (jnp.empty((0,)), jnp.empty((0,)), jnp.inf)),
        (inverse_gamma, (jnp.empty((0,)), 0.0, jnp.empty((0,)))),
        (laplace, (jnp.empty((0,)), 0.0, 0.0)),
        (laplace, (jnp.empty((0,)), jnp.inf, 1.0)),
        (laplace, (jnp.empty((0,)), jnp.empty((0,)), jnp.inf)),
        (laplace, (jnp.empty((0,)), jnp.inf, jnp.empty((0,)))),
        (student_t, (jnp.empty((0,)), 0.0, 0.0, 1.0)),
        (student_t, (jnp.empty((0,)), 5.0, jnp.inf, 1.0)),
        (student_t, (jnp.empty((0,)), 5.0, 0.0, jnp.inf)),
        (uniform, (jnp.empty((0,)), 0.0, 0.0)),
        (uniform, (jnp.empty((0,)), -jnp.inf, 1.0)),
        (uniform, (jnp.empty((0,)), jnp.empty((0,)), jnp.inf)),
        (uniform, (jnp.empty((0,)), -jnp.inf, jnp.empty((0,)))),
    ],
)
def test_invalid_parameters_remain_nan_for_empty_batch(density, arguments) -> None:
    eager = density(*arguments)
    compiled = jax.jit(density)(*arguments)

    assert jnp.isnan(eager)
    assert jnp.isnan(compiled)


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
    assert inverse_gamma_logpdf(values + 1, dtype(1.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert laplace_logpdf(values, dtype(0.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert student_t_logpdf(values, dtype(5.0), dtype(0.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)
    assert uniform_logpdf(values, dtype(0.0), dtype(1.0)).dtype == jnp.dtype(expected_dtype)


def test_densities_promote_integer_inputs_to_float32() -> None:
    values = jnp.array([0, 1], dtype=jnp.int32)

    assert normal_logpdf(values, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert half_normal_logpdf(values, 1).dtype == jnp.dtype(jnp.float32)
    assert lognormal_logpdf(values + 1, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert exponential_logpdf(values, 1).dtype == jnp.dtype(jnp.float32)
    assert gamma_logpdf(values + 1, 1, 1).dtype == jnp.dtype(jnp.float32)
    assert beta_logpdf(values, 1, 1).dtype == jnp.dtype(jnp.float32)
    assert inverse_gamma_logpdf(values + 1, 1, 1).dtype == jnp.dtype(jnp.float32)
    assert laplace_logpdf(values, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert student_t_logpdf(values, 5, 0, 1).dtype == jnp.dtype(jnp.float32)
    assert uniform_logpdf(values, 0, 1).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("logpdf", "logpdf_arguments", "rng", "rng_arguments"),
    [
        pytest.param(normal_logpdf, (0.0, 0.0, 1.0), normal_rng, (0.0, 1.0), id="normal"),
        pytest.param(half_normal_logpdf, (0.0, 1.0), half_normal_rng, (1.0,), id="half-normal"),
        pytest.param(lognormal_logpdf, (1.0, 0.0, 1.0), lognormal_rng, (0.0, 1.0), id="lognormal"),
        pytest.param(exponential_logpdf, (1.0, 1.0), exponential_rng, (1.0,), id="exponential"),
        pytest.param(gamma_logpdf, (1.0, 1.0, 1.0), gamma_rng, (1.0, 1.0), id="gamma"),
        pytest.param(beta_logpdf, (0.5, 1.0, 1.0), beta_rng, (1.0, 1.0), id="beta"),
        pytest.param(
            inverse_gamma_logpdf,
            (1.0, 1.0, 1.0),
            inverse_gamma_rng,
            (1.0, 1.0),
            id="inverse-gamma",
        ),
        pytest.param(laplace_logpdf, (0.0, 0.0, 1.0), laplace_rng, (0.0, 1.0), id="laplace"),
        pytest.param(
            student_t_logpdf,
            (0.0, 5.0, 0.0, 1.0),
            student_t_rng,
            (5.0, 0.0, 1.0),
            id="student-t",
        ),
        pytest.param(uniform_logpdf, (0.5, 0.0, 1.0), uniform_rng, (0.0, 1.0), id="uniform"),
    ],
)
def test_python_scalar_inputs_follow_jax_default_dtype(logpdf, logpdf_arguments, rng, rng_arguments) -> None:
    expected_dtype = jnp.asarray(0.0).dtype

    assert logpdf(*logpdf_arguments).dtype == expected_dtype
    assert rng(jax.random.key(0), *rng_arguments).dtype == expected_dtype


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
    assert inverse_gamma_logpdf(values + 1, 1.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert laplace_logpdf(values, 0.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert student_t_logpdf(values, 5.0, 0.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert uniform_logpdf(values, 0.0, 1.0).dtype == jnp.dtype(jnp.float64)
    assert normal_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert half_normal_rng(key, jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert lognormal_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert exponential_rng(key, jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert gamma_rng(key, jnp.float64(1.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert beta_rng(key, jnp.float64(1.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert inverse_gamma_rng(key, jnp.float64(1.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert laplace_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert student_t_rng(key, jnp.float64(5.0), jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)
    assert uniform_rng(key, jnp.float64(0.0), jnp.float64(1.0)).dtype == jnp.dtype(jnp.float64)


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
        (inverse_gamma_logpdf, (jnp.array([0.5, 1.0]), 2.0, 1.5)),
        (inverse_gamma, (jnp.array([0.5, 1.0]), 2.0, 1.5)),
        (laplace_logpdf, (jnp.array([0.0, 1.0]), 0.5, 2.0)),
        (laplace, (jnp.array([0.0, 1.0]), 0.5, 2.0)),
        (student_t_logpdf, (jnp.array([0.0, 1.0]), 5.0, 0.5, 2.0)),
        (student_t, (jnp.array([0.0, 1.0]), 5.0, 0.5, 2.0)),
        (uniform_logpdf, (jnp.array([0.0, 1.0]), -0.5, 2.0)),
        (uniform, (jnp.array([0.0, 1.0]), -0.5, 2.0)),
    ],
)
def test_density_functions_can_be_jitted(function, arguments) -> None:
    assert jnp.allclose(jax.jit(function)(*arguments), function(*arguments))


def test_rngs_return_scalar_for_scalar_parameters() -> None:
    key = jax.random.key(0)

    assert normal_rng(key, 0.0, 1.0).shape == ()
    assert half_normal_rng(key, 1.0).shape == ()
    assert lognormal_rng(key, 0.0, 1.0).shape == ()
    assert exponential_rng(key, 1.0).shape == ()
    assert gamma_rng(key, 1.0, 1.0).shape == ()
    assert beta_rng(key, 1.0, 1.0).shape == ()
    assert inverse_gamma_rng(key, 1.0, 1.0).shape == ()
    assert laplace_rng(key, 0.0, 1.0).shape == ()
    assert student_t_rng(key, 5.0, 0.0, 1.0).shape == ()
    assert uniform_rng(key, 0.0, 1.0).shape == ()


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
    inverse_gamma_shape_result = inverse_gamma_rng(
        key,
        jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]),
        1.0,
    )
    inverse_gamma_scale_result = inverse_gamma_rng(
        key,
        1.0,
        jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]),
    )
    laplace_result = laplace_rng(key, jnp.array([0.0, 0.0, jnp.inf]), jnp.array([1.0, 0.0, 1.0]))
    student_degrees_result = student_t_rng(
        key,
        jnp.array([5.0, 0.0, -1.0, jnp.inf, jnp.nan]),
        0.0,
        1.0,
    )
    student_location_result = student_t_rng(key, 5.0, jnp.array([0.0, jnp.inf, -jnp.inf, jnp.nan]), 1.0)
    student_scale_result = student_t_rng(
        key,
        5.0,
        0.0,
        jnp.array([1.0, 0.0, -1.0, jnp.inf, jnp.nan]),
    )
    uniform_result = uniform_rng(
        key,
        jnp.array([0.0, 0.0, 1.0, -jnp.inf, 0.0]),
        jnp.array([1.0, 0.0, 0.0, 1.0, jnp.nan]),
    )

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
    assert jnp.isfinite(inverse_gamma_shape_result[0])
    assert jnp.all(jnp.isnan(inverse_gamma_shape_result[1:]))
    assert jnp.isfinite(inverse_gamma_scale_result[0])
    assert jnp.all(jnp.isnan(inverse_gamma_scale_result[1:]))
    assert jnp.isfinite(laplace_result[0])
    assert jnp.all(jnp.isnan(laplace_result[1:]))
    assert jnp.isfinite(student_degrees_result[0])
    assert jnp.all(jnp.isnan(student_degrees_result[1:]))
    assert jnp.isfinite(student_location_result[0])
    assert jnp.all(jnp.isnan(student_location_result[1:]))
    assert jnp.isfinite(student_scale_result[0])
    assert jnp.all(jnp.isnan(student_scale_result[1:]))
    assert jnp.isfinite(uniform_result[0])
    assert jnp.all(jnp.isnan(uniform_result[1:]))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_rngs_compute_with_float32_for_low_precision_parameters(dtype) -> None:
    key = jax.random.key(0)
    location = jnp.array([0.0, 1.0], dtype=dtype)
    scale = jnp.array([1.0, 2.0], dtype=dtype)
    rate = jnp.array([0.5, 2.0], dtype=dtype)
    degrees = jnp.array([5.0, 7.0], dtype=dtype)
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
    expected_inverse_gamma = jnp.exp(
        jnp.log(rate.astype(jnp.float32))
        - jax.random.loggamma(key, scale.astype(jnp.float32), shape=(2,), dtype=jnp.float32)
    )
    expected_laplace = location.astype(jnp.float32) + scale.astype(jnp.float32) * jax.random.laplace(
        key, shape=(2,), dtype=jnp.float32
    )
    uniform_lower = jnp.zeros(2, dtype=dtype)
    uniform_upper = jnp.ones(2, dtype=dtype)
    expected_uniform = jax.random.uniform(key, shape=(2,), dtype=jnp.float32)

    normal_result = normal_rng(key, location, scale)
    half_normal_result = half_normal_rng(key, scale)
    lognormal_result = lognormal_rng(key, location, scale)
    exponential_result = exponential_rng(key, rate)
    gamma_result = gamma_rng(key, scale, rate)
    beta_result = beta_rng(key, scale, rate)
    inverse_gamma_result = inverse_gamma_rng(key, scale, rate)
    laplace_result = laplace_rng(key, location, scale)
    student_result = student_t_rng(key, degrees, location, scale)
    uniform_result = uniform_rng(key, uniform_lower, uniform_upper)

    assert normal_result.dtype == jnp.dtype(jnp.float32)
    assert half_normal_result.dtype == jnp.dtype(jnp.float32)
    assert lognormal_result.dtype == jnp.dtype(jnp.float32)
    assert exponential_result.dtype == jnp.dtype(jnp.float32)
    assert gamma_result.dtype == jnp.dtype(jnp.float32)
    assert beta_result.dtype == jnp.dtype(jnp.float32)
    assert inverse_gamma_result.dtype == jnp.dtype(jnp.float32)
    assert laplace_result.dtype == jnp.dtype(jnp.float32)
    assert student_result.dtype == jnp.dtype(jnp.float32)
    assert uniform_result.dtype == jnp.dtype(jnp.float32)
    assert jnp.array_equal(normal_result, expected_normal)
    assert jnp.array_equal(half_normal_result, expected_half_normal)
    assert jnp.array_equal(lognormal_result, expected_lognormal)
    assert jnp.array_equal(exponential_result, expected_exponential)
    assert jnp.array_equal(gamma_result, expected_gamma)
    assert jnp.array_equal(beta_result, expected_beta)
    assert jnp.array_equal(inverse_gamma_result, expected_inverse_gamma)
    assert jnp.array_equal(laplace_result, expected_laplace)
    assert jnp.all(jnp.isfinite(student_result))
    assert jnp.array_equal(uniform_result, expected_uniform)


def test_rngs_can_be_jitted() -> None:
    key = jax.random.key(0)
    compiled_normal = jax.jit(partial(normal_rng, location=0.0, scale=1.0, sample_shape=(2,)))
    compiled_half_normal = jax.jit(partial(half_normal_rng, scale=1.0, sample_shape=(2,)))
    compiled_lognormal = jax.jit(partial(lognormal_rng, location=0.0, scale=1.0, sample_shape=(2,)))
    compiled_exponential = jax.jit(partial(exponential_rng, rate=1.0, sample_shape=(2,)))
    compiled_gamma = jax.jit(partial(gamma_rng, shape=2.0, rate=1.0, sample_shape=(2,)))
    compiled_beta = jax.jit(partial(beta_rng, alpha=2.0, beta=1.0, sample_shape=(2,)))
    compiled_inverse_gamma = jax.jit(partial(inverse_gamma_rng, shape=2.0, scale=1.0, sample_shape=(2,)))
    compiled_laplace = jax.jit(partial(laplace_rng, location=0.0, scale=1.0, sample_shape=(2,)))
    compiled_student = jax.jit(
        partial(student_t_rng, degrees_of_freedom=5.0, location=0.0, scale=1.0, sample_shape=(2,))
    )
    compiled_uniform = jax.jit(partial(uniform_rng, lower=0.0, upper=1.0, sample_shape=(2,)))

    assert jnp.array_equal(compiled_normal(key), normal_rng(key, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_half_normal(key), half_normal_rng(key, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_lognormal(key), lognormal_rng(key, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_exponential(key), exponential_rng(key, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_gamma(key), gamma_rng(key, 2.0, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_beta(key), beta_rng(key, 2.0, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_inverse_gamma(key), inverse_gamma_rng(key, 2.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_laplace(key), laplace_rng(key, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.allclose(compiled_student(key), student_t_rng(key, 5.0, 0.0, 1.0, sample_shape=(2,)))
    assert jnp.array_equal(compiled_uniform(key), uniform_rng(key, 0.0, 1.0, sample_shape=(2,)))


def test_rngs_can_be_vectorized_over_keys() -> None:
    keys = jax.random.split(jax.random.key(0), 3)

    normal_result = jax.vmap(normal_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    half_normal_result = jax.vmap(half_normal_rng, in_axes=(0, None))(keys, 1.0)
    lognormal_result = jax.vmap(lognormal_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    exponential_result = jax.vmap(exponential_rng, in_axes=(0, None))(keys, 1.0)
    gamma_result = jax.vmap(gamma_rng, in_axes=(0, None, None))(keys, 2.0, 1.0)
    beta_result = jax.vmap(beta_rng, in_axes=(0, None, None))(keys, 2.0, 1.0)
    inverse_gamma_result = jax.vmap(inverse_gamma_rng, in_axes=(0, None, None))(keys, 2.0, 1.0)
    laplace_result = jax.vmap(laplace_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    student_result = jax.vmap(student_t_rng, in_axes=(0, None, None, None))(keys, 5.0, 0.0, 1.0)
    uniform_result = jax.vmap(uniform_rng, in_axes=(0, None, None))(keys, 0.0, 1.0)
    expected_normal = jnp.stack([normal_rng(key, 0.0, 1.0) for key in keys])
    expected_half_normal = jnp.stack([half_normal_rng(key, 1.0) for key in keys])
    expected_lognormal = jnp.stack([lognormal_rng(key, 0.0, 1.0) for key in keys])
    expected_exponential = jnp.stack([exponential_rng(key, 1.0) for key in keys])
    expected_gamma = jnp.stack([gamma_rng(key, 2.0, 1.0) for key in keys])
    expected_beta = jnp.stack([beta_rng(key, 2.0, 1.0) for key in keys])
    expected_inverse_gamma = jnp.stack([inverse_gamma_rng(key, 2.0, 1.0) for key in keys])
    expected_laplace = jnp.stack([laplace_rng(key, 0.0, 1.0) for key in keys])
    expected_student = jnp.stack([student_t_rng(key, 5.0, 0.0, 1.0) for key in keys])
    expected_uniform = jnp.stack([uniform_rng(key, 0.0, 1.0) for key in keys])

    assert jnp.array_equal(normal_result, expected_normal)
    assert jnp.array_equal(half_normal_result, expected_half_normal)
    assert jnp.allclose(lognormal_result, expected_lognormal)
    assert jnp.array_equal(exponential_result, expected_exponential)
    assert jnp.allclose(gamma_result, expected_gamma)
    assert jnp.array_equal(beta_result, expected_beta)
    assert jnp.allclose(inverse_gamma_result, expected_inverse_gamma)
    assert jnp.array_equal(laplace_result, expected_laplace)
    assert jnp.array_equal(student_result, expected_student)
    assert jnp.array_equal(uniform_result, expected_uniform)


def test_rngs_are_deterministic_for_a_given_key() -> None:
    key, different_key = jax.random.split(jax.random.key(0))

    first = normal_rng(key, 0.0, 1.0, sample_shape=(8,))
    repeated = normal_rng(key, 0.0, 1.0, sample_shape=(8,))
    different = normal_rng(different_key, 0.0, 1.0, sample_shape=(8,))

    assert jnp.array_equal(first, repeated)
    assert not jnp.array_equal(first, different)


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


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (normal_logpdf, (0.0, 0.0, 1.0 + 0.0j), "scale"),
        (half_normal_logpdf, (0.0, 1.0 + 0.0j), "scale"),
        (lognormal_logpdf, (1.0, 0.0, 1.0 + 0.0j), "scale"),
        (exponential_logpdf, (0.0, 1.0 + 0.0j), "rate"),
        (gamma_logpdf, (1.0, 1.0 + 0.0j, 1.0), "shape"),
        (beta_logpdf, (0.5, 1.0 + 0.0j, 1.0), "alpha"),
        (inverse_gamma_logpdf, (1.0, 1.0 + 0.0j, 1.0), "shape"),
        (laplace_logpdf, (0.0, 0.0, 1.0 + 0.0j), "scale"),
        (normal_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "scale"),
        (half_normal_rng, (jax.random.key(0), 1.0 + 0.0j), "scale"),
        (lognormal_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "scale"),
        (exponential_rng, (jax.random.key(0), 1.0 + 0.0j), "rate"),
        (gamma_rng, (jax.random.key(0), 1.0, 1.0 + 0.0j), "rate"),
        (beta_rng, (jax.random.key(0), 1.0, 1.0 + 0.0j), "beta"),
        (inverse_gamma_rng, (jax.random.key(0), 1.0, 1.0 + 0.0j), "scale"),
        (laplace_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "scale"),
        (student_t_logpdf, (0.0, 5.0 + 0.0j, 0.0, 1.0), "degrees_of_freedom"),
        (student_t_rng, (jax.random.key(0), 5.0, 0.0, 1.0 + 0.0j), "scale"),
        (uniform_logpdf, (0.0, 0.0 + 0.0j, 1.0), "lower"),
        (uniform_rng, (jax.random.key(0), 0.0, 1.0 + 0.0j), "upper"),
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
