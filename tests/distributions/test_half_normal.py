"""Tests for HalfNormal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import (
    half_normal,
    half_normal_logcdf,
    half_normal_logpdf,
    half_normal_logsf,
    half_normal_rng,
)


def test_half_normal_logpdf_matches_known_values() -> None:
    values = jnp.array([0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = jnp.array(
        [-0.6312564607528918, -0.6868120163084473, -1.5201453496417807],
        dtype=jnp.float32,
    )

    result = half_normal_logpdf(values, 1.5)

    assert jnp.allclose(result, expected)


def test_half_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 0.5, 2.0])

    result = half_normal(values, 1.5)

    assert result.shape == ()
    assert jnp.allclose(result, -2.83821382670312)


@pytest.mark.parametrize(
    ("function", "reference", "values"),
    [
        pytest.param(
            half_normal_logcdf,
            stats.halfnorm.logcdf,
            [0.0, np.finfo(np.float32).tiny, 1e-30, 1e-10, 0.1, 1.0, 10.0, np.inf],
            id="logcdf",
        ),
        pytest.param(
            half_normal_logsf,
            stats.halfnorm.logsf,
            [0.0, 1e-10, 0.1, 1.0, 10.0, 40.0, np.inf],
            id="logsf",
        ),
    ],
)
def test_half_normal_log_probabilities_match_scipy(function, reference, values: list[float]) -> None:
    values_array = np.asarray(values, dtype=np.float32)
    scale = np.float32(1.7)
    expected = reference(values_array.astype(np.float64), scale=float(scale))

    result = function(values_array, scale)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_half_normal_log_probabilities_are_complements() -> None:
    values = jnp.concatenate((jnp.asarray([0.0]), jnp.geomspace(1e-30, 40.0, 31)))

    log_cdf = half_normal_logcdf(values, 1.7)
    log_survival = half_normal_logsf(values, 1.7)

    assert jnp.allclose(jnp.logaddexp(log_cdf, log_survival), 0, atol=2e-7)


def test_half_normal_log_probabilities_broadcast_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    scales = jnp.array([0.5, 1.0, 2.0])

    assert half_normal_logcdf(values, scales).shape == (2, 3)
    assert half_normal_logsf(values, scales).shape == (2, 3)


def test_half_normal_log_probabilities_handle_support_boundaries_and_nan() -> None:
    values = jnp.array([-jnp.inf, -1.0, -0.0, 0.0, jnp.inf, jnp.nan])

    log_cdf = half_normal_logcdf(values, 1.7)
    log_survival = half_normal_logsf(values, 1.7)

    assert jnp.all(jnp.isneginf(log_cdf[:4]))
    assert log_cdf[4] == 0
    assert jnp.isnan(log_cdf[5])
    assert jnp.all(log_survival[:4] == 0)
    assert jnp.isneginf(log_survival[4])
    assert jnp.isnan(log_survival[5])


@pytest.mark.parametrize("function", [half_normal_logcdf, half_normal_logsf])
def test_half_normal_log_probabilities_reject_invalid_scale(function) -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    result = function(-1.0, scales)

    assert jnp.all(jnp.isnan(result))


@pytest.mark.parametrize(
    ("function", "direction"),
    [(half_normal_logcdf, 1), (half_normal_logsf, -1)],
)
@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("value", [0.3, 5.0], ids=["ordinary", "tail"])
def test_half_normal_log_probability_gradients_match_density_ratio(
    function,
    direction: int,
    differentiate,
    value: float,
) -> None:
    scale = 1.7
    expected_value = direction * jnp.exp(half_normal_logpdf(value, scale) - function(value, scale))
    expected_scale = -(value / scale) * expected_value

    result = differentiate(function, argnums=(0, 1))(value, scale)
    compiled_result = jax.jit(differentiate(function, argnums=(0, 1)))(value, scale)

    assert jnp.allclose(jnp.asarray(result), jnp.asarray([expected_value, expected_scale]), rtol=3e-6, atol=0)
    assert jnp.allclose(jnp.asarray(compiled_result), jnp.asarray(result), rtol=3e-6, atol=0)


@pytest.mark.parametrize("function", [half_normal_logcdf, half_normal_logsf])
def test_half_normal_log_probabilities_match_scalar_evaluation_in_mixed_batches(function) -> None:
    values = jnp.array([np.finfo(np.float32).tiny, 0.5, 5.0, 40.0])

    result = jax.jit(function)(values, 1.0)
    expected = jnp.stack([jax.jit(function)(value, 1.0) for value in values])

    assert jnp.allclose(result, expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize(
    ("function", "reference"),
    [
        (half_normal_logcdf, stats.halfnorm.logcdf),
        (half_normal_logsf, stats.halfnorm.logsf),
    ],
)
def test_half_normal_log_probabilities_match_scipy_around_formula_boundary(function, reference) -> None:
    boundary = np.float32(1)
    values = np.array(
        [
            np.nextafter(boundary, np.float32(-np.inf)),
            boundary,
            np.nextafter(boundary, np.float32(np.inf)),
        ]
    )

    result = function(values, 1.0)
    expected = reference(values.astype(np.float64))

    np.testing.assert_allclose(
        result,
        expected,
        rtol=3e-6,
        atol=np.finfo(np.float32).eps,
    )


def test_half_normal_logcdf_preserves_values_below_erf_range() -> None:
    value = jnp.float32(np.finfo(np.float32).tiny)
    scale = jnp.float32(1.7)
    expected = jnp.log(value) - jnp.log(scale) + jnp.asarray(np.log(2 / np.pi) / 2, dtype=jnp.float32)

    result = half_normal_logcdf(value, scale)
    compiled_result = jax.jit(half_normal_logcdf)(value, scale)
    gradients = jax.jit(jax.grad(half_normal_logcdf, argnums=(0, 1)))(value, scale)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(compiled_result, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(gradients[0], 1 / value, rtol=3e-6, atol=0)
    assert jnp.allclose(gradients[1], -1 / scale, rtol=3e-6, atol=0)


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


def test_half_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    expected = jnp.asarray(-0.2257913526447274, dtype=jnp.float32) - jnp.log(scales)

    result = half_normal_logpdf(jnp.zeros(2, dtype=jnp.float32), scales)

    assert jnp.all(jnp.isfinite(result))
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


def test_half_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [0.5, 2.0]])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(half_normal)(values, scales)
    expected = jnp.stack([half_normal(value, scale) for value, scale in zip(values, scales, strict=True)])

    assert jnp.allclose(result, expected)


def test_half_normal_rng_folds_standard_normal_draws() -> None:
    key = jax.random.key(42)
    scale = jnp.array([0.5, 2.0], dtype=jnp.float32)
    expected = jnp.abs(scale * jax.random.normal(key, shape=(3, 2), dtype=jnp.float32))

    result = half_normal_rng(key, scale, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all(result >= 0)
