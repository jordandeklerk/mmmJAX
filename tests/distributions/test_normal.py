"""Tests for Normal distribution functions."""

import jax
import jax.numpy as jnp
import pytest

from mmmjax import (
    normal,
    normal_logcdf,
    normal_logpdf,
    normal_logsf,
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


def test_normal_returns_scalar_sum() -> None:
    values = jnp.array([0.0, 1.0, -2.0])

    result = normal(values, 0.0, 1.0)

    assert result.shape == ()
    assert jnp.allclose(result, jnp.sum(normal_logpdf(values, 0.0, 1.0)))


def test_normal_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.0], [1.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    result = normal_logpdf(values, locations, 2.0)

    assert result.shape == (2, 3)
    assert jnp.allclose(normal(values, locations, 2.0), jnp.sum(result))


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


def test_normal_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    half_log_two_pi = jnp.asarray(0.9189385332046727, dtype=jnp.float32)
    expected = -jnp.log(scales) - half_log_two_pi

    result = normal_logpdf(jnp.zeros(2, dtype=jnp.float32), 0.0, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_normal_log_probabilities_are_complements() -> None:
    values = jnp.array([-20.0, -2.0, 0.0, 2.0, 20.0])

    log_cdf = normal_logcdf(values, 0.0, 1.0)
    log_survival = normal_logsf(values, 0.0, 1.0)

    assert jnp.allclose(jnp.logaddexp(log_cdf, log_survival), 0.0, atol=1e-6)


def test_normal_log_probabilities_follow_distribution_symmetry() -> None:
    values = jnp.array([-20.0, -2.0, 0.0, 2.0, 20.0])
    location = 0.4
    scale = 1.7

    assert jnp.allclose(
        normal_logsf(values, location, scale),
        normal_logcdf(-values, -location, scale),
    )


def test_normal_log_probabilities_remain_finite_in_extreme_tails() -> None:
    lower_tail = normal_logcdf(-40.0, 0.0, 1.0)
    upper_tail = normal_logsf(40.0, 0.0, 1.0)

    assert jnp.isfinite(lower_tail)
    assert jnp.isfinite(upper_tail)
    assert jnp.allclose(lower_tail, upper_tail)


@pytest.mark.parametrize(
    ("function", "direction"),
    [(normal_logcdf, -1), (normal_logsf, 1)],
)
@pytest.mark.parametrize("tangent", [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
def test_normal_log_probabilities_preserve_translation_invariant_tangents(
    function,
    direction: int,
    tangent: tuple[float, float, float],
) -> None:
    arguments = (
        jnp.asarray(direction * 8.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
    )
    tangent_arrays = tuple(jnp.asarray(item) for item in tangent)

    _, directional_derivative = jax.jvp(function, arguments, tangent_arrays)
    compiled_derivative = jax.jit(lambda current: jax.jvp(function, arguments, current)[1])(tangent_arrays)

    assert directional_derivative == 0
    assert compiled_derivative == 0


@pytest.mark.parametrize(
    ("function", "direction"),
    [(normal_logcdf, -1), (normal_logsf, 1)],
)
def test_normal_log_probabilities_preserve_scale_invariant_tangents(function, direction: int) -> None:
    standardized_value = jnp.asarray(direction * 8.0)
    arguments = (standardized_value, jnp.asarray(0.0), jnp.asarray(1.0))
    tangent = arguments

    _, directional_derivative = jax.jvp(function, arguments, tangent)
    compiled_derivative = jax.jit(lambda current: jax.jvp(function, arguments, current)[1])(tangent)

    assert directional_derivative == 0
    assert compiled_derivative == 0


@pytest.mark.parametrize("function", [normal_logcdf, normal_logsf])
def test_normal_log_probability_jvp_is_linear(function) -> None:
    arguments = (jnp.asarray(1.25), jnp.asarray(-0.3), jnp.asarray(1.7))
    first_tangent = (jnp.asarray(0.2), jnp.asarray(-0.1), jnp.asarray(0.05))
    second_tangent = (jnp.asarray(-0.4), jnp.asarray(0.3), jnp.asarray(-0.15))
    combined_tangent = tuple(first + second for first, second in zip(first_tangent, second_tangent, strict=True))

    first = jax.jvp(function, arguments, first_tangent)[1]
    second = jax.jvp(function, arguments, second_tangent)[1]
    combined = jax.jvp(function, arguments, combined_tangent)[1]
    compiled = jax.jit(lambda tangent: jax.jvp(function, arguments, tangent)[1])(combined_tangent)

    assert jnp.allclose(combined, first + second)
    assert jnp.allclose(compiled, combined)


def test_normal_log_probabilities_broadcast_arguments() -> None:
    values = jnp.array([[-1.0], [1.0]])
    locations = jnp.array([-1.0, 0.0, 1.0])

    assert normal_logcdf(values, locations, 2.0).shape == (2, 3)
    assert normal_logsf(values, locations, 2.0).shape == (2, 3)


def test_normal_log_probabilities_handle_infinite_values_and_nan() -> None:
    values = jnp.array([-jnp.inf, jnp.inf, jnp.nan])

    log_cdf = normal_logcdf(values, 0.0, 1.0)
    log_survival = normal_logsf(values, 0.0, 1.0)

    assert jnp.isneginf(log_cdf[0])
    assert log_cdf[1] == 0
    assert jnp.isnan(log_cdf[2])
    assert log_survival[0] == 0
    assert jnp.isneginf(log_survival[1])
    assert jnp.isnan(log_survival[2])


@pytest.mark.parametrize("function", [normal_logcdf, normal_logsf])
def test_normal_log_probabilities_reject_invalid_parameters(function) -> None:
    scales = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_scales = function(0.0, 0.0, scales)
    invalid_locations = function(0.0, jnp.array([jnp.inf, -jnp.inf, jnp.nan]), 1.0)

    assert jnp.all(jnp.isnan(invalid_scales))
    assert jnp.all(jnp.isnan(invalid_locations))
    assert jnp.isnan(function(jnp.nan, 0.0, 1.0))


@pytest.mark.parametrize("function", [normal_logcdf, normal_logsf])
@pytest.mark.parametrize("bound", [-jnp.inf, jnp.inf])
def test_normal_log_probabilities_have_zero_parameter_gradients_at_infinite_bounds(function, bound) -> None:
    parameters = jnp.array([5.0, jnp.finfo(jnp.float32).tiny])

    gradients = jax.grad(lambda current: function(bound, current[0], current[1]))(parameters)

    assert jnp.array_equal(gradients, jnp.zeros_like(parameters))


@pytest.mark.parametrize(
    ("function", "direction"),
    [(normal_logcdf, 1.0), (normal_logsf, -1.0)],
)
def test_normal_log_probability_value_gradients_match_density_ratio(function, direction) -> None:
    value = 1.25
    location = -0.3
    scale = 1.7
    expected = direction * jnp.exp(normal_logpdf(value, location, scale) - function(value, location, scale))

    result = jax.grad(lambda current: function(current, location, scale))(value)

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("function", [normal_logcdf, normal_logsf])
def test_normal_log_probabilities_can_be_vectorized(function) -> None:
    values = jnp.array([-1.0, 2.0])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(function)(values, locations, scales)
    expected = jnp.stack(
        [function(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
    )

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("function", [normal_logcdf, normal_logsf])
def test_normal_log_probability_gradients_match_between_arrays_and_vmap(function) -> None:
    values = jnp.array([-2.0, 0.25, 3.0])
    locations = jnp.array([0.0, -0.5, 0.75])
    scales = jnp.array([1.0, 1.5, 2.0])
    argnums = (0, 1, 2)

    array_jacobians = jax.jit(jax.jacfwd(function, argnums=argnums))(values, locations, scales)
    mapped_gradients = jax.jit(jax.vmap(jax.grad(function, argnums=argnums)))(values, locations, scales)

    for jacobian, gradients in zip(array_jacobians, mapped_gradients, strict=True):
        assert jnp.allclose(jnp.diag(jacobian), gradients)


@pytest.mark.parametrize(
    ("function", "values", "direction"),
    [
        (normal_logcdf, jnp.array([-7.0, 13.0]), 1),
        (normal_logsf, jnp.array([7.0, -13.0]), -1),
    ],
)
def test_normal_log_probabilities_preserve_small_derivatives_in_mixed_tail_batches(
    function,
    values: jax.Array,
    direction: int,
) -> None:
    expected = direction * jnp.exp(-0.5 * jnp.square(jnp.asarray(13.0))) / jnp.sqrt(2 * jnp.pi)
    differentiate = jax.jacfwd(lambda current: function(current, jnp.zeros(2), jnp.ones(2)))

    result = differentiate(values)[1, 1]
    compiled_result = jax.jit(differentiate)(values)[1, 1]

    assert result != 0
    assert jnp.allclose(result, expected, rtol=3e-6, atol=0)
    assert jnp.allclose(compiled_result, expected, rtol=3e-6, atol=0)


def test_normal_is_differentiable_with_respect_to_location() -> None:
    values = jnp.array([0.0, 1.0, 2.0])
    location = 0.5
    scale = 2.0
    expected = jnp.sum(values - location) / scale**2

    result = jax.grad(lambda current_location: normal(values, current_location, scale))(location)

    assert jnp.allclose(result, expected)


def test_normal_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(normal)(values, locations, scales)
    expected = jnp.stack(
        [normal(value, location, scale) for value, location, scale in zip(values, locations, scales, strict=True)]
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


def test_normal_rng_uses_broadcast_parameter_shape() -> None:
    location = jnp.zeros((2, 1))
    scale = jnp.ones(3)

    result = normal_rng(jax.random.key(0), location, scale, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_normal_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        normal_rng(jax.random.key(0), jnp.zeros(2), jnp.ones(3))
