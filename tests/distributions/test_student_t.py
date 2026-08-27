"""Tests for Student-t distribution functions."""

import jax
import jax.numpy as jnp
import pytest

from mmmjax import (
    Positive,
    normal_logpdf,
    student_t,
    student_t_logpdf,
    student_t_rng,
)


def test_student_t_logpdf_matches_known_values() -> None:
    values = jnp.array([-2.0, 0.5, 3.0], dtype=jnp.float32)
    expected = jnp.array(
        [-3.0850290200062593, -1.2508512163496537, -2.6885176662184307],
        dtype=jnp.float32,
    )

    result = student_t_logpdf(values, 4.5, 0.7, 1.3)

    assert jnp.allclose(result, expected)


def test_student_t_returns_scalar_sum() -> None:
    values = jnp.array([-2.0, 0.5, 3.0])

    result = student_t(values, 4.5, 0.7, 1.3)

    assert result.shape == ()
    assert jnp.allclose(result, -7.024397902574344)


def test_student_t_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[-1.0], [2.0]])
    degrees = jnp.array([1.0, 5.0, 30.0])
    locations = jnp.array([-0.5, 0.0, 0.5])
    scales = jnp.array([0.75, 1.5, 2.5])
    expected = jnp.array(
        [
            [-1.2247725935229363, -1.6295581221838091, -2.028453905664747],
            [-3.3511711182905435, -2.286718820371863, -2.028453905664747],
        ]
    )

    result = student_t_logpdf(values, degrees, locations, scales)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected)
    assert jnp.allclose(student_t(values, degrees, locations, scales), -12.549128465698647)


def test_student_t_logpdf_rejects_invalid_parameters_without_repairing_them() -> None:
    invalid = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_degrees = student_t_logpdf(0.0, invalid, 0.0, 1.0)
    invalid_locations = student_t_logpdf(0.0, 5.0, jnp.array([jnp.inf, -jnp.inf, jnp.nan]), 1.0)
    invalid_scales = student_t_logpdf(0.0, 5.0, 0.0, invalid)

    assert jnp.all(jnp.isnan(invalid_degrees))
    assert jnp.all(jnp.isnan(invalid_locations))
    assert jnp.all(jnp.isnan(invalid_scales))


def test_student_t_logpdf_handles_nonfinite_values() -> None:
    values = jnp.array([jnp.inf, -jnp.inf, jnp.nan])

    result = student_t_logpdf(values, 4.5, 0.7, 1.3)

    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.isnan(result[2])


def test_student_t_logpdf_remains_finite_in_extreme_valid_tails() -> None:
    values = jnp.array([-1e20, 1e20], dtype=jnp.float32)

    result = student_t_logpdf(values, 4.5, 0.0, 1.0)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, -250.12220807750055)


def test_student_t_logpdf_handles_finite_subtraction_overflow() -> None:
    maximum = jnp.finfo(jnp.float32).max
    values = jnp.array([maximum, -maximum], dtype=jnp.float32)
    locations = -values

    result = student_t_logpdf(values, 4.5, locations, 1.0)
    compiled = jax.jit(student_t_logpdf)(values, 4.5, locations, 1.0)
    gradients = jax.vmap(jax.grad(lambda value, location: student_t_logpdf(value, 4.5, location, 1.0)))(
        values,
        locations,
    )

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, -488.6257721276112)
    assert jnp.allclose(compiled, result)
    assert jnp.all(jnp.isfinite(gradients))


def test_student_t_logpdf_handles_finite_subtraction_underflow() -> None:
    value = jnp.float32(-3.5653085e-35)
    location = jnp.float32(-3.5653065e-35)
    degrees = jnp.float32(6.07383e-38)
    scale = jnp.float32(1.1537141e-32)

    result = student_t_logpdf(value, degrees, location, scale)
    compiled = jax.jit(student_t_logpdf)(value, degrees, location, scale)

    assert value != location
    assert jnp.allclose(result, 7.3210094489)
    assert jnp.allclose(compiled, result)


def test_student_t_logpdf_remains_finite_for_extreme_valid_scales() -> None:
    scales = jnp.array([1e-30, 1e20], dtype=jnp.float32)
    expected = jnp.array([68.10349210053107, -47.02576254917121], dtype=jnp.float32)

    result = student_t_logpdf(0.7, 4.5, 0.7, scales)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_student_t_with_one_degree_of_freedom_matches_cauchy_values() -> None:
    values = jnp.array([-2.0, 0.0, 1.5])
    expected = jnp.array([-2.936489355077455, -1.432411958301181, -1.9369679690535764])

    result = student_t_logpdf(values, 1.0, 0.4, 1.2)

    assert jnp.allclose(result, expected)


def test_student_t_approaches_normal_for_large_degrees_of_freedom() -> None:
    values = jnp.array([-3.0, -0.5, 0.0, 2.0])

    result = student_t_logpdf(values, 1e7, 0.0, 1.0)
    expected = normal_logpdf(values, 0.0, 1.0)

    assert jnp.allclose(result, expected, rtol=0, atol=3e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_student_t_approaches_normal_at_maximum_degrees_of_freedom(dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    values = jnp.array([-2.0, -0.1, 0.0, 1.0, 3.0], dtype=dtype)

    result = student_t_logpdf(values, jnp.finfo(dtype).max, dtype(0.0), dtype(1.0))
    expected = normal_logpdf(values, 0.0, 1.0)

    tolerance = 5e-7 if dtype == jnp.float32 else 1e-14
    assert jnp.allclose(result, expected, rtol=0, atol=tolerance)


def test_student_t_logpdf_supports_smallest_normal_degrees_of_freedom() -> None:
    result = student_t_logpdf(0.0, jnp.finfo(jnp.float32).tiny, 0.0, 1.0)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, -44.361419555836505)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_student_t_normalizer_and_gradient_match_float64_reference() -> None:
    degrees = jnp.float64(16)

    result = student_t_logpdf(jnp.float64(0), degrees, jnp.float64(0), jnp.float64(1))
    gradient = jax.grad(lambda current: student_t_logpdf(jnp.float64(0), current, jnp.float64(0), jnp.float64(1)))(
        degrees
    )

    assert jnp.allclose(result, -0.9345534078090085, rtol=0, atol=1e-14)
    assert jnp.allclose(gradient, 0.0009746698119050823, rtol=0, atol=1e-14)


def test_student_t_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([-2.0, 0.5, 3.0])
    expected = jnp.array([0.9969788519637463, 0.14388489208633087, -0.9810003877471888])

    result = jax.grad(lambda current_values: student_t(current_values, 4.5, 0.7, 1.3))(values)

    assert jnp.allclose(result, expected)


def test_student_t_is_differentiable_with_respect_to_parameters() -> None:
    values = jnp.array([-2.0, 0.5, 3.0])

    degrees_gradient = jax.grad(lambda current: student_t(values, current, 0.7, 1.3))(4.5)
    location_gradient = jax.grad(lambda current: student_t(values, 4.5, current, 1.3))(0.7)
    scale_gradient = jax.grad(lambda current: student_t(values, 4.5, 0.7, current))(1.3)

    assert jnp.allclose(degrees_gradient, -0.013559942952256928, atol=1e-6)
    assert jnp.allclose(location_gradient, -0.15986335630288828)
    assert jnp.allclose(scale_gradient, 1.5207082850291656)


def test_student_t_has_zero_finite_gradient_at_location() -> None:
    result = jax.grad(lambda value: student_t_logpdf(value, 4.5, 0.7, 1.3))(0.7)

    assert jnp.isfinite(result)
    assert result == 0


def test_student_t_has_correct_curvature_at_location() -> None:
    result = jax.grad(jax.grad(lambda value: student_t_logpdf(value, 5.0, 0.0, 1.0)))(jnp.float32(0))

    assert jnp.allclose(result, -1.2)


@pytest.mark.parametrize("position", [-50.0, -87.0])
def test_student_t_has_finite_gradient_through_degrees_parameterization(position: float) -> None:
    parameter = Positive()

    def target(current_position):
        degrees = parameter.constrain(current_position)
        return student_t_logpdf(0.3, degrees, 0.0, 1.0) + parameter.log_density_adjustment(current_position)

    result = jax.grad(target)(jnp.float32(position))

    assert jnp.isfinite(result)
    assert jnp.allclose(result, 2.0)


@pytest.mark.parametrize(("value", "expected"), [(0.0, 0.0), (0.3, 6.0)])
@pytest.mark.parametrize("position", [-50.0, -87.0])
def test_student_t_has_finite_gradient_through_scale_parameterization(
    value: float,
    expected: float,
    position: float,
) -> None:
    parameter = Positive()

    def target(current_position):
        scale = parameter.constrain(current_position)
        return student_t_logpdf(value, 5.0, 0.0, scale) + parameter.log_density_adjustment(current_position)

    result = jax.grad(target)(jnp.float32(position))

    assert jnp.isfinite(result)
    assert jnp.allclose(result, expected)


def test_student_t_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[-1.0, 0.0], [0.5, 2.0]])
    degrees = jnp.array([3.0, 8.0])
    locations = jnp.array([0.0, 0.5])
    scales = jnp.array([1.0, 2.0])

    result = jax.vmap(student_t)(values, degrees, locations, scales)
    expected = jnp.stack(
        [
            student_t(value, degree, location, scale)
            for value, degree, location, scale in zip(values, degrees, locations, scales, strict=True)
        ]
    )

    assert jnp.allclose(result, expected)


def test_student_t_rng_uses_normal_and_log_gamma_draws() -> None:
    key = jax.random.key(42)
    degrees = jnp.array([3.0, 7.0], dtype=jnp.float32)
    locations = jnp.array([0.5, -1.0], dtype=jnp.float32)
    scales = jnp.array([0.75, 2.0], dtype=jnp.float32)
    normal_key, gamma_key = jax.random.split(key)
    standard_normal = jax.random.normal(normal_key, shape=(3, 2), dtype=jnp.float32)
    log_unit_gamma = jax.random.loggamma(gamma_key, degrees / 2, shape=(3, 2), dtype=jnp.float32)
    log_magnitude = (
        jnp.log(scales) + jnp.log(jnp.abs(standard_normal)) + 0.5 * (jnp.log(degrees) - jnp.log(2.0) - log_unit_gamma)
    )
    expected = locations + jnp.copysign(jnp.exp(log_magnitude), standard_normal)

    result = student_t_rng(key, degrees, locations, scales, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)


def test_student_t_rng_uses_broadcast_parameter_shape() -> None:
    degrees = jnp.ones((2, 1)) * 5
    locations = jnp.zeros(3)

    result = student_t_rng(jax.random.key(0), degrees, locations, 1.0, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_student_t_rng_matches_reference_central_coverage() -> None:
    samples = student_t_rng(jax.random.key(7), 5.0, 0.7, 1.3, sample_shape=(50_000,))
    standardized = jnp.abs((samples - 0.7) / 1.3)
    central_coverage = jnp.mean(standardized <= 2.5705818356363146)

    assert jnp.allclose(central_coverage, 0.95, rtol=0, atol=0.01)


def test_student_t_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\), \(\)\)",
    ):
        student_t_rng(jax.random.key(0), jnp.ones(2), jnp.zeros(3), 1.0)
