"""Tests for Dirichlet distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mmmjax import Simplex, beta_logpdf, dirichlet, dirichlet_logpdf


def test_dirichlet_logpdf_matches_known_value() -> None:
    result = dirichlet_logpdf(jnp.array([0.2, 0.3, 0.5]), jnp.ones(3))

    assert jnp.allclose(result, jnp.log(2.0))


def test_two_component_dirichlet_matches_beta() -> None:
    values = jnp.array([0.1, 0.4, 0.8])
    simplexes = jnp.stack((values, 1 - values), axis=-1)

    result = dirichlet_logpdf(simplexes, jnp.array([2.3, 4.7]))
    expected = beta_logpdf(values, 2.3, 4.7)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_dirichlet_logpdf_matches_scipy_reference_batch() -> None:
    values = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
            [0.6, 0.1, 0.3],
        ],
        dtype=np.float32,
    )
    concentrations = np.array(
        [
            [1.0, 1.0, 1.0],
            [6.2, 3.5, 9.1],
            [2.5, 7.4, 6.1],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            stats.dirichlet.logpdf(value, concentration)
            for value, concentration in zip(values, concentrations, strict=True)
        ]
    )

    result = dirichlet_logpdf(values, concentrations)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_dirichlet_returns_scalar_sum() -> None:
    values = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
    concentration = jnp.array([1.0, 1.0, 1.0])

    result = dirichlet(values, concentration)

    assert result.shape == ()
    assert jnp.allclose(result, 2 * jnp.log(2.0))


def test_dirichlet_logpdf_broadcasts_leading_axes() -> None:
    values = jnp.array(
        [
            [[0.2, 0.3, 0.5]],
            [[0.1, 0.8, 0.1]],
        ]
    )
    concentrations = jnp.array([[[1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5], [4.0, 2.0, 1.0]]])

    result = dirichlet_logpdf(values, concentrations)
    expected = np.empty((2, 4))
    for value_index in range(2):
        for concentration_index in range(4):
            expected[value_index, concentration_index] = stats.dirichlet.logpdf(
                np.asarray(values[value_index, 0]),
                np.asarray(concentrations[0, concentration_index]),
            )

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_dirichlet_logpdf_gradients_match_analytic_result() -> None:
    value = jnp.array([0.2, 0.3, 0.5])
    concentration = jnp.array([2.5, 3.5, 4.5])
    concentration_sum = np.sum(np.asarray(concentration, dtype=np.float64))
    expected_value_gradient = (np.asarray(concentration) - 1) / np.asarray(value)
    expected_concentration_gradient = (
        special.digamma(concentration_sum)
        - special.digamma(np.asarray(concentration, dtype=np.float64))
        + np.log(np.asarray(value, dtype=np.float64))
    )

    value_gradient, concentration_gradient = jax.grad(dirichlet_logpdf, argnums=(0, 1))(value, concentration)

    np.testing.assert_allclose(value_gradient, expected_value_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(concentration_gradient, expected_concentration_gradient, rtol=3e-6, atol=3e-6)


def test_dirichlet_logpdf_is_stable_for_concentrated_large_simplex() -> None:
    event_size = 465
    value = jnp.full((event_size,), 1 / event_size, dtype=jnp.float32)
    concentration = jnp.full((event_size,), 1e8, dtype=jnp.float32)
    reference_value = np.full(event_size, 1 / event_size, dtype=np.float64)
    reference_concentration = np.full(event_size, 1e8, dtype=np.float64)
    expected_log_density = stats.dirichlet.logpdf(reference_value, reference_concentration)
    expected_gradient = special.digamma(event_size * 1e8) - special.digamma(1e8) + np.log(1 / event_size)

    result, gradient = jax.jit(jax.value_and_grad(dirichlet_logpdf, argnums=1))(
        value,
        concentration,
    )

    np.testing.assert_allclose(result, expected_log_density, rtol=2e-7)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-7)


def test_dirichlet_logpdf_is_stable_below_component_cutoff() -> None:
    event_size = 465
    value = jnp.full((event_size,), 1 / event_size, dtype=jnp.float32)
    concentration = jnp.full((event_size,), 7.9, dtype=jnp.float32)
    expected = stats.dirichlet.logpdf(
        np.full(event_size, 1 / event_size, dtype=np.float64),
        np.full(event_size, 7.9, dtype=np.float64),
    )

    result = dirichlet_logpdf(value, concentration)

    np.testing.assert_allclose(result, expected, rtol=2e-7)


def test_concentrated_two_component_dirichlet_matches_beta() -> None:
    value = jnp.asarray(0.5001000165939331, dtype=jnp.float32)
    concentration = jnp.array([1e8, 1e8], dtype=jnp.float32)

    def dirichlet_from_first_component(first_component, shape):
        simplex = jnp.stack((first_component, 1 - first_component))
        return dirichlet_logpdf(simplex, shape)

    def beta_from_shapes(first_component, shape):
        return beta_logpdf(first_component, shape[0], shape[1])

    zero_concentration_tangent = jnp.zeros_like(concentration)
    dirichlet_value, dirichlet_tangent = jax.jvp(
        dirichlet_from_first_component,
        (value, concentration),
        (jnp.ones_like(value), zero_concentration_tangent),
    )
    beta_value, beta_tangent = jax.jvp(
        beta_from_shapes,
        (value, concentration),
        (jnp.ones_like(value), zero_concentration_tangent),
    )
    dirichlet_gradient = jax.grad(dirichlet_from_first_component, argnums=1)(value, concentration)
    beta_gradient = jax.grad(beta_from_shapes, argnums=1)(value, concentration)

    np.testing.assert_allclose(dirichlet_value, beta_value, rtol=3e-7)
    np.testing.assert_allclose(dirichlet_tangent, beta_tangent, rtol=3e-7)
    np.testing.assert_allclose(dirichlet_gradient, beta_gradient, rtol=3e-7)


def test_dirichlet_logpdf_has_correct_second_derivatives() -> None:
    value = jnp.array([0.2, 0.3, 0.5])
    concentration = jnp.array([20.0, 30.0, 50.0])
    value_array = np.asarray(value, dtype=np.float64)
    concentration_array = np.asarray(concentration, dtype=np.float64)
    concentration_sum = np.sum(concentration_array)

    expected_value_hessian = np.diag(-(concentration_array - 1) / np.square(value_array))
    expected_cross_derivative = np.diag(1 / value_array)
    expected_concentration_hessian = np.full(
        (3, 3),
        special.polygamma(1, concentration_sum),
    )
    expected_concentration_hessian[np.diag_indices(3)] -= special.polygamma(
        1,
        concentration_array,
    )

    value_hessian = jax.hessian(dirichlet_logpdf, argnums=0)(value, concentration)
    cross_derivative = jax.jacfwd(
        jax.grad(dirichlet_logpdf, argnums=0),
        argnums=1,
    )(value, concentration)
    concentration_hessian = jax.hessian(dirichlet_logpdf, argnums=1)(
        value,
        concentration,
    )

    np.testing.assert_allclose(value_hessian, expected_value_hessian, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(cross_derivative, expected_cross_derivative, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(
        concentration_hessian,
        expected_concentration_hessian,
        rtol=3e-6,
        atol=1e-7,
    )


def test_dirichlet_logpdf_handles_concentration_sum_overflow() -> None:
    maximum = jnp.finfo(jnp.float32).max
    value = jnp.array([0.5, 0.5])
    concentration = jnp.array([maximum, maximum])

    result, gradient = jax.value_and_grad(dirichlet_logpdf, argnums=1)(
        value,
        concentration,
    )
    expected = beta_logpdf(value[0], concentration[0], concentration[1])
    expected_gradient = jax.grad(lambda shape: beta_logpdf(value[0], shape[0], shape[1]))(concentration)

    assert jnp.isfinite(result)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(result, expected, rtol=3e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6)


def test_dirichlet_logpdf_mixes_standard_and_stable_batches_under_vmap() -> None:
    values = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    concentrations = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [1e8, 1e8, 1e8],
        ]
    )

    batched_value, batched_gradient = jax.jit(
        jax.value_and_grad(lambda shape: jnp.sum(dirichlet_logpdf(values, shape)))
    )(concentrations)
    mapped_values, mapped_gradients = jax.jit(jax.vmap(jax.value_and_grad(dirichlet_logpdf, argnums=1)))(
        values, concentrations
    )

    np.testing.assert_allclose(batched_value, jnp.sum(mapped_values), rtol=3e-6)
    np.testing.assert_allclose(batched_gradient, mapped_gradients, rtol=3e-6)


def test_dirichlet_logpdf_uses_boundary_limits() -> None:
    value = jnp.array([0.0, 1.0])
    concentrations = jnp.array(
        [
            [0.5, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )

    result = dirichlet_logpdf(value, concentrations)

    assert jnp.isposinf(result[0])
    assert jnp.allclose(result[1], jnp.log(2.0))
    assert jnp.isneginf(result[2])


def test_dirichlet_logpdf_uses_stable_finite_boundary_limit() -> None:
    value = jnp.array([0.0, 0.5, 0.5])
    concentration = jnp.array([1.0, 1e8, 1e8])
    expected = stats.dirichlet.logpdf(
        np.array([0.0, 0.5, 0.5], dtype=np.float64),
        np.array([1.0, 1e8, 1e8], dtype=np.float64),
    )

    result = jax.jit(dirichlet_logpdf)(value, concentration)

    np.testing.assert_allclose(result, expected, rtol=3e-6)


def test_dirichlet_logpdf_preserves_defined_boundary_derivatives() -> None:
    value = jnp.array([0.0, 0.5, 0.5])
    concentration = jnp.array([1.0, 1e8, 1e8])
    expected_value_gradient = (np.asarray(concentration[1:]) - 1) / np.asarray(value[1:])
    expected_concentration_gradient = (
        special.digamma(np.sum(np.asarray(concentration, dtype=np.float64)))
        - special.digamma(np.asarray(concentration[1:], dtype=np.float64))
        + np.log(np.asarray(value[1:], dtype=np.float64))
    )

    value_gradient, concentration_gradient = jax.grad(
        dirichlet_logpdf,
        argnums=(0, 1),
    )(value, concentration)
    _, zero_tangent = jax.jvp(
        dirichlet_logpdf,
        (value, concentration),
        (jnp.zeros_like(value), jnp.zeros_like(concentration)),
    )
    _, positive_concentration_tangent = jax.jvp(
        dirichlet_logpdf,
        (value, concentration),
        (jnp.zeros_like(value), jnp.array([0.0, 1.0, 0.0])),
    )

    assert value_gradient[0] == 0
    assert concentration_gradient[0] == 0
    assert zero_tangent == 0
    np.testing.assert_allclose(value_gradient[1:], expected_value_gradient, rtol=3e-6)
    np.testing.assert_allclose(
        concentration_gradient[1:],
        expected_concentration_gradient,
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        positive_concentration_tangent,
        expected_concentration_gradient[0],
        rtol=3e-6,
    )


def test_dirichlet_logpdf_boundary_jvp_is_zero_preserving_below_cutoff() -> None:
    value = jnp.array([0.0, 1.0])
    concentration = jnp.array([1.0, 2.0])

    result, zero_tangent = jax.jvp(
        dirichlet_logpdf,
        (value, concentration),
        (jnp.zeros_like(value), jnp.zeros_like(concentration)),
    )
    _, positive_concentration_tangent = jax.jvp(
        dirichlet_logpdf,
        (value, concentration),
        (jnp.zeros_like(value), jnp.array([0.0, 1.0])),
    )

    np.testing.assert_allclose(result, jnp.log(2.0), rtol=3e-6)
    assert zero_tangent == 0
    np.testing.assert_allclose(positive_concentration_tangent, 0.5, rtol=3e-6)


@pytest.mark.parametrize("first_concentration", [0.5, 2.0])
def test_nonfinite_dirichlet_boundary_gradients_match_beta(first_concentration: float) -> None:
    value = jnp.array([0.0, 1.0])
    concentration = jnp.array([first_concentration, 20.0])

    result = dirichlet_logpdf(value, concentration)
    gradient = jax.grad(dirichlet_logpdf, argnums=1)(value, concentration)
    expected = beta_logpdf(value[0], concentration[0], concentration[1])
    expected_gradient = jnp.asarray(
        jax.grad(beta_logpdf, argnums=(1, 2))(
            value[0],
            concentration[0],
            concentration[1],
        )
    )

    assert jnp.array_equal(result, expected)
    assert jnp.array_equal(gradient, expected_gradient)


def test_dirichlet_logpdf_marks_path_dependent_boundary_limit_as_nan() -> None:
    result = dirichlet_logpdf(jnp.array([0.0, 0.0, 1.0]), jnp.array([0.5, 2.0, 1.0]))

    assert jnp.isnan(result)


def test_dirichlet_logpdf_enforces_simplex_support() -> None:
    values = jnp.array(
        [
            [-0.1, 0.6, 0.5],
            [0.2, 0.3, 0.4],
            [0.2, 0.3, jnp.inf],
            [0.2, 0.3, jnp.nan],
        ]
    )

    result = dirichlet_logpdf(values, jnp.ones(3))

    assert jnp.all(jnp.isneginf(result[:3]))
    assert jnp.isnan(result[3])


def test_dirichlet_logpdf_rejects_invalid_concentration_before_support_check() -> None:
    concentrations = jnp.array(
        [
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [jnp.inf, 1.0, 1.0],
            [jnp.nan, 1.0, 1.0],
        ]
    )

    result = dirichlet_logpdf(jnp.array([-0.1, 0.5, 0.6]), concentrations)

    assert jnp.all(jnp.isnan(result))


@pytest.mark.parametrize("concentration", [0.5, 1.0, 2.0, 1e8])
def test_single_component_dirichlet_has_zero_log_density(concentration: float) -> None:
    value = jnp.ones((1,))
    shape = jnp.array([concentration])

    result, shape_gradient = jax.value_and_grad(dirichlet_logpdf, argnums=1)(
        value,
        shape,
    )
    value_gradient = jax.grad(dirichlet_logpdf, argnums=0)(value, shape)

    assert result == 0
    assert shape_gradient[0] == 0
    np.testing.assert_allclose(value_gradient, concentration - 1, rtol=3e-6)


def test_dirichlet_logpdf_handles_empty_batch() -> None:
    result = dirichlet_logpdf(jnp.empty((0, 3)), jnp.ones(3))

    assert result.shape == (0,)
    assert dirichlet(jnp.empty((0, 3)), jnp.ones(3)) == 0


def test_dirichlet_logpdf_integrates_with_large_simplex_parameterization() -> None:
    parameterization = Simplex(shape=(8, 465))
    position = parameterization.initialize(jax.random.key(0))
    value = parameterization.constrain(position)
    concentration = jnp.linspace(0.5, 2.0, 465)

    result = jax.jit(dirichlet_logpdf)(value, concentration)

    assert result.shape == (8,)
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    ("value", "concentration", "message"),
    [
        (0.5, jnp.ones(2), "value must include a final Dirichlet event axis"),
        (jnp.array([0.5, 0.5]), 1.0, "concentration must include a final Dirichlet event axis"),
        (jnp.empty((0,)), jnp.empty((0,)), "Dirichlet event size must be positive"),
        (jnp.ones(2) / 2, jnp.ones(3), "must have the same final event size"),
        (jnp.ones((2, 3)) / 3, jnp.ones((4, 3)), "Dirichlet batch shapes must be broadcastable"),
    ],
)
def test_dirichlet_logpdf_rejects_invalid_shapes(value, concentration, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dirichlet_logpdf(value, concentration)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_dirichlet_logpdf_uses_float32_for_low_precision_inputs(dtype) -> None:
    result = dirichlet_logpdf(
        jnp.array([0.2, 0.3, 0.5], dtype=dtype),
        jnp.ones(3, dtype=dtype),
    )

    assert result.dtype == jnp.dtype(jnp.float32)
