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


def test_single_component_dirichlet_has_zero_log_density() -> None:
    result = jax.jit(dirichlet_logpdf)(jnp.ones((2, 1)), jnp.array([0.5]))

    assert jnp.array_equal(result, jnp.zeros(2))


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
