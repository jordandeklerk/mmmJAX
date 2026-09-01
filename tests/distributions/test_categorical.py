"""Tests for Categorical distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import bernoulli_logpmf, categorical, categorical_logpmf


def test_categorical_logpmf_matches_scipy_multinomial_reference() -> None:
    probabilities = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    values = np.arange(probabilities.size)
    expected = stats.multinomial.logpmf(
        np.eye(probabilities.size, dtype=np.int32),
        n=1,
        p=np.array([0.2, 0.3, 0.5], dtype=np.float64),
    )

    result = categorical_logpmf(values, probabilities)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_two_category_categorical_matches_bernoulli() -> None:
    values = jnp.array([0, 1])
    success_probabilities = jnp.array([0.1, 0.4, 0.8])
    categorical_probabilities = jnp.stack(
        (1 - success_probabilities, success_probabilities),
        axis=-1,
    )

    result = categorical_logpmf(
        values[:, None],
        categorical_probabilities,
    )
    expected = bernoulli_logpmf(values[:, None], success_probabilities)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_categorical_logpmf_broadcasts_batch_axes() -> None:
    values = jnp.array([[0], [2]])
    probabilities = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2],
            [0.6, 0.1, 0.3],
            [0.25, 0.25, 0.5],
        ]
    )
    expected = np.log(
        np.array(
            [
                [0.2, 0.1, 0.6, 0.25],
                [0.5, 0.2, 0.3, 0.5],
            ]
        )
    )

    result = categorical_logpmf(values, probabilities)

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_categorical_probabilities_normalize_over_categories() -> None:
    probabilities = jnp.array([0.05, 0.15, 0.3, 0.5])

    log_masses = categorical_logpmf(jnp.arange(4), probabilities)

    np.testing.assert_allclose(jnp.sum(jnp.exp(log_masses)), 1, rtol=3e-6)


def test_categorical_returns_scalar_sum() -> None:
    values = jnp.array([0, 2, 1, 2])
    probabilities = jnp.array([0.2, 0.3, 0.5])
    expected = jnp.log(0.2) + 2 * jnp.log(0.5) + jnp.log(0.3)

    result = categorical(values, probabilities)

    assert result.shape == ()
    np.testing.assert_allclose(result, expected, rtol=3e-6)


def test_categorical_logpmf_handles_zero_probabilities() -> None:
    probabilities = jnp.array([1.0, 0.0, 0.0])

    result = categorical_logpmf(jnp.arange(3), probabilities)
    gradient = jax.grad(lambda current: categorical_logpmf(0, current))(probabilities)

    assert result[0] == 0
    assert jnp.all(jnp.isneginf(result[1:]))
    assert jnp.array_equal(gradient, jnp.array([1.0, 0.0, 0.0]))


def test_categorical_logpmf_requires_supported_integer_categories() -> None:
    values = jnp.array(
        [
            -jnp.inf,
            -1.0,
            -0.0,
            0.5,
            0.0,
            2.0,
            3.0,
            jnp.inf,
            jnp.nan,
        ]
    )

    result = categorical_logpmf(values, jnp.array([0.2, 0.3, 0.5]))

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 3, 6, 7])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([2, 4, 5])]))
    assert jnp.isnan(result[8])


def test_categorical_logpmf_rejects_invalid_probability_events_before_support() -> None:
    probabilities = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.4],
            [-0.1, 0.6, 0.5],
            [jnp.inf, 0.0, 0.0],
            [jnp.nan, 0.3, 0.7],
        ]
    )

    result = categorical_logpmf(3, probabilities)

    assert jnp.isneginf(result[0])
    assert jnp.all(jnp.isnan(result[1:]))


def test_categorical_probability_derivatives_match_closed_form() -> None:
    probabilities = jnp.array([0.2, 0.3, 0.5])
    expected_gradient = jnp.array([0.0, 1 / 0.3, 0.0])
    expected_hessian = jnp.diag(jnp.array([0.0, -1 / 0.3**2, 0.0]))

    gradient = jax.jit(jax.grad(categorical_logpmf, argnums=1))(
        1,
        probabilities,
    )
    hessian = jax.jit(jax.hessian(categorical_logpmf, argnums=1))(
        1,
        probabilities,
    )

    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6)


def test_categorical_gradient_accumulates_repeated_categories() -> None:
    values = jnp.array([0, 2, 2, 1, 2])
    probabilities = jnp.array([0.2, 0.3, 0.5])
    expected = jnp.array([1, 1, 3]) / probabilities

    gradient = jax.jit(jax.grad(categorical, argnums=1))(
        values,
        probabilities,
    )

    np.testing.assert_allclose(gradient, expected, rtol=3e-6)


def test_single_category_categorical_has_zero_log_mass() -> None:
    probabilities = jnp.ones((1,))

    result = categorical_logpmf(jnp.array([0, 1]), probabilities)

    assert result[0] == 0
    assert jnp.isneginf(result[1])


def test_categorical_handles_empty_batch() -> None:
    values = jnp.empty((0,), dtype=jnp.int32)

    result = categorical_logpmf(values, jnp.array([0.2, 0.3, 0.5]))

    assert result.shape == (0,)
    assert categorical(values, jnp.array([0.2, 0.3, 0.5])) == 0


def test_categorical_logpmf_composes_with_jit_and_vmap() -> None:
    values = jnp.array([0, 2])
    probabilities = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])

    compiled = jax.jit(categorical_logpmf)(values, probabilities)
    mapped = jax.vmap(categorical_logpmf)(values, probabilities)
    expected = jnp.array([jnp.log(0.2), jnp.log(0.2)])

    np.testing.assert_allclose(compiled, expected, rtol=3e-6)
    np.testing.assert_allclose(mapped, expected, rtol=3e-6)


@pytest.mark.parametrize(
    ("value", "probabilities", "message"),
    [
        (0, 1.0, "probabilities must include a final Categorical event axis"),
        (0, jnp.empty((0,)), "Categorical event size must be positive"),
        (
            jnp.ones(2),
            jnp.ones((3, 4)) / 4,
            "Categorical batch shapes must be broadcastable",
        ),
    ],
)
def test_categorical_logpmf_rejects_invalid_shapes(
    value,
    probabilities,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        categorical_logpmf(value, probabilities)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_categorical_logpmf_uses_float32_for_low_precision_inputs(dtype) -> None:
    result = categorical_logpmf(
        1,
        jnp.array([0.2, 0.3, 0.5], dtype=dtype),
    )

    assert result.dtype == jnp.dtype(jnp.float32)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_categorical_observations_do_not_control_parameter_dtype() -> None:
    result = categorical_logpmf(
        jnp.int64(1),
        jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32),
    )

    assert result.dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("value", "probabilities", "argument_name"),
    [
        (0.0 + 0.0j, jnp.array([0.2, 0.8]), "value"),
        (0, jnp.array([0.2 + 0.0j, 0.8 + 0.0j]), "probabilities"),
    ],
)
def test_categorical_functions_reject_complex_arguments(
    value,
    probabilities,
    argument_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"argument '{argument_name}' must have a real numeric dtype, got complex",
    ):
        categorical_logpmf(value, probabilities)
