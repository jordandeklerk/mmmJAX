"""Tests for Categorical distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mmmjax import (
    bernoulli_logit_logpmf,
    bernoulli_logpmf,
    categorical,
    categorical_logit,
    categorical_logit_logpmf,
    categorical_logit_rng,
    categorical_logpmf,
    categorical_rng,
)


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


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (categorical_logpmf, jnp.array([0.2, 0.3, 0.5])),
        (categorical_logit_logpmf, jnp.array([-1.0, 0.0, 1.0])),
    ],
)
def test_categorical_logpmfs_require_supported_integer_categories(
    function,
    parameters,
) -> None:
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

    result = function(values, parameters)

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

    supported_result = categorical_logpmf(0, probabilities)
    unsupported_result = categorical_logpmf(3, probabilities)

    assert jnp.isfinite(supported_result[0])
    assert jnp.isneginf(unsupported_result[0])
    assert jnp.all(jnp.isnan(supported_result[1:]))
    assert jnp.all(jnp.isnan(unsupported_result[1:]))


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


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (categorical_logpmf, jnp.array([0.2, 0.3, 0.5])),
        (categorical_logpmf, jnp.array([0.0, 0.5, 0.5])),
        (categorical_logit_logpmf, jnp.array([-1.0, 0.0, 1.0])),
        (categorical_logit_logpmf, jnp.array([-jnp.inf, 0.0, 1.0])),
    ],
)
def test_unsupported_categorical_values_have_zero_parameter_derivatives(function, parameters) -> None:
    values = jnp.array([-1.0, 0.5, 3.0, jnp.inf, jnp.nan])

    def evaluate(current_parameters):
        return function(values, current_parameters)

    forward_jacobian = jax.jacfwd(evaluate)(parameters)
    reverse_jacobian = jax.jacrev(evaluate)(parameters)

    assert jnp.array_equal(forward_jacobian, jnp.zeros((values.size, parameters.size)))
    assert jnp.array_equal(reverse_jacobian, jnp.zeros((values.size, parameters.size)))


def test_single_category_categorical_has_zero_log_mass() -> None:
    probabilities = jnp.ones((1,))

    result = categorical_logpmf(jnp.array([0, 1]), probabilities)

    assert result[0] == 0
    assert jnp.isneginf(result[1])


def test_categorical_handles_empty_batch() -> None:
    values = jnp.empty((0,), dtype=jnp.int32)

    probability_result = categorical_logpmf(values, jnp.array([0.2, 0.3, 0.5]))
    logit_result = categorical_logit_logpmf(values, jnp.array([-1.0, 0.0, 1.0]))

    assert probability_result.shape == (0,)
    assert logit_result.shape == (0,)
    assert categorical(values, jnp.array([0.2, 0.3, 0.5])) == 0
    assert categorical_logit(values, jnp.array([-1.0, 0.0, 1.0])) == 0


def test_categorical_logpmf_composes_with_jit_and_vmap() -> None:
    values = jnp.array([0, 2])
    probabilities = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])

    compiled = jax.jit(categorical_logpmf)(values, probabilities)
    mapped = jax.vmap(categorical_logpmf)(values, probabilities)
    expected = jnp.array([jnp.log(0.2), jnp.log(0.2)])

    np.testing.assert_allclose(compiled, expected, rtol=3e-6)
    np.testing.assert_allclose(mapped, expected, rtol=3e-6)


def test_categorical_logit_logpmf_matches_scipy_log_softmax() -> None:
    values = np.array([0, 2, 1])
    logits = np.array(
        [
            [2.0, -1.0, 0.5],
            [-4.0, 3.0, 1.0],
            [0.2, 0.8, -0.5],
        ],
        dtype=np.float32,
    )
    expected = special.log_softmax(logits.astype(np.float64), axis=-1)[
        np.arange(values.size),
        values,
    ]

    result = categorical_logit_logpmf(values, logits)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_categorical_logit_matches_probability_parameterization() -> None:
    values = jnp.array([0, 2, 1])
    probabilities = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2],
            [0.6, 0.1, 0.3],
        ]
    )
    logits = jnp.log(probabilities) + jnp.array([[100.0], [-50.0], [2.5]])

    result = categorical_logit_logpmf(values, logits)
    expected = categorical_logpmf(values, probabilities)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_categorical_logit_logpmf_broadcasts_batch_axes() -> None:
    values = jnp.array([[0], [2]])
    logits = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-4.0, 3.0, 1.0],
            [0.2, 0.8, -0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    expected = special.log_softmax(np.asarray(logits, dtype=np.float64), axis=-1)[:, [0, 2]].T

    result = categorical_logit_logpmf(values, logits)

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_two_category_categorical_logit_matches_bernoulli_logit() -> None:
    values = jnp.array([0, 1])
    logits = jnp.array([-1000.0, -2.0, 0.0, 3.0, 1000.0])
    categorical_logits = jnp.stack((jnp.zeros_like(logits), logits), axis=-1)

    result = categorical_logit_logpmf(values[:, None], categorical_logits)
    expected = bernoulli_logit_logpmf(values[:, None], logits)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_categorical_logit_treats_positive_infinite_logits_as_undefined() -> None:
    values = jnp.array([0, 1])

    result = categorical_logit_logpmf(values, jnp.array([0.0, jnp.inf]))
    bernoulli_result = bernoulli_logit_logpmf(values, jnp.inf)

    assert jnp.all(jnp.isnan(result))
    assert jnp.array_equal(bernoulli_result, jnp.array([-jnp.inf, 0.0]))


def test_categorical_logit_logpmf_is_stable_for_extreme_finite_logits() -> None:
    logits = jnp.array([1000.0, 0.0, -1000.0])

    result = categorical_logit_logpmf(jnp.arange(3), logits)

    np.testing.assert_allclose(result, jnp.array([0.0, -1000.0, -2000.0]), rtol=3e-6)


def test_categorical_logit_logpmf_supports_masked_categories() -> None:
    logits = jnp.array([0.0, -jnp.inf, jnp.log(2.0)])
    expected = jnp.array([-jnp.log(3.0), -jnp.inf, jnp.log(2 / 3)])

    result = categorical_logit_logpmf(jnp.arange(3), logits)

    np.testing.assert_allclose(result, expected, rtol=3e-6)


def test_categorical_logit_logpmf_supports_one_available_category() -> None:
    logits = jnp.array([-jnp.inf, 2.0, -jnp.inf])

    result = categorical_logit_logpmf(jnp.arange(3), logits)

    assert jnp.isneginf(result[0])
    assert result[1] == 0
    assert jnp.isneginf(result[2])


@pytest.mark.parametrize(
    ("logits", "probabilities"),
    [
        (jnp.array([0.0, -jnp.inf, jnp.log(2.0)]), jnp.array([1 / 3, 0.0, 2 / 3])),
        (jnp.array([-jnp.inf, 2.0, -jnp.inf]), jnp.array([0.0, 1.0, 0.0])),
    ],
)
def test_categorical_logit_masked_category_derivatives_match_limiting_values(logits, probabilities) -> None:
    expected_jacobian = jnp.eye(3) - probabilities
    expected_hessian = -(jnp.diag(probabilities) - jnp.outer(probabilities, probabilities))

    def evaluate(current_logits):
        return categorical_logit_logpmf(jnp.arange(3), current_logits)

    forward_jacobian = jax.jit(jax.jacfwd(evaluate))(logits)
    reverse_jacobian = jax.jit(jax.jacrev(evaluate))(logits)
    hessian = jax.jit(jax.hessian(categorical_logit_logpmf, argnums=1))(1, logits)

    np.testing.assert_allclose(forward_jacobian, expected_jacobian, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(reverse_jacobian, expected_jacobian, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


def test_categorical_logit_logpmf_isolates_invalid_events_in_a_batch() -> None:
    values = jnp.array([1, 2, 0, 0, 0])
    logits = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, -jnp.inf, 1.0],
            [0.0, jnp.inf, 1.0],
            [0.0, jnp.nan, 1.0],
            [-jnp.inf, -jnp.inf, -jnp.inf],
        ]
    )
    valid_logits = np.array([[0.0, 1.0, 2.0], [0.0, -np.inf, 1.0]])
    expected = special.log_softmax(valid_logits, axis=-1)[[0, 1], [1, 2]]
    expected_gradient = np.eye(3)[[1, 2]] - special.softmax(valid_logits, axis=-1)

    def evaluate(current_logits):
        return categorical_logit_logpmf(values, current_logits)

    compiled = jax.jit(evaluate)(logits)
    mapped = jax.vmap(categorical_logit_logpmf)(values, logits)
    gradient = jax.jit(jax.grad(lambda current: jnp.sum(evaluate(current)[:2])))(logits)

    np.testing.assert_allclose(compiled[:2], expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(mapped[:2], expected, rtol=3e-6, atol=3e-6)
    assert jnp.all(jnp.isnan(compiled[2:]))
    assert jnp.all(jnp.isnan(mapped[2:]))
    np.testing.assert_allclose(gradient[:2], expected_gradient, rtol=3e-6, atol=3e-6)
    assert jnp.array_equal(gradient[2:], jnp.zeros((3, 3)))


def test_categorical_logit_logpmf_rejects_undefined_logit_events_before_support() -> None:
    logits = jnp.array(
        [
            [0.0, -jnp.inf, 1.0],
            [0.0, jnp.inf, 1.0],
            [0.0, jnp.nan, 1.0],
            [-jnp.inf, -jnp.inf, -jnp.inf],
        ]
    )

    supported_result = categorical_logit_logpmf(0, logits)
    unsupported_result = categorical_logit_logpmf(3, logits)

    assert jnp.isfinite(supported_result[0])
    assert jnp.isneginf(unsupported_result[0])
    assert jnp.all(jnp.isnan(supported_result[1:]))
    assert jnp.all(jnp.isnan(unsupported_result[1:]))


def test_categorical_logit_derivatives_match_closed_form() -> None:
    logits = jnp.array([-1.0, 0.5, 2.0])
    probabilities = jax.nn.softmax(logits)
    expected_gradient = jnp.array([0.0, 1.0, 0.0]) - probabilities
    expected_hessian = -(jnp.diag(probabilities) - jnp.outer(probabilities, probabilities))

    gradient = jax.jit(jax.grad(categorical_logit_logpmf, argnums=1))(
        1,
        logits,
    )
    hessian = jax.jit(jax.hessian(categorical_logit_logpmf, argnums=1))(
        1,
        logits,
    )

    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


def test_categorical_logit_sums_log_masses() -> None:
    values = jnp.array([0, 2, 2, 1])
    logits = jnp.array([0.2, -0.5, 1.3])
    expected = np.sum(special.log_softmax(np.array([0.2, -0.5, 1.3]))[np.array([0, 2, 2, 1])])

    result = categorical_logit(values, logits)

    assert result.shape == ()
    np.testing.assert_allclose(result, expected, rtol=3e-6)


def test_categorical_logit_gradient_accumulates_repeated_categories() -> None:
    values = jnp.array([0, 2, 2, 1, 2])
    logits = jnp.array([0.2, -0.5, 1.3])
    expected = np.array([1.0, 1.0, 3.0]) - values.size * special.softmax(np.array([0.2, -0.5, 1.3]))

    gradient = jax.jit(jax.grad(categorical_logit, argnums=1))(values, logits)

    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=3e-6)


def test_categorical_logit_propagates_impossible_and_undefined_terms() -> None:
    logits = jnp.array([0.0, -jnp.inf, 1.0])

    assert jnp.isneginf(categorical_logit(jnp.array([0, 1, 2]), logits))
    assert jnp.isnan(categorical_logit(jnp.array([0.0, jnp.nan]), logits))
    assert jnp.isnan(categorical_logit(0, jnp.array([0.0, jnp.nan, 1.0])))


def test_categorical_logit_logpmf_composes_with_jit_and_vmap() -> None:
    values = jnp.array([0, 2])
    logits = jnp.array([[2.0, -1.0, 0.5], [-4.0, 3.0, 1.0]])
    expected = jnp.array(
        [
            jax.nn.log_softmax(logits[0])[0],
            jax.nn.log_softmax(logits[1])[2],
        ]
    )

    compiled = jax.jit(categorical_logit_logpmf)(values, logits)
    mapped = jax.vmap(categorical_logit_logpmf)(values, logits)

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


@pytest.mark.parametrize(
    ("value", "logits", "message"),
    [
        (0, 1.0, "logits must include a final Categorical event axis"),
        (0, jnp.empty((0,)), "Categorical event size must be positive"),
        (
            jnp.ones(2),
            jnp.ones((3, 4)),
            "Categorical batch shapes must be broadcastable",
        ),
    ],
)
def test_categorical_logit_logpmf_rejects_invalid_shapes(
    value,
    logits,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        categorical_logit_logpmf(value, logits)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_categorical_logpmfs_use_float32_for_low_precision_inputs(dtype) -> None:
    probability_result = categorical_logpmf(
        1,
        jnp.array([0.2, 0.3, 0.5], dtype=dtype),
    )
    logit_result = categorical_logit_logpmf(
        1,
        jnp.array([-1.0, 0.0, 1.0], dtype=dtype),
    )

    assert probability_result.dtype == jnp.dtype(jnp.float32)
    assert logit_result.dtype == jnp.dtype(jnp.float32)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_categorical_observations_do_not_control_parameter_dtype() -> None:
    probability_result = categorical_logpmf(
        jnp.int64(1),
        jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32),
    )
    logit_result = categorical_logit_logpmf(
        jnp.int64(1),
        jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32),
    )

    assert probability_result.dtype == jnp.dtype(jnp.float32)
    assert logit_result.dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (categorical_logpmf, (0.0 + 0.0j, jnp.array([0.2, 0.8])), "value"),
        (categorical_logpmf, (0, jnp.array([0.2 + 0.0j, 0.8 + 0.0j])), "probabilities"),
        (categorical_logit_logpmf, (0, jnp.array([0.0 + 0.0j, 1.0 + 0.0j])), "logits"),
        (categorical_rng, (jax.random.key(0), jnp.array([0.2 + 0.0j, 0.8 + 0.0j])), "probabilities"),
        (categorical_logit_rng, (jax.random.key(0), jnp.array([0.0 + 0.0j, 1.0 + 0.0j])), "logits"),
    ],
)
def test_categorical_functions_reject_complex_arguments(
    function,
    arguments,
    argument_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=rf"argument '{argument_name}' must have a real numeric dtype, got complex",
    ):
        function(*arguments)


def test_categorical_rng_matches_centered_jax_categorical_high_mode_and_shape() -> None:
    key = jax.random.key(42)
    probabilities = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=jnp.float32)
    logits = jnp.log(probabilities)
    centered_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    expected = jax.random.categorical(
        key,
        centered_logits,
        shape=(4, 2),
        mode="high",
    ).astype(jnp.int32)

    result = categorical_rng(key, probabilities, sample_shape=(4,))

    assert result.shape == (4, 2)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)


def test_categorical_logit_rng_matches_centered_jax_categorical_high_mode_and_shape() -> None:
    key = jax.random.key(5)
    logits = jnp.array([[2.0, -1.0, 0.5], [-4.0, 3.0, 1.0]])
    centered_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    expected = jax.random.categorical(key, centered_logits, shape=(8, 2), mode="high").astype(jnp.int32)

    result = categorical_logit_rng(key, logits, sample_shape=(8,))

    assert result.shape == (8, 2)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)


def test_categorical_rngs_prepend_sample_axes_and_drop_the_event_axis() -> None:
    key = jax.random.key(3)

    probability_samples = categorical_rng(key, jnp.full((2, 3, 4), 0.25), sample_shape=(5, 6))
    logit_samples = categorical_logit_rng(key, jnp.zeros((2, 3, 4)), sample_shape=(5, 6))

    assert probability_samples.shape == (5, 6, 2, 3)
    assert logit_samples.shape == (5, 6, 2, 3)


def test_categorical_rng_handles_deterministic_probabilities() -> None:
    probabilities = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = categorical_rng(jax.random.key(0), probabilities, sample_shape=(32,))

    assert jnp.all(result[:, 0] == 0)
    assert jnp.all(result[:, 1] == 2)


def test_categorical_logit_rng_never_draws_masked_categories() -> None:
    logits = jnp.array([[-jnp.inf, 2.0, -jnp.inf], [0.0, -jnp.inf, 0.0]])

    result = categorical_logit_rng(jax.random.key(0), logits, sample_shape=(64,))

    assert jnp.all(result[:, 0] == 1)
    assert jnp.all(result[:, 1] != 1)


def test_categorical_logit_rng_is_shift_invariant_for_large_logits() -> None:
    key = jax.random.key(9)
    logits = jnp.array([0.0, 1.0, 2.0])

    result = categorical_logit_rng(key, logits + 1e7, sample_shape=(256,))
    expected = categorical_logit_rng(key, logits, sample_shape=(256,))

    assert jnp.array_equal(result, expected)


def test_categorical_rngs_match_expected_frequencies() -> None:
    probabilities = jnp.array([0.2, 0.3, 0.5])
    logits = jnp.log(probabilities) + 7.0

    probability_samples = categorical_rng(jax.random.key(7), probabilities, sample_shape=(50_000,))
    logit_samples = categorical_logit_rng(jax.random.key(8), logits, sample_shape=(50_000,))
    probability_frequencies = jnp.bincount(probability_samples, length=3) / probability_samples.size
    logit_frequencies = jnp.bincount(logit_samples, length=3) / logit_samples.size

    np.testing.assert_allclose(probability_frequencies, probabilities, rtol=0, atol=0.01)
    np.testing.assert_allclose(logit_frequencies, probabilities, rtol=0, atol=0.01)


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (categorical_rng, jnp.array([0.2, 0.3, 0.5])),
        (categorical_logit_rng, jnp.array([-1.0, 0.0, 1.0])),
    ],
)
def test_categorical_rngs_support_jit_and_vmap_over_keys(function, parameters) -> None:
    keys = jax.random.split(jax.random.key(1), 3)

    compiled = jax.jit(function, static_argnames="sample_shape")(keys[0], parameters, sample_shape=(4,))
    mapped = jax.vmap(lambda key: function(key, parameters, sample_shape=(4,)))(keys)

    assert jnp.array_equal(compiled, function(keys[0], parameters, sample_shape=(4,)))
    assert mapped.shape == (3, 4)
    assert jnp.array_equal(mapped[2], function(keys[2], parameters, sample_shape=(4,)))


@pytest.mark.parametrize(
    ("function", "parameters", "message"),
    [
        (categorical_rng, 1.0, "probabilities must include a final Categorical event axis"),
        (categorical_rng, jnp.empty((0,)), "Categorical event size must be positive"),
        (categorical_logit_rng, 1.0, "logits must include a final Categorical event axis"),
        (categorical_logit_rng, jnp.empty((0,)), "Categorical event size must be positive"),
    ],
)
def test_categorical_rngs_reject_invalid_event_shapes(function, parameters, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function(jax.random.key(0), parameters)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_categorical_rngs_compute_with_float32_for_low_precision_inputs(dtype) -> None:
    key = jax.random.key(0)
    probabilities = jnp.array([0.2, 0.3, 0.5], dtype=dtype)
    logits = jnp.array([-1.0, 0.0, 1.0], dtype=dtype)

    probability_result = categorical_rng(key, probabilities, sample_shape=(256,))
    logit_result = categorical_logit_rng(key, logits, sample_shape=(256,))

    assert probability_result.dtype == jnp.dtype(jnp.int32)
    assert logit_result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(
        probability_result, categorical_rng(key, probabilities.astype(jnp.float32), sample_shape=(256,))
    )
    assert jnp.array_equal(logit_result, categorical_logit_rng(key, logits.astype(jnp.float32), sample_shape=(256,)))
