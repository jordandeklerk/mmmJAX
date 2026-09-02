"""Tests for Multinomial distribution functions."""

from collections.abc import Callable
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multinomial as jax_multinomial_distribution
from scipy import special, stats

from mmmjax import (
    binomial_logit_logpmf,
    binomial_logpmf,
    categorical_logit_logpmf,
    categorical_logpmf,
    multinomial,
    multinomial_logit,
    multinomial_logit_logpmf,
    multinomial_logit_rng,
    multinomial_logpmf,
    multinomial_rng,
)


def test_multinomial_logpmf_matches_scipy_across_broadcast_batches() -> None:
    values = np.array([[[2, 1, 0]], [[1, 3, 1]]], dtype=np.int32)
    probabilities = np.array(
        [
            [0.25, 0.25, 0.5],
            [0.125, 0.625, 0.25],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [stats.multinomial.logpmf(value[0], n=np.sum(value), p=probability) for probability in probabilities]
            for value in values
        ]
    )

    result = multinomial_logpmf(values, probabilities)
    compiled = jax.jit(multinomial_logpmf)(values, probabilities)

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-6)


def test_multinomial_logpmf_matches_jax_for_one_event() -> None:
    values = jnp.array([2, 1, 3])
    probabilities = jnp.array([0.25, 0.25, 0.5])
    expected = jax_multinomial_distribution.logpmf(values, jnp.sum(values), probabilities)

    result = multinomial_logpmf(values, probabilities)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_multinomial_logit_logpmf_matches_explicit_reference_across_broadcast_batches() -> None:
    values = np.array([[[2, 1, 0]], [[1, 3, 1]]], dtype=np.int32)
    logits = np.array(
        [
            [-2.0, -0.5, 1.0],
            [0.25, 0.5, -1.0],
            [2.0, -3.0, 0.75],
            [-0.5, -0.25, 0.0],
        ],
        dtype=np.float32,
    )
    expected = np.array([[_multinomial_logit_reference(value[0], logit) for logit in logits] for value in values])

    result = multinomial_logit_logpmf(values, logits)
    compiled = jax.jit(multinomial_logit_logpmf)(values, logits)

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(compiled, expected, rtol=3e-6, atol=3e-6)


def test_multinomial_logpmfs_support_vmap() -> None:
    values = jnp.array([[2, 1, 0], [1, 3, 1], [4, 0, 2]])
    probabilities = jnp.array(
        [
            [0.25, 0.25, 0.5],
            [0.125, 0.625, 0.25],
            [0.5, 0.25, 0.25],
        ]
    )
    logits = jnp.log(probabilities)

    probability_result = jax.jit(jax.vmap(multinomial_logpmf))(values, probabilities)
    logit_result = jax.jit(jax.vmap(multinomial_logit_logpmf))(values, logits)
    expected = np.array(
        [
            stats.multinomial.logpmf(value, n=np.sum(value), p=probability)
            for value, probability in zip(np.asarray(values), np.asarray(probabilities), strict=True)
        ]
    )

    np.testing.assert_allclose(probability_result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(logit_result, expected, rtol=3e-6, atol=3e-6)


def test_one_draw_multinomial_matches_categorical() -> None:
    values = jnp.eye(3, dtype=jnp.int32)
    categories = jnp.arange(3)
    probabilities = jnp.array([0.2, 0.3, 0.5])
    logits = jnp.array([-1.0, 0.5, 2.0])

    np.testing.assert_allclose(
        multinomial_logpmf(values, probabilities),
        categorical_logpmf(categories, probabilities),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        multinomial_logit_logpmf(values, logits),
        categorical_logit_logpmf(categories, logits),
        rtol=3e-6,
        atol=3e-6,
    )


def test_two_category_multinomial_matches_binomial() -> None:
    successes = jnp.array([0, 2, 7])
    trials = jnp.array([4, 5, 7])
    values = jnp.stack((trials - successes, successes), axis=-1)
    probability = jnp.array([0.2, 0.6, 0.85])
    probabilities = jnp.stack((1 - probability, probability), axis=-1)
    logits = jnp.log(probabilities)
    binomial_logits = logits[..., 1] - logits[..., 0]

    np.testing.assert_allclose(
        multinomial_logpmf(values, probabilities),
        binomial_logpmf(successes, trials, probability),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        multinomial_logit_logpmf(values, logits),
        binomial_logit_logpmf(successes, trials, binomial_logits),
        rtol=3e-6,
        atol=3e-6,
    )


def test_multinomial_probability_mass_normalizes() -> None:
    values = jnp.asarray(_count_compositions(total=4, categories=3))
    probabilities = jnp.array([0.2, 0.3, 0.5])
    logits = jnp.array([-1.0, 0.5, 2.0])

    probability_total = jnp.sum(jnp.exp(multinomial_logpmf(values, probabilities)))
    logit_total = jnp.sum(jnp.exp(multinomial_logit_logpmf(values, logits)))

    np.testing.assert_allclose(probability_total, 1, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(logit_total, 1, rtol=3e-6, atol=3e-6)


def test_multinomial_functions_return_scalar_sums() -> None:
    values = jnp.array([[2, 1, 0], [1, 3, 1]])
    probabilities = jnp.array([0.25, 0.25, 0.5])
    logits = jnp.log(probabilities)

    probability_result = multinomial(values, probabilities)
    logit_result = multinomial_logit(values, logits)
    expected = jnp.sum(multinomial_logpmf(values, probabilities))

    assert probability_result.shape == ()
    assert logit_result.shape == ()
    np.testing.assert_allclose(probability_result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(logit_result, expected, rtol=3e-6, atol=3e-6)


def test_multinomial_probability_derivatives_match_closed_form() -> None:
    values = jnp.array([[2, 1, 0], [1, 3, 3]])
    probabilities = jnp.array([0.2, 0.3, 0.5])
    total_count = jnp.sum(values, axis=0)
    expected_gradient = total_count / probabilities
    expected_hessian = jnp.diag(-total_count / jnp.square(probabilities))

    gradient = jax.jit(jax.jacrev(multinomial, argnums=1))(values, probabilities)
    forward_gradient = jax.jit(jax.jacfwd(multinomial, argnums=1))(values, probabilities)
    hessian = jax.jit(jax.hessian(multinomial, argnums=1))(values, probabilities)

    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(forward_gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


def test_multinomial_logit_derivatives_match_closed_form() -> None:
    values = jnp.array([[2, 1, 0], [1, 3, 3]])
    logits = jnp.array([-1.0, 0.5, 2.0])
    probabilities = jax.nn.softmax(logits)
    category_counts = jnp.sum(values, axis=0)
    total_count = jnp.sum(values)
    expected_gradient = category_counts - total_count * probabilities
    expected_hessian = -total_count * (jnp.diag(probabilities) - jnp.outer(probabilities, probabilities))

    gradient = jax.jit(jax.jacrev(multinomial_logit, argnums=1))(values, logits)
    forward_gradient = jax.jit(jax.jacfwd(multinomial_logit, argnums=1))(values, logits)
    hessian = jax.jit(jax.hessian(multinomial_logit, argnums=1))(values, logits)

    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(forward_gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("function", [multinomial_logpmf, multinomial_logit_logpmf])
def test_multinomial_logpmfs_require_nonnegative_integer_counts(
    function: Callable[[Any, Any], jax.Array],
) -> None:
    values = jnp.array(
        [
            [0.0, 2.0, 3.0],
            [-0.0, 2.0, 3.0],
            [-1.0, 2.0, 3.0],
            [0.5, 2.0, 3.0],
            [jnp.inf, 2.0, 3.0],
            [-jnp.inf, 2.0, 3.0],
            [jnp.nan, 2.0, 3.0],
        ]
    )
    parameters = jnp.array([0.2, 0.3, 0.5]) if function is multinomial_logpmf else jnp.array([-1.0, 0.5, 2.0])

    result = jax.jit(function)(values, parameters)

    assert jnp.all(jnp.isfinite(result[:2]))
    assert jnp.all(jnp.isneginf(result[2:6]))
    assert jnp.isnan(result[6])


def test_multinomial_logpmf_rejects_invalid_probabilities_before_count_support() -> None:
    values = jnp.array([-1, 2, 3])
    probabilities = jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.4],
            [-0.1, 0.6, 0.5],
            [jnp.inf, 0.0, 0.0],
            [jnp.nan, 0.3, 0.7],
        ]
    )

    result = multinomial_logpmf(values, probabilities)

    assert jnp.isneginf(result[0])
    assert jnp.all(jnp.isnan(result[1:]))


def test_multinomial_logpmf_uses_dtype_specific_simplex_tolerance() -> None:
    values = jnp.array([1, 2, 3])
    accepted = jnp.array([0.2, 0.3, 0.5 + 5e-7], dtype=jnp.float32)
    rejected = jnp.array([0.2, 0.3, 0.5 + 2e-6], dtype=jnp.float32)
    accepted_float64 = np.asarray(accepted, dtype=np.float64)
    expected = (
        special.gammaln(7)
        - np.sum(special.gammaln(np.asarray(values, dtype=np.float64) + 1))
        + np.dot(np.asarray(values, dtype=np.float64), np.log(accepted_float64))
    )

    accepted_result = multinomial_logpmf(values, accepted)
    rejected_result = multinomial_logpmf(values, rejected)
    gradient = jax.grad(multinomial_logpmf, argnums=1)(values, accepted)

    assert jnp.abs(jnp.sum(accepted) - 1) <= 1e-6
    assert jnp.abs(jnp.sum(rejected) - 1) > 1e-6
    np.testing.assert_allclose(accepted_result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gradient, values / accepted, rtol=3e-6, atol=3e-6)
    assert jnp.isnan(rejected_result)


def test_multinomial_logit_logpmf_rejects_undefined_logits_before_count_support() -> None:
    values = jnp.array([-1, 2, 3])
    logits = jnp.array(
        [
            [-jnp.inf, 0.0, 1.0],
            [jnp.inf, 0.0, 1.0],
            [jnp.nan, 0.0, 1.0],
            [-jnp.inf, -jnp.inf, -jnp.inf],
        ]
    )

    result = multinomial_logit_logpmf(values, logits)

    assert jnp.isneginf(result[0])
    assert jnp.all(jnp.isnan(result[1:]))


def test_invalid_multinomial_parameter_batches_have_zero_cotangents() -> None:
    values = jnp.array([2, 1, 0])
    probabilities = jnp.array([[0.25, 0.25, 0.5], [0.2, 0.3, 0.4]])
    logits = jnp.array([[0.0, 1.0, 2.0], [jnp.inf, 0.0, 1.0]])

    probability_result = multinomial_logpmf(values, probabilities)
    logit_result = multinomial_logit_logpmf(values, logits)
    probability_gradient = jax.grad(lambda current: multinomial_logpmf(values, current)[1])(probabilities)
    logit_gradient = jax.grad(lambda current: multinomial_logit_logpmf(values, current)[1])(logits)

    assert jnp.isnan(probability_result[1])
    assert jnp.isnan(logit_result[1])
    assert jnp.array_equal(probability_gradient, jnp.zeros_like(probability_gradient))
    assert jnp.array_equal(logit_gradient, jnp.zeros_like(logit_gradient))


def test_multinomial_boundaries_and_degenerate_events() -> None:
    zero_count = jnp.array([0, 0, 0])
    deterministic_count = jnp.array([5, 0, 0])
    impossible_count = jnp.array([4, 1, 0])
    probabilities = jnp.array([1.0, 0.0, 0.0])
    logits = jnp.array([0.0, -jnp.inf, -jnp.inf])

    assert multinomial_logpmf(zero_count, probabilities) == 0
    assert multinomial_logit_logpmf(zero_count, logits) == 0
    assert multinomial_logpmf(deterministic_count, probabilities) == 0
    assert multinomial_logit_logpmf(deterministic_count, logits) == 0
    assert jnp.isneginf(multinomial_logpmf(impossible_count, probabilities))
    assert jnp.isneginf(multinomial_logit_logpmf(impossible_count, logits))
    assert multinomial_logpmf(jnp.array([8]), jnp.array([1.0])) == 0
    assert multinomial_logit_logpmf(jnp.array([8]), jnp.array([2.0])) == 0


def test_multinomial_probability_boundary_derivatives_match_closed_form() -> None:
    values = jnp.array([0, 2, 3])
    probabilities = jnp.array([0.0, 0.4, 0.6])
    expected_gradient = jnp.array([0.0, 5.0, 5.0])
    expected_hessian = jnp.diag(jnp.array([0.0, -12.5, -25 / 3]))

    gradient = jax.grad(multinomial_logpmf, argnums=1)(values, probabilities)
    hessian = jax.hessian(multinomial_logpmf, argnums=1)(values, probabilities)

    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


def test_multinomial_logit_mask_derivatives_match_limiting_values() -> None:
    values = jnp.array([1, 1, 1])
    logits = jnp.array([-jnp.inf, 0.0, jnp.log(2.0)])
    expected_gradient = jnp.array([1.0, 0.0, -1.0])
    probabilities = jnp.array([0.0, 1 / 3, 2 / 3])
    expected_hessian = -3 * (jnp.diag(probabilities) - jnp.outer(probabilities, probabilities))

    result = multinomial_logit_logpmf(values, logits)
    gradient = jax.grad(multinomial_logit_logpmf, argnums=1)(values, logits)
    hessian = jax.hessian(multinomial_logit_logpmf, argnums=1)(values, logits)

    assert jnp.isneginf(result)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=3e-6, atol=3e-6)


def test_multinomial_logit_logpmf_handles_extreme_logits_and_constant_shifts() -> None:
    values = np.array([2, 3, 5], dtype=np.int32)
    logits = np.array([1_000.0, 0.0, -1_000.0], dtype=np.float32)
    expected = _multinomial_logit_reference(values, logits)

    result = multinomial_logit_logpmf(values, logits)
    shifted = multinomial_logit_logpmf(values, logits + np.float32(10_000))

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(shifted, result, rtol=3e-6, atol=3e-6)


def test_multinomial_logit_logpmf_stays_accurate_with_large_counts_and_offset_logits() -> None:
    logits = np.array([1_000_000.0, 1_000_001.0, 999_998.0], dtype=np.float32)
    probabilities = special.softmax(logits.astype(np.float64))
    total = 1_000_000_000
    values = np.rint(total * probabilities).astype(np.int64)
    values[-1] = total - np.sum(values[:-1])
    values = values.astype(np.int32)
    expected = _sequential_multinomial_logpmf(values, probabilities)

    result = multinomial_logit_logpmf(values, logits)

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize(
    ("elementwise_function", "summed_function", "parameters"),
    [
        (multinomial_logpmf, multinomial, jnp.array([-0.1, 0.4, 0.7])),
        (multinomial_logit_logpmf, multinomial_logit, jnp.array([jnp.inf, 0.0, 1.0])),
    ],
)
def test_multinomial_empty_batch_returns_scalar_zero(
    elementwise_function: Callable[[Any, Any], jax.Array],
    summed_function: Callable[[Any, Any], jax.Array],
    parameters: jax.Array,
) -> None:
    values = jnp.empty((0, 3), dtype=jnp.int32)

    elementwise = jax.jit(elementwise_function)(values, parameters)
    summed = jax.jit(summed_function)(values, parameters)

    assert elementwise.shape == (0,)
    assert summed.shape == ()
    assert summed == 0


@pytest.mark.parametrize(
    ("function", "value", "parameters", "message"),
    [
        (multinomial_logpmf, 1, jnp.array([0.5, 0.5]), "value must include a final Multinomial event axis"),
        (multinomial_logpmf, jnp.array([1, 2]), 0.5, "probabilities must include a final Multinomial event axis"),
        (
            multinomial_logit_logpmf,
            jnp.empty((0,)),
            jnp.empty((0,)),
            "Multinomial event size must be positive",
        ),
        (
            multinomial_logpmf,
            jnp.array([1, 2]),
            jnp.array([0.2, 0.3, 0.5]),
            "value and probabilities must have the same final event size",
        ),
        (
            multinomial_logit_logpmf,
            jnp.ones((2, 3)),
            jnp.ones((4, 3)),
            "Multinomial batch shapes must be broadcastable",
        ),
    ],
)
def test_multinomial_logpmfs_raise_targeted_shape_errors(
    function: Callable[[Any, Any], jax.Array],
    value: Any,
    parameters: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(value, parameters)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("function", [multinomial_logpmf, multinomial_logit_logpmf])
def test_multinomial_logpmfs_promote_low_precision_parameters(
    function: Callable[[Any, Any], jax.Array],
    dtype: jnp.dtype,
) -> None:
    values = jnp.array([1, 1, 2])
    parameters = (
        jnp.array([0.25, 0.25, 0.5], dtype=dtype)
        if function is multinomial_logpmf
        else jnp.array([-1.0, 0.0, 1.0], dtype=dtype)
    )

    result = function(values, parameters)

    assert result.dtype == jnp.dtype(jnp.float32)
    assert jnp.isfinite(result)


@pytest.mark.parametrize(
    ("function", "value", "parameters", "argument"),
    [
        (multinomial_logpmf, jnp.array([1 + 1j, 2, 3]), jnp.array([0.2, 0.3, 0.5]), "value"),
        (multinomial_logpmf, jnp.array([1, 2, 3]), jnp.array([0.2 + 0j, 0.3, 0.5]), "probabilities"),
        (multinomial_logit_logpmf, jnp.array([1, 2, 3]), jnp.array([0.0 + 0j, 1.0, 2.0]), "logits"),
    ],
)
def test_multinomial_logpmfs_reject_complex_arguments(
    function: Callable[[Any, Any], jax.Array],
    value: Any,
    parameters: Any,
    argument: str,
) -> None:
    with pytest.raises(TypeError, match=rf"distribution argument '{argument}' must have a real numeric dtype"):
        function(value, parameters)


def test_multinomial_logpmfs_remain_accurate_for_large_counts() -> None:
    cases = [
        (
            np.array([200_000, 300_000, 500_000], dtype=np.int32),
            np.array([0.2, 0.3, 0.5], dtype=np.float32),
        ),
        (
            np.array([250_000_000, 250_000_000, 500_000_000], dtype=np.int32),
            np.array([0.25, 0.25, 0.5], dtype=np.float32),
        ),
        (
            np.array([1_024, 268_435_456, 805_305_344], dtype=np.int32),
            np.array([2**-20, 0.25, 0.75 - 2**-20], dtype=np.float32),
        ),
    ]

    for values, probabilities in cases:
        logits = np.log(probabilities)
        expected_probability = _sequential_multinomial_logpmf(values, probabilities)
        expected_logit = _multinomial_logit_reference(values, logits)
        probability_result = multinomial_logpmf(values, probabilities)
        logit_result = multinomial_logit_logpmf(values, logits)
        permutation = np.array([2, 0, 1])

        assert probability_result <= 0
        assert logit_result <= 0
        np.testing.assert_allclose(probability_result, expected_probability, rtol=1e-6, atol=5e-6)
        np.testing.assert_allclose(logit_result, expected_logit, rtol=1e-6, atol=5e-6)
        np.testing.assert_allclose(
            multinomial_logpmf(values[permutation], probabilities[permutation]),
            probability_result,
            rtol=1e-6,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            multinomial_logit_logpmf(values[permutation], logits[permutation]),
            logit_result,
            rtol=1e-6,
            atol=5e-6,
        )


def test_multinomial_logpmf_preserves_float32_simplex_residual_at_large_counts() -> None:
    values = np.full(10, 200_000_000, dtype=np.int32)
    probabilities = np.full(10, 0.1, dtype=np.float32)
    expected = _sequential_multinomial_logpmf(values, probabilities)

    result = multinomial_logpmf(values, probabilities)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=5e-6)


@pytest.mark.skipif(jax.config.x64_enabled, reason="the total fits when 64-bit counts are enabled")
def test_multinomial_logpmfs_reject_event_totals_beyond_int32() -> None:
    values = jnp.array([2_000_000_000, 200_000_000], dtype=jnp.int32)
    probabilities = jnp.array([0.5, 0.5])
    logits = jnp.zeros(2)

    assert jnp.isneginf(multinomial_logpmf(values, probabilities))
    assert jnp.isneginf(multinomial_logit_logpmf(values, logits))


def test_multinomial_logpmfs_preserve_small_counts_beside_float32_boundary() -> None:
    values = jnp.array([16_777_216, 1], dtype=jnp.int32)
    probabilities = jnp.array([1.0, 0.0], dtype=jnp.float32)
    logits = jnp.array([0.0, -jnp.inf], dtype=jnp.float32)

    assert jnp.isneginf(multinomial_logpmf(values, probabilities))
    assert jnp.isneginf(multinomial_logit_logpmf(values, logits))


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (multinomial_rng, jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.25] * 4])),
        (multinomial_logit_rng, jnp.array([[-2.0, -1.0, 0.0, 1.0], [1.0, 0.0, -1.0, -2.0], [0.0] * 4])),
    ],
)
def test_multinomial_rngs_preserve_sample_batch_and_event_axes(function, parameters) -> None:
    trials = jnp.array([[5], [9]])

    result = function(
        jax.random.key(4),
        parameters,
        trials,
        sample_shape=(6,),
    )
    expected_totals = jnp.broadcast_to(trials, (6, 2, 3))

    assert result.shape == (6, 2, 3, 4)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.all(result >= 0)
    assert jnp.array_equal(jnp.sum(result, axis=-1), expected_totals)


def test_multinomial_rng_matches_jax_when_largest_category_is_last() -> None:
    key = jax.random.key(8)
    trials = jnp.array([5, 11])
    probabilities = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.25, 0.65]])
    expected = jax.random.multinomial(
        key,
        trials.astype(probabilities.dtype),
        probabilities,
        shape=(7, 2, 3),
        dtype=probabilities.dtype,
    ).astype(jnp.int32)

    result = multinomial_rng(key, probabilities, trials, sample_shape=(7,))

    assert jnp.array_equal(result, expected)


def test_multinomial_rng_matches_analytic_moments() -> None:
    sample_size = 30_000
    trials = 20
    probabilities = np.array([0.5, 0.2, 0.3], dtype=np.float64)
    expected_mean = trials * probabilities
    expected_covariance = trials * (np.diag(probabilities) - np.outer(probabilities, probabilities))

    samples = np.asarray(
        multinomial_rng(
            jax.random.key(7),
            probabilities.astype(np.float32),
            trials,
            sample_shape=(sample_size,),
        ),
        dtype=np.float64,
    )
    sample_mean = np.mean(samples, axis=0)
    centered_samples = samples - expected_mean
    centered_products = centered_samples[:, :, None] * centered_samples[:, None, :]
    sample_covariance = np.mean(centered_products, axis=0)
    mean_standard_error = np.sqrt(np.diag(expected_covariance) / sample_size)
    covariance_standard_error = np.std(centered_products, axis=0, ddof=1) / np.sqrt(sample_size)

    assert np.all(np.abs(sample_mean - expected_mean) <= 6 * mean_standard_error)
    assert np.all(np.abs(sample_covariance - expected_covariance) <= 6 * covariance_standard_error)


def test_multinomial_rngs_handle_zero_trials_and_deterministic_events() -> None:
    probabilities = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    logits = jnp.array([[0.0, -jnp.inf, -jnp.inf], [-jnp.inf, -jnp.inf, 0.0]])

    zero_samples = multinomial_rng(
        jax.random.key(0),
        probabilities,
        0,
        sample_shape=(8,),
    )
    deterministic_samples = multinomial_rng(
        jax.random.key(1),
        probabilities,
        11,
        sample_shape=(8,),
    )
    logit_samples = multinomial_logit_rng(
        jax.random.key(2),
        logits,
        11,
        sample_shape=(8,),
    )

    assert jnp.array_equal(zero_samples, jnp.zeros((8, 2, 3), dtype=jnp.int32))
    assert jnp.all(deterministic_samples[:, 0] == jnp.array([11, 0, 0]))
    assert jnp.all(deterministic_samples[:, 1] == jnp.array([0, 0, 11]))
    assert jnp.array_equal(logit_samples, deterministic_samples)


def test_single_category_multinomial_rng_returns_every_trial() -> None:
    trials = jnp.array([0, 3, 7])

    probability_result = multinomial_rng(jax.random.key(0), jnp.array([1.0]), trials, sample_shape=(4,))
    logit_result = multinomial_logit_rng(jax.random.key(1), jnp.array([2.0]), trials, sample_shape=(4,))
    expected = jnp.broadcast_to(trials, (4, 3))[..., None]

    assert jnp.array_equal(probability_result, expected)
    assert jnp.array_equal(logit_result, expected)


def test_multinomial_rngs_preserve_the_largest_exact_float32_trial_count() -> None:
    trials = 16_777_216

    probability_result = multinomial_rng(jax.random.key(0), jnp.array([1.0, 0.0]), trials)
    logit_result = multinomial_logit_rng(jax.random.key(1), jnp.array([0.0, -jnp.inf]), trials)
    expected = jnp.array([trials, 0], dtype=jnp.int32)

    assert jnp.array_equal(probability_result, expected)
    assert jnp.array_equal(logit_result, expected)


def test_multinomial_logit_rng_preserves_rare_categories_in_either_order() -> None:
    key = jax.random.key(12)
    logits = jnp.array([17.0, 0.0])

    result = multinomial_logit_rng(key, logits, 1_000_000, sample_shape=(1_000,))
    reversed_result = multinomial_logit_rng(key, logits[::-1], 1_000_000, sample_shape=(1_000,))

    assert jnp.array_equal(result, reversed_result[..., ::-1])
    assert jnp.sum(result[..., 1]) > 0


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (multinomial_rng, jnp.array([0.2, 0.3, 0.5])),
        (multinomial_logit_rng, jnp.array([-1.0, 0.5, 2.0])),
    ],
)
def test_multinomial_rngs_support_jit_and_vmap_over_keys(function, parameters) -> None:
    keys = jax.random.split(jax.random.key(3), 3)
    compiled = jax.jit(function, static_argnames="sample_shape")(
        keys[0],
        parameters,
        10,
        sample_shape=(5,),
    )
    mapped = jax.vmap(lambda key: function(key, parameters, 10, sample_shape=(5,)))(keys)

    assert jnp.array_equal(compiled, function(keys[0], parameters, 10, sample_shape=(5,)))
    assert mapped.shape == (3, 5, 3)
    assert jnp.array_equal(mapped[2], function(keys[2], parameters, 10, sample_shape=(5,)))


@pytest.mark.parametrize(
    ("function", "parameters", "message"),
    [
        (multinomial_rng, 0.5, "probabilities must include a final Multinomial event axis"),
        (multinomial_rng, jnp.empty((0,)), "Multinomial event size must be positive"),
        (multinomial_logit_rng, 0.5, "logits must include a final Multinomial event axis"),
        (multinomial_logit_rng, jnp.empty((0,)), "Multinomial event size must be positive"),
    ],
)
def test_multinomial_rngs_raise_targeted_event_shape_errors(function, parameters, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function(jax.random.key(0), parameters, 5)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_multinomial_rngs_compute_with_float32_for_low_precision_inputs(dtype) -> None:
    key = jax.random.key(5)
    probabilities = jnp.array([0.25, 0.25, 0.5], dtype=dtype)
    logits = jnp.array([-1.0, 0.0, 1.0], dtype=dtype)

    probability_result = multinomial_rng(key, probabilities, 10, sample_shape=(32,))
    logit_result = multinomial_logit_rng(key, logits, 10, sample_shape=(32,))

    assert probability_result.dtype == jnp.dtype(jnp.int32)
    assert logit_result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(
        probability_result,
        multinomial_rng(key, probabilities.astype(jnp.float32), 10, sample_shape=(32,)),
    )
    assert jnp.array_equal(
        logit_result,
        multinomial_logit_rng(key, logits.astype(jnp.float32), 10, sample_shape=(32,)),
    )


@pytest.mark.parametrize(
    ("function", "parameters", "trials", "argument"),
    [
        (multinomial_rng, jnp.array([0.2, 0.3, 0.5]), 5 + 0j, "trials"),
        (multinomial_rng, jnp.array([0.2 + 0j, 0.3, 0.5]), 5, "probabilities"),
        (multinomial_logit_rng, jnp.array([0.0 + 0j, 1.0, 2.0]), 5, "logits"),
    ],
)
def test_multinomial_rngs_reject_complex_arguments(function, parameters, trials, argument: str) -> None:
    with pytest.raises(TypeError, match=rf"distribution argument '{argument}' must have a real numeric dtype"):
        function(jax.random.key(0), parameters, trials)


def _multinomial_logit_reference(value: np.ndarray, logits: np.ndarray) -> np.float64:
    counts = np.asarray(value, dtype=np.float64)
    log_probabilities = special.log_softmax(np.asarray(logits, dtype=np.float64))
    total = np.sum(counts)
    return special.gammaln(total + 1) - np.sum(special.gammaln(counts + 1)) + np.dot(counts, log_probabilities)


def _sequential_multinomial_logpmf(value: np.ndarray, probabilities: np.ndarray) -> np.float64:
    counts = np.asarray(value, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    remaining_count = int(np.sum(counts, dtype=np.int64))
    remaining_probability = np.sum(probability)
    log_mass = np.float64(0)

    for count, current_probability in zip(counts[:-1], probability[:-1], strict=True):
        conditional_probability = current_probability / remaining_probability if remaining_probability > 0 else 0.5
        log_mass += stats.binom.logpmf(int(count), remaining_count, conditional_probability)
        remaining_count -= int(count)
        remaining_probability -= current_probability

    return log_mass + np.sum(counts, dtype=np.float64) * np.log(np.sum(probability))


def _count_compositions(total: int, categories: int) -> list[tuple[int, ...]]:
    return [counts for counts in product(range(total + 1), repeat=categories) if sum(counts) == total]
