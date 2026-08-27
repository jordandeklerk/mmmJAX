"""Tests for Beta distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from mmmjax import (
    beta,
    beta_logpdf,
    beta_rng,
)


def test_beta_logpdf_matches_known_values() -> None:
    values = jnp.array([0.1, 0.4, 0.8], dtype=jnp.float32)
    expected = jnp.array(
        [0.30546208190868773, 0.6074238513643362, -2.556350281979741],
        dtype=jnp.float32,
    )

    result = beta_logpdf(values, 2.3, 4.7)

    assert jnp.allclose(result, expected)


def test_beta_logpdf_matches_scipy_reference_grid() -> None:
    values = np.array([1e-20, 0.01, 0.1, 0.4, 0.5, 0.8, 0.99, 1 - 1e-6], dtype=np.float32)
    alphas = np.array([0.1, 0.5, 1.0, 2.3, 7.999, 8.0, 50.0, 3.0], dtype=np.float32)
    betas = np.array([3.0, 0.2, 1.0, 4.7, 8.001, 25.0, 9.0, 0.1], dtype=np.float32)
    expected = stats.beta.logpdf(
        values.astype(np.float64),
        alphas.astype(np.float64),
        betas.astype(np.float64),
    )

    result = beta_logpdf(values, alphas, betas)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_beta_logpdf_gradients_match_analytic_reference_grid() -> None:
    values = np.array([0.01, 0.2, 0.5, 0.7, 0.99], dtype=np.float32)
    alphas = np.array([0.2, 1.0, 2.3, 8.0, 50.0], dtype=np.float32)
    betas = np.array([4.0, 0.5, 4.7, 25.0, 9.0], dtype=np.float32)
    shape_sums = alphas + betas
    expected = np.stack(
        [
            (alphas - 1) / values - (betas - 1) / (1 - values),
            np.log(values) - special.digamma(alphas) + special.digamma(shape_sums),
            np.log1p(-values) - special.digamma(betas) + special.digamma(shape_sums),
        ],
        axis=-1,
    )

    gradients = jax.vmap(jax.grad(beta_logpdf, argnums=(0, 1, 2)))(values, alphas, betas)
    result = np.stack(gradients, axis=-1)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize(
    ("value", "alpha", "beta_parameter", "expected"),
    [
        (0.9990000128746033, 2.5, 1.0, 1.5015014819800854),
        (0.0010000000474974513, 1.0, 0.5, 0.5005005005480932),
    ],
)
def test_beta_logpdf_value_gradient_uses_ordinary_shape_formula(
    value: float,
    alpha: float,
    beta_parameter: float,
    expected: float,
) -> None:
    result = jax.grad(beta_logpdf)(
        jnp.float32(value),
        jnp.float32(alpha),
        jnp.float32(beta_parameter),
    )

    assert jnp.allclose(result, expected, rtol=3e-6, atol=0)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_beta_logpdf_matches_scipy_across_float64_betaln_cutoff() -> None:
    cutoff = np.float64(8.0)
    beta_parameters = np.array(
        [
            np.nextafter(cutoff, -np.inf),
            cutoff,
            np.nextafter(cutoff, np.inf),
        ]
    )
    expected = stats.beta.logpdf(0.37, 2.5, beta_parameters)

    result = beta_logpdf(
        jnp.float64(0.37),
        jnp.float64(2.5),
        jnp.asarray(beta_parameters),
    )

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=5e-13)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_beta_logpdf_matches_scipy_for_ordinary_float64_tail() -> None:
    expected = stats.beta.logpdf(0.001, 8.0, 2.5)

    result = beta_logpdf(jnp.float64(0.001), jnp.float64(8.0), jnp.float64(2.5))

    assert jnp.allclose(result, expected, rtol=1e-12, atol=5e-13)


def test_beta_returns_scalar_sum() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = beta(values, 2.3, 4.7)

    assert result.shape == ()
    assert jnp.allclose(result, -1.6434643487067171)


def test_beta_logpdf_broadcasts_arguments() -> None:
    values = jnp.array([[0.2], [0.75]])
    alphas = jnp.array([0.5, 1.0, 3.0])
    betas = jnp.array([2.0, 1.0, 0.5])
    expected = jnp.array(
        [
            [0.29389333245105947, 0.0, -3.1718425703486668],
            [-1.530135397345781, 0.0, 0.05324451451881232],
        ]
    )

    result = beta_logpdf(values, alphas, betas)

    assert result.shape == (2, 3)
    assert jnp.allclose(result, expected, atol=1e-6)
    assert jnp.allclose(beta(values, alphas, betas), jnp.sum(expected))


def test_beta_logpdf_uses_boundary_limits() -> None:
    alphas = jnp.array([0.5, 1.0, 2.0])
    betas = jnp.array([0.5, 1.0, 2.0])
    maximum = jnp.asarray(jnp.finfo(jnp.float32).max)

    lower_result = beta_logpdf(jnp.array([[0.0], [-0.0]]), alphas, 4.7)
    upper_result = beta_logpdf(1.0, 2.3, betas)
    extreme_equality_result = beta_logpdf(jnp.array([0.0, 1.0]), jnp.array([1.0, maximum]), jnp.array([maximum, 1.0]))
    extreme_concentrated_result = beta_logpdf(jnp.array([0.0, 1.0]), maximum, maximum)

    assert lower_result.shape == (2, 3)
    assert jnp.all(jnp.isposinf(lower_result[:, 0]))
    assert jnp.allclose(lower_result[:, 1], jnp.log(4.7))
    assert jnp.all(jnp.isneginf(lower_result[:, 2]))
    assert jnp.isposinf(upper_result[0])
    assert jnp.allclose(upper_result[1], jnp.log(2.3))
    assert jnp.isneginf(upper_result[2])
    assert jnp.allclose(extreme_equality_result, jnp.log(maximum))
    assert jnp.all(jnp.isneginf(extreme_concentrated_result))


def test_beta_logpdf_enforces_support_and_propagates_nan() -> None:
    values = jnp.array([-jnp.inf, -0.1, 1.1, jnp.inf, jnp.nan])

    result = beta_logpdf(values, 2.3, 4.7)

    assert jnp.all(jnp.isneginf(result[:4]))
    assert jnp.isnan(result[4])


def test_beta_logpdf_rejects_invalid_parameters_before_support_check() -> None:
    invalid_parameters = jnp.array([0.0, -1.0, jnp.inf, jnp.nan])

    invalid_alphas = beta_logpdf(-1.0, invalid_parameters, 1.0)
    invalid_betas = beta_logpdf(-1.0, 1.0, invalid_parameters)

    assert jnp.all(jnp.isnan(invalid_alphas))
    assert jnp.all(jnp.isnan(invalid_betas))


def test_beta_logpdf_preserves_small_tail_terms() -> None:
    result = beta_logpdf(jnp.float32(1e-20), 1.0, jnp.float32(1e20))

    assert jnp.isfinite(result)
    assert jnp.allclose(result, 45.051701859880914)


def test_beta_logpdf_remains_accurate_for_concentrated_shapes() -> None:
    value = jnp.float32(0.5)
    alpha = jnp.float32(1e8)
    beta_parameter = jnp.float32(1e8)

    result = beta_logpdf(value, alpha, beta_parameter)
    gradients = jax.grad(beta, argnums=(0, 1, 2))(value, alpha, beta_parameter)

    assert jnp.allclose(result, 9.331122398376465, rtol=2e-7, atol=0)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([0.0, 2.5000002068509275e-9, 2.5000002068509275e-9]),
        rtol=2e-7,
        atol=0,
    )


def test_beta_logpdf_preserves_concentrated_off_mode_terms() -> None:
    value = jnp.float32(0.5001000165939331)
    alpha = jnp.float32(1e8)
    beta_parameter = jnp.float32(1e8)

    result = beta_logpdf(value, alpha, beta_parameter)
    gradients = jax.grad(beta_logpdf, argnums=(0, 1, 2))(value, alpha, beta_parameter)

    assert jnp.allclose(result, 5.329794883728027, rtol=3e-7, atol=0)
    assert jnp.allclose(
        jnp.asarray(gradients),
        jnp.array([-80013.28125, 0.00020001568167936057, -0.00020005069382023066]),
        rtol=3e-7,
        atol=0,
    )


def test_beta_logpdf_handles_maximum_finite_concentrated_shapes() -> None:
    shape = jnp.asarray(jnp.finfo(jnp.float32).max)

    result = beta_logpdf(jnp.float32(0.5), shape, shape)
    gradients = jax.grad(beta_logpdf, argnums=(0, 1, 2))(jnp.float32(0.5), shape, shape)

    assert jnp.isfinite(result)
    assert jnp.allclose(result, 44.482200622558594, rtol=2e-7, atol=0)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_beta_logpdf_remains_accurate_at_extreme_float64_shapes() -> None:
    shape = jnp.float64(1e20)

    result = beta_logpdf(jnp.float64(0.5), shape, shape)

    assert jnp.allclose(result, 23.1466331675757, rtol=1e-14, atol=0)


def test_beta_is_differentiable_with_respect_to_value() -> None:
    values = jnp.array([0.1, 0.4, 0.8])
    expected = jnp.array([8.888888888888886, -2.9166666666666674, -16.875000000000004])

    result = jax.grad(lambda current_values: beta(current_values, 2.3, 4.7))(values)

    assert jnp.allclose(result, expected)


def test_beta_is_differentiable_with_respect_to_alpha() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = jax.grad(lambda current_alpha: beta(values, current_alpha, 4.7))(2.3)

    assert jnp.allclose(result, 0.37621398802108175)


def test_beta_is_differentiable_with_respect_to_beta() -> None:
    values = jnp.array([0.1, 0.4, 0.8])

    result = jax.grad(lambda current_beta: beta(values, 2.3, current_beta))(4.7)

    assert jnp.allclose(result, -0.91954247545786227)


def test_beta_logpdf_supports_forward_mode_differentiation() -> None:
    primals = (jnp.float32(0.4), jnp.float32(2.3), jnp.float32(4.7))
    tangents = (jnp.float32(0.2), jnp.float32(-0.3), jnp.float32(0.4))

    _, result = jax.jvp(beta_logpdf, primals, tangents)

    assert jnp.allclose(result, -0.72045548951)


def test_beta_logpdf_supports_higher_order_differentiation() -> None:
    parameters = jnp.array([0.4, 2.3, 4.7], dtype=jnp.float32)
    expected = jnp.array(
        [
            [-18.40277778, 2.5, -1.66666667],
            [2.5, -0.38899228, 0.15354518],
            [-1.66666667, 0.15354518, -0.08344666],
        ]
    )

    result = jax.hessian(lambda arguments: beta_logpdf(*arguments))(parameters)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize(
    "arguments",
    [
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (jnp.nan, 1.0, 1.0),
    ],
)
def test_beta_logpdf_gradients_propagate_invalid_inputs(arguments) -> None:
    gradients = jax.grad(beta_logpdf, argnums=(0, 1, 2))(*arguments)

    assert jnp.all(jnp.isnan(jnp.asarray(gradients)))


def test_beta_can_be_vectorized_over_datasets() -> None:
    values = jnp.array([[0.1, 0.4], [0.5, 0.8]])
    alphas = jnp.array([1.5, 3.0])
    betas = jnp.array([0.5, 2.0])

    result = jax.vmap(beta)(values, alphas, betas)
    expected = jnp.stack(
        [beta(value, alpha, beta_value) for value, alpha, beta_value in zip(values, alphas, betas, strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_beta_rng_wraps_jax_sampler() -> None:
    key = jax.random.key(42)
    alphas = jnp.array([0.5, 2.5], dtype=jnp.float32)
    betas = jnp.array([1.7, 0.8], dtype=jnp.float32)
    expected = jax.random.beta(key, alphas, betas, shape=(3, 2), dtype=jnp.float32)

    result = beta_rng(key, alphas, betas, sample_shape=(3,))

    assert result.shape == (3, 2)
    assert jnp.array_equal(result, expected)
    assert jnp.all((result >= 0) & (result <= 1))


def test_beta_rng_uses_broadcast_parameter_shape() -> None:
    alphas = jnp.ones((2, 1))
    betas = jnp.ones(3)

    result = beta_rng(jax.random.key(0), alphas, betas, sample_shape=(4,))

    assert result.shape == (4, 2, 3)


def test_beta_rng_matches_distribution_moments() -> None:
    samples = beta_rng(jax.random.key(7), 2.0, 5.0, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples), 2 / 7, rtol=0, atol=0.007)
    assert jnp.allclose(jnp.var(samples), 10 / 392, rtol=0, atol=0.002)


def test_beta_rng_rejects_incompatible_parameter_shapes() -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter shapes cannot be broadcast together: \(\(2,\), \(3,\)\)",
    ):
        beta_rng(jax.random.key(0), jnp.ones(2), jnp.ones(3))
