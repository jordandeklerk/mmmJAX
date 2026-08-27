"""Tests for Beta distribution functions."""

import jax
import jax.numpy as jnp
import pytest

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
