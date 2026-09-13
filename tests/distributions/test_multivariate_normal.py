"""Tests for multivariate Normal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mmmjax import multivariate_normal, multivariate_normal_logpdf, multivariate_normal_rng, normal_logpdf


@pytest.fixture
def scale_tril():
    return jnp.array([[1.5, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.3, 0.2, 1.2]])


def test_multivariate_normal_matches_normalized_reference(scale_tril):
    values = np.array([[0.2, -0.4, 1.0], [1.0, 0.0, -0.5]])
    location = np.array([0.4, 0.1, -0.3])
    expected = stats.multivariate_normal.logpdf(values, mean=location, cov=np.asarray(scale_tril @ scale_tril.T))
    result = jax.jit(multivariate_normal_logpdf)(values, location, scale_tril)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(multivariate_normal(values, location, scale_tril), expected.sum(), rtol=3e-6)


def test_diagonal_covariance_matches_independent_normals():
    values = jnp.array([[0.2, 0.4], [1.0, -2.0]])
    scales = jnp.array([0.7, 2.0])
    result = multivariate_normal_logpdf(values, jnp.zeros(2), jnp.diag(scales))
    np.testing.assert_allclose(result, normal_logpdf(values, 0.0, scales).sum(axis=-1), rtol=3e-6)


def test_multivariate_normal_broadcasts_batches_not_event_axes(scale_tril):
    values = jnp.arange(12.0).reshape(4, 1, 3) / 4
    means = jnp.arange(6.0).reshape(2, 3) / 3
    factors = jnp.stack((scale_tril, 2 * scale_tril))
    actual = multivariate_normal_logpdf(values, means, factors)
    expected = jnp.stack([multivariate_normal_logpdf(values[:, 0], means[i], factors[i]) for i in range(2)], axis=-1)
    assert actual.shape == (4, 2)
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


def test_multivariate_normal_gradients_match_analytic_scores(scale_tril):
    value = jnp.array([0.3, -0.7, 0.2])
    location = jnp.array([0.0, 0.2, 0.8])
    gradients = jax.jit(jax.grad(multivariate_normal, argnums=(0, 1, 2)))(value, location, scale_tril)
    whitened = jnp.linalg.solve(scale_tril, value - location)
    score = jnp.linalg.solve(scale_tril.T, whitened)
    scale_score = jnp.linalg.solve(scale_tril.T, jnp.outer(whitened, whitened) - jnp.eye(3))
    np.testing.assert_allclose(gradients[0], -score, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gradients[1], score, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gradients[2], jnp.tril(scale_score), rtol=3e-6, atol=3e-6)


def test_multivariate_normal_rng_recovers_mean_and_covariance(scale_tril):
    mean = jnp.array([0.3, -0.8, 2.0])
    samples = multivariate_normal_rng(jax.random.key(4), mean, scale_tril, sample_shape=(20_000,))
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.04)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), scale_tril @ scale_tril.T, atol=0.06)


def test_multivariate_normal_rng_sample_batch_event_axes_and_keys(scale_tril):
    location = jnp.ones((4, 1, 3))
    factors = jnp.stack((scale_tril, 2 * scale_tril))
    draw = jax.jit(lambda key: multivariate_normal_rng(key, location, factors, sample_shape=(2, 3)))
    first = draw(jax.random.key(1))
    assert first.shape == (2, 3, 4, 2, 3)
    np.testing.assert_array_equal(first, draw(jax.random.key(1)))
    assert not jnp.array_equal(first, draw(jax.random.key(2)))


@pytest.mark.parametrize(
    "factor",
    [
        [[1.0, 0.1], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [np.inf, 1.0]],
        [[1.0, 0.0], [np.nan, 1.0]],
    ],
)
def test_multivariate_normal_invalid_covariance_factors_are_isolated(factor):
    factors = jnp.stack((jnp.eye(2), jnp.asarray(factor)))
    result = multivariate_normal_logpdf(jnp.zeros(2), jnp.zeros(2), factors)
    assert jnp.isfinite(result[0])
    assert jnp.isnan(result[1])
    samples = multivariate_normal_rng(jax.random.key(0), jnp.zeros(2), factors, sample_shape=(3,))
    assert jnp.all(jnp.isfinite(samples[:, 0]))
    assert jnp.all(jnp.isnan(samples[:, 1]))


def test_multivariate_normal_handles_nonfinite_observations_and_means():
    values = jnp.array([[jnp.inf, 0.0], [-jnp.inf, jnp.inf], [jnp.nan, 0.0]])
    result = multivariate_normal_logpdf(values, jnp.zeros(2), jnp.eye(2))
    assert jnp.all(jnp.isneginf(result[:2]))
    assert jnp.isnan(result[2])
    assert jnp.isnan(multivariate_normal_logpdf(jnp.zeros(2), jnp.array([0.0, jnp.inf]), jnp.eye(2)))


@pytest.mark.parametrize(
    "value,location,factor",
    [
        (0.0, [0.0], [[1.0]]),
        ([0.0], 0.0, [[1.0]]),
        ([0.0], [0.0], [1.0]),
        ([0.0], [0.0], [[1.0, 0.0]]),
        ([0.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([0.0], [0.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([], [], np.zeros((0, 0))),
    ],
)
def test_multivariate_normal_rejects_mismatched_events(value, location, factor):
    with pytest.raises(ValueError):
        multivariate_normal_logpdf(value, location, factor)


@pytest.mark.parametrize("sample_shape,error", [(3, TypeError), ((-1,), ValueError), ((True,), TypeError)])
def test_multivariate_normal_rng_validates_sample_shape(sample_shape, error):
    with pytest.raises(error, match="sample_shape"):
        multivariate_normal_rng(jax.random.key(0), jnp.zeros(2), jnp.eye(2), sample_shape=sample_shape)
