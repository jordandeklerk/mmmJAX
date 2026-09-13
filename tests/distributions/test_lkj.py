"""Tests for LKJ correlation Cholesky distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import betaln

from mmmjax import CorrelationCholesky, lkj_cholesky, lkj_cholesky_logpdf, lkj_cholesky_rng


@pytest.mark.parametrize("concentration", [0.5, 1.0, 2.0, 5.0])
def test_two_dimensional_lkj_has_normalized_beta_density(concentration):
    correlations = np.array([-0.8, -0.3, 0.0, 0.4, 0.9])
    factors = np.zeros((5, 2, 2))
    factors[:, 0, 0] = 1
    factors[:, 1, 0] = correlations
    factors[:, 1, 1] = np.sqrt(1 - correlations**2)

    expected = (concentration - 1) * np.log1p(-(correlations**2)) - betaln(0.5, concentration)
    actual = jax.jit(lkj_cholesky_logpdf)(factors, concentration)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(lkj_cholesky(factors, concentration), expected.sum(), atol=1e-5)


def test_lkj_density_is_in_factor_space_not_correlation_matrix_space():
    factor = jnp.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    # Uniform correlation matrices still induce a nonconstant factor density.
    difference = lkj_cholesky_logpdf(factor, 1.0) - lkj_cholesky_logpdf(jnp.eye(3), 1.0)
    np.testing.assert_allclose(difference, np.log(0.8), atol=2e-6)


def test_lkj_broadcasts_concentrations_and_factors():
    factors = lkj_cholesky_rng(jax.random.key(5), 3, 2.0, sample_shape=(4, 1))
    concentration = jnp.array([0.5, 1.0, 3.0])
    result = lkj_cholesky_logpdf(factors, concentration)
    assert result.shape == (4, 3)
    expected = jax.vmap(lambda factor: lkj_cholesky_logpdf(factor[0], concentration))(factors)
    np.testing.assert_allclose(result, expected, atol=2e-6)


def test_lkj_rng_axes_support_and_marginal_moments():
    concentration = jnp.array([1.0, 4.0])
    factors = jax.jit(lambda key: lkj_cholesky_rng(key, 3, concentration, sample_shape=(12_000,)))(jax.random.key(8))
    assert factors.shape == (12_000, 2, 3, 3)
    np.testing.assert_allclose(jnp.sum(factors**2, axis=-1), 1, atol=3e-6)
    assert jnp.all(jnp.triu(factors, 1) == 0)
    assert jnp.all(jnp.diagonal(factors, axis1=-2, axis2=-1) > 0)

    correlations = factors @ jnp.swapaxes(factors, -1, -2)
    pair = correlations[..., 0, 2]
    # Each off-diagonal correlation has variance 1 / (2 * eta + K - 1).
    np.testing.assert_allclose(pair.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose(pair.var(axis=0), 1 / (2 * concentration + 2), atol=0.012)


def test_lkj_rng_multiaxis_sample_shape_and_reproducibility():
    key = jax.random.key(4)
    first = lkj_cholesky_rng(key, 2, jnp.ones((2, 1)), sample_shape=(3, 4))
    assert first.shape == (3, 4, 2, 1, 2, 2)
    np.testing.assert_array_equal(first, lkj_cholesky_rng(key, 2, jnp.ones((2, 1)), sample_shape=(3, 4)))
    assert not jnp.array_equal(first, lkj_cholesky_rng(jax.random.key(5), 2, jnp.ones((2, 1)), sample_shape=(3, 4)))


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_lkj_position_and_concentration_gradients(dimension):
    declaration = CorrelationCholesky((dimension, dimension))
    position = jnp.linspace(-0.4, 0.5, declaration.position_shape[-1])

    def density(position, concentration):
        return lkj_cholesky(declaration.constrain(position), concentration) + declaration.log_density_adjustment(
            position
        )

    value, gradients = jax.jit(jax.value_and_grad(density, argnums=(0, 1)))(position, 2.0)
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
    step = 0.01
    expected = (density(position, 2.0 + step) - density(position, 2.0 - step)) / (2 * step)
    np.testing.assert_allclose(gradients[1], expected, atol=3e-4, rtol=2e-3)


@pytest.mark.parametrize("concentration", [0.0, -1.0, np.inf, np.nan])
def test_lkj_invalid_concentration_is_isolated_in_batches(concentration):
    parameters = jnp.array([2.0, concentration])
    result = lkj_cholesky_logpdf(jnp.eye(2), parameters)
    assert jnp.isfinite(result[0])
    assert jnp.isnan(result[1])
    samples = lkj_cholesky_rng(jax.random.key(1), 2, parameters, sample_shape=(3,))
    assert jnp.all(jnp.isfinite(samples[:, 0]))
    assert jnp.all(jnp.isnan(samples[:, 1]))


@pytest.mark.parametrize(
    "factor",
    [
        [[1.0, 0.1], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.5, 1.0]],
        [[1.0, 0.0], [np.inf, 1.0]],
    ],
)
def test_lkj_rejects_factors_outside_support(factor):
    values = jnp.stack((jnp.eye(2), jnp.asarray(factor)))
    result = lkj_cholesky_logpdf(values, 2.0)
    assert jnp.isfinite(result[0])
    assert jnp.isneginf(result[1])


def test_lkj_nan_and_single_dimension():
    assert jnp.isnan(lkj_cholesky_logpdf(jnp.full((2, 2), jnp.nan), 1.0))
    assert lkj_cholesky_logpdf(jnp.ones((1, 1)), 2.0) == 0
    np.testing.assert_array_equal(lkj_cholesky_rng(jax.random.key(0), 1, 2.0), [[1.0]])


@pytest.mark.parametrize("shape", [(), (2,), (2, 3), (0, 0)])
def test_lkj_requires_square_factors(shape):
    with pytest.raises(ValueError, match="nonempty square"):
        lkj_cholesky_logpdf(jnp.zeros(shape), 1.0)


@pytest.mark.parametrize("dimension,error", [(0, ValueError), (-1, ValueError), (2.0, TypeError), (True, TypeError)])
def test_lkj_rng_requires_static_positive_dimension(dimension, error):
    with pytest.raises(error, match="dimension"):
        lkj_cholesky_rng(jax.random.key(0), dimension, 2.0)


@pytest.mark.parametrize("sample_shape,error", [(3, TypeError), ((-1,), ValueError), ((True,), TypeError)])
def test_lkj_rng_validates_sample_shape(sample_shape, error):
    with pytest.raises(error, match="sample_shape"):
        lkj_cholesky_rng(jax.random.key(0), 2, 1.0, sample_shape=sample_shape)
