"""Tests for incomplete Beta shape derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betainc
from scipy import integrate, special, stats

from mmmjax.distributions._beta_cdf import _betainc


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_betainc_preserves_jax_values_and_matches_scipy(dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    alpha = jnp.array([0.2, 1.0, 2.5, 20.0], dtype=dtype)
    beta = jnp.array([0.5, 2.0, 4.7, 30.0], dtype=dtype)
    value = jnp.array([[0.01], [0.4], [0.9]], dtype=dtype)
    expected = special.betainc(*(np.asarray(arg, dtype=np.float64) for arg in (alpha, beta, value)))

    result = jax.jit(_betainc)(alpha, beta, value)

    np.testing.assert_array_equal(result, jax.jit(betainc)(alpha, beta, value))
    tolerance = 3e-5 if dtype == jnp.float32 else 2e-12
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_betainc_gradients_match_independent_integrals(dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    # Integer shapes exercise derivatives beyond the point where the value fraction terminates
    points = jnp.array(
        [
            [1.0, 1.0, 0.4],
            [2.0, 1.0, 0.4],
            [0.5, 2.0, 0.3],
            [2.0, 3.0, 0.4],
            [0.2, 0.3, 0.01],
            [0.2, 0.3, 0.9],
            [20.0, 30.0, 0.4],
            [11.0, 3.8, 0.32],
            [2.3, 4.7, 0.001],
            [4.7, 2.3, 0.999],
        ],
        dtype=dtype,
    )
    expected = np.array([_integrated_beta_gradient(*point) for point in np.asarray(points, dtype=np.float64)])

    result = jax.jit(jax.vmap(jax.grad(_betainc, argnums=(0, 1, 2))))(*points.T)

    tolerance = 3e-5 if dtype == jnp.float32 else 2e-11
    np.testing.assert_allclose(np.stack(result, axis=-1), expected, rtol=tolerance, atol=0)


def test_betainc_broadcast_jvp_and_reverse_mode_match_integrals() -> None:
    alpha = jnp.array([0.5, 2.0, 4.0], dtype=jnp.float32)
    beta = jnp.asarray(1.0, dtype=jnp.float32)
    value = jnp.array([[0.2], [0.8]], dtype=jnp.float32)
    tangents = (
        jnp.array([0.3, -0.2, 0.1], dtype=alpha.dtype),
        jnp.asarray(-0.4, dtype=beta.dtype),
        jnp.array([[0.1], [-0.2]], dtype=value.dtype),
    )
    broadcast = np.broadcast_arrays(*(np.asarray(arg, dtype=np.float64) for arg in (alpha, beta, value)))
    derivatives = np.array(
        [_integrated_beta_gradient(*point) for point in zip(*(arg.flat for arg in broadcast), strict=True)]
    )
    derivatives = derivatives.reshape(2, 3, 3)
    expected_tangent = sum(derivatives[..., index] * np.asarray(tangent) for index, tangent in enumerate(tangents))

    result, tangent = jax.jit(lambda *args: jax.jvp(_betainc, args, tangents))(alpha, beta, value)
    gradient = jax.jit(jax.grad(lambda *args: jnp.sum(_betainc(*args)), argnums=(0, 1, 2)))(alpha, beta, value)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(gradient[0], derivatives[..., 0].sum(axis=0), rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(gradient[1], derivatives[..., 1].sum(), rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(gradient[2], derivatives[..., 2].sum(axis=1, keepdims=True), rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("point", [(1.0, 1.0, 0.4), (2.0, 3.0, 0.4), (0.2, 0.3, 0.9)])
def test_betainc_second_derivatives_match_independent_gradient_differences(point, dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    arguments = jnp.asarray(point, dtype=dtype)
    parameters = np.asarray(arguments, dtype=np.float64)
    expected = np.empty((3, 3))
    for index, argument in enumerate(parameters):
        offset = np.zeros(3)
        offset[index] = argument * 1e-4
        # Differences of independently integrated scores do not reuse our continued fraction
        gradients = [_integrated_beta_gradient(*(parameters + multiplier * offset)) for multiplier in (-2, -1, 1, 2)]
        expected[:, index] = (gradients[0] - 8 * gradients[1] + 8 * gradients[2] - gradients[3]) / (12 * offset[index])

    result = jax.jit(jax.hessian(lambda parameters: _betainc(*parameters)))(arguments)

    tolerance = 1e-4 if dtype == jnp.float32 else 2e-8
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=1e-9)
    np.testing.assert_allclose(result, result.T, rtol=tolerance, atol=1e-12)


@pytest.mark.parametrize(
    "threshold, mean, concentration", [(0, 2.0, 1.0), (1, 4.0, 0.5), (8, 6.0, 2.0), (30, 25.0, 20.0)]
)
def test_betainc_negative_binomial_gradients_match_finite_mass_sums(threshold, mean, concentration) -> None:
    counts = np.arange(threshold + 1)
    masses = stats.nbinom.pmf(counts, concentration, concentration / (concentration + mean))
    mean_scores = counts / mean - (counts + concentration) / (mean + concentration)
    concentration_scores = (
        special.digamma(counts + concentration)
        - special.digamma(concentration)
        + np.log(concentration)
        + 1
        - np.log(mean + concentration)
        - (counts + concentration) / (mean + concentration)
    )
    expected_gradient = np.array([masses @ mean_scores, masses @ concentration_scores])

    def cdf(mean, concentration):
        return _betainc(
            concentration, jnp.asarray(threshold + 1, dtype=mean.dtype), concentration / (concentration + mean)
        )

    probability, gradient = jax.jit(jax.value_and_grad(cdf, argnums=(0, 1)))(
        jnp.float32(mean), jnp.float32(concentration)
    )

    np.testing.assert_allclose(probability, masses.sum(), rtol=3e-5, atol=0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=8e-5, atol=2e-7)


def _integrated_beta_gradient(alpha: float, beta: float, value: float) -> np.ndarray:
    # A nearly complete integral cancels positive and negative scores, so integrate its complement instead
    if value > 0.5:
        reflected = _integrated_beta_gradient(beta, alpha, 1 - value)
        return np.array([-reflected[1], -reflected[0], reflected[2]])

    # Integrating density scores checks shape derivatives without an incomplete Beta algorithm
    # Substituting t = value * u lets QUADPACK handle the integrable singularity at zero
    def weighted_integral(function, weight="alg"):
        return integrate.quad(function, 0, 1, weight=weight, wvar=(alpha - 1, 0), epsabs=1e-12, epsrel=1e-12)[0]

    mass = weighted_integral(lambda u: (1 - value * u) ** (beta - 1))
    log_value_integral = weighted_integral(lambda u: (1 - value * u) ** (beta - 1), weight="alg-loga")
    log_complement_integral = weighted_integral(lambda u: (1 - value * u) ** (beta - 1) * np.log1p(-value * u))
    factor = np.exp(alpha * np.log(value) - special.betaln(alpha, beta))
    alpha_derivative = factor * (
        log_value_integral + (np.log(value) - special.digamma(alpha) + special.digamma(alpha + beta)) * mass
    )
    beta_derivative = factor * (
        log_complement_integral + (special.digamma(alpha + beta) - special.digamma(beta)) * mass
    )
    return np.array([alpha_derivative, beta_derivative, stats.beta.pdf(value, alpha, beta)])
