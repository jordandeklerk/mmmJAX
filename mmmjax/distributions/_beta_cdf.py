"""Incomplete Beta derivatives for distribution tails."""

from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, digamma

from mmmjax.distributions._beta import _beta_logpdf


@jax.custom_jvp
def _betainc(alpha: jax.Array, beta: jax.Array, value: jax.Array) -> jax.Array:
    """Evaluate regularized incomplete Beta with differentiable shape parameters.

    Callers must supply finite positive shapes and values strictly between
    zero and one, handling support boundaries in the distribution function.
    """
    return betainc(alpha, beta, value)


@_betainc.defjvp
def _betainc_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    probability = _betainc(*primals)
    alpha, beta, value = jnp.broadcast_arrays(*primals)
    alpha_tangent, beta_tangent, value_tangent = tangents

    reflected = value > (alpha + 1) / (alpha + beta + 2)
    first_shape = jnp.where(reflected, beta, alpha)
    second_shape = jnp.where(reflected, alpha, beta)
    argument = jnp.where(reflected, 1 - value, value)
    fraction, first_derivative, second_derivative = _beta_fraction(first_shape, second_shape, argument)

    # Reuse the density's normalization and derivative rules instead of differentiating betaln approximations
    # Reflecting the value and swapping shapes leaves the Beta density unchanged
    log_density = _beta_logpdf(value, alpha, beta)
    log_argument = jnp.log(argument)
    log_complement = jnp.log1p(-argument)
    prefactor = jnp.exp(log_density + log_argument + log_complement - jnp.log(first_shape))
    sum_derivative = digamma(first_shape + second_shape)
    first_derivative = prefactor * (
        first_derivative + fraction * (log_argument - 1 / first_shape + sum_derivative - digamma(first_shape))
    )
    second_derivative = prefactor * (
        second_derivative + fraction * (log_complement + sum_derivative - digamma(second_shape))
    )
    alpha_derivative = jnp.where(reflected, -second_derivative, first_derivative)
    beta_derivative = jnp.where(reflected, -first_derivative, second_derivative)
    value_derivative = jnp.exp(log_density)

    tangent = alpha_derivative * alpha_tangent + beta_derivative * beta_tangent + value_derivative * value_tangent
    return probability, tangent


def _beta_fraction(
    alpha: jax.Array,
    beta: jax.Array,
    value: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Modified Lentz evaluation of the continued fraction in DLMF 8.17.22-23
    # The square-root floor also leaves room for reciprocals in the differentiated recurrence
    minimum = jnp.sqrt(jnp.finfo(value.dtype).tiny)
    tolerance = jnp.finfo(value.dtype).eps
    ones = jnp.ones_like(value)
    zeros = jnp.zeros_like(value)

    def protect(denominator: jax.Array) -> jax.Array:
        return jnp.where(jnp.abs(denominator) < minimum, minimum, denominator)

    def initialize(alpha: jax.Array, beta: jax.Array) -> jax.Array:
        denominator = 1 / protect(1 - (alpha + beta) * value / (alpha + 1))
        return jnp.stack((ones, denominator, denominator))

    initial, alpha_derivative = jax.jvp(initialize, (alpha, beta), (ones, zeros))
    _, beta_derivative = jax.jvp(initialize, (alpha, beta), (zeros, ones))
    initial_state = jnp.stack((initial, alpha_derivative, beta_derivative))

    def step(index: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        order = jnp.asarray(index, dtype=value.dtype)

        def update(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            previous, active = state

            def advance(alpha: jax.Array, beta: jax.Array, terms: jax.Array) -> jax.Array:
                numerator, denominator, fraction = terms
                even = order * (beta - order) * value / ((alpha + 2 * order - 1) * (alpha + 2 * order))
                numerator = protect(1 + even / numerator)
                denominator = 1 / protect(1 + even * denominator)
                fraction = fraction * numerator * denominator

                odd = (
                    -(alpha + order) * (alpha + beta + order) * value / ((alpha + 2 * order) * (alpha + 2 * order + 1))
                )
                numerator = protect(1 + odd / numerator)
                denominator = 1 / protect(1 + odd * denominator)
                return jnp.stack((numerator, denominator, fraction * numerator * denominator))

            # Forward-mode propagates elementwise partials without constructing a batch-sized Jacobian
            updated, alpha_derivative = jax.jvp(advance, (alpha, beta, previous[0]), (ones, zeros, previous[1]))
            _, beta_derivative = jax.jvp(advance, (alpha, beta, previous[0]), (zeros, ones, previous[2]))
            current = jnp.stack((updated, alpha_derivative, beta_derivative))

            # Integer shapes can terminate the value fraction before its shape derivatives converge
            change = jnp.abs(current[:, 2] - previous[:, 2])
            converged = change[0] <= tolerance * jnp.abs(current[0, 2])
            converged &= jnp.all(change[1:] <= tolerance * jnp.maximum(1.0, jnp.abs(current[1:, 2])), axis=0)
            return jnp.where(active, current, previous), active & ~converged

        return cast(
            tuple[jax.Array, jax.Array],
            jax.lax.cond(jnp.any(state[1]), update, lambda state: state, state),
        )

    # These caps match 200/600 alternating terms in JAX's continued fraction
    # Static loop bounds keep reverse-mode differentiation available for higher derivatives
    iterations = 300 if value.dtype == jnp.float64 else 100
    result, unfinished = jax.lax.fori_loop(1, iterations + 1, step, (initial_state, jnp.ones_like(value, dtype=bool)))
    result = jnp.where(unfinished, jnp.nan, result)
    return result[0, 2], result[1, 2], result[2, 2]
