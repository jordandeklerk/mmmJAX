"""Incomplete Beta derivatives for distribution tails."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, digamma

from mmmjax.distributions._beta import _beta_logpdf


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _log_betainc(alpha: jax.Array, beta: jax.Array, value: jax.Array, log_value: bool = False) -> jax.Array:
    r"""Evaluate log regularized incomplete Beta without reflection.

    Callers must provide finite positive shapes and a value satisfying
    :math:`0 < x \leq (\alpha + 1) / (\alpha + \beta + 2)`.
    With ``log_value=True``, the last array holds :math:`\log x` instead.
    """
    log_argument = value if log_value else jnp.log(value)
    value = jnp.exp(value) if log_value else value
    alpha, beta, value = jnp.broadcast_arrays(alpha, beta, value)
    fraction = _beta_fraction(alpha, beta, value, with_derivatives=False)[0]
    # Native betainc can lose normalization accuracy at large shapes or underflow before taking a log
    return _beta_logpdf(value, alpha, beta) + log_argument + jnp.log1p(-value) - jnp.log(alpha) + jnp.log(fraction)


@_log_betainc.defjvp
def _log_betainc_jvp(
    log_value: bool,
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    alpha, beta, value = primals
    alpha_tangent, beta_tangent, value_tangent = tangents
    log_argument = value if log_value else jnp.log(value)
    value = jnp.exp(value) if log_value else value
    fraction, alpha_derivative, beta_derivative = _beta_fraction(*jnp.broadcast_arrays(alpha, beta, value))
    log_density, shape_tangent = jax.jvp(
        _beta_logpdf,
        (value, alpha, beta),
        (jnp.zeros_like(value), alpha_tangent, beta_tangent),
    )
    log_probability = log_density + log_argument + jnp.log1p(-value) - jnp.log(alpha) + jnp.log(fraction)

    # Reuse the fraction's partials instead of reverse-differentiating its iteration history
    # The value derivative is the Beta density divided by its cumulative probability
    # Cancel dx/dlog(x) analytically so a tiny argument does not overflow an intermediate derivative
    value_derivative = (alpha / fraction) / (1 - value)
    if not log_value:
        value_derivative = value_derivative / value
    tangent = (
        shape_tangent
        + (alpha_derivative / fraction - 1 / alpha) * alpha_tangent
        + (beta_derivative / fraction) * beta_tangent
        + value_derivative * value_tangent
    )
    return log_probability, tangent


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
    *,
    with_derivatives: bool = True,
) -> tuple[jax.Array, ...]:
    """Evaluate the fraction and optionally its two shape derivatives."""
    # Evaluate the incomplete Beta continued fraction from DLMF 8.17.22-23 using modified Lentz
    # The square-root floor also leaves room for reciprocals in the differentiated recurrence
    minimum = jnp.sqrt(jnp.finfo(value.dtype).tiny)
    tolerance = jnp.finfo(value.dtype).eps
    ones = jnp.ones_like(value)
    zeros = jnp.zeros_like(value)

    def protect(denominator: jax.Array) -> jax.Array:
        return jnp.where(jnp.abs(denominator) < minimum, minimum, denominator)

    def initialize(alpha: jax.Array, beta: jax.Array) -> jax.Array:
        # This rearrangement avoids subtracting nearly equal shape ratios when beta is close to one
        denominator = 1 / protect((1 - value) - (beta - 1) * value / (alpha + 1))
        return jnp.stack((ones, denominator, denominator))

    if with_derivatives:
        initial, alpha_derivative = jax.jvp(initialize, (alpha, beta), (ones, zeros))
        _, beta_derivative = jax.jvp(initialize, (alpha, beta), (zeros, ones))
        initial_state = jnp.stack((initial, alpha_derivative, beta_derivative))
    else:
        initial_state = initialize(alpha, beta)[None, ...]

    def step(index: int | jax.Array, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        order = jnp.asarray(index, dtype=value.dtype)

        def update(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            previous, active = state

            def advance(alpha: jax.Array, beta: jax.Array, terms: jax.Array) -> jax.Array:
                numerator, denominator, fraction = terms
                even = (order / (alpha + 2 * order)) * ((beta - order) / (alpha + 2 * order - 1)) * value
                numerator = protect(1 + even / numerator)
                denominator = 1 / protect(1 + even * denominator)
                fraction = fraction * numerator * denominator

                odd = (
                    -((alpha + order) / (alpha + 2 * order))
                    * ((alpha + beta + order) / (alpha + 2 * order + 1))
                    * value
                )
                numerator = protect(1 + odd / numerator)
                denominator = 1 / protect(1 + odd * denominator)
                return jnp.stack((numerator, denominator, fraction * numerator * denominator))

            if with_derivatives:
                # Forward-mode propagates elementwise partials without constructing a batch-sized Jacobian
                updated, alpha_derivative = jax.jvp(advance, (alpha, beta, previous[0]), (ones, zeros, previous[1]))
                _, beta_derivative = jax.jvp(advance, (alpha, beta, previous[0]), (zeros, ones, previous[2]))
                current = jnp.stack((updated, alpha_derivative, beta_derivative))
            else:
                current = advance(alpha, beta, previous[0])[None, ...]

            converged = jnp.abs(current[0, 0] * current[0, 1] - 1) <= tolerance
            if with_derivatives:
                # Integer shapes can terminate the value fraction before its shape derivatives converge
                change = jnp.abs(current[1:, 2] - previous[1:, 2])
                converged &= jnp.all(change <= tolerance * jnp.maximum(1.0, jnp.abs(current[1:, 2])), axis=0)
            return jnp.where(active, current, previous), active & ~converged

        if not with_derivatives:
            return update(state)
        return cast(
            tuple[jax.Array, jax.Array],
            jax.lax.cond(jnp.any(state[1]), update, lambda state: state, state),
        )

    # Each step evaluates two coefficients, so use paired-step limits based on JAX's iteration budget
    iterations = 300 if value.dtype == jnp.float64 else 100
    active = jnp.ones_like(value, dtype=bool)
    if with_derivatives:
        # Static bounds let reverse mode differentiate the shape partials when computing Hessians
        result, _ = jax.lax.fori_loop(1, iterations + 1, step, (initial_state, active))
    else:
        # The custom JVP supplies derivatives, so value-only calls can stop as soon as the batch converges
        def unfinished_batch(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            index, _, unfinished = state
            return (index <= iterations) & jnp.any(unfinished)

        def advance_batch(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
            index, terms, unfinished = state
            terms, unfinished = step(index, (terms, unfinished))
            return index + 1, terms, unfinished

        _, result, _ = jax.lax.while_loop(unfinished_batch, advance_batch, (jnp.asarray(1), initial_state, active))

    # Keep the final estimate at the iteration cap, as JAX's continued fraction does
    # Roundoff can keep the stopping check active even when the estimate is accurate
    return tuple(result[:, 2])
