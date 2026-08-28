"""Beta distribution functions."""

from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.stats import beta as beta_distribution
from jax.typing import ArrayLike

from mmmjax.distributions._utils import (
    _asymptotic_gamma_shape_log_derivative,
    _asymptotic_gamma_shape_normalizer,
    _gamma_shape_log_derivative,
    _gamma_shape_normalizer,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


def beta_logpdf(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Beta log density elementwise.

    For value :math:`x \in (0, 1)` and shape parameters :math:`\alpha > 0`
    and :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = (\alpha - 1)\log(x)
          + (\beta - 1)\log(1 - x)
          - \log\mathrm{B}(\alpha, \beta).

    At :math:`x = 0`, the log density is :math:`+\infty` when
    :math:`\alpha < 1`, :math:`\log(\beta)` when :math:`\alpha = 1`, and
    :math:`-\infty` when :math:`\alpha > 1`. At :math:`x = 1`, the log
    density is :math:`+\infty` when :math:`\beta < 1`, :math:`\log(\alpha)`
    when :math:`\beta = 1`, and :math:`-\infty` when :math:`\beta > 1`.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values outside ``[0, 1]`` produce ``-inf``. A nonpositive or
        nonfinite shape parameter produces ``nan``.
    """
    value_array, alpha_array, beta_array = _promote_inexact(
        ("value", value),
        ("alpha", alpha),
        ("beta", beta),
    )
    return _beta_logpdf(value_array, alpha_array, beta_array)


def beta(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Beta log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_densities = beta_logpdf(value, alpha, beta)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    alpha_array = jnp.asarray(alpha)
    beta_array = jnp.asarray(beta)
    valid_parameters = jnp.all(jnp.isfinite(alpha_array) & (alpha_array > 0)) & jnp.all(
        jnp.isfinite(beta_array) & (beta_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def beta_rng(
    key: jax.Array,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Beta distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape parameter produces ``nan``.
    """
    alpha_array, beta_array = _promote_inexact(("alpha", alpha), ("beta", beta))
    output_shape = _random_shape(sample_shape, alpha_array, beta_array)

    valid_alpha = jnp.isfinite(alpha_array) & (alpha_array > 0)
    valid_beta = jnp.isfinite(beta_array) & (beta_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_alpha = jnp.where(valid_alpha, alpha_array, jnp.ones_like(alpha_array))
    safe_beta = jnp.where(valid_beta, beta_array, jnp.ones_like(beta_array))

    samples = jax.random.beta(
        key,
        safe_alpha,
        safe_beta,
        shape=output_shape,
        dtype=alpha_array.dtype,
    )

    return jnp.where(valid_alpha & valid_beta, samples, jnp.nan)


@jax.custom_jvp
def _beta_logpdf(
    value: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    valid_value = jnp.isfinite(value) & (value > 0) & (value < 1)
    valid_alpha = jnp.isfinite(alpha) & (alpha > 0)
    valid_beta = jnp.isfinite(beta) & (beta > 0)
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    uses_standard_formula = (
        jnp.all(valid_value)
        & jnp.all(valid_alpha)
        & jnp.all(valid_beta)
        & jnp.all(jnp.maximum(alpha, beta) < asymptotic_threshold)
    )
    # Ordinary homogeneous batches do not need the heavier stability calculation
    return cast(
        jax.Array,
        jax.lax.cond(
            uses_standard_formula,
            _standard_beta_logpdf,
            _stable_beta_logpdf,
            value,
            alpha,
            beta,
        ),
    )


@_beta_logpdf.defjvp
def _beta_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    log_density = _beta_logpdf(*primals)

    # The robust rule preserves derivatives at boundaries and extreme shapes
    _, log_density_tangent = jax.jvp(_stable_beta_logpdf, primals, tangents)
    return log_density, log_density_tangent


@jax.custom_jvp
def _stable_beta_logpdf(
    value: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    valid_value = jnp.isfinite(value) & (value > 0) & (value < 1)
    valid_alpha = jnp.isfinite(alpha) & (alpha > 0)
    valid_beta = jnp.isfinite(beta) & (beta > 0)

    # Sanitizing every candidate branch keeps invalid inputs out of JAX derivatives
    safe_value = jnp.where(valid_value, value, jnp.full_like(value, 0.5))
    safe_alpha = jnp.where(valid_alpha, alpha, jnp.ones_like(alpha))
    safe_beta = jnp.where(valid_beta, beta, jnp.ones_like(beta))
    one_minus_value = 1 - safe_value

    direct_sum, log_sum, inverse_sum, exact_sum_region = _beta_shape_sum(safe_alpha, safe_beta)
    sum_normalizer = _beta_sum_normalizer(
        direct_sum,
        log_sum,
        inverse_sum,
        exact_sum_region,
    )
    normalizer = _gamma_shape_normalizer(safe_alpha) + _gamma_shape_normalizer(safe_beta) - sum_normalizer

    largest_shape = jnp.maximum(safe_alpha, safe_beta)
    scaled_alpha = safe_alpha / largest_shape
    scaled_beta = safe_beta / largest_shape
    scaled_sum = scaled_alpha + scaled_beta
    alpha_mean = scaled_alpha / scaled_sum
    beta_mean = scaled_beta / scaled_sum

    log_value = jnp.log(safe_value)
    log_one_minus_value = jnp.log1p(-safe_value)
    raw_alpha_log_ratio = log_value + log_sum - jnp.log(safe_alpha)
    raw_beta_log_ratio = log_one_minus_value + log_sum - jnp.log(safe_beta)
    alpha_log_ratio, _, _ = _stable_log_ratio(safe_value, alpha_mean, raw_alpha_log_ratio)
    beta_log_ratio, _, _ = _stable_log_ratio(one_minus_value, beta_mean, raw_beta_log_ratio)

    value_deviation = safe_value - alpha_mean
    shape_difference = -largest_shape * (scaled_sum * value_deviation)
    alpha_contribution = _weighted_log_ratio_deviance(
        safe_alpha,
        alpha_log_ratio,
        -shape_difference,
    )
    beta_contribution = _weighted_log_ratio_deviance(
        safe_beta,
        beta_log_ratio,
        shape_difference,
    )
    centered_log_density = normalizer + alpha_contribution + beta_contribution - log_value - log_one_minus_value

    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    centered_region = jnp.minimum(safe_alpha, safe_beta) >= asymptotic_threshold
    uses_standard_formula = largest_shape < asymptotic_threshold

    # The standard lgamma formula avoids the approximation JAX betaln uses at its cutoff
    standard_value = jnp.where(uses_standard_formula, safe_value, jnp.full_like(safe_value, 0.5))
    standard_alpha = jnp.where(uses_standard_formula, safe_alpha, jnp.ones_like(safe_alpha))
    standard_beta = jnp.where(uses_standard_formula, safe_beta, jnp.ones_like(safe_beta))
    standard_log_density = (
        gammaln(standard_alpha + standard_beta)
        - gammaln(standard_alpha)
        - gammaln(standard_beta)
        + (standard_alpha - 1) * jnp.log(standard_value)
        + (standard_beta - 1) * jnp.log1p(-standard_value)
    )

    # JAX betaln protects the mixed large-small regime from lgamma cancellation
    mixed_region = ~centered_region & ~uses_standard_formula
    mixed_value = jnp.where(mixed_region, safe_value, jnp.full_like(safe_value, 0.5))
    mixed_alpha = jnp.where(mixed_region, safe_alpha, jnp.ones_like(safe_alpha))
    mixed_beta = jnp.where(mixed_region, safe_beta, jnp.ones_like(safe_beta))
    mixed_log_density = beta_distribution.logpdf(mixed_value, mixed_alpha, mixed_beta)
    ordinary_log_density = jnp.where(uses_standard_formula, standard_log_density, mixed_log_density)
    interior_log_density = jnp.where(centered_region, centered_log_density, ordinary_log_density)

    lower_boundary_log_density = jnp.where(
        safe_alpha < 1,
        jnp.inf,
        jnp.where(safe_alpha == 1, jnp.log(safe_beta), -jnp.inf),
    )
    upper_boundary_log_density = jnp.where(
        safe_beta < 1,
        jnp.inf,
        jnp.where(safe_beta == 1, jnp.log(safe_alpha), -jnp.inf),
    )
    supported_log_density = jnp.where(
        value == 0,
        lower_boundary_log_density,
        jnp.where(
            value == 1,
            upper_boundary_log_density,
            jnp.where(valid_value, interior_log_density, jnp.where(jnp.isnan(value), jnp.nan, -jnp.inf)),
        ),
    )
    return jnp.where(valid_alpha & valid_beta, supported_log_density, jnp.nan)


@_stable_beta_logpdf.defjvp
def _stable_beta_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, alpha, beta = primals
    value_tangent, alpha_tangent, beta_tangent = tangents
    log_density = _stable_beta_logpdf(value, alpha, beta)

    valid_value = jnp.isfinite(value) & (value > 0) & (value < 1)
    valid_alpha = jnp.isfinite(alpha) & (alpha > 0)
    valid_beta = jnp.isfinite(beta) & (beta > 0)
    safe_value = jnp.where(valid_value, value, jnp.full_like(value, 0.5))
    safe_alpha = jnp.where(valid_alpha, alpha, jnp.ones_like(alpha))
    safe_beta = jnp.where(valid_beta, beta, jnp.ones_like(beta))
    one_minus_value = 1 - safe_value

    direct_sum, log_sum, inverse_sum, exact_sum_region = _beta_shape_sum(safe_alpha, safe_beta)
    sum_log_derivative = _beta_sum_log_derivative(
        direct_sum,
        inverse_sum,
        exact_sum_region,
    )

    largest_shape = jnp.maximum(safe_alpha, safe_beta)
    scaled_alpha = safe_alpha / largest_shape
    scaled_beta = safe_beta / largest_shape
    scaled_sum = scaled_alpha + scaled_beta
    alpha_mean = scaled_alpha / scaled_sum
    beta_mean = scaled_beta / scaled_sum

    raw_alpha_log_ratio = jnp.log(safe_value) + log_sum - jnp.log(safe_alpha)
    raw_beta_log_ratio = jnp.log1p(-safe_value) + log_sum - jnp.log(safe_beta)
    alpha_log_ratio, _, _ = _stable_log_ratio(safe_value, alpha_mean, raw_alpha_log_ratio)
    beta_log_ratio, _, _ = _stable_log_ratio(one_minus_value, beta_mean, raw_beta_log_ratio)

    value_deviation = safe_value - alpha_mean
    shape_difference = -largest_shape * (scaled_sum * value_deviation)
    centered_value_derivative = (shape_difference + 2 * safe_value - 1) / (safe_value * one_minus_value)
    ordinary_value_derivative = (safe_alpha - 1) / safe_value - (safe_beta - 1) / one_minus_value
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    centered_region = jnp.minimum(safe_alpha, safe_beta) >= asymptotic_threshold
    value_derivative = jnp.where(
        centered_region,
        centered_value_derivative,
        ordinary_value_derivative,
    )
    alpha_derivative = alpha_log_ratio + _gamma_shape_log_derivative(safe_alpha) - sum_log_derivative
    beta_derivative = beta_log_ratio + _gamma_shape_log_derivative(safe_beta) - sum_log_derivative

    value_derivative = jnp.where(valid_value, value_derivative, jnp.zeros_like(value_derivative))
    lower_alpha_derivative = jnp.zeros_like(alpha_derivative)
    lower_beta_derivative = jnp.where(
        (value == 0) & (safe_alpha == 1),
        1 / safe_beta,
        jnp.zeros_like(beta_derivative),
    )
    upper_alpha_derivative = jnp.where(
        (value == 1) & (safe_beta == 1),
        1 / safe_alpha,
        lower_alpha_derivative,
    )
    upper_beta_derivative = jnp.zeros_like(beta_derivative)
    alpha_derivative = jnp.where(
        valid_value,
        alpha_derivative,
        jnp.where(value == 1, upper_alpha_derivative, lower_alpha_derivative),
    )
    beta_derivative = jnp.where(
        valid_value,
        beta_derivative,
        jnp.where(value == 0, lower_beta_derivative, upper_beta_derivative),
    )

    defined_derivatives = valid_alpha & valid_beta & ~jnp.isnan(value)
    value_derivative = jnp.where(defined_derivatives, value_derivative, jnp.nan)
    alpha_derivative = jnp.where(defined_derivatives, alpha_derivative, jnp.nan)
    beta_derivative = jnp.where(defined_derivatives, beta_derivative, jnp.nan)

    log_density_tangent = (
        value_derivative * value_tangent + alpha_derivative * alpha_tangent + beta_derivative * beta_tangent
    )
    return log_density, log_density_tangent


def _beta_shape_sum(
    alpha: jax.Array,
    beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    largest_shape = jnp.maximum(alpha, beta)
    scaled_sum = alpha / largest_shape + beta / largest_shape
    log_sum = jnp.log(largest_shape) + jnp.log(scaled_sum)
    inverse_sum = (1 / largest_shape) / scaled_sum

    direct_sum = alpha + beta
    dtype_bits = jax.dtypes.itemsize_bits(alpha.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=alpha.dtype)
    exact_sum_region = jnp.isfinite(direct_sum) & (direct_sum < asymptotic_threshold)
    return direct_sum, log_sum, inverse_sum, exact_sum_region


def _beta_sum_normalizer(
    direct_sum: jax.Array,
    log_sum: jax.Array,
    inverse_sum: jax.Array,
    exact_sum_region: jax.Array,
) -> jax.Array:
    exact_sum = jnp.where(exact_sum_region, direct_sum, jnp.ones_like(direct_sum))
    exact_normalizer = _gamma_shape_normalizer(exact_sum)
    asymptotic_normalizer = _asymptotic_gamma_shape_normalizer(log_sum, inverse_sum)
    return jnp.where(exact_sum_region, exact_normalizer, asymptotic_normalizer)


def _beta_sum_log_derivative(
    direct_sum: jax.Array,
    inverse_sum: jax.Array,
    exact_sum_region: jax.Array,
) -> jax.Array:
    exact_sum = jnp.where(exact_sum_region, direct_sum, jnp.ones_like(direct_sum))
    exact_derivative = _gamma_shape_log_derivative(exact_sum)
    asymptotic_derivative = _asymptotic_gamma_shape_log_derivative(inverse_sum)
    return jnp.where(exact_sum_region, exact_derivative, asymptotic_derivative)


def _standard_beta_logpdf(
    value: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    dtype_bits = jax.dtypes.itemsize_bits(value.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value.dtype)
    valid_value = jnp.isfinite(value) & (value > 0) & (value < 1)
    valid_alpha = jnp.isfinite(alpha) & (alpha > 0) & (alpha < asymptotic_threshold)
    valid_beta = jnp.isfinite(beta) & (beta > 0) & (beta < asymptotic_threshold)

    # vmap can turn the conditional into a select, so keep the inactive branch finite
    safe_value = jnp.where(valid_value, value, jnp.full_like(value, 0.5))
    safe_alpha = jnp.where(valid_alpha, alpha, jnp.ones_like(alpha))
    safe_beta = jnp.where(valid_beta, beta, jnp.ones_like(beta))
    return (
        gammaln(safe_alpha + safe_beta)
        - gammaln(safe_alpha)
        - gammaln(safe_beta)
        + (safe_alpha - 1) * jnp.log(safe_value)
        + (safe_beta - 1) * jnp.log1p(-safe_value)
    )
