"""Inverse Gamma distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._gamma import _gamma_log_probability
from mmmjax.distributions._utils import (
    _gamma_shape_log_derivative,
    _gamma_shape_normalizer,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


def inverse_gamma_logpdf(
    value: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Inverse Gamma log density elementwise.

    For value :math:`x > 0`, shape :math:`\alpha > 0`, and scale
    :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = \alpha\log(\beta)
          - \log\Gamma(\alpha)
          - (\alpha + 1)\log(x)
          - \frac{\beta}{x}.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Nonpositive values and either infinity produce ``-inf``. A
        nonpositive or nonfinite shape or scale produces ``nan``. A ``nan``
        value also produces ``nan``.

    Notes
    -----
    Highly concentrated distributions can require JAX 64-bit mode. If
    ``scale`` and ``shape * value`` differ by less than their dtype can
    represent, that difference cannot be recovered by the density calculation.
    """
    value_array, shape_array, scale_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("scale", scale),
    )
    return _inverse_gamma_logpdf_core(value_array, shape_array, scale_array)


def inverse_gamma(
    value: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Inverse Gamma log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    return jnp.sum(inverse_gamma_logpdf(value, shape, scale))


def inverse_gamma_logcdf(
    value: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Inverse Gamma log cumulative distribution function elementwise.

    For value :math:`x > 0`, shape :math:`\alpha > 0`, and scale
    :math:`\beta > 0`, the log cumulative probability is

    .. math::

        \log F(x \mid \alpha, \beta)
        = \log Q\left(\alpha, \frac{\beta}{x}\right),

    where :math:`Q` is the regularized upper incomplete Gamma function. For
    :math:`x \leq 0`, the log cumulative probability is :math:`-\infty`.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    shape
        Positive shape parameter.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonpositive or nonfinite shape or scale produces ``nan``.
    """
    return _inverse_gamma_log_probability(value, shape, scale, survival=False)


def inverse_gamma_logsf(
    value: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Inverse Gamma log survival function elementwise.

    For value :math:`x > 0`, shape :math:`\alpha > 0`, and scale
    :math:`\beta > 0`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \alpha, \beta)
        = \log P\left(\alpha, \frac{\beta}{x}\right),

    where :math:`P` is the regularized lower incomplete Gamma function. For
    :math:`x \leq 0`, the log survival probability is zero.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    shape
        Positive shape parameter.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonpositive or nonfinite shape or scale produces ``nan``.
    """
    return _inverse_gamma_log_probability(value, shape, scale, survival=True)


def inverse_gamma_rng(
    key: jax.Array,
    shape: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from an Inverse Gamma distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    shape
        Positive shape parameter.
    scale
        Positive scale parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape or scale produces ``nan``.
    """
    shape_array, scale_array = _promote_inexact(("shape", shape), ("scale", scale))
    output_shape = _random_shape(sample_shape, shape_array, scale_array)

    valid_shape = jnp.isfinite(shape_array) & (shape_array > 0)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_shape = jnp.where(valid_shape, shape_array, jnp.ones_like(shape_array))

    # Work in log space so tiny Gamma draws can be inverted before they underflow
    log_unit_rate_samples = jax.random.loggamma(
        key,
        safe_shape,
        shape=output_shape,
        dtype=shape_array.dtype,
    )
    samples = jnp.exp(jnp.log(scale_array) - log_unit_rate_samples)

    return jnp.where(valid_shape & valid_scale, samples, jnp.nan)


@jax.custom_jvp
def _inverse_gamma_logpdf_core(
    value: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    valid_value = jnp.isfinite(value) & (value > 0)
    valid_shape = jnp.isfinite(shape) & (shape > 0)
    valid_scale = jnp.isfinite(scale) & (scale > 0)

    # Sanitizing every candidate branch keeps invalid inputs out of JAX derivatives
    safe_value = jnp.where(valid_value, value, jnp.ones_like(value))
    safe_shape = jnp.where(valid_shape, shape, jnp.ones_like(shape))
    safe_scale = jnp.where(valid_scale, scale, jnp.ones_like(scale))

    log_ratio, density_deviation, _, _, has_density_deviation = _inverse_gamma_ratio_terms(
        safe_value,
        safe_shape,
        safe_scale,
    )

    density_contribution = _weighted_log_ratio_deviance(
        safe_shape,
        log_ratio,
        jnp.where(has_density_deviation, density_deviation, jnp.inf),
    )

    interior_log_density = _gamma_shape_normalizer(safe_shape) + density_contribution - jnp.log(safe_value)
    supported_log_density = jnp.where(
        valid_value,
        interior_log_density,
        jnp.where(jnp.isnan(value), jnp.nan, -jnp.inf),
    )
    return jnp.where(valid_shape & valid_scale, supported_log_density, jnp.nan)


@_inverse_gamma_logpdf_core.defjvp
def _inverse_gamma_logpdf_core_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, shape, scale = primals
    value_tangent, shape_tangent, scale_tangent = tangents
    log_density = _inverse_gamma_logpdf_core(value, shape, scale)

    valid_value = jnp.isfinite(value) & (value > 0)
    valid_shape = jnp.isfinite(shape) & (shape > 0)
    valid_scale = jnp.isfinite(scale) & (scale > 0)
    safe_value = jnp.where(valid_value, value, jnp.ones_like(value))
    safe_shape = jnp.where(valid_shape, shape, jnp.ones_like(shape))
    safe_scale = jnp.where(valid_scale, scale, jnp.ones_like(scale))

    log_ratio, density_deviation, ratio_deviation, near_unit_ratio, _ = _inverse_gamma_ratio_terms(
        safe_value,
        safe_shape,
        safe_scale,
    )

    value_derivative = (density_deviation - 1) / safe_value
    shape_derivative = log_ratio + _gamma_shape_log_derivative(safe_shape)

    direct_scale_derivative = safe_shape / safe_scale - 1 / safe_value
    near_ratio_deviation = jnp.where(
        near_unit_ratio,
        ratio_deviation,
        jnp.zeros_like(ratio_deviation),
    )
    near_scale_derivative = -near_ratio_deviation / (safe_value * (1 + near_ratio_deviation))
    scale_derivative = jnp.where(
        near_unit_ratio,
        near_scale_derivative,
        direct_scale_derivative,
    )

    valid_interior = valid_value & valid_shape & valid_scale
    defined_support = valid_shape & valid_scale & ~jnp.isnan(value)
    undefined_derivative = jnp.where(
        defined_support,
        jnp.zeros_like(value_derivative),
        jnp.nan,
    )
    value_derivative = jnp.where(
        valid_interior,
        value_derivative,
        undefined_derivative,
    )
    shape_derivative = jnp.where(
        valid_interior,
        shape_derivative,
        undefined_derivative,
    )
    scale_derivative = jnp.where(
        valid_interior,
        scale_derivative,
        undefined_derivative,
    )
    log_density_tangent = (
        value_derivative * value_tangent + shape_derivative * shape_tangent + scale_derivative * scale_tangent
    )
    return log_density, log_density_tangent


def _inverse_gamma_ratio_terms(
    value: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    product_denominator = value * shape
    product_difference = scale - product_denominator
    product_density_deviation = product_difference / value

    scaled_inverse_value = scale / value
    sequential_density_deviation = scaled_inverse_value - shape

    raw_log_ratio = jnp.log(scale) - jnp.log(value) - jnp.log(shape)
    product_log_ratio, product_ratio_deviation, valid_product_ratio = _stable_log_ratio(
        scale,
        product_denominator,
        raw_log_ratio,
    )
    sequential_log_ratio, sequential_ratio_deviation, valid_sequential_ratio = _stable_log_ratio(
        scaled_inverse_value,
        shape,
        raw_log_ratio,
    )
    has_finite_ratio = valid_product_ratio | valid_sequential_ratio
    log_ratio = jnp.where(valid_product_ratio, product_log_ratio, sequential_log_ratio)
    ratio_deviation = jnp.where(
        valid_product_ratio,
        product_ratio_deviation,
        sequential_ratio_deviation,
    )

    valid_product_density_deviation = (
        jnp.isfinite(product_denominator) & (product_denominator > 0) & jnp.isfinite(product_density_deviation)
    )
    valid_sequential_density_deviation = jnp.isfinite(sequential_density_deviation)
    has_density_deviation = valid_product_density_deviation | valid_sequential_density_deviation
    density_deviation = jnp.where(
        valid_product_density_deviation,
        product_density_deviation,
        sequential_density_deviation,
    )

    near_unit_ratio = has_finite_ratio & (jnp.abs(ratio_deviation) < 0.5)
    return log_ratio, density_deviation, ratio_deviation, near_unit_ratio, has_density_deviation


def _inverse_gamma_log_probability(
    value: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
    *,
    survival: bool,
) -> jax.Array:
    value_array, shape_array, scale_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("scale", scale),
    )

    valid_shape = jnp.isfinite(shape_array) & (shape_array > 0)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    interior_value = jnp.isfinite(value_array) & (value_array > 0)
    evaluate_probability = valid_shape & valid_scale & interior_value
    safe_value = jnp.where(evaluate_probability, value_array, jnp.ones_like(value_array))
    safe_shape = jnp.where(evaluate_probability, shape_array, jnp.ones_like(shape_array))
    safe_scale = jnp.where(evaluate_probability, scale_array, jnp.ones_like(scale_array))
    scaled_inverse_value = safe_scale / safe_value
    log_scaled_inverse_value = jnp.log(safe_scale) - jnp.log(safe_value)
    unit_rate = jnp.ones((), dtype=value_array.dtype)

    if survival:
        log_probability = _gamma_log_probability(
            scaled_inverse_value,
            safe_shape,
            unit_rate,
            upper_tail=False,
            log_scaled_value=log_scaled_inverse_value,
        )
        boundary_log_probability = jnp.where(value_array <= 0, 0, -jnp.inf)
    else:
        log_probability = _gamma_log_probability(
            scaled_inverse_value,
            safe_shape,
            unit_rate,
            upper_tail=True,
            log_scaled_value=log_scaled_inverse_value,
        )
        boundary_log_probability = jnp.where(value_array <= 0, -jnp.inf, 0)

    supported_log_probability = jnp.where(
        evaluate_probability,
        log_probability,
        boundary_log_probability,
    )
    supported_log_probability = jnp.where(jnp.isnan(value_array), jnp.nan, supported_log_probability)
    return jnp.where(valid_shape & valid_scale, supported_log_probability, jnp.nan)
