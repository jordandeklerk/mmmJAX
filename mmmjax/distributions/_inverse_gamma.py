"""Inverse Gamma distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


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
    """
    value_array, shape_array, scale_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("scale", scale),
    )

    outside_support = (value_array <= 0) | jnp.isinf(value_array)
    # Keep unsupported values out of logarithms and division
    safe_value = jnp.where(outside_support, jnp.ones_like(value_array), value_array)

    log_value = jnp.log(safe_value)
    log_scale_to_value = jnp.log(scale_array) - log_value

    # The first omitted Stirling term is below the dtype precision at these cutoffs
    dtype_bits = jax.dtypes.itemsize_bits(value_array.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=value_array.dtype)
    large_shape = jnp.isfinite(shape_array) & (shape_array >= asymptotic_threshold)
    centered_shape = jnp.where(large_shape, shape_array, asymptotic_threshold)

    # Centering around scale / value = shape avoids large terms canceling near the mode
    product_denominator = safe_value * centered_shape
    product_deviation = (scale_array - product_denominator) / product_denominator
    scaled_inverse_value = scale_array / safe_value
    sequential_deviation = (scaled_inverse_value - centered_shape) / centered_shape
    valid_product_deviation = (
        jnp.isfinite(product_denominator)
        & (product_denominator > 0)
        & jnp.isfinite(product_deviation)
        & (product_deviation > -1)
    )
    valid_sequential_deviation = (
        jnp.isfinite(scaled_inverse_value)
        & (scaled_inverse_value > 0)
        & jnp.isfinite(sequential_deviation)
        & (sequential_deviation > -1)
    )
    has_finite_deviation = valid_product_deviation | valid_sequential_deviation
    scale_to_shape_deviation = jnp.where(
        valid_product_deviation,
        product_deviation,
        jnp.where(valid_sequential_deviation, sequential_deviation, jnp.zeros_like(product_deviation)),
    )
    centered_region = large_shape & has_finite_deviation

    centered_value = jnp.where(centered_region, safe_value, jnp.ones_like(safe_value))
    centered_scale = jnp.where(centered_region, scale_array, centered_shape)
    scale_to_shape_deviation = jnp.where(
        centered_region,
        scale_to_shape_deviation,
        jnp.zeros_like(scale_to_shape_deviation),
    )
    logarithmic_ratio = jnp.log(centered_scale) - jnp.log(centered_value) - jnp.log(centered_shape)
    direct_logarithmic_ratio = jnp.log1p(scale_to_shape_deviation)
    # The direct value keeps nearby ratios precise while the log path avoids overflow in AD
    log_scale_to_shape_ratio = logarithmic_ratio + jax.lax.stop_gradient(direct_logarithmic_ratio - logarithmic_ratio)
    # Terms through seventh order put the first omitted term below the dtype precision here
    series_threshold = jnp.asarray(0.01 if dtype_bits == 64 else 0.1, dtype=value_array.dtype)
    series_region = jnp.abs(log_scale_to_shape_ratio) < series_threshold
    series_argument = jnp.where(
        series_region,
        log_scale_to_shape_ratio,
        jnp.zeros_like(log_scale_to_shape_ratio),
    )
    squared_series_argument = jnp.square(series_argument)
    log_ratio_deviation_series = -squared_series_argument * (
        0.5
        + series_argument
        * (
            1 / 6
            + series_argument
            * (1 / 24 + series_argument * (1 / 120 + series_argument * (1 / 720 + series_argument / 5040)))
        )
    )
    direct_argument = jnp.where(
        series_region,
        jnp.zeros_like(log_scale_to_shape_ratio),
        log_scale_to_shape_ratio,
    )
    log_ratio_deviation = jnp.where(
        series_region,
        log_ratio_deviation_series,
        direct_argument - jnp.expm1(direct_argument),
    )

    # Stirling's correction keeps the Gamma normalizer accurate for large shapes
    inverse_shape = 1 / centered_shape
    squared_inverse_shape = jnp.square(inverse_shape)
    stirling_correction = inverse_shape * (
        1 / 12
        + squared_inverse_shape * (-1 / 360 + squared_inverse_shape * (1 / 1260 + squared_inverse_shape * (-1 / 1680)))
    )
    centered_normalizer = (
        0.5 * (jnp.log(centered_shape) - jnp.asarray(math.log(2 * math.pi), dtype=value_array.dtype))
        - stirling_correction
    )
    centered_log_density = centered_shape * log_ratio_deviation + centered_normalizer - log_value

    standard_shape = jnp.where(centered_region, jnp.ones_like(shape_array), shape_array)
    standard_log_scale_to_value = jnp.where(
        centered_region,
        jnp.zeros_like(log_scale_to_value),
        log_scale_to_value,
    )
    standard_log_density = (
        standard_shape * standard_log_scale_to_value
        - gammaln(standard_shape)
        - log_value
        - jnp.exp(standard_log_scale_to_value)
    )
    interior_log_density = jnp.where(centered_region, centered_log_density, standard_log_density)
    supported_log_density = jnp.where(outside_support, -jnp.inf, interior_log_density)

    valid_parameters = jnp.isfinite(shape_array) & (shape_array > 0) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


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
    log_density = jnp.sum(inverse_gamma_logpdf(value, shape, scale))

    shape_array, scale_array = _promote_inexact(("shape", shape), ("scale", scale))
    valid_shape = jnp.all(jnp.isfinite(shape_array) & (shape_array > 0))
    valid_scale = jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_shape & valid_scale, log_density, jnp.nan)


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
    safe_scale = jnp.where(valid_scale, scale_array, jnp.ones_like(scale_array))

    # Work in log space so tiny Gamma draws can be inverted before they underflow
    log_unit_rate_samples = jax.random.loggamma(
        key,
        safe_shape,
        shape=output_shape,
        dtype=shape_array.dtype,
    )
    samples = jnp.exp(jnp.log(safe_scale) - log_unit_rate_samples)

    return jnp.where(valid_shape & valid_scale, samples, jnp.nan)
