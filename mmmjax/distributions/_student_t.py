"""Student-t distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def student_t_logpdf(
    value: ArrayLike,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Student-t log density elementwise.

    For value :math:`x \in \mathbb{R}`, degrees of freedom :math:`\nu > 0`,
    location :math:`\mu \in \mathbb{R}`, and scale :math:`\sigma > 0`, the
    log density is

    .. math::

        \log p(x \mid \nu, \mu, \sigma)
        = \log\Gamma\left(\frac{\nu + 1}{2}\right)
          - \log\Gamma\left(\frac{\nu}{2}\right)
          - \frac{1}{2}\log(\nu\pi)
          - \log(\sigma)
          - \frac{\nu + 1}{2}
            \log\left(1 + \frac{1}{\nu}
            \left(\frac{x - \mu}{\sigma}\right)^2\right).

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    degrees_of_freedom
        Positive degrees of freedom.
    location
        Location of the distribution.
    scale
        Positive scale parameter. This is not the standard deviation except
        in the Normal limit.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments. A
        nonpositive or nonfinite degrees of freedom or scale, or a nonfinite
        location, produces ``nan``.
    """
    value_array, degrees_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("degrees_of_freedom", degrees_of_freedom),
        ("location", location),
        ("scale", scale),
    )

    valid_degrees = jnp.isfinite(degrees_array) & (degrees_array > 0)
    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)

    log_scale = jnp.log(scale_array)

    one = jnp.ones((), dtype=value_array.dtype)
    log_two = jnp.asarray(math.log(2), dtype=value_array.dtype)

    residual = value_array - location_array
    at_location = value_array == location_array
    scaled_residual_region = jnp.isfinite(value_array) & valid_location & ~at_location

    scaled_value = jnp.where(scaled_residual_region, value_array, jnp.ones_like(value_array))
    scaled_location = jnp.where(scaled_residual_region, location_array, jnp.zeros_like(location_array))
    residual_magnitude = jnp.maximum(jnp.abs(scaled_value), jnp.abs(scaled_location))
    _, residual_exponent = jnp.frexp(residual_magnitude)
    residual_scale = jnp.ldexp(
        jnp.ones_like(residual_magnitude),
        residual_exponent - 1,
    )
    scaled_residual = scaled_value / residual_scale - scaled_location / residual_scale

    direct_residual = jnp.where(
        scaled_residual_region | at_location,
        jnp.ones_like(residual),
        residual,
    )
    residual_for_log = jnp.where(scaled_residual_region, scaled_residual, direct_residual)
    log_scale_adjustment = (residual_exponent - 1).astype(value_array.dtype) * log_two
    log_absolute_residual = jnp.log(jnp.abs(residual_for_log)) + jnp.where(
        scaled_residual_region,
        log_scale_adjustment,
        jnp.zeros_like(log_scale_adjustment),
    )

    epsilon = jnp.spacing(one)
    tail_weight = (degrees_array + 1) / 2
    log_squared_ratio = 2 * (log_absolute_residual - log_scale) - jnp.log(degrees_array)
    log_squared_ratio = jnp.where(at_location, -jnp.inf, log_squared_ratio)
    small_ratio_region = log_squared_ratio < jnp.log(epsilon)

    # The leading expansion keeps the Normal limit when 1 / nu rounds to zero
    small_noncentral_region = small_ratio_region & ~at_location
    noncentral_log_squared_standardized = jnp.where(
        small_noncentral_region,
        2 * (log_absolute_residual - log_scale),
        jnp.zeros_like(log_squared_ratio),
    )
    noncentral_squared_standardized = jnp.exp(noncentral_log_squared_standardized)
    noncentral_squared_ratio = jnp.exp(
        jnp.where(
            small_noncentral_region,
            log_squared_ratio,
            jnp.zeros_like(log_squared_ratio),
        )
    )

    center_residual = jnp.where(at_location, residual, jnp.zeros_like(residual))
    center_scale = jnp.where(
        at_location,
        jax.lax.stop_gradient(scale_array),
        jnp.ones_like(scale_array),
    )
    center_degrees = jnp.where(
        at_location,
        jax.lax.stop_gradient(degrees_array),
        jnp.ones_like(degrees_array),
    )
    center_standardized = center_residual / center_scale
    center_squared_standardized = jnp.square(center_standardized)
    center_squared_ratio = center_squared_standardized / center_degrees

    squared_standardized = jnp.where(
        at_location,
        center_squared_standardized,
        noncentral_squared_standardized,
    )
    squared_ratio = jnp.where(at_location, center_squared_ratio, noncentral_squared_ratio)
    small_ratio_kernel = 0.5 * (squared_standardized + squared_ratio)

    log_kernel_ratio = jnp.where(
        small_ratio_region,
        jnp.zeros_like(log_squared_ratio),
        log_squared_ratio,
    )
    log_kernel = tail_weight * jnp.logaddexp(jnp.zeros_like(log_kernel_ratio), log_kernel_ratio)

    density_kernel = jnp.where(small_ratio_region, small_ratio_kernel, log_kernel)

    # These cutoffs keep the first omitted asymptotic term below the dtype precision
    dtype_bits = jax.dtypes.itemsize_bits(value_array.dtype)
    asymptotic_threshold = jnp.asarray(
        64 if dtype_bits == 64 else 8,
        dtype=value_array.dtype,
    )
    small_degrees_region = degrees_array < 1
    asymptotic_region = degrees_array >= asymptotic_threshold
    gamma_normalizer_degrees = jnp.where(asymptotic_region, jnp.ones_like(degrees_array), degrees_array)

    half = jnp.asarray(0.5, dtype=value_array.dtype)
    log_pi = jnp.asarray(math.log(math.pi), dtype=value_array.dtype)
    half_log_two_pi = jnp.asarray(math.log(2 * math.pi) / 2, dtype=value_array.dtype)

    gamma_numerator = jnp.where(
        small_degrees_region,
        1 + gamma_normalizer_degrees / 2,
        gamma_normalizer_degrees / 2,
    )
    gamma_ratio = gammaln(gamma_numerator) - gammaln((gamma_normalizer_degrees + 1) / 2)
    gamma_normalizer = gamma_ratio + jnp.where(
        small_degrees_region,
        -half * jnp.log(gamma_normalizer_degrees) + log_two + half * log_pi,
        half * (jnp.log(gamma_normalizer_degrees) + log_pi),
    )

    # The asymptotic form avoids subtracting nearly equal log-Gamma values
    asymptotic_degrees = jnp.where(
        asymptotic_region,
        degrees_array,
        jnp.full_like(degrees_array, asymptotic_threshold),
    )
    inverse_degrees = 1 / asymptotic_degrees
    squared_inverse_degrees = jnp.square(inverse_degrees)
    asymptotic_normalizer = half_log_two_pi + inverse_degrees * (
        0.25
        + squared_inverse_degrees
        * (-1 / 24 + squared_inverse_degrees * (1 / 20 + squared_inverse_degrees * (-17 / 112)))
    )

    degrees_normalizer = jnp.where(asymptotic_region, asymptotic_normalizer, gamma_normalizer)

    log_density = -log_scale - degrees_normalizer - density_kernel
    valid_parameters = valid_degrees & valid_location & valid_scale
    return jnp.where(valid_parameters, log_density, jnp.nan)


def student_t(
    value: ArrayLike,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Student-t log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    degrees_of_freedom
        Positive degrees of freedom.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    return jnp.sum(student_t_logpdf(value, degrees_of_freedom, location, scale))


def student_t_rng(
    key: jax.Array,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Student-t distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    degrees_of_freedom
        Positive degrees of freedom.
    location
        Location of the distribution.
    scale
        Positive scale parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite degrees of freedom or scale, or a nonfinite
        location, produces ``nan``.
    """
    degrees_array, location_array, scale_array = _promote_inexact(
        ("degrees_of_freedom", degrees_of_freedom),
        ("location", location),
        ("scale", scale),
    )
    output_shape = _random_shape(sample_shape, degrees_array, location_array, scale_array)

    valid_degrees = jnp.isfinite(degrees_array) & (degrees_array > 0)
    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    # Keep invalid degrees of freedom out of JAX's rejection sampler
    safe_degrees = jnp.where(valid_degrees, degrees_array, jnp.ones_like(degrees_array))

    normal_key, gamma_key = jax.random.split(key)
    standard_normal = jax.random.normal(normal_key, shape=output_shape, dtype=degrees_array.dtype)
    log_unit_gamma = jax.random.loggamma(
        gamma_key,
        safe_degrees / 2,
        shape=output_shape,
        dtype=degrees_array.dtype,
    )

    nonzero_normal = standard_normal != 0
    safe_absolute_normal = jnp.where(nonzero_normal, jnp.abs(standard_normal), jnp.ones_like(standard_normal))
    log_magnitude = (
        jnp.log(scale_array)
        + jnp.log(safe_absolute_normal)
        + 0.5 * (jnp.log(safe_degrees) - math.log(2) - log_unit_gamma)
    )
    centered_samples = jnp.where(
        nonzero_normal,
        jnp.copysign(jnp.exp(log_magnitude), standard_normal),
        jnp.zeros_like(standard_normal),
    )

    samples = location_array + centered_samples
    valid_parameters = valid_degrees & valid_location & valid_scale
    return jnp.where(valid_parameters, samples, jnp.nan)
