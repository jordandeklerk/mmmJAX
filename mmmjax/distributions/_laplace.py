"""Laplace distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def laplace_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Laplace log density elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`b > 0`, the log density is

    .. math::

        \log p(x \mid \mu, b)
        = -\log(2b) - \frac{|x - \mu|}{b},
        \qquad b > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive scale parameter. The standard deviation is
        :math:`\sqrt{2}b`.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        The value and location gradients use a zero subgradient when
        ``value == location``. A nonfinite location or a nonpositive or
        nonfinite scale produces ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    safe_location = jnp.where(valid_location, location_array, jnp.zeros_like(location_array))
    safe_scale = jnp.where(valid_scale, scale_array, jnp.ones_like(scale_array))

    crosses_zero = ((value_array < 0) & (safe_location > 0)) | ((value_array > 0) & (safe_location < 0))

    # Separate magnitudes keep opposite-sign values finite near the dtype limits
    cross_value = jnp.where(crosses_zero, value_array, jnp.zeros_like(value_array))
    cross_location = jnp.where(crosses_zero, safe_location, jnp.zeros_like(safe_location))
    cross_zero_distance = jnp.abs(cross_value) / safe_scale + jnp.abs(cross_location) / safe_scale

    direct_value = jnp.where(crosses_zero, jnp.zeros_like(value_array), value_array)
    direct_location = jnp.where(crosses_zero, jnp.zeros_like(safe_location), safe_location)
    residual = direct_value - direct_location
    # The constant branch gives the same symmetric subgradient Stan uses at the cusp
    residual_magnitude = jnp.where(residual == 0, jnp.zeros_like(residual), jnp.abs(residual))
    direct_distance = residual_magnitude / safe_scale
    standardized_distance = jnp.where(crosses_zero, cross_zero_distance, direct_distance)

    log_two = jnp.asarray(math.log(2), dtype=value_array.dtype)
    log_density = -log_two - jnp.log(safe_scale) - standardized_distance
    return jnp.where(valid_location & valid_scale, log_density, jnp.nan)


def laplace(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Laplace log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
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
    log_densities = laplace_logpdf(value, location, scale)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    location_array, scale_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
    )
    valid_parameters = jnp.all(jnp.isfinite(location_array)) & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_parameters, log_density, jnp.nan)


def laplace_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Laplace distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
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
        nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    location_array, scale_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
    )
    output_shape = _random_shape(sample_shape, location_array, scale_array)

    standard_samples = jax.random.laplace(key, shape=output_shape, dtype=location_array.dtype)
    samples = location_array + scale_array * standard_samples

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)
