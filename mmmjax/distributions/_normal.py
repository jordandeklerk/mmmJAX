"""Normal distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def normal_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Normal log density elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \mu, \sigma)
        = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2
          - \log(\sigma)
          - \frac{1}{2}\log(2\pi),
        \qquad \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    log_density = _normal_logpdf_kernel(value_array, location_array, scale_array)

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, log_density, jnp.nan)


def normal(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Normal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_densities = normal_logpdf(value, location, scale)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    location_array = jnp.asarray(location)
    scale_array = jnp.asarray(scale)
    valid_parameters = jnp.all(jnp.isfinite(location_array)) & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_parameters, log_density, jnp.nan)


def normal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Normal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.
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

    shape = _random_shape(sample_shape, location_array, scale_array)
    standard_normal = jax.random.normal(key, shape=shape, dtype=location_array.dtype)
    samples = location_array + scale_array * standard_normal

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)


def _normal_logpdf_kernel(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    # Keep the scale out of the square so extreme values stay finite
    standardized = (value - location) / scale
    half_log_two_pi = jnp.asarray(math.log(2 * math.pi) / 2, dtype=value.dtype)
    return -0.5 * jnp.square(standardized) - jnp.log(scale) - half_log_two_pi
