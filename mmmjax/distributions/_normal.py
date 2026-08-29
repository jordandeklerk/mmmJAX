"""Normal distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
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


def normal_logcdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Normal log cumulative distribution function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log cumulative probability is

    .. math::

        \log F(x \mid \mu, \sigma)
        = \log \Phi\left(\frac{x - \mu}{\sigma}\right),
        \qquad \sigma > 0,

    where :math:`\Phi` is the standard Normal cumulative distribution
    function.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonfinite location or a nonpositive or nonfinite scale
        produces ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    return _normal_logcdf_kernel(value_array, location_array, scale_array)


def normal_logsf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Normal log survival function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \mu, \sigma)
        = \log\left[1 - \Phi\left(\frac{x - \mu}{\sigma}\right)\right]
        = \log \Phi\left(\frac{\mu - x}{\sigma}\right),
        \qquad \sigma > 0,

    where :math:`\Phi` is the standard Normal cumulative distribution
    function.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    return _normal_logsf_kernel(value_array, location_array, scale_array)


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
    standardized = _standardize(value, location, scale)
    half_log_two_pi = jnp.asarray(math.log(2 * math.pi) / 2, dtype=value.dtype)
    return -0.5 * jnp.square(standardized) - jnp.log(scale) - half_log_two_pi


def _normal_logcdf_kernel(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return _normal_log_probability(value, location, scale, direction=1)


def _normal_logsf_kernel(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return _normal_log_probability(value, location, scale, direction=-1)


def _normal_log_probability(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    direction: int,
) -> jax.Array:
    valid_parameters = jnp.isfinite(location) & jnp.isfinite(scale) & (scale > 0)
    infinite_value = jnp.isinf(value)
    evaluate_probability = valid_parameters & ~infinite_value
    safe_value = jnp.where(infinite_value, jnp.zeros_like(value), value)
    safe_location = jnp.where(evaluate_probability, location, jnp.zeros_like(location))
    safe_scale = jnp.where(evaluate_probability, scale, jnp.ones_like(scale))
    log_probability = log_ndtr(direction * _standardize(safe_value, safe_location, safe_scale))

    endpoint_probability = jnp.where(direction * value > 0, jnp.zeros_like(value), -jnp.inf)
    supported_log_probability = jnp.where(infinite_value, endpoint_probability, log_probability)
    return jnp.where(valid_parameters, supported_log_probability, jnp.nan)


@jax.custom_jvp
def _standardize(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    difference = value - location
    subtraction_overflowed = jnp.isinf(difference) & jnp.isfinite(value) & jnp.isfinite(location)
    zero = jnp.zeros_like(difference)
    standardized = jnp.where(subtraction_overflowed, zero, difference) / scale

    # Halving both parts keeps the overflow path in range without changing the ratio
    half = jnp.asarray(0.5, dtype=difference.dtype)
    scaled_value = jnp.where(subtraction_overflowed, value, zero) * half
    scaled_location = jnp.where(subtraction_overflowed, location, zero) * half
    scaled_scale = jnp.where(subtraction_overflowed, scale, jnp.ones_like(scale)) * half
    overflow_standardized = (scaled_value - scaled_location) / scaled_scale

    return jnp.where(subtraction_overflowed, overflow_standardized, standardized)


@_standardize.defjvp
def _standardize_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, location, scale = primals
    value_tangent, location_tangent, scale_tangent = tangents
    standardized = _standardize(value, location, scale)

    # The analytic rule keeps masked overflow work out of model gradients
    difference_tangent = value_tangent - location_tangent
    standardized_tangent = difference_tangent / scale - standardized * (scale_tangent / scale)
    return standardized, standardized_tangent
