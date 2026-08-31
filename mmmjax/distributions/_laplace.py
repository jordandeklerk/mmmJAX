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

    standardized = _standardize(value_array, safe_location, safe_scale)
    # The constant branch gives the same symmetric subgradient Stan uses at the cusp
    standardized_distance = jnp.where(
        value_array == safe_location,
        jnp.zeros_like(standardized),
        jnp.where(value_array < safe_location, -standardized, standardized),
    )

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
    return jnp.sum(laplace_logpdf(value, location, scale))


def laplace_logcdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Laplace log cumulative distribution function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`b > 0`, the log cumulative probability is

    .. math::

        \log F(x \mid \mu, b)
        = \begin{cases}
            -\log(2) + \dfrac{x - \mu}{b}, & x < \mu, \\
            \log\left[1 - \dfrac{1}{2}
            \exp\left(-\dfrac{x - \mu}{b}\right)\right], & x \geq \mu.
          \end{cases}

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

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
    return _laplace_log_probability(
        value_array,
        location_array,
        scale_array,
        survival=False,
    )


def laplace_logsf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Laplace log survival function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`b > 0`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \mu, b)
        = \begin{cases}
            \log\left[1 - \dfrac{1}{2}
            \exp\left(\dfrac{x - \mu}{b}\right)\right], & x < \mu, \\
            -\log(2) - \dfrac{x - \mu}{b}, & x \geq \mu.
          \end{cases}

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

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
    return _laplace_log_probability(
        value_array,
        location_array,
        scale_array,
        survival=True,
    )


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


def _laplace_log_probability(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    survival: bool,
) -> jax.Array:
    valid_parameters = jnp.isfinite(location) & jnp.isfinite(scale) & (scale > 0)
    safe_location = jnp.where(valid_parameters, location, jnp.zeros_like(location))
    safe_scale = jnp.where(valid_parameters, scale, jnp.ones_like(scale))
    standardized = _standardize(value, safe_location, safe_scale)
    below_location = value < safe_location

    log_two = jnp.asarray(math.log(2), dtype=value.dtype)
    if survival:
        safe_lower_standardized = jnp.where(
            below_location,
            standardized,
            jnp.zeros_like(standardized),
        )
        lower_log_probability = jnp.log1p(-0.5 * jnp.exp(safe_lower_standardized))
        upper_log_probability = -log_two - standardized
    else:
        lower_log_probability = standardized - log_two
        safe_upper_standardized = jnp.where(
            below_location,
            jnp.zeros_like(standardized),
            standardized,
        )
        upper_log_probability = jnp.log1p(-0.5 * jnp.exp(-safe_upper_standardized))

    log_probability = jnp.where(
        below_location,
        lower_log_probability,
        upper_log_probability,
    )
    return jnp.where(valid_parameters, log_probability, jnp.nan)


@jax.custom_jvp
def _standardize(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    crosses_zero = ((value < 0) & (location > 0)) | ((value > 0) & (location < 0))

    # Scaling before subtraction keeps opposite-sign values inside the dtype range
    cross_value = jnp.where(crosses_zero, value, jnp.zeros_like(value))
    cross_location = jnp.where(crosses_zero, location, jnp.zeros_like(location))
    cross_zero_standardized = cross_value / scale - cross_location / scale

    direct_value = jnp.where(crosses_zero, jnp.zeros_like(value), value)
    direct_location = jnp.where(crosses_zero, jnp.zeros_like(location), location)
    direct_standardized = (direct_value - direct_location) / scale
    return jnp.where(crosses_zero, cross_zero_standardized, direct_standardized)


@_standardize.defjvp
def _standardize_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, location, scale = primals
    value_tangent, location_tangent, scale_tangent = tangents
    standardized = _standardize(value, location, scale)
    crosses_zero = ((value < 0) & (location > 0)) | ((value > 0) & (location < 0))

    # Opposite-sign values must divide before subtraction, just like the primal path
    cross_tangent = value_tangent / scale - location_tangent / scale - standardized * (scale_tangent / scale)
    direct_tangent = (value_tangent - location_tangent - standardized * scale_tangent) / scale
    return standardized, jnp.where(crosses_zero, cross_tangent, direct_tangent)
