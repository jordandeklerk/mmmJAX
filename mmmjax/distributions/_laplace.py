"""Laplace distribution functions."""

import distrax
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
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive scale parameter. The standard deviation is
        :math:`\sqrt{2}b`.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        The value and location gradients use a zero subgradient when
        ``value == location``. A nonfinite location or a nonpositive or
        nonfinite scale produces ``nan``.

    Examples
    --------
    Evaluate one log density per value.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import laplace_logpdf
           ...: values = jnp.array([-1.0, 0.0, 1.0])
           ...: laplace_logpdf(values, location=0.0, scale=1.0)
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)

    standardized = _standardize(value_array, location_array, scale_array)
    # The constant branch gives a symmetric zero subgradient at the cusp
    standardized_distance = jnp.where(
        value_array == location_array,
        jnp.zeros_like(standardized),
        jnp.where(value_array < location_array, -standardized, standardized),
    )

    distribution = distrax.Laplace(
        loc=jnp.zeros((), dtype=value_array.dtype),
        scale=jnp.ones((), dtype=value_array.dtype),
    )
    log_density = distribution.log_prob(standardized_distance) - jnp.log(scale_array)
    return jnp.where(valid_location & valid_scale, log_density, jnp.nan)


def laplace(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Laplace log densities.

    Parameters
    ----------
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.

    Examples
    --------
    Sum the log densities of independent observations into a single log
    likelihood.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import laplace
           ...: values = jnp.array([-1.0, 0.0, 1.0])
           ...: laplace(values, location=0.0, scale=1.0)
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
    value : array_like
        Values at which to evaluate the cumulative probability.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonfinite location or a nonpositive or nonfinite scale
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import laplace_logcdf
           ...: thresholds = jnp.array([-1.0, 0.0, 1.0])
           ...: log_prob = laplace_logcdf(thresholds, location=0.0, scale=1.0)
           ...: jnp.exp(log_prob)
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
    value : array_like
        Values at which to evaluate the survival probability.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import laplace_logsf
           ...: thresholds = jnp.array([-1.0, 0.0, 1.0])
           ...: log_prob = laplace_logsf(thresholds, location=0.0, scale=1.0)
           ...: jnp.exp(log_prob)
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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive scale parameter.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.

    Examples
    --------
    Draw five samples using a key to make the draw reproducible.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import laplace_rng
           ...: key = random.key(0)
           ...: laplace_rng(key, location=0.0, scale=1.0, sample_shape=(5,))
    """
    location_array, scale_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
    )
    _random_shape(sample_shape, location_array, scale_array)
    samples = distrax.Laplace(location_array, scale_array).sample(
        seed=key,
        sample_shape=sample_shape,
    )

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
    standardized = _standardize(value, location, scale)
    distribution = distrax.Laplace(
        loc=jnp.zeros((), dtype=value.dtype),
        scale=jnp.ones((), dtype=value.dtype),
    )
    log_probability = (
        distribution.log_survival_function(standardized) if survival else distribution.log_cdf(standardized)
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
