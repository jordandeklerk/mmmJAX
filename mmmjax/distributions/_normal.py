"""Normal distribution functions."""

from typing import cast

import distrax
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
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.

    Examples
    --------
    Evaluate a separate log density for each value under a standard Normal
    distribution:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import normal_logpdf
           ...: values = jnp.array([-1.0, 0.0, 1.0])
           ...: normal_logpdf(values, location=0.0, scale=1.0)
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
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.

    Examples
    --------
    Compute a single log likelihood by summing the log densities of
    independent observations:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import normal
           ...: values = jnp.array([-1.0, 0.0, 1.0])
           ...: normal(values, location=0.0, scale=1.0)

    Differentiate with respect to location, the second argument, and compile
    the gradient function for repeated evaluation:

    .. ipython::

        In [2]: from jax import grad, jit
           ...: location_gradient = jit(grad(normal, argnums=1))
           ...: location_gradient(values, 0.5, 1.0)
    """
    return jnp.sum(normal_logpdf(value, location, scale))


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
    value : array_like
        Values at which to evaluate the cumulative probability.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonfinite location or a nonpositive or nonfinite scale
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds under a standard
    Normal distribution, then convert the results to probabilities:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import normal_logcdf
           ...: thresholds = jnp.array([-1.0, 0.0, 1.0])
           ...: log_prob = normal_logcdf(thresholds, location=0.0, scale=1.0)
           ...: jnp.exp(log_prob)
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
    value : array_like
        Values at which to evaluate the survival probability.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds under a standard
    Normal distribution, then convert the results to probabilities:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import normal_logsf
           ...: thresholds = jnp.array([-1.0, 0.0, 1.0])
           ...: log_prob = normal_logsf(thresholds, location=0.0, scale=1.0)
           ...: jnp.exp(log_prob)
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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location : array_like
        Location of the distribution.
    scale : array_like
        Positive standard deviation of the distribution.
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
    Draw five samples from a standard Normal distribution. The key makes
    the draw reproducible:

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import normal_rng
           ...: key = random.key(0)
           ...: normal_rng(key, location=0.0, scale=1.0, sample_shape=(5,))
    """
    location_array, scale_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
    )

    _random_shape(sample_shape, location_array, scale_array)
    samples = distrax.Normal(location_array, scale_array).sample(
        seed=key,
        sample_shape=sample_shape,
    )

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)


def _normal_logpdf_kernel(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    standardized = _standardize(value, location, scale)
    distribution = distrax.Normal(
        loc=jnp.zeros((), dtype=value.dtype),
        scale=jnp.ones((), dtype=value.dtype),
    )
    return cast(jax.Array, distribution.log_prob(standardized) - jnp.log(scale))


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
    standardized = _standardize(safe_value, safe_location, safe_scale)
    distribution = distrax.Normal(
        loc=jnp.zeros((), dtype=value.dtype),
        scale=jnp.ones((), dtype=value.dtype),
    )
    log_probability = distribution.log_cdf(direction * standardized)

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
