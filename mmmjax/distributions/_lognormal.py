"""LogNormal distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._normal import (
    _normal_log_probability,
    _normal_logpdf_kernel,
    normal_rng,
)
from mmmjax.distributions._utils import _promote_inexact


def lognormal_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the LogNormal log density elementwise.

    For value :math:`x > 0`, log-scale location :math:`\mu \in \mathbb{R}`,
    and log-scale standard deviation :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \mu, \sigma)
        = -\frac{1}{2}\left(\frac{\log(x) - \mu}{\sigma}\right)^2
          - \log(\sigma)
          - \log(x)
          - \frac{1}{2}\log(2\pi),
        \qquad x > 0,\; \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values at or below zero produce ``-inf``. A nonfinite location or a
        nonpositive or nonfinite scale produces ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    outside_support = value_array <= 0
    # Avoid an indeterminate expression at zero without changing NaN inputs
    safe_value = jnp.where(outside_support, jnp.ones_like(value_array), value_array)
    log_value = jnp.log(safe_value)
    log_density = _normal_logpdf_kernel(log_value, location_array, scale_array) - log_value
    supported_log_density = jnp.where(outside_support, -jnp.inf, log_density)

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def lognormal(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of LogNormal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    return jnp.sum(lognormal_logpdf(value, location, scale))


def lognormal_logcdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the LogNormal log cumulative distribution function elementwise.

    For value :math:`x > 0`, log-scale location :math:`\mu \in \mathbb{R}`,
    and log-scale standard deviation :math:`\sigma > 0`, the log cumulative
    probability is

    .. math::

        \log F(x \mid \mu, \sigma)
        = \log \Phi\left(\frac{\log(x) - \mu}{\sigma}\right),
        \qquad x > 0,\; \sigma > 0,

    where :math:`\Phi` is the standard Normal cumulative distribution
    function. For :math:`x \leq 0`, the cumulative probability is zero and
    its logarithm is :math:`-\infty`.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

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

    return _lognormal_log_probability(value_array, location_array, scale_array, direction=1)


def lognormal_logsf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the LogNormal log survival function elementwise.

    For value :math:`x > 0`, log-scale location :math:`\mu \in \mathbb{R}`,
    and log-scale standard deviation :math:`\sigma > 0`, the log survival
    probability is

    .. math::

        \log \overline{F}(x \mid \mu, \sigma)
        = \log \Phi\left(\frac{\mu - \log(x)}{\sigma}\right),
        \qquad x > 0,\; \sigma > 0,

    where :math:`\Phi` is the standard Normal cumulative distribution
    function. For :math:`x \leq 0`, the survival probability is one and its
    logarithm is zero.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

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

    return _lognormal_log_probability(value_array, location_array, scale_array, direction=-1)


def lognormal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a LogNormal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.
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
    return jnp.exp(normal_rng(key, location, scale, sample_shape=sample_shape))


def _lognormal_log_probability(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    direction: int,
) -> jax.Array:
    valid_parameters = jnp.isfinite(location) & jnp.isfinite(scale) & (scale > 0)
    supported_boundary = (value <= 0) & valid_parameters
    safe_value = jnp.where(supported_boundary, jnp.ones_like(value), value)
    safe_location = jnp.where(supported_boundary, jnp.zeros_like(location), location)
    safe_scale = jnp.where(supported_boundary, jnp.ones_like(scale), scale)
    log_probability = _normal_log_probability(
        jnp.log(safe_value),
        safe_location,
        safe_scale,
        direction=direction,
    )
    boundary_probability = -jnp.inf if direction == 1 else 0
    return jnp.where(supported_boundary, boundary_probability, log_probability)
