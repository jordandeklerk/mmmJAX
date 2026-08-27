"""LogNormal distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._normal import normal_logpdf, normal_rng
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
    log_density = normal_logpdf(log_value, location_array, scale_array) - log_value
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
    log_density = jnp.sum(lognormal_logpdf(value, location, scale))
    location_array = jnp.asarray(location)
    scale_array = jnp.asarray(scale)
    valid_parameters = jnp.all(jnp.isfinite(location_array)) & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_parameters, log_density, jnp.nan)


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
