"""HalfNormal distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._normal import normal_logpdf, normal_rng
from mmmjax.distributions._utils import _promote_inexact


def half_normal_logpdf(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    r"""Evaluate the HalfNormal log density elementwise.

    For value :math:`x \geq 0` and scale :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \sigma)
        = \frac{1}{2}\log\left(\frac{2}{\pi}\right)
          - \log(\sigma)
          - \frac{1}{2}\left(\frac{x}{\sigma}\right)^2,
        \qquad x \geq 0,\; \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values below zero produce ``-inf`` and a nonpositive or nonfinite scale
        produces ``nan``.
    """
    value_array, scale_array = _promote_inexact(("value", value), ("scale", scale))

    log_two = jnp.asarray(math.log(2), dtype=value_array.dtype)
    log_density = normal_logpdf(value_array, 0, scale_array) + log_two
    supported_log_density = jnp.where(value_array < 0, -jnp.inf, log_density)

    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_scale, supported_log_density, jnp.nan)


def half_normal(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    """Return the scalar sum of HalfNormal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(half_normal_logpdf(value, scale))

    scale_array = jnp.asarray(scale)
    valid_scale = jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_scale, log_density, jnp.nan)


def half_normal_rng(
    key: jax.Array,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a HalfNormal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + scale.shape``. A
        nonpositive or nonfinite scale produces ``nan``.
    """
    return jnp.abs(normal_rng(key, 0, scale, sample_shape=sample_shape))
