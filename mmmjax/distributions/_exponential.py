"""Exponential distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def exponential_logpdf(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    r"""Evaluate the Exponential log density elementwise.

    For value :math:`x \in \mathbb{R}` and rate :math:`\lambda > 0`, the log
    density is

    .. math::

        \log p(x \mid \lambda)
        = \begin{cases}
            \log(\lambda) - \lambda x, & x \ge 0, \\
            -\infty, & x < 0,
          \end{cases}
        \qquad \lambda > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values below zero produce ``-inf`` and a nonpositive or nonfinite rate
        produces ``nan``.
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    log_density = jnp.log(rate_array) - rate_array * value_array
    supported_log_density = jnp.where(value_array < 0, -jnp.inf, log_density)

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, supported_log_density, jnp.nan)


def exponential(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    """Return the scalar sum of Exponential log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(exponential_logpdf(value, rate))

    rate_array = jnp.asarray(rate)
    valid_rate = jnp.all(jnp.isfinite(rate_array) & (rate_array > 0))
    return jnp.where(valid_rate, log_density, jnp.nan)


def exponential_rng(
    key: jax.Array,
    rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from an Exponential distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    rate
        Positive rate parameter, equal to the inverse scale.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + rate.shape``. A nonpositive
        or nonfinite rate produces ``nan``.
    """
    (rate_array,) = _promote_inexact(("rate", rate))

    shape = _random_shape(sample_shape, rate_array)
    standard_exponential = jax.random.exponential(key, shape=shape, dtype=rate_array.dtype)
    samples = standard_exponential / rate_array

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, samples, jnp.nan)
