"""Exponential distribution functions."""

import math

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
    return jnp.sum(exponential_logpdf(value, rate))


def exponential_logcdf(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    r"""Evaluate the Exponential log cumulative distribution function elementwise.

    For value :math:`x \in \mathbb{R}` and rate :math:`\lambda > 0`, the log
    cumulative probability is

    .. math::

        \log F(x \mid \lambda)
        = \begin{cases}
            \log\left(1 - \exp(-\lambda x)\right), & x > 0, \\
            -\infty, & x \leq 0,
          \end{cases}
        \qquad \lambda > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonpositive or nonfinite rate produces ``nan``.
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    lower_boundary = value_array <= 0
    upper_boundary = jnp.isposinf(value_array)
    evaluate_probability = valid_rate & ~lower_boundary & ~upper_boundary
    safe_value = jnp.where(evaluate_probability, value_array, jnp.ones_like(value_array))
    safe_rate = jnp.where(evaluate_probability, rate_array, jnp.ones_like(rate_array))
    scaled_value = safe_rate * safe_value

    # Each form keeps precision in one tail, and safe inputs stop the unused branch from leaking NaN derivatives
    log_two = jnp.asarray(math.log(2), dtype=scaled_value.dtype)
    use_expm1 = scaled_value < log_two
    expm1_input = jnp.where(use_expm1, scaled_value, jnp.ones_like(scaled_value))
    log1p_input = jnp.where(use_expm1, jnp.ones_like(scaled_value), scaled_value)
    near_zero = jnp.log(-jnp.expm1(-expm1_input))
    upper_tail = jnp.log1p(-jnp.exp(-log1p_input))
    log_cdf = jnp.where(use_expm1, near_zero, upper_tail)

    boundary_log_cdf = jnp.where(lower_boundary, -jnp.inf, 0)
    supported_log_cdf = jnp.where(evaluate_probability, log_cdf, boundary_log_cdf)
    return jnp.where(valid_rate, supported_log_cdf, jnp.nan)


def exponential_logsf(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    r"""Evaluate the Exponential log survival function elementwise.

    For value :math:`x \in \mathbb{R}` and rate :math:`\lambda > 0`, the log
    survival probability is

    .. math::

        \log \overline{F}(x \mid \lambda)
        = \begin{cases}
            -\lambda x, & x > 0, \\
            0, & x \leq 0,
          \end{cases}
        \qquad \lambda > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonpositive or nonfinite rate produces ``nan``.
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    lower_boundary = value_array <= 0
    upper_boundary = jnp.isposinf(value_array)
    evaluate_probability = valid_rate & ~lower_boundary & ~upper_boundary
    safe_value = jnp.where(evaluate_probability, value_array, jnp.zeros_like(value_array))
    safe_rate = jnp.where(evaluate_probability, rate_array, jnp.ones_like(rate_array))
    log_survival = -safe_rate * safe_value

    boundary_log_survival = jnp.where(lower_boundary, 0, -jnp.inf)
    supported_log_survival = jnp.where(evaluate_probability, log_survival, boundary_log_survival)
    return jnp.where(valid_rate, supported_log_survival, jnp.nan)


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
