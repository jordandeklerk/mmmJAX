"""Exponential distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import math as tfm

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
    value : array_like
        Values at which to evaluate the density.
    rate : array_like
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values below zero produce ``-inf`` and a nonpositive or nonfinite rate
        produces ``nan``.

    Examples
    --------
    Evaluate one log density per value. The rate is the reciprocal of the mean.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import exponential_logpdf
           ...: values = jnp.array([0.5, 1.0, 2.0])
           ...: exponential_logpdf(values, rate=2.0)

    Broadcasting evaluates all three values at two rates. The extra axis
    keeps values in rows and rates in columns:

    .. ipython::

        In [2]: rates = jnp.array([1.0, 2.0])
           ...: exponential_logpdf(values[:, None], rate=rates)
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    log_density = tfd.Exponential(rate=rate_array).log_prob(value_array)
    supported_log_density = jnp.where(value_array < 0, -jnp.inf, log_density)

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, supported_log_density, jnp.nan)


def exponential(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    """Return the scalar sum of Exponential log densities.

    Parameters
    ----------
    value : array_like
        Values at which to evaluate the density.
    rate : array_like
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.

    Examples
    --------
    Sum the log densities of independent observations into a single log
    likelihood. The rate is the reciprocal of the mean.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import exponential
           ...: values = jnp.array([0.5, 1.0, 2.0])
           ...: exponential(values, rate=2.0)
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
    value : array_like
        Values at which to evaluate the cumulative probability.
    rate : array_like
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonpositive or nonfinite rate produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import exponential_logcdf
           ...: thresholds = jnp.array([0.5, 1.0, 2.0])
           ...: log_prob = exponential_logcdf(thresholds, rate=2.0)
           ...: jnp.exp(log_prob)
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    lower_boundary = value_array <= 0
    upper_boundary = jnp.isposinf(value_array)
    evaluate_probability = valid_rate & ~lower_boundary & ~upper_boundary
    safe_value = jnp.where(evaluate_probability, value_array, jnp.ones_like(value_array))
    safe_rate = jnp.where(evaluate_probability, rate_array, jnp.ones_like(rate_array))
    scaled_value = safe_rate * safe_value
    use_direct = scaled_value < jnp.log(jnp.asarray(2, dtype=value_array.dtype))
    direct_value = jnp.where(use_direct, scaled_value, jnp.ones_like(scaled_value))
    tail_value = jnp.where(use_direct, jnp.ones_like(scaled_value), scaled_value)

    distribution = tfd.Exponential(rate=jnp.ones((), dtype=value_array.dtype))
    direct_log_cdf = distribution.log_cdf(direct_value)
    tail_log_cdf = tfm.log1mexp(-tail_value)
    log_cdf = jnp.where(use_direct, direct_log_cdf, tail_log_cdf)

    boundary_log_cdf = jnp.where(lower_boundary, -jnp.inf, 0)
    supported_log_cdf = jnp.where(evaluate_probability, log_cdf, boundary_log_cdf)
    return jnp.where(valid_rate & ~jnp.isnan(value_array), supported_log_cdf, jnp.nan)


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
    value : array_like
        Values at which to evaluate the survival probability.
    rate : array_like
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonpositive or nonfinite rate produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import exponential_logsf
           ...: thresholds = jnp.array([0.5, 1.0, 2.0])
           ...: log_prob = exponential_logsf(thresholds, rate=2.0)
           ...: jnp.exp(log_prob)
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    lower_boundary = value_array <= 0
    upper_boundary = jnp.isposinf(value_array)
    evaluate_probability = valid_rate & ~lower_boundary & ~upper_boundary
    safe_value = jnp.where(evaluate_probability, value_array, jnp.zeros_like(value_array))
    safe_rate = jnp.where(evaluate_probability, rate_array, jnp.ones_like(rate_array))
    # The exact survival formula avoids cancellation of density normalizers near zero
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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    rate : array_like
        Positive rate parameter, equal to the inverse scale.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + rate.shape``. A nonpositive
        or nonfinite rate produces ``nan``.

    Examples
    --------
    Draw five samples using a key to make the draw reproducible.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import exponential_rng
           ...: key = random.key(0)
           ...: exponential_rng(key, rate=2.0, sample_shape=(5,))
    """
    (rate_array,) = _promote_inexact(("rate", rate))

    _random_shape(sample_shape, rate_array)
    samples = tfd.Exponential(rate=rate_array).sample(seed=key, sample_shape=sample_shape)

    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, samples, jnp.nan)
