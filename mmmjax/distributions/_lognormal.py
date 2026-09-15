"""LogNormal distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import distributions as tfd

from mmmjax.distributions._distribution import _bind_distribution
from mmmjax.distributions._normal import normal_logcdf, normal_logpdf, normal_logsf
from mmmjax.distributions._utils import _promote_inexact, _random_shape


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
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Mean of the underlying Normal distribution for ``log(value)``.
    scale : array_like
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values at or below zero produce ``-inf``. A nonfinite location or a
        nonpositive or nonfinite scale produces ``nan``.

    Examples
    --------
    Evaluate a separate log density for each positive value. Here, the
    logarithms follow a Normal distribution with mean zero and standard
    deviation 0.5:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lognormal_logpdf
           ...: values = jnp.array([0.5, 1.0, 2.0])
           ...: lognormal_logpdf(values, location=0.0, scale=0.5)
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
    value : array_like
        Values at which to evaluate the density.
    location : array_like
        Mean of the underlying Normal distribution for ``log(value)``.
    scale : array_like
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.

    Examples
    --------
    Compute a single log likelihood by summing the log densities of
    independent positive observations:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lognormal
           ...: values = jnp.array([0.5, 1.0, 2.0])
           ...: lognormal(values, location=0.0, scale=0.5)
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
    value : array_like
        Values at which to evaluate the cumulative probability.
    location : array_like
        Mean of the underlying Normal distribution for ``log(value)``.
    scale : array_like
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonfinite location or a nonpositive or nonfinite scale
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three positive thresholds, then
    convert the results to probabilities:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lognormal_logcdf
           ...: thresholds = jnp.array([0.5, 1.0, 2.0])
           ...: log_prob = lognormal_logcdf(thresholds, location=0.0, scale=0.5)
           ...: jnp.exp(log_prob)
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
    value : array_like
        Values at which to evaluate the survival probability.
    location : array_like
        Mean of the underlying Normal distribution for ``log(value)``.
    scale : array_like
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three positive thresholds, then
    convert the results to probabilities:

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lognormal_logsf
           ...: thresholds = jnp.array([0.5, 1.0, 2.0])
           ...: log_prob = lognormal_logsf(thresholds, location=0.0, scale=0.5)
           ...: jnp.exp(log_prob)
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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location : array_like
        Mean of the underlying Normal distribution for ``log(value)``.
    scale : array_like
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.
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
    Draw five positive samples. The location and scale describe the
    underlying Normal distribution, not the samples themselves:

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import lognormal_rng
           ...: key = random.key(0)
           ...: lognormal_rng(key, location=0.0, scale=0.5, sample_shape=(5,))
    """
    location_array, scale_array = _promote_inexact(("location", location), ("scale", scale))
    _random_shape(sample_shape, location_array, scale_array)
    samples = tfd.LogNormal(loc=location_array, scale=scale_array).sample(seed=key, sample_shape=sample_shape)

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)


def _lognormal_log_probability(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
    *,
    direction: int,
) -> jax.Array:
    valid_parameters = jnp.isfinite(location) & jnp.isfinite(scale) & (scale > 0)
    supported_boundary = ((value <= 0) | jnp.isposinf(value)) & valid_parameters
    safe_value = jnp.where(supported_boundary, jnp.ones_like(value), value)
    safe_location = jnp.where(supported_boundary, jnp.zeros_like(location), location)
    safe_scale = jnp.where(supported_boundary, jnp.ones_like(scale), scale)
    log_value = jnp.log(safe_value)
    log_probability = (
        normal_logcdf(log_value, safe_location, safe_scale)
        if direction == 1
        else normal_logsf(log_value, safe_location, safe_scale)
    )
    boundary_probability = jnp.where(value <= 0, -jnp.inf, 0) if direction == 1 else jnp.where(value <= 0, 0, -jnp.inf)
    supported_log_probability = jnp.where(supported_boundary, boundary_probability, log_probability)
    return jnp.where(valid_parameters, supported_log_probability, jnp.nan)


_bind_distribution(lognormal, lognormal_logpdf, lognormal_rng, tfd.LogNormal, location="loc", scale="scale")
