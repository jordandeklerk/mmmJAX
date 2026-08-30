"""HalfNormal distribution functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc, log_ndtr
from jax.typing import ArrayLike

from mmmjax.distributions._normal import _normal_logpdf_kernel, _standardize, normal_rng
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
    location = jnp.asarray(0, dtype=value_array.dtype)
    log_density = _normal_logpdf_kernel(value_array, location, scale_array) + log_two
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
    log_densities = half_normal_logpdf(value, scale)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    scale_array = jnp.asarray(scale)
    valid_scale = jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_scale, log_density, jnp.nan)


def half_normal_logcdf(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    r"""Evaluate the HalfNormal log cumulative distribution function elementwise.

    For value :math:`x \geq 0` and scale :math:`\sigma > 0`, the log
    cumulative probability is

    .. math::

        \log F(x \mid \sigma)
        = \log\operatorname{erf}\left(\frac{x}{\sigma\sqrt{2}}\right),
        \qquad x \geq 0,\; \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Values at or below zero produce ``-inf``. A nonpositive or
        nonfinite scale produces ``nan``.
    """
    value_array, scale_array = _promote_inexact(("value", value), ("scale", scale))
    return _half_normal_logcdf_kernel(value_array, scale_array)


def half_normal_logsf(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    r"""Evaluate the HalfNormal log survival function elementwise.

    For value :math:`x \geq 0` and scale :math:`\sigma > 0`, the log survival
    probability is

    .. math::

        \log \overline{F}(x \mid \sigma)
        = \log\left[2\Phi\left(-\frac{x}{\sigma}\right)\right],
        \qquad x \geq 0,\; \sigma > 0,

    where :math:`\Phi` is the standard Normal cumulative distribution
    function.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        Values below zero produce zero. A nonpositive or nonfinite scale
        produces ``nan``.
    """
    value_array, scale_array = _promote_inexact(("value", value), ("scale", scale))
    return _half_normal_logsf_kernel(value_array, scale_array)


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


def _half_normal_logcdf_kernel(value: jax.Array, scale: jax.Array) -> jax.Array:
    valid_scale = jnp.isfinite(scale) & (scale > 0)
    positive_finite_value = jnp.isfinite(value) & (value > 0)
    safe_value = jnp.where(positive_finite_value, value, jnp.ones_like(value))
    safe_scale = jnp.where(valid_scale, scale, jnp.ones_like(scale))
    standardized = _standardize(safe_value, jnp.zeros_like(safe_value), safe_scale)

    half_log_two_over_pi = jnp.asarray(math.log(2 / math.pi) / 2, dtype=value.dtype)
    # The leading log limit stays finite when erf falls below the dtype's normal range
    small_threshold = jnp.sqrt(jnp.asarray(np.finfo(value.dtype).tiny, dtype=value.dtype))
    small_logcdf = jnp.log(safe_value) - jnp.log(safe_scale) + half_log_two_over_pi

    ordinary_region = (standardized >= small_threshold) & (standardized <= 1)
    ordinary_standardized = jnp.where(ordinary_region, standardized, jnp.ones_like(standardized))
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=value.dtype))
    ordinary_logcdf = jnp.log(erf(ordinary_standardized / sqrt_two))

    upper_standardized = jnp.where(standardized > 1, standardized, jnp.ones_like(standardized))
    upper_logcdf = jnp.log1p(-erfc(upper_standardized / sqrt_two))

    interior_logcdf = jnp.where(
        standardized < small_threshold,
        small_logcdf,
        jnp.where(standardized <= 1, ordinary_logcdf, upper_logcdf),
    )
    supported_logcdf = jnp.where(
        jnp.isposinf(value),
        jnp.zeros_like(value),
        jnp.where(value > 0, interior_logcdf, -jnp.inf),
    )
    supported_logcdf = jnp.where(jnp.isnan(value), jnp.nan, supported_logcdf)
    return jnp.where(valid_scale, supported_logcdf, jnp.nan)


def _half_normal_logsf_kernel(value: jax.Array, scale: jax.Array) -> jax.Array:
    valid_scale = jnp.isfinite(scale) & (scale > 0)
    nonnegative_finite_value = jnp.isfinite(value) & (value >= 0)
    safe_value = jnp.where(nonnegative_finite_value, value, jnp.ones_like(value))
    safe_scale = jnp.where(valid_scale, scale, jnp.ones_like(scale))
    standardized = _standardize(safe_value, jnp.zeros_like(safe_value), safe_scale)

    ordinary_region = standardized <= 1
    ordinary_standardized = jnp.where(ordinary_region, standardized, jnp.ones_like(standardized))
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=value.dtype))
    ordinary_logsf = jnp.log1p(-erf(ordinary_standardized / sqrt_two))

    tail_standardized = jnp.where(ordinary_region, jnp.ones_like(standardized), standardized)
    log_two = jnp.asarray(math.log(2), dtype=value.dtype)
    tail_logsf = log_two + log_ndtr(-tail_standardized)
    interior_logsf = jnp.where(ordinary_region, ordinary_logsf, tail_logsf)

    supported_logsf = jnp.where(
        value < 0,
        jnp.zeros_like(value),
        jnp.where(jnp.isposinf(value), -jnp.inf, interior_logsf),
    )
    supported_logsf = jnp.where(jnp.isnan(value), jnp.nan, supported_logsf)
    return jnp.where(valid_scale, supported_logsf, jnp.nan)
