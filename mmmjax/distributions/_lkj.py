"""LKJ correlation Cholesky distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import distributions as tfd

from mmmjax.distributions._utils import (
    _is_lower_cholesky,
    _promote_inexact,
    _random_shape,
    _validate_cholesky_shape,
)


def lkj_cholesky_logpdf(value: ArrayLike, concentration: ArrayLike) -> jax.Array:
    r"""Evaluate the normalized LKJ density in Cholesky-factor space.

    For a correlation factor :math:`L` of size :math:`K` and concentration
    :math:`\eta > 0`, the density is proportional to
    :math:`\prod_{i=2}^{K} L_{ii}^{K-i+2\eta-2}`.

    Parameters
    ----------
    value : array_like
        Lower Cholesky factors of correlation matrices. The final two axes
        are square, with positive diagonals and unit-length rows.
    concentration : array_like
        Positive concentration. One gives a uniform distribution over
        correlation matrices. Larger values favor weaker correlations.
        Broadcasts with the leading axes of ``value``.

    Returns
    -------
    jax.Array
        One log density per broadcast batch. Invalid factors give ``-inf``.
        NaN factors or nonpositive or nonfinite concentrations give ``nan``.

    Examples
    --------
    Evaluate the log density of a correlation factor.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lkj_cholesky_logpdf
           ...: correlation = jnp.array([[1.0, 0.4], [0.4, 1.0]])
           ...: factor = jnp.linalg.cholesky(correlation)
           ...: lkj_cholesky_logpdf(factor, concentration=2.0)
    """
    value, concentration = _promote_inexact(("value", value), ("concentration", concentration))
    _validate_cholesky_shape(value, name="value")
    jnp.broadcast_shapes(value.shape[:-2], concentration.shape)

    tolerance = 32 * jnp.finfo(value.dtype).eps
    unit_rows = jnp.all(jnp.abs(jnp.sum(value**2, axis=-1) - 1) <= tolerance, axis=-1)
    supported = _is_lower_cholesky(value) & unit_rows
    valid_concentration = jnp.isfinite(concentration) & (concentration > 0)

    safe_value = jnp.where(supported[..., None, None], value, jnp.eye(value.shape[-1], dtype=value.dtype))
    safe_concentration = jnp.where(valid_concentration, concentration, 1)
    result = tfd.CholeskyLKJ(value.shape[-1], safe_concentration).log_prob(safe_value)

    result = jnp.where(supported, result, -jnp.inf)
    invalid = ~valid_concentration | jnp.any(jnp.isnan(value), axis=(-2, -1))
    return jnp.where(invalid, jnp.nan, result)


def lkj_cholesky(value: ArrayLike, concentration: ArrayLike) -> jax.Array:
    """Return the scalar sum of LKJ Cholesky log densities.

    Parameters
    ----------
    value : array_like
        Correlation Cholesky factors along the final two axes.
    concentration : array_like
        Positive concentration, broadcast over the factor batch.

    Returns
    -------
    jax.Array
        Scalar sum of the normalized log densities.

    Examples
    --------
    Add an LKJ prior to a model's log density.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import lkj_cholesky
           ...: factor = jnp.eye(3)
           ...: target = lkj_cholesky(factor, concentration=2.0)
           ...: target
    """
    return jnp.sum(lkj_cholesky_logpdf(value, concentration))


def lkj_cholesky_rng(
    key: jax.Array,
    dimension: int,
    concentration: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw random correlation Cholesky factors.

    Parameters
    ----------
    key : jax.Array
        JAX random key. Use a fresh key for independent draws.
    dimension : int
        Positive number of rows and columns. Must be static under JIT.
    concentration : array_like
        Positive concentration. Its shape defines the batch dimensions.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the batch shape.
        Must be static under JIT.

    Returns
    -------
    jax.Array
        Factors with shape ``sample_shape + batch_shape + (dimension, dimension)``.
        Invalid concentrations produce ``nan`` factors.

    Examples
    --------
    Draw a correlation factor for three effects.

    .. ipython::
        :okwarning:

        In [1]: from jax import random
           ...: from mmmjax import lkj_cholesky_rng
           ...: factor = lkj_cholesky_rng(random.key(0), 3, concentration=2.0)
           ...: factor @ factor.T
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be a positive integer")
    if dimension < 1:
        raise ValueError("dimension must be positive")

    (concentration,) = _promote_inexact(("concentration", concentration))
    _random_shape(sample_shape, concentration)
    valid = jnp.isfinite(concentration) & (concentration > 0)
    safe_concentration = jnp.where(valid, concentration, 1)
    result = tfd.CholeskyLKJ(dimension, safe_concentration).sample(sample_shape, seed=key)
    return jnp.where(valid[..., None, None], result, jnp.nan)
