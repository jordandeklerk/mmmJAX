"""Multivariate Normal distribution functions."""

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


def multivariate_normal_logpdf(value: ArrayLike, location: ArrayLike, scale_tril: ArrayLike) -> jax.Array:
    r"""Evaluate the multivariate Normal log density along the final axis.

    The covariance is :math:`\Sigma = LL^\mathsf{T}`, where :math:`L`
    is ``scale_tril``. For event size :math:`K`, the log density is

    .. math::

        \log p(x \mid \mu, L) = -\frac{K}{2}\log(2\pi)
        - \sum_i \log L_{ii} - \frac{1}{2}\|L^{-1}(x-\mu)\|^2.

    Parameters
    ----------
    value : array_like
        Observations with event components along the final axis.
    location : array_like
        Mean vectors with the same final event size as ``value``.
    scale_tril : array_like
        Lower Cholesky factors of the covariance, with positive diagonals.
        Final axes match the event size. Leading parameter axes broadcast.

    Returns
    -------
    jax.Array
        One normalized log density per broadcast batch. Infinite observations
        give ``-inf``. NaN observations or invalid parameters give ``nan``.

    Examples
    --------
    Evaluate two observations with correlated coordinates.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import multivariate_normal_logpdf
           ...: values = jnp.array([[0.0, 1.0], [1.0, 0.5]])
           ...: factor = jnp.array([[1.0, 0.0], [0.5, 1.0]])
           ...: multivariate_normal_logpdf(values, jnp.zeros(2), factor)
    """
    value, location, scale_tril = _promote_inexact(("value", value), ("location", location), ("scale_tril", scale_tril))
    _validate_parameters(location, scale_tril)
    if value.ndim < 1 or value.shape[-1] != location.shape[-1]:
        raise ValueError("value and location must have the same final event size")
    jnp.broadcast_shapes(value.shape[:-1], location.shape[:-1], scale_tril.shape[:-2])

    safe_location, safe_scale, valid = _safe_parameters(location, scale_tril)
    finite_value = jnp.all(jnp.isfinite(value), axis=-1)
    safe_value = jnp.where(finite_value[..., None], value, 0)
    result = tfd.MultivariateNormalTriL(safe_location, safe_scale).log_prob(safe_value)

    result = jnp.where(finite_value, result, -jnp.inf)
    return jnp.where(valid & ~jnp.any(jnp.isnan(value), axis=-1), result, jnp.nan)


def multivariate_normal(value: ArrayLike, location: ArrayLike, scale_tril: ArrayLike) -> jax.Array:
    """Return the scalar sum of multivariate Normal log densities.

    Parameters
    ----------
    value : array_like
        Observations with event components along the final axis.
    location : array_like
        Mean vectors with the same final event size as ``value``.
    scale_tril : array_like
        Lower Cholesky factors of the covariance, with positive diagonals.

    Returns
    -------
    jax.Array
        Scalar sum of the normalized log densities.

    Examples
    --------
    Add a joint Normal prior for two coefficients.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import multivariate_normal
           ...: coefficients = jnp.array([0.2, -0.3])
           ...: factor = jnp.array([[1.0, 0.0], [0.5, 1.0]])
           ...: target = multivariate_normal(coefficients, jnp.zeros(2), factor)
           ...: target
    """
    return jnp.sum(multivariate_normal_logpdf(value, location, scale_tril))


def multivariate_normal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale_tril: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw random multivariate Normal vectors.

    Parameters
    ----------
    key : jax.Array
        JAX random key. Use a fresh key for independent draws.
    location : array_like
        Mean vectors with event components along the final axis.
    scale_tril : array_like
        Lower Cholesky factors of the covariance, with positive diagonals.
        Leading axes broadcast with the location batch.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the batch shape.
        Must be static under JIT.

    Returns
    -------
    jax.Array
        Samples with shape ``sample_shape + batch_shape + (event_size,)``.
        Invalid parameters produce ``nan`` vectors.

    Examples
    --------
    Draw five correlated two-dimensional observations.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from jax import random
           ...: from mmmjax import multivariate_normal_rng
           ...: factor = jnp.array([[1.0, 0.0], [0.5, 1.0]])
           ...: multivariate_normal_rng(
           ...:     random.key(0), jnp.zeros(2), factor, sample_shape=(5,)
           ...: )
    """
    location, scale_tril = _promote_inexact(("location", location), ("scale_tril", scale_tril))
    _validate_parameters(location, scale_tril)
    _random_shape(sample_shape, location[..., 0], scale_tril[..., 0, 0])

    safe_location, safe_scale, valid = _safe_parameters(location, scale_tril)
    result = tfd.MultivariateNormalTriL(safe_location, safe_scale).sample(sample_shape, seed=key)
    return jnp.where(valid[..., None], result, jnp.nan)


def _validate_parameters(location: jax.Array, scale_tril: jax.Array) -> None:
    _validate_cholesky_shape(scale_tril, name="scale_tril")
    if location.ndim < 1 or location.shape[-1] != scale_tril.shape[-1]:
        raise ValueError("location must have a final event size matching scale_tril")
    jnp.broadcast_shapes(location.shape[:-1], scale_tril.shape[:-2])


def _safe_parameters(location: jax.Array, scale_tril: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    valid_location = jnp.all(jnp.isfinite(location), axis=-1)
    valid_scale = _is_lower_cholesky(scale_tril)
    safe_location = jnp.where(valid_location[..., None], location, 0)
    safe_scale = jnp.where(
        valid_scale[..., None, None], scale_tril, jnp.eye(scale_tril.shape[-1], dtype=scale_tril.dtype)
    )
    return safe_location, safe_scale, valid_location & valid_scale
