"""Categorical distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _as_real_array, _promote_inexact


def categorical_logpmf(
    value: ArrayLike,
    probabilities: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Categorical log probability mass.

    For category :math:`k \in \{0, \ldots, K - 1\}` and probability vector
    :math:`\mathbf{p}` on the :math:`K`-simplex, the log probability mass is

    .. math::

        \log p(k \mid \mathbf{p}) = \log(p_k).

    Parameters
    ----------
    value
        Zero-based category indices. Every axis is a batch dimension.
    probabilities
        Category probabilities. The final axis contains the categories and
        every leading axis is a batch dimension.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast batch shape.
        Values outside the integer support produce ``-inf`` and a ``nan``
        value produces ``nan``. A probability vector outside the simplex
        produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    (probability_array,) = _promote_inexact(("probabilities", probabilities))
    if probability_array.ndim == 0:
        raise ValueError("probabilities must include a final Categorical event axis, got shape ()")
    if probability_array.shape[-1] == 0:
        raise ValueError(f"Categorical event size must be positive, got probabilities.shape={probability_array.shape}")

    try:
        batch_shape = jnp.broadcast_shapes(
            value_array.shape,
            probability_array.shape[:-1],
        )
    except ValueError as exc:
        raise ValueError(
            "Categorical batch shapes must be broadcastable, "
            f"got value.shape={value_array.shape} and probabilities.shape={probability_array.shape}"
        ) from exc

    category_count = probability_array.shape[-1]
    value_array = jnp.broadcast_to(value_array, batch_shape)
    probability_array = jnp.broadcast_to(
        probability_array,
        (*batch_shape, category_count),
    )

    tolerance = jnp.asarray(
        1e-8 if jax.dtypes.itemsize_bits(probability_array.dtype) == 64 else 1e-6,
        dtype=probability_array.dtype,
    )
    valid_probability = jnp.all(
        jnp.isfinite(probability_array) & (probability_array >= 0),
        axis=-1,
    ) & (jnp.abs(jnp.sum(probability_array, axis=-1) - 1) <= tolerance)
    safe_probability = jnp.where(
        valid_probability[..., None],
        probability_array,
        jnp.full_like(probability_array, 1 / category_count),
    )

    supported = (
        jnp.isfinite(value_array)
        & (value_array == jnp.floor(value_array))
        & (value_array >= 0)
        & (value_array < category_count)
    )
    safe_index = jnp.where(supported, value_array, jnp.zeros_like(value_array)).astype(jnp.int32)
    selected_probability = jnp.take_along_axis(
        safe_probability,
        safe_index[..., None],
        axis=-1,
    )[..., 0]
    supported_probability = jnp.where(
        supported,
        selected_probability,
        jnp.ones_like(selected_probability),
    )

    log_mass = jnp.where(supported, jnp.log(supported_probability), -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_probability, log_mass, jnp.nan)


def categorical(
    value: ArrayLike,
    probabilities: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Categorical log probability masses.

    Parameters
    ----------
    value
        Zero-based category indices.
    probabilities
        Category probabilities with categories along the final axis.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        broadcast batch dimension.
    """
    return jnp.sum(categorical_logpmf(value, probabilities))
