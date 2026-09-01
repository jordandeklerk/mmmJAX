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
    value_array, probability_array, supported, safe_index = _prepare_categorical_inputs(
        value,
        probabilities,
        parameter_name="probabilities",
    )
    category_count = probability_array.shape[-1]
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


def categorical_logit_logpmf(
    value: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    r"""Evaluate the logit-parameterized Categorical log probability mass.

    For category :math:`k \in \{0, \ldots, K - 1\}` and unnormalized log
    probabilities :math:`\boldsymbol{\eta}`, the log probability mass is

    .. math::

        \log p(k \mid \boldsymbol{\eta})
        = \eta_k - \log\!\left(\sum_{j=0}^{K-1} e^{\eta_j}\right).

    Parameters
    ----------
    value
        Zero-based category indices. Every axis is a batch dimension.
    logits
        Unnormalized category log probabilities. The final axis contains the
        categories and every leading axis is a batch dimension. A ``-inf``
        logit masks that category when another category has finite weight.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast batch shape.
        Values outside the integer support produce ``-inf`` and a ``nan``
        value produces ``nan``. A logit event containing ``nan`` or ``+inf``,
        or containing no finite logit, produces ``nan``.
    """
    value_array, logits_array, supported, safe_index = _prepare_categorical_inputs(
        value,
        logits,
        parameter_name="logits",
    )
    has_undefined_logit = jnp.any(
        jnp.isnan(logits_array) | jnp.isposinf(logits_array),
        axis=-1,
    )
    has_finite_logit = jnp.any(jnp.isfinite(logits_array), axis=-1)
    valid_logits = ~has_undefined_logit & has_finite_logit
    safe_logits = jnp.where(
        valid_logits[..., None],
        logits_array,
        jnp.zeros_like(logits_array),
    )
    log_probabilities = jax.nn.log_softmax(safe_logits, axis=-1)

    selected_log_probability = jnp.take_along_axis(
        log_probabilities,
        safe_index[..., None],
        axis=-1,
    )[..., 0]
    supported_log_probability = jnp.where(
        supported,
        selected_log_probability,
        jnp.zeros_like(selected_log_probability),
    )

    log_mass = jnp.where(supported, supported_log_probability, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_logits, log_mass, jnp.nan)


def categorical_logit(
    value: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of logit-parameterized Categorical log masses.

    Parameters
    ----------
    value
        Zero-based category indices.
    logits
        Unnormalized category log probabilities with categories along the
        final axis.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        broadcast batch dimension.
    """
    return jnp.sum(categorical_logit_logpmf(value, logits))


def _prepare_categorical_inputs(
    value: ArrayLike,
    parameters: ArrayLike,
    *,
    parameter_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    value_array = _as_real_array("value", value)
    (parameter_array,) = _promote_inexact((parameter_name, parameters))
    if parameter_array.ndim == 0:
        raise ValueError(f"{parameter_name} must include a final Categorical event axis, got shape ()")
    if parameter_array.shape[-1] == 0:
        raise ValueError(f"Categorical event size must be positive, got {parameter_name}.shape={parameter_array.shape}")

    try:
        batch_shape = jnp.broadcast_shapes(
            value_array.shape,
            parameter_array.shape[:-1],
        )
    except ValueError as exc:
        raise ValueError(
            "Categorical batch shapes must be broadcastable, "
            f"got value.shape={value_array.shape} and {parameter_name}.shape={parameter_array.shape}"
        ) from exc

    category_count = parameter_array.shape[-1]
    value_array = jnp.broadcast_to(value_array, batch_shape)
    parameter_array = jnp.broadcast_to(
        parameter_array,
        (*batch_shape, category_count),
    )
    supported = (
        jnp.isfinite(value_array)
        & (value_array == jnp.floor(value_array))
        & (value_array >= 0)
        & (value_array < category_count)
    )
    safe_index = jnp.where(
        supported,
        value_array,
        jnp.zeros_like(value_array),
    ).astype(jnp.int32)
    return value_array, parameter_array, supported, safe_index
