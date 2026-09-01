"""Dirichlet distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact


def dirichlet_logpdf(
    value: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Dirichlet log density along the final axis.

    For a :math:`K`-component simplex :math:`\mathbf{x}` and positive
    concentration vector :math:`\boldsymbol{\alpha}`, the log density is

    .. math::

        \log p(\mathbf{x} \mid \boldsymbol{\alpha})
        = \log\Gamma\!\left(\sum_{i=1}^{K}\alpha_i\right)
          - \sum_{i=1}^{K}\log\Gamma(\alpha_i)
          + \sum_{i=1}^{K}(\alpha_i - 1)\log(x_i).

    Parameters
    ----------
    value
        Simplex values. The final axis contains the components and every
        leading axis is a batch dimension.
    concentration
        Positive concentration parameters. The final axis must match the
        event size of ``value`` and leading axes are broadcast as batches.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast batch shape. Values
        outside the simplex produce ``-inf`` and a ``nan`` value produces
        ``nan``. A nonpositive or nonfinite concentration produces ``nan``.

        At a zero-valued component, the corresponding density term is
        ``inf``, zero, or ``-inf`` when its concentration is below, equal to,
        or above one. Opposing infinite terms produce ``nan`` because the
        multivariate boundary limit depends on the path of approach.
    """
    value_array, concentration_array = _promote_inexact(
        ("value", value),
        ("concentration", concentration),
    )
    if value_array.ndim == 0:
        raise ValueError("value must include a final Dirichlet event axis, got shape ()")
    if concentration_array.ndim == 0:
        raise ValueError("concentration must include a final Dirichlet event axis, got shape ()")
    if value_array.shape[-1] == 0 or concentration_array.shape[-1] == 0:
        raise ValueError(
            "Dirichlet event size must be positive, "
            f"got value.shape={value_array.shape} and concentration.shape={concentration_array.shape}"
        )
    if value_array.shape[-1] != concentration_array.shape[-1]:
        raise ValueError(
            "value and concentration must have the same final event size, "
            f"got {value_array.shape[-1]} and {concentration_array.shape[-1]}"
        )

    try:
        batch_shape = jnp.broadcast_shapes(value_array.shape[:-1], concentration_array.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "Dirichlet batch shapes must be broadcastable, "
            f"got value.shape={value_array.shape} and concentration.shape={concentration_array.shape}"
        ) from exc

    event_size = value_array.shape[-1]
    output_shape = (*batch_shape, event_size)
    value_array = jnp.broadcast_to(value_array, output_shape)
    concentration_array = jnp.broadcast_to(concentration_array, output_shape)

    valid_concentration = jnp.all(jnp.isfinite(concentration_array) & (concentration_array > 0), axis=-1)
    finite_nonnegative_value = jnp.all(jnp.isfinite(value_array) & (value_array >= 0), axis=-1)
    tolerance = jnp.asarray(
        1e-8 if jax.dtypes.itemsize_bits(value_array.dtype) == 64 else 1e-6,
        dtype=value_array.dtype,
    )
    on_simplex = finite_nonnegative_value & (jnp.abs(jnp.sum(value_array, axis=-1) - 1) <= tolerance)

    safe_event = valid_concentration & on_simplex
    safe_concentration = jnp.where(safe_event[..., None], concentration_array, jnp.ones_like(concentration_array))
    safe_value = jnp.where(
        safe_event[..., None],
        value_array,
        jnp.full_like(value_array, 1 / event_size),
    )

    concentration_sum = jnp.sum(safe_concentration, axis=-1)
    log_density = (
        gammaln(concentration_sum)
        - jnp.sum(gammaln(safe_concentration), axis=-1)
        + jnp.sum(xlogy(safe_concentration - 1, safe_value), axis=-1)
    )

    supported_log_density = jnp.where(on_simplex, log_density, -jnp.inf)
    supported_log_density = jnp.where(jnp.any(jnp.isnan(value_array), axis=-1), jnp.nan, supported_log_density)
    return jnp.where(valid_concentration, supported_log_density, jnp.nan)


def dirichlet(
    value: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Dirichlet log densities.

    Parameters
    ----------
    value
        Simplex values with event components along the final axis.
    concentration
        Positive concentration parameters with the same final event size.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every broadcast batch dimension.
    """
    return jnp.sum(dirichlet_logpdf(value, concentration))
