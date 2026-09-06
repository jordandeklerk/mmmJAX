"""Carryover transformations for marketing inputs."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["geometric_adstock"]


def geometric_adstock(
    media: ArrayLike,
    alpha: ArrayLike,
    *,
    max_lag: int,
    axis: int = 0,
    normalize: bool = True,
) -> jax.Array:
    r"""Apply finite geometric carryover along a time axis.

    For a series :math:`x_t`, retention :math:`0 \leq \alpha \leq 1`,
    and maximum lag :math:`L`, the transformed series is

    .. math::

        a_t = \frac{\sum_{\ell=0}^{L} \alpha^\ell x_{t-\ell}}{Z},
        \qquad
        Z = \begin{cases}
            \sum_{\ell=0}^{L} \alpha^\ell & \text{if normalized}, \\
            1 & \text{otherwise}.
        \end{cases}

    The lag-zero weight is one, including when :math:`\alpha = 0`.
    Values before the start of the supplied series are zero. Normalization
    always uses the full lag window, even near the start of a short series.
    To include observed history, prepend it to ``media`` and slice the
    corresponding periods off the result.

    Parameters
    ----------
    media
        Real-valued array with at least one dimension. Observations along
        ``axis`` must represent equally spaced time periods. Other axes
        identify independent series, such as channels and geographies.
    alpha
        Finite retention parameter between zero and one, inclusive. Its
        shape must broadcast to the shape of ``media`` with the time axis
        removed. For ``media.shape == (time, geo, channel)``, channel-level
        parameters have shape ``(channel,)`` and geo-channel parameters
        have shape ``(geo, channel)``. Use ``jax.vmap`` for additional draws.
    max_lag
        Nonnegative number of previous periods to include. The window has
        ``max_lag + 1`` weights, including the current period.
    axis
        Time axis in ``media``. Defaults to the first axis.
    normalize
        Whether to divide the weights by their sum. Defaults to ``True``.
        ``max_lag``, ``axis``, and ``normalize`` must be static under JIT.

    Returns
    -------
    jax.Array
        Transformed values with the same shape as ``media``. Inputs are
        promoted to a common floating-point dtype of at least float32.
        Invalid retention parameters produce ``nan`` for their series.
    """
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError("max_lag must be a static nonnegative integer")
    if max_lag < 0:
        raise ValueError(f"max_lag must be nonnegative, got {max_lag}")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be a static integer identifying the time axis")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a static boolean")

    arrays = []
    for name, value in (("media", media), ("alpha", alpha)):
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be real numeric and array-like, got {type(value).__name__}") from exc
        if not (
            jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)
            or array.dtype == jnp.bool_
        ):
            raise TypeError(f"{name} must have a real numeric dtype, got {array.dtype}")
        arrays.append(array)

    media_array, alpha_array = arrays
    if media_array.ndim == 0:
        raise ValueError("media must have at least one dimension for time, got a scalar")
    if not -media_array.ndim <= axis < media_array.ndim:
        raise ValueError(f"axis {axis} is out of range for media with {media_array.ndim} dimensions")
    axis %= media_array.ndim

    dtype = jnp.result_type(media_array, alpha_array)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
    media_array = jnp.asarray(media_array, dtype=dtype)
    alpha_array = jnp.asarray(alpha_array, dtype=dtype)

    time_last = jnp.moveaxis(media_array, axis, -1)
    batch_shape = time_last.shape[:-1]
    try:
        alpha_array = jnp.broadcast_to(alpha_array, batch_shape)
    except ValueError as exc:
        raise ValueError(
            f"alpha shape {alpha_array.shape} must broadcast to the non-time media shape {batch_shape}; "
            f"media has shape {media_array.shape} with time axis {axis}"
        ) from exc

    if media_array.size == 0:
        return media_array

    valid_alpha = jnp.isfinite(alpha_array) & (alpha_array >= 0) & (alpha_array <= 1)
    safe_alpha = jnp.where(valid_alpha, alpha_array, 0.0)
    n_times = time_last.shape[-1]

    def convolve_series(series: jax.Array, retention: jax.Array) -> jax.Array:
        # Products keep derivatives at zero retention well-defined without differentiating 0**0
        factors = jnp.concatenate((jnp.ones((1,), dtype=dtype), jnp.full((max_lag,), retention, dtype=dtype)))
        weights = jnp.cumprod(factors)
        if normalize:
            weights = weights / jnp.sum(weights)
        # The leading part of full convolution is causal; 'same' would center the window
        return jnp.convolve(series, weights, mode="full", precision=jax.lax.Precision.HIGHEST)[:n_times]

    transformed = jax.vmap(convolve_series)(time_last.reshape((-1, n_times)), safe_alpha.reshape(-1))
    transformed = jnp.where(valid_alpha[..., None], transformed.reshape(time_last.shape), jnp.nan)
    return jnp.moveaxis(transformed, -1, axis)
