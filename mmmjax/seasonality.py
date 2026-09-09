"""Seasonal features for recurring patterns over time."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["fourier_features"]


def fourier_features(time: ArrayLike, *, period: ArrayLike, order: int) -> jax.Array:
    r"""Build sine and cosine features for a repeating seasonal pattern.

    For time positions :math:`t`, cycle length :math:`P > 0`, and harmonics
    :math:`k = 1, \ldots, K`, the features are

    .. math::

        \sin\left(\frac{2\pi kt}{P}\right), \qquad
        \cos\left(\frac{2\pi kt}{P}\right).

    Multiply the features by model coefficients to obtain a seasonal
    contribution. Coefficients and their priors are specified separately.

    Parameters
    ----------
    time : array_like
        One-dimensional, finite numeric positions for the modeling periods.
        For dated observations, use elapsed time from a fixed reference date.
        Keep that reference date and the time units unchanged for prediction.
        Negative and fractional positions are accepted.
    period : array_like
        Positive, finite scalar giving the cycle length in the same units as
        ``time``. For example, use 7 for weekly seasonality when time is in
        days, or 365.25 for an annual cycle measured in days.
    order : int
        Positive number of harmonics. Each adds a sine and a cosine term.
        Higher orders allow more detailed patterns within a cycle. Keep
        this argument static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Features with shape ``(len(time), 2 * order)``. Sine terms come first
        in increasing harmonic order, followed by cosine terms in the same
        order. No intercept column is included. Values use a common floating
        dtype of at least float32. Invalid numeric inputs produce ``nan`` in
        the affected rows.

        A coefficient vector of shape ``(2 * order,)`` gives one shared
        seasonal curve through ``features @ coefficients``. Coefficients
        shaped ``(2 * order, group)`` give separate group curves without
        duplicating the features.

    Examples
    --------
    Build annual features from weekly dates and combine them with illustrative
    coefficients. Reuse ``origin`` when preparing later dates for prediction.

    .. ipython::

        In [1]: from datetime import date
           ...: import jax.numpy as jnp
           ...: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import fourier_features
           ...: frame = pl.DataFrame({
           ...:     "week": [date(2026, 1, 1), date(2026, 1, 8),
           ...:              date(2026, 1, 15)],
           ...:     "sales": [100.0, 120.0, 110.0],
           ...: })
           ...: origin = date(2026, 1, 1)
           ...: days = (frame["week"] - origin).dt.total_days().to_numpy()
           ...: features = fourier_features(days, period=365.25, order=2)
           ...: coefficients = jnp.array([0.2, -0.1, 0.3, 0.1])
           ...: seasonal = features @ coefficients
           ...: frame.with_columns(
           ...:     pl.Series("seasonality", np.asarray(seasonal)),
           ...: )
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be a positive Python integer and stay fixed during JIT compilation")
    if order <= 0:
        raise ValueError(f"order must be at least 1, got {order}")

    leaves = []
    for name, value in (("time", time), ("period", period)):
        value_leaves = jax.tree_util.tree_leaves(value)
        try:
            argument_dtype = jnp.result_type(*value_leaves)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be real numeric and array-like") from error
        if value is None or not (
            jnp.issubdtype(argument_dtype, jnp.floating) or jnp.issubdtype(argument_dtype, jnp.integer)
        ):
            raise TypeError(f"{name} must have a real numeric dtype, got {argument_dtype}")
        leaves.extend(value_leaves)

    dtype = jnp.result_type(*leaves)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
    time_array = jnp.asarray(time, dtype=dtype)
    period_array = jnp.asarray(period, dtype=dtype)
    if time_array.ndim != 1:
        raise ValueError(f"time must be one-dimensional, got shape {time_array.shape}")
    if period_array.ndim != 0:
        raise ValueError(f"period must be a scalar cycle length, got shape {period_array.shape}")

    valid = jnp.isfinite(time_array) & jnp.isfinite(period_array) & (period_array > 0)
    safe_time = jnp.where(valid, time_array, 0.0)
    safe_period = jnp.where(jnp.isfinite(period_array) & (period_array > 0), period_array, 1.0)
    harmonics = jnp.arange(1, order + 1, dtype=dtype)
    angles = (2 * jnp.pi * (safe_time / safe_period))[:, None] * harmonics
    features = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    return jnp.where(valid[:, None], features, jnp.nan)
