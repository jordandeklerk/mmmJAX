"""Saturation transformations for marketing inputs."""

from math import log

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["hill_saturation", "log_saturation", "logistic_saturation", "root_saturation"]


def hill_saturation(
    media: ArrayLike,
    half_saturation: ArrayLike,
    slope: ArrayLike,
) -> jax.Array:
    r"""Apply a Hill response curve to nonnegative media inputs.

    For exposure :math:`x \geq 0`, half-saturation point :math:`k > 0`,
    and slope :math:`s > 0`, the response is

    .. math::

        h(x; k, s) = \frac{x^s}{x^s + k^s}.

    Zero exposure gives zero response, exposure equal to ``half_saturation``
    gives one half, and the curve approaches one as exposure increases.
    Slopes at most one give a concave curve. Larger slopes give an S-shaped
    curve with an initial period of increasing marginal response.

    Parameters
    ----------
    media : array_like
        Finite, nonnegative exposures, either raw or transformed. Scalars
        and arrays are accepted. For grouped data, use the prepared layout
        ``(time, group, channel)``. This function acts elementwise and does
        not mix periods, groups, or channels.
    half_saturation : array_like
        Positive, finite exposure at which the response is one half.
        Use the same units as ``media``, including any scaling applied
        beforehand. Supply a scalar to share it across inputs or an array
        such as ``(channel,)`` or ``(group, channel)`` for separate curves.
    slope : array_like
        Positive, finite parameter controlling the curve's shape. Supply
        a scalar or an array that broadcasts with the other inputs.
        Slopes below one give an infinite exposure gradient at zero.
        Use ``jax.vmap`` to evaluate additional parameter draws.

    Returns
    -------
    jax.Array
        Responses between zero and one with the broadcast shape of the
        inputs. Values use a common floating-point dtype of at least
        float32. Invalid numeric inputs produce ``nan`` at the affected
        positions. Multiply the response by a separate model coefficient
        to express its contribution in outcome units.

    Examples
    --------
    Give two impression columns separate half-saturation points and add
    their response curves to the dataframe.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import hill_saturation
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "search": [0.0, 1_000.0, 3_000.0],
           ...:     "video": [1_000.0, 2_000.0, 4_000.0],
           ...: })
           ...: response = hill_saturation(
           ...:     frame.select("search", "video").to_numpy(),
           ...:     half_saturation=np.array([1_000.0, 2_000.0]),
           ...:     slope=np.array([1.0, 2.0]),
           ...: )
           ...: frame.with_columns(
           ...:     pl.Series("search_response", np.asarray(response[:, 0])),
           ...:     pl.Series("video_response", np.asarray(response[:, 1])),
           ...: )
    """
    media_array, half_array, slope_array = _broadcast_saturation_inputs(
        ("media", media), ("half_saturation", half_saturation), ("slope", slope)
    )
    valid = (
        jnp.isfinite(media_array)
        & (media_array >= 0)
        & jnp.isfinite(half_array)
        & (half_array > 0)
        & jnp.isfinite(slope_array)
        & (slope_array > 0)
    )
    safe_media = jnp.where(valid, media_array, 1.0)
    safe_half = jnp.where(valid, half_array, 1.0)
    safe_slope = jnp.where(valid, slope_array, 1.0)

    # A common factor cancels from the ratio and keeps both power bases at most one
    # Treat the cancelling factor as constant during differentiation
    common = jax.lax.stop_gradient(jnp.maximum(safe_media, safe_half))
    media_power = (safe_media / common) ** safe_slope
    half_power = (safe_half / common) ** safe_slope
    response = media_power / (media_power + half_power)
    return jnp.where(valid, response, jnp.nan)


def logistic_saturation(media: ArrayLike, half_saturation: ArrayLike) -> jax.Array:
    r"""Apply a logistic response curve to nonnegative media inputs.

    For exposure :math:`x \geq 0` and half-saturation point :math:`k > 0`,
    the response is

    .. math::

        f(x; k) = \frac{1 - \exp(-\log(3)x/k)}{1 + \exp(-\log(3)x/k)}
                = \tanh\left(\frac{\log(3)x}{2k}\right).

    Zero exposure gives zero response, exposure equal to ``half_saturation``
    gives one half, and the curve approaches one as exposure increases.
    Marginal response decreases with increasing exposure. The
    half-saturation point has the same meaning as in :func:`hill_saturation`.

    Parameters
    ----------
    media : array_like
        Finite, nonnegative exposures, either raw or transformed. Scalars
        and arrays are accepted. For grouped data, use the prepared layout
        ``(time, group, channel)``. The function acts elementwise without
        mixing periods, groups, or channels.
    half_saturation : array_like
        Positive, finite exposure at which the response is one half.
        Use the same units as ``media``, including any scaling applied
        beforehand. Supply a scalar to share a curve or an array such as
        ``(channel,)`` or ``(group, channel)`` for separate curves.
        Larger values require more exposure to reach the same response.

    Returns
    -------
    jax.Array
        Responses between zero and one with the broadcast shape of the
        inputs. Values use a common floating-point dtype of at least
        float32. Invalid numeric inputs produce ``nan`` at the affected
        positions. Multiply the response by a separate model coefficient
        to express its contribution in outcome units.

    Examples
    --------
    Apply separate response curves to two impression columns.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import logistic_saturation
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "search": [0.0, 1_000.0, 3_000.0],
           ...:     "video": [1_000.0, 2_000.0, 4_000.0],
           ...: })
           ...: response = logistic_saturation(
           ...:     frame.select("search", "video").to_numpy(),
           ...:     half_saturation=np.array([1_000.0, 2_000.0]),
           ...: )
           ...: frame.with_columns(
           ...:     pl.Series("search_response", np.asarray(response[:, 0])),
           ...:     pl.Series("video_response", np.asarray(response[:, 1])),
           ...: )
    """
    media_array, half_array = _broadcast_saturation_inputs(("media", media), ("half_saturation", half_saturation))
    valid = jnp.isfinite(media_array) & (media_array >= 0) & jnp.isfinite(half_array) & (half_array > 0)
    safe_media = jnp.where(valid, media_array, 1.0)
    safe_half = jnp.where(valid, half_array, 1.0)

    # tanh(z / 2) is the logistic quotient without subtracting nearly equal values near zero
    # log(3) sets the response at media == half_saturation to one half
    response = jnp.tanh((log(3) / 2) * (safe_media / safe_half))
    return jnp.where(valid, response, jnp.nan)


def root_saturation(media: ArrayLike, exponent: ArrayLike) -> jax.Array:
    r"""Apply a root response curve to nonnegative media inputs.

    For exposure :math:`x \geq 0` and exponent :math:`0 < a \leq 1`,
    the response is

    .. math::

        f(x; a) = x^a.

    An exponent of one half gives a square-root response. Smaller
    exponents give stronger diminishing returns, while one leaves the
    input values unchanged. Unlike Hill and logistic curves, root curves
    have no fixed upper limit and their responses can exceed one.

    Parameters
    ----------
    media : array_like
        Finite, nonnegative exposures, either raw or transformed. Scalars
        and arrays are accepted. For grouped data, use the prepared layout
        ``(time, group, channel)``. Input scaling remains a separate step.
    exponent : array_like
        Finite exponent greater than zero and at most one. Supply a scalar
        for a shared curve or an array such as ``(channel,)`` or
        ``(group, channel)`` that broadcasts with ``media``.

    Returns
    -------
    jax.Array
        Nonnegative responses with the broadcast shape of the inputs.
        Values use a common floating-point dtype of at least float32.
        Invalid numeric inputs produce ``nan`` at the affected positions.
        Multiply the response by a separate model coefficient to express
        its contribution in outcome units.

    Examples
    --------
    Apply a square-root response to an impression column.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import root_saturation
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "video": [0.0, 100.0, 400.0],
           ...: })
           ...: response = root_saturation(
           ...:     frame["video"].to_numpy(), exponent=0.5,
           ...: )
           ...: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response)),
           ...: )
    """
    media_array, exponent_array = _broadcast_saturation_inputs(("media", media), ("exponent", exponent))
    valid = (
        jnp.isfinite(media_array)
        & (media_array >= 0)
        & jnp.isfinite(exponent_array)
        & (exponent_array > 0)
        & (exponent_array <= 1)
    )
    safe_media = jnp.where(valid, media_array, 1.0)
    safe_exponent = jnp.where(valid, exponent_array, 1.0)

    # Mask zero inputs before fractional powers to keep upstream gradients finite
    positive = safe_media > 0
    power_input = jnp.where(positive, safe_media, 1.0)
    # Retain the identity derivative when the exponent is exactly one
    zero_response = jnp.where(safe_exponent == 1, safe_media, 0.0)
    response = jnp.where(positive, power_input**safe_exponent, zero_response)
    return jnp.where(valid, response, jnp.nan)


def log_saturation(media: ArrayLike) -> jax.Array:
    r"""Apply a logarithmic response curve to nonnegative media inputs.

    For exposure :math:`x \geq 0`, the response is

    .. math::

        f(x) = \log(1 + x).

    Zero exposure gives zero response. The response increases with
    diminishing marginal returns but has no fixed upper limit. Unlike
    Hill and logistic curves, its values can exceed one.

    Parameters
    ----------
    media : array_like
        Finite, nonnegative exposures, either raw or transformed. Scalars
        and arrays are accepted. For grouped data, use the prepared layout
        ``(time, group, channel)``. Choose the input units before applying
        this function, since rescaling the inputs changes the response.

    Returns
    -------
    jax.Array
        Nonnegative responses with the same shape as ``media`` and a
        floating-point dtype of at least float32. Invalid numeric inputs
        produce ``nan`` at the affected positions. Multiply the response
        by a separate model coefficient to express its contribution in
        outcome units.

    Examples
    --------
    Apply a logarithmic response to an impression column, including a
    period with no exposure.

    .. ipython::

        In [1]: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import log_saturation
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "video": [0.0, 100.0, 400.0],
           ...: })
           ...: response = log_saturation(frame["video"].to_numpy())
           ...: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response)),
           ...: )
    """
    (media_array,) = _broadcast_saturation_inputs(("media", media))
    valid = jnp.isfinite(media_array) & (media_array >= 0)
    safe_media = jnp.where(valid, media_array, 0.0)

    response = jnp.log1p(safe_media)
    return jnp.where(valid, response, jnp.nan)


def _broadcast_saturation_inputs(*arguments: tuple[str, ArrayLike]) -> list[jax.Array]:
    """Promote real inputs before conversion and check their broadcast shapes."""
    leaves = []
    for name, value in arguments:
        value_leaves = jax.tree_util.tree_leaves(value)
        try:
            argument_dtype = jnp.result_type(*value_leaves)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be real numeric and array-like") from error
        if not (
            jnp.issubdtype(argument_dtype, jnp.floating)
            or jnp.issubdtype(argument_dtype, jnp.integer)
            or argument_dtype == jnp.bool_
        ):
            raise TypeError(f"{name} must have a real numeric dtype, got {argument_dtype}")
        leaves.extend(value_leaves)

    dtype = jnp.result_type(*leaves)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
    arrays = []
    for name, value in arguments:
        try:
            # Convert counts directly to floating point to avoid narrowing them to int32
            arrays.append(jnp.asarray(value, dtype=dtype))
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be real numeric and array-like") from error

    try:
        return jnp.broadcast_arrays(*arrays)
    except ValueError as error:
        shapes = ", ".join(f"{name} shape {array.shape}" for (name, _), array in zip(arguments, arrays, strict=True))
        raise ValueError(f"{shapes} must broadcast together") from error
