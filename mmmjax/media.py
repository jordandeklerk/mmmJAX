"""Media response primitives for combining carryover and saturation."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["media_response", "reach_frequency_response"]


def media_response(
    media: ArrayLike,
    *,
    adstock: Callable[[ArrayLike], ArrayLike],
    saturation: Callable[[ArrayLike], ArrayLike],
    n_periods: int | None = None,
    adstock_first: bool = True,
) -> jax.Array:
    """Apply carryover and saturation and keep the requested modeling periods.

    Apply both transformations to the full exposure history before selecting
    modeling periods. Scaling and coefficient multiplication remain separate.

    Parameters
    ----------
    media : array_like
        Exposure values with time on the first axis. Prepared inputs use
        ``(time, channel)`` or ``(time, group, channel)``. Include any earlier
        exposure history needed by the carryover transformation.
    adstock : callable
        Function applying carryover along axis zero while preserving shape.
        Supply parameters with a function or ``functools.partial``. Keep this
        argument static when using ``jax.jit``.
    saturation : callable
        Shape-preserving response curve with parameters supplied as for
        ``adstock``. For learned parameters, define both callbacks inside
        ``transformed_parameters`` or the log density using current values.
        Keep this argument static when using ``jax.jit``.
    n_periods : int, optional
        Number of final periods to return, such as ``len(data.time_values)``.
        Must be positive and not exceed the exposure window. Omit to return
        all periods, including history. Keep this argument static when using
        ``jax.jit``.
    adstock_first : bool, default True
        Apply carryover before saturation, or reverse the order with ``False``.
        Keep this argument static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Media responses over the selected periods with all other axes
        unchanged. Precision and numeric domain behavior follow the supplied
        transformations. Channel contributions are not summed.

    Examples
    --------
    Start with three modeling weeks and one earlier week of impressions
    that should count towards carryover.

    .. ipython::

        In [1]: from functools import partial
           ...: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import (
           ...:     geometric_adstock,
           ...:     hill_saturation,
           ...:     media_response,
           ...:     prepare_data,
           ...: )

        In [2]: frame = pl.DataFrame({
           ...:     "week": [2, 3, 4],
           ...:     "sales": [120.0, 140.0, 110.0],
           ...:     "video": [200.0, 400.0, 100.0],
           ...: })
           ...: history = pl.DataFrame({"week": [1], "video": [100.0]})

        In [3]: data = prepare_data(
           ...:     frame, time="week", outcome="sales", media=["video"],
           ...:     media_history=history,
           ...: )

    Apply adstock and then saturation over the full media history and keep
    one response for each modeling period.

    .. ipython::

        In [4]: response = media_response(
           ...:     data.arrays["media"],
           ...:     adstock=partial(geometric_adstock, alpha=0.5, max_lag=2),
           ...:     saturation=partial(
           ...:         hill_saturation, half_saturation=200.0, slope=1.0,
           ...:     ),
           ...:     n_periods=len(data.time_values),
           ...: )

        In [5]: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response[:, 0])),
           ...: )
    """
    if not callable(adstock):
        raise TypeError("adstock must be a function that preserves the media shape")
    if not callable(saturation):
        raise TypeError("saturation must be a function that preserves the media shape")
    if not isinstance(adstock_first, bool):
        raise TypeError("adstock_first must be a static boolean")

    media_shape = np.shape(media)
    if not media_shape:
        raise ValueError("media must have at least one dimension with time on the first axis")
    if n_periods is not None:
        if isinstance(n_periods, bool) or not isinstance(n_periods, int):
            raise TypeError("n_periods must be a static positive integer or None")
        if not 0 < n_periods <= media_shape[0]:
            raise ValueError(
                f"n_periods must be between 1 and the {media_shape[0]} supplied exposure periods, got {n_periods}. "
                "Use the number of modeling periods without counting earlier history"
            )
    else:
        n_periods = media_shape[0]

    transformations = (("adstock", adstock), ("saturation", saturation))
    if not adstock_first:
        transformations = transformations[::-1]

    response = media
    for name, transform in transformations:
        response = transform(response)
        response_shape = np.shape(response)
        if response_shape != media_shape:
            raise ValueError(
                f"{name} must preserve media shape {media_shape}, got {response_shape}. "
                "Keep all exposure periods and retain the group and channel axes"
            )

    # Keep historical exposures available to both transformations before selecting modeling periods
    return jnp.asarray(response)[media_shape[0] - n_periods :]


def reach_frequency_response(
    reach: ArrayLike,
    frequency: ArrayLike,
    *,
    adstock: Callable[[ArrayLike], ArrayLike],
    saturation: Callable[[ArrayLike], ArrayLike],
    n_periods: int | None = None,
) -> jax.Array:
    r"""Saturate frequency and weight it by reach before applying carryover.

    For reach :math:`r` and frequency :math:`f`, the response is

    .. math::

        \operatorname{Adstock}\bigl(r \odot S(f)\bigr),

    where :math:`S` is the supplied saturation function and
    :math:`\odot` denotes elementwise multiplication. Earlier reach and
    frequency contribute to carryover before the modeling periods are
    selected. The same calculation supports paid and organic channels.

    Parameters
    ----------
    reach : array_like
        Finite, nonnegative audience reached, including earlier history.
        Prepared shapes are ``(time, channel)`` or ``(time, group, channel)``.
        Supply raw reach or reach transformed using fitted scales.
    frequency : array_like
        Finite, nonnegative average exposures per person. Must match the shape
        and order of ``reach``. Keep its original units when scaling inputs.
        Use :func:`prepare_data` to validate and order reach and frequency.
    adstock : callable
        Function applying carryover along axis zero without changing the
        input shape. It receives reach multiplied by saturated frequency.
        Keep this argument static when using ``jax.jit``.
    saturation : callable
        Shape-preserving response curve with thresholds in frequency units.
        Supply parameters with a function or ``functools.partial``. For
        learned parameters, define callbacks inside ``transformed_parameters``
        or the log density using current values. Keep this argument static
        when using ``jax.jit``.
    n_periods : int, optional
        Number of final modeling periods to return, such as
        ``len(data.time_values)``. Must be positive and not exceed the
        exposure window. Omit to return all periods. Keep this argument
        static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Reach-weighted responses over the selected periods, with other
        axes unchanged. Reach and saturated frequency are multiplied using
        a common floating-point dtype of at least float32. Scaling, model
        coefficients, and channel summation remain separate.

    Examples
    --------
    Start with reach and frequency for three modeling weeks, the last with
    no new exposure, plus one earlier week of history.

    .. ipython::

        In [1]: from functools import partial
           ...: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import (
           ...:     geometric_adstock,
           ...:     hill_saturation,
           ...:     prepare_data,
           ...:     reach_frequency_response,
           ...: )

        In [2]: frame = pl.DataFrame({
           ...:     "week": [2, 3, 4],
           ...:     "video_reach": [1_000, 1_500, 0],
           ...:     "video_frequency": [2.0, 3.0, 0.0],
           ...: })
           ...: history = pl.DataFrame({
           ...:     "week": [1], "video_reach": [800],
           ...:     "video_frequency": [1.5],
           ...: })

        In [3]: data = prepare_data(
           ...:     frame, time="week", reach=["video_reach"],
           ...:     media_frequency=["video_frequency"], media_history=history,
           ...: )

    Saturation applies to frequency and the result is weighted by reach.
    Adstock carries earlier exposure into the week with none.

    .. ipython::

        In [4]: response = reach_frequency_response(
           ...:     data.arrays["reach"], data.arrays["media_frequency"],
           ...:     adstock=partial(geometric_adstock, alpha=0.5, max_lag=2),
           ...:     saturation=partial(
           ...:         hill_saturation, half_saturation=2.0, slope=1.0,
           ...:     ),
           ...:     n_periods=len(data.time_values),
           ...: )

        In [5]: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response[:, 0])),
           ...: )
    """
    reach_shape, frequency_shape = np.shape(reach), np.shape(frequency)
    if not reach_shape:
        raise ValueError("reach must have at least one dimension with time on the first axis")
    if reach_shape != frequency_shape:
        raise ValueError(
            f"reach shape {reach_shape} and frequency shape {frequency_shape} must match. "
            "Supply the same periods, groups and channels, including earlier history"
        )
    if not callable(adstock):
        raise TypeError("adstock must be a function that preserves the reach and frequency shape")

    reach_leaves = jax.tree_util.tree_leaves(reach)
    try:
        reach_dtype = jnp.result_type(*reach_leaves)
    except (TypeError, ValueError) as error:
        raise TypeError("reach must be real numeric and array-like") from error
    if not (
        jnp.issubdtype(reach_dtype, jnp.floating)
        or jnp.issubdtype(reach_dtype, jnp.integer)
        or reach_dtype == jnp.bool_
    ):
        raise TypeError(f"reach must have a real numeric dtype, got {reach_dtype}")

    def carry_response(saturated_frequency: ArrayLike) -> ArrayLike:
        # Convert integer reach directly to floating point before multiplying
        dtype = jnp.result_type(*reach_leaves, *jax.tree_util.tree_leaves(saturated_frequency))
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
        dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
        weighted = jnp.asarray(reach, dtype=dtype) * jnp.asarray(saturated_frequency, dtype=dtype)
        return adstock(weighted)

    return media_response(
        frequency,
        adstock=carry_response,
        saturation=saturation,
        n_periods=n_periods,
        adstock_first=False,
    )
