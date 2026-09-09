"""Composition of carryover and saturation for media exposures."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["media_response"]


def media_response(
    media: ArrayLike,
    *,
    adstock: Callable[[ArrayLike], ArrayLike],
    saturation: Callable[[ArrayLike], ArrayLike],
    n_periods: int | None = None,
    adstock_first: bool = True,
) -> jax.Array:
    """Apply carryover and saturation, retaining the requested modeling periods.

    Both transformations receive the full exposure window, including any
    earlier history. Historical periods are removed from the response only
    after both transformations have been applied. Scaling and multiplication
    by model coefficients remain separate steps.

    Parameters
    ----------
    media : array_like
        Exposure values with time on the first axis. Prepared inputs use
        ``(time, channel)`` or ``(time, group, channel)``. Include any earlier
        exposure history needed by the carryover transformation.
    adstock : callable
        Function accepting exposure values and applying carryover along
        axis zero. It must preserve the input shape. Supply its parameters
        with a function or ``functools.partial``.
    saturation : callable
        Function accepting exposure values and applying a response curve
        without changing the input shape. Channel or group-channel
        parameters can be supplied in the same way as for ``adstock``.
        For learned parameters, define the callbacks inside the model's
        log-density function using its current parameter values.
    n_periods : int, optional
        Number of final periods to return. Use ``len(data.time_values)``
        for data returned by :func:`prepare_data`, including prediction
        data without outcomes. Must be positive and cannot exceed the
        supplied number of exposure periods. If omitted, all periods are
        returned, including history.
    adstock_first : bool, default True
        Apply carryover before saturation. Set to ``False`` to apply
        saturation before carryover. Callbacks, transformation order, and
        ``n_periods`` must be fixed when compiling a model function.

    Returns
    -------
    jax.Array
        Media responses over the selected periods with all other axes
        unchanged. Precision and numeric domain behavior follow the supplied
        transformations. Channel contributions are not summed.

    Examples
    --------
    Include earlier impressions when calculating carryover, then return
    one response for each modeling period.

    .. ipython::

        In [1]: from functools import partial
           ...: import numpy as np
           ...: import polars as pl
           ...: import mmmjax as mm
           ...: frame = pl.DataFrame({
           ...:     "week": [2, 3, 4],
           ...:     "sales": [120.0, 140.0, 110.0],
           ...:     "video": [200.0, 400.0, 100.0],
           ...: })
           ...: history = pl.DataFrame({"week": [1], "video": [100.0]})
           ...: data = mm.prepare_data(
           ...:     frame, time="week", outcome="sales", media=["video"],
           ...:     media_history=history,
           ...: )
           ...: response = mm.media_response(
           ...:     data.arrays["media"],
           ...:     adstock=partial(mm.geometric_adstock, alpha=0.5, max_lag=2),
           ...:     saturation=partial(
           ...:         mm.hill_saturation, half_saturation=200.0, slope=1.0,
           ...:     ),
           ...:     n_periods=len(data.time_values),
           ...: )
           ...: frame.with_columns(
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
