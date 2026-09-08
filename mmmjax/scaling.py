"""Reusable centering and scaling for model inputs."""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax.data import PreparedData, _DataLayout

__all__ = ["DataScaling", "Scaling", "fit_data_scaling", "fit_media_scaling", "fit_scaling"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class Scaling:
    r"""Store an offset and scale for repeated transformations.

    For an offset :math:`c` and a positive scale :math:`s`, transformation
    and inversion use

    .. math::

        z = \frac{x - c}{s}, \qquad x = z s + c.

    Use :func:`fit_scaling` or :func:`fit_media_scaling` to estimate these
    values from training data. Applying them to new data does not update
    them. Direct construction does not validate the supplied values.

    Attributes
    ----------
    offset : jax.Array
        Values subtracted before scaling. Reduced axes have length one
        so the offsets broadcast across new observations.
    scale : jax.Array
        Positive divisors with the same shape as ``offset``. Their values
        depend on the fitting function used.

    Notes
    -----
    This object can be passed to JAX transformations with its arrays as
    dynamic inputs. Its fields cannot be reassigned.
    """

    offset: jax.Array
    scale: jax.Array

    def transform(self, values: ArrayLike) -> jax.Array:
        """Center and scale values using the stored training statistics.

        Parameters
        ----------
        values : array_like
            Real-valued inputs. Stored statistics must broadcast to their
            shape without adding axes or enlarging any dimension. Use
            the same feature and group ordering as the training data.

        Returns
        -------
        jax.Array
            Transformed values with the same shape as ``values``. Inputs
            and statistics are promoted to a common floating-point dtype.

        Examples
        --------
        Apply training statistics to a new observation.

        .. ipython::

            In [1]: from mmmjax import fit_scaling
               ...: scaling = fit_scaling([2.0, 4.0, 6.0])
               ...: scaling.transform([8.0])
        """
        values_array = self._prepare_values(values)
        return (values_array - self.offset) / self.scale

    def inverse_transform(self, values: ArrayLike) -> jax.Array:
        """Return scaled values to their original units.

        Parameters
        ----------
        values : array_like
            Real-valued scaled inputs or predictions. Stored statistics
            must broadcast to their shape without adding axes or enlarging
            any dimension. Keep the training feature and group ordering.

        Returns
        -------
        jax.Array
            Values in the original units, with the same shape as ``values``.
            Inputs and statistics use a common floating-point dtype.

        Examples
        --------
        Convert predictions from the scaled units back to the original units.

        .. ipython::

            In [1]: from mmmjax import fit_scaling
               ...: scaling = fit_scaling([2.0, 4.0, 6.0])
               ...: scaling.inverse_transform([0.0, 1.0])
        """
        values_array = self._prepare_values(values)
        return values_array * self.scale + self.offset

    def _prepare_values(self, values: ArrayLike) -> jax.Array:
        """Check numeric inputs and broadcasting for either transformation."""
        try:
            # Choose floating-point precision before JAX can narrow integer observations
            dtype = jnp.result_type(*jax.tree_util.tree_leaves(values), self.offset, self.scale)
            values_array = jnp.asarray(values, dtype=dtype)
        except (TypeError, ValueError) as error:
            raise TypeError("values must be a real numeric array or sequence") from error
        if not (
            jnp.issubdtype(values_array.dtype, jnp.floating)
            or jnp.issubdtype(values_array.dtype, jnp.integer)
            or values_array.dtype == jnp.bool_
        ):
            raise TypeError(f"values must have a real numeric dtype, got {values_array.dtype}")
        try:
            result_shape = jnp.broadcast_shapes(values_array.shape, self.offset.shape, self.scale.shape)
        except ValueError as error:
            raise ValueError(
                f"values with shape {values_array.shape} do not match scaling statistics with shapes "
                f"{self.offset.shape} and {self.scale.shape}. Keep the training feature and group axes"
            ) from error
        if result_shape != values_array.shape:
            raise ValueError(
                f"scaling statistics would expand values from shape {values_array.shape} to {result_shape}. "
                "Keep the training feature and group axes, including axes of length one"
            )
        return values_array


@dataclass(frozen=True, slots=True, eq=False)
class DataScaling:
    """Store training transformations together with their input labels.

    Use :func:`fit_data_scaling` to create this object. Transformations are
    fitted once and reused for later observations. Groups and columns are
    matched internally, so their order in a new dataframe may differ.
    New groups or changed channel-to-column assignments are not accepted.

    Use :attr:`transformations` to access individual fitted transformations
    for JAX computations in the training group and feature order.
    """

    _fitted: tuple[tuple[str, Scaling], ...]
    _layout: _DataLayout
    _population: NDArray[np.generic] | None

    @property
    def transformations(self) -> dict[str, Scaling]:
        """Return the fitted transformations for the selected inputs.

        Returns
        -------
        dict of str to Scaling
            Transformations keyed by input name, such as ``media`` or
            ``controls``. Inputs left in their original units have no entry.
            Returns a new dictionary without changing the fitted state.
        """
        return dict(self._fitted)

    def transform(self, data: PreparedData) -> PreparedData:
        """Apply training scales and match the training input order.

        Parameters
        ----------
        data : PreparedData
            Data returned by :func:`prepare_data`. New periods and media
            history are allowed. Inputs may be omitted, such as outcomes
            not yet observed. Supplied inputs must retain their training
            columns, channel assignments, and groups.

        Returns
        -------
        PreparedData
            New prepared inputs containing

            - **arrays** : Scaled selected inputs and copies of untouched inputs
            - **columns** : Source columns in training order
            - **group_values** : Groups in training order
            - **time_values** : Supplied modeling periods
            - **media_time_values** : Supplied exposure periods, including history

            Channel labels retain their training assignments. Original data
            is unchanged. Alignment and scaling run before model evaluation,
            outside JAX transformations.
        """
        return self._apply(data, inverse=False)

    def inverse_transform(self, data: PreparedData) -> PreparedData:
        """Return selected inputs to their original units.

        Parameters
        ----------
        data : PreparedData
            Scaled data with the same input identities as the training data.
            Supplied groups and columns are matched to their training order.

        Returns
        -------
        PreparedData
            New prepared inputs with selected arrays restored to their
            original units and copies of untouched arrays. Labels follow
            the same rules as :meth:`transform`. Restored arrays remain
            floating-point, including inputs originally stored as integers.
        """
        return self._apply(data, inverse=True)

    def _apply(self, data: PreparedData, *, inverse: bool) -> PreparedData:
        """Align once and apply fitted factors at the data preparation boundary."""
        if not isinstance(data, PreparedData):
            raise TypeError("data must be PreparedData returned by prepare_data")
        aligned = data._align_to(self._layout)
        if (
            self._population is not None
            and "population" in aligned.arrays
            and any(name in aligned.arrays for name in ("media", "organic_media", "reach", "organic_reach"))
            and not np.array_equal(aligned.arrays["population"], self._population)
        ):
            raise ValueError(
                "population differs from the estimates used to fit media scaling. "
                "Keep the training population estimates when reusing these transformations"
            )

        for name, scaling in self._fitted:
            if name not in aligned.arrays:
                continue
            array = aligned.arrays[name]
            dtype = jnp.result_type(array, scaling.offset, scaling.scale)
            # Prepared data stays on the host until it is passed into model evaluation
            offset, divisor = np.asarray(scaling.offset), np.asarray(scaling.scale)
            with np.errstate(over="ignore", invalid="ignore"):
                values = array.astype(dtype)
                result = values * divisor + offset if inverse else (values - offset) / divisor
            if not np.isfinite(result).all():
                raise ValueError(
                    f"input {name!r} produces nonfinite values after scaling. "
                    "Check its units and magnitudes or use float64 inputs with JAX 64-bit mode"
                )
            aligned.arrays[name] = result

        return aligned


def fit_scaling(
    values: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = 0,
    center: bool = True,
    scale: bool = True,
) -> Scaling:
    """Estimate reusable centering and scaling from training observations.

    By default, subtract the training mean and divide by the training
    standard deviation. Statistics use all observations along the selected
    axes, with no degrees-of-freedom correction. Constant training series
    use a scale of one, preserving differences in future observations.

    Parameters
    ----------
    values : array_like
        Finite real-valued training observations with at least one dimension
        and no empty axes. Boolean inputs are treated as zeros and ones.
        The original values are not modified.
    axis : int or tuple of int or None, default 0
        Axes over which to estimate statistics. For values shaped
        ``(time, group, feature)``, use ``0`` for separate group-feature
        statistics or ``(0, 1)`` to pool observations across groups.
        ``None`` uses all axes. Negative axes are accepted.
    center : bool, default True
        Whether to subtract the training mean. Set to ``False`` to use
        an offset of zero.
    scale : bool, default True
        Whether to divide by the training standard deviation. Set to
        ``False`` to use a divisor of one.

    Returns
    -------
    Scaling
        Stored transformation containing

        - **offset** : Training mean, or zeros when centering is disabled
        - **scale** : Training standard deviation, with ones for constant
          series or when scaling is disabled

        Reduced axes have length one. Arrays use at least float32 precision,
        with float64 available when JAX 64-bit mode is enabled. Fitting runs
        outside JAX transformations. The returned object's ``transform`` and
        ``inverse_transform`` methods support JIT, gradients, and vectorization.

    Examples
    --------
    Fit separate statistics for two columns, then reuse them on later
    observations without changing the training scales.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import fit_scaling
           ...: training = pl.DataFrame({
           ...:     "temperature": [10.0, 12.0, 14.0],
           ...:     "price": [8.0, 10.0, 12.0],
           ...: })
           ...: scaling = fit_scaling(training.to_numpy())
           ...: future = pl.DataFrame({"temperature": [16.0], "price": [9.0]})
           ...: scaled = scaling.transform(future.to_numpy())
           ...: scaling.inverse_transform(scaled)
    """
    if not isinstance(center, bool):
        raise TypeError(f"center must be True or False, got {center!r}")
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be True or False, got {scale!r}")

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError("values must be a real numeric array or sequence") from error
    if array.dtype.kind not in "biuf":
        raise TypeError(f"values must have a real numeric dtype, got {array.dtype}")
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"values must have at least one dimension and no empty axes, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinite entries. Resolve them before fitting scaling")

    if axis is None:
        axes = tuple(range(array.ndim))
    else:
        requested_axes = axis if isinstance(axis, tuple) else (axis,)
        if not requested_axes or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in requested_axes
        ):
            raise TypeError("axis must be an integer, a nonempty tuple of integers, or None")
        if any(item < -array.ndim or item >= array.ndim for item in requested_axes):
            raise ValueError(f"axis {axis!r} is out of bounds for values with {array.ndim} dimensions")
        axes = tuple(int(item) % array.ndim for item in requested_axes)
        if len(set(axes)) != len(axes):
            raise ValueError(f"axis {axis!r} selects an axis more than once. Select each axis only once")

    dtype = array.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64 if dtype.itemsize == 8 else np.float32)
    dtype = jax.dtypes.canonicalize_dtype(np.promote_types(dtype, np.float32))
    statistic_shape = tuple(1 if index in axes else size for index, size in enumerate(array.shape))

    # Estimate once on the host with a wider accumulator, then store the chosen JAX precision
    with np.errstate(over="ignore", invalid="ignore"):
        offset = np.mean(array, axis=axes, dtype=np.float64, keepdims=True) if center else np.zeros(statistic_shape)
        deviation = np.std(array, axis=axes, dtype=np.float64, keepdims=True) if scale else np.ones(statistic_shape)
        if scale:
            minimum = np.min(array, axis=axes, keepdims=True)
            constant = minimum == np.max(array, axis=axes, keepdims=True)
            # Even identical decimal values can have a tiny computed standard deviation
            deviation = np.where(constant | (deviation == 0), 1.0, deviation)
            if center:
                offset = np.where(constant, minimum, offset)
        scale_values = deviation.astype(dtype)
        offset = offset.astype(dtype)
    if not np.isfinite(offset).all() or not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
        raise ValueError(
            f"values do not produce a finite offset and positive scale in {dtype}. "
            "Rescale the training values or use float64 inputs with JAX 64-bit mode"
        )

    return Scaling(offset=jnp.asarray(offset), scale=jnp.asarray(scale_values))


def fit_media_scaling(
    media: ArrayLike,
    *,
    method: Literal["median", "mean", "max"] = "median",
    population: ArrayLike | None = None,
) -> Scaling:
    r"""Estimate reusable channel scales from media exposures.

    Each channel is divided by a training statistic, pooled across periods
    and groups. By default, use the median of positive exposures. Mean and
    maximum scaling are also available. Media is not centered, so periods
    with no exposure remain zero after transformation.

    With population estimates :math:`p_g`, first calculate exposure per
    person. For training exposures :math:`x_{t,g,c}` and the selected
    statistic :math:`T`, the stored divisor is

    .. math::

        a_c = T_{t,g}\!\left(\frac{x_{t,g,c}}{p_g}\right),
        \qquad s_{g,c} = p_g a_c.

    The median excludes zero exposures, while the mean includes them.
    Transformed exposures are :math:`x_{t,g,c}/s_{g,c}`. Without population,
    use :math:`p_g = 1`. A single series uses one statistic per channel
    without a group axis.

    Parameters
    ----------
    media : array_like
        Finite, nonnegative training exposures shaped ``(time, channel)``
        or ``(time, group, channel)``. Keep the channel axis even for a
        single channel. Each channel needs at least one positive observation
        across the training periods and groups. Suitable for paid or organic
        impressions and reach. Apply frequency transformations separately.
    method : {"median", "mean", "max"}, default "median"
        Statistic used to scale each channel. ``"median"`` uses only positive
        exposures, reducing the influence of unusually large observations.
        ``"mean"`` includes periods with no exposure. ``"max"`` divides by
        the largest training exposure. Future values can exceed one.
    population : array_like, optional
        Positive, finite population estimates, not boolean. Supply a scalar
        for ``(time, channel)`` inputs, or one value per group with shape
        ``(n_groups,)`` for grouped inputs. Omit to skip population adjustment.
        A scalar population cancels from the normalized result for a single
        series.

    Returns
    -------
    Scaling
        Stored transformation containing

        - **offset** : Zeros so absent exposure stays zero
        - **scale** : Selected channel statistics multiplied by population
          where supplied

        Statistics retain a length-one time axis. Grouped inputs have a
        length-one group axis unless population-specific factors are used.
        Both channel statistics and population factors stay fixed for future calls.
        Keep the training group and channel order. The original inputs are
        unchanged. Fitting runs outside JAX transformations, while applying
        the stored transformation supports JIT, gradients, and vectorization.
        Statistics use a common floating-point dtype of at least float32.

    Examples
    --------
    Scale two impression columns without changing periods with no activity,
    then reuse the training scales for a later week.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import fit_media_scaling
           ...: training = pl.DataFrame({
           ...:     "search": [0.0, 1_000.0, 3_000.0],
           ...:     "video": [2_000.0, 0.0, 6_000.0],
           ...: })
           ...: scaling = fit_media_scaling(training.to_numpy())
           ...: future = pl.DataFrame({"search": [4_000.0], "video": [0.0]})
           ...: scaling.transform(future.to_numpy())
           ...: maximum = fit_media_scaling(training.to_numpy(), method="max")
           ...: maximum.transform(future.to_numpy())
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be 'median', 'mean' or 'max', got {method!r}")
    if method not in ("median", "mean", "max"):
        raise ValueError(f"method must be 'median', 'mean' or 'max', got {method!r}")

    try:
        media_array = np.asarray(media)
    except (TypeError, ValueError) as error:
        raise TypeError("media must be a real numeric array or sequence") from error
    if media_array.dtype.kind not in "biuf":
        raise TypeError(f"media must have a real numeric dtype, got {media_array.dtype}")
    if media_array.ndim not in (2, 3) or media_array.size == 0:
        raise ValueError(
            "media must have shape (time, channel) or (time, group, channel) with no empty axes. "
            f"Got shape {media_array.shape}"
        )
    if not np.isfinite(media_array).all() or np.any(media_array < 0):
        raise ValueError("media must contain finite nonnegative exposures. Resolve negative, NaN or infinite values")

    population_factors = np.ones((1,) * media_array.ndim)
    dtype = jnp.result_type(media_array)
    if population is not None:
        try:
            population_array = np.asarray(population)
        except (TypeError, ValueError) as error:
            raise TypeError("population must contain real numeric estimates") from error
        if population_array.dtype.kind not in "iuf":
            raise TypeError(f"population must have a real numeric dtype, not boolean, got {population_array.dtype}")
        expected_shape = (media_array.shape[1],) if media_array.ndim == 3 else ()
        if population_array.shape != expected_shape:
            raise ValueError(
                f"population must have shape {expected_shape} for media with shape {media_array.shape}. "
                f"Got shape {population_array.shape}"
            )
        if not np.isfinite(population_array).all() or np.any(population_array <= 0):
            raise ValueError("population must contain positive finite estimates. Check each group's population")
        dtype = jnp.result_type(media_array, population_array)
        population_factors = population_array.astype(np.float64).reshape(1, *expected_shape, 1)

    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64 if dtype.itemsize == 8 else np.float32)
    dtype = jax.dtypes.canonicalize_dtype(np.promote_types(dtype, np.float32))
    axes = tuple(range(media_array.ndim - 1))

    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = media_array.astype(np.float64) / population_factors
    if not np.isfinite(adjusted).all():
        raise ValueError("media and population produce nonfinite adjusted exposures. Check their units and magnitudes")
    positive = adjusted > 0
    inactive_channels = np.flatnonzero(~np.any(positive, axis=axes)).tolist()
    if inactive_channels:
        raise ValueError(
            f"media channels at indices {inactive_channels} have no positive observations to estimate a scale. "
            "Check media and population values or remove inactive channels"
        )

    # Fit once on the host, keeping a separate statistic for each channel
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "median":
            channel_scale = np.nanmedian(np.where(positive, adjusted, np.nan), axis=axes, keepdims=True)
        elif method == "mean":
            channel_scale = np.mean(adjusted, axis=axes, keepdims=True)
        else:
            channel_scale = np.max(adjusted, axis=axes, keepdims=True)
        scale_values = (channel_scale * population_factors).astype(dtype)
    if not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
        raise ValueError(
            f"media and population do not produce a positive finite scale in {dtype}. "
            "Rescale the inputs or use float64 inputs with JAX 64-bit mode"
        )

    return Scaling(offset=jnp.zeros(scale_values.shape, dtype=dtype), scale=jnp.asarray(scale_values))


def fit_data_scaling(
    data: PreparedData,
    *,
    media_method: Literal["median", "mean", "max"] | None = "median",
    scale_outcome: bool = False,
    scale_controls: bool = True,
    scale_treatments: bool = True,
    adjust_population: bool = False,
) -> DataScaling:
    """Fit reusable transformations for labeled model inputs.

    By default, scale exposure inputs by their positive medians and
    standardize controls and non-media treatments. Outcomes stay in their
    original units unless scaling is requested. Spend, population, revenue
    per outcome, and both frequency inputs are always left unchanged.

    Parameters
    ----------
    data : PreparedData
        Training data returned by :func:`prepare_data`. Statistics pool
        periods and groups, retaining one statistic per feature. Exposure
        statistics include all supplied training media history. Other
        statistics use only the modeling periods. Inputs are not modified.
    media_method : {"median", "mean", "max"} or None, default "median"
        Scaling method for paid and organic media and reach. ``"median"``
        excludes zeros, ``"mean"`` includes them, and ``"max"`` uses the
        largest exposure. Each channel needs positive training exposure.
        Use ``None`` to leave these inputs in their original units.
    scale_outcome : bool, default False
        Center the outcome and divide by its training standard deviation.
        Enable for a model defined on a standardized continuous response.
        Leave disabled for count likelihoods that need the original counts.
        Requires an outcome in the training data.
    scale_controls : bool, default True
        Standardize each control using its training mean and standard
        deviation. Omitted controls do not create a transformation.
    scale_treatments : bool, default True
        Standardize each non-media treatment, such as product price, using
        its training mean and standard deviation. Omitted treatments do
        not create a transformation.
    adjust_population : bool, default False
        Adjust exposures by population before estimating channel scales.
        Requires population in the training data. Does not adjust controls,
        treatments, or outcomes. Population factors remain fixed when
        applying the fitted scales. Prediction data may omit population
        to reuse those factors, but supplied estimates must match the
        training values when exposure inputs are transformed.

    Returns
    -------
    DataScaling
        Fitted scaling object containing

        - **transformations** : Per-input offsets and divisors, available
          as individual :class:`Scaling` objects

        The object also retains the input labels needed to align future
        data, without storing training observations. Its ``transform`` and
        ``inverse_transform`` methods return new :class:`PreparedData`
        objects. No statistics are refitted when transforming new data.

    Examples
    --------
    Prepare impressions, costs, and a control from a dataframe. Fit the
    scaling once, then reuse it on a later week with no observed outcome.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import fit_data_scaling, prepare_data
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "sales": [100, 120, 140],
           ...:     "impressions": [0, 1_000, 3_000],
           ...:     "cost": [0.0, 20.0, 60.0],
           ...:     "temperature": [10.0, 12.0, 14.0],
           ...: })
           ...: data = prepare_data(
           ...:     frame, time="week", outcome="sales",
           ...:     media=["impressions"], spend=["cost"],
           ...:     controls=["temperature"],
           ...: )
           ...: scaling = fit_data_scaling(data)
           ...: scaled = scaling.transform(data)
           ...: future = prepare_data(
           ...:     pl.DataFrame({"week": [4], "impressions": [4_000]}),
           ...:     time="week", media=["impressions"],
           ...: )
           ...: scaling.transform(future).arrays["media"]
    """
    if not isinstance(data, PreparedData):
        raise TypeError("data must be PreparedData returned by prepare_data")
    for name, value in (
        ("scale_outcome", scale_outcome),
        ("scale_controls", scale_controls),
        ("scale_treatments", scale_treatments),
        ("adjust_population", adjust_population),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False, got {value!r}")
    if media_method is not None:
        if not isinstance(media_method, str):
            raise TypeError(f"media_method must be 'median', 'mean', 'max' or None, got {media_method!r}")
        if media_method not in ("median", "mean", "max"):
            raise ValueError(f"media_method must be 'median', 'mean', 'max' or None, got {media_method!r}")
    if scale_outcome and "outcome" not in data.arrays:
        raise ValueError("scale_outcome requires an outcome selected in prepare_data")
    if adjust_population and "population" not in data.arrays:
        raise ValueError("adjust_population requires population selected in prepare_data")

    fitted = {}
    population = data.arrays["population"] if adjust_population else None
    media_inputs = ("media", "organic_media", "reach", "organic_reach")
    if media_method is not None:
        for name in media_inputs:
            if name in data.arrays:
                try:
                    fitted[name] = fit_media_scaling(data.arrays[name], method=media_method, population=population)
                except ValueError as error:
                    raise ValueError(f"Cannot fit scaling for {name!r}. {error}") from error

    observation_axes = (0, 1) if data.group_columns else (0,)
    for name, enabled in (("outcome", scale_outcome), ("controls", scale_controls), ("treatments", scale_treatments)):
        if enabled and name in data.arrays:
            fitted[name] = fit_scaling(data.arrays[name], axis=observation_axes)

    stored_population = (
        population.copy() if population is not None and any(name in fitted for name in media_inputs) else None
    )
    return DataScaling(_fitted=tuple(fitted.items()), _layout=data._layout(), _population=stored_population)
