"""Reusable centering and scaling for model inputs."""

from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax.data.prepare import PreparedData, _DataLayout

__all__ = ["DataScaling", "Scaling", "fit_data_scaling"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class Scaling:
    r"""Store an offset and scale for repeated transformations.

    For an offset :math:`c` and a positive scale :math:`s`, transformation
    and inversion use

    .. math::

        z = \frac{x - c}{s}, \qquad x = z s + c.

    :func:`fit_data_scaling` fits one for each scaled input role, and blocks
    receive the outcome's as ``outcome_scaling``. Direct construction skips
    validation. The object is registered as a JAX pytree, so it can pass
    through JAX transformations and its methods work inside them.

    Attributes
    ----------
    offset : jax.Array
        Values subtracted before scaling. Reduced axes have length one
        so the offsets broadcast across new observations.
    scale : jax.Array
        Positive divisors with the same shape as ``offset``.
    """

    offset: jax.Array
    scale: jax.Array

    def transform(self, values: ArrayLike) -> jax.Array:
        """Center and scale values using the stored training statistics.

        Parameters
        ----------
        values : array_like
            Real-valued inputs in the fitted feature and group order.
            Stored statistics must broadcast without expanding the input shape.

        Returns
        -------
        jax.Array
            Transformed values with the input shape and a floating-point dtype.
        """
        values_array = self._prepare_values(values)
        return (values_array - self.offset) / self.scale

    def inverse_transform(self, values: ArrayLike) -> jax.Array:
        """Return scaled values to their original units.

        Parameters
        ----------
        values : array_like
            Scaled inputs or predictions in the fitted feature and group order.
            Stored statistics must broadcast without expanding the input shape.

        Returns
        -------
        jax.Array
            Values in original units with the input shape and a floating-point dtype.
        """
        values_array = self._prepare_values(values)
        return values_array * self.scale + self.offset

    def _prepare_values(self, values: ArrayLike) -> jax.Array:
        """Check numeric inputs and broadcasting for either transformation."""
        try:
            # Choose floating-point precision before JAX can narrow integer observations
            dtype = jnp.result_type(*jax.tree.leaves(values), self.offset, self.scale)
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

    :func:`fit_data_scaling` builds this object from training data. It keeps
    one :class:`Scaling` for each scaled input role, and
    :attr:`transformations` returns them by role.

    ``transform`` and ``inverse_transform`` put new data in the training
    group and column order before applying the fitted statistics. The data
    must hold exactly the training groups, and a changed channel-to-column
    assignment raises an error.
    """

    _fitted: tuple[tuple[str, Scaling], ...]
    _layout: _DataLayout
    _population: NDArray[np.generic] | None
    _population_roles: tuple[str, ...] = ()

    @property
    def transformations(self) -> dict[str, Scaling]:
        """Return the fitted transformations for the selected inputs.

        Returns
        -------
        dict of str to Scaling
            A new dictionary keyed by input role, such as ``media`` or
            ``controls``. Unscaled inputs have no entry.
        """
        return dict(self._fitted)

    def transform(self, data: PreparedData) -> PreparedData:
        """Apply training scales and match the training input order.

        The call runs on the host, outside JAX transformations, and leaves
        the original data unchanged. Data that was already scaled raises an
        error.

        Parameters
        ----------
        data : PreparedData
            Unscaled inputs from :func:`prepare_data` that retain the fitted
            columns, channel assignments, and groups. New periods, different
            ordering, and omitted inputs are allowed.

        Returns
        -------
        PreparedData
            New prepared inputs with the following fields.

            - **arrays** — Scaled selected inputs and copies of untouched inputs
            - **columns** — Source columns in training order
            - **group_values** — Groups in training order
            - **time_values** — Supplied modeling periods
            - **media_time_values** — Supplied exposure periods including
              history
        """
        return self._apply(data, inverse=False)

    def inverse_transform(self, data: PreparedData) -> PreparedData:
        """Return selected inputs to their original units.

        Supplied groups and columns are matched to their training order as in
        :meth:`transform`. Restored arrays stay floating-point even where the
        original inputs were integers. Data scaled by a different
        ``DataScaling`` raises an error.

        Parameters
        ----------
        data : PreparedData
            Scaled data with the same input identities as the training data.

        Returns
        -------
        PreparedData
            New inputs in original units.
        """
        return self._apply(data, inverse=True)

    def _apply(self, data: PreparedData, *, inverse: bool) -> PreparedData:
        """Align once and apply fitted factors at the data preparation boundary."""
        if not isinstance(data, PreparedData):
            raise TypeError("data must be PreparedData returned by prepare_data")
        if data._scaling is not None:
            if not inverse:
                raise ValueError("These inputs have already been scaled. Supply raw prepared data to transform")
            if data._scaling is not self:
                raise ValueError("These inputs use different fitted scales. Invert them with their original scaling")
        aligned = data._align_to(self._layout)
        if (
            self._population is not None
            and "population" in aligned.arrays
            and any(name in aligned.arrays for name in self._population_roles)
            and not np.array_equal(aligned.arrays["population"], self._population)
        ):
            raise ValueError(
                "population differs from the estimates used to fit scaling. "
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

        return replace(aligned, _scaling=None if inverse else self)


def fit_data_scaling(
    data: PreparedData,
    *,
    media_method: Literal["median", "mean", "max"] | None = "median",
    scale_outcome: bool | Literal["population"] = False,
    scale_controls: bool = True,
    scale_treatments: bool = True,
    adjust_population: bool = False,
) -> DataScaling:
    """Fit reusable transformations for labeled model inputs.

    By default, paid and organic impressions and reach are divided by their
    positive medians, and controls and non-media treatments are standardized.
    The outcome stays in its original units unless ``scale_outcome`` is set.
    Spend, population, revenue per outcome, and both frequency inputs are
    always left unchanged.

    Statistics pool periods and groups per feature. Exposure statistics
    include the media history, and the others use only the modeling periods.
    The data itself is not modified.

    The returned ``DataScaling`` applies the training statistics to new
    prepared data, and its ``inverse_transform`` restores the original units.
    New data may omit population. Any population it supplies must match the
    training population when it also holds population-adjusted inputs.

    Parameters
    ----------
    data : PreparedData
        Unscaled inputs from :func:`prepare_data`.
    media_method : {"median", "mean", "max"} or None, default "median"
        Statistic for paid and organic impressions and reach. The median
        excludes zeros and the mean includes them. Each channel needs positive
        exposure. ``None`` leaves them in their original units.
    scale_outcome : bool or {"population"}, default False
        ``True`` standardizes the outcome and ``"population"`` first divides
        it by population. Requires an outcome and, for ``"population"``,
        population estimates. Leave disabled for count likelihoods.
    scale_controls : bool, default True
        Standardize each supplied control using its mean and standard deviation.
    scale_treatments : bool, default True
        Standardize each supplied non-media treatment using its mean and
        standard deviation.
    adjust_population : bool, default False
        Divide exposures by population before fitting channel statistics.
        Requires population and affects exposures only.

    Returns
    -------
    DataScaling
        Fitted transformations by input role with their training labels.

    Examples
    --------
    Prepare impressions, costs, and a control from a dataframe.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import fit_data_scaling, prepare_data

        In [2]: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3],
           ...:     "sales": [100, 120, 140],
           ...:     "impressions": [0, 1_000, 3_000],
           ...:     "cost": [0.0, 20.0, 60.0],
           ...:     "temperature": [10.0, 12.0, 14.0],
           ...: })

        In [3]: data = prepare_data(
           ...:     frame, time="week", outcome="sales",
           ...:     media=["impressions"], spend=["cost"],
           ...:     controls=["temperature"],
           ...: )

    Fit the scaling once on the training data. Each role gets the
    transformation suited to it.

    .. ipython::

        In [4]: scaling = fit_data_scaling(data)
           ...: scaled = scaling.transform(data)

    A later week with no observed outcome is scaled with the same fitted
    statistics.

    .. ipython::

        In [5]: future = prepare_data(
           ...:     pl.DataFrame({"week": [4], "impressions": [4_000]}),
           ...:     time="week", media=["impressions"],
           ...: )
           ...: scaling.transform(future).arrays["media"]
    """
    if not isinstance(data, PreparedData):
        raise TypeError("data must be PreparedData returned by prepare_data")
    if data._scaling is not None:
        raise ValueError("These inputs have already been scaled. Fit transformations using raw prepared data")
    population_outcome = isinstance(scale_outcome, str) and scale_outcome == "population"
    if not isinstance(scale_outcome, bool) and not population_outcome:
        raise TypeError("scale_outcome must be True, False, or 'population'")

    for name, value in (
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
    if population_outcome and "population" not in data.arrays:
        raise ValueError("scale_outcome='population' requires population selected in prepare_data")
    if adjust_population and "population" not in data.arrays:
        raise ValueError("adjust_population requires population selected in prepare_data")

    fitted = {}
    population_roles = []
    population = data.arrays["population"] if adjust_population else None
    media_inputs = ("media", "organic_media", "reach", "organic_reach")
    if media_method is not None:
        for name in media_inputs:
            if name in data.arrays:
                try:
                    fitted[name] = _fit_media_scaling(data.arrays[name], method=media_method, population=population)
                except ValueError as error:
                    raise ValueError(f"Cannot fit scaling for {name!r}. {error}") from error
                if adjust_population:
                    population_roles.append(name)

    observation_axes = (0, 1) if data.group_columns else (0,)
    for name, enabled in (("outcome", scale_outcome), ("controls", scale_controls), ("treatments", scale_treatments)):
        if enabled and name in data.arrays:
            if name == "outcome" and population_outcome:
                fitted[name] = _fit_population_outcome(data.arrays[name], data.arrays["population"])
                population_roles.append(name)
            else:
                fitted[name] = _fit_standardization(data.arrays[name], axis=observation_axes)

    stored_population = data.arrays["population"].copy() if population_roles else None
    result = DataScaling(
        _fitted=tuple(fitted.items()),
        _layout=data._layout(),
        _population=stored_population,
        _population_roles=tuple(population_roles),
    )
    return result


def _fit_media_scaling(
    media: ArrayLike,
    *,
    method: Literal["median", "mean", "max"] = "median",
    population: ArrayLike | None = None,
) -> Scaling:
    """Divide each channel by one statistic of its exposures so that zero exposure stays zero."""
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


def _fit_population_outcome(outcome: NDArray[np.generic], population: NDArray[np.generic]) -> Scaling:
    """Combine per-person standardization into one original-unit affine transform."""
    if outcome.dtype.kind not in "biuf" or not np.isfinite(outcome).all():
        raise ValueError("outcome must contain finite real values for population scaling")
    if outcome.ndim not in (1, 2) or outcome.size == 0:
        raise ValueError("outcome must have a nonempty time axis and optional group axis")
    if population.dtype.kind not in "iuf":
        raise TypeError("population must contain real numeric estimates, not boolean values")
    if population.shape != outcome.shape[1:]:
        raise ValueError(f"population must have shape {outcome.shape[1:]} for outcome shape {outcome.shape}")
    population_values = population.astype(np.float64)
    if not np.isfinite(population_values).all() or np.any(population_values <= 0):
        raise ValueError("population must contain positive finite estimates")

    dtype = jnp.result_type(outcome, population)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64 if dtype.itemsize == 8 else np.float32)
    dtype = jax.dtypes.canonicalize_dtype(np.promote_types(dtype, np.float32))
    factors = population_values.reshape(1, *population.shape)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        adjusted = outcome.astype(np.float64) / factors
    if not np.isfinite(adjusted).all():
        raise ValueError("outcome and population produce nonfinite per-person values. Check their units and magnitudes")

    mean, deviation = _scaling_statistics(adjusted, tuple(range(adjusted.ndim)))
    # Folding population into both factors lets inversion restore original units directly
    with np.errstate(over="ignore", invalid="ignore"):
        offset = (mean * factors).astype(dtype)
        scale = (deviation * factors).astype(dtype)
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(
            f"outcome and population do not produce a finite offset and positive scale in {dtype}. "
            "Rescale the inputs or use float64 inputs with JAX 64-bit mode"
        )

    result = Scaling(offset=jnp.asarray(offset), scale=jnp.asarray(scale))
    return result


def _fit_standardization(
    values: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = 0,
    center: bool = True,
    scale: bool = True,
) -> Scaling:
    """Estimate centering and scaling statistics over the given axes of training values."""
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
    # Estimate once on the host with a wider accumulator, then store the chosen JAX precision
    offset, deviation = _scaling_statistics(array.astype(np.float64, copy=False), axes, center=center, scale=scale)
    with np.errstate(over="ignore", invalid="ignore"):
        scale_values = deviation.astype(dtype)
        offset = offset.astype(dtype)
    if not np.isfinite(offset).all() or not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
        raise ValueError(
            f"values do not produce a finite offset and positive scale in {dtype}. "
            "Rescale the training values or use float64 inputs with JAX 64-bit mode"
        )

    return Scaling(offset=jnp.asarray(offset), scale=jnp.asarray(scale_values))


def _scaling_statistics(
    array: NDArray[np.float64],
    axes: tuple[int, ...],
    *,
    center: bool = True,
    scale: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate host statistics before converting to the requested storage precision."""
    statistic_shape = tuple(1 if index in axes else size for index, size in enumerate(array.shape))
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
    return np.asarray(offset), np.asarray(deviation)
