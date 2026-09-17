"""Dataframe preparation for marketing mix models."""

from calendar import monthrange
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from keyword import iskeyword
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias, cast, get_args

import jax
import jax.numpy as jnp
import narwhals as nw
import numpy as np
import xarray as xr
from jax.typing import ArrayLike, DTypeLike
from narwhals.dependencies import is_into_dataframe
from narwhals.typing import IntoDataFrameT
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mmmjax.scaling import DataScaling

__all__ = ["Data", "ModelInput", "PreparedData", "Reference", "prepare_data", "select_channels"]

_TimeInput: TypeAlias = Literal["time", "media_time", "day_of_year", "media_day_of_year"]


class ModelInput(NamedTuple):
    """Describe one name that model functions can request from prepared data.

    Attributes
    ----------
    kind : str
        ``array`` for JAX arrays, ``integer`` for static Python integers,
        ``object`` for the ``outcome_scaling`` transform and the ``reference``
        namespace, or ``constant`` for declared Python values passed through
        unchanged.
    axes : tuple of str
        Named axes of an array input, empty for scalars. Training references
        use ``reference_time`` and ``reference_media_time`` for their time
        axes. Outcome conversions gain a group axis under population scaling.
    source : str
        ``data`` for selected roles, ``time`` for positions computed from the
        observation labels, ``builtin`` for model-supplied values,
        ``reference`` for training inputs retained when evaluating new data,
        ``input`` for auxiliary datasets, or ``constant`` for declared values.
    """

    kind: Literal["array", "integer", "object", "constant"]
    axes: tuple[str, ...]
    source: Literal["data", "time", "builtin", "reference", "input", "constant"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Reference:
    """Training inputs retained while a model evaluates new data.

    Request ``reference`` in a program block and read the training arrays as
    attributes, such as ``reference.spend`` or ``reference.time``, together
    with the training period count ``reference.n_periods``. These never
    change for scenarios, forecasts, or response curves, so calculations
    that must stay anchored to the fitted data use them in place of the
    current inputs.

    Attributes
    ----------
    values : dict of str to jax.Array
        Training arrays by role and time input name.
    n_periods : int
        Number of training periods.
    """

    values: dict[str, jax.Array]
    n_periods: int = field(default=0, metadata={"static": True})

    def __getattr__(self, name: str) -> jax.Array:
        """Return the training array stored under ``name`` or explain which roles exist."""
        values: dict[str, jax.Array] = object.__getattribute__(self, "values")
        if name in values:
            return values[name]
        available = ", ".join((*sorted(values), "n_periods"))
        raise AttributeError(f"reference has no input {name!r}. Available reference inputs are {available}")


@dataclass(frozen=True, slots=True, eq=False)
class PreparedData:
    """Store prepared model inputs together with their observation labels.

    Create validated inputs with :func:`prepare_data`. Direct construction
    skips validation. If editing arrays, preserve their shapes and label order.

    Attributes
    ----------
    arrays : dict of str to numpy.ndarray
        Independent arrays ordered by time, optional group, then feature.
        Outcome and revenue per outcome have no feature axis. Population
        is a group vector or a scalar. Exposure inputs include history,
        while other time-varying inputs cover only the modeling periods.
    time_column : str
        Source column identifying observation periods.
    time_values : tuple
        Sorted modeling periods.
    media_time_values : tuple
        Shared exposure periods, including history. Matches ``time_values``
        without history and is empty without exposure inputs.
    group_columns : tuple of str
        Source columns identifying each series. Empty for a single series.
    group_values : tuple of tuple
        Observed group combinations in first-appearance order.
        Empty when no group columns were supplied.
    columns : dict of str to tuple of str
        Source columns for each input, in the selected order.
    channels : tuple of str
        Channel labels shared by media and spend. Empty without media.
    organic_channels : tuple of str
        Labels for the separate organic media axis. Empty without organic media.
    rf_channels : tuple of str
        Labels for the separate paid reach, frequency, and spend axis.
        Empty without these inputs.
    organic_rf_channels : tuple of str
        Labels for the separate organic reach and frequency axis.
        Empty without these inputs.
    frequency : str or None
        Inferred or declared observation spacing. None for numeric labels,
        fewer than three dates without a declared spacing, or disabled checks.
    time_positions : numpy.ndarray
        Elapsed positions of the modeling periods from the first period, in
        days for dates and in label units otherwise. These are the ``time``
        values that models receive.
    media_time_positions : numpy.ndarray
        Elapsed positions of the exposure periods on the same origin, with
        history periods negative. Empty without exposure inputs. These are
        the ``media_time`` values that models receive.
    day_of_year : numpy.ndarray
        Calendar day of the year for each modeling period, from one through
        366, matching the ``day_of_year`` model input. Requires dates.
    media_day_of_year : numpy.ndarray
        Calendar day of the year for each exposure period, matching the
        ``media_day_of_year`` model input. Empty without exposure inputs.
    model_inputs : dict of str to ModelInput
        Every name that model functions can request from this data, with its
        kind, axes, and source. Use it to choose ``Data`` variable names.
    """

    arrays: dict[str, NDArray[np.generic]]
    time_column: str
    time_values: tuple[object, ...]
    media_time_values: tuple[object, ...]
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]
    columns: dict[str, tuple[str, ...]]
    channels: tuple[str, ...]
    organic_channels: tuple[str, ...] = ()
    rf_channels: tuple[str, ...] = ()
    organic_rf_channels: tuple[str, ...] = ()
    frequency: str | None = None
    # Record applied transformations to prevent scaling these arrays twice
    _scaling: "DataScaling | None" = field(default=None, repr=False)

    @property
    def time_positions(self) -> NDArray[np.float64]:
        """Return elapsed positions of the modeling periods from the first period."""
        positions, _ = _time_positions(self.time_values)
        return positions

    @property
    def media_time_positions(self) -> NDArray[np.float64]:
        """Return elapsed positions of the exposure periods, with history negative."""
        if not self.media_time_values:
            return np.empty(0, dtype=np.float64)
        _, origin = _time_positions(self.time_values)
        positions, _ = _time_positions(self.media_time_values, origin=origin)
        return positions

    @property
    def day_of_year(self) -> NDArray[np.float64]:
        """Return the calendar day of the year for each modeling period."""
        return _day_of_year(self.time_values)

    @property
    def media_day_of_year(self) -> NDArray[np.float64]:
        """Return the calendar day of the year for each exposure period."""
        if not self.media_time_values:
            return np.empty(0, dtype=np.float64)
        return _day_of_year(self.media_time_values)

    @property
    def model_inputs(self) -> dict[str, ModelInput]:
        """Return the names model functions can request, with kind, axes, and source."""
        return _model_inputs(self)

    def _to_jax(
        self,
        *,
        dtype: DTypeLike = float,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ) -> dict[str, jax.Array]:
        """Copy input arrays to JAX before model evaluation.

        Preserve keys and shapes, keeping integer and boolean inputs non-floating.
        Floating-point inputs use ``dtype``, with ``float`` following JAX's precision setting.
        Reject unavailable precision or overflowing casts. ``device`` selects
        the destination device or sharding, with None using JAX's default.
        """
        try:
            requested_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as error:
            raise TypeError(f"dtype must be float32 or float64, or float for the JAX default. Got {dtype!r}") from error
        if dtype is None or requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"dtype must be float32 or float64, or float for the JAX default. Got {dtype!r}")

        floating_dtype = jax.dtypes.canonicalize_dtype(requested_dtype)
        if dtype is not float and floating_dtype != requested_dtype:
            raise ValueError(
                "explicit dtype float64 requires JAX 64-bit mode. "
                "Set JAX_ENABLE_X64=true before starting Python or use dtype=float32"
            )

        # CPU transfers can reuse NumPy buffers, so hand JAX snapshots of our editable arrays
        converted: dict[str, NDArray[np.generic]] = {}
        for name, array in self.arrays.items():
            if np.issubdtype(array.dtype, np.floating):
                # Check on the host so overflow is caught before allocating device buffers
                with np.errstate(over="ignore"):
                    values = array.astype(floating_dtype, copy=True)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"block {name!r} contains values that are not finite as {floating_dtype}. "
                        "Check or rescale the data, or use float64 with JAX 64-bit mode"
                    )
            else:
                target_dtype = jax.dtypes.canonicalize_dtype(array.dtype)
                if target_dtype != array.dtype and np.issubdtype(array.dtype, np.integer):
                    limits = np.iinfo(target_dtype)
                    if int(array.min()) < limits.min or int(array.max()) > limits.max:
                        raise ValueError(
                            f"block {name!r} contains integers that do not fit {target_dtype}. "
                            "Set JAX_ENABLE_X64=true before starting Python to preserve these counts"
                        )
                values = array.astype(target_dtype, copy=True)
            converted[name] = values

        return cast(dict[str, jax.Array], jax.device_put(converted, device=device))

    def _layout(self) -> "_DataLayout":
        """Snapshot input identities and axes without retaining observations."""
        return _DataLayout(
            axis_counts={name: array.ndim for name, array in self.arrays.items()},
            columns=self.columns.copy(),
            group_columns=self.group_columns,
            group_values=self.group_values,
            channels=self.channels,
            organic_channels=self.organic_channels,
            rf_channels=self.rf_channels,
            organic_rf_channels=self.organic_rf_channels,
        )

    def _align_to(self, reference: "PreparedData | _DataLayout") -> "PreparedData":
        """Match reference ordering for internal prediction preparation."""
        if isinstance(reference, PreparedData):
            reference = reference._layout()
        if not isinstance(reference, _DataLayout):
            raise TypeError("reference must be PreparedData returned by prepare_data")
        if self.group_columns != reference.group_columns:
            raise ValueError(
                f"group columns must match the reference {reference.group_columns}, got {self.group_columns}. "
                "Use the same groups selection when calling prepare_data"
            )

        observation_indices = [np.arange(len(self.time_values))]
        if self.group_columns:
            try:
                positions = {label: index for index, label in enumerate(self.group_values)}
                reference_groups = set(reference.group_values)
            except TypeError as error:
                raise TypeError(
                    f"group columns {self.group_columns} contain labels that cannot be matched. "
                    "Use scalar group labels such as strings, numbers or dates"
                ) from error
            missing = [label for label in reference.group_values if label not in positions]
            unexpected = [label for label in self.group_values if label not in reference_groups]
            if missing or unexpected:
                raise ValueError(
                    f"group labels do not match the reference for {self.group_columns}. "
                    f"Missing groups {missing} and unexpected groups {unexpected}. "
                    "Supply exactly the reference groups before alignment"
                )
            observation_indices.append(np.asarray([positions[label] for label in reference.group_values]))

        unknown = [name for name in self.arrays if name not in reference.axis_counts]
        if unknown:
            raise ValueError(f"inputs {unknown} do not exist in the reference. Supply inputs used by the model")

        arrays: dict[str, NDArray[np.generic]] = {}
        columns: dict[str, tuple[str, ...]] = {}
        media_inputs = ("media", "organic_media", "reach", "media_frequency", "organic_reach", "organic_frequency")
        for name, reference_ndim in reference.axis_counts.items():
            if name not in self.arrays:
                continue
            array = self.arrays[name]
            reference_columns = reference.columns[name]
            reference_column_set = set(reference_columns)
            column_positions = {column: index for index, column in enumerate(self.columns[name])}
            missing_columns = [column for column in reference_columns if column not in column_positions]
            unexpected_columns = [column for column in self.columns[name] if column not in reference_column_set]
            if missing_columns or unexpected_columns:
                raise ValueError(
                    f"columns for {name!r} do not match the reference. "
                    f"Missing columns {missing_columns} and unexpected columns {unexpected_columns}. "
                    "Select the same source columns before alignment"
                )
            if array.ndim != reference_ndim:
                raise ValueError(
                    f"input {name!r} has a different number of axes from the reference. "
                    "Prepare both inputs with prepare_data without changing their array shapes"
                )

            if name == "population":
                arrays[name] = array[observation_indices[1]] if self.group_columns else array.copy()
                columns[name] = reference_columns
                continue

            indices = observation_indices
            if name in media_inputs:
                indices = [np.arange(len(self.media_time_values)), *observation_indices[1:]]
            if array.ndim > len(observation_indices):
                feature_order = [column_positions[column] for column in reference_columns]
                if name in media_inputs or name in ("spend", "rf_spend"):
                    if name in ("organic_reach", "organic_frequency"):
                        channel_labels, reference_labels = self.organic_rf_channels, reference.organic_rf_channels
                    elif name in ("reach", "media_frequency", "rf_spend"):
                        channel_labels, reference_labels = self.rf_channels, reference.rf_channels
                    elif name == "organic_media":
                        channel_labels, reference_labels = self.organic_channels, reference.organic_channels
                    else:
                        channel_labels, reference_labels = self.channels, reference.channels
                    aligned_channels = tuple(channel_labels[index] for index in feature_order)
                    if aligned_channels != reference_labels:
                        raise ValueError(
                            f"channel labels for {name!r} do not match the reference. "
                            "Use the same channel-to-column assignments when calling prepare_data"
                        )
                indices = [*indices, np.asarray(feature_order)]
            # Reorder both axes in one copy instead of gathering whole blocks twice
            arrays[name] = array[np.ix_(*indices)]
            columns[name] = reference_columns

        return PreparedData(
            arrays=arrays,
            time_column=self.time_column,
            time_values=self.time_values,
            media_time_values=self.media_time_values if any(name in arrays for name in media_inputs) else (),
            group_columns=self.group_columns,
            group_values=reference.group_values,
            columns=columns,
            channels=reference.channels if "media" in arrays else (),
            organic_channels=reference.organic_channels if "organic_media" in arrays else (),
            rf_channels=reference.rf_channels if "reach" in arrays else (),
            organic_rf_channels=reference.organic_rf_channels if "organic_reach" in arrays else (),
            frequency=self.frequency,
            _scaling=self._scaling,
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Data:
    """Declare model data, variable names, and fixed preprocessing together.

    Supply this object to ``Model(data=...)``. Observations, variable names,
    and auxiliary inputs are copied. Fitted scaling objects are reused.

    Parameters
    ----------
    data : PreparedData
        Observations and labels returned by :func:`prepare_data`.
    variables : mapping of str to str, optional
        Model function argument names mapped to prepared or auxiliary input
        names. ``PreparedData.model_inputs`` lists the available names. If
        omitted, functions use the standard input names. When supplied, only
        the declared names are available to model functions.
    inputs : xarray.Dataset, optional
        Additional fixed inputs, such as experiment measurements.
    constants : mapping of str to object, optional
        Fixed specification values that model functions request by name,
        such as a carryover length or a prepared HSGP approximation. Values
        must be hashable and pass through unchanged, so integers stay usable
        as shapes under ``jax.jit``. They never change across scenarios.
    scaling : DataScaling or {"auto"}, optional
        Fitted transformations or automatic scaling. None uses unchanged data.

    Attributes
    ----------
    observations : PreparedData
        A copy of the prepared observations.
    variables : dict of str to str or None
        A copy of the variable declarations.
    inputs : xarray.Dataset or None
        A copy of the auxiliary inputs.
    constants : dict of str to object
        A copy of the declared constants.
    scaling : DataScaling or str or None
        The fitted transformations or requested scaling mode.
    model_inputs : dict of str to ModelInput
        Every name model functions can request, combining the prepared
        observations, auxiliary inputs, and constants.
    """

    _observations: PreparedData
    _variables: dict[str, str] | None
    _inputs: xr.Dataset | None
    _constants: dict[str, object]
    _scaling: "DataScaling | Literal['auto'] | None"

    def __init__(
        self,
        data: PreparedData,
        *,
        variables: Mapping[str, str] | None = None,
        inputs: xr.Dataset | None = None,
        constants: Mapping[str, object] | None = None,
        scaling: "DataScaling | Literal['auto'] | None" = None,
    ) -> None:
        from mmmjax.scaling import DataScaling

        if not isinstance(data, PreparedData):
            raise TypeError("Data requires PreparedData returned by prepare_data")
        if variables is not None:
            if not isinstance(variables, Mapping):
                raise TypeError("variables must map model function input names to data source names")
            for name, source in variables.items():
                if not isinstance(name, str) or not name.isidentifier() or iskeyword(name):
                    raise ValueError("Each data variable name must be a valid Python identifier")
                if not isinstance(source, str) or not source:
                    raise TypeError(f"Source for data variable {name!r} must be a nonempty string")
        if inputs is not None and not isinstance(inputs, xr.Dataset):
            raise TypeError("inputs must be an xarray.Dataset")
        if constants is not None:
            if not isinstance(constants, Mapping):
                raise TypeError("constants must map model function input names to fixed values")
            for name, value in constants.items():
                if not isinstance(name, str) or not name.isidentifier() or iskeyword(name):
                    raise ValueError("Each constant name must be a valid Python identifier")
                try:
                    hash(value)
                except TypeError as error:
                    raise TypeError(
                        f"Constant {name!r} must be hashable, such as a number, string, or frozen configuration. "
                        "Supply arrays through inputs or the prepared data"
                    ) from error
        if isinstance(scaling, str) and scaling != "auto":
            raise ValueError("scaling must be DataScaling, 'auto', or None")
        if scaling is not None and not isinstance(scaling, (DataScaling, str)):
            raise TypeError("scaling must be DataScaling, 'auto', or None")

        scaling_memo = {id(value): value for value in (data._scaling, scaling) if isinstance(value, DataScaling)}
        observations = deepcopy(data, scaling_memo)
        object.__setattr__(self, "_observations", observations)
        object.__setattr__(self, "_variables", None if variables is None else dict(variables))
        object.__setattr__(self, "_inputs", None if inputs is None else inputs.copy(deep=True))
        object.__setattr__(self, "_constants", {} if constants is None else dict(constants))
        object.__setattr__(self, "_scaling", scaling)

    @property
    def observations(self) -> PreparedData:
        """Return a copy of the prepared observations."""
        return deepcopy(self._observations, {id(self._observations._scaling): self._observations._scaling})

    @property
    def variables(self) -> dict[str, str] | None:
        """Return a copy of the model variable declarations."""
        if self._variables is None:
            return None
        return self._variables.copy()

    @property
    def inputs(self) -> xr.Dataset | None:
        """Return a copy of the auxiliary model inputs."""
        if self._inputs is None:
            return None
        return self._inputs.copy(deep=True)

    @property
    def constants(self) -> dict[str, object]:
        """Return a copy of the declared constants."""
        return dict(self._constants)

    @property
    def scaling(self) -> "DataScaling | Literal['auto'] | None":
        """Return the fitted transformations or requested scaling mode."""
        return self._scaling

    @property
    def model_inputs(self) -> dict[str, ModelInput]:
        """Return every requestable name across observations, inputs, and constants."""
        inputs = self._observations.model_inputs
        if self._inputs is not None:
            for name, variable in self._inputs.data_vars.items():
                inputs[str(name)] = ModelInput("array", tuple(str(axis) for axis in variable.dims), "input")
        for name in self._constants:
            inputs[name] = ModelInput("constant", (), "constant")
        return inputs

    def _snapshot(
        self,
    ) -> tuple[
        PreparedData,
        dict[str, str] | None,
        xr.Dataset | None,
        dict[str, object],
        "DataScaling | Literal['auto'] | None",
    ]:
        """Copy declarations together to preserve applied-scaling identity."""
        snapshot = (self.observations, self.variables, self.inputs, self.constants, self.scaling)
        return snapshot


def _data_dimensions(data: PreparedData) -> dict[str, tuple[str, ...]]:
    """Name each prepared role's axes independently of their lengths."""
    group_axes = ("group",) if data.group_columns else ()
    observation_axes = ("time", *group_axes)
    exposure_axes = ("media_time", *group_axes)
    return {
        "outcome": observation_axes,
        "revenue_per_outcome": observation_axes,
        "media": (*exposure_axes, "channel"),
        "organic_media": (*exposure_axes, "organic_channel"),
        "reach": (*exposure_axes, "rf_channel"),
        "media_frequency": (*exposure_axes, "rf_channel"),
        "organic_reach": (*exposure_axes, "organic_rf_channel"),
        "organic_frequency": (*exposure_axes, "organic_rf_channel"),
        "spend": (*observation_axes, "channel"),
        "rf_spend": (*observation_axes, "rf_channel"),
        "controls": (*observation_axes, "control"),
        "treatments": (*observation_axes, "treatment"),
        "population": group_axes,
    }


def _time_input_names(data: PreparedData) -> tuple[_TimeInput, ...]:
    """List the time inputs the data can supply, with calendar names only for dates."""
    try:
        _, origin = _time_positions(data.time_values)
    except (TypeError, ValueError):
        # Categorical period labels carry no positions, so no time input exists.
        return ()
    dated = isinstance(origin, datetime)
    names: list[_TimeInput] = []
    for name in get_args(_TimeInput):
        if name.startswith("media_") and not data.media_time_values:
            continue
        if name.endswith("day_of_year") and not dated:
            continue
        names.append(name)
    return tuple(names)


def _time_input_axes(name: _TimeInput) -> tuple[str, ...]:
    """Name the axis of a time input by whether it follows the media periods."""
    return ("media_time",) if name.startswith("media_") else ("time",)


def _reference_dimensions(data: PreparedData) -> dict[str, tuple[str, ...]]:
    """Name the axes of every training array the reference namespace can hold."""
    dimensions = _data_dimensions(data)
    dimensions.update({name: _time_input_axes(name) for name in get_args(_TimeInput)})
    return dimensions


def _model_inputs(data: PreparedData) -> dict[str, ModelInput]:
    """Enumerate the requestable inputs once for models, results, and users."""
    role_axes = _data_dimensions(data)
    inputs = {role: ModelInput("array", role_axes[role], "data") for role in data.arrays}
    for name in _time_input_names(data):
        inputs[name] = ModelInput("array", _time_input_axes(name), "time")
    inputs["n_periods"] = ModelInput("integer", (), "builtin")
    if "outcome" in data.arrays:
        inputs["outcome_scaling"] = ModelInput("object", (), "builtin")
    inputs["reference"] = ModelInput("object", (), "reference")
    return inputs


@dataclass(frozen=True, slots=True)
class _DataLayout:
    """Retain input ordering for reuse independently of training observations."""

    axis_counts: dict[str, int]
    columns: dict[str, tuple[str, ...]]
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]
    channels: tuple[str, ...]
    organic_channels: tuple[str, ...]
    rf_channels: tuple[str, ...]
    organic_rf_channels: tuple[str, ...]


def _prepare_model_frame(
    frame: object,
    layout: _DataLayout,
    *,
    time: str,
    frequency: str | None,
) -> PreparedData:
    """Reuse selected source columns without inferring roles from new column names."""
    if not is_into_dataframe(frame):
        raise TypeError("New data must be an eager dataframe or PreparedData")
    native = nw.from_native(frame, eager_only=True)
    available = set(native.columns)
    columns = {}
    for role, names in layout.columns.items():
        if not available.intersection(names):
            continue
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"New data is missing selected {role} columns {missing}. Supply all columns for this input"
            )
        columns[role] = names

    return prepare_data(
        native,
        time=time,
        groups=layout.group_columns,
        frequency=frequency,
        outcome=columns["outcome"][0] if "outcome" in columns else None,
        revenue_per_outcome=columns["revenue_per_outcome"][0] if "revenue_per_outcome" in columns else None,
        population=columns["population"][0] if "population" in columns else None,
        media=columns.get("media"),
        organic_media=columns.get("organic_media"),
        reach=columns.get("reach"),
        media_frequency=columns.get("media_frequency"),
        organic_reach=columns.get("organic_reach"),
        organic_frequency=columns.get("organic_frequency"),
        spend=columns.get("spend"),
        rf_spend=columns.get("rf_spend"),
        controls=columns.get("controls"),
        treatments=columns.get("treatments"),
        channels=layout.channels if "media" in columns else None,
        organic_channels=layout.organic_channels if "organic_media" in columns else None,
        rf_channels=layout.rf_channels if "reach" in columns else None,
        organic_rf_channels=layout.organic_rf_channels if "organic_reach" in columns else None,
    )


def prepare_data(
    frame: IntoDataFrameT,
    *,
    time: str,
    outcome: str | None = None,
    revenue_per_outcome: str | None = None,
    population: str | None = None,
    media: Sequence[str] | None = None,
    organic_media: Sequence[str] | None = None,
    reach: Sequence[str] | None = None,
    media_frequency: Sequence[str] | None = None,
    organic_reach: Sequence[str] | None = None,
    organic_frequency: Sequence[str] | None = None,
    media_history: IntoDataFrameT | None = None,
    spend: Sequence[str] | None = None,
    rf_spend: Sequence[str] | None = None,
    controls: Sequence[str] | None = None,
    treatments: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    organic_channels: Sequence[str] | None = None,
    rf_channels: Sequence[str] | None = None,
    organic_rf_channels: Sequence[str] | None = None,
    groups: Sequence[str] = (),
    frequency: str | None = "auto",
) -> PreparedData:
    """Prepare a dataframe for modeling while keeping its observation labels.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals, including pandas and Polars
        DataFrames and PyArrow Tables. Selected values must be finite,
        nonmissing, and numeric or boolean. The input is not modified.
        Omit value selections to prepare only time and group labels.
    time : str
        Observation periods that sort chronologically. Each time-group
        combination must be unique.
        Calendar checks accept dates, timezone-naive datetimes, or strings
        written as ``YYYY-MM-DD``. Original labels are retained.
    outcome : str, optional
        Column containing the response, such as sales or conversions.
        Omit it when preparing prediction data without observed outcomes.
    revenue_per_outcome : str, optional
        Nonnegative revenue per sale or conversion, varying by period and
        group. Stored separately without converting the outcome to revenue.
        Boolean values are not accepted. Does not require an outcome.
    population : str, optional
        Positive population, constant across periods within each group.
        Without groups, repeat one value across all periods. Boolean values
        are not accepted. Supplying population does not apply scaling.
    media : sequence of str, optional
        Columns containing nonnegative paid media inputs, such as impressions
        or spending. Their order defines the channel axis. Use a list
        even for one channel.
    organic_media : sequence of str, optional
        Nonnegative unpaid exposures, such as email clicks or organic social
        impressions. Their order defines a separate channel axis. Use a list
        even for one channel. Paid media and spend are not required.
    reach : sequence of str, optional
        Nonnegative audience reached per paid channel and period. Pair with
        ``media_frequency`` in matching order. Uses a separate channel axis
        and does not require ``media``.
    media_frequency : sequence of str, optional
        Nonnegative average exposures per person reached, ordered to match
        ``reach``. This is advertising frequency, not calendar spacing.
    organic_reach : sequence of str, optional
        Nonnegative audience reached by unpaid channels. Pair with
        ``organic_frequency`` in matching order. Uses a separate channel axis
        and does not require other media inputs.
    organic_frequency : sequence of str, optional
        Nonnegative average organic exposures per person reached, ordered to
        match ``organic_reach``. This is separate from calendar spacing.
    media_history : dataframe-like, optional
        Earlier exposures for carryover into the first modeling periods.
        Include all selected exposure columns and all groups, using the same
        time and group columns as ``frame``. All periods must precede ``frame``.
        Other inputs are not required. Requires at least one exposure selection.
        Calendar checks cover both frames. Preparation does not apply adstock.
    spend : sequence of str, optional
        Nonnegative spending, with one column per ``media`` channel in matching
        order. The same columns may be selected for media and spend.
    rf_spend : sequence of str, optional
        Nonnegative spending, with one column per ``reach`` channel in matching
        order. Covers only the modeling periods.
    controls : sequence of str, optional
        Columns containing adjustment variables, such as temperature or
        economic indicators. Their order defines the control axis.
        Negative values are allowed for controls and the outcome.
    treatments : sequence of str, optional
        Non-media inputs whose effects the model estimates, such as product
        prices or promotions. Negative and boolean values are accepted.
        Their order defines the treatment axis. Use a list even for one
        treatment. Media and spend are not required.
    channels : sequence of str, optional
        Unique labels shared by media and spend, in column order.
        Defaults to the ``media`` column names. Requires ``media``.
    organic_channels : sequence of str, optional
        Unique labels in ``organic_media`` column order. Defaults to those
        column names. Requires ``organic_media``.
    rf_channels : sequence of str, optional
        Unique labels shared by reach, media frequency, and their spend, in
        column order. Defaults to the ``reach`` column names. Requires ``reach``.
    organic_rf_channels : sequence of str, optional
        Unique labels shared by organic reach and frequency, in column order.
        Defaults to the ``organic_reach`` column names. Requires ``organic_reach``.
    groups : sequence of str, optional
        Columns identifying each series, such as ``["region"]``. All groups
        must have the same periods. Observed combinations share one array
        axis in first-appearance order. Omit for a single series.
    frequency : str or None, default "auto"
        Infer ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``,
        or ``"yearly"`` from at least three dates, including history.
        Calendar periods follow the first date's day or month-end.
        Specify the expected spacing to check for missing periods, including
        short series. Numeric labels are not assigned calendar units.
        Set to ``None`` to skip calendar checks.

    Returns
    -------
    PreparedData
        Prepared inputs containing

        - **arrays** : NumPy arrays keyed by input role. Omitted inputs have no entry
        - **time_column** : Name of the source column identifying time periods
        - **time_values** : Sorted modeling periods
        - **media_time_values** : Shared periods for all paid and organic
          exposure inputs, including history. Empty without exposure inputs
        - **group_columns** : Selected group-column names
        - **group_values** : Observed group-label tuples in array order
        - **columns** : Source column names by input role, in selected order
        - **channels** : Labels shared by media and spend
        - **organic_channels** : Organic media labels
        - **rf_channels** : Labels shared by paid reach, frequency, and spend
        - **organic_rf_channels** : Labels shared by organic reach and frequency
        - **frequency** : Inferred or declared calendar spacing, or None

        Time-varying arrays are ordered by time, group (if supplied), then
        feature. Outcome and revenue per outcome have no final feature axis.
        Population has shape ``(n_groups,)``, or ``()`` without groups.
        All other inputs keep a feature axis even for a single column.
        Integer outcomes and population estimates retain their dtype
        separately from continuous inputs. Arrays do not share memory with
        either dataframe. Unused group and channel labels are empty tuples.
    """
    selections = {
        "outcome": outcome,
        "revenue_per_outcome": revenue_per_outcome,
        "population": population,
        "media": media,
        "organic_media": organic_media,
        "reach": reach,
        "media_frequency": media_frequency,
        "organic_reach": organic_reach,
        "organic_frequency": organic_frequency,
        "spend": spend,
        "rf_spend": rf_spend,
        "controls": controls,
        "treatments": treatments,
    }
    columns: dict[str, tuple[str, ...]] = {}
    for name, selection in selections.items():
        if selection is None:
            continue
        if name in ("outcome", "revenue_per_outcome", "population"):
            if not isinstance(selection, str):
                example = {"outcome": "sales", "revenue_per_outcome": "unit_revenue", "population": "residents"}[name]
                raise TypeError(f"{name} must be a column name, such as {example!r}")
            names: tuple[str, ...] = (selection,)
        elif isinstance(selection, Sequence) and not isinstance(selection, str):
            names = tuple(selection)
        else:
            raise TypeError(f"{name} must be a sequence of column names. Use a list even for one column")
        if not names or any(not isinstance(column, str) or not column for column in names):
            raise ValueError(f"{name} must select at least one nonempty string column name. Got {selection!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"{name} contains repeated columns {names}. Select each column only once per input")
        columns[name] = names

    media_inputs = tuple(
        name
        for name in ("media", "organic_media", "reach", "media_frequency", "organic_reach", "organic_frequency")
        if name in columns
    )
    if media_history is not None and not media_inputs:
        raise ValueError(
            "media_history requires media, organic_media, reach or organic_reach columns. "
            "Select the same exposure columns in both dataframes"
        )
    for reach_input, frequency_input in (("reach", "media_frequency"), ("organic_reach", "organic_frequency")):
        if (reach_input in columns) != (frequency_input in columns):
            missing = frequency_input if reach_input in columns else reach_input
            raise ValueError(
                f"{reach_input} and {frequency_input} must be supplied together. "
                f"Add {missing} columns in matching channel order"
            )
    for name, media_input in (
        ("spend", "media"),
        ("rf_spend", "reach"),
        ("media_frequency", "reach"),
        ("organic_frequency", "organic_reach"),
    ):
        if name not in columns:
            continue
        if media_input not in columns:
            raise ValueError(f"{name} requires {media_input} columns so spending can be associated with each channel")
        if len(columns[name]) != len(columns[media_input]):
            raise ValueError(f"{name} must select one column per {media_input} channel in the same order")

    channel_names: dict[str, tuple[str, ...]] = {}
    for name, selection, media_input in (
        ("channels", channels, "media"),
        ("organic_channels", organic_channels, "organic_media"),
        ("rf_channels", rf_channels, "reach"),
        ("organic_rf_channels", organic_rf_channels, "organic_reach"),
    ):
        labels = columns.get(media_input, ())
        if selection is not None:
            if media_input not in columns:
                raise ValueError(f"{name} requires {media_input} columns to label")
            if isinstance(selection, str) or not isinstance(selection, Sequence):
                raise TypeError(f"{name} must be a sequence of names. Use a list even for one channel")
            labels = tuple(selection)
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError(f"{name} must contain only nonempty string names")
            if len(set(labels)) != len(labels):
                raise ValueError(f"{name} must contain unique names. Give each media channel a different name")
            if len(labels) != len(columns[media_input]):
                raise ValueError(f"{name} must contain one name per {media_input} column in the same order")
        channel_names[media_input] = labels

    selected = _prepare_panel(
        frame,
        time=time,
        groups=groups,
        values=list(dict.fromkeys(column for names in columns.values() for column in names)),
        # History and modeling periods need one calendar anchor, especially across February
        frequency=frequency if media_history is None and frequency != "auto" else None,
    )
    time_values = tuple(selected.get_column(time).unique(maintain_order=True).to_list())
    group_values = tuple(selected.select(groups).unique(maintain_order=True).rows()) if groups else ()
    observation_shape = (len(time_values), len(group_values)) if groups else (len(time_values),)

    arrays: dict[str, NDArray[np.generic]] = {}
    for name, names in columns.items():
        if name == "population":
            values = selected.get_column(names[0]).to_numpy().reshape(observation_shape)
            if values.dtype == np.bool_:
                raise TypeError(
                    f"population column {names[0]!r} contains boolean values. Use numeric population estimates"
                )
            if np.any(values <= 0):
                raise ValueError(
                    f"population column {names[0]!r} contains zero or negative values. "
                    "Provide a positive population estimate for each group"
                )
            changing = np.any(values != values[0], axis=0)
            if np.any(changing):
                affected = [group_values[index] for index in np.flatnonzero(changing)] if groups else []
                context = f" for groups {affected}" if groups else ""
                raise ValueError(
                    f"population column {names[0]!r} changes across periods{context}. "
                    "Choose a fixed population estimate for each series before preparing data"
                )
            arrays[name] = values[:1].reshape(observation_shape[1:]).copy()
        elif name in ("outcome", "revenue_per_outcome"):
            arrays[name] = selected.get_column(names[0]).to_numpy().copy().reshape(observation_shape)
            if name == "revenue_per_outcome":
                if arrays[name].dtype == np.bool_:
                    raise TypeError(
                        f"revenue_per_outcome column {names[0]!r} contains boolean values. Use numeric revenue amounts"
                    )
                if np.any(np.less(arrays[name], 0)):
                    raise ValueError(
                        f"revenue_per_outcome column {names[0]!r} contains negative values. "
                        "Provide nonnegative revenue per response unit"
                    )
        else:
            # Stacking numeric series avoids object arrays from mixed pandas column dtypes
            values = np.stack([selected.get_column(column).to_numpy() for column in names], axis=-1)
            arrays[name] = values.reshape(*observation_shape, len(names))
        if name in media_inputs or name in ("spend", "rf_spend"):
            negative = np.any(np.less(arrays[name], 0), axis=tuple(range(arrays[name].ndim - 1)))
            negative_columns = [names[index] for index in np.flatnonzero(negative)]
            if negative_columns:
                raise ValueError(
                    f"{name} columns {negative_columns} contain negative values. "
                    "Use nonnegative media measurements and spending before preparing data"
                )

    data = PreparedData(
        arrays=arrays,
        time_column=time,
        time_values=time_values,
        media_time_values=time_values if media_inputs else (),
        group_columns=tuple(groups),
        group_values=group_values,
        columns=columns,
        channels=channel_names["media"],
        organic_channels=channel_names["organic_media"],
        rf_channels=channel_names["reach"],
        organic_rf_channels=channel_names["organic_reach"],
        frequency=_resolve_frequency(time_values, time=time, frequency=frequency) if media_history is None else None,
    )
    if media_history is None:
        return data

    history = prepare_data(
        media_history,
        time=time,
        groups=groups,
        media=columns.get("media"),
        organic_media=columns.get("organic_media"),
        reach=columns.get("reach"),
        media_frequency=columns.get("media_frequency"),
        organic_reach=columns.get("organic_reach"),
        organic_frequency=columns.get("organic_frequency"),
        channels=channel_names["media"] if "media" in columns else None,
        organic_channels=channel_names["organic_media"] if "organic_media" in columns else None,
        rf_channels=channel_names["reach"] if "reach" in columns else None,
        organic_rf_channels=channel_names["organic_reach"] if "organic_reach" in columns else None,
        frequency=None,
    )._align_to(data)
    try:
        overlaps = history.time_values[-1] >= time_values[0]
    except TypeError as error:
        raise TypeError(
            f"media_history and frame must use the same kind of time labels in column {time!r}. "
            "Use comparable period numbers or the same date representation in both dataframes"
        ) from error
    if overlaps:
        raise ValueError(
            f"media_history must contain only periods before the first modeling period {time_values[0]!r}. "
            f"Its last period is {history.time_values[-1]!r}. Remove overlapping or later observations"
        )

    media_times = (*history.time_values, *time_values)
    return replace(
        data,
        arrays={
            **arrays,
            **{name: np.concatenate((history.arrays[name], arrays[name]), axis=0) for name in media_inputs},
        },
        media_time_values=media_times,
        frequency=_resolve_frequency(media_times, time=time, frequency=frequency),
    )


def select_channels(
    values: ArrayLike,
    *,
    channels: Sequence[str],
    select: str | Sequence[str],
) -> jax.Array:
    """Select named channels without changing the preceding dimensions.

    Select parameters for channel-specific priors or retrieve contributions
    by channel name.

    Parameters
    ----------
    values : array_like
        Parameters or contributions with channels on the final axis.
    channels : sequence of str
        Unique channel labels in the array's current order, such as
        ``data.channels``. Keep these labels fixed when compiling.
    select : str or sequence of str
        One channel name or a nonempty sequence of unique names to retain,
        in the desired output order. Keep this selection fixed when compiling.

    Returns
    -------
    jax.Array
        Selected values with the input's preceding dimensions and dtype.
        Selecting one name retains a final channel axis of length one.

    Examples
    --------
    Use different prior families for two entries of the same coefficient
    array. For group-specific coefficients, the same calls select the
    channels across every group.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import half_normal, lognormal, select_channels
           ...: channels = ("video", "search")
           ...: coefficient = jnp.array([0.6, 0.4])
           ...: video = select_channels(
           ...:     coefficient, channels=channels, select="video",
           ...: )
           ...: search = select_channels(
           ...:     coefficient, channels=channels, select="search",
           ...: )
           ...: target = half_normal(video, scale=1.0)
           ...: target += lognormal(search, location=0.0, scale=0.5)
           ...: target
    """
    channel_names = _channel_names(channels, argument="channels")
    selected_names = _channel_names((select,) if isinstance(select, str) else select, argument="select")
    positions = {name: index for index, name in enumerate(channel_names)}
    unknown = [name for name in selected_names if name not in positions]
    if unknown:
        raise ValueError(f"select contains unknown channel names {unknown}. Use names from channels")

    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must have a final channel axis")
    if array.shape[-1] != len(channel_names):
        raise ValueError(
            f"The final axis of values has length {array.shape[-1]} but channels contains {len(channel_names)} names"
        )

    indices = tuple(positions[name] for name in selected_names)
    return jnp.take(array, jnp.asarray(indices, dtype=jnp.int32), axis=-1, unique_indices=True)


def _channel_names(value: object, *, argument: str) -> tuple[str, ...]:
    """Validate ordered labels before constructing array indices."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{argument} must be an ordered sequence of channel names")
    if not value:
        raise ValueError(f"{argument} must contain at least one channel name")

    names = []
    for name in value:
        if not isinstance(name, str):
            raise TypeError(f"{argument} must contain only string channel names")
        if not name:
            raise ValueError(f"{argument} must contain only nonempty channel names")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{argument} must contain unique channel names")
    return tuple(names)


def _prepare_panel(
    frame: IntoDataFrameT,
    *,
    time: str,
    groups: Sequence[str] = (),
    values: Sequence[str],
    frequency: str | None = None,
) -> nw.DataFrame[IntoDataFrameT]:
    """Validate and sort selected columns into a complete time-major panel.

    Retain observed group combinations in first-appearance order. Reject
    missing time-group pairs without filling them. When supplied, ``frequency``
    also checks for missing periods between the first and last observations.
    """
    if isinstance(groups, str) or not isinstance(groups, Sequence):
        raise TypeError("groups must be a sequence of column names, such as ['region'], or () for a single series")
    selected = _prepare_frame(frame, keys=[time, *groups], values=values)
    if frequency is not None:
        _validate_calendar(selected.get_column(time).unique().to_list(), time=time, frequency=frequency)
    if not groups:
        return selected.sort(time)

    series = selected.select(groups).unique(keep="first", maintain_order=True)
    n_times = selected.get_column(time).n_unique()
    missing = n_times * series.shape[0] - selected.shape[0]
    # Keys are unique, so this count proves every observed series has every observed time
    if missing:
        raise ValueError(
            f"data is missing {missing} combinations of {time!r} and {list(groups)}. "
            "Each observed group must contain the same time values"
        )

    # Sorting by an explicit rank avoids backend-specific ordering of categorical labels
    rank = nw.generate_temporary_column_name(n_bytes=8, columns=selected.columns)
    series = series.with_row_index(rank)
    return selected.join(series, on=list(groups), how="left").sort([time, rank]).select(selected.columns)


def _prepare_frame(
    frame: IntoDataFrameT,
    *,
    keys: Sequence[str],
    values: Sequence[str],
) -> nw.DataFrame[IntoDataFrameT]:
    """Select and validate an eager observation table without changing its rows.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals. The input is not modified.
    keys : sequence of str
        Columns identifying each observation, such as time and geography.
        Their combinations must be unique and cannot contain missing values.
    values : sequence of str
        Numeric or boolean columns to retain in the supplied order. Values
        must be finite and nonmissing. Negative values are allowed.

    Returns
    -------
    narwhals.DataFrame
        Selected columns, keys first, with the original row order and dtypes.
        Calendar validation, alignment and array conversion happen separately.
    """
    for argument, names in (("keys", keys), ("values", values)):
        if isinstance(names, str) or not isinstance(names, Sequence):
            raise TypeError(f"{argument} must be a sequence of column names, such as ['week']")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{argument} must contain only nonempty string column names. Got {names!r}")
    if not keys:
        raise ValueError("keys must include at least one column identifying observations, such as 'week'")

    columns = [*keys, *values]
    repeated = [name for name, count in Counter(columns).items() if count > 1]
    if repeated:
        raise ValueError(f"column declarations contain repeated names {repeated}. Select each column only once")

    # Collecting a lazy or distributed input here could unexpectedly load the entire dataset
    selected = nw.from_native(frame, eager_only=True)
    available = set(selected.columns)
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"data is missing columns {missing}. The available columns are {selected.columns}")
    selected = selected.select(columns)
    if selected.is_empty():
        raise ValueError("data must contain at least one observation")

    nulls = selected.null_count().rows(named=True)[0]
    missing_values = [name for name, count in nulls.items() if count]
    if missing_values:
        raise ValueError(f"columns {missing_values} contain missing values. Resolve them before preparing the data")

    schema = selected.schema
    invalid_types = {
        name: schema[name]
        for name in values
        if not (schema[name].is_integer() or schema[name].is_float() or schema[name] == nw.Boolean)
    }
    if invalid_types:
        raise TypeError(f"value columns must be integer, floating-point or boolean. Got {invalid_types}")

    # Null checks alone miss NaNs in Polars and Arrow, including NaNs used as observation keys
    float_columns = [name for name, dtype in schema.items() if dtype.is_float()]
    if float_columns:
        finite = selected.select(nw.col(*float_columns).is_finite().all()).rows(named=True)[0]
        nonfinite = [name for name, valid in finite.items() if not valid]
        if nonfinite:
            raise ValueError(f"columns {nonfinite} contain NaN or infinite values. Provide finite data")

    duplicates = selected.select(keys).is_duplicated()
    if duplicates.any():
        example = selected.select(keys).filter(duplicates).head(1).rows(named=True)[0]
        raise ValueError(f"data contains duplicate observations for {example}. Provide one row per key combination")

    return selected


def _calendar_dates(labels: Sequence[object], *, time: str, frequency: str) -> list[datetime]:
    """Parse calendar labels without changing their date or time of day."""
    # Calendar metadata is small even when the panel has many groups and channels
    times = []
    representations = set()
    for label in labels:
        if isinstance(label, str):
            representations.add("string")
            try:
                parsed = date.fromisoformat(label)
                if parsed.isoformat() != label:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"time column {time!r} contains {label!r}. Use valid dates written as YYYY-MM-DD "
                    "or convert the column to a date/datetime dtype"
                ) from error
            times.append(datetime.combine(parsed, datetime.min.time()))
        elif isinstance(label, datetime):
            representations.add("datetime")
            if label.utcoffset() is not None:
                raise ValueError(
                    f"time column {time!r} contains timezone-aware timestamps. "
                    "Convert them to observation dates in the intended timezone before preparing the panel"
                )
            times.append(label)
        elif isinstance(label, date):
            representations.add("date")
            times.append(datetime.combine(label, datetime.min.time()))
        else:
            raise TypeError(
                f"frequency={frequency!r} requires dates in time column {time!r}, got {label!r}. "
                "Use dates, timezone-naive datetimes, or YYYY-MM-DD strings"
            )

    if len(representations) > 1:
        raise TypeError(
            f"time column {time!r} mixes date representations. "
            "Convert the entire column to a single date/datetime dtype before preparing the panel"
        )

    return sorted(times)


def _time_positions(
    labels: tuple[object, ...], *, origin: float | datetime | None = None
) -> tuple[NDArray[np.float64], float | datetime]:
    """Convert time labels to fixed-origin positions, using days for dates."""
    if not labels:
        raise ValueError("Time positions require at least one observation time")
    numeric = all(
        isinstance(label, (int, float, np.integer, np.floating)) and not isinstance(label, (bool, np.bool_))
        for label in labels
    )
    if numeric:
        if isinstance(origin, datetime):
            raise TypeError("Prediction times must remain calendar dates as in training")
        times = np.asarray(labels, dtype=np.float64)
        if not np.all(np.isfinite(times)):
            raise ValueError("Observation time positions must be finite")
        numeric_origin = float(times.min()) if origin is None else origin
        return times - numeric_origin, numeric_origin

    if origin is not None and not isinstance(origin, datetime):
        raise TypeError("Prediction times must remain numeric and use the training time units")
    dates = _parse_dates(labels)
    date_origin = min(dates) if origin is None else origin
    positions = np.asarray([(value - date_origin).total_seconds() / 86400 for value in dates], dtype=np.float64)
    return positions, date_origin


def _day_of_year(labels: tuple[object, ...]) -> NDArray[np.float64]:
    """Anchor seasonal features to the calendar rather than to the first observation."""
    if not labels:
        raise ValueError("Day of year positions require at least one observation time")
    if all(
        isinstance(label, (int, float, np.integer, np.floating)) and not isinstance(label, (bool, np.bool_))
        for label in labels
    ):
        raise ValueError("day_of_year requires calendar dates. Observation times are numeric positions")
    return np.asarray([float(value.timetuple().tm_yday) for value in _parse_dates(labels)], dtype=np.float64)


def _parse_dates(labels: tuple[object, ...]) -> list[datetime]:
    """Read date labels in the forms accepted by prepare_data."""
    dates = []
    for label in labels:
        if isinstance(label, str):
            try:
                parsed = date.fromisoformat(label)
                if parsed.isoformat() != label:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"Invalid observation date {label!r}. Use YYYY-MM-DD strings or date columns"
                ) from error
            timestamp = datetime.combine(parsed, datetime.min.time())
        elif isinstance(label, datetime):
            timestamp = label
        elif isinstance(label, date):
            timestamp = datetime.combine(label, datetime.min.time())
        else:
            raise TypeError(
                "Observation times must be numeric positions, dates, naive datetimes, or YYYY-MM-DD strings"
            )
        if timestamp.utcoffset() is not None:
            raise ValueError("Convert timezone-aware timestamps to observation dates in the intended timezone")
        dates.append(timestamp)
    return dates


def _resolve_frequency(labels: Sequence[object], *, time: str, frequency: str | None) -> str | None:
    """Infer only complete supported calendars, leaving numeric periods unassigned."""
    if frequency != "auto":
        if frequency is not None:
            _validate_calendar(labels, time=time, frequency=frequency)
        return frequency
    if not any(isinstance(label, (str, date)) for label in labels):
        return None
    times = _calendar_dates(labels, time=time, frequency=frequency)
    if len(times) < 3:
        return None
    for candidate in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        try:
            _validate_calendar_spacing(times, time=time, frequency=candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError(
        f"Cannot infer a supported calendar frequency from time column {time!r}. "
        "Check for missing or irregular periods, specify the expected frequency, "
        "or set frequency=None to skip calendar checks"
    )


def _validate_calendar(labels: Sequence[object], *, time: str, frequency: str) -> None:
    """Check the distinct observation times against a declared calendar spacing."""
    frequencies = ("daily", "weekly", "monthly", "quarterly", "yearly")
    if frequency not in frequencies:
        raise ValueError(f"frequency must be one of {frequencies}, got {frequency!r}")
    _validate_calendar_spacing(_calendar_dates(labels, time=time, frequency=frequency), time=time, frequency=frequency)


def _validate_calendar_spacing(times: Sequence[datetime], *, time: str, frequency: str) -> None:
    """Check every period against the same calendar anchor."""
    first = times[0]
    months = {"monthly": 1, "quarterly": 3, "yearly": 12}.get(frequency)
    at_month_end = first.day == monthrange(first.year, first.month)[1]
    for index, observed in enumerate(times[1:], start=1):
        if months is None:
            expected = first + timedelta(days=index * (7 if frequency == "weekly" else 1))
        else:
            # Anchor to the first observation so February does not shift later month-end labels
            year, month = divmod(first.year * 12 + first.month - 1 + index * months, 12)
            last_day = monthrange(year, month + 1)[1]
            day = last_day if at_month_end else min(first.day, last_day)
            expected = first.replace(year=year, month=month + 1, day=day)
        if observed != expected:
            raise ValueError(
                f"time column {time!r} does not follow frequency={frequency!r}, "
                f"expected {expected.isoformat(sep=' ')}, found {observed.isoformat(sep=' ')}. "
                "Check for missing periods or an incorrect frequency"
            )
