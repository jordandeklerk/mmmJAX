"""Dataframe preparation for marketing mix models."""

from calendar import monthrange
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import cast

import jax
import narwhals as nw
import numpy as np
from jax.typing import DTypeLike
from narwhals.typing import IntoDataFrameT
from numpy.typing import NDArray

__all__ = ["PreparedData", "prepare_data"]


@dataclass(frozen=True, slots=True, eq=False)
class PreparedData:
    """Store prepared model inputs together with their observation labels.

    Use :func:`prepare_data` to create this object from a dataframe with
    validation and labeled arrays. Constructing it directly does not validate data.

    Attributes
    ----------
    arrays : dict of str to numpy.ndarray
        Selected outcome, media, organic media, spend, controls, and
        treatments, ordered by time, group if supplied, then feature.
        Outcome has no final feature axis. Each array is independent of
        the source dataframe and other inputs. Paid and organic media
        include earlier observations when history is supplied. Other
        inputs cover only the modeling periods.
    time_column : str
        Source column identifying observation periods.
    time_values : tuple
        Sorted modeling periods for outcome, spend, controls, and treatments.
    media_time_values : tuple
        Shared time labels for paid and organic media, including any earlier
        history. Without history these match ``time_values``. Empty when
        neither media input was supplied.
    group_columns : tuple of str
        Source columns identifying each series. Empty for a single series.
    group_values : tuple of tuple
        Observed group combinations matching the group axis. Preparation
        uses first-appearance order. Empty when no group columns were supplied.
    columns : dict of str to tuple of str
        Source columns for each input, in the selected order. The outcome
        column is recorded as a one-element tuple.
    channels : tuple of str
        Shared channel labels for the final axis of media and spend.
        Empty when media was not supplied.
    organic_channels : tuple of str
        Channel labels for the final axis of organic media, independent of
        paid channel labels. Empty when organic media was not supplied.

    Notes
    -----
    The arrays and dictionaries remain editable. Keep their shapes and
    ordering consistent with the labels.
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

    def _to_jax(
        self,
        *,
        dtype: DTypeLike = float,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ) -> dict[str, jax.Array]:
        """Convert numerical inputs for internal model evaluation.

        Parameters
        ----------
        dtype : data-type, default float
            Precision for floating-point blocks, either float32 or float64.
            The default ``float`` follows JAX's precision setting. Explicit
            float64 requires JAX 64-bit mode to be enabled. Integer and
            boolean blocks stay integer and boolean. Integers that cannot
            fit JAX's available dtype raise an error rather than wrapping.
        device : jax.Device or jax.sharding.Sharding, optional
            Destination for the arrays. If omitted, JAX chooses the device.

        Returns
        -------
        dict of str to jax.Array
            Dictionary containing the supplied inputs:

            - **outcome** : Observed response values without a feature axis
            - **media** : Media values, including any earlier history
            - **organic_media** : Organic exposure values over the same media periods
            - **spend** : Spending for the modeling periods, ordered by channel
            - **controls** : Additional predictors in the selected column order
            - **treatments** : Non-media inputs in the selected column order

            Omitted inputs have no entry. Shapes and axis order are unchanged.
            The host arrays and labels are retained on this object. Call this
            once before using JAX transformations on a model.
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

    def _align_to(self, reference: "PreparedData") -> "PreparedData":
        """Match reference ordering for internal prediction preparation."""
        if not isinstance(reference, PreparedData):
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

        unknown = [name for name in self.arrays if name not in reference.arrays]
        if unknown:
            raise ValueError(f"inputs {unknown} do not exist in the reference. Supply inputs used by the model")

        arrays: dict[str, NDArray[np.generic]] = {}
        columns: dict[str, tuple[str, ...]] = {}
        for name, reference_array in reference.arrays.items():
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
            if array.ndim != reference_array.ndim:
                raise ValueError(
                    f"input {name!r} has a different number of axes from the reference. "
                    "Prepare both inputs with prepare_data without changing their array shapes"
                )

            indices = observation_indices
            if name in ("media", "organic_media"):
                indices = [np.arange(len(self.media_time_values)), *observation_indices[1:]]
            if array.ndim > len(observation_indices):
                feature_order = [column_positions[column] for column in reference_columns]
                if name in ("media", "spend", "organic_media"):
                    channel_labels = self.organic_channels if name == "organic_media" else self.channels
                    reference_labels = reference.organic_channels if name == "organic_media" else reference.channels
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
            media_time_values=self.media_time_values if "media" in arrays or "organic_media" in arrays else (),
            group_columns=self.group_columns,
            group_values=reference.group_values,
            columns=columns,
            channels=reference.channels if "media" in arrays else (),
            organic_channels=reference.organic_channels if "organic_media" in arrays else (),
        )


def prepare_data(
    frame: IntoDataFrameT,
    *,
    time: str,
    outcome: str | None = None,
    media: Sequence[str] | None = None,
    organic_media: Sequence[str] | None = None,
    media_history: IntoDataFrameT | None = None,
    spend: Sequence[str] | None = None,
    controls: Sequence[str] | None = None,
    treatments: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    organic_channels: Sequence[str] | None = None,
    groups: Sequence[str] = (),
    frequency: str | None = None,
) -> PreparedData:
    """Prepare a dataframe for modeling while keeping its observation labels.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals, including pandas and Polars
        DataFrames and PyArrow Tables. Selected values must be numeric or
        boolean, finite, and nonmissing. The input is not modified.
    time : str
        Column identifying observation periods, with values that sort
        chronologically. Each combination of time and group must be unique.
        Calendar checks accept dates, timezone-naive datetimes, or strings
        written as ``YYYY-MM-DD``. Original labels are retained.
    outcome : str, optional
        Column containing the response, such as sales or conversions.
        Omit it when preparing prediction data without observed outcomes.
    media : sequence of str, optional
        Columns containing nonnegative paid media inputs, such as impressions
        or spending. Their order defines the channel axis. Use a list
        even for one channel.
    organic_media : sequence of str, optional
        Columns containing nonnegative exposure from unpaid media, such
        as email clicks or impressions from organic social posts. Their
        order defines a separate organic channel axis. Use a list even
        for one channel. Paid media and spend are not required.
    media_history : dataframe-like, optional
        Earlier paid and organic media observations used to calculate
        carryover into the first modeling periods. Include every selected
        ``media`` and ``organic_media`` column, with the same time and group
        columns as ``frame``. Both media inputs share this history window.
        Include only periods before ``frame`` and all its groups. Outcomes,
        controls, treatments, and separate spend columns are not required.
        Supply ``frequency`` to check for missing periods across both
        dataframes. Requires ``media`` or ``organic_media``.
    spend : sequence of str, optional
        Columns containing nonnegative spending for the selected media.
        Supply one column per media channel in the same order. If media
        already contains spending, the same columns can be selected here.
        Omit this argument when separate spending inputs are not needed.
    controls : sequence of str, optional
        Columns containing adjustment variables, such as temperature or
        economic indicators. Their order defines the control axis.
        Negative values are allowed for controls and the outcome.
    treatments : sequence of str, optional
        Columns containing non-media inputs whose effects the model will
        estimate, such as product prices or promotions. Their order defines
        the treatment axis. Numeric and boolean values are accepted,
        including negative values. Use a list even for one treatment.
        These inputs cover only the modeling periods and do not require
        media or spend. The model determines how their effects are represented.
    channels : sequence of str, optional
        Unique channel names shared by media and spend, such as
        ``["video", "search"]``. These match the selected columns by
        position. Defaults to the media column names. Requires ``media``.
    organic_channels : sequence of str, optional
        Unique names for the organic media channels, such as
        ``["email", "social"]``. These match ``organic_media`` columns by
        position and default to their column names. Requires ``organic_media``.
    groups : sequence of str, optional
        Columns identifying each observed series, such as ``["region"]``.
        Every observed group must have the same time periods. Group
        combinations retain their first-appearance order and share one
        array axis. Omit this argument for a single series with no group axis.
    frequency : str, optional
        Expected spacing given as ``"daily"``, ``"weekly"``, ``"monthly"``,
        ``"quarterly"``, or ``"yearly"``. Calendar periods follow the first
        observation's day, including history if supplied, or month-end if
        it starts at month-end. Supply this to detect periods missing from
        every group. Without it, only coverage of the observed times is
        checked.

    Returns
    -------
    PreparedData
        Prepared inputs containing:

        - **arrays** : Dictionary of NumPy arrays for the supplied ``outcome``,
          ``media``, ``organic_media``, ``spend``, ``controls``, and
          ``treatments``. Omitted inputs have no entry
        - **time_column** : Name of the source column identifying time periods
        - **time_values** : Sorted modeling periods for outcome, spend,
          controls, and treatments
        - **media_time_values** : Shared paid and organic media periods,
          including any earlier history. Matches ``time_values`` without
          history. Empty when neither media input was supplied
        - **group_columns** : Selected group-column names. Empty for a single
          series without groups
        - **group_values** : Observed group-label tuples in array order,
          preserving first appearance in ``frame``. Empty without groups
        - **columns** : Dictionary mapping each supplied input to its source
          column names in the selected order
        - **channels** : Channel names shared by media and spend in array
          order. Empty without media
        - **organic_channels** : Organic channel names in array order.
          Empty without organic media

        Arrays are ordered by time, group (if supplied), then feature.
        Outcome has no final feature axis. All other inputs keep that axis
        even for a single column. Integer outcomes retain their dtype
        separately from continuous inputs. Columns within an input use NumPy
        type promotion. Arrays do not share memory with either dataframe.

    Examples
    --------
    Prepare weekly sales with paid impressions, spending, and organic email
    clicks. Use product price as a treatment and temperature as a control.
    Rows are sorted by week while features keep the requested order.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import prepare_data
           ...: df = pl.DataFrame({
           ...:     "week": ["2026-01-12", "2026-01-05", "2026-01-19"],
           ...:     "sales": [140, 100, 120],
           ...:     "video_impressions": [14_000, 10_000, 12_000],
           ...:     "video_spend": [130.0, 80.0, 95.0],
           ...:     "search_impressions": [6_000, 5_000, 4_500],
           ...:     "search_spend": [60.0, 40.0, 50.0],
           ...:     "email_clicks": [90, 60, 80],
           ...:     "temperature": [12.0, 10.0, 14.0],
           ...:     "product_price": [10.0, 12.0, 11.0],
           ...: })

        In [2]: data = prepare_data(
           ...:     df,
           ...:     time="week",
           ...:     outcome="sales",
           ...:     media=["video_impressions", "search_impressions"],
           ...:     organic_media=["email_clicks"],
           ...:     spend=["video_spend", "search_spend"],
           ...:     controls=["temperature"],
           ...:     treatments=["product_price"],
           ...:     channels=["video", "search"],
           ...:     organic_channels=["Email"],
           ...:     frequency="weekly",
           ...: )
           ...: data.channels, data.organic_channels

    To include earlier exposure data, pass it as ``media_history`` without
    adding historical sales values. Include both the paid impressions and
    organic email clicks.

    .. ipython::

        In [3]: history = pl.DataFrame({
           ...:     "week": ["2025-12-22", "2025-12-29"],
           ...:     "video_impressions": [8_000, 9_000],
           ...:     "search_impressions": [3_000, 4_000],
           ...:     "email_clicks": [30, 45],
           ...: })
           ...: data = prepare_data(
           ...:     df,
           ...:     time="week",
           ...:     outcome="sales",
           ...:     media=["video_impressions", "search_impressions"],
           ...:     organic_media=["email_clicks"],
           ...:     spend=["video_spend", "search_spend"],
           ...:     controls=["temperature"],
           ...:     treatments=["product_price"],
           ...:     channels=["video", "search"],
           ...:     organic_channels=["Email"],
           ...:     frequency="weekly",
           ...:     media_history=history,
           ...: )
           ...: data.media_time_values
    """
    selections = {
        "outcome": outcome,
        "media": media,
        "organic_media": organic_media,
        "spend": spend,
        "controls": controls,
        "treatments": treatments,
    }
    columns: dict[str, tuple[str, ...]] = {}
    for name, selection in selections.items():
        if selection is None:
            continue
        if name == "outcome":
            if not isinstance(selection, str):
                raise TypeError("outcome must be a column name, such as 'sales'")
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

    media_inputs = tuple(name for name in ("media", "organic_media") if name in columns)
    if media_history is not None and not media_inputs:
        raise ValueError(
            "media_history requires media or organic_media columns. Select the same media columns in both dataframes"
        )
    if not columns:
        raise ValueError(
            "select at least one of outcome, media, organic_media, controls or treatments when preparing data"
        )
    if spend is not None and "media" not in columns:
        raise ValueError("spend requires media columns so spending can be associated with each channel")
    if spend is not None and len(columns["spend"]) != len(columns["media"]):
        raise ValueError("spend must select one column per media channel in the same order")

    channel_names: dict[str, tuple[str, ...]] = {}
    for name, selection, media_input in (
        ("channels", channels, "media"),
        ("organic_channels", organic_channels, "organic_media"),
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
        frequency=frequency if media_history is None else None,
    )
    time_values = tuple(selected.get_column(time).unique(maintain_order=True).to_list())
    group_values = tuple(selected.select(groups).unique(maintain_order=True).rows()) if groups else ()
    observation_shape = (len(time_values), len(group_values)) if groups else (len(time_values),)

    arrays: dict[str, NDArray[np.generic]] = {}
    for name, names in columns.items():
        if name == "outcome":
            arrays[name] = selected.get_column(names[0]).to_numpy().copy().reshape(observation_shape)
        else:
            # Stacking numeric series avoids object arrays from mixed pandas column dtypes
            values = np.stack([selected.get_column(column).to_numpy() for column in names], axis=-1)
            arrays[name] = values.reshape(*observation_shape, len(names))
        if name in ("media", "organic_media", "spend"):
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
    )
    if media_history is None:
        return data

    history = prepare_data(
        media_history,
        time=time,
        groups=groups,
        media=columns.get("media"),
        organic_media=columns.get("organic_media"),
        channels=channel_names["media"] if "media" in columns else None,
        organic_channels=channel_names["organic_media"] if "organic_media" in columns else None,
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
    if frequency is not None:
        _validate_calendar(media_times, time=time, frequency=frequency)
    return replace(
        data,
        arrays={
            **arrays,
            **{name: np.concatenate((history.arrays[name], arrays[name]), axis=0) for name in media_inputs},
        },
        media_time_values=media_times,
    )


def _prepare_panel(
    frame: IntoDataFrameT,
    *,
    time: str,
    groups: Sequence[str] = (),
    values: Sequence[str],
    frequency: str | None = None,
) -> nw.DataFrame[IntoDataFrameT]:
    """Arrange a complete observation panel in time-major order.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals. The input is not modified.
    time : str
        Time column, represented in a form that sorts chronologically.
        Calendar checks accept dates, timezone-naive datetimes, or date
        strings written as ``YYYY-MM-DD``. The original labels are retained.
    groups : sequence of str, optional
        Columns identifying each observed series. Combinations retain their
        first-appearance order across time. With no groups, the input is a
        single series.
    values : sequence of str
        Numeric or boolean columns to retain in the supplied order.
    frequency : str, optional
        Expected spacing given as ``"daily"``, ``"weekly"``, ``"monthly"``,
        ``"quarterly"``, or ``"yearly"``. Calendar periods follow the first
        observation's day, or month-end if it starts at month-end. Without
        this argument, only coverage of the observed times is checked.

    Returns
    -------
    narwhals.DataFrame
        Rows sorted by time, with the same observed groups in the same order
        at every time. Nested group labels stay together rather than forming
        every possible combination. No missing observations are filled.

    Notes
    -----
    Supply ``frequency`` to detect periods missing from every group. Missing
    observations are reported, not filled or treated as zero. Only the span
    between the first and last supplied times can be checked.
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


def _validate_calendar(labels: Sequence[object], *, time: str, frequency: str) -> None:
    """Check the distinct observation times against a declared calendar spacing."""
    frequencies = ("daily", "weekly", "monthly", "quarterly", "yearly")
    if frequency not in frequencies:
        raise ValueError(f"frequency must be one of {frequencies}, got {frequency!r}")

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

    times.sort()
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
