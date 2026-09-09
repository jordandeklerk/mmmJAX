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
        Time-varying inputs are ordered by time, group if supplied, then
        feature. Outcome and revenue per outcome have no final feature axis.
        Population is stored once per group, or as a scalar without groups.
        Each array is independent of the source dataframe and other inputs.
        All paid and organic exposure inputs, including reach and frequency,
        include any earlier history. Other time-varying inputs cover only
        the modeling periods.
    time_column : str
        Source column identifying observation periods.
    time_values : tuple
        Sorted modeling periods for outcome, revenue per outcome, both spend
        inputs, controls, and treatments.
    media_time_values : tuple
        Shared time labels for all paid and organic exposure inputs, including
        any earlier history. Without history these match ``time_values``.
        Empty when no exposure inputs were supplied.
    group_columns : tuple of str
        Source columns identifying each series. Empty for a single series.
    group_values : tuple of tuple
        Observed group combinations matching the group axis. Preparation
        uses first-appearance order. Empty when no group columns were supplied.
    columns : dict of str to tuple of str
        Source columns for each input, in the selected order. Outcome,
        revenue per outcome, and population each use a one-element tuple.
    channels : tuple of str
        Shared channel labels for the final axis of media and spend.
        Empty when media was not supplied.
    organic_channels : tuple of str
        Channel labels for the final axis of organic media, independent of
        paid channel labels. Empty when organic media was not supplied.
    rf_channels : tuple of str
        Shared channel labels for reach, media frequency, and their spend.
        These use a separate axis from media and organic media channels.
        Empty when reach and frequency inputs were not supplied.
    organic_rf_channels : tuple of str
        Shared channel labels for organic reach and frequency. These use a
        separate axis from the other media inputs. Empty when organic reach
        and frequency inputs were not supplied.

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
    rf_channels: tuple[str, ...] = ()
    organic_rf_channels: tuple[str, ...] = ()

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
            - **revenue_per_outcome** : Revenue per response unit for each modeling period and group
            - **population** : One population estimate per group, or a scalar without groups
            - **media** : Media values, including any earlier history
            - **organic_media** : Organic exposure values over the same media periods
            - **reach** : Audience reached, including any earlier history
            - **media_frequency** : Average exposures per person over the same media periods
            - **organic_reach** : Audience reached by organic channels over the media periods
            - **organic_frequency** : Average organic exposures per person over the media periods
            - **spend** : Spending for the modeling periods, ordered by channel
            - **rf_spend** : Spending for reach and frequency channels over modeling periods
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
        )


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
    frequency: str | None = None,
) -> PreparedData:
    """Prepare a dataframe for modeling while keeping its observation labels.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals, including pandas and Polars
        DataFrames and PyArrow Tables. Selected values must be numeric or
        boolean, finite, and nonmissing. The input is not modified.
        For predictions that need only dates and group labels, omit all
        value selections. The result retains the observation labels without
        adding placeholder outcomes or other inputs.
    time : str
        Column identifying observation periods, with values that sort
        chronologically. Each combination of time and group must be unique.
        Calendar checks accept dates, timezone-naive datetimes, or strings
        written as ``YYYY-MM-DD``. Original labels are retained.
    outcome : str, optional
        Column containing the response, such as sales or conversions.
        Omit it when preparing prediction data without observed outcomes.
    revenue_per_outcome : str, optional
        Column containing the average revenue per response unit, such as
        revenue per sale or conversion. Values can vary by period and group
        and must be numeric, finite, and nonnegative, not boolean.
        Stored separately without converting the outcome to revenue.
        Can also be supplied for prediction data without observed outcomes.
    population : str, optional
        Column containing a positive population estimate for each group.
        Repeat the same value across that group's modeling periods. Without
        groups, supply one value repeated across all periods. Integer and
        floating-point estimates are accepted, but boolean values are not.
        Population is stored once per group, or as a scalar without groups.
        Preparation does not scale other inputs or average changing values.
    media : sequence of str, optional
        Columns containing nonnegative paid media inputs, such as impressions
        or spending. Their order defines the channel axis. Use a list
        even for one channel.
    organic_media : sequence of str, optional
        Columns containing nonnegative exposure from unpaid media, such
        as email clicks or impressions from organic social posts. Their
        order defines a separate organic channel axis. Use a list even
        for one channel. Paid media and spend are not required.
    reach : sequence of str, optional
        Columns containing the nonnegative audience reached by each paid
        channel in each period. Supply these together with ``media_frequency``
        in matching channel order. These channels have their own axis and
        do not require a separate ``media`` selection.
    media_frequency : sequence of str, optional
        Columns containing the nonnegative average number of exposures per
        person reached. Supply one column per ``reach`` channel in the same
        order. This describes advertising exposure, not the observation
        calendar controlled by ``frequency``.
    organic_reach : sequence of str, optional
        Columns containing the nonnegative audience reached by unpaid channels,
        such as readers of an email newsletter. Supply these together with
        ``organic_frequency`` in matching channel order. They have their own
        channel axis and do not require paid media, spend, or ``organic_media``.
    organic_frequency : sequence of str, optional
        Columns containing the nonnegative average number of organic exposures
        per person reached. Supply one column per ``organic_reach`` channel
        in the same order. This is separate from the observation calendar
        controlled by ``frequency``.
    media_history : dataframe-like, optional
        Earlier exposure observations used to calculate carryover into the
        first modeling periods. Include every selected paid and organic
        exposure column, including both reach and frequency where supplied,
        with the same time and group columns as ``frame``. All exposure
        inputs share this history window.
        Include only periods before ``frame`` and all its groups. Outcomes,
        revenue per outcome, population, controls, treatments, and separate
        spend columns are not required.
        Supply ``frequency`` to check for missing periods across both
        dataframes. Requires ``media``, ``organic_media``, ``reach``, or
        ``organic_reach``.
    spend : sequence of str, optional
        Columns containing nonnegative spending for the selected media.
        Supply one column per media channel in the same order. If media
        already contains spending, the same columns can be selected here.
        Omit this argument when separate spending inputs are not needed.
    rf_spend : sequence of str, optional
        Columns containing nonnegative spending for reach and frequency
        channels. Supply one column per ``reach`` channel in the same order.
        These values cover only the modeling periods. Omit this argument
        when spending inputs are not needed.
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
    rf_channels : sequence of str, optional
        Unique names shared by reach, media frequency, and their spend.
        These match the selected columns by position and default to the
        ``reach`` column names. Requires ``reach`` and ``media_frequency``.
    organic_rf_channels : sequence of str, optional
        Unique names shared by organic reach and frequency. These match the
        selected columns by position and default to the ``organic_reach``
        column names. Requires ``organic_reach`` and ``organic_frequency``.
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
          ``revenue_per_outcome``, ``population``, ``media``, ``organic_media``, ``reach``,
          ``media_frequency``, ``organic_reach``, ``organic_frequency``,
          ``spend``, ``rf_spend``, ``controls``, and ``treatments``.
          Omitted inputs have no entry
        - **time_column** : Name of the source column identifying time periods
        - **time_values** : Sorted modeling periods for outcome, revenue per
          outcome, both spend inputs, controls, and treatments
        - **media_time_values** : Shared periods for all paid and organic
          exposure inputs, including any earlier history. Matches
          ``time_values`` without history. Empty without exposure inputs
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
        - **rf_channels** : Shared reach, media frequency, and spend channel
          names in array order. Empty without reach and frequency inputs
        - **organic_rf_channels** : Shared organic reach and frequency channel
          names in array order. Empty without organic reach and frequency inputs

        Time-varying arrays are ordered by time, group (if supplied), then
        feature. Outcome and revenue per outcome have no final feature axis.
        Population has shape ``(n_groups,)``, or ``()`` without groups, and
        follows ``group_values``.
        All other inputs keep a feature axis even for a single column.
        Integer outcomes and population estimates retain their dtype
        separately from continuous inputs. Columns within an input use NumPy
        type promotion. Arrays do not share memory with either dataframe.

    Examples
    --------
    Prepare weekly sales for two regions with paid and organic media,
    earlier exposure history, and a population estimate for each region.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import prepare_data
           ...: # Search uses impressions, while video uses reach and frequency
           ...: # Email and social are organic channels without associated spend
           ...: # Sales counts units sold, and unit_revenue is revenue per sale
           ...: df = pl.DataFrame({
           ...:     "week": ["2026-01-05", "2026-01-05",
           ...:              "2026-01-12", "2026-01-12"],
           ...:     "region": ["west", "east", "west", "east"],
           ...:     "sales": [100, 80, 140, 90],
           ...:     "unit_revenue": [11.5, 10.5, 9.5, 10.0],
           ...:     "residents": [50_000, 30_000, 50_000, 30_000],
           ...:     "search_impressions": [5_000, 3_000, 6_000, 4_000],
           ...:     "search_spend": [40.0, 30.0, 60.0, 45.0],
           ...:     "video_reach": [5_000, 3_000, 7_000, 4_000],
           ...:     "video_frequency": [2.0, 1.5, 2.5, 2.0],
           ...:     "video_spend": [80.0, 50.0, 130.0, 70.0],
           ...:     "email_clicks": [60, 40, 90, 50],
           ...:     "social_reach": [300, 200, 350, 250],
           ...:     "social_frequency": [1.0, 1.5, 2.0, 1.5],
           ...:     "temperature": [10.0, 8.0, 12.0, 9.0],
           ...:     "product_price": [12.0, 11.0, 10.0, 11.0],
           ...: })
           ...: # History needs the same exposure columns, but no sales or spend
           ...: history = pl.DataFrame({
           ...:     "week": ["2025-12-29", "2025-12-29"],
           ...:     "region": ["west", "east"],
           ...:     "search_impressions": [4_000, 2_500],
           ...:     "video_reach": [4_000, 2_500],
           ...:     "video_frequency": [1.5, 1.0],
           ...:     "email_clicks": [45, 30],
           ...:     "social_reach": [250, 150],
           ...:     "social_frequency": [1.0, 1.0],
           ...: })
           ...: # Select each input and give its channels readable names
           ...: # Product price is a treatment, and temperature is a control
           ...: data = prepare_data(
           ...:     df,
           ...:     time="week",
           ...:     groups=["region"],
           ...:     frequency="weekly",
           ...:     outcome="sales",
           ...:     revenue_per_outcome="unit_revenue",
           ...:     population="residents",
           ...:     media=["search_impressions"],
           ...:     spend=["search_spend"],
           ...:     channels=["search"],
           ...:     reach=["video_reach"],
           ...:     media_frequency=["video_frequency"],
           ...:     rf_spend=["video_spend"],
           ...:     rf_channels=["video"],
           ...:     organic_media=["email_clicks"],
           ...:     organic_channels=["email"],
           ...:     organic_reach=["social_reach"],
           ...:     organic_frequency=["social_frequency"],
           ...:     organic_rf_channels=["social"],
           ...:     controls=["temperature"],
           ...:     treatments=["product_price"],
           ...:     media_history=history,
           ...: )
           ...: data.group_values
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
        frequency=frequency if media_history is None else None,
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
