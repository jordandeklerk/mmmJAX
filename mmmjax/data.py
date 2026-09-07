"""Dataframe preparation for marketing mix models."""

from calendar import monthrange
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    validation and alignment. Constructing it directly does not validate data.

    Attributes
    ----------
    arrays : dict of str to numpy.ndarray
        Named numerical blocks, ordered by time, group if supplied, then
        feature for sequence selections. Each array is independent of
        the source dataframe and other blocks.
    time_column : str
        Source column identifying observation periods.
    time_values : tuple
        Sorted time labels corresponding to the first array axis.
    group_columns : tuple of str
        Source columns identifying each series. Empty for a single series.
    group_values : tuple of tuple
        Observed group combinations in first-appearance order, matching
        the group axis. Empty when no group columns were supplied.
    columns : dict of str to tuple of str
        Source columns for each block, in the selected order. A single
        column selection is recorded as a one-element tuple.

    Notes
    -----
    The arrays and dictionaries remain editable. Keep their shapes and
    ordering consistent with the labels. :meth:`to_jax` produces an
    independent dictionary of JAX arrays for model calculations.
    """

    arrays: dict[str, NDArray[np.generic]]
    time_column: str
    time_values: tuple[object, ...]
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]
    columns: dict[str, tuple[str, ...]]

    def to_jax(
        self,
        *,
        dtype: DTypeLike = float,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ) -> dict[str, jax.Array]:
        """Convert numerical blocks to JAX arrays without transferring labels.

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
            Named arrays with unchanged shapes and axis order. The host
            arrays and labels are retained on this object. Call this once
            before using JAX transformations on a model.
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


def prepare_data(
    frame: IntoDataFrameT,
    *,
    time: str,
    blocks: Mapping[str, str | Sequence[str]],
    groups: Sequence[str] = (),
    frequency: str | None = None,
) -> PreparedData:
    """Prepare a dataframe for modeling while keeping its observation labels.

    Parameters
    ----------
    frame : dataframe-like
        Eager dataframe supported by Narwhals, including pandas and Polars
        DataFrames and PyArrow Tables. Block values must be numeric or
        boolean, finite, and nonmissing. The input is not modified.
    time : str
        Column identifying observation periods, with values that sort
        chronologically. Each combination of time and group must be unique.
        Calendar checks accept dates, timezone-naive datetimes, or strings
        written as ``YYYY-MM-DD``. Original labels are retained.
    blocks : mapping of str to str or sequence of str
        Names for model inputs and their source columns, for example
        ``{"outcome": "sales", "media": ["search", "video"]}``. A column
        name produces one value per observation. A sequence keeps a final
        feature axis, even for a single column. Columns may appear in more
        than one block, but cannot be repeated within the same block.
    groups : sequence of str, optional
        Columns identifying each observed series, such as ``["region"]``.
        Every observed group must have the same time periods. Group
        combinations retain their first-appearance order and share one
        array axis. Omit this argument for a single series with no group axis.
    frequency : str, optional
        Expected spacing given as ``"daily"``, ``"weekly"``, ``"monthly"``,
        ``"quarterly"``, or ``"yearly"``. Calendar periods follow the first
        observation's day, or month-end if it starts at month-end. Supply
        this to detect periods missing from every group. Without it, only
        coverage of the observed times is checked.

    Returns
    -------
    PreparedData
        Arrays ordered as time, group (if supplied), then feature (for
        sequence selections). Labels follow that exact order. Each block
        has its own dtype. Columns within a block use NumPy type promotion.
        Keep count outcomes in a separate block from continuous features
        to preserve their integer dtype. Arrays do not share memory with
        the input dataframe.

    Notes
    -----
    All blocks share the same observation periods. Missing observations
    are reported, not filled or treated as zero. Calendar checks cover only
    the span between the first and last supplied times.

    Block names describe your model inputs but do not add role-specific
    checks. For example, naming a block ``"media"`` does not require its
    values to be nonnegative or associate it with spending columns.

    Each call prepares its input independently. Group and feature ordering
    must match the model before using separately prepared prediction data.
    Call :meth:`PreparedData.to_jax` once before model calculations. Dataframe
    preparation belongs outside JAX transformations such as ``jax.jit``.

    Examples
    --------
    Select an outcome and two media channels from weekly observations.
    Rows are sorted by week while channels keep the requested order.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import prepare_data
           ...: spend = pl.DataFrame({
           ...:     "week": ["2026-01-12", "2026-01-05", "2026-01-19"],
           ...:     "sales": [140, 100, 120],
           ...:     "search": [60.0, 40.0, 50.0],
           ...:     "video": [80.0, 60.0, 70.0],
           ...: })

        In [2]: data = prepare_data(
           ...:     spend,
           ...:     time="week",
           ...:     blocks={
           ...:         "outcome": "sales",
           ...:         "media": ["video", "search"],
           ...:     },
           ...:     frequency="weekly",
           ...: )
           ...: data.columns["media"]

    Convert the numerical blocks for use in a model. Labels remain on
    ``data`` so you can identify the observations and channels later.

    .. ipython::

        In [3]: inputs = data.to_jax()
           ...: inputs["media"]
    """
    if not isinstance(blocks, Mapping) or not blocks:
        raise TypeError("blocks must be a nonempty mapping, such as {'outcome': 'sales', 'media': ['video']}")

    columns: dict[str, tuple[str, ...]] = {}
    for name, selection in blocks.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"block names must be nonempty strings, got {name!r}")
        if isinstance(selection, str):
            names: tuple[str, ...] = (selection,)
        elif isinstance(selection, Sequence):
            names = tuple(selection)
        else:
            raise TypeError(f"block {name!r} must select a column name or a sequence of column names")
        if not names or any(not isinstance(column, str) or not column for column in names):
            raise ValueError(f"block {name!r} must select at least one nonempty string column name. Got {selection!r}")
        if len(set(names)) != len(names):
            raise ValueError(
                f"block {name!r} contains repeated columns {names}. Select each column only once per block"
            )
        columns[name] = names

    selected = _prepare_panel(
        frame,
        time=time,
        groups=groups,
        values=list(dict.fromkeys(column for names in columns.values() for column in names)),
        frequency=frequency,
    )
    time_values = tuple(selected.get_column(time).unique(maintain_order=True).to_list())
    group_values = tuple(selected.select(groups).unique(maintain_order=True).rows()) if groups else ()
    observation_shape = (len(time_values), len(group_values)) if groups else (len(time_values),)

    arrays: dict[str, NDArray[np.generic]] = {}
    for name, names in columns.items():
        if isinstance(blocks[name], str):
            arrays[name] = selected.get_column(names[0]).to_numpy().copy().reshape(observation_shape)
        else:
            # Stacking numeric series avoids object arrays from mixed pandas column dtypes
            values = np.stack([selected.get_column(column).to_numpy() for column in names], axis=-1)
            arrays[name] = values.reshape(*observation_shape, len(names))

    return PreparedData(
        arrays=arrays,
        time_column=time,
        time_values=time_values,
        group_columns=tuple(groups),
        group_values=group_values,
        columns=columns,
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
        _validate_calendar(selected, time=time, frequency=frequency)
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


def _validate_calendar(frame: nw.DataFrame[IntoDataFrameT], *, time: str, frequency: str) -> None:
    """Check the distinct observation times against a declared calendar spacing."""
    frequencies = ("daily", "weekly", "monthly", "quarterly", "yearly")
    if frequency not in frequencies:
        raise ValueError(f"frequency must be one of {frequencies}, got {frequency!r}")

    # Calendar metadata is small even when the panel has many groups and channels
    labels = frame.get_column(time).unique().to_list()
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
