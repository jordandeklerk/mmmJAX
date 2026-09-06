"""Dataframe preparation for marketing mix models."""

from collections import Counter
from collections.abc import Sequence

import narwhals as nw
from narwhals.typing import IntoDataFrameT


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
        must be finite and nonmissing; negative values are allowed.

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
            raise ValueError(f"{argument} must contain only nonempty string column names; got {names!r}")
    if not keys:
        raise ValueError("keys must include at least one column identifying observations, such as 'week'")

    columns = [*keys, *values]
    repeated = [name for name, count in Counter(columns).items() if count > 1]
    if repeated:
        raise ValueError(f"column declarations contain repeated names {repeated}; select each column only once")

    # Collecting a lazy or distributed input here could unexpectedly load the entire dataset
    selected = nw.from_native(frame, eager_only=True)
    available = set(selected.columns)
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"data is missing columns {missing}; available columns are {selected.columns}")
    selected = selected.select(columns)
    if selected.is_empty():
        raise ValueError("data must contain at least one observation")

    nulls = selected.null_count().rows(named=True)[0]
    missing_values = [name for name, count in nulls.items() if count]
    if missing_values:
        raise ValueError(f"columns {missing_values} contain missing values; resolve them before preparing the data")

    schema = selected.schema
    invalid_types = {
        name: schema[name]
        for name in values
        if not (schema[name].is_integer() or schema[name].is_float() or schema[name] == nw.Boolean)
    }
    if invalid_types:
        raise TypeError(f"value columns must be integer, floating-point or boolean; got {invalid_types}")

    # Null checks alone miss NaNs in Polars and Arrow, including NaNs used as observation keys
    float_columns = [name for name, dtype in schema.items() if dtype.is_float()]
    if float_columns:
        finite = selected.select(nw.col(*float_columns).is_finite().all()).rows(named=True)[0]
        nonfinite = [name for name, valid in finite.items() if not valid]
        if nonfinite:
            raise ValueError(f"columns {nonfinite} contain NaN or infinite values; provide finite data")

    duplicates = selected.select(keys).is_duplicated()
    if duplicates.any():
        example = selected.select(keys).filter(duplicates).head(1).rows(named=True)[0]
        raise ValueError(f"data contains duplicate observations for {example}; provide one row per key combination")

    return selected
