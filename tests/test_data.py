"""Tests for dataframe input preparation."""

from datetime import date

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from mmmjax.data import _prepare_frame, _prepare_panel


@pytest.fixture(params=["pandas", "pandas_nullable", "pandas_arrow", "polars", "pyarrow"])
def frame_factory(request):
    if request.param == "pandas":
        return pd.DataFrame
    if request.param == "pandas_nullable":
        return lambda columns: pd.DataFrame(columns).convert_dtypes(dtype_backend="numpy_nullable")
    if request.param == "pandas_arrow":
        return lambda columns: pd.DataFrame(columns).convert_dtypes(dtype_backend="pyarrow")
    if request.param == "polars":
        return pl.DataFrame
    return pa.table


def test_prepare_panel_aligns_groups_without_changing_values_or_dtypes(frame_factory):
    columns = {
        "sales": [40, 10, 30, 20],
        "week": [2, 1, 1, 2],
        "geo": ["west", "east", "west", "east"],
        "video": [4.5, 1.5, 3.5, 2.5],
    }
    source = frame_factory(columns)
    original = nw.from_native(source)
    original_values = original.to_dict(as_series=False)

    result = _prepare_panel(source, time="week", groups=["geo"], values=["video", "sales"])

    expected = {
        "week": [1, 1, 2, 2],
        "geo": ["west", "east", "west", "east"],
        "video": [3.5, 1.5, 4.5, 2.5],
        "sales": [30, 10, 40, 20],
    }
    assert result.columns == list(expected)
    assert result.to_dict(as_series=False) == expected
    assert result.implementation == original.implementation
    assert result.schema == original.select(list(expected)).schema
    np.testing.assert_equal(nw.from_native(source).to_dict(as_series=False), original_values)


def test_prepare_panel_keeps_only_observed_nested_groups(frame_factory):
    source = frame_factory(
        {
            "week": [2, 1, 2, 1, 2, 1],
            "region": ["west", "east", "east", "west", "west", "west"],
            "store": ["north", "central", "central", "central", "central", "north"],
            "sales": [60, 10, 20, 30, 40, 50],
        }
    )

    result = _prepare_panel(source, time="week", groups=["region", "store"], values=["sales"])

    assert result.to_dict(as_series=False) == {
        "week": [1, 1, 1, 2, 2, 2],
        "region": ["west", "east", "west", "west", "east", "west"],
        "store": ["north", "central", "central", "north", "central", "central"],
        "sales": [50, 10, 30, 60, 20, 40],
    }


def test_prepare_panel_preserves_hundreds_of_channels_across_groups(frame_factory):
    media = np.arange(3 * 8 * 465, dtype=np.int64).reshape(3, 8, 465)
    time_order = [2, 0, 1]
    group_order = [7, 3, 0, 6, 2, 1, 5, 4]
    rows = [(time, group) for time in time_order for group in group_order]
    columns = {
        "week": [time for time, _ in rows],
        "geo": [group for _, group in rows],
        **{f"channel_{channel}": [media[time, group, channel] for time, group in rows] for channel in range(465)},
    }
    channel_order = [f"channel_{channel}" for channel in reversed(range(465))]

    result = _prepare_panel(frame_factory(columns), time="week", groups=["geo"], values=channel_order)

    assert result["week"].to_list() == [0] * 8 + [1] * 8 + [2] * 8
    assert result["geo"].to_list() == group_order * 3
    aligned = result.select(channel_order).to_numpy().reshape(3, 8, 465)
    np.testing.assert_array_equal(aligned, media[:, group_order, ::-1])


def test_prepare_panel_sorts_dates_for_a_single_series(frame_factory):
    source = frame_factory({"week": [date(2026, 1, 12), date(2026, 1, 5)], "sales": [20, 10]})

    result = _prepare_panel(source, time="week", values=["sales"])

    assert result.to_dict(as_series=False) == {
        "week": [date(2026, 1, 5), date(2026, 1, 12)],
        "sales": [10, 20],
    }


@pytest.mark.parametrize(
    "weeks,geos", [([1, 2, 1], ["east", "east", "west"]), ([1, 2, 2, 3], ["east", "east", "west", "west"])]
)
def test_prepare_panel_rejects_missing_or_mismatched_group_times(frame_factory, weeks, geos):
    source = frame_factory({"week": weeks, "geo": geos, "sales": list(range(len(weeks)))})

    # Equal observation counts per group are not enough if the actual time values differ
    with pytest.raises(ValueError, match=r"missing [12] combinations of 'week'.*geo.*same time values"):
        _prepare_panel(source, time="week", groups=["geo"], values=["sales"])


def test_prepare_panel_rejects_duplicates_before_checking_coverage(frame_factory):
    source = frame_factory({"week": [1, 1, 1, 2], "geo": ["east", "east", "west", "west"], "sales": [10, 10, 20, 30]})

    # The row count matches a full panel, but a duplicate cannot stand in for a missing observation
    with pytest.raises(ValueError, match="duplicate observations"):
        _prepare_panel(source, time="week", groups=["geo"], values=["sales"])


def test_prepare_panel_does_not_fill_calendar_gaps_shared_by_all_groups(frame_factory):
    source = frame_factory({"week": [1, 3, 1, 3], "geo": ["east", "east", "west", "west"]})

    result = _prepare_panel(source, time="week", groups=["geo"], values=[])

    assert result.to_dict(as_series=False) == {"week": [1, 1, 3, 3], "geo": ["east", "west", "east", "west"]}


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_prepare_panel_uses_observed_order_for_categorical_groups(backend):
    columns = {"week": [2, 1, 1, 2], "geo": ["west", "east", "west", "east"], "sales": [40, 10, 30, 20]}
    if backend == "pandas":
        source = pd.DataFrame(columns)
        source["geo"] = pd.Categorical(source["geo"], categories=["east", "west", "unused"], ordered=True)
    elif backend == "polars":
        source = pl.DataFrame(columns).with_columns(pl.col("geo").cast(pl.Enum(["east", "west", "unused"])))
    else:
        source = pa.table(columns)
        labels = pa.DictionaryArray.from_arrays(pa.array([1, 0, 1, 0]), pa.array(["east", "west", "unused"]))
        source = source.set_column(1, "geo", labels)
    original_schema = nw.from_native(source).schema

    result = _prepare_panel(source, time="week", groups=["geo"], values=["sales"])

    assert result.to_dict(as_series=False) == {
        "week": [1, 1, 2, 2],
        "geo": ["west", "east", "west", "east"],
        "sales": [30, 10, 40, 20],
    }
    assert result.schema == original_schema


@pytest.mark.parametrize("groups", ["geo", None, 1])
def test_prepare_panel_rejects_invalid_group_declarations(groups):
    with pytest.raises(TypeError, match="groups must be a sequence of column names"):
        _prepare_panel(pl.DataFrame({"week": [1]}), time="week", groups=groups, values=[])


def test_prepare_frame_preserves_observations_and_declared_column_order(frame_factory):
    columns = {
        "sales": [30, 10, 20, 40],
        "week": [date(2026, 1, 12), date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 12)],
        "geo": ["west", "east", "west", "east"],
        "video": [0.0, 100.5, 40.5, 80.0],
        "price_change": [-2.5, 0.0, 1.5, -1.0],
        "promotion": [False, True, False, True],
        "unused": [None, "note", None, "note"],
    }
    source = frame_factory(columns)
    original = nw.from_native(source)
    original_values = original.to_dict(as_series=False)
    names = ["geo", "week", "video", "promotion", "price_change", "sales"]

    result = _prepare_frame(source, keys=["geo", "week"], values=names[2:])

    assert result.columns == names
    assert result.to_dict(as_series=False) == {name: columns[name] for name in names}
    assert result.implementation == original.implementation
    assert result.schema == original.select(names).schema
    np.testing.assert_equal(nw.from_native(source).to_dict(as_series=False), original_values)


def test_prepare_frame_accepts_a_single_series_without_geo(frame_factory):
    source = frame_factory({"week": [2, 1], "sales": [10, -5]})

    result = _prepare_frame(source, keys=("week",), values=("sales",))

    assert result.to_dict(as_series=False) == {"week": [2, 1], "sales": [10, -5]}


def test_prepare_frame_accepts_keys_without_value_columns(frame_factory):
    result = _prepare_frame(frame_factory({"week": [1, 2]}), keys=["week"], values=[])

    assert result.to_dict(as_series=False) == {"week": [1, 2]}


def test_prepare_frame_reports_missing_columns(frame_factory):
    source = frame_factory({"week": [1], "sales": [10]})

    with pytest.raises(ValueError, match=r"missing columns \['video', 'price'\].*available columns.*week.*sales"):
        _prepare_frame(source, keys=["week"], values=["video", "price"])


def test_prepare_frame_rejects_empty_data(frame_factory):
    with pytest.raises(ValueError, match="at least one observation"):
        _prepare_frame(frame_factory({"week": [], "sales": []}), keys=["week"], values=["sales"])


@pytest.mark.parametrize("column", ["week", "sales"])
def test_prepare_frame_rejects_missing_selected_values(frame_factory, column):
    columns = {"week": [1, 2], "sales": [10, 20]}
    columns[column][1] = None

    with pytest.raises(ValueError, match=rf"{column}.*missing values"):
        _prepare_frame(frame_factory(columns), keys=["week"], values=["sales"])


@pytest.mark.parametrize("column", ["week", "sales"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_prepare_frame_rejects_nonfinite_keys_and_values(frame_factory, column, invalid):
    columns = {"week": [1.0, 2.0], "sales": [10.0, 20.0]}
    columns[column][1] = invalid
    # pandas tries an integer cast while inferring nullable dtypes for infinite inputs
    with np.errstate(invalid="ignore"):
        source = frame_factory(columns)

    # NumPy-backed pandas treats NaN as missing, while Polars and Arrow distinguish the two
    with pytest.raises(ValueError, match=rf"{column}.*(missing|NaN or infinite) values"):
        _prepare_frame(source, keys=["week"], values=["sales"])


@pytest.mark.parametrize("value", ["10", date(2026, 1, 5), [10]])
def test_prepare_frame_does_not_silently_coerce_value_columns(frame_factory, value):
    source = frame_factory({"week": [1], "sales": [value]})

    with pytest.raises(TypeError, match=r"value columns must be.*sales"):
        _prepare_frame(source, keys=["week"], values=["sales"])


@pytest.mark.parametrize("sales", [[10, 10], [10, 20]])
def test_prepare_frame_rejects_duplicate_keys_even_when_values_agree(frame_factory, sales):
    source = frame_factory({"week": [1, 1], "geo": ["east", "east"], "sales": sales})

    with pytest.raises(ValueError, match=r"duplicate observations.*week.*1.*geo.*east.*one row"):
        _prepare_frame(source, keys=["week", "geo"], values=["sales"])


def test_prepare_frame_uses_columns_not_the_pandas_index():
    source = pd.DataFrame({"week": [3, 1, 2], "sales": [30, 10, 20]}, index=[7, 7, 2])
    original = source.copy(deep=True)

    result = _prepare_frame(source, keys=["week"], values=["sales"])

    assert result.to_dict(as_series=False) == {"week": [3, 1, 2], "sales": [30, 10, 20]}
    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(result.to_native(), original)


def test_prepare_frame_preserves_numeric_precision(frame_factory):
    columns = {
        "week": np.array([1, 2], dtype=np.int32),
        "sales": np.array([2**53, 2**53 + 1], dtype=np.int64),
        "video": np.array([0.1, 0.2], dtype=np.float32),
    }
    source = frame_factory(columns)
    original = nw.from_native(source)

    result = _prepare_frame(source, keys=["week"], values=["sales", "video"])

    assert result.schema == original.schema
    assert result["sales"].to_list() == [2**53, 2**53 + 1]


@pytest.mark.parametrize(
    "keys,values,error,message",
    [
        ("week", ["sales"], TypeError, "keys must be a sequence"),
        (["week"], "sales", TypeError, "values must be a sequence"),
        (None, ["sales"], TypeError, "keys must be a sequence"),
        (["week"], None, TypeError, "values must be a sequence"),
        ([], ["sales"], ValueError, "keys must include at least one column"),
        ([""], ["sales"], ValueError, "keys must contain only nonempty string"),
        ([1], ["sales"], ValueError, "keys must contain only nonempty string"),
        (["week"], [1], ValueError, "values must contain only nonempty string"),
        (["week", "week"], ["sales"], ValueError, "repeated names.*week"),
        (["week"], ["sales", "sales"], ValueError, "repeated names.*sales"),
        (["week"], ["week"], ValueError, "repeated names.*week"),
    ],
)
def test_prepare_frame_validates_column_declarations(keys, values, error, message):
    with pytest.raises(error, match=message):
        _prepare_frame(pl.DataFrame({"week": [1], "sales": [10]}), keys=keys, values=values)


def test_prepare_frame_does_not_collect_lazy_inputs():
    source = pl.DataFrame({"week": [1], "sales": [10]}).lazy()

    with pytest.raises(TypeError, match=r"[Ll]azy"):
        _prepare_frame(source, keys=["week"], values=["sales"])


@pytest.mark.parametrize("source", [{"week": [1]}, [1], np.array([1]), pd.Series([1]), pl.Series([1])])
def test_prepare_frame_rejects_inputs_that_are_not_dataframes(source):
    with pytest.raises(TypeError):
        _prepare_frame(source, keys=["week"], values=[])
