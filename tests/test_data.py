"""Tests for dataframe input preparation."""

from datetime import date

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from mmmjax.data import _prepare_frame


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
