"""Tests for dataframe input preparation."""

from datetime import UTC, date, datetime

import jax
import jax.numpy as jnp
import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import mmmjax
from mmmjax import Model, PreparedData, Real, geometric_adstock, normal, prepare_data
from mmmjax.data import _prepare_frame, _prepare_panel


def test_data_api_exports_public_entry_points():
    assert mmmjax.data.__all__ == ["PreparedData", "prepare_data"]
    assert {"PreparedData", "prepare_data"}.issubset(mmmjax.__all__)
    assert PreparedData is mmmjax.data.PreparedData
    assert not hasattr(PreparedData, "align_to")
    assert not hasattr(PreparedData, "to_jax")
    assert prepare_data is mmmjax.data.prepare_data


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


def test_align_to_reorders_groups_and_features_without_changing_dates(frame_factory):
    training = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 1],
                "region": ["west", "east"],
                "sales": [10, 20],
                "video": [80.0, 60.0],
                "search": [40.0, 30.0],
                "promotion": [False, False],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        controls=["promotion"],
    )
    prediction = prepare_data(
        frame_factory(
            {
                "week": [4, 3, 3, 4, 5, 5],
                "region": ["east", "east", "west", "west", "east", "west"],
                "video": [40.0, 30.0, 300.0, 400.0, 50.0, 500.0],
                "search": [4.0, 3.0, 30.0, 40.0, 5.0, 50.0],
                "promotion": [True, False, True, False, True, False],
            }
        ),
        time="week",
        groups=["region"],
        controls=["promotion"],
        media=["search", "video"],
    )

    aligned = prediction._align_to(training)

    assert aligned.time_column == "week"
    assert aligned.time_values == (3, 4, 5)
    assert aligned.group_columns == ("region",)
    assert aligned.group_values == (("west",), ("east",))
    assert list(aligned.arrays) == ["media", "controls"]
    assert aligned.columns == {"media": ("video", "search"), "controls": ("promotion",)}
    assert aligned.channels == ("video", "search")
    np.testing.assert_array_equal(
        aligned.arrays["media"], [[[300, 30], [30, 3]], [[400, 40], [40, 4]], [[500, 50], [50, 5]]]
    )
    np.testing.assert_array_equal(aligned.arrays["controls"], [[[True], [False]], [[False], [True]], [[False], [True]]])
    assert aligned.arrays["media"].dtype == prediction.arrays["media"].dtype
    assert aligned.arrays["controls"].dtype == np.bool_
    assert prediction.group_values == (("east",), ("west",))
    assert prediction.columns["media"] == ("search", "video")
    assert prediction.channels == ("search", "video")
    assert list(prediction.arrays) == ["media", "controls"]
    np.testing.assert_array_equal(training.arrays["media"], [[[80, 40], [60, 30]]])

    # Distinct group and channel coefficients make a swapped label change the predictions
    coefficients = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    predict = jax.jit(lambda inputs: jnp.sum(inputs["media"] * coefficients, axis=-1))
    np.testing.assert_array_equal(predict(aligned._to_jax()), [[360, 102], [480, 136], [600, 170]])


@pytest.mark.parametrize("grouped", [False, True])
def test_align_to_preserves_singleton_axes_and_copies_unchanged_blocks(frame_factory, grouped):
    source = frame_factory({"week": [2, 1], "region": ["west", "west"], "count": [20, 10]})
    data = prepare_data(
        source,
        time="week",
        groups=["region"] if grouped else (),
        outcome="count",
        media=["count"],
    )

    aligned = data._align_to(data)

    assert aligned is not data
    assert aligned.arrays is not data.arrays
    assert aligned.columns is not data.columns
    for name, array in aligned.arrays.items():
        assert array.shape == data.arrays[name].shape
        assert array.dtype == data.arrays[name].dtype
        assert not np.shares_memory(array, data.arrays[name])
        np.testing.assert_array_equal(array, data.arrays[name])
        array[...] = 0
        assert np.all(data.arrays[name] > 0)


def test_align_to_matches_nested_group_combinations():
    training = prepare_data(
        pl.DataFrame({"week": [1, 1, 1], "region": ["west", "east", "east"], "store": [1, 2, 1], "value": [1, 2, 3]}),
        time="week",
        groups=["region", "store"],
        outcome="value",
    )
    prediction = prepare_data(
        pd.DataFrame(
            {"week": [2, 2, 2], "region": ["east", "west", "east"], "store": [1, 1, 2], "value": [30, 10, 20]}
        ),
        time="week",
        groups=["region", "store"],
        outcome="value",
    )

    aligned = prediction._align_to(training)

    assert aligned.group_values == (("west", 1), ("east", 2), ("east", 1))
    np.testing.assert_array_equal(aligned.arrays["outcome"], [[10, 20, 30]])


@pytest.mark.parametrize("groups", [["east"], ["east", "west", "north"], ["east", "north"]])
def test_align_to_rejects_missing_or_unexpected_groups(groups):
    training = prepare_data(
        pl.DataFrame({"week": [1, 1], "region": ["west", "east"], "value": [1, 2]}),
        time="week",
        groups=["region"],
        outcome="value",
    )
    prediction = prepare_data(
        pl.DataFrame({"week": [2] * len(groups), "region": groups, "value": [3] * len(groups)}),
        time="week",
        groups=["region"],
        outcome="value",
    )

    with pytest.raises(ValueError, match=r"group labels do not match.*region") as error:
        prediction._align_to(training)

    if "west" not in groups:
        assert "Missing groups [('west',)]" in str(error.value)
    if "north" in groups:
        assert "unexpected groups [('north',)]" in str(error.value)


@pytest.mark.parametrize("groups", [(), ["region"], ["store", "region"]])
def test_align_to_requires_matching_group_columns(groups):
    source = pl.DataFrame({"week": [1], "region": ["west"], "store": [1], "value": [1]})
    training = prepare_data(source, time="week", groups=["region", "store"], outcome="value")
    prediction = prepare_data(source, time="week", groups=groups, outcome="value")

    with pytest.raises(ValueError, match=r"group columns must match.*same groups selection"):
        prediction._align_to(training)


@pytest.mark.parametrize("reference_has_lists", [False, True])
def test_align_to_reports_list_valued_group_cells(reference_has_lists):
    scalar = prepare_data(
        pl.DataFrame({"week": [1, 1], "region": ["west", "east"], "value": [1, 2]}),
        time="week",
        groups=["region"],
        outcome="value",
    )
    nested = prepare_data(
        pl.DataFrame({"week": [1, 1], "region": [["west"], ["east"]], "value": [1, 2]}),
        time="week",
        groups=["region"],
        outcome="value",
    )
    prediction, reference = (scalar, nested) if reference_has_lists else (nested, scalar)

    with pytest.raises(TypeError, match=r"group columns.*region.*Use scalar group labels"):
        prediction._align_to(reference)


@pytest.mark.parametrize("columns", [["video"], ["video", "search", "social"], ["video", "social"]])
def test_align_to_rejects_missing_or_unexpected_columns(columns):
    source = pl.DataFrame({"week": [1], "video": [1.0], "search": [2.0], "social": [3.0]})
    training = prepare_data(source, time="week", media=["video", "search"])
    prediction = prepare_data(source, time="week", media=columns)

    with pytest.raises(ValueError, match="columns for 'media' do not match") as error:
        prediction._align_to(training)

    if "search" not in columns:
        assert "Missing columns ['search']" in str(error.value)
    if "social" in columns:
        assert "unexpected columns ['social']" in str(error.value)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("reshape_reference", [False, True])
def test_align_to_rejects_manually_changed_array_layout(grouped, reshape_reference):
    source = pl.DataFrame({"week": [1], "region": ["west"], "sales": [10]})
    groups = ["region"] if grouped else ()
    training = prepare_data(source, time="week", groups=groups, outcome="sales")
    prediction = prepare_data(source, time="week", groups=groups, outcome="sales")
    changed = training if reshape_reference else prediction
    changed.arrays["outcome"] = changed.arrays["outcome"][..., None]

    with pytest.raises(ValueError, match="input 'outcome' has a different number of axes"):
        prediction._align_to(training)


def test_align_to_rejects_additional_inputs_and_different_outcome_columns():
    source = pl.DataFrame({"week": [1], "sales": [10], "other": [20]})
    training = prepare_data(source, time="week", outcome="sales")
    unknown = prepare_data(source, time="week", controls=["sales"])
    renamed = prepare_data(source, time="week", outcome="other")

    with pytest.raises(ValueError, match=r"inputs.*controls.*do not exist in the reference"):
        unknown._align_to(training)
    with pytest.raises(ValueError, match=r"'outcome'.*Missing columns.*sales.*unexpected columns.*other"):
        renamed._align_to(training)


@pytest.mark.parametrize("reference", [None, {}, "training"])
def test_align_to_requires_prepared_reference_data(reference):
    data = prepare_data(pl.DataFrame({"week": [1], "sales": [10]}), time="week", outcome="sales")

    with pytest.raises(TypeError, match="reference must be PreparedData returned by prepare_data"):
        data._align_to(reference)


def test_prepare_data_example_keeps_channel_order_and_sorts_observations(frame_factory):
    source = frame_factory(
        {
            "week": ["2026-01-12", "2026-01-05", "2026-01-19"],
            "sales": [140, 100, 120],
            "search": [60.0, 40.0, 50.0],
            "video": [80.0, 60.0, 70.0],
        }
    )

    data = prepare_data(
        source,
        time="week",
        outcome="sales",
        media=["video", "search"],
        frequency="weekly",
    )
    inputs = data.arrays

    assert isinstance(data, PreparedData)
    assert data.time_values == ("2026-01-05", "2026-01-12", "2026-01-19")
    assert data.group_columns == data.group_values == ()
    assert data.columns == {"outcome": ("sales",), "media": ("video", "search")}
    np.testing.assert_array_equal(inputs["media"], [[60.0, 40.0], [80.0, 60.0], [70.0, 50.0]])
    np.testing.assert_array_equal(inputs["outcome"], [100, 140, 120])


@pytest.mark.parametrize("x64", [False, True])
def test_prepared_data_to_jax_follows_precision_setting_without_changing_host_data(frame_factory, x64):
    source = frame_factory(
        {
            "week": [2, 1],
            "sales": np.array([20, 10], dtype=np.int64),
            "video": np.array([2.5, 1.5], dtype=np.float64),
            "promotion": [True, False],
        }
    )
    data = prepare_data(source, time="week", outcome="sales", media=["video"], controls=["promotion"])
    before = {name: array.copy() for name, array in data.arrays.items()}

    with jax.enable_x64(x64):
        result = data._to_jax()
        assert jax.config.x64_enabled == x64

    assert set(result) == set(data.arrays)
    assert result["media"].dtype == (np.float64 if x64 else np.float32)
    assert result["outcome"].dtype == (np.int64 if x64 else np.int32)
    assert result["controls"].dtype == np.bool_
    for name, array in result.items():
        assert isinstance(array, jax.Array)
        assert array.shape == before[name].shape
        np.testing.assert_array_equal(array, before[name])
        np.testing.assert_array_equal(data.arrays[name], before[name])
        assert data.arrays[name].dtype == before[name].dtype
    assert data.time_values == (1, 2)
    assert data.columns == {"outcome": ("sales",), "media": ("video",), "controls": ("promotion",)}


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.float64, "float64"])
def test_prepared_data_to_jax_respects_explicit_floating_precision(dtype):
    data = prepare_data(pl.DataFrame({"week": [1], "value": [1.5]}), time="week", outcome="value")

    with jax.enable_x64(True):
        result = data._to_jax(dtype=dtype)

    assert result["outcome"].dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, "float64", np.dtype("float64")])
def test_prepared_data_to_jax_does_not_silently_narrow_explicit_float64(dtype):
    data = prepare_data(pl.DataFrame({"week": [1], "value": [1.5]}), time="week", outcome="value")

    with jax.enable_x64(False), pytest.raises(ValueError, match=r"explicit dtype float64.*JAX_ENABLE_X64=true"):
        data._to_jax(dtype=dtype)


@pytest.mark.parametrize("dtype", [None, np.int32, np.bool_, np.complex64, np.float16, "not-a-dtype"])
def test_prepared_data_to_jax_requires_a_supported_floating_dtype(dtype):
    data = prepare_data(pl.DataFrame({"week": [1], "value": [1.5]}), time="week", outcome="value")

    with pytest.raises(TypeError, match="dtype must be float32 or float64"):
        data._to_jax(dtype=dtype)


@pytest.mark.parametrize(
    "source_dtype,expected_dtype,values",
    [
        (np.int8, np.int8, [-128, 127]),
        (np.int16, np.int16, [-32768, 32767]),
        (np.int32, np.int32, [np.iinfo(np.int32).min, np.iinfo(np.int32).max]),
        (np.int64, np.int32, [np.iinfo(np.int32).min, np.iinfo(np.int32).max]),
        (np.uint64, np.uint32, [0, np.iinfo(np.uint32).max]),
    ],
)
def test_prepared_data_to_jax_preserves_representable_integers(source_dtype, expected_dtype, values):
    original = np.array(values, dtype=source_dtype)
    data = prepare_data(pl.DataFrame({"week": [1, 2], "counts": original}), time="week", outcome="counts")

    with jax.enable_x64(False):
        result = data._to_jax()

    assert result["outcome"].dtype == expected_dtype
    np.testing.assert_array_equal(result["outcome"], original)


@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.int64, int(np.iinfo(np.int32).min) - 1),
        (np.int64, int(np.iinfo(np.int32).max) + 1),
        (np.uint64, int(np.iinfo(np.uint32).max) + 1),
    ],
)
def test_prepared_data_to_jax_rejects_integer_wraparound(dtype, value):
    data = prepare_data(
        pl.DataFrame({"week": [1], "sales": np.array([value], dtype=dtype)}), time="week", outcome="sales"
    )

    with jax.enable_x64(False), pytest.raises(ValueError, match=r"block 'outcome'.*integers.*JAX_ENABLE_X64=true"):
        data._to_jax()

    with jax.enable_x64(True):
        result = data._to_jax()

    assert result["outcome"].dtype == dtype
    assert int(np.asarray(result["outcome"])[0]) == value


def test_prepared_data_to_jax_keeps_large_counts_exact_in_64_bit_mode():
    counts = np.array([2**53, 2**53 + 1], dtype=np.int64)
    data = prepare_data(pl.DataFrame({"week": [1, 2], "counts": counts}), time="week", outcome="counts")

    with jax.enable_x64(True):
        result = data._to_jax(dtype=np.float32)

    # The floating-point precision choice must not turn count outcomes into rounded floats
    assert result["outcome"].dtype == np.int64
    np.testing.assert_array_equal(result["outcome"], counts)


@pytest.mark.parametrize("value", [1e100, -1e100])
def test_prepared_data_to_jax_reports_floating_overflow_before_transfer(value):
    data = prepare_data(pl.DataFrame({"week": [1], "response": [value]}), time="week", outcome="response")

    with jax.enable_x64(False), pytest.raises(ValueError, match=r"block 'outcome'.*not finite as float32.*rescale"):
        data._to_jax()

    with jax.enable_x64(True):
        result = data._to_jax(dtype=np.float64)

    np.testing.assert_array_equal(result["outcome"], [value])


@pytest.mark.parametrize("sharded", [False, True])
def test_prepared_data_to_jax_places_all_blocks_on_the_requested_device(sharded):
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "sales": [10, 20], "video": [1.5, 2.5]}),
        time="week",
        outcome="sales",
        media=["video"],
    )
    device = jax.devices("cpu")[0]
    destination = jax.sharding.SingleDeviceSharding(device) if sharded else device

    result = data._to_jax(device=destination)

    for array in result.values():
        assert array.devices() == {device}
        assert array.committed


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
def test_prepared_data_to_jax_does_not_share_mutable_host_buffers(dtype):
    original = np.array([0, 1], dtype=dtype)
    data = prepare_data(pl.DataFrame({"week": [1, 2], "value": original}), time="week", outcome="value")
    result = data._to_jax(dtype=np.float32, device=jax.devices("cpu")[0])
    jax.block_until_ready(result)

    data.arrays["outcome"][:] = 0

    np.testing.assert_array_equal(result["outcome"], original)


def test_prepare_data_keeps_block_axes_and_labels_aligned(frame_factory):
    source = frame_factory(
        {
            "week": [2, 1, 1, 2],
            "geo": ["west", "east", "west", "east"],
            "sales": [40, 10, 30, 20],
            "video": np.array([4.5, 1.5, 3.5, 2.5], dtype=np.float32),
            "promotion": [True, False, False, True],
            "price": np.array([-0.5, 1.5, 2.5, 0.5], dtype=np.float32),
        }
    )
    before = nw.from_native(source).to_dict(as_series=False)

    result = prepare_data(
        source,
        time="week",
        groups=["geo"],
        outcome="sales",
        media=["video"],
        controls=["price", "promotion"],
    )

    assert result.time_column == "week"
    assert result.time_values == (1, 2)
    assert result.group_columns == ("geo",)
    assert result.group_values == (("west",), ("east",))
    assert result.columns == {"outcome": ("sales",), "media": ("video",), "controls": ("price", "promotion")}
    np.testing.assert_array_equal(result.arrays["outcome"], [[30, 10], [40, 20]])
    np.testing.assert_array_equal(result.arrays["media"], [[[3.5], [1.5]], [[4.5], [2.5]]])
    np.testing.assert_array_equal(result.arrays["controls"], [[[2.5, 0], [1.5, 0]], [[-0.5, 1], [0.5, 1]]])
    assert result.arrays["outcome"].dtype == np.int64
    assert result.arrays["media"].dtype == np.float32
    assert result.arrays["controls"].dtype == np.float32
    assert nw.from_native(source).to_dict(as_series=False) == before


@pytest.mark.parametrize("grouped", [False, True])
def test_prepare_data_distinguishes_a_column_from_a_single_column_block(frame_factory, grouped):
    source = frame_factory({"date": ["2026-01-05"], "geo": ["east"], "sales": [10]})
    groups = ["geo"] if grouped else []

    result = prepare_data(source, time="date", groups=groups, outcome="sales", media=["sales"], frequency="weekly")

    assert result.time_values == ("2026-01-05",)
    assert result.group_values == ((("east",),) if grouped else ())
    assert result.arrays["outcome"].shape == ((1, 1) if grouped else (1,))
    assert result.arrays["media"].shape == ((1, 1, 1) if grouped else (1, 1))
    np.testing.assert_array_equal(result.arrays["media"][..., 0], result.arrays["outcome"])


def test_prepare_data_keeps_nested_groups_on_one_observed_series_axis(frame_factory):
    source = frame_factory(
        {
            "week": [2, 1, 2, 1],
            "region": ["west", "east", "east", "west"],
            "store": [1, 1, 1, 1],
            "sales": [40, 10, 20, 30],
        }
    )

    result = prepare_data(source, time="week", groups=["region", "store"], outcome="sales")

    assert result.group_columns == ("region", "store")
    assert result.group_values == (("west", 1), ("east", 1))
    np.testing.assert_array_equal(result.arrays["outcome"], [[30, 10], [40, 20]])


def test_prepared_data_keeps_hundreds_of_channels_aligned(frame_factory):
    expected = np.arange(3 * 8 * 465, dtype=np.float32).reshape(3, 8, 465)
    rows = [(time, group) for time in [2, 0, 1] for group in reversed(range(8))]
    source = frame_factory(
        {
            "week": [time for time, _ in rows],
            "geo": [group for _, group in rows],
            **{
                f"channel_{channel}": [expected[time, group, channel] for time, group in rows] for channel in range(465)
            },
        }
    )
    channels = [f"channel_{channel}" for channel in reversed(range(465))]

    result = prepare_data(source, time="week", groups=["geo"], media=channels)

    assert result.columns["media"] == tuple(channels)
    assert result.group_values == tuple((group,) for group in reversed(range(8)))
    np.testing.assert_array_equal(result.arrays["media"], expected[:, ::-1, ::-1])
    assert len(jax.tree.leaves(result.arrays)) == 1

    reference = prepare_data(
        nw.from_native(source).sort(["week", "geo"]).to_native(),
        time="week",
        groups=["geo"],
        media=list(reversed(channels)),
    )
    aligned = result._align_to(reference)

    np.testing.assert_array_equal(aligned.arrays["media"], expected)
    source_dtype = nw.from_native(source).get_column(channels[0]).to_numpy().dtype
    assert aligned.arrays["media"].dtype == source_dtype
    assert aligned.columns == reference.columns
    assert aligned.group_values == reference.group_values
    assert len(jax.tree.leaves(aligned.arrays)) == 1


def test_prepare_data_preserves_integer_counts_separately_from_floating_features(frame_factory):
    source = frame_factory({"week": [1, 2], "counts": [2**53, 2**53 + 1], "media": [0.5, 1.5], "flag": [True, False]})

    result = prepare_data(source, time="week", outcome="counts", media=["media"], controls=["flag"])

    assert result.arrays["outcome"].dtype == np.int64
    assert result.arrays["controls"].dtype == np.bool_
    np.testing.assert_array_equal(result.arrays["outcome"], np.array([2**53, 2**53 + 1], dtype=np.int64))


def test_prepare_data_arrays_do_not_modify_the_input_or_each_other(frame_factory):
    source = frame_factory({"week": [1, 2], "sales": [10, 20]})
    result = prepare_data(source, time="week", outcome="sales", media=["sales"])

    result.arrays["outcome"][0] = 0

    assert nw.from_native(source)["sales"].to_list() == [10, 20]
    np.testing.assert_array_equal(result.arrays["media"], [[10], [20]])
    result.arrays["media"][1, 0] = 0
    assert nw.from_native(source)["sales"].to_list() == [10, 20]


def test_prepare_data_keeps_media_spend_and_channel_labels_together(frame_factory):
    source = frame_factory(
        {
            "week": [3, 1, 2],
            "sales": [30, 10, 20],
            "video_views": [300.0, 100.0, 200.0],
            "search_clicks": [60.0, 20.0, 40.0],
            "video_cost": [90.0, 30.0, 60.0],
            "search_cost": [120.0, 40.0, 80.0],
            "price": [-3.0, -1.0, -2.0],
            "promotion": [True, False, True],
        }
    )
    training = prepare_data(
        source,
        time="week",
        outcome="sales",
        media=["video_views", "search_clicks"],
        spend=["video_cost", "search_cost"],
        channels=["video", "search"],
        controls=["price", "promotion"],
    )
    prediction = prepare_data(
        source,
        time="week",
        media=["search_clicks", "video_views"],
        spend=["search_cost", "video_cost"],
        channels=["search", "video"],
        controls=["promotion", "price"],
    )

    aligned = prediction._align_to(training)

    assert aligned.channels == ("video", "search")
    assert aligned.columns == {
        "media": ("video_views", "search_clicks"),
        "spend": ("video_cost", "search_cost"),
        "controls": ("price", "promotion"),
    }
    np.testing.assert_array_equal(aligned.arrays["media"], [[100, 20], [200, 40], [300, 60]])
    np.testing.assert_array_equal(aligned.arrays["spend"], [[30, 40], [60, 80], [90, 120]])
    np.testing.assert_array_equal(aligned.arrays["controls"], [[-1, 0], [-2, 1], [-3, 1]])
    assert prediction.channels == ("search", "video")
    np.testing.assert_array_equal(prediction.arrays["spend"], [[40, 30], [80, 60], [120, 90]])
    assert set(aligned._to_jax()) == {"media", "spend", "controls"}


@pytest.mark.parametrize(
    "media,spend,channels",
    [
        (["video", "search"], ["video_cost", "search_cost"], ["video", "social"]),
        (["search", "video"], ["video_cost", "search_cost"], ["video", "search"]),
        (["video", "search"], ["search_cost", "video_cost"], ["video", "search"]),
    ],
)
def test_align_to_rejects_changed_channel_assignments(media, spend, channels):
    source = pl.DataFrame({"week": [1], "video": [10], "search": [20], "video_cost": [30], "search_cost": [40]})
    training = prepare_data(source, time="week", media=["video", "search"], spend=["video_cost", "search_cost"])
    prediction = prepare_data(source, time="week", media=media, spend=spend, channels=channels)

    with pytest.raises(ValueError, match=r"channel labels for.*same channel-to-column assignments"):
        prediction._align_to(training)


@pytest.mark.parametrize("grouped", [False, True])
def test_prepare_data_can_use_spending_as_media_without_sharing_buffers(frame_factory, grouped):
    source = frame_factory({"week": [2, 1], "region": ["west", "west"], "video": [20.0, 10.0]})
    data = prepare_data(source, time="week", groups=["region"] if grouped else (), media=("video",), spend=("video",))

    assert data.channels == ("video",)
    expected = [[[10]], [[20]]] if grouped else [[10], [20]]
    np.testing.assert_array_equal(data.arrays["media"], expected)
    np.testing.assert_array_equal(data.arrays["spend"], expected)
    data.arrays["media"][...] = 0
    np.testing.assert_array_equal(data.arrays["spend"], expected)
    assert nw.from_native(source)["video"].to_list() == [20.0, 10.0]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("role", ["media", "spend"])
def test_prepare_data_reports_negative_media_and_spending_columns(frame_factory, role, grouped):
    source = frame_factory({"week": [2, 1], "region": ["west", "west"], "video": [0, 10], "search": [-1, 20]})
    selections = {"media": ["video"], role: ["search"]}

    with pytest.raises(ValueError, match=rf"{role} columns.*search.*negative values"):
        prepare_data(source, time="week", groups=["region"] if grouped else (), **selections)


def test_prepare_data_allows_signed_outcomes_and_controls(frame_factory):
    data = prepare_data(
        frame_factory({"week": [2, 1], "response": [-10, 5], "price": [-0.5, 1.5], "video": [0, 10]}),
        time="week",
        outcome="response",
        controls=["price"],
        media=["video"],
    )
    np.testing.assert_array_equal(data.arrays["outcome"], [5, -10])
    np.testing.assert_array_equal(data.arrays["controls"], [[1.5], [-0.5]])
    np.testing.assert_array_equal(data.arrays["media"], [[10], [0]])


@pytest.mark.parametrize(
    "selection,error,message",
    [
        ({}, ValueError, "select at least one of outcome, media or controls"),
        ({"media": None}, ValueError, "select at least one"),
        ({"outcome": ["sales"]}, TypeError, "outcome must be a column name"),
        ({"outcome": ""}, ValueError, "outcome must select at least one nonempty"),
        ({"media": "sales"}, TypeError, "media must be a sequence"),
        ({"controls": "sales"}, TypeError, "controls must be a sequence"),
        ({"media": []}, ValueError, "media must select at least one"),
        ({"media": [1]}, ValueError, "media must select at least one"),
        ({"media": ["sales", "sales"]}, ValueError, "media contains repeated columns"),
        ({"controls": ["sales", "sales"]}, ValueError, "controls contains repeated columns"),
        ({"spend": ["sales"]}, ValueError, "spend requires media"),
        ({"media": ["sales"], "spend": "sales"}, TypeError, "spend must be a sequence"),
        ({"media": ["sales"], "spend": ["sales", "other"]}, ValueError, "spend must select one column per media"),
        ({"outcome": "sales", "channels": ["video"]}, ValueError, "channels requires media"),
        ({"media": ["sales"], "channels": "video"}, TypeError, "channels must be a sequence"),
        ({"media": ["sales"], "channels": []}, ValueError, "channels must contain one name per media"),
        ({"media": ["sales"], "channels": [1]}, ValueError, "channels must contain only nonempty string"),
        ({"media": ["sales"], "channels": [""]}, ValueError, "channels must contain only nonempty string"),
        ({"media": ["sales"], "channels": ["video", "video"]}, ValueError, "channels must contain unique names"),
        ({"media": ["sales"], "channels": ["video", "search"]}, ValueError, "channels must contain one name per media"),
        ({"outcome": "missing"}, ValueError, "data is missing columns"),
        ({"outcome": "week"}, ValueError, "column declarations contain repeated names"),
    ],
)
def test_prepare_data_reports_invalid_input_selections(selection, error, message):
    with pytest.raises(error, match=message):
        prepare_data(pl.DataFrame({"week": [1], "sales": [10]}), time="week", **selection)


def test_prepare_data_checks_calendar_and_panel_coverage():
    source = pl.DataFrame({"week": ["2026-01-05", "2026-01-19"], "sales": [10, 20]})

    with pytest.raises(ValueError, match="missing periods"):
        prepare_data(source, time="week", outcome="sales", frequency="weekly")

    source = pl.DataFrame({"week": [1, 2, 1], "geo": ["east", "east", "west"], "sales": [10, 20, 30]})
    with pytest.raises(ValueError, match="same time values"):
        prepare_data(source, time="week", groups=["geo"], outcome="sales")


def test_prepare_data_works_with_model_densities_and_gradients():
    data = prepare_data(
        pl.DataFrame({"week": [2, 1], "sales": [5.0, 2.0], "video": [2.0, 1.0], "search": [1.0, 3.0]}),
        time="week",
        outcome="sales",
        media=["video", "search"],
    )

    def log_density(data, beta):
        return normal(data["outcome"], data["media"] @ beta, 1.0)

    model = Model({"beta": Real(shape=(2,))}, log_density)
    position = {"beta": jnp.array([0.5, 1.0])}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, data._to_jax())

    residual = np.array([2.0, 5.0]) - np.array([[1.0, 3.0], [2.0, 1.0]]) @ np.array([0.5, 1.0])
    expected_density = -0.5 * np.sum(residual**2) - np.log(2 * np.pi)
    np.testing.assert_allclose(value, expected_density, rtol=1e-6)
    np.testing.assert_allclose(gradient["beta"], np.array([[1.0, 2.0], [3.0, 1.0]]) @ residual)


def test_prepare_data_keeps_labels_out_of_jax_compilation():
    first = prepare_data(pl.DataFrame({"week": [1, 2], "sales": [10.0, 20.0]}), time="week", outcome="sales")
    second = prepare_data(
        pl.DataFrame({"date": ["2026-01-05", "2026-01-12"], "revenue": [30.0, 40.0]}),
        time="date",
        outcome="revenue",
    )
    traces = []

    @jax.jit
    def total(data):
        # Changing only labels and values should reuse the same compiled numerical function
        traces.append(None)
        return data["outcome"].sum()

    assert float(total(first._to_jax())) == 30.0
    assert float(total(second._to_jax())) == 70.0
    assert len(traces) == 1


def test_prepare_data_works_with_batched_adstock(frame_factory):
    source = frame_factory(
        {"week": [2, 1, 1, 2], "geo": ["west", "east", "west", "east"], "video": [4.0, 1.0, 3.0, 2.0]}
    )
    data = prepare_data(source, time="week", groups=["geo"], media=["video"])

    carried = jax.jit(lambda media: geometric_adstock(media, 0.5, max_lag=1, normalize=False))(data._to_jax()["media"])

    np.testing.assert_array_equal(carried, [[[3.0], [1.0]], [[5.5], [2.5]]])


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


@pytest.mark.parametrize(
    "frequency,dates",
    [
        ("daily", ["2024-02-28", "2024-02-29", "2024-03-01"]),
        ("weekly", ["2025-12-31", "2026-01-07", "2026-01-14"]),
        ("monthly", ["2024-01-01", "2024-02-01", "2024-03-01"]),
        ("monthly", ["2024-01-31", "2024-02-29", "2024-03-31"]),
        ("monthly", ["2024-01-30", "2024-02-29", "2024-03-30"]),
        ("quarterly", ["2023-11-30", "2024-02-29", "2024-05-31"]),
        ("yearly", ["2024-02-29", "2025-02-28", "2026-02-28"]),
    ],
)
def test_prepare_panel_checks_calendar_without_changing_labels_or_values(frame_factory, frequency, dates):
    source = frame_factory({"date": dates[::-1], "sales": [30, 20, 10]})
    original = nw.from_native(source)
    before = original.to_dict(as_series=False)

    result = _prepare_panel(source, time="date", values=["sales"], frequency=frequency)

    assert result.to_dict(as_series=False) == {"date": dates, "sales": [10, 20, 30]}
    assert result.schema == original.schema
    assert nw.from_native(source).to_dict(as_series=False) == before


@pytest.mark.parametrize("representation", [date.fromisoformat, datetime.fromisoformat])
def test_prepare_panel_accepts_native_dates_and_datetimes(frame_factory, representation):
    dates = [representation(value) for value in ["2024-01-31", "2024-02-29", "2024-03-31"]]
    source = frame_factory({"date": dates[::-1], "sales": [30, 20, 10]})

    result = _prepare_panel(source, time="date", values=["sales"], frequency="monthly")

    assert result["date"].to_list() == dates
    assert result.schema == nw.from_native(source).schema


@pytest.mark.parametrize(
    "frequency,offset", [("monthly", pd.offsets.MonthEnd()), ("quarterly", pd.offsets.QuarterEnd())]
)
def test_prepare_panel_accepts_established_calendar_ranges(frame_factory, frequency, offset):
    # A calendar-generated reference spans leap years and months of different lengths
    dates = pd.date_range("2023-01-01", periods=36, freq=offset).to_pydatetime().tolist()
    source = frame_factory({"date": dates[::-1]})

    result = _prepare_panel(source, time="date", values=[], frequency=frequency)

    assert result["date"].to_list() == dates


def test_prepare_panel_detects_a_week_missing_from_every_group(frame_factory):
    source = frame_factory({"week": ["2026-01-05", "2026-01-19"] * 2, "geo": ["east", "east", "west", "west"]})

    with pytest.raises(ValueError, match=r"week.*weekly.*expected 2026-01-12.*found 2026-01-19.*missing periods"):
        _prepare_panel(source, time="week", groups=["geo"], values=[], frequency="weekly")


def test_prepare_panel_checks_distinct_times_in_a_balanced_panel(frame_factory):
    source = frame_factory({"week": ["2026-01-12", "2026-01-05"] * 2, "geo": ["west", "west", "east", "east"]})

    result = _prepare_panel(source, time="week", groups=["geo"], values=[], frequency="weekly")

    assert result.to_dict(as_series=False) == {
        "week": ["2026-01-05", "2026-01-05", "2026-01-12", "2026-01-12"],
        "geo": ["west", "east", "west", "east"],
    }


@pytest.mark.parametrize(
    "frequency,dates,expected,observed",
    [
        ("daily", ["2024-02-28", "2024-03-01"], "2024-02-29", "2024-03-01"),
        ("weekly", ["2026-01-05", "2026-01-13"], "2026-01-12", "2026-01-13"),
        ("monthly", ["2026-01-31", "2026-03-31"], "2026-02-28", "2026-03-31"),
        ("monthly", ["2026-01-31", "2026-02-28", "2026-03-28"], "2026-03-31", "2026-03-28"),
        ("quarterly", ["2026-01-01", "2026-07-01"], "2026-04-01", "2026-07-01"),
        ("yearly", ["2024-02-29", "2026-02-28"], "2025-02-28", "2026-02-28"),
    ],
)
def test_prepare_panel_rejects_missing_or_shifted_calendar_periods(frame_factory, frequency, dates, expected, observed):
    with pytest.raises(ValueError, match=rf"date.*{frequency}.*expected {expected}.*found {observed}"):
        _prepare_panel(frame_factory({"date": dates}), time="date", values=[], frequency=frequency)


@pytest.mark.parametrize("label", ["01/02/2026", "2026-02-30", "20260105", "2026-W02-1", "2026-01-05T01:00:00"])
def test_prepare_panel_does_not_guess_or_silently_adjust_date_strings(frame_factory, label):
    with pytest.raises(ValueError, match=r"time column 'date'.*YYYY-MM-DD"):
        _prepare_panel(frame_factory({"date": [label]}), time="date", values=[], frequency="weekly")


def test_prepare_panel_does_not_interpret_period_numbers_as_calendar_dates(frame_factory):
    with pytest.raises(TypeError, match=r"weekly.*requires dates.*week"):
        _prepare_panel(frame_factory({"week": [1, 2]}), time="week", values=[], frequency="weekly")


def test_prepare_panel_requires_explicit_timezone_conversion(frame_factory):
    source = frame_factory({"date": [datetime(2026, 1, 5, tzinfo=UTC)]})

    with pytest.raises(ValueError, match=r"timezone-aware.*observation dates.*intended timezone"):
        _prepare_panel(source, time="date", values=[], frequency="weekly")


def test_prepare_panel_does_not_discard_time_of_day(frame_factory):
    source = frame_factory({"date": [datetime(2026, 1, 5, 12), datetime(2026, 1, 12, 13)]})

    with pytest.raises(ValueError, match="expected 2026-01-12 12:00:00, found 2026-01-12 13:00:00"):
        _prepare_panel(source, time="date", values=[], frequency="weekly")


@pytest.mark.parametrize("second", [datetime(2026, 1, 12), "2026-01-12"])
def test_prepare_panel_reports_mixed_time_representations(second):
    # pandas object columns can mix individually valid labels that cannot be sorted together
    source = pd.DataFrame({"date": [date(2026, 1, 5), second]})

    with pytest.raises(TypeError, match=r"date.*mixes date representations.*entire column"):
        _prepare_panel(source, time="date", values=[], frequency="weekly")


def test_prepare_panel_accepts_one_calendar_period_without_inventing_dates(frame_factory):
    result = _prepare_panel(frame_factory({"date": ["2026-01-05"]}), time="date", values=[], frequency="weekly")

    assert result["date"].to_list() == ["2026-01-05"]


@pytest.mark.parametrize("frequency", ["", "fortnightly", 7, ["weekly"]])
def test_prepare_panel_reports_invalid_calendar_frequency(frequency):
    with pytest.raises(ValueError, match=r"frequency must be one of.*daily.*weekly"):
        _prepare_panel(pl.DataFrame({"date": ["2026-01-05"]}), time="date", values=[], frequency=frequency)


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
