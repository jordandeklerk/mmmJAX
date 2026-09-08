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


def test_prepare_data_keeps_treatments_separate_and_orders_their_axes(frame_factory):
    source = frame_factory(
        {
            "week": [2, 1, 1, 2],
            "region": ["west", "east", "west", "east"],
            "store": [1, 2, 1, 2],
            "sales": [40, 10, 30, 20],
            "video": [4.5, 1.5, 3.5, 2.5],
            "cost": [45, 15, 35, 25],
            "temperature": [-4.0, -1.0, -3.0, -2.0],
            "promotion": [True, False, True, False],
            "price_change": [-0.5, 1.5, 2.5, 0.5],
        }
    )
    before = nw.from_native(source).to_dict(as_series=False)

    data = prepare_data(
        source,
        time="week",
        groups=["region", "store"],
        outcome="sales",
        media=["video"],
        spend=["cost"],
        controls=["temperature"],
        treatments=["price_change", "promotion"],
        channels=["Video"],
    )

    assert set(data.arrays) == {"outcome", "media", "spend", "controls", "treatments"}
    assert data.columns["treatments"] == ("price_change", "promotion")
    assert data.columns["controls"] == ("temperature",)
    assert data.time_values == (1, 2)
    assert data.group_values == (("west", 1), ("east", 2))
    assert data.channels == ("Video",)
    assert data.rf_channels == ()
    assert data.organic_rf_channels == ()
    np.testing.assert_array_equal(data.arrays["treatments"], [[[2.5, 1], [1.5, 0]], [[-0.5, 1], [0.5, 0]]])
    np.testing.assert_array_equal(data.arrays["outcome"], [[30, 10], [40, 20]])
    np.testing.assert_array_equal(data.arrays["media"], [[[3.5], [1.5]], [[4.5], [2.5]]])
    np.testing.assert_array_equal(data.arrays["spend"], [[[35], [15]], [[45], [25]]])
    np.testing.assert_array_equal(data.arrays["controls"], [[[-3], [-1]], [[-4], [-2]]])
    assert np.issubdtype(data.arrays["outcome"].dtype, np.integer)
    assert np.issubdtype(data.arrays["treatments"].dtype, np.floating)
    assert nw.from_native(source).to_dict(as_series=False) == before

    data.arrays["treatments"][...] = 0
    assert nw.from_native(source).to_dict(as_series=False) == before
    np.testing.assert_array_equal(data.arrays["controls"], [[[-3], [-1]], [[-4], [-2]]])


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("values", [[True, False], [-2, 3], [-0.5, 1.25]])
def test_prepare_data_accepts_treatments_without_other_inputs(frame_factory, grouped, values):
    source = frame_factory({"week": [2, 1], "region": ["west", "west"], "promotion": values})
    data = prepare_data(source, time="week", groups=["region"] if grouped else (), treatments=("promotion",))

    assert set(data.arrays) == {"treatments"}
    assert data.columns == {"treatments": ("promotion",)}
    assert data.time_values == (1, 2)
    assert data.media_time_values == ()
    assert data.channels == ()
    expected = [[[values[1]]], [[values[0]]]] if grouped else [[values[1]], [values[0]]]
    np.testing.assert_array_equal(data.arrays["treatments"], expected)
    assert data.arrays["treatments"].dtype == np.asarray(values).dtype


def test_prepare_data_ignores_unselected_treatments(frame_factory):
    source = frame_factory({"week": [1, 2], "sales": [10, 20], "promotion": [None, "unknown"]})
    data = prepare_data(source, time="week", outcome="sales", treatments=None)

    assert set(data.arrays) == {"outcome"}
    assert data.columns == {"outcome": ("sales",)}


@pytest.mark.parametrize("role", ["treatments", "organic_media"])
@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([1.0, None], ValueError, r"promotion.*missing values"),
        ([1.0, np.nan], ValueError, r"promotion.*(missing|NaN or infinite) values"),
        ([1.0, np.inf], ValueError, r"promotion.*NaN or infinite values"),
        ([1.0, -np.inf], ValueError, r"promotion.*NaN or infinite values"),
        (["yes", "no"], TypeError, r"value columns must be.*promotion"),
    ],
)
def test_prepare_data_validates_optional_input_values(frame_factory, role, values, error, message):
    # Nullable pandas attempts an integer cast while inferring columns containing infinity
    with np.errstate(invalid="ignore"):
        source = frame_factory({"week": [1, 2], "promotion": values})
    with pytest.raises(error, match=message):
        prepare_data(source, time="week", **{role: ["promotion"]})


def test_prepare_data_rejects_duplicate_treatment_observations(frame_factory):
    source = frame_factory({"week": [1, 1], "promotion": [True, False]})
    with pytest.raises(ValueError, match=r"duplicate observations.*week"):
        prepare_data(source, time="week", treatments=["promotion"])


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("values", [[True, False], [2, 3], [0.5, 1.25]])
def test_prepare_data_accepts_organic_only_inputs(frame_factory, grouped, with_history, values):
    source = frame_factory({"week": [3, 2], "region": ["west", "west"], "newsletter": values})
    history = frame_factory({"week": [1], "region": ["west"], "newsletter": [values[0]]})
    data = prepare_data(
        source,
        time="week",
        groups=["region"] if grouped else (),
        organic_media=("newsletter",),
        media_history=history if with_history else None,
    )

    assert set(data.arrays) == {"organic_media"}
    assert data.columns == {"organic_media": ("newsletter",)}
    assert data.channels == ()
    assert data.organic_channels == ("newsletter",)
    assert data.time_values == (2, 3)
    assert data.media_time_values == ((1, 2, 3) if with_history else (2, 3))
    expected = ([values[0]] if with_history else []) + [values[1], values[0]]
    expected = [[[value]] for value in expected] if grouped else [[value] for value in expected]
    np.testing.assert_array_equal(data.arrays["organic_media"], expected)
    assert data.arrays["organic_media"].dtype == np.asarray(values).dtype


def test_prepare_data_rejects_negative_organic_exposure(frame_factory):
    source = frame_factory({"week": [1, 2], "newsletter": [1.0, -1.0]})
    with pytest.raises(ValueError, match=r"organic_media.*newsletter.*negative"):
        prepare_data(source, time="week", organic_media=["newsletter"])


def test_prepare_data_keeps_reach_frequency_separate_and_aligns_history(frame_factory):
    source = frame_factory(
        {
            "week": ["2026-01-19", "2026-01-12", "2026-01-12", "2026-01-19"],
            "region": ["west", "east", "west", "east"],
            "sales": [40, 3, 30, 4],
            "video": [4.0, 0.3, 3.0, 0.4],
            "newsletter": [40, 3, 30, 4],
            "reach_a": [40.5, 3.5, 30.5, 4.5],
            "reach_b": [4.0, 0.0, 3.0, 0.5],
            "frequency_a": [4.0, 0.3, 3.0, 0.4],
            "frequency_b": [0.0, 0.75, 0.25, 0.5],
            "cost_a": [8.0, 0.6, 6.0, 0.8],
            "cost_b": [4.0, 0.3, 3.0, 0.4],
            "organic_a": [40, 3, 30, 4],
            "organic_b": [80, 6, 60, 8],
            "organic_c": [120, 9, 90, 12],
            "views_a": [0.4, 0.03, 0.3, 0.04],
            "views_b": [0.8, 0.06, 0.6, 0.08],
            "views_c": [1.2, 0.09, 0.9, 0.12],
        }
    )
    history = frame_factory(
        {
            "week": ["2026-01-05", "2026-01-05"],
            "region": ["east", "west"],
            "video": [0.2, 2.0],
            "newsletter": [2, 20],
            "reach_a": [2.5, 20.5],
            "reach_b": [2, 20],
            "frequency_a": [0.2, 2.0],
            "frequency_b": [0.5, 5.0],
            "organic_a": [2, 20],
            "organic_b": [4, 40],
            "organic_c": [6, 60],
            "views_a": [0.02, 0.2],
            "views_b": [0.04, 0.4],
            "views_c": [0.06, 0.6],
        }
    )
    before = nw.from_native(source).to_dict(as_series=False)
    data = prepare_data(
        source,
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video"],
        spend=["cost_a"],
        organic_media=["newsletter"],
        reach=["reach_b", "reach_a"],
        media_frequency=["frequency_b", "frequency_a"],
        rf_spend=["cost_b", "cost_a"],
        channels=["Video"],
        organic_channels=["Email"],
        rf_channels=["B", "A"],
        organic_reach=["organic_c", "organic_a", "organic_b"],
        organic_frequency=["views_c", "views_a", "views_b"],
        organic_rf_channels=["C", "A", "B"],
        frequency="weekly",
        media_history=history,
    )

    assert data.time_values == ("2026-01-12", "2026-01-19")
    assert data.media_time_values == ("2026-01-05", "2026-01-12", "2026-01-19")
    assert data.group_values == (("west",), ("east",))
    assert (data.channels, data.organic_channels, data.rf_channels) == (("Video",), ("Email",), ("B", "A"))
    assert data.columns["reach"] == ("reach_b", "reach_a")
    assert data.columns["media_frequency"] == ("frequency_b", "frequency_a")
    assert data.columns["rf_spend"] == ("cost_b", "cost_a")
    assert data.organic_rf_channels == ("C", "A", "B")
    assert data.columns["organic_reach"] == ("organic_c", "organic_a", "organic_b")
    assert data.columns["organic_frequency"] == ("views_c", "views_a", "views_b")
    np.testing.assert_array_equal(
        data.arrays["reach"], [[[20, 20.5], [2, 2.5]], [[3, 30.5], [0, 3.5]], [[4, 40.5], [0.5, 4.5]]]
    )
    np.testing.assert_array_equal(
        data.arrays["media_frequency"], [[[5, 2], [0.5, 0.2]], [[0.25, 3], [0.75, 0.3]], [[0, 4], [0.5, 0.4]]]
    )
    np.testing.assert_array_equal(data.arrays["rf_spend"], [[[3, 6], [0.3, 0.6]], [[4, 8], [0.4, 0.8]]])
    np.testing.assert_array_equal(data.arrays["media"], [[[2], [0.2]], [[3], [0.3]], [[4], [0.4]]])
    np.testing.assert_array_equal(data.arrays["organic_media"], [[[20], [2]], [[30], [3]], [[40], [4]]])
    np.testing.assert_array_equal(data.arrays["spend"], [[[6], [0.6]], [[8], [0.8]]])
    np.testing.assert_array_equal(data.arrays["outcome"], [[30, 3], [40, 4]])
    np.testing.assert_array_equal(
        data.arrays["organic_reach"],
        [[[60, 20, 40], [6, 2, 4]], [[90, 30, 60], [9, 3, 6]], [[120, 40, 80], [12, 4, 8]]],
    )
    np.testing.assert_array_equal(
        data.arrays["organic_frequency"],
        [
            [[0.6, 0.2, 0.4], [0.06, 0.02, 0.04]],
            [[0.9, 0.3, 0.6], [0.09, 0.03, 0.06]],
            [[1.2, 0.4, 0.8], [0.12, 0.04, 0.08]],
        ],
    )
    data.arrays["rf_spend"][...] = 0
    np.testing.assert_array_equal(data.arrays["spend"], [[[6], [0.6]], [[8], [0.8]]])
    assert nw.from_native(source).to_dict(as_series=False) == before


def test_prepare_data_accepts_reach_frequency_without_spend_or_other_media():
    data = prepare_data(
        pl.DataFrame({"week": [2, 1], "audience": [0.0, 2.5], "views": [0.0, 0.5]}),
        time="week",
        reach=("audience",),
        media_frequency=("views",),
    )

    assert set(data.arrays) == {"reach", "media_frequency"}
    assert data.channels == data.organic_channels == ()
    assert data.rf_channels == ("audience",)
    assert data.media_time_values == data.time_values == (1, 2)
    np.testing.assert_array_equal(data.arrays["reach"], [[2.5], [0]])
    np.testing.assert_array_equal(data.arrays["media_frequency"], [[0.5], [0]])


@pytest.mark.parametrize(
    "selection,error,message",
    [
        ({"reach": ["r"]}, ValueError, "reach.*media_frequency"),
        ({"media_frequency": ["f"]}, ValueError, "reach.*media_frequency"),
        ({"rf_spend": ["c"]}, ValueError, "rf_spend requires.*reach"),
        ({"reach": ["r", "c"], "media_frequency": ["f"]}, ValueError, "media_frequency.*one column per reach"),
        ({"reach": ["r"], "media_frequency": ["f"], "rf_spend": ["c", "r"]}, ValueError, "rf_spend.*one column"),
        ({"outcome": "r", "rf_channels": ["R"]}, ValueError, "rf_channels requires reach"),
    ],
)
def test_prepare_data_rejects_unpaired_reach_frequency_selections(selection, error, message):
    with pytest.raises(error, match=message):
        prepare_data(pl.DataFrame({"week": [1], "r": [2], "f": [0.5], "c": [1]}), time="week", **selection)


@pytest.mark.parametrize(
    "selection,error,message",
    [
        ({"reach": "r"}, TypeError, "reach must be a sequence"),
        ({"media_frequency": []}, ValueError, "media_frequency must select at least one"),
        ({"rf_spend": [""]}, ValueError, "rf_spend must select at least one"),
        ({"reach": ["r", "r"]}, ValueError, "reach contains repeated columns"),
        ({"media_frequency": ["missing"]}, ValueError, "missing columns.*missing"),
        ({"rf_spend": ["week"]}, ValueError, "column declarations contain repeated names.*week"),
        ({"rf_channels": "R"}, TypeError, "rf_channels must be a sequence"),
        ({"rf_channels": []}, ValueError, "rf_channels must contain one name"),
        ({"rf_channels": [""]}, ValueError, "rf_channels must contain only nonempty"),
        ({"rf_channels": ["R", "R"]}, ValueError, "rf_channels must contain unique names"),
    ],
)
def test_prepare_data_validates_reach_frequency_selectors_and_labels(selection, error, message):
    inputs = {"reach": ["r"], "media_frequency": ["f"], **selection}
    with pytest.raises(error, match=message):
        prepare_data(pl.DataFrame({"week": [1], "r": [2], "f": [0.5]}), time="week", **inputs)


@pytest.mark.parametrize("role", ["reach", "media_frequency", "rf_spend", "organic_reach", "organic_frequency"])
@pytest.mark.parametrize(
    "value,error,message",
    [
        (-1.0, ValueError, "negative"),
        (None, ValueError, "missing values"),
        (np.inf, ValueError, "NaN or infinite"),
        (np.nan, ValueError, "NaN or infinite"),
        ("bad", TypeError, "value columns must be"),
    ],
)
def test_prepare_data_validates_reach_frequency_values(role, value, error, message):
    columns = {
        "week": [1],
        "reach": [2.5],
        "media_frequency": [0.5],
        "rf_spend": [1.0],
        "organic_reach": [1.5],
        "organic_frequency": [0.25],
        role: [value],
    }
    with pytest.raises(error, match=message):
        prepare_data(
            pl.DataFrame(columns),
            time="week",
            reach=["reach"],
            media_frequency=["media_frequency"],
            rf_spend=["rf_spend"],
            organic_reach=["organic_reach"],
            organic_frequency=["organic_frequency"],
        )


def test_align_to_reorders_reach_frequency_groups_features_and_history():
    source = pl.DataFrame(
        {
            "week": [3, 2, 2, 3],
            "region": ["east", "east", "west", "west"],
            "ra": [4, 3, 30, 40],
            "rb": [8, 6, 60, 80],
            "fa": [0.4, 0.3, 3.0, 4.0],
            "fb": [0.8, 0.6, 6.0, 8.0],
            "ca": [40, 30, 300, 400],
            "cb": [80, 60, 600, 800],
        }
    )
    reference = prepare_data(
        source.reverse(),
        time="week",
        groups=["region"],
        reach=["ra", "rb"],
        media_frequency=["fa", "fb"],
        rf_spend=["ca", "cb"],
        rf_channels=["A", "B"],
    )
    prediction = prepare_data(
        source,
        time="week",
        groups=["region"],
        reach=["rb", "ra"],
        media_frequency=["fb", "fa"],
        rf_spend=["cb", "ca"],
        rf_channels=["B", "A"],
        media_history=pl.DataFrame(
            {
                "week": [1, 1],
                "region": ["west", "east"],
                "ra": [20, 2],
                "rb": [40, 4],
                "fa": [2.0, 0.2],
                "fb": [4.0, 0.4],
            }
        ),
    )
    aligned = prediction._align_to(reference)

    assert aligned.rf_channels == ("A", "B")
    assert aligned.columns == reference.columns
    assert aligned.group_values == (("west",), ("east",))
    assert aligned.time_values == (2, 3)
    assert aligned.media_time_values == (1, 2, 3)
    np.testing.assert_array_equal(aligned.arrays["reach"], [[[20, 40], [2, 4]], [[30, 60], [3, 6]], [[40, 80], [4, 8]]])
    np.testing.assert_array_equal(
        aligned.arrays["media_frequency"], [[[2, 4], [0.2, 0.4]], [[3, 6], [0.3, 0.6]], [[4, 8], [0.4, 0.8]]]
    )
    np.testing.assert_array_equal(aligned.arrays["rf_spend"], [[[300, 600], [30, 60]], [[400, 800], [40, 80]]])
    assert prediction.rf_channels == ("B", "A")
    for role in ("reach", "media_frequency", "rf_spend"):
        assert not np.shares_memory(aligned.arrays[role], prediction.arrays[role])


@pytest.mark.parametrize("role", ["reach", "media_frequency", "rf_spend", "organic_reach", "organic_frequency"])
def test_align_to_rejects_changed_reach_frequency_channel_assignments(role):
    source = pl.DataFrame({"week": [1], "a": [1], "b": [2]})
    selection = {
        "reach": ["a", "b"],
        "media_frequency": ["a", "b"],
        "rf_spend": ["a", "b"],
        "organic_reach": ["a", "b"],
        "organic_frequency": ["a", "b"],
    }
    labels = {"rf_channels": ["A", "B"], "organic_rf_channels": ["A", "B"]}
    reference = prepare_data(source, time="week", **labels, **selection)
    prediction = prepare_data(source, time="week", **labels, **{**selection, role: ["b", "a"]})

    with pytest.raises(ValueError, match=rf"channel labels for '{role}'.*channel-to-column assignments"):
        prediction._align_to(reference)


def test_prepared_reach_frequency_to_jax_preserves_inputs_and_uses_float_precision():
    data = prepare_data(
        pl.DataFrame({"week": [1], "r": [2.5], "f": [0.5], "c": [1.25]}),
        time="week",
        reach=["r"],
        media_frequency=["f"],
        rf_spend=["c"],
        organic_reach=["r"],
        organic_frequency=["f"],
    )
    device = jax.devices("cpu")[0]
    inputs = data._to_jax(dtype=np.float32, device=device)
    jax.block_until_ready(inputs)

    assert set(inputs) == {"reach", "media_frequency", "rf_spend", "organic_reach", "organic_frequency"}
    for role, expected in (
        ("reach", [[2.5]]),
        ("media_frequency", [[0.5]]),
        ("rf_spend", [[1.25]]),
        ("organic_reach", [[2.5]]),
        ("organic_frequency", [[0.5]]),
    ):
        assert inputs[role].dtype == np.float32
        assert inputs[role].devices() == {device}
        np.testing.assert_array_equal(data.arrays[role], expected)
        data.arrays[role][...] = 0
        np.testing.assert_array_equal(inputs[role], expected)


def test_prepare_data_accepts_only_organic_reach_frequency_with_history():
    data = prepare_data(
        pl.DataFrame({"week": [3, 2], "audience": [0.0, 2.5], "views": [0.0, 0.5]}),
        time="week",
        organic_reach=("audience",),
        organic_frequency=("views",),
        media_history=pl.DataFrame({"week": [1], "audience": [1.5], "views": [0.25]}),
    )

    assert set(data.arrays) == {"organic_reach", "organic_frequency"}
    assert data.channels == data.organic_channels == data.rf_channels == ()
    assert data.organic_rf_channels == ("audience",)
    assert data.time_values == (2, 3)
    assert data.media_time_values == (1, 2, 3)
    np.testing.assert_array_equal(data.arrays["organic_reach"], [[1.5], [2.5], [0]])
    np.testing.assert_array_equal(data.arrays["organic_frequency"], [[0.25], [0.5], [0]])


@pytest.mark.parametrize(
    "selection,message",
    [
        ({"organic_reach": ["r"]}, "organic_reach.*organic_frequency.*together"),
        ({"organic_frequency": ["f"]}, "organic_reach.*organic_frequency.*together"),
        ({"organic_reach": ["r", "f"], "organic_frequency": ["f"]}, "organic_frequency.*one column per"),
        ({"outcome": "r", "organic_rf_channels": ["R"]}, "organic_rf_channels requires organic_reach"),
    ],
)
def test_prepare_data_rejects_unpaired_organic_reach_frequency_selections(selection, message):
    with pytest.raises(ValueError, match=message):
        prepare_data(pl.DataFrame({"week": [1], "r": [2], "f": [0.5]}), time="week", **selection)


@pytest.mark.parametrize(
    "selection,error,message",
    [
        ({"organic_reach": "r"}, TypeError, "organic_reach must be a sequence"),
        ({"organic_frequency": []}, ValueError, "organic_frequency must select at least one"),
        ({"organic_reach": ["r", "r"]}, ValueError, "organic_reach contains repeated columns"),
        ({"organic_frequency": ["missing"]}, ValueError, "missing columns.*missing"),
        ({"organic_reach": ["week"]}, ValueError, "column declarations contain repeated names.*week"),
        ({"organic_rf_channels": "R"}, TypeError, "organic_rf_channels must be a sequence"),
        ({"organic_rf_channels": []}, ValueError, "organic_rf_channels must contain one name"),
        ({"organic_rf_channels": [""]}, ValueError, "organic_rf_channels must contain only nonempty"),
        ({"organic_rf_channels": ["R", "R"]}, ValueError, "organic_rf_channels must contain unique names"),
    ],
)
def test_prepare_data_validates_organic_reach_frequency_selectors_and_labels(selection, error, message):
    inputs = {"organic_reach": ["r"], "organic_frequency": ["f"], **selection}
    with pytest.raises(error, match=message):
        prepare_data(pl.DataFrame({"week": [1], "r": [2], "f": [0.5]}), time="week", **inputs)


def test_align_to_reorders_organic_reach_frequency_history_and_leaves_paid_inputs_omitted():
    source = pl.DataFrame(
        {
            "week": [3, 2, 2, 3],
            "region": ["east", "east", "west", "west"],
            "ra": [4, 3, 30, 40],
            "rb": [8, 6, 60, 80],
            "fa": [0.4, 0.3, 3.0, 4.0],
            "fb": [0.8, 0.6, 6.0, 8.0],
        }
    )
    reference = prepare_data(
        source.reverse(),
        time="week",
        groups=["region"],
        organic_reach=["ra", "rb"],
        organic_frequency=["fa", "fb"],
        organic_rf_channels=["A", "B"],
        reach=["ra"],
        media_frequency=["fa"],
    )
    prediction = prepare_data(
        source,
        time="week",
        groups=["region"],
        organic_reach=["rb", "ra"],
        organic_frequency=["fb", "fa"],
        organic_rf_channels=["B", "A"],
        media_history=pl.DataFrame(
            {
                "week": [1, 1],
                "region": ["west", "east"],
                "ra": [20, 2],
                "rb": [40, 4],
                "fa": [2.0, 0.2],
                "fb": [4.0, 0.4],
            }
        ),
    )
    aligned = prediction._align_to(reference)

    assert set(aligned.arrays) == {"organic_reach", "organic_frequency"}
    assert aligned.rf_channels == ()
    assert aligned.organic_rf_channels == ("A", "B")
    assert aligned.columns == {"organic_reach": ("ra", "rb"), "organic_frequency": ("fa", "fb")}
    assert aligned.group_values == (("west",), ("east",))
    assert aligned.time_values == (2, 3)
    assert aligned.media_time_values == (1, 2, 3)
    np.testing.assert_array_equal(
        aligned.arrays["organic_reach"], [[[20, 40], [2, 4]], [[30, 60], [3, 6]], [[40, 80], [4, 8]]]
    )
    np.testing.assert_array_equal(
        aligned.arrays["organic_frequency"], [[[2, 4], [0.2, 0.4]], [[3, 6], [0.3, 0.6]], [[4, 8], [0.4, 0.8]]]
    )
    assert prediction.organic_rf_channels == ("B", "A")
    for role in ("organic_reach", "organic_frequency"):
        assert not np.shares_memory(aligned.arrays[role], prediction.arrays[role])


@pytest.mark.parametrize("with_media", [False, True])
def test_prepare_data_without_history_keeps_the_existing_time_window(frame_factory, with_media):
    data = prepare_data(
        frame_factory({"week": [2, 1], "sales": [20, 10], "video": [4.0, 2.0]}),
        time="week",
        outcome="sales",
        media=["video"] if with_media else None,
    )

    assert data.time_values == (1, 2)
    assert data.media_time_values == ((1, 2) if with_media else ())
    np.testing.assert_array_equal(data.arrays["outcome"], [10, 20])
    if with_media:
        np.testing.assert_array_equal(data.arrays["media"], [[2.0], [4.0]])


def test_media_history_extends_only_media_and_aligns_nested_groups(frame_factory):
    frame = frame_factory(
        {
            "week": [4, 3, 3, 4],
            "region": ["west", "east", "west", "east"],
            "store": [1, 2, 1, 2],
            "sales": [40, 3, 30, 4],
            "video": [40.5, 3.0, 30.0, 4.0],
            "search": [400.0, 30.0, 300.0, 40.0],
            "cost": [8.0, 0.6, 6.0, 0.8],
            "promotion": [True, False, True, False],
            "price_change": [-1.0, 0.5, -2.0, 0.25],
            "newsletter": [4000, 300, 3000, 400],
            "blog": [80, 6, 60, 8],
        }
    )
    history = pl.DataFrame(
        {
            "week": [2, 1, 2, 1],
            "region": ["east", "west", "west", "east"],
            "store": [2, 1, 1, 2],
            "search": [20, 100, 200, 10],
            "video": [2, 10, 20, 1],
            "newsletter": [200, 1000, 2000, 100],
            "blog": [4, 20, 40, 2],
        }
    )
    data = prepare_data(
        frame,
        time="week",
        groups=["region", "store"],
        outcome="sales",
        media=["video", "search"],
        spend=["cost", "search"],
        controls=["promotion"],
        treatments=["price_change"],
        channels=["Video", "Search"],
        organic_media=["blog", "newsletter"],
        organic_channels=["Blog", "Email"],
        media_history=history,
    )

    assert data.time_values == (3, 4)
    assert data.media_time_values == (1, 2, 3, 4)
    assert data.group_values == (("west", 1), ("east", 2))
    assert data.channels == ("Video", "Search")
    assert data.organic_channels == ("Blog", "Email")
    assert data.columns["organic_media"] == ("blog", "newsletter")
    np.testing.assert_array_equal(
        data.arrays["organic_media"],
        [[[20, 1000], [2, 100]], [[40, 2000], [4, 200]], [[60, 3000], [6, 300]], [[80, 4000], [8, 400]]],
    )
    assert data.columns["media"] == ("video", "search")
    np.testing.assert_array_equal(
        data.arrays["media"],
        [[[10, 100], [1, 10]], [[20, 200], [2, 20]], [[30, 300], [3, 30]], [[40.5, 400], [4, 40]]],
    )
    np.testing.assert_array_equal(data.arrays["outcome"], [[30, 3], [40, 4]])
    np.testing.assert_array_equal(data.arrays["spend"], [[[6, 300], [0.6, 30]], [[8, 400], [0.8, 40]]])
    np.testing.assert_array_equal(data.arrays["controls"], [[[True], [False]], [[True], [False]]])
    np.testing.assert_array_equal(data.arrays["treatments"], [[[-2.0], [0.5]], [[-1.0], [0.25]]])
    assert np.issubdtype(data.arrays["outcome"].dtype, np.integer)
    assert data.arrays["controls"].dtype == np.bool_
    assert np.issubdtype(data.arrays["media"].dtype, np.floating)

    # History must survive internal reordering even though outcomes have fewer periods
    reference = prepare_data(
        pl.DataFrame({"week": [5, 5], "region": ["east", "west"], "store": [2, 1], "video": [0, 0], "search": [0, 0]}),
        time="week",
        groups=["region", "store"],
        media=["search", "video"],
        channels=["Search", "Video"],
    )
    aligned = reference._align_to(data)
    assert aligned.media_time_values == (5,)
    assert aligned.organic_channels == ()
    assert aligned.arrays["media"].shape == (1, 2, 2)
    reordered = data._align_to(data)
    assert reordered.media_time_values == data.media_time_values
    np.testing.assert_array_equal(reordered.arrays["media"], data.arrays["media"])
    assert not np.shares_memory(reordered.arrays["media"], data.arrays["media"])

    media_only = prepare_data(
        frame,
        time="week",
        groups=["region", "store"],
        media=["video", "search"],
        channels=["Video", "Search"],
        media_history=history,
    )._align_to(reference)
    assert media_only.time_values == (3, 4)
    assert media_only.media_time_values == (1, 2, 3, 4)
    assert media_only.group_values == (("east", 2), ("west", 1))
    assert media_only.channels == ("Search", "Video")
    np.testing.assert_array_equal(
        media_only.arrays["media"],
        [[[10, 1], [100, 10]], [[20, 2], [200, 20]], [[30, 3], [300, 30]], [[40, 4], [400, 40.5]]],
    )

    data.arrays["media"][...] = -1
    np.testing.assert_array_equal(history["video"].to_numpy(), [2, 10, 20, 1])
    np.testing.assert_array_equal(nw.from_native(frame)["video"].to_numpy(), [40.5, 3, 30, 4])
    np.testing.assert_array_equal(data.arrays["spend"][..., 1], [[300, 30], [400, 40]])


def test_media_history_supplies_carryover_without_creating_outcomes(frame_factory):
    data = prepare_data(
        pl.DataFrame({"week": [3, 4], "sales": [80, 40], "video": [20.0, 10.0]}),
        time="week",
        outcome="sales",
        media=["video"],
        spend=["video"],
        media_history=frame_factory({"week": [2, 1], "video": [60.0, 100.0]}),
    )
    inputs = data._to_jax()
    n_outcomes = len(data.time_values)

    def carried_media(alpha):
        carried = geometric_adstock(inputs["media"], alpha, max_lag=2, normalize=False)
        return carried[-n_outcomes:]

    # Check the finite weighted sum directly, without using another adstock implementation
    np.testing.assert_allclose(jax.jit(carried_media)(0.5), [[75.0], [35.0]])
    _, gradient = jax.jit(jax.value_and_grad(lambda alpha: carried_media(alpha).sum()))(0.5)
    np.testing.assert_allclose(gradient, 240.0)
    np.testing.assert_array_equal(inputs["outcome"], [80, 40])
    np.testing.assert_array_equal(inputs["spend"], [[20.0], [10.0]])


@pytest.mark.parametrize("last_history_time", [3, 4, 5])
def test_media_history_rejects_overlapping_or_later_periods(last_history_time):
    with pytest.raises(ValueError, match=r"media_history.*before.*3"):
        prepare_data(
            pl.DataFrame({"week": [3, 4], "video": [1, 2]}),
            time="week",
            media=["video"],
            media_history=pl.DataFrame({"week": [1, last_history_time], "video": [3, 4]}),
        )


def test_media_history_requires_media_columns():
    with pytest.raises(ValueError, match="media_history requires media, organic_media, reach or organic_reach columns"):
        prepare_data(
            pl.DataFrame({"week": [2], "sales": [10]}),
            time="week",
            outcome="sales",
            media_history=pl.DataFrame({"week": [1], "video": [5]}),
        )


@pytest.mark.parametrize("history_groups", [["west"], ["west", "east", "north"]])
def test_media_history_requires_the_same_groups(history_groups):
    with pytest.raises(ValueError, match="group labels do not match"):
        prepare_data(
            pl.DataFrame({"week": [2, 2], "region": ["west", "east"], "video": [10, 20]}),
            time="week",
            groups=["region"],
            media=["video"],
            media_history=pl.DataFrame(
                {"week": [1] * len(history_groups), "region": history_groups, "video": [1] * len(history_groups)}
            ),
        )


@pytest.mark.parametrize("role", ["media", "organic_media"])
@pytest.mark.parametrize(
    ("history", "message"),
    [
        ({"week": [1], "video": [-1.0]}, "negative"),
        ({"week": [1], "video": [None]}, "missing values"),
        ({"week": [1], "video": [np.inf]}, "infinite"),
        ({"week": [1], "other": [1]}, "missing columns.*video"),
        ({"week": [], "video": []}, "at least one observation"),
        ({"week": [1, 1], "video": [1, 2]}, "duplicate observations"),
    ],
)
def test_media_history_validates_selected_observations(frame_factory, role, history, message):
    # Nullable pandas inference tries an integer cast on infinity before our validation runs
    with np.errstate(invalid="ignore"):
        source = frame_factory(history)
    with pytest.raises(ValueError, match=message):
        prepare_data(
            pl.DataFrame({"week": [3], "video": [1]}),
            time="week",
            **{role: ["video"]},
            media_history=source,
        )


@pytest.mark.parametrize("missing", ["video", "newsletter", "audience", "views", "organic_audience", "organic_views"])
def test_media_history_requires_every_selected_exposure_column(missing):
    history = {
        "week": [1],
        "video": [10],
        "newsletter": [20],
        "audience": [30],
        "views": [0.5],
        "organic_audience": [15],
        "organic_views": [0.25],
    }
    del history[missing]
    with pytest.raises(ValueError, match=rf"missing columns.*{missing}"):
        prepare_data(
            pl.DataFrame(
                {
                    "week": [2],
                    "video": [30],
                    "newsletter": [40],
                    "audience": [50],
                    "views": [0.5],
                    "organic_audience": [25],
                    "organic_views": [0.25],
                }
            ),
            time="week",
            media=["video"],
            organic_media=["newsletter"],
            reach=["audience"],
            media_frequency=["views"],
            organic_reach=["organic_audience"],
            organic_frequency=["organic_views"],
            media_history=pl.DataFrame(history),
        )


@pytest.mark.parametrize("role", ["media", "organic_media"])
@pytest.mark.parametrize("representation", [str, date.fromisoformat, datetime.fromisoformat])
@pytest.mark.parametrize(
    ("frequency", "history_times", "times"),
    [
        ("weekly", ["2025-12-22", "2025-12-29"], ["2026-01-05", "2026-01-12"]),
        ("monthly", ["2026-01-30"], ["2026-02-28", "2026-03-30"]),
    ],
)
def test_media_history_checks_one_calendar_across_both_windows(
    frame_factory, role, representation, frequency, history_times, times
):
    data = prepare_data(
        frame_factory({"time": list(map(representation, times)), "video": [1, 2]}),
        time="time",
        **{role: ["video"]},
        frequency=frequency,
        media_history=frame_factory(
            {"time": list(map(representation, history_times)), "video": [3] * len(history_times)}
        ),
    )

    assert data.time_values == tuple(map(representation, times))
    assert data.media_time_values == tuple(map(representation, history_times + times))


@pytest.mark.parametrize("history_times", [["2025-12-15", "2025-12-22"], ["2025-12-15", "2025-12-29"]])
def test_media_history_rejects_calendar_gaps_within_history_or_at_the_boundary(history_times):
    with pytest.raises(ValueError, match="does not follow frequency='weekly'"):
        prepare_data(
            pl.DataFrame({"week": ["2026-01-05", "2026-01-12"], "video": [1, 2]}),
            time="week",
            media=["video"],
            frequency="weekly",
            media_history=pl.DataFrame({"week": history_times, "video": [3, 4]}),
        )


def test_media_history_reports_incompatible_time_labels():
    with pytest.raises(TypeError, match=r"media_history.*same.*time"):
        prepare_data(
            pl.DataFrame({"week": [date(2026, 1, 5)], "video": [1]}),
            time="week",
            media=["video"],
            media_history=pl.DataFrame({"week": ["2025-12-29"], "video": [2]}),
        )


def test_media_history_rejects_missing_group_periods(frame_factory):
    with pytest.raises(ValueError, match="missing 1 combinations"):
        prepare_data(
            pl.DataFrame({"week": [3, 3], "region": ["east", "west"], "video": [1, 2]}),
            time="week",
            groups=["region"],
            media=["video"],
            media_history=frame_factory({"week": [1, 1, 2], "region": ["east", "west", "west"], "video": [3, 4, 5]}),
        )


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


def test_align_to_reorders_treatments_independently_of_media_history(frame_factory):
    training = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 1],
                "region": ["west", "east"],
                "video": [10.0, 20.0],
                "promotion": [False, False],
                "price_change": [0.5, 0.25],
            }
        ),
        time="week",
        groups=["region"],
        media=["video"],
        treatments=["promotion", "price_change"],
    )
    prediction = prepare_data(
        frame_factory(
            {
                "week": [3, 2, 2, 3],
                "region": ["east", "east", "west", "west"],
                "video": [4.0, 3.0, 30.0, 40.0],
                "promotion": [True, False, True, False],
                "price_change": [-4.0, -3.0, -30.0, -40.0],
            }
        ),
        time="week",
        groups=["region"],
        media=["video"],
        treatments=["price_change", "promotion"],
        media_history=pl.DataFrame({"week": [1, 1], "region": ["east", "west"], "video": [2.0, 20.0]}),
    )
    before = prediction.arrays["treatments"].copy()

    aligned = prediction._align_to(training)

    assert aligned.time_values == (2, 3)
    assert aligned.media_time_values == (1, 2, 3)
    assert aligned.group_values == (("west",), ("east",))
    assert aligned.columns["treatments"] == ("promotion", "price_change")
    assert aligned.channels == ("video",)
    np.testing.assert_array_equal(aligned.arrays["treatments"], [[[1, -30], [0, -3]], [[0, -40], [1, -4]]])
    np.testing.assert_array_equal(aligned.arrays["media"], [[[20], [2]], [[30], [3]], [[40], [4]]])
    assert aligned.arrays["treatments"].dtype == prediction.arrays["treatments"].dtype
    assert prediction.group_values == (("east",), ("west",))
    assert prediction.columns["treatments"] == ("price_change", "promotion")

    aligned.arrays["treatments"][...] = 0
    np.testing.assert_array_equal(prediction.arrays["treatments"], before)
    np.testing.assert_array_equal(training.arrays["treatments"], [[[0, 0.5], [0, 0.25]]])


def test_align_to_reorders_organic_history_without_requiring_paid_media(frame_factory):
    reference = prepare_data(
        pl.DataFrame({"week": [1, 1], "region": ["west", "east"], "video": [1, 2], "email": [3, 4], "blog": [5, 6]}),
        time="week",
        groups=["region"],
        media=["video"],
        channels=["Paid"],
        organic_media=["email", "blog"],
        organic_channels=["Email", "Blog"],
    )
    prediction = prepare_data(
        frame_factory(
            {
                "week": [3, 2, 2, 3],
                "region": ["east", "east", "west", "west"],
                "email": [4, 3, 30, 40],
                "blog": [8, 6, 60, 80],
            }
        ),
        time="week",
        groups=["region"],
        organic_media=["blog", "email"],
        organic_channels=["Blog", "Email"],
        media_history=frame_factory({"week": [1, 1], "region": ["east", "west"], "email": [2, 20], "blog": [4, 40]}),
    )
    before = prediction.arrays["organic_media"].copy()
    aligned = prediction._align_to(reference)

    assert set(aligned.arrays) == {"organic_media"}
    assert aligned.columns == {"organic_media": ("email", "blog")}
    assert aligned.channels == ()
    assert aligned.organic_channels == ("Email", "Blog")
    assert aligned.group_values == (("west",), ("east",))
    assert aligned.time_values == (2, 3)
    assert aligned.media_time_values == (1, 2, 3)
    np.testing.assert_array_equal(
        aligned.arrays["organic_media"], [[[20, 40], [2, 4]], [[30, 60], [3, 6]], [[40, 80], [4, 8]]]
    )
    assert aligned.arrays["organic_media"].dtype == prediction.arrays["organic_media"].dtype
    assert prediction.organic_channels == ("Blog", "Email")
    aligned.arrays["organic_media"][...] = 0
    np.testing.assert_array_equal(prediction.arrays["organic_media"], before)


@pytest.mark.parametrize("labels", [["Email", "Blog"], ["Blog", "Other"]])
def test_align_to_rejects_changed_organic_channel_assignments(labels):
    source = pl.DataFrame({"week": [1], "email": [10], "blog": [20]})
    reference = prepare_data(source, time="week", organic_media=["email", "blog"], organic_channels=["Email", "Blog"])
    prediction = prepare_data(source, time="week", organic_media=["blog", "email"], organic_channels=labels)

    with pytest.raises(ValueError, match=r"channel labels for 'organic_media'.*channel-to-column assignments"):
        prediction._align_to(reference)


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


@pytest.mark.parametrize("role", ["media", "treatments", "organic_media"])
@pytest.mark.parametrize("columns", [["video"], ["video", "search", "social"], ["video", "social"]])
def test_align_to_rejects_missing_or_unexpected_columns(role, columns):
    source = pl.DataFrame({"week": [1], "video": [1.0], "search": [2.0], "social": [3.0]})
    training = prepare_data(source, time="week", **{role: ["video", "search"]})
    prediction = prepare_data(source, time="week", **{role: columns})

    with pytest.raises(ValueError, match=f"columns for '{role}' do not match") as error:
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


@pytest.mark.parametrize("role", ["controls", "treatments", "organic_media"])
def test_align_to_rejects_additional_inputs_and_different_outcome_columns(role):
    source = pl.DataFrame({"week": [1], "sales": [10], "other": [20]})
    training = prepare_data(source, time="week", outcome="sales")
    unknown = prepare_data(source, time="week", **{role: ["sales"]})
    renamed = prepare_data(source, time="week", outcome="other")

    with pytest.raises(ValueError, match=rf"inputs.*{role}.*do not exist in the reference"):
        unknown._align_to(training)
    with pytest.raises(ValueError, match=r"'outcome'.*Missing columns.*sales.*unexpected columns.*other"):
        renamed._align_to(training)


def test_align_to_does_not_fill_omitted_treatments():
    source = pl.DataFrame({"week": [1, 2], "video": [10.0, 20.0], "promotion": [True, False]})
    reference = prepare_data(source, time="week", media=["video"], treatments=["promotion"])
    selected = prepare_data(source, time="week", media=["video"])

    aligned = selected._align_to(reference)

    assert aligned.columns == {"media": ("video",)}
    assert set(aligned.arrays) == {"media"}
    assert aligned.channels == ("video",)
    np.testing.assert_array_equal(aligned.arrays["media"], [[10.0], [20.0]])


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
            "video_impressions": [14_000, 10_000, 12_000],
            "video_spend": [130.0, 80.0, 95.0],
            "search_impressions": [6_000, 5_000, 4_500],
            "search_spend": [60.0, 40.0, 50.0],
            "email_clicks": [90, 60, 80],
            "temperature": [12.0, 10.0, 14.0],
            "product_price": [10.0, 12.0, 11.0],
        }
    )

    data = prepare_data(
        source,
        time="week",
        outcome="sales",
        media=["video_impressions", "search_impressions"],
        organic_media=["email_clicks"],
        spend=["video_spend", "search_spend"],
        controls=["temperature"],
        treatments=["product_price"],
        channels=["video", "search"],
        organic_channels=["Email"],
        frequency="weekly",
    )
    inputs = data.arrays

    assert isinstance(data, PreparedData)
    assert data.time_values == ("2026-01-05", "2026-01-12", "2026-01-19")
    assert data.group_columns == data.group_values == ()
    assert data.channels == ("video", "search")
    assert data.organic_channels == ("Email",)
    assert data.columns == {
        "outcome": ("sales",),
        "media": ("video_impressions", "search_impressions"),
        "organic_media": ("email_clicks",),
        "spend": ("video_spend", "search_spend"),
        "controls": ("temperature",),
        "treatments": ("product_price",),
    }
    np.testing.assert_array_equal(inputs["media"], [[10_000, 5_000], [14_000, 6_000], [12_000, 4_500]])
    np.testing.assert_array_equal(inputs["organic_media"], [[60], [90], [80]])
    np.testing.assert_array_equal(inputs["spend"], [[80.0, 40.0], [130.0, 60.0], [95.0, 50.0]])
    np.testing.assert_array_equal(inputs["outcome"], [100, 140, 120])
    np.testing.assert_array_equal(inputs["controls"], [[10.0], [12.0], [14.0]])
    np.testing.assert_array_equal(inputs["treatments"], [[12.0], [10.0], [11.0]])


@pytest.mark.parametrize("x64", [False, True])
def test_prepared_data_to_jax_follows_precision_setting_without_changing_host_data(frame_factory, x64):
    source = frame_factory(
        {
            "week": [2, 1],
            "sales": np.array([20, 10], dtype=np.int64),
            "video": np.array([2.5, 1.5], dtype=np.float64),
            "email_clicks": np.array([12.25, 10.5], dtype=np.float64),
            "promotion": [True, False],
            "price_change": np.array([-0.5, 0.25], dtype=np.float64),
        }
    )
    data = prepare_data(
        source,
        time="week",
        outcome="sales",
        media=["video"],
        organic_media=["email_clicks"],
        controls=["promotion"],
        treatments=["price_change"],
    )
    before = {name: array.copy() for name, array in data.arrays.items()}

    with jax.enable_x64(x64):
        result = data._to_jax()
        assert jax.config.x64_enabled == x64

    assert set(result) == set(data.arrays)
    assert result["media"].dtype == (np.float64 if x64 else np.float32)
    assert result["organic_media"].dtype == (np.float64 if x64 else np.float32)
    assert result["outcome"].dtype == (np.int64 if x64 else np.int32)
    assert result["controls"].dtype == np.bool_
    assert result["treatments"].dtype == (np.float64 if x64 else np.float32)
    for name, array in result.items():
        assert isinstance(array, jax.Array)
        assert array.shape == before[name].shape
        np.testing.assert_array_equal(array, before[name])
        np.testing.assert_array_equal(data.arrays[name], before[name])
        assert data.arrays[name].dtype == before[name].dtype
    assert data.time_values == (1, 2)
    assert data.columns == {
        "outcome": ("sales",),
        "media": ("video",),
        "controls": ("promotion",),
        "treatments": ("price_change",),
        "organic_media": ("email_clicks",),
    }


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
        pl.DataFrame(
            {"week": [1, 2], "sales": [10, 20], "video": [1.5, 2.5], "email": [2, 1], "promotion": [True, False]}
        ),
        time="week",
        outcome="sales",
        media=["video"],
        organic_media=["email"],
        treatments=["promotion"],
    )
    device = jax.devices("cpu")[0]
    destination = jax.sharding.SingleDeviceSharding(device) if sharded else device

    result = data._to_jax(device=destination)

    for array in result.values():
        assert array.devices() == {device}
        assert array.committed


@pytest.mark.parametrize("role", ["outcome", "treatments", "organic_media"])
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
def test_prepared_data_to_jax_does_not_share_mutable_host_buffers(role, dtype):
    original = np.array([0, 1], dtype=dtype)
    selection = {role: "value" if role == "outcome" else ["value"]}
    data = prepare_data(pl.DataFrame({"week": [1, 2], "value": original}), time="week", **selection)
    result = data._to_jax(dtype=np.float32, device=jax.devices("cpu")[0])
    jax.block_until_ready(result)

    data.arrays[role][:] = 0

    assert result[role].dtype == dtype
    np.testing.assert_array_equal(result[role], original if role == "outcome" else original[:, None])


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


@pytest.mark.parametrize(
    "role,frequency_role,channel_axis",
    [
        ("media", None, "channels"),
        ("reach", "media_frequency", "rf_channels"),
        ("organic_reach", "organic_frequency", "organic_rf_channels"),
    ],
)
def test_prepared_data_keeps_hundreds_of_channels_aligned(frame_factory, role, frequency_role, channel_axis):
    expected = np.arange(3 * 8 * 465, dtype=np.float32).reshape(3, 8, 465)
    expected_frequency = np.arange(3 * 8 * 465, dtype=np.float32).reshape(3, 8, 465) / 8 + 0.125
    rows = [(time, group) for time in [2, 0, 1] for group in reversed(range(8))]
    source = frame_factory(
        {
            "week": [time for time, _ in rows],
            "geo": [group for _, group in rows],
            **{
                f"channel_{channel}": [expected[time, group, channel] for time, group in rows] for channel in range(465)
            },
            **(
                {
                    f"frequency_{channel}": [expected_frequency[time, group, channel] for time, group in rows]
                    for channel in range(465)
                }
                if frequency_role
                else {}
            ),
        }
    )
    channels = [f"channel_{channel}" for channel in reversed(range(465))]
    selections = {role: channels}
    expected_arrays = {role: expected}
    if frequency_role:
        selections[frequency_role] = [f"frequency_{channel}" for channel in reversed(range(465))]
        expected_arrays[frequency_role] = expected_frequency

    result = prepare_data(source, time="week", groups=["geo"], **selections)

    assert result.columns == {name: tuple(columns) for name, columns in selections.items()}
    assert getattr(result, channel_axis) == tuple(channels)
    assert result.group_values == tuple((group,) for group in reversed(range(8)))
    for name, values in expected_arrays.items():
        np.testing.assert_array_equal(result.arrays[name], values[:, ::-1, ::-1])
    assert len(jax.tree.leaves(result.arrays)) == len(expected_arrays)

    reference = prepare_data(
        nw.from_native(source).sort(["week", "geo"]).to_native(),
        time="week",
        groups=["geo"],
        **{name: list(reversed(columns)) for name, columns in selections.items()},
    )
    aligned = result._align_to(reference)

    for name, values in expected_arrays.items():
        np.testing.assert_array_equal(aligned.arrays[name], values)
        source_dtype = nw.from_native(source).get_column(selections[name][0]).to_numpy().dtype
        assert aligned.arrays[name].dtype == source_dtype
    assert getattr(aligned, channel_axis) == tuple(reversed(channels))
    assert aligned.columns == reference.columns
    assert aligned.group_values == reference.group_values
    assert len(jax.tree.leaves(aligned.arrays)) == len(expected_arrays)


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
        (
            {},
            ValueError,
            "select at least one of outcome, media, organic_media, reach, organic_reach, controls or treatments",
        ),
        ({"media": None}, ValueError, "select at least one"),
        ({"outcome": ["sales"]}, TypeError, "outcome must be a column name"),
        ({"outcome": ""}, ValueError, "outcome must select at least one nonempty"),
        ({"media": "sales"}, TypeError, "media must be a sequence"),
        ({"controls": "sales"}, TypeError, "controls must be a sequence"),
        ({"treatments": "sales"}, TypeError, "treatments must be a sequence"),
        ({"treatments": []}, ValueError, "treatments must select at least one"),
        ({"treatments": [1]}, ValueError, "treatments must select at least one"),
        ({"treatments": [""]}, ValueError, "treatments must select at least one"),
        ({"treatments": ["sales", "sales"]}, ValueError, "treatments contains repeated columns"),
        ({"treatments": ["missing"]}, ValueError, "data is missing columns.*missing"),
        ({"treatments": ["week"]}, ValueError, "column declarations contain repeated names.*week"),
        ({"organic_media": "sales"}, TypeError, "organic_media must be a sequence"),
        ({"organic_media": []}, ValueError, "organic_media must select at least one"),
        ({"organic_media": [1]}, ValueError, "organic_media must select at least one"),
        ({"organic_media": [""]}, ValueError, "organic_media must select at least one"),
        ({"organic_media": ["sales", "sales"]}, ValueError, "organic_media contains repeated columns"),
        ({"organic_media": ["missing"]}, ValueError, "data is missing columns.*missing"),
        ({"organic_media": ["week"]}, ValueError, "column declarations contain repeated names.*week"),
        ({"organic_media": ["sales"], "spend": ["sales"]}, ValueError, "spend requires media"),
        ({"organic_media": ["sales"], "channels": ["Email"]}, ValueError, "channels requires media"),
        ({"outcome": "sales", "organic_channels": ["Email"]}, ValueError, "organic_channels requires organic_media"),
        ({"organic_media": ["sales"], "organic_channels": "Email"}, TypeError, "organic_channels must be a sequence"),
        ({"organic_media": ["sales"], "organic_channels": []}, ValueError, "organic_channels must contain one name"),
        (
            {"organic_media": ["sales"], "organic_channels": [1]},
            ValueError,
            "organic_channels must contain only nonempty",
        ),
        (
            {"organic_media": ["sales"], "organic_channels": [""]},
            ValueError,
            "organic_channels must contain only nonempty",
        ),
        (
            {"organic_media": ["sales"], "organic_channels": ["Email", "Email"]},
            ValueError,
            "organic_channels must contain unique names",
        ),
        (
            {"organic_media": ["sales"], "organic_channels": ["Email", "Blog"]},
            ValueError,
            "organic_channels must contain one name",
        ),
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


@pytest.mark.parametrize("role", ["outcome", "treatments", "organic_media"])
def test_prepare_data_checks_calendar_and_panel_coverage(role):
    selection = {role: "sales" if role == "outcome" else ["sales"]}
    source = pl.DataFrame({"week": ["2026-01-05", "2026-01-19"], "sales": [10, 20]})

    with pytest.raises(ValueError, match="missing periods"):
        prepare_data(source, time="week", frequency="weekly", **selection)

    source = pl.DataFrame({"week": [1, 2, 1], "geo": ["east", "east", "west"], "sales": [10, 20, 30]})
    with pytest.raises(ValueError, match="same time values"):
        prepare_data(source, time="week", groups=["geo"], **selection)


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


def test_prepared_treatments_work_with_grouped_model_densities_and_gradients():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [2, 1, 1, 2],
                "region": ["west", "east", "west", "east"],
                "sales": [4.0, 2.0, 1.0, 5.0],
                "video": [4.0, 1.0, 3.0, 2.0],
                "promotion": [False, False, True, True],
                "price_change": [0.5, 0.25, -0.25, -0.5],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video"],
        treatments=["promotion", "price_change"],
    )

    def log_density(inputs, media_beta, treatment_beta):
        mean = jnp.sum(inputs["media"] * media_beta, axis=-1)
        mean += jnp.sum(inputs["treatments"] * treatment_beta, axis=-1)
        return normal(inputs["outcome"], mean, 2.0)

    model = Model({"media_beta": Real(shape=(2, 1)), "treatment_beta": Real(shape=(2, 2))}, log_density)
    media_beta = np.array([[0.5], [1.0]])
    treatment_beta = np.array([[2.0, -1.0], [3.0, 0.5]])
    position = {"media_beta": jnp.asarray(media_beta), "treatment_beta": jnp.asarray(treatment_beta)}
    inputs = data._to_jax()
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, inputs)

    # Use independently ordered arrays and the Normal score to catch swapped features or groups
    media = np.array([[[3.0], [1.0]], [[4.0], [2.0]]])
    treatments = np.array([[[1.0, -0.25], [0.0, 0.25]], [[0.0, 0.5], [1.0, -0.5]]])
    mean = np.sum(media * media_beta, axis=-1) + np.sum(treatments * treatment_beta, axis=-1)
    residual = np.array([[1.0, 2.0], [4.0, 5.0]]) - mean
    expected_density = -0.5 * np.sum((residual / 2.0) ** 2) - residual.size * np.log(2.0 * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(value, expected_density, rtol=1e-6)
    np.testing.assert_allclose(gradient["media_beta"], np.sum(media * residual[..., None] / 4.0, axis=0), rtol=1e-6)
    np.testing.assert_allclose(
        gradient["treatment_beta"], np.sum(treatments * residual[..., None] / 4.0, axis=0), rtol=1e-6
    )


def test_paid_and_organic_history_work_with_adstock_and_model_gradients():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [3, 2, 2, 3],
                "region": ["west", "east", "west", "east"],
                "sales": [15.0, 12.0, 10.0, 25.0],
                "video": [4.0, 2.0, 3.0, 5.0],
                "email": [2.0, 3.0, 1.0, 4.0],
                "social": [4.0, 1.0, 2.0, 3.0],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video"],
        organic_media=["email", "social"],
        media_history=pl.DataFrame(
            {"week": [1, 1], "region": ["east", "west"], "video": [1.0, 2.0], "email": [2.0, 4.0], "social": [3.0, 1.0]}
        ),
    )

    def expected_sales(inputs, media_decay, organic_decay, media_beta, organic_beta):
        paid = geometric_adstock(inputs["media"], media_decay, max_lag=1, normalize=False)
        organic = geometric_adstock(inputs["organic_media"], organic_decay, max_lag=1, normalize=False)
        n_times = inputs["outcome"].shape[0]
        return paid[-n_times:, ..., 0] * media_beta + jnp.sum(organic[-n_times:] * organic_beta, axis=-1)

    def log_density(inputs, **parameters):
        return normal(inputs["outcome"], expected_sales(inputs, **parameters), 1.0)

    def generate(key, inputs, **parameters):
        return {"mean": expected_sales(inputs, **parameters)}

    model = Model(
        {
            "media_decay": Real(),
            "organic_decay": Real(shape=(2,)),
            "media_beta": Real(),
            "organic_beta": Real(shape=(2,)),
        },
        log_density,
        generate,
    )
    position = {
        "media_decay": jnp.asarray(0.5),
        "organic_decay": jnp.array([0.25, 0.75]),
        "media_beta": jnp.asarray(2.0),
        "organic_beta": jnp.array([1.5, 0.5]),
    }
    inputs = data._to_jax()
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, inputs)
    generated = jax.jit(model.generate)(jax.random.key(0), position, inputs)

    # One lag gives current + decay * previous, including the history row for the first period
    paid = np.array([[4.0, 2.5], [5.5, 6.0]])
    organic = np.array([[[2.0, 2.75], [3.5, 3.25]], [[2.25, 5.5], [4.75, 3.75]]])
    mean = 2.0 * paid + np.sum(organic * np.array([1.5, 0.5]), axis=-1)
    residual = np.array([[10.0, 12.0], [15.0, 25.0]]) - mean
    expected_density = -0.5 * np.sum(residual**2) - 0.5 * residual.size * np.log(2.0 * np.pi)
    np.testing.assert_allclose(generated["mean"], mean, rtol=1e-6)
    np.testing.assert_allclose(value, expected_density, rtol=1e-6)
    np.testing.assert_allclose(gradient["media_beta"], np.sum(paid * residual), rtol=1e-6)
    np.testing.assert_allclose(gradient["organic_beta"], np.sum(organic * residual[..., None], axis=(0, 1)), rtol=1e-6)

    # The decay derivative uses the previous raw exposure multiplied by its coefficient
    previous_paid = np.array([[2.0, 1.0], [3.0, 2.0]])
    previous_organic = np.array([[[4.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 1.0]]])
    np.testing.assert_allclose(gradient["media_decay"], np.sum(2.0 * previous_paid * residual), rtol=1e-6)
    np.testing.assert_allclose(
        gradient["organic_decay"],
        np.sum(previous_organic * np.array([1.5, 0.5]) * residual[..., None], axis=(0, 1)),
        rtol=1e-6,
    )
    assert generated["mean"].shape == (len(data.time_values), len(data.group_values))
    assert inputs["media"].shape == (3, 2, 1)
    assert inputs["organic_media"].shape == (3, 2, 2)


def test_reach_frequency_history_works_with_grouped_model_and_adstock_gradients():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [3, 2, 2, 3],
                "region": ["west", "east", "west", "east"],
                "sales": [17.0, 9.0, 14.0, 16.0],
                "video_reach": [4, 2, 3, 5],
                "audio_reach": [3, 1, 2, 2],
                "video_frequency": [0.5, 0.5, 2.0, 1.0],
                "audio_frequency": [2.0, 2.0, 1.5, 1.5],
                "email_reach": [2, 3, 1, 4],
                "email_frequency": [1.5, 1.0, 2.0, 0.5],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        reach=["video_reach", "audio_reach"],
        media_frequency=["video_frequency", "audio_frequency"],
        rf_channels=["Video", "Audio"],
        organic_reach=["email_reach"],
        organic_frequency=["email_frequency"],
        organic_rf_channels=["Email"],
        media_history=pl.DataFrame(
            {
                "week": [1, 1],
                "region": ["east", "west"],
                "video_reach": [1, 2],
                "audio_reach": [3, 4],
                "video_frequency": [2.0, 1.5],
                "audio_frequency": [1.0, 0.5],
                "email_reach": [2, 6],
                "email_frequency": [2.5, 0.5],
            }
        ),
    )

    def expected_sales(inputs, paid_decay, organic_decay, paid_beta, organic_beta):
        # This test models total exposure so both reach and frequency affect the likelihood
        paid_exposure = inputs["reach"] * inputs["media_frequency"]
        organic_exposure = inputs["organic_reach"] * inputs["organic_frequency"]
        paid = geometric_adstock(paid_exposure, paid_decay, max_lag=1, normalize=False)
        organic = geometric_adstock(organic_exposure, organic_decay, max_lag=1, normalize=False)
        n_times = inputs["outcome"].shape[0]
        return jnp.sum(paid[-n_times:] * paid_beta, axis=-1) + jnp.sum(organic[-n_times:] * organic_beta, axis=-1)

    def log_density(inputs, **parameters):
        return normal(inputs["outcome"], expected_sales(inputs, **parameters), 2.0)

    def generate(key, inputs, **parameters):
        return {"mean": expected_sales(inputs, **parameters)}

    model = Model(
        {
            "paid_decay": Real(shape=(2,)),
            "organic_decay": Real(),
            "paid_beta": Real(shape=(2, 2)),
            "organic_beta": Real(shape=(2, 1)),
        },
        log_density,
        generate,
    )
    paid_beta = np.array([[0.5, 1.5], [2.0, 0.25]])
    organic_beta = np.array([[1.25], [0.5]])
    position = {
        "paid_decay": jnp.array([0.5, 0.25]),
        "organic_decay": jnp.asarray(0.75),
        "paid_beta": jnp.asarray(paid_beta),
        "organic_beta": jnp.asarray(organic_beta),
    }
    inputs = data._to_jax()
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, inputs)
    generated = jax.jit(model.generate)(jax.random.key(0), position, inputs)

    # Work out the one-lag sums independently so a swapped input or lost history changes the result
    paid = np.array([[[7.5, 3.5], [2.0, 2.75]], [[5.0, 6.75], [5.5, 3.5]]])
    organic = np.array([[[4.25], [6.75]], [[4.5], [4.25]]])
    mean = np.array([[14.3125, 8.0625], [18.25, 14.0]])
    residual = np.array([[14.0, 9.0], [17.0, 16.0]]) - mean
    score = residual / 4.0
    expected_density = -0.5 * np.sum((residual / 2.0) ** 2) - residual.size * np.log(2.0 * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(generated["mean"], mean, rtol=1e-6)
    np.testing.assert_allclose(value, expected_density, rtol=1e-6)
    np.testing.assert_allclose(gradient["paid_beta"], np.sum(paid * score[..., None], axis=0), rtol=1e-6)
    np.testing.assert_allclose(gradient["organic_beta"], np.sum(organic * score[..., None], axis=0), rtol=1e-6)

    # With one lag, differentiating decay leaves the previous period's raw exposure
    previous_paid = np.array([[[3.0, 2.0], [2.0, 3.0]], [[6.0, 3.0], [1.0, 2.0]]])
    previous_organic = np.array([[[3.0], [5.0]], [[2.0], [3.0]]])
    np.testing.assert_allclose(
        gradient["paid_decay"], np.sum(previous_paid * paid_beta * score[..., None], axis=(0, 1)), rtol=1e-6
    )
    np.testing.assert_allclose(
        gradient["organic_decay"], np.sum(previous_organic * organic_beta * score[..., None]), rtol=1e-6
    )
    assert generated["mean"].shape == (2, 2)
    assert inputs["reach"].shape == inputs["media_frequency"].shape == (3, 2, 2)
    assert inputs["organic_reach"].shape == inputs["organic_frequency"].shape == (3, 2, 1)
    assert data.rf_channels == ("Video", "Audio")
    assert data.organic_rf_channels == ("Email",)


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


@pytest.mark.parametrize("role", ["media", "organic_media"])
def test_prepare_data_works_with_batched_adstock(frame_factory, role):
    source = frame_factory(
        {"week": [2, 1, 1, 2], "geo": ["west", "east", "west", "east"], "video": [4.0, 1.0, 3.0, 2.0]}
    )
    data = prepare_data(source, time="week", groups=["geo"], **{role: ["video"]})

    carried = jax.jit(lambda media: geometric_adstock(media, 0.5, max_lag=1, normalize=False))(data._to_jax()[role])

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
