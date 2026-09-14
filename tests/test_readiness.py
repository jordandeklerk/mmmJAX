"""Tests for labeled marketing data readiness checks."""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

from mmmjax import fit_data_scaling, prepare_data
from mmmjax.readiness import check_data


def _pair(report, first, second):
    pairs = report["pairs"].to_dataset()
    left, right = pairs["feature_a"].values, pairs["feature_b"].values
    indices = np.flatnonzero(((left == first) & (right == second)) | ((left == second) & (right == first)))
    assert indices.size == 1
    return pairs.isel(pair=int(indices[0]))


def test_check_data_reports_hand_calculated_series_statistics():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 2, 3, 4, 5],
                "sales": [-2, 0, 2, 0, 0],
                "impressions": [0, 0, 2, 0, 4],
                "inactive": [0, 0, 0, 0, 0],
                "constant": [7, 7, 7, 7, 7],
                "promotion": [True, False, False, True, False],
            }
        ),
        time="week",
        outcome="sales",
        media=["impressions", "inactive", "constant"],
        channels=["Search", "Unused", "Always on"],
        treatments=["promotion"],
    )

    report = check_data(data, max_zero_fraction=0.6)

    assert isinstance(report, xr.DataTree)
    assert set(report.children) == {"coverage", "series", "pairs"}
    assert report["series"]["mean"].dims == ("feature",)
    series = report["series"].to_dataset()
    media = series.sel(feature="media.impressions")
    assert media["minimum"].item() == 0
    assert media["maximum"].item() == 4
    assert media["mean"].item() == pytest.approx(1.2)
    assert media["std"].item() == pytest.approx(1.6)
    assert media["nonzero_periods"].item() == 2
    assert media["zero_fraction"].item() == pytest.approx(0.6)
    assert media["longest_zero_run"].item() == 2
    assert not media["constant"].item()
    assert media["sparse"].item()
    assert media["role"].item() == "media"
    assert media["column"].item() == "impressions"
    assert media["channel"].item() == "Search"

    inactive = series.sel(feature="media.inactive")
    assert inactive["constant"].item()
    assert inactive["sparse"].item()
    assert inactive["std"].item() == 0
    assert inactive["longest_zero_run"].item() == 5
    constant = series.sel(feature="media.constant")
    assert constant["constant"].item()
    assert not constant["sparse"].item()
    assert constant["longest_zero_run"].item() == 0
    assert constant["std"].item() == 0

    outcome = series.sel(feature="outcome.sales")
    assert outcome["minimum"].item() == -2
    assert outcome["nonzero_periods"].item() == 2
    assert outcome["channel"].item() == ""
    promotion = series.sel(feature="treatments.promotion")
    assert promotion["mean"].item() == pytest.approx(0.4)
    assert promotion["longest_zero_run"].item() == 2
    assert promotion["channel"].item() == ""


def test_check_data_excludes_exposure_history_from_statistics_and_pairs():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": ["2026-01-19", "2026-01-05", "2026-01-12"],
                "impressions": [0, 0, 2],
                "clicks": [0, 0, 4],
                "cost": [0, 1, 0],
            }
        ),
        time="week",
        media=["impressions"],
        organic_media=["clicks"],
        spend=["cost"],
        media_history=pl.DataFrame(
            {
                "week": ["2025-12-22", "2025-12-29"],
                "impressions": [100, 200],
                "clicks": [500, 0],
            }
        ),
    )

    report = check_data(data)

    coverage = report["coverage"]
    assert coverage.attrs["n_periods"] == 3
    assert coverage.attrs["n_groups"] == 1
    assert coverage.attrs["n_history_periods"] == 2
    assert coverage.attrs["frequency"] == "weekly"
    np.testing.assert_array_equal(coverage["time"], data.time_values)
    np.testing.assert_array_equal(coverage["media_time"], data.media_time_values)
    media = report["series"].to_dataset().sel(feature="media.impressions")
    assert media["maximum"].item() == 2
    assert media["mean"].item() == pytest.approx(2 / 3)
    assert media["zero_fraction"].item() == pytest.approx(2 / 3)
    assert media["longest_zero_run"].item() == 1
    spend = report["series"].to_dataset().sel(feature="spend.cost")
    assert spend["mean"].item() == pytest.approx(1 / 3)
    pair = _pair(report, "media.impressions", "organic_media.clicks")
    assert pair["correlation"].item() == pytest.approx(1)
    assert pair["joint_active_observations"].item() == 1
    assert pair["matching_activity"].item()


def test_check_data_labels_every_role_without_merging_shared_channel_names():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 2, 3],
                "sales": [2, 4, 6],
                "value": [1.0, 1.5, 2.0],
                "population": [100, 100, 100],
                "impressions": [0, 2, 3],
                "clicks": [0, 3, 4],
                "audience": [0, 4, 5],
                "views": [0.5, 0, 0.25],
                "organic_audience": [0, 2, 4],
                "organic_views": [0.0, 0.5, 1.5],
                "cost": [0, 1, 2],
                "temperature": [-2, 0, 2],
                "promotion": [False, True, False],
            }
        ),
        time="week",
        outcome="sales",
        revenue_per_outcome="value",
        population="population",
        media=["impressions"],
        channels=["Shared"],
        organic_media=["clicks"],
        organic_channels=["Shared"],
        reach=["audience"],
        media_frequency=["views"],
        rf_channels=["Shared"],
        organic_reach=["organic_audience"],
        organic_frequency=["organic_views"],
        organic_rf_channels=["Shared"],
        spend=["cost"],
        rf_spend=["cost"],
        controls=["temperature"],
        treatments=["promotion"],
    )

    report = check_data(data)

    series = report["series"].to_dataset()
    expected = {
        "outcome.sales",
        "media.impressions",
        "organic_media.clicks",
        "reach.audience",
        "media_frequency.views",
        "organic_reach.organic_audience",
        "organic_frequency.organic_views",
        "spend.cost",
        "rf_spend.cost",
        "controls.temperature",
        "treatments.promotion",
    }
    assert set(series["feature"].values) == expected
    for feature in expected:
        role, column = feature.split(".", maxsplit=1)
        row = series.sel(feature=feature)
        assert row["role"].item() == role
        assert row["column"].item() == column
        assert row["channel"].item() == ("" if role in ("outcome", "controls", "treatments") else "Shared")
    predictors = expected - {"outcome.sales", "spend.cost", "rf_spend.cost"}
    pairs = report["pairs"]
    assert pairs.sizes["pair"] == len(predictors) * (len(predictors) - 1) // 2
    actual_pairs = [frozenset(pair) for pair in zip(pairs["feature_a"].values, pairs["feature_b"].values, strict=True)]
    assert len(set(actual_pairs)) == len(actual_pairs)
    assert all(len(pair) == 2 and pair <= predictors for pair in actual_pairs)
    for first, second in (
        ("reach.audience", "organic_reach.organic_audience"),
        ("media.impressions", "organic_media.clicks"),
    ):
        assert _pair(report, first, second)["activity_overlap"].item() == 1
    frequency_pair = _pair(report, "reach.audience", "media_frequency.views")
    assert np.isnan(frequency_pair["activity_overlap"].item())


def test_check_data_computes_correlation_and_temporal_activity_overlap():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 2, 3, 4],
                "a": [0, 1, 0, 2],
                "b": [0, 2, 0, 4],
                "c": [0, 0, 3, 0],
                "always": [1, 1, 1, 1],
                "baseline": [2, 2, 2, 2],
                "negative": [0, -1, 0, -2],
            }
        ),
        time="week",
        media=["a", "b", "c", "always", "baseline"],
        controls=["negative"],
    )

    report = check_data(data)

    matching = _pair(report, "media.a", "media.b")
    assert matching["correlation"].item() == pytest.approx(1)
    assert matching["high_correlation"].item()
    assert matching["activity_overlap"].item() == 1
    assert matching["matching_activity"].item()
    assert matching["joint_active_observations"].item() == 2
    assert matching["exclusive_active_observations"].item() == 0
    exclusive = _pair(report, "media.a", "media.c")
    assert exclusive["activity_overlap"].item() == 0
    assert not exclusive["matching_activity"].item()
    assert exclusive["joint_active_observations"].item() == 0
    assert exclusive["exclusive_active_observations"].item() == 3
    constants = _pair(report, "media.always", "media.baseline")
    assert np.isnan(constants["correlation"].item())
    assert not constants["high_correlation"].item()
    assert constants["activity_overlap"].item() == 1
    assert not constants["matching_activity"].item()
    assert constants["joint_active_observations"].item() == 4
    non_exposure = _pair(report, "media.a", "controls.negative")
    assert non_exposure["correlation"].item() == pytest.approx(-1)
    assert non_exposure["high_correlation"].item()
    assert np.isnan(non_exposure["activity_overlap"].item())
    assert not non_exposure["matching_activity"].item()
    assert non_exposure["joint_active_observations"].item() == -1
    assert non_exposure["exclusive_active_observations"].item() == -1


@pytest.mark.parametrize("nested", [False, True])
def test_check_data_keeps_group_order_and_removes_group_means_for_correlation(nested):
    rows = []
    for region, store, offset in (("west", 2, 100), ("east", 1, 0)):
        for period in (4, 1, 3, 2):
            rows.append(
                {
                    "week": period,
                    "region": region,
                    "store": store,
                    "first": offset + period,
                    "second": offset - period,
                    "fixed_a": offset,
                    "fixed_b": 2 * offset,
                }
            )
    data = prepare_data(
        pl.DataFrame(rows),
        time="week",
        groups=["region", "store"] if nested else ["region"],
        controls=["first", "second"],
        media=["fixed_a", "fixed_b"],
    )

    report = check_data(data)

    assert report["coverage"].attrs["n_groups"] == 2
    assert report["coverage"].attrs["n_periods"] == 4
    series = report["series"].to_dataset()
    assert series["mean"].dims == ("feature", "group")
    np.testing.assert_array_equal(series["group"], [0, 1] if nested else ["west", "east"])
    np.testing.assert_array_equal(series["mean"].sel(feature="controls.first"), [102.5, 2.5])
    np.testing.assert_array_equal(series["constant"].sel(feature="media.fixed_a"), [True, True])
    np.testing.assert_array_equal(series["zero_fraction"].sel(feature="media.fixed_a"), [0, 1])
    if nested:
        for node in (report["coverage"], report["series"]):
            np.testing.assert_array_equal(node["group_region"], ["west", "east"])
            np.testing.assert_array_equal(node["group_store"], [2, 1])
    varying = _pair(report, "controls.first", "controls.second")
    assert varying["correlation"].item() > 0.99
    assert varying["high_correlation"].item()
    assert varying["within_group_correlation"].item() == pytest.approx(-1)
    assert varying["high_within_group_correlation"].item()
    fixed = _pair(report, "media.fixed_a", "media.fixed_b")
    assert fixed["correlation"].item() == pytest.approx(1)
    assert np.isnan(fixed["within_group_correlation"].item())
    assert not fixed["high_within_group_correlation"].item()
    # Group differences alone do not supply temporal activity changes.
    assert not fixed["matching_activity"].item()


@pytest.mark.parametrize("grouped", [False, True])
def test_check_data_handles_single_period_without_undefined_correlation_flags(grouped):
    data = prepare_data(
        pl.DataFrame({"week": [1], "region": ["west"], "email": [0], "social": [2]}),
        time="week",
        groups=["region"] if grouped else (),
        organic_media=["email", "social"],
    )

    report = check_data(data)

    assert report["coverage"].attrs["frequency"] == "unknown"
    assert report["coverage"].attrs["n_periods"] == 1
    assert report["series"]["constant"].dims == (("feature", "group") if grouped else ("feature",))
    assert report["series"]["constant"].values.all()
    pair = _pair(report, "organic_media.email", "organic_media.social")
    assert np.isnan(pair["correlation"].item())
    assert not pair["high_correlation"].item()
    assert not pair["matching_activity"].item()


@pytest.mark.parametrize("grouped", [False, True])
def test_check_data_accepts_time_labels_without_selected_arrays(grouped):
    data = prepare_data(
        pl.DataFrame({"week": [1, 3], "region": ["west", "west"]}),
        time="week",
        groups=["region"] if grouped else (),
    )

    report = check_data(data)

    assert report["coverage"].attrs["n_periods"] == 2
    assert report["coverage"].attrs["n_groups"] == 1
    assert report["coverage"].attrs["n_history_periods"] == 0
    assert report["coverage"].attrs["frequency"] == "unknown"
    assert "media_time" not in report["coverage"].coords
    assert report["series"].sizes["feature"] == 0
    assert report["pairs"].sizes["pair"] == 0


def test_check_data_does_not_modify_inputs_or_retain_mutable_observations():
    data = prepare_data(
        pl.DataFrame({"week": [3, 1, 2], "impressions": [2, 0, 1], "cost": [2, 0, 1]}),
        time="week",
        media=["impressions"],
        spend=["cost"],
    )
    arrays_before = {name: values.copy() for name, values in data.arrays.items()}
    columns_before = data.columns.copy()
    labels_before = (data.time_values, data.media_time_values, data.channels, data.group_values)

    report = check_data(data)

    for name, values in arrays_before.items():
        np.testing.assert_array_equal(data.arrays[name], values)
    assert data.columns == columns_before
    assert (data.time_values, data.media_time_values, data.channels, data.group_values) == labels_before
    assert data._scaling is None
    snapshot = report.copy(deep=True)
    data.arrays["media"][...] = 999
    data.arrays["spend"][...] = 0
    xr.testing.assert_identical(report, snapshot)


def test_check_data_computes_large_integer_statistics_without_overflow():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 2, 3, 4],
                "impressions": [0, 2**62, 0, 2**62],
                "promotion": [False, True, False, True],
            }
        ),
        time="week",
        media=["impressions"],
        treatments=["promotion"],
    )

    report = check_data(data)

    media = report["series"].to_dataset().sel(feature="media.impressions")
    assert media["mean"].item() == 2**61
    assert media["std"].item() == 2**61
    pair = _pair(report, "media.impressions", "treatments.promotion")
    assert pair["correlation"].item() == pytest.approx(1)


@pytest.mark.parametrize("factory", [pd.DataFrame, pl.DataFrame, pa.table])
def test_check_data_has_equivalent_results_across_dataframe_backends(factory):
    columns = {
        "week": [3, 1, 2],
        "impressions": [3, 0, 2],
        "promotion": [True, False, True],
    }
    selection = {"time": "week", "media": ["impressions"], "treatments": ["promotion"]}
    expected = check_data(prepare_data(pd.DataFrame(columns), **selection))

    actual = check_data(prepare_data(factory(columns), **selection))

    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("option", ["max_zero_fraction", "correlation_threshold"])
@pytest.mark.parametrize("value", [0, -0.1, 1.01, np.inf, np.nan, True, "0.8"])
def test_check_data_rejects_invalid_thresholds(option, value):
    data = prepare_data(pl.DataFrame({"week": [1, 2]}), time="week")

    with pytest.raises((TypeError, ValueError), match=option):
        check_data(data, **{option: value})


def test_check_data_accepts_upper_threshold_boundaries():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2, 3], "unused": [0, 0, 0]}),
        time="week",
        media=["unused"],
    )

    report = check_data(data, max_zero_fraction=1, correlation_threshold=1)

    assert report["series"]["sparse"].sel(feature="media.unused").item()


@pytest.mark.parametrize("grouped", [False, True])
def test_check_data_flags_perfect_correlations_at_threshold_one(grouped):
    columns = {"week": [1, 2, 3], "video": [0, 1, 2], "social": [0, 2, 4]}
    if grouped:
        columns["region"] = ["north"] * 3
    data = prepare_data(
        pl.DataFrame(columns), time="week", media=["video", "social"], groups=["region"] if grouped else ()
    )

    pair = _pair(check_data(data, correlation_threshold=1), "media.video", "media.social")

    assert pair["correlation"].item() == 1
    assert pair["high_correlation"].item()
    if grouped:
        assert pair["within_group_correlation"].item() == 1
        assert pair["high_within_group_correlation"].item()


def test_check_data_rejects_scaled_prepared_data_even_when_transformations_are_disabled():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2, 3], "impressions": [1, 2, 3]}),
        time="week",
        media=["impressions"],
    )
    scaling = fit_data_scaling(data, media_method=None, scale_controls=False, scale_treatments=False)
    scaled = scaling.transform(data)

    with pytest.raises(ValueError, match=r"[Rr]aw|[Ss]cal"):
        check_data(scaled)


@pytest.mark.parametrize("value", [None, {}, pl.DataFrame({"week": [1]})])
def test_check_data_requires_prepared_data(value):
    with pytest.raises(TypeError, match="PreparedData"):
        check_data(value)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_check_data_rejects_nonfinite_edited_arrays(value):
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "impressions": [1.0, 2.0]}),
        time="week",
        media=["impressions"],
    )
    data.arrays["media"][0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        check_data(data)


def test_check_data_rejects_edited_array_shapes():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "impressions": [1.0, 2.0]}),
        time="week",
        media=["impressions"],
    )
    data.arrays["media"] = np.ones(2)

    with pytest.raises(ValueError, match="shape"):
        check_data(data)
