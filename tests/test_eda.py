"""Tests for exploratory analysis of labeled marketing data."""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

from mmmjax import Data, Model, Real, fit_data_scaling, normal_rng, prepare_data, sample_prior
from mmmjax.eda import _predictor_checks, check_data, check_prior


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
    assert set(report.children) == {"coverage", "series", "pairs", "spend", "predictors"}
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
    spending = report["spend"].to_dataset().sel(feature="spend.cost")
    np.testing.assert_array_equal(spending["time"], data.time_values)
    np.testing.assert_array_equal(spending["spend_without_exposure"], [True, False, False])
    np.testing.assert_array_equal(spending["exposure_without_spend"], [False, True, False])
    assert spending["cost_observations"].item() == 1


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
    assert report["spend"].sizes["feature"] == 0
    assert report["predictors"].sizes["feature"] == 0


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


def test_predictor_checks_detect_multivariate_dependence_beyond_pairwise_correlations():
    rows = np.arange(16)[:, None]
    independent = 2.0 * ((rows // (2 ** np.arange(4))) % 2) - 1
    values = np.column_stack((independent[:, :3], independent[:, :3].sum(axis=1), independent[:, 3]))
    features = np.array(["first", "second", "third", "combined", "unrelated"])
    correlations = np.corrcoef(values, rowvar=False)
    np.testing.assert_array_less(np.abs(correlations[np.triu_indices(5, k=1)]), 0.6)

    result = _predictor_checks(values[:, None], features, False)

    np.testing.assert_array_equal(result["feature"], features)
    assert np.isinf(result["vif"].sel(feature=["first", "second", "third", "combined"])).all()
    assert result["vif"].sel(feature="unrelated").item() == pytest.approx(1)
    assert set(result.data_vars) == {"vif"}


def test_predictor_checks_keep_unrelated_vif_finite_when_other_columns_are_duplicates():
    first = np.array([-1.0, -1.0, 1.0, 1.0])
    independent = np.array([-1.0, 1.0, -1.0, 1.0])
    values = np.column_stack((first, 2 * first + 5, independent + first, np.ones(4)))

    result = _predictor_checks(values[:, None], np.array(["first", "duplicate", "third", "constant"]), False)

    assert np.isinf(result["vif"].sel(feature=["first", "duplicate"])).all()
    assert result["vif"].sel(feature="third").item() == pytest.approx(2)
    assert np.isnan(result["vif"].sel(feature="constant").item())


@pytest.mark.parametrize("factors", [[1, 1, 1, 1], [1e-12, 1e12, -1e6, 1e-6]])
def test_predictor_checks_vif_matches_auxiliary_regressions_and_is_invariant_to_units(factors):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(12, 4))
    values[:, 1] += 2 * values[:, 0]
    expected = []
    for column in range(values.shape[1]):
        predictor = values[:, column]
        design = np.column_stack((np.ones(len(values)), np.delete(values, column, axis=1)))
        fitted = design @ np.linalg.lstsq(design, predictor, rcond=None)[0]
        expected.append(np.sum((predictor - predictor.mean()) ** 2) / np.sum((predictor - fitted) ** 2))

    result = _predictor_checks((values * factors)[:, None], np.array(["a", "b", "c", "d"]), False)

    np.testing.assert_allclose(result["vif"], expected, rtol=1e-12)


@pytest.mark.parametrize("n_observations", [1, 2, 3])
def test_predictor_checks_saturated_designs_are_not_given_finite_vifs(n_observations):
    rng = np.random.default_rng(14)
    values = rng.normal(size=(n_observations, 5))

    result = _predictor_checks(values[:, None], np.array(["a", "b", "c", "d", "e"]), False)

    if n_observations == 1:
        assert np.isnan(result["vif"]).all()
    else:
        assert np.isinf(result["vif"]).all()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("n_features", [0, 1])
def test_predictor_checks_support_empty_and_single_feature_inputs(grouped, n_features):
    n_groups = 2 if grouped else 1
    values = np.arange(3 * n_groups, dtype=float).reshape(3, n_groups, 1)[..., :n_features]

    result = _predictor_checks(values, np.array(["media"][:n_features], dtype=str), grouped)

    assert result.sizes["feature"] == n_features
    assert result["vif"].dims == ("feature",)
    if n_features:
        assert result["vif"].item() == pytest.approx(1)


def test_predictor_checks_distinguish_group_time_and_interaction_variation():
    time = np.array([-1.0, 0.0, 1.0])[:, None]
    group = np.array([-1.0, 1.0])[None, :]
    values = np.stack(
        (
            np.broadcast_to(time, (3, 2)),
            np.broadcast_to(group, (3, 2)),
            time + group,
            time * group,
            np.ones((3, 2)),
        ),
        axis=-1,
    )
    features = np.array(["time", "group", "additive", "interaction", "constant"])

    result = _predictor_checks(values, features, True)

    np.testing.assert_allclose(result["group_r_squared"], [0, 1, 0.6, 0, np.nan], atol=1e-15)
    np.testing.assert_allclose(result["time_r_squared"], [1, 0, 0.4, 0, np.nan], atol=1e-15)
    np.testing.assert_allclose(result["group_time_r_squared"], [1, 1, 1, 0, np.nan], atol=1e-15)
    assert result.attrs["n_observations"] == 6
    assert result.attrs["n_predictors"] == 5
    assert "Unadjusted" in result["group_time_r_squared"].attrs["description"]


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_predictor_checks_group_time_summaries_allow_one_period_or_one_group(shape):
    values = np.arange(3.0).reshape(*shape, 1)

    result = _predictor_checks(values, np.array(["feature"]), True)

    assert result["group_r_squared"].item() == pytest.approx(int(shape[0] == 1))
    assert result["time_r_squared"].item() == pytest.approx(int(shape[1] == 1))
    assert result["group_time_r_squared"].item() == pytest.approx(1)


def test_predictor_checks_group_time_summaries_match_indicator_regressions():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(5, 3, 2))
    values[:, :, 0] += np.arange(5)[:, None]
    values[:, :, 1] += 2 * np.arange(3)[None, :]
    flattened = values.reshape(15, 2)
    group = np.tile(np.eye(3), (5, 1))
    time = np.repeat(np.eye(5), 3, axis=0)

    result = _predictor_checks(values, np.array(["a", "b"]), True)

    for name, design in (
        ("group_r_squared", group),
        ("time_r_squared", time),
        ("group_time_r_squared", np.column_stack((group, time))),
    ):
        fitted = design @ np.linalg.lstsq(design, flattened, rcond=None)[0]
        expected = 1 - np.sum((flattened - fitted) ** 2, axis=0) / np.sum(
            (flattened - flattened.mean(axis=0)) ** 2, axis=0
        )
        np.testing.assert_allclose(result[name], expected, rtol=1e-12)


def test_check_data_predictor_checks_exclude_outcomes_spend_and_history():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [3, 4, 5, 6],
                "impressions": [0, 1, 0, 1],
                "sales": [0, 1, 0, 1],
                "cost": [0, 1, 0, 1],
                "price": [-1, 1, 1, -1],
            }
        ),
        time="week",
        outcome="sales",
        media=["impressions"],
        spend=["cost"],
        channels=["video"],
        controls=["price"],
        media_history=pl.DataFrame({"week": [1, 2], "impressions": [300, 400]}),
    )

    result = check_data(data)["predictors"].to_dataset()

    np.testing.assert_array_equal(result["feature"], ["media.impressions", "controls.price"])
    np.testing.assert_array_equal(result["role"], ["media", "controls"])
    np.testing.assert_array_equal(result["column"], ["impressions", "price"])
    np.testing.assert_array_equal(result["channel"], ["video", ""])
    np.testing.assert_allclose(result["vif"], [1, 1])
    assert result.attrs["n_observations"] == 4
    assert result.attrs["n_predictors"] == 2


def test_check_data_reports_dated_spend_mismatches_and_cost_outliers():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": np.arange(1, 10),
                "video": [0, 0, 10, 10, 10, 10, 10, 10, 10],
                "cost": [0, 5, 0, 10, 10, 10, 10, 10, 100],
            }
        ),
        time="week",
        media=["video"],
        spend=["cost"],
        channels=["Online video"],
    )

    spend = check_data(data)["spend"].to_dataset().sel(feature="spend.cost")

    assert spend["cost_per_exposure"].dims == ("time",)
    assert spend["channel"].item() == "Online video"
    assert spend["column"].item() == "cost"
    assert spend["exposure_column"].item() == "video"
    assert spend["frequency_column"].item() == ""
    np.testing.assert_array_equal(spend["spend_without_exposure"], [False, True, *([False] * 7)])
    np.testing.assert_array_equal(spend["exposure_without_spend"], [False, False, True, *([False] * 6)])
    np.testing.assert_allclose(spend["cost_per_exposure"], [np.nan, np.nan, 0, 1, 1, 1, 1, 1, 10], equal_nan=True)
    np.testing.assert_array_equal(spend["cost_outlier"], [False, False, True, *([False] * 5), True])
    assert spend["cost_observations"].item() == 7
    assert spend["cost_lower_fence"].item() == 1
    assert spend["cost_upper_fence"].item() == 1


@pytest.mark.parametrize("nested", [False, True])
def test_check_data_pairs_rf_spend_and_uses_group_local_cost_fences(nested):
    rows = []
    for region, store, unit_cost in (("west", 2, 10.0), ("east", 1, 1.0)):
        for period in range(1, 9):
            frequency = 0 if period == 1 else 2
            rows.append(
                {
                    "week": period,
                    "region": region,
                    "store": store,
                    "views": 10,
                    "audience": 5,
                    "frequency": frequency,
                    "cost": 10 * unit_cost,
                }
            )
    data = prepare_data(
        pl.DataFrame(rows),
        time="week",
        groups=["region", "store"] if nested else ["region"],
        media=["views"],
        spend=["cost"],
        channels=["Shared"],
        reach=["audience"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
        rf_channels=["Shared"],
    )

    spend = check_data(data)["spend"].to_dataset()

    np.testing.assert_array_equal(spend["feature"], ["spend.cost", "rf_spend.cost"])
    np.testing.assert_array_equal(spend["group"], [0, 1] if nested else ["west", "east"])
    assert spend["cost_per_exposure"].dims == ("time", "group", "feature")
    assert spend["cost_lower_fence"].dims == ("feature", "group")
    assert not spend["cost_outlier"].values.any()
    np.testing.assert_allclose(spend["cost_lower_fence"], [[10, 1], [10, 1]])
    rf = spend.sel(feature="rf_spend.cost")
    assert rf["exposure_column"].item() == "audience"
    assert rf["frequency_column"].item() == "frequency"
    assert rf["spend_without_exposure"].isel(time=0).values.all()
    assert not rf["spend_without_exposure"].isel(time=slice(1, None)).values.any()
    np.testing.assert_allclose(rf["cost_per_exposure"].isel(time=1), [10, 1])
    np.testing.assert_array_equal(rf["cost_observations"], [7, 7])
    if nested:
        np.testing.assert_array_equal(spend["group_region"], ["west", "east"])
        np.testing.assert_array_equal(spend["group_store"], [2, 1])


def test_check_data_keeps_undefined_costs_unflagged_and_handles_one_observation():
    data = prepare_data(
        pl.DataFrame(
            {
                "week": [1, 2, 3],
                "inactive": [0, 0, 0],
                "active": [0, 10, 0],
                "free": [0, 0, 0],
                "cost": [1, 2, 3],
            }
        ),
        time="week",
        media=["inactive", "active"],
        spend=["free", "cost"],
    )

    spend = check_data(data)["spend"].to_dataset()

    inactive = spend.sel(feature="spend.free")
    assert inactive["cost_per_exposure"].isnull().all()
    assert inactive["cost_lower_fence"].isnull()
    assert inactive["cost_upper_fence"].isnull()
    assert not inactive["cost_outlier"].any()
    assert not inactive["spend_without_exposure"].any()
    assert not inactive["exposure_without_spend"].any()
    assert inactive["cost_observations"].item() == 0
    active = spend.sel(feature="spend.cost")
    assert active["cost_lower_fence"].item() == pytest.approx(0.2)
    assert active["cost_upper_fence"].item() == pytest.approx(0.2)
    assert not active["cost_outlier"].any()


def test_check_data_keeps_costs_on_iqr_fences_unflagged():
    data = prepare_data(
        pl.DataFrame({"week": np.arange(1, 10), "media": np.ones(9), "cost": [0, 3, 3, 4, 4, 5, 5, 5, 8]}),
        time="week",
        media=["media"],
        spend=["cost"],
    )

    spend = check_data(data)["spend"].to_dataset()

    assert spend["cost_lower_fence"].item() == 0
    assert spend["cost_upper_fence"].item() == 8
    assert not spend["cost_outlier"].any()


def test_check_data_reports_hand_calculated_robust_variability_for_signed_controls():
    data = prepare_data(
        pl.DataFrame({"week": np.arange(7), "temperature": [-100, -2, -1, 0, 1, 2, 100]}),
        time="week",
        controls=["temperature"],
    )

    series = check_data(data)["series"].to_dataset().sel(feature="controls.temperature")

    assert series["lower_quartile"].item() == pytest.approx(-1.5)
    assert series["upper_quartile"].item() == pytest.approx(1.5)
    assert series["interquartile_range"].item() == pytest.approx(3)
    assert series["outlier_periods"].item() == 2
    assert series["std_without_outliers"].item() == pytest.approx(np.sqrt(2))
    assert not series["outlier_driven_variation"].item()


@pytest.mark.parametrize(
    ("observations", "outliers", "driven"),
    [
        ([0, 0, 0, 0, 10], 1, True),
        ([1, 1, 1, 1, 10], 1, True),
        ([0, 0, 0, 0, 0], 0, False),
        ([7, 7, 7, 7, 7], 0, False),
    ],
)
def test_check_data_distinguishes_outlier_driven_variability_from_constant_series(observations, outliers, driven):
    data = prepare_data(
        pl.DataFrame({"week": np.arange(len(observations)), "impressions": observations}),
        time="week",
        media=["impressions"],
    )

    series = check_data(data)["series"].to_dataset().sel(feature="media.impressions")

    assert series["interquartile_range"].item() == 0
    assert series["outlier_periods"].item() == outliers
    assert series["std_without_outliers"].item() == 0
    assert series["outlier_driven_variation"].item() is driven


def test_check_data_retains_observations_on_iqr_fences():
    observations = np.array([1, 4, 4, 5, 6, 6, 9])
    data = prepare_data(
        pl.DataFrame({"week": np.arange(len(observations)), "impressions": observations}),
        time="week",
        media=["impressions"],
    )

    series = check_data(data)["series"].to_dataset().sel(feature="media.impressions")

    assert series["lower_quartile"].item() == pytest.approx(4)
    assert series["upper_quartile"].item() == pytest.approx(6)
    assert series["outlier_periods"].item() == 0
    assert series["std_without_outliers"].item() == pytest.approx(observations.std())
    assert not series["outlier_driven_variation"].item()


def test_check_data_keeps_fence_boundary_when_rescaling_would_introduce_roundoff():
    observations = np.array([18, 16, -4, 8, -19, 11, 16])
    data = prepare_data(
        pl.DataFrame({"week": np.arange(len(observations)), "temperature": observations}),
        time="week",
        controls=["temperature"],
    )

    series = check_data(data)["series"].to_dataset().sel(feature="controls.temperature")

    assert series["lower_quartile"].item() == 2
    assert series["upper_quartile"].item() == 16
    assert series["outlier_periods"].item() == 0
    assert series["std_without_outliers"].item() == pytest.approx(observations.std())


@pytest.mark.parametrize("observations", [[2], [1, 9]])
def test_check_data_handles_short_series_for_robust_variability(observations):
    data = prepare_data(
        pl.DataFrame({"week": np.arange(len(observations)), "impressions": observations}),
        time="week",
        media=["impressions"],
    )

    series = check_data(data)["series"].to_dataset().sel(feature="media.impressions")

    assert series["outlier_periods"].item() == 0
    assert series["std_without_outliers"].item() == pytest.approx(np.std(observations))
    assert not series["outlier_driven_variation"].item()


def test_check_data_computes_robust_variability_separately_by_group_and_excludes_history():
    frame = pl.concat(
        [
            pl.DataFrame({"week": [1, 2, 3, 4, 5], "geo": [geo] * 5, "impressions": observations})
            for geo, observations in (("west", [1, 1, 1, 1, 10]), ("east", [100, 200, 300, 400, 500]))
        ]
    )
    data = prepare_data(
        frame,
        time="week",
        groups=["geo"],
        media=["impressions"],
        media_history=pl.DataFrame({"week": [0, 0], "geo": ["west", "east"], "impressions": [9999, 9999]}),
    )

    series = check_data(data)["series"].to_dataset().sel(feature="media.impressions")

    assert series["outlier_periods"].dims == ("group",)
    np.testing.assert_array_equal(series["group"], ["west", "east"])
    np.testing.assert_array_equal(series["outlier_periods"], [1, 0])
    np.testing.assert_allclose(series["interquartile_range"], [0, 200])
    np.testing.assert_allclose(series["std_without_outliers"], [0, np.sqrt(20000)])
    np.testing.assert_array_equal(series["outlier_driven_variation"], [True, False])


def test_check_prior_is_exported_from_the_package():
    import mmmjax
    import mmmjax.eda

    assert mmmjax.check_prior is check_prior
    assert "check_prior" in mmmjax.__all__
    assert "check_prior" in mmmjax.eda.__all__


def test_check_prior_reports_hand_calculated_probabilities_and_inclusive_bounds():
    draws = xr.DataArray(
        [[-1, 0], [0, 1], [2, 4], [4, 2]],
        dims=("draw", "channel"),
        coords={"channel": ["video", "search"]},
        name="roi",
    )

    report = check_prior(draws, lower=0, upper=2)

    assert isinstance(report, xr.Dataset)
    assert set(report.data_vars) == {
        "finite_draws",
        "nonfinite_fraction",
        "probability_below",
        "probability_above",
        "probability_outside",
        "lower",
        "upper",
    }
    assert report.attrs["sample_count"] == 4
    assert report.attrs["quantity"] == "roi"
    assert report.attrs["probability_scope"] == "finite draws only"
    np.testing.assert_array_equal(report["channel"], ["video", "search"])
    np.testing.assert_array_equal(report["finite_draws"], [4, 4])
    np.testing.assert_array_equal(report["nonfinite_fraction"], [0, 0])
    np.testing.assert_allclose(report["probability_below"], [0.25, 0])
    np.testing.assert_allclose(report["probability_above"], [0.25, 0.25])
    np.testing.assert_allclose(report["probability_outside"], [0.5, 0.25])
    np.testing.assert_array_equal(report["lower"], [0, 0])
    np.testing.assert_array_equal(report["upper"], [2, 2])


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_check_prior_supports_one_sided_scalar_and_boolean_quantities(bound):
    draws = xr.DataArray([False, True, True, False], dims="draw")

    report = check_prior(draws, **{bound: 0.5})

    assert report.attrs["quantity"] == "unnamed"
    assert report.sizes == {}
    assert report["probability_outside"].item() == 0.5
    assert report["finite_draws"].item() == 4
    assert report[bound].item() == 0.5
    absent = ("upper", "probability_above") if bound == "lower" else ("lower", "probability_below")
    assert not set(absent) & set(report.data_vars)


def test_check_prior_uses_both_sampling_axes_regardless_of_dimension_order():
    values = np.arange(24).reshape(2, 4, 3)
    draws = xr.DataArray(
        values,
        dims=("chain", "draw", "channel"),
        coords={"chain": [7, 9], "draw": [10, 20, 30, 40], "channel": ["search", "video", "tv"]},
        name="contribution",
    )

    report = check_prior(draws.transpose("channel", "draw", "chain"), lower=5, upper=17)
    expected = check_prior(draws, lower=5, upper=17)

    xr.testing.assert_identical(report, expected)
    assert report.attrs["sample_count"] == 8
    np.testing.assert_allclose(report["probability_below"], (values < 5).mean(axis=(0, 1)))
    np.testing.assert_allclose(report["probability_above"], (values > 17).mean(axis=(0, 1)))
    assert "draw" not in report.coords
    assert "chain" not in report.coords


def test_check_prior_preserves_pointwise_axes_and_leaves_aggregation_to_the_user():
    draws = xr.DataArray([[-2, 2], [2, -2]], dims=("draw", "time"), coords={"time": [1, 2]})

    pointwise = check_prior(draws, lower=0)
    total = check_prior(draws.sum("time"), lower=0)

    np.testing.assert_array_equal(pointwise["probability_below"], [0.5, 0.5])
    assert total["probability_below"].item() == 0


def test_check_prior_excludes_nonfinite_draws_from_probabilities_but_reports_their_fraction():
    draws = xr.DataArray(
        [[np.nan, np.nan, -1], [1, np.inf, np.inf], [3, -np.inf, 2], [np.inf, np.nan, 4]],
        dims=("draw", "channel"),
        coords={"channel": ["search", "video", "tv"]},
    )

    report = check_prior(draws, lower=0, upper=2)

    np.testing.assert_array_equal(report["finite_draws"], [2, 0, 3])
    np.testing.assert_allclose(report["nonfinite_fraction"], [0.5, 1, 0.25])
    np.testing.assert_allclose(report["probability_below"], [0, np.nan, 1 / 3], equal_nan=True)
    np.testing.assert_allclose(report["probability_above"], [0.5, np.nan, 1 / 3], equal_nan=True)
    np.testing.assert_allclose(report["probability_outside"], [0.5, np.nan, 2 / 3], equal_nan=True)


def test_check_prior_aligns_labeled_bounds_and_preserves_auxiliary_coordinates_without_mutation():
    draws = xr.DataArray(
        np.arange(24).reshape(4, 2, 3),
        dims=("draw", "geo", "channel"),
        coords={
            "geo": ["east", "west"],
            "channel": ["search", "video", "tv"],
            "region": ("geo", ["coastal", "inland"]),
            "family": ("channel", ["digital", "digital", "offline"]),
            "currency": "USD",
        },
        name="revenue",
    )
    lower = xr.DataArray([3, 1, 2], dims="channel", coords={"channel": ["tv", "search", "video"]})
    upper = xr.DataArray([20, 15], dims="geo", coords={"geo": ["west", "east"]})
    original, original_lower, original_upper = draws.copy(deep=True), lower.copy(deep=True), upper.copy(deep=True)

    report = check_prior(draws, lower=lower, upper=upper)

    assert report["probability_outside"].dims == ("geo", "channel")
    for name in draws.coords:
        xr.testing.assert_identical(report.coords[name], draws.coords[name])
    np.testing.assert_array_equal(report["lower"], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(report["upper"], [[15, 15, 15], [20, 20, 20]])
    np.testing.assert_allclose(report["probability_below"], (draws.values < np.array([1, 2, 3])).mean(axis=0))
    np.testing.assert_allclose(report["probability_above"], (draws.values > np.array([[15], [20]])).mean(axis=0))
    xr.testing.assert_identical(draws, original)
    xr.testing.assert_identical(lower, original_lower)
    xr.testing.assert_identical(upper, original_upper)


def test_check_prior_accepts_equal_bounds_and_scalar_dataarray_bounds():
    draws = xr.DataArray([-1, 0, 1], dims="draw")

    report = check_prior(draws, lower=xr.DataArray(0.0), upper=np.float64(0.0))

    assert report["probability_outside"].item() == pytest.approx(2 / 3)
    assert report["probability_below"].item() == pytest.approx(1 / 3)
    assert report["probability_above"].item() == pytest.approx(1 / 3)


@pytest.mark.parametrize("draws", [[1, 2], np.array([1, 2]), xr.Dataset(), xr.DataTree(), 1.0])
def test_check_prior_requires_a_dataarray(draws):
    with pytest.raises(TypeError):
        check_prior(draws, lower=0)


@pytest.mark.parametrize("dims,shape", [((), ()), (("chain",), (2,)), (("time", "channel"), (2, 3))])
def test_check_prior_requires_a_draw_axis(dims, shape):
    draws = xr.DataArray(np.ones(shape), dims=dims)

    with pytest.raises(ValueError):
        check_prior(draws, lower=0)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_check_prior_rejects_empty_sampling_axes(shape):
    draws = xr.DataArray(np.empty(shape), dims=("chain", "draw"))

    with pytest.raises(ValueError):
        check_prior(draws, lower=0)


@pytest.mark.parametrize(
    "values", [["1", "2"], [1 + 2j], np.array([1], dtype=object), np.array(["2026-01-01"], dtype="datetime64[D]")]
)
def test_check_prior_requires_real_numeric_draws(values):
    with pytest.raises(TypeError):
        check_prior(xr.DataArray(values, dims="draw"), lower=0)


def test_check_prior_requires_at_least_one_bound():
    with pytest.raises(ValueError):
        check_prior(xr.DataArray([1, 2], dims="draw"))


@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize("value", [True, "1", 1 + 2j, [0, 1], np.array([0, 1]), np.array(0)])
def test_check_prior_rejects_invalid_or_unlabeled_bounds(bound, value):
    with pytest.raises(TypeError):
        check_prior(xr.DataArray([0, 1], dims="draw"), **{bound: value})


@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_check_prior_rejects_nonfinite_scalar_bounds(bound, value):
    with pytest.raises(ValueError):
        check_prior(xr.DataArray([0, 1], dims="draw"), **{bound: value})


@pytest.mark.parametrize("values", [[True, False], ["0", "1"], [0j, 1j], np.array([0, 1], dtype=object)])
def test_check_prior_rejects_nonnumeric_or_boolean_labeled_bounds(values):
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"), coords={"channel": ["video", "search"]})
    lower = xr.DataArray(values, dims="channel", coords={"channel": ["video", "search"]})

    with pytest.raises(TypeError):
        check_prior(draws, lower=lower)


@pytest.mark.parametrize("values", [[0, np.nan], [0, np.inf], [-np.inf, 1]])
def test_check_prior_rejects_nonfinite_labeled_bounds(values):
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"), coords={"channel": ["video", "search"]})
    upper = xr.DataArray(values, dims="channel", coords={"channel": ["video", "search"]})

    with pytest.raises(ValueError):
        check_prior(draws, upper=upper)


@pytest.mark.parametrize("labels", [["video"], ["video", "radio"], ["video", "search", "tv"], ["video", "video"]])
def test_check_prior_rejects_missing_extra_or_duplicate_bound_labels(labels):
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"), coords={"channel": ["video", "search"]})
    lower = xr.DataArray(np.zeros(len(labels)), dims="channel", coords={"channel": labels})

    with pytest.raises(ValueError):
        check_prior(draws, lower=lower)


@pytest.mark.parametrize("dim", ["draw", "chain", "unrelated"])
def test_check_prior_rejects_bounds_over_sampling_or_unrelated_axes(dim):
    draws = xr.DataArray(np.zeros((2, 2, 2)), dims=("chain", "draw", "channel"))
    lower = xr.DataArray([0, 0], dims=dim, coords={dim: [0, 1]})

    with pytest.raises(ValueError):
        check_prior(draws, lower=lower)


@pytest.mark.parametrize("labeled_draws", [False, True])
def test_check_prior_rejects_bounds_without_axis_labels(labeled_draws):
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"))
    if labeled_draws:
        draws = draws.assign_coords(channel=["video", "search"])
    lower = xr.DataArray([0, 0], dims="channel")

    with pytest.raises(ValueError):
        check_prior(draws, lower=lower)


def test_check_prior_rejects_ambiguous_duplicate_labels_on_checked_draws():
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"), coords={"channel": ["video", "video"]})
    lower = xr.DataArray([0, 1], dims="channel", coords={"channel": ["video", "search"]})

    with pytest.raises(ValueError):
        check_prior(draws, lower=lower)


def test_check_prior_rejects_crossed_bounds_after_label_alignment():
    draws = xr.DataArray([[0, 1]], dims=("draw", "channel"), coords={"channel": ["video", "search"]})
    lower = xr.DataArray([3, 1], dims="channel", coords={"channel": ["search", "video"]})
    upper = xr.DataArray([2, 2], dims="channel", coords={"channel": ["video", "search"]})

    with pytest.raises(ValueError):
        check_prior(draws, lower=lower, upper=upper)


def test_check_prior_checks_original_unit_quantities_saved_from_actual_prior_draws():
    data = prepare_data(pl.DataFrame({"week": [1, 2, 3], "sales": [100.0, 110.0, 120.0]}), time="week", outcome="sales")
    scaling = fit_data_scaling(data, scale_outcome=True)

    def density(location):
        raise AssertionError("Prior checks must not evaluate the likelihood")

    def quantities(location):
        return {"expected_sales": scaling.transformations["outcome"].inverse_transform(location[None])[0]}

    def prior(key):
        return {"location": normal_rng(key, location=0, scale=1)}

    model = Model(
        parameters={"location": Real()},
        log_density=density,
        data=Data(data, scaling=scaling),
        transformed_parameters=quantities,
        prior=prior,
        save=("expected_sales",),
    )
    results = sample_prior(model, draws=8, seed=4)
    draws = results["prior_generated_quantities"]["expected_sales"]

    report = check_prior(draws, lower=105, upper=115)

    assert report.attrs["sample_count"] == 8
    assert report.attrs["quantity"] == "expected_sales"
    assert report["finite_draws"].item() == 8
    assert report["probability_below"].item() == pytest.approx((draws.values < 105).mean())
    assert report["probability_above"].item() == pytest.approx((draws.values > 115).mean())
    expected = 110 + np.std([100, 110, 120]) * results["prior"]["location"].values
    np.testing.assert_allclose(draws, expected, rtol=2e-6)


@pytest.mark.parametrize("reverse_limits", [False, True])
def test_check_prior_aligns_stacked_location_bounds_and_preserves_index_levels(reverse_limits):
    draws = xr.DataArray(
        np.arange(24).reshape(4, 3, 2),
        dims=("draw", "time", "geo"),
        coords={"time": [1, 2, 3], "geo": ["east", "west"]},
        name="revenue",
    ).stack(location=("time", "geo"))
    lower = draws.isel(draw=0, drop=True) + 6
    upper = lower + 6
    if reverse_limits:
        lower = lower.isel(location=slice(None, None, -1))
        upper = upper.isel(location=slice(None, None, -1))
    original_lower = lower.copy(deep=True)

    report = check_prior(draws, lower=lower, upper=upper)

    assert report["probability_outside"].dims == ("location",)
    assert report.indexes["location"].equals(draws.indexes["location"])
    for name in ("location", "time", "geo"):
        xr.testing.assert_identical(report.coords[name], draws.coords[name])
    np.testing.assert_array_equal(report["lower"], np.arange(6, 12))
    np.testing.assert_array_equal(report["upper"], np.arange(12, 18))
    np.testing.assert_allclose(report["probability_below"], 0.25)
    np.testing.assert_allclose(report["probability_above"], 0.25)
    np.testing.assert_allclose(report["probability_outside"], 0.5)
    xr.testing.assert_identical(lower, original_lower)


@pytest.mark.parametrize("reserved", ["probability_outside", "lower"])
@pytest.mark.parametrize("coordinate_only", [False, True])
def test_check_prior_rejects_output_names_used_as_retained_coordinates_or_dimensions(reserved, coordinate_only):
    dim = "channel" if coordinate_only else reserved
    draws = xr.DataArray([[0, 1], [1, 2]], dims=("draw", dim))
    draws = draws.assign_coords({reserved: (dim, ["video", "search"])})

    with pytest.raises(ValueError, match="reserved"):
        check_prior(draws, lower=0, upper=2)
