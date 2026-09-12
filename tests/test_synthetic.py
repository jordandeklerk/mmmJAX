"""Tests for synthetic marketing data and its independently labeled truth."""

import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

import mmmjax
from mmmjax import SyntheticData, geometric_adstock, hill_saturation, prepare_data, simulate_data

CHANNELS = (
    "meta",
    "tiktok",
    "snapchat",
    "youtube",
    "streaming",
    "linear_tv",
    "branded_search",
    "generic_search",
    "influencer",
    "display",
    "email",
)
PAID_CHANNELS = CHANNELS[:-1]
EXPOSURE_COLUMNS = [f"{channel}_impressions" for channel in PAID_CHANNELS] + ["email_sends"]
SPEND_COLUMNS = [f"{channel}_spend" for channel in PAID_CHANNELS]
CONTROLS = ["demand", "price", "promotion", "holiday"]
COMPONENTS = (
    "baseline",
    "seasonality",
    "demand_effect",
    "price_effect",
    "promotion_effect",
    "holiday_effect",
)


@pytest.fixture(scope="module")
def grouped_data():
    return simulate_data(seed=37, n_periods=104, groups=("west", "east"))


def _numpy_contribution(truth, exposure):
    series = exposure.reshape(exposure.shape[0], -1, len(CHANNELS))
    series = series / np.asarray(truth.population.values).reshape(1, -1, 1)
    carried = np.empty_like(series)
    for channel, retention in enumerate(truth.retention.values):
        weights = retention ** np.arange(9)
        weights /= weights.sum()
        for group in range(series.shape[1]):
            carried[:, group, channel] = np.convolve(series[:, group, channel], weights)[: series.shape[0]]
    power = carried[8:] ** truth.slope.values
    response = power / (power + truth.half_saturation.values**truth.slope.values)
    coefficient = truth.coefficient.values.reshape(1, -1, len(CHANNELS))
    return (response * coefficient).reshape(truth.contribution.shape)


def test_synthetic_api_is_exported_and_result_is_frozen(grouped_data):
    assert {"SyntheticData", "simulate_data"}.issubset(mmmjax.__all__)
    assert isinstance(grouped_data, SyntheticData)
    assert isinstance(grouped_data.frame, pd.DataFrame)
    assert isinstance(grouped_data.media_history, pd.DataFrame)
    assert isinstance(grouped_data.channels, pd.DataFrame)
    assert isinstance(grouped_data.truth, xr.Dataset)
    with pytest.raises(FrozenInstanceError):
        grouped_data.frame = pd.DataFrame()


def test_generator_works_without_polars():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] == "polars":
                    raise ModuleNotFoundError(f"No module named {fullname!r}")

        sys.meta_path.insert(0, MissingOptional())
        import mmmjax

        assert "polars" not in sys.modules
        result = mmmjax.simulate_data(n_periods=1)
        import pandas as pd
        assert isinstance(result.frame, pd.DataFrame)
        assert isinstance(result.media_history, pd.DataFrame)
        assert isinstance(result.channels, pd.DataFrame)
        assert "polars" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_simulation_has_three_years_and_three_regions():
    result = simulate_data()
    assert isinstance(result.frame, pd.DataFrame)
    assert isinstance(result.media_history, pd.DataFrame)
    assert isinstance(result.channels, pd.DataFrame)
    assert len(result.frame) == 156 * 3
    assert len(result.media_history) == 8 * 3
    assert result.truth.group.values.tolist() == ["north", "south", "west"]
    assert np.datetime64(result.frame["week"][0], "D") == np.datetime64("2022-01-03")


@pytest.mark.parametrize("groups", [None, ("west", "east"), ["one"]])
@pytest.mark.parametrize("start", ["2024-02-26", date(2024, 2, 26)])
def test_simulation_preserves_weekly_dates_group_order_and_labeled_axes(groups, start):
    result = simulate_data(seed=4, n_periods=5, groups=groups, start=start)
    truth = result.truth
    n_groups = 1 if groups is None else len(groups)
    dates = [date(2024, 2, 26) + timedelta(weeks=index) for index in range(5)]
    history_dates = [dates[0] + timedelta(weeks=index) for index in range(-8, 0)]
    assert len(result.frame) == 5 * n_groups
    assert len(result.media_history) == 8 * n_groups
    assert all(type(value) is date for value in result.frame["week"])
    assert result.frame["week"].to_list() == [week for week in dates for _ in range(n_groups)]
    assert result.media_history["week"].to_list() == [week for week in history_dates for _ in range(n_groups)]
    np.testing.assert_array_equal(truth.time.values, np.asarray(dates, dtype="datetime64[ns]"))
    np.testing.assert_array_equal(truth.media_time.values, np.asarray(history_dates + dates, dtype="datetime64[ns]"))
    assert truth.channel.values.tolist() == list(CHANNELS)
    assert truth.paid_channel.values.tolist() == list(PAID_CHANNELS)

    observation_dims = ("time",) if groups is None else ("time", "group")
    media_dims = ("media_time",) if groups is None else ("media_time", "group")
    coefficient_dims = () if groups is None else ("group",)
    for name in (*COMPONENTS, "expected_revenue", "noise", "revenue"):
        assert truth[name].dims == observation_dims
    for name in ("exposure", "observed_exposure"):
        assert truth[name].dims == (*media_dims, "channel")
    for name in ("spend", "cpm"):
        assert truth[name].dims == (*media_dims, "paid_channel")
    assert truth.contribution.dims == (*observation_dims, "channel")
    assert truth.coefficient.dims == (*coefficient_dims, "channel")
    assert truth.roi.dims == (*coefficient_dims, "paid_channel")
    for name in ("retention", "half_saturation", "slope"):
        assert truth[name].dims == ("channel",)

    if groups is None:
        assert "region" not in result.frame.columns
        assert "region" not in result.media_history.columns
        assert "group" not in truth.dims
    else:
        assert result.frame["region"].to_list() == list(groups) * 5
        assert result.media_history["region"].to_list() == list(groups) * 8
        assert truth.group.values.tolist() == list(groups)


@pytest.mark.parametrize("start", ["1500-01-04", "2500-01-04"])
def test_truth_dates_do_not_wrap_outside_nanosecond_datetime_range(start):
    result = simulate_data(n_periods=2, groups=None, start=start)
    dates = np.asarray(result.frame["week"].to_list(), dtype="datetime64[D]")
    history = np.asarray(result.media_history["week"].to_list(), dtype="datetime64[D]")
    np.testing.assert_array_equal(result.truth.time.values.astype("datetime64[D]"), dates)
    np.testing.assert_array_equal(
        result.truth.media_time.values.astype("datetime64[D]"), np.concatenate([history, dates])
    )


def test_channel_catalog_matches_observable_columns_without_truth_leakage(grouped_data):
    result = grouped_data
    catalog = result.channels
    assert catalog["channel"].to_list() == list(CHANNELS)
    assert {
        "family",
        "platform",
        "tactic",
        "kind",
        "exposure_column",
        "spend_column",
        "unit",
        "activity",
    }.issubset(catalog.columns)
    paid = catalog.loc[catalog["kind"] == "paid"]
    organic = catalog.loc[catalog["kind"] == "organic"]
    assert paid["channel"].to_list() == list(PAID_CHANNELS)
    assert paid["exposure_column"].to_list() == EXPOSURE_COLUMNS[:-1]
    assert paid["spend_column"].to_list() == SPEND_COLUMNS
    assert organic["channel"].to_list() == ["email"]
    assert organic["exposure_column"].to_list() == ["email_sends"]
    assert organic["spend_column"].isna().all()
    assert set(result.frame.columns) == {
        "week",
        "region",
        "revenue",
        "population",
        *CONTROLS,
        *EXPOSURE_COLUMNS,
        *SPEND_COLUMNS,
    }
    assert set(EXPOSURE_COLUMNS).issubset(result.media_history.columns)
    assert "revenue" not in result.media_history.columns
    for column in ["population", "revenue", *CONTROLS, *EXPOSURE_COLUMNS, *SPEND_COLUMNS]:
        assert np.isfinite(result.frame[column].to_numpy()).all()
    assert (result.frame[EXPOSURE_COLUMNS + SPEND_COLUMNS].to_numpy() >= 0).all()
    assert (result.frame["population"].to_numpy() > 0).all()
    assert (result.frame.groupby("region")["population"].nunique() == 1).all()


@pytest.mark.parametrize("groups", [None, ("west", "east")])
def test_revenue_components_and_paid_media_costs_reconcile(groups):
    result = simulate_data(seed=11, n_periods=30, groups=groups)
    truth = result.truth
    expected = sum(truth[name] for name in COMPONENTS) + truth.contribution.sum("channel")
    xr.testing.assert_allclose(truth.expected_revenue, expected)
    xr.testing.assert_allclose(truth.revenue, truth.expected_revenue + truth.noise)
    np.testing.assert_array_equal(result.frame["revenue"].to_numpy(), truth.revenue.values.reshape(-1))

    paid_exposure = truth.exposure.sel(channel=list(PAID_CHANNELS)).values
    np.testing.assert_allclose(truth.spend.values, paid_exposure / 1_000 * truth.cpm.values, rtol=1e-12)
    assert (truth.cpm.values > 0).all()
    assert np.isfinite(truth.cpm.values).all()
    np.testing.assert_array_equal(
        result.frame[SPEND_COLUMNS].to_numpy(), truth.spend.values[8:].reshape(-1, len(PAID_CHANNELS))
    )
    np.testing.assert_array_equal(
        result.frame[EXPOSURE_COLUMNS].to_numpy(), truth.observed_exposure.values[8:].reshape(-1, len(CHANNELS))
    )
    np.testing.assert_array_equal(
        result.media_history[EXPOSURE_COLUMNS].to_numpy(),
        truth.observed_exposure.values[:8].reshape(-1, len(CHANNELS)),
    )


@pytest.mark.parametrize("groups", [None, ("west", "east")])
def test_baseline_is_positive_smooth_and_nonperiodic(groups):
    truth = simulate_data(seed=41, n_periods=130, groups=groups).truth
    baseline = np.asarray(truth.baseline).reshape(len(truth.time), -1)

    assert np.all(baseline > 0)
    assert np.all(np.ptp(baseline, axis=0) > 0)

    log_baseline = np.log(baseline)
    weekly_changes = np.abs(np.diff(log_baseline, axis=0))
    quarterly_changes = np.abs(log_baseline[13:] - log_baseline[:-13])
    assert np.all(np.median(weekly_changes, axis=0) < np.median(quarterly_changes, axis=0))
    assert not np.allclose(baseline[52:], baseline[:-52])

    if groups is not None:
        relative_paths = baseline / baseline[:1]
        assert not np.allclose(relative_paths[:, 0], relative_paths[:, 1])


@pytest.mark.parametrize("groups", [None, ("west", "east")])
def test_contributions_match_independent_convolution_and_public_transformations(groups):
    truth = simulate_data(seed=37, n_periods=30, groups=groups).truth
    independent = _numpy_contribution(truth, truth.exposure.values)
    np.testing.assert_allclose(truth.contribution.values, independent, rtol=3e-6, atol=1e-7)

    population = truth.population.values if groups is None else truth.population.values[None, :, None]
    carried = geometric_adstock(truth.exposure.values / population, truth.retention.values, max_lag=8, normalize=True)
    response = hill_saturation(carried[8:], truth.half_saturation.values, truth.slope.values)
    np.testing.assert_allclose(truth.response.values, response, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(truth.contribution.values, np.asarray(response) * truth.coefficient.values, rtol=3e-6)

    without_history = truth.exposure.values.copy()
    without_history[:8] = 0
    cold_contribution = _numpy_contribution(truth, without_history)
    assert (truth.contribution.values[0, ..., 0] > cold_contribution[0, ..., 0]).all()


def test_roi_matches_channel_removal_including_history_and_model_window(grouped_data):
    truth = grouped_data.truth
    nonmedia = sum(truth[name].values for name in COMPONENTS)
    for index, channel in enumerate(PAID_CHANNELS):
        counterfactual = truth.exposure.values.copy()
        counterfactual[..., index] = 0
        expected_without_channel = nonmedia + _numpy_contribution(truth, counterfactual).sum(axis=-1)
        incremental = (truth.expected_revenue.values - expected_without_channel).sum(axis=0)
        spend = truth.spend.sel(paid_channel=channel).values[8:].sum(axis=0)
        np.testing.assert_allclose(truth.roi.sel(paid_channel=channel).values, incremental / spend, rtol=2e-5)


def test_catalog_activity_distinguishes_always_on_media_from_inactive_flights(grouped_data):
    truth = grouped_data.truth
    for channel in grouped_data.channels.to_dict(orient="records"):
        exposure = truth.exposure.sel(channel=channel["channel"]).values[8:]
        if channel["activity"] == "always_on":
            assert (exposure > 0).all()
        else:
            assert channel["activity"] == "flights"
            assert (exposure == 0).any()
            assert (exposure > 0).any()
        if channel["kind"] == "paid":
            spend = truth.spend.sel(paid_channel=channel["channel"]).values[8:]
            np.testing.assert_array_equal(spend == 0, exposure == 0)


def test_seed_controls_reproducibility_without_touching_numpy_global_rng():
    first = simulate_data(seed=5, n_periods=20)
    second = simulate_data(seed=5, n_periods=20)
    different = simulate_data(seed=6, n_periods=20)
    assert_frame_equal(first.frame, second.frame)
    assert_frame_equal(first.media_history, second.media_history)
    assert_frame_equal(first.channels, second.channels)
    xr.testing.assert_identical(first.truth, second.truth)
    assert not np.array_equal(first.truth.exposure.values, different.truth.exposure.values)
    assert not np.array_equal(first.truth.revenue.values, different.truth.revenue.values)

    state = np.random.get_state()  # noqa: NPY002 - verify the shared legacy RNG is untouched
    simulate_data(seed=7, n_periods=3, groups=None)
    after = np.random.get_state()  # noqa: NPY002
    assert after[0] == state[0]
    np.testing.assert_array_equal(after[1], state[1])
    assert after[2:] == state[2:]


def test_measurement_error_changes_only_observed_exposures():
    exact = simulate_data(seed=19, n_periods=52, measurement_error=0)
    noisy = simulate_data(seed=19, n_periods=52, measurement_error=0.3)
    np.testing.assert_array_equal(exact.truth.observed_exposure.values, exact.truth.exposure.values)
    assert not np.array_equal(noisy.truth.observed_exposure.values, exact.truth.observed_exposure.values)
    assert (noisy.truth.observed_exposure.values >= 0).all()
    assert np.isfinite(noisy.truth.observed_exposure.values).all()
    np.testing.assert_array_equal(noisy.truth.observed_exposure.values == 0, exact.truth.exposure.values == 0)
    for name in exact.truth.data_vars:
        if name != "observed_exposure":
            xr.testing.assert_identical(noisy.truth[name], exact.truth[name])
    assert_frame_equal(noisy.frame.drop(columns=EXPOSURE_COLUMNS), exact.frame.drop(columns=EXPOSURE_COLUMNS))
    assert_frame_equal(noisy.channels, exact.channels)


def test_campaign_overlap_changes_media_and_noise_scale_changes_outcome_noise():
    independent = simulate_data(seed=8, n_periods=52, groups=None, campaign_overlap=0)
    shared = simulate_data(seed=8, n_periods=52, groups=None, campaign_overlap=1)
    assert not np.array_equal(independent.truth.exposure.values, shared.truth.exposure.values)
    assert not np.array_equal(independent.truth.spend.values, shared.truth.spend.values)

    noiseless = simulate_data(seed=8, n_periods=52, groups=None, noise_scale=0)
    noisy = simulate_data(seed=8, n_periods=52, groups=None, noise_scale=0.1)
    for name in noiseless.truth.data_vars:
        if name not in ("noise", "revenue"):
            xr.testing.assert_identical(noisy.truth[name], noiseless.truth[name])
    assert not np.array_equal(noisy.truth.noise.values, noiseless.truth.noise.values)
    assert_frame_equal(noisy.frame.drop(columns="revenue"), noiseless.frame.drop(columns="revenue"))


def test_full_campaign_overlap_shares_flight_timing_including_media_history():
    shared = simulate_data(seed=8, n_periods=52, campaign_overlap=1)
    independent = simulate_data(seed=8, n_periods=52, campaign_overlap=0)
    flight_channels = shared.channels.loc[
        (shared.channels["activity"] == "flights") & (shared.channels["channel"] != "snapchat"), "channel"
    ].to_list()
    shared_activity = shared.truth.campaign.values > 0
    xr.testing.assert_identical(independent.truth.campaign, shared.truth.campaign)
    for channel in flight_channels:
        np.testing.assert_array_equal(shared.truth.exposure.sel(channel=channel).values > 0, shared_activity)
    assert any(
        not np.array_equal(independent.truth.exposure.sel(channel=channel).values > 0, shared_activity)
        for channel in flight_channels
    )


@pytest.mark.parametrize("groups", [None, ("west", "east")])
def test_generated_frames_prepare_paid_organic_and_history_inputs(groups):
    result = simulate_data(seed=23, n_periods=20, groups=groups, measurement_error=0.2)
    paid = result.channels.loc[result.channels["kind"] == "paid"]
    organic = result.channels.loc[result.channels["kind"] == "organic"]
    data = prepare_data(
        result.frame,
        time="week",
        groups=[] if groups is None else ["region"],
        outcome="revenue",
        population="population",
        media=paid["exposure_column"].to_list(),
        spend=paid["spend_column"].to_list(),
        organic_media=organic["exposure_column"].to_list(),
        channels=paid["channel"].to_list(),
        organic_channels=organic["channel"].to_list(),
        controls=CONTROLS,
        media_history=result.media_history,
        frequency="weekly",
    )

    assert data.channels == PAID_CHANNELS
    assert data.organic_channels == ("email",)
    assert data.group_values == (() if groups is None else tuple((name,) for name in groups))
    assert len(data.time_values) == 20
    assert len(data.media_time_values) == 28
    assert set(data.arrays) == {"outcome", "population", "media", "spend", "organic_media", "controls"}
    np.testing.assert_array_equal(data.arrays["outcome"], result.truth.revenue.values)
    np.testing.assert_array_equal(data.arrays["media"], result.truth.observed_exposure.values[..., :-1])
    np.testing.assert_array_equal(data.arrays["organic_media"], result.truth.observed_exposure.values[..., -1:])
    np.testing.assert_array_equal(data.arrays["spend"], result.truth.spend.values[8:])
    np.testing.assert_array_equal(data.arrays["controls"].reshape(-1, len(CONTROLS)), result.frame[CONTROLS].to_numpy())


@pytest.mark.parametrize(
    "argument,value,error",
    [
        ("seed", -1, ValueError),
        ("seed", 1.5, TypeError),
        ("seed", True, TypeError),
        ("n_periods", 0, ValueError),
        ("n_periods", -1, ValueError),
        ("n_periods", 1.5, TypeError),
        ("n_periods", True, TypeError),
        ("groups", [], ValueError),
        ("groups", ["west", "west"], ValueError),
        ("groups", [""], ValueError),
        ("groups", ["west", 1], TypeError),
        ("groups", "west", TypeError),
        ("groups", {"west"}, TypeError),
        ("groups", {"west": 1}, TypeError),
        ("start", "2024-02-30", ValueError),
        ("start", "2024-01-01T00:00:00", ValueError),
        ("start", date.min, ValueError),
        ("start", date.max, ValueError),
        ("start", datetime(2024, 1, 1), TypeError),
        ("start", None, TypeError),
        ("campaign_overlap", -0.1, ValueError),
        ("campaign_overlap", 1.1, ValueError),
    ],
)
def test_simulation_rejects_invalid_arguments(argument, value, error):
    arguments = {"n_periods": 3, "groups": None, argument: value}
    with pytest.raises(error, match=argument):
        simulate_data(**arguments)


@pytest.mark.parametrize("argument", ["campaign_overlap", "noise_scale", "measurement_error"])
@pytest.mark.parametrize(
    "value,error",
    [(np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError), ("0.1", TypeError), (True, TypeError)],
)
def test_simulation_requires_finite_numeric_scale_arguments(argument, value, error):
    with pytest.raises(error, match=argument):
        simulate_data(n_periods=3, groups=None, **{argument: value})


@pytest.mark.parametrize("argument", ["noise_scale", "measurement_error"])
def test_simulation_rejects_negative_noise_scales(argument):
    with pytest.raises(ValueError, match=argument):
        simulate_data(n_periods=3, groups=None, **{argument: -0.1})


@pytest.mark.parametrize("campaign_overlap", [0, 1])
def test_simulation_supports_one_period_and_boundary_overlap_without_noise(campaign_overlap):
    result = simulate_data(seed=0, n_periods=1, groups=None, campaign_overlap=campaign_overlap, noise_scale=0)
    assert len(result.frame) == 1
    np.testing.assert_array_equal(result.truth.noise.values, 0)
    xr.testing.assert_allclose(result.truth.revenue, result.truth.expected_revenue)
    assert not np.isinf(result.truth.roi.values).any()
    spend = result.truth.spend.values[8:].sum(axis=0)
    contribution = result.truth.contribution.values[..., :-1].sum(axis=0)
    expected_roi = np.divide(contribution, spend, out=np.full_like(spend, np.nan), where=spend > 0)
    np.testing.assert_allclose(result.truth.roi.values, expected_roi, equal_nan=True)
