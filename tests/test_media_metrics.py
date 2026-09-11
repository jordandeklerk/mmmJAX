"""Tests for posterior media ROI and finite-increment marginal ROI."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

import mmmjax
from mmmjax import Model, Real, fit_data_scaling, media_metrics, prepare_data
from mmmjax._results import _collect_results


def _data(*, grouped=False, multiplier=1.0):
    rows = []
    for week, exposure in enumerate([[4.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]):
        for group in range(2 if grouped else 1):
            video, search = np.asarray(exposure) * (group + 1) * (multiplier if week else 1)
            rows.append(
                {
                    "week": week,
                    "region": ("east", "west")[group],
                    "video": video,
                    "search": search,
                    "video_cost": video / 2,
                    "search_cost": search / 4,
                    "outcome": 100.0 + 10 * week + 20 * group,
                    "control": 2.0 + week + group,
                    "population": 100.0 + 200 * group,
                }
            )
    frame = pl.DataFrame(rows)
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region"] if grouped else (),
        outcome="outcome",
        population="population",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Online video", "Paid search"],
        controls=["control"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _model(data, *, scaling=None):
    def transformed(media, spend, controls, coefficient):
        carried = media[1:] + 0.5 * media[:-1]
        expected = (
            3.0
            + controls[..., 0]
            + (carried @ coefficient) ** 2
            + 0.7 * carried[..., 0] * carried[..., 1]
            + 0.25 * spend.sum(axis=-1)
        )
        return {"expected_users": expected, "aggregate": expected.sum(), "exposure": media}

    def density(expected_users):
        raise AssertionError("Media metrics must not evaluate the log density")

    def generated(key, expected_users):
        raise AssertionError("Media metrics must not generate observations")

    return Model(
        {"coefficient": Real((2,))},
        density,
        generated,
        data=data,
        components=[],
        transformed_parameters=transformed,
        scaling=scaling,
        dims={"coefficient": ("channel",)},
    )


def _results(data):
    return _collect_results(
        {
            "coefficient": np.array(
                [[[0.5, 1.0], [1.0, 0.5], [2.0, 1.5]], [[0.8, 1.1], [1.5, 1.8], [2.5, 2.2]]], dtype=np.float32
            )
        },
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30]},
    )


@pytest.fixture
def case():
    data = _data()
    return data, _model(data), _results(data)


def _expected(
    data, coefficient, *, channel=None, multiplier=1.0, spend_periods=None, response_periods=None, scaling=None
):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    spending = (
        np.array([data.time_values.index(t) for t in spend_periods]) if spend_periods is not None else np.arange(3)
    )
    measurement = (
        np.array([data.time_values.index(t) for t in response_periods])
        if response_periods is not None
        else np.arange(3)
    )
    if channel is not None:
        arrays["spend"][spending, ..., channel] *= multiplier
        arrays["media"][1 + spending, ..., channel] *= multiplier
    scenario = replace(data, arrays=arrays)
    values = (scaling.transform(scenario) if scaling is not None else scenario).arrays
    carried = values["media"][1:] + 0.5 * values["media"][:-1]
    response = (
        3.0
        + values["controls"][..., 0]
        + (carried @ coefficient) ** 2
        + 0.7 * carried[..., 0] * carried[..., 1]
        + 0.25 * values["spend"].sum(axis=-1)
    )
    return response[measurement].sum()


def _assert_closed_form(metrics, data, results, **options):
    for chain in range(metrics.sizes["chain"]):
        for draw in range(metrics.sizes["draw"]):
            coefficient = results["posterior"]["coefficient"].values[chain, draw]
            reference = _expected(data, coefficient, **options)
            np.testing.assert_allclose(metrics["reference_response"][chain, draw], reference, rtol=3e-6)
            for index, label in enumerate(metrics.channel.values):
                channel = data.channels.index(label)
                spend = metrics["reference_spend"].values[index]
                increase = metrics["incremental_spend"].values[index]
                zero = _expected(data, coefficient, channel=channel, multiplier=0.0, **options)
                boosted = _expected(data, coefficient, channel=channel, multiplier=1.0 + increase / spend, **options)
                for name, expected in {
                    "incremental_response": reference - zero,
                    "roi": (reference - zero) / spend,
                    "marginal_response": boosted - reference,
                    "marginal_roi": (boosted - reference) / increase,
                }.items():
                    np.testing.assert_allclose(metrics[name][chain, draw, index], expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("grouped", [False, True])
def test_media_metrics_evaluate_full_nonlinear_model_for_every_paired_draw(grouped):
    data = _data(grouped=grouped)
    model, results = _model(data), _results(data)
    original_results = results.copy(deep=True)
    original_inputs = {name: np.asarray(value).copy() for name, value in model.data.values.items()}

    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25, batch_size=2)

    assert isinstance(metrics, xr.Dataset)
    assert "media_metrics" in mmmjax.__all__
    assert set(metrics.data_vars) == {
        "incremental_response",
        "roi",
        "marginal_response",
        "marginal_roi",
        "reference_spend",
        "incremental_spend",
        "reference_response",
    }
    for name in ("incremental_response", "roi", "marginal_response", "marginal_roi"):
        assert metrics[name].dims == ("chain", "draw", "channel")
    assert metrics["reference_response"].dims == ("chain", "draw")
    for name in ("reference_spend", "incremental_spend"):
        assert metrics[name].dims == ("channel",)
    for coord, expected in {
        "chain": [4, 8],
        "draw": [10, 20, 30],
        "channel": data.channels,
        "spend_period": [1, 2, 3],
        "response_period": [1, 2, 3],
    }.items():
        np.testing.assert_array_equal(metrics.coords[coord], expected)
    assert metrics.attrs["incremental_increase"] == 0.25
    assert isinstance(metrics.attrs["incremental_increase"], float)
    assert metrics.attrs["quantity"] == "expected_users"
    assert metrics.attrs["spend_to_media"] == "proportional"
    assert metrics.attrs["history"] == "fixed"
    assert metrics.attrs["response_units"] == "as returned by the transformed quantity"
    np.testing.assert_allclose(metrics["reference_spend"], data.arrays["spend"].sum(axis=(0, 1) if grouped else 0))
    _assert_closed_form(metrics, data, results)
    mean_parameters = results["posterior"]["coefficient"].values.mean(axis=(0, 1))
    assert not np.isclose(metrics["reference_response"].mean(), _expected(data, mean_parameters))
    xr.testing.assert_identical(results, original_results)
    for name, original in original_inputs.items():
        np.testing.assert_array_equal(model.data.values[name], original)


def test_media_metrics_preserve_posterior_labels_and_requested_channel_order(case):
    data, model, results = case
    selected = results.isel(chain=[1], draw=[2, 0])
    metrics = media_metrics(model, selected, quantity="expected_users", channels=["Paid search", "Online video"])
    np.testing.assert_array_equal(metrics.channel, ["Paid search", "Online video"])
    np.testing.assert_array_equal(metrics.chain, [8])
    np.testing.assert_array_equal(metrics.draw, [30, 10])
    one = media_metrics(model, selected, quantity="expected_users", channels=["Online video"])
    xr.testing.assert_allclose(one, metrics.sel(channel=["Online video"]))
    _assert_closed_form(metrics, data, selected)


@pytest.mark.parametrize("already_scaled", [False, True])
def test_media_metrics_reuse_fitted_scales_and_report_transformed_quantity_units(already_scaled):
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, scale_outcome=True, adjust_population=True)
    model = _model(scaling.transform(data) if already_scaled else data, scaling=scaling)
    results = _results(data)
    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25)
    np.testing.assert_allclose(metrics["reference_spend"], data.arrays["spend"].sum(axis=(0, 1)), rtol=2e-6)
    _assert_closed_form(metrics, data, results, scaling=scaling)


def test_media_metrics_prepare_new_data_with_existing_fitted_scales(case):
    data, _, results = case
    model = _model(data, scaling="auto")
    new_data = _data(multiplier=2.0)
    metrics = media_metrics(model, results, quantity="expected_users", new_data=new_data, incremental_increase=0.25)
    np.testing.assert_allclose(metrics["reference_spend"], new_data.arrays["spend"].sum(axis=0), rtol=2e-6)
    _assert_closed_form(metrics, new_data, results, scaling=model.scaling)
    np.testing.assert_array_equal(model.data.values["spend"], data.arrays["spend"])


def test_media_metrics_keep_history_and_unselected_periods_fixed(case):
    data, model, results = case
    windows = {"spend_periods": [2, 1], "response_periods": [3, 2]}
    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25, **windows)
    np.testing.assert_array_equal(metrics.spend_period, [1, 2])
    np.testing.assert_array_equal(metrics.response_period, [2, 3])
    np.testing.assert_allclose(metrics["reference_spend"], data.arrays["spend"][:2].sum(axis=0))
    _assert_closed_form(metrics, data, results, **windows)
    carryover = media_metrics(
        model, results, quantity="expected_users", spend_periods=[1], response_periods=[2], incremental_increase=0.25
    )
    _assert_closed_form(carryover, data, results, spend_periods=[1], response_periods=[2])
    assert np.all(carryover["incremental_response"].values > 0)


def _linear_model(*, media=(1, 3), spend=(1.0, 3.0), baseline=0.0, scaling=None):
    data = prepare_data(
        pl.DataFrame({"week": np.arange(len(media)), "views": media, "cost": spend}),
        time="week",
        media=["views"],
        spend=["cost"],
        channels=["Video"],
    )
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected_users": baseline + media[:, 0] * coefficient},
        scaling=scaling,
    )
    return model, _collect_results({"coefficient": np.array([[1.0, 2.0]], dtype=np.float32)}, data=data)


def test_media_metrics_custom_conversion_keeps_fractional_integer_media():
    model, results = _linear_model()
    assert jnp.issubdtype(model.data.values["media"].dtype, jnp.integer)
    metrics = media_metrics(
        model, results, quantity="expected_users", incremental_increase=0.25, spend_to_media=lambda spend: 0.5 * spend
    )
    assert metrics.attrs["spend_to_media"] == "custom"
    np.testing.assert_allclose(metrics["reference_response"], [[2.0, 4.0]])
    np.testing.assert_allclose(metrics["incremental_response"][0, :, 0], [2.0, 4.0])
    np.testing.assert_allclose(metrics["roi"][0, :, 0], [0.5, 1.0])
    np.testing.assert_allclose(metrics["marginal_roi"][0, :, 0], [0.5, 1.0])
    np.testing.assert_array_equal(model.data.values["media"], [[1], [3]])


def test_media_metrics_use_custom_converter_with_raw_new_dataframe_spend():
    model, results = _linear_model(scaling="auto")
    new_data = pl.DataFrame({"week": [5, 4], "views": [8, 6], "cost": [16.0, 12.0]})
    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        new_data=new_data,
        spend_to_media=lambda spend: 0.5 * spend,
        incremental_increase=0.25,
    )
    expected = model.scaling.transformations["media"].transform(jnp.array([[6.0], [8.0]])).sum()
    np.testing.assert_allclose(metrics["reference_response"], [[expected, 2 * expected]], rtol=2e-6)
    np.testing.assert_allclose(metrics["reference_spend"], [28.0])
    np.testing.assert_allclose(metrics["roi"][0, :, 0], np.array([expected, 2 * expected]) / 28, rtol=2e-6)
    np.testing.assert_allclose(metrics["marginal_roi"], metrics["roi"], rtol=2e-6)
    np.testing.assert_array_equal(metrics.spend_period, [4, 5])


def test_media_metrics_allow_negative_and_zero_model_lift():
    model, results = _linear_model()
    results["posterior"]["coefficient"][:] = [[-2.0, 0.0]]
    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25)
    np.testing.assert_allclose(metrics["incremental_response"][0, :, 0], [-8.0, 0.0])
    np.testing.assert_allclose(metrics["roi"][0, :, 0], [-2.0, 0.0])
    np.testing.assert_allclose(metrics["marginal_response"][0, :, 0], [-2.0, 0.0])
    np.testing.assert_allclose(metrics["marginal_roi"][0, :, 0], [-2.0, 0.0])


def test_media_metrics_subtract_paired_observations_before_summing_large_baselines():
    model, results = _linear_model(media=np.ones(100), spend=np.ones(100), baseline=1_000_000.0)
    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=0.5)
    np.testing.assert_allclose(metrics["incremental_response"][0, :, 0], [100.0, 200.0])
    np.testing.assert_allclose(metrics["marginal_response"][0, :, 0], [50.0, 100.0])
    np.testing.assert_allclose(metrics["roi"][0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(metrics["marginal_roi"][0, :, 0], [1.0, 2.0])


def test_media_metrics_use_actual_representable_spend_increment():
    model, results = _linear_model(media=(7.0,), spend=(7.0,))
    increase = 1.25 * np.finfo(model._dtype).eps
    metrics = media_metrics(model, results, quantity="expected_users", incremental_increase=increase)
    reference = np.asarray(7.0, dtype=model._dtype)
    actual_increase = reference * np.asarray(1.0 + increase, dtype=model._dtype) - reference
    np.testing.assert_array_equal(metrics["incremental_spend"], [actual_increase])
    assert not np.isclose(actual_increase, 7 * increase, rtol=0.01, atol=0)
    np.testing.assert_allclose(metrics["marginal_roi"][0, :, 0], [1.0, 2.0], rtol=1e-6)


def test_media_metrics_defaults_and_batches_preserve_values(case):
    _, model, results = case
    default = media_metrics(model, results, quantity="expected_users")
    explicit = media_metrics(
        model, results, quantity="expected_users", incremental_increase=0.01, spend_to_media="proportional"
    )
    xr.testing.assert_identical(default, explicit)
    for batch_size in (1, 4):
        batched = media_metrics(model, results, quantity="expected_users", batch_size=batch_size)
        xr.testing.assert_allclose(default, batched, rtol=5e-5, atol=2e-5)


@pytest.mark.parametrize("increase", [0, -0.1, np.nan, np.inf, True, "0.01", [0.01], np.array(0.01), 0.01j])
def test_media_metrics_reject_invalid_incremental_increases(case, increase):
    _, model, results = case
    with pytest.raises((TypeError, ValueError), match="incremental_increase"):
        media_metrics(model, results, quantity="expected_users", incremental_increase=increase)


def test_media_metrics_reject_unrepresentably_small_and_overflowing_increases(case):
    _, model, results = case
    with pytest.raises(ValueError, match="too small"):
        media_metrics(model, results, quantity="expected_users", incremental_increase=np.finfo(model._dtype).eps / 8)
    with pytest.raises(ValueError, match=r"large|precision|finite|overflow"):
        media_metrics(model, results, quantity="expected_users", incremental_increase=np.finfo(model._dtype).max)


def test_media_metrics_require_positive_spend_only_for_selected_channels(case):
    data, _, results = case
    data.arrays["spend"][..., 0] = 0
    data.arrays["media"][1:, ..., 0] = 0
    model = _model(data)
    with pytest.raises(ValueError, match="positive reference spending"):
        media_metrics(model, results, quantity="expected_users")
    metrics = media_metrics(model, results, quantity="expected_users", channels=["Paid search"])
    assert metrics.sizes["channel"] == 1
    assert np.isfinite(metrics["roi"]).all()


@pytest.mark.parametrize("quantity", ["missing", "aggregate", "exposure"])
def test_media_metrics_require_observation_shaped_quantities(case, quantity):
    _, model, results = case
    with pytest.raises(ValueError, match=r"quantity|shape"):
        media_metrics(model, results, quantity=quantity)


@pytest.mark.parametrize("conversion", [lambda spend: spend.sum(), lambda spend: -spend, lambda spend: spend * jnp.nan])
def test_media_metrics_reject_invalid_converted_media(case, conversion):
    _, model, results = case
    with pytest.raises((TypeError, ValueError), match=r"media|shape|finite"):
        media_metrics(model, results, quantity="expected_users", spend_to_media=conversion)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"channels": ["unknown"]}, "channel"),
        ({"channels": ["Online video", "Online video"]}, "channel"),
        ({"spend_periods": [99]}, "spend_periods"),
        ({"response_periods": [1, 1]}, "response_periods"),
        ({"batch_size": True}, "batch_size"),
        ({"spend_to_media": "automatic"}, "spend_to_media"),
    ],
)
def test_media_metrics_share_response_context_validation(case, options, message):
    _, model, results = case
    with pytest.raises((TypeError, ValueError), match=message):
        media_metrics(model, results, quantity="expected_users", **options)
