"""Tests for prior and posterior media ROI and finite-increment marginal ROI."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

import mmmjax
from mmmjax import Data, Model, Positive, Real, fit_data_scaling, media_metrics, prepare_data
from mmmjax._results import _collect_results


def _data(*, grouped=False, multiplier=1.0, compound=False, reverse=False):
    rows = []
    for week, exposure in enumerate([[4.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]):
        for group in range(3 if compound else 2 if grouped else 1):
            video, search = np.asarray(exposure) * (group + 1) * (multiplier if week else 1)
            rows.append(
                {
                    "week": week,
                    "region": ("east", "west", "east")[group],
                    "store": ("retail", "online", "online")[group],
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
    if reverse:
        frame = frame.reverse()
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region", "store"] if compound else ["region"] if grouped else (),
        outcome="outcome",
        population="population",
        media=["search", "video"] if reverse else ["video", "search"],
        spend=["search_cost", "video_cost"] if reverse else ["video_cost", "search_cost"],
        channels=["Paid search", "Online video"] if reverse else ["Online video", "Paid search"],
        controls=["control"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _model(data, *, scaling=None, cross_group=False):
    def transformed(media, spend, controls, coefficient):
        carried = media[1:] + 0.5 * media[:-1]
        expected = (
            3.0
            + controls[..., 0]
            + (carried @ coefficient) ** 2
            + 0.7 * carried[..., 0] * carried[..., 1]
            + 0.25 * spend.sum(axis=-1)
        )
        if cross_group:
            expected = expected + 0.2 * carried[..., 0].sum(axis=1, keepdims=True) ** 2
        return {"expected_users": expected, "aggregate": expected.sum(), "exposure": media}

    def density(expected_users):
        raise AssertionError("Media metrics must not evaluate the log density")

    def generated(key, expected_users):
        raise AssertionError("Media metrics must not generate observations")

    return Model(
        parameters={"coefficient": Real((2,))},
        log_density=density,
        generated_quantities=generated,
        data=Data(data, scaling=scaling),
        transformed_parameters=transformed,
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
    data,
    coefficient,
    *,
    channel=None,
    multiplier=1.0,
    spend_periods=None,
    response_periods=None,
    scaling=None,
    by=(),
    cross_group=False,
):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    spending = (
        np.array([data.time_values.index(t) for t in spend_periods])
        if spend_periods is not None
        else np.arange(len(data.time_values))
    )
    measurement = (
        np.array(sorted(data.time_values.index(t) for t in response_periods))
        if response_periods is not None
        else np.arange(len(data.time_values))
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
    if cross_group:
        response = response + 0.2 * carried[..., 0].sum(axis=1, keepdims=True) ** 2
    axes = ("time", "group") if data.group_columns else ("time",)
    return response[measurement].sum(axis=tuple(index for index, axis in enumerate(axes) if axis not in by))


def _assert_closed_form(metrics, data, results, *, group="posterior", **options):
    by = metrics["reference_response"].dims[2:]
    for chain in range(metrics.sizes["chain"]):
        for draw in range(metrics.sizes["draw"]):
            coefficient = results[group]["coefficient"].values[chain, draw]
            reference = _expected(data, coefficient, by=by, **options)
            np.testing.assert_allclose(metrics["reference_response"][chain, draw], reference, rtol=3e-6)
            for index, label in enumerate(metrics.channel.values):
                channel = data.channels.index(label)
                spend = metrics["reference_spend"].values[index]
                increase = metrics["incremental_spend"].values[index]
                zero = _expected(data, coefficient, channel=channel, multiplier=0.0, by=by, **options)
                boosted = _expected(
                    data, coefficient, channel=channel, multiplier=1.0 + increase / spend, by=by, **options
                )
                for name, expected in {
                    "incremental_response": reference - zero,
                    "roi": (reference - zero).sum() / spend,
                    "marginal_response": boosted - reference,
                    "marginal_roi": (boosted - reference).sum() / increase,
                }.items():
                    np.testing.assert_allclose(
                        metrics[name].isel(chain=chain, draw=draw, channel=index), expected, rtol=3e-5, atol=3e-5
                    )


@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_media_metrics_selects_prior_or_posterior_draws_with_detailed_scaled_scenarios(group):
    data = _data(compound=True)
    scaling = fit_data_scaling(data, adjust_population=True)
    model = _model(data, scaling=scaling, cross_group=True)
    prior = _collect_results(
        {"coefficient": np.array([[[0.2, 0.7], [1.4, 0.4], [0.8, 1.9], [1.7, 1.0]]], dtype=np.float32)},
        sample_group="prior",
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [42], "draw": [9, 3, 7, 5]},
    )
    results = xr.DataTree.from_dict(
        {"prior": prior["prior"].to_dataset(), "posterior": _results(data)["posterior"].to_dataset()}
    )
    new_data = _data(compound=True, multiplier=2.0, reverse=True)
    expected_data = _data(compound=True, multiplier=2.0)
    options = {
        "quantity": "expected_users",
        "group": group,
        "new_data": new_data,
        "spend_periods": [3, 1],
        "response_periods": [3, 2],
        "incremental_increase": 0.25,
    }
    detailed = media_metrics(model, results, by=("group", "time"), batch_size=3, **options)
    aggregate = media_metrics(model, results, batch_size=1, **options)
    assert detailed.attrs["group"] == group
    assert detailed["incremental_response"].dims == ("chain", "draw", "time", "group", "channel")
    np.testing.assert_array_equal(detailed.chain, results[group].chain)
    np.testing.assert_array_equal(detailed.draw, results[group].draw)
    np.testing.assert_array_equal(detailed.time, [2, 3])
    np.testing.assert_array_equal(detailed.group_region, ["east", "west", "east"])
    np.testing.assert_array_equal(detailed.group_store, ["retail", "online", "online"])
    _assert_closed_form(
        detailed,
        expected_data,
        results,
        group=group,
        scaling=scaling,
        cross_group=True,
        spend_periods=[1, 3],
        response_periods=[2, 3],
    )
    for name in ("incremental_response", "marginal_response", "reference_response"):
        xr.testing.assert_allclose(detailed[name].sum(("time", "group")), aggregate[name], rtol=3e-5, atol=3e-5)
    for name in ("roi", "marginal_roi", "reference_spend", "incremental_spend"):
        xr.testing.assert_allclose(detailed[name], aggregate[name], rtol=3e-5, atol=3e-5)

    if group == "prior":
        prior_only = media_metrics(model, prior, by=("group", "time"), batch_size=3, **options)
        xr.testing.assert_identical(detailed, prior_only)
        results["posterior"]["coefficient"] = xr.full_like(results["posterior"]["coefficient"], np.nan)
        ignored = media_metrics(model, results, by=("group", "time"), batch_size=3, **options)
        xr.testing.assert_identical(detailed, ignored)


@pytest.mark.parametrize("group", [None, "prior_predictive", ["prior"]])
def test_media_metrics_rejects_invalid_draw_group(case, group):
    _, model, results = case
    with pytest.raises(ValueError, match="group"):
        media_metrics(model, results, quantity="expected_users", group=group)


def test_media_metrics_requires_selected_prior_group(case):
    _, model, results = case
    with pytest.raises(ValueError, match="prior"):
        media_metrics(model, results, quantity="expected_users", group="prior")


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


@pytest.mark.parametrize("by", ["time", "group", ("time", "group"), ["group", "time"]])
def test_media_metrics_breakdowns_preserve_full_model_interactions_and_aggregate_roi(by):
    data = _data(grouped=True)
    model, results = _model(data, cross_group=True), _results(data)
    options = {"quantity": "expected_users", "incremental_increase": 0.25}
    aggregate = media_metrics(model, results, **options)
    detailed = media_metrics(model, results, by=by, batch_size=4, **options)
    axes = tuple(axis for axis in ("time", "group") if axis in by)

    assert detailed["reference_response"].dims == ("chain", "draw", *axes)
    for name in ("incremental_response", "marginal_response"):
        assert detailed[name].dims == ("chain", "draw", *axes, "channel")
    for name in ("reference_response", "incremental_response", "marginal_response"):
        xr.testing.assert_allclose(detailed[name].sum(axes), aggregate[name], rtol=3e-6, atol=3e-5)
    for name in ("roi", "marginal_roi", "reference_spend", "incremental_spend"):
        xr.testing.assert_allclose(detailed[name], aggregate[name], rtol=3e-6, atol=3e-5)
    if "time" in axes:
        np.testing.assert_array_equal(detailed.time, [1, 2, 3])
    if "group" in axes:
        np.testing.assert_array_equal(detailed.group, ["east", "west"])
    np.testing.assert_array_equal(detailed.chain, [4, 8])
    np.testing.assert_array_equal(detailed.draw, [10, 20, 30])
    _assert_closed_form(detailed, data, results, cross_group=True)
    if by == ("time", "group"):
        unbatched = media_metrics(model, results, by=by, **options)
        xr.testing.assert_allclose(detailed, unbatched, rtol=3e-6, atol=3e-5)


def test_media_metrics_national_time_breakdown_preserves_posterior_and_channel_selection(case):
    data, model, results = case
    selected = results.isel(chain=[1], draw=[2, 0])
    metrics = media_metrics(
        model,
        selected,
        quantity="expected_users",
        by="time",
        channels=["Paid search", "Online video"],
        response_periods=[3, 1],
        incremental_increase=0.25,
        batch_size=3,
    )
    assert metrics["incremental_response"].dims == ("chain", "draw", "time", "channel")
    assert metrics["reference_response"].dims == ("chain", "draw", "time")
    assert "group" not in metrics.coords
    np.testing.assert_array_equal(metrics.time, [1, 3])
    np.testing.assert_array_equal(metrics.response_period, [1, 3])
    np.testing.assert_array_equal(metrics.channel, ["Paid search", "Online video"])
    np.testing.assert_array_equal(metrics.chain, [8])
    np.testing.assert_array_equal(metrics.draw, [30, 10])
    _assert_closed_form(metrics, data, selected, response_periods=[3, 1])


@pytest.mark.parametrize("compound", [False, True])
def test_media_metrics_breakdown_new_data_aligns_group_and_channel_labels_with_fitted_scales(compound):
    data = _data(grouped=True, compound=compound)
    scaling = fit_data_scaling(data, scale_outcome=True, adjust_population=True)
    model, results = _model(data, scaling=scaling, cross_group=True), _results(data)
    original = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    new_data = _data(grouped=True, compound=compound, multiplier=2.0, reverse=True)
    expected_data = _data(grouped=True, compound=compound, multiplier=2.0)
    assert new_data.group_values == data.group_values[::-1]
    assert new_data.channels == data.channels[::-1]

    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        new_data=new_data,
        by=("group", "time"),
        incremental_increase=0.25,
        batch_size=4,
    )

    assert metrics["incremental_response"].dims == ("chain", "draw", "time", "group", "channel")
    np.testing.assert_array_equal(metrics.channel, data.channels)
    np.testing.assert_array_equal(metrics.time, expected_data.time_values)
    if compound:
        np.testing.assert_array_equal(metrics.group, [0, 1, 2])
        np.testing.assert_array_equal(metrics.group_region, ["east", "west", "east"])
        np.testing.assert_array_equal(metrics.group_store, ["retail", "online", "online"])
        assert metrics.group_region.dims == metrics.group_store.dims == ("group",)
    else:
        np.testing.assert_array_equal(metrics.group, ["east", "west"])
        assert "group_region" not in metrics.coords
    np.testing.assert_allclose(metrics["reference_spend"], expected_data.arrays["spend"].sum(axis=(0, 1)), rtol=2e-6)
    _assert_closed_form(metrics, expected_data, results, scaling=scaling, cross_group=True)
    for name, value in original.items():
        np.testing.assert_array_equal(model.data.values[name], value)


def test_media_metrics_breakdown_reports_carryover_on_zero_spend_response_dates():
    data = _data(grouped=True)
    data.arrays["spend"][1] = 0.0
    data.arrays["media"][2] = 0.0
    model, results = _model(data, cross_group=True), _results(data)
    windows = {"spend_periods": [1], "response_periods": [3, 2]}
    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        by=("time", "group"),
        incremental_increase=0.25,
        **windows,
    )

    np.testing.assert_array_equal(metrics.time, [2, 3])
    np.testing.assert_array_equal(metrics.spend_period, [1])
    np.testing.assert_array_equal(metrics.response_period, [2, 3])
    np.testing.assert_allclose(metrics["reference_spend"], data.arrays["spend"][0].sum(axis=0))
    for name in ("incremental_response", "marginal_response"):
        assert np.all(metrics[name].sel(time=2).values > 0)
        np.testing.assert_array_equal(metrics[name].sel(time=3), 0.0)
    assert metrics["roi"].dims == metrics["marginal_roi"].dims == ("chain", "draw", "channel")
    assert np.isfinite(metrics["roi"]).all()
    assert np.isfinite(metrics["marginal_roi"]).all()
    _assert_closed_form(metrics, data, results, cross_group=True, **windows)


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


@pytest.mark.parametrize("already_scaled", [False, True])
@pytest.mark.parametrize("new_scenario", [False, True])
@pytest.mark.parametrize("by", [(), ("time",), ("group",), ("time", "group")])
def test_media_metrics_population_outcome_inversion_restores_revenue_and_roi(already_scaled, new_scenario, by):
    frame = pl.DataFrame(
        {
            "week": [0, 0, 1, 1, 2, 2],
            "region": ["east", "west"] * 3,
            "population": [100.0, 400.0] * 3,
            "outcome": [100.0, 1200.0, 200.0, 1600.0, 300.0, 2000.0],
            "revenue": [2.0, 3.0, 4.0, 2.0, 3.0, 5.0],
            "video": [10.0, 20.0, 30.0, 60.0, 50.0, 100.0],
            "search": [5.0, 15.0, 10.0, 30.0, 20.0, 60.0],
        }
    ).with_columns(video_cost=pl.col("video") / 2, search_cost=pl.col("search") / 4)
    data = prepare_data(
        frame,
        time="week",
        groups=["region"],
        population="population",
        outcome="outcome",
        revenue_per_outcome="revenue",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
    )
    scaling = fit_data_scaling(data, scale_outcome="population", adjust_population=True)
    outcome_scaling = scaling.transformations["outcome"]

    def transformed(media, revenue_per_outcome, coefficient):
        expected_standardized = 0.25 + media @ coefficient
        expected_outcome = outcome_scaling.inverse_transform(expected_standardized)
        return {"expected_revenue": expected_outcome * revenue_per_outcome}

    def density(coefficient):
        raise AssertionError("Media metrics must not evaluate the log density")

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=density,
        data=Data(scaling.transform(data) if already_scaled else data, scaling=scaling),
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    results = _results(data)
    reference_data = data
    new_data = None
    if new_scenario:
        arrays = {name: value.copy() for name, value in data.arrays.items()}
        arrays["media"] *= 1.5
        arrays["spend"] *= 1.5
        arrays["outcome"] *= 5.0
        reference_data = replace(data, arrays=arrays)
        new_data = scaling.transform(reference_data) if already_scaled else reference_data

    metrics = media_metrics(
        model,
        results,
        quantity="expected_revenue",
        new_data=new_data,
        by=by,
        incremental_increase=0.25,
        batch_size=4,
    )

    population = data.arrays["population"]
    per_person_outcome = data.arrays["outcome"] / population
    mean, deviation = per_person_outcome.mean(), per_person_outcome.std()
    median_media = np.median(data.arrays["media"] / population[:, None], axis=(0, 1))
    scaled_media = reference_data.arrays["media"] / (population[:, None] * median_media)
    revenue_factor = deviation * population * reference_data.arrays["revenue_per_outcome"]
    coefficient = results["posterior"]["coefficient"].values
    lift = np.einsum("tgc,adc->adtgc", scaled_media, coefficient) * revenue_factor[None, None, ..., None]
    baseline = (mean + 0.25 * deviation) * population * reference_data.arrays["revenue_per_outcome"]
    reference = baseline[None, None] + lift.sum(axis=-1)
    spend = reference_data.arrays["spend"].sum(axis=(0, 1))
    axes = tuple(index + 2 for index, axis in enumerate(("time", "group")) if axis not in by)
    expected = {
        "reference_response": reference.sum(axis=axes),
        "incremental_response": lift.sum(axis=axes),
        "marginal_response": 0.25 * lift.sum(axis=axes),
        "reference_spend": spend,
        "incremental_spend": 0.25 * spend,
        "roi": lift.sum(axis=(2, 3)) / spend,
        "marginal_roi": lift.sum(axis=(2, 3)) / spend,
    }
    assert metrics["reference_response"].dims == ("chain", "draw", *by)
    assert metrics["incremental_response"].dims == ("chain", "draw", *by, "channel")
    for name, values in expected.items():
        np.testing.assert_allclose(metrics[name], values, rtol=3e-6, atol=3e-3)


def test_media_metrics_keep_model_reference_inputs_fixed_when_changing_channel_spend():
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, scale_outcome="population", adjust_population=True)

    def quantities(media, reference, outcome_scaling, n_periods, roi):
        original = reference.media[-reference.n_periods :]
        reference_response = original / (1.0 + original)
        denominator = jnp.sum(reference_response * outcome_scaling.scale[None, :, None], axis=(0, 1))
        coefficient = roi * reference.spend.sum(axis=(0, 1)) / denominator
        current = media[-n_periods:]
        standardized = 0.25 + jnp.sum(current / (1.0 + current) * coefficient, axis=-1)
        return {"expected_revenue": outcome_scaling.inverse_transform(standardized)}

    def density(roi):
        raise AssertionError("Media metrics must not evaluate the log density")

    model = Model(
        parameters={"roi": Positive(dims="channel")},
        log_density=density,
        data=Data(data, scaling=scaling),
        transformed_parameters=quantities,
    )
    roi = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    results = _collect_results({"roi": roi}, data=data, dims={"roi": ("channel",)})
    options = {"quantity": "expected_revenue", "incremental_increase": 0.25, "by": ("time", "group")}
    baseline = media_metrics(model, results, **options)
    changed_data = _data(grouped=True, multiplier=2.0)
    changed = media_metrics(model, results, new_data=changed_data, **options)

    reference_media = scaling.transform(data).arrays["media"][1:]
    outcome_scale = np.asarray(scaling.transformations["outcome"].scale)[..., None]
    reference_weight = reference_media / (1.0 + reference_media) * outcome_scale
    denominator = reference_weight.sum(axis=(0, 1))
    spend = data.arrays["spend"].sum(axis=(0, 1))
    coefficient = roi * spend / denominator
    np.testing.assert_allclose(baseline["roi"], roi, rtol=2e-5)
    np.testing.assert_allclose(baseline["reference_spend"], spend)
    np.testing.assert_allclose(
        baseline["incremental_response"], reference_weight * coefficient[:, :, None, None, :], rtol=2e-5, atol=2e-5
    )

    changed_media = scaling.transform(changed_data).arrays["media"][1:]
    changed_weight = changed_media / (1.0 + changed_media) * outcome_scale
    expected_lift = changed_weight * coefficient[:, :, None, None, :]
    np.testing.assert_allclose(changed["reference_spend"], 2.0 * spend)
    np.testing.assert_allclose(changed["incremental_response"], expected_lift, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(changed["roi"], expected_lift.sum(axis=(2, 3)) / (2.0 * spend), rtol=2e-5)
    marginal_weight = 1.25 * changed_media / (1.0 + 1.25 * changed_media) * outcome_scale - changed_weight
    expected_marginal = marginal_weight * coefficient[:, :, None, None, :]
    np.testing.assert_allclose(changed["marginal_response"], expected_marginal, rtol=2e-5, atol=2e-5)
    assert np.all(changed["reference_response"].values > baseline["reference_response"].values)
    assert np.all(changed["roi"].values < roi)
    assert np.all(changed["marginal_roi"].values > 0.0)
    np.testing.assert_array_equal(model.data.reference.spend, data.arrays["spend"])


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
        parameters={"coefficient": Real()},
        log_density=lambda coefficient: -(coefficient**2),
        data=Data(data, scaling=scaling),
        transformed_parameters=lambda media, coefficient: {"expected_users": baseline + media[:, 0] * coefficient},
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
    detailed = media_metrics(model, results, quantity="expected_users", incremental_increase=0.5, by="time")
    for name in ("incremental_response", "marginal_response"):
        xr.testing.assert_allclose(detailed[name].sum("time"), metrics[name])


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
        model,
        results,
        quantity="expected_users",
        incremental_increase=0.01,
        spend_to_media="proportional",
        group="posterior",
    )
    xr.testing.assert_identical(default, explicit)
    for by in (None, (), []):
        xr.testing.assert_identical(default, media_metrics(model, results, quantity="expected_users", by=by))
    for batch_size in (1, 4):
        batched = media_metrics(model, results, quantity="expected_users", batch_size=batch_size)
        xr.testing.assert_allclose(default, batched, rtol=5e-5, atol=2e-5)


@pytest.mark.parametrize(
    "by",
    ["region", "", ("time", "time"), ("time", 1), ("time", ["group"]), 1, b"time", {"time"}, {"time": True}],
)
def test_media_metrics_reject_invalid_breakdown_axes(case, by):
    _, model, results = case
    with pytest.raises(ValueError, match="by"):
        media_metrics(model, results, quantity="expected_users", by=by)


@pytest.mark.parametrize("by", ["group", ("time", "group")])
def test_media_metrics_reject_group_breakdown_for_national_data(case, by):
    _, model, results = case
    with pytest.raises(ValueError, match="group"):
        media_metrics(model, results, quantity="expected_users", by=by)


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


def _rf_case(*, mixed=False, scaled=False):
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3],
            "audience": [3.0, 4.0, 6.0, 8.0],
            "frequency": [1.0, 2.0, 3.0, 4.0],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "search": [6.0, 2.0, 4.0, 6.0],
            "search_cost": [3.0, 1.0, 2.0, 3.0],
        }
    )
    options = {"media": ["search"], "spend": ["search_cost"], "channels": ["Search"]} if mixed else {}
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        reach=["audience"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
        rf_channels=["Video"],
        media_history=frame.filter(pl.col("week") == 0),
        **options,
    )

    def response(reach, frequency, coefficient, media=None):
        saturated = reach[:, 0] * frequency[:, 0] / (1.0 + frequency[:, 0])
        carried = saturated[1:] + 0.5 * saturated[:-1]
        expected = 3.0 + coefficient**2 * carried
        if media is not None:
            expected = expected + coefficient * media[1:, 0] * (1.0 + carried)
        return {"expected_users": expected}

    if mixed:

        def transformed(reach, media_frequency, media, coefficient):
            return response(reach, media_frequency, coefficient, media)

    else:

        def transformed(reach, media_frequency, coefficient):
            return response(reach, media_frequency, coefficient)

    def density(expected_users):
        raise AssertionError("Media metrics must only evaluate transformed quantities")

    model = Model(
        parameters={"coefficient": Real()},
        log_density=density,
        data=Data(data, scaling="auto" if scaled else None),
        transformed_parameters=transformed,
    )
    results = _collect_results(
        {"coefficient": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)},
        coords={"chain": [4, 8], "draw": [10, 30]},
    )
    return data, model, results


def _rf_response(data, coefficient, *, channel=None, multiplier=1.0, mode="reach", scaling=None):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    # Spend only in the first period and measure its carryover in the next period.
    if channel == "Search":
        arrays["spend"][0, 0] *= multiplier
        arrays["media"][1, 0] *= multiplier
    elif channel == "Video":
        arrays["rf_spend"][0, 0] *= multiplier
        if not callable(mode):
            arrays["reach" if mode == "reach" else "media_frequency"][1, 0] *= multiplier
    if callable(mode):
        reach, frequency = mode(arrays["rf_spend"])
        arrays["reach"][1] = np.asarray(reach)[0]
        arrays["media_frequency"][1] = np.asarray(frequency)[0]
    scenario = replace(data, arrays=arrays)
    values = (scaling.transform(scenario) if scaling is not None else scenario).arrays
    saturated = values["reach"][:, 0] * values["media_frequency"][:, 0] / (1.0 + values["media_frequency"][:, 0])
    carried = saturated[2] + 0.5 * saturated[1]
    expected = 3.0 + coefficient**2 * carried
    if "media" in values:
        expected = expected + coefficient * values["media"][2, 0] * (1.0 + carried)
    return expected


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("mode", ["reach", "frequency", lambda spend: (3.0 * spend, jnp.ones_like(spend) * 2.0)])
def test_media_metrics_rf_roi_and_marginal_roi_follow_paired_nonlinear_scenarios(mixed, scaled, mode):
    data, model, results = _rf_case(mixed=mixed, scaled=scaled)
    original = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    labels = ["Video", "Search"] if mixed else ["Video"]
    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        channels=labels,
        incremental_increase=0.25,
        spend_to_rf=mode,
        spend_periods=[1],
        response_periods=[2],
        batch_size=3,
    )
    np.testing.assert_array_equal(metrics.channel, labels)
    np.testing.assert_array_equal(metrics.channel_type, ["reach_frequency", "media"] if mixed else ["reach_frequency"])
    np.testing.assert_array_equal(metrics.chain, [4, 8])
    np.testing.assert_array_equal(metrics.draw, [10, 30])
    np.testing.assert_allclose(metrics["reference_spend"], [2.0, 1.0] if mixed else [2.0])
    np.testing.assert_allclose(metrics["incremental_spend"], [0.5, 0.25] if mixed else [0.5])
    assert metrics.attrs["spend_to_rf"] == ("custom" if callable(mode) else mode)
    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        reference = _rf_response(data, coefficient, mode=mode, scaling=model.scaling)
        np.testing.assert_allclose(metrics["reference_response"][chain, draw], reference, rtol=2e-6)
        for index, channel in enumerate(labels):
            zero = _rf_response(data, coefficient, channel=channel, multiplier=0.0, mode=mode, scaling=model.scaling)
            increased = _rf_response(
                data, coefficient, channel=channel, multiplier=1.25, mode=mode, scaling=model.scaling
            )
            spend = 2.0 if channel == "Video" else 1.0
            for name, expected in {
                "incremental_response": reference - zero,
                "roi": (reference - zero) / spend,
                "marginal_response": increased - reference,
                "marginal_roi": (increased - reference) / (spend * 0.25),
            }.items():
                np.testing.assert_allclose(metrics[name][chain, draw, index], expected, rtol=2e-5, atol=2e-5)
    assert np.all(metrics["roi"].sel(channel="Video") > 0)
    for name, value in original.items():
        np.testing.assert_array_equal(model.data.values[name], value)


def test_media_metrics_time_breakdown_preserves_mixed_rf_channel_scenarios():
    data, model, results = _rf_case(mixed=True, scaled=True)
    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        by="time",
        channels=["Video", "Search"],
        spend_to_rf="frequency",
        spend_periods=[1],
        response_periods=[2],
        incremental_increase=0.25,
        batch_size=3,
    )
    np.testing.assert_array_equal(metrics.channel, ["Video", "Search"])
    np.testing.assert_array_equal(metrics.channel_type, ["reach_frequency", "media"])
    np.testing.assert_array_equal(metrics.time, [2])
    assert metrics["reference_response"].dims == ("chain", "draw", "time")
    for name in ("incremental_response", "marginal_response"):
        assert metrics[name].dims == ("chain", "draw", "time", "channel")
    for name in ("roi", "marginal_roi"):
        assert metrics[name].dims == ("chain", "draw", "channel")
    for name in ("reference_spend", "incremental_spend"):
        assert metrics[name].dims == ("channel",)
    np.testing.assert_allclose(metrics["reference_spend"], [2.0, 1.0])
    np.testing.assert_allclose(metrics["incremental_spend"], [0.5, 0.25])
    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        options = {"mode": "frequency", "scaling": model.scaling}
        reference = _rf_response(data, coefficient, **options)
        np.testing.assert_allclose(metrics["reference_response"][chain, draw, 0], reference, rtol=2e-6)
        for channel, spend in (("Video", 2.0), ("Search", 1.0)):
            zero = _rf_response(data, coefficient, channel=channel, multiplier=0.0, **options)
            increased = _rf_response(data, coefficient, channel=channel, multiplier=1.25, **options)
            for name, expected in {
                "incremental_response": reference - zero,
                "marginal_response": increased - reference,
                "roi": (reference - zero) / spend,
                "marginal_roi": (increased - reference) / (0.25 * spend),
            }.items():
                actual = metrics[name].sel(channel=channel).isel(chain=chain, draw=draw)
                if "time" in actual.dims:
                    actual = actual.isel(time=0)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_media_metrics_evaluates_prior_rf_metrics_with_custom_raw_spend_conversion():
    data, model, _ = _rf_case(mixed=True, scaled=True)
    results = _collect_results(
        {"coefficient": np.array([[0.25, 1.1, 2.0]], dtype=np.float32)},
        sample_group="prior",
        coords={"chain": [42], "draw": [9, 3, 7]},
    )

    def convert(spend):
        return 3.0 * spend, jnp.ones_like(spend) * 2.0

    metrics = media_metrics(
        model,
        results,
        group="prior",
        quantity="expected_users",
        channels=["Video", "Search"],
        spend_to_rf=convert,
        spend_periods=[1],
        response_periods=[2],
        by="time",
        incremental_increase=0.25,
        batch_size=2,
    )
    assert metrics.attrs["group"] == "prior"
    assert metrics.attrs["spend_to_rf"] == "custom"
    np.testing.assert_array_equal(metrics.chain, [42])
    np.testing.assert_array_equal(metrics.draw, [9, 3, 7])
    for draw in range(3):
        coefficient = results["prior"]["coefficient"].values[0, draw]
        reference = _rf_response(data, coefficient, mode=convert, scaling=model.scaling)
        np.testing.assert_allclose(metrics["reference_response"][0, draw, 0], reference, rtol=2e-6)
        for channel, spend in (("Video", 2.0), ("Search", 1.0)):
            zero = _rf_response(data, coefficient, channel=channel, multiplier=0.0, mode=convert, scaling=model.scaling)
            boosted = _rf_response(
                data, coefficient, channel=channel, multiplier=1.25, mode=convert, scaling=model.scaling
            )
            for name, expected in {
                "incremental_response": reference - zero,
                "marginal_response": boosted - reference,
                "roi": (reference - zero) / spend,
                "marginal_roi": (boosted - reference) / (0.25 * spend),
            }.items():
                actual = metrics[name].sel(channel=channel).isel(chain=0, draw=draw)
                if "time" in actual.dims:
                    actual = actual.isel(time=0)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_media_metrics_rf_default_reach_and_selected_channels_match_full_result():
    _, model, results = _rf_case(mixed=True)
    default = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25)
    explicit = media_metrics(model, results, quantity="expected_users", incremental_increase=0.25, spend_to_rf="reach")
    xr.testing.assert_identical(default, explicit)
    selected = media_metrics(
        model,
        results.isel(chain=[1], draw=[1, 0]),
        quantity="expected_users",
        channels=["Video"],
        incremental_increase=0.25,
    )
    xr.testing.assert_allclose(selected, default.sel(channel=["Video"]).isel(chain=[1], draw=[1, 0]))


@pytest.mark.parametrize("channel", ["Search", "Video"])
@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_media_metrics_preserve_unselected_family_exposure_with_zero_spend(channel, mode):
    data, model, results = _rf_case(mixed=True)
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    arrays["rf_spend" if channel == "Search" else "spend"][:] = 0.0
    new_data = replace(data, arrays=arrays)
    metrics = media_metrics(
        model,
        results,
        quantity="expected_users",
        channels=[channel],
        new_data=new_data,
        spend_to_rf=mode,
        incremental_increase=0.25,
    )
    coefficient = results["posterior"]["coefficient"].values[..., None]

    def response(multiplier):
        reach = arrays["reach"][:, 0].copy()
        frequency = arrays["media_frequency"][:, 0].copy()
        media = arrays["media"][:, 0].copy()
        if channel == "Search":
            media[1:] *= multiplier
        elif mode == "reach":
            reach[1:] *= multiplier
        else:
            frequency[1:] *= multiplier
        saturated = reach * frequency / (1.0 + frequency)
        carried = saturated[1:] + 0.5 * saturated[:-1]
        return (3.0 + coefficient**2 * carried + coefficient * media[1:] * (1.0 + carried)).sum(axis=-1)

    reference = response(1.0)
    incremental = reference - response(0.0)
    marginal = response(1.25) - reference
    spend = 6.0 if channel == "Search" else 9.0
    np.testing.assert_array_equal(metrics.channel, [channel])
    np.testing.assert_array_equal(metrics.channel_type, ["media" if channel == "Search" else "reach_frequency"])
    np.testing.assert_allclose(metrics["reference_spend"], [spend])
    np.testing.assert_allclose(metrics["reference_response"], reference, rtol=2e-6)
    for name, expected in {
        "incremental_response": incremental,
        "roi": incremental / spend,
        "marginal_response": marginal,
        "marginal_roi": marginal / (0.25 * spend),
    }.items():
        np.testing.assert_allclose(metrics[name].sel(channel=channel), expected, rtol=2e-5, atol=2e-5)
    for name in ("media", "reach", "media_frequency"):
        np.testing.assert_array_equal(new_data.arrays[name], data.arrays[name])


@pytest.mark.parametrize(("channel", "role", "conversion"), [("Search", "spend", "media"), ("Video", "rf_spend", "rf")])
@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_media_metrics_reject_selected_positive_exposure_at_zero_spend(channel, role, conversion, mode):
    data, model, results = _rf_case(mixed=True)
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    arrays[role][0, 0] = 0.0

    with pytest.raises(ValueError, match=f"explicit spend_to_{conversion}"):
        media_metrics(
            model,
            results,
            quantity="expected_users",
            channels=[channel],
            new_data=replace(data, arrays=arrays),
            spend_to_rf=mode,
        )


def test_media_metrics_rf_validate_custom_conversion_at_increased_spending():
    data, model, results = _rf_case()
    reference = jnp.asarray(data.arrays["rf_spend"])

    def conversion(spend):
        frequency = jnp.where(spend > 1.1 * reference, -1.0, 2.0)
        return 2.0 * spend, frequency

    with pytest.raises(ValueError, match=r"invalid|finite|frequency"):
        media_metrics(
            model,
            results,
            quantity="expected_users",
            incremental_increase=0.25,
            spend_to_rf=conversion,
        )
