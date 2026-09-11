"""Tests for posterior response curves with proportional or custom spending assumptions."""

from dataclasses import replace
from datetime import date, datetime

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

import mmmjax
from mmmjax import MediaEffect, Model, Real, fit_data_scaling, prepare_data, response_curves
from mmmjax._results import _collect_results
from mmmjax.response import _BudgetResponse, _prepare_response


def _data(*, grouped=False, multiplier=1.0):
    rows = []
    media = np.array([[8.0, 12.0], [2.0, 3.0], [0.0, 5.0], [7.0, 2.0]])
    for time, exposure in enumerate(media):
        for group in range(2 if grouped else 1):
            exposure = media[time] * (group + 1)
            if time:
                exposure = exposure * multiplier
            rows.append(
                {
                    "week": time,
                    "region": ("east", "west")[group],
                    "video": exposure[0],
                    "search": exposure[1],
                    "video_spend": exposure[0] / 2,
                    "search_spend": exposure[1] / 3,
                    "sales": 100.0 + 10 * time + 20 * group,
                    "temperature": 10.0 + time + group,
                    "residents": 100.0 + 200 * group,
                }
            )
    frame = pl.DataFrame(rows)
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region"] if grouped else (),
        population="residents",
        outcome="sales",
        media=["video", "search"],
        spend=["video_spend", "search_spend"],
        channels=["Online video", "Paid search"],
        controls=["temperature"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _model(data, *, scaling=None):
    def transformed(media, spend, controls, coefficient):
        carried = media[1:] + 0.5 * media[:-1]
        predictor = (
            10.0
            + controls[..., 0]
            + carried @ coefficient
            + 0.03 * carried[..., 0] * carried[..., 1]
            + 0.2 * spend.sum(axis=-1)
        )
        expected = jnp.exp(0.01 * predictor)
        return {"expected": expected, "aggregate": expected.sum(), "exposure": media}

    def density(expected):
        raise AssertionError("Response curves must not evaluate the log density")

    def generated(key, expected):
        raise AssertionError("Response curves must not evaluate generated observations")

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


def _results(model, data):
    values = np.array([[[0.5, 0.7], [1.0, 1.2], [2.0, 1.5]], [[0.8, 1.1], [1.5, 1.8], [2.5, 2.2]]], dtype=np.float32)
    return _collect_results(
        {"coefficient": values},
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30]},
        sample_stats={"diverging": np.zeros((2, 3), dtype=bool)},
    )


@pytest.fixture
def integer_media_model():
    frame = pl.DataFrame({"week": [0, 1], "views": [1, 3], "cost": [1.0, 3.0]})
    data = prepare_data(frame, time="week", media=["views"], spend=["cost"])
    return Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
    )


def _expected(data, coefficient, *, scaling=None):
    values = (scaling.transform(data) if scaling is not None else data).arrays
    carried = values["media"][1:] + 0.5 * values["media"][:-1]
    predictor = (
        10.0
        + values["controls"][..., 0]
        + carried @ coefficient
        + 0.03 * carried[..., 0] * carried[..., 1]
        + 0.2 * values["spend"].sum(axis=-1)
    )
    return np.exp(0.01 * predictor).sum()


def _scenario(data, channel, multiplier, *, conversion=None):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    arrays["spend"][..., channel] *= multiplier
    if conversion is None:
        arrays["media"][1:, ..., channel] *= multiplier
    else:
        arrays["media"][1:] = np.asarray(conversion(arrays["spend"]))
    return replace(data, arrays=arrays)


def test_response_curves_default_to_proportional_media():
    data = _data()
    model = _model(data)
    results = _results(model, data)

    curves = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0])
    explicit = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], spend_to_media="proportional"
    )

    xr.testing.assert_identical(curves, explicit)


@pytest.mark.parametrize("conversion", ["proportional", lambda spend: spend])
def test_response_curves_keep_fractional_exposures_and_gradients_with_integer_inputs(integer_media_model, conversion):
    model = integer_media_model
    results = _collect_results({"coefficient": jnp.ones((1, 1))})
    assert jnp.issubdtype(model.data.values["media"].dtype, jnp.integer)

    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 0.5, 1.0, 1.5], spend_to_media=conversion
    )
    np.testing.assert_allclose(curves["response"][0, 0, 0], [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(curves["incremental_response"][0, 0, 0], [0.0, 2.0, 4.0, 6.0])
    context = _prepare_response(model, results, quantity="expected", spend_to_media=conversion)
    budgets = context.reference_spend * 0.5
    value, gradient = jax.jit(jax.value_and_grad(lambda allocation: context.evaluator(allocation).mean()))(budgets)
    np.testing.assert_allclose(value, 2.0)
    np.testing.assert_allclose(gradient, [1.0])
    scenario, valid = jax.jit(context.evaluator._scenario_inputs)(budgets)
    assert valid
    assert scenario.values["media"].dtype == model._dtype
    np.testing.assert_array_equal(scenario.values["media"], [[0.5], [1.5]])
    np.testing.assert_array_equal(model.data.values["media"], [[1], [3]])


def test_response_curves_reject_fractional_negative_exposures_with_integer_inputs(integer_media_model):
    results = _collect_results({"coefficient": jnp.ones((1, 1))})
    with pytest.raises(ValueError, match="invalid media"):
        response_curves(
            integer_media_model,
            results,
            quantity="expected",
            multipliers=[1.0],
            spend_to_media=lambda spend: jnp.full_like(spend, -0.25),
        )


@pytest.mark.parametrize("grouped", [False, True])
def test_response_curves_evaluate_full_model_per_draw_with_fixed_history(grouped):
    data = _data(grouped=grouped)
    model = _model(data)
    results = _results(model, data)
    original_results = results.copy(deep=True)
    original_inputs = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    multipliers = [0.0, 0.5, 1.0, 2.0]

    curves = response_curves(
        model, results, quantity="expected", multipliers=multipliers, spend_to_media="proportional", batch_size=2
    )

    assert isinstance(curves, xr.Dataset)
    assert "response_curves" in mmmjax.__all__
    assert curves["response"].dims == ("chain", "draw", "channel", "multiplier")
    assert curves["incremental_response"].dims == curves["response"].dims
    assert curves["spend"].dims == ("channel", "multiplier")
    assert curves["reference_spend"].dims == ("channel",)
    assert curves["reference_response"].dims == ("chain", "draw")
    np.testing.assert_array_equal(curves.channel, data.channels)
    np.testing.assert_array_equal(curves.chain, [4, 8])
    np.testing.assert_array_equal(curves.draw, [10, 20, 30])
    np.testing.assert_array_equal(curves.multiplier, multipliers)

    reference_spend = data.arrays["spend"].sum(axis=tuple(range(data.arrays["spend"].ndim - 1)))
    np.testing.assert_allclose(curves["reference_spend"], reference_spend, rtol=1e-6)
    np.testing.assert_allclose(curves["spend"], reference_spend[:, None] * multipliers, rtol=1e-6)
    for chain in range(2):
        for draw in range(3):
            coefficient = results["posterior"]["coefficient"].values[chain, draw]
            reference = _expected(data, coefficient)
            np.testing.assert_allclose(curves["reference_response"][chain, draw], reference, rtol=2e-6)
            for channel in range(2):
                zero = _expected(_scenario(data, channel, 0.0), coefficient)
                for index, multiplier in enumerate(multipliers):
                    expected = _expected(_scenario(data, channel, multiplier), coefficient)
                    np.testing.assert_allclose(curves["response"][chain, draw, channel, index], expected, rtol=2e-6)
                    np.testing.assert_allclose(
                        curves["incremental_response"][chain, draw, channel, index], expected - zero, atol=3e-6
                    )
    np.testing.assert_allclose(curves["incremental_response"].sel(multiplier=0), 0.0, atol=1e-6)
    expected_reference = (
        curves["reference_response"].expand_dims(channel=curves.channel).transpose("chain", "draw", "channel")
    )
    np.testing.assert_allclose(curves["response"].sel(multiplier=1), expected_reference, rtol=1e-6)
    xr.testing.assert_identical(results, original_results)
    for name, original in original_inputs.items():
        np.testing.assert_array_equal(model.data.values[name], original)


def test_response_curves_average_responses_not_parameters():
    data = _data()
    model = _model(data)
    results = _results(model, data)
    curves = response_curves(model, results, quantity="expected", multipliers=[2.0], spend_to_media="proportional")
    coefficient = results["posterior"]["coefficient"].values.mean(axis=(0, 1))
    evaluated_mean = _expected(_scenario(data, 0, 2.0), coefficient)
    assert not np.isclose(curves["response"].isel(channel=0, multiplier=0).mean(), evaluated_mean, rtol=1e-4)


def test_response_curves_recompute_adstock_and_saturation_without_a_generation_callback():
    data = _data()

    def transformed(paid_media_total):
        return {"expected": jnp.exp(paid_media_total)}

    model = Model(
        {},
        lambda expected: expected.sum(),
        data=data,
        components=[MediaEffect(max_lag=1, normalize=False)],
        transformed_parameters=transformed,
    )
    parameters = {
        "paid_media_coefficient": [0.8, 1.2],
        "paid_media_retention": [0.3, 0.6],
        "paid_media_half_saturation": [2.0, 3.0],
        "paid_media_slope": [1.0, 2.0],
    }
    results = _collect_results(
        {name: np.asarray(value, dtype=np.float32)[None, None, :] for name, value in parameters.items()},
        data=data,
        dims=dict.fromkeys(parameters, ("channel",)),
    )
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], spend_to_media="proportional"
    )
    for channel in range(2):
        for index, multiplier in enumerate([0.0, 1.0, 2.0]):
            scenario = _scenario(data, channel, multiplier)
            media = scenario.arrays["media"]
            carried = media[1:] + media[:-1] * parameters["paid_media_retention"]
            numerator = carried ** parameters["paid_media_slope"]
            denominator = (
                numerator + np.asarray(parameters["paid_media_half_saturation"]) ** parameters["paid_media_slope"]
            )
            expected = np.exp((numerator / denominator) @ parameters["paid_media_coefficient"]).sum()
            np.testing.assert_allclose(curves["response"][0, 0, channel, index], expected, rtol=2e-6)


def test_response_curves_preserve_channel_selection_and_posterior_coordinates():
    data = _data()
    model = _model(data)
    posterior = _results(model, data)["posterior"].to_dataset().isel(chain=[1], draw=[2, 0])
    selected = xr.DataTree.from_dict({"posterior": posterior.transpose("channel", "draw", "chain")})
    curves = response_curves(
        model,
        selected,
        quantity="expected",
        multipliers=[1.0, 0.0],
        channels=["Paid search", "Online video"],
        spend_to_media="proportional",
    )
    np.testing.assert_array_equal(curves.channel, ["Paid search", "Online video"])
    np.testing.assert_array_equal(curves.chain, [8])
    np.testing.assert_array_equal(curves.draw, [30, 10])
    one = response_curves(
        model,
        selected,
        quantity="expected",
        multipliers=[1.0, 0.0],
        channels=["Online video"],
        spend_to_media="proportional",
    )
    xr.testing.assert_allclose(one["response"], curves["response"].sel(channel=["Online video"]))


@pytest.mark.parametrize("already_scaled", [False, True])
def test_response_curves_reuse_fitted_scaling_and_leave_response_units_to_model(already_scaled):
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, scale_outcome=True, adjust_population=True)
    model = _model(scaling.transform(data) if already_scaled else data, scaling=scaling)
    results = _results(model, data)
    curves = response_curves(model, results, quantity="expected", multipliers=[0.0, 2.0], spend_to_media="proportional")
    for channel in range(2):
        for index, multiplier in enumerate([0.0, 2.0]):
            expected = _expected(
                _scenario(data, channel, multiplier), results["posterior"]["coefficient"].values[0, 0], scaling=scaling
            )
            np.testing.assert_allclose(curves["response"][0, 0, channel, index], expected, rtol=2e-6)
    np.testing.assert_allclose(curves["reference_spend"], data.arrays["spend"].sum(axis=(0, 1)), rtol=1e-6)


def test_response_curves_use_custom_spend_conversion_in_raw_units():
    data = _data()
    model = _model(data, scaling="auto")
    results = _results(model, data)
    reference_spend = jnp.asarray(data.arrays["spend"])

    def conversion(spend):
        relative_increase = (spend - reference_spend) / (1.0 + reference_spend)
        return spend * jnp.array([2.0, 3.0]) / (1.0 + 0.1 * relative_increase)

    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], spend_to_media=conversion
    )
    for channel in range(2):
        scenario = _scenario(data, channel, 2.0, conversion=conversion)
        expected = _expected(scenario, results["posterior"]["coefficient"].values[0, 0], scaling=model.scaling)
        np.testing.assert_allclose(curves["response"][0, 0, channel, 2], expected, rtol=2e-6)


def test_response_curves_prepare_scenarios_without_changing_training_reference():
    data = _data()
    model = _model(data, scaling="auto")
    results = _results(model, data)
    scenario = _data(multiplier=2.0)
    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[1.0, 2.0],
        spend_to_media="proportional",
        new_data=scenario,
    )
    expected = _expected(scenario, results["posterior"]["coefficient"].values[0, 0], scaling=model.scaling)
    np.testing.assert_allclose(curves["reference_response"][0, 0], expected, rtol=2e-6)
    np.testing.assert_allclose(curves["reference_spend"], scenario.arrays["spend"].sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(model.data.values["spend"], data.arrays["spend"], rtol=1e-6)


def test_response_curves_batches_produce_the_same_draws_and_values():
    data = _data()
    model = _model(data)
    results = _results(model, data)
    options = {"quantity": "expected", "multipliers": [0.0, 1.0, 2.0], "spend_to_media": "proportional"}
    single = response_curves(model, results, batch_size=1, **options)
    uneven = response_curves(model, results, batch_size=4, **options)
    large = response_curves(model, results, batch_size=64, **options)
    # Batch shapes can change floating-point evaluation order by a few ulps.
    xr.testing.assert_allclose(single, uneven, rtol=2e-6, atol=2e-7)
    xr.testing.assert_allclose(single, large, rtol=2e-6, atol=2e-7)


def test_budget_response_supports_joint_allocations_and_compiled_gradients():
    data = _data()
    model = _model(data)
    results = _results(model, data)
    spend = data.arrays["spend"]
    reference_budgets = spend.sum(axis=0)
    weights = spend / reference_budgets
    posterior = {"coefficient": jnp.asarray(results["posterior"]["coefficient"].values)}
    evaluator = _BudgetResponse(
        model=model,
        inputs=model.data,
        posterior=posterior,
        quantity="expected",
        spend_weights=jnp.asarray(weights),
        convert=lambda candidate: candidate * jnp.array([2.0, 3.0]),
        observation_shape=spend.shape[:-1],
        batch_size=4,
    )
    budgets = reference_budgets * np.array([1.2, 0.8])
    value, gradient = jax.jit(jax.value_and_grad(lambda allocation: evaluator(allocation).mean()))(jnp.asarray(budgets))

    current_media = weights * budgets * [2.0, 3.0]
    media = np.concatenate((data.arrays["media"][:1], current_media))
    carried = media[1:] + 0.5 * media[:-1]
    current_derivative = weights * [2.0, 3.0]
    carried_derivative = current_derivative + 0.5 * np.concatenate((np.zeros((1, 2)), current_derivative[:-1]))
    expected_values = []
    expected_gradients = []
    for coefficient in np.asarray(posterior["coefficient"]).reshape(-1, 2):
        predictor = (
            10.0
            + data.arrays["controls"][..., 0]
            + carried @ coefficient
            + 0.03 * carried[..., 0] * carried[..., 1]
            + 0.2 * (weights * budgets).sum(axis=-1)
        )
        response = np.exp(0.01 * predictor)
        derivative = carried_derivative * coefficient + 0.03 * carried_derivative * carried[:, ::-1] + 0.2 * weights
        expected_values.append(response.sum())
        expected_gradients.append((0.01 * response[:, None] * derivative).sum(axis=0))

    np.testing.assert_allclose(value, np.mean(expected_values), rtol=2e-6)
    np.testing.assert_allclose(gradient, np.mean(expected_gradients, axis=0), rtol=2e-6)
    assert np.all(np.asarray(gradient) > 0)


def test_response_curves_subtract_paired_observations_before_aggregation():
    frame = pl.DataFrame({"week": np.arange(100), "video": np.ones(100), "cost": np.ones(100)})
    data = prepare_data(frame, time="week", media=["video"], spend=["cost"])

    def transformed(media, coefficient):
        return {"expected": 1_000_000.0 + media[:, 0] * coefficient}

    model = Model(
        {"coefficient": Real()},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)}, data=data)
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], spend_to_media="proportional"
    )
    np.testing.assert_allclose(curves["incremental_response"][0, 0, 0], [0.0, 100.0, 200.0], rtol=1e-6)


@pytest.mark.parametrize("multipliers", [[], [-1.0], [float("nan")], [float("inf")], [[0.0, 1.0]], [True], [1.0, 1.0]])
def test_response_curves_reject_invalid_multiplier_grids(multipliers):
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match="multipliers"):
        response_curves(
            model, _results(model, data), quantity="expected", multipliers=multipliers, spend_to_media="proportional"
        )


@pytest.mark.parametrize("channels", [[], ["Unknown"], ["Online video", "Online video"], "Online video"])
def test_response_curves_reject_invalid_channel_selections(channels):
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match="channel"):
        response_curves(
            model,
            _results(model, data),
            quantity="expected",
            multipliers=[1.0],
            channels=channels,
            spend_to_media="proportional",
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_response_curves_reject_invalid_batch_sizes(batch_size):
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        response_curves(
            model,
            _results(model, data),
            quantity="expected",
            multipliers=[1.0],
            spend_to_media="proportional",
            batch_size=batch_size,
        )


@pytest.mark.parametrize("quantity", ["missing", "aggregate", "exposure"])
def test_response_curves_require_an_observation_shaped_transformed_quantity(quantity):
    data = _data()
    model = _model(data)
    with pytest.raises(ValueError, match=r"quantity|shape"):
        response_curves(
            model, _results(model, data), quantity=quantity, multipliers=[1.0], spend_to_media="proportional"
        )


def test_response_curves_reject_unknown_spend_conversion():
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match="spend_to_media"):
        response_curves(
            model, _results(model, data), quantity="expected", multipliers=[1.0], spend_to_media="automatic"
        )


@pytest.mark.parametrize("conversion", [lambda spend: spend.sum(), lambda spend: -spend, lambda spend: spend * jnp.nan])
def test_response_curves_reject_invalid_converted_exposures(conversion):
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match=r"media|exposure|shape|finite"):
        response_curves(model, _results(model, data), quantity="expected", multipliers=[1.0], spend_to_media=conversion)


def test_response_curves_require_paid_media_and_spend():
    data = _data()
    no_spend = replace(data, arrays={name: value for name, value in data.arrays.items() if name != "spend"})

    def transformed(media, coefficient):
        return {"expected": media[1:] @ coefficient}

    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=no_spend,
        components=[],
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    with pytest.raises(ValueError, match="spend"):
        response_curves(
            model, _results(model, no_spend), quantity="expected", multipliers=[1.0], spend_to_media="proportional"
        )


def test_response_curves_require_spending_to_define_selected_channel_allocation():
    data = _data()
    data.arrays["spend"][..., 0] = 0
    data.arrays["media"][1:, ..., 0] = 0
    model = _model(data)
    results = _results(model, data)
    options = {"quantity": "expected", "multipliers": [0.0, 1.0], "spend_to_media": "proportional"}
    with pytest.raises(ValueError, match="positive reference spending"):
        response_curves(model, results, **options)
    curves = response_curves(model, results, channels=["Paid search"], **options)
    assert curves.sizes["channel"] == 1
    assert np.isfinite(curves["response"]).all()


def test_response_curves_require_custom_conversion_for_exposures_without_spend():
    data = _data()
    data.arrays["spend"][0, 0] = 0
    model = _model(data)
    with pytest.raises(ValueError, match="zero spend"):
        response_curves(
            model, _results(model, data), quantity="expected", multipliers=[1.0], spend_to_media="proportional"
        )


def test_response_curves_accept_dataframes_and_evaluate_the_supplied_periods_only():
    frame = pl.DataFrame({"week": [0, 1, 2], "views": [10.0, 20.0, 40.0], "cost": [2.0, 4.0, 8.0]})
    data = prepare_data(frame, time="week", media=["views"], spend=["cost"])
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
        scaling="auto",
    )
    results = _collect_results({"coefficient": jnp.ones((1, 1))})
    scenario = pl.DataFrame({"week": [5, 4], "views": [80.0, 60.0], "cost": [16.0, 12.0]})
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0], spend_to_media="proportional", new_data=scenario
    )
    expected = model.scaling.transformations["media"].transform(jnp.array([[60.0], [80.0]])).sum()
    np.testing.assert_allclose(curves["reference_response"], expected, rtol=1e-6)
    np.testing.assert_allclose(curves["reference_spend"], [28.0])
    np.testing.assert_array_equal(model.data.values["spend"], data.arrays["spend"])


def _window_scenario(data, channel, multiplier, periods, *, conversion=None):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    indices = np.array([data.time_values.index(period) for period in periods])
    arrays["spend"][indices, ..., channel] *= multiplier
    history_length = len(data.media_time_values) - len(data.time_values)
    if conversion is None:
        arrays["media"][history_length + indices, ..., channel] *= multiplier
    else:
        arrays["media"][history_length + indices] = np.asarray(conversion(arrays["spend"]))[indices]
    return replace(data, arrays=arrays)


def _window_expected(data, coefficient, periods, *, scaling=None):
    values = (scaling.transform(data) if scaling is not None else data).arrays
    carried = values["media"][1:] + 0.5 * values["media"][:-1]
    predictor = (
        10.0
        + values["controls"][..., 0]
        + carried @ coefficient
        + 0.03 * carried[..., 0] * carried[..., 1]
        + 0.2 * values["spend"].sum(axis=-1)
    )
    indices = [data.time_values.index(period) for period in periods]
    return np.exp(0.01 * predictor)[indices].sum()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_response_curves_use_separate_noncontiguous_spend_and_response_periods(grouped, scaled):
    data = _data(grouped=grouped)
    scaling = fit_data_scaling(data, adjust_population=grouped, scale_outcome=True) if scaled else None
    model = _model(data, scaling=scaling)
    results = _results(model, data)
    original = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        spend_to_media="proportional",
        spend_periods=[3, 1],
        response_periods=[3, 2],
    )
    np.testing.assert_array_equal(curves["spend_period"], [1, 3])
    np.testing.assert_array_equal(curves["response_period"], [2, 3])
    assert curves["spend_period"].dims == ("spend_period",)
    assert curves["response_period"].dims == ("response_period",)
    reference_spend = data.arrays["spend"][[0, 2]].sum(axis=tuple(range(data.arrays["spend"].ndim - 1)))
    np.testing.assert_allclose(curves["reference_spend"], reference_spend, rtol=1e-6)
    np.testing.assert_allclose(curves["spend"], reference_spend[:, None] * [0.0, 1.0, 2.0], rtol=1e-6)

    for chain in range(2):
        for draw in range(3):
            coefficient = results["posterior"]["coefficient"].values[chain, draw]
            expected_reference = _window_expected(data, coefficient, [2, 3], scaling=scaling)
            np.testing.assert_allclose(curves["reference_response"][chain, draw], expected_reference, rtol=2e-6)
            for channel in range(2):
                zero = _window_scenario(data, channel, 0.0, [1, 3])
                comparison = _window_expected(zero, coefficient, [2, 3], scaling=scaling)
                for index, multiplier in enumerate([0.0, 1.0, 2.0]):
                    scenario = _window_scenario(data, channel, multiplier, [1, 3])
                    expected = _window_expected(scenario, coefficient, [2, 3], scaling=scaling)
                    np.testing.assert_allclose(curves["response"][chain, draw, channel, index], expected, rtol=2e-6)
                    np.testing.assert_allclose(
                        curves["incremental_response"][chain, draw, channel, index], expected - comparison, atol=3e-6
                    )
    for name, value in original.items():
        np.testing.assert_array_equal(model.data.values[name], value)


def test_response_curves_measure_carryover_after_spending_has_ended():
    data = _data()

    def transformed(media, coefficient):
        return {"expected": (media[1:] + 0.5 * media[:-1]) @ coefficient}

    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    results = _results(model, data)
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 1.0, 2.0],
        "spend_to_media": "proportional",
        "channels": ["Online video"],
        "spend_periods": [1],
    }
    curves = response_curves(model, results, response_periods=[2], **options)
    coefficient = results["posterior"]["coefficient"].values
    # Week two retains half of week one's exposure even though spending changes only in week one.
    expected_increment = coefficient[..., 0, None] * [0.0, 1.0, 2.0]
    expected_total = expected_increment + 6.5 * coefficient[..., 1, None]
    np.testing.assert_allclose(curves["incremental_response"].isel(channel=0), expected_increment, rtol=1e-6)
    np.testing.assert_allclose(curves["response"].isel(channel=0), expected_total, rtol=1e-6)
    np.testing.assert_allclose(curves["spend"], [[0.0, 1.0, 2.0]], rtol=1e-6)

    after_carryover = response_curves(model, results, response_periods=[3], **options)
    np.testing.assert_array_equal(after_carryover["incremental_response"], 0.0)


def test_response_curves_confine_custom_conversion_to_selected_spending_periods():
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, adjust_population=True)
    model = _model(data, scaling=scaling)
    results = _results(model, data)

    def conversion(spend):
        # Conversion sees the whole schedule, but only the selected rows replace observed exposure.
        return spend * jnp.array([2.0, 3.0]) + spend.sum(axis=(0, 1), keepdims=True)

    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        channels=["Paid search"],
        spend_to_media=conversion,
        spend_periods=[2],
        response_periods=[1, 3],
    )
    coefficient = results["posterior"]["coefficient"].values[0, 0]
    zero = _window_scenario(data, 1, 0.0, [2], conversion=conversion)
    comparison = _window_expected(zero, coefficient, [1, 3], scaling=scaling)
    for index, multiplier in enumerate([0.0, 1.0, 2.0]):
        scenario = _window_scenario(data, 1, multiplier, [2], conversion=conversion)
        expected = _window_expected(scenario, coefficient, [1, 3], scaling=scaling)
        np.testing.assert_allclose(curves["response"][0, 0, 0, index], expected, rtol=2e-6)
        np.testing.assert_allclose(curves["incremental_response"][0, 0, 0, index], expected - comparison, atol=3e-6)


@pytest.mark.parametrize("argument", ["spend_periods", "response_periods"])
@pytest.mark.parametrize("periods", [[], [1, 1], [0], [4], [True], [1, False], "1", ["1"]])
def test_response_curves_reject_invalid_period_labels(argument, periods):
    data = _data()
    model = _model(data)
    with pytest.raises((TypeError, ValueError), match=argument):
        response_curves(
            model,
            _results(model, data),
            quantity="expected",
            multipliers=[1.0],
            spend_to_media="proportional",
            **{argument: periods},
        )


@pytest.mark.parametrize("time_type", [date, datetime])
def test_response_curves_accept_iso_selections_for_native_datetime_labels(time_type):
    labels = [time_type(2026, 1, day) for day in [5, 12, 19]]
    frame = pl.DataFrame({"week": labels, "views": [10.0, 20.0, 40.0], "cost": [2.0, 4.0, 8.0]})
    data = prepare_data(frame, time="week", media=["views"], spend=["cost"])
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
    )
    results = _collect_results({"coefficient": jnp.ones((1, 1))})
    options = {"quantity": "expected", "multipliers": [1.0, 2.0], "spend_to_media": "proportional"}
    native = response_curves(
        model, results, spend_periods=[labels[2], labels[0]], response_periods=[labels[2]], **options
    )
    iso = response_curves(
        model, results, spend_periods=["2026-01-19", "2026-01-05"], response_periods=["2026-01-19"], **options
    )
    xr.testing.assert_identical(native, iso)
    np.testing.assert_allclose(iso["reference_spend"], [10.0])
    np.testing.assert_allclose(iso["response"][0, 0, 0], [40.0, 80.0])


def test_response_curves_resolve_period_labels_from_new_data():
    frame = pl.DataFrame({"week": [0, 1, 2], "views": [10.0, 20.0, 40.0], "cost": [2.0, 4.0, 8.0]})
    data = prepare_data(frame, time="week", media=["views"], spend=["cost"])
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
        scaling="auto",
    )
    results = _collect_results({"coefficient": jnp.ones((1, 1))})
    scenario = pl.DataFrame({"week": [5, 4], "views": [80.0, 60.0], "cost": [16.0, 12.0]})
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 1.0, 2.0],
        "spend_to_media": "proportional",
        "new_data": scenario,
    }
    curves = response_curves(model, results, spend_periods=[4], response_periods=[4, 5], **options)
    expected = model.scaling.transformations["media"].transform(jnp.array([[60.0], [80.0]]))[:, 0]
    np.testing.assert_allclose(curves["response"][0, 0, 0], expected[1] + expected[0] * np.array([0, 1, 2]), rtol=1e-6)
    np.testing.assert_allclose(curves["reference_spend"], [12.0])
    np.testing.assert_array_equal(curves["spend_period"], [4])
    np.testing.assert_array_equal(curves["response_period"], [4, 5])
    with pytest.raises(ValueError, match="spend_periods"):
        response_curves(model, results, spend_periods=[1], **options)


def test_response_curves_require_positive_selected_window_spending_only():
    data = _data()
    model = _model(data)
    results = _results(model, data)
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 1.0],
        "spend_to_media": "proportional",
        "spend_periods": [2],
    }
    with pytest.raises(ValueError, match="positive reference spending"):
        response_curves(model, results, channels=["Online video"], **options)
    curves = response_curves(model, results, channels=["Paid search"], **options)
    np.testing.assert_allclose(curves["reference_spend"], [5.0 / 3.0], rtol=1e-6)


def test_response_curves_ignore_unspendable_exposure_outside_spending_periods():
    data = _data()
    data.arrays["spend"][0, 0] = 0.0
    model = _model(data)
    curves = response_curves(
        model,
        _results(model, data),
        quantity="expected",
        multipliers=[0.0, 1.0],
        spend_to_media="proportional",
        spend_periods=[3],
    )
    assert np.isfinite(curves["response"]).all()
    np.testing.assert_allclose(curves["reference_spend"], data.arrays["spend"][2], rtol=1e-6)


@pytest.mark.parametrize("scaled", [False, True])
def test_budget_response_keeps_exact_stored_values_outside_spending_periods(scaled):
    data = _data(grouped=True)
    data.arrays["media"] = data.arrays["media"] + 0.137 if scaled else data.arrays["media"].astype(np.int64)
    scaling = fit_data_scaling(data, adjust_population=True) if scaled else None
    model = _model(data, scaling=scaling)
    spend = jnp.asarray(data.arrays["spend"], dtype=model.data.values["spend"].dtype)
    mask = jnp.array([True, False, False])
    budgets = spend[0].sum(axis=0)
    evaluator = _BudgetResponse(
        model=model,
        inputs=model.data,
        posterior={"coefficient": jnp.asarray(_results(model, data)["posterior"]["coefficient"].values)},
        quantity="expected",
        spend_weights=jnp.where(mask[:, None, None], spend / budgets, 0),
        convert=lambda candidate: candidate * jnp.array([2.0, 3.0]) + 7.0,
        observation_shape=spend.shape[:-1],
        batch_size=4,
        spend_mask=mask,
        reference_spend=spend,
        response_indices=jnp.array([1, 2]),
    )
    scenario, valid = jax.jit(evaluator._scenario_inputs)(2.0 * budgets)
    assert valid
    # History and both later periods retain their stored values without an inverse-scaling round trip.
    np.testing.assert_array_equal(
        np.asarray(scenario.values["media"])[[0, 2, 3]], np.asarray(model.data.values["media"])[[0, 2, 3]]
    )
    np.testing.assert_array_equal(scenario.values["spend"][1:], model.data.values["spend"][1:])
    for name in model.data.values.keys() - {"media", "spend"}:
        np.testing.assert_array_equal(scenario.values[name], model.data.values[name])
    assert not np.array_equal(scenario.values["media"][1], model.data.values["media"][1])


def test_budget_response_differentiates_carryover_with_a_separate_response_window():
    data = _data()

    def transformed(media, coefficient):
        return {"expected": (media[1:] + 0.5 * media[:-1]) @ coefficient}

    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    posterior = {"coefficient": jnp.asarray(_results(model, data)["posterior"]["coefficient"].values)}
    spend = jnp.asarray(data.arrays["spend"], dtype=model.data.values["spend"].dtype)
    mask = jnp.array([True, False, False])
    budgets = spend[0]
    evaluator = _BudgetResponse(
        model=model,
        inputs=model.data,
        posterior=posterior,
        quantity="expected",
        spend_weights=jnp.where(mask[:, None], spend / budgets, 0),
        convert=lambda candidate: candidate * jnp.array([2.0, 3.0]),
        observation_shape=spend.shape[:-1],
        batch_size=4,
        spend_mask=mask,
        reference_spend=spend,
        response_indices=jnp.array([1]),
    )
    candidate = budgets * jnp.array([1.2, 0.8])
    value, gradient = jax.jit(jax.value_and_grad(lambda allocation: evaluator(allocation).mean()))(candidate)
    coefficient = np.asarray(posterior["coefficient"]).mean(axis=(0, 1))
    expected_gradient = 0.5 * np.array([2.0, 3.0]) * coefficient
    expected_value = (data.arrays["media"][2] @ coefficient) + (np.asarray(candidate) @ expected_gradient)
    np.testing.assert_allclose(value, expected_value, rtol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)
    assert np.all(np.asarray(gradient) > 0)
