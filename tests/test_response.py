"""Tests for prior and posterior response curves with explicit spending assumptions."""

from dataclasses import replace
from datetime import date, datetime

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

import mmmjax
from mmmjax import (
    Data,
    Interval,
    Model,
    Positive,
    Real,
    fit_data_scaling,
    frequency_curves,
    geometric_adstock,
    hill_saturation,
    prepare_data,
    response_curves,
    sample_prior,
)
from mmmjax._results import _collect_results
from mmmjax.response import _BudgetResponse, _prepare_response


def _data(*, grouped=False, multiplier=1.0, compound=False, reverse=False):
    rows = []
    media = np.array([[8.0, 12.0], [2.0, 3.0], [0.0, 5.0], [7.0, 2.0]])
    for time, exposure in enumerate(media):
        for group in range(3 if compound else 2 if grouped else 1):
            exposure = media[time] * (group + 1)
            if time:
                exposure = exposure * multiplier
            rows.append(
                {
                    "week": time,
                    "region": ("east", "west", "east")[group],
                    "store": ("retail", "online", "online")[group],
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
    if reverse:
        frame = frame.reverse()
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region", "store"] if compound else ["region"] if grouped else (),
        population="residents",
        outcome="sales",
        media=["search", "video"] if reverse else ["video", "search"],
        spend=["search_spend", "video_spend"] if reverse else ["video_spend", "search_spend"],
        channels=["Paid search", "Online video"] if reverse else ["Online video", "Paid search"],
        controls=["temperature"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _model(data, *, scaling=None, cross_group=False):
    def transformed(media, spend, controls, coefficient):
        carried = media[1:] + 0.5 * media[:-1]
        predictor = (
            10.0
            + controls[..., 0]
            + carried @ coefficient
            + 0.03 * carried[..., 0] * carried[..., 1]
            + 0.2 * spend.sum(axis=-1)
        )
        if cross_group:
            predictor = predictor + 0.04 * carried[..., 0].sum(axis=1, keepdims=True) * carried[..., 1]
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
        data=Data(data, scaling=scaling),
        transformed_parameters=transformed,
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
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        spend_to_media="proportional",
        group="posterior",
    )

    xr.testing.assert_identical(curves, explicit)
    for by in (None, (), []):
        xr.testing.assert_identical(
            curves, response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], by=by)
        )


@pytest.mark.parametrize("new_data", [False, True])
def test_response_curves_retain_fixed_inputs_during_budget_changes(new_data):
    data = _data()
    inputs = xr.Dataset(
        {"weights": (("experiment", "channel"), [[2.0, 3.0], [4.0, 5.0]])},
        coords={"experiment": ["first", "second"], "channel": list(data.channels)},
    )

    def transformed(media, spend, weights, coefficient):
        return {"expected": media[-spend.shape[0] :] @ (coefficient * weights.mean(axis=0))}

    model = Model(
        {"coefficient": Real(dims="channel")},
        lambda expected: -expected.sum(),
        data=Data(data, inputs=inputs),
        transformed_parameters=transformed,
    )
    results = _results(model, data)
    reference = _data(multiplier=1.5) if new_data else data
    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        new_data=reference if new_data else None,
    )

    coefficients = results["posterior"]["coefficient"].values
    for channel in range(2):
        for index, multiplier in enumerate([0.0, 1.0, 2.0]):
            media = reference.arrays["media"][-len(reference.time_values) :].copy()
            media[:, channel] *= multiplier
            expected = (coefficients * np.array([3.0, 4.0]) * media.sum(axis=0)).sum(axis=-1)
            np.testing.assert_allclose(curves["response"][:, :, channel, index], expected, rtol=1e-6)

    context = _prepare_response(model, results, quantity="expected", spend_to_media="proportional")
    gradient = jax.jit(jax.grad(lambda budgets: context.evaluator(budgets).mean()))(context.reference_spend)
    np.testing.assert_allclose(gradient, coefficients.mean(axis=(0, 1)) * [3.0, 4.0] * [2.0, 3.0], rtol=1e-6)
    np.testing.assert_array_equal(model.data.values["weights"], inputs["weights"])


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

    def transformed(
        media,
        spend,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
    ):
        carried = geometric_adstock(media, alpha=paid_media_retention, max_lag=1, normalize=False)
        response = hill_saturation(carried, half_saturation=paid_media_half_saturation, slope=paid_media_slope)[
            -spend.shape[0] :
        ]
        return {"expected": jnp.exp(response @ paid_media_coefficient)}

    model = Model(
        {
            "paid_media_coefficient": Positive(dims="channel"),
            "paid_media_retention": Interval(0.0, 1.0, dims="channel"),
            "paid_media_half_saturation": Positive(dims="channel"),
            "paid_media_slope": Positive(dims="channel"),
        },
        lambda expected: expected.sum(),
        data=data,
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
        samples=posterior,
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
        data=Data(data, scaling="auto"),
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
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


def _window_expected(data, coefficient, periods, *, scaling=None, by=(), cross_group=False):
    values = (scaling.transform(data) if scaling is not None else data).arrays
    carried = values["media"][1:] + 0.5 * values["media"][:-1]
    predictor = (
        10.0
        + values["controls"][..., 0]
        + carried @ coefficient
        + 0.03 * carried[..., 0] * carried[..., 1]
        + 0.2 * values["spend"].sum(axis=-1)
    )
    if cross_group:
        predictor = predictor + 0.04 * carried[..., 0].sum(axis=1, keepdims=True) * carried[..., 1]
    indices = [data.time_values.index(period) for period in periods]
    expected = np.exp(0.01 * predictor)[indices]
    axes = tuple(index for index in range(expected.ndim) if ("time", "group")[index] not in by)
    return expected.sum(axis=axes)


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


@pytest.mark.parametrize("by", ["time", "group", ("time", "group"), ["group", "time"]])
def test_response_curves_breakdowns_preserve_global_interventions_and_posterior_draws(by):
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, adjust_population=True, scale_outcome=True)
    model = _model(data, scaling=scaling, cross_group=True)
    results = _results(model, data)
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 0.5, 1.0, 2.0],
        "spend_periods": [3, 1],
        "response_periods": [3, 2],
    }
    detailed = response_curves(model, results, by=by, batch_size=4, **options)
    aggregate = response_curves(model, results, **options)
    retained = tuple(name for name in ("time", "group") if name in by)

    assert detailed["response"].dims == ("chain", "draw", *retained, "channel", "multiplier")
    assert detailed["incremental_response"].dims == detailed["response"].dims
    assert detailed["reference_response"].dims == ("chain", "draw", *retained)
    assert detailed["spend"].dims == ("channel", "multiplier")
    assert detailed["reference_spend"].dims == ("channel",)
    np.testing.assert_array_equal(detailed.chain, [4, 8])
    np.testing.assert_array_equal(detailed.draw, [10, 20, 30])
    np.testing.assert_array_equal(detailed.spend_period, [1, 3])
    np.testing.assert_array_equal(detailed.response_period, [2, 3])
    if "time" in retained:
        np.testing.assert_array_equal(detailed.time, [2, 3])
    if "group" in retained:
        np.testing.assert_array_equal(detailed.group, ["east", "west"])

    for name in ("response", "incremental_response", "reference_response"):
        xr.testing.assert_allclose(detailed[name].sum(retained), aggregate[name], rtol=3e-5, atol=2e-6)
    for name in ("spend", "reference_spend"):
        xr.testing.assert_identical(detailed[name], aggregate[name])

    for chain, draw in np.ndindex(2, 3):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        expected_reference = _window_expected(data, coefficient, [2, 3], scaling=scaling, by=retained, cross_group=True)
        np.testing.assert_allclose(detailed["reference_response"][chain, draw], expected_reference, rtol=3e-6)
        for channel in range(2):
            zero = _window_expected(
                _window_scenario(data, channel, 0.0, [1, 3]),
                coefficient,
                [2, 3],
                scaling=scaling,
                by=retained,
                cross_group=True,
            )
            for multiplier in options["multipliers"]:
                expected = _window_expected(
                    _window_scenario(data, channel, multiplier, [1, 3]),
                    coefficient,
                    [2, 3],
                    scaling=scaling,
                    by=retained,
                    cross_group=True,
                )
                actual = detailed.sel(multiplier=multiplier).isel(chain=chain, draw=draw, channel=channel)
                np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
                np.testing.assert_allclose(actual["incremental_response"], expected - zero, atol=2e-6)

    if by == "time":
        unbatched = response_curves(model, results, by=by, **options)
        xr.testing.assert_allclose(detailed, unbatched, rtol=3e-5, atol=2e-6)


def test_response_curves_national_time_breakdown_preserves_selected_labels():
    data = _data()
    model = _model(data)
    results = _results(model, data).isel(chain=[1], draw=[2, 0])
    results = xr.DataTree.from_dict({"posterior": results["posterior"].to_dataset().assign_coords(group=["unrelated"])})
    curves = response_curves(
        model,
        results,
        quantity="expected",
        by="time",
        multipliers=[2.0, 0.0, 1.0],
        channels=["Paid search", "Online video"],
        response_periods=[3, 1],
        batch_size=3,
    )

    assert curves["response"].dims == ("chain", "draw", "time", "channel", "multiplier")
    assert "group" not in curves.coords
    np.testing.assert_array_equal(curves.chain, [8])
    np.testing.assert_array_equal(curves.draw, [30, 10])
    np.testing.assert_array_equal(curves.time, [1, 3])
    np.testing.assert_array_equal(curves.channel, ["Paid search", "Online video"])
    np.testing.assert_array_equal(curves.multiplier, [2.0, 0.0, 1.0])
    for channel, label in enumerate(data.channels):
        for draw in range(2):
            coefficient = results["posterior"]["coefficient"].values[0, draw]
            expected = _window_expected(_scenario(data, channel, 2.0), coefficient, [1, 3], by=("time",))
            actual = curves["response"].sel(channel=label, multiplier=2).isel(chain=0, draw=draw)
            np.testing.assert_allclose(actual, expected, rtol=3e-6)

    with pytest.raises(ValueError, match="group"):
        response_curves(model, results, quantity="expected", multipliers=[1.0], by="group")


@pytest.mark.parametrize("compound", [False, True])
def test_response_curves_breakdown_aligns_new_data_groups_and_channels_with_fitted_scaling(compound):
    data = _data(grouped=True, compound=compound)
    scaling = fit_data_scaling(data, adjust_population=True)
    model = _model(data, scaling=scaling, cross_group=True)
    results = _results(model, data)
    new_data = _data(grouped=True, compound=compound, multiplier=2.0, reverse=True)
    expected_data = _data(grouped=True, compound=compound, multiplier=2.0)
    assert new_data.group_values == data.group_values[::-1]
    assert new_data.channels == data.channels[::-1]

    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        by=("group", "time"),
        new_data=new_data,
        batch_size=4,
    )
    assert curves["response"].dims == ("chain", "draw", "time", "group", "channel", "multiplier")
    np.testing.assert_array_equal(curves.time, expected_data.time_values)
    np.testing.assert_array_equal(curves.channel, expected_data.channels)
    if compound:
        np.testing.assert_array_equal(curves.group, [0, 1, 2])
        np.testing.assert_array_equal(curves.group_region, ["east", "west", "east"])
        np.testing.assert_array_equal(curves.group_store, ["retail", "online", "online"])
        assert curves.group_region.dims == curves.group_store.dims == ("group",)
    else:
        np.testing.assert_array_equal(curves.group, ["east", "west"])
        assert "group_region" not in curves.coords

    for channel in range(2):
        coefficient = results["posterior"]["coefficient"].values[0, 0]
        expected = _window_expected(
            _scenario(expected_data, channel, 2.0),
            coefficient,
            [1, 2, 3],
            scaling=scaling,
            by=("time", "group"),
            cross_group=True,
        )
        actual = curves["response"].sel(multiplier=2).isel(chain=0, draw=0, channel=channel)
        np.testing.assert_allclose(actual, expected, rtol=3e-6)
    np.testing.assert_allclose(curves["reference_spend"], expected_data.arrays["spend"].sum(axis=(0, 1)), rtol=2e-6)


@pytest.mark.parametrize(
    "by",
    ["region", "", ("time", "time"), ("time", 1), ("time", ["group"]), 1, b"time", {"time"}, {"time": True}],
)
def test_response_curves_reject_invalid_breakdown_axes(by):
    data = _data(grouped=True)
    model = _model(data)
    with pytest.raises(ValueError, match="by"):
        response_curves(model, _results(model, data), quantity="expected", multipliers=[1.0], by=by)


def test_response_curves_measure_carryover_after_spending_has_ended():
    data = _data()

    def transformed(media, coefficient):
        return {"expected": (media[1:] + 0.5 * media[:-1]) @ coefficient}

    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=data,
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

    detailed = response_curves(model, results, response_periods=[2, 3], by="time", **options)
    np.testing.assert_allclose(detailed["incremental_response"].sel(time=2), curves["incremental_response"], rtol=1e-6)
    np.testing.assert_array_equal(detailed["incremental_response"].sel(time=3), 0.0)
    assert data.arrays["spend"][1, 0] == 0.0


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
        data=Data(data, scaling="auto"),
        transformed_parameters=lambda media, coefficient: {"expected": media[:, 0] * coefficient},
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
        samples={"coefficient": jnp.asarray(_results(model, data)["posterior"]["coefficient"].values)},
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
        samples=posterior,
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


def _rf_data(*, mixed=False, grouped=False):
    rows = []
    for week in range(4):
        for group in range(2 if grouped else 1):
            rows.append(
                {
                    "week": week,
                    "region": ("east", "west")[group],
                    "audience_video": (3.0 + week) * (group + 1),
                    "audience_audio": (8.0 - week) * (group + 1),
                    "frequency_video": 1.0 + week,
                    "frequency_audio": 2.0 + 0.5 * week,
                    "cost_video": (2.0 + week) * (group + 1),
                    "cost_audio": (4.0 + week) * (group + 1),
                    "search": (2.0 + week) * (group + 1),
                    "cost_search": (1.0 + week) * (group + 1),
                    "population": 100.0 + 200 * group,
                    "sales": 100.0 + 10 * week,
                }
            )
    frame = pl.DataFrame(rows)
    options = {"media": ["search"], "spend": ["cost_search"], "channels": ["Search"]} if mixed else {}
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region"] if grouped else (),
        outcome="sales",
        population="population",
        reach=["audience_video", "audience_audio"],
        media_frequency=["frequency_video", "frequency_audio"],
        rf_spend=["cost_video", "cost_audio"],
        rf_channels=["Video", "Audio"],
        media_history=frame.filter(pl.col("week") == 0),
        **options,
    )


def _rf_model(data, *, scaling=None):
    def expected(reach, media_frequency, coefficient, rf_spend=None, media=None, spend=None):
        exposure = reach * media_frequency / (1.0 + media_frequency)
        carried = (exposure[1:] + 0.5 * exposure[:-1]) @ jnp.array([1.0, 2.0])
        expected = 5.0 + coefficient**2 * carried
        if rf_spend is not None:
            expected = expected + 0.1 * rf_spend.sum(axis=-1)
        if media is not None:
            expected = expected + coefficient * media[1:, ..., 0] * (1.0 + 0.1 * carried)
        if spend is not None:
            expected = expected + 0.2 * spend.sum(axis=-1)
        return {"expected": expected}

    if "media" not in data.arrays:

        def transformed(reach, media_frequency, rf_spend, coefficient):
            return expected(reach, media_frequency, coefficient, rf_spend=rf_spend)

    elif "spend" not in data.arrays:

        def transformed(reach, media_frequency, rf_spend, media, coefficient):
            return expected(reach, media_frequency, coefficient, rf_spend=rf_spend, media=media)

    elif "rf_spend" not in data.arrays:

        def transformed(reach, media_frequency, media, spend, coefficient):
            return expected(reach, media_frequency, coefficient, media=media, spend=spend)

    else:

        def transformed(reach, media_frequency, rf_spend, media, spend, coefficient):
            return expected(reach, media_frequency, coefficient, rf_spend=rf_spend, media=media, spend=spend)

    def density(expected):
        raise AssertionError("Response curves must only evaluate transformed quantities")

    return Model({"coefficient": Real()}, density, data=Data(data, scaling=scaling), transformed_parameters=transformed)


def _rf_results():
    return _collect_results(
        {"coefficient": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)},
        coords={"chain": [4, 8], "draw": [10, 30]},
    )


def _rf_expected(data, coefficient, *, scaling=None, response_periods=(1, 2, 3), by=()):
    values = (scaling.transform(data) if scaling is not None else data).arrays
    exposure = values["reach"] * values["media_frequency"] / (1.0 + values["media_frequency"])
    carried = (exposure[1:] + 0.5 * exposure[:-1]) @ np.array([1.0, 2.0])
    expected = 5.0 + coefficient**2 * carried
    if "rf_spend" in values:
        expected = expected + 0.1 * values["rf_spend"].sum(axis=-1)
    if "media" in values:
        expected = expected + coefficient * values["media"][1:, ..., 0] * (1.0 + 0.1 * carried)
    if "spend" in values:
        expected = expected + 0.2 * values["spend"].sum(axis=-1)
    expected = expected[[data.time_values.index(period) for period in response_periods]]
    axes = tuple(index for index in range(expected.ndim) if ("time", "group")[index] not in by)
    return expected.sum(axis=axes)


def _rf_scenario(data, channel, multiplier, *, mode="reach", spend_periods=(1, 2, 3), media_conversion=None):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    periods = np.array([data.time_values.index(period) for period in spend_periods])
    if channel in data.rf_channels:
        index = data.rf_channels.index(channel)
        arrays["rf_spend"][periods, ..., index] *= multiplier
        if not callable(mode):
            role = "reach" if mode == "reach" else "media_frequency"
            arrays[role][1 + periods, ..., index] *= multiplier
    elif channel is not None:
        index = data.channels.index(channel)
        arrays["spend"][periods, ..., index] *= multiplier
        if media_conversion is None:
            arrays["media"][1 + periods, ..., index] *= multiplier
    if callable(mode):
        reach, frequency = mode(arrays["rf_spend"])
        arrays["reach"][1 + periods] = np.asarray(reach)[periods]
        arrays["media_frequency"][1 + periods] = np.asarray(frequency)[periods]
    if media_conversion is not None:
        arrays["media"][1 + periods] = np.asarray(media_conversion(arrays["spend"]))[periods]
    return replace(data, arrays=arrays)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_response_curves_support_rf_channels_with_explicit_spending_assumptions(mixed, grouped, mode):
    data = _rf_data(mixed=mixed, grouped=grouped)
    model, results = _rf_model(data), _rf_results()
    original = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], spend_to_rf=mode, batch_size=3
    )
    labels = [*data.channels, *data.rf_channels]
    np.testing.assert_array_equal(curves.channel, labels)
    np.testing.assert_array_equal(curves.channel_type, (["media"] if mixed else []) + ["reach_frequency"] * 2)
    np.testing.assert_array_equal(curves.chain, [4, 8])
    np.testing.assert_array_equal(curves.draw, [10, 30])
    assert curves.attrs["spend_to_rf"] == mode
    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        np.testing.assert_allclose(
            curves["reference_response"][chain, draw], _rf_expected(data, coefficient), rtol=2e-6
        )
        for channel in labels:
            zero = _rf_expected(_rf_scenario(data, channel, 0.0, mode=mode), coefficient)
            for multiplier in [0.0, 1.0, 2.0]:
                expected = _rf_expected(_rf_scenario(data, channel, multiplier, mode=mode), coefficient)
                actual = curves.sel(channel=channel, multiplier=multiplier).isel(chain=chain, draw=draw)
                np.testing.assert_allclose(actual["response"], expected, rtol=2e-6)
                np.testing.assert_allclose(actual["incremental_response"], expected - zero, rtol=3e-6, atol=3e-5)
    for name, value in original.items():
        np.testing.assert_array_equal(model.data.values[name], value)


def test_response_curves_default_rf_assumption_and_channel_order():
    data = _rf_data(mixed=True)
    model, results = _rf_model(data), _rf_results()
    default = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0])
    explicit = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0], spend_to_rf="reach")
    xr.testing.assert_identical(default, explicit)
    selected = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0], channels=["Audio", "Search", "Video"]
    )
    xr.testing.assert_allclose(selected, default.sel(channel=["Audio", "Search", "Video"]))
    single = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0], channels=["Video"])
    xr.testing.assert_allclose(single, default.sel(channel=["Video"]))


@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_response_curves_breakdown_preserves_mixed_rf_channel_order_and_spending_windows(mode):
    data = _rf_data(mixed=True, grouped=True)
    model, results = _rf_model(data), _rf_results()
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 1.0, 2.0],
        "channels": ["Audio", "Search", "Video"],
        "spend_to_rf": mode,
        "spend_periods": [1],
        "response_periods": [2, 3],
    }
    curves = response_curves(model, results, by=("time", "group"), batch_size=3, **options)
    aggregate = response_curves(model, results, **options)

    np.testing.assert_array_equal(curves.channel, ["Audio", "Search", "Video"])
    np.testing.assert_array_equal(curves.channel_type, ["reach_frequency", "media", "reach_frequency"])
    np.testing.assert_array_equal(curves.time, [2, 3])
    np.testing.assert_array_equal(curves.group, ["east", "west"])
    for name in ("response", "incremental_response", "reference_response"):
        xr.testing.assert_allclose(curves[name].sum(("time", "group")), aggregate[name], rtol=3e-6, atol=3e-5)
    for name in ("spend", "reference_spend"):
        xr.testing.assert_identical(curves[name], aggregate[name])

    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        for channel in curves.channel.values:
            zero = _rf_expected(
                _rf_scenario(data, channel, 0.0, mode=mode, spend_periods=[1]),
                coefficient,
                response_periods=[2, 3],
                by=("time", "group"),
            )
            for multiplier in options["multipliers"]:
                expected = _rf_expected(
                    _rf_scenario(data, channel, multiplier, mode=mode, spend_periods=[1]),
                    coefficient,
                    response_periods=[2, 3],
                    by=("time", "group"),
                )
                actual = curves.sel(channel=channel, multiplier=multiplier).isel(chain=chain, draw=draw)
                np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
                np.testing.assert_allclose(actual["incremental_response"], expected - zero, rtol=3e-6, atol=3e-5)
    np.testing.assert_array_equal(curves["incremental_response"].sel(time=3), 0.0)


@pytest.mark.parametrize("already_scaled", [False, True])
def test_response_curves_rf_reuse_population_scaling_and_fixed_history_windows(already_scaled):
    data = _rf_data(mixed=True, grouped=True)
    scaling = fit_data_scaling(data, adjust_population=True, scale_outcome=True)
    model = _rf_model(scaling.transform(data) if already_scaled else data, scaling=scaling)
    results = _rf_results()
    options = {"spend_periods": [1], "response_periods": [2, 3], "spend_to_rf": "frequency"}
    curves = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], **options)
    np.testing.assert_allclose(
        curves["reference_spend"], np.concatenate((data.arrays["spend"][0].sum(0), data.arrays["rf_spend"][0].sum(0)))
    )
    for channel in curves.channel.values:
        scenario = _rf_scenario(data, channel, 2.0, mode="frequency", spend_periods=[1])
        for chain, draw in np.ndindex(2, 2):
            expected = _rf_expected(
                scenario,
                results["posterior"]["coefficient"].values[chain, draw],
                scaling=scaling,
                response_periods=[2, 3],
            )
            np.testing.assert_allclose(
                curves["response"].sel(channel=channel, multiplier=2).isel(chain=chain, draw=draw), expected, rtol=3e-6
            )
    assert np.all(curves["incremental_response"].sel(channel=["Video", "Audio"], multiplier=1) > 0)
    context = _prepare_response(model, results, quantity="expected", spend_to_media="proportional", **options)
    scenario, valid = jax.jit(context.evaluator._scenario_inputs)(context.reference_spend * 2)
    assert valid
    for name in ["media", "reach", "media_frequency"]:
        np.testing.assert_array_equal(
            np.asarray(scenario.values[name])[[0, 2, 3]], np.asarray(model.data.values[name])[[0, 2, 3]]
        )
    for name in ["spend", "rf_spend"]:
        np.testing.assert_array_equal(scenario.values[name][1:], model.data.values[name][1:])


def test_response_curves_custom_rf_conversion_receives_raw_family_spend_and_preserves_windows():
    data = _rf_data(mixed=True, grouped=True)
    model = _rf_model(data, scaling="auto")
    results = _rf_results()

    def rf_conversion(spend):
        assert spend.shape == data.arrays["rf_spend"].shape
        return spend * jnp.array([3.0, 4.0]), jnp.ones_like(spend) * jnp.array([2.0, 3.0])

    def media_conversion(spend):
        assert spend.shape == data.arrays["spend"].shape
        return 2.5 * spend

    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0, 2.0],
        channels=["Audio", "Search"],
        spend_to_media=media_conversion,
        spend_to_rf=rf_conversion,
        spend_periods=[3, 1],
        response_periods=[2, 3],
    )
    assert curves.attrs["spend_to_rf"] == "custom"
    for channel in curves.channel.values:
        for multiplier in [0.0, 1.0, 2.0]:
            scenario = _rf_scenario(
                data, channel, multiplier, mode=rf_conversion, media_conversion=media_conversion, spend_periods=[1, 3]
            )
            expected = _rf_expected(scenario, 0.5, scaling=model.scaling, response_periods=[2, 3])
            np.testing.assert_allclose(
                curves["response"].sel(channel=channel, multiplier=multiplier).isel(chain=0, draw=0),
                expected,
                rtol=3e-6,
            )


@pytest.mark.parametrize(
    "conversion",
    [
        "automatic",
        lambda spend: spend,
        lambda spend: (spend.sum(), spend),
        lambda spend: (spend, spend[..., :1]),
        lambda spend: (-spend, jnp.ones_like(spend)),
        lambda spend: (spend, -jnp.ones_like(spend)),
        lambda spend: (spend * jnp.nan, jnp.ones_like(spend)),
        lambda spend: (spend, jnp.full_like(spend, jnp.inf)),
    ],
)
def test_response_curves_reject_invalid_rf_conversions(conversion):
    data = _rf_data()
    with pytest.raises((TypeError, ValueError), match=r"reach|frequency|shape|finite|invalid"):
        response_curves(_rf_model(data), _rf_results(), quantity="expected", multipliers=[1.0], spend_to_rf=conversion)


@pytest.mark.parametrize("mode", ["reach", "frequency"])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(
    "channel, unchanged_spend", [("Search", "rf_spend"), ("Audio", "rf_spend"), ("Audio", "spend")]
)
def test_response_curves_preserve_unselected_exposures_with_zero_spend(channel, unchanged_spend, scaled, mode):
    data = _rf_data(mixed=True, grouped=True)
    data.arrays[unchanged_spend][0, ..., 0] = 0.0
    scaling = fit_data_scaling(data, adjust_population=True) if scaled else None
    model, results = _rf_model(data, scaling=scaling), _rf_results()
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], channels=[channel], spend_to_rf=mode
    )

    for multiplier in [0.0, 1.0, 2.0]:
        scenario = _rf_scenario(data, channel, multiplier, mode=mode)
        expected = _rf_expected(scenario, 0.5, scaling=scaling)
        np.testing.assert_allclose(
            curves["response"].sel(channel=channel, multiplier=multiplier).isel(chain=0, draw=0),
            expected,
            rtol=3e-6,
        )

    context = _prepare_response(
        model, results, quantity="expected", channels=[channel], spend_to_media="proportional", spend_to_rf=mode
    )
    budgets = context.reference_spend.at[context.indices].multiply(2.0)
    scenario, valid = jax.jit(context.evaluator._scenario_inputs)(budgets)
    assert valid
    for roles, labels in [
        (("media", "spend"), data.channels),
        (("reach", "media_frequency", "rf_spend"), data.rf_channels),
    ]:
        unchanged = [index for index, name in enumerate(labels) if name != channel]
        for role in roles:
            np.testing.assert_array_equal(
                scenario.values[role][..., unchanged], model.data.values[role][..., unchanged]
            )


@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_response_curves_rf_positive_exposure_at_zero_spend_needs_custom_conversion(mode):
    data = _rf_data()
    data.arrays["rf_spend"][0, 0] = 0.0
    model, results = _rf_model(data), _rf_results()
    with pytest.raises(ValueError, match="zero spend"):
        response_curves(model, results, quantity="expected", multipliers=[1.0], spend_to_rf=mode)
    curves = response_curves(
        model,
        results,
        quantity="expected",
        multipliers=[0.0, 1.0],
        spend_to_rf=lambda spend: (2.0 * spend, jnp.ones_like(spend)),
    )
    assert np.isfinite(curves["response"]).all()
    outside = response_curves(
        model, results, quantity="expected", multipliers=[1.0], spend_periods=[3], spend_to_rf=mode
    )
    assert np.isfinite(outside["response"]).all()


def test_response_curves_rf_zero_spend_cells_and_unselected_zero_budget_channels():
    data = _rf_data()
    data.arrays["rf_spend"][:, 0] = 0.0
    data.arrays["reach"][1:, 0] = 0.0
    data.arrays["rf_spend"][0, 1] = 0.0
    data.arrays["reach"][1, 1] = 0.0
    model, results = _rf_model(data), _rf_results()
    with pytest.raises(ValueError, match="positive reference spending"):
        response_curves(model, results, quantity="expected", multipliers=[1.0])
    curves = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], channels=["Audio"])
    np.testing.assert_allclose(curves["reference_spend"], [13.0])
    assert np.isfinite(curves["response"]).all()


@pytest.mark.parametrize("unpriced", ["spend", "rf_spend"])
def test_response_curves_keep_unpriced_input_families_fixed(unpriced):
    data = _rf_data(mixed=True)
    data = replace(data, arrays={name: value for name, value in data.arrays.items() if name != unpriced})
    model, results = _rf_model(data), _rf_results()
    curves = response_curves(model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0])
    labels = ["Video", "Audio"] if unpriced == "spend" else ["Search"]
    np.testing.assert_array_equal(curves.channel, labels)
    for channel in labels:
        expected = _rf_expected(_rf_scenario(data, channel, 2.0), 0.5)
        np.testing.assert_allclose(
            curves["response"].sel(channel=channel, multiplier=2).isel(chain=0, draw=0), expected
        )


def test_response_curves_reject_ambiguous_labels_across_priced_input_families():
    data = replace(_rf_data(mixed=True), channels=("Video",))
    with pytest.raises(ValueError, match=r"channel|unique|ambiguous|duplicate"):
        response_curves(_rf_model(data), _rf_results(), quantity="expected", multipliers=[1.0])


def test_response_curves_rf_new_data_uses_training_scales_and_raw_reference_spend():
    training = _rf_data(grouped=True)
    model, results = _rf_model(training, scaling="auto"), _rf_results()
    arrays = {name: value.copy() for name, value in training.arrays.items()}
    arrays["reach"][1:] *= 3.0
    arrays["rf_spend"] *= 2.0
    new_data = replace(training, arrays=arrays)
    curves = response_curves(
        model, results, quantity="expected", multipliers=[0.0, 1.0, 2.0], new_data=new_data, channels=["Audio"]
    )
    np.testing.assert_allclose(curves["reference_spend"], [new_data.arrays["rf_spend"][..., 1].sum()])
    for multiplier in [0.0, 1.0, 2.0]:
        scenario = _rf_scenario(new_data, "Audio", multiplier)
        expected = _rf_expected(scenario, 0.5, scaling=model.scaling)
        np.testing.assert_allclose(
            curves["response"].sel(channel="Audio", multiplier=multiplier).isel(chain=0, draw=0), expected, rtol=3e-6
        )
    np.testing.assert_array_equal(model.data.values["rf_spend"], training.arrays["rf_spend"])


@pytest.mark.parametrize("mode", ["reach", "frequency"])
def test_response_curves_rf_integer_inputs_preserve_fractional_scenarios_and_gradients(mode):
    data = prepare_data(
        pl.DataFrame({"week": [0, 1], "audience": [2, 6], "frequency": [1, 3], "cost": [1.0, 3.0]}),
        time="week",
        reach=["audience"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
    )

    def transformed(reach, media_frequency, coefficient):
        return {"expected": coefficient * reach[:, 0] * media_frequency[:, 0] / (1.0 + media_frequency[:, 0])}

    model = Model(
        {"coefficient": Real()}, lambda expected: expected.sum(), data=data, transformed_parameters=transformed
    )
    results = _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)})
    context = _prepare_response(model, results, quantity="expected", spend_to_media="proportional", spend_to_rf=mode)
    value, gradient = jax.jit(jax.value_and_grad(lambda budget: context.evaluator(budget).mean()))(
        context.reference_spend * 0.5
    )
    reach = np.array([2.0, 6.0]) * (0.5 if mode == "reach" else 1.0)
    frequency = np.array([1.0, 3.0]) * (0.5 if mode == "frequency" else 1.0)
    expected = (reach * frequency / (1.0 + frequency)).sum()
    derivative = (
        np.array([2.0, 6.0]) / 4.0 * frequency / (1.0 + frequency)
        if mode == "reach"
        else reach / (1.0 + frequency) ** 2 * np.array([1.0, 3.0]) / 4.0
    ).sum()
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(gradient, [derivative], rtol=2e-6)
    scenario, valid = jax.jit(context.evaluator._scenario_inputs)(context.reference_spend * 0.5)
    assert valid
    np.testing.assert_allclose(scenario.values["reach"][:, 0], reach)
    np.testing.assert_allclose(scenario.values["media_frequency"][:, 0], frequency)
    np.testing.assert_array_equal(model.data.values["reach"][:, 0], [2, 6])
    np.testing.assert_array_equal(model.data.values["media_frequency"][:, 0], [1, 3])


def _frequency_scenario(data, channel, frequency, *, periods=(1, 2, 3)):
    scenario = _rf_scenario(data, None, 1.0)
    index = data.rf_channels.index(channel)
    history = len(data.media_time_values) - len(data.time_values)
    for period in periods:
        row = history + data.time_values.index(period)
        reach = data.arrays["reach"][row, ..., index]
        original_frequency = data.arrays["media_frequency"][row, ..., index]
        impressions = reach * original_frequency
        scenario.arrays["reach"][row, ..., index] = np.where(impressions > 0, impressions / frequency, reach)
        scenario.arrays["media_frequency"][row, ..., index] = np.where(impressions > 0, frequency, original_frequency)
    return scenario


def test_frequency_curves_are_public_and_evaluate_conditional_responses_per_draw():
    assert mmmjax.frequency_curves is frequency_curves
    data = _rf_data(mixed=True)
    model, results = _rf_model(data), _rf_results()
    frequencies = [4.0, 0.5, 2.0]
    curves = frequency_curves(model, results, quantity="expected", frequencies=frequencies)
    explicit = frequency_curves(model, results, quantity="expected", frequencies=frequencies, group="posterior")
    xr.testing.assert_identical(curves, explicit)
    assert curves.attrs["group"] == "posterior"

    assert set(curves.data_vars) == {
        "response",
        "response_change",
        "reference_response",
        "reference_spend",
        "best_frequency",
    }
    assert curves["response"].dims == ("chain", "draw", "channel", "frequency")
    assert curves["response_change"].dims == curves["response"].dims
    assert curves["reference_response"].dims == ("chain", "draw")
    assert curves["reference_spend"].dims == curves["best_frequency"].dims == ("channel",)
    for coordinate, expected in {
        "chain": [4, 8],
        "draw": [10, 30],
        "channel": ["Video", "Audio"],
        "frequency": frequencies,
        "channel_type": ["reach_frequency"] * 2,
        "frequency_period": [1, 2, 3],
        "response_period": [1, 2, 3],
    }.items():
        np.testing.assert_array_equal(curves[coordinate], expected)
    np.testing.assert_allclose(curves["reference_spend"], data.arrays["rf_spend"].sum(axis=0))
    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        reference = _rf_expected(data, coefficient)
        np.testing.assert_allclose(curves["reference_response"][chain, draw], reference, rtol=2e-6)
        for channel in data.rf_channels:
            for frequency in frequencies:
                expected = _rf_expected(_frequency_scenario(data, channel, frequency), coefficient)
                actual = curves.sel(channel=channel, frequency=frequency).isel(chain=chain, draw=draw)
                np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
                np.testing.assert_allclose(actual["response_change"], expected - reference, rtol=3e-6, atol=3e-5)
    selected = frequency_curves(
        model, results, quantity="expected", frequencies=frequencies, channels=["Audio", "Video"]
    )
    xr.testing.assert_allclose(selected, curves.sel(channel=["Audio", "Video"]))


@pytest.mark.parametrize("reference_kind", ["raw", "scaled", "new_data"])
def test_frequency_curves_reuse_fitted_population_scales_and_separate_windows(reference_kind):
    training = _rf_data(mixed=True, grouped=True)
    scaling = fit_data_scaling(training, adjust_population=True, scale_outcome=True)
    model = _rf_model(scaling.transform(training) if reference_kind == "scaled" else training, scaling=scaling)
    reference = training
    if reference_kind == "new_data":
        arrays = {name: value.copy() for name, value in training.arrays.items()}
        arrays["reach"][1:] *= 3.0
        arrays["rf_spend"] *= 2.0
        reference = replace(training, arrays=arrays)
    original = {name: np.asarray(value).copy() for name, value in model.data.values.items()}
    results = _rf_results()
    curves = frequency_curves(
        model,
        results,
        quantity="expected",
        frequencies=[0.25, 3.0],
        channels=["Audio"],
        periods=[1],
        response_periods=[3, 2],
        new_data=reference if reference_kind == "new_data" else None,
    )
    np.testing.assert_array_equal(curves["frequency_period"], [1])
    np.testing.assert_array_equal(curves["response_period"], [2, 3])
    np.testing.assert_allclose(curves["reference_spend"], [reference.arrays["rf_spend"][0, ..., 1].sum()])
    for chain, draw in np.ndindex(2, 2):
        coefficient = results["posterior"]["coefficient"].values[chain, draw]
        expected_reference = _rf_expected(reference, coefficient, scaling=scaling, response_periods=[2, 3])
        np.testing.assert_allclose(curves["reference_response"][chain, draw], expected_reference, rtol=3e-6)
        for frequency in [0.25, 3.0]:
            scenario = _frequency_scenario(reference, "Audio", frequency, periods=[1])
            expected = _rf_expected(scenario, coefficient, scaling=scaling, response_periods=[2, 3])
            actual = curves.sel(channel="Audio", frequency=frequency).isel(chain=chain, draw=draw)
            np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
            # The paired change can be much smaller than the observations evaluated in model precision.
            rounding = np.finfo(curves["response"].dtype).eps * abs(expected_reference)
            np.testing.assert_allclose(
                actual["response_change"], expected - expected_reference, rtol=3e-6, atol=rounding
            )
    for name, value in original.items():
        np.testing.assert_array_equal(model.data.values[name], value)


@pytest.mark.parametrize("half_saturation, best", [([[2.0, 2.0]], 2.0), ([[1.0, 7.0]], 1.0)])
def test_frequency_curves_recover_hill_optimum_and_average_posterior_responses(half_saturation, best):
    data = _rf_data()

    def transformed(reach, media_frequency, half_saturation):
        return {"expected": (reach * hill_saturation(media_frequency, half_saturation, 2.0))[1:, 0]}

    def density(expected):
        raise AssertionError("Frequency curves must not evaluate the density")

    def generated(key, expected):
        raise AssertionError("Frequency curves must not evaluate generated observations")

    model = Model({"half_saturation": Positive()}, density, generated, data=data, transformed_parameters=transformed)
    results = _collect_results({"half_saturation": np.asarray(half_saturation, dtype=np.float32)})
    frequencies = np.array([4.0, 1.0, 2.0, 7.0])
    curves = frequency_curves(model, results, quantity="expected", frequencies=frequencies, channels=["Video"])
    impressions = (data.arrays["reach"][1:, 0] * data.arrays["media_frequency"][1:, 0]).sum()
    # Holding I = reach * frequency fixed gives I*f/(f**2 + half**2), with optimum f = half.
    expected = impressions * frequencies / (frequencies**2 + np.asarray(half_saturation)[..., None] ** 2)
    np.testing.assert_allclose(curves["response"].sel(channel="Video"), expected, rtol=2e-6)
    assert curves["best_frequency"].sel(channel="Video").item() == best
    if best == 1.0:
        at_mean_parameters = impressions * frequencies / (frequencies**2 + np.mean(half_saturation) ** 2)
        assert frequencies[np.argmax(at_mean_parameters)] == 4.0
        assert not np.allclose(curves["response"].mean(("chain", "draw")), at_mean_parameters[None, :])


def test_frequency_curves_preserve_zero_impression_cells_and_all_fixed_inputs_exactly():
    data = _rf_data(mixed=True, grouped=True)
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    arrays["reach"][1, 0, 0] = 0.0
    arrays["media_frequency"][2, 1, 0] = 0.0
    arrays["rf_spend"][0, 1, 0] = 0.0  # Positive impressions at a zero-spend cell are valid here.
    arrays.update(
        organic_media=arrays["media"].copy(),
        organic_reach=arrays["reach"][..., :1].copy(),
        organic_frequency=arrays["media_frequency"][..., :1].copy(),
        controls=arrays["spend"].copy(),
    )
    data = replace(
        data,
        arrays=arrays,
        organic_channels=("Email",),
        organic_rf_channels=("Social",),
        columns={
            **data.columns,
            "organic_media": ("email",),
            "organic_reach": ("social_reach",),
            "organic_frequency": ("social_frequency",),
            "controls": ("temperature",),
        },
    )
    mutable = np.zeros_like(arrays["reach"], dtype=bool)
    mutable[1:3, ..., 0] = (arrays["reach"] * arrays["media_frequency"])[1:3, ..., 0] > 0
    fixed_names = ("rf_spend", "spend", "media", "organic_media", "organic_reach", "organic_frequency", "controls")

    def transformed(
        reach,
        media_frequency,
        rf_spend,
        spend,
        media,
        organic_media,
        organic_reach,
        organic_frequency,
        controls,
        coefficient,
    ):
        values = locals()
        valid = jnp.all(jnp.where(mutable, True, reach == arrays["reach"]))
        valid &= jnp.all(jnp.where(mutable, True, media_frequency == arrays["media_frequency"]))
        valid &= jnp.allclose(reach * media_frequency, arrays["reach"] * arrays["media_frequency"])
        for name in fixed_names:
            valid &= jnp.all(values[name] == arrays[name])
        exposure = reach * media_frequency / (1.0 + media_frequency)
        # The cross-channel interaction distinguishes conditional curves from joint interventions.
        expected = coefficient * exposure[1:, ..., 0] * exposure[1:, ..., 1]
        return {"expected": jnp.where(valid, expected, jnp.nan)}

    model = Model(
        {"coefficient": Real()}, lambda expected: expected.sum(), data=data, transformed_parameters=transformed
    )
    curves = frequency_curves(
        model, _rf_results(), quantity="expected", frequencies=[0.125, 2.5], channels=["Video"], periods=[1, 2]
    )
    for frequency in [0.125, 2.5]:
        scenario = _frequency_scenario(data, "Video", frequency, periods=[1, 2])
        exposure = (
            scenario.arrays["reach"] * scenario.arrays["media_frequency"] / (1 + scenario.arrays["media_frequency"])
        )
        expected = 0.5 * (exposure[1:, ..., 0] * exposure[1:, ..., 1]).sum()
        np.testing.assert_allclose(
            curves["response"].sel(channel="Video", frequency=frequency).isel(chain=0, draw=0), expected, rtol=2e-6
        )
    for name, original in arrays.items():
        np.testing.assert_array_equal(model.data.values[name], original)


def test_frequency_curves_constant_frequency_reference_and_batching():
    data = _rf_data(mixed=True)
    data.arrays["media_frequency"][1:, 0] = 2.0
    model, results = _rf_model(data), _rf_results()
    options = {"quantity": "expected", "frequencies": [4.0, 2.0, 1.0], "channels": ["Video"]}
    single = frequency_curves(model, results, batch_size=1, **options)
    for batch_size in [3, 64]:
        batched = frequency_curves(model, results, batch_size=batch_size, **options)
        xr.testing.assert_allclose(single, batched, rtol=2e-6, atol=2e-5)
    rounding = np.finfo(single["response"].dtype).eps * float(abs(single["reference_response"]).max())
    np.testing.assert_allclose(single["response_change"].sel(frequency=2.0), 0.0, rtol=0, atol=rounding)
    np.testing.assert_allclose(
        single["response"].sel(channel="Video", frequency=2.0), single["reference_response"], rtol=0, atol=rounding
    )


def test_frequency_curves_exact_ties_choose_first_supplied_frequency():
    data = _rf_data()
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        transformed_parameters=lambda rf_spend, coefficient: {"expected": coefficient * rf_spend.sum(axis=-1)},
    )
    curves = frequency_curves(model, _rf_results(), quantity="expected", frequencies=[7.0, 1.0, 3.0])
    np.testing.assert_array_equal(curves["response_change"], 0.0)
    np.testing.assert_array_equal(curves["best_frequency"], [7.0, 7.0])


def test_frequency_curves_subtract_paired_observations_before_aggregation():
    data = prepare_data(
        pl.DataFrame({"week": np.arange(100), "reach": np.ones(100), "frequency": np.ones(100), "cost": np.ones(100)}),
        time="week",
        reach=["reach"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
    )
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        transformed_parameters=lambda media_frequency, coefficient: {
            "expected": 1_000_000.0 + coefficient * media_frequency[:, 0]
        },
    )
    curves = frequency_curves(
        model,
        _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)}),
        quantity="expected",
        frequencies=[1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(curves["response_change"][0, 0, 0], [0.0, 100.0, 200.0], rtol=1e-6)


def test_frequency_curves_preserve_date_labels_and_fractional_reach_from_integer_inputs():
    dates = [date(2026, 1, day) for day in [1, 2, 3]]
    data = prepare_data(
        pl.DataFrame({"week": dates, "reach": [2, 6, 10], "frequency": [1, 3, 5], "cost": [1.0, 3.0, 5.0]}),
        time="week",
        reach=["reach"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
    )
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        transformed_parameters=lambda reach, media_frequency, coefficient: {
            "expected": coefficient * (reach * media_frequency / (1 + media_frequency))[:, 0]
        },
    )
    curves = frequency_curves(
        model,
        _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)}),
        quantity="expected",
        frequencies=[4.0, 2.0],
        periods=[dates[1]],
        response_periods=[dates[2], dates[1]],
    )
    np.testing.assert_array_equal(curves["frequency_period"], np.asarray(dates[1:2], dtype="datetime64[D]"))
    np.testing.assert_array_equal(curves["response_period"], np.asarray(dates[1:], dtype="datetime64[D]"))
    np.testing.assert_allclose(curves["reference_spend"], [3.0])
    np.testing.assert_allclose(curves["response"][0, 0, 0], 18.0 / np.array([5.0, 3.0]) + 50.0 / 6.0)
    np.testing.assert_array_equal(model.data.values["reach"][:, 0], [2, 6, 10])


@pytest.fixture
def integer_frequency_data():
    data = _rf_data()
    for role in ("reach", "media_frequency"):
        data.arrays[role] = data.arrays[role].astype(np.int64)
    return data


def _integer_frequency_model(data):
    return Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=data,
        transformed_parameters=lambda reach, media_frequency, coefficient: {
            "expected": coefficient * reach[1:, 1] + coefficient * media_frequency[1:, 1]
        },
    )


@pytest.mark.parametrize(
    "role, location, new_data",
    [
        ("reach", "selected", False),
        ("media_frequency", "selected", False),
        ("reach", "unselected", False),
        ("media_frequency", "history", False),
        ("reach", "outside_periods", False),
        ("reach", "zero_impressions", False),
        ("media_frequency", "zero_impressions", False),
        ("reach", "history", True),
        ("media_frequency", "unselected", True),
    ],
)
def test_frequency_curves_reject_integer_precision_loss(integer_frequency_data, role, location, new_data):
    data = integer_frequency_data
    with jax.enable_x64(False):
        training_model = _integer_frequency_model(data) if new_data else None
        row, channel = {"history": (0, 0), "unselected": (1, 1), "outside_periods": (3, 0)}.get(location, (1, 0))
        data.arrays[role][row, channel] = 2**24 + 1
        if location == "zero_impressions":
            other_role = "media_frequency" if role == "reach" else "reach"
            data.arrays[other_role][row, channel] = 0
        model = training_model if new_data else _integer_frequency_model(data)
        with pytest.raises(ValueError, match=r"precision|64-bit"):
            frequency_curves(
                model,
                _rf_results(),
                quantity="expected",
                frequencies=[2.0, 4.0],
                channels=["Video"],
                periods=[1, 2],
                response_periods=[1, 2],
                new_data=data if new_data else None,
            )


@pytest.mark.parametrize("role", ["reach", "media_frequency"])
def test_frequency_curves_preserve_large_integer_inputs_with_x64(integer_frequency_data, role):
    data = integer_frequency_data
    data.arrays[role][:, 1] = 2**24 + 1
    with jax.enable_x64(True):
        model = _integer_frequency_model(data)
        results = _collect_results({"coefficient": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)})
        curves = frequency_curves(model, results, quantity="expected", frequencies=[2.0, 4.0], channels=["Video"])
        expected = results["posterior"]["coefficient"].values.astype(np.float64) * (
            data.arrays["reach"][1:, 1].sum() + data.arrays["media_frequency"][1:, 1].sum()
        )
        np.testing.assert_array_equal(curves["reference_response"], expected)
        np.testing.assert_array_equal(curves["response_change"], 0.0)
        for frequency in [2.0, 4.0]:
            np.testing.assert_array_equal(curves["response"].sel(channel="Video", frequency=frequency), expected)
        np.testing.assert_array_equal(model.data.values[role], data.arrays[role])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_frequency_curves_reject_integer_boundaries_rounded_out_of_range(integer_frequency_data, dtype):
    data = integer_frequency_data
    data.arrays["reach"] = data.arrays["reach"].astype(dtype)
    data.arrays["reach"][0, 1] = np.iinfo(dtype).max
    with jax.enable_x64(True):
        model = _integer_frequency_model(data)
        with pytest.raises(ValueError, match=r"precision|64-bit"):
            frequency_curves(model, _rf_results(), quantity="expected", frequencies=[2.0], channels=["Video"])


def test_frequency_curves_evaluate_reference_with_original_integer_arithmetic(integer_frequency_data):
    data = integer_frequency_data
    data.arrays["reach"][:, 1] = 2**24
    data.arrays["media_frequency"][:, 1] = 1
    with jax.enable_x64(False):
        model = Model(
            {"coefficient": Real()},
            lambda coefficient: -(coefficient**2),
            data=data,
            transformed_parameters=lambda reach, media_frequency, coefficient: {
                "expected": coefficient * ((reach[1:, 1] + media_frequency[1:, 1]) % 3)
            },
        )
        results = _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)})
        curves = frequency_curves(model, results, quantity="expected", frequencies=[2.0], channels=["Video"])
    # Every input is exactly representable, but converting before the addition changes the remainder.
    np.testing.assert_array_equal(curves["reference_response"], [[6.0]])


@pytest.mark.parametrize(
    "frequencies", [[], [0.0], [-1.0], [np.nan], [np.inf], [[1.0, 2.0]], [True], [1.0, 1.0], [1j], ["2"]]
)
def test_frequency_curves_reject_invalid_frequency_grids(frequencies):
    data = _rf_data()
    with pytest.raises((TypeError, ValueError), match="frequenc"):
        frequency_curves(_rf_model(data), _rf_results(), quantity="expected", frequencies=frequencies)


@pytest.mark.parametrize("channels", [[], ["Search"], ["Unknown"], ["Video", "Video"], "Video"])
def test_frequency_curves_reject_invalid_or_non_rf_channel_selections(channels):
    data = _rf_data(mixed=True)
    with pytest.raises((TypeError, ValueError), match=r"channel|reach.frequency"):
        frequency_curves(_rf_model(data), _rf_results(), quantity="expected", frequencies=[1.0], channels=channels)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_frequency_curves_reject_invalid_batch_sizes(batch_size):
    data = _rf_data()
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        frequency_curves(_rf_model(data), _rf_results(), quantity="expected", frequencies=[1.0], batch_size=batch_size)


@pytest.mark.parametrize("quantity", [None, "", "missing", "aggregate", "exposure"])
def test_frequency_curves_require_an_observation_shaped_transformed_quantity(quantity):
    data = _rf_data()

    def transformed(reach, coefficient):
        return {"expected": coefficient * reach[1:, 0], "aggregate": reach.sum(), "exposure": reach}

    model = Model(
        {"coefficient": Real()}, lambda expected: expected.sum(), data=data, transformed_parameters=transformed
    )
    with pytest.raises((TypeError, ValueError), match=r"quantity|shape"):
        frequency_curves(model, _rf_results(), quantity=quantity, frequencies=[1.0])


@pytest.mark.parametrize("missing", ["rf_spend", "impressions", "selected_spend"])
def test_frequency_curves_require_paired_spend_and_positive_selected_totals(missing):
    data = _rf_data(mixed=True)
    if missing == "rf_spend":
        data = replace(data, arrays={name: value for name, value in data.arrays.items() if name != "rf_spend"})
    elif missing == "impressions":
        data.arrays["reach"][1, 0] = 0.0
    else:
        data.arrays["rf_spend"][0, 0] = 0.0
    model = _rf_model(data)
    with pytest.raises(ValueError, match=r"spend|impression|positive"):
        frequency_curves(model, _rf_results(), quantity="expected", frequencies=[1.0], channels=["Video"], periods=[1])
    if missing != "rf_spend":
        curves = frequency_curves(
            model, _rf_results(), quantity="expected", frequencies=[1.0], channels=["Audio"], periods=[1]
        )
        assert np.isfinite(curves["response"]).all()


def test_frequency_curves_require_reach_frequency_inputs():
    data = _data()
    model = _model(data)
    with pytest.raises(ValueError, match=r"reach|frequency|channel"):
        frequency_curves(model, _results(model, data), quantity="expected", frequencies=[1.0])


@pytest.mark.parametrize("scenario", ["raw_underflow", "scaled_overflow"])
def test_frequency_curves_reject_unrepresentable_exposures(scenario):
    precision = np.finfo(np.float64 if jax.config.jax_enable_x64 else np.float32)
    reach = 10 * float(precision.tiny) if scenario == "raw_underflow" else 1e-5
    frequency = 1.0 if scenario == "raw_underflow" else 1e10
    candidate = float(precision.max) / 2 if scenario == "raw_underflow" else float(precision.tiny) * 1e8
    data = prepare_data(
        pl.DataFrame({"time": [0], "reach": [reach], "frequency": [frequency], "cost": [1.0]}),
        time="time",
        reach=["reach"],
        media_frequency=["frequency"],
        rf_spend=["cost"],
    )
    model = Model(
        {"coefficient": Real()},
        lambda coefficient: -(coefficient**2),
        data=Data(data, scaling=fit_data_scaling(data) if scenario == "scaled_overflow" else None),
        transformed_parameters=lambda reach, media_frequency, coefficient: {
            "expected": coefficient * (jnp.tanh(reach) * media_frequency)[:, 0]
        },
    )
    with pytest.raises(ValueError, match="invalid exposures"):
        frequency_curves(
            model,
            _collect_results({"coefficient": np.ones((1, 1), dtype=precision.dtype)}),
            quantity="expected",
            frequencies=[candidate],
        )


@pytest.mark.parametrize("kind", ["spending", "frequency"])
@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_response_analysis_selects_independent_parameter_groups(kind, group):
    data = _rf_data(mixed=True, grouped=True)
    scaling = fit_data_scaling(data, adjust_population=True)
    model = _rf_model(data, scaling=scaling)
    prior = _collect_results(
        {"coefficient": np.array([[0.3, 0.9, 1.8]], dtype=np.float32)},
        sample_group="prior",
        coords={"chain": [42], "draw": [9, 3, 7]},
    )
    results = xr.DataTree.from_dict(
        {"prior": prior["prior"].to_dataset(), "posterior": _rf_results()["posterior"].to_dataset()}
    )
    original = results.copy(deep=True)
    options = {"quantity": "expected", "group": group, "channels": ["Audio", "Video"], "response_periods": [2, 3]}
    if kind == "spending":
        curves = response_curves(
            model, results, multipliers=[0.0, 1.0, 2.0], spend_periods=[1], batch_size=2, **options
        )
    else:
        curves = frequency_curves(model, results, frequencies=[0.5, 2.0, 4.0], periods=[1], batch_size=2, **options)

    draws = results[group]["coefficient"]
    assert curves.attrs["group"] == group
    np.testing.assert_array_equal(curves.chain, draws.chain)
    np.testing.assert_array_equal(curves.draw, draws.draw)
    for chain, draw in np.ndindex(draws.shape):
        coefficient = draws.values[chain, draw]
        reference = _rf_expected(data, coefficient, scaling=scaling, response_periods=[2, 3])
        np.testing.assert_allclose(curves["reference_response"][chain, draw], reference, rtol=3e-6)
        for channel in curves.channel.values:
            if kind == "spending":
                zero = _rf_expected(
                    _rf_scenario(data, channel, 0.0, spend_periods=[1]),
                    coefficient,
                    scaling=scaling,
                    response_periods=[2, 3],
                )
                for multiplier in curves.multiplier.values:
                    expected = _rf_expected(
                        _rf_scenario(data, channel, multiplier, spend_periods=[1]),
                        coefficient,
                        scaling=scaling,
                        response_periods=[2, 3],
                    )
                    actual = curves.sel(channel=channel, multiplier=multiplier).isel(chain=chain, draw=draw)
                    np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
                    np.testing.assert_allclose(actual["incremental_response"], expected - zero, rtol=3e-6, atol=3e-5)
            else:
                for frequency in curves.frequency.values:
                    expected = _rf_expected(
                        _frequency_scenario(data, channel, frequency, periods=[1]),
                        coefficient,
                        scaling=scaling,
                        response_periods=[2, 3],
                    )
                    actual = curves.sel(channel=channel, frequency=frequency).isel(chain=chain, draw=draw)
                    np.testing.assert_allclose(actual["response"], expected, rtol=3e-6)
                    np.testing.assert_allclose(actual["response_change"], expected - reference, rtol=3e-6, atol=3e-5)
    if kind == "frequency":
        best = curves["response"].mean(("chain", "draw")).argmax("frequency")
        np.testing.assert_array_equal(curves["best_frequency"], curves.frequency.values[best.values])
    if group == "prior":
        if kind == "spending":
            prior_only = response_curves(
                model, prior, multipliers=[0.0, 1.0, 2.0], spend_periods=[1], batch_size=2, **options
            )
        else:
            prior_only = frequency_curves(
                model, prior, frequencies=[0.5, 2.0, 4.0], periods=[1], batch_size=2, **options
            )
        xr.testing.assert_identical(curves, prior_only)
    xr.testing.assert_identical(results, original)


def test_response_curves_use_sampled_priors_without_generation_or_resampling():
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, adjust_population=True)
    model = _model(data, scaling=scaling, cross_group=True)
    allow_sampling = True

    def prior(key):
        if not allow_sampling:
            raise AssertionError("Response analysis must reuse the supplied draws")
        return {"coefficient": jax.random.uniform(key, shape=(2,), minval=0.2, maxval=2.0)}

    results = sample_prior(model, prior, draws=5, seed=42, generate=False, batch_size=3)
    allow_sampling = False
    assert "posterior" not in results.children
    assert "prior_generated_quantities" not in results.children
    original = results.copy(deep=True)
    options = {
        "quantity": "expected",
        "multipliers": [0.0, 0.5, 2.0],
        "group": "prior",
        "spend_periods": [3, 1],
        "response_periods": [3, 2],
    }
    detailed = response_curves(model, results, by=("group", "time"), batch_size=3, **options)
    aggregate = response_curves(model, results, batch_size=1, **options)
    assert detailed["response"].dims == ("chain", "draw", "time", "group", "channel", "multiplier")
    assert detailed.attrs["group"] == "prior"
    np.testing.assert_array_equal(detailed.time, [2, 3])
    np.testing.assert_array_equal(detailed.group, ["east", "west"])
    for name in ("response", "incremental_response", "reference_response"):
        xr.testing.assert_allclose(detailed[name].sum(("time", "group")), aggregate[name], rtol=3e-5, atol=3e-6)
    for draw in range(5):
        coefficient = results["prior"]["coefficient"].values[0, draw]
        for channel in range(2):
            expected = _window_expected(
                _window_scenario(data, channel, 2.0, [1, 3]),
                coefficient,
                [2, 3],
                scaling=scaling,
                by=("time", "group"),
                cross_group=True,
            )
            actual = detailed["response"].sel(multiplier=2).isel(chain=0, draw=draw, channel=channel)
            np.testing.assert_allclose(actual, expected, rtol=3e-6)
    xr.testing.assert_identical(results, original)


@pytest.mark.parametrize("kind", ["spending", "frequency"])
@pytest.mark.parametrize("group", [None, "Prior", "prior_predictive", 1, ["prior"]])
def test_response_analysis_rejects_invalid_result_groups(kind, group):
    data = _rf_data()
    model = _rf_model(data)
    function, grid = (
        (response_curves, {"multipliers": [1.0]}) if kind == "spending" else (frequency_curves, {"frequencies": [1.0]})
    )
    with pytest.raises(ValueError, match="group"):
        function(model, _rf_results(), quantity="expected", group=group, **grid)


@pytest.mark.parametrize("kind", ["spending", "frequency"])
def test_response_analysis_requires_the_selected_group_without_falling_back(kind):
    data = _rf_data()
    model = _rf_model(data)
    function, grid = (
        (response_curves, {"multipliers": [1.0]}) if kind == "spending" else (frequency_curves, {"frequencies": [1.0]})
    )
    with pytest.raises(ValueError, match="prior"):
        function(model, _rf_results(), quantity="expected", group="prior", **grid)
    prior = _collect_results({"coefficient": np.ones((1, 1), dtype=np.float32)}, sample_group="prior")
    with pytest.raises(ValueError, match="posterior"):
        function(model, prior, quantity="expected", **grid)


@pytest.mark.parametrize("invalid", ["shape", "labels", "nonfinite", "precision"])
def test_response_curves_validate_prior_parameter_shape_labels_and_values(invalid):
    data = _data()
    model = _model(data)
    values = np.array([[[0.5, 1.0]]], dtype=np.float32)
    labels = list(data.channels)
    if invalid == "shape":
        values = values[..., :1]
        labels = labels[:1]
    elif invalid == "labels":
        labels.reverse()
    elif invalid == "nonfinite":
        values[0, 0, 0] = np.nan
    else:
        values = values.astype(np.float64)
    results = xr.DataTree.from_dict(
        {"prior": xr.Dataset({"coefficient": (("chain", "draw", "channel"), values)}, coords={"channel": labels})}
    )
    if invalid == "precision" and jax.config.jax_enable_x64:
        curves = response_curves(model, results, quantity="expected", group="prior", multipliers=[1.0])
        assert curves["response"].dtype == np.dtype(np.float64)
        np.testing.assert_allclose(curves["reference_response"], _expected(data, values[0, 0]), rtol=1e-12)
        return
    with pytest.raises(ValueError, match=r"shape|coordinate|finite|64-bit"):
        response_curves(model, results, quantity="expected", group="prior", multipliers=[1.0])
