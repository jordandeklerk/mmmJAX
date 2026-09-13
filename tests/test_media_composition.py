"""Tests for composing learned media and seasonal effects in public models."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import (
    Interval,
    Model,
    Positive,
    Real,
    beta,
    fourier_features,
    generate_quantities,
    geometric_adstock,
    half_normal,
    hill_saturation,
    media_response,
    normal,
    normal_rng,
    prepare_data,
    reach_frequency_response,
    sample_prior,
)
from mmmjax._results import _collect_results

_POSITIVE = ("coefficient", "half_saturation", "slope")


@pytest.fixture
def media_values():
    return np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
            [0.0, 2.0],
            [4.0, 1.0],
            [2.0, 3.0],
        ]
    )


@pytest.fixture
def reach_values():
    return np.array(
        [
            [45.0, 12.0],
            [4.0, 25.0],
            [0.0, 35.0],
            [18.0, 6.0],
            [9.0, 40.0],
        ]
    )


@pytest.fixture
def frequency_values():
    return np.array(
        [
            [1.0, 3.0],
            [2.0, 1.0],
            [0.0, 4.0],
            [3.0, 2.0],
            [1.0, 5.0],
        ]
    )


def _data(media, *, periods=3, start=0, reverse=False, observed=True):
    grouped = media.ndim == 3
    groups = [1, 0] if grouped and reverse else ([0, 1] if grouped else [0])
    rows = []
    for time in range(len(media)):
        for group in groups:
            exposure = media[time, group] if grouped else media[time]
            row = {
                "time": start + time,
                "video": exposure[0],
                "search": exposure[1],
                "sales": 0.3 + time / 5 + group,
                "temperature": (start + time) / 10 + group,
                "price": (start + time) / 5 - group,
                "video_cost": exposure[0] / 2,
                "search_cost": exposure[1] / 2,
            }
            if grouped:
                row["region"] = ("west", "east")[group]
            rows.append(row)
    history = (len(media) - periods) * len(groups)
    channels = ["search", "video"] if reverse else ["video", "search"]
    controls = ["price", "temperature"] if reverse else ["temperature", "price"]
    return prepare_data(
        pl.DataFrame(rows[history:]),
        time="time",
        groups=["region"] if grouped else [],
        media=channels,
        controls=controls,
        outcome="sales" if observed else None,
        spend=[channel + "_cost" for channel in channels] if observed else None,
        media_history=pl.DataFrame(rows[:history]) if history else None,
    )


def _prior(coefficient, retention, half_saturation, slope, annual_coefficients):
    return (
        half_normal(coefficient, 1.5)
        + beta(retention, 1.0, 3.0)
        + half_normal(half_saturation, 1.5)
        + half_normal(slope, 1.5)
        + normal(annual_coefficients, 0.0, 1.0)
    )


def _model(data, *, keyword_only_generation=True, grouped=False, include_priors=True):
    def density(
        outcome,
        paid_media,
        annual,
        intercept,
        sigma,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
        annual_coefficients,
    ):
        target = normal(outcome, intercept + paid_media.sum(-1) + annual, sigma) + normal(intercept, 0.0, 2.0)
        if include_priors:
            target += _prior(
                paid_media_coefficient,
                paid_media_retention,
                paid_media_half_saturation,
                paid_media_slope,
                annual_coefficients,
            )
        return target

    def quantities(key, *, paid_media, annual, controls, intercept, sigma):
        mean = intercept + paid_media.sum(-1) + annual
        return {
            "mean": mean,
            "paid_media": paid_media,
            "annual": annual,
            "controls": controls,
            "draw": normal_rng(key, mean, sigma),
        }

    def ordinary_quantities(key, controls, sigma, paid_media, intercept, annual):
        return quantities(
            key, controls=controls, sigma=sigma, paid_media=paid_media, intercept=intercept, annual=annual
        )

    def transform(
        media,
        time,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
        annual_coefficients,
    ):
        response = media_response(
            media,
            adstock=lambda exposure: geometric_adstock(exposure, alpha=paid_media_retention, max_lag=2),
            saturation=lambda exposure: hill_saturation(
                exposure,
                half_saturation=paid_media_half_saturation,
                slope=paid_media_slope,
            ),
            n_periods=time.shape[0],
        )
        annual = fourier_features(time, period=8.0, order=1) @ annual_coefficients
        return {"paid_media": response * paid_media_coefficient, "annual": annual}

    return Model(
        {
            "intercept": Real(),
            "sigma": Positive(),
            "paid_media_coefficient": Positive(shape=(2, 2) if grouped else (2,)),
            "paid_media_retention": Interval(0.0, 1.0, shape=(2,)),
            "paid_media_half_saturation": Positive(shape=(2,)),
            "paid_media_slope": Positive(shape=(2,)),
            "annual_coefficients": Real(shape=(2, 2) if grouped else (2,)),
        },
        density,
        quantities if keyword_only_generation else ordinary_quantities,
        data=data,
        transformed_parameters=transform,
    )


def _position(*, grouped=False):
    coefficient = [[0.7, 1.1], [0.5, 1.3]] if grouped else [0.7, 1.1]
    return {
        "paid_media_coefficient": jnp.log(jnp.asarray(coefficient)),
        "paid_media_retention": jnp.log(jnp.array([0.25, 0.6]) / jnp.array([0.75, 0.4])),
        "paid_media_half_saturation": jnp.log(jnp.array([1.5, 2.5])),
        "paid_media_slope": jnp.log(jnp.array([1.2, 0.8])),
        "annual_coefficients": jnp.array([[0.3, -0.2], [0.7, 1.1]]) if grouped else jnp.array([0.3, -0.2]),
        "intercept": jnp.array(0.25),
        "sigma": jnp.log(jnp.array(0.7)),
    }


def _normal(value, scale=1.0):
    return (-0.5 * (np.asarray(value, dtype=np.float64) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def _reference(position, media, times):
    position = {name: np.asarray(value, dtype=np.float64) for name, value in position.items()}
    parameters = {
        name: np.exp(value) if name == "sigma" or any(name == "paid_media_" + role for role in _POSITIVE) else value
        for name, value in position.items()
    }
    retention = 1 / (1 + np.exp(-position["paid_media_retention"]))
    weights = retention ** np.arange(3)[:, None]
    weights /= weights.sum(axis=0)
    carried = np.zeros_like(media, dtype=np.float64)
    for time in range(len(media)):
        for lag in range(min(time, 2) + 1):
            carried[time] += weights[lag] * media[time - lag]
    half, slope = parameters["paid_media_half_saturation"], parameters["paid_media_slope"]
    powered = carried**slope
    paid = (powered / (powered + half**slope))[-len(times) :] * parameters["paid_media_coefficient"]
    angle = 2 * np.pi * (np.asarray(times, dtype=np.float64) - 2) / 8
    annual = np.column_stack((np.sin(angle), np.cos(angle))) @ position["annual_coefficients"]
    mean = position["intercept"] + paid.sum(axis=-1) + annual
    prior = _normal(position["annual_coefficients"]) + _normal(position["intercept"], 2.0)
    prior += (np.log(3) + 2 * np.log1p(-retention)).sum()
    adjustment = position["sigma"] + (np.log(retention) + np.log1p(-retention)).sum()
    for role in _POSITIVE:
        value = parameters["paid_media_" + role]
        prior += _normal(value, 1.5) + value.size * np.log(2)
        adjustment += position["paid_media_" + role].sum()
    return mean, paid, annual, prior + adjustment, parameters["sigma"]


@pytest.mark.parametrize("keyword_only_generation", [False, True])
def test_explicit_media_and_seasonality_match_full_density_gradients_and_generated_breakdowns(
    keyword_only_generation, media_values
):
    data = _data(media_values)
    model, position = _model(data, keyword_only_generation=keyword_only_generation), _position()
    mean, paid, annual, prior, sigma = _reference(position, media_values, [2, 3, 4])
    outcome = np.asarray(data.arrays["outcome"], dtype=np.float64)

    def density_reference(values):
        expected_mean, _, _, expected_prior, scale = _reference(values, media_values, [2, 3, 4])
        return _normal(outcome - expected_mean, scale) + expected_prior

    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    assert set(model.parameters) == set(position)
    assert model.parameters["paid_media_coefficient"].shape == (2,)
    assert model.parameters["annual_coefficients"].shape == (2,)
    np.testing.assert_allclose(value, _normal(outcome - mean, sigma) + prior, rtol=4e-6, atol=3e-6)
    values64 = {name: np.asarray(value, dtype=np.float64) for name, value in position.items()}
    for name, values in values64.items():
        expected = np.empty_like(values)
        for index in np.ndindex(values.shape):
            delta = np.zeros_like(values)
            delta[index] = 1e-4
            expected[index] = (
                density_reference({**values64, name: values + delta})
                - density_reference({**values64, name: values - delta})
            ) / 2e-4
        np.testing.assert_allclose(gradient[name], expected, rtol=4e-5, atol=5e-6)
    key = jax.random.key(7)
    constrained = model.constrain(position)
    generated = jax.jit(model.generate)(key, constrained, model.data)
    np.testing.assert_allclose(generated["paid_media"], paid, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(generated["annual"], annual, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(generated["mean"], mean, rtol=4e-6, atol=2e-6)
    noise = np.asarray(jax.random.normal(key, mean.shape, dtype=constrained["sigma"].dtype))
    np.testing.assert_allclose(generated["draw"], mean + sigma * noise, rtol=4e-6, atol=2e-6)


def test_posterior_vmap_evaluates_dynamic_prediction_media_and_parameters(media_values):
    model = _model(_data(media_values))
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.1)), _position())
    evaluate = jax.jit(jax.vmap(model.log_density, in_axes=(0, None)))
    generate = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))
    keys = jax.random.split(jax.random.key(3), 2)
    constrained = jax.vmap(model.constrain)(draws)
    for multiplier in (0.6, 1.4):
        future_media = media_values * multiplier
        future = _data(future_media, start=5)
        inputs = model.prepare_data(future)
        actual_density, actual = evaluate(draws, inputs), generate(keys, constrained, inputs)
        for index in range(2):
            position = {name: value[index] for name, value in draws.items()}
            mean, paid, _, prior, sigma = _reference(position, future_media, [7, 8, 9])
            expected = _normal(future.arrays["outcome"] - mean, sigma) + prior
            np.testing.assert_allclose(actual_density[index], expected, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(actual["mean"][index], mean, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(actual["paid_media"][index], paid, rtol=5e-6, atol=3e-6)


def test_grouped_prediction_aligns_history_channels_controls_and_preserves_float_precision(media_values):
    media = np.stack((media_values, media_values * 1.5 + 0.2), axis=1)
    with jax.enable_x64(False):
        model = _model(_data(media), grouped=True)
        position = _position(grouped=True)
        parameters = model.constrain(position)
    future_media = media[:4] * 0.8
    future = _data(future_media, periods=2, start=6, reverse=True, observed=False)
    with jax.enable_x64(True):
        generated = jax.jit(model.generate)(jax.random.key(0), parameters, model.prepare_data(future))
    mean, paid, annual, _, _ = _reference(position, future_media, [8, 9])
    controls = np.array([[[0.8, 1.6], [1.8, 0.6]], [[0.9, 1.8], [1.9, 0.8]]])

    assert generated["paid_media"].shape == (2, 2, 2)
    assert generated["annual"].shape == (2, 2)
    assert future.group_values == (("east",), ("west",))
    assert "outcome" not in future.arrays and "spend" not in future.arrays
    assert all(value.dtype == jnp.float32 for value in generated.values())
    for name, expected in (("mean", mean), ("paid_media", paid), ("annual", annual), ("controls", controls)):
        np.testing.assert_allclose(generated[name], expected, rtol=5e-6, atol=3e-6)


def test_media_model_snapshots_training_arrays_and_retains_layout_after_source_mutation(media_values):
    data = _data(media_values)
    model, position = _model(data), _position()
    parameters = model.constrain(position)
    data.arrays["media"][:] = 99
    data.arrays.clear()
    data.columns.clear()
    training = jax.jit(model.generate)(jax.random.key(0), parameters, model.data)
    np.testing.assert_allclose(training["mean"], _reference(position, media_values, [2, 3, 4])[0], rtol=4e-6, atol=2e-6)
    future_media = media_values * 0.7
    future = _data(future_media, start=5, observed=False)
    inputs = model.prepare_data(future)
    future.arrays["media"][:] = 99
    forecast = jax.jit(model.generate)(jax.random.key(0), parameters, inputs)
    np.testing.assert_allclose(forecast["mean"], _reference(position, future_media, [7, 8, 9])[0], rtol=4e-6, atol=2e-6)


def _rf_data(
    reach,
    frequency,
    *,
    media=None,
    periods=3,
    start=0,
    reverse=False,
    observed=True,
    frequency_columns=("tv_frequency", "social_frequency"),
):
    grouped = reach.ndim == 3
    groups = ([1, 0] if reverse else [0, 1]) if grouped else [0]
    rows = []
    for time in range(len(reach)):
        for group in groups:
            reached = reach[time, group] if grouped else reach[time]
            repeated = frequency[time, group] if grouped else frequency[time]
            row = {"time": start + time, "sales": 1.0 + time / 10 + group}
            for index, channel in enumerate(("tv", "social")):
                row[channel + "_reach"] = reached[index]
                row[frequency_columns[index]] = repeated[index]
                row[channel + "_cost"] = reached[index] * repeated[index] / 10
            if media is not None:
                row["video"] = media[time, group, 0] if grouped else media[time, 0]
            if grouped:
                row["region"] = ("west", "east")[group]
            rows.append(row)
    history = (len(reach) - periods) * len(groups)
    order = [1, 0] if reverse else [0, 1]
    channels = [("tv", "social")[index] for index in order]
    return prepare_data(
        pl.DataFrame(rows[history:]),
        time="time",
        groups=["region"] if grouped else [],
        outcome="sales" if observed else None,
        media=["video"] if media is not None else None,
        reach=[channel + "_reach" for channel in channels],
        media_frequency=[frequency_columns[index] for index in order],
        rf_channels=channels,
        rf_spend=[channel + "_cost" for channel in channels] if observed else None,
        media_history=pl.DataFrame(rows[:history]) if history else None,
    )


def _rf_parameters(*, mixed=False, grouped=False):
    values = {
        "intercept": jnp.array(0.25),
        "paid_rf_coefficient": jnp.array([[0.07, 0.12], [0.11, 0.05]] if grouped else [0.07, 0.12]),
        "paid_rf_retention": jnp.array([0.25, 0.6]),
        "paid_rf_half_saturation": jnp.array([1.5, 2.5]),
        "paid_rf_slope": jnp.array([1.2, 0.8]),
    }
    if mixed:
        values.update(
            paid_media_coefficient=jnp.array([[0.4], [0.7]] if grouped else [0.4]),
            paid_media_retention=jnp.array([0.3]),
            paid_media_half_saturation=jnp.array([1.2]),
            paid_media_slope=jnp.array([1.4]),
        )
    return values


def _rf_model(data, *, grouped=False, scaling=None, include_priors=True):
    def density(
        outcome,
        mean,
        intercept,
        paid_rf_coefficient,
        paid_rf_retention,
        paid_rf_half_saturation,
        paid_rf_slope,
    ):
        value = normal(outcome, mean, 0.7) + normal(intercept, 0.0, 2.0)
        if include_priors:
            value += half_normal(paid_rf_coefficient, 1.5) + beta(paid_rf_retention, 1.0, 3.0)
            value += half_normal(paid_rf_half_saturation, 1.5) + half_normal(paid_rf_slope, 1.5)
        return value

    def rf_quantities(
        reach,
        media_frequency,
        time,
        paid_rf_coefficient,
        paid_rf_retention,
        paid_rf_half_saturation,
        paid_rf_slope,
        intercept,
    ):
        response = reach_frequency_response(
            reach,
            media_frequency,
            adstock=lambda exposure: geometric_adstock(exposure, alpha=paid_rf_retention, max_lag=2),
            saturation=lambda frequency: hill_saturation(
                frequency,
                half_saturation=paid_rf_half_saturation,
                slope=paid_rf_slope,
            ),
            n_periods=time.shape[0],
        )
        paid_rf = response * paid_rf_coefficient
        return {"paid_rf": paid_rf, "paid_rf_total": paid_rf.sum(-1), "mean": intercept + paid_rf.sum(-1)}

    def mixed_quantities(
        reach,
        media_frequency,
        time,
        paid_rf_coefficient,
        paid_rf_retention,
        paid_rf_half_saturation,
        paid_rf_slope,
        intercept,
        media,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
    ):
        quantities = rf_quantities(
            reach,
            media_frequency,
            time,
            paid_rf_coefficient,
            paid_rf_retention,
            paid_rf_half_saturation,
            paid_rf_slope,
            intercept,
        )
        response = media_response(
            media,
            adstock=lambda exposure: geometric_adstock(exposure, alpha=paid_media_retention, max_lag=2),
            saturation=lambda exposure: hill_saturation(
                exposure,
                half_saturation=paid_media_half_saturation,
                slope=paid_media_slope,
            ),
            n_periods=time.shape[0],
        )
        paid_media = response * paid_media_coefficient
        return {
            **quantities,
            "paid_media": paid_media,
            "paid_media_total": paid_media.sum(-1),
            "mean": quantities["mean"] + paid_media.sum(-1),
        }

    def generate(key, mean, paid_rf_coefficient, paid_rf_retention):
        return {"prediction": mean, "coefficient_copy": paid_rf_coefficient, "retention_copy": paid_rf_retention}

    parameters = {
        "intercept": Real(),
        "paid_rf_coefficient": Positive(dims=("group", "rf_channel") if grouped else ("rf_channel",)),
        "paid_rf_retention": Interval(0.0, 1.0, dims="rf_channel"),
        "paid_rf_half_saturation": Positive(dims="rf_channel"),
        "paid_rf_slope": Positive(dims="rf_channel"),
    }
    observation_dims = ("time", "group") if grouped else ("time",)
    saved = ["paid_rf", "paid_rf_total", "mean"]
    generated_dims = {
        "paid_rf": (*observation_dims, "rf_channel"),
        "paid_rf_total": observation_dims,
        "mean": observation_dims,
    }
    mixed = "media" in data.arrays
    if mixed:
        parameters.update(
            paid_media_coefficient=Positive(dims=("group", "channel") if grouped else ("channel",)),
            paid_media_retention=Interval(0.0, 1.0, dims="channel"),
            paid_media_half_saturation=Positive(dims="channel"),
            paid_media_slope=Positive(dims="channel"),
        )
        saved.extend(("paid_media", "paid_media_total"))
        generated_dims.update(paid_media=(*observation_dims, "channel"), paid_media_total=observation_dims)

    return Model(
        parameters,
        density,
        generate,
        data=data,
        scaling=scaling,
        transformed_parameters=mixed_quantities if mixed else rf_quantities,
        save=saved,
        generated_dims=generated_dims,
        predictive=("prediction",),
    )


def _rf_reference(parameters, arrays, periods):
    parameters = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}

    def saturate(values, name):
        powered = np.asarray(values, dtype=np.float64) ** parameters[name + "_slope"]
        return powered / (powered + parameters[name + "_half_saturation"] ** parameters[name + "_slope"])

    def carryover(values, name):
        weights = parameters[name + "_retention"] ** np.arange(3)[:, None]
        weights /= weights.sum(axis=0)
        response = np.zeros_like(values, dtype=np.float64)
        for time in range(len(values)):
            for lag in range(min(time, 2) + 1):
                response[time] += weights[lag] * values[time - lag]
        return response

    effective_reach = np.asarray(arrays["reach"]) * saturate(arrays["media_frequency"], "paid_rf")
    paid_rf = carryover(effective_reach, "paid_rf")[-periods:] * parameters["paid_rf_coefficient"]
    results = {"paid_rf": paid_rf, "paid_rf_total": paid_rf.sum(-1)}
    mean = parameters["intercept"] + results["paid_rf_total"]
    if "media" in arrays:
        media = saturate(carryover(arrays["media"], "paid_media"), "paid_media")[-periods:]
        results["paid_media"] = media * parameters["paid_media_coefficient"]
        results["paid_media_total"] = results["paid_media"].sum(-1)
        mean = mean + results["paid_media_total"]
    return {**results, "mean": mean}


@pytest.mark.parametrize("mixed", [False, True], ids=["rf-only", "mixed"])
@pytest.mark.parametrize("include_priors", [False, True], ids=["likelihood-only", "explicit-rf-priors"])
def test_reach_frequency_models_match_direct_density_and_all_unconstrained_gradients(
    mixed, include_priors, media_values, reach_values, frequency_values
):
    data = _rf_data(reach_values, frequency_values, media=media_values[:, :1] if mixed else None)
    model = _rf_model(data, include_priors=include_priors)
    parameters = _rf_parameters(mixed=mixed)
    position = model.unconstrain(parameters)

    def reference(values):
        constrained = {}
        adjustment = 0.0
        for name, value in values.items():
            value = np.asarray(value, dtype=np.float64)
            if name == "intercept":
                constrained[name] = value
            elif name.endswith("_retention"):
                constrained[name] = 1 / (1 + np.exp(-value))
                adjustment += (np.log(constrained[name]) + np.log1p(-constrained[name])).sum()
            else:
                constrained[name] = np.exp(value)
                adjustment += value.sum()
        quantities = _rf_reference(constrained, data.arrays, 3)
        target = _normal(data.arrays["outcome"] - quantities["mean"], 0.7) + _normal(constrained["intercept"], 2)
        if include_priors:
            target += (np.log(3) + 2 * np.log1p(-constrained["paid_rf_retention"])).sum()
            for role in _POSITIVE:
                value = constrained["paid_rf_" + role]
                target += _normal(value, 1.5) + value.size * np.log(2)
        return target + adjustment, target, quantities

    expected_density, expected_log_prob, expected = reference(position)
    evaluated = jax.jit(model.evaluate)(parameters)
    assert set(model.parameters) == set(parameters)
    assert set(evaluated) == set(expected)
    assert model.parameters["paid_rf_coefficient"].shape == (2,)
    if mixed:
        assert model.parameters["paid_media_coefficient"].shape == (1,)
    for name, value in expected.items():
        np.testing.assert_allclose(evaluated[name], value, rtol=5e-6, atol=3e-6)
    np.testing.assert_allclose(jax.jit(model.log_prob)(parameters), expected_log_prob, rtol=5e-6, atol=3e-6)
    actual, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    np.testing.assert_allclose(actual, expected_density, rtol=5e-6, atol=3e-6)
    for name, value in position.items():
        value = np.asarray(value, dtype=np.float64)
        expected_gradient = np.empty_like(value)
        for index in np.ndindex(value.shape):
            delta = np.zeros_like(value)
            delta[index] = 1e-4
            expected_gradient[index] = (
                reference({**position, name: value + delta})[0] - reference({**position, name: value - delta})[0]
            ) / 2e-4
        assert np.isfinite(gradient[name]).all()
        np.testing.assert_allclose(gradient[name], expected_gradient, rtol=5e-5, atol=8e-6)


@pytest.mark.parametrize("mixed", [False, True], ids=["rf-only", "mixed"])
def test_grouped_reach_frequency_predictions_reuse_scaling_and_align_shorter_reordered_history(
    mixed, media_values, reach_values, frequency_values
):
    reach = np.stack((reach_values, reach_values * 1.7 + 2), axis=1)
    frequency = np.stack((frequency_values, frequency_values + 0.5), axis=1)
    media = np.stack((media_values[:, :1], media_values[:, :1] * 1.5), axis=1) if mixed else None
    training = _rf_data(reach, frequency, media=media)
    model = _rf_model(training, grouped=True, scaling="auto")
    parameters = _rf_parameters(mixed=mixed, grouped=True)
    divisor = np.array([np.median(reach[..., index][reach[..., index] > 0]) for index in range(2)])
    np.testing.assert_allclose(model.data.values["reach"], reach / divisor, rtol=2e-6)
    np.testing.assert_array_equal(model.data.values["media_frequency"], frequency)
    np.testing.assert_array_equal(
        model.data.values["rf_spend"], training.arrays["rf_spend"].astype(model.data.values["rf_spend"].dtype)
    )
    assert "media_frequency" not in model.scaling.transformations

    future_reach, future_frequency = reach[:4] * 2.3, frequency[:4] * 0.7
    future_media = media[:4] * 1.8 if mixed else None
    future = _rf_data(
        future_reach,
        future_frequency,
        media=future_media,
        periods=2,
        start=6,
        reverse=True,
        observed=False,
    )
    inputs = model.prepare_data(future)
    assert future.rf_channels == ("social", "tv")
    assert future.group_values == (("east",), ("west",))
    assert "outcome" not in inputs.values and "rf_spend" not in inputs.values
    np.testing.assert_allclose(inputs.values["reach"], future_reach / divisor, rtol=2e-6)
    np.testing.assert_array_equal(
        inputs.values["media_frequency"], future_frequency.astype(inputs.values["media_frequency"].dtype)
    )
    if mixed:
        media_divisor = np.median(media[media > 0])
        np.testing.assert_allclose(inputs.values["media"], future_media / media_divisor, rtol=2e-6)
    draws = jax.tree.map(lambda value: jnp.stack((value, value * 1.1)), parameters)
    actual = jax.jit(jax.vmap(model.evaluate, in_axes=(0, None)))(draws, inputs)
    generated = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))(
        jax.random.split(jax.random.key(12), 2), draws, inputs
    )
    assert actual["paid_rf"].shape == (2, 2, 2, 2)
    assert actual["paid_rf_total"].shape == (2, 2, 2)
    for draw in range(2):
        values = {name: value[draw] for name, value in draws.items()}
        expected = _rf_reference(values, inputs.values, 2)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name][draw], value, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(generated[name][draw], value, rtol=5e-6, atol=3e-6)
        np.testing.assert_allclose(generated["prediction"][draw], expected["mean"], rtol=5e-6, atol=3e-6)


@pytest.mark.parametrize("mixed", [False, True], ids=["rf-only", "mixed"])
def test_reach_frequency_prior_and_posterior_replay_retain_saved_totals_and_distinct_channel_labels(
    mixed, media_values, reach_values, frequency_values
):
    reach = np.stack((reach_values, reach_values * 1.7 + 2), axis=1)
    frequency = np.stack((frequency_values, frequency_values + 0.5), axis=1)
    media = np.stack((media_values[:, :1], media_values[:, :1] * 1.5), axis=1) if mixed else None
    training = _rf_data(reach, frequency, media=media)
    model = _rf_model(training, grouped=True)
    parameters = _rf_parameters(mixed=mixed, grouped=True)

    def prior(key):
        return {**parameters, "paid_rf_coefficient": parameters["paid_rf_coefficient"] * jax.random.uniform(key) + 0.01}

    result = sample_prior(model, prior, draws=2, seed=8)
    prior_draws = result["prior"]
    assert prior_draws["paid_rf_coefficient"].dims == ("chain", "draw", "group", "rf_channel")
    for role in ("retention", "half_saturation", "slope"):
        assert prior_draws["paid_rf_" + role].dims == ("chain", "draw", "rf_channel")
    assert result["constant_data"]["reach"].dims == ("media_time", "group", "rf_channel")
    assert result["constant_data"]["media_frequency"].dims == ("media_time", "group", "rf_channel")
    assert result["constant_data"]["rf_spend"].dims == ("time", "group", "rf_channel")
    np.testing.assert_array_equal(result["constant_data"]["media_time"], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(prior_draws["rf_channel"], ["tv", "social"])
    posterior = _collect_results(
        {name: prior_draws[name].values for name in model.parameters},
        data=training,
        dims={name: prior_draws[name].dims[2:] for name in model.parameters},
    )
    future = _rf_data(
        reach[:4] * 1.8,
        frequency[:4] * 0.6,
        media=media[:4] * 1.2 if mixed else None,
        periods=2,
        start=6,
        reverse=True,
        observed=False,
    )
    replay = generate_quantities(model, posterior, new_data=future)
    assert "observed_data" not in replay.children
    assert "rf_spend" not in replay["constant_data"]
    for tree, group, predictive, times, inputs in (
        (result, "prior_generated_quantities", "prior_predictive", [2, 3, 4], model.data),
        (replay, "generated_quantities", "posterior_predictive", [8, 9], model.prepare_data(future)),
    ):
        saved = tree[group]
        assert saved["paid_rf"].dims == ("chain", "draw", "time", "group", "rf_channel")
        assert saved["paid_rf_total"].dims == ("chain", "draw", "time", "group")
        assert saved["coefficient_copy"].dims == ("chain", "draw", "group", "rf_channel")
        assert saved["retention_copy"].dims == ("chain", "draw", "rf_channel")
        assert tree[predictive]["prediction"].dims == ("chain", "draw", "time", "group")
        np.testing.assert_array_equal(saved["rf_channel"], ["tv", "social"])
        np.testing.assert_array_equal(saved["group"], ["west", "east"])
        np.testing.assert_array_equal(tree[predictive]["time"], times)
        np.testing.assert_allclose(saved["paid_rf_total"], saved["paid_rf"].sum("rf_channel"), rtol=2e-6)
        np.testing.assert_array_equal(saved["coefficient_copy"], prior_draws["paid_rf_coefficient"])
        if mixed:
            assert tree["prior" if group.startswith("prior") else "posterior"]["paid_media_coefficient"].dims == (
                "chain",
                "draw",
                "group",
                "channel",
            )
            assert saved["paid_media"].dims == ("chain", "draw", "time", "group", "channel")
            np.testing.assert_array_equal(saved["channel"], ["video"])
        else:
            assert "channel" not in saved.dims
        for draw in range(2):
            values = {name: prior_draws[name].values[0, draw] for name in model.parameters}
            expected = _rf_reference(values, inputs.values, len(times))
            for name, value in expected.items():
                np.testing.assert_allclose(saved[name].values[0, draw], value, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(tree[predictive]["prediction"].values[0, draw], expected["mean"], rtol=5e-6)


def test_reach_frequency_prediction_rejects_missing_inputs_and_changed_frequency_source_columns(
    reach_values, frequency_values
):
    model = _rf_model(_rf_data(reach_values, frequency_values))
    for role in ("reach", "media_frequency"):
        incomplete = _rf_data(reach_values, frequency_values, observed=False)
        incomplete.arrays.pop(role)
        inputs = model.prepare_data(incomplete)
        with pytest.raises(ValueError, match=r"reach|frequency"):
            model.evaluate(_rf_parameters(), inputs)
    renamed = _rf_data(
        reach_values,
        frequency_values,
        observed=False,
        frequency_columns=("tv_repeats", "social_repeats"),
    )
    with pytest.raises(ValueError, match=r"frequency|columns"):
        model.prepare_data(renamed)
