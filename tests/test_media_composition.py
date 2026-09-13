"""Tests for composing learned media and seasonal effects in public models."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import (
    FourierSeasonality,
    MediaEffect,
    Model,
    Positive,
    ReachFrequencyEffect,
    Real,
    beta,
    delayed_adstock,
    generate_quantities,
    half_normal,
    logistic_saturation,
    normal,
    normal_rng,
    prepare_data,
    root_saturation,
    sample_prior,
    uniform,
    weibull_cdf_adstock,
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


def _component_prior(coefficient, retention, half_saturation, slope, annual_coefficients):
    return (
        half_normal(coefficient, 1.5)
        + beta(retention, 1.0, 3.0)
        + half_normal(half_saturation, 1.5)
        + half_normal(slope, 1.5)
        + normal(annual_coefficients, 0.0, 1.0)
    )


def _model(data, *, keyword_only_generation=True, grouped=False, include_component_priors=True):
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
        if include_component_priors:
            target += _component_prior(
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

    return Model(
        {"intercept": Real(), "sigma": Positive()},
        density,
        quantities if keyword_only_generation else ordinary_quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2, group_specific_coefficients=grouped),
            FourierSeasonality(period=8, order=1, name="annual", group_specific_coefficients=grouped),
        ],
    )


def _position(*, grouped=False):
    coefficient = [[0.7, 1.1], [0.5, 1.3]] if grouped else [0.7, 1.1]
    return {
        "paid_media_coefficient": jnp.log(jnp.asarray(coefficient)),
        "paid_media_retention": jnp.log(jnp.array([0.25, 0.6]) / jnp.array([0.75, 0.4])),
        "paid_media_half_saturation": jnp.log(jnp.array([1.5, 2.5])),
        "paid_media_slope": jnp.log(jnp.array([1.2, 0.8])),
        "annual": jnp.array([[0.3, -0.2], [0.7, 1.1]]) if grouped else jnp.array([0.3, -0.2]),
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
    annual = np.column_stack((np.sin(angle), np.cos(angle))) @ position["annual"]
    mean = position["intercept"] + paid.sum(axis=-1) + annual
    prior = _normal(position["annual"]) + _normal(position["intercept"], 2.0)
    prior += (np.log(3) + 2 * np.log1p(-retention)).sum()
    adjustment = position["sigma"] + (np.log(retention) + np.log1p(-retention)).sum()
    for role in _POSITIVE:
        value = parameters["paid_media_" + role]
        prior += _normal(value, 1.5) + value.size * np.log(2)
        adjustment += position["paid_media_" + role].sum()
    return mean, paid, annual, prior + adjustment, parameters["sigma"]


@pytest.mark.parametrize("keyword_only_generation", [False, True])
def test_mixed_components_match_full_density_gradients_and_generated_breakdowns(keyword_only_generation, media_values):
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
    assert model.parameters["annual"].shape == (2,)
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


def test_media_models_require_exposures_and_reject_foreign_component_bundles(media_values):
    model, position = _model(_data(media_values)), _position()
    missing = prepare_data(pl.DataFrame({"time": [5, 6]}), time="time")
    with pytest.raises(ValueError, match="media"):
        model.prepare_data(missing)
    foreign = _model(_data(media_values))
    with pytest.raises(ValueError):
        model.log_density(position, foreign.data)
    with pytest.raises(ValueError):
        model.generate(jax.random.key(0), model.constrain(position), foreign.data)


def test_media_components_reject_legacy_density_inputs_with_guidance(media_values):
    with pytest.raises(TypeError, match=r"no longer receives data or effects.*Request individual inputs by name"):
        Model(
            {},
            lambda data, effects: normal(data["outcome"], effects["paid_media"].sum(-1), 1.0),
            data=_data(media_values),
            components=[MediaEffect(max_lag=2)],
        )


def test_generated_parameter_and_effect_names_cannot_be_shadowed(media_values):
    data = _data(media_values)

    def callback():
        return jnp.array(0.0)

    for name in ("paid_media", "paid_media_coefficient"):
        with pytest.raises(ValueError, match=name):
            Model({name: Real()}, callback, data=data, components=[MediaEffect(max_lag=2)])
    for reverse in (False, True):
        components = [MediaEffect(max_lag=2), FourierSeasonality(period=8, name="paid_media_coefficient")]
        with pytest.raises(ValueError, match="paid_media_coefficient"):
            Model({}, callback, data=data, components=components[::-1] if reverse else components)


def _total_model(data, *, transformed=False, grouped=False):
    def density(
        outcome,
        paid_media_total,
        annual,
        intercept,
        sigma,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
        annual_coefficients,
    ):
        return (
            normal(outcome, intercept + paid_media_total + annual, sigma)
            + normal(intercept, 0.0, 2.0)
            + _component_prior(
                paid_media_coefficient,
                paid_media_retention,
                paid_media_half_saturation,
                paid_media_slope,
                annual_coefficients,
            )
        )

    def transform(paid_media_total, annual, intercept):
        return {"mean": intercept + paid_media_total + annual}

    def transformed_density(
        outcome,
        mean,
        intercept,
        sigma,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
        annual_coefficients,
    ):
        return (
            normal(outcome, mean, sigma)
            + normal(intercept, 0.0, 2.0)
            + _component_prior(
                paid_media_coefficient,
                paid_media_retention,
                paid_media_half_saturation,
                paid_media_slope,
                annual_coefficients,
            )
        )

    def quantities(key, paid_media, paid_media_total):
        return {"channels": paid_media, "total": paid_media_total}

    def transformed_quantities(key, paid_media, paid_media_total, mean):
        return {"channels": paid_media, "total": paid_media_total, "mean": mean}

    return Model(
        {"intercept": Real(), "sigma": Positive()},
        transformed_density if transformed else density,
        transformed_quantities if transformed else quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2, group_specific_coefficients=grouped),
            FourierSeasonality(period=8, order=1, name="annual", group_specific_coefficients=grouped),
        ],
        transformed_parameters=transform if transformed else None,
    )


@pytest.mark.parametrize("transformed", [False, True])
def test_named_media_totals_preserve_density_and_all_parameter_gradients(transformed, media_values):
    data = _data(media_values)
    model = _total_model(data, transformed=transformed)
    original, position = _model(data), _position()
    evaluate = jax.jit(jax.value_and_grad(model.log_density))
    actual, gradient = evaluate(position, model.data)
    expected, expected_gradient = jax.jit(jax.value_and_grad(original.log_density))(position, original.data)
    mean, paid, _, prior, sigma = _reference(position, media_values, [2, 3, 4])

    assert set(model.parameters) == set(original.parameters) == set(position)
    np.testing.assert_allclose(actual, _normal(data.arrays["outcome"] - mean, sigma) + prior, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-6)
    for name in position:
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)

    generated = jax.jit(model.generate)(jax.random.key(0), model.constrain(position), model.data)
    assert isinstance(generated["channels"], jax.Array)
    assert generated["channels"].shape == (3, 2)
    assert generated["total"].shape == (3,)
    np.testing.assert_allclose(generated["channels"], paid, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(generated["total"], paid.sum(axis=-1), rtol=4e-6, atol=3e-6)
    if transformed:
        np.testing.assert_allclose(generated["mean"], mean, rtol=4e-6, atol=3e-6)


@pytest.mark.parametrize(
    ("adstock", "saturation", "adstock_arguments", "saturation_arguments"),
    [
        (
            weibull_cdf_adstock,
            root_saturation,
            {"adstock_shape": "shape", "adstock_scale": "scale"},
            {"exponent": "exponent"},
        ),
        (
            delayed_adstock,
            logistic_saturation,
            {"retention": "alpha", "delay": "theta"},
            {"half_saturation": "half_saturation"},
        ),
    ],
    ids=["weibull-root", "delayed-logistic"],
)
def test_selected_media_functions_flow_through_model_transforms_priors_and_jit_gradients(
    adstock, saturation, adstock_arguments, saturation_arguments, media_values
):
    data = _data(media_values)

    def transform(paid_media_total, intercept):
        return {"mean": intercept + paid_media_total}

    if adstock is weibull_cdf_adstock:

        def density(
            outcome,
            mean,
            intercept,
            paid_media_coefficient,
            paid_media_adstock_shape,
            paid_media_adstock_scale,
            paid_media_exponent,
        ):
            return (
                normal(outcome, mean, 0.7)
                + normal(intercept, 0.0, 2.0)
                + half_normal(paid_media_coefficient, 1.5)
                + half_normal(paid_media_adstock_shape, 1.5)
                + half_normal(paid_media_adstock_scale, 1.5)
                + uniform(paid_media_exponent, 0.0, 1.0)
            )
    else:

        def density(
            outcome,
            mean,
            intercept,
            paid_media_coefficient,
            paid_media_retention,
            paid_media_delay,
            paid_media_half_saturation,
        ):
            return (
                normal(outcome, mean, 0.7)
                + normal(intercept, 0.0, 2.0)
                + half_normal(paid_media_coefficient, 1.5)
                + beta(paid_media_retention, 1.0, 3.0)
                + uniform(paid_media_delay, 0.0, 2.0)
                + half_normal(paid_media_half_saturation, 1.5)
            )

    def quantities(key, paid_media, paid_media_total, mean):
        return {"channels": paid_media, "total": paid_media_total, "mean": mean}

    model = Model(
        {"intercept": Real()},
        density,
        quantities,
        data=data,
        components=[MediaEffect(max_lag=2, adstock=adstock, saturation=saturation)],
        transformed_parameters=transform,
    )
    roles = ("coefficient", *adstock_arguments, *saturation_arguments)
    position = {
        f"paid_media_{role}": jnp.array([-0.5 + index / 10, 0.2 + index / 5]) for index, role in enumerate(roles)
    }
    position["intercept"] = jnp.array(0.25)

    def reference(unconstrained):
        parameters = {}
        prior = jnp.array(0.0)
        for role in roles:
            value = unconstrained[f"paid_media_{role}"]
            if role in ("retention", "delay", "exponent"):
                width = 2.0 if role == "delay" else 1.0
                fraction = jax.nn.sigmoid(value)
                parameters[role] = width * fraction
                prior += jnp.sum(jnp.log(width) + jnp.log(fraction) + jnp.log1p(-fraction))
                if role == "retention":
                    prior += jnp.sum(jnp.log(3.0) + 2 * jnp.log1p(-fraction))
                else:
                    prior -= value.size * jnp.log(width)
            else:
                parameters[role] = jnp.exp(value)
                prior += normal(parameters[role], 0.0, 1.5) + value.size * jnp.log(2.0) + value.sum()
        carried = adstock(
            jnp.asarray(media_values),
            **{argument: parameters[role] for role, argument in adstock_arguments.items()},
            max_lag=2,
        )
        response = saturation(
            carried, **{argument: parameters[role] for role, argument in saturation_arguments.items()}
        )
        channels = response[-len(data.time_values) :] * parameters["coefficient"]
        mean = unconstrained["intercept"] + channels.sum(axis=-1)
        value = normal(data.arrays["outcome"], mean, 0.7) + normal(unconstrained["intercept"], 0.0, 2.0) + prior
        return value, (parameters, channels, mean)

    actual, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    (expected, (parameters, channels, mean)), expected_gradient = jax.value_and_grad(reference, has_aux=True)(position)
    assert set(model.parameters) == set(position)
    constrained = jax.jit(model.constrain)(position)
    for role in roles:
        name = f"paid_media_{role}"
        assert model.parameters[name].shape == (2,)
        np.testing.assert_allclose(constrained[name], parameters[role], rtol=3e-6)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-6)
    for name in position:
        assert np.all(np.isfinite(gradient[name]))
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)
    generated = jax.jit(model.generate)(jax.random.key(0), constrained, model.data)
    for name, expected in (("channels", channels), ("total", channels.sum(axis=-1)), ("mean", mean)):
        np.testing.assert_allclose(generated[name], expected, rtol=4e-6, atol=3e-6)


@pytest.mark.parametrize("transformed", [False, True])
def test_media_totals_preserve_group_and_prediction_axes_under_jit_vmap(transformed, media_values):
    media = np.stack((media_values, media_values * 1.5 + 0.2), axis=1)
    model = _total_model(_data(media), transformed=transformed, grouped=True)
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.1)), _position(grouped=True))
    constrained = jax.vmap(model.constrain)(draws)
    generate = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))
    keys = jax.random.split(jax.random.key(2), 2)

    for multiplier in (0.6, 1.4):
        future_media = media[:4] * multiplier
        future = _data(future_media, periods=2, start=6, reverse=True, observed=False)
        generated = generate(keys, constrained, model.prepare_data(future))
        assert "outcome" not in future.arrays
        assert generated["channels"].shape == (2, 2, 2, 2)
        assert generated["total"].shape == (2, 2, 2)
        for index in range(2):
            position = {name: value[index] for name, value in draws.items()}
            mean, paid, _, _, _ = _reference(position, future_media, [8, 9])
            np.testing.assert_allclose(generated["channels"][index], paid, rtol=5e-6, atol=3e-6)
            np.testing.assert_allclose(generated["total"][index], paid.sum(axis=-1), rtol=5e-6, atol=3e-6)
            if transformed:
                np.testing.assert_allclose(generated["mean"][index], mean, rtol=5e-6, atol=3e-6)


def test_custom_media_names_have_independent_totals_and_keep_single_channel_axes():
    data = prepare_data(pl.DataFrame({"time": [0, 1, 2], "video": [1.0, 2.0, 3.0]}), time="time", media=["video"])

    def density(first_total, campaign_total_total):
        return normal(first_total + campaign_total_total, 0.0, 1.0)

    def quantities(key, first, first_total, campaign_total, campaign_total_total):
        return {
            "first": first,
            "first_total": first_total,
            "campaign_total": campaign_total,
            "campaign_total_total": campaign_total_total,
        }

    model = Model(
        {},
        density,
        quantities,
        data=data,
        components=[MediaEffect(name="first", max_lag=0), MediaEffect(name="campaign_total", max_lag=0)],
    )
    position = {name: jnp.zeros(parameter.shape) for name, parameter in model.parameters.items()}
    position["campaign_total_coefficient"] = jnp.log(jnp.array([2.0]))
    generated = jax.jit(model.generate)(jax.random.key(0), model.constrain(position), model.data)
    for name, coefficient in (("first", 1.0), ("campaign_total", 2.0)):
        expected = coefficient * np.array([1 / 2, 2 / 3, 3 / 4])
        assert generated[name].shape == (3, 1)
        assert generated[name + "_total"].shape == (3,)
        np.testing.assert_allclose(generated[name][:, 0], expected, rtol=3e-6)
        np.testing.assert_allclose(generated[name + "_total"], expected, rtol=3e-6)


def test_media_total_names_are_reserved_even_when_callbacks_do_not_request_them(media_values):
    data = _data(media_values)

    def callback():
        return jnp.array(0.0)

    media = MediaEffect(max_lag=2)
    with pytest.raises(ValueError, match="paid_media_total"):
        Model({"paid_media_total": Real()}, callback, data=data, components=[media])

    for other in (
        FourierSeasonality(period=8, name="paid_media_total"),
        MediaEffect(max_lag=2, name="paid_media_total"),
    ):
        for components in ([media, other], [other, media]):
            with pytest.raises(ValueError, match="paid_media_total"):
                Model({}, callback, data=data, components=components)

    data.arrays["paid_media_total"] = np.ones(3)
    with pytest.raises(ValueError, match="paid_media_total"):
        Model({}, callback, data=data, components=[media])


def test_transformed_outputs_cannot_replace_unrequested_media_totals(media_values):
    model = Model(
        {},
        lambda *, mean: jnp.sum(mean),
        lambda key, *, mean: {"mean": mean},
        data=_data(media_values),
        components=[MediaEffect(max_lag=2)],
        transformed_parameters=lambda: {"mean": jnp.zeros(3), "paid_media_total": jnp.zeros(3)},
    )
    position = {name: value for name, value in _position().items() if name.startswith("paid_media_")}
    with pytest.raises(ValueError, match="paid_media_total"):
        model.log_density(position, model.data)
    with pytest.raises(ValueError, match="paid_media_total"):
        model.generate(jax.random.key(0), model.constrain(position), model.data)


def test_media_total_names_do_not_replace_random_keys_or_create_fourier_totals(media_values):
    data = _data(media_values)
    with pytest.raises(TypeError, match=r"paid_media_total.*random key"):
        Model(
            {},
            lambda *, paid_media_total: jnp.sum(paid_media_total),
            lambda paid_media_total, *, paid_media: {"channels": paid_media},
            data=data,
            components=[MediaEffect(max_lag=2)],
        )
    with pytest.raises(ValueError, match="unknown input 'annual_total'"):
        Model(
            {},
            lambda *, annual_total: jnp.sum(annual_total),
            data=data,
            components=[FourierSeasonality(period=8, name="annual")],
        )


@pytest.mark.parametrize("annual_scale", [0.4, 1.7])
def test_explicit_component_prior_choices_control_density_and_gradients(annual_scale, media_values):
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
        prior = half_normal(paid_media_coefficient, 1.5) + beta(paid_media_retention, 1.0, 3.0)
        prior += half_normal(paid_media_half_saturation, 1.5) + half_normal(paid_media_slope, 1.5)
        prior += normal(annual_coefficients, 0.0, annual_scale)
        return normal(outcome, intercept + paid_media.sum(-1) + annual, sigma) + normal(intercept, 0.0, 2.0) + prior

    def quantities(key, paid_media_coefficient, paid_media_retention, annual_coefficients, annual):
        return {
            "coefficient": paid_media_coefficient,
            "retention": paid_media_retention,
            "seasonal_coefficients": annual_coefficients,
            "annual": annual,
        }

    data, position = _data(media_values), _position()
    model = Model(
        {"intercept": Real(), "sigma": Positive()},
        density,
        quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2),
            FourierSeasonality(period=8, order=1, name="annual"),
        ],
    )
    baseline = _model(data)

    def reference(values):
        target = baseline.log_density(values, baseline.data)
        return target + normal(values["annual"], 0.0, annual_scale) - normal(values["annual"], 0.0, 1.0)

    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(reference))(position)
    assert set(model.parameters) == set(baseline.parameters) == set(position)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, rtol=5e-6, atol=3e-6)

    parameters = model.constrain(position)
    generated = jax.jit(model.generate)(jax.random.key(0), parameters, model.data)
    for output, parameter in (
        ("coefficient", "paid_media_coefficient"),
        ("retention", "paid_media_retention"),
        ("seasonal_coefficients", "annual"),
    ):
        np.testing.assert_allclose(generated[output], parameters[parameter], rtol=3e-6)
    np.testing.assert_allclose(
        generated["annual"], _reference(position, media_values, [2, 3, 4])[2], rtol=4e-6, atol=2e-6
    )


def test_density_without_component_priors_retains_only_explicit_terms_and_jacobians(media_values):
    data = _data(media_values)
    model = _model(data, include_component_priors=False)
    position = _position()

    def reference(values):
        parameters = model.constrain(values)
        generated = model.generate(jax.random.key(0), parameters, model.data)
        retention = parameters["paid_media_retention"]
        jacobian = values["sigma"] + jnp.sum(jnp.log(retention) + jnp.log1p(-retention))
        jacobian += sum(values["paid_media_" + role].sum() for role in _POSITIVE)
        return (
            normal(data.arrays["outcome"], generated["mean"], parameters["sigma"])
            + normal(parameters["intercept"], 0.0, 2.0)
            + jacobian
        )

    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(reference))(position)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, rtol=5e-6, atol=3e-6)


def test_declared_component_parameters_reach_transforms_and_forecast_generation_under_jit_vmap(media_values):
    def transform(paid_media_coefficient, paid_media_retention, annual_coefficients):
        return {"parameter_summary": paid_media_coefficient * paid_media_retention + annual_coefficients}

    def quantities(key, paid_media, annual, parameter_summary, paid_media_coefficient, annual_coefficients):
        return {
            "media": paid_media,
            "annual": annual,
            "summary": parameter_summary,
            "coefficient": paid_media_coefficient,
            "seasonal_coefficients": annual_coefficients,
        }

    media = np.stack((media_values, media_values * 1.5 + 0.2), axis=1)
    model = Model(
        {},
        lambda *, parameter_summary: normal(parameter_summary, 0.0, 2.0),
        quantities,
        data=_data(media),
        components=[
            MediaEffect(max_lag=2, group_specific_coefficients=True),
            FourierSeasonality(period=8, order=1, name="annual", group_specific_coefficients=True),
        ],
        transformed_parameters=transform,
    )
    position = {name: value for name, value in _position(grouped=True).items() if name in model.parameters}
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.2)), position)
    parameters = jax.vmap(model.constrain)(draws)
    future = _data(media[:4] * 0.8, periods=2, start=6, reverse=True, observed=False)
    generated = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))(
        jax.random.split(jax.random.key(0), 2), parameters, model.prepare_data(future)
    )
    expected_summary = parameters["paid_media_coefficient"] * parameters["paid_media_retention"][:, None, :]
    expected_summary += parameters["annual"]
    assert generated["media"].shape == (2, 2, 2, 2)
    assert generated["annual"].shape == (2, 2, 2)
    np.testing.assert_allclose(generated["summary"], expected_summary, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(generated["coefficient"], parameters["paid_media_coefficient"], rtol=3e-6)
    np.testing.assert_allclose(generated["seasonal_coefficients"], parameters["annual"], rtol=3e-6)


def test_selected_media_parameters_support_direct_priors_and_reject_inactive_roles(media_values):
    def direct_density(
        outcome,
        paid_media_total,
        paid_media_coefficient,
        paid_media_adstock_shape,
        paid_media_adstock_scale,
        paid_media_exponent,
    ):
        return (
            normal(outcome, paid_media_total, 0.7)
            + half_normal(paid_media_coefficient, 1.5)
            + half_normal(paid_media_adstock_shape, 1.5)
            + half_normal(paid_media_adstock_scale, 1.5)
            + uniform(paid_media_exponent, 0.0, 1.0)
        )

    data = _data(media_values)
    component = MediaEffect(max_lag=2, adstock=weibull_cdf_adstock, saturation=root_saturation)
    model = Model({}, direct_density, data=data, components=[component])
    roles = ("coefficient", "adstock_shape", "adstock_scale", "exponent")
    position = {
        f"paid_media_{role}": jnp.array([-0.5 + index / 10, 0.2 + index / 5]) for index, role in enumerate(roles)
    }
    assert set(model.parameters) == set(position)

    def reference(values):
        coefficient = jnp.exp(values["paid_media_coefficient"])
        shape = jnp.exp(values["paid_media_adstock_shape"])
        scale = jnp.exp(values["paid_media_adstock_scale"])
        exponent = jax.nn.sigmoid(values["paid_media_exponent"])
        carried = weibull_cdf_adstock(jnp.asarray(media_values), shape, scale, max_lag=2)
        total = (root_saturation(carried, exponent)[-3:] * coefficient).sum(axis=-1)
        jacobian = sum(values[f"paid_media_{role}"].sum() for role in roles[:-1])
        jacobian += jnp.sum(jnp.log(exponent) + jnp.log1p(-exponent))
        return (
            normal(data.arrays["outcome"], total, 0.7)
            + half_normal(coefficient, 1.5)
            + half_normal(shape, 1.5)
            + half_normal(scale, 1.5)
            + uniform(exponent, 0.0, 1.0)
            + jacobian
        )

    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(reference))(position)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, rtol=5e-6, atol=3e-6)
    with pytest.raises(ValueError, match="paid_media_retention"):
        Model({}, lambda *, paid_media_retention: jnp.sum(paid_media_retention), data=data, components=[component])


def test_requested_media_parameter_names_cannot_collide_with_data(media_values):
    data = _data(media_values)
    data.arrays["paid_media_coefficient"] = np.ones(3)
    with pytest.raises(ValueError, match=r"paid_media_coefficient.*ambiguous"):
        Model(
            {},
            lambda *, paid_media_coefficient: normal(paid_media_coefficient, 0.0, 1.0),
            data=data,
            components=[MediaEffect(max_lag=2)],
        )


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

    def transform(paid_rf_total, intercept):
        return {"mean": intercept + paid_rf_total}

    def mixed_transform(paid_media_total, paid_rf_total, intercept):
        return {"mean": intercept + paid_media_total + paid_rf_total}

    def generate(key, mean, paid_rf_coefficient, paid_rf_retention):
        return {"prediction": mean, "coefficient_copy": paid_rf_coefficient, "retention_copy": paid_rf_retention}

    components = [ReachFrequencyEffect(max_lag=2, group_specific_coefficients=grouped)]
    saved = ["paid_rf", "paid_rf_total", "mean"]
    mixed = "media" in data.arrays
    if mixed:
        components.insert(0, MediaEffect(max_lag=2, group_specific_coefficients=grouped))
        saved.extend(("paid_media", "paid_media_total"))
    return Model(
        {"intercept": Real()},
        density,
        generate,
        data=data,
        components=components,
        scaling=scaling,
        transformed_parameters=mixed_transform if mixed else transform,
        save=saved,
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
        with pytest.raises(ValueError, match=r"reach|frequency"):
            model.prepare_data(incomplete)
    renamed = _rf_data(
        reach_values,
        frequency_values,
        observed=False,
        frequency_columns=("tv_repeats", "social_repeats"),
    )
    with pytest.raises(ValueError, match=r"frequency|columns"):
        model.prepare_data(renamed)
