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
    Real,
    beta,
    delayed_adstock,
    half_normal,
    logistic_saturation,
    normal,
    normal_rng,
    prepare_data,
    root_saturation,
    uniform,
    weibull_cdf_adstock,
)

_MEDIA = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 2.0], [4.0, 1.0], [2.0, 3.0]])
_POSITIVE = ("coefficient", "half_saturation", "slope")


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


def _model(data, *, named=True, grouped=False, automatic_priors=True):
    def density(*, outcome, paid_media, annual, intercept, sigma):
        return normal(outcome, intercept + paid_media.sum(-1) + annual, sigma) + normal(intercept, 0.0, 2.0)

    def quantities(key, *, paid_media, annual, controls, intercept, sigma):
        mean = intercept + paid_media.sum(-1) + annual
        return {
            "mean": mean,
            "paid_media": paid_media,
            "annual": annual,
            "controls": controls,
            "draw": normal_rng(key, mean, sigma),
        }

    def positional_density(data, effects, **parameters):
        assert set(parameters) == {"intercept", "sigma"}
        return density(outcome=data["outcome"], **effects, **parameters)

    def positional_quantities(key, data, effects, **parameters):
        assert set(parameters) == {"intercept", "sigma"}
        return quantities(key, controls=data["controls"], **effects, **parameters)

    return Model(
        {"intercept": Real(), "sigma": Positive()},
        density if named else positional_density,
        quantities if named else positional_quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2, group_specific=grouped, automatic_priors=automatic_priors),
            FourierSeasonality(
                period=8, order=1, name="annual", group_specific=grouped, automatic_priors=automatic_priors
            ),
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


@pytest.mark.parametrize("named", [False, True])
def test_mixed_components_match_full_density_gradients_and_generated_breakdowns(named):
    data = _data(_MEDIA)
    model, position = _model(data, named=named), _position()
    mean, paid, annual, prior, sigma = _reference(position, _MEDIA, [2, 3, 4])
    outcome = np.asarray(data.arrays["outcome"], dtype=np.float64)

    def density_reference(values):
        expected_mean, _, _, expected_prior, scale = _reference(values, _MEDIA, [2, 3, 4])
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


def test_posterior_vmap_evaluates_dynamic_prediction_media_and_parameters():
    model = _model(_data(_MEDIA))
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.1)), _position())
    evaluate = jax.jit(jax.vmap(model.log_density, in_axes=(0, None)))
    generate = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))
    keys = jax.random.split(jax.random.key(3), 2)
    constrained = jax.vmap(model.constrain)(draws)
    for multiplier in (0.6, 1.4):
        future_media = _MEDIA * multiplier
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


def test_grouped_prediction_aligns_history_channels_controls_and_preserves_float_precision():
    media = np.stack((_MEDIA, _MEDIA * 1.5 + 0.2), axis=1)
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


def test_media_model_snapshots_training_arrays_and_retains_layout_after_source_mutation():
    data = _data(_MEDIA)
    model, position = _model(data), _position()
    parameters = model.constrain(position)
    data.arrays["media"][:] = 99
    data.arrays.clear()
    data.columns.clear()
    training = jax.jit(model.generate)(jax.random.key(0), parameters, model.data)
    np.testing.assert_allclose(training["mean"], _reference(position, _MEDIA, [2, 3, 4])[0], rtol=4e-6, atol=2e-6)
    future_media = _MEDIA * 0.7
    future = _data(future_media, start=5, observed=False)
    inputs = model.prepare_data(future)
    future.arrays["media"][:] = 99
    forecast = jax.jit(model.generate)(jax.random.key(0), parameters, inputs)
    np.testing.assert_allclose(forecast["mean"], _reference(position, future_media, [7, 8, 9])[0], rtol=4e-6, atol=2e-6)


def test_media_models_require_exposures_and_reject_foreign_component_bundles():
    model, position = _model(_data(_MEDIA)), _position()
    missing = prepare_data(pl.DataFrame({"time": [5, 6]}), time="time")
    with pytest.raises(ValueError, match="media"):
        model.prepare_data(missing)
    foreign = _model(_data(_MEDIA))
    with pytest.raises(ValueError):
        model.log_density(position, foreign.data)
    with pytest.raises(ValueError):
        model.generate(jax.random.key(0), model.constrain(position), foreign.data)


def test_generated_parameter_and_effect_names_cannot_be_shadowed():
    data = _data(_MEDIA)

    def callback(data, effects, **parameters):
        return jnp.array(0.0)

    for name in ("paid_media", "paid_media_coefficient"):
        with pytest.raises(ValueError, match=name):
            Model({name: Real()}, callback, data=data, components=[MediaEffect(max_lag=2)])
    for reverse in (False, True):
        components = [MediaEffect(max_lag=2), FourierSeasonality(period=8, name="paid_media_coefficient")]
        with pytest.raises(ValueError, match="paid_media_coefficient"):
            Model({}, callback, data=data, components=components[::-1] if reverse else components)


def _total_model(data, *, transformed=False, grouped=False):
    def density(*, outcome, paid_media_total, annual, intercept, sigma):
        return normal(outcome, intercept + paid_media_total + annual, sigma) + normal(intercept, 0.0, 2.0)

    def transform(*, paid_media_total, annual, intercept):
        return {"mean": intercept + paid_media_total + annual}

    def transformed_density(*, outcome, mean, intercept, sigma):
        return normal(outcome, mean, sigma) + normal(intercept, 0.0, 2.0)

    def quantities(key, *, paid_media, paid_media_total):
        return {"channels": paid_media, "total": paid_media_total}

    def transformed_quantities(key, *, paid_media, paid_media_total, mean):
        return {"channels": paid_media, "total": paid_media_total, "mean": mean}

    return Model(
        {"intercept": Real(), "sigma": Positive()},
        transformed_density if transformed else density,
        transformed_quantities if transformed else quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2, group_specific=grouped),
            FourierSeasonality(period=8, order=1, name="annual", group_specific=grouped),
        ],
        transformed_parameters=transform if transformed else None,
    )


@pytest.mark.parametrize("transformed", [False, True])
def test_named_media_totals_preserve_density_and_all_parameter_gradients(transformed):
    data = _data(_MEDIA)
    model = _total_model(data, transformed=transformed)
    original, position = _model(data), _position()
    evaluate = jax.jit(jax.value_and_grad(model.log_density))
    actual, gradient = evaluate(position, model.data)
    expected, expected_gradient = jax.jit(jax.value_and_grad(original.log_density))(position, original.data)
    mean, paid, _, prior, sigma = _reference(position, _MEDIA, [2, 3, 4])

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
    adstock, saturation, adstock_arguments, saturation_arguments
):
    data = _data(_MEDIA)

    def transform(*, paid_media_total, intercept):
        return {"mean": intercept + paid_media_total}

    def density(*, outcome, mean, intercept):
        return normal(outcome, mean, 0.7) + normal(intercept, 0.0, 2.0)

    def quantities(key, *, paid_media, paid_media_total, mean):
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
            jnp.asarray(_MEDIA),
            **{argument: parameters[role] for role, argument in adstock_arguments.items()},
            max_lag=2,
        )
        response = saturation(
            carried, **{argument: parameters[role] for role, argument in saturation_arguments.items()}
        )
        channels = response[-len(data.time_values) :] * parameters["coefficient"]
        mean = unconstrained["intercept"] + channels.sum(axis=-1)
        value = density(outcome=data.arrays["outcome"], mean=mean, intercept=unconstrained["intercept"]) + prior
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
def test_media_totals_preserve_group_and_prediction_axes_under_jit_vmap(transformed):
    media = np.stack((_MEDIA, _MEDIA * 1.5 + 0.2), axis=1)
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

    def density(*, first_total, campaign_total_total):
        return normal(first_total + campaign_total_total, 0.0, 1.0)

    def quantities(key, *, first, first_total, campaign_total, campaign_total_total):
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


def test_media_total_names_are_reserved_even_when_callbacks_do_not_request_them():
    data = _data(_MEDIA)

    def callback(data, effects, **parameters):
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


def test_transformed_outputs_cannot_replace_unrequested_media_totals():
    model = Model(
        {},
        lambda *, mean: jnp.sum(mean),
        lambda key, *, mean: {"mean": mean},
        data=_data(_MEDIA),
        components=[MediaEffect(max_lag=2)],
        transformed_parameters=lambda: {"mean": jnp.zeros(3), "paid_media_total": jnp.zeros(3)},
    )
    position = {name: value for name, value in _position().items() if name.startswith("paid_media_")}
    with pytest.raises(ValueError, match="paid_media_total"):
        model.log_density(position, model.data)
    with pytest.raises(ValueError, match="paid_media_total"):
        model.generate(jax.random.key(0), model.constrain(position), model.data)


def test_media_total_names_do_not_replace_random_keys_or_create_fourier_totals():
    data = _data(_MEDIA)
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


@pytest.mark.parametrize("manual_annual", [False, True])
def test_direct_component_priors_match_automatic_density_and_gradients(manual_annual):
    def density(
        *,
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
        if manual_annual:
            prior += normal(annual_coefficients, 0.0, 1.0)
        return normal(outcome, intercept + paid_media.sum(-1) + annual, sigma) + normal(intercept, 0.0, 2.0) + prior

    def quantities(key, *, paid_media_coefficient, paid_media_retention, annual_coefficients, annual):
        return {
            "coefficient": paid_media_coefficient,
            "retention": paid_media_retention,
            "seasonal_coefficients": annual_coefficients,
            "annual": annual,
        }

    data, position = _data(_MEDIA), _position()
    model = Model(
        {"intercept": Real(), "sigma": Positive()},
        density,
        quantities,
        data=data,
        components=[
            MediaEffect(max_lag=2, automatic_priors=False),
            FourierSeasonality(period=8, order=1, name="annual", automatic_priors=not manual_annual),
        ],
    )
    automatic = _model(data)
    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(automatic.log_density))(position, automatic.data)
    assert set(model.parameters) == set(automatic.parameters) == set(position)
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
    np.testing.assert_allclose(generated["annual"], _reference(position, _MEDIA, [2, 3, 4])[2], rtol=4e-6, atol=2e-6)


def test_disabling_component_priors_retains_jacobians_and_legacy_callback_inputs():
    data = _data(_MEDIA)
    model = _model(data, named=False, automatic_priors=False)
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


def test_automatic_component_parameters_reach_transforms_and_forecast_generation_under_jit_vmap():
    def transform(*, paid_media_coefficient, paid_media_retention, annual_coefficients):
        return {"parameter_summary": paid_media_coefficient * paid_media_retention + annual_coefficients}

    def quantities(key, *, paid_media, annual, parameter_summary, paid_media_coefficient, annual_coefficients):
        return {
            "media": paid_media,
            "annual": annual,
            "summary": parameter_summary,
            "coefficient": paid_media_coefficient,
            "seasonal_coefficients": annual_coefficients,
        }

    media = np.stack((_MEDIA, _MEDIA * 1.5 + 0.2), axis=1)
    model = Model(
        {},
        lambda *, parameter_summary: normal(parameter_summary, 0.0, 2.0),
        quantities,
        data=_data(media),
        components=[
            MediaEffect(max_lag=2, group_specific=True),
            FourierSeasonality(period=8, order=1, name="annual", group_specific=True),
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


def test_selected_media_parameters_support_direct_priors_and_reject_inactive_roles():
    def direct_density(
        *,
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

    def component(automatic_priors):
        return MediaEffect(
            max_lag=2, adstock=weibull_cdf_adstock, saturation=root_saturation, automatic_priors=automatic_priors
        )

    data = _data(_MEDIA)
    model = Model({}, direct_density, data=data, components=[component(False)])
    automatic = Model(
        {},
        lambda *, outcome, paid_media_total: normal(outcome, paid_media_total, 0.7),
        data=data,
        components=[component(True)],
    )
    roles = ("coefficient", "adstock_shape", "adstock_scale", "exponent")
    position = {
        f"paid_media_{role}": jnp.array([-0.5 + index / 10, 0.2 + index / 5]) for index, role in enumerate(roles)
    }
    assert set(model.parameters) == set(position)
    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(automatic.log_density))(position, automatic.data)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, rtol=5e-6, atol=3e-6)
    with pytest.raises(ValueError, match="paid_media_retention"):
        Model(
            {}, lambda *, paid_media_retention: jnp.sum(paid_media_retention), data=data, components=[component(False)]
        )


def test_requested_media_parameter_names_cannot_collide_with_data():
    data = _data(_MEDIA)
    data.arrays["paid_media_coefficient"] = np.ones(3)
    with pytest.raises(ValueError, match=r"paid_media_coefficient.*ambiguous"):
        Model(
            {},
            lambda *, paid_media_coefficient: normal(paid_media_coefficient, 0.0, 1.0),
            data=data,
            components=[MediaEffect(max_lag=2)],
        )
