"""Tests for composing learned media and seasonal effects in public models."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import FourierSeasonality, MediaEffect, Model, Positive, Real, normal, normal_rng, prepare_data

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


def _model(data, *, named=True, grouped=False):
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
            MediaEffect(max_lag=2, group_specific=grouped),
            FourierSeasonality(period=8, order=1, name="annual", group_specific=grouped),
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
