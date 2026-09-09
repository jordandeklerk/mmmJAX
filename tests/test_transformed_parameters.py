"""Tests for deterministic transformed parameters in prepared models."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import FourierSeasonality, MediaEffect, Model, Positive, Real, normal, normal_rng, prepare_data


def _data(controls=None, *, observed=True):
    controls = np.array([[1.0, 0.3], [0.5, 1.2], [1.5, -0.4]]) if controls is None else np.asarray(controls)
    frame = pl.DataFrame(
        {
            "time": np.arange(len(controls)),
            "first": controls[:, 0],
            "second": controls[:, 1],
            "sales": np.linspace(0.4, 1.1, len(controls)),
        }
    )
    return prepare_data(frame, time="time", outcome="sales" if observed else None, controls=["first", "second"])


def _normal(value, scale=1.0):
    return (-0.5 * (np.asarray(value, dtype=np.float64) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def _regression():
    def transformed(*, controls, beta, scale):
        return {"signal": controls @ beta, "spread": jnp.sqrt(scale**2 + 0.4)}

    def density(*, outcome, signal, spread, intercept, beta):
        return normal(outcome, signal + intercept, spread) + normal(beta, 0.0, 1.0) + normal(intercept, 0.0, 2.0)

    def quantities(key, *, signal, spread, intercept):
        return {
            "signal": signal,
            "spread": spread,
            "mean": signal + intercept,
            "draw": normal_rng(key, signal + intercept, spread),
        }

    return Model(
        {"beta": Real(shape=(2,)), "scale": Positive(), "intercept": Real()},
        density,
        quantities,
        data=_data(),
        components=[],
        transformed_parameters=transformed,
    )


def test_transformed_regression_has_correct_nonlinear_chain_rule_and_single_jacobian():
    model = _regression()
    position = {"beta": jnp.array([0.3, -0.2]), "scale": jnp.log(jnp.array(0.7)), "intercept": jnp.array(0.25)}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    controls, outcome = _data().arrays["controls"], _data().arrays["outcome"]
    beta = np.asarray(position["beta"], dtype=np.float64)
    scale, intercept = np.exp(float(position["scale"])), float(position["intercept"])
    variance = scale**2 + 0.4
    signal = controls @ beta
    residual = outcome - signal - intercept
    expected = _normal(residual, np.sqrt(variance)) + _normal(beta) + _normal(intercept, 2.0) + float(position["scale"])
    expected_gradient = {
        "beta": controls.T @ residual / variance - beta,
        "intercept": residual.sum() / variance - intercept / 4,
        "scale": scale**2 / variance * (-len(outcome) + np.sum(residual**2) / variance) + 1,
    }

    np.testing.assert_allclose(value, expected, rtol=4e-6, atol=3e-6)
    for name in position:
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)
    key = jax.random.key(5)
    generated = jax.jit(model.generate)(key, model.constrain(position), model.data)
    np.testing.assert_allclose(generated["signal"], signal, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(generated["spread"], np.sqrt(variance), rtol=3e-6)
    noise = np.asarray(jax.random.normal(key, (3,), dtype=generated["signal"].dtype))
    np.testing.assert_allclose(generated["draw"], signal + intercept + np.sqrt(variance) * noise, rtol=4e-6, atol=2e-6)


def test_transformed_outputs_change_with_positions_and_outcome_free_forecasts_under_jit_vmap():
    model = _regression()
    position = {"beta": jnp.array([0.3, -0.2]), "scale": jnp.log(jnp.array(0.7)), "intercept": jnp.array(0.25)}
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.1)), position)
    constrained = jax.vmap(model.constrain)(draws)
    generate = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))
    for controls in (np.array([[2.0, 0.5], [0.3, 1.1]]), np.array([[0.4, 1.7], [1.5, 0.2]])):
        future = _data(controls, observed=False)
        actual = generate(jax.random.split(jax.random.key(1), 2), constrained, model.prepare_data(future))
        assert "outcome" not in future.arrays
        for index in range(2):
            expected = controls @ np.asarray(draws["beta"][index], dtype=np.float64) + float(draws["intercept"][index])
            np.testing.assert_allclose(actual["mean"][index], expected, rtol=4e-6, atol=2e-6)
    assert set(model.parameters) == {"beta", "scale", "intercept"}


def _media_model(transformed):
    data = prepare_data(
        pl.DataFrame(
            {"time": [10, 11, 12], "video": [1.0, 3.0, 2.0], "search": [2.0, 1.0, 4.0], "sales": [0.4, 0.7, 1.1]}
        ),
        time="time",
        media=["video", "search"],
        outcome="sales",
    )
    model = Model(
        {"intercept": Real(), "sigma": Positive()},
        lambda *, outcome, mu, sigma: normal(outcome, mu, sigma),
        lambda key, *, mu: {"mu": mu},
        data=data,
        components=[MediaEffect(max_lag=1), FourierSeasonality(period=8, order=1, name="annual")],
        transformed_parameters=transformed,
    )
    position = {
        "intercept": jnp.array(0.2),
        "sigma": jnp.log(jnp.array(0.6)),
        "annual": jnp.array([0.1, -0.2]),
        "paid_media_coefficient": jnp.log(jnp.array([0.7, 1.1])),
        "paid_media_retention": jnp.log(jnp.array([0.2, 0.5]) / jnp.array([0.8, 0.5])),
        "paid_media_half_saturation": jnp.log(jnp.array([1.5, 2.0])),
        "paid_media_slope": jnp.log(jnp.array([1.2, 0.8])),
    }
    return model, data, position


def test_transform_combines_media_and_fourier_effects_without_duplicate_priors():
    def transformed(*, paid_media, annual, intercept):
        return {"mu": intercept + paid_media.sum(axis=-1) + annual}

    model, data, position = _media_model(transformed)
    values = {name: np.asarray(value, dtype=np.float64) for name, value in position.items()}
    retention = 1 / (1 + np.exp(-values["paid_media_retention"]))
    media = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    carried = (media + retention * np.vstack((np.zeros(2), media[:-1]))) / (1 + retention)
    half, slope = np.exp(values["paid_media_half_saturation"]), np.exp(values["paid_media_slope"])
    paid = carried**slope / (carried**slope + half**slope) * np.exp(values["paid_media_coefficient"])
    angles = 2 * np.pi * np.arange(3) / 8
    annual = np.column_stack((np.sin(angles), np.cos(angles))) @ values["annual"]
    mean = values["intercept"] + paid.sum(-1) + annual
    expected = _normal(data.arrays["outcome"] - mean, np.exp(values["sigma"])) + _normal(values["annual"])
    expected += (np.log(3) + 2 * np.log1p(-retention) + np.log(retention) + np.log1p(-retention)).sum()
    expected += values["sigma"]
    for role in ("coefficient", "half_saturation", "slope"):
        value = values["paid_media_" + role]
        expected += _normal(np.exp(value), 1.5) + value.size * np.log(2) + value.sum()

    np.testing.assert_allclose(jax.jit(model.log_density)(position, model.data), expected, rtol=5e-6, atol=3e-6)
    generated = jax.jit(model.generate)(jax.random.key(0), model.constrain(position), model.data)
    np.testing.assert_allclose(generated["mu"], mean, rtol=4e-6, atol=2e-6)


def test_construction_does_not_probe_callbacks_and_runs_the_whole_transform_for_generation():
    calls = []

    def transformed(*, outcome):
        calls.append("transformed")
        return {"summary": outcome.mean(), "unused": outcome.sum()}

    model = Model(
        {},
        lambda *, summary: summary,
        lambda key: {"constant": jnp.array(1.0)},
        data=_data(),
        components=[],
        transformed_parameters=transformed,
    )
    assert calls == []
    model.log_density({}, model.data)
    model.generate(jax.random.key(0), {}, model.data)
    assert calls == ["transformed", "transformed"]
    future = model.prepare_data(_data(observed=False))
    with pytest.raises(ValueError, match="outcome"):
        model.generate(jax.random.key(0), {}, future)


def test_unknown_density_and_generation_inputs_are_deferred_to_runtime_even_with_defaults():
    for density in (lambda *, typo: typo, lambda *, typo=1.0: jnp.asarray(typo)):
        model = Model(
            {}, density, data=_data(), components=[], transformed_parameters=lambda: {"actual": jnp.array(0.0)}
        )
        with pytest.raises(ValueError, match="typo"):
            model.log_density({}, model.data)
    model = Model(
        {},
        lambda: jnp.array(0.0),
        lambda key, *, typo=1.0: {"value": typo},
        data=_data(),
        components=[],
        transformed_parameters=lambda: {"actual": jnp.array(0.0)},
    )
    with pytest.raises(ValueError, match="typo"):
        model.generate(jax.random.key(0), {}, model.data)


def test_transformed_names_cannot_shadow_sources_even_when_unused_or_absent_in_forecast():
    def returning(name):
        def transformed(*, intercept):
            return {name: intercept}

        return transformed

    for name in ("outcome", "intercept", "annual", "paid_media", "paid_media_coefficient"):
        model, _, position = _media_model(returning(name))
        with pytest.raises(ValueError, match=name):
            model.log_density(position, model.data)
    model, _, position = _media_model(returning("outcome"))
    future = prepare_data(
        pl.DataFrame({"time": [13], "video": [2.0], "search": [1.0]}), time="time", media=["video", "search"]
    )
    with pytest.raises(ValueError, match="outcome"):
        model.generate(jax.random.key(0), model.constrain(position), model.prepare_data(future))


def test_transform_outputs_must_be_a_mapping_of_valid_names_to_array_like_values():
    def make_transform(result):
        return lambda: result

    for result in (None, [1.0], {1: 1.0}, {"bad-name": 1.0}, {"class": 1.0}, {"valid": object()}):
        model = Model(
            {},
            lambda: jnp.array(0.0),
            lambda key: {},
            data=_data(),
            components=[],
            transformed_parameters=make_transform(result),
        )
        with pytest.raises((TypeError, ValueError)):
            model.log_density({}, model.data)
        with pytest.raises((TypeError, ValueError)):
            model.generate(jax.random.key(0), {}, model.data)


def test_empty_transforms_and_python_scalar_list_boolean_outputs_are_supported():
    empty = Model(
        {},
        lambda: jnp.array(1.5),
        lambda key: {"constant": 2.5},
        data=_data(),
        components=[],
        transformed_parameters=lambda: {},
    )
    np.testing.assert_array_equal(jax.jit(empty.log_density)({}, empty.data), 1.5)
    np.testing.assert_array_equal(jax.jit(empty.generate)(jax.random.key(0), {}, empty.data)["constant"], 2.5)

    def transformed():
        return {"constant": 2.5, "values": [1.0, 2.0], "mask": [True, False]}

    def density(*, constant, values, mask):
        return constant + jnp.where(mask, values, 0.0).sum()

    def quantities(key, *, constant, values, mask):
        return {"constant": constant, "values": values, "mask": mask}

    model = Model({}, density, quantities, data=_data(), components=[], transformed_parameters=transformed)
    np.testing.assert_array_equal(jax.jit(model.log_density)({}, model.data), 3.5)
    generated = jax.jit(model.generate)(jax.random.key(0), {}, model.data)
    assert generated["constant"].shape == ()
    assert generated["mask"].dtype == jnp.bool_
    np.testing.assert_array_equal(generated["values"], [1.0, 2.0])
    np.testing.assert_array_equal(generated["mask"], [True, False])


def test_parameter_usage_is_checked_across_transform_and_density_not_generation():
    with pytest.raises(ValueError, match="unused"):
        Model(
            {"unused": Real()},
            lambda: jnp.array(0.0),
            lambda key, *, unused: {"unused": unused},
            data=_data(),
            components=[],
            transformed_parameters=lambda: {},
        )
    with pytest.raises(ValueError, match="paid_media_retention"):
        _media_model(lambda *, paid_media_retention, intercept: {"mu": paid_media_retention + intercept})
    with pytest.raises(ValueError, match="second_stage"):
        Model(
            {},
            lambda *, first_stage: first_stage,
            data=_data(),
            components=[],
            transformed_parameters=lambda *, second_stage: {"first_stage": second_stage},
        )


def test_transformed_stage_requires_prepared_named_callbacks_and_one_callable():
    for transformed in ([lambda: {}], 0.5):
        with pytest.raises(TypeError):
            Model({}, lambda: jnp.array(0.0), data=_data(), components=[], transformed_parameters=transformed)
    with pytest.raises((TypeError, ValueError)):
        Model({}, lambda: jnp.array(0.0), transformed_parameters=lambda: {})
    with pytest.raises(TypeError):
        Model({}, lambda data, effects: jnp.array(0.0), data=_data(), components=[], transformed_parameters=lambda: {})
    with pytest.raises(TypeError):
        Model(
            {},
            lambda: jnp.array(0.0),
            lambda key, data, effects: {},
            data=_data(),
            components=[],
            transformed_parameters=lambda: {},
        )
    for transformed in (lambda controls: {}, lambda **values: {}, lambda *values: {}):
        with pytest.raises(TypeError):
            Model({}, lambda: jnp.array(0.0), data=_data(), components=[], transformed_parameters=transformed)
