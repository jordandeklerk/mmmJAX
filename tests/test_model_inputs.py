"""Tests for binding prepared model inputs to keyword-only callbacks."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import FourierSeasonality, Model, Positive, Real, normal, normal_rng, prepare_data


def _data():
    return prepare_data(
        pl.DataFrame({"time": [10.0, 11.0, 13.0, 15.0], "sales": [0.4, 1.0, 1.4, 0.7]}), time="time", outcome="sales"
    )


def _features(elapsed):
    angles = 2 * np.pi * np.asarray(elapsed, dtype=np.float64) / 8
    return np.column_stack((np.sin(angles), np.cos(angles)))


def _normal(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    return (-0.5 * (values / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def _density(*, outcome, annual, intercept, sigma):
    return normal(outcome, annual + intercept, sigma) + normal(intercept, 0.0, 2.0)


def _model(density=_density, generate=None):
    return Model(
        {"intercept": Real(), "sigma": Positive()},
        density,
        generate,
        data=_data(),
        components=[FourierSeasonality(period=8, order=1, name="annual")],
    )


def test_named_inputs_provide_evaluated_curves_and_count_priors_and_jacobian_once():
    def quantities(rng, /, *, annual, intercept, sigma):
        return {"mean": annual + intercept, "draw": normal_rng(rng, annual + intercept, sigma)}

    model = _model(generate=quantities)
    position = {"annual": jnp.array([0.3, -0.2]), "intercept": jnp.array(0.25), "sigma": jnp.log(jnp.array(0.7))}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    features = _features([0, 1, 3, 5])
    coefficients = np.asarray(position["annual"], dtype=np.float64)
    intercept, log_scale = float(position["intercept"]), float(position["sigma"])
    scale = np.exp(log_scale)
    mean = features @ coefficients + intercept
    residual = np.array([0.4, 1.0, 1.4, 0.7]) - mean
    expected = _normal(residual, scale) + _normal(coefficients) + _normal(intercept, 2.0) + log_scale
    expected_gradient = {
        "annual": features.T @ residual / scale**2 - coefficients,
        "intercept": residual.sum() / scale**2 - intercept / 4,
        "sigma": np.sum(residual**2) / scale**2 - 4 + 1,
    }

    assert model.parameters["annual"].shape == (2,)
    np.testing.assert_allclose(value, expected, rtol=4e-6, atol=3e-6)
    for name in position:
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)
    key = jax.random.key(13)
    constrained = model.constrain(position)
    generated = jax.jit(model.generate)(key, constrained, model.data)
    assert generated["mean"].shape == (4,)
    np.testing.assert_allclose(generated["mean"], mean, rtol=4e-6, atol=2e-6)
    noise = np.asarray(jax.random.normal(key, (4,), dtype=constrained["sigma"].dtype))
    np.testing.assert_allclose(generated["draw"], mean + scale * noise, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("named_density", [False, True])
def test_positional_and_named_callbacks_mix_and_generation_requests_a_parameter_subset(named_density):
    def positional_density(data, effects, intercept, sigma):
        return _density(outcome=data["outcome"], annual=effects["annual"], intercept=intercept, sigma=sigma)

    def named_generation(random_key, *, annual, intercept):
        return {"mean": annual + intercept}

    def positional_generation(key, data, effects, intercept):
        return {"mean": effects["annual"] + intercept}

    model = _model(
        _density if named_density else positional_density, positional_generation if named_density else named_generation
    )
    position = {"annual": jnp.array([0.3, -0.2]), "intercept": jnp.array(0.25), "sigma": jnp.array(0.0)}
    generated = jax.jit(model.generate)(jax.random.key(0), model.constrain(position), model.data)
    expected = _features([0, 1, 3, 5]) @ np.array([0.3, -0.2]) + 0.25
    np.testing.assert_allclose(generated["mean"], expected, rtol=4e-6, atol=2e-6)
    assert np.isfinite(jax.jit(model.log_density)(position, model.data))


def test_named_forecast_inputs_align_groups_channels_controls_and_retain_training_phase():
    frame = pl.DataFrame(
        {
            "time": [10, 10, 11, 11],
            "region": ["west", "east"] * 2,
            "video": [1, 2, 3, 4],
            "search": [5, 6, 7, 8],
            "temperature": [8, 7, 6, 5],
            "price": [1, 2, 3, 4],
            "sales": [0.3, 0.5, 0.7, 0.9],
        }
    )
    training = prepare_data(
        frame,
        time="time",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        controls=["temperature", "price"],
    )

    def mean(annual, media, controls, media_beta, control_beta):
        return annual + jnp.sum(media * media_beta + controls * control_beta, axis=-1)

    def density(*, outcome, annual, media, controls, media_beta, control_beta):
        return normal(outcome, mean(annual, media, controls, media_beta, control_beta), 1.0)

    def quantities(key, *, annual, media, controls, media_beta, control_beta):
        return {"mean": mean(annual, media, controls, media_beta, control_beta), "annual": annual}

    model = Model(
        {"media_beta": Real(shape=(2, 2)), "control_beta": Real(shape=(2, 2))},
        density,
        quantities,
        data=training,
        components=[FourierSeasonality(period=8, order=1, name="annual", group_specific=True)],
    )
    future = prepare_data(
        pl.DataFrame(
            {
                "time": [12, 12, 14, 14],
                "region": ["east", "west"] * 2,
                "video": [6, 5, 8, 7],
                "search": [10, 9, 12, 11],
                "temperature": [3, 4, 1, 2],
                "price": [6, 5, 8, 7],
            }
        ),
        time="time",
        groups=["region"],
        media=["search", "video"],
        controls=["price", "temperature"],
    )
    parameters = {
        "annual": jnp.array([[0.3, -0.2], [0.7, 1.1]]),
        "media_beta": jnp.array([[0.2, 0.3], [-0.1, 0.4]]),
        "control_beta": jnp.array([[0.5, -0.2], [0.3, 0.1]]),
    }
    actual = jax.jit(model.generate)(jax.random.key(0), parameters, model.prepare_data(future))
    media = np.array([[[5, 9], [6, 10]], [[7, 11], [8, 12]]], dtype=np.float64)
    controls = np.array([[[4, 5], [3, 6]], [[2, 7], [1, 8]]], dtype=np.float64)
    annual = _features([2, 4]) @ np.asarray(parameters["annual"], dtype=np.float64)
    expected = annual + (
        media * np.asarray(parameters["media_beta"]) + controls * np.asarray(parameters["control_beta"])
    ).sum(-1)

    assert "outcome" not in future.arrays
    assert future.group_values == (("east",), ("west",))
    np.testing.assert_allclose(actual["annual"], annual, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(actual["mean"], expected, rtol=4e-6, atol=3e-6)


def test_named_callbacks_work_with_empty_components_and_require_requested_inputs_despite_defaults():
    def density(*, outcome=0.0):
        return normal(outcome, 0.0, 1.0)

    def quantities(key, *, outcome=0.0):
        return {"observed": outcome}

    model = Model({}, density, quantities, data=_data(), components=[])
    np.testing.assert_allclose(jax.jit(model.log_density)({}, model.data), _normal([0.4, 1.0, 1.4, 0.7]), rtol=3e-6)
    future = model.prepare_data(prepare_data(pl.DataFrame({"time": [17, 18]}), time="time"))
    with pytest.raises(ValueError, match="outcome"):
        model.log_density({}, future)
    with pytest.raises(ValueError, match="outcome"):
        model.generate(jax.random.key(0), {}, future)


def test_named_density_must_request_every_user_parameter():
    with pytest.raises(ValueError, match="intercept"):
        Model({"intercept": Real()}, lambda *, outcome: normal(outcome, 0.0, 1.0), data=_data(), components=[])


def test_undeclared_names_are_rejected_even_when_defaulted_or_matching_source_columns():
    for density in (lambda *, sales: jnp.array(0.0), lambda *, outcome, unknown=0.0: jnp.array(0.0)):
        with pytest.raises(ValueError, match=r"sales|unknown"):
            Model({}, density, data=_data(), components=[])
    with pytest.raises(ValueError, match="unknown"):
        Model({}, lambda *, outcome: jnp.array(0.0), lambda key, *, unknown=0.0: {}, data=_data(), components=[])


def test_named_binding_rejects_collisions_between_data_components_and_parameters():
    cases = (
        ({"outcome": Real()}, []),
        ({}, [FourierSeasonality(period=8, name="outcome")]),
        ({"annual": Real()}, [FourierSeasonality(period=8, name="annual")]),
    )
    for parameters, components in cases:
        with pytest.raises(ValueError, match=r"ambig|colli|conflict"):
            Model(parameters, lambda *, outcome: jnp.array(0.0), data=_data(), components=components)

    def positional_density(data, effects, outcome):
        return normal(data["outcome"], outcome, 1.0)

    with pytest.raises(ValueError, match=r"ambig|colli|conflict"):
        Model(
            {"outcome": Real()},
            positional_density,
            lambda key, *, outcome: {"value": outcome},
            data=_data(),
            components=[],
        )


def test_named_callbacks_reject_variadic_parameters_and_a_keyword_only_random_key():
    densities = (lambda *, outcome, **values: jnp.array(0.0), lambda *values, outcome: jnp.array(0.0))
    for density in densities:
        with pytest.raises(TypeError):
            Model({}, density, data=_data(), components=[])
    generators = (
        lambda key, *, outcome, **values: {},
        lambda key, *values, outcome: {},
        lambda *, key, outcome: {},
        lambda outcome: {},
    )
    for generate in generators:
        with pytest.raises(TypeError):
            Model({}, lambda *, outcome: jnp.array(0.0), generate, data=_data(), components=[])


def test_keyword_only_input_resolution_is_not_enabled_for_legacy_models():
    with pytest.raises(TypeError):
        Model({}, lambda *, outcome: normal(outcome, 0.0, 1.0))


def test_unrequested_name_collisions_preserve_mixed_callback_modes():
    def density(data, effects, outcome):
        return normal(data["outcome"], outcome + effects["annual"], 1.0)

    model = Model(
        {"outcome": Real()},
        density,
        lambda key, *, annual: {"annual": annual},
        data=_data(),
        components=[FourierSeasonality(period=8, order=1, name="annual")],
    )
    parameters = {"outcome": jnp.array(0.2), "annual": jnp.array([0.3, -0.2])}
    generated = jax.jit(model.generate)(jax.random.key(0), parameters, model.data)
    np.testing.assert_allclose(
        generated["annual"], _features([0, 1, 3, 5]) @ np.array([0.3, -0.2]), rtol=4e-6, atol=2e-6
    )
    assert np.isfinite(model.log_density(parameters, model.data))


def test_prepared_callbacks_may_request_no_named_inputs():
    model = Model(
        {},
        lambda: jnp.array(1.25),
        lambda key: {"constant": jnp.array(3.0)},
        data=_data(),
        components=[FourierSeasonality(period=8, order=1, name="annual")],
    )
    parameters = {"annual": jnp.array([0.3, -0.2])}

    np.testing.assert_allclose(
        jax.jit(model.log_density)(parameters, model.data), 1.25 + _normal(parameters["annual"]), rtol=3e-6
    )
    generated = jax.jit(model.generate)(jax.random.key(0), parameters, model.data)
    np.testing.assert_array_equal(generated["constant"], 3.0)


def test_fourier_coefficient_aliases_follow_custom_names_without_changing_parameter_keys():
    def density(*, outcome, annual_coefficients, annual_coefficients_coefficients):
        return normal(outcome, annual_coefficients, 0.7) + normal(annual_coefficients_coefficients, 0.0, 0.8)

    def quantities(key, *, annual_coefficients, annual_coefficients_coefficients):
        return {"curve": annual_coefficients, "coefficients": annual_coefficients_coefficients}

    data = _data()
    model = Model(
        {},
        density,
        quantities,
        data=data,
        components=[FourierSeasonality(period=8, order=1, name="annual_coefficients", automatic_priors=False)],
    )
    position = {"annual_coefficients": jnp.array([0.3, -0.2])}
    features = _features([0, 1, 3, 5])
    coefficients = np.asarray(position["annual_coefficients"], dtype=np.float64)
    residual = data.arrays["outcome"] - features @ coefficients
    actual, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    assert set(model.parameters) == {"annual_coefficients"}
    np.testing.assert_allclose(actual, _normal(residual, 0.7) + _normal(coefficients, 0.8), rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(
        gradient["annual_coefficients"],
        features.T @ residual / 0.7**2 - coefficients / 0.8**2,
        rtol=5e-6,
        atol=3e-6,
    )
    generated = jax.jit(model.generate)(jax.random.key(0), position, model.data)
    np.testing.assert_allclose(generated["curve"], features @ coefficients, rtol=4e-6, atol=2e-6)
    np.testing.assert_array_equal(generated["coefficients"], position["annual_coefficients"])


def test_requested_fourier_coefficient_aliases_reject_parameter_component_and_data_collisions():
    seasonal = FourierSeasonality(period=8, order=1, name="annual")
    cases = (
        ({"annual_coefficients": Real()}, [seasonal], _data()),
        ({}, [seasonal, FourierSeasonality(period=8, order=1, name="annual_coefficients")], _data()),
    )
    data_collision = _data()
    data_collision.arrays["annual_coefficients"] = np.ones(4)
    for parameters, components, data in (*cases, ({}, [seasonal], data_collision)):
        with pytest.raises(ValueError, match=r"annual_coefficients.*ambiguous"):
            Model(
                parameters,
                lambda *, annual_coefficients: normal(annual_coefficients, 0.0, 1.0),
                data=data,
                components=components,
            )


def test_unrequested_fourier_coefficient_aliases_preserve_legacy_parameter_bindings():
    def density(data, effects, annual_coefficients):
        assert set(effects) == {"annual"}
        return normal(data["outcome"], effects["annual"] + annual_coefficients, 1.0)

    def quantities(key, *, annual):
        return {"annual": annual}

    model = Model(
        {"annual_coefficients": Real()},
        density,
        quantities,
        data=_data(),
        components=[FourierSeasonality(period=8, order=1, name="annual")],
    )
    position = {"annual": jnp.array([0.3, -0.2]), "annual_coefficients": jnp.array(0.25)}
    curve = _features([0, 1, 3, 5]) @ np.asarray(position["annual"], dtype=np.float64)
    expected = _normal(_data().arrays["outcome"] - curve - 0.25) + _normal(position["annual"])
    np.testing.assert_allclose(jax.jit(model.log_density)(position, model.data), expected, rtol=4e-6, atol=3e-6)
    generated = jax.jit(model.generate)(jax.random.key(0), position, model.data)
    np.testing.assert_allclose(generated["annual"], curve, rtol=4e-6, atol=2e-6)


def test_transformed_outputs_cannot_shadow_unrequested_fourier_coefficient_aliases():
    model = Model(
        {},
        lambda *, summary: jnp.sum(summary),
        lambda key, *, summary: {"summary": summary},
        data=_data(),
        components=[FourierSeasonality(period=8, order=1, name="annual")],
        transformed_parameters=lambda: {"summary": jnp.ones(4), "annual_coefficients": jnp.ones(2)},
    )
    position = {"annual": jnp.array([0.3, -0.2])}
    with pytest.raises(ValueError, match="annual_coefficients"):
        model.log_density(position, model.data)
    with pytest.raises(ValueError, match="annual_coefficients"):
        model.generate(jax.random.key(0), position, model.data)


def test_fourier_coefficient_aliases_cannot_replace_the_generation_random_key():
    with pytest.raises(TypeError, match=r"annual_coefficients.*random key"):
        Model(
            {},
            lambda *, annual: normal(annual, 0.0, 1.0),
            lambda annual_coefficients, *, annual: {"annual": annual},
            data=_data(),
            components=[FourierSeasonality(period=8, order=1, name="annual")],
        )
