"""Tests for automatic component preparation in JAX probability models."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import FourierSeasonality, Model, Positive, Real, normal, normal_rng, prepare_data


def _features(time, period, order):
    angles = 2 * np.pi * np.asarray(time, dtype=np.float64)[:, None] * np.arange(1, order + 1) / period
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def _normal(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    return (-0.5 * (values / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def _national_model(generate=None):
    data = prepare_data(
        pl.DataFrame({"time": [10.0, 11.0, 13.0, 15.0], "sales": [0.4, 1.0, 1.4, 0.7]}),
        time="time",
        outcome="sales",
    )

    def custom_prior(coefficients):
        return -0.5 * jnp.square(coefficients / 0.6) - jnp.log(0.6) - 0.5 * jnp.log(2 * jnp.pi)

    def log_density(data, effects, intercept, sigma):
        mean = intercept + effects["annual"] + effects["promotion"]
        return normal(data["outcome"], mean, sigma) + normal(intercept, 0.0, 2.0)

    model = Model(
        {"intercept": Real(), "sigma": Positive()},
        log_density,
        generate,
        data=data,
        components=[
            FourierSeasonality(period=8, order=1, name="annual"),
            FourierSeasonality(period=12, order=2, name="promotion", prior=custom_prior),
        ],
    )
    position = {
        "annual": jnp.array([0.3, -0.2]),
        "promotion": jnp.array([0.1, -0.3, 0.2, 0.4]),
        "intercept": jnp.array(0.25),
        "sigma": jnp.log(jnp.array(0.7)),
    }
    return model, data, position


def test_composition_adds_two_component_priors_and_positive_jacobian_exactly_once():
    model, data, position = _national_model()
    annual, promotion = _features([0, 1, 3, 5], 8, 1), _features([0, 1, 3, 5], 12, 2)
    outcome = np.asarray(data.arrays["outcome"], dtype=np.float64)

    def reference(values):
        scale = np.exp(values["sigma"])
        residual = outcome - values["intercept"] - annual @ values["annual"] - promotion @ values["promotion"]
        return (
            _normal(residual, scale)
            + _normal(values["intercept"], 2.0)
            + _normal(values["annual"])
            + _normal(values["promotion"], 0.6)
            + values["sigma"]
        )

    values = jax.tree.map(lambda value: np.asarray(value, dtype=np.float64), position)
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    scale = np.exp(values["sigma"])
    residual = outcome - values["intercept"] - annual @ values["annual"] - promotion @ values["promotion"]
    expected_gradient = {
        "intercept": residual.sum() / scale**2 - values["intercept"] / 4,
        "sigma": -len(outcome) + np.sum(residual**2) / scale**2 + 1,
        "annual": annual.T @ residual / scale**2 - values["annual"],
        "promotion": promotion.T @ residual / scale**2 - values["promotion"] / 0.6**2,
    }
    np.testing.assert_allclose(value, reference(values), rtol=4e-6, atol=3e-6)
    for name in position:
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.1)), position)
    actual = jax.jit(jax.vmap(model.log_density, in_axes=(0, None)))(draws, model.data)
    expected = [
        reference({name: np.asarray(value[index], dtype=np.float64) for name, value in draws.items()})
        for index in range(2)
    ]
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-6)


@pytest.mark.parametrize("mapping_callback", [False, True])
def test_generation_receives_effects_and_only_requested_user_parameters(mapping_callback):
    def subset(key, data, effects, intercept):
        return {"mean": intercept + effects["annual"] + effects["promotion"], "observed": data["outcome"]}

    def mapping(key, data, effects, **parameters):
        assert set(parameters) == {"intercept", "sigma"}
        assert set(effects) == {"annual", "promotion"}
        mean = parameters["intercept"] + effects["annual"] + effects["promotion"]
        return {"mean": mean, "draw": normal_rng(key, mean, parameters["sigma"])}

    model, data, position = _national_model(mapping if mapping_callback else subset)
    constrained = model.constrain(position)
    key = jax.random.key(17)
    result = jax.jit(model.generate)(key, constrained, model.data)
    expected = (
        float(position["intercept"])
        + _features([0, 1, 3, 5], 8, 1) @ np.asarray(position["annual"])
        + _features([0, 1, 3, 5], 12, 2) @ np.asarray(position["promotion"])
    )

    np.testing.assert_allclose(result["mean"], expected, rtol=4e-6, atol=2e-6)
    if mapping_callback:
        noise = np.asarray(jax.random.normal(key, (4,), dtype=constrained["sigma"].dtype))
        np.testing.assert_allclose(result["draw"], expected + float(constrained["sigma"]) * noise, rtol=4e-6, atol=2e-6)
    else:
        np.testing.assert_array_equal(result["observed"], data.arrays["outcome"].astype(result["observed"].dtype))


@pytest.mark.parametrize("group_specific", [False, True])
def test_grouped_composition_preserves_effect_shapes_and_counts_shared_priors_once(group_specific):
    data = prepare_data(
        pl.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-03", "2026-01-06"] * 2,
                "region": ["west"] * 3 + ["east"] * 3,
                "sales": [0.3, 0.7, 1.1, 0.5, 0.9, 1.3],
            }
        ),
        time="date",
        groups=["region"],
        outcome="sales",
    )

    def log_density(data, effects):
        return normal(data["outcome"], effects["seasonality"], 1.0)

    def generate(key, data, effects):
        return {"seasonality": effects["seasonality"]}

    model = Model(
        {},
        log_density,
        generate,
        data=data,
        components=[FourierSeasonality(period="weekly", order=1, group_specific=group_specific)],
    )
    coefficients = jnp.array([[0.3, -0.2], [0.7, 1.1]]) if group_specific else jnp.array([0.3, 0.7])
    position = {"seasonality": coefficients}
    features = _features([0, 2, 5], 7, 1)
    expected = features @ np.asarray(coefficients, dtype=np.float64)
    if not group_specific:
        expected = np.broadcast_to(expected[:, None], (3, 2))
    residual = np.asarray(data.arrays["outcome"], dtype=np.float64) - expected
    expected_gradient = features.T @ (residual if group_specific else residual.sum(axis=1)) - np.asarray(coefficients)

    assert model.parameters["seasonality"].shape == ((2, 2) if group_specific else (2,))
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    generated = jax.jit(model.generate)(jax.random.key(0), model.constrain(position), model.data)
    assert generated["seasonality"].shape == (3, 2)
    np.testing.assert_allclose(generated["seasonality"], expected, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(value, _normal(residual) + _normal(coefficients), rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(gradient["seasonality"], expected_gradient, rtol=5e-6, atol=3e-6)


def test_all_component_parameters_participate_in_initialization_and_transforms():
    model, _, position = _national_model()
    initialized = jax.jit(model.initialize_random)(jax.random.key(3))
    assert list(model.parameters) == ["annual", "intercept", "promotion", "sigma"]
    assert set(initialized) == set(position)
    assert all(isinstance(model.parameters[name], Real) for name in ("annual", "promotion"))
    for name, parameter in model.parameters.items():
        assert initialized[name].shape == parameter.position_shape
        assert np.isfinite(initialized[name]).all()
    round_trip = model.unconstrain(model.constrain(position))
    for name in position:
        np.testing.assert_allclose(round_trip[name], position[name], rtol=2e-6, atol=2e-6)


def test_complete_parameter_names_and_shapes_are_required_even_for_subset_generation():
    model, _, position = _national_model(lambda key, data, effects: {"mean": effects["annual"]})
    missing = {name: value for name, value in position.items() if name != "promotion"}
    extra = {**position, "unexpected": jnp.array(0.0)}
    malformed = {**position, "annual": jnp.zeros(3)}
    for values in (missing, extra, malformed):
        for operation in (
            lambda values: model.constrain(values),
            lambda values: model.log_density(values, model.data),
            lambda values: model.generate(jax.random.key(0), values, model.data),
        ):
            with pytest.raises(ValueError):
                operation(values)


def test_empty_components_use_effect_callback_and_omitted_components_keep_legacy_callbacks():
    data = prepare_data(pl.DataFrame({"time": [0, 1], "sales": [0.3, 0.7]}), time="time", outcome="sales")

    def composed(data, effects, **parameters):
        assert effects == {}
        assert set(parameters) == {"location"}
        return normal(data["outcome"], parameters["location"], 1.0)

    def legacy(data, location):
        return normal(data["outcome"], location, 1.0)

    position = {"location": jnp.array(0.2)}
    model = Model({"location": Real()}, composed, data=data, components=[])
    old_model = Model({"location": Real()}, legacy)
    actual = jax.jit(model.log_density)(position, model.data)
    expected = old_model.log_density(position, data._to_jax())
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_composed_model_snapshots_prepared_arrays_and_component_selection():
    data = prepare_data(pl.DataFrame({"time": [0, 1], "sales": [0.3, 0.7]}), time="time", outcome="sales")
    components = [FourierSeasonality(period=7, order=1)]
    parameters = {"location": Real()}

    def log_density(data, effects, **values):
        assert set(values) == {"location"}
        return normal(data["outcome"], values["location"] + effects["seasonality"], 1.0)

    model = Model(parameters, log_density, data=data, components=components)
    position = {"location": jnp.array(0.2), "seasonality": jnp.array([0.3, -0.7])}
    expected = model.log_density(position, model.data)
    data.arrays["outcome"][:] = 99
    components.clear()
    parameters.clear()
    returned_parameters = model.parameters
    returned_parameters.clear()

    assert set(model.parameters) == {"location", "seasonality"}
    np.testing.assert_allclose(model.log_density(position, model.data), expected, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(model.log_density)(position, model.data), expected, rtol=2e-6)


def test_composition_rejects_duplicate_names_and_user_parameter_collisions():
    _, data, _ = _national_model()
    for parameters, components in (
        ({}, [FourierSeasonality(period=7), FourierSeasonality(period=12)]),
        ({"seasonality": Real()}, [FourierSeasonality(period=7)]),
    ):
        with pytest.raises(ValueError):
            Model(parameters, lambda data, effects, **values: jnp.array(0.0), data=data, components=components)


def test_composition_requires_prepared_data_and_supported_component_instances():
    _, data, _ = _national_model()

    def callback(data, effects):
        return jnp.array(0.0)

    with pytest.raises((TypeError, ValueError)):
        Model({}, callback, components=[FourierSeasonality(period=7)])
    for invalid_data in ({"outcome": jnp.ones(2)}, pl.DataFrame({"time": [0]})):
        with pytest.raises(TypeError):
            Model({}, callback, data=invalid_data, components=[])
    for components in ([object()], [FourierSeasonality], "seasonality"):
        with pytest.raises(TypeError):
            Model({}, callback, data=data, components=components)


def test_component_callbacks_validate_user_parameters_and_required_effect_argument():
    _, data, _ = _national_model()
    bad_densities = (
        lambda data, location: jnp.array(0.0),
        lambda data, *, effects, location: jnp.array(0.0),
        lambda data, effects: jnp.array(0.0),
        lambda data, effects, location, seasonality: jnp.array(0.0),
        lambda data, effects, location, **parameters: jnp.array(0.0),
    )
    for callback in bad_densities:
        with pytest.raises((TypeError, ValueError)):
            Model({"location": Real()}, callback, data=data, components=[FourierSeasonality(period=7)])
    for callback in (lambda key, data, location: {}, lambda key, data, effects, seasonality: {}):
        with pytest.raises((TypeError, ValueError)):
            Model(
                {"location": Real()},
                lambda data, effects, location: jnp.array(0.0),
                callback,
                data=data,
                components=[FourierSeasonality(period=7)],
            )


def test_component_priors_do_not_hide_an_invalid_integer_callback_density():
    _, data, _ = _national_model()
    model = Model({}, lambda data, effects: jnp.array(1), data=data, components=[FourierSeasonality(period=7, order=1)])

    with pytest.raises(TypeError):
        model.log_density({"seasonality": jnp.array([0.3, -0.7])}, model.data)


def test_runtime_inputs_must_preserve_component_training_origin_and_group_identity():
    component = FourierSeasonality(period=7, order=1, group_specific=True)

    def build(times, groups, specification):
        data = prepare_data(
            pl.DataFrame(
                {
                    "time": [time for time in times for _ in groups],
                    "region": list(groups) * len(times),
                    "sales": [0.5] * (len(times) * len(groups)),
                }
            ),
            time="time",
            groups=["region"],
            outcome="sales",
        )
        model = Model(
            {},
            lambda data, effects: normal(data["outcome"], effects["seasonality"], 1.0),
            lambda key, data, effects: {"seasonality": effects["seasonality"]},
            data=data,
            components=[specification],
        )
        return model, data

    model, data = build((0, 1), ("west", "east"), component)
    foreign, _ = build((0, 1), ("west", "east"), FourierSeasonality(period=7, order=1, group_specific=True))
    reordered, _ = build((0, 1), ("east", "west"), component)
    shifted, _ = build((10, 11), ("west", "east"), component)
    position = {"seasonality": jnp.array([[0.3, -0.2], [0.7, 1.1]])}
    constrained = model.constrain(position)

    for supplied, error in (
        (data, TypeError),
        (data._to_jax(), TypeError),
        (foreign.data, ValueError),
        (reordered.data, ValueError),
        (shifted.data, ValueError),
    ):
        with pytest.raises(error):
            model.log_density(position, supplied)
        with pytest.raises(error):
            model.generate(jax.random.key(0), constrained, supplied)


def test_generation_skips_priors_and_accepts_renamed_positional_only_inputs():
    data = prepare_data(pl.DataFrame({"time": [10, 11], "sales": [0.3, 0.7]}), time="time", outcome="sales")

    def prior(coefficients):
        raise AssertionError("Generation must not evaluate coefficient priors")

    def log_density(observations, contributions, /):
        return normal(observations["outcome"], contributions["seasonality"], 1.0)

    def generate(rng, observations, contributions, /):
        return {"effect": contributions["seasonality"]}

    model = Model({}, log_density, generate, data=data, components=[FourierSeasonality(period=7, order=1, prior=prior)])
    coefficients = jnp.array([0.3, -0.7])
    constrained = model.constrain({"seasonality": coefficients})
    result = jax.jit(model.generate)(jax.random.key(0), constrained, model.data)

    np.testing.assert_allclose(
        result["effect"], _features([0, 1], 7, 1) @ np.asarray(coefficients), rtol=3e-6, atol=2e-6
    )


def _prediction_data(times, *, reverse=False, outcome=True):
    groups = ("east", "west") if reverse else ("west", "east")
    rows = []
    for time in times:
        for group in groups:
            index = int(group == "east")
            rows.append(
                {
                    "time": time,
                    "region": group,
                    "video": 10 * time + index,
                    "search": 2 * time + 3 * index,
                    "video_cost": time + 0.1 * index,
                    "search_cost": 0.4 * time + 0.6 * index,
                    "temperature": time + 4 * index,
                    "price": 2 * time - index,
                    "residents": 100 + 100 * index,
                    "sales": 0.2 * time + index,
                }
            )
    media, spend, controls = ["video", "search"], ["video_cost", "search_cost"], ["temperature", "price"]
    if reverse:
        media, spend, controls = media[::-1], spend[::-1], controls[::-1]
    return prepare_data(
        pl.DataFrame(rows),
        time="time",
        groups=["region"],
        outcome="sales" if outcome else None,
        media=media,
        spend=spend,
        controls=controls,
        population="residents",
        channels=media,
    )


@pytest.fixture
def prediction_model():
    training = _prediction_data((10, 11))

    def mean(data, effects, parameters):
        return (
            effects["seasonality"]
            + jnp.sum(data["media"] * parameters["media_beta"], axis=-1)
            + jnp.sum(data["spend"] * parameters["spend_beta"], axis=-1)
            + jnp.sum(data["controls"] * parameters["control_beta"], axis=-1)
            + data["population"] * parameters["population_beta"]
        )

    def density(data, effects, **parameters):
        return normal(data["outcome"], mean(data, effects, parameters), 1.0)

    def generate(key, data, effects, **parameters):
        return {"mean": mean(data, effects, parameters), "seasonality": effects["seasonality"]}

    model = Model(
        {
            "media_beta": Real(shape=(2, 2)),
            "spend_beta": Real(shape=(2, 2)),
            "control_beta": Real(shape=(2, 2)),
            "population_beta": Real(shape=(2,)),
        },
        density,
        generate,
        data=training,
        components=[FourierSeasonality(period=7, order=1, group_specific=True)],
    )
    coefficients = {
        "media_beta": [[0.02, -0.01], [0.03, 0.015]],
        "spend_beta": [[0.01, 0.02], [-0.015, 0.025]],
        "control_beta": [[0.1, -0.05], [0.07, 0.03]],
        "population_beta": [0.001, -0.002],
        "seasonality": [[0.3, -0.2], [0.7, 1.1]],
    }
    return model, training, {name: jnp.asarray(value) for name, value in coefficients.items()}


def _prediction_reference(times, coefficients):
    time, group = np.asarray(times, dtype=np.float64)[:, None], np.arange(2)[None, :]
    media = np.stack((10 * time + group, 2 * time + 3 * group), axis=-1)
    spend = np.stack((time + 0.1 * group, 0.4 * time + 0.6 * group), axis=-1)
    controls = np.stack((time + 4 * group, 2 * time - group), axis=-1)
    coefficients = {name: np.asarray(value, dtype=np.float64) for name, value in coefficients.items()}
    effect = _features(np.asarray(times) - 10, 7, 1) @ coefficients["seasonality"]
    mean = (
        effect
        + (media * coefficients["media_beta"]).sum(axis=-1)
        + (spend * coefficients["spend_beta"]).sum(axis=-1)
        + (controls * coefficients["control_beta"]).sum(axis=-1)
        + np.array([100, 200]) * coefficients["population_beta"]
    )
    return mean, effect


def test_prediction_aligns_all_inputs_and_retains_training_phase_across_calls(prediction_model):
    model, _, coefficients = prediction_model
    generate = jax.jit(model.generate)
    for times in ((12, 13), (15, 17)):
        future = _prediction_data(times, reverse=True, outcome=False)
        inputs = model.prepare_data(future)
        actual = generate(jax.random.key(0), coefficients, inputs)
        expected_mean, expected_effect = _prediction_reference(times, coefficients)
        assert "outcome" not in future.arrays
        assert future.group_values == (("east",), ("west",))
        assert future.columns["media"] == ("search", "video")
        np.testing.assert_allclose(actual["mean"], expected_mean, rtol=4e-6, atol=3e-6)
        np.testing.assert_allclose(actual["seasonality"], expected_effect, rtol=4e-6, atol=3e-6)

    observed = model.prepare_data(_prediction_data((12, 13), reverse=True))
    expected_mean, _ = _prediction_reference((12, 13), coefficients)
    expected_outcome = np.array([[2.4, 3.4], [2.6, 3.6]])
    expected_density = _normal(expected_outcome - expected_mean) + _normal(coefficients["seasonality"])
    np.testing.assert_allclose(
        jax.jit(model.log_density)(coefficients, observed), expected_density, rtol=5e-6, atol=3e-6
    )


def test_prediction_supports_jitted_posterior_vmap(prediction_model):
    model, _, coefficients = prediction_model
    inputs = model.prepare_data(_prediction_data((12, 13), reverse=True, outcome=False))
    draws = jax.tree.map(lambda value: jnp.stack((value, value + 0.03)), coefficients)
    generated = jax.jit(jax.vmap(model.generate, in_axes=(0, 0, None)))(
        jax.random.split(jax.random.key(1), 2), draws, inputs
    )

    assert generated["mean"].shape == (2, 2, 2)
    for index in range(2):
        expected, _ = _prediction_reference((12, 13), {name: value[index] for name, value in draws.items()})
        np.testing.assert_allclose(generated["mean"][index], expected, rtol=4e-6, atol=3e-6)


def test_prediction_uses_training_layout_snapshot_and_copies_future_inputs(prediction_model):
    model, training, coefficients = prediction_model
    training.arrays["media"][:] = 999
    training.arrays.clear()
    training.columns.clear()
    future = _prediction_data((12, 13), reverse=True, outcome=False)
    inputs = model.prepare_data(future)
    future.arrays["media"][:] = 999
    future.arrays["population"][:] = 999
    future.arrays.clear()
    future.columns.clear()

    expected, _ = _prediction_reference((12, 13), coefficients)
    actual = jax.jit(model.generate)(jax.random.key(0), coefficients, inputs)
    np.testing.assert_allclose(actual["mean"], expected, rtol=4e-6, atol=3e-6)


def test_prediction_rejects_new_roles_columns_groups_and_channel_assignments(prediction_model):
    model, _, _ = prediction_model
    future = _prediction_data((12, 13), outcome=False)
    invalid = (
        replace(future, columns={**future.columns, "media": ("new_video", "search")}),
        replace(future, group_values=(("west",), ("central",))),
        replace(future, group_values=(("west",),)),
        replace(future, channels=("search", "video")),
        replace(future, group_columns=("country",)),
        replace(future, time_column="new_time"),
        replace(
            future,
            arrays={**future.arrays, "treatments": np.ones((2, 2, 1))},
            columns={**future.columns, "treatments": ("promotion",)},
        ),
    )
    for data in invalid:
        with pytest.raises(ValueError):
            model.prepare_data(data)


def test_empty_component_models_prepare_partial_prediction_roles_and_validate_data():
    training = _prediction_data((10, 11))
    model = Model(
        {},
        lambda data, effects: jnp.array(0.0),
        lambda key, data, effects: {"population": data["population"]},
        data=training,
        components=[],
    )
    future = prepare_data(
        pl.DataFrame({"time": [12, 12], "region": ["east", "west"], "residents": [300, 150]}),
        time="time",
        groups=["region"],
        population="residents",
    )
    generated = jax.jit(model.generate)(jax.random.key(0), {}, model.prepare_data(future))
    np.testing.assert_array_equal(generated["population"], [150, 300])
    with pytest.raises(ValueError):
        model.prepare_data(replace(future, time_column="new_time"))
    for wrong in ({"population": [150, 300]}, None):
        with pytest.raises(TypeError):
            model.prepare_data(wrong)
    legacy = Model({}, lambda data: jnp.array(0.0))
    with pytest.raises(RuntimeError):
        legacy.prepare_data(future)


def test_outcome_only_seasonal_model_forecasts_from_calendar_labels_alone():
    training = prepare_data(
        pl.DataFrame({"time": ["2026-01-01", "2026-01-03"], "sales": [0.3, 0.7]}), time="time", outcome="sales"
    )
    model = Model(
        {},
        lambda data, effects: normal(data["outcome"], effects["seasonality"], 1.0),
        lambda key, data, effects: {"effect": effects["seasonality"]},
        data=training,
        components=[FourierSeasonality(period="weekly", order=1)],
    )
    future = prepare_data(pl.DataFrame({"time": ["2026-01-05", "2026-01-08"]}), time="time")
    coefficients = jnp.array([0.3, -0.7])
    generated = jax.jit(model.generate)(jax.random.key(0), {"seasonality": coefficients}, model.prepare_data(future))

    assert future.arrays == {}
    np.testing.assert_allclose(
        generated["effect"], _features([4, 7], 7, 1) @ np.asarray(coefficients), rtol=4e-6, atol=2e-6
    )


def test_prediction_retains_training_float_precision_when_global_precision_changes():
    training = prepare_data(pl.DataFrame({"time": [10, 11], "sales": [0.3, 0.7]}), time="time", outcome="sales")
    with jax.enable_x64(False):
        model = Model(
            {},
            lambda data, effects: normal(data["outcome"], effects["seasonality"], 1.0),
            lambda key, data, effects: {"effect": effects["seasonality"], "observed": data["outcome"]},
            data=training,
            components=[FourierSeasonality(period=7, order=1)],
        )
        coefficients = jnp.array([0.3, -0.7])
    future = prepare_data(pl.DataFrame({"time": [12, 13], "sales": [0.5, 0.9]}), time="time", outcome="sales")
    with jax.enable_x64(True):
        inputs = model.prepare_data(future)
        generated = jax.jit(model.generate)(jax.random.key(0), {"seasonality": coefficients}, inputs)

    assert generated["effect"].dtype == generated["observed"].dtype == jnp.float32
    np.testing.assert_allclose(
        generated["effect"], _features([2, 3], 7, 1) @ np.asarray(coefficients), rtol=4e-6, atol=2e-6
    )
    np.testing.assert_array_equal(generated["observed"], np.array([0.5, 0.9], dtype=np.float32))
