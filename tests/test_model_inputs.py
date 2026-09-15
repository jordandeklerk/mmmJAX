"""Tests for binding prepared model inputs to named callbacks."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    DataBlock,
    Model,
    Positive,
    Real,
    fit_data_scaling,
    fourier_features,
    generate_quantities,
    normal,
    normal_rng,
    prepare_data,
)


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


@pytest.mark.parametrize("keyword", ["inputs", "scaling", "data_variables"])
def test_model_data_options_are_declared_in_the_data_block(keyword):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
        Model({}, lambda: jnp.array(0.0), data=_data(), **{keyword: None})


def _density(outcome, annual, annual_coefficients, intercept, sigma):
    return (
        normal(outcome, annual + intercept, sigma) + normal(intercept, 0.0, 2.0) + normal(annual_coefficients, 0.0, 1.0)
    )


def _model(density=_density, generate=None):
    return Model(
        {"intercept": Real(), "sigma": Positive(), "annual_coefficients": Real((2,))},
        density,
        generate,
        data=_data(),
        transformed_parameters=lambda time, annual_coefficients: {
            "annual": fourier_features(time, period=8, order=1) @ annual_coefficients
        },
    )


def test_named_inputs_provide_evaluated_curves_and_count_priors_and_jacobian_once():
    def quantities(rng, /, *, annual, intercept, sigma):
        return {"mean": annual + intercept, "draw": normal_rng(rng, annual + intercept, sigma)}

    model = _model(generate=quantities)
    position = {
        "annual_coefficients": jnp.array([0.3, -0.2]),
        "intercept": jnp.array(0.25),
        "sigma": jnp.log(jnp.array(0.7)),
    }
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    features = _features([0, 1, 3, 5])
    coefficients = np.asarray(position["annual_coefficients"], dtype=np.float64)
    intercept, log_scale = float(position["intercept"]), float(position["sigma"])
    scale = np.exp(log_scale)
    mean = features @ coefficients + intercept
    residual = np.array([0.4, 1.0, 1.4, 0.7]) - mean
    expected = _normal(residual, scale) + _normal(coefficients) + _normal(intercept, 2.0) + log_scale
    expected_gradient = {
        "annual_coefficients": features.T @ residual / scale**2 - coefficients,
        "intercept": residual.sum() / scale**2 - intercept / 4,
        "sigma": np.sum(residual**2) / scale**2 - 4 + 1,
    }

    assert model.parameters["annual_coefficients"].shape == (2,)
    np.testing.assert_allclose(value, expected, rtol=4e-6, atol=3e-6)
    for name in position:
        np.testing.assert_allclose(gradient[name], expected_gradient[name], rtol=5e-6, atol=3e-6)
    key = jax.random.key(13)
    constrained = model.constrain(position)
    generated = jax.jit(model.generate_quantities)(key, constrained, model.data)
    assert generated["mean"].shape == (4,)
    np.testing.assert_allclose(generated["mean"], mean, rtol=4e-6, atol=2e-6)
    noise = np.asarray(jax.random.normal(key, (4,), dtype=constrained["sigma"].dtype))
    np.testing.assert_allclose(generated["draw"], mean + scale * noise, rtol=4e-6, atol=2e-6)


@pytest.mark.parametrize("keyword_only", [False, True])
def test_named_density_supports_generation_modes_and_parameter_subsets(keyword_only):
    def keyword_generation(random_key, *, annual, intercept):
        return {"mean": annual + intercept}

    def ordinary_generation(key, intercept, annual):
        return {"mean": annual + intercept}

    model = _model(generate=keyword_generation if keyword_only else ordinary_generation)
    position = {"annual_coefficients": jnp.array([0.3, -0.2]), "intercept": jnp.array(0.25), "sigma": jnp.array(0.0)}
    generated = jax.jit(model.generate_quantities)(jax.random.key(0), model.constrain(position), model.data)
    expected = _features([0, 1, 3, 5]) @ np.array([0.3, -0.2]) + 0.25
    np.testing.assert_allclose(generated["mean"], expected, rtol=4e-6, atol=2e-6)
    assert np.isfinite(jax.jit(model.log_density)(position, model.data))


def test_density_arguments_can_mix_ordinary_and_keyword_only_inputs_in_any_order():
    def density(sigma, intercept, *, annual_coefficients, annual, outcome):
        return _density(outcome, annual, annual_coefficients, intercept, sigma)

    model = _model(density=density)
    ordinary_model = _model()
    position = {"annual_coefficients": jnp.array([0.3, -0.2]), "intercept": jnp.array(0.25), "sigma": jnp.array(0.0)}
    actual = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected = jax.jit(jax.value_and_grad(ordinary_model.log_density))(position, ordinary_model.data)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(result, reference)


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
        {"media_beta": Real(shape=(2, 2)), "control_beta": Real(shape=(2, 2)), "annual_coefficients": Real((2, 2))},
        density,
        quantities,
        data=training,
        transformed_parameters=lambda time, annual_coefficients: {
            "annual": fourier_features(time, period=8, order=1) @ annual_coefficients
        },
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
        "annual_coefficients": jnp.array([[0.3, -0.2], [0.7, 1.1]]),
        "media_beta": jnp.array([[0.2, 0.3], [-0.1, 0.4]]),
        "control_beta": jnp.array([[0.5, -0.2], [0.3, 0.1]]),
    }
    actual = jax.jit(model.generate_quantities)(jax.random.key(0), parameters, model.prepare_data(future))
    media = np.array([[[5, 9], [6, 10]], [[7, 11], [8, 12]]], dtype=np.float64)
    controls = np.array([[[4, 5], [3, 6]], [[2, 7], [1, 8]]], dtype=np.float64)
    annual = _features([2, 4]) @ np.asarray(parameters["annual_coefficients"], dtype=np.float64)
    expected = annual + (
        media * np.asarray(parameters["media_beta"]) + controls * np.asarray(parameters["control_beta"])
    ).sum(-1)

    assert "outcome" not in future.arrays
    assert future.group_values == (("east",), ("west",))
    np.testing.assert_allclose(actual["annual"], annual, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(actual["mean"], expected, rtol=4e-6, atol=3e-6)


def test_named_callbacks_require_requested_inputs_despite_defaults():
    def density(outcome=0.0):
        return normal(outcome, 0.0, 1.0)

    def quantities(key, outcome=0.0):
        return {"observed": outcome}

    model = Model({}, density, quantities, data=_data())
    np.testing.assert_allclose(jax.jit(model.log_density)({}, model.data), _normal([0.4, 1.0, 1.4, 0.7]), rtol=3e-6)
    future = model.prepare_data(prepare_data(pl.DataFrame({"time": [17, 18]}), time="time"))
    with pytest.raises(ValueError, match="outcome"):
        model.log_density({}, future)
    with pytest.raises(ValueError, match="outcome"):
        model.generate_quantities(jax.random.key(0), {}, future)


def test_named_density_must_request_every_user_parameter():
    with pytest.raises(ValueError, match="intercept"):
        Model({"intercept": Real()}, lambda *, outcome: normal(outcome, 0.0, 1.0), data=_data())


def test_undeclared_names_are_rejected_even_when_defaulted_or_matching_source_columns():
    for density in (lambda *, sales: jnp.array(0.0), lambda *, outcome, unknown=0.0: jnp.array(0.0)):
        with pytest.raises(ValueError, match=r"sales|unknown"):
            Model({}, density, data=_data())
    with pytest.raises(ValueError, match="unknown"):
        Model({}, lambda *, outcome: jnp.array(0.0), lambda key, *, unknown=0.0: {}, data=_data())


def test_named_binding_rejects_collisions_between_data_and_parameters():
    with pytest.raises(ValueError, match=r"ambig|colli|conflict"):
        Model({"outcome": Real()}, lambda outcome: jnp.sum(outcome), data=_data())

    with pytest.raises(ValueError, match=r"ambig|colli|conflict"):
        Model(
            {"outcome": Real()},
            lambda outcome: normal(outcome, 0.0, 1.0),
            lambda key, outcome: {"value": outcome},
            data=_data(),
        )


def test_named_callbacks_reject_variadic_parameters_and_a_keyword_only_random_key():
    densities = (
        lambda *, outcome, **values: jnp.array(0.0),
        lambda *values, outcome: jnp.array(0.0),
        lambda outcome, /: jnp.array(0.0),
    )
    for density in densities:
        with pytest.raises(TypeError):
            Model({}, density, data=_data())
    generators = (
        lambda key, *, outcome, **values: {},
        lambda key, *values, outcome: {},
        lambda *, key, outcome: {},
        lambda outcome: {},
        lambda key, outcome, /: {},
    )
    for generate in generators:
        with pytest.raises(TypeError, match="generated_quantities"):
            Model({}, lambda *, outcome: jnp.array(0.0), generated_quantities=generate, data=_data())


def test_keyword_only_input_resolution_is_not_enabled_for_legacy_models():
    with pytest.raises(TypeError):
        Model({}, lambda *, outcome: normal(outcome, 0.0, 1.0))


def test_prepared_callbacks_cannot_request_raw_data_or_effects():
    def density(data, effects):
        return normal(data["outcome"], effects["annual"], 1.0)

    with pytest.raises(TypeError, match="no longer receives data or effects"):
        Model({}, density, data=_data())


def test_prepared_callbacks_may_request_no_inputs_without_hidden_priors():
    model = Model(
        {},
        lambda: jnp.array(1.25),
        lambda key: {"constant": jnp.array(3.0)},
        data=_data(),
    )
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))({}, model.data)
    np.testing.assert_array_equal(value, 1.25)
    assert gradient == {}
    generated = jax.jit(model.generate_quantities)(jax.random.key(0), {}, model.data)
    np.testing.assert_array_equal(generated["constant"], 3.0)


def test_transformed_outputs_cannot_shadow_declared_parameters():
    model = Model(
        {"annual_coefficients": Real((2,))},
        lambda summary: jnp.sum(summary),
        lambda key, summary: {"summary": summary},
        data=_data(),
        transformed_parameters=lambda annual_coefficients: {
            "summary": jnp.ones(4),
            "annual_coefficients": annual_coefficients,
        },
    )
    position = {"annual_coefficients": jnp.array([0.3, -0.2])}
    with pytest.raises(ValueError, match="annual_coefficients"):
        model.log_density(position, model.data)
    with pytest.raises(ValueError, match="annual_coefficients"):
        model.generate_quantities(jax.random.key(0), position, model.data)


def test_parameter_names_cannot_replace_the_generation_random_key():
    with pytest.raises(TypeError, match=r"annual_coefficients.*random key"):
        Model(
            {"annual_coefficients": Real((2,))},
            lambda annual_coefficients: normal(annual_coefficients, 0.0, 1.0),
            lambda annual_coefficients: {},
            data=_data(),
        )


def test_explicit_data_variables_bind_user_names_in_all_program_blocks_under_jit():
    data = _data()

    def transformed_parameters(elapsed, level, seasonal_coefficients):
        seasonal = fourier_features(elapsed, period=8, order=1) @ seasonal_coefficients
        mean = level + seasonal
        return {"mean": mean}

    def log_density(revenue, mean, level, seasonal_coefficients):
        target = normal(level, 0.0, 2.0)
        target += normal(seasonal_coefficients, 0.0, 1.0)
        target += normal(revenue, mean, 1.0)
        return target

    def generated_quantities(key, revenue, mean):
        residual = revenue - mean
        return {"residual": residual, "mean": mean}

    model = Model(
        {"level": Real(), "seasonal_coefficients": Real((2,))},
        log_density,
        generated_quantities,
        data=DataBlock(data, variables={"revenue": "outcome", "elapsed": "time"}),
        transformed_parameters=transformed_parameters,
    )
    position = {"level": jnp.array(0.25), "seasonal_coefficients": jnp.array([0.3, -0.2])}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), position, model.data)
    features = _features([0, 1, 3, 5])
    mean = 0.25 + features @ np.array([0.3, -0.2])
    residual = data.arrays["outcome"] - mean
    expected = _normal(0.25, 2.0) + _normal([0.3, -0.2]) + _normal(residual)

    np.testing.assert_allclose(value, expected, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(gradient["level"], residual.sum() - 0.25 / 4, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(
        gradient["seasonal_coefficients"], features.T @ residual - [0.3, -0.2], rtol=4e-6, atol=3e-6
    )
    np.testing.assert_allclose(outputs["mean"], mean, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(outputs["residual"], residual, rtol=4e-6, atol=3e-6)


def test_explicit_data_variables_alias_auxiliary_inputs_and_keep_their_snapshot():
    inputs = xr.Dataset({"measured_lift": ("experiment", [1.0, 2.0])})

    def density(experiment_lift, location):
        target = normal(location, 0.0, 1.0)
        target += normal(experiment_lift, location, 0.5)
        return target

    model = Model(
        {"location": Real()},
        density,
        lambda key, experiment_lift: {"lift": experiment_lift},
        data=DataBlock(_data(), inputs=inputs, variables={"experiment_lift": "measured_lift"}),
    )
    inputs["measured_lift"].values[:] = 100
    position = {"location": jnp.array(0.25)}
    value = jax.jit(model.log_density)(position, model.data)
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), position, model.data)
    expected = _normal(0.25) + _normal(np.array([1.0, 2.0]) - 0.25, 0.5)

    np.testing.assert_allclose(value, expected, rtol=3e-6)
    np.testing.assert_array_equal(outputs["lift"], [1.0, 2.0])


def test_auxiliary_source_may_share_a_parameter_name_when_declared_under_a_different_name():
    inputs = xr.Dataset({"roi": ("experiment", [1.0, 2.0])}, coords={"experiment": ["video", "search"]})

    def density(roi, measured_roi):
        target = normal(roi, 0.0, 1.0)
        target += normal(measured_roi, roi, 0.5)
        return target

    model = Model(
        {"roi": Real()},
        density,
        lambda key, roi, measured_roi: {"estimated_roi": roi, "measured_roi": measured_roi},
        data=DataBlock(_data(), inputs=inputs, variables={"measured_roi": "roi"}),
    )
    value = jax.jit(model.log_density)({"roi": jnp.array(0.25)}, model.data)
    expected = _normal(0.25) + _normal(np.array([1.0, 2.0]) - 0.25, 0.5)
    draws = np.array([[0.25, 0.75]], dtype=model.parameters["roi"].dtype)
    posterior = xr.Dataset({"roi": (("chain", "draw"), draws)})
    results = generate_quantities(model, xr.DataTree.from_dict({"posterior": posterior}), batch_size=1)

    np.testing.assert_allclose(value, expected, rtol=3e-6)
    assert results["posterior"]["roi"].dims == ("chain", "draw")
    assert results["constant_data"]["roi"].dims == ("experiment",)
    assert results["generated_quantities"]["measured_roi"].dims == ("chain", "draw", "experiment")
    np.testing.assert_array_equal(results["constant_data"]["roi"], [1.0, 2.0])
    np.testing.assert_array_equal(results["generated_quantities"]["estimated_roi"], [[0.25, 0.75]])
    np.testing.assert_array_equal(results["generated_quantities"]["measured_roi"], [[[1.0, 2.0], [1.0, 2.0]]])


def test_explicit_data_variable_mapping_is_copied_at_construction_and_inspection():
    declarations = {"revenue": "outcome"}
    model = Model({}, lambda revenue: jnp.sum(revenue), data=DataBlock(_data(), variables=declarations))
    declarations["revenue"] = "media"
    inspected = model.data_variables
    inspected["revenue"] = "spend"

    assert model.data_variables == {"revenue": "outcome"}
    np.testing.assert_allclose(model.log_density({}, model.data), 3.5)


@pytest.mark.parametrize("declarations", ["outcome", [("revenue", "outcome")], 1])
def test_data_variables_requires_a_mapping(declarations):
    with pytest.raises(TypeError, match="variables must map"):
        Model({}, lambda: jnp.array(0.0), data=DataBlock(_data(), variables=declarations))


@pytest.mark.parametrize("name", ["", "weekly sales", "class", 1])
def test_data_variable_names_must_be_valid_function_argument_names(name):
    with pytest.raises(ValueError, match="data variable name must be a valid Python identifier"):
        Model({}, lambda: jnp.array(0.0), data=DataBlock(_data(), variables={name: "outcome"}))


@pytest.mark.parametrize("source", ["", 1, np.ones(4)])
def test_data_variable_sources_must_be_nonempty_names(source):
    with pytest.raises((TypeError, ValueError), match="Source for data variable 'revenue'"):
        Model({}, lambda: jnp.array(0.0), data=DataBlock(_data(), variables={"revenue": source}))


@pytest.mark.parametrize("source", ["media", "sales", "reference_media", "unknown"])
def test_unavailable_data_variable_sources_are_rejected_even_if_unused(source):
    with pytest.raises(ValueError, match="exposure"):
        Model({}, lambda: jnp.array(0.0), data=DataBlock(_data(), variables={"exposure": source}))


def test_explicit_data_variables_require_prepared_data():
    with pytest.raises(TypeError, match="DataBlock data must be PreparedData"):
        Model({}, lambda data: jnp.array(0.0), data=DataBlock(None, variables={"revenue": "outcome"}))


def test_declared_data_variables_cannot_share_parameter_names():
    with pytest.raises(ValueError, match="revenue"):
        Model(
            {"revenue": Real()},
            lambda revenue: normal(revenue, 0.0, 1.0),
            data=DataBlock(_data(), variables={"revenue": "outcome"}),
        )


@pytest.mark.parametrize("block", ["log_density", "transformed_parameters", "generated_quantities"])
def test_explicit_data_variables_do_not_fall_back_to_implicit_source_names(block):
    callbacks = {
        "log_density": lambda: jnp.array(0.0),
        "transformed_parameters": None,
        "generated_quantities": None,
    }
    if block == "log_density":
        callbacks[block] = lambda outcome: jnp.sum(outcome)
    elif block == "transformed_parameters":
        callbacks[block] = lambda outcome: {"mean": outcome}
    else:
        callbacks[block] = lambda key, outcome: {"observed": outcome}

    with pytest.raises(ValueError, match="outcome"):
        Model({}, data=DataBlock(_data(), variables={"revenue": "outcome"}), **callbacks)


def test_missing_transformed_quantity_does_not_fall_back_to_an_undeclared_data_source():
    model = Model(
        {},
        lambda outcome: jnp.sum(outcome),
        data=DataBlock(_data(), variables={"revenue": "outcome"}),
        transformed_parameters=lambda revenue: {"mean": revenue},
    )
    with pytest.raises(ValueError, match="outcome"):
        model.log_density({}, model.data)


def test_empty_data_variables_declares_no_inputs():
    model = Model({}, lambda: jnp.array(1.25), data=DataBlock(_data(), variables={}))
    assert model.data_variables == {}
    np.testing.assert_array_equal(jax.jit(model.log_density)({}, model.data), 1.25)

    with pytest.raises(ValueError, match="outcome"):
        Model({}, lambda outcome: jnp.sum(outcome), data=DataBlock(_data(), variables={}))


def test_undeclared_source_identifiers_can_be_used_as_parameter_names():
    def density(revenue, outcome, outcome_scale, n_periods, reference_outcome):
        mean = outcome + outcome_scale + n_periods + reference_outcome
        target = normal(outcome, 0.0, 1.0)
        target += normal(outcome_scale, 0.0, 1.0)
        target += normal(n_periods, 0.0, 1.0)
        target += normal(reference_outcome, 0.0, 1.0)
        target += normal(revenue, mean, 1.0)
        return target

    names = ("outcome", "outcome_scale", "n_periods", "reference_outcome")
    model = Model({name: Real() for name in names}, density, data=DataBlock(_data(), variables={"revenue": "outcome"}))
    position = {name: jnp.array(0.25) for name in names}
    expected = _normal(np.full(4, 0.25)) + _normal(_data().arrays["outcome"] - 1.0)
    np.testing.assert_allclose(jax.jit(model.log_density)(position, model.data), expected, rtol=4e-6)


def test_undeclared_source_identifiers_can_be_used_as_transformed_output_names():
    model = Model(
        {},
        lambda outcome: jnp.sum(outcome),
        data=DataBlock(_data(), variables={"revenue": "outcome"}),
        transformed_parameters=lambda revenue: {"outcome": revenue * 2},
    )
    np.testing.assert_allclose(jax.jit(model.log_density)({}, model.data), 7.0)


def test_transformed_outputs_cannot_shadow_explicit_data_variables():
    model = Model(
        {},
        lambda revenue: jnp.sum(revenue),
        data=DataBlock(_data(), variables={"revenue": "outcome"}),
        transformed_parameters=lambda revenue: {"revenue": revenue * 2},
    )
    with pytest.raises(ValueError, match="revenue"):
        model.log_density({}, model.data)


def test_generated_quantities_key_may_use_an_undeclared_source_identifier():
    model = Model(
        {},
        lambda revenue: jnp.sum(revenue),
        lambda outcome, revenue: {"observed": revenue},
        data=DataBlock(_data(), variables={"revenue": "outcome"}),
    )
    output = jax.jit(model.generate_quantities)(jax.random.key(0), {}, model.data)
    np.testing.assert_allclose(output["observed"], _data().arrays["outcome"])


def test_generated_quantities_key_cannot_replace_an_explicit_data_variable():
    with pytest.raises(TypeError, match=r"revenue.*random key"):
        Model(
            {},
            lambda revenue: jnp.sum(revenue),
            lambda revenue: {},
            data=DataBlock(_data(), variables={"revenue": "outcome"}),
        )


@pytest.mark.parametrize("scaled", [False, True])
def test_aliased_outcome_scale_and_offset_restore_units_under_jit(scaled):
    data = _data()
    scaling = fit_data_scaling(data, scale_outcome=True) if scaled else None

    def generated_quantities(key, standardized_revenue, revenue_scale, revenue_offset):
        revenue = standardized_revenue * revenue_scale + revenue_offset
        return {"revenue": revenue, "scale": revenue_scale, "offset": revenue_offset}

    model = Model(
        {},
        lambda standardized_revenue: jnp.sum(standardized_revenue),
        generated_quantities,
        data=DataBlock(
            data,
            scaling=scaling,
            variables={
                "standardized_revenue": "outcome",
                "revenue_scale": "outcome_scale",
                "revenue_offset": "outcome_offset",
            },
        ),
    )
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), {}, model.data)

    np.testing.assert_allclose(outputs["revenue"], data.arrays["outcome"], rtol=4e-6, atol=3e-6)
    assert outputs["scale"].shape == ()
    assert outputs["offset"].shape == ()
    if not scaled:
        np.testing.assert_array_equal(outputs["scale"], 1.0)
        np.testing.assert_array_equal(outputs["offset"], 0.0)


@pytest.mark.parametrize("groups", [1, 2])
def test_aliased_population_unit_conversions_preserve_the_group_axis(groups):
    rows = [
        {"time": time, "region": f"region_{group}", "sales": (group + 1) * (time + 2), "population": 10 * (group + 1)}
        for time in range(3)
        for group in range(groups)
    ]
    data = prepare_data(pl.DataFrame(rows), time="time", groups=["region"], outcome="sales", population="population")
    scaling = fit_data_scaling(data, scale_outcome="population")

    def generated_quantities(key, y, scale, offset):
        original_units = y * scale + offset
        return {"original_units": original_units, "scale": scale, "offset": offset}

    model = Model(
        {},
        lambda y: jnp.sum(y),
        generated_quantities,
        data=DataBlock(
            data, scaling=scaling, variables={"y": "outcome", "scale": "outcome_scale", "offset": "outcome_offset"}
        ),
    )
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), {}, model.data)

    assert outputs["scale"].shape == (groups,)
    assert outputs["offset"].shape == (groups,)
    np.testing.assert_allclose(outputs["original_units"], data.arrays["outcome"], rtol=4e-6, atol=3e-6)


def test_declared_current_inputs_update_for_scenarios_while_references_remain_fixed():
    training = prepare_data(
        pl.DataFrame({"time": [1, 2, 3], "video": [10.0, 20.0, 30.0], "sales": [1.0, 2.0, 3.0]}),
        time="time",
        media=["video"],
        outcome="sales",
    )
    scenario = prepare_data(pl.DataFrame({"time": [4, 5], "video": [40.0, 50.0]}), time="time", media=["video"])

    def generated_quantities(key, exposure, original_exposure, elapsed, period_count, original_period_count):
        window = jnp.ones(period_count)
        original_window = jnp.ones(original_period_count)
        return {
            "exposure": exposure,
            "original_exposure": original_exposure,
            "elapsed": elapsed,
            "window": window,
            "original_window": original_window,
        }

    model = Model(
        {},
        lambda revenue: jnp.sum(revenue),
        generated_quantities,
        data=DataBlock(
            training,
            variables={
                "revenue": "outcome",
                "exposure": "media",
                "original_exposure": "reference_media",
                "elapsed": "time",
                "period_count": "n_periods",
                "original_period_count": "reference_n_periods",
            },
        ),
    )
    inputs = model.prepare_data(scenario)
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), {}, inputs)

    np.testing.assert_array_equal(outputs["exposure"], [[40.0], [50.0]])
    np.testing.assert_array_equal(outputs["original_exposure"], [[10.0], [20.0], [30.0]])
    np.testing.assert_array_equal(outputs["elapsed"], [3.0, 4.0])
    assert outputs["window"].shape == (2,)
    assert outputs["original_window"].shape == (3,)
    with pytest.raises(ValueError, match="revenue"):
        model.log_density({}, inputs)
