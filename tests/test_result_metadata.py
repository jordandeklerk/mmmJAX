"""Tests for result labels attached to model definitions without invoking a sampler."""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    Data,
    Model,
    Positive,
    Prior,
    Real,
    Simplex,
    dirichlet,
    fit_data_scaling,
    fourier_features,
    generate_quantities,
    normal,
    prepare_data,
    sample_prior,
)
from mmmjax._results import _collect_results


def _plain_model(parameters=None, generate=None, **metadata):
    declarations = {"coefficient": Real(shape=(2,)), "intercept": Real()} if parameters is None else parameters

    def density(data, **values):
        return -sum((jnp.square(value).sum() for value in values.values()), start=jnp.array(0.0))

    return Model(parameters=declarations, log_density=density, generated_quantities=generate, **metadata)


def _generate(key, data, **parameters):
    return {"outcome": jnp.ones(3), "pointwise": -jnp.ones(3), "mean": jnp.ones(3)}


def _prepared_data(reverse=False):
    groups = [("west", "online"), ("east", "retail")]
    if reverse:
        groups.reverse()
    rows = [
        {
            "week": time,
            "region": region,
            "segment": segment,
            "sales": 10.0 * time + (region == "west"),
            "video": float(time * 3 + (region == "west")),
            "search": float(time * 2 + (region == "east")),
        }
        for time in (1, 2, 3, 4)
        for region, segment in groups
    ]
    frame = pl.DataFrame(rows)
    return prepare_data(
        frame.filter(pl.col("week") > 1),
        time="week",
        groups=["region", "segment"],
        outcome="sales",
        media=["search", "video"] if reverse else ["video", "search"],
        media_history=frame.filter(pl.col("week") == 1),
    )


def test_empty_metadata_preserves_existing_models():
    model = _plain_model()
    assert model._result_dims == {}
    assert model._result_coords == {}
    assert model._generated_dims == {}
    assert model._time_values == ()
    assert model._media_time_values == ()


@pytest.mark.parametrize("names", [(), ("coefficient",), ("coefficient", "intercept", "extra")])
def test_prior_mappings_require_exact_parameter_names(names):
    priors = {name: Prior(normal, location=0.0, scale=1.0) for name in names}
    with pytest.raises(ValueError, match=r"prior|parameter"):
        sample_prior(_plain_model(), priors, draws=2)


@pytest.mark.parametrize("distribution", [normal, 1.0, None])
def test_prior_mapping_entries_require_prior_objects(distribution):
    priors = {"coefficient": distribution, "intercept": Prior(normal, location=0.0, scale=1.0)}
    with pytest.raises(TypeError, match="Prior"):
        sample_prior(_plain_model(), priors, draws=2)


@pytest.mark.parametrize("event", [False, True])
def test_prior_mapping_shapes_must_match_parameter_declarations(event):
    parameter = Simplex((3,)) if event else Real((3,))
    distribution = (
        Prior(dirichlet, concentration=jnp.ones(2)) if event else Prior(normal, location=jnp.zeros(2), scale=1.0)
    )
    with pytest.raises(ValueError, match=r"shape.*coefficient"):
        sample_prior(_plain_model(parameters={"coefficient": parameter}), {"coefficient": distribution}, draws=2)


def test_result_metadata_copies_mappings_axes_coordinates_and_output_names():
    dims = {"coefficient": ["channel"], "intercept": []}
    labels = np.array(["video", "search"])
    coords = {"channel": labels, "time": [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15)]}
    generated_dims = {"outcome": ["time"], "pointwise": ["time"], "mean": ["time"]}
    model = _plain_model(generate=_generate, dims=dims, coords=coords, generated_dims=generated_dims)
    dims["coefficient"].append("another")
    dims["extra"] = []
    labels[:] = "edited"
    coords["time"].clear()
    coords["extra"] = [1]
    generated_dims["outcome"].clear()

    assert model._result_dims == {"coefficient": ("channel",), "intercept": ()}
    np.testing.assert_array_equal(model._result_coords["channel"], ["video", "search"])
    np.testing.assert_array_equal(
        model._result_coords["time"], np.array(["2026-01-01", "2026-01-08", "2026-01-15"], dtype="datetime64[us]")
    )
    assert set(model._result_coords) == {"channel", "time"}
    assert model._generated_dims == {"outcome": ("time",), "pointwise": ("time",), "mean": ("time",)}


@pytest.mark.parametrize(
    "metadata,error,message",
    [
        ({"dims": []}, TypeError, "dims must map"),
        ({"dims": {"unknown": ()}}, ValueError, "undeclared parameter"),
        ({"dims": {"coefficient": 1}}, TypeError, "sequence"),
        ({"dims": {"coefficient": ()}}, ValueError, "constrained shape"),
        ({"dims": {"intercept": ("channel",)}}, ValueError, "constrained shape"),
        ({"dims": {"coefficient": ("channel", "channel")}}, ValueError, "must not repeat"),
        ({"dims": {"coefficient": ("draw",)}}, ValueError, "reserved sample"),
        ({"dims": {"coefficient": ("",)}}, ValueError, "nonempty"),
        ({"coords": []}, TypeError, "coords must map"),
        ({"coords": {"channel": [[1, 2]]}}, ValueError, "one-dimensional"),
        (
            {"dims": {"coefficient": ("channel",)}, "coords": {"channel": ["only"]}},
            ValueError,
            "must have length 2",
        ),
    ],
)
def test_invalid_parameter_metadata_is_rejected(metadata, error, message):
    with pytest.raises(error, match=message):
        _plain_model(**metadata)


def test_shared_dimensions_must_have_consistent_sizes_without_coordinates():
    with pytest.raises(ValueError, match="Dimension 'feature' must have length"):
        _plain_model(
            parameters={"first": Real(shape=(2,)), "second": Real(shape=(3,))},
            dims={"first": ("feature",), "second": ("feature",)},
        )


def test_simplex_metadata_uses_constrained_shape():
    model = _plain_model(
        parameters={"weights": Simplex(shape=(3,))},
        dims={"weights": ("channel",)},
        coords={"channel": ["video", "search", "social"]},
    )
    assert model.parameters["weights"].position_shape == (2,)
    assert model.parameters["weights"].shape == (3,)
    with pytest.raises(ValueError, match="must have length 3"):
        _plain_model(
            parameters={"weights": Simplex(shape=(3,))},
            dims={"weights": ("channel",)},
            coords={"channel": ["video", "search"]},
        )


@pytest.mark.parametrize("group", ["predictive", "log_likelihood", "log_prior"])
@pytest.mark.parametrize(
    "value,error,message",
    [
        (jnp.ones(3), TypeError, "mapping"),
        (["outcome"], TypeError, "mapping"),
        ({"": jnp.ones(3)}, ValueError, "identifier"),
        ({1: jnp.ones(3)}, TypeError, "string"),
    ],
)
def test_generated_group_values_are_validated_when_returned(group, value, error, message):
    model = _plain_model(generate=lambda key, data, **parameters: {group: value})
    position = {"coefficient": jnp.zeros(2), "intercept": jnp.array(0.0)}
    with pytest.raises(error, match=message):
        model.generate_quantities(jax.random.key(0), position, None)


def test_the_same_output_name_can_be_returned_in_several_groups():
    def generate(key, data, **parameters):
        value = jnp.ones(3)
        return {"outcome": value, "predictive": {"outcome": value}, "log_likelihood": {"outcome": -value}}

    model = _plain_model(generate=generate)
    position = {"coefficient": jnp.zeros(2), "intercept": jnp.array(0.0)}
    outputs = model.generate_quantities(jax.random.key(0), position, None)
    np.testing.assert_array_equal(outputs["outcome"], np.ones(3))
    np.testing.assert_array_equal(outputs["predictive"]["outcome"], np.ones(3))
    np.testing.assert_array_equal(outputs["log_likelihood"]["outcome"], -np.ones(3))


def test_generated_dimensions_require_a_generation_callback():
    with pytest.raises(ValueError, match="requires a generated_quantities callback"):
        _plain_model(generated_dims={"mean": ("time",)})


@pytest.mark.parametrize(
    "generated_dims,error",
    [([], TypeError), ({"mean": 1}, TypeError), ({"mean": ("chain",)}, ValueError)],
)
def test_generated_dimensions_use_the_same_axis_validation(generated_dims, error):
    with pytest.raises(error):
        _plain_model(generate=_generate, generated_dims=generated_dims)


def test_metadata_does_not_execute_generated_quantities_at_construction():
    def unavailable_until_evaluation(key, data, **parameters):
        raise RuntimeError("This callback should not run during construction")

    model = _plain_model(
        generate=unavailable_until_evaluation,
        generated_dims={"later_output": ("new_axis",)},
    )
    assert model._generated_dims == {"later_output": ("new_axis",)}


def test_parameters_and_generated_names_have_separate_dimension_maps():
    data = _prepared_data()

    def density(outcome, annual, paid_media_coefficient):
        return -jnp.square(outcome).sum() - jnp.square(annual).sum() - jnp.square(paid_media_coefficient).sum()

    def generate(key, time, annual):
        return {"annual": fourier_features(time, period=52, order=2) @ annual}

    model = Model(
        parameters={"annual": Real((4, 2)), "paid_media_coefficient": Real((2, 2))},
        log_density=density,
        generated_quantities=generate,
        data=data,
        dims={"annual": ("annual_mode", "group"), "paid_media_coefficient": ("group", "channel")},
        coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
        generated_dims={"annual": ("time", "group")},
    )
    assert model._result_dims["annual"] == ("annual_mode", "group")
    assert model._generated_dims["annual"] == ("time", "group")
    assert model.parameters["annual"].shape == (4, 2)
    assert model.parameters["paid_media_coefficient"].shape == (2, 2)
    with pytest.raises(ValueError, match="undeclared parameter 'annual_coefficients'"):
        Model(
            parameters={"annual": Real((4,))},
            log_density=lambda annual: jnp.sum(annual),
            data=data,
            dims={"annual_coefficients": ("annual_mode",)},
        )


def test_aligned_time_history_and_group_metadata_follow_actual_scaled_model_inputs():
    reference = _prepared_data()
    incoming = _prepared_data(reverse=True)
    scaling = fit_data_scaling(reference, scale_outcome=True)
    expected = scaling.transform(incoming)
    model = Model(
        parameters={},
        log_density=lambda outcome: -jnp.square(outcome).sum(),
        data=Data(incoming, scaling=scaling),
    )
    assert model._time_values == expected.time_values == (2, 3, 4)
    assert model._media_time_values == expected.media_time_values == (1, 2, 3, 4)
    assert model._layout.group_columns == ("region", "segment")
    assert model._layout.group_values == reference.group_values
    assert model._layout.group_values != incoming.group_values
    assert model._layout.channels == ("video", "search")
    assert model._layout.columns == expected.columns
    assert model._data.values["outcome"].shape == (3, 2)
    assert model._data.values["media"].shape == (4, 2, 2)
    for name, values in expected.arrays.items():
        np.testing.assert_allclose(model._data.values[name], values, rtol=1e-6)
    incoming.arrays["outcome"][:] = -999
    incoming.columns["media"] = ("changed",)
    np.testing.assert_allclose(model._data.values["outcome"], expected.arrays["outcome"], rtol=1e-6)
    assert model._layout.columns["media"] == ("video", "search")


def test_result_labels_do_not_change_density_gradients_or_generation():
    plain = _plain_model(generate=_generate)
    labeled = _plain_model(
        generate=_generate,
        dims={"coefficient": ("channel",)},
        coords={"channel": ["video", "search"], "time": [1, 2, 3]},
        generated_dims={"outcome": ("time",), "pointwise": ("time",), "mean": ("time",)},
    )
    position = {"coefficient": jnp.array([0.2, -0.3]), "intercept": jnp.array(0.4)}
    expected = jax.jit(jax.value_and_grad(plain.log_density))(position, None)
    actual = jax.jit(jax.value_and_grad(labeled.log_density))(position, None)
    for actual_array, expected_array in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    key = jax.random.key(0)
    for name, expected_array in plain.generate_quantities(key, plain.constrain(position), None).items():
        np.testing.assert_array_equal(
            labeled.generate_quantities(key, labeled.constrain(position), None)[name], expected_array
        )


def _named_axis_data():
    rows = [
        {
            "week": week,
            "region": region,
            "sales": 10.0 + week,
            "video": 3.0 + week,
            "search": 2.0 + week,
            "temperature": 15.0 + week,
            "holiday": float(week == 2),
            "price": 5.0 + week,
            "email": 4.0 + week,
            "tv_reach": 100.0 + week,
            "tv_frequency": 2.0,
            "social_reach": 50.0 + week,
            "social_frequency": 1.0,
        }
        for week in (1, 2, 3)
        for region in ("west", "east")
    ]
    return prepare_data(
        pl.DataFrame(rows),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        controls=["temperature", "holiday"],
        treatments=["price"],
        organic_media=["email"],
        reach=["tv_reach"],
        media_frequency=["tv_frequency"],
        rf_channels=["tv"],
        organic_reach=["social_reach"],
        organic_frequency=["social_frequency"],
        organic_rf_channels=["social"],
    )


@pytest.mark.parametrize(
    "axis,labels",
    [
        ("channel", ["video", "search"]),
        ("group", ["west", "east"]),
        ("control", ["temperature", "holiday"]),
        ("treatment", ["price"]),
        ("organic_channel", ["email"]),
        ("rf_channel", ["tv"]),
        ("organic_rf_channel", ["social"]),
    ],
)
def test_parameter_axes_infer_shapes_and_labels_from_prepared_data(axis, labels):
    data = _named_axis_data()
    model = Model(
        parameters={"coefficient": Real(dims=axis)},
        log_density=lambda coefficient: -jnp.square(coefficient).sum(),
        data=data,
    )

    assert model.parameters["coefficient"].shape == (len(labels),)
    assert model.parameters["coefficient"].position_shape == (len(labels),)
    assert model._result_dims["coefficient"] == (axis,)
    results = _collect_results(
        {"coefficient": np.zeros((1, 1, len(labels)))},
        data=data,
        dims=model._result_dims,
        coords=model._result_coords,
    )
    np.testing.assert_array_equal(results["posterior"][axis], labels)


def test_named_simplex_axes_resolve_constrained_and_unconstrained_batch_shapes():
    model = Model(
        parameters={"weights": Simplex(dims=("group", "channel"))},
        log_density=lambda weights: jnp.log(weights).sum(),
        data=_prepared_data(),
    )

    assert model.parameters["weights"].shape == (2, 2)
    assert model.parameters["weights"].position_shape == (2, 1)
    assert model._result_dims["weights"] == ("group", "channel")
    position = {"weights": jnp.zeros((2, 1))}
    constrained = model.constrain(position)
    np.testing.assert_allclose(constrained["weights"], np.full((2, 2), 0.5))
    assert np.isfinite(jax.jit(model.log_density)(position, model.data))


def test_named_declaration_can_be_reused_without_mutation_for_different_coordinate_sizes():
    declaration = Positive(dims="feature")
    short = _plain_model(parameters={"scale": declaration}, coords={"feature": ["price"]})
    long = _plain_model(parameters={"scale": declaration}, coords={"feature": ["price", "promotion"]})

    assert declaration.shape == ()
    assert declaration.dims == ("feature",)
    assert short.parameters["scale"].shape == (1,)
    assert long.parameters["scale"].shape == (2,)
    assert short.parameters["scale"] is not declaration
    assert long.parameters["scale"] is not declaration
    assert short._result_dims == long._result_dims == {"scale": ("feature",)}


@pytest.mark.parametrize("shape", [(), (2,)])
def test_matching_model_and_declaration_axes_are_accepted(shape):
    model = _plain_model(
        parameters={"coefficient": Real(shape=shape, dims="feature")},
        dims={"coefficient": ("feature",)},
        coords={"feature": ["price", "promotion"]},
    )

    assert model.parameters["coefficient"].shape == (2,)
    assert model._result_dims == {"coefficient": ("feature",)}


def test_model_dimensions_cannot_relabel_declared_axes_even_when_lengths_match():
    with pytest.raises(ValueError, match="coefficient"):
        _plain_model(
            parameters={"coefficient": Real(dims="channel")},
            dims={"coefficient": ("control",)},
            coords={"channel": ["video", "search"], "control": ["price", "promotion"]},
        )


def test_explicit_parameter_shape_must_agree_with_declared_coordinate_length():
    with pytest.raises(ValueError, match=r"channel|coefficient"):
        Model(
            parameters={"coefficient": Real(shape=(3,), dims="channel")},
            log_density=lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
        )


@pytest.mark.parametrize("axis", ["unknown", "control", "treatment", "organic_channel"])
def test_declared_axes_must_be_available_at_model_construction(axis):
    with pytest.raises(ValueError, match=axis):
        Model(
            parameters={"coefficient": Real(dims=axis)},
            log_density=lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
        )


def test_declared_custom_axis_requires_coordinates_without_prepared_data():
    with pytest.raises(ValueError, match="feature"):
        _plain_model(parameters={"coefficient": Real(dims="feature")})


def test_explicit_shape_supplies_named_axis_length_without_coordinates():
    model = _plain_model(parameters={"coefficient": Real(shape=(2,), dims="feature")})

    assert model.parameters["coefficient"].shape == (2,)
    assert model._result_dims == {"coefficient": ("feature",)}


def test_unresolved_named_shape_cannot_borrow_another_parameters_shape():
    with pytest.raises(ValueError, match="feature"):
        _plain_model(parameters={"first": Real(shape=(2,), dims="feature"), "second": Real(dims="feature")})


def test_explicit_coordinates_cannot_reorder_prepared_channel_labels():
    with pytest.raises(ValueError, match="channel"):
        Model(
            parameters={"coefficient": Real(dims="channel")},
            log_density=lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
            coords={"channel": ["search", "video"]},
        )


def test_named_parameter_axes_survive_reordered_scenario_inputs():
    data = _prepared_data()
    model = Model(
        parameters={"coefficient": Real(dims=("group", "channel"))},
        log_density=lambda coefficient: -jnp.square(coefficient).sum(),
        generated_quantities=lambda key, media, coefficient: {
            "coefficient_copy": coefficient,
            "response": media[-3:] * coefficient,
        },
        data=data,
        generated_dims={"response": ("time", "group", "channel")},
    )
    coefficient = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=jax.dtypes.canonicalize_dtype(float))
    results = _collect_results(
        {"coefficient": coefficient[None, None]},
        data=data,
        dims=model._result_dims,
        coords=model._result_coords,
    )

    evaluated = generate_quantities(model, results, new_data=_prepared_data(reverse=True))
    generated = evaluated["generated_quantities"]

    assert generated["coefficient_copy"].dims == ("chain", "draw", "group", "channel")
    np.testing.assert_array_equal(generated["channel"], ["video", "search"])
    np.testing.assert_array_equal(generated["coefficient_copy"], coefficient[None, None])
    np.testing.assert_allclose(
        generated["response"],
        (data.arrays["media"][-3:] * coefficient)[None, None],
    )


def test_auxiliary_inputs_remain_fixed_across_shorter_reordered_scenarios():
    data = _prepared_data()
    inputs = xr.Dataset(
        {"experiment_spend": (("experiment", "channel"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        coords={"experiment": ["north", "south", "national"], "channel": ["video", "search"]},
    )
    model = Model(
        parameters={"coefficient": Real(dims="channel")},
        log_density=lambda coefficient: -jnp.square(coefficient).sum(),
        generated_quantities=lambda key, media, experiment_spend, experiment_response: {
            "media_copy": media,
            "experiment_copy": experiment_spend,
            "experiment_response": experiment_response,
        },
        data=Data(data, inputs=inputs),
        transformed_parameters=lambda experiment_spend, coefficient: {
            "experiment_response": experiment_spend @ coefficient
        },
        generated_dims={"experiment_response": ("experiment",)},
    )
    coefficient = np.array([2.0, 3.0], dtype=jax.dtypes.canonicalize_dtype(float))
    results = _collect_results(
        {"coefficient": coefficient[None, None]}, data=data, inputs=inputs, dims=model._result_dims
    )
    scenario = pl.DataFrame(
        [
            {"week": week, "region": region, "segment": segment, "video": 20.0, "search": 10.0}
            for week in (8, 9)
            for region, segment in reversed(data.group_values)
        ]
    )
    prepared = model.prepare_data(scenario)
    np.testing.assert_array_equal(prepared.values["experiment_spend"], model.data.values["experiment_spend"])
    evaluated = generate_quantities(model, results, new_data=scenario)

    assert "observed_data" not in evaluated.children
    assert evaluated["constant_data"]["experiment_spend"].dims == ("experiment", "channel")
    np.testing.assert_array_equal(evaluated["constant_data"]["experiment"], inputs.experiment)
    np.testing.assert_array_equal(evaluated["constant_data"]["channel"], ["video", "search"])
    np.testing.assert_array_equal(evaluated["constant_data"]["media_time"], [8, 9])
    np.testing.assert_array_equal(evaluated["constant_data"]["experiment_spend"], inputs.experiment_spend)
    generated = evaluated["generated_quantities"]
    assert generated["experiment_copy"].dims == ("chain", "draw", "experiment", "channel")
    np.testing.assert_array_equal(generated["experiment_response"], [[[8.0, 18.0, 28.0]]])
    np.testing.assert_array_equal(generated["media_time"], [8, 9])
    np.testing.assert_array_equal(model.data.values["experiment_spend"], inputs.experiment_spend)
    np.testing.assert_array_equal(results["constant_data"]["media_time"], [1, 2, 3, 4])


@pytest.mark.parametrize("periods", [(8, 9), (8, 9, 10), (8, 9, 10, 11, 12)])
def test_generated_reference_inputs_keep_training_labels_for_new_observation_windows(periods):
    data = _prepared_data()
    scaling = fit_data_scaling(data, scale_outcome=True)

    def generated(key, media, outcome, reference, outcome_scaling, n_periods):
        predictive = {
            "restored_outcome": outcome_scaling.inverse_transform(outcome),
            "original_outcome": reference.outcome,
        }
        return {
            "current_media": media,
            "original_media": reference.media,
            "original_elapsed": reference.time,
            "original_media_elapsed": reference.media_time,
            "outcome_divisor": outcome_scaling.scale,
            "current_count": jnp.asarray(n_periods),
            "original_count": jnp.asarray(reference.n_periods),
            "predictive": predictive,
        }

    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=generated,
        data=Data(data, scaling=scaling),
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
    scenario = pl.DataFrame(
        [
            {
                "week": week,
                "region": region,
                "segment": segment,
                "sales": 100.0 + week,
                "video": 20.0,
                "search": 10.0,
            }
            for week in periods
            for region, segment in reversed(data.group_values)
        ]
    )
    evaluated = generate_quantities(model, results, new_data=scenario, batch_size=1)
    generated_values = evaluated["generated_quantities"]
    predictive = evaluated["posterior_predictive"]

    assert generated_values["current_media"].dims == ("chain", "draw", "media_time", "group", "channel")
    assert generated_values["original_media"].dims == ("chain", "draw", "reference_media_time", "group", "channel")
    assert predictive["original_outcome"].dims == ("chain", "draw", "reference_time", "group")
    assert predictive["restored_outcome"].dims == ("chain", "draw", "time", "group")
    assert generated_values["original_elapsed"].dims == ("chain", "draw", "reference_time")
    assert generated_values["original_media_elapsed"].dims == ("chain", "draw", "reference_media_time")
    assert generated_values["outcome_divisor"].dims == ("chain", "draw")
    np.testing.assert_array_equal(generated_values["media_time"], periods)
    np.testing.assert_array_equal(generated_values["reference_media_time"], data.media_time_values)
    np.testing.assert_array_equal(predictive["time"], periods)
    np.testing.assert_array_equal(predictive["reference_time"], data.time_values)
    np.testing.assert_array_equal(generated_values["channel"], data.channels)
    np.testing.assert_array_equal(generated_values["group_region"], ["west", "east"])
    np.testing.assert_array_equal(generated_values["current_count"], [[len(periods)] * 2])
    np.testing.assert_array_equal(generated_values["original_count"], [[len(data.time_values)] * 2])
    np.testing.assert_array_equal(generated_values["original_elapsed"][0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(generated_values["original_media_elapsed"][0, 0], [-1.0, 0.0, 1.0, 2.0])
    expected = scaling.transform(data)
    np.testing.assert_array_equal(generated_values["original_media"][0, 0], expected.arrays["media"])
    np.testing.assert_array_equal(predictive["original_outcome"][0, 0], expected.arrays["outcome"])
    restored = np.repeat(100.0 + np.array(periods), 2).reshape(-1, 2)
    np.testing.assert_allclose(predictive["restored_outcome"][0, 0], restored)
    assert set(evaluated["observed_data"].data_vars) == {"outcome"}
    assert set(evaluated["constant_data"].data_vars) == {"media"}


@pytest.mark.parametrize("prior", [False, True])
def test_generated_training_arrays_distinguish_current_and_reference_provenance(prior):
    data = _prepared_data()
    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=lambda key, media, reference: {"current_media": media, "original_media": reference.media},
        data=data,
    )
    if prior:
        evaluated = sample_prior(model, lambda key: {"level": jnp.asarray(0.0)}, draws=2)
        generated = evaluated["prior_generated_quantities"]
    else:
        results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
        evaluated = generate_quantities(model, results)
        generated = evaluated["generated_quantities"]

    assert generated["current_media"].dims == ("chain", "draw", "media_time", "group", "channel")
    assert generated["original_media"].dims == ("chain", "draw", "reference_media_time", "group", "channel")
    np.testing.assert_array_equal(generated["current_media"], generated["original_media"])
    np.testing.assert_array_equal(generated["reference_media_time"], data.media_time_values)
    assert set(evaluated["constant_data"].data_vars) == {"media"}


@pytest.mark.parametrize("groups", [1, 2])
def test_generated_population_outcome_scale_retains_group_axis_even_for_one_group(groups):
    frame = pl.DataFrame(
        [
            {"week": week, "region": str(group), "population": 100.0 * (group + 1), "sales": 50.0 * week}
            for week in (1, 2, 3)
            for group in range(groups)
        ]
    )
    data = prepare_data(frame, time="week", groups=["region"], outcome="sales", population="population")
    scaling = fit_data_scaling(data, scale_outcome="population")
    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=lambda key, outcome_scaling: {"outcome_divisor": outcome_scaling.scale},
        data=Data(data, scaling=scaling),
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
    generated = generate_quantities(model, results)["generated_quantities"]

    assert generated["outcome_divisor"].dims == ("chain", "draw", "group")
    np.testing.assert_array_equal(generated["group"], [str(group) for group in range(groups)])
    np.testing.assert_array_equal(generated["outcome_divisor"][0, 0], scaling.transformations["outcome"].scale[0])


@pytest.mark.parametrize("prior", [False, True])
def test_declared_data_variables_keep_prepared_and_auxiliary_result_labels(prior):
    data = _prepared_data()
    inputs = xr.Dataset(
        {"experiment_spend": (("experiment", "channel"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        coords={"experiment": ["north", "south", "national"], "channel": ["video", "search"]},
    )

    def generated(key, revenue, impressions, experiment_costs):
        predictive = {"sales": revenue, "experiment_costs": experiment_costs}
        quantities = {"exposure": impressions, "predictive": predictive}
        return quantities

    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=generated,
        data=Data(
            data,
            inputs=inputs,
            variables={
                "revenue": "outcome",
                "impressions": "media",
                "experiment_costs": "experiment_spend",
            },
        ),
    )
    if prior:
        evaluated = sample_prior(model, {"level": Prior(normal, location=0.0, scale=1.0)}, draws=2, batch_size=1)
        generated_values = evaluated["prior_generated_quantities"]
        predictive = evaluated["prior_predictive"]
    else:
        results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data, inputs=inputs)
        evaluated = generate_quantities(model, results, batch_size=1)
        generated_values = evaluated["generated_quantities"]
        predictive = evaluated["posterior_predictive"]

    assert predictive["sales"].dims == ("chain", "draw", "time", "group")
    assert generated_values["exposure"].dims == ("chain", "draw", "media_time", "group", "channel")
    assert predictive["experiment_costs"].dims == ("chain", "draw", "experiment", "channel")
    np.testing.assert_array_equal(predictive["time"], data.time_values)
    np.testing.assert_array_equal(predictive["group_region"], ["west", "east"])
    np.testing.assert_array_equal(generated_values["media_time"], data.media_time_values)
    np.testing.assert_array_equal(generated_values["channel"], data.channels)
    np.testing.assert_array_equal(predictive["experiment"], inputs.experiment)
    np.testing.assert_array_equal(predictive["sales"][0, 0], model.data.values["outcome"])
    np.testing.assert_array_equal(generated_values["exposure"][0, 0], model.data.values["media"])
    np.testing.assert_array_equal(predictive["experiment_costs"][0, 0], inputs.experiment_spend)
    assert set(evaluated["observed_data"].data_vars) == {"outcome"}
    assert set(evaluated["constant_data"].data_vars) == {"media", "experiment_spend"}


@pytest.mark.parametrize("groups", [1, 2])
def test_declared_outcome_scaling_variables_keep_group_labels(groups):
    frame = pl.DataFrame(
        [
            {"week": week, "region": str(group), "population": 100.0 * (group + 1), "sales": 50.0 * week}
            for week in (1, 2, 3)
            for group in range(groups)
        ]
    )
    data = prepare_data(frame, time="week", groups=["region"], outcome="sales", population="population")
    scaling = fit_data_scaling(data, scale_outcome="population")

    def generated(key, revenue, revenue_scaling):
        restored = revenue_scaling.inverse_transform(revenue)
        quantities = {
            "divisor": revenue_scaling.scale,
            "offset": revenue_scaling.offset,
            "predictive": {"restored": restored},
        }
        return quantities

    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=generated,
        data=Data(
            data,
            scaling=scaling,
            variables={"revenue": "outcome", "revenue_scaling": "outcome_scaling"},
        ),
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
    evaluated = generate_quantities(model, results, batch_size=1)
    generated_values = evaluated["generated_quantities"]

    assert generated_values["divisor"].dims == ("chain", "draw", "group")
    assert generated_values["offset"].dims == ("chain", "draw", "group")
    np.testing.assert_array_equal(generated_values["group"], [str(group) for group in range(groups)])
    np.testing.assert_allclose(generated_values["divisor"][0, 0], scaling.transformations["outcome"].scale[0])
    np.testing.assert_allclose(generated_values["offset"][0, 0], scaling.transformations["outcome"].offset[0])
    np.testing.assert_allclose(evaluated["posterior_predictive"]["restored"][0, 0], data.arrays["outcome"])


@pytest.mark.parametrize("periods", [(8, 9), (8, 9, 10), (8, 9, 10, 11, 12)])
def test_declared_reference_variables_keep_original_labels_in_scenarios(periods):
    data = _prepared_data()

    def generated(key, impressions, training):
        quantities = {
            "current": impressions,
            "original": training.media,
            "original_elapsed": training.time,
            "predictive": {"original_sales": training.outcome},
        }
        return quantities

    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=generated,
        data=Data(
            data,
            variables={"impressions": "media", "training": "reference"},
        ),
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
    scenario = pl.DataFrame(
        [
            {"week": week, "region": region, "segment": segment, "video": 20.0, "search": 10.0}
            for week in periods
            for region, segment in reversed(data.group_values)
        ]
    )
    evaluated = generate_quantities(model, results, new_data=scenario, batch_size=1)
    generated_values = evaluated["generated_quantities"]
    predictive = evaluated["posterior_predictive"]

    assert generated_values["current"].dims == ("chain", "draw", "media_time", "group", "channel")
    assert generated_values["original"].dims == ("chain", "draw", "reference_media_time", "group", "channel")
    assert predictive["original_sales"].dims == ("chain", "draw", "reference_time", "group")
    assert generated_values["original_elapsed"].dims == ("chain", "draw", "reference_time")
    np.testing.assert_array_equal(generated_values["media_time"], periods)
    np.testing.assert_array_equal(generated_values["reference_media_time"], data.media_time_values)
    np.testing.assert_array_equal(predictive["reference_time"], data.time_values)
    np.testing.assert_array_equal(generated_values["channel"], data.channels)
    np.testing.assert_array_equal(generated_values["group_region"], ["west", "east"])
    np.testing.assert_array_equal(generated_values["current"][0, 0], model.prepare_data(scenario).values["media"])
    np.testing.assert_array_equal(generated_values["original"][0, 0], model.data.values["media"])
    np.testing.assert_array_equal(predictive["original_sales"][0, 0], model.data.values["outcome"])
    np.testing.assert_array_equal(generated_values["original_elapsed"][0, 0], [0.0, 1.0, 2.0])
    assert "observed_data" not in evaluated.children
    assert set(evaluated["constant_data"].data_vars) == {"media"}


def test_result_collection_rejects_auxiliary_coordinate_alignment():
    inputs = xr.Dataset({"offset": ("channel", [1.0, 2.0])}, coords={"channel": ["search", "video"]})
    with pytest.raises(ValueError, match="Coordinate 'channel' conflicts"):
        _collect_results({"location": np.zeros((1, 1))}, data=_prepared_data(), inputs=inputs)


@pytest.mark.parametrize("name", ["outcome", "media"])
def test_result_collection_rejects_auxiliary_overwrite_of_prepared_data(name):
    with pytest.raises(ValueError, match="conflict with prepared data variables"):
        _collect_results({"location": np.zeros((1, 1))}, data=_prepared_data(), inputs=xr.Dataset({name: 1.0}))


def test_single_names_are_accepted_for_dims_and_generated_dims():
    data = _prepared_data()

    def transformed(outcome, level):
        return {"mu": outcome * 0 + level}

    def build(level_axes, mu_axes):
        return Model(
            parameters={"level": Real((2,))},
            log_density=lambda outcome, mu, level: -jnp.square(outcome - mu).sum() - jnp.square(level).sum(),
            generated_quantities=lambda key, mu: {"mu": mu},
            data=data,
            transformed_parameters=transformed,
            dims={"level": level_axes},
            coords={"unit": ["first", "second"]},
            generated_dims={"mu": mu_axes},
        )

    single = build("unit", "time")
    sequence = build(("unit",), ("time",))
    assert single._result_dims == sequence._result_dims == {"level": ("unit",)}
    assert single._generated_dims == sequence._generated_dims == {"mu": ("time",)}
