"""Tests for result labels attached to model definitions without invoking a sampler."""

from datetime import date

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
    Prior,
    Real,
    Simplex,
    dirichlet,
    fit_data_scaling,
    fourier_features,
    generate_quantities,
    lognormal,
    normal,
    prepare_data,
    sample_prior,
)
from mmmjax._results import _collect_results


def _plain_model(parameters=None, generate=None, **metadata):
    declarations = {"coefficient": Real(shape=(2,)), "intercept": Real()} if parameters is None else parameters

    def density(data, **values):
        return -sum((jnp.square(value).sum() for value in values.values()), start=jnp.array(0.0))

    return Model(declarations, density, generate, **metadata)


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
    assert model._predictive_names == ()
    assert model._likelihood_names == ()
    assert model._log_prior_names == ()
    assert model._time_values == ()
    assert model._media_time_values == ()


@pytest.mark.parametrize("names", [(), ("coefficient",), ("coefficient", "intercept", "extra")])
def test_registered_priors_require_exact_parameter_names(names):
    with pytest.raises(ValueError, match=r"prior|parameter"):
        _plain_model(prior={name: Prior(normal, location=0.0, scale=1.0) for name in names})


@pytest.mark.parametrize("distribution", [normal, 1.0, None])
def test_registered_prior_entries_require_prior_objects(distribution):
    with pytest.raises(TypeError, match="Prior"):
        _plain_model(prior={"coefficient": distribution, "intercept": Prior(normal, location=0.0, scale=1.0)})


@pytest.mark.parametrize("event", [False, True])
def test_registered_prior_shapes_must_match_parameter_declarations(event):
    parameter = Simplex((3,)) if event else Real((3,))
    distribution = (
        Prior(dirichlet, concentration=jnp.ones(2)) if event else Prior(normal, location=jnp.zeros(2), scale=1.0)
    )
    with pytest.raises(ValueError, match=r"shape.*coefficient"):
        _plain_model(parameters={"coefficient": parameter}, prior={"coefficient": distribution})


@pytest.mark.parametrize("include_prior", [False, True])
def test_registered_priors_do_not_implicitly_change_jitted_density_or_gradient(include_prior):
    prior = Prior(lognormal, location=0.2, scale=0.9)

    def density(data, scale):
        likelihood = normal(data, scale, 0.5)
        return likelihood + prior(scale) if include_prior else likelihood

    plain = Model({"scale": Positive()}, density)
    registered = Model({"scale": Positive()}, density, prior={"scale": prior})
    values = {"scale": jnp.array(2.0)}
    position = {"scale": jnp.log(values["scale"])}
    for method, arguments in (("log_prob", values), ("log_density", position)):
        expected = jax.jit(jax.value_and_grad(getattr(plain, method)))(arguments, 1.5)
        actual = jax.jit(jax.value_and_grad(getattr(registered, method)))(arguments, 1.5)
        for actual_array, expected_array in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.parametrize("selection", ["predictive", "log_likelihood"])
def test_registered_log_priors_cannot_be_selected_as_other_generated_groups(selection):
    with pytest.raises(ValueError, match=r"log_prior|outputs"):
        _plain_model(
            parameters={"coefficient": Real()},
            prior={"coefficient": Prior(normal, location=0.0, scale=1.0)},
            **{selection: ("log_prior_coefficient",)},
        )


def test_result_metadata_copies_mappings_axes_coordinates_and_output_names():
    dims = {"coefficient": ["channel"], "intercept": []}
    labels = np.array(["video", "search"])
    coords = {"channel": labels, "time": [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15)]}
    generated_dims = {"outcome": ["time"], "pointwise": ["time"], "mean": ["time"]}
    predictive = ["outcome"]
    likelihood = ["pointwise"]
    prior = ["lp_coefficient"]
    model = _plain_model(
        generate=_generate,
        dims=dims,
        coords=coords,
        generated_dims=generated_dims,
        predictive=predictive,
        log_likelihood=likelihood,
        log_prior=prior,
    )
    dims["coefficient"].append("another")
    dims["extra"] = []
    labels[:] = "edited"
    coords["time"].clear()
    coords["extra"] = [1]
    generated_dims["outcome"].clear()
    predictive.append("mean")
    likelihood.clear()
    prior.clear()

    assert model._result_dims == {"coefficient": ("channel",), "intercept": ()}
    np.testing.assert_array_equal(model._result_coords["channel"], ["video", "search"])
    np.testing.assert_array_equal(
        model._result_coords["time"], np.array(["2026-01-01", "2026-01-08", "2026-01-15"], dtype="datetime64[us]")
    )
    assert set(model._result_coords) == {"channel", "time"}
    assert model._generated_dims == {"outcome": ("time",), "pointwise": ("time",), "mean": ("time",)}
    assert model._predictive_names == ("outcome",)
    assert model._likelihood_names == ("pointwise",)
    assert model._log_prior_names == ("lp_coefficient",)


@pytest.mark.parametrize(
    "metadata,error,message",
    [
        ({"dims": []}, TypeError, "dims must map"),
        ({"dims": {"unknown": ()}}, ValueError, "undeclared parameter"),
        ({"dims": {"coefficient": "channel"}}, TypeError, "sequence"),
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


@pytest.mark.parametrize("option", ["predictive", "log_likelihood", "log_prior", "save"])
@pytest.mark.parametrize(
    "value,error,message",
    [
        ("outcome", TypeError, "sequence"),
        (None, TypeError, "sequence"),
        ({"outcome"}, TypeError, "sequence"),
        ([""], ValueError, "nonempty string"),
        ([1], ValueError, "nonempty string"),
        (["outcome", "outcome"], ValueError, "duplicate"),
    ],
)
def test_generated_group_names_are_validated(option, value, error, message):
    with pytest.raises(error, match=message):
        _plain_model(generate=_generate, **{option: value})


def test_save_requires_prepared_data():
    with pytest.raises(ValueError, match="save requires prepared data"):
        _plain_model(save=("mu",))


def test_save_selection_is_copied_without_probing_callbacks():
    def transformed():
        raise AssertionError("Construction must not evaluate transformed quantities")

    selection = ["mu"]
    model = Model(
        {},
        lambda mu: mu.sum(),
        data=_prepared_data(),
        transformed_parameters=transformed,
        save=selection,
        generated_dims={"mu": ("time", "group")},
    )
    selection.clear()
    assert model._saved_inputs == (("mu", "transformed"),)


@pytest.mark.parametrize("name", ["outcome", "intercept", "not a name", "class"])
def test_save_rejects_data_parameter_and_invalid_names(name):
    with pytest.raises(ValueError, match=r"save|saved"):
        Model(
            {"intercept": Real()},
            lambda intercept: -(intercept**2),
            data=_prepared_data(),
            transformed_parameters=lambda: {"mu": jnp.zeros(3)},
            save=(name,),
        )


def test_save_requires_transformed_parameters():
    with pytest.raises(ValueError, match="transformed quantity"):
        Model({}, lambda: jnp.array(0.0), data=_prepared_data(), save=("missing",))


@pytest.mark.parametrize(
    "first,second", [("predictive", "log_likelihood"), ("predictive", "log_prior"), ("log_likelihood", "log_prior")]
)
def test_generated_group_names_cannot_overlap(first, second):
    with pytest.raises(ValueError, match="outputs"):
        _plain_model(generate=_generate, **{first: ["outcome"], second: ["outcome"]})


@pytest.mark.parametrize(
    "metadata",
    [
        {"predictive": ["outcome"]},
        {"log_likelihood": ["pointwise"]},
        {"log_prior": ["lp_coefficient"]},
        {"generated_dims": {"mean": ("time",)}},
    ],
)
def test_generated_metadata_requires_generation_callback(metadata):
    with pytest.raises(ValueError, match="requires a generated_quantities callback"):
        _plain_model(**metadata)


@pytest.mark.parametrize(
    "generated_dims,error",
    [([], TypeError), ({"mean": "time"}, TypeError), ({"mean": ("chain",)}, ValueError)],
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
        predictive=["later_output"],
        log_likelihood=["pointwise"],
        log_prior=["lp_coefficient"],
    )
    assert model._generated_dims == {"later_output": ("new_axis",)}
    assert model._predictive_names == ("later_output",)


def test_parameters_and_generated_names_have_separate_dimension_maps():
    data = _prepared_data()

    def density(outcome, annual, paid_media_coefficient):
        return -jnp.square(outcome).sum() - jnp.square(annual).sum() - jnp.square(paid_media_coefficient).sum()

    def generate(key, time, annual):
        return {"annual": fourier_features(time, period=52, order=2) @ annual}

    model = Model(
        {"annual": Real((4, 2)), "paid_media_coefficient": Real((2, 2))},
        density,
        generate,
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
            {"annual": Real((4,))},
            lambda annual: jnp.sum(annual),
            data=data,
            dims={"annual_coefficients": ("annual_mode",)},
        )


def test_aligned_time_history_and_group_metadata_follow_actual_scaled_model_inputs():
    reference = _prepared_data()
    incoming = _prepared_data(reverse=True)
    scaling = fit_data_scaling(reference, scale_outcome=True)
    expected = scaling.transform(incoming)
    model = Model(
        {},
        lambda outcome: -jnp.square(outcome).sum(),
        data=DataBlock(incoming, scaling=scaling),
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
        predictive=["outcome"],
        log_likelihood=["pointwise"],
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
        {"coefficient": Real(dims=axis)},
        lambda coefficient: -jnp.square(coefficient).sum(),
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
        {"weights": Simplex(dims=("group", "channel"))},
        lambda weights: jnp.log(weights).sum(),
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
            {"coefficient": Real(shape=(3,), dims="channel")},
            lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
        )


@pytest.mark.parametrize("axis", ["unknown", "control", "treatment", "organic_channel"])
def test_declared_axes_must_be_available_at_model_construction(axis):
    with pytest.raises(ValueError, match=axis):
        Model(
            {"coefficient": Real(dims=axis)},
            lambda coefficient: coefficient.sum(),
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
            {"coefficient": Real(dims="channel")},
            lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
            coords={"channel": ["search", "video"]},
        )


def test_named_parameter_axes_survive_reordered_scenario_inputs():
    data = _prepared_data()
    model = Model(
        {"coefficient": Real(dims=("group", "channel"))},
        lambda coefficient: -jnp.square(coefficient).sum(),
        lambda key, media, coefficient: {
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
        {"coefficient": Real(dims="channel")},
        lambda coefficient: -jnp.square(coefficient).sum(),
        lambda key, media, experiment_spend: {"media_copy": media, "experiment_copy": experiment_spend},
        data=DataBlock(data, inputs=inputs),
        transformed_parameters=lambda experiment_spend, coefficient: {
            "experiment_response": experiment_spend @ coefficient
        },
        save=("experiment_response",),
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

    def generated(
        key,
        media,
        outcome,
        reference_media,
        reference_outcome,
        reference_time,
        reference_media_time,
        outcome_scale,
        unscale_outcome,
        n_periods,
        reference_n_periods,
    ):
        return {
            "current_media": media,
            "original_media": reference_media,
            "original_outcome": reference_outcome,
            "original_elapsed": reference_time,
            "original_media_elapsed": reference_media_time,
            "restored_outcome": unscale_outcome(outcome),
            "outcome_divisor": outcome_scale,
            "current_count": jnp.asarray(n_periods),
            "original_count": jnp.asarray(reference_n_periods),
        }

    model = Model(
        {"level": Real()},
        lambda level: -jnp.square(level),
        generated,
        data=DataBlock(data, scaling=scaling),
        predictive=("restored_outcome", "original_outcome"),
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
        {"level": Real()},
        lambda level: -jnp.square(level),
        lambda key, media, reference_media: {"current_media": media, "original_media": reference_media},
        data=data,
        prior=lambda key: {"level": jnp.asarray(0.0)},
    )
    if prior:
        evaluated = sample_prior(model, draws=2)
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
        {"level": Real()},
        lambda level: -jnp.square(level),
        lambda key, outcome_scale: {"outcome_divisor": outcome_scale},
        data=DataBlock(data, scaling=scaling),
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
        quantities = {
            "sales": revenue,
            "exposure": impressions,
            "experiment_costs": experiment_costs,
        }
        return quantities

    model = Model(
        {"level": Real()},
        lambda level: -jnp.square(level),
        generated,
        data=DataBlock(
            data,
            inputs=inputs,
            variables={
                "revenue": "outcome",
                "impressions": "media",
                "experiment_costs": "experiment_spend",
            },
        ),
        prior={"level": Prior(normal, location=0.0, scale=1.0)},
        predictive=("sales", "experiment_costs"),
    )
    if prior:
        evaluated = sample_prior(model, draws=2, batch_size=1)
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

    def generated(key, revenue, revenue_scale, revenue_offset):
        restored = revenue * revenue_scale + revenue_offset
        quantities = {"divisor": revenue_scale, "offset": revenue_offset, "restored": restored}
        return quantities

    model = Model(
        {"level": Real()},
        lambda level: -jnp.square(level),
        generated,
        data=DataBlock(
            data,
            scaling=scaling,
            variables={
                "revenue": "outcome",
                "revenue_scale": "outcome_scale",
                "revenue_offset": "outcome_offset",
            },
        ),
        predictive=("restored",),
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

    def generated(key, impressions, original_impressions, original_revenue, original_elapsed):
        quantities = {
            "current": impressions,
            "original": original_impressions,
            "original_sales": original_revenue,
            "original_elapsed": original_elapsed,
        }
        return quantities

    model = Model(
        {"level": Real()},
        lambda level: -jnp.square(level),
        generated,
        data=DataBlock(
            data,
            variables={
                "impressions": "media",
                "original_impressions": "reference_media",
                "original_revenue": "reference_outcome",
                "original_elapsed": "reference_time",
            },
        ),
        predictive=("original_sales",),
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
