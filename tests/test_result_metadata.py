"""Tests for result labels attached to model definitions without invoking a sampler."""

from datetime import date

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
    Simplex,
    fit_data_scaling,
    generate_quantities,
    prepare_data,
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
    assert model._time_values == ()
    assert model._media_time_values == ()


def test_result_metadata_copies_mappings_axes_coordinates_and_output_names():
    dims = {"coefficient": ["channel"], "intercept": []}
    labels = np.array(["video", "search"])
    coords = {"channel": labels, "time": [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15)]}
    generated_dims = {"outcome": ["time"], "pointwise": ["time"], "mean": ["time"]}
    predictive = ["outcome"]
    likelihood = ["pointwise"]
    model = _plain_model(
        generate=_generate,
        dims=dims,
        coords=coords,
        generated_dims=generated_dims,
        predictive=predictive,
        log_likelihood=likelihood,
    )
    dims["coefficient"].append("another")
    dims["extra"] = []
    labels[:] = "edited"
    coords["time"].clear()
    coords["extra"] = [1]
    generated_dims["outcome"].clear()
    predictive.append("mean")
    likelihood.clear()

    assert model._result_dims == {"coefficient": ("channel",), "intercept": ()}
    np.testing.assert_array_equal(model._result_coords["channel"], ["video", "search"])
    np.testing.assert_array_equal(
        model._result_coords["time"], np.array(["2026-01-01", "2026-01-08", "2026-01-15"], dtype="datetime64[us]")
    )
    assert set(model._result_coords) == {"channel", "time"}
    assert model._generated_dims == {"outcome": ("time",), "pointwise": ("time",), "mean": ("time",)}
    assert model._predictive_names == ("outcome",)
    assert model._likelihood_names == ("pointwise",)


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


@pytest.mark.parametrize("option", ["predictive", "log_likelihood", "save"])
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
        components=[],
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
            components=[],
            transformed_parameters=lambda: {"mu": jnp.zeros(3)},
            save=(name,),
        )


def test_save_rejects_unknown_component_without_transformed_parameters():
    with pytest.raises(ValueError, match="transformed quantity or component contribution"):
        Model({}, lambda: jnp.array(0.0), data=_prepared_data(), components=[], save=("missing",))


def test_predictive_and_likelihood_names_cannot_overlap():
    with pytest.raises(ValueError, match="different generated outputs"):
        _plain_model(generate=_generate, predictive=["outcome"], log_likelihood=["outcome"])


@pytest.mark.parametrize(
    "metadata",
    [
        {"predictive": ["outcome"]},
        {"log_likelihood": ["pointwise"]},
        {"generated_dims": {"mean": ("time",)}},
    ],
)
def test_generated_metadata_requires_generation_callback(metadata):
    with pytest.raises(ValueError, match="requires a generate callback"):
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
    )
    assert model._generated_dims == {"later_output": ("new_axis",)}
    assert model._predictive_names == ("later_output",)


def test_component_parameters_and_generated_names_have_separate_dimension_maps():
    data = _prepared_data()

    def density(outcome, annual, paid_media_total):
        return -jnp.square(outcome - annual - paid_media_total).sum()

    def generate(key, annual):
        return {"annual": annual}

    model = Model(
        {},
        density,
        generate,
        data=data,
        components=[
            FourierSeasonality(period=52, name="annual", group_specific_coefficients=True),
            MediaEffect(max_lag=1, group_specific_coefficients=True),
        ],
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
            {},
            lambda annual: jnp.sum(annual),
            data=data,
            components=[FourierSeasonality(period=52, name="annual")],
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
        data=incoming,
        components=[],
        scaling=scaling,
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
    for name, expected_array in plain.generate(key, plain.constrain(position), None).items():
        np.testing.assert_array_equal(labeled.generate(key, labeled.constrain(position), None)[name], expected_array)


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
        components=[],
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
        components=[],
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
            components=[],
        )


@pytest.mark.parametrize("axis", ["unknown", "control", "treatment", "organic_channel"])
def test_declared_axes_must_be_available_at_model_construction(axis):
    with pytest.raises(ValueError, match=axis):
        Model(
            {"coefficient": Real(dims=axis)},
            lambda coefficient: coefficient.sum(),
            data=_prepared_data(),
            components=[],
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
            components=[],
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
        components=[],
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
