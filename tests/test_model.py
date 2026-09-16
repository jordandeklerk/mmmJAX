"""Tests for composing parameter declarations into JAX models."""

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    CorrelationCholesky,
    Data,
    Model,
    Positive,
    Real,
    Simplex,
    exponential,
    half_normal,
    lkj_cholesky,
    lognormal,
    multivariate_normal,
    normal,
    normal_rng,
    prepare_data,
)


def test_explicit_noncentered_correlated_hierarchy_evaluates_and_differentiates(auxiliary_data):
    def quantities(location, scales, factor, offsets, media):
        scale_tril = scales[:, None] * factor
        coefficients = location + offsets @ scale_tril.T
        mu = jnp.sum(media * coefficients, axis=-1)
        return {"coefficients": coefficients, "mu": mu}

    def density(outcome, mu, location, scales, factor, offsets):
        target = normal(location, 0.0, 1.0)
        target += half_normal(scales, 1.0)
        target += lkj_cholesky(factor, 2.0)
        target += normal(offsets, 0.0, 1.0)
        target += normal(outcome, mu, 1.0)
        return target

    model = Model(
        {
            "location": Real(dims="channel"),
            "scales": Positive(dims="channel"),
            "factor": CorrelationCholesky(dims=("channel", "channel_to")),
            "offsets": Real((2, 2)),
        },
        density,
        data=auxiliary_data,
        transformed_parameters=quantities,
        coords={"channel_to": auxiliary_data.channels},
        save=("mu",),
    )
    values = {
        "location": jnp.array([0.2, 0.3]),
        "scales": jnp.array([0.5, 0.8]),
        "factor": jnp.array([[1.0, 0.0], [0.6, 0.8]]),
        "offsets": jnp.array([[0.1, -0.2], [0.4, 0.3]]),
    }
    result = model.evaluate(values)
    covariance_factor = values["scales"][:, None] * values["factor"]
    expected = values["location"] + values["offsets"] @ covariance_factor.T
    np.testing.assert_allclose(result["coefficients"], expected, rtol=2e-6)

    # Changing from group coefficients to standard-Normal offsets adds one
    # scale determinant per group, recovering the independent offset density.
    centered = multivariate_normal(expected, values["location"], covariance_factor)
    coefficient_adjustment = 2 * jnp.log(jnp.diag(covariance_factor)).sum()
    np.testing.assert_allclose(centered + coefficient_adjustment, normal(values["offsets"], 0, 1), rtol=3e-6)

    position = model.unconstrain(values)
    adjusted, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    correction = sum(
        declaration.log_density_adjustment(position[name]) for name, declaration in model.parameters.items()
    )
    np.testing.assert_allclose(adjusted, model.log_prob(values) + correction, rtol=3e-6)
    assert all(jnp.all(jnp.isfinite(value)) for value in gradient.values())


@pytest.fixture
def auxiliary_data():
    return prepare_data(
        pl.DataFrame(
            {
                "period": [0, 0, 1, 1],
                "region": ["east", "west", "east", "west"],
                "sales": [2.0, 3.0, 4.0, 5.0],
                "video": [1.0, 2.0, 3.0, 4.0],
                "search": [5.0, 6.0, 7.0, 8.0],
            }
        ),
        time="period",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
    )


@pytest.fixture
def auxiliary_inputs():
    return xr.Dataset(
        {
            "lift": ("experiment", [1.5, -0.25, 0.75]),
            "indices": ("experiment", np.array([1, 0, 1], dtype=np.int64)),
            "active": ("experiment", [True, False, True]),
            "offset": 0.5,
        },
        coords={"experiment": ["first", "second", "third"]},
    )


def test_labeled_inputs_reach_named_callbacks_and_have_analytic_jit_gradients(auxiliary_data, auxiliary_inputs):
    def transformed(theta, beta, indices, offset):
        return {"trial_mean": theta + beta[indices] + offset}

    def density(outcome, lift, active, trial_mean, theta, beta):
        residual = jnp.where(active, lift - trial_mean, 0.0)
        return -0.5 * (jnp.square(outcome - theta.sum()).sum() + jnp.square(residual).sum() + jnp.square(beta).sum())

    def generated(key, *, trial_mean, indices, active, offset):
        return {"prediction": normal_rng(key, trial_mean, offset), "indices": indices, "active": active}

    model = Model(
        {"theta": Real(dims="experiment"), "beta": Real(dims="channel")},
        density,
        generated,
        data=Data(auxiliary_data, inputs=auxiliary_inputs),
        transformed_parameters=transformed,
        save=("trial_mean",),
    )
    parameters = {"theta": jnp.array([0.2, -0.1, 0.3]), "beta": jnp.array([0.4, -0.2])}
    mean = np.array([0.5, 0.8, 0.6])
    residual = np.array([1.0, 0.0, 0.15])
    observations = np.array([2.0, 3.0, 4.0, 5.0]) - 0.4
    expected = -0.5 * (np.square(observations).sum() + np.square(residual).sum() + 0.2)
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(parameters, model.data)

    assert model.parameters["theta"].shape == (3,)
    assert model.parameters["beta"].shape == (2,)
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(model.log_prob(parameters), expected, rtol=2e-6)
    np.testing.assert_allclose(gradient["theta"], observations.sum() + residual, rtol=2e-6)
    np.testing.assert_allclose(gradient["beta"], [-0.4, 1.35], rtol=2e-6)
    np.testing.assert_allclose(jax.jit(model.evaluate)(parameters)["trial_mean"], mean, rtol=2e-6)
    key = jax.random.key(17)
    generated_values = jax.jit(model.generate_quantities)(key, parameters, model.data)
    np.testing.assert_allclose(generated_values["trial_mean"], mean, rtol=2e-6)
    np.testing.assert_allclose(generated_values["prediction"], normal_rng(key, jnp.asarray(mean), 0.5), rtol=2e-6)
    np.testing.assert_array_equal(generated_values["indices"], [1, 0, 1])
    np.testing.assert_array_equal(generated_values["active"], [True, False, True])
    assert generated_values["indices"].dtype == jax.dtypes.canonicalize_dtype(np.int64)
    assert generated_values["active"].dtype == np.bool_


def test_labeled_inputs_copy_source_values_labels_and_returned_data(auxiliary_data, auxiliary_inputs):
    model = Model({}, lambda lift: -jnp.square(lift).sum(), data=Data(auxiliary_data, inputs=auxiliary_inputs))
    auxiliary_inputs["lift"].values[:] = 100.0
    auxiliary_inputs["indices"].values[:] = 0
    auxiliary_inputs["experiment"].values[:] = ["other", "value", "names"]
    returned = model.data
    returned.values.pop("active")

    np.testing.assert_array_equal(model.data.values["lift"], [1.5, -0.25, 0.75])
    np.testing.assert_array_equal(model.data.values["indices"], [1, 0, 1])
    np.testing.assert_array_equal(model._input_coords["experiment"], ["first", "second", "third"])
    assert "active" in model.data.values
    np.testing.assert_allclose(jax.jit(model.log_prob)({}), -2.875)


def test_labeled_inputs_remain_unscaled_and_fixed_in_new_observation_scenarios(auxiliary_data, auxiliary_inputs):
    model = Model(
        {"theta": Real(dims="experiment")},
        lambda outcome, trial_mean: normal(outcome, trial_mean.sum(), 1.0),
        lambda key, trial_mean, lift, indices, active, media: {
            "trial_mean": trial_mean,
            "lift": lift,
            "indices": indices,
            "active": active,
            "media": media,
        },
        data=Data(auxiliary_data, inputs=auxiliary_inputs, scaling="auto"),
        transformed_parameters=lambda theta, lift: {"trial_mean": theta + lift},
    )
    future = pl.DataFrame(
        {
            "period": [2, 2, 3, 3, 4, 4],
            "region": ["west", "east"] * 3,
            "video": [6.0, 5.0, 8.0, 7.0, 10.0, 9.0],
            "search": [10.0, 9.0, 12.0, 11.0, 14.0, 13.0],
        }
    )
    parameters = {"theta": jnp.array([0.2, -0.1, 0.3])}
    scenario = model.prepare_data(future)
    generated = jax.jit(model.generate_quantities)(jax.random.key(0), parameters, scenario)

    assert "outcome" not in scenario.values
    assert generated["media"].shape == (3, 2, 2)
    assert model.data.values["media"].shape == (2, 2, 2)
    for name in ("lift", "indices", "active"):
        np.testing.assert_array_equal(generated[name], auxiliary_inputs[name].values)
        np.testing.assert_array_equal(scenario.values[name], model.data.values[name])
    np.testing.assert_allclose(generated["trial_mean"], [1.7, -0.35, 1.05], rtol=2e-6)
    with pytest.raises(ValueError, match="outcome"):
        model.log_prob(parameters, scenario)


def test_labeled_inputs_require_prepared_data(auxiliary_inputs):
    with pytest.raises(TypeError, match="Data requires PreparedData"):
        Model({}, lambda data: jnp.array(0.0), data=Data(None, inputs=auxiliary_inputs))


@pytest.mark.parametrize("inputs", [{"lift": [1.0]}, xr.DataArray([1.0]), [1.0]])
def test_labeled_inputs_require_an_xarray_dataset(auxiliary_data, inputs):
    with pytest.raises(TypeError, match=r"inputs must be an xarray\.Dataset"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))


@pytest.mark.parametrize(
    "name",
    [
        "outcome",
        "media",
        "organic_media",
        "reach",
        "media_frequency",
        "organic_reach",
        "organic_frequency",
        "spend",
        "rf_spend",
        "controls",
        "treatments",
        "population",
        "revenue_per_outcome",
        "time",
        "media_time",
    ],
)
def test_labeled_inputs_cannot_use_prepared_roles_even_when_unselected(auxiliary_data, name):
    with pytest.raises(ValueError, match="conflicts with a data role"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=xr.Dataset({name: 1.0})))


@pytest.mark.parametrize("name", ["media", "controls", "group_region", "group_segment"])
@pytest.mark.parametrize("placement", ["variable", "dimension", "coordinate"])
def test_labeled_inputs_reject_role_and_group_label_namespaces(name, placement):
    data = prepare_data(
        pl.DataFrame(
            {
                "period": [0, 0, 1, 1],
                "region": ["east", "west"] * 2,
                "segment": ["retail"] * 4,
                "video": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        time="period",
        groups=["region", "segment"],
        media=["video"],
    )
    if placement == "variable":
        inputs = xr.Dataset({name: 1.0})
        message = "conflicts with a data role"
    else:
        inputs = xr.Dataset({"weights": (name, [1.0, 2.0])})
        if placement == "coordinate":
            inputs = inputs.assign_coords({name: ["first", "second"]})
        message = "conflict with prepared data variables"
    with pytest.raises(ValueError, match=message):
        Model({}, lambda: jnp.array(0.0), data=Data(data, inputs=inputs))


@pytest.mark.parametrize("name", ["theta", "group", "channel", "external_axis"])
def test_labeled_input_names_cannot_shadow_parameters_or_coordinates(auxiliary_data, name):
    with pytest.raises(ValueError, match="conflicts with a data role, parameter, or coordinate"):
        Model(
            {"theta": Real()},
            lambda theta: -(theta**2),
            data=Data(auxiliary_data, inputs=xr.Dataset({name: 1.0})),
            coords={"external_axis": ["first"]},
        )


@pytest.mark.parametrize("name", ["", "not-valid", "class", 1])
def test_labeled_input_names_must_be_valid_identifiers(auxiliary_data, name):
    with pytest.raises(ValueError, match=r"nonempty string|valid non-keyword Python identifier"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=xr.Dataset({name: 1.0})))


@pytest.mark.parametrize(
    "values",
    [np.array([1.0 + 0.0j]), np.array(["1"]), np.array([1], dtype=object), [np.nan], [np.inf], [-np.inf]],
)
def test_labeled_inputs_require_finite_real_values_or_booleans(auxiliary_data, values):
    with pytest.raises(ValueError, match="finite real numbers or booleans"):
        Model(
            {},
            lambda: jnp.array(0.0),
            data=Data(auxiliary_data, inputs=xr.Dataset({"lift": ("experiment", values)})),
        )


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint32", "uint64"])
@pytest.mark.parametrize("byteorder", ["=", "S"], ids=["native", "non-native"])
@pytest.mark.parametrize("enable_x64", [False, True])
def test_labeled_inputs_normalize_byte_order_without_changing_values(auxiliary_data, dtype, byteorder, enable_x64):
    values = np.array([1, 3, 7], dtype=np.dtype(dtype).newbyteorder(byteorder))
    inputs = xr.Dataset({"lift": ("experiment", values)})

    with jax.enable_x64(enable_x64):
        model = Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))
        converted = model.data.values["lift"]

        assert converted.dtype == jax.dtypes.canonicalize_dtype(dtype)
        np.testing.assert_array_equal(converted, values)

    assert inputs["lift"].dtype == values.dtype
    np.testing.assert_array_equal(inputs["lift"].values, values)


@pytest.mark.parametrize("byteorder", ["=", "S"], ids=["native", "non-native"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.array([2**31], dtype=np.int64), "integers outside the JAX dtype range"),
        (np.array([-(2**31) - 1], dtype=np.int64), "integers outside the JAX dtype range"),
        (np.array([2**32], dtype=np.uint64), "integers outside the JAX dtype range"),
        (np.array([1e100], dtype=np.float64), "not finite at the current JAX precision"),
    ],
)
def test_labeled_inputs_reject_values_that_overflow_jax_precision(auxiliary_data, value, message, byteorder):
    value = value.astype(value.dtype.newbyteorder(byteorder))
    inputs = xr.Dataset({"lift": ("experiment", value)})
    with jax.enable_x64(False), pytest.raises(ValueError, match=message):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))
    with jax.enable_x64(True):
        model = Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))
        np.testing.assert_array_equal(model.data.values["lift"], value)


@pytest.mark.parametrize("axis", ["group", "channel"])
@pytest.mark.parametrize("mismatch", ["reordered", "different", "missing", "length"])
def test_shared_input_axes_require_exact_labels_without_alignment(auxiliary_data, axis, mismatch):
    inputs = xr.Dataset(
        {"weights": (("group", "channel"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"group": ["east", "west"], "channel": ["video", "search"]},
    )
    model = Model({}, lambda media, weights: -(media * weights).sum(), data=Data(auxiliary_data, inputs=inputs))
    expected = -(auxiliary_data.arrays["media"] * inputs["weights"].values).sum()
    np.testing.assert_allclose(jax.jit(model.log_prob)({}), expected)

    if mismatch == "reordered":
        inputs = inputs.isel({axis: [1, 0]})
    elif mismatch == "different":
        inputs = inputs.assign_coords({axis: ["first", "second"]})
    elif mismatch == "missing":
        inputs = inputs.drop_vars(axis)
    else:
        inputs = inputs.isel({axis: [0]})
    with pytest.raises(ValueError, match=f"Input coordinate '{axis}' must match the model labels and ordering"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))


def test_input_axis_labels_must_agree_with_explicit_model_coordinates(auxiliary_data, auxiliary_inputs):
    with pytest.raises(ValueError, match="Input coordinate 'experiment' must match"):
        Model(
            {},
            lambda: jnp.array(0.0),
            data=Data(auxiliary_data, inputs=auxiliary_inputs),
            coords={"experiment": ["third", "second", "first"]},
        )
    with pytest.raises(ValueError, match="Dimension 'experiment' must have length 4"):
        Model(
            {"theta": Real(shape=(4,), dims="experiment")},
            lambda theta: -jnp.square(theta).sum(),
            data=Data(auxiliary_data, inputs=auxiliary_inputs),
        )


def test_unlabeled_independent_input_axes_supply_positional_coordinates(auxiliary_data):
    model = Model(
        {"theta": Real(dims="experiment")},
        lambda theta, lift: normal(lift, theta, 1.0),
        data=Data(auxiliary_data, inputs=xr.Dataset({"lift": ("experiment", [1.0, 2.0, 3.0])})),
    )
    assert model.parameters["theta"].shape == (3,)
    np.testing.assert_array_equal(model._input_coords["experiment"], [0, 1, 2])


@pytest.mark.parametrize("axis", ["time", "media_time", "chain", "draw", "sample", "pred_id"])
def test_labeled_inputs_reject_observation_and_sample_axes(auxiliary_data, axis):
    with pytest.raises(ValueError, match=r"fixed across scenarios|sample dimensions"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=xr.Dataset({"lift": (axis, [1.0])})))


@pytest.mark.parametrize("coordinate", [0.5, ("experiment", ["a", "b", "c"])])
def test_labeled_inputs_reject_auxiliary_coordinates(auxiliary_data, auxiliary_inputs, coordinate):
    inputs = auxiliary_inputs.assign_coords(note=coordinate)
    with pytest.raises(ValueError, match="Input coordinate 'note' must label only its own dimension"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))


def test_labeled_inputs_require_unique_axis_labels(auxiliary_data, auxiliary_inputs):
    inputs = auxiliary_inputs.assign_coords(experiment=["first", "first", "third"])
    with pytest.raises(ValueError, match="Input coordinate 'experiment' must have unique labels"):
        Model({}, lambda: jnp.array(0.0), data=Data(auxiliary_data, inputs=inputs))


def test_transformed_quantities_cannot_shadow_auxiliary_inputs(auxiliary_data, auxiliary_inputs):
    model = Model(
        {},
        lambda: jnp.array(0.0),
        data=Data(auxiliary_data, inputs=auxiliary_inputs),
        transformed_parameters=lambda: {"lift": 0.0},
    )
    with pytest.raises(ValueError, match="Transformed quantity 'lift' conflicts"):
        model.evaluate({})


@pytest.mark.parametrize("dated", [False, True], ids=["numeric", "dates"])
def test_model_time_inputs_keep_the_training_origin_for_scenarios(dated):
    def labels(offsets):
        return [date(2026, 1, 1) + timedelta(days=offset) if dated else 100 + offset for offset in offsets]

    frame = pl.DataFrame({"date": labels([0, 7, 14]), "sales": [2.0, 3.0, 4.0], "search": [4.0, 5.0, 6.0]})
    history = pl.DataFrame({"date": labels([-14, -7]), "search": [1.0, 2.0]})
    data = prepare_data(frame, time="date", outcome="sales", media=["search"], media_history=history)

    def quantities(time, media_time, slope):
        return {"curve": slope * time, "history_curve": slope * media_time}

    def density(outcome, curve, slope):
        return normal(slope, 0.0, 1.0) + normal(outcome, curve, 1.0)

    model = Model({"slope": Real()}, density, data=data, transformed_parameters=quantities)
    values = {"slope": jnp.asarray(2.0)}
    actual = jax.jit(model.evaluate)(values)
    np.testing.assert_array_equal(actual["curve"], [0.0, 14.0, 28.0])
    np.testing.assert_array_equal(actual["history_curve"], [-28.0, -14.0, 0.0, 14.0, 28.0])

    scenario = pl.DataFrame({"date": labels([7, 14]), "sales": [3.0, 4.0], "search": [5.0, 6.0]})
    scenario_inputs = model.prepare_data(scenario)
    changed = jax.jit(model.evaluate)(values, scenario_inputs)
    np.testing.assert_array_equal(changed["curve"], [14.0, 28.0])
    np.testing.assert_array_equal(changed["history_curve"], [14.0, 28.0])
    np.testing.assert_array_equal(model.evaluate(values)["curve"], actual["curve"])
    gradient = jax.jit(jax.grad(model.log_density))(values, scenario_inputs)
    assert jnp.isfinite(gradient["slope"])


def test_model_does_not_interpret_time_labels_unless_requested():
    data = prepare_data(
        pl.DataFrame({"period": ["early", "late"], "sales": [1.0, 2.0]}),
        time="period",
        outcome="sales",
        frequency=None,
    )
    model = Model({"level": Real()}, lambda outcome, level: normal(outcome, level, 1.0), data=data)
    assert jnp.isfinite(model.log_prob({"level": 0.0}))
    assert set(model.data.values) == {"outcome"}

    with pytest.raises(ValueError, match="Invalid observation date"):
        Model({"level": Real()}, lambda time, level: normal(time, level, 1.0), data=data)


def test_model_time_and_declared_parameter_names_must_be_distinct():
    data = prepare_data(pl.DataFrame({"period": [1, 2]}), time="period")
    with pytest.raises(ValueError, match="ambiguous"):
        Model({"time": Real()}, lambda time: normal(time, 0.0, 1.0), data=data)


def test_model_media_time_requires_exposure_periods():
    data = prepare_data(pl.DataFrame({"period": [1, 2]}), time="period")
    with pytest.raises(ValueError, match="unknown input 'media_time'"):
        Model({}, lambda media_time: jnp.sum(media_time), data=data)


def test_generation_key_name_does_not_request_time_coordinates():
    data = prepare_data(pl.DataFrame({"period": [1, 2]}), time="period")

    def generate(time):
        return {"draw": jax.random.normal(time)}

    model = Model({}, lambda: jnp.array(0.0), generate, data=data)
    key = jax.random.key(10)
    assert model._time_inputs == ()
    actual = jax.jit(model.generate_quantities)(key, {}, model.data)
    np.testing.assert_array_equal(actual["draw"], jax.random.normal(key))


def test_model_copies_and_canonicalizes_parameters() -> None:
    parameters = {"s": Positive(), "a": Real()}

    specification = Model(parameters, _scalar_log_density)
    parameters["extra"] = Real()
    returned_parameters = specification.parameters
    returned_parameters.pop("a")

    assert isinstance(specification, Model)
    assert list(specification.parameters) == ["a", "s"]


def test_model_density_and_gradient_do_not_retain_tracers():
    def density(data, scale):
        return lognormal(scale, 0.0, 1.0) + normal(data, 0.0, scale)

    model = Model({"scale": Positive()}, density)
    position = {"scale": jnp.array(0.3, dtype=jnp.float32)}
    observations = jnp.array([-0.5, 1.2], dtype=jnp.float32)

    with jax.check_tracer_leaks():
        compiled_density = jax.jit(model.log_density)(position, observations)
        value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, observations)

    assert jnp.allclose(compiled_density, value)
    assert jnp.allclose(value, model.log_density(position, observations))
    assert jnp.isfinite(gradient["scale"])


def test_model_is_immutable() -> None:
    specification = _make_scalar_model()

    with pytest.raises(FrozenInstanceError):
        specification._log_density = _scalar_log_density


@pytest.mark.parametrize(("name", "type_name"), [(1, "int"), (None, "NoneType")])
def test_model_requires_string_parameter_names(name, type_name: str) -> None:
    with pytest.raises(TypeError, match=rf"must be a string, got {type_name}"):
        Model({name: Real()}, _one_parameter_log_density)


@pytest.mark.parametrize("name", ["", "not-valid", "class"])
def test_model_requires_parameter_names_to_be_python_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="must be a valid non-keyword Python identifier"):
        Model({name: Real()}, _one_parameter_log_density)


def test_model_requires_parameter_mapping() -> None:
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        Model([("a", Real())], _one_parameter_log_density)


def test_model_requires_parameterizations() -> None:
    with pytest.raises(TypeError, match="parameter 'a' must implement Parameterization, got object"):
        Model({"a": object()}, _one_parameter_log_density)


def test_named_builtin_axes_do_not_change_the_custom_parameterization_contract() -> None:
    base = Real(shape=(2,))
    custom = SimpleNamespace(
        **{
            name: getattr(base, name)
            for name in (
                "shape",
                "position_shape",
                "dtype",
                "constrain",
                "unconstrain",
                "log_density_adjustment",
                "initialize",
            )
        }
    )

    def density(data, coefficient):
        return normal(coefficient, 0.0, 1.0)

    coordinates = {"feature": ["first", "second"]}
    explicit = Model({"coefficient": custom}, density, dims={"coefficient": ("feature",)}, coords=coordinates)
    named = Model({"coefficient": Real(dims="feature")}, density, coords=coordinates)
    position = {"coefficient": jnp.array([0.2, -0.3])}

    assert explicit.parameters["coefficient"] is custom
    assert not hasattr(custom, "dims")
    expected = jax.jit(jax.value_and_grad(explicit.log_density))(position, None)
    actual = jax.jit(jax.value_and_grad(named.log_density))(position, None)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.array_equal(left, right)


def test_model_requires_callable_log_density() -> None:
    with pytest.raises(TypeError, match="log_density must be callable"):
        Model({"a": Real()}, None)


def test_model_requires_inspectable_log_density() -> None:
    with pytest.raises(TypeError, match="log_density must expose an inspectable Python signature, got type"):
        Model({"a": Real()}, type)


def test_log_density_must_accept_data_first() -> None:
    def log_density(a):
        return normal(a, 0.0, 1.0)

    with pytest.raises(TypeError, match="places declared model parameter 'a' where data is required"):
        Model({"a": Real()}, log_density)


def test_log_density_must_accept_data_positionally() -> None:
    def log_density(*, data, a):
        return normal(a, 0.0, 1.0)

    with pytest.raises(TypeError, match="must accept data as positional argument 1"):
        Model({"a": Real()}, log_density)


def test_log_density_parameters_must_match_model_parameters() -> None:
    def wrong_name(data, b):
        return jnp.asarray(0.0)

    with pytest.raises(
        ValueError,
        match=r"missing \['a'\].*unexpected \['b'\]",
    ):
        Model({"a": Real()}, wrong_name)


def test_log_density_must_use_named_parameters() -> None:
    def positional_only(data, a, /):
        return jnp.asarray(a)

    with pytest.raises(TypeError, match=r"unsupported model parameter arguments \['a'\]"):
        Model({"a": Real()}, positional_only)


def test_log_density_accepts_parameter_mapping_callback() -> None:
    def log_density(data, **parameters):
        return normal(data["target"], parameters["a"], parameters["s"])

    specification = Model({"a": Real(), "s": Positive()}, log_density)
    position = {"a": jnp.array(0.5), "s": jnp.log(jnp.array(1.5))}
    data = {"target": jnp.array([0.25, 1.0])}

    value, gradient = jax.jit(jax.value_and_grad(specification.log_density))(position, data)

    assert value.shape == ()
    assert jax.tree.all(jax.tree.map(lambda leaf: jnp.all(jnp.isfinite(leaf)), gradient))


def test_log_density_rejects_mixed_named_and_mapping_parameters() -> None:
    def log_density(data, a, **parameters):
        return jnp.asarray(a)

    with pytest.raises(TypeError, match="cannot combine named model parameters"):
        Model({"a": Real()}, log_density)


def test_generate_must_accept_random_key() -> None:
    def generate():
        return {}

    with pytest.raises(TypeError, match="generated_quantities must accept key as positional argument 1"):
        Model({}, _empty_log_density, generated_quantities=generate)


def test_generate_must_accept_data() -> None:
    def generate(key):
        return {}

    with pytest.raises(TypeError, match="generated_quantities must accept data as positional argument 2"):
        Model({}, _empty_log_density, generated_quantities=generate)


@pytest.mark.parametrize(
    "generate",
    [
        lambda *, key, data, a: {"a": a},
        lambda key, a: {"a": a},
    ],
)
def test_generate_runtime_inputs_must_precede_model_parameters(generate) -> None:
    with pytest.raises(
        TypeError,
        match=r"(must accept key as positional argument 1|where data is required)",
    ):
        Model({"a": Real()}, _one_parameter_log_density, generate)


def test_generate_random_key_name_is_not_restricted() -> None:
    def generate(rng, prediction_data, a):
        return {"a": a}

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)

    result = specification.generate_quantities(jax.random.key(0), {"a": 1.0}, {})

    assert result["a"] == 1.0


def test_generate_mapping_callback_preserves_supplied_parameter_order() -> None:
    def density(data, **parameters):
        return jnp.asarray(0.0)

    def generate(key, data, **parameters):
        return {"values": jnp.stack(list(parameters.values()))}

    model = Model({"a": Real(), "b": Real()}, density, generate)
    quantities = model.generate_quantities(jax.random.key(0), {"b": 2.0, "a": 1.0}, {})

    assert jnp.array_equal(quantities["values"], jnp.array([2.0, 1.0]))


def test_generate_parameters_must_belong_to_model() -> None:
    def generate(key, data, b):
        return {"b": b}

    with pytest.raises(ValueError, match=r"unexpected \['b'\]"):
        Model({"a": Real()}, _one_parameter_log_density, generate)


def test_generate_rejects_mixed_named_and_mapping_parameters() -> None:
    def generate(key, data, a, **parameters):
        return {"a": a}

    with pytest.raises(TypeError, match="cannot combine named model parameters"):
        Model({"a": Real()}, _one_parameter_log_density, generate)


def test_constrain_and_unconstrain_complete_parameter_mapping() -> None:
    specification = _make_scalar_model()
    position = {"a": jnp.array(-0.5), "s": jnp.log(jnp.array(2.0))}

    parameters = specification.constrain(position)
    round_trip = specification.unconstrain(parameters)

    assert jnp.allclose(parameters["a"], -0.5)
    assert jnp.allclose(parameters["s"], 2.0)
    assert jax.tree.all(jax.tree.map(jnp.allclose, round_trip, position))


@pytest.mark.parametrize("method_name", ["constrain", "unconstrain", "evaluate", "log_prob"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"a": 0.0}, r"missing \['s'\]"),
        ({"a": 0.0, "s": 1.0, "extra": 0.0}, r"unexpected \['extra'\]"),
        (
            {"a": 0.0, "scale": 1.0},
            r"missing \['s'\].*unexpected \['scale'\]",
        ),
    ],
)
def test_parameter_mappings_must_have_exact_names(method_name: str, values, message: str) -> None:
    specification = _make_scalar_model()
    method = getattr(specification, method_name)

    with pytest.raises(ValueError, match=message):
        method(values)


def test_parameter_values_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="position must be a mapping"):
        _make_scalar_model().constrain((0.0, 1.0))


@pytest.mark.parametrize(("invalid_name", "type_name"), [(1, "int"), (None, "NoneType")])
def test_parameter_value_names_must_be_strings(invalid_name, type_name: str) -> None:
    with pytest.raises(TypeError, match=rf"non-string parameter name .* of type {type_name}"):
        _make_scalar_model().constrain({"a": 0.0, invalid_name: 1.0})


def test_parameter_shape_validation_is_preserved() -> None:
    specification = Model({"a": Real(shape=(2,))}, _one_parameter_log_density)

    with pytest.raises(ValueError, match=r"position must have shape \(2,\), got \(3,\)"):
        specification.constrain({"a": jnp.ones(3)})


def test_random_initialization_splits_key_in_canonical_name_order() -> None:
    key = jax.random.key(42)
    parameters = {"s": Positive(shape=(2,)), "a": Real(shape=(2,))}
    specification = Model(parameters, _vector_log_density)
    a_key, s_key = jax.random.split(key, 2)

    result = specification.initialize_random(key)

    assert jnp.array_equal(result["a"], parameters["a"].initialize(a_key))
    assert jnp.array_equal(result["s"], parameters["s"].initialize(s_key))
    assert not jnp.array_equal(result["a"], result["s"])


def test_random_initialization_is_independent_of_declaration_order() -> None:
    key = jax.random.key(42)
    first = Model({"a": Real(), "s": Positive()}, _scalar_log_density)
    reordered = Model({"s": Positive(), "a": Real()}, _scalar_log_density)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, first.initialize_random(key), reordered.initialize_random(key)))


def test_random_initialization_can_be_jitted() -> None:
    specification = _make_scalar_model()
    key = jax.random.key(42)

    result = jax.jit(specification.initialize_random)(key)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, result, specification.initialize_random(key)))


def test_model_outputs_follow_jax_default_dtype() -> None:
    specification = _make_regression_model()
    position = specification.initialize_random(jax.random.key(0))
    data = _regression_data()
    parameters = specification.constrain(position)
    expected_dtype = jnp.asarray(0.0).dtype

    density = specification.log_density(position, data)
    generated = specification.generate_quantities(jax.random.key(1), parameters, data)

    assert jax.tree.all(jax.tree.map(lambda value: value.dtype == expected_dtype, position))
    assert jax.tree.all(jax.tree.map(lambda value: value.dtype == expected_dtype, parameters))
    assert density.dtype == expected_dtype
    assert jax.tree.all(jax.tree.map(lambda value: value.dtype == expected_dtype, generated))


def test_model_preserves_explicit_float32() -> None:
    def generate(key, data, a):
        return {"a": a}

    specification = Model({"a": Real(dtype=jnp.float32)}, _one_parameter_log_density, generate)
    position = {"a": jnp.asarray(0.25, dtype=jnp.float32)}
    parameters = specification.constrain(position)

    density = specification.log_density(position, {})
    generated = specification.generate_quantities(jax.random.key(0), parameters, {})

    assert density.dtype == jnp.dtype(jnp.float32)
    assert generated["a"].dtype == jnp.dtype(jnp.float32)


def test_model_without_parameters_can_be_initialized() -> None:
    specification = Model({}, _empty_log_density)

    assert specification.initialize_random(jax.random.key(0)) == {}
    assert specification.log_density({}, {}) == 0.0


def test_log_density_constrains_parameters_and_adds_adjustments_once() -> None:
    specification = _make_scalar_model()
    position = {"a": jnp.array(0.5), "s": jnp.log(jnp.array(2.0))}
    expected = _scalar_log_density({}, a=position["a"], s=jnp.exp(position["s"])) + position["s"]

    result = specification.log_density(position, {})

    assert result.shape == ()
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("method_name", ["log_density", "log_prob"])
def test_log_density_must_return_scalar(method_name: str) -> None:
    def vector_density(data, a):
        return jnp.array([a, a])

    specification = Model({"a": Real()}, vector_density)

    with pytest.raises(ValueError, match=r"must return a scalar, got shape \(2,\)"):
        getattr(specification, method_name)({"a": 0.0}, {})


@pytest.mark.parametrize("method_name", ["log_density", "log_prob"])
def test_log_density_must_return_floating_point_value(method_name: str) -> None:
    def integer_density(data, a):
        return jnp.asarray(1, dtype=jnp.int32)

    specification = Model({"a": Real()}, integer_density)

    with pytest.raises(TypeError, match="must return a real floating-point value"):
        getattr(specification, method_name)({"a": 0.0}, {})


@pytest.mark.parametrize("method_name", ["log_density", "log_prob"])
def test_log_density_must_return_array_like_value(method_name: str) -> None:
    def object_density(data, a):
        return object()

    specification = Model({"a": Real()}, object_density)

    with pytest.raises(TypeError, match="must return an array-like floating-point scalar, got object"):
        getattr(specification, method_name)({"a": 0.0}, {})


def test_log_prob_evaluates_constrained_values_and_gradients_without_adjustments() -> None:
    specification = _make_scalar_model()
    parameters = {"a": jnp.array(0.5), "s": jnp.array(2.0)}

    value, gradient = jax.jit(jax.value_and_grad(specification.log_prob))(parameters)
    position = specification.unconstrain(parameters)

    assert jnp.allclose(value, _scalar_log_density(None, **parameters))
    assert jnp.allclose(specification.log_density(position, None), value + jnp.log(parameters["s"]))
    assert jnp.allclose(gradient["a"], -parameters["a"] / 4.0)
    assert jnp.allclose(gradient["s"], -0.5)


def test_log_prob_passes_dynamic_data_to_plain_callbacks_under_jit_vmap() -> None:
    specification = _make_regression_model()
    parameters = {
        "a": jnp.array([0.25, -0.5]),
        "b": jnp.array([[0.5, -0.25], [1.0, 0.5]]),
        "s": jnp.array([1.5, 0.75]),
    }
    data = _regression_data()
    compiled = jax.jit(jax.vmap(specification.log_prob, in_axes=(0, None)))

    for offset in (0.0, 2.0):
        current = {"media": data["media"], "target": data["target"] + offset}
        actual = compiled(parameters, current)
        expected = jnp.stack(
            [
                specification._log_density(current, **jax.tree.map(lambda value, index=index: value[index], parameters))
                for index in range(2)
            ]
        )
        assert jnp.allclose(actual, expected)


def test_direct_evaluation_uses_model_shapes_without_inference_transforms() -> None:
    def unavailable(*args):
        raise AssertionError("Inspection must not call inference transformations")

    declaration = SimpleNamespace(
        shape=(3,),
        position_shape=(2,),
        dtype=jnp.float32,
        constrain=unavailable,
        unconstrain=unavailable,
        log_density_adjustment=unavailable,
        initialize=unavailable,
    )

    def density(data, weights):
        return -jnp.square(weights).sum()

    specification = Model({"weights": declaration}, density)
    parameters = {"weights": [0, 0, 1]}

    assert specification.evaluate(parameters) == {}
    value = specification.log_prob(parameters)
    assert value == -1.0
    assert value.dtype == jnp.float32


def test_log_prob_preserves_simplex_boundary_values() -> None:
    def density(data, weights):
        return -jnp.square(weights).sum()

    specification = Model({"weights": Simplex(shape=(3,))}, density)

    assert jax.jit(specification.log_prob)({"weights": jnp.array([0.0, 0.0, 1.0])}) == -1.0


@pytest.mark.parametrize("method_name", ["evaluate", "log_prob"])
@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([0.0, 1.0], TypeError, "parameters must be a mapping"),
        ({1: 0.0}, TypeError, "non-string parameter name"),
        ({"a": [0.0], "s": 1.0}, ValueError, r"parameter 'a' must have shape \(\), got \(1,\)"),
        ({"a": object(), "s": 1.0}, TypeError, "parameter 'a' must be array-like"),
    ],
)
def test_direct_evaluation_validates_values_before_callbacks(method_name, values, error, message) -> None:
    with pytest.raises(error, match=message):
        getattr(_make_scalar_model(), method_name)(values)


def test_log_prob_converts_python_values_to_declaration_dtype() -> None:
    def density(data, coefficient):
        assert coefficient.dtype == jnp.float32
        return normal(coefficient, 0.0, 1.0)

    specification = Model({"coefficient": Real(shape=(2,), dtype=jnp.float32)}, density)

    assert specification.log_prob({"coefficient": [1, 2]}).dtype == jnp.float32


def test_direct_evaluation_supports_parameter_free_models() -> None:
    specification = Model({}, _empty_log_density)

    assert jax.jit(specification.evaluate)({}) == {}
    assert jax.jit(specification.log_prob)({}) == 0.0


def test_log_density_can_be_jitted() -> None:
    specification = _make_regression_model()
    position = {
        "a": jnp.array(0.25),
        "b": jnp.array([0.5, -0.25]),
        "s": jnp.log(jnp.array(1.5)),
    }
    data = _regression_data()

    eager = specification.log_density(position, data)
    compiled = jax.jit(specification.log_density)(position, data)

    assert jnp.allclose(compiled, eager)


def test_compiled_log_density_accepts_new_data() -> None:
    specification = _make_regression_model()
    position = {
        "a": jnp.array(0.25),
        "b": jnp.array([0.5, -0.25]),
        "s": jnp.log(jnp.array(1.5)),
    }
    first_data = _regression_data()
    second_data = {
        "media": first_data["media"],
        "target": first_data["target"] + 2.0,
    }
    compiled = jax.jit(specification.log_density)

    first = compiled(position, first_data)
    second = compiled(position, second_data)

    assert not jnp.allclose(first, second)


def test_log_density_gradient_matches_analytical_result() -> None:
    specification = _make_scalar_model()
    position = {"a": jnp.array(0.5), "s": jnp.log(jnp.array(2.0))}

    gradient = jax.grad(specification.log_density)(position, {})

    assert jnp.allclose(gradient["a"], -position["a"] / 4.0)
    assert jnp.allclose(gradient["s"], 1.0 - 0.5 * jnp.exp(position["s"]))


def test_log_density_can_be_vectorized_over_positions() -> None:
    specification = _make_scalar_model()
    positions = {
        "a": jnp.array([-1.0, 0.5, 2.0]),
        "s": jnp.log(jnp.array([0.5, 1.0, 2.0])),
    }

    result = jax.vmap(specification.log_density, in_axes=(0, None))(positions, {})
    expected = jnp.stack(
        [specification.log_density({"a": a, "s": s}, {}) for a, s in zip(positions["a"], positions["s"], strict=True)]
    )

    assert jnp.allclose(result, expected)


def test_tensor_parameter_blocks_support_noncentered_geo_hierarchies() -> None:
    n_geos = 8
    n_channels = 465
    specification = Model(
        {
            "channel_location": Real(shape=(n_channels,)),
            "channel_scale": Positive(shape=(n_channels,)),
            "geo_channel_raw": Real(shape=(n_geos, n_channels)),
            "sigma": Positive(),
        },
        _hierarchical_log_density,
    )
    data = {"target": jnp.ones((n_geos, n_channels))}
    position = specification.initialize_random(jax.random.key(5))

    value, gradient = jax.jit(jax.value_and_grad(specification.log_density))(position, data)

    assert value.shape == ()
    assert jax.tree.structure(gradient) == jax.tree.structure(position)
    assert jax.tree.all(jax.tree.map(lambda leaf: jnp.all(jnp.isfinite(leaf)), gradient))
    assert gradient["geo_channel_raw"].shape == (n_geos, n_channels)


def test_generate_receives_unchanged_key_data_and_constrained_parameters() -> None:
    specification = _make_regression_model()
    key = jax.random.key(7)
    parameters = {
        "a": jnp.array(0.25),
        "b": jnp.array([0.5, -0.25]),
        "s": jnp.array(1.5),
    }
    data = {"media": jnp.array([[1.5, -0.5]])}
    expected_location = parameters["a"] + data["media"] @ parameters["b"]

    result = specification.generate_quantities(key, parameters, data)
    expected = normal_rng(key, expected_location, parameters["s"])

    assert list(result) == ["y_new"]
    assert jnp.array_equal(result["y_new"], expected)


def test_generate_can_request_parameter_subset() -> None:
    def generate(key, data, a, s):
        return {"y_new": normal_rng(key, data["location"] + a, s)}

    specification = Model(
        {"a": Real(), "prior_only": Real(), "s": Positive()},
        _three_parameter_log_density,
        generate,
    )
    parameters = {"a": 0.25, "prior_only": -1.0, "s": 1.5}
    data = {"location": jnp.array([0.0, 1.0])}

    result = specification.generate_quantities(jax.random.key(7), parameters, data)

    assert result["y_new"].shape == (2,)


def test_generate_accepts_parameter_mapping_callback() -> None:
    def generate(key, data, **parameters):
        return {"total": data["offset"] + parameters["a"] + parameters["b"]}

    specification = Model(
        {"a": Real(), "b": Real()},
        _two_parameter_log_density,
        generate,
    )

    result = specification.generate_quantities(
        jax.random.key(0),
        {"a": 1.0, "b": 2.0},
        {"offset": jnp.array(3.0)},
    )

    assert result["total"] == 6.0


@pytest.mark.parametrize("callback_binding", ["positional", "keyword"])
def test_generate_quantities_can_be_jitted(callback_binding) -> None:
    specification = _make_regression_model(callback_binding=callback_binding)
    key = jax.random.key(7)
    parameters = {"a": jnp.array(0.25), "b": jnp.array([0.5, -0.25]), "s": jnp.array(1.5)}
    data = {"media": jnp.array([[1.5, -0.5]])}

    result = jax.jit(specification.generate_quantities)(key, parameters, data)

    assert jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            result,
            specification.generate_quantities(key, parameters, data),
        )
    )


def test_generate_can_be_vectorized_over_keys_and_parameters() -> None:
    specification = _make_regression_model()
    keys = jax.random.split(jax.random.key(7), 2)
    parameters = {
        "a": jnp.array([0.25, -0.5]),
        "b": jnp.array([[0.5, -0.25], [1.0, 0.5]]),
        "s": jnp.array([1.5, 0.75]),
    }
    data = {"media": jnp.array([[1.5, -0.5]])}

    result = jax.vmap(specification.generate_quantities, in_axes=(0, 0, None))(keys, parameters, data)

    assert result["y_new"].shape == (2, 1)


def test_generate_is_optional() -> None:
    with pytest.raises(RuntimeError, match="model has no generated_quantities callback"):
        _make_scalar_model().generate_quantities(jax.random.key(0), {"a": 0.0, "s": 1.0}, {})


def test_generate_parameter_values_must_match_model() -> None:
    specification = _make_regression_model()

    with pytest.raises(ValueError, match=r"missing \['b', 's'\]"):
        specification.generate_quantities(jax.random.key(0), {"a": 0.0}, {"media": jnp.ones((1, 2))})


def test_generate_must_return_mapping() -> None:
    def generate(key, data, a):
        return a

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)

    with pytest.raises(TypeError, match="generated_quantities must return a mapping"):
        specification.generate_quantities(jax.random.key(0), {"a": 0.0}, {})


def test_generated_quantity_values_must_be_array_like() -> None:
    def generate(key, data, a):
        return {"bad_value": object()}

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)

    with pytest.raises(TypeError, match="generated quantity 'bad_value' must be array-like, got object"):
        specification.generate_quantities(jax.random.key(0), {"a": 0.0}, {})


@pytest.mark.parametrize("quantity_name", [1, "not-valid"])
def test_generated_quantity_names_are_valid_identifiers(quantity_name) -> None:
    def generate(key, data, a):
        return {quantity_name: a}

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)
    error = TypeError if not isinstance(quantity_name, str) else ValueError

    with pytest.raises(error, match="generated quantity"):
        specification.generate_quantities(jax.random.key(0), {"a": 0.0}, {})


def _empty_log_density(data):
    return jnp.asarray(0.0)


def _one_parameter_log_density(data, a):
    return normal(a, 0.0, 1.0)


def _two_parameter_log_density(data, a, b):
    return normal(a, 0.0, 1.0) + normal(b, 0.0, 1.0)


def _three_parameter_log_density(data, a, prior_only, s):
    return normal(a, 0.0, 1.0) + normal(prior_only, 0.0, 1.0) + exponential(s, 1.0)


def _scalar_log_density(data, a, s):
    return normal(a, 0.0, 2.0) + exponential(s, 0.5)


def _vector_log_density(data, a, s):
    return normal(a, 0.0, 2.0) + exponential(s, 0.5)


def _hierarchical_log_density(data, channel_location, channel_scale, geo_channel_raw, sigma):
    geo_channel = channel_location + channel_scale * geo_channel_raw
    lp = normal(channel_location, 0.0, 1.0)
    lp += exponential(channel_scale, 1.0)
    lp += normal(geo_channel_raw, 0.0, 1.0)
    lp += exponential(sigma, 1.0)
    lp += normal(data["target"], geo_channel, sigma)
    return lp


def _make_scalar_model() -> Model:
    return Model({"a": Real(), "s": Positive()}, _scalar_log_density)


def _make_regression_model(*, callback_binding="positional") -> Model:
    def log_density(data, a, b, s):
        lp = normal(a, 0.0, 2.0)
        lp += normal(b, 0.0, 1.0)
        lp += exponential(s, 0.5)
        lp += normal(data["target"], a + data["media"] @ b, s)
        return lp

    def generate(key, data, a, b, s):
        return {"y_new": normal_rng(key, a + data["media"] @ b, s)}

    callbacks = (generate,) if callback_binding == "positional" else ()
    options = {"generated_quantities": generate} if callback_binding == "keyword" else {}
    return Model(
        {"a": Real(), "b": Real(shape=(2,)), "s": Positive()},
        log_density,
        *callbacks,
        **options,
    )


def _regression_data() -> dict[str, jax.Array]:
    return {
        "media": jnp.array([[1.0, -1.0], [0.5, 2.0], [-0.5, 1.0]]),
        "target": jnp.array([0.25, 1.5, -0.75]),
    }
