"""Tests for composing parameter declarations into JAX models."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest

from mmmjax import Model, Positive, Real, exponential, normal, normal_rng


def test_model_copies_and_canonicalizes_parameters() -> None:
    parameters = {"s": Positive(), "a": Real()}

    specification = Model(parameters, _scalar_log_density)
    parameters["extra"] = Real()
    returned_parameters = specification.parameters
    returned_parameters.pop("a")

    assert isinstance(specification, Model)
    assert list(specification.parameters) == ["a", "s"]


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
        match=r"missing \['a'\]; unexpected \['b'\]",
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

    with pytest.raises(TypeError, match="must accept key as positional argument 1"):
        Model({}, _empty_log_density, generate)


def test_generate_must_accept_data() -> None:
    def generate(key):
        return {}

    with pytest.raises(TypeError, match="must accept data as positional argument 2"):
        Model({}, _empty_log_density, generate)


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

    result = specification.generate(jax.random.key(0), {"a": 1.0}, {})

    assert result["a"] == 1.0


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


@pytest.mark.parametrize("method_name", ["constrain", "unconstrain"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"a": 0.0}, r"missing \['s'\]"),
        ({"a": 0.0, "s": 1.0, "extra": 0.0}, r"unexpected \['extra'\]"),
        (
            {"a": 0.0, "scale": 1.0},
            r"missing \['s'\]; unexpected \['scale'\]",
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


def test_log_density_must_return_scalar() -> None:
    def vector_density(data, a):
        return jnp.array([a, a])

    specification = Model({"a": Real()}, vector_density)

    with pytest.raises(ValueError, match=r"must return a scalar, got shape \(2,\)"):
        specification.log_density({"a": 0.0}, {})


def test_log_density_must_return_floating_point_value() -> None:
    def integer_density(data, a):
        return jnp.asarray(1, dtype=jnp.int32)

    specification = Model({"a": Real()}, integer_density)

    with pytest.raises(TypeError, match="must return a real floating-point value"):
        specification.log_density({"a": 0.0}, {})


def test_log_density_must_return_array_like_value() -> None:
    def object_density(data, a):
        return object()

    specification = Model({"a": Real()}, object_density)

    with pytest.raises(TypeError, match="must return an array-like floating-point scalar, got object"):
        specification.log_density({"a": 0.0}, {})


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

    result = specification.generate(key, parameters, data)
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

    result = specification.generate(jax.random.key(7), parameters, data)

    assert result["y_new"].shape == (2,)


def test_generate_accepts_parameter_mapping_callback() -> None:
    def generate(key, data, **parameters):
        return {"total": data["offset"] + parameters["a"] + parameters["b"]}

    specification = Model(
        {"a": Real(), "b": Real()},
        _two_parameter_log_density,
        generate,
    )

    result = specification.generate(
        jax.random.key(0),
        {"a": 1.0, "b": 2.0},
        {"offset": jnp.array(3.0)},
    )

    assert result["total"] == 6.0


def test_generate_can_be_jitted() -> None:
    specification = _make_regression_model()
    key = jax.random.key(7)
    parameters = {"a": jnp.array(0.25), "b": jnp.array([0.5, -0.25]), "s": jnp.array(1.5)}
    data = {"media": jnp.array([[1.5, -0.5]])}

    result = jax.jit(specification.generate)(key, parameters, data)

    assert jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            result,
            specification.generate(key, parameters, data),
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

    result = jax.vmap(specification.generate, in_axes=(0, 0, None))(keys, parameters, data)

    assert result["y_new"].shape == (2, 1)


def test_generate_is_optional() -> None:
    with pytest.raises(RuntimeError, match="model has no generate callback"):
        _make_scalar_model().generate(jax.random.key(0), {"a": 0.0, "s": 1.0}, {})


def test_generate_parameter_values_must_match_model() -> None:
    specification = _make_regression_model()

    with pytest.raises(ValueError, match=r"missing \['b', 's'\]"):
        specification.generate(jax.random.key(0), {"a": 0.0}, {"media": jnp.ones((1, 2))})


def test_generate_must_return_mapping() -> None:
    def generate(key, data, a):
        return a

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)

    with pytest.raises(TypeError, match="generate must return a mapping"):
        specification.generate(jax.random.key(0), {"a": 0.0}, {})


def test_generated_quantity_values_must_be_array_like() -> None:
    def generate(key, data, a):
        return {"bad_value": object()}

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)

    with pytest.raises(TypeError, match="generated quantity 'bad_value' must be array-like, got object"):
        specification.generate(jax.random.key(0), {"a": 0.0}, {})


@pytest.mark.parametrize("quantity_name", [1, "not-valid"])
def test_generated_quantity_names_are_valid_identifiers(quantity_name) -> None:
    def generate(key, data, a):
        return {quantity_name: a}

    specification = Model({"a": Real()}, _one_parameter_log_density, generate)
    error = TypeError if not isinstance(quantity_name, str) else ValueError

    with pytest.raises(error, match="generated quantity"):
        specification.generate(jax.random.key(0), {"a": 0.0}, {})


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


def _make_regression_model() -> Model:
    def log_density(data, a, b, s):
        lp = normal(a, 0.0, 2.0)
        lp += normal(b, 0.0, 1.0)
        lp += exponential(s, 0.5)
        lp += normal(data["target"], a + data["media"] @ b, s)
        return lp

    def generate(key, data, a, b, s):
        return {"y_new": normal_rng(key, a + data["media"] @ b, s)}

    return Model(
        {"a": Real(), "b": Real(shape=(2,)), "s": Positive()},
        log_density,
        generate,
    )


def _regression_data() -> dict[str, jax.Array]:
    return {
        "media": jnp.array([[1.0, -1.0], [0.5, 2.0], [-0.5, 1.0]]),
        "target": jnp.array([0.25, 1.5, -0.75]),
    }
