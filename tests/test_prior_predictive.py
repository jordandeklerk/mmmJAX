"""Tests for explicit prior draws and labeled prior predictive quantities."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mmmjax
import mmmjax.sampling as sampling
from mmmjax import (
    FourierSeasonality,
    Interval,
    MediaEffect,
    Model,
    Positive,
    Real,
    Simplex,
    fit_data_scaling,
    prepare_data,
    sample_prior,
)


@pytest.fixture
def scalar_model():
    def forbidden_density(data, location):
        raise AssertionError("Prior sampling must not evaluate the model density")

    return Model({"location": Real()}, forbidden_density)


def _normal_prior(key):
    return {"location": jax.random.normal(key)}


def _prepared_model(*, scaling=False):
    data = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [100.0, 200.0, 300.0], "price": [5.0, 10.0, 20.0]}),
        time="week",
        outcome="sales",
        controls=["price"],
    )

    def forbidden_density(outcome, location):
        raise AssertionError("Prior sampling must not evaluate the model density")

    def generate(key, outcome, controls, location):
        mean = jnp.full_like(outcome, location)
        return {
            "prediction": mean + jax.random.normal(key, outcome.shape),
            "pointwise": jnp.zeros_like(outcome),
            "mean": mean,
            "inputs": controls,
        }

    model = Model(
        {"location": Real()},
        forbidden_density,
        generate,
        data=data,
        components=[],
        scaling=fit_data_scaling(data, scale_outcome=True) if scaling else None,
        predictive=("prediction",),
        log_likelihood=("pointwise",),
        generated_dims={"mean": ("time",)},
    )
    return model, data


def test_public_default_draws_do_not_use_density_nuts_or_initialization(scalar_model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Prior sampling must use the supplied prior directly")

    monkeypatch.setattr(sampling, "_sample_nuts", forbidden)
    monkeypatch.setattr(Model, "log_density", forbidden)
    monkeypatch.setattr(Model, "initialize_random", forbidden)
    assert "sample_prior" in mmmjax.__all__
    assert sampling.sample_prior is sample_prior
    result = sample_prior(scalar_model, _normal_prior)
    assert isinstance(result, xr.DataTree)
    assert set(result.children) == {"prior"}
    assert result["prior"]["location"].dims == ("chain", "draw")
    assert result["prior"]["location"].shape == (1, 500)
    assert isinstance(result["prior"]["location"].data, np.ndarray)
    assert np.isfinite(result["prior"]["location"]).all()


def test_seeded_draws_and_generation_are_reproducible_and_independent():
    model = Model(
        {"location": Real()},
        lambda data, location: jnp.nan,
        lambda key, data, location: {"prediction": location + jax.random.normal(key)},
        predictive=("prediction",),
    )
    first = sample_prior(model, _normal_prior, draws=32, seed=19)
    second = sample_prior(model, _normal_prior, draws=32, seed=19)
    changed = sample_prior(model, _normal_prior, draws=32, seed=20)
    disabled = sample_prior(model, _normal_prior, draws=32, seed=19, generate=False)
    xr.testing.assert_equal(first, second)
    xr.testing.assert_equal(first["prior"], disabled["prior"])
    values = first["prior"]["location"].values
    assert np.unique(values).size == 32
    assert not np.array_equal(values, changed["prior"]["location"])
    noise = first["prior_predictive"]["prediction"].values - values
    assert np.unique(noise).size == 32
    assert not np.allclose(noise, values)
    assert not np.array_equal(first["prior_predictive"]["prediction"], changed["prior_predictive"]["prediction"])


def test_attached_prior_matches_explicit_callback_and_supports_generation_toggle():
    model = Model(
        {"location": Real()},
        lambda data, location: jnp.nan,
        lambda key, data, location: {"prediction": location + jax.random.normal(key)},
        prior=_normal_prior,
        predictive=("prediction",),
    )
    attached = sample_prior(model, draws=4, seed=12)
    explicit = sample_prior(model, _normal_prior, draws=4, seed=12)
    explicit_none = sample_prior(model, None, draws=4, seed=12)
    disabled = sample_prior(model, draws=4, seed=12, generate=False)
    xr.testing.assert_equal(attached, explicit)
    xr.testing.assert_equal(attached, explicit_none)
    xr.testing.assert_equal(attached["prior"], disabled["prior"])
    assert set(attached.children) == {"prior", "prior_predictive"}
    assert set(disabled.children) == {"prior"}


def test_per_call_prior_override_does_not_replace_the_attached_callback():
    model = Model({"location": Real()}, lambda data, location: jnp.nan, prior=_normal_prior)
    original = sample_prior(model, draws=3, seed=5)
    override = sample_prior(model, lambda key: {"location": 42.0}, draws=3, seed=5)
    again = sample_prior(model, draws=3, seed=5)
    np.testing.assert_array_equal(override["prior"]["location"], np.full((1, 3), 42.0))
    xr.testing.assert_equal(original, again)
    assert not np.array_equal(original["prior"]["location"], override["prior"]["location"])


def test_attached_prior_preserves_parameter_and_generated_metadata():
    model = Model(
        {"coefficient": Real((2,))},
        lambda data, coefficient: jnp.nan,
        lambda key, data, coefficient: {"copy": coefficient},
        prior=lambda key: {"coefficient": jax.random.normal(key, (2,))},
        dims={"coefficient": ("feature",)},
        coords={"feature": ["price", "promotion"]},
    )
    result = sample_prior(model, draws=3)
    assert result["prior"]["coefficient"].dims == ("chain", "draw", "feature")
    assert result["prior_generated_quantities"]["copy"].dims == ("chain", "draw", "feature")
    np.testing.assert_array_equal(result["prior"]["feature"], ["price", "promotion"])
    np.testing.assert_array_equal(result["prior_generated_quantities"]["copy"], result["prior"]["coefficient"])


def test_attached_prior_is_not_executed_by_construction_density_or_posterior_sampling(monkeypatch):
    def forbidden_prior(key):
        raise AssertionError("Attaching a prior must not add or evaluate a model density")

    def stationary_draws(logdensity, positions, keys, *, draws, **options):
        repeated = jax.tree.map(lambda value: jnp.repeat(value[:, None], draws, axis=1), positions)
        shape = (len(keys), draws)
        return repeated, {
            "lp": jax.vmap(jax.vmap(logdensity))(repeated),
            "diverging": jnp.zeros(shape, dtype=bool),
            "reached_max_treedepth": jnp.zeros(shape, dtype=bool),
        }

    monkeypatch.setattr(sampling, "_sample_nuts", stationary_draws)
    model = Model({"location": Real()}, lambda data, location: -0.5 * location**2, prior=forbidden_prior)
    position = {"location": jnp.array(1.5)}
    np.testing.assert_allclose(model.log_density(position, None), -1.125)
    result = sampling.sample(model, draws=2, warmup=1, chains=1, initial_values=position)
    np.testing.assert_array_equal(result["sample_stats"]["lp"], np.full((1, 2), -1.125))
    assert set(result.children) == {"posterior", "sample_stats"}


def test_attached_prior_shapes_are_validated_even_when_generation_is_disabled():
    def forbidden_generate(key, data, location):
        raise AssertionError("Generation was disabled")

    model = Model(
        {"location": Real()},
        lambda data, location: jnp.nan,
        forbidden_generate,
        prior=lambda key: {"location": jnp.ones(2)},
    )
    with pytest.raises(ValueError, match="shape"):
        sample_prior(model, draws=2, generate=False)


def test_constrained_draws_preserve_full_parameter_shapes_and_labels():
    model = Model(
        {"scale": Positive(), "fraction": Interval(0.0, 1.0), "weights": Simplex((3,))},
        lambda data, scale, fraction, weights: jnp.nan,
        dims={"weights": ("category",)},
        coords={"category": ["video", "search", "radio"]},
    )

    def prior(key):
        scale_key, fraction_key, weights_key = jax.random.split(key, 3)
        return {
            "scale": jnp.exp(jax.random.normal(scale_key)),
            "fraction": jax.random.uniform(fraction_key, minval=0.1, maxval=0.9),
            "weights": jax.random.dirichlet(weights_key, jnp.array([2.0, 3.0, 4.0])),
        }

    result = sample_prior(model, prior, draws=12)["prior"]
    assert result["weights"].dims == ("chain", "draw", "category")
    assert result["weights"].shape == (1, 12, 3)
    np.testing.assert_array_equal(result["category"], ["video", "search", "radio"])
    np.testing.assert_allclose(result["weights"].sum("category"), 1.0, rtol=2e-6)
    assert (result["weights"].values > 0).all()
    assert (result["scale"].values > 0).all()
    assert ((result["fraction"].values > 0) & (result["fraction"].values < 1)).all()


def test_unprepared_generation_receives_supplied_data():
    model = Model(
        {"location": Real()},
        lambda data, location: jnp.nan,
        lambda key, data, location: {"prediction": data["offset"] + location},
        predictive=("prediction",),
        generated_dims={"prediction": ("scenario",)},
        coords={"scenario": ["base", "expanded"]},
    )
    result = sample_prior(model, lambda key: {"location": 2.0}, data={"offset": jnp.array([1.0, 3.0])}, draws=2)
    assert set(result.children) == {"prior", "prior_predictive"}
    assert result["prior_predictive"]["prediction"].dims == ("chain", "draw", "scenario")
    np.testing.assert_array_equal(result["prior_predictive"]["prediction"], [[[3.0, 5.0], [3.0, 5.0]]])


def test_prepared_groups_infer_data_axes_and_exclude_selected_likelihoods():
    model, data = _prepared_model()
    result = sample_prior(model, lambda key: {"location": 1.25}, draws=3)
    assert set(result.children) == {
        "prior",
        "prior_predictive",
        "prior_generated_quantities",
        "observed_data",
        "constant_data",
    }
    generated = result["prior_generated_quantities"]
    assert set(generated.data_vars) == {"mean", "inputs"}
    assert generated["mean"].dims == ("chain", "draw", "time")
    assert generated["inputs"].dims == ("chain", "draw", "time", "control")
    assert result["prior_predictive"]["prediction"].dims == ("chain", "draw", "time")
    np.testing.assert_array_equal(result["prior_predictive"]["time"], [10, 11, 12])
    np.testing.assert_array_equal(generated["control"], ["price"])
    np.testing.assert_array_equal(generated["mean"], np.full((1, 3, 3), 1.25))
    np.testing.assert_array_equal(result["observed_data"]["outcome"], data.arrays["outcome"])
    np.testing.assert_array_equal(result["constant_data"]["controls"], data.arrays["controls"])
    for group in result.children.values():
        assert "pointwise" not in group.data_vars
        for variable in group.data_vars.values():
            assert isinstance(variable.data, np.ndarray)


def test_generation_toggle_does_not_execute_callback_and_keeps_prepared_data():
    prepared, _ = _prepared_model()

    def forbidden_generate(key, location):
        raise AssertionError("Generation was disabled")

    model = Model(
        {"location": Real()},
        lambda location: jnp.nan,
        forbidden_generate,
        data=prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales"),
        components=[],
    )
    result = sample_prior(model, _normal_prior, draws=2, generate=False)
    assert set(result.children) == {"prior", "observed_data"}
    result_with_controls = sample_prior(prepared, _normal_prior, draws=2, generate=False)
    assert set(result_with_controls.children) == {"prior", "observed_data", "constant_data"}


def test_scaled_prepared_inputs_are_snapshotted_without_rescaling_prior_values():
    model, original = _prepared_model(scaling=True)
    expected_outcome = np.array(model.data.values["outcome"])
    expected_controls = np.array(model.data.values["controls"])
    assert not np.allclose(expected_outcome, original.arrays["outcome"])
    assert not np.allclose(expected_controls, original.arrays["controls"])
    original.arrays["outcome"][...] = -999.0
    original.arrays["controls"][...] = -999.0
    result = sample_prior(model, lambda key: {"location": 1.25}, draws=2)
    np.testing.assert_array_equal(result["observed_data"]["outcome"], expected_outcome)
    np.testing.assert_array_equal(result["constant_data"]["controls"], expected_controls)
    np.testing.assert_array_equal(result["prior"]["location"], np.full((1, 2), 1.25))
    np.testing.assert_array_equal(result["prior_generated_quantities"]["mean"], np.full((1, 2, 3), 1.25))
    assert result.attrs["data_scale"] == "model"
    result["observed_data"]["outcome"].values[...] = -50.0
    np.testing.assert_array_equal(model.data.values["outcome"], expected_outcome)


def test_components_and_transformed_parameters_run_for_each_prior_draw():
    data = prepare_data(
        pd.DataFrame(
            {"week": [0, 1, 2], "sales": [1.0, 2.0, 3.0], "video": [1.0, 3.0, 2.0], "search": [2.0, 1.0, 4.0]}
        ),
        time="week",
        outcome="sales",
        media=["video", "search"],
    )

    def transformed(intercept, paid_total, annual):
        return {"mean": intercept + paid_total + annual}

    def generate(key, mean, paid, annual, annual_coefficients):
        return {"prediction": mean, "paid": paid, "seasonal": annual, "weights": annual_coefficients}

    model = Model(
        {"intercept": Real()},
        lambda outcome, mean: jnp.nan,
        generate,
        data=data,
        components=[MediaEffect(max_lag=1, name="paid"), FourierSeasonality(period=8, order=1, name="annual")],
        transformed_parameters=transformed,
        predictive=("prediction",),
    )

    def prior(key):
        intercept_key, annual_key = jax.random.split(key)
        return {
            "intercept": jax.random.normal(intercept_key),
            "paid_coefficient": jnp.array([0.7, 1.1]),
            "paid_retention": jnp.array([0.2, 0.5]),
            "paid_half_saturation": jnp.array([1.5, 2.0]),
            "paid_slope": jnp.array([1.2, 0.8]),
            "annual": jax.random.normal(annual_key, (2,)),
        }

    result = sample_prior(model, prior, draws=3)
    parameters = result["prior"]
    assert set(parameters.data_vars) == set(model.parameters)
    assert parameters["paid_retention"].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(parameters["channel"], ["video", "search"])
    generated = result["prior_generated_quantities"]
    assert generated["paid"].dims == ("chain", "draw", "time", "channel")
    assert generated["seasonal"].dims == ("chain", "draw", "time")
    assert generated["weights"].dims == parameters["annual"].dims
    for draw in range(3):
        constrained = {name: jnp.asarray(parameters[name].values[0, draw]) for name in model.parameters}
        expected = model.generate(jax.random.key(0), constrained, model.data)
        np.testing.assert_allclose(
            result["prior_predictive"]["prediction"].values[0, draw], expected["prediction"], rtol=3e-6, atol=2e-6
        )
    prediction = result["prior_predictive"]["prediction"].values
    assert not np.array_equal(prediction[0, 0], prediction[0, 1])


def test_generated_aliases_inherit_parameter_axes_without_guessing_equal_shapes():
    model = Model(
        {"coefficient": Real((2,)), "other": Real((2,))},
        lambda data, coefficient, other: jnp.nan,
        lambda key, data, coefficient, other: {"copy": coefficient, "other_copy": other, "sum": coefficient + other},
        dims={"coefficient": ("feature",)},
        coords={"feature": ["first", "second"]},
    )
    result = sample_prior(model, lambda key: {"coefficient": jnp.ones(2), "other": jnp.zeros(2)}, draws=2)
    generated = result["prior_generated_quantities"]
    assert generated["copy"].dims == ("chain", "draw", "feature")
    assert generated["other_copy"].dims == ("chain", "draw", "other_dim_0")
    assert generated["sum"].dims == ("chain", "draw", "sum_dim_0")
    np.testing.assert_array_equal(generated["feature"], ["first", "second"])


@pytest.mark.parametrize("draws", [0, True, 1.5])
def test_invalid_draw_counts_are_rejected(scalar_model, draws):
    with pytest.raises(ValueError, match=r"draws.*positive integer"):
        sample_prior(scalar_model, _normal_prior, draws=draws)


@pytest.mark.parametrize("seed", [-1, True])
def test_invalid_seeds_are_rejected(scalar_model, seed):
    with pytest.raises(ValueError, match=r"seed.*nonnegative integer"):
        sample_prior(scalar_model, _normal_prior, seed=seed)


def test_invalid_model_is_rejected():
    with pytest.raises(TypeError, match=r"model.*Model"):
        sample_prior(object(), _normal_prior, draws=2)


@pytest.mark.parametrize("options", [{}, {"prior": None}])
def test_missing_prior_requires_an_attached_or_per_call_callback(scalar_model, options):
    with pytest.raises(ValueError, match="prior"):
        sample_prior(scalar_model, draws=2, **options)


@pytest.mark.parametrize("prior", [False, {"location": 1.0}])
def test_prior_argument_must_be_callable(scalar_model, prior):
    with pytest.raises(TypeError, match=r"prior.*(callable|function)"):
        sample_prior(scalar_model, prior, draws=2)


def test_model_rejects_noncallable_prior():
    with pytest.raises(TypeError, match=r"prior.*(callable|function)"):
        Model({"location": Real()}, lambda data, location: jnp.nan, prior={"location": 1.0})


def test_generate_flag_must_be_boolean(scalar_model):
    with pytest.raises(TypeError, match=r"generate.*bool"):
        sample_prior(scalar_model, _normal_prior, generate=1)


@pytest.mark.parametrize("value", [None, 1.0, (1.0,)])
def test_prior_output_must_be_a_mapping(scalar_model, value):
    with pytest.raises(TypeError, match="mapping"):
        sample_prior(scalar_model, lambda key: value, draws=2)


@pytest.mark.parametrize("value", [{}, {"location": 1.0, "extra": 2.0}])
def test_prior_output_requires_exact_parameter_names(scalar_model, value):
    with pytest.raises(ValueError, match=r"(parameter|missing|extra|name)"):
        sample_prior(scalar_model, lambda key: value, draws=2)


@pytest.mark.parametrize("value", [jnp.ones(1), jnp.ones((2, 2))])
def test_prior_output_requires_declared_shapes(scalar_model, value):
    with pytest.raises(ValueError, match="shape"):
        sample_prior(scalar_model, lambda key: {"location": value}, draws=2)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, 1.0 + 2.0j])
def test_prior_values_must_be_finite_and_real(scalar_model, value):
    with pytest.raises((TypeError, ValueError), match=r"(finite|real|complex|floating)"):
        sample_prior(scalar_model, lambda key: {"location": value}, draws=2)


@pytest.mark.parametrize(
    "parameter,value",
    [
        (Positive(), 0.0),
        (Positive(), -1.0),
        (Interval(0.0, 1.0), 1.0),
        (Simplex((3,)), jnp.array([0.2, 0.3, 0.8])),
        (Simplex((3,)), jnp.array([0.0, 0.5, 0.5])),
    ],
)
def test_prior_values_must_satisfy_declared_support(parameter, value):
    model = Model({"location": parameter}, lambda data, location: jnp.nan)
    with pytest.raises(ValueError, match=r"(constraint|finite|support)"):
        sample_prior(model, lambda key: {"location": value}, draws=2)


@pytest.mark.parametrize("invalid", [-1.0, jnp.nan])
def test_support_validation_covers_later_draws(invalid):
    model = Model({"location": Positive()}, lambda data, location: jnp.nan)

    def valid_prior(key):
        return {"location": 0.5 + jax.random.uniform(key)}

    valid = sample_prior(model, valid_prior, draws=8, seed=11)
    last_value = valid["prior"]["location"].values[0, -1]
    assert last_value != valid["prior"]["location"].values[0, 0]

    def invalid_later_prior(key):
        value = valid_prior(key)["location"]
        return {"location": jnp.where(value == last_value, invalid, value)}

    with pytest.raises(ValueError, match=r"(constraint|finite|support)"):
        sample_prior(model, invalid_later_prior, draws=8, seed=11)


@pytest.mark.parametrize("axis", ["chain", "draw"])
def test_model_coordinates_cannot_override_sample_axes(axis):
    model = Model({"location": Real()}, lambda data, location: jnp.nan, coords={axis: [10]})
    with pytest.raises(ValueError, match=r"(chain|draw)"):
        sample_prior(model, _normal_prior, draws=2)


def test_prepared_models_reject_replacement_data():
    model, data = _prepared_model()
    with pytest.raises(ValueError, match=r"(stored|prepared|Prepared)"):
        sample_prior(model, _normal_prior, data=data, draws=2)


def test_generated_dimensions_must_match_generated_arrays():
    model = Model(
        {"location": Real()},
        lambda data, location: jnp.nan,
        lambda key, data, location: {"prediction": jnp.ones(3) * location},
        predictive=("prediction",),
        generated_dims={"prediction": ("scenario",)},
        coords={"scenario": ["first", "second"]},
    )
    with pytest.raises(ValueError, match=r"(scenario|coordinate|Coordinate)"):
        sample_prior(model, _normal_prior, draws=2)
