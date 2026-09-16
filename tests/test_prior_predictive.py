"""Tests for explicit prior draws and labeled prior predictive quantities."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mmmjax
import mmmjax.sampling as sampling
from mmmjax import (
    CorrelationCholesky,
    Data,
    Interval,
    Model,
    Positive,
    Prior,
    Real,
    Simplex,
    dirichlet,
    fit_data_scaling,
    fourier_features,
    geometric_adstock,
    hill_saturation,
    lkj_cholesky,
    lkj_cholesky_rng,
    lognormal,
    multivariate_normal,
    multivariate_normal_rng,
    normal,
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


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_registered_priors_draw_declared_shapes_with_independent_keys_and_copied_mapping(batch_size):
    def forbidden_density(data, first, second, scale, weights, joint, factor):
        raise AssertionError("Registered prior draws must not evaluate the model density")

    shared = Prior(normal, location=jnp.array([-0.5, 0.0, 0.5]), scale=0.5)
    priors = {
        "first": shared,
        "second": shared,
        "scale": Prior(lognormal, location=0.2, scale=0.9),
        "weights": Prior(dirichlet, concentration=jnp.array([2.0, 3.0, 4.0])),
        "joint": Prior(multivariate_normal, location=jnp.zeros(3), scale_tril=jnp.eye(3)),
        "factor": Prior(lkj_cholesky, concentration=2.0),
    }
    model = Model(
        {
            "first": Real(dims=("group", "channel")),
            "second": Real(dims=("group", "channel")),
            "scale": Positive(),
            "weights": Simplex(dims=("group", "channel")),
            "joint": Real(dims=("group", "channel")),
            "factor": CorrelationCholesky(dims=("group", "channel", "channel_to")),
        },
        forbidden_density,
        prior=priors,
        coords={
            "group": ["west", "east"],
            "channel": ["search", "video", "radio"],
            "channel_to": ["search", "video", "radio"],
        },
    )
    priors.clear()
    first = sample_prior(model, draws=5, seed=19, batch_size=batch_size)
    repeated = sample_prior(model, draws=5, seed=19, batch_size=64)
    xr.testing.assert_allclose(first, repeated)
    assert set(first.children) == {"prior"}
    assert set(first["prior"].data_vars) == {"first", "second", "scale", "weights", "joint", "factor"}
    for name in ("first", "second", "weights", "joint"):
        assert first["prior"][name].dims == ("chain", "draw", "group", "channel")
        assert first["prior"][name].shape == (1, 5, 2, 3)
    assert first["prior"]["scale"].dims == ("chain", "draw")
    factor = first["prior"]["factor"]
    assert factor.dims == ("chain", "draw", "group", "channel", "channel_to")
    assert factor.shape == (1, 5, 2, 3, 3)
    np.testing.assert_array_equal(factor, np.tril(factor))
    np.testing.assert_allclose(np.square(factor).sum("channel_to"), 1.0, rtol=2e-6)
    np.testing.assert_array_equal(first["prior"]["channel"], ["search", "video", "radio"])
    np.testing.assert_array_equal(first["prior"]["group"], ["west", "east"])
    assert not np.array_equal(first["prior"]["first"], first["prior"]["second"])
    assert np.unique(first["prior"]["first"]).size == 30
    assert np.all(first["prior"]["scale"] > 0)
    np.testing.assert_allclose(first["prior"]["weights"].sum("channel"), 1.0, rtol=2e-6)


def test_registered_prior_draws_still_require_parameter_support():
    model = Model(
        {"scale": Positive()},
        lambda data, scale: normal(scale, 0.0, 1.0),
        prior={"scale": Prior(normal, location=-100.0, scale=0.01)},
    )
    with pytest.raises(ValueError, match=r"scale|support"):
        sample_prior(model, draws=3)


def test_registered_prior_only_draws_skip_unsaved_transforms():
    def forbidden_transform(location):
        raise AssertionError("Prior draws must not evaluate transforms when no outputs are requested")

    _, data = _prepared_model()
    model = Model(
        {"location": Real()},
        lambda mean: mean.sum(),
        data=data,
        transformed_parameters=forbidden_transform,
        prior={"location": Prior(normal, location=0.0, scale=1.0)},
    )
    result = sample_prior(model, draws=3)
    assert set(result.children) == {"prior", "observed_data", "constant_data"}
    assert result["prior"]["location"].shape == (1, 3)


def test_correlated_joint_prior_draws_keep_parameter_and_predictive_axes():
    def density(data, factor, coefficients):
        return lkj_cholesky(factor, 2.0) + multivariate_normal(coefficients, jnp.zeros(2), factor)

    def prior(key):
        factor_key, coefficient_key = jax.random.split(key)
        factor = lkj_cholesky_rng(factor_key, 2, 2.0)
        coefficients = multivariate_normal_rng(coefficient_key, jnp.zeros(2), factor)
        return {"factor": factor, "coefficients": coefficients}

    model = Model(
        {"factor": CorrelationCholesky(dims=("effect", "effect_to")), "coefficients": Real(dims="effect")},
        density,
        lambda key, data, coefficients: {"prediction": coefficients},
        prior=prior,
        coords={"effect": ["search", "video"], "effect_to": ["search", "video"]},
        generated_dims={"prediction": ("effect",)},
        predictive=("prediction",),
    )
    results = sample_prior(model, draws=12, seed=4)
    assert results["prior"]["factor"].dims == ("chain", "draw", "effect", "effect_to")
    assert results["prior"]["coefficients"].dims == ("chain", "draw", "effect")
    np.testing.assert_array_equal(results["prior"]["coefficients"], results["prior_predictive"]["prediction"])
    assert np.all(np.isfinite(results["prior"]["factor"]))


@pytest.mark.parametrize(
    "factor",
    [
        [[1.0, 0.1], [0.0, 1.0]],
        [[1.0, 0.0], [0.5, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
)
def test_prior_correlation_factors_must_already_satisfy_support(factor):
    model = Model({"factor": CorrelationCholesky((2, 2))}, lambda data, factor: lkj_cholesky(factor, 1.0))
    with pytest.raises(ValueError, match="factor"):
        sample_prior(model, lambda key: {"factor": jnp.asarray(factor)}, draws=2)


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
            "lp_location": -0.5 * location**2,
            "mean": mean,
            "inputs": controls,
        }

    model = Model(
        {"location": Real()},
        forbidden_density,
        generate,
        data=Data(data, scaling=fit_data_scaling(data, scale_outcome=True) if scaling else None),
        predictive=("prediction",),
        log_likelihood=("pointwise",),
        log_prior=("lp_location",),
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


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_prior_batches_preserve_seeded_draws_generated_groups_and_labels(batch_size):
    model, _ = _prepared_model()
    result = sample_prior(model, _normal_prior, draws=10, seed=19, batch_size=batch_size)
    expected = sample_prior(model, _normal_prior, draws=10, seed=19, batch_size=64)
    xr.testing.assert_allclose(result, expected)

    prior_key, generation_key, _ = jax.random.split(jax.random.key(19), 3)
    parameters = jax.jit(jax.vmap(_normal_prior))(jax.random.split(prior_key, 10))
    generated = jax.jit(jax.vmap(lambda key, values: model.generate_quantities(key, values, model.data)))(
        jax.random.split(generation_key, 10), parameters
    )
    np.testing.assert_allclose(result["prior"]["location"].values[0], parameters["location"], rtol=2e-6)
    assert set(result.children) == {
        "prior",
        "prior_predictive",
        "prior_generated_quantities",
        "observed_data",
        "constant_data",
    }
    assert "pointwise" not in result["prior_generated_quantities"]
    assert "lp_location" not in result["prior_generated_quantities"]
    assert "log_prior" not in result
    for group, name in (("prior_predictive", "prediction"), ("prior_generated_quantities", "mean")):
        np.testing.assert_allclose(result[group][name].values[0], generated[name], rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(result[group]["time"], [10, 11, 12])
        assert result[group][name].dims == ("chain", "draw", "time")
        assert isinstance(result[group][name].data, np.ndarray)


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


@pytest.mark.parametrize("registered", [False, True])
def test_per_call_prior_override_does_not_replace_the_attached_prior(registered):
    prior = {"location": Prior(normal, location=0.0, scale=1.0)} if registered else _normal_prior
    model = Model({"location": Real()}, lambda data, location: jnp.nan, prior=prior)
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


@pytest.mark.parametrize("generate", [False, True])
def test_prior_auxiliary_inputs_supply_parameter_axes_and_saved_calculations(generate):
    _, data = _prepared_model()
    inputs = xr.Dataset(
        {"lift": ("experiment", [0.1, 0.2, 0.3]), "basis": (("experiment", "component"), np.eye(3))},
        coords={"experiment": ["north", "south", "national"]},
    )

    def forbidden_density(effect):
        raise AssertionError("Prior sampling must not evaluate the model density")

    model = Model(
        {"effect": Real(dims="component")},
        forbidden_density,
        lambda key, lift: {"lift_copy": lift},
        prior=lambda key: {"effect": jnp.array([1.0, 2.0, 3.0])},
        data=Data(data, inputs=inputs),
        transformed_parameters=lambda basis, effect: {"expected_lift": basis @ effect},
        save=("expected_lift",),
        predictive=("lift_copy",),
        generated_dims={"expected_lift": ("experiment",)},
    )
    result = sample_prior(model, draws=2, generate=generate)

    assert result["prior"]["effect"].dims == ("chain", "draw", "component")
    np.testing.assert_array_equal(result["prior"]["component"], np.arange(3))
    assert set(result["observed_data"].data_vars) == {"outcome"}
    for name in inputs.data_vars:
        np.testing.assert_array_equal(result["constant_data"][name], model.data.values[name])
        assert result["constant_data"][name].dtype == model.data.values[name].dtype
    if generate:
        assert result["prior_predictive"]["lift_copy"].dims == ("chain", "draw", "experiment")
        np.testing.assert_array_equal(result["prior_predictive"]["experiment"], inputs.experiment)
        np.testing.assert_array_equal(
            result["prior_generated_quantities"]["expected_lift"], [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]
        )
    else:
        assert "prior_generated_quantities" not in result.children


def test_generation_toggle_does_not_execute_callback_and_keeps_prepared_data():
    prepared, _ = _prepared_model()

    def forbidden_generate(key, location):
        raise AssertionError("Generation was disabled")

    model = Model(
        {"location": Real()},
        lambda location: jnp.nan,
        forbidden_generate,
        data=prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales"),
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


def test_explicit_media_and_seasonality_run_for_each_prior_draw():
    data = prepare_data(
        pd.DataFrame(
            {"week": [0, 1, 2], "sales": [1.0, 2.0, 3.0], "video": [1.0, 3.0, 2.0], "search": [2.0, 1.0, 4.0]}
        ),
        time="week",
        outcome="sales",
        media=["video", "search"],
    )

    def transformed(
        media,
        time,
        intercept,
        paid_coefficient,
        paid_retention,
        paid_half_saturation,
        paid_slope,
        annual_coefficients,
    ):
        carried = geometric_adstock(media, alpha=paid_retention, max_lag=1)
        paid = hill_saturation(carried, half_saturation=paid_half_saturation, slope=paid_slope) * paid_coefficient
        annual = fourier_features(time, period=8, order=1) @ annual_coefficients
        return {"mean": intercept + paid.sum(-1) + annual, "paid": paid, "annual": annual}

    def generate(key, mean, paid, annual, annual_coefficients):
        return {"prediction": mean, "paid": paid, "seasonal": annual, "weights": annual_coefficients}

    model = Model(
        {
            "intercept": Real(),
            "paid_coefficient": Positive(dims="channel"),
            "paid_retention": Interval(0.0, 1.0, dims="channel"),
            "paid_half_saturation": Positive(dims="channel"),
            "paid_slope": Positive(dims="channel"),
            "annual_coefficients": Real(dims="annual_mode"),
        },
        lambda outcome, mean: jnp.nan,
        generate,
        data=data,
        transformed_parameters=transformed,
        coords={"annual_mode": ["sin_1", "cos_1"]},
        generated_dims={"paid": ("time", "channel"), "seasonal": ("time",)},
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
            "annual_coefficients": jax.random.normal(annual_key, (2,)),
        }

    result = sample_prior(model, prior, draws=3)
    parameters = result["prior"]
    assert set(parameters.data_vars) == set(model.parameters)
    assert parameters["paid_retention"].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(parameters["channel"], ["video", "search"])
    generated = result["prior_generated_quantities"]
    assert generated["paid"].dims == ("chain", "draw", "time", "channel")
    assert generated["seasonal"].dims == ("chain", "draw", "time")
    assert generated["weights"].dims == parameters["annual_coefficients"].dims
    for draw in range(3):
        constrained = {name: jnp.asarray(parameters[name].values[0, draw]) for name in model.parameters}
        expected = model.generate_quantities(jax.random.key(0), constrained, model.data)
        np.testing.assert_allclose(
            result["prior_predictive"]["prediction"].values[0, draw], expected["prediction"], rtol=3e-6, atol=2e-6
        )
    prediction = result["prior_predictive"]["prediction"].values
    assert not np.array_equal(prediction[0, 0], prediction[0, 1])


@pytest.mark.parametrize("declared_axes", [False, True])
def test_generated_aliases_inherit_parameter_axes_without_guessing_equal_shapes(declared_axes):
    model = Model(
        {"coefficient": Real(dims="feature") if declared_axes else Real((2,)), "other": Real((2,))},
        lambda data, coefficient, other: jnp.nan,
        lambda key, data, coefficient, other: {"copy": coefficient, "other_copy": other, "sum": coefficient + other},
        dims=None if declared_axes else {"coefficient": ("feature",)},
        coords={"feature": ["first", "second"]},
    )
    result = sample_prior(model, lambda key: {"coefficient": jnp.ones(2), "other": jnp.zeros(2)}, draws=2)
    generated = result["prior_generated_quantities"]
    assert generated["copy"].dims == ("chain", "draw", "feature")
    assert generated["other_copy"].dims == ("chain", "draw", "other_dim_0")
    assert generated["sum"].dims == ("chain", "draw", "sum_dim_0")
    np.testing.assert_array_equal(generated["feature"], ["first", "second"])


def test_prior_draws_use_resolved_simplex_axes_without_requiring_explicit_shapes():
    model = Model(
        {"weights": Simplex(dims=("region", "category"))},
        lambda data, weights: jnp.nan,
        lambda key, data, weights: {"weights_copy": weights},
        coords={"region": ["west", "east"], "category": ["video", "search", "radio"]},
    )

    def prior(key):
        return {"weights": jax.random.dirichlet(key, jnp.ones(model.parameters["weights"].shape))}

    result = sample_prior(model, prior, draws=3, seed=12)
    weights = result["prior"]["weights"]

    assert weights.dims == ("chain", "draw", "region", "category")
    assert weights.shape == (1, 3, 2, 3)
    assert model.parameters["weights"].position_shape == (2, 2)
    np.testing.assert_array_equal(weights["region"], ["west", "east"])
    np.testing.assert_array_equal(weights["category"], ["video", "search", "radio"])
    np.testing.assert_allclose(weights.sum("category"), 1.0, rtol=2e-6)
    xr.testing.assert_equal(result["prior_generated_quantities"]["weights_copy"].rename("weights"), weights)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_prior_saves_transformed_quantities_without_generation_callback_or_density_evaluation(batch_size):
    _, data = _prepared_model()

    def forbidden_density(mean):
        raise AssertionError("Prior saved quantities must not evaluate the model density")

    def transformed(location, controls, outcome):
        mean = location + controls[:, 0]
        return {
            "mean": mean,
            "pointwise": -0.5 * (outcome - mean) ** 2,
            "lp_location": -0.5 * location**2,
            "total": mean.sum(),
        }

    model = Model(
        {"location": Real()},
        forbidden_density,
        data=data,
        prior=_normal_prior,
        transformed_parameters=transformed,
        save=("mean", "pointwise", "lp_location", "total"),
        predictive=("mean",),
        log_likelihood=("pointwise",),
        log_prior=("lp_location",),
    )
    result = sample_prior(model, draws=6, seed=23, batch_size=batch_size)

    assert set(result["prior"].data_vars) == {"location"}
    assert set(result["prior_generated_quantities"].data_vars) == {"total"}
    assert "log_likelihood" not in result
    assert "log_prior" not in result
    assert result["prior_predictive"]["mean"].dims == ("chain", "draw", "time")
    np.testing.assert_array_equal(result["prior_predictive"]["time"], [10, 11, 12])
    expected = result["prior"]["location"].values[..., None] + data.arrays["controls"][:, 0]
    np.testing.assert_allclose(result["prior_predictive"]["mean"], expected, rtol=2e-6)
    np.testing.assert_allclose(result["prior_generated_quantities"]["total"], expected.sum(-1), rtol=2e-6)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_prior_generation_toggle_skips_saved_transforms_entirely(batch_size):
    _, data = _prepared_model()

    def forbidden_transform(location):
        raise AssertionError("Saved transforms must not execute when generation is disabled")

    model = Model(
        {"location": Real()},
        lambda mean: jnp.nan,
        data=data,
        prior=_normal_prior,
        transformed_parameters=forbidden_transform,
        save=("mean",),
    )
    result = sample_prior(model, draws=6, generate=False, batch_size=batch_size)

    assert set(result["prior"].data_vars) == {"location"}
    assert "prior_generated_quantities" not in result
    assert "prior_predictive" not in result


def test_prior_saves_explicit_media_and_seasonal_outputs_with_named_axes():
    data = prepare_data(
        pd.DataFrame({"week": [0, 1, 2], "sales": [1.0, 2.0, 3.0], "video": [1.0, 3.0, 2.0]}),
        time="week",
        outcome="sales",
        media=["video"],
    )

    def transformed(
        media,
        time,
        paid_media_coefficient,
        paid_media_retention,
        paid_media_half_saturation,
        paid_media_slope,
        annual_coefficients,
    ):
        carried = geometric_adstock(media, alpha=paid_media_retention, max_lag=1)
        paid_media = (
            hill_saturation(carried, half_saturation=paid_media_half_saturation, slope=paid_media_slope)
            * paid_media_coefficient
        )
        annual = fourier_features(time, period=8, order=1) @ annual_coefficients
        return {"paid_media": paid_media, "paid_media_total": paid_media.sum(-1), "annual": annual}

    model = Model(
        {
            "paid_media_coefficient": Positive(dims="channel"),
            "paid_media_retention": Interval(0.0, 1.0, dims="channel"),
            "paid_media_half_saturation": Positive(dims="channel"),
            "paid_media_slope": Positive(dims="channel"),
            "annual_coefficients": Real(dims="annual_mode"),
        },
        lambda outcome: jnp.nan,
        data=data,
        transformed_parameters=transformed,
        coords={"annual_mode": ["sin_1", "cos_1"]},
        generated_dims={
            "paid_media": ("time", "channel"),
            "paid_media_total": ("time",),
            "annual": ("time",),
        },
        save=("paid_media", "paid_media_total", "annual"),
    )

    def prior(key):
        return {
            "paid_media_coefficient": jnp.array([0.7]),
            "paid_media_retention": jnp.array([0.2]),
            "paid_media_half_saturation": jnp.array([1.5]),
            "paid_media_slope": jnp.array([1.2]),
            "annual_coefficients": jax.random.normal(key, (2,)),
        }

    result = sample_prior(model, prior, draws=3, seed=25)
    generated = result["prior_generated_quantities"]

    assert set(result["prior"].data_vars) == set(model.parameters)
    assert generated["paid_media"].dims == ("chain", "draw", "time", "channel")
    assert generated["paid_media_total"].dims == ("chain", "draw", "time")
    assert generated["annual"].dims == ("chain", "draw", "time")
    assert result["prior"]["annual_coefficients"].dims == ("chain", "draw", "annual_mode")
    np.testing.assert_array_equal(generated["channel"], ["video"])
    np.testing.assert_allclose(generated["paid_media_total"], generated["paid_media"].sum("channel"), rtol=2e-6)

    for draw in range(3):
        parameters = {name: jnp.asarray(result["prior"][name].values[0, draw]) for name in model.parameters}
        expected = model.generate_quantities(jax.random.key(0), parameters, model.data)
        np.testing.assert_allclose(generated["annual"].values[0, draw], expected["annual"], rtol=2e-6)


@pytest.mark.parametrize("draws", [0, True, 1.5])
def test_invalid_draw_counts_are_rejected(scalar_model, draws):
    with pytest.raises(ValueError, match=r"draws.*positive integer"):
        sample_prior(scalar_model, _normal_prior, draws=draws)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "4", None])
def test_prior_batch_sizes_require_positive_integers_before_evaluation(scalar_model, batch_size):
    def forbidden_prior(key):
        raise AssertionError("Invalid batch sizes must be rejected before evaluating priors")

    with pytest.raises(ValueError, match=r"batch_size.*positive integer"):
        sample_prior(scalar_model, forbidden_prior, draws=2, batch_size=batch_size)


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


@pytest.mark.parametrize("definition", [1.0, normal, "normal"])
def test_model_rejects_invalid_registered_prior_entries(definition):
    with pytest.raises(TypeError, match="Prior") as error:
        Model({"location": Real()}, lambda data, location: jnp.nan, prior={"location": definition})
    assert "location" in str(error.value)


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("form", ["instance", "logpdf"])
def test_registered_prior_density_requires_parameter_mapping(scalar_model, override, form):
    prior = Prior(normal, location=0.0, scale=1.0)
    value = prior if form == "instance" else prior.logpdf
    with pytest.raises(TypeError, match="mapping"):
        if override:
            sample_prior(scalar_model, value, draws=2)
        else:
            Model({"location": Real()}, lambda data, location: jnp.nan, prior=value)


@pytest.mark.parametrize("value", [("lp_location",), ["lp_location"], "lp_location"])
@pytest.mark.parametrize("override", [False, True])
def test_output_names_passed_as_prior_point_to_log_prior(scalar_model, value, override):
    with pytest.raises(TypeError, match="log_prior"):
        if override:
            sample_prior(scalar_model, value, draws=2)
        else:
            Model({"location": Real()}, lambda data, location: jnp.nan, prior=value)


@pytest.mark.parametrize("form", ["instance", "mapping", "sequence", "density"])
def test_prior_definitions_passed_as_log_prior_point_to_prior(form):
    prior = Prior(normal, location=0.0, scale=1.0)
    value = {"instance": prior, "mapping": {"location": prior}, "sequence": [prior], "density": normal}[form]
    with pytest.raises(TypeError, match=r"log_prior.*\bprior\b"):
        Model(
            {"location": Real()},
            lambda data, location: jnp.nan,
            lambda key, data, location: {"lp_location": prior(location)},
            log_prior=value,
        )


@pytest.mark.parametrize("form", ["no_key", "required_positional", "required_keyword", "density"])
@pytest.mark.parametrize("override", [False, True])
def test_prior_callback_signatures_are_rejected_without_execution(scalar_model, form, override):
    def no_key():
        raise AssertionError("Signature validation must not run prior callbacks")

    def required_positional(key, location):
        raise AssertionError("Signature validation must not run prior callbacks")

    def required_keyword(key, *, location):
        raise AssertionError("Signature validation must not run prior callbacks")

    prior = {
        "no_key": no_key,
        "required_positional": required_positional,
        "required_keyword": required_keyword,
        "density": normal,
    }[form]
    with pytest.raises(TypeError, match=r"prior\(key\)") as error:
        if override:
            sample_prior(scalar_model, prior, draws=2)
        else:
            Model({"location": Real()}, lambda data, location: jnp.nan, prior=prior)
    assert "mapping" in str(error.value)
    assert "density" in str(error.value)


@pytest.mark.parametrize(
    "form", ["positional_only", "defaults", "callable_object", "callable_sequence", "partial", "jit"]
)
@pytest.mark.parametrize("override", [False, True])
def test_compatible_prior_callbacks_are_validated_without_executing_at_construction(form, override):
    calls = []

    def positional_only(key, /):
        calls.append(True)
        return _normal_prior(key)

    def defaults(key, location=0.0, *, scale=1.0):
        calls.append(True)
        return {"location": location + scale * jax.random.normal(key)}

    def with_location(location, key):
        return defaults(key, location)

    class Sampler:
        def __call__(self, key):
            return positional_only(key)

    class SamplerSequence(list):
        def __call__(self, key):
            return positional_only(key)

    prior = {
        "positional_only": positional_only,
        "defaults": defaults,
        "callable_object": Sampler(),
        "callable_sequence": SamplerSequence(),
        "partial": partial(with_location, 0.0),
        "jit": jax.jit(positional_only),
    }[form]
    model = Model({"location": Real()}, lambda data, location: jnp.nan, prior=None if override else prior)
    assert calls == []
    result = sample_prior(model, prior if override else None, draws=3)
    assert calls
    assert result["prior"]["location"].shape == (1, 3)
    assert np.unique(result["prior"]["location"]).size == 3


def test_opaque_prior_callback_keeps_runtime_output_validation():
    calls = []

    class OpaqueSampler:
        __signature__ = "unavailable"

        def __call__(self, key):
            calls.append(True)
            return jnp.array(-1.0)

    prior = OpaqueSampler()
    model = Model({"location": Real()}, lambda data, location: jnp.nan, prior=prior)
    assert calls == []
    with pytest.raises(TypeError, match="mapping"):
        sample_prior(model, draws=2)
    assert calls


@pytest.mark.parametrize("names", [(), ("extra",), ("location", "extra")])
@pytest.mark.parametrize("registered", [False, True])
def test_prior_key_errors_identify_missing_and_unexpected_names(scalar_model, names, registered):
    values = {name: Prior(normal, location=0.0, scale=1.0) if registered else 1.0 for name in names}
    with pytest.raises(ValueError) as error:
        if registered:
            Model({"location": Real()}, lambda data, location: jnp.nan, prior=values)
        else:
            sample_prior(scalar_model, lambda key: values, draws=2)
    message = str(error.value).lower()
    if "location" not in names:
        assert "missing" in message and "location" in message
    if "extra" in names:
        assert "unexpected" in message and "extra" in message


@pytest.mark.parametrize("registered", [False, True])
def test_prior_mapping_keys_must_be_strings(scalar_model, registered):
    value = Prior(normal, location=0.0, scale=1.0) if registered else 1.0
    values = {"location": value, 1: value}
    with pytest.raises((TypeError, ValueError), match=r"name|string"):
        if registered:
            Model({"location": Real()}, lambda data, location: jnp.nan, prior=values)
        else:
            sample_prior(scalar_model, lambda key: values, draws=2)


def test_generate_flag_must_be_boolean(scalar_model):
    with pytest.raises(TypeError, match=r"generate.*bool"):
        sample_prior(scalar_model, _normal_prior, generate=1)


@pytest.mark.parametrize("value", [None, 1.0, (1.0,)])
def test_prior_output_must_be_a_mapping(scalar_model, value):
    with pytest.raises(TypeError, match="mapping"):
        sample_prior(scalar_model, lambda key: value, draws=2)


@pytest.mark.parametrize("form", ["prior", "density", "output_name"])
def test_prior_draw_callbacks_reject_definitions_and_output_names(scalar_model, form):
    value = {
        "prior": Prior(normal, location=0.0, scale=1.0),
        "density": normal,
        "output_name": "lp_location",
    }[form]
    with pytest.raises(TypeError, match="location") as error:
        sample_prior(scalar_model, lambda key: {"location": value}, draws=2)
    message = str(error.value)
    assert "prior=" in message
    assert "log_prior" in message


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
@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_support_validation_covers_later_draws(invalid, batch_size):
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
        sample_prior(model, invalid_later_prior, draws=8, seed=11, batch_size=batch_size)


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
