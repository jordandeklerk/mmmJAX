"""Tests for generated quantities from existing posterior draws."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

import mmmjax
import mmmjax.inference.sampling as sampling
from mmmjax import (
    CorrelationCholesky,
    Model,
    Positive,
    Prior,
    Real,
    Simplex,
    dirichlet,
    dirichlet_logpdf,
    generate_quantities,
    lkj_cholesky,
    lkj_cholesky_logpdf,
    lognormal,
    multivariate_normal,
    multivariate_normal_logpdf,
    normal,
    normal_logpdf,
    prepare_data,
)
from mmmjax.data._results import _collect_results


@pytest.mark.parametrize("callback_binding", ["positional", "keyword"])
def test_generated_time_inputs_retain_date_labels_without_becoming_observed_data(callback_binding):
    frame = pl.DataFrame({"period": [10, 11, 12], "sales": [1.0, 2.0, 3.0], "search": [5.0, 6.0, 7.0]})
    data = prepare_data(frame, time="period", outcome="sales", media=["search"])

    def density(outcome, level):
        return -jnp.sum((outcome - level) ** 2)

    def generate(key, time, media_time, level):
        return {"elapsed": time, "exposure_elapsed": media_time, "predictive": {"prediction": time + level}}

    if callback_binding == "positional":
        model = Model(data, None, {"level": Real()}, None, density, generate)
    else:
        model = Model(data=data, parameters={"level": Real()}, log_density=density, generated_quantities=generate)
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)})
    for new_data in (None, frame.slice(1)):
        evaluated = generate_quantities(model, results, new_data=new_data)
        expected = [0.0, 1.0, 2.0] if new_data is None else [1.0, 2.0]
        assert evaluated["generated_quantities"]["elapsed"].dims == ("chain", "draw", "time")
        assert evaluated["generated_quantities"]["exposure_elapsed"].dims == ("chain", "draw", "media_time")
        np.testing.assert_array_equal(evaluated["generated_quantities"]["elapsed"][0, 0], expected)
        np.testing.assert_array_equal(evaluated["generated_quantities"]["exposure_elapsed"][0, 0], expected)
        if new_data is None:
            assert set(evaluated["observed_data"].data_vars) == {"outcome"}
            assert set(evaluated["constant_data"].data_vars) == {"media"}
        else:
            assert set(evaluated["predictions_constant_data"].data_vars) == {"outcome", "media"}


def _unused_density(data, scale):
    raise AssertionError("Generation must not evaluate the log density")


def _generate(key, data, scale):
    mean = data["value"] * scale
    return {"mean": mean, "predictive": {"prediction": mean + jax.random.normal(key, mean.shape)}}


@pytest.fixture
def model():
    return Model(
        parameters={"scale": Positive((2,))},
        log_density=_unused_density,
        generated_quantities=_generate,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims={"mean": ("channel",), "prediction": ("channel",)},
    )


@pytest.fixture
def results():
    values = np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2)
    tree = _collect_results(
        {"scale": values},
        sample_stats={"lp": np.zeros((2, 3), dtype=np.float32)},
        generated_quantities={"old": np.ones((2, 3), dtype=np.float32)},
        dims={"scale": ("channel",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30], "channel": ["search", "video"]},
    )
    tree.attrs.update(inference_method="nuts", seed=19)
    return tree


def test_generate_quantities_reuses_constrained_draws_without_sampling(model, results, monkeypatch):
    def no_sampling(*args, **kwargs):
        raise AssertionError("Generation must not run the sampler")

    monkeypatch.setattr(sampling, "_sample_nuts", no_sampling)
    original = results.copy(deep=True)
    evaluated = generate_quantities(model, results, new_data={"value": jnp.array([2.0, 3.0])}, seed=8)

    assert mmmjax.generate_quantities is sampling.generate_quantities
    assert "generate_quantities" in mmmjax.__all__
    assert "sample_stats" not in evaluated
    assert "old" not in evaluated["generated_quantities"]
    assert "inference_method" not in evaluated.attrs
    assert "seed" not in evaluated.attrs
    assert evaluated.attrs["generation_seed"] == 8
    xr.testing.assert_identical(evaluated["posterior"], original["posterior"])
    np.testing.assert_array_equal(
        evaluated["generated_quantities"]["mean"], results["posterior"]["scale"] * np.array([2.0, 3.0])
    )
    xr.testing.assert_identical(results, original)
    evaluated["posterior"]["scale"].values[...] = 0
    xr.testing.assert_identical(results, original)


def test_generate_quantities_random_draws_are_reproducible_and_independent(model, results):
    first = generate_quantities(model, results, new_data={"value": 0.0}, seed=9)
    second = generate_quantities(model, results, new_data={"value": 0.0}, seed=9)
    different = generate_quantities(model, results, new_data={"value": 0.0}, seed=10)
    xr.testing.assert_identical(first, second)
    predictions = first["posterior_predictive"]["prediction"].values
    assert not np.array_equal(predictions, different["posterior_predictive"]["prediction"].values)
    assert np.unique(predictions).size == predictions.size


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_generated_batches_preserve_key_assignment_draw_order_and_labels(model, results, batch_size):
    new_data = {"value": np.array([2.0, 3.0], dtype=np.float32)}
    original = results.copy(deep=True)
    evaluated = generate_quantities(model, results, new_data=new_data, seed=9, batch_size=batch_size)
    expected = generate_quantities(model, results, new_data=new_data, seed=9, batch_size=64)
    xr.testing.assert_allclose(evaluated, expected)
    xr.testing.assert_identical(results, original)

    generation_key, _ = jax.random.split(jax.random.key(9))
    keys = jax.random.split(generation_key, (2, 3))
    posterior = {"scale": jnp.asarray(results["posterior"]["scale"].values)}
    reference = jax.jit(jax.vmap(jax.vmap(lambda key, values: model.generate_quantities(key, values, new_data))))(
        keys, posterior
    )
    direct = {"mean": reference["mean"], "prediction": reference["predictive"]["prediction"]}
    for group, name in (("generated_quantities", "mean"), ("posterior_predictive", "prediction")):
        np.testing.assert_allclose(evaluated[group][name], direct[name], rtol=2e-6, atol=2e-6)
        assert evaluated[group][name].dims == ("chain", "draw", "channel")
        assert isinstance(evaluated[group][name].data, np.ndarray)
        np.testing.assert_array_equal(evaluated[group]["chain"], [4, 8])
        np.testing.assert_array_equal(evaluated[group]["draw"], [10, 20, 30])
        np.testing.assert_array_equal(evaluated[group]["channel"], ["search", "video"])


def test_generate_quantities_preserves_selected_sample_labels_and_transposed_axes(model, results):
    posterior = results["posterior"].to_dataset().isel(chain=[1], draw=[2, 0]).transpose("channel", "draw", "chain")
    selected = xr.DataTree.from_dict({"posterior": posterior})
    evaluated = generate_quantities(model, selected, new_data={"value": 2.0})
    for group in ("posterior", "posterior_predictive", "generated_quantities"):
        np.testing.assert_array_equal(evaluated[group].coords["chain"], [8])
        np.testing.assert_array_equal(evaluated[group].coords["draw"], [30, 10])
    assert evaluated["posterior"]["scale"].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(
        evaluated["posterior"]["scale"], posterior["scale"].transpose("chain", "draw", "channel")
    )


@pytest.mark.parametrize("seed", [-1, True, 0.5, "1", 2**32, np.int64(2**40), 2**63])
def test_generate_quantities_rejects_invalid_seeds(model, results, seed):
    with pytest.raises(ValueError, match=r"seed must be a nonnegative integer below 2\*\*32, got"):
        generate_quantities(model, results, seed=seed)


@pytest.mark.parametrize("seed", [2**32 - 1, np.uint32(2**32 - 1)])
def test_generate_quantities_accepts_the_largest_32_bit_seed(model, results, seed):
    new_data = {"value": np.array([2.0, 3.0], dtype=np.float32)}
    generation_key, _ = jax.random.split(jax.random.key(2**32 - 1))
    keys = jax.random.split(generation_key, (2, 3))
    posterior = {"scale": jnp.asarray(results["posterior"]["scale"].values)}
    expected = jax.vmap(jax.vmap(lambda key, values: model.generate_quantities(key, values, new_data)))(keys, posterior)

    evaluated = generate_quantities(model, results, new_data=new_data, seed=seed)

    assert evaluated.attrs["generation_seed"] == 2**32 - 1
    np.testing.assert_allclose(
        evaluated["posterior_predictive"]["prediction"], expected["predictive"]["prediction"], rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "4", None])
def test_generated_batch_sizes_require_positive_integers(model, results, batch_size):
    with pytest.raises(ValueError, match=r"batch_size.*positive integer"):
        generate_quantities(model, results, new_data={"value": 1.0}, batch_size=batch_size)


def test_generate_quantities_requires_model_callback_and_result_tree(model, results):
    with pytest.raises(TypeError, match="model must be a Model"):
        generate_quantities(None, results)
    no_callback = Model(parameters={"scale": Positive((2,))}, log_density=_unused_density)
    with pytest.raises(ValueError, match="generated_quantities callback"):
        generate_quantities(no_callback, results)
    with pytest.raises(TypeError, match=r"xarray\.DataTree"):
        generate_quantities(model, results["posterior"].to_dataset())
    with pytest.raises(ValueError, match="posterior group"):
        generate_quantities(model, xr.DataTree())


@pytest.mark.parametrize("axis", ["chain", "draw"])
def test_generate_quantities_rejects_empty_sample_axes(model, results, axis):
    empty = xr.DataTree.from_dict({"posterior": results["posterior"].to_dataset().isel({axis: slice(0, 0)})})
    with pytest.raises(ValueError, match=f"at least one {axis}"):
        generate_quantities(model, empty)


@pytest.mark.parametrize("change", ["missing", "extra", "dimension", "shape", "labels", "order"])
def test_generate_quantities_rejects_incompatible_posterior(model, results, change):
    posterior = results["posterior"].to_dataset().copy(deep=True)
    if change == "missing":
        posterior = posterior.drop_vars("scale")
    elif change == "extra":
        posterior["unknown"] = posterior["scale"]
    elif change == "dimension":
        posterior = posterior.rename(channel="feature")
    elif change == "shape":
        posterior = posterior.isel(channel=[0])
    elif change == "labels":
        posterior = posterior.assign_coords(channel=["other", "video"])
    else:
        posterior = posterior.isel(channel=[1, 0])
    with pytest.raises(ValueError, match="Posterior"):
        generate_quantities(model, xr.DataTree.from_dict({"posterior": posterior}))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_generate_quantities_rejects_nonfinite_posterior(model, results, value):
    results["posterior"]["scale"].values[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite real numbers"):
        generate_quantities(model, results)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_generate_quantities_does_not_silently_reduce_posterior_precision(model, results, batch_size):
    posterior = results["posterior"].to_dataset().astype(np.float64)
    double = xr.DataTree.from_dict({"posterior": posterior})
    with jax.enable_x64(False), pytest.raises(ValueError, match="64-bit mode"):
        generate_quantities(model, double, new_data={"value": 1.0}, batch_size=batch_size)
    with jax.enable_x64(True):
        evaluated = generate_quantities(model, double, new_data={"value": 1.0}, batch_size=batch_size)
    assert evaluated["posterior"]["scale"].dtype == np.float64
    assert evaluated["generated_quantities"]["mean"].dtype == model.parameters["scale"].dtype


def test_generate_quantities_keeps_custom_parameter_and_output_axes(model, results):
    def generate(key, data, scale):
        return {"scale_copy": scale, "average": jnp.mean(scale)}

    custom = Model(
        parameters={"scale": Positive((2,))},
        log_density=_unused_density,
        generated_quantities=generate,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
    )
    evaluated = generate_quantities(custom, results)
    assert evaluated["generated_quantities"]["scale_copy"].dims == ("chain", "draw", "channel")
    assert evaluated["generated_quantities"]["average"].dims == ("chain", "draw")


def test_generate_quantities_does_not_silently_drop_missing_declared_outputs(results):
    model = Model(
        parameters={"scale": Positive((2,))},
        log_density=_unused_density,
        generated_quantities=lambda key, data, scale: {"scale_copy": scale},
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims={"pointwise": ("channel",)},
    )
    with pytest.raises(ValueError, match="missing outputs"):
        generate_quantities(model, results)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_hierarchical_log_prior_terms_preserve_draw_values_and_labels(batch_size):
    def forbidden_density(data, location, scale, coefficient):
        raise AssertionError("Generating log prior terms must not evaluate the model density")

    def generate(key, data, location, scale, coefficient):
        log_prior = {
            "lp_location": normal(location, 0.0, 2.0),
            "lp_scale": -scale,
            "lp_coefficient": normal_logpdf(coefficient, location, scale),
        }
        return {"log_prior": log_prior, "mean": jnp.mean(coefficient)}

    model = Model(
        parameters={"location": Real(), "scale": Positive(), "coefficient": Real((2,))},
        log_density=forbidden_density,
        generated_quantities=generate,
        dims={"coefficient": ("region",)},
        coords={"region": ["west", "east"]},
        generated_dims={"lp_coefficient": ("region",)},
    )
    location = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    scale = 0.5 + location
    coefficient = np.stack([location - 0.25, location + 0.75], axis=-1)
    original = _collect_results(
        {"location": location, "scale": scale, "coefficient": coefficient},
        dims={"coefficient": ("region",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30], "region": ["west", "east"]},
    )
    evaluated = generate_quantities(model, original, batch_size=batch_size)
    log_prior = evaluated["log_prior"]
    expected = {
        "lp_location": -0.5 * (location / 2) ** 2 - np.log(2) - 0.5 * np.log(2 * np.pi),
        "lp_scale": -scale,
        "lp_coefficient": (
            -0.5 * ((coefficient - location[..., None]) / scale[..., None]) ** 2
            - np.log(scale[..., None])
            - 0.5 * np.log(2 * np.pi)
        ),
    }
    assert set(log_prior.data_vars) == set(expected)
    assert set(evaluated["generated_quantities"].data_vars) == {"mean"}
    assert log_prior.attrs["sample_dims"] == ["chain", "draw"]
    np.testing.assert_array_equal(log_prior["chain"], [4, 8])
    np.testing.assert_array_equal(log_prior["draw"], [10, 20, 30])
    np.testing.assert_array_equal(log_prior["region"], ["west", "east"])
    for name, values in expected.items():
        np.testing.assert_allclose(log_prior[name], values, rtol=2e-6)
        assert isinstance(log_prior[name].data, np.ndarray)
        expected_dims = ("chain", "draw", "region") if name == "lp_coefficient" else ("chain", "draw")
        assert log_prior[name].dims == expected_dims
    xr.testing.assert_identical(evaluated["posterior"], original["posterior"])


@pytest.mark.parametrize("explicit_dims", [False, True])
def test_log_prior_axes_follow_parameter_names_not_matching_shapes(results, explicit_dims):
    model = Model(
        parameters={"scale": Positive((2,))},
        log_density=_unused_density,
        generated_quantities=lambda key, data, scale: {"log_prior": {"scale": -scale, "lp_scale": -scale}},
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims={"scale": ("channel",), "lp_scale": ("channel",)} if explicit_dims else None,
    )
    evaluated = generate_quantities(model, results)
    for name in ("scale", "lp_scale"):
        expected_axis = "channel" if explicit_dims or name == "scale" else f"{name}_dim_0"
        assert evaluated["log_prior"][name].dims == ("chain", "draw", expected_axis)
        np.testing.assert_array_equal(evaluated["log_prior"][name], -results["posterior"]["scale"])


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_prior_log_densities_preserve_batch_axes_and_reduce_distribution_events(batch_size):
    def forbidden_density(data, **parameters):
        raise AssertionError("Generating prior terms must not evaluate the model density")

    def generate(key, data, scale, independent, coefficient, weights, factor):
        values = {
            "scale": scale,
            "independent": independent,
            "coefficient": coefficient,
            "weights": weights,
            "factor": factor,
        }
        return {"log_prior": {f"log_prior_{name}": priors[name].logpdf(values[name]) for name in priors}}

    priors = {
        "scale": Prior(lognormal, location=0.2, scale=0.9),
        "independent": Prior(normal, location=jnp.zeros(3), scale=1.0),
        "coefficient": Prior(multivariate_normal, location=jnp.zeros(3), scale_tril=jnp.eye(3)),
        "weights": Prior(dirichlet, concentration=jnp.array([2.0, 3.0, 4.0])),
        "factor": Prior(lkj_cholesky, concentration=2.0),
    }
    dims = {
        "independent": ("group", "channel"),
        "coefficient": ("group", "channel"),
        "weights": ("group", "channel"),
        "factor": ("group", "channel", "channel_to"),
    }
    coords = {
        "group": ["west", "east"],
        "channel": ["search", "video", "radio"],
        "channel_to": ["search", "video", "radio"],
    }
    model = Model(
        parameters={
            "scale": Positive(),
            "independent": Real(dims=("group", "channel")),
            "coefficient": Real(dims=("group", "channel")),
            "weights": Simplex(dims=("group", "channel")),
            "factor": CorrelationCholesky(dims=("group", "channel", "channel_to")),
        },
        log_density=forbidden_density,
        generated_quantities=generate,
        coords=coords,
        generated_dims={
            "log_prior_independent": ("group", "channel"),
            "log_prior_coefficient": ("group",),
            "log_prior_weights": ("group",),
            "log_prior_factor": ("group",),
        },
    )
    coefficients = np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3) / 10
    posterior = {
        "scale": 0.5 + np.arange(6, dtype=np.float32).reshape(2, 3) / 4,
        "independent": coefficients,
        "coefficient": coefficients,
        "weights": (coefficients + 1) / (coefficients + 1).sum(axis=-1, keepdims=True),
        "factor": np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 2, 3, 3)).copy(),
    }
    original = _collect_results(posterior, dims=dims, coords=coords | {"chain": [4, 8], "draw": [10, 20, 30]})
    evaluated = generate_quantities(model, original, batch_size=batch_size)
    assert set(evaluated.children) == {"posterior", "log_prior"}
    expected = {
        "scale": (
            -0.5 * ((np.log(posterior["scale"]) - 0.2) / 0.9) ** 2
            - np.log(posterior["scale"])
            - np.log(0.9)
            - 0.5 * np.log(2 * np.pi)
        ),
        "independent": -0.5 * coefficients**2 - 0.5 * np.log(2 * np.pi),
        "coefficient": multivariate_normal_logpdf(coefficients, jnp.zeros(3), jnp.eye(3)),
        "weights": dirichlet_logpdf(posterior["weights"], jnp.array([2.0, 3.0, 4.0])),
        "factor": lkj_cholesky_logpdf(posterior["factor"], 2.0),
    }
    log_prior = evaluated["log_prior"]
    assert set(log_prior.data_vars) == {f"log_prior_{name}" for name in priors}
    np.testing.assert_array_equal(log_prior["chain"], [4, 8])
    np.testing.assert_array_equal(log_prior["draw"], [10, 20, 30])
    np.testing.assert_array_equal(log_prior["group"], ["west", "east"])
    np.testing.assert_array_equal(log_prior["channel"], ["search", "video", "radio"])
    for name, values in expected.items():
        variable = log_prior[f"log_prior_{name}"]
        axes = () if name == "scale" else ("group", "channel") if name == "independent" else ("group",)
        assert variable.dims == ("chain", "draw", *axes)
        np.testing.assert_allclose(variable, values, rtol=2e-6, atol=2e-6)
    xr.testing.assert_identical(evaluated["posterior"], original["posterior"])


def test_log_prior_terms_keep_their_names_and_axes(results):
    scale_prior = Prior(lognormal, location=0.2, scale=0.9)
    model = Model(
        parameters={"scale": Positive((2,))},
        log_density=_unused_density,
        generated_quantities=lambda key, data, scale: {
            "log_prior": {"log_prior_scale": scale_prior.logpdf(scale), "manual": normal(scale, 0.0, 1.0)},
            "mean": jnp.mean(scale),
        },
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims={"log_prior_scale": ("channel",)},
    )
    evaluated = generate_quantities(model, results)
    assert set(evaluated["log_prior"].data_vars) == {"log_prior_scale", "manual"}
    assert evaluated["log_prior"]["log_prior_scale"].dims == ("chain", "draw", "channel")
    assert evaluated["log_prior"]["manual"].dims == ("chain", "draw")
    assert set(evaluated["generated_quantities"].data_vars) == {"mean"}


def _saved_data(*, start=0, observations=3):
    return prepare_data(
        pl.DataFrame(
            {
                "week": np.arange(start, start + observations),
                "sales": np.arange(1.0, observations + 1),
                "price": np.arange(2.0, observations + 2),
                "promotion": np.full(observations, 0.5),
            }
        ),
        time="week",
        outcome="sales",
        controls=["price", "promotion"],
    )


def _saved_model(*, generated_dims=None, generate=None):
    def forbidden_density(signal):
        raise AssertionError("Generating quantities must not evaluate the model density")

    def keep_everything(key, signal, pointwise, total):
        return {"signal": signal, "pointwise": pointwise, "total": total}

    def transformed(controls, scale, outcome):
        signal = controls @ scale
        return {"signal": signal, "pointwise": -0.5 * (outcome - signal) ** 2, "total": signal.sum()}

    return Model(
        parameters={"scale": Positive((2,))},
        log_density=forbidden_density,
        generated_quantities=keep_everything if generate is None else generate,
        data=_saved_data(),
        transformed_parameters=transformed,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims=generated_dims,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_generate_quantities_returns_transforms_and_recomputes_scenarios(results, monkeypatch, batch_size):
    def forbidden_sampler(*args, **kwargs):
        raise AssertionError("Generating quantities must not run the sampler")

    monkeypatch.setattr(sampling, "_sample_nuts", forbidden_sampler)
    model = _saved_model(generated_dims={"signal": ("time",), "pointwise": ("time",)})
    original = results.copy(deep=True)
    original_controls = np.array(model.data.values["controls"], copy=True)
    baseline = generate_quantities(model, results, batch_size=batch_size)
    scenario = _saved_data(start=6, observations=2)
    changed = generate_quantities(model, results, new_data=scenario, batch_size=batch_size)

    assert set(changed["generated_quantities"].data_vars) == {"signal", "pointwise", "total"}
    assert changed["generated_quantities"]["signal"].dims == ("chain", "draw", "time")
    assert changed["generated_quantities"]["signal"].shape == (2, 3, 2)
    assert baseline["generated_quantities"]["signal"].shape == (2, 3, 3)
    np.testing.assert_array_equal(changed["generated_quantities"]["time"], [6, 7])
    expected = np.asarray(results["posterior"]["scale"]) @ scenario.arrays["controls"].T
    np.testing.assert_allclose(changed["generated_quantities"]["signal"], expected, rtol=2e-6)
    np.testing.assert_allclose(changed["generated_quantities"]["total"], expected.sum(-1), rtol=2e-6)
    xr.testing.assert_identical(changed["posterior"], original["posterior"])
    xr.testing.assert_identical(results, original)
    np.testing.assert_array_equal(model.data.values["controls"], original_controls)


def test_transformed_outputs_routed_to_result_groups_keep_observation_labels(results):
    def generate(key, signal, pointwise, total):
        return {"predictive": {"signal": signal}, "log_likelihood": {"pointwise": pointwise}, "total": total}

    model = _saved_model(generate=generate)
    result = generate_quantities(model, results)

    assert result["posterior_predictive"]["signal"].dims == ("chain", "draw", "time")
    assert result["log_likelihood"]["pointwise"].dims == ("chain", "draw", "time")
    assert set(result["generated_quantities"].data_vars) == {"total"}
    for group in ("posterior_predictive", "log_likelihood"):
        np.testing.assert_array_equal(result[group]["time"], [0, 1, 2])
        np.testing.assert_array_equal(result[group]["chain"], results["posterior"]["chain"])
        np.testing.assert_array_equal(result[group]["draw"], results["posterior"]["draw"])


def test_transformed_log_prior_terms_are_collected_through_the_generation_callback(results):
    model = Model(
        parameters={"scale": Positive((2,))},
        log_density=lambda scale: -scale.sum(),
        generated_quantities=lambda key, lp_scale: {"log_prior": {"lp_scale": lp_scale}},
        data=_saved_data(),
        transformed_parameters=lambda scale: {"lp_scale": -scale.sum()},
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
    )
    evaluated = generate_quantities(model, results)
    assert evaluated["log_prior"]["lp_scale"].dims == ("chain", "draw")
    np.testing.assert_allclose(evaluated["log_prior"]["lp_scale"], -results["posterior"]["scale"].sum("channel"))
    assert "generated_quantities" not in evaluated


def test_returned_custom_outputs_keep_fallback_axes_without_guessing_observation_dimensions(results):
    result = generate_quantities(_saved_model(), results)

    assert result["generated_quantities"]["signal"].dims == ("chain", "draw", "signal_dim_0")
    assert result["generated_quantities"]["pointwise"].dims == ("chain", "draw", "pointwise_dim_0")
    assert result["generated_quantities"]["total"].dims == ("chain", "draw")


def test_scenarios_for_prepared_models_use_arviz_prediction_groups():
    frame = pl.DataFrame({"period": [10, 11, 12], "sales": [1.0, 2.0, 3.0], "search": [5.0, 6.0, 7.0]})
    data = prepare_data(frame, time="period", outcome="sales", media=["search"])

    def generate(key, outcome, media, level):
        mean = level + media[:, 0]
        return {"predictive": {"outcome": mean}, "log_likelihood": {"outcome": normal_logpdf(outcome, mean, 1.0)}}

    model = Model(
        data=data,
        parameters={"level": Real()},
        log_density=lambda outcome, level: normal(outcome, level, 1.0),
        generated_quantities=generate,
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)})

    fitted = generate_quantities(model, results)
    scenario = generate_quantities(model, results, new_data=frame.slice(1))

    in_sample = {"posterior", "posterior_predictive", "log_likelihood", "observed_data", "constant_data"}
    predictions = {"posterior", "predictions", "predictions_log_likelihood", "predictions_constant_data"}
    assert set(fitted.children) == in_sample
    assert set(scenario.children) == predictions
    np.testing.assert_array_equal(scenario["predictions"]["time"], [11, 12])
    np.testing.assert_array_equal(scenario["predictions_constant_data"]["outcome"], [2.0, 3.0])
    np.testing.assert_array_equal(scenario["predictions_constant_data"]["media"], [[6.0], [7.0]])
    np.testing.assert_allclose(
        scenario["predictions_log_likelihood"]["outcome"], fitted["log_likelihood"]["outcome"].isel(time=[1, 2])
    )


def test_log_prior_terms_named_after_parameters_take_their_axes():
    def density(data, scale, offset):
        raise AssertionError("Generation must not evaluate the log density")

    def generate(key, data, scale, offset):
        terms = normal_logpdf(scale, 0.0, 1.0)
        return {"log_prior": {"scale": terms, "offset": normal_logpdf(offset, 0.0, 1.0), "total": terms.sum()}}

    model = Model(
        parameters={"scale": Positive((2,)), "offset": Real()},
        log_density=density,
        generated_quantities=generate,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
    )
    results = _collect_results(
        {"scale": np.ones((1, 2, 2), dtype=np.float32), "offset": np.zeros((1, 2), dtype=np.float32)},
        dims=model._result_dims,
        coords=model._result_coords,
    )

    evaluated = generate_quantities(model, results, new_data={"value": 0.0})

    assert evaluated["log_prior"]["scale"].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(evaluated["log_prior"]["channel"], ["search", "video"])
    assert evaluated["log_prior"]["offset"].dims == ("chain", "draw")
    assert evaluated["log_prior"]["total"].dims == ("chain", "draw")
