"""Tests for generated quantities from existing posterior draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import mmmjax
import mmmjax.sampling as sampling
from mmmjax import Model, Positive, generate_quantities
from mmmjax._results import _collect_results


def _unused_density(data, scale):
    raise AssertionError("Generation must not evaluate the log density")


def _generate(key, data, scale):
    mean = data["value"] * scale
    return {"mean": mean, "prediction": mean + jax.random.normal(key, mean.shape)}


@pytest.fixture
def model():
    return Model(
        {"scale": Positive((2,))},
        _unused_density,
        _generate,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        generated_dims={"mean": ("channel",), "prediction": ("channel",)},
        predictive=("prediction",),
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


@pytest.mark.parametrize("seed", [-1, True, 0.5, "1"])
def test_generate_quantities_rejects_invalid_seeds(model, results, seed):
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        generate_quantities(model, results, seed=seed)


def test_generate_quantities_requires_model_callback_and_result_tree(model, results):
    with pytest.raises(TypeError, match="model must be a Model"):
        generate_quantities(None, results)
    no_callback = Model({"scale": Positive((2,))}, _unused_density)
    with pytest.raises(ValueError, match="generation callback"):
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


def test_generate_quantities_does_not_silently_reduce_posterior_precision(model, results):
    posterior = results["posterior"].to_dataset().astype(np.float64)
    double = xr.DataTree.from_dict({"posterior": posterior})
    with jax.enable_x64(False), pytest.raises(ValueError, match="64-bit mode"):
        generate_quantities(model, double, new_data={"value": 1.0})
    with jax.enable_x64(True):
        evaluated = generate_quantities(model, double, new_data={"value": 1.0})
    assert evaluated["posterior"]["scale"].dtype == np.float64


def test_generate_quantities_keeps_custom_parameter_and_output_axes(model, results):
    def generate(key, data, scale):
        return {"scale_copy": scale, "average": jnp.mean(scale)}

    custom = Model(
        {"scale": Positive((2,))},
        _unused_density,
        generate,
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
    )
    evaluated = generate_quantities(custom, results)
    assert evaluated["generated_quantities"]["scale_copy"].dims == ("chain", "draw", "channel")
    assert evaluated["generated_quantities"]["average"].dims == ("chain", "draw")


def test_generate_quantities_does_not_silently_drop_missing_declared_outputs(results):
    model = Model(
        {"scale": Positive((2,))},
        _unused_density,
        lambda key, data, scale: {"scale_copy": scale},
        dims={"scale": ("channel",)},
        coords={"channel": ["search", "video"]},
        log_likelihood=("pointwise",),
    )
    with pytest.raises(ValueError, match="missing outputs"):
        generate_quantities(model, results)
