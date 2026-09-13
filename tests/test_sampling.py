"""Tests for sampling JAX models and collecting constrained, labeled draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mmmjax
import mmmjax.sampling as sampling
from mmmjax import (
    Interval,
    Model,
    Positive,
    Real,
    Simplex,
    beta,
    dirichlet,
    fit_data_scaling,
    fourier_features,
    geometric_adstock,
    half_normal,
    hill_saturation,
    lognormal,
    normal,
    normal_logpdf,
    normal_rng,
    prepare_data,
    sample,
)


@pytest.fixture
def normal_model():
    return Model({"location": Real()}, lambda data, location: normal(location, data, 1.0))


@pytest.fixture
def nuts_calls(monkeypatch):
    calls = []

    def stationary_draws(logdensity, initial_positions, keys, *, draws, warmup, target_accept, max_tree_depth):
        calls.append(
            {
                "positions": jax.tree.map(np.array, initial_positions),
                "keys": np.asarray(jax.random.key_data(keys)),
                "draws": draws,
                "warmup": warmup,
                "target_accept": target_accept,
                "max_tree_depth": max_tree_depth,
            }
        )
        positions = jax.tree.map(
            lambda value: jnp.broadcast_to(value[:, None], (value.shape[0], draws, *value.shape[1:])), initial_positions
        )
        chains = next(iter(initial_positions.values())).shape[0]
        return positions, {
            "diverging": jnp.zeros((chains, draws), dtype=bool),
            "reached_max_treedepth": jnp.zeros((chains, draws), dtype=bool),
            "lp": jax.vmap(jax.vmap(logdensity))(positions),
        }

    monkeypatch.setattr(sampling, "_sample_nuts", stationary_draws)
    return calls


def _prepared_model():
    data = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [100.0, 150.0, 200.0], "price": [5.0, 4.0, 3.0]}),
        time="week",
        outcome="sales",
        controls=["price"],
    )

    def density(outcome, intercept):
        return normal(outcome, intercept, 10.0) + normal(intercept, 150.0, 50.0)

    def generate(key, outcome, intercept):
        mean = jnp.full_like(outcome, intercept)
        return {
            "prediction": normal_rng(key, mean, 10.0),
            "pointwise": normal_logpdf(outcome, mean, 10.0),
            "mean": mean,
        }

    model = Model(
        {"intercept": Real()},
        density,
        generate,
        data=data,
        generated_dims={"mean": ("time",)},
        predictive=("prediction",),
        log_likelihood=("pointwise",),
    )
    return model, data


def test_public_sampling_replaces_manual_result_collection():
    assert "sample" in mmmjax.__all__
    assert "collect_results" not in mmmjax.__all__
    assert not hasattr(mmmjax, "collect_results")


def test_normal_nuts_produces_reproducible_independent_chains(normal_model):
    options = {"data": 0.0, "draws": 200, "warmup": 150, "chains": 2, "seed": 19, "target_accept": 0.8}
    first = sample(normal_model, **options)
    second = sample(normal_model, **options)
    assert isinstance(first, xr.DataTree)
    posterior = first["posterior"]["location"]
    assert posterior.dims == ("chain", "draw")
    assert posterior.shape == (2, 200)
    assert np.isfinite(posterior).all()
    assert abs(float(posterior.mean())) < 0.3
    assert 0.65 < float(posterior.std()) < 1.4
    assert not np.array_equal(posterior.values[0], posterior.values[1])
    xr.testing.assert_equal(first["posterior"].to_dataset(), second["posterior"].to_dataset())
    assert first["sample_stats"]["diverging"].dtype == np.bool_
    assert first["sample_stats"]["diverging"].shape == (2, 200)
    for group in first.children.values():
        for variable in group.data_vars.values():
            assert isinstance(variable.data, np.ndarray)


@pytest.mark.parametrize("named_axes", [False, True])
def test_nuts_allows_parameters_named_after_sampler_diagnostics(named_axes):
    model = Model(
        {"energy": Real(shape=(2,))},
        lambda data, energy: normal(energy, 0.0, 1.0),
        dims={"energy": ("feature",)} if named_axes else None,
        coords={"feature": ["first", "second"]} if named_axes else None,
    )
    results = sample(model, draws=10, warmup=50, chains=1, seed=19, generate=False)
    parameter_axis = "feature" if named_axes else "energy_dim_0"
    assert results["posterior"]["energy"].dims == ("chain", "draw", parameter_axis)
    assert results["posterior"]["energy"].shape == (1, 10, 2)
    assert results["sample_stats"]["energy"].dims == ("chain", "draw")
    assert results["sample_stats"]["energy"].shape == (1, 10)
    assert parameter_axis not in results["sample_stats"].coords
    assert np.isfinite(results["posterior"]["energy"]).all()
    assert np.isfinite(results["sample_stats"]["energy"]).all()
    if named_axes:
        np.testing.assert_array_equal(results["posterior"]["feature"], ["first", "second"])


def test_nuts_returns_positive_parameters_and_full_simplex_events():
    model = Model(
        {"scale": Positive(), "weights": Simplex((3,))},
        lambda data, scale, weights: half_normal(scale, 1.0) + dirichlet(weights, jnp.array([2.0, 3.0, 4.0])),
        dims={"weights": ("category",)},
        coords={"category": ["a", "b", "c"]},
    )
    result = sample(
        model,
        draws=40,
        warmup=80,
        chains=1,
        seed=5,
        initial_values={"scale": 1.0, "weights": np.array([0.2, 0.3, 0.5])},
    )
    assert result["posterior"]["scale"].shape == (1, 40)
    assert (result["posterior"]["scale"].values > 0).all()
    assert result["posterior"]["weights"].shape == (1, 40, 3)
    assert (result["posterior"]["weights"].values > 0).all()
    np.testing.assert_allclose(result["posterior"]["weights"].sum("category"), 1.0, rtol=2e-6)
    np.testing.assert_array_equal(result["posterior"]["category"], ["a", "b", "c"])


def test_default_initialization_and_keys_are_independent_per_chain(normal_model, nuts_calls):
    sample(normal_model, data=0.0, draws=3, warmup=5, chains=3, seed=12)
    call = nuts_calls[0]
    assert call["positions"]["location"].shape == (3,)
    assert len(np.unique(call["positions"]["location"])) == 3
    assert len(np.unique(call["keys"], axis=0)) == 3


def test_explicit_constrained_initial_values_are_replicated_and_sampler_options_forwarded(normal_model, nuts_calls):
    initial = {"location": np.array(1.5)}
    result = sample(
        normal_model,
        data=3.0,
        draws=4,
        warmup=7,
        chains=3,
        seed=11,
        target_accept=0.9,
        max_tree_depth=6,
        initial_values=initial,
    )
    np.testing.assert_array_equal(nuts_calls[0]["positions"]["location"], [1.5, 1.5, 1.5])
    assert nuts_calls[0]["draws"] == 4
    assert nuts_calls[0]["warmup"] == 7
    assert nuts_calls[0]["target_accept"] == 0.9
    assert nuts_calls[0]["max_tree_depth"] == 6
    np.testing.assert_array_equal(result["posterior"]["location"], np.full((3, 4), 1.5))
    np.testing.assert_allclose(result["sample_stats"]["lp"], normal(1.5, 3.0, 1.0), rtol=1e-6)
    assert initial["location"] == 1.5


def test_prepared_data_and_selected_generated_outputs_are_collected_automatically(nuts_calls):
    model, data = _prepared_model()
    result = sample(model, draws=4, warmup=5, chains=2, seed=7, initial_values={"intercept": 150.0})
    assert set(result.children) == {
        "posterior",
        "sample_stats",
        "posterior_predictive",
        "log_likelihood",
        "generated_quantities",
        "observed_data",
        "constant_data",
    }
    for group, name in (
        ("posterior_predictive", "prediction"),
        ("log_likelihood", "pointwise"),
        ("generated_quantities", "mean"),
    ):
        assert result[group][name].dims == ("chain", "draw", "time")
        assert result[group][name].shape == (2, 4, 3)
        np.testing.assert_array_equal(result[group]["time"], [10, 11, 12])
    np.testing.assert_array_equal(result["observed_data"]["outcome"], data.arrays["outcome"])
    np.testing.assert_array_equal(result["constant_data"]["controls"], data.arrays["controls"])
    np.testing.assert_array_equal(result["generated_quantities"]["mean"], np.full((2, 4, 3), 150.0))
    expected = np.broadcast_to(normal_logpdf(data.arrays["outcome"], 150.0, 10.0), (2, 4, 3))
    np.testing.assert_allclose(result["log_likelihood"]["pointwise"], expected, rtol=1e-6)
    predictions = result["posterior_predictive"]["prediction"].values
    assert not np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[:, 0], predictions[:, 1])


def test_generation_can_be_disabled_without_executing_the_callback(nuts_calls):
    def forbidden_generate(key, data, location):
        raise AssertionError("Generation was disabled")

    model = Model({"location": Real()}, lambda data, location: normal(location, 0.0, 1.0), forbidden_generate)
    result = sample(model, draws=2, warmup=3, chains=1, generate=False)
    assert set(result.children) == {"posterior", "sample_stats"}


@pytest.mark.parametrize("generate", [False, True])
def test_auxiliary_inputs_keep_evaluated_values_and_experiment_labels(nuts_calls, generate):
    _, data = _prepared_model()
    inputs = xr.Dataset(
        {
            "lift": ("experiment", np.array([0.1, 0.2, 0.3], dtype=np.float64)),
            "uncertainty": ("experiment", [0.5, 0.25, 0.125]),
            "reference": 2.0,
        },
        coords={"experiment": ["north", "south", "national"]},
    )
    model = Model(
        {"effect": Real(dims="experiment")},
        lambda lift, expected_lift, uncertainty, effect: (
            normal(lift, expected_lift, uncertainty) + normal(effect, 0.0, 1.0)
        ),
        lambda key, lift, expected_lift, uncertainty: {
            "lift_copy": lift,
            "pointwise": normal_logpdf(lift, expected_lift, uncertainty),
        },
        data=data,
        inputs=inputs,
        scaling=fit_data_scaling(data, scale_outcome=True),
        transformed_parameters=lambda effect, reference: {"expected_lift": effect * reference},
        save=("expected_lift",),
        predictive=("lift_copy",),
        log_likelihood=("pointwise",),
        generated_dims={"expected_lift": ("experiment",), "pointwise": ("experiment",)},
    )
    effect = np.array([1.0, 2.0, 3.0])
    result = sample(model, draws=2, warmup=1, chains=1, initial_values={"effect": effect}, generate=generate)

    assert result["posterior"]["effect"].dims == ("chain", "draw", "experiment")
    np.testing.assert_array_equal(result["posterior"]["experiment"], inputs.experiment)
    assert set(result["observed_data"].data_vars) == {"outcome"}
    np.testing.assert_allclose(result["observed_data"]["outcome"], model.data.values["outcome"])
    for name in inputs.data_vars:
        actual = result["constant_data"][name]
        assert actual.dims == inputs[name].dims
        assert actual.dtype == model.data.values[name].dtype
        np.testing.assert_array_equal(actual, model.data.values[name])
    np.testing.assert_array_equal(result["constant_data"]["time"], data.time_values)
    if generate:
        assert result["posterior_predictive"]["lift_copy"].dims == ("chain", "draw", "experiment")
        assert "time" not in result["posterior_predictive"].dims
        np.testing.assert_allclose(result["generated_quantities"]["expected_lift"], [[effect * 2, effect * 2]])
        expected = normal_logpdf(model.data.values["lift"], effect * 2, model.data.values["uncertainty"])
        np.testing.assert_allclose(result["log_likelihood"]["pointwise"], [[expected, expected]])
    else:
        assert "generated_quantities" not in result.children
        assert "posterior_predictive" not in result.children


def test_unspecified_vector_axes_receive_parameter_specific_names(nuts_calls):
    model = Model({"coefficient": Real((2,))}, lambda data, coefficient: normal(coefficient, 0.0, 1.0))
    result = sample(model, draws=3, warmup=5, chains=1)
    variable = result["posterior"]["coefficient"]
    assert variable.dims[:2] == ("chain", "draw")
    assert variable.dims[2].startswith("coefficient")
    assert variable.shape == (1, 3, 2)


def test_generated_parameter_aliases_keep_explicit_and_neutral_axes(nuts_calls):
    model = Model(
        {"coefficient": Real((2,)), "other": Real((2,))},
        lambda data, coefficient, other: normal(coefficient, 0.0, 1.0) + normal(other, 0.0, 1.0),
        lambda key, data, coefficient, other: {"copy": coefficient, "other_copy": other, "sum": coefficient + other},
        dims={"coefficient": ("feature",)},
        coords={"feature": ["first", "second"]},
    )
    result = sample(model, draws=2, warmup=3, chains=1)["generated_quantities"]
    assert result["copy"].dims == ("chain", "draw", "feature")
    assert result["other_copy"].dims == ("chain", "draw", "other_dim_0")
    assert result["sum"].dims == ("chain", "draw", "sum_dim_0")
    np.testing.assert_array_equal(result["feature"], ["first", "second"])


def test_custom_generation_does_not_infer_axes_from_names_or_equal_lengths(nuts_calls):
    data = prepare_data(
        pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0], "a": [5.0, 6.0], "b": [7.0, 8.0]}),
        time="week",
        outcome="sales",
        controls=["a", "b"],
        treatments=["a", "b"],
    )

    def generate(key, outcome, controls, treatments, coefficient):
        return {
            "controls_copy": controls,
            "treatments_copy": treatments,
            "controls": controls.T,
            "outcome": outcome[::-1],
            "mean": coefficient + outcome,
            "prediction": jnp.array([1.0, 2.0, 3.0]),
        }

    model = Model(
        {"coefficient": Real((2,))},
        lambda coefficient: normal(coefficient, 0.0, 1.0),
        generate,
        data=data,
        predictive=("prediction",),
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    generated = result["generated_quantities"]
    assert result["posterior"]["coefficient"].dims == ("chain", "draw", "coefficient_dim_0")
    assert generated["controls_copy"].dims == ("chain", "draw", "time", "control")
    assert generated["treatments_copy"].dims == ("chain", "draw", "time", "treatment")
    assert generated["controls"].dims == ("chain", "draw", "controls_dim_0", "controls_dim_1")
    assert generated["outcome"].dims == ("chain", "draw", "outcome_dim_0")
    assert generated["mean"].dims == ("chain", "draw", "mean_dim_0")
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "prediction_dim_0")


def test_generated_dimension_overrides_take_precedence(nuts_calls):
    data = prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales")
    model = Model(
        {"intercept": Real()},
        lambda intercept: normal(intercept, 0.0, 1.0),
        lambda key, outcome, intercept: {"copy": outcome, "prediction": outcome + intercept},
        data=data,
        generated_dims={"copy": ("custom",), "prediction": ("custom",)},
        coords={"custom": ["early", "late"]},
        predictive=("prediction",),
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    assert result["generated_quantities"]["copy"].dims == ("chain", "draw", "custom")
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "custom")
    np.testing.assert_array_equal(result["posterior_predictive"]["custom"], ["early", "late"])


def test_predictive_outputs_use_declared_observation_order(nuts_calls):
    data = prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales")
    model = Model(
        {"levels": Real((2,))},
        lambda levels: normal(levels, 0.0, 1.0),
        lambda key, levels: {"prediction": levels, "copy": levels},
        data=data,
        predictive=("prediction",),
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time")
    assert result["generated_quantities"]["copy"].dims == ("chain", "draw", "levels_dim_0")


def test_missing_selected_generated_output_is_reported(nuts_calls):
    model = Model(
        {"location": Real()},
        lambda data, location: normal(location, 0.0, 1.0),
        lambda key, data, location: {"mean": location},
        predictive=("prediction",),
    )
    with pytest.raises(ValueError, match="prediction"):
        sample(model, draws=2, warmup=3, chains=1)


@pytest.mark.parametrize("group_specific", [False, True])
def test_explicit_media_and_fourier_parameters_retain_declared_axes(nuts_calls, group_specific):
    data = prepare_data(
        pd.DataFrame(
            {
                "week": [1, 1, 2, 2, 3, 3],
                "region": ["west", "east"] * 3,
                "sales": [10.0, 15.0, 12.0, 18.0, 13.0, 17.0],
                "video": [100.0, 200.0, 120.0, 160.0, 180.0, 140.0],
                "search": [60.0, 80.0, 90.0, 70.0, 50.0, 90.0],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        channels=["Video", "Search"],
        media_history=pd.DataFrame(
            {"week": [0, 0], "region": ["west", "east"], "video": [80.0, 120.0], "search": [40.0, 50.0]}
        ),
    )

    def density(
        outcome,
        paid_total,
        annual,
        paid_coefficient,
        paid_retention,
        paid_half_saturation,
        paid_slope,
        annual_coefficients,
    ):
        target = normal(outcome, paid_total + annual, 10.0)
        target += half_normal(paid_coefficient, 1.0)
        target += beta(paid_retention, 2.0, 3.0)
        target += lognormal(paid_half_saturation, 0.0, 0.5)
        target += lognormal(paid_slope, 0.0, 0.5)
        target += normal(annual_coefficients, 0.0, 0.5)
        return target

    def generate(key, outcome, media, paid, paid_total, annual, annual_coefficients, paid_retention):
        mean = paid_total + annual
        return {
            "prediction": normal_rng(key, mean, 10.0),
            "pointwise": normal_logpdf(outcome, mean, 10.0),
            "contribution": paid,
            "total": paid_total,
            "seasonal": annual,
            "weights": annual_coefficients,
            "retention": paid_retention,
            "exposure": media,
        }

    def transformed(
        media,
        time,
        paid_coefficient,
        paid_retention,
        paid_half_saturation,
        paid_slope,
        annual_coefficients,
    ):
        carried = geometric_adstock(media, alpha=paid_retention, max_lag=1)
        response = hill_saturation(carried, half_saturation=paid_half_saturation, slope=paid_slope)[-time.shape[0] :]
        paid = response * paid_coefficient
        annual = fourier_features(time, period=52, order=2) @ annual_coefficients
        if not group_specific:
            annual = jnp.broadcast_to(annual[:, None], paid.shape[:2])
        return {"paid": paid, "paid_total": paid.sum(-1), "annual": annual}

    coefficient_axes = ("group", "channel") if group_specific else ("channel",)
    annual_axes = ("annual_mode", "group") if group_specific else ("annual_mode",)
    model = Model(
        {
            "paid_coefficient": Positive(dims=coefficient_axes),
            "paid_retention": Interval(0.0, 1.0, dims="channel"),
            "paid_half_saturation": Positive(dims="channel"),
            "paid_slope": Positive(dims="channel"),
            "annual_coefficients": Real(dims=annual_axes),
        },
        density,
        generate,
        data=data,
        transformed_parameters=transformed,
        coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
        generated_dims={
            "contribution": ("time", "group", "channel"),
            "total": ("time", "group"),
            "seasonal": ("time", "group"),
        },
        predictive=("prediction",),
        log_likelihood=("pointwise",),
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    posterior = result["posterior"]
    coefficient_axes = ("group", "channel") if group_specific else ("channel",)
    assert posterior["paid_coefficient"].dims == ("chain", "draw", *coefficient_axes)
    for name in ("paid_retention", "paid_half_saturation", "paid_slope"):
        assert posterior[name].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(posterior["channel"], ["Video", "Search"])
    modes = posterior["annual_coefficients"].dims[2]
    np.testing.assert_array_equal(posterior[modes], ["sin_1", "sin_2", "cos_1", "cos_2"])
    annual_axes = (modes, "group") if group_specific else (modes,)
    assert posterior["annual_coefficients"].dims == ("chain", "draw", *annual_axes)
    if group_specific:
        np.testing.assert_array_equal(posterior["group"], ["west", "east"])
    generated = result["generated_quantities"]
    assert generated["contribution"].dims == ("chain", "draw", "time", "group", "channel")
    for name in ("total", "seasonal"):
        assert generated[name].dims == ("chain", "draw", "time", "group")
    assert generated["weights"].dims == ("chain", "draw", *annual_axes)
    assert generated["retention"].dims == ("chain", "draw", "channel")
    assert generated["exposure"].dims == ("chain", "draw", "media_time", "group", "channel")
    np.testing.assert_array_equal(generated["media_time"], [0, 1, 2, 3])
    np.testing.assert_array_equal(generated["time"], [1, 2, 3])
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time", "group")
    assert result["log_likelihood"]["pointwise"].dims == ("chain", "draw", "time", "group")
    np.testing.assert_array_equal(generated["channel"], ["Video", "Search"])


def test_collected_data_uses_model_scaling_and_does_not_rescale_draws_or_generated_values(nuts_calls):
    data = prepare_data(
        pd.DataFrame({"time": [1, 2, 3], "sales": [100.0, 200.0, 300.0], "price": [10.0, 20.0, 40.0]}),
        time="time",
        outcome="sales",
        controls=["price"],
    )
    scaling = fit_data_scaling(data, scale_outcome=True)

    def density(outcome, controls, location):
        return normal(outcome, location + controls[..., 0], 1.0) + normal(location, 0.0, 1.0)

    def generate(key, outcome, location):
        return {"mean": jnp.full_like(outcome, location)}

    model = Model(
        {"location": Real()},
        density,
        generate,
        data=data,
        scaling=scaling,
        generated_dims={"mean": ("time",)},
    )
    expected_outcome = np.array(model.data.values["outcome"])
    expected_controls = np.array(model.data.values["controls"])
    assert not np.allclose(expected_outcome, data.arrays["outcome"])
    assert not np.allclose(expected_controls, data.arrays["controls"])
    data.arrays["outcome"][...] = -999.0
    data.arrays["controls"][...] = -999.0
    results = sample(model, draws=2, warmup=3, chains=1, initial_values={"location": 1.25})
    np.testing.assert_array_equal(results["observed_data"]["outcome"], expected_outcome)
    np.testing.assert_array_equal(results["constant_data"]["controls"], expected_controls)
    np.testing.assert_array_equal(results["posterior"]["location"], np.full((1, 2), 1.25))
    np.testing.assert_array_equal(results["generated_quantities"]["mean"], np.full((1, 2, 3), 1.25))
    assert results.attrs["data_scale"] == "model"


@pytest.mark.parametrize(
    "statistic,message",
    [("diverging", "1 divergent transition"), ("reached_max_treedepth", "max_tree_depth on 1 draw")],
)
def test_problematic_sampler_diagnostics_warn_without_dropping_draws(
    normal_model, nuts_calls, monkeypatch, statistic, message
):
    backend = sampling._sample_nuts

    def flagged_draws(*args, **kwargs):
        positions, statistics = backend(*args, **kwargs)
        statistics[statistic] = statistics[statistic].at[0, 0].set(True)
        return positions, statistics

    monkeypatch.setattr(sampling, "_sample_nuts", flagged_draws)
    with pytest.warns(RuntimeWarning, match=message):
        results = sample(normal_model, data=0.0, draws=3, warmup=4, chains=2)
    assert results["posterior"]["location"].shape == (2, 3)
    assert int(results["sample_stats"][statistic].sum()) == 1


def test_nonfinite_sampled_log_density_raises_instead_of_returning_results(normal_model, nuts_calls, monkeypatch):
    backend = sampling._sample_nuts

    def invalid_draws(*args, **kwargs):
        positions, statistics = backend(*args, **kwargs)
        statistics["lp"] = statistics["lp"].at[0, 0].set(-jnp.inf)
        return positions, statistics

    monkeypatch.setattr(sampling, "_sample_nuts", invalid_draws)
    with pytest.raises(RuntimeError, match="nonfinite log densities"):
        sample(normal_model, data=0.0, draws=3, warmup=4, chains=1)


@pytest.mark.parametrize("option", ["draws", "warmup", "chains", "max_tree_depth"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_counts_require_positive_integers_before_sampling(normal_model, nuts_calls, option, value):
    arguments = {"draws": 2, "warmup": 3, "chains": 1, option: value}
    with pytest.raises(ValueError, match=option):
        sample(normal_model, data=0.0, **arguments)
    assert not nuts_calls


@pytest.mark.parametrize("value", [0.0, 1.0, np.nan, np.inf, True, "0.8"])
def test_acceptance_target_must_be_a_probability(normal_model, nuts_calls, value):
    with pytest.raises(ValueError, match="target_accept"):
        sample(normal_model, data=0.0, target_accept=value, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_seed_requires_a_nonnegative_integer(normal_model, nuts_calls, value):
    with pytest.raises(ValueError, match="seed"):
        sample(normal_model, data=0.0, seed=value, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize(
    "initial", [{}, {"unexpected": 0.0}, {"location": [0.0, 1.0]}, {"location": np.nan}, {"location": np.inf}]
)
def test_initial_values_must_be_complete_finite_constrained_parameters(normal_model, nuts_calls, initial):
    with pytest.raises((ValueError, TypeError)):
        sample(normal_model, data=0.0, initial_values=initial, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_initial_positive_parameters_must_be_in_the_interior(nuts_calls, value):
    model = Model({"scale": Positive()}, lambda data, scale: half_normal(scale, 1.0))
    with pytest.raises(ValueError):
        sample(model, initial_values={"scale": value}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize(
    "weights", [[1.0, 1.0, 1.0], [0.2, 0.3, 0.4], [np.nan, 0.5, 0.5], [np.inf, 0.5, 0.5], [0.3], [np.nan], [np.inf]]
)
def test_initial_simplex_values_cannot_be_renormalized_or_discarded(nuts_calls, weights):
    model = Model(
        {"weights": Simplex((len(weights),)), "location": Real()},
        lambda data, weights, location: normal(weights, 0.0, 1.0) + normal(location, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="weights"):
        sample(model, initial_values={"weights": weights, "location": 0.0}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def test_valid_single_category_simplex_is_kept_with_other_free_parameters(nuts_calls):
    model = Model(
        {"weights": Simplex((1,)), "location": Real()},
        lambda data, weights, location: normal(weights, 0.0, 1.0) + normal(location, 0.0, 1.0),
    )
    results = sample(model, initial_values={"weights": [1.0], "location": 0.0}, draws=2, warmup=3, chains=1)
    assert nuts_calls[0]["positions"]["weights"].shape == (1, 0)
    np.testing.assert_array_equal(results["posterior"]["weights"], np.ones((1, 2, 1)))


@pytest.mark.parametrize("failure", ["density", "gradient"])
def test_nonfinite_initial_density_or_gradient_is_rejected_before_sampling(nuts_calls, failure):
    def density(data, location):
        return jnp.log(-jnp.square(location) - 1.0) if failure == "density" else -jnp.sqrt(jnp.abs(location))

    model = Model({"location": Real()}, density)
    with pytest.raises(ValueError, match=r"finite|density|gradient"):
        sample(model, initial_values={"location": 0.0}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def test_prepared_models_do_not_accept_separate_sampling_data(nuts_calls):
    model, data = _prepared_model()
    with pytest.raises(ValueError, match="data"):
        sample(model, data=data, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def _saved_model(**options):
    _, data = _prepared_model()

    def transformed(controls, intercept):
        return {"mu": intercept + controls[:, 0], "unused": intercept**2}

    def density(outcome, mu, intercept):
        return normal(outcome, mu, 10.0) + normal(intercept, 0.0, 1.0)

    settings = {"save": ("mu",), "generated_dims": {"mu": ("time",)}} | options
    return Model(
        {"intercept": Real()},
        density,
        data=data,
        transformed_parameters=transformed,
        **settings,
    )


def test_sampling_collects_selected_transformed_quantities_without_callback(nuts_calls):
    model = _saved_model()
    options = {"draws": 3, "warmup": 4, "chains": 2, "seed": 9, "initial_values": {"intercept": 2.0}}
    results = sample(model, **options)
    disabled = sample(model, **options, generate=False)

    assert set(results["posterior"].data_vars) == {"intercept"}
    assert set(results["generated_quantities"].data_vars) == {"mu"}
    assert results["generated_quantities"]["mu"].dims == ("chain", "draw", "time")
    np.testing.assert_allclose(results["generated_quantities"]["mu"], np.broadcast_to([7.0, 6.0, 5.0], (2, 3, 3)))
    np.testing.assert_array_equal(results["generated_quantities"].coords["time"], [10, 11, 12])
    xr.testing.assert_identical(results["posterior"], disabled["posterior"])
    xr.testing.assert_identical(results["sample_stats"], disabled["sample_stats"])
    assert "generated_quantities" not in disabled
    assert len(nuts_calls) == 2


@pytest.mark.parametrize("problem", ["missing", "collision", "dimensions"])
def test_saved_quantities_are_validated_before_chain_adaptation(nuts_calls, problem):
    if problem == "missing":
        model = _saved_model(save=("unknown",))
        message = "unknown"
    elif problem == "collision":
        model = _saved_model(generate=lambda key, mu: {"mu": mu})
        message = "also returned by generate"
    else:
        model = _saved_model(generated_dims={"mu": ()})
        message = "dims matching"

    with pytest.raises(ValueError, match=message):
        sample(model, draws=2, warmup=3, chains=1)
    assert not nuts_calls
