"""Tests for posterior scenarios with fitted components and labeled inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

from mmmjax import (
    FourierSeasonality,
    MediaEffect,
    Model,
    Positive,
    Real,
    fit_data_scaling,
    generate_quantities,
    normal,
    normal_logpdf,
    normal_rng,
    prepare_data,
)
from mmmjax._results import _collect_results


def _data(media, *, start=0, reverse=False, observed=True):
    rows = []
    for index, exposure in enumerate(media):
        for group in [1, 0] if reverse else [0, 1]:
            rows.append(
                {
                    "week": start + index,
                    "region": ("west", "east")[group],
                    "video": exposure[group, 0],
                    "search": exposure[group, 1],
                    "sales": 100.0 + 5 * (start + index) + 20 * group,
                    "temperature": 10.0 + start + index + group,
                    "price": 20.0 + 2 * (start + index) - group,
                }
            )
    channels = ["search", "video"] if reverse else ["video", "search"]
    controls = ["price", "temperature"] if reverse else ["temperature", "price"]
    return prepare_data(
        pd.DataFrame(rows[4:]),
        time="week",
        groups=["region"],
        outcome="sales" if observed else None,
        media=channels,
        controls=controls,
        media_history=pd.DataFrame(rows[:4]),
    )


def _media(periods):
    values = np.arange(1, 1 + periods * 4, dtype=float).reshape(periods, 2, 2)
    return values * np.array([2.0, 3.0])


def _model(data):
    def transformed(paid_media_total, annual, controls, intercept):
        return {"mean": intercept + paid_media_total + annual + 0.1 * controls[..., 0]}

    def density(outcome, mean, intercept, sigma):
        return normal(outcome, mean, sigma) + normal(intercept, 0.0, 1.0)

    def generated(key, mean, paid_media, annual, controls, sigma):
        return {
            "mean": mean,
            "contribution": paid_media,
            "seasonal": annual,
            "control_values": controls,
            "prediction": normal_rng(key, mean, sigma),
        }

    return Model(
        {"intercept": Real(), "sigma": Positive()},
        density,
        generated,
        data=data,
        components=[
            MediaEffect(max_lag=2, group_specific_coefficients=True),
            FourierSeasonality(period=8, order=1, name="annual", group_specific_coefficients=True),
        ],
        transformed_parameters=transformed,
        scaling=fit_data_scaling(data, scale_outcome=True),
        generated_dims={"mean": ("time", "group")},
        predictive=("prediction",),
    )


def _results(model, data):
    parameters = {
        "intercept": np.array(0.25),
        "sigma": np.array(0.7),
        "paid_media_coefficient": np.array([[0.7, 1.1], [0.5, 1.3]]),
        "paid_media_retention": np.array([0.25, 0.6]),
        "paid_media_half_saturation": np.array([1.5, 2.5]),
        "paid_media_slope": np.array([1.2, 0.8]),
        "annual": np.array([[0.3, -0.2], [0.7, 1.1]]),
    }
    posterior = {
        name: np.broadcast_to(value, (2, 3, *value.shape)).astype(jax.dtypes.canonicalize_dtype(float))
        for name, value in parameters.items()
    }
    posterior["intercept"] += np.arange(6).reshape(2, 3) / 10
    assert set(posterior) == set(model.parameters)
    dimensions = {
        "paid_media_coefficient": ("group", "channel"),
        "paid_media_retention": ("channel",),
        "paid_media_half_saturation": ("channel",),
        "paid_media_slope": ("channel",),
        "annual": ("annual_mode", "group"),
    }
    result = _collect_results(
        posterior,
        data=model.scaling.transform(data),
        dims=dimensions,
        coords={"annual_mode": ["sin_1", "cos_1"]},
        sample_stats={"diverging": np.zeros((2, 3), dtype=bool)},
        generated_quantities={"obsolete": np.full((2, 3), -999.0)},
    )
    result.attrs.update(inference_library="blackjax", inference_method="nuts", warmup_steps=500)
    return result


def _assert_direct_quantities(model, results, scenario, generated):
    inputs = model.prepare_data(scenario)
    for chain in range(2):
        for draw in range(3):
            parameters = {
                name: jnp.asarray(variable.values[chain, draw])
                for name, variable in results["posterior"].data_vars.items()
            }
            expected = model.generate(jax.random.key(0), parameters, inputs)
            for name in ("mean", "contribution", "seasonal", "control_values"):
                np.testing.assert_allclose(
                    generated["generated_quantities"][name].values[chain, draw],
                    expected[name],
                    rtol=4e-6,
                    atol=3e-6,
                )


def _frame(periods=4):
    rows = []
    for index, exposure in enumerate(_media(periods)):
        for group, region in enumerate(("west", "east")):
            rows.append(
                {
                    "week": index,
                    "region": region,
                    "sales": 100.0 + 5 * index + 20 * group,
                    "video": exposure[group, 0],
                    "search": exposure[group, 1],
                    "video_cost": exposure[group, 0] / 10,
                    "search_cost": exposure[group, 1] / 5,
                    "temperature": 10.0 + index + group,
                    "price": 20.0 + 2 * index - group,
                }
            )
    return pd.DataFrame(rows)


def _prepare_frame(frame, **kwargs):
    return prepare_data(
        frame,
        time="week",
        groups=["region"],
        outcome="sales" if "sales" in frame.columns else None,
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Online video", "Paid search"],
        controls=["temperature", "price"],
        **kwargs,
    )


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_dataframe_scenario_matches_explicit_preparation_with_saved_selections(backend):
    training = _prepare_frame(_frame())
    model = _model(training)
    results = _results(model, training)
    scenario = _frame(5).drop(columns="sales").iloc[::-1, ::-1].reset_index(drop=True)
    scenario["video"] *= 1.2
    scenario["video_cost"] *= 1.2
    scenario["unused_column"] = -1.0
    original = scenario.copy(deep=True)
    prepared = _prepare_frame(scenario)
    if backend == "polars":
        supplied = pl.from_pandas(scenario)
    elif backend == "pyarrow":
        supplied = pa.Table.from_pandas(scenario, preserve_index=False)
    else:
        supplied = scenario

    actual = generate_quantities(model, results, new_data=supplied, seed=11)
    expected = generate_quantities(model, results, new_data=prepared, seed=11)

    xr.testing.assert_identical(actual, expected)
    assert "observed_data" not in actual.children
    np.testing.assert_array_equal(actual["constant_data"]["channel"], ["Online video", "Paid search"])
    np.testing.assert_array_equal(actual["posterior_predictive"]["group"], ["west", "east"])
    np.testing.assert_array_equal(actual["posterior_predictive"]["time"], np.arange(5))
    pd.testing.assert_frame_equal(scenario, original)


@pytest.mark.parametrize("missing", ["search", "price"])
def test_dataframe_scenario_rejects_partially_missing_selected_roles(missing):
    frame = _frame()
    training = _prepare_frame(frame)
    model = _model(training)
    results = _results(model, training)

    with pytest.raises(ValueError, match=missing):
        generate_quantities(model, results, new_data=frame.drop(columns=missing))


def test_dataframe_scenario_does_not_infer_exposure_from_spend():
    frame = _frame()
    training = _prepare_frame(frame)
    model = _model(training)
    results = _results(model, training)
    higher_spend = frame.copy(deep=True)
    higher_spend["video_cost"] *= 1.2
    higher_exposure = higher_spend.copy(deep=True)
    higher_exposure["video"] *= 1.2

    baseline = generate_quantities(model, results, new_data=frame, seed=6)
    cost_only = generate_quantities(model, results, new_data=higher_spend, seed=6)
    exposure_changed = generate_quantities(model, results, new_data=higher_exposure, seed=6)

    xr.testing.assert_identical(cost_only["generated_quantities"], baseline["generated_quantities"])
    xr.testing.assert_identical(cost_only["posterior_predictive"], baseline["posterior_predictive"])
    np.testing.assert_array_equal(cost_only["constant_data"]["media"], baseline["constant_data"]["media"])
    np.testing.assert_allclose(
        cost_only["constant_data"]["spend"].values[..., 0], frame["video_cost"].to_numpy().reshape(4, 2) * 1.2
    )
    assert np.all(
        exposure_changed["generated_quantities"]["mean"].values > baseline["generated_quantities"]["mean"].values
    )
    np.testing.assert_array_equal(
        exposure_changed["generated_quantities"]["contribution"].values[..., 1],
        baseline["generated_quantities"]["contribution"].values[..., 1],
    )


def test_dataframe_scenario_does_not_reuse_original_media_history():
    frame = _frame(5)
    observations = frame.iloc[4:].copy()
    training = _prepare_frame(observations, media_history=frame.iloc[:4])
    model = _model(training)
    results = _results(model, training)

    original = generate_quantities(model, results, seed=15)
    raw = generate_quantities(model, results, new_data=observations, seed=15)
    explicit = generate_quantities(model, results, new_data=_prepare_frame(observations), seed=15)

    xr.testing.assert_identical(raw, explicit)
    np.testing.assert_array_equal(raw["constant_data"]["time"], [2, 3, 4])
    assert raw["constant_data"]["media"].shape[0] == 3
    assert original["constant_data"]["media"].shape[0] == 5
    assert not np.allclose(
        raw["generated_quantities"]["mean"].values[:, :, :2], original["generated_quantities"]["mean"].values[:, :, :2]
    )
    np.testing.assert_allclose(
        raw["generated_quantities"]["mean"].values[:, :, 2:],
        original["generated_quantities"]["mean"].values[:, :, 2:],
        rtol=3e-6,
    )


def test_dataframe_scenario_preserves_additional_data_roles_and_channel_labels():
    frame = pd.DataFrame(
        {
            "week": [1, 2, 3],
            "sales": [100.0, 110.0, 120.0],
            "email_clicks": [20.0, 25.0, 30.0],
            "video_reach": [1000.0, 2000.0, 1500.0],
            "video_frequency": [2.0, 3.0, 4.0],
            "video_cost": [100.0, 200.0, 150.0],
            "social_reach": [50.0, 70.0, 60.0],
            "social_frequency": [1.0, 2.0, 1.0],
            "price": [10.0, 9.0, 11.0],
            "residents": [10000.0] * 3,
        }
    )
    selections = {
        "time": "week",
        "outcome": "sales",
        "population": "residents",
        "organic_media": ["email_clicks"],
        "organic_channels": ["Email"],
        "reach": ["video_reach"],
        "media_frequency": ["video_frequency"],
        "rf_spend": ["video_cost"],
        "rf_channels": ["Streaming video"],
        "organic_reach": ["social_reach"],
        "organic_frequency": ["social_frequency"],
        "organic_rf_channels": ["Organic social"],
        "treatments": ["price"],
    }
    training = prepare_data(frame, **selections)

    def density(outcome, intercept):
        return normal(outcome, intercept, 1.0)

    def generated(key, organic_media, reach, media_frequency, organic_reach, organic_frequency, treatments, population):
        return {
            "email": organic_media,
            "video_exposure": reach * media_frequency,
            "social_exposure": organic_reach * organic_frequency,
            "prices": treatments,
            "population_copy": population,
        }

    model = Model({"intercept": Real()}, density, generated, data=training, components=[], scaling=None)
    results = _collect_results({"intercept": jnp.array([[100.0, 110.0]])}, data=training)
    scenario = frame.iloc[::-1, ::-1].copy()
    scenario["video_frequency"] += 1
    actual = generate_quantities(model, results, new_data=scenario)
    expected = generate_quantities(model, results, new_data=prepare_data(scenario, **selections))

    xr.testing.assert_identical(actual, expected)
    np.testing.assert_array_equal(actual["constant_data"]["organic_channel"], ["Email"])
    np.testing.assert_array_equal(actual["constant_data"]["rf_channel"], ["Streaming video"])
    np.testing.assert_array_equal(actual["constant_data"]["organic_rf_channel"], ["Organic social"])


def test_grouped_scenario_reuses_training_scaling_and_preserves_original_objects():
    training = _data(_media(5))
    model = _model(training)
    results = _results(model, training)
    scenario_media = _media(6) * 4
    scenario = _data(scenario_media, start=10, reverse=True)
    results_before = results.copy(deep=True)
    training_before = {name: value.copy() for name, value in training.arrays.items()}
    scenario_before = {name: value.copy() for name, value in scenario.arrays.items()}
    model_before = {name: np.array(value) for name, value in model.data.values.items()}
    scaling = model.scaling

    generated = generate_quantities(model, results, new_data=scenario, seed=17)
    ordered = generate_quantities(model, results, new_data=_data(scenario_media, start=10), seed=17)
    xr.testing.assert_identical(generated, ordered)
    assert set(generated.children) == {
        "posterior",
        "posterior_predictive",
        "generated_quantities",
        "observed_data",
        "constant_data",
    }
    assert not {"inference_library", "inference_method", "warmup_steps"} & generated.attrs.keys()
    xr.testing.assert_identical(generated["posterior"].to_dataset(), results["posterior"].to_dataset())
    np.testing.assert_array_equal(generated["posterior"]["group"], ["west", "east"])
    np.testing.assert_array_equal(generated["posterior"]["channel"], ["video", "search"])
    np.testing.assert_array_equal(generated["generated_quantities"]["time"], [12, 13, 14, 15])
    np.testing.assert_array_equal(generated["constant_data"]["media_time"], [10, 11, 12, 13, 14, 15])
    assert generated["generated_quantities"]["contribution"].dims == ("chain", "draw", "time", "group", "channel")
    assert generated["generated_quantities"]["seasonal"].dims == ("chain", "draw", "time", "group")
    expected_data = scaling.transform(scenario)
    for name, expected in expected_data.arrays.items():
        group = "observed_data" if name == "outcome" else "constant_data"
        np.testing.assert_allclose(generated[group][name], expected, rtol=2e-6)
    refitted = fit_data_scaling(scenario, scale_outcome=True).transform(scenario)
    assert not np.allclose(generated["constant_data"]["media"], refitted.arrays["media"][:, ::-1, ::-1])
    _assert_direct_quantities(model, results, scenario, generated)

    xr.testing.assert_identical(results, results_before)
    assert model.scaling is scaling
    assert scenario.group_values == (("east",), ("west",))
    assert scenario.channels == ("search", "video")
    for name, expected in scenario_before.items():
        np.testing.assert_array_equal(scenario.arrays[name], expected)
    for name, expected in training_before.items():
        np.testing.assert_array_equal(training.arrays[name], expected)
    for name, expected in model_before.items():
        np.testing.assert_array_equal(model.data.values[name], expected)

    generated["posterior"]["intercept"].values[:] = -999.0
    generated["constant_data"]["media"].values[:] = -999.0
    xr.testing.assert_identical(results, results_before)
    np.testing.assert_array_equal(model.data.values["media"], model_before["media"])
    np.testing.assert_array_equal(scenario.arrays["media"], scenario_before["media"])


def test_explicit_scenario_history_changes_only_affected_periods_and_is_retained():
    training = _data(_media(5))
    model = _model(training)
    results = _results(model, training)
    baseline_media = _media(6)
    changed_media = baseline_media.copy()
    changed_media[:2] *= 10
    baseline = generate_quantities(model, results, new_data=_data(baseline_media, start=20), seed=9)
    scenario = _data(changed_media, start=20)
    changed = generate_quantities(model, results, new_data=scenario, seed=9)

    baseline_paid = baseline["generated_quantities"]["contribution"].values
    changed_paid = changed["generated_quantities"]["contribution"].values
    assert np.all(changed_paid[:, :, :2] > baseline_paid[:, :, :2])
    np.testing.assert_allclose(changed_paid[:, :, 2:], baseline_paid[:, :, 2:], rtol=2e-6)
    np.testing.assert_array_equal(changed["generated_quantities"]["time"], [22, 23, 24, 25])
    np.testing.assert_array_equal(changed["constant_data"]["media_time"], [20, 21, 22, 23, 24, 25])
    np.testing.assert_allclose(changed["constant_data"]["media"], model.scaling.transform(scenario).arrays["media"])
    np.testing.assert_array_equal(changed["observed_data"]["outcome"], baseline["observed_data"]["outcome"])
    _assert_direct_quantities(model, results, scenario, changed)


def test_outcome_free_scenario_infers_predictive_time_and_group_axes():
    training = _data(_media(5))
    model = _model(training)
    results = _results(model, training)
    future = _data(_media(6), start=30, reverse=True, observed=False)

    generated = generate_quantities(model, results, new_data=future, seed=12)

    assert "outcome" not in future.arrays
    assert "observed_data" not in generated.children
    assert "log_likelihood" not in generated.children
    prediction = generated["posterior_predictive"]["prediction"]
    assert prediction.dims == ("chain", "draw", "time", "group")
    assert prediction.shape == (2, 3, 4, 2)
    np.testing.assert_array_equal(prediction["time"], [32, 33, 34, 35])
    np.testing.assert_array_equal(prediction["group"], ["west", "east"])
    assert np.isfinite(prediction).all()
    _assert_direct_quantities(model, results, future, generated)


@pytest.mark.parametrize("required_by", ["generate", "transformed_parameters"])
def test_missing_outcome_is_rejected_when_any_evaluated_callback_requires_it(required_by):
    training = _data(_media(5))

    def transform(outcome, intercept):
        return {"unused_residual": outcome - intercept}

    def density(outcome, intercept):
        return normal(outcome, intercept, 1.0)

    def generate(key, outcome, intercept):
        return {"pointwise": normal_logpdf(outcome, intercept, 1.0)}

    def independent_generate(key, controls, intercept):
        return {"prediction": controls[..., 0] + intercept}

    model = Model(
        {"intercept": Real()},
        density,
        generate if required_by == "generate" else independent_generate,
        data=training,
        components=[],
        transformed_parameters=transform if required_by == "transformed_parameters" else None,
        log_likelihood=("pointwise",) if required_by == "generate" else (),
        predictive=("prediction",) if required_by == "transformed_parameters" else (),
    )
    results = _collect_results({"intercept": jnp.ones((1, 2))}, data=training)
    scenario = _data(_media(6), start=40, observed=False)

    with pytest.raises(ValueError, match=rf"{required_by}.*outcome"):
        generate_quantities(model, results, new_data=scenario)


def test_default_data_uses_stored_inputs_and_recomputes_likelihood_outputs():
    training = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [1.0, 2.0, 4.0], "price": [10.0, 20.0, 30.0]}),
        time="week",
        outcome="sales",
        controls=["price"],
    )

    def density(outcome, intercept):
        return normal(outcome, intercept, 1.0)

    def generated(key, outcome, intercept):
        return {"pointwise": normal_logpdf(outcome, intercept, 1.0), "outcome_copy": outcome}

    model = Model(
        {"intercept": Real()},
        density,
        generated,
        data=training,
        components=[],
        log_likelihood=("pointwise",),
    )
    results = _collect_results(
        {"intercept": jnp.array([[0.5, 1.5]])},
        data=training,
        log_likelihood={name: np.full((1, 2, 3), -999.0) for name in ("pointwise", "obsolete")},
        generated_dims={name: ("time",) for name in ("pointwise", "obsolete")},
        sample_stats={"diverging": np.ones((1, 2), dtype=bool)},
    )
    original = results.copy(deep=True)
    training.arrays["outcome"][:] = -100.0
    training.arrays["controls"][:] = -100.0

    regenerated = generate_quantities(model, results, new_data=None, seed=4)

    assert "sample_stats" not in regenerated.children
    assert set(regenerated["log_likelihood"].data_vars) == {"pointwise"}
    np.testing.assert_array_equal(regenerated["observed_data"]["outcome"], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(regenerated["constant_data"]["controls"], [[10.0], [20.0], [30.0]])
    np.testing.assert_array_equal(regenerated["log_likelihood"]["time"], [10, 11, 12])
    expected = normal_logpdf(jnp.array([1.0, 2.0, 4.0]), jnp.array([[[0.5], [1.5]]]), 1.0)
    np.testing.assert_allclose(regenerated["log_likelihood"]["pointwise"], expected, rtol=2e-6)
    np.testing.assert_array_equal(
        regenerated["generated_quantities"]["outcome_copy"], np.broadcast_to([1.0, 2.0, 4.0], (1, 2, 3))
    )
    xr.testing.assert_identical(results, original)


@pytest.mark.parametrize(
    ("coordinate", "mutation"),
    [
        (None, None),
        ("group_region", "changed"),
        ("group_store", "changed"),
        ("group_region", "missing"),
        ("group_store", "missing"),
    ],
)
def test_multicolumn_group_posterior_requires_matching_auxiliary_identities(coordinate, mutation):
    training = prepare_data(
        pd.DataFrame(
            {
                "week": [1, 1, 2, 2],
                "region": ["west", "east"] * 2,
                "store": ["mall", "outlet"] * 2,
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        time="week",
        groups=["region", "store"],
        outcome="sales",
    )

    def density(outcome, intercept):
        return normal(outcome, intercept, 1.0)

    def generated(key, outcome, intercept):
        return {"mean": jnp.broadcast_to(intercept, outcome.shape)}

    model = Model(
        {"intercept": Real((2,))},
        density,
        generated,
        data=training,
        components=[],
        dims={"intercept": ("group",)},
        generated_dims={"mean": ("time", "group")},
    )
    results = _collect_results(
        {"intercept": jnp.array([[[0.2, 1.2], [0.4, 1.4]]])}, data=training, dims={"intercept": ("group",)}
    )
    if coordinate is not None:
        posterior = results["posterior"].to_dataset()
        if mutation == "changed":
            posterior = posterior.assign_coords({coordinate: ("group", posterior[coordinate].values[::-1])})
        else:
            posterior = posterior.drop_vars(coordinate)
        results["posterior"] = posterior
        np.testing.assert_array_equal(results["posterior"]["group"], [0, 1])
        with pytest.raises(ValueError, match=coordinate):
            generate_quantities(model, results)
    else:
        regenerated = generate_quantities(model, results)
        xr.testing.assert_identical(regenerated["posterior"].to_dataset(), results["posterior"].to_dataset())
        for group in ("posterior", "generated_quantities", "observed_data"):
            np.testing.assert_array_equal(regenerated[group]["group_region"], ["west", "east"])
            np.testing.assert_array_equal(regenerated[group]["group_store"], ["mall", "outlet"])
        np.testing.assert_allclose(regenerated["generated_quantities"]["mean"].values[0, 0], [[0.2, 1.2]] * 2)
