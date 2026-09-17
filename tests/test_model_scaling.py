"""Tests for fitted scaling at the model data boundary."""

from dataclasses import replace
from datetime import date, timedelta

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mmmjax import Data, Model, Real, fit_data_scaling, generate_quantities, normal, prepare_data
from mmmjax._results import _collect_results


def _data(*, multiplier=1.0, reverse=False, outcome=True):
    frame = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "sales": [10, 20, 30],
            "video": np.array([0.0, 100.0, 300.0]) * multiplier,
            "search": np.array([40.0, 80.0, 120.0]) * multiplier,
            "video_cost": [0.0, 2.0, 6.0],
            "search_cost": [1.0, 2.0, 3.0],
            "temperature": [10.0, 15.0, 20.0],
            "price": [3.0, 2.0, 1.0],
        }
    )
    media = ["search", "video"] if reverse else ["video", "search"]
    return prepare_data(
        frame,
        time="time",
        outcome="sales" if outcome else None,
        media=media,
        spend=[f"{channel}_cost" for channel in media],
        controls=["temperature"],
        treatments=["price"],
    )


def _model(data, *, scaling=None):
    def density(*, outcome, media, controls, treatments, intercept):
        mu = intercept + media.sum(-1) + controls[..., 0] + treatments[..., 0]
        return normal(outcome, mu, 2.0) + normal(intercept, 0.0, 2.0)

    def generate(key, *, media, controls, treatments, spend, intercept):
        return {
            "media": media,
            "controls": controls,
            "treatments": treatments,
            "spend": spend,
            "mu": intercept + media.sum(-1) + controls[..., 0] + treatments[..., 0],
        }

    return Model({"intercept": Real()}, density, generate, data=Data(data, scaling=scaling))


def _population_data(*, start=1, reverse=False, include_population=True):
    groups = [("west", 10.0), ("east", 20.0)]
    if reverse:
        groups.reverse()
    rows = [
        {
            "time": time,
            "region": region,
            "sales": population * (time + (region == "east")),
            "residents": population,
            "video": 10.0 * time,
            "search": 20.0 * time,
        }
        for time in range(start, start + 3)
        for region, population in groups
    ]
    return prepare_data(
        pl.DataFrame(rows),
        time="time",
        groups=["region"],
        outcome="sales",
        population="residents" if include_population else None,
        media=["search", "video"] if reverse else ["video", "search"],
    )


def _population_model(data, *, scaling=None):
    return Model(
        {"intercept": Real()},
        lambda outcome, intercept: normal(outcome, intercept, 1.0) + normal(intercept, 0.0, 1.0),
        lambda key, outcome, intercept: {"prediction": jnp.full_like(outcome, intercept)},
        data=Data(data, scaling=scaling),
        predictive=("prediction",),
    )


def _reference_data(*, grouped=True, start=1, periods=3, observed=True, reverse=False, history=False):
    groups = [("west", 10.0), ("east", 20.0)] if grouped else [("national", 10.0)]
    if reverse:
        groups.reverse()
    rows = [
        {
            "time": time,
            "region": region,
            "sales": population * (time + (region == "east")),
            "residents": population,
            "video": 10.0 * (time + 1),
            "search": 20.0 * (time + 2),
            "video_cost": 2.0 * (time + 1),
            "search_cost": 3.0 * (time + 2),
        }
        for time in range(start - int(history), start + periods)
        for region, population in groups
    ]
    frame = pl.DataFrame(rows)
    channels = ["search", "video"] if reverse else ["video", "search"]
    return prepare_data(
        frame.filter(pl.col("time") >= start),
        time="time",
        groups=["region"] if grouped else (),
        outcome="sales" if observed else None,
        population="residents",
        media=channels,
        spend=[f"{channel}_cost" for channel in channels],
        media_history=frame.filter(pl.col("time") < start) if history else None,
    )


def test_reference_roles_are_scaled_snapshots_with_raw_spend_and_observation_period_counts():
    raw = _reference_data(history=True)
    scaling = fit_data_scaling(raw, scale_outcome="population", adjust_population=True)
    expected = scaling.transform(raw)
    original_spend = raw.arrays["spend"].copy()

    def generate(key, media, spend, reference, n_periods):
        return {
            "current_media": media,
            "current_spend": spend,
            "baseline_media": reference.media,
            "baseline_spend": reference.spend,
            "baseline_outcome": reference.outcome,
            "baseline_population": reference.population,
            "baseline_time": reference.time,
            "baseline_media_time": reference.media_time,
            "current_window": jnp.ones((n_periods,)),
            "reference_window": jnp.ones((reference.n_periods,)),
        }

    model = Model({}, lambda reference: jnp.sum(reference.outcome), generate, data=Data(raw, scaling=scaling))
    raw.arrays["media"][...] = -1
    raw.arrays["spend"][...] = -1
    raw.arrays["outcome"][...] = -1
    edited_bundle = model.data
    for role in ("media", "spend", "outcome", "population"):
        edited_bundle.values[role] = jnp.zeros_like(edited_bundle.values[role])
    edited = model.generate_quantities(jax.random.key(0), {}, edited_bundle)
    np.testing.assert_array_equal(edited["current_media"], 0.0)
    np.testing.assert_array_equal(edited["current_spend"], 0.0)
    for role in ("media", "spend", "outcome", "population"):
        np.testing.assert_array_equal(edited[f"baseline_{role}"], expected.arrays[role])
    edited_bundle.reference.values["media"] = jnp.zeros_like(edited_bundle.reference.media)

    generate_compiled = jax.jit(model.generate_quantities)
    for periods in (1, 5):
        future = _reference_data(start=5, periods=periods, observed=False, reverse=True, history=True)
        canonical = _reference_data(start=5, periods=periods, observed=False, history=True)
        current = scaling.transform(canonical)
        prepared = model.prepare_data(future)
        actual = generate_compiled(jax.random.key(0), {}, prepared)
        assert "outcome" not in prepared.values
        for role in ("media", "spend", "outcome", "population"):
            np.testing.assert_array_equal(actual[f"baseline_{role}"], expected.arrays[role])
        np.testing.assert_array_equal(actual["baseline_spend"], original_spend)
        np.testing.assert_array_equal(actual["baseline_time"], [0, 1, 2])
        np.testing.assert_array_equal(actual["baseline_media_time"], [-1, 0, 1, 2])
        np.testing.assert_allclose(actual["current_media"], current.arrays["media"], rtol=1e-6)
        np.testing.assert_array_equal(actual["current_spend"], canonical.arrays["spend"])
        assert actual["baseline_media"].shape == (4, 2, 2)
        assert actual["baseline_spend"].shape == (3, 2, 2)
        assert actual["current_window"].shape == (periods,)
        assert actual["reference_window"].shape == (3,)
        np.testing.assert_allclose(
            jax.jit(model.log_density)({}, prepared), expected.arrays["outcome"].sum(), atol=1e-6
        )


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", [None, False, True, "population"])
def test_outcome_helpers_restore_levels_and_expose_only_marginal_scales(grouped, mode):
    raw = _reference_data(grouped=grouped)
    scaling = None if mode is None else fit_data_scaling(raw, scale_outcome=mode)

    def density(outcome, intercept, outcome_scaling):
        prediction = jnp.full_like(outcome, intercept)
        raw_outcome = outcome_scaling.inverse_transform(outcome)
        raw_prediction = outcome_scaling.inverse_transform(prediction)
        return normal(raw_outcome, raw_prediction, outcome_scaling.scale) + normal(intercept, 0.0, 2.0)

    def generate(key, intercept, reference, outcome_scaling, n_periods):
        prediction = jnp.full((n_periods, *reference.outcome.shape[1:]), intercept)
        return {
            "prediction": prediction,
            "raw_prediction": outcome_scaling.inverse_transform(prediction),
            "raw_reference": outcome_scaling.inverse_transform(reference.outcome),
            "batched_raw_prediction": outcome_scaling.inverse_transform(jnp.stack((prediction, prediction + 1))),
            "unit_scale": outcome_scaling.scale,
        }

    model = Model({"intercept": Real()}, density, generate, data=Data(raw, scaling=scaling))
    transformation = None if scaling is None else scaling.transformations.get("outcome")
    if transformation is None:
        expected_scale, expected_offset = np.asarray(1.0), np.asarray(0.0)
    else:
        expected_scale = (
            np.asarray(transformation.scale).reshape(-1)
            if grouped and mode == "population"
            else np.asarray(transformation.scale).reshape(())
        )
        expected_offset = (
            np.asarray(transformation.offset).reshape(-1)
            if grouped and mode == "population"
            else np.asarray(transformation.offset).reshape(())
        )
    position = {"intercept": jnp.array(0.3)}
    generated = jax.jit(model.generate_quantities)(jax.random.key(0), position, model.data)
    assert generated["unit_scale"].shape == ((2,) if grouped and mode == "population" else ())
    np.testing.assert_allclose(generated["unit_scale"], expected_scale, rtol=1e-6)
    np.testing.assert_allclose(generated["raw_reference"], raw.arrays["outcome"], rtol=1e-6)
    expected_prediction = np.broadcast_to(0.3 * expected_scale + expected_offset, generated["raw_prediction"].shape)
    np.testing.assert_allclose(generated["raw_prediction"], expected_prediction, rtol=1e-6)
    np.testing.assert_allclose(
        generated["batched_raw_prediction"],
        np.stack((expected_prediction, expected_prediction + expected_scale)),
        rtol=1e-6,
    )

    def expected_density(intercept):
        return normal(raw.arrays["outcome"], intercept * expected_scale + expected_offset, expected_scale) + normal(
            intercept, 0.0, 2.0
        )

    actual_value, actual_gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected_value, expected_gradient = jax.value_and_grad(expected_density)(position["intercept"])
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(actual_gradient["intercept"], expected_gradient, rtol=1e-6, atol=2e-6)

    draws = {"intercept": jnp.array([-0.2, 0.7])}
    generate_draws = jax.jit(jax.vmap(model.generate_quantities, in_axes=(0, 0, None)))
    for periods in (1, 5):
        future = _reference_data(grouped=grouped, start=5, periods=periods, observed=False, reverse=True)
        prepared = model.prepare_data(future)
        actual = generate_draws(jax.random.split(jax.random.key(0), 2), draws, prepared)
        assert actual["raw_prediction"].shape == (2, periods) + ((2,) if grouped else ())
        predicted = (
            np.asarray(draws["intercept"]).reshape((2, 1, 1) if grouped else (2, 1)) * expected_scale + expected_offset
        )
        np.testing.assert_allclose(
            actual["raw_prediction"], np.broadcast_to(predicted, actual["raw_prediction"].shape), rtol=1e-6
        )
        np.testing.assert_allclose(
            actual["raw_reference"], np.broadcast_to(raw.arrays["outcome"], actual["raw_reference"].shape), rtol=1e-6
        )


@pytest.mark.parametrize("shape", [(), (3,), (3, 1)])
def test_outcome_inverse_helper_preserves_grouped_broadcast_shape_guards(shape):
    raw = _reference_data()
    scaling = fit_data_scaling(raw, scale_outcome="population")
    model = Model(
        {},
        lambda: jnp.array(0.0),
        lambda key, outcome_scaling: {"raw_prediction": outcome_scaling.inverse_transform(jnp.zeros(shape))},
        data=Data(raw, scaling=scaling),
    )
    with pytest.raises(ValueError, match=r"shape|broadcast"):
        jax.jit(model.generate_quantities)(jax.random.key(0), {}, model.data)


def test_auto_scaling_matches_explicit_fitting_without_changing_counts_or_costs():
    data = _data()
    model = _model(data, scaling="auto")
    expected = fit_data_scaling(data).transform(data)
    assert model.scaling is not None
    assert set(model.scaling.transformations) == {"media", "controls", "treatments"}
    for role, values in expected.arrays.items():
        np.testing.assert_allclose(model.data.values[role], values, rtol=1e-6)
    np.testing.assert_array_equal(model.data.values["outcome"], data.arrays["outcome"])
    np.testing.assert_array_equal(model.data.values["spend"], data.arrays["spend"])
    np.testing.assert_array_equal(data.arrays["media"][:, 0], [0, 100, 300])


def test_model_without_scaling_preserves_original_inputs():
    data = _data()
    model = _model(data)
    assert model.scaling is None
    for role, values in data.arrays.items():
        np.testing.assert_array_equal(model.data.values[role], values)


def test_supplied_scaling_can_standardize_continuous_outcomes():
    data = _data()
    scaling = fit_data_scaling(data, scale_outcome=True)
    model = _model(data, scaling=scaling)
    assert model.scaling is scaling
    np.testing.assert_allclose(model.data.values["outcome"], [-np.sqrt(1.5), 0, np.sqrt(1.5)], rtol=1e-6)
    restored = scaling.transformations["outcome"].inverse_transform(model.data.values["outcome"])
    np.testing.assert_allclose(restored, data.arrays["outcome"])


def test_population_outcome_scaling_matches_manual_preparation_and_jit_gradients():
    raw = _population_data()
    original = raw.arrays["outcome"].copy()
    fitted = fit_data_scaling(raw, scale_outcome="population", media_method=None)
    model = _population_model(raw, scaling=fitted)
    per_capita = original / raw.arrays["population"]
    expected = (per_capita - per_capita.mean()) / per_capita.std(ddof=0)
    manual = _population_model(replace(raw, arrays=raw.arrays | {"outcome": expected}))
    position = {"intercept": jnp.array(0.3)}
    actual_value, actual_gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    expected_value, expected_gradient = jax.jit(jax.value_and_grad(manual.log_density))(position, manual.data)

    assert model.scaling is fitted
    np.testing.assert_allclose(model.data.values["outcome"], expected, rtol=1e-6)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6)
    np.testing.assert_allclose(actual_gradient["intercept"], expected_gradient["intercept"], rtol=1e-6)
    np.testing.assert_array_equal(raw.arrays["outcome"], original)
    restored = fitted.transformations["outcome"].inverse_transform(model.data.values["outcome"])
    np.testing.assert_allclose(restored, original, rtol=1e-6)


@pytest.mark.parametrize("include_population", [False, True])
def test_population_outcome_scaling_reuses_training_factors_and_result_labels(include_population):
    raw = _population_data()
    fitted = fit_data_scaling(raw, scale_outcome="population", media_method=None)
    model = _population_model(raw, scaling=fitted)
    incoming = _population_data(start=4, reverse=True, include_population=include_population)
    canonical = _population_data(start=4)
    posterior = _collect_results({"intercept": np.zeros((1, 2), dtype=np.float32)})
    evaluated = generate_quantities(model, posterior, new_data=incoming)
    per_capita = raw.arrays["outcome"] / raw.arrays["population"]
    expected = (canonical.arrays["outcome"] / raw.arrays["population"] - per_capita.mean()) / per_capita.std()

    np.testing.assert_allclose(evaluated["observed_data"]["outcome"], expected, rtol=1e-6)
    np.testing.assert_array_equal(evaluated["constant_data"]["media"], canonical.arrays["media"])
    for group in ("observed_data", "posterior_predictive"):
        np.testing.assert_array_equal(evaluated[group]["time"], [4, 5, 6])
        np.testing.assert_array_equal(evaluated[group]["group"], ["west", "east"])
    assert evaluated["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time", "group")
    assert incoming.group_values == (("east",), ("west",))
    np.testing.assert_array_equal(evaluated["constant_data"]["channel"], ["video", "search"])
    np.testing.assert_allclose(model.data.values["outcome"], (per_capita - per_capita.mean()) / per_capita.std())


@pytest.mark.parametrize("explicit", [False, True])
def test_population_outcome_scaling_provenance_prevents_double_transformation(explicit):
    raw = _population_data()
    fitted = fit_data_scaling(raw, scale_outcome="population")
    scaled = fitted.transform(raw)
    model = _population_model(scaled, scaling=fitted if explicit else None)
    assert model.scaling is fitted
    for incoming in (raw, scaled):
        prepared = model.prepare_data(incoming)
        for role, values in model.data.values.items():
            np.testing.assert_array_equal(prepared.values[role], values)


def test_auto_model_scaling_keeps_outcomes_in_raw_units_when_population_is_available():
    raw = _population_data()
    model = _population_model(raw, scaling="auto")
    assert "outcome" not in model.scaling.transformations
    np.testing.assert_array_equal(model.data.values["outcome"], raw.arrays["outcome"])


def test_new_data_reuses_fitted_statistics_and_channel_order_without_outcomes():
    original = _data()
    model = _model(original, scaling="auto")
    incoming = _data(multiplier=10.0, reverse=True, outcome=False)
    prepared = model.prepare_data(incoming)
    result = jax.jit(model.generate_quantities)(jax.random.key(0), {"intercept": jnp.array(0.0)}, prepared)
    np.testing.assert_allclose(result["media"], original.arrays["media"] * 10 / [200, 80], rtol=1e-6)
    np.testing.assert_array_equal(result["spend"], original.arrays["spend"])
    np.testing.assert_array_equal(model.data.values["media"], original.arrays["media"] / [200, 80])


@pytest.mark.parametrize("explicit", [False, True])
def test_already_scaled_data_is_not_transformed_twice(explicit):
    raw = _data()
    scaling = fit_data_scaling(raw, scale_outcome=True)
    scaled = scaling.transform(raw)
    model = _model(scaled, scaling=scaling if explicit else None)
    assert model.scaling is scaling
    for incoming in (raw, scaled):
        prepared = model.prepare_data(incoming)
        for role, values in model.data.values.items():
            np.testing.assert_array_equal(prepared.values[role], values)


def test_scaled_provenance_rejects_double_fitting_and_incompatible_transformations():
    raw = _data()
    fitted = fit_data_scaling(raw)
    scaled = fitted.transform(raw)
    other = fit_data_scaling(raw, media_method="max")
    with pytest.raises(ValueError, match="already been scaled"):
        fitted.transform(scaled)
    with pytest.raises(ValueError, match="already been scaled"):
        fit_data_scaling(scaled)
    with pytest.raises(ValueError, match="already been scaled"):
        _model(scaled, scaling="auto")
    with pytest.raises(ValueError, match="already been scaled"):
        _model(scaled, scaling=other)
    with pytest.raises(ValueError, match="different fitted scales"):
        other.inverse_transform(scaled)
    model = _model(raw, scaling=fitted)
    with pytest.raises(ValueError, match="already been scaled"):
        model.prepare_data(other.transform(raw))
    with pytest.raises(ValueError, match="unscaled inputs"):
        _model(raw).prepare_data(scaled)


def test_inverse_transform_clears_provenance_and_alignment_preserves_it():
    raw = _data()
    scaling = fit_data_scaling(raw)
    scaled = scaling.transform(raw)
    aligned = scaled._align_to(raw)
    assert aligned._scaling is scaling
    restored = scaling.inverse_transform(aligned)
    assert restored._scaling is None
    for role, values in raw.arrays.items():
        np.testing.assert_allclose(restored.arrays[role], values, rtol=1e-6)
    assert fit_data_scaling(restored) is not None


def test_scaling_does_not_change_log_density_or_gradients_relative_to_manual_preparation():
    raw = _data()
    fitted = fit_data_scaling(raw, scale_outcome=True)
    automatic = _model(raw, scaling=fitted)
    manual_data = replace(fitted.transform(raw), _scaling=None)
    manual = _model(manual_data)
    position = {"intercept": jnp.array(0.3)}
    actual, actual_grad = jax.jit(jax.value_and_grad(automatic.log_density))(position, automatic.data)
    expected, expected_grad = jax.jit(jax.value_and_grad(manual.log_density))(position, manual.data)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(actual_grad["intercept"], expected_grad["intercept"], rtol=1e-6)


@pytest.mark.parametrize("scaling,error", [(False, TypeError), ({}, TypeError), ("unknown", ValueError)])
def test_invalid_scaling_options_fail_at_construction(scaling, error):
    with pytest.raises(error, match="scaling"):
        _model(_data(), scaling=scaling)


def test_scaling_requires_a_prepared_model():
    with pytest.raises(TypeError, match="Data requires PreparedData"):
        Model({}, lambda data: jnp.array(0.0), data=Data(None, scaling="auto"))


def test_models_cannot_exchange_scaled_input_bundles():
    data = _data()
    first = _model(data, scaling="auto")
    second = _model(data)
    with pytest.raises(ValueError, match="this model"):
        jax.jit(first.log_density)({"intercept": jnp.array(0.0)}, second.data)


def test_media_models_reject_changes_to_known_observation_spacing():
    def dated_data(step):
        return prepare_data(
            pl.DataFrame(
                {
                    "date": [date(2026, 1, 1) + timedelta(days=step * index) for index in range(4)],
                    "video": [1.0, 2.0, 3.0, 4.0],
                    "search": [2.0, 3.0, 1.0, 4.0],
                }
            ),
            time="date",
            media=["video", "search"],
        )

    model = Model(
        {},
        lambda media: normal(media.sum(-1), 0.0, 1.0),
        data=Data(dated_data(7), scaling="auto"),
    )
    with pytest.raises(ValueError, match="weekly observation spacing"):
        model.prepare_data(dated_data(1))
    assert model.prepare_data(dated_data(7)) is not None
