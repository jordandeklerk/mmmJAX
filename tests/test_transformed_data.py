"""Tests for fixed data-only calculations declared through transformed_data."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import (
    Data,
    Model,
    Real,
    generate_quantities,
    normal,
    optimize_budget,
    prepare_data,
    response_curves,
)
from mmmjax._results import _collect_results


def _data(periods=4, *, observed=True, multiplier=1.0):
    video = multiplier * np.arange(1.0, periods + 1.0)
    search = multiplier * np.linspace(2.0, 4.0, periods)
    frame = pl.DataFrame(
        {
            "week": np.arange(periods),
            "video": video,
            "search": search,
            "video_cost": video / 2.0,
            "search_cost": search / 4.0,
            "sales": np.linspace(1.0, 2.0, periods),
        }
    )
    return prepare_data(
        frame,
        time="week",
        outcome="sales" if observed else None,
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
    )


def _normal(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64) / scale
    return (-0.5 * values**2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def test_transformed_data_runs_once_and_feeds_every_program_block_under_jit():
    data = _data()
    calls = []

    def transformed_data(media):
        calls.append(media.shape)
        return {"log_media": jnp.log1p(media), "media_total": media.sum()}

    def transformed_parameters(log_media, coefficient):
        return {"expected": log_media @ coefficient}

    def log_density(outcome, expected, media_total, coefficient):
        return normal(outcome, expected, 1.0) + normal(coefficient, 0.0, 1.0) + 0.0 * media_total

    def generated_quantities(key, expected, media_total):
        return {"mean": expected, "total": media_total}

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=log_density,
        generated_quantities=generated_quantities,
        data=data,
        transformed_data=transformed_data,
        transformed_parameters=transformed_parameters,
    )
    assert calls == [(4, 2)]

    coefficient = np.array([0.5, -0.25])
    parameters = {"coefficient": jnp.asarray(coefficient)}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(parameters, model.data)
    outputs = jax.jit(model.generate_quantities)(jax.random.key(0), parameters, model.data)
    evaluated = jax.jit(model.evaluate)(parameters, model.data)
    log_media = np.log1p(data.arrays["media"])
    expected = log_media @ coefficient
    residual = data.arrays["outcome"] - expected

    assert calls == [(4, 2)]
    np.testing.assert_allclose(value, _normal(residual) + _normal(coefficient), rtol=2e-5)
    np.testing.assert_allclose(gradient["coefficient"], log_media.T @ residual - coefficient, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(outputs["mean"], expected, rtol=2e-5)
    np.testing.assert_allclose(outputs["total"], data.arrays["media"].sum(), rtol=2e-5)
    np.testing.assert_allclose(evaluated["expected"], expected, rtol=2e-5)


def test_transformed_data_is_recomputed_for_new_observations_while_references_stay_fixed():
    data = _data()

    def transformed_data(media, reference):
        return {"cumulative": jnp.cumsum(media, axis=0), "reference_total": reference.media.sum()}

    def generated_quantities(key, cumulative, reference_total):
        return {"cumulative": cumulative, "reference_total": reference_total}

    model = Model(
        parameters={},
        log_density=lambda cumulative: cumulative.sum(),
        generated_quantities=generated_quantities,
        data=data,
        transformed_data=transformed_data,
    )
    scenario = _data(periods=2, observed=False, multiplier=3.0)
    inputs = model.prepare_data(scenario)
    generate = jax.jit(model.generate_quantities)
    training = generate(jax.random.key(0), {}, model.data)
    outputs = generate(jax.random.key(0), {}, inputs)

    np.testing.assert_allclose(training["cumulative"], np.cumsum(data.arrays["media"], axis=0))
    np.testing.assert_allclose(outputs["cumulative"], np.cumsum(scenario.arrays["media"], axis=0))
    np.testing.assert_allclose(training["reference_total"], data.arrays["media"].sum())
    np.testing.assert_allclose(outputs["reference_total"], data.arrays["media"].sum())
    np.testing.assert_allclose(jax.jit(model.log_prob)({}, inputs), np.cumsum(scenario.arrays["media"], axis=0).sum())


def test_static_integer_and_boolean_outputs_define_shapes_and_follow_new_period_counts():
    data = _data()

    def transformed_data(n_periods, media):
        return {"window": n_periods, "half": n_periods // 2, "exposed": bool(np.any(np.asarray(media) > 0))}

    def transformed_parameters(window, half, exposed, level):
        return {"trend": jnp.linspace(0.0, 1.0, window) * level, "late": jnp.ones(half) * level, "flag": exposed}

    model = Model(
        parameters={"level": Real()},
        log_density=lambda trend, level: trend.sum(),
        data=data,
        transformed_data=transformed_data,
        transformed_parameters=transformed_parameters,
    )
    evaluate = jax.jit(model.evaluate)
    training = evaluate({"level": jnp.array(2.0)}, model.data)
    scenario = evaluate({"level": jnp.array(2.0)}, model.prepare_data(_data(periods=6, observed=False)))

    assert training["trend"].shape == (4,)
    assert training["late"].shape == (2,)
    assert bool(training["flag"])
    assert scenario["trend"].shape == (6,)
    assert scenario["late"].shape == (3,)
    np.testing.assert_allclose(scenario["trend"], np.linspace(0.0, 1.0, 6) * 2.0, rtol=1e-6)


def test_transformed_data_must_return_the_same_names_for_every_dataset():
    def transformed_data(media, n_periods):
        outputs = {"total": media.sum()}
        if n_periods > 3:
            outputs["late"] = media[3:].sum()
        return outputs

    model = Model(parameters={}, log_density=lambda total: total, data=_data(), transformed_data=transformed_data)

    with pytest.raises(ValueError, match="same names for each dataset"):
        model.prepare_data(_data(periods=2, observed=False))


def test_transformed_data_requires_prepared_data():
    with pytest.raises(ValueError, match="transformed_data requires prepared data"):
        Model(parameters={}, log_density=lambda data: jnp.array(0.0), transformed_data=lambda: {})


@pytest.mark.parametrize(
    "transformed_data, error, message",
    [
        (lambda coefficient: {}, ValueError, "unknown input 'coefficient'"),
        (lambda media: {"media": media}, ValueError, "conflicts with an input or parameter"),
        (lambda media: {"coefficient": media}, ValueError, "conflicts with an input or parameter"),
        (lambda media: media, TypeError, "must return a mapping"),
        (lambda media: {"label": "text"}, TypeError, "must contain real numeric values"),
        (lambda media: {"missing": None}, TypeError, "must contain real numeric values"),
        (lambda media: {"bad": jnp.array([jnp.nan])}, ValueError, "must contain finite values"),
        (lambda media: {"not valid": media}, ValueError, "valid non-keyword Python identifier"),
        (
            lambda media: {"big": np.array([2**32 + 1], dtype=np.int64)},
            ValueError,
            "contains integers outside the JAX dtype range",
        ),
        (lambda media: {"big": [2**32 + 1]}, ValueError, "contains integers outside the JAX dtype range"),
        (lambda media: {"big": [2**70]}, TypeError, "must contain real numeric values"),
    ],
)
def test_transformed_data_outputs_are_validated_at_construction(transformed_data, error, message):
    with jax.enable_x64(False), pytest.raises(error, match=message):
        Model(
            parameters={"coefficient": Real()},
            log_density=lambda coefficient: -jnp.square(coefficient),
            data=_data(),
            transformed_data=transformed_data,
        )


@pytest.mark.parametrize(
    "value",
    [np.array([2**32 + 1], dtype=np.int64), [2**32 + 1]],
    ids=["int64_array", "list"],
)
def test_transformed_data_accepts_and_preserves_large_integers_under_64_bit_mode(value):
    def transformed_data(media):
        return {"big": value}

    def generated_quantities(key, big):
        return {"big": big}

    with jax.enable_x64(True):
        model = Model(
            parameters={"coefficient": Real()},
            log_density=lambda coefficient: -jnp.square(coefficient),
            generated_quantities=generated_quantities,
            data=_data(),
            transformed_data=transformed_data,
        )
        outputs = model.generate_quantities(jax.random.key(0), {"coefficient": jnp.array(0.0)}, model.data)

    assert int(np.asarray(outputs["big"])[0]) == 2**32 + 1


def test_transformed_data_preserves_in_range_integers_at_either_precision():
    def transformed_data(media):
        return {"count": np.array([5], dtype=np.int64)}

    def generated_quantities(key, count):
        return {"count": count}

    for precision in (False, True):
        with jax.enable_x64(precision):
            model = Model(
                parameters={"coefficient": Real()},
                log_density=lambda coefficient: -jnp.square(coefficient),
                generated_quantities=generated_quantities,
                data=_data(),
                transformed_data=transformed_data,
            )
            outputs = model.generate_quantities(jax.random.key(0), {"coefficient": jnp.array(0.0)}, model.data)

        assert int(np.asarray(outputs["count"])[0]) == 5


def test_transformed_data_uses_declared_variable_names_only():
    data = _data()
    block = Data(data, variables={"revenue": "outcome", "impressions": "media"})

    def log_density(revenue, log_impressions, coefficient):
        return normal(revenue, log_impressions @ coefficient, 1.0)

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=log_density,
        data=block,
        transformed_data=lambda impressions: {"log_impressions": jnp.log1p(impressions)},
    )
    coefficient = np.array([0.1, 0.2])
    residual = data.arrays["outcome"] - np.log1p(data.arrays["media"]) @ coefficient

    assert model.data_variables == {"revenue": "outcome", "impressions": "media"}
    np.testing.assert_allclose(model.log_prob({"coefficient": jnp.asarray(coefficient)}), _normal(residual), rtol=2e-5)
    with pytest.raises(ValueError, match="unknown input 'media'"):
        Model(
            parameters={"coefficient": Real((2,))},
            log_density=log_density,
            data=block,
            transformed_data=lambda media: {"log_impressions": jnp.log1p(media)},
        )


def test_transformed_data_outputs_returned_unchanged_keep_their_data_labels():
    data = _data()

    def transformed_data(media, spend):
        return {"exposure": media, "cost_share": spend / spend.sum()}

    def generated_quantities(key, exposure, cost_share, level):
        return {"exposure_copy": exposure, "share_copy": cost_share, "level_copy": level}

    model = Model(
        parameters={"level": Real()},
        log_density=lambda level: -jnp.square(level),
        generated_quantities=generated_quantities,
        data=data,
        transformed_data=transformed_data,
    )
    results = _collect_results({"level": np.zeros((1, 2), dtype=np.float32)}, data=data)
    generated = generate_quantities(model, results)["generated_quantities"]

    assert generated["exposure_copy"].dims == ("chain", "draw", "media_time", "channel")
    np.testing.assert_array_equal(generated["channel"], ["video", "search"])
    np.testing.assert_allclose(generated["exposure_copy"][0, 0], data.arrays["media"])
    np.testing.assert_allclose(generated["share_copy"][0, 0], data.arrays["spend"] / data.arrays["spend"].sum())


def _budget_models(data):
    curvature = jnp.array([[2.0, 0.6], [0.6, 1.5]])

    def quadratic(totals, periods, coefficient):
        difference = totals - coefficient**2
        response = 1000.0 - 0.5 * difference @ curvature @ difference
        return {"expected": jnp.full(periods, response / periods)}

    def density(expected):
        raise AssertionError("Response analysis must not evaluate the log density")

    reference = Model(
        parameters={"coefficient": Real((2,))},
        log_density=density,
        data=data,
        transformed_parameters=lambda media, coefficient: quadratic(media.sum(axis=0), media.shape[0], coefficient),
        dims={"coefficient": ("channel",)},
    )
    fixed = Model(
        parameters={"coefficient": Real((2,))},
        log_density=density,
        data=data,
        transformed_data=lambda media: {"total_exposure": media.sum(axis=0)},
        transformed_parameters=lambda total_exposure, n_periods, coefficient: quadratic(
            total_exposure, n_periods, coefficient
        ),
        dims={"coefficient": ("channel",)},
    )
    coefficients = np.array([[[0.5, 1.5], [1.0, 0.5]]], dtype=np.float32)
    results = _collect_results(
        {"coefficient": coefficients},
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [0], "draw": [0, 1]},
    )
    return reference, fixed, results


def test_response_curves_recompute_transformed_data_for_each_spending_level():
    reference, fixed, results = _budget_models(_data())

    expected = response_curves(reference, results, quantity="expected", multipliers=[0.0, 1.0, 2.0])
    curves = response_curves(fixed, results, quantity="expected", multipliers=[0.0, 1.0, 2.0])

    xr.testing.assert_allclose(curves, expected)
    assert not np.allclose(curves["response"].isel(multiplier=0), curves["response"].isel(multiplier=2))


def test_budget_optimization_recomputes_transformed_data_for_candidate_allocations():
    reference, fixed, results = _budget_models(_data())
    options = {"budget": 10.0, "quantity": "expected", "bounds": (0.0, 20.0)}

    expected = optimize_budget(reference, results, **options)
    allocation = optimize_budget(fixed, results, **options)

    xr.testing.assert_allclose(allocation, expected)
    assert not np.allclose(
        allocation["spend"].sel(allocation="optimized"), allocation["spend"].sel(allocation="reference")
    )


def test_data_variables_list_every_standard_name_without_being_requested():
    data = _data()
    model = Model(
        parameters={"level": Real()}, log_density=lambda outcome, level: normal(outcome, level, 1.0), data=data
    )
    variables = model.data_variables

    assert variables == {name: name for name in variables}
    assert {"outcome", "media", "spend", "n_periods", "outcome_scaling", "reference"} <= variables.keys()
    assert {"time", "media_time"} <= variables.keys()
    assert not [name for name in variables if name.startswith("reference_") or name.startswith("unscale")]
    assert variables.keys() == data.model_inputs.keys()
