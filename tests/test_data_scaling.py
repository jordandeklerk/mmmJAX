"""Tests for fitted scaling of labeled model inputs."""

import weakref
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pytest

from mmmjax import DataScaling, PreparedData, fit_data_scaling, prepare_data


@pytest.fixture
def training_data():
    frame = pl.DataFrame(
        {
            "week": [1, 1, 2, 2],
            "region": ["west", "east", "west", "east"],
            "residents": [10, 20, 10, 20],
            "sales": [2, 4, 6, 8],
            "revenue": [2.0, 3.0, 4.0, 5.0],
            "video": [20, 80, 40, 120],
            "search": [0, 40, 30, 80],
            "video_spend": [2.0, 8.0, 4.0, 12.0],
            "search_spend": [0.0, 4.0, 3.0, 8.0],
            "organic": [5, 10, 15, 20],
            "audience": [100, 200, 300, 400],
            "frequency": [2.0, 3.0, 4.0, 5.0],
            "audience_spend": [1.0, 2.0, 3.0, 4.0],
            "organic_audience": [20, 40, 60, 80],
            "organic_frequency": [1.0, 2.0, 3.0, 4.0],
            "control": [1.0, 3.0, 5.0, 7.0],
            "price": [10.0, 20.0, 30.0, 40.0],
        }
    )
    history = pl.DataFrame(
        {
            "week": [0, 0],
            "region": ["west", "east"],
            "video": [10, 20],
            "search": [10, 20],
            "organic": [0, 0],
            "audience": [0, 0],
            "frequency": [1.0, 1.0],
            "organic_audience": [0, 0],
            "organic_frequency": [1.0, 1.0],
        }
    )
    return prepare_data(
        frame,
        time="week",
        groups=["region"],
        population="residents",
        outcome="sales",
        revenue_per_outcome="revenue",
        media=["video", "search"],
        spend=["video_spend", "search_spend"],
        channels=["Video", "Search"],
        organic_media=["organic"],
        reach=["audience"],
        media_frequency=["frequency"],
        rf_spend=["audience_spend"],
        organic_reach=["organic_audience"],
        organic_frequency=["organic_frequency"],
        controls=["control"],
        treatments=["price"],
        media_history=history,
    )


def test_default_data_scaling_preserves_labels_and_uses_all_exposure_history(training_data):
    fitted = fit_data_scaling(training_data)
    transformed = fitted.transform(training_data)

    assert isinstance(fitted, DataScaling)
    assert isinstance(transformed, PreparedData)
    assert set(fitted.transformations) == {"media", "organic_media", "reach", "organic_reach", "controls", "treatments"}
    assert transformed.time_values == (1, 2)
    assert transformed.media_time_values == (0, 1, 2)
    assert transformed.group_values == training_data.group_values
    assert transformed.columns == training_data.columns
    assert transformed.channels == ("Video", "Search")
    assert transformed.organic_channels == training_data.organic_channels
    assert transformed.rf_channels == training_data.rf_channels
    assert transformed.organic_rf_channels == training_data.organic_rf_channels
    for role, divisor in [("media", 30.0), ("organic_media", 12.5), ("reach", 250.0), ("organic_reach", 50.0)]:
        np.testing.assert_allclose(transformed.arrays[role], training_data.arrays[role] / divisor, rtol=1e-6)
    np.testing.assert_allclose(transformed.arrays["controls"], (training_data.arrays["controls"] - 4) / np.sqrt(5))
    np.testing.assert_allclose(
        transformed.arrays["treatments"], (training_data.arrays["treatments"] - 25) / np.sqrt(125)
    )


def test_data_scaling_preserves_unselected_values_and_independent_storage(training_data):
    original = deepcopy(training_data)
    transformed = fit_data_scaling(training_data).transform(training_data)
    for role, values in transformed.arrays.items():
        assert isinstance(values, np.ndarray)
        assert not np.shares_memory(values, training_data.arrays[role])
    for role in (
        "outcome",
        "spend",
        "rf_spend",
        "revenue_per_outcome",
        "population",
        "media_frequency",
        "organic_frequency",
    ):
        np.testing.assert_array_equal(transformed.arrays[role], original.arrays[role])
        assert transformed.arrays[role].dtype == original.arrays[role].dtype
    transformed.arrays["spend"][...] = -1
    transformed.columns["media"] = ("changed",)
    np.testing.assert_array_equal(training_data.arrays["spend"], original.arrays["spend"])
    assert training_data.columns == original.columns


@pytest.mark.parametrize("adjust_population", [False, True])
@pytest.mark.parametrize("scale_outcome", [True, "population"])
def test_inverse_data_scaling_recovers_values_with_zero_exposures(training_data, adjust_population, scale_outcome):
    fitted = fit_data_scaling(training_data, scale_outcome=scale_outcome, adjust_population=adjust_population)
    restored = fitted.inverse_transform(fitted.transform(training_data))
    for role, values in training_data.arrays.items():
        np.testing.assert_allclose(restored.arrays[role], values, rtol=1e-6, atol=1e-6)
    assert restored.time_values == training_data.time_values
    assert restored.media_time_values == training_data.media_time_values
    assert restored.columns == training_data.columns


def test_outcome_scaling_is_explicit_and_does_not_adjust_for_population(training_data):
    transformed = fit_data_scaling(training_data, scale_outcome=True, adjust_population=True).transform(training_data)
    assert np.issubdtype(transformed.arrays["outcome"].dtype, np.floating)
    np.testing.assert_allclose(transformed.arrays["outcome"], (training_data.arrays["outcome"] - 5) / np.sqrt(5))
    assert np.issubdtype(training_data.arrays["outcome"].dtype, np.integer)


@pytest.mark.parametrize("adjust_population", [False, True])
@pytest.mark.parametrize("media_method", [None, "median"])
def test_population_outcome_scaling_pools_per_capita_values(training_data, adjust_population, media_method):
    options = {"adjust_population": adjust_population, "media_method": media_method}
    fitted = fit_data_scaling(training_data, scale_outcome="population", **options)
    transformed = fitted.transform(training_data)
    unscaled_outcome = fit_data_scaling(training_data, **options).transform(training_data)
    per_capita = training_data.arrays["outcome"] / training_data.arrays["population"]
    mean, deviation = per_capita.mean(), per_capita.std(ddof=0)
    outcome = fitted.transformations["outcome"]

    assert outcome.offset.shape == outcome.scale.shape == (1, 2)
    np.testing.assert_allclose(outcome.offset, training_data.arrays["population"][None] * mean, rtol=1e-6)
    np.testing.assert_allclose(outcome.scale, training_data.arrays["population"][None] * deviation, rtol=1e-6)
    np.testing.assert_allclose(transformed.arrays["outcome"], (per_capita - mean) / deviation, rtol=1e-6)
    np.testing.assert_allclose(transformed.arrays["outcome"].mean(), 0.0, atol=2e-7)
    np.testing.assert_allclose(transformed.arrays["outcome"].std(ddof=0), 1.0, rtol=1e-6)
    assert transformed.group_values == training_data.group_values
    for role in transformed.arrays:
        if role != "outcome":
            np.testing.assert_array_equal(transformed.arrays[role], unscaled_outcome.arrays[role])


def test_population_outcome_inverse_preserves_batched_raw_units_and_jit_gradients(training_data):
    fitted = fit_data_scaling(training_data, scale_outcome="population").transformations["outcome"]
    population = training_data.arrays["population"]
    per_capita = training_data.arrays["outcome"] / population
    raw = np.broadcast_to(training_data.arrays["outcome"], (2, 3, 2, 2)).copy().astype(np.float32)
    raw += np.arange(3, dtype=np.float32)[None, :, None, None]
    transformed = jax.jit(fitted.transform)(raw)
    restored = jax.jit(fitted.inverse_transform)(transformed)
    forward_gradient = jax.jit(jax.grad(lambda values: fitted.transform(values).sum()))(jnp.asarray(raw))
    inverse_gradient = jax.jit(jax.grad(lambda values: fitted.inverse_transform(values).sum()))(transformed)

    assert transformed.shape == restored.shape == raw.shape
    np.testing.assert_allclose(transformed, (raw / population - per_capita.mean()) / per_capita.std(), rtol=1e-6)
    np.testing.assert_allclose(restored, raw, rtol=1e-6, atol=1e-6)
    factors = np.broadcast_to(population * per_capita.std(ddof=0), raw.shape)
    np.testing.assert_allclose(inverse_gradient, factors, rtol=1e-6)
    np.testing.assert_allclose(forward_gradient, 1 / factors, rtol=1e-6)


def test_national_population_outcome_scaling_keeps_a_length_one_statistic_axis():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2, 3], "sales": [100.0, 200.0, 300.0], "residents": [10.0] * 3}),
        time="week",
        outcome="sales",
        population="residents",
    )
    fitted = fit_data_scaling(data, scale_outcome="population")
    outcome = fitted.transformations["outcome"]
    ordinary = fit_data_scaling(data, scale_outcome=True).transformations["outcome"]
    assert outcome.offset.shape == outcome.scale.shape == (1,)
    np.testing.assert_allclose(outcome.offset, ordinary.offset, rtol=1e-6)
    np.testing.assert_allclose(outcome.scale, ordinary.scale, rtol=1e-6)
    np.testing.assert_allclose(fitted.inverse_transform(fitted.transform(data)).arrays["outcome"], [100, 200, 300])


def test_constant_per_capita_outcome_uses_unit_deviation(training_data):
    training_data.arrays["outcome"] = np.broadcast_to(2 * training_data.arrays["population"], (2, 2)).copy()
    fitted = fit_data_scaling(training_data, scale_outcome="population")
    outcome = fitted.transformations["outcome"]
    np.testing.assert_array_equal(outcome.offset, [[20, 40]])
    np.testing.assert_array_equal(outcome.scale, [[10, 20]])
    np.testing.assert_array_equal(fitted.transform(training_data).arrays["outcome"], np.zeros((2, 2)))


@pytest.mark.parametrize("include_population", [False, True])
def test_population_outcome_scaling_reuses_factors_after_group_alignment(training_data, include_population):
    fitted = fit_data_scaling(training_data, scale_outcome="population", media_method=None)
    prediction = prepare_data(
        pl.DataFrame(
            {
                "week": [4, 3, 3, 4],
                "region": ["east", "west", "east", "west"],
                "sales": [20.0, 10.0, 16.0, 12.0],
                "residents": [20, 10, 20, 10],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        population="residents" if include_population else None,
    )
    transformed = fitted.transform(prediction)
    per_capita = training_data.arrays["outcome"] / training_data.arrays["population"]
    raw = np.array([[10.0, 16.0], [12.0, 20.0]])
    expected = (raw / training_data.arrays["population"] - per_capita.mean()) / per_capita.std(ddof=0)
    np.testing.assert_allclose(transformed.arrays["outcome"], expected, rtol=1e-6)
    np.testing.assert_allclose(fitted.inverse_transform(transformed).arrays["outcome"], raw, rtol=1e-6)
    assert transformed.time_values == (3, 4)
    assert transformed.group_values == training_data.group_values
    assert prediction.group_values == (("east",), ("west",))
    assert ("population" in transformed.arrays) is include_population


@pytest.mark.parametrize("operation", ["transform", "inverse_transform"])
def test_population_outcome_scaling_rejects_changed_population_without_scaled_media(training_data, operation):
    fitted = fit_data_scaling(training_data, scale_outcome="population", media_method=None)
    prediction = deepcopy(training_data)
    prediction.arrays["population"][0] += 1
    with pytest.raises(ValueError, match="population"):
        getattr(fitted, operation)(prediction)


def test_population_outcome_scaling_allows_changed_population_when_outcome_is_omitted(training_data):
    fitted = fit_data_scaling(training_data, scale_outcome="population", media_method=None)
    prediction = deepcopy(training_data)
    prediction.arrays.pop("outcome")
    prediction.columns.pop("outcome")
    prediction.arrays["population"] *= 2
    transformed = fitted.transform(prediction)
    np.testing.assert_array_equal(transformed.arrays["population"], prediction.arrays["population"])
    np.testing.assert_array_equal(transformed.arrays["media"], prediction.arrays["media"])


@pytest.mark.parametrize("missing", ["outcome", "population"])
def test_population_outcome_scaling_requires_both_inputs(missing):
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "sales": [1.0, 2.0], "residents": [10.0, 10.0]}),
        time="week",
        outcome=None if missing == "outcome" else "sales",
        population=None if missing == "population" else "residents",
    )
    with pytest.raises(ValueError, match=missing):
        fit_data_scaling(data, scale_outcome="population")


@pytest.mark.parametrize("population", [0.0, -1.0, np.nan, np.inf])
def test_population_outcome_scaling_revalidates_population_values(training_data, population):
    training_data.arrays["population"] = np.array([population, 20.0])
    with pytest.raises(ValueError, match="population"):
        fit_data_scaling(training_data, scale_outcome="population", media_method=None)


@pytest.mark.parametrize("case", ["adjusted_overflow", "factor_overflow", "factor_underflow"])
def test_population_outcome_scaling_requires_finite_adjusted_values_and_representable_factors(case):
    populations = {"adjusted_overflow": 1e-308, "factor_overflow": 1e40, "factor_underflow": 1e-46}
    population = populations[case]
    outcomes = [1e308, 5e307] if case == "adjusted_overflow" else [population, 3 * population]
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "sales": outcomes, "residents": [population, population]}),
        time="week",
        outcome="sales",
        population="residents",
    )
    with jax.enable_x64(False), pytest.raises(ValueError, match=r"finite|positive|scale|outcome"):
        fit_data_scaling(data, scale_outcome="population")
    if case != "adjusted_overflow":
        with jax.enable_x64(True):
            fitted = fit_data_scaling(data, scale_outcome="population")
            np.testing.assert_allclose(fitted.transform(data).arrays["outcome"], [-1.0, 1.0], atol=1e-14)


@pytest.mark.parametrize(
    "method,video_divisor,search_divisor", [("median", 30.0, 30.0), ("mean", 290 / 6, 30.0), ("max", 120.0, 80.0)]
)
def test_data_scaling_uses_requested_media_method(training_data, method, video_divisor, search_divisor):
    transformed = fit_data_scaling(training_data, media_method=method).transform(training_data)
    np.testing.assert_allclose(
        transformed.arrays["media"], training_data.arrays["media"] / [video_divisor, search_divisor], rtol=1e-6
    )


def test_data_scaling_can_leave_every_input_unchanged(training_data):
    fitted = fit_data_scaling(training_data, media_method=None, scale_controls=False, scale_treatments=False)
    assert fitted.transformations == {}
    transformed = fitted.transform(training_data)
    for role, values in training_data.arrays.items():
        np.testing.assert_array_equal(transformed.arrays[role], values)
        assert transformed.arrays[role].dtype == values.dtype


@pytest.mark.parametrize("include_population", [False, True])
def test_predictions_align_groups_and_channels_without_refitting(training_data, include_population):
    fitted = fit_data_scaling(training_data, adjust_population=True)
    prediction = prepare_data(
        pd.DataFrame(
            {
                "week": [4, 3, 3, 4],
                "region": ["east", "west", "east", "west"],
                "search": [80, 20, 40, 40],
                "video": [120, 30, 60, 60],
                "residents": [20, 10, 20, 10],
            }
        ),
        time="week",
        groups=["region"],
        media=["search", "video"],
        channels=["Search", "Video"],
        population="residents" if include_population else None,
        media_history=pd.DataFrame({"week": [2, 2], "region": ["east", "west"], "search": [20, 10], "video": [30, 15]}),
    )
    transformed = fitted.transform(prediction)

    assert transformed.time_values == (3, 4)
    assert transformed.media_time_values == (2, 3, 4)
    assert transformed.group_values == (("west",), ("east",))
    assert transformed.channels == ("Video", "Search")
    expected_columns = {"media": ("video", "search")}
    if include_population:
        expected_columns["population"] = ("residents",)
        np.testing.assert_array_equal(transformed.arrays["population"], [10, 20])
    assert transformed.columns == expected_columns
    assert set(transformed.arrays) == set(expected_columns)
    # Training per-capita positive medians are 3 for video and 2 for search
    expected = np.array([[[0.5, 0.5], [0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])
    np.testing.assert_allclose(transformed.arrays["media"], expected)
    assert prediction.group_values == (("east",), ("west",))
    assert prediction.columns["media"] == ("search", "video")
    restored = fitted.inverse_transform(transformed)
    np.testing.assert_array_equal(
        restored.arrays["media"], [[[15, 10], [30, 20]], [[30, 20], [60, 40]], [[60, 40], [120, 80]]]
    )


def test_fitted_statistics_and_labels_survive_mutation_of_training_data(training_data):
    prediction = deepcopy(training_data)
    fitted = fit_data_scaling(training_data, adjust_population=True)
    expected = fitted.transform(prediction)
    for values in training_data.arrays.values():
        values[...] = 1
    training_data.columns["media"] = ("changed", "columns")
    fitted.transformations.clear()
    actual = fitted.transform(prediction)
    for role in actual.arrays:
        np.testing.assert_array_equal(actual.arrays[role], expected.arrays[role])
    assert actual.columns == expected.columns
    assert "media" in fitted.transformations
    transformations = fitted.transformations
    # CPython gh-105936 can raise TypeError instead of AttributeError for frozen slotted properties
    with pytest.raises((AttributeError, TypeError)):
        fitted.transformations = {}
    assert fitted.transformations == transformations


def test_fitted_data_scaling_does_not_retain_training_observation_buffers():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "impressions": [10, 30], "residents": [100, 100]}),
        time="week",
        media=["impressions"],
        population="residents",
    )
    observations = [weakref.ref(array) for array in data.arrays.values()]
    fitted = fit_data_scaling(data, adjust_population=True)

    del data

    assert all(reference() is None for reference in observations)
    np.testing.assert_allclose(fitted.transformations["media"].scale, [[20]])


@pytest.mark.parametrize("operation", ["transform", "inverse_transform"])
def test_population_adjusted_scaling_rejects_changed_population(training_data, operation):
    fitted = fit_data_scaling(training_data, adjust_population=True)
    prediction = deepcopy(training_data)
    prediction.arrays["population"][0] += 1
    with pytest.raises(ValueError, match="population"):
        getattr(fitted, operation)(prediction)


@pytest.mark.parametrize("options", [{}, {"media_method": None, "adjust_population": True}])
def test_unchanged_media_policy_does_not_restrict_prediction_population(training_data, options):
    fitted = fit_data_scaling(training_data, **options)
    prediction = deepcopy(training_data)
    prediction.arrays["population"] *= 2
    np.testing.assert_array_equal(fitted.transform(prediction).arrays["population"], prediction.arrays["population"])


def test_population_change_is_allowed_when_prediction_has_no_scaled_exposure(training_data):
    prediction = prepare_data(
        pl.DataFrame({"week": [3, 3], "region": ["east", "west"], "residents": [40, 20], "control": [9.0, 11.0]}),
        time="week",
        groups=["region"],
        population="residents",
        controls=["control"],
    )
    transformed = fit_data_scaling(training_data, adjust_population=True).transform(prediction)
    np.testing.assert_array_equal(transformed.arrays["population"], [20, 40])
    np.testing.assert_allclose(transformed.arrays["controls"], [[[7 / np.sqrt(5)], [5 / np.sqrt(5)]]])


def test_large_integer_exposures_are_scaled_before_jitted_computation():
    data = prepare_data(
        pl.DataFrame({"week": [1, 2], "impressions": [3_000_000_000, 5_000_000_000], "sales": [1, 3]}),
        time="week",
        media=["impressions"],
        outcome="sales",
    )
    transformed = fit_data_scaling(data).transform(data)
    inputs = transformed._to_jax()
    result = jax.jit(lambda blocks: jnp.sum(blocks["media"][..., 0] * blocks["outcome"]))(inputs)
    np.testing.assert_allclose(inputs["media"], [[0.75], [1.25]])
    np.testing.assert_allclose(result, 4.5)


@pytest.mark.parametrize("option", ["scale_outcome", "scale_controls", "scale_treatments", "adjust_population"])
@pytest.mark.parametrize("value", [0, "yes"])
def test_data_scaling_rejects_non_boolean_options(training_data, option, value):
    with pytest.raises(TypeError, match=option):
        fit_data_scaling(training_data, **{option: value})


@pytest.mark.parametrize(
    "options,match",
    [
        ({"scale_outcome": True}, "outcome"),
        ({"adjust_population": True}, "population"),
        ({"media_method": "unknown"}, "media_method"),
    ],
)
def test_data_scaling_reports_missing_inputs_and_invalid_methods(options, match):
    data = prepare_data(pl.DataFrame({"week": [1, 2], "impressions": [2, 6]}), time="week", media=["impressions"])
    with pytest.raises(ValueError, match=match):
        fit_data_scaling(data, **options)


@pytest.mark.parametrize(
    "change,match",
    [
        ("groups", "group labels"),
        ("features", "columns.*media"),
        ("channels", "channel labels"),
        ("roles", "do not exist"),
    ],
)
def test_data_scaling_rejects_incompatible_prediction_labels(change, match):
    values = {"week": [3, 3], "region": ["west", "east"], "video": [30, 60], "search": [20, 40], "new": [1, 2]}
    selection = {"media": ["video", "search"], "channels": ["Video", "Search"]}
    reference = prepare_data(pl.DataFrame(values), time="week", groups=["region"], **selection)
    fitted = fit_data_scaling(reference)
    if change == "groups":
        values["region"] = ["west", "new"]
    elif change == "features":
        selection["media"] = ["video", "new"]
    elif change == "channels":
        selection["channels"] = ["Search", "Video"]
    else:
        selection["controls"] = ["new"]
    prediction = prepare_data(pl.DataFrame(values), time="week", groups=["region"], **selection)
    with pytest.raises(ValueError, match=match):
        fitted.transform(prediction)
