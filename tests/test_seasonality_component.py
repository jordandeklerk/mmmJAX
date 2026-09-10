"""Tests for prepared Fourier seasonality in labeled JAX models."""

from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, date, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import mmmjax
from mmmjax import Model, Real, normal, normal_rng, prepare_data
from mmmjax.seasonality import FourierSeasonality


def _data(times, *, groups=(), outcome=True, frequency="auto"):
    labels = tuple(times)
    group_count = len(groups) if groups else 1
    values = {"time": [label for label in labels for _ in range(group_count)]}
    if groups:
        values["region"] = list(groups) * len(labels)
    if outcome:
        values["sales"] = (np.arange(len(labels) * group_count) / 5 + 0.3).tolist()
    else:
        values["temperature"] = [15.0] * (len(labels) * group_count)
    return prepare_data(
        pl.DataFrame(values),
        time="time",
        groups=["region"] if groups else [],
        outcome="sales" if outcome else None,
        controls=None if outcome else ["temperature"],
        frequency=frequency,
    )


def _fourier_reference(positions, period, order):
    positions = np.asarray(positions, dtype=np.float64)
    angles = 2 * np.pi * positions[:, None] * np.arange(1, order + 1, dtype=np.float64) / np.float64(period)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)


def _normal_reference(values, location=0.0, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    return (-0.5 * ((values - location) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)).sum()


def test_fourier_seasonality_is_exported_and_has_immutable_defaults():
    spec = FourierSeasonality()

    assert mmmjax.FourierSeasonality is FourierSeasonality
    assert "FourierSeasonality" in mmmjax.__all__
    assert spec.period == "yearly"
    assert spec.order == 2
    assert spec.prior is None
    assert spec.name == "seasonality"
    assert spec.group_specific is False
    assert spec.automatic_priors is True
    for attribute, value in (("period", 7.0), ("order", 3), ("name", "weekly"), ("group_specific", True)):
        with pytest.raises(FrozenInstanceError):
            setattr(spec, attribute, value)


def test_numeric_seasonality_uses_original_time_units_and_earliest_training_origin():
    data = _data([13.5, 10.0, 11.25])
    prepared = FourierSeasonality(period=7.5, order=2, name="cycle")._prepare(data)
    coefficients = jnp.array([0.3, -0.7, 1.1, 0.5])
    expected_features = _fourier_reference([0.0, 1.25, 3.5], 7.5, 2)
    expected = expected_features @ np.asarray(coefficients, dtype=np.float64)

    assert list(prepared.parameters) == ["cycle"]
    assert isinstance(prepared.parameters["cycle"], Real)
    assert prepared.parameters["cycle"].shape == (4,)
    for actual in (
        prepared.apply(coefficients),
        jax.jit(lambda component, beta: component.apply(beta))(prepared, coefficients),
    ):
        assert actual.shape == (3,)
        np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(prepared.features, expected_features, rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("period,days", [("yearly", 365.25), ("monthly", 365.25 / 12), ("weekly", 7.0)])
def test_named_calendar_periods_match_independent_features_across_leap_day(period, days):
    data = _data(["2024-03-04", "2024-02-28", "2024-03-01"], frequency=None)
    prepared = FourierSeasonality(period=period, order=2)._prepare(data)
    coefficients = jnp.array([0.7, -0.2, 0.4, 1.1])
    expected = _fourier_reference([0, 2, 5], days, 2) @ np.asarray(coefficients, dtype=np.float64)

    np.testing.assert_allclose(prepared.apply(coefficients), expected, rtol=3e-6, atol=2e-6)


def test_date_and_naive_datetime_labels_preserve_elapsed_fractional_days():
    examples = (
        ([date(2024, 2, 28), date(2024, 3, 1), date(2024, 3, 4)], [0, 2, 5]),
        ([datetime(2024, 2, 28, 6), datetime(2024, 2, 29, 18), datetime(2024, 3, 1)], [0, 1.5, 1.75]),
    )
    coefficients = jnp.array([0.7, -0.3])
    for labels, elapsed in examples:
        prepared = FourierSeasonality(period="weekly", order=1)._prepare(_data(labels, frequency=None))
        expected = _fourier_reference(elapsed, 7, 1) @ np.asarray(coefficients, dtype=np.float64)
        np.testing.assert_allclose(prepared.apply(coefficients), expected, rtol=3e-6, atol=2e-6)


def test_dated_grouped_seasonality_matches_across_dataframe_backends():
    columns = {
        "time": [datetime(2024, 3, 1, 6)] * 2 + [datetime(2024, 2, 28, 18)] * 2 + [datetime(2024, 2, 29, 12)] * 2,
        "region": ["west", "east"] * 3,
        "sales": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
    }
    frames = (
        pd.DataFrame(columns),
        pd.DataFrame(columns).convert_dtypes(dtype_backend="pyarrow"),
        pl.DataFrame(columns),
        pa.table(columns),
    )
    spec = FourierSeasonality(period="weekly", order=1, group_specific=True)
    coefficients = jnp.array([[0.3, -0.2], [0.7, 1.1]])
    expected_features = _fourier_reference([0.0, 0.75, 1.5], 7, 1)
    expected_effect = expected_features @ np.asarray(coefficients, dtype=np.float64)
    results = []

    for frame in frames:
        data = prepare_data(frame, time="time", groups=["region"], outcome="sales", frequency=None)
        prepared = spec._prepare(data)
        effect = prepared.apply(coefficients)
        assert data.group_values == (("west",), ("east",))
        np.testing.assert_allclose(prepared.features, expected_features, rtol=3e-6, atol=2e-6)
        np.testing.assert_allclose(effect, expected_effect, rtol=3e-6, atol=2e-6)
        results.append((prepared.features, effect))
    for features, effect in results[1:]:
        np.testing.assert_array_equal(features, results[0][0])
        np.testing.assert_array_equal(effect, results[0][1])


def test_observation_time_features_ignore_earlier_media_history():
    frame = pl.DataFrame({"week": [11, 10, 12], "sales": [0.5, 0.7, 1.2], "video": [2.0, 3.0, 4.0]})
    history = pl.DataFrame({"week": [8, 9], "video": [1.0, 2.0]})
    data = prepare_data(frame, time="week", outcome="sales", media=["video"], media_history=history)
    without_history = prepare_data(frame, time="week", outcome="sales", media=["video"])
    spec = FourierSeasonality(period=7, order=1)
    prepared = spec._prepare(data)
    coefficients = jnp.array([0.3, 0.7])
    expected = _fourier_reference([0, 1, 2], 7, 1) @ np.asarray(coefficients, dtype=np.float64)

    assert len(data.media_time_values) == 5
    assert prepared.features.shape == (3, 2)
    np.testing.assert_allclose(prepared.apply(coefficients), expected, rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(prepared.features, spec._prepare(without_history).features)


def test_shared_seasonality_broadcasts_one_curve_over_groups():
    data = _data([2.0, 3.5, 6.0], groups=("west", "east", "central"))
    prepared = FourierSeasonality(period=8, order=1)._prepare(data)
    coefficients = jnp.array([0.5, 1.3])
    expected = _fourier_reference([0, 1.5, 4], 8, 1) @ np.asarray(coefficients, dtype=np.float64)
    result = jax.jit(lambda component, beta: component.apply(beta))(prepared, coefficients)

    assert prepared.parameters["seasonality"].shape == (2,)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, np.broadcast_to(expected[:, None], (3, 3)), rtol=3e-6, atol=2e-6)


def test_shared_seasonality_accepts_new_forecast_groups_and_reuses_training_origin():
    prepared = FourierSeasonality(period=8, order=1)._prepare(_data([2, 4], groups=("west", "east")))
    future_data = _data([6, 7], groups=("central",), outcome=False)
    forecast = prepared.for_data(future_data)
    coefficients = jnp.array([0.5, 1.3])
    expected = _fourier_reference([4, 5], 8, 1) @ np.asarray(coefficients, dtype=np.float64)

    assert forecast.apply(coefficients).shape == (2, 1)
    np.testing.assert_allclose(forecast.apply(coefficients)[:, 0], expected, rtol=3e-6, atol=2e-6)


def test_group_specific_forecasts_preserve_coefficient_identity_and_future_group_order():
    group_locations = jnp.array([[0.3, 1.1]])

    def prior(beta):
        return -0.5 * jnp.square(beta - group_locations) - 0.5 * jnp.log(2 * jnp.pi)

    spec = FourierSeasonality(period="weekly", order=2, name="weekly", group_specific=True, prior=prior)
    training = _data(["2026-01-01", "2026-01-03", "2026-01-04"], groups=("west", "east"), frequency=None)
    future_data = _data(["2026-01-05", "2026-01-07"], groups=("east", "west"), outcome=False)
    prepared = spec._prepare(training)
    forecast = prepared.for_data(future_data)
    coefficients = jnp.array([[0.3, -0.2], [0.7, 1.1], [-0.5, 0.4], [0.2, -0.7]])
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    expected_training = _fourier_reference([0, 2, 3], 7, 2) @ coefficient_values
    expected_future = _fourier_reference([4, 6], 7, 2) @ coefficient_values[:, [1, 0]]

    assert prepared.parameters["weekly"].shape == forecast.parameters["weekly"].shape == (4, 2)
    assert future_data.group_values == (("east",), ("west",))
    np.testing.assert_allclose(prepared.apply(coefficients), expected_training, rtol=4e-6, atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(lambda component, beta: component.apply(beta))(forecast, coefficients),
        expected_future,
        rtol=4e-6,
        atol=2e-6,
    )
    expected_prior = _normal_reference(coefficient_values, location=np.asarray(group_locations, dtype=np.float64))
    np.testing.assert_allclose(forecast.log_prior(coefficients), expected_prior, rtol=2e-6)
    np.testing.assert_array_equal(forecast.log_prior(coefficients), prepared.log_prior(coefficients))
    # A forecast prepared from another forecast still refers to the original
    # coefficient groups and origin, even when its group order changes again.
    returned = forecast.for_data(_data(["2026-01-08"], groups=("west", "east"), outcome=False))
    np.testing.assert_allclose(
        returned.apply(coefficients), _fourier_reference([7], 7, 2) @ coefficient_values, rtol=4e-6, atol=2e-6
    )


def test_default_prior_matches_standard_normal_value_and_gradient():
    prepared = FourierSeasonality(period=7, order=2, group_specific=True)._prepare(
        _data([0, 1], groups=("west", "east"))
    )
    coefficients = jnp.array([[0.3, -0.7], [0.0, 1.2], [-0.5, 0.2], [0.9, -1.1]])
    value, gradient = jax.jit(jax.value_and_grad(lambda beta, component: component.log_prior(beta)))(
        coefficients, prepared
    )

    assert value.shape == ()
    np.testing.assert_allclose(value, _normal_reference(coefficients), rtol=2e-6)
    np.testing.assert_allclose(gradient, -np.asarray(coefficients), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_disabled_automatic_priors_preserve_seasonal_coefficients_and_effect(enable_x64):
    data = _data([0, 1], groups=("west", "east"))
    spec = FourierSeasonality(period=7, order=1, name="weekly", group_specific=True)
    with jax.enable_x64(enable_x64):
        default = spec._prepare(data)
        manual = replace(spec, automatic_priors=False)._prepare(data)
        coefficients = jnp.array([[0.3, -0.7], [0.0, 1.2]])
        value, gradient = jax.jit(jax.value_and_grad(manual.log_prior))(coefficients)

        assert value.shape == () and value.dtype == coefficients.dtype
        np.testing.assert_array_equal(value, 0.0)
        assert isinstance(manual.parameters["weekly"], Real)
        assert manual.parameters["weekly"].shape == default.parameters["weekly"].shape
        assert manual.parameters["weekly"].dtype == default.parameters["weekly"].dtype
        np.testing.assert_array_equal(gradient, np.zeros_like(coefficients))
        np.testing.assert_array_equal(manual.apply(coefficients), default.apply(coefficients))
        future = manual.for_data(_data([2, 3], groups=("west", "east"), outcome=False))
        np.testing.assert_array_equal(future.log_prior(coefficients), value)


def test_disabled_automatic_priors_reject_a_seasonal_prior_callback():
    with pytest.raises(ValueError, match=r"prior.*automatic_priors=False"):
        FourierSeasonality(prior=lambda value: -(value**2), automatic_priors=False)


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", np.bool_(False)])
def test_seasonal_automatic_priors_requires_a_python_boolean(invalid):
    with pytest.raises(TypeError, match="automatic_priors"):
        FourierSeasonality(automatic_priors=invalid)


@pytest.mark.parametrize("scalar_prior", [False, True])
def test_custom_prior_accepts_elementwise_or_scalar_log_density(scalar_prior):
    def prior(coefficients):
        values = -jnp.abs(coefficients - 0.5) / 2 - jnp.log(4.0)
        return values.sum() if scalar_prior else values

    prepared = FourierSeasonality(period=8, order=1, prior=prior)._prepare(_data([0, 1]))
    coefficients = jnp.array([-0.3, 1.1])
    value, gradient = jax.jit(jax.value_and_grad(lambda beta, component: component.log_prior(beta)))(
        coefficients, prepared
    )
    expected = (-np.abs(np.asarray(coefficients, dtype=np.float64) - 0.5) / 2 - np.log(4)).sum()

    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(gradient, [0.5, -0.5], rtol=2e-6)


def test_fresh_components_with_array_valued_callable_priors_work_in_one_jitted_function():
    @dataclass
    class Prior:
        scale: np.ndarray

        def __call__(self, coefficients):
            return -0.5 * jnp.square(coefficients / self.scale) - jnp.log(self.scale) - 0.5 * jnp.log(2 * jnp.pi)

    data = _data([0, 1])
    scales = (np.array([0.5, 1.5]), np.array([1.0, 2.0]))
    components = [FourierSeasonality(period=7, order=1, prior=Prior(scale))._prepare(data) for scale in scales]
    coefficients = jnp.array([0.3, -0.7])
    evaluate = jax.jit(lambda component, beta: component.log_prior(beta))
    results = []

    for component, scale in zip(components, scales, strict=True):
        result = evaluate(component, coefficients)
        np.testing.assert_allclose(result, _normal_reference(coefficients, scale=scale), rtol=2e-6)
        results.append(result)
    assert not np.isclose(results[0], results[1])
    np.testing.assert_array_equal(evaluate(components[0], coefficients), results[0])


@pytest.mark.parametrize("group_specific", [False, True])
def test_prepared_component_composes_with_jitted_model_density_gradient_and_generation(group_specific):
    data = _data([10.0, 11.5, 13.0], groups=("west", "east"))
    prepared = FourierSeasonality(period=9, order=1, name="cycle", group_specific=group_specific)._prepare(data)

    def log_density(inputs, cycle):
        component = inputs["seasonal"]
        return component.log_prior(cycle) + normal(inputs["outcome"], component.apply(cycle), 0.7)

    def generate(key, inputs, cycle):
        effect = inputs["seasonal"].apply(cycle)
        return {"seasonality": effect, "outcome": normal_rng(key, effect, 0.7)}

    model = Model(prepared.parameters, log_density, generate)
    coefficients = jnp.array([[0.3, -0.5], [0.7, 1.1]]) if group_specific else jnp.array([0.3, 0.7])
    position = {"cycle": coefficients}
    inputs = {"outcome": data._to_jax()["outcome"], "seasonal": prepared}
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, inputs)
    features = _fourier_reference([0, 1.5, 3], 9, 1)
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    expected_effect = features @ coefficient_values
    if not group_specific:
        expected_effect = np.broadcast_to(expected_effect[:, None], (3, 2))
    residual = np.asarray(data.arrays["outcome"], dtype=np.float64) - expected_effect
    expected_value = _normal_reference(residual, scale=0.7) + _normal_reference(coefficient_values)
    expected_gradient = (
        features.T @ (residual if group_specific else residual.sum(axis=1)) / 0.7**2 - coefficient_values
    )

    np.testing.assert_allclose(value, expected_value, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gradient["cycle"], expected_gradient, rtol=5e-6, atol=3e-6)
    key = jax.random.key(42)
    generation = jax.jit(model.generate)(key, model.constrain(position), inputs)
    np.testing.assert_allclose(generation["seasonality"], expected_effect, rtol=3e-6, atol=2e-6)
    expected_draw = expected_effect + 0.7 * np.asarray(jax.random.normal(key, (3, 2), dtype=coefficients.dtype))
    np.testing.assert_allclose(generation["outcome"], expected_draw, rtol=3e-6, atol=2e-6)

    future = _data([13.5, 15.0], groups=("east", "west"), outcome=False)
    future_inputs = {"seasonal": prepared.for_data(future)}
    generated_future = jax.jit(model.generate)(key, model.constrain(position), future_inputs)
    future_features = _fourier_reference([3.5, 5], 9, 1)
    expected_future = future_features @ (coefficient_values[:, [1, 0]] if group_specific else coefficient_values)
    if not group_specific:
        expected_future = np.broadcast_to(expected_future[:, None], (2, 2))
    assert generated_future["outcome"].shape == (2, 2)
    np.testing.assert_allclose(generated_future["seasonality"], expected_future, rtol=4e-6, atol=2e-6)


def test_seasonality_validates_configuration_names_order_and_group_option():
    for name in ("", "not-valid", "class"):
        with pytest.raises(ValueError, match="name"):
            FourierSeasonality(name=name)
    with pytest.raises(TypeError, match="name"):
        FourierSeasonality(name=1)
    for order in (True, 1.5, np.int64(2)):
        with pytest.raises(TypeError, match="order"):
            FourierSeasonality(order=order)
    for order in (0, -1):
        with pytest.raises(ValueError, match="order"):
            FourierSeasonality(order=order)
    for grouped in (1, "yes"):
        with pytest.raises(TypeError, match="group_specific"):
            FourierSeasonality(group_specific=grouped)


def test_seasonality_validates_named_and_numeric_periods():
    for period in ("annual", "fortnightly", "", 0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="period"):
            FourierSeasonality(period=period)
    for period in (True, 1j, [7.0], None):
        with pytest.raises(TypeError, match="period"):
            FourierSeasonality(period=period)
    with pytest.raises(ValueError, match=r"period|numeric|calendar"):
        FourierSeasonality(period="yearly")._prepare(_data([0, 1]))


def test_seasonality_requires_callable_prior_with_compatible_output_shape():
    with pytest.raises(TypeError, match="prior"):
        FourierSeasonality(prior=0.5)
    prepared = FourierSeasonality(period=7, order=2, prior=lambda coefficients: jnp.zeros(2))._prepare(_data([0, 1]))

    with pytest.raises(ValueError, match=r"prior|shape"):
        prepared.log_prior(jnp.ones(4))

    for result in (None, "invalid", True, 1j):
        invalid = FourierSeasonality(period=7, prior=lambda coefficients, result=result: result)._prepare(_data([0, 1]))
        with pytest.raises(TypeError, match="prior must return real numeric log densities"):
            invalid.log_prior(jnp.ones(4))


def test_seasonality_requires_prepared_data_and_groups_for_group_specific_effects():
    with pytest.raises(TypeError, match=r"data|PreparedData"):
        FourierSeasonality(period=7)._prepare({"time": [0, 1]})
    with pytest.raises(ValueError, match=r"group_specific|groups"):
        FourierSeasonality(period=7, group_specific=True)._prepare(_data([0, 1]))


def test_seasonality_rejects_invalid_time_labels_and_timezone_aware_datetimes():
    data = _data([0, 1])
    invalid_labels = (
        (),
        (True, False),
        (0.0, np.nan),
        (0.0, np.inf),
        ("January", "February"),
        (0, date(2026, 1, 1)),
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)),
    )
    for labels in invalid_labels:
        with pytest.raises((TypeError, ValueError), match=r"time|date|calendar|numeric|timezone"):
            FourierSeasonality(period=7)._prepare(replace(data, time_values=labels))


def test_forecasts_cannot_switch_between_calendar_and_numeric_time():
    numeric = _data([0, 1])
    calendar = _data(["2026-01-01", "2026-01-02"])
    for training, future in ((numeric, calendar), (calendar, numeric)):
        prepared = FourierSeasonality(period=7)._prepare(training)
        with pytest.raises(TypeError, match=r"time|calendar|numeric"):
            prepared.for_data(future)


def test_group_specific_forecasts_reject_unknown_missing_or_different_group_columns():
    prepared = FourierSeasonality(period=7, group_specific=True)._prepare(_data([0, 1], groups=("west", "east")))
    for groups in (("west",), ("west", "central"), ("west", "east", "central")):
        future = _data([2, 3], groups=groups, outcome=False)
        with pytest.raises(ValueError, match="group"):
            prepared.for_data(future)
    future = replace(_data([2, 3], groups=("west", "east"), outcome=False), group_columns=("country",))
    with pytest.raises(ValueError, match="group"):
        prepared.for_data(future)


@pytest.mark.parametrize("automatic_priors", [False, True])
def test_apply_and_log_prior_require_exact_coefficient_shapes(automatic_priors):
    shared = FourierSeasonality(period=7, order=2, automatic_priors=automatic_priors)._prepare(_data([0, 1]))
    grouped = FourierSeasonality(period=7, order=2, group_specific=True, automatic_priors=automatic_priors)._prepare(
        _data([0, 1], groups=("west", "east"))
    )
    for prepared, shapes in ((shared, [(), (3,), (4, 1)]), (grouped, [(4,), (2, 4), (4, 1)])):
        for shape in shapes:
            for method in (prepared.apply, prepared.log_prior):
                with pytest.raises(ValueError, match=r"coefficient|shape"):
                    method(jnp.ones(shape))
        with pytest.raises(TypeError, match="real numeric"):
            prepared.log_prior(jnp.ones(prepared.parameters["seasonality"].shape, dtype=jnp.complex64))
