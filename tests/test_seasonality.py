"""Tests for Fourier seasonality features."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mmmjax
from mmmjax.seasonality import fourier_features


def _fourier_reference(time, period, order):
    time64 = np.asarray(time, dtype=np.float64)
    period64 = np.asarray(period, dtype=np.float64)
    harmonics = np.arange(1, order + 1, dtype=np.float64)
    angles = 2 * np.pi * np.outer(time64, harmonics) / period64
    return np.column_stack((np.sin(angles), np.cos(angles)))


def test_fourier_features_is_exported():
    assert mmmjax.fourier_features is fourier_features
    assert "fourier_features" in mmmjax.__all__


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fourier_features_matches_float64_reference_on_irregular_positions(dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    time = jnp.asarray([-3.25, 0.0, 0.375, 2.0, 7.25, 11.5], dtype=dtype)
    period = jnp.asarray(8.75, dtype=dtype)
    expected = _fourier_reference(time, period, 3)
    function = partial(fourier_features, order=3)
    tolerance = 4e-6 if dtype == jnp.float32 else 2e-14

    for result in (function(time, period=period), jax.jit(function)(time, period=period)):
        assert result.shape == (6, 6)
        assert result.dtype == dtype
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


def test_fourier_features_quarter_cycle_values_and_column_order():
    expected = np.array([[0, 0, 1, 1], [1, 0, 0, -1], [0, 0, -1, 1], [-1, 0, 0, -1]])

    result = fourier_features([0, 1, 2, 3], period=4, order=2)

    # Float32 arguments near multiples of pi leave small residuals at analytic zeros.
    np.testing.assert_allclose(result, expected, rtol=0, atol=5e-7)


def test_fourier_features_regular_cycle_has_orthogonal_zero_mean_columns():
    sample_count = 64
    result = fourier_features(jnp.arange(sample_count), period=sample_count, order=5)
    features64 = np.asarray(result, dtype=np.float64)

    np.testing.assert_allclose(features64.mean(axis=0), 0, rtol=0, atol=2e-7)
    np.testing.assert_allclose(features64.T @ features64 / sample_count, np.eye(10) / 2, rtol=0, atol=5e-7)


def test_fourier_features_is_periodic():
    time = jnp.array([-2.5, 0.0, 1.25, 3.0])
    function = partial(fourier_features, period=8.0, order=3)

    np.testing.assert_allclose(function(time + 16), function(time), rtol=0, atol=8e-6)


def test_fourier_features_is_invariant_to_common_time_units():
    days = jnp.array([0.0, 7.0, 14.0, 42.0, 91.0])

    daily = fourier_features(days, period=364.0, order=3)
    weekly = fourier_features(days / 7, period=52.0, order=3)

    # Equivalent units can round angles differently; allow float32 trig error at zeros.
    np.testing.assert_allclose(daily, weekly, rtol=2e-6, atol=2e-6)


def test_fourier_features_preserves_fixed_origin_across_training_and_prediction_slices():
    time = np.array([-7.0, 0.0, 3.0, 8.0, 13.0, 14.0, 18.0], dtype=np.float32)
    function = partial(fourier_features, period=16.0, order=2)
    complete = function(time)
    training = function(time[:4])
    prediction = function(time[4:])

    np.testing.assert_array_equal(training, complete[:4])
    np.testing.assert_array_equal(prediction, complete[4:])
    np.testing.assert_allclose(prediction, _fourier_reference(time[4:], 16.0, 2), rtol=2e-6, atol=2e-6)
    assert not np.allclose(prediction[0], [0, 0, 1, 1])


@pytest.mark.parametrize("time", [[], [2.5]])
def test_fourier_features_handles_empty_and_singleton_time(time):
    result = jax.jit(partial(fourier_features, order=1))(jnp.asarray(time), period=7.0)

    assert result.shape == (len(time), 2)
    np.testing.assert_allclose(result, _fourier_reference(time, 7.0, 1), rtol=2e-6, atol=5e-7)


def test_fourier_features_jit_accepts_dynamic_time_and_period_with_static_order():
    compiled = jax.jit(fourier_features, static_argnames=("order",))
    for time, period in [([0.0, 1.5, 2.0], 4.0), ([1.0, 3.5, 5.0], 7.0)]:
        result = compiled(jnp.asarray(time), period=jnp.asarray(period), order=2)
        np.testing.assert_allclose(result, _fourier_reference(time, period, 2), rtol=2e-6, atol=2e-6)


def test_fourier_features_vectorizes_over_periods():
    time = jnp.array([-0.5, 0.0, 1.25, 3.0])
    periods = jnp.array([4.0, 8.0, 12.0])

    vectorized = jax.vmap(lambda time, period: fourier_features(time, period=period, order=2), in_axes=(None, 0))
    result = jax.jit(vectorized)(time, periods)
    expected = np.stack([_fourier_reference(time, period, 2) for period in periods])

    assert result.shape == (3, 4, 4)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_fourier_features_time_and_period_derivatives_match_analytic_gradient_and_hessian():
    arguments = jnp.array([0.7, 4.3], dtype=jnp.float32)
    sine_weights = np.array([0.25, -0.5], dtype=np.float64)
    cosine_weights = np.array([1.25, -0.75], dtype=np.float64)
    coefficients = jnp.asarray(np.concatenate((sine_weights, cosine_weights)), dtype=jnp.float32)
    time, period = np.asarray(arguments, dtype=np.float64)
    frequencies = 2 * np.pi * np.arange(1, 3, dtype=np.float64) / period
    angles = frequencies * time
    derivative_by_angle = sine_weights * np.cos(angles) - cosine_weights * np.sin(angles)
    second_by_angle = -sine_weights * np.sin(angles) - cosine_weights * np.cos(angles)
    angle_gradients = np.column_stack((frequencies, -angles / period))
    angle_hessians = np.zeros((2, 2, 2), dtype=np.float64)
    angle_hessians[:, 0, 1] = -frequencies / period
    angle_hessians[:, 1, 0] = -frequencies / period
    angle_hessians[:, 1, 1] = 2 * angles / period**2
    expected_gradient = derivative_by_angle @ angle_gradients
    expected_hessian = np.einsum("k,ki,kj->ij", second_by_angle, angle_gradients, angle_gradients)
    expected_hessian += np.einsum("k,kij->ij", derivative_by_angle, angle_hessians)

    def response(values):
        return fourier_features(values[:1], period=values[1], order=2)[0] @ coefficients

    value, gradient = jax.jit(jax.value_and_grad(response))(arguments)
    hessian = jax.jit(jax.hessian(response))(arguments)

    expected_value = (_fourier_reference([time], period, 2) @ np.asarray(coefficients, dtype=np.float64))[0]
    np.testing.assert_allclose(value, expected_value, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=5e-6, atol=2e-6)


def test_fourier_features_supports_shared_and_group_specific_coefficients():
    time = jnp.array([0.0, 1.0, 2.5, 4.0, 7.0])
    coefficients = jnp.array([[0.5, -0.25, 1.0, 0.75], [1.5, 0.25, -0.5, 0.0], [-0.5, 0.0, 0.5, 1.0]])
    features = fourier_features(time, period=8.0, order=2)
    reference = _fourier_reference(time, 8.0, 2)

    shared = jax.jit(lambda basis, weights: basis @ weights)(features, coefficients[0])
    grouped = jax.jit(lambda basis, weights: basis @ weights.T)(features, coefficients)
    gradient = jax.jit(jax.grad(lambda basis, weights: (basis @ weights.T).sum(), argnums=1))(features, coefficients)

    assert shared.shape == (5,)
    assert grouped.shape == (5, 3)
    np.testing.assert_allclose(shared, reference @ np.asarray(coefficients[0], dtype=np.float64), rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(grouped, reference @ np.asarray(coefficients, dtype=np.float64).T, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, np.broadcast_to(reference.sum(axis=0), (3, 4)), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float16, jnp.bfloat16, jnp.float32])
def test_fourier_features_uses_at_least_float32(dtype):
    result = fourier_features(jnp.asarray([0, 1, 2], dtype=dtype), period=dtype(8), order=1)

    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, _fourier_reference([0, 1, 2], 8, 1), rtol=2e-6, atol=5e-7)


def test_fourier_features_uses_common_float64_dtype_when_enabled():
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    time = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    period = jnp.asarray(7.0, dtype=jnp.float64)

    result = fourier_features(time, period=period, order=2)

    assert result.dtype == jnp.float64
    np.testing.assert_allclose(result, _fourier_reference(time, period, 2), rtol=2e-14, atol=2e-14)


def test_fourier_features_converts_large_integer_positions_before_narrowing():
    time = np.array([0, 3_000_000_000, 6_000_000_000], dtype=np.int64)

    result = fourier_features(time, period=12_000_000_000, order=1)

    np.testing.assert_allclose(result, _fourier_reference(time, 12_000_000_000, 1), rtol=2e-6, atol=5e-7)


def test_fourier_features_masks_only_nonfinite_time_rows():
    time = jnp.array([-1.0, jnp.nan, 0.0, jnp.inf, -jnp.inf, 2.5])
    valid_indices = [0, 2, 5]
    function = partial(fourier_features, period=7.0, order=2)

    for result in (function(time), jax.jit(function)(time)):
        np.testing.assert_allclose(
            np.asarray(result)[valid_indices],
            _fourier_reference(np.asarray(time)[valid_indices], 7.0, 2),
            rtol=2e-6,
            atol=2e-6,
        )
        assert np.isnan(np.asarray(result)[[1, 3, 4]]).all()


@pytest.mark.parametrize("period", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_fourier_features_invalid_scalar_period_masks_all_features(period):
    function = partial(fourier_features, order=2)
    time = jnp.array([-1.0, 0.0, 1.0])

    for result in (function(time, period=period), jax.jit(function)(time, period=period)):
        assert result.shape == (3, 4)
        assert np.isnan(result).all()


@pytest.mark.parametrize(
    "argument,value",
    [
        ("time", [True, False]),
        ("time", [1 + 2j]),
        ("time", ["1", "2"]),
        ("time", np.array([1], dtype=object)),
        ("period", True),
        ("period", 1 + 2j),
        ("period", "7"),
    ],
)
def test_fourier_features_names_nonreal_inputs_in_errors(argument, value):
    arguments = {"time": [0.0, 1.0], "period": 7.0, "order": 2}
    arguments[argument] = value

    with pytest.raises(TypeError, match=argument):
        fourier_features(**arguments)


@pytest.mark.parametrize("argument,values", [("time", [1.0, [[1.0], [2.0]]]), ("period", [[7.0], [[7.0]]])])
def test_fourier_features_rejects_invalid_input_dimensions(argument, values):
    for value in values:
        arguments = {"time": [0.0, 1.0], "period": 7.0, "order": 2}
        arguments[argument] = value
        with pytest.raises(ValueError, match=argument):
            fourier_features(**arguments)


@pytest.mark.parametrize("order", [True, 1.5, np.int64(2)])
def test_fourier_features_requires_python_integer_order(order):
    with pytest.raises(TypeError, match="order"):
        fourier_features([0.0, 1.0], period=7.0, order=order)


@pytest.mark.parametrize("order", [0, -1])
def test_fourier_features_requires_positive_order(order):
    with pytest.raises(ValueError, match="order"):
        fourier_features([0.0, 1.0], period=7.0, order=order)


def test_fourier_features_explains_that_jit_order_must_be_static():
    with pytest.raises(TypeError, match="order"):
        jax.jit(fourier_features)(jnp.array([0.0, 1.0]), period=7.0, order=2)
