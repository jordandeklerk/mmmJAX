"""Tests for fitted centering and scaling."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import mmmjax
from mmmjax.scaling import Scaling, fit_scaling


def test_scaling_exports_public_entry_points():
    assert mmmjax.Scaling is Scaling
    assert mmmjax.fit_scaling is fit_scaling
    assert {"Scaling", "fit_scaling"}.issubset(mmmjax.__all__)


def test_scaling_stores_population_statistics_with_reduced_axes_retained():
    values = [[0, 2], [2, 6], [4, 10]]

    fitted = fit_scaling(values)

    assert fitted.offset.shape == fitted.scale.shape == (1, 2)
    assert isinstance(fitted.offset, jax.Array)
    assert isinstance(fitted.scale, jax.Array)
    np.testing.assert_array_equal(fitted.offset, [[2, 6]])
    np.testing.assert_allclose(fitted.scale, [[np.sqrt(8 / 3), np.sqrt(32 / 3)]], rtol=1e-7)
    with pytest.raises(FrozenInstanceError):
        fitted.offset = jnp.zeros((1, 2))


@pytest.mark.parametrize("axis", [0, 1, -1, np.int64(0)])
def test_scaling_matches_scipy_standardization(axis):
    values = np.array([[1, 5, 2], [4, 7, 3], [8, 12, 9], [3, 6, 4]], dtype=np.float32)
    fitted = fit_scaling(values, axis=axis)

    transformed = fitted.transform(values)

    np.testing.assert_allclose(transformed, stats.zscore(values, axis=axis, ddof=0), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(fitted.inverse_transform(transformed), values, rtol=2e-6, atol=1e-7)


def test_scaling_prediction_reuses_training_statistics_without_refitting():
    training = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    prediction = np.array([[20.0, 40.0], [21.0, 44.0]])
    fitted = fit_scaling(training)

    result = fitted.transform(prediction)

    assert result.shape == prediction.shape
    np.testing.assert_allclose(result, stats.zmap(prediction, training, ddof=0), rtol=2e-6)
    np.testing.assert_array_equal(fitted.offset, [[3.0, 14.0]])
    np.testing.assert_allclose(fitted.inverse_transform(result), prediction, rtol=2e-6)


@pytest.mark.parametrize("center,scale", [(False, False), (False, True), (True, False), (True, True)])
def test_scaling_flags_control_offset_and_standard_deviation_separately(center, scale):
    fitted = fit_scaling([[2.0, 10.0], [6.0, 22.0]], center=center, scale=scale)
    offset = np.array([[4.0, 16.0]]) if center else np.zeros((1, 2))
    divisor = np.array([[2.0, 6.0]]) if scale else np.ones((1, 2))

    np.testing.assert_array_equal(fitted.offset, offset)
    np.testing.assert_array_equal(fitted.scale, divisor)
    np.testing.assert_allclose(fitted.transform([[8.0, 28.0]]), (np.array([[8.0, 28.0]]) - offset) / divisor)


def test_scaling_constant_columns_keep_changes_in_new_values():
    fitted = fit_scaling([[7.0, 0.0, 2.0], [7.0, 0.0, 6.0]])

    np.testing.assert_array_equal(fitted.scale, [[1.0, 1.0, 2.0]])
    np.testing.assert_array_equal(fitted.transform([[8.0, 3.0, 8.0]]), [[1.0, 3.0, 2.0]])
    np.testing.assert_array_equal(fitted.inverse_transform([[1.0, 3.0, 2.0]]), [[8.0, 3.0, 8.0]])


@pytest.mark.parametrize("count,constant", [(104, 9.99), (3, 0.1)])
def test_scaling_decimal_constants_do_not_acquire_a_spurious_standard_deviation(count, constant):
    training = np.column_stack([np.full(count, constant), np.arange(count)])
    fitted = fit_scaling(training)
    varying_mean = (count - 1) / 2
    varying_scale = np.sqrt((count**2 - 1) / 12)
    future = np.array([[constant + 2, varying_mean + 2 * varying_scale]])

    np.testing.assert_array_equal(fitted.scale[:, 0], [1.0])
    np.testing.assert_array_equal(fitted.transform(training)[:, 0], np.zeros(count))
    np.testing.assert_allclose(fitted.scale[:, 1], [varying_scale], rtol=1e-7)
    np.testing.assert_allclose(fitted.transform(future), [[2.0, 2.0]], rtol=2e-6)
    np.testing.assert_allclose(fitted.inverse_transform([[2.0, 2.0]]), future, rtol=2e-6)


@pytest.mark.parametrize("axis", [(0, 1), (-3, -2), (np.int64(1), np.int64(0))])
def test_scaling_multiple_axes_share_statistics_without_mixing_channels(axis):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    fitted = fit_scaling(values, axis=axis)

    assert fitted.offset.shape == fitted.scale.shape == (1, 1, 4)
    np.testing.assert_array_equal(fitted.offset, [[[10, 11, 12, 13]]])
    np.testing.assert_allclose(fitted.scale, np.full((1, 1, 4), np.sqrt(140 / 3)), rtol=1e-7)


def test_scaling_all_axes_and_single_observation():
    fitted = fit_scaling(np.arange(24).reshape(2, 3, 4), axis=None)

    assert fitted.offset.shape == fitted.scale.shape == (1, 1, 1)
    np.testing.assert_array_equal(fitted.offset, [[[11.5]]])
    np.testing.assert_allclose(fitted.scale, [[[np.sqrt(575 / 12)]]], rtol=1e-7)
    single = fit_scaling([9.0])
    np.testing.assert_array_equal(single.transform([9.0, 10.0]), [0.0, 1.0])


def test_scaling_supports_prediction_batches_and_parameter_draws():
    fitted = fit_scaling([[2.0, 10.0], [6.0, 22.0]])
    draws = jnp.arange(30, dtype=jnp.float32).reshape(5, 3, 2)
    expected = (np.asarray(draws) - [4.0, 16.0]) / [2.0, 6.0]

    result = jax.jit(jax.vmap(fitted.transform))(draws)

    np.testing.assert_allclose(result, expected, rtol=1e-7)
    np.testing.assert_allclose(fitted.transform(draws), expected, rtol=1e-7)
    np.testing.assert_allclose(jax.jit(fitted.inverse_transform)(result), draws, atol=1e-6)


def test_scaling_is_a_dynamic_pytree_and_supports_gradients():
    fitted = fit_scaling([[2.0, 10.0], [6.0, 22.0]])
    values = jnp.array([[8.0, 28.0]])
    leaves, structure = jax.tree_util.tree_flatten(fitted)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert len(leaves) == 2
    transformed = jax.jit(lambda scaling, x: scaling.transform(x))(restored, values)
    np.testing.assert_array_equal(transformed, [[2.0, 2.0]])
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(fitted.transform(x) ** 2)))(values)
    np.testing.assert_allclose(gradient, [[2.0, 2 / 3]], rtol=1e-7)
    inverse_gradient = jax.jit(jax.grad(lambda x: fitted.inverse_transform(x).sum()))(values)
    np.testing.assert_array_equal(inverse_gradient, [[2.0, 6.0]])


def test_scaling_does_not_keep_a_mutable_view_of_training_values():
    training = np.array([[2.0, 10.0], [6.0, 22.0]], dtype=np.float32)
    original = training.copy()
    fitted = fit_scaling(training)

    np.testing.assert_array_equal(training, original)
    training[...] = -100

    np.testing.assert_array_equal(fitted.offset, [[4.0, 16.0]])
    np.testing.assert_array_equal(fitted.scale, [[2.0, 6.0]])
    np.testing.assert_array_equal(fitted.transform(original), [[-1.0, -1.0], [1.0, 1.0]])


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float16, np.float32, np.float64])
def test_scaling_statistics_follow_jax_precision_policy(dtype):
    fitted = fit_scaling(np.array([0, 1], dtype=dtype))
    expected = np.float64 if jax.config.x64_enabled and dtype in (np.int64, np.float64) else np.float32

    assert fitted.offset.dtype == fitted.scale.dtype == expected
    np.testing.assert_array_equal(fitted.transform(np.array([0, 1], dtype=dtype)), [-1.0, 1.0])


def test_scaling_preserves_float64_statistics_when_enabled():
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    values = np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10], dtype=np.float64)
    fitted = fit_scaling(values)

    np.testing.assert_allclose(fitted.transform(values), stats.zscore(values), rtol=1e-14)
    assert fitted.transform(values).dtype == jnp.float64


@pytest.mark.parametrize("axis", [True, 0.0, "time", [], (), (False,)])
def test_scaling_rejects_noninteger_or_empty_reduction_axes(axis):
    with pytest.raises(TypeError, match="axis"):
        fit_scaling(np.ones((3, 2)), axis=axis)


@pytest.mark.parametrize("axis", [(0, 0), (0, -2), 2, -3])
def test_scaling_rejects_repeated_or_out_of_bounds_reduction_axes(axis):
    with pytest.raises(ValueError, match="axis"):
        fit_scaling(np.ones((3, 2)), axis=axis)


@pytest.mark.parametrize("flag", ["center", "scale"])
@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_scaling_rejects_nonboolean_options(flag, value):
    with pytest.raises(TypeError, match=flag):
        fit_scaling([1.0, 2.0], **{flag: value})


@pytest.mark.parametrize("values", [1.0, [], np.empty((2, 0)), [1.0, np.nan], [np.inf], [-np.inf]])
def test_scaling_rejects_missing_or_nonfinite_training_values(values):
    with pytest.raises(ValueError, match="values"):
        fit_scaling(values)


def test_scaling_reports_training_values_whose_statistics_overflow():
    largest = np.finfo(np.float64).max

    with pytest.raises(ValueError, match=r"values.*finite.*scale"):
        fit_scaling([-largest, largest])


@pytest.mark.parametrize("values", [["1", "2"], [1 + 2j], np.array([1, 2], dtype=object)])
def test_scaling_rejects_nonreal_numeric_values(values):
    with pytest.raises(TypeError, match="values"):
        fit_scaling(values)
    fitted = fit_scaling([1.0, 2.0])
    for method in (fitted.transform, fitted.inverse_transform):
        with pytest.raises(TypeError, match="values"):
            method(values)


@pytest.mark.parametrize("shape", [(), (2,), (3, 1), (3, 3)])
def test_scaling_rejects_broadcasting_that_would_change_input_shape(shape):
    fitted = fit_scaling([[2.0, 10.0], [6.0, 22.0]])
    values = jnp.ones(shape)

    for method in (fitted.transform, fitted.inverse_transform):
        with pytest.raises(ValueError, match="shape"):
            method(values)


def test_scaling_transform_leaves_nonfinite_predictions_visible():
    fitted = fit_scaling([2.0, 6.0])
    values = jnp.array([jnp.nan, jnp.inf, -jnp.inf])

    for method in (fitted.transform, fitted.inverse_transform):
        result = jax.jit(method)(values)
        assert np.isnan(result[0])
        assert np.isposinf(result[1])
        assert np.isneginf(result[2])
