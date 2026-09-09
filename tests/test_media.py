"""Tests for composing media transformations over observed exposure history."""

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

import mmmjax
from mmmjax import geometric_adstock, hill_saturation, media_response


@pytest.fixture
def prepared_media():
    exposures = np.array(
        [
            [[80, 10], [20, 40]],
            [[40, 20], [80, 10]],
            [[0, 40], [20, 20]],
            [[160, 80], [40, 0]],
            [[80, 20], [40, 40]],
        ],
        dtype=float,
    )
    frame = pl.DataFrame(
        {
            "week": [3, 3, 4, 4, 5, 5],
            "region": ["west", "east"] * 3,
            "video": exposures[2:, :, 0].ravel(),
            "search": exposures[2:, :, 1].ravel(),
            "sales": [0.5, 1.0, 1.5, 0.8, 1.0, 1.2],
        }
    )
    history = pl.DataFrame(
        {
            "week": [1, 1, 2, 2],
            "region": ["west", "east"] * 2,
            "video": exposures[:2, :, 0].ravel(),
            "search": exposures[:2, :, 1].ravel(),
        }
    )
    return mmmjax.prepare_data(
        frame, time="week", groups=["region"], outcome="sales", media=["video", "search"], media_history=history
    )


def _reference_response(exposures, alpha, half, *, max_lag, adstock_first):
    weights = np.asarray(alpha) ** np.arange(max_lag + 1)[:, None]
    weights = weights / weights.sum(axis=0)
    source = exposures if adstock_first else exposures / (exposures + half)
    carried = np.zeros_like(exposures, dtype=float)
    for time in range(len(exposures)):
        for lag in range(min(time, max_lag) + 1):
            carried[time] += source[time - lag] * weights[lag]
    return carried / (carried + half) if adstock_first else carried


@pytest.mark.parametrize("adstock_first", [False, True])
def test_media_history_composes_with_scaling_and_learned_model_parameters(prepared_media, adstock_first):
    original = deepcopy(prepared_media)
    scaled = mmmjax.fit_data_scaling(prepared_media, media_method="max").transform(prepared_media)
    n_periods = len(scaled.time_values)

    def log_density(data, alpha, half):
        response = media_response(
            data["media"],
            adstock=partial(geometric_adstock, alpha=alpha, max_lag=3),
            saturation=partial(hill_saturation, half_saturation=half, slope=1.0),
            n_periods=n_periods,
            adstock_first=adstock_first,
        )
        return mmmjax.normal(data["outcome"], (response * jnp.array([0.7, 1.3])).sum(axis=-1), 1.0)

    model = mmmjax.Model(
        {"alpha": mmmjax.Interval(0.0, 1.0, shape=(2,)), "half": mmmjax.Positive(shape=(2,))}, log_density
    )

    def objective(position, data):
        return model.log_density({"alpha": position[:2], "half": position[2:]}, data)

    exposures = original.arrays["media"] / original.arrays["media"].max(axis=(0, 1))

    def reference(position):
        alpha, half = 1 / (1 + np.exp(-position[:2])), np.exp(position[2:])
        response = _reference_response(exposures, alpha, half, max_lag=3, adstock_first=adstock_first)[-n_periods:]
        residual = original.arrays["outcome"] - (response * [0.7, 1.3]).sum(axis=-1)
        likelihood = (-0.5 * residual**2 - 0.5 * np.log(2 * np.pi)).sum()
        return likelihood + (np.log(alpha) + np.log1p(-alpha)).sum() + position[2:].sum()

    position = np.array([-0.8, 0.4, -0.2, 0.3])
    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(position), scaled._to_jax())
    # Finite differences use the independent causal sum and include the parameter Jacobians.
    shifts = np.eye(4) * 1e-4
    expected_gradient = np.array([(reference(position + step) - reference(position - step)) / 2e-4 for step in shifts])
    np.testing.assert_allclose(value, reference(position), rtol=3e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-6)
    assert scaled.arrays["media"].shape == (5, 2, 2)
    assert scaled.arrays["outcome"].shape == (3, 2)
    for name, values in original.arrays.items():
        np.testing.assert_array_equal(prepared_media.arrays[name], values)


def test_prediction_reuses_scaling_and_aligns_reordered_groups_and_channels(prepared_media):
    fitted = mmmjax.fit_data_scaling(prepared_media, media_method="max")
    prediction = mmmjax.prepare_data(
        pl.DataFrame(
            {
                "week": [7, 6, 7, 6],
                "region": ["east", "west", "west", "east"],
                "search": [40, 10, 20, 20],
                "video": [160, 40, 80, 80],
            }
        ),
        time="week",
        groups=["region"],
        media=["search", "video"],
        media_history=pl.DataFrame({"week": [5, 5], "region": ["east", "west"], "search": [10, 5], "video": [40, 20]}),
    )
    original = deepcopy(prediction)
    scaled = fitted.transform(prediction)
    alpha, half = np.array([0.2, 0.6]), np.array([0.5, 1.5])
    response = media_response(
        scaled.arrays["media"],
        adstock=partial(geometric_adstock, alpha=alpha, max_lag=2),
        saturation=partial(hill_saturation, half_saturation=half, slope=1.0),
        n_periods=len(scaled.time_values),
    )
    exposures = np.array([[[20, 5], [40, 10]], [[40, 10], [80, 20]], [[80, 20], [160, 40]]]) / [160, 80]
    expected = _reference_response(exposures, alpha, half, max_lag=2, adstock_first=True)[1:]
    np.testing.assert_allclose(response, expected, rtol=2e-6)
    assert response.shape == (2, 2, 2)
    assert scaled.time_values == (6, 7)
    assert scaled.channels == ("video", "search")
    assert scaled.group_values == (("west",), ("east",))
    assert "outcome" not in scaled.arrays
    assert prediction.channels == original.channels
    assert prediction.group_values == original.group_values
    np.testing.assert_array_equal(prediction.arrays["media"], original.arrays["media"])


@pytest.mark.parametrize("adstock_first", [False, True])
@pytest.mark.parametrize("max_lag", [0, 2, 7])
def test_media_without_history_and_full_window_normalization(adstock_first, max_lag):
    exposures = np.array([[0.0, 2.0], [8.0, 0.0], [2.0, 4.0]])
    alpha, half = np.array([0.25, 0.6]), np.array([1.0, 2.0])
    function = partial(
        media_response,
        adstock=partial(geometric_adstock, alpha=alpha, max_lag=max_lag),
        saturation=partial(hill_saturation, half_saturation=half, slope=1.0),
        adstock_first=adstock_first,
    )
    expected = _reference_response(exposures, alpha, half, max_lag=max_lag, adstock_first=adstock_first)
    np.testing.assert_allclose(function(exposures), expected, rtol=2e-6)
    np.testing.assert_allclose(
        jax.jit(function, static_argnames="n_periods")(exposures, n_periods=3), expected, rtol=2e-6
    )


def test_media_response_vectorization_retains_group_and_channel_axes():
    exposures = np.arange(3 * 8 * 465, dtype=float).reshape(3, 8, 465) / 1000
    alphas = jnp.linspace(0.1, 0.9, 465)[None, :] * jnp.array([[0.4], [0.8]])
    half = np.linspace(0.5, 1.5, 8)[:, None]

    def response(alpha):
        return media_response(
            exposures,
            adstock=partial(geometric_adstock, alpha=alpha, max_lag=1),
            saturation=partial(hill_saturation, half_saturation=half, slope=1.0),
            n_periods=2,
        )

    actual = jax.jit(jax.vmap(response))(alphas)
    expected = np.stack([_reference_response(exposures, a, half, max_lag=1, adstock_first=True)[1:] for a in alphas])
    assert actual.shape == (2, 2, 8, 465)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("adstock_first", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_media_response_preserves_primitive_dtype_promotion(adstock_first, dtype):
    if dtype == np.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    result = media_response(
        np.array([3.0, 6.0], dtype=dtype),
        adstock=partial(geometric_adstock, alpha=0.5, max_lag=0),
        saturation=partial(hill_saturation, half_saturation=3.0, slope=1.0),
        adstock_first=adstock_first,
    )
    assert result.dtype == jnp.result_type(dtype, jnp.float32)
    np.testing.assert_allclose(result, [0.5, 2 / 3], rtol=2e-6)


@pytest.mark.parametrize("adstock_first", [False, True])
def test_media_response_leaves_large_integer_conversion_to_first_primitive(adstock_first):
    result = media_response(
        np.array([3_000_000_000, 6_000_000_000], dtype=np.int64),
        adstock=partial(geometric_adstock, alpha=0.5, max_lag=0),
        saturation=partial(hill_saturation, half_saturation=3_000_000_000, slope=1.0),
        adstock_first=adstock_first,
    )
    assert jnp.issubdtype(result.dtype, jnp.floating)
    np.testing.assert_allclose(result, [0.5, 2 / 3], rtol=2e-6)


def test_media_response_export_and_empty_series():
    from mmmjax.media import media_response as module_function

    assert media_response is module_function
    assert "media_response" in mmmjax.__all__
    result = media_response(np.empty((0, 2)), adstock=jnp.asarray, saturation=jnp.asarray)
    assert isinstance(result, jax.Array)
    assert result.shape == (0, 2)


@pytest.mark.parametrize("count", [True, 1.5, np.int64(1), jnp.array(1)])
def test_media_response_rejects_nonstatic_or_noninteger_period_count(count):
    with pytest.raises(TypeError, match="n_periods"):
        media_response([1.0, 2.0], adstock=jnp.asarray, saturation=jnp.asarray, n_periods=count)


@pytest.mark.parametrize("count", [0, -1, 3])
def test_media_response_rejects_period_count_outside_observation_window(count):
    with pytest.raises(ValueError, match="n_periods"):
        media_response([1.0, 2.0], adstock=jnp.asarray, saturation=jnp.asarray, n_periods=count)


def test_media_response_rejects_dynamic_period_count_under_jit():
    function = partial(media_response, adstock=jnp.asarray, saturation=jnp.asarray)
    with pytest.raises(TypeError, match="n_periods"):
        jax.jit(function)(jnp.ones(3), n_periods=2)


@pytest.mark.parametrize("name", ["adstock", "saturation"])
def test_media_response_requires_callable_transformations(name):
    callbacks = {"adstock": jnp.asarray, "saturation": jnp.asarray, name: None}
    with pytest.raises(TypeError, match=name):
        media_response([1.0, 2.0], **callbacks)


@pytest.mark.parametrize("name", ["adstock", "saturation"])
@pytest.mark.parametrize("adstock_first", [False, True])
def test_media_response_rejects_each_callback_changing_full_input_shape(name, adstock_first):
    callbacks = {"adstock": jnp.asarray, "saturation": jnp.asarray, name: lambda values: values[1:]}
    with pytest.raises(ValueError, match=rf"{name}.*shape"):
        media_response(np.ones((3, 2)), **callbacks, n_periods=2, adstock_first=adstock_first)


@pytest.mark.parametrize("order", [0, None, "adstock", jnp.array(True)])
def test_media_response_requires_static_boolean_order(order):
    with pytest.raises(TypeError, match="adstock_first"):
        media_response([1.0, 2.0], adstock=jnp.asarray, saturation=jnp.asarray, adstock_first=order)


def test_media_response_requires_time_axis():
    with pytest.raises(ValueError, match=r"media.*dimension"):
        media_response(1.0, adstock=jnp.asarray, saturation=jnp.asarray)
