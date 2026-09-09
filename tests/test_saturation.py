"""Tests for bounded media-response transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scipy import stats

import mmmjax
from mmmjax.saturation import hill_saturation, logistic_saturation


def test_hill_saturation_is_exported():
    assert mmmjax.hill_saturation is hill_saturation
    assert "hill_saturation" in mmmjax.__all__


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("slope", [0.5, 1.0, 2.0, 4.5])
def test_hill_saturation_matches_direct_equation(dtype, slope):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    media = jnp.asarray([0.0, 0.125, 0.5, 1.0, 3.0, 12.0], dtype=dtype)
    half_saturation = jnp.asarray(2.0, dtype=dtype)
    media64 = np.asarray(media, dtype=np.float64)
    expected = media64**slope / (media64**slope + 2.0**slope)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14

    for function in (hill_saturation, jax.jit(hill_saturation)):
        result = function(media, half_saturation, slope)
        assert result.dtype == dtype
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)


def test_hill_saturation_has_zero_origin_half_response_and_unit_limit():
    result = hill_saturation(jnp.array([0.0, 4.0, 4e6]), 4.0, 2.0)
    np.testing.assert_array_equal(result[:2], [0.0, 0.5])
    np.testing.assert_allclose(result[2], 1.0, rtol=1e-10)
    np.testing.assert_allclose(hill_saturation([1.0, 2.0, 6.0], 2.0, 1.0), [1 / 3, 1 / 2, 3 / 4])


@pytest.mark.parametrize("slope", [0.4, 1.0, 3.0, 8.0])
def test_hill_saturation_matches_scipy_fisk_for_widely_spaced_exposures(slope):
    # The Fisk CDF is the same bounded response and provides an independent implementation
    media = np.geomspace(1e-12, 1e12, 41).astype(np.float32)
    expected = stats.fisk.cdf(media.astype(np.float64), c=slope, scale=1e4)
    result = jax.jit(hill_saturation)(media, 1e4, slope)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=1e-37)
    assert np.all(np.diff(result) >= 0)
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.parametrize("magnitude", [1e-20, 1.0, 1e20])
def test_hill_saturation_is_invariant_to_common_units(magnitude):
    media = jnp.asarray([0.0, 1.0, 2.0, 4.0]) * magnitude
    result = jax.jit(hill_saturation)(media, 2.0 * magnitude, 4.0)

    np.testing.assert_allclose(result, [0.0, 1 / 17, 1 / 2, 16 / 17], rtol=2e-6, atol=0)


@pytest.mark.parametrize("group_specific", [False, True])
def test_hill_saturation_broadcasts_across_hundreds_of_channels(group_specific):
    half_saturation = np.linspace(1e5, 2e6, 465, dtype=np.float32)
    if group_specific:
        half_saturation = np.arange(1, 9, dtype=np.float32)[:, None] * half_saturation
    ratios = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)[:, None, None]
    media = np.broadcast_to(ratios * half_saturation, (4, 8, 465))
    slope = np.where(np.arange(465) % 2 == 0, 1.0, 2.0).astype(np.float32)
    expected = np.broadcast_to(ratios**slope / (1 + ratios**slope), media.shape)

    result = jax.jit(hill_saturation)(media, half_saturation, slope)

    assert result.shape == media.shape
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_hill_saturation_allows_general_broadcasting_and_scalar_inputs():
    result = hill_saturation(2.0, np.array([[1.0], [2.0]]), np.array([1.0, 2.0, 3.0]))

    assert hill_saturation(2.0, 2.0, 1.0).shape == ()
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, [[2 / 3, 4 / 5, 8 / 9], [0.5, 0.5, 0.5]], rtol=1e-6)


def test_hill_saturation_vectorizes_parameter_draws():
    media = jnp.array([[0.0, 1.0], [2.0, 4.0], [3.0, 6.0]])
    half_saturation = jnp.array([[0.5, 1.0], [1.0, 2.0], [2.0, 4.0]])
    slopes = jnp.array([[0.5, 1.0], [1.0, 2.0], [2.0, 3.0]])
    expected = np.stack(
        [
            np.asarray(media, dtype=np.float64) ** s / (np.asarray(media, dtype=np.float64) ** s + k**s)
            for k, s in zip(np.asarray(half_saturation), np.asarray(slopes), strict=True)
        ]
    )

    result = jax.jit(jax.vmap(hill_saturation, in_axes=(None, 0, 0)))(media, half_saturation, slopes)

    assert result.shape == (3, 3, 2)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("media,half_saturation,slope", [(0.3, 2.0, 0.5), (2.0, 2.0, 1.5), (8.0, 3.0, 2.0)])
def test_hill_saturation_gradients_match_analytic_derivatives(media, half_saturation, slope):
    response = media**slope / (media**slope + half_saturation**slope)
    common = response * (1 - response)
    expected = np.array(
        [slope * common / media, -slope * common / half_saturation, common * np.log(media / half_saturation)]
    )
    arguments = tuple(jnp.asarray(value) for value in (media, half_saturation, slope))

    gradient = jax.jit(jax.grad(hill_saturation, argnums=(0, 1, 2)))(*arguments)
    _, tangent = jax.jit(lambda *args: jax.jvp(hill_saturation, args, (0.2, -0.3, 0.4)))(*arguments)

    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=2e-8)
    np.testing.assert_allclose(tangent, expected @ np.array([0.2, -0.3, 0.4]), rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize("slope,media_derivative", [(0.5, np.inf), (1.0, 0.25), (2.0, 0.0)])
def test_hill_saturation_derivatives_at_zero_follow_right_hand_limits(slope, media_derivative):
    arguments = (jnp.asarray(0.0), jnp.asarray(4.0), jnp.asarray(slope))
    gradients = jax.jit(jax.grad(hill_saturation, argnums=(0, 1, 2)))(*arguments)
    np.testing.assert_array_equal(gradients, [media_derivative, 0.0, 0.0])

    def vary_parameters(half_saturation, exponent):
        return hill_saturation(0.0, half_saturation, exponent)

    _, tangent = jax.jit(lambda k, s: jax.jvp(vary_parameters, (k, s), (0.3, 0.4)))(*arguments[1:])
    assert tangent == 0


@pytest.mark.parametrize("slope", [0.5, 1.0, 3.0])
def test_hill_saturation_midpoint_hessian_is_smooth(slope):
    half_saturation = 2.0
    expected = np.array([[-slope / 16, 0, 1 / 8], [0, slope / 16, -1 / 8], [1 / 8, -1 / 8, 0]])

    def response(arguments):
        return hill_saturation(*arguments)

    result = jax.jit(jax.hessian(response))(jnp.array([half_saturation, half_saturation, slope]))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize("argument", ["media", "half_saturation", "slope"])
@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf, -np.inf])
def test_hill_saturation_masks_invalid_values_without_affecting_valid_cells(argument, bad_value):
    arguments = {"media": jnp.array([1.0, 2.0]), "half_saturation": jnp.array([2.0, 2.0]), "slope": jnp.ones(2)}
    arguments[argument] = arguments[argument].at[1].set(bad_value)

    result = jax.jit(hill_saturation)(**arguments)
    gradient = jax.jit(jax.grad(lambda x: hill_saturation(x, arguments["half_saturation"], arguments["slope"]).sum()))(
        arguments["media"]
    )

    np.testing.assert_allclose(result[0], 1 / 3, rtol=1e-6)
    assert np.isnan(result[1])
    np.testing.assert_allclose(gradient[0], 2 / 9, rtol=1e-6)


@pytest.mark.parametrize("argument", ["half_saturation", "slope"])
def test_hill_saturation_requires_strictly_positive_parameters(argument):
    arguments = {"media": [0.0, 1.0], "half_saturation": 2.0, "slope": 1.0}
    arguments[argument] = 0.0
    assert np.isnan(hill_saturation(**arguments)).all()


@pytest.mark.parametrize("argument", ["media", "half_saturation", "slope"])
@pytest.mark.parametrize("value", [["1", "2"], [1 + 2j], np.array([1], dtype=object)])
def test_hill_saturation_names_nonreal_inputs_in_errors(argument, value):
    arguments = {"media": [1.0, 2.0], "half_saturation": 2.0, "slope": 1.0}
    arguments[argument] = value
    with pytest.raises(TypeError, match=argument):
        hill_saturation(**arguments)


def test_hill_saturation_explains_incompatible_shapes():
    with pytest.raises(ValueError) as caught:
        hill_saturation(np.ones((3, 2)), np.ones(4), np.ones(2))
    message = str(caught.value)
    for detail in ("media", "half_saturation", "slope", "(3, 2)", "(4,)", "(2,)"):
        assert detail in message


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float16, np.float32])
def test_hill_saturation_promotes_low_precision_and_integer_inputs(dtype):
    result = hill_saturation(np.array([1, 2, 4], dtype=dtype), 2, 1)
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, [1 / 3, 1 / 2, 2 / 3], rtol=1e-6)


def test_hill_saturation_converts_large_counts_before_narrowing_integers():
    media = np.array([3_000_000_000, 6_000_000_000], dtype=np.int64)
    np.testing.assert_allclose(hill_saturation(media, 3_000_000_000, 1), [1 / 2, 2 / 3], rtol=1e-6)


def test_hill_saturation_composes_with_prepared_data_scaling_adstock_and_model():
    frame = pl.DataFrame(
        {"week": [1, 2, 3], "sales": [1.0, 1.5, 2.0], "video": [10.0, 20.0, 30.0], "search": [20.0, 40.0, 80.0]}
    )
    prepared = mmmjax.prepare_data(frame, time="week", outcome="sales", media=["video", "search"])
    scaled = mmmjax.fit_data_scaling(prepared).transform(prepared)

    def log_density(data, half_saturation, slope):
        carried = mmmjax.geometric_adstock(data["media"], 0.5, max_lag=1)
        prediction = hill_saturation(carried, half_saturation, slope).sum(axis=-1)
        return mmmjax.normal(data["outcome"], prediction, 1.0)

    model = mmmjax.Model(
        {"half_saturation": mmmjax.Positive(shape=(2,)), "slope": mmmjax.Positive(shape=(2,))}, log_density
    )
    position = {"half_saturation": jnp.zeros(2), "slope": jnp.zeros(2)}
    value, gradients = jax.jit(jax.value_and_grad(model.log_density))(position, scaled._to_jax())
    media = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 2.0]])
    carried = np.stack([np.convolve(column, [1.0, 0.5])[:3] / 1.5 for column in media.T], axis=-1)
    expected_prediction = (carried / (carried + 1)).sum(axis=-1)
    expected = stats.norm.logpdf([1.0, 1.5, 2.0], loc=expected_prediction).sum()

    np.testing.assert_allclose(value, expected, rtol=2e-6)
    for gradient in gradients.values():
        assert gradient.shape == (2,)
        assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(prepared.arrays["media"], [[10, 20], [20, 40], [30, 80]])


def test_logistic_saturation_is_exported():
    assert mmmjax.logistic_saturation is logistic_saturation
    assert "logistic_saturation" in mmmjax.__all__


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_logistic_saturation_matches_exponential_equation_and_scipy(dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    media = jnp.asarray([0.0, 0.125, 1.0, 2.0, 4.0, 20.0], dtype=dtype)
    decay = np.exp(-np.log(3) * np.asarray(media, dtype=np.float64) / 2)
    expected = (1 - decay) / (1 + decay)
    scipy_expected = stats.halflogistic.cdf(np.asarray(media, dtype=np.float64), scale=2 / np.log(3))
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14

    for function in (logistic_saturation, jax.jit(logistic_saturation)):
        result = function(media, 2.0)
        assert result.dtype == dtype
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)
        np.testing.assert_allclose(result, scipy_expected, rtol=tolerance, atol=0)
        assert result[0] == 0
        np.testing.assert_allclose(result[3], 0.5, rtol=tolerance)
        assert np.all(np.diff(result) > 0)


def test_logistic_saturation_preserves_small_responses_and_approaches_one():
    media = jnp.array([1e-12, 1e6], dtype=jnp.float32)
    result = jax.jit(logistic_saturation)(media, 1.0)

    # At this exposure, the first-order limit has negligible cubic truncation error
    np.testing.assert_allclose(result[0], np.log(3) * float(media[0]) / 2, rtol=2e-6, atol=0)
    assert result[1] == 1


@pytest.mark.parametrize("magnitude", [1e-6, 1e6])
def test_logistic_saturation_is_invariant_to_common_units(magnitude):
    media = jnp.array([0.0, 2.0, 4.0]) * magnitude
    result = jax.jit(logistic_saturation)(media, 2.0 * magnitude)

    np.testing.assert_allclose(result, [0.0, 0.5, 0.8], rtol=2e-6, atol=0)


@pytest.mark.parametrize("group_specific", [False, True])
def test_logistic_saturation_broadcasts_across_hundreds_of_channels(group_specific):
    half = np.linspace(1e5, 2e6, 465, dtype=np.float32)
    if group_specific:
        half = np.arange(1, 9, dtype=np.float32)[:, None] * half
    ratios = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)[:, None, None]
    media = np.broadcast_to(ratios * half, (4, 8, 465))
    expected = np.broadcast_to((1 - 3.0 ** (-ratios)) / (1 + 3.0 ** (-ratios)), media.shape)

    result = jax.jit(logistic_saturation)(media, half)

    assert result.shape == media.shape
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_logistic_saturation_sums_shared_channel_parameter_gradients_over_batches():
    media = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)
    half = jnp.array([2.0, 4.0])
    constant = np.log(3) / 2
    scaled = constant * np.asarray(media, dtype=np.float64) / np.asarray(half)
    expected = (-scaled / (np.asarray(half) * np.cosh(scaled) ** 2)).sum(axis=(0, 1))

    gradient = jax.jit(jax.grad(lambda parameters: logistic_saturation(media, parameters).sum()))(half)

    assert gradient.shape == (2,)
    np.testing.assert_allclose(gradient, expected, rtol=3e-6)


def test_logistic_saturation_supports_scalar_general_broadcast_and_parameter_draws():
    media = jnp.array([[0.0], [2.0], [4.0]])
    half = jnp.array([[1.0, 2.0], [2.0, 4.0]])
    result = jax.jit(jax.vmap(logistic_saturation, in_axes=(None, 0)))(media, half)
    expected = stats.halflogistic.cdf(np.asarray(media)[None, :, :], scale=np.asarray(half)[:, None, :] / np.log(3))

    assert logistic_saturation(1.0, 1.0).shape == ()
    assert result.shape == (2, 3, 2)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)
    assert logistic_saturation(np.empty((0, 2)), 1.0).shape == (0, 2)


@pytest.mark.parametrize("media,half", [(0.0, 2.0), (0.25, 2.0), (2.0, 2.0), (5.0, 1.5)])
def test_logistic_saturation_derivatives_match_analytic_expressions(media, half):
    constant = np.log(3) / 2
    response = np.tanh(constant * media / half)
    squared_sech = 1 / np.cosh(constant * media / half) ** 2
    gradient = squared_sech * np.array([constant / half, -constant * media / half**2])
    mixed = -constant * squared_sech / half**2 + 2 * constant**2 * media * response * squared_sech / half**3
    hessian = np.array(
        [
            [-2 * constant**2 * response * squared_sech / half**2, mixed],
            [mixed, 2 * constant * media * squared_sech / half**3 * (1 - constant * media * response / half)],
        ]
    )
    arguments = jnp.array([media, half])

    def response_function(values):
        return logistic_saturation(*values)

    actual_gradient = jax.jit(jax.grad(response_function))(arguments)
    actual_hessian = jax.jit(jax.hessian(response_function))(arguments)
    _, tangent = jax.jit(lambda values: jax.jvp(response_function, (values,), (jnp.array([0.3, -0.2]),)))(arguments)

    np.testing.assert_allclose(actual_gradient, gradient, rtol=4e-6, atol=2e-8)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=4e-6, atol=2e-8)
    np.testing.assert_allclose(tangent, gradient @ np.array([0.3, -0.2]), rtol=4e-6, atol=2e-8)


@pytest.mark.parametrize("argument", ["media", "half_saturation"])
@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf, -np.inf])
def test_logistic_saturation_masks_invalid_cells_and_keeps_valid_gradients(argument, bad_value):
    arguments = {"media": jnp.array([2.0, 1.0]), "half_saturation": jnp.array([2.0, 2.0])}
    arguments[argument] = arguments[argument].at[1].set(bad_value)
    result = jax.jit(logistic_saturation)(**arguments)
    gradient = jax.jit(jax.grad(lambda media: logistic_saturation(media, arguments["half_saturation"]).sum()))(
        arguments["media"]
    )

    np.testing.assert_allclose(result[0], 0.5, rtol=1e-6)
    assert np.isnan(result[1])
    np.testing.assert_allclose(gradient[0], 3 * np.log(3) / 16, rtol=2e-6)
    assert np.isnan(logistic_saturation([0.0, 1.0], 0.0)).all()


@pytest.mark.parametrize("argument", ["media", "half_saturation"])
@pytest.mark.parametrize("value", [["1", "2"], [1 + 2j], np.array([1], dtype=object)])
def test_logistic_saturation_names_nonreal_inputs_in_errors(argument, value):
    arguments = {"media": [1.0, 2.0], "half_saturation": 2.0}
    arguments[argument] = value
    with pytest.raises(TypeError, match=argument):
        logistic_saturation(**arguments)


def test_logistic_saturation_explains_incompatible_shapes():
    with pytest.raises(ValueError) as caught:
        logistic_saturation(np.ones((3, 2)), np.ones(4))
    assert all(detail in str(caught.value) for detail in ("media", "half_saturation", "(3, 2)", "(4,)"))


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.int32, np.float16, np.float32])
def test_logistic_saturation_promotes_inputs_to_at_least_float32(dtype):
    result = logistic_saturation(np.array([0, 1], dtype=dtype), np.asarray(1, dtype=dtype))
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, [0.0, 0.5], rtol=1e-6)


def test_logistic_saturation_converts_large_counts_without_integer_narrowing():
    media = np.array([3_000_000_000, 6_000_000_000], dtype=np.int64)
    np.testing.assert_allclose(logistic_saturation(media, 3_000_000_000), [0.5, 0.8], rtol=2e-6)


def test_logistic_saturation_composes_with_prepared_data_scaling_adstock_and_model():
    frame = pl.DataFrame(
        {"week": [1, 2, 3], "sales": [0.0, 1.0, 2.0], "video": [0.0, 10.0, 20.0], "search": [0.0, 20.0, 40.0]}
    )
    prepared = mmmjax.prepare_data(frame, time="week", outcome="sales", media=["video", "search"])
    scaled = mmmjax.fit_data_scaling(prepared).transform(prepared)

    def log_density(data, half_saturation):
        carried = mmmjax.geometric_adstock(data["media"], 0.5, max_lag=1)
        prediction = logistic_saturation(carried, half_saturation).sum(axis=-1)
        return mmmjax.normal(data["outcome"], prediction, 1.0)

    model = mmmjax.Model({"half_saturation": mmmjax.Positive(shape=(2,))}, log_density)
    value, gradients = jax.jit(jax.value_and_grad(model.log_density))(
        {"half_saturation": jnp.zeros(2)}, scaled._to_jax()
    )
    carried = np.convolve([0.0, 2 / 3, 4 / 3], [1.0, 0.5])[:3] / 1.5
    prediction = 2 * stats.halflogistic.cdf(carried, scale=1 / np.log(3))

    np.testing.assert_allclose(value, stats.norm.logpdf([0, 1, 2], loc=prediction).sum(), rtol=2e-6)
    assert gradients["half_saturation"].shape == (2,)
    assert np.isfinite(gradients["half_saturation"]).all()
    np.testing.assert_array_equal(prepared.arrays["media"], [[0, 0], [10, 20], [20, 40]])
