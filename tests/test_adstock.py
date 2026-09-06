"""Tests for finite carryover transformations."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy import stats

import mmmjax
from mmmjax.adstock import delayed_adstock, geometric_adstock, weibull_cdf_adstock, weibull_pdf_adstock


def test_geometric_adstock_is_exported() -> None:
    assert mmmjax.geometric_adstock is geometric_adstock
    assert "geometric_adstock" in mmmjax.__all__


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("alpha,max_lag", [(0.0, 4), (0.3, 0), (0.3, 3), (0.8, 12), (1.0, 3)])
def test_geometric_adstock_matches_numpy_convolution(alpha, max_lag, normalize, dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    media = jnp.array([0.0, 2.0, 1.0, 0.5, 3.0, 0.0, 0.0], dtype=dtype)
    retention = jnp.asarray(alpha, dtype=dtype)
    expected = _convolution_reference(media, retention, max_lag=max_lag, normalize=normalize)
    function = partial(geometric_adstock, max_lag=max_lag, normalize=normalize)

    eager = function(media, retention)
    compiled = jax.jit(function)(media, retention)

    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14
    for result in (eager, compiled):
        assert result.shape == media.shape
        assert result.dtype == media.dtype
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=1e-15)


@pytest.mark.parametrize("normalize", [False, True])
def test_geometric_adstock_impulse_is_causal_and_stops_at_max_lag(normalize) -> None:
    media = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    expected = np.array([0.0, 0.0, 1.0, 0.5, 0.25, 0.0, 0.0])
    if normalize:
        expected /= 1.75

    result = geometric_adstock(media, 0.5, max_lag=2, normalize=normalize)

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


def test_geometric_adstock_normalizes_the_full_window_for_short_series() -> None:
    result = geometric_adstock(jnp.ones(3), 1.0, max_lag=4)

    np.testing.assert_allclose(result, [0.2, 0.4, 0.6], rtol=1e-7, atol=0)


def test_geometric_adstock_accepts_history_by_prepending_and_slicing() -> None:
    media = np.array([1.0, 4.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 5.0])
    expected = _convolution_reference(media, 0.6, max_lag=3, normalize=True)[5:]

    result = geometric_adstock(media[2:], 0.6, max_lag=3)[3:]

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("alpha_shape", [(), (3,), (2, 1), (2, 3)])
def test_geometric_adstock_broadcasts_across_non_time_axes(axis, alpha_shape) -> None:
    media = np.moveaxis(np.arange(42, dtype=np.float32).reshape(2, 3, 7) / 10, -1, axis)
    alpha = np.full(alpha_shape, 0.4, dtype=np.float32)
    if alpha_shape:
        alpha = np.linspace(0.1, 0.9, alpha.size, dtype=np.float32).reshape(alpha_shape)
    expected = _convolution_reference(media, alpha, max_lag=3, normalize=True, axis=axis)

    result = jax.jit(partial(geometric_adstock, max_lag=3, axis=axis))(media, alpha)

    assert result.shape == media.shape
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_geometric_adstock_handles_hundreds_of_channels_without_mixing_series() -> None:
    media = np.random.default_rng(23).uniform(size=(8, 10, 465)).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, 465, dtype=np.float32)
    expected = _convolution_reference(media, alpha, max_lag=4, normalize=True, axis=1)

    result = jax.jit(partial(geometric_adstock, max_lag=4, axis=1))(media, alpha)

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_geometric_adstock_vectorizes_over_parameter_draws() -> None:
    media = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3) / 10
    alpha = jnp.linspace(0.1, 0.9, 24).reshape(4, 2, 3)
    expected = np.stack([_convolution_reference(media, draw, max_lag=2, normalize=True) for draw in alpha])
    function = partial(geometric_adstock, max_lag=2)

    result = jax.jit(jax.vmap(function, in_axes=(None, 0)))(media, alpha)

    assert result.shape == (4, 5, 2, 3)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


def test_geometric_adstock_broadcast_parameter_jvp_matches_finite_differences() -> None:
    media = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3) / 10
    alpha = jnp.array([0.2, 0.5, 0.8])
    direction = jnp.array([0.3, -0.2, 0.1])

    def transformed(retention):
        return geometric_adstock(media, retention, max_lag=3, axis=1)

    _, tangent = jax.jit(lambda a: jax.jvp(transformed, (a,), (direction,)))(alpha)
    gradient = jax.jit(jax.grad(lambda a: transformed(a).sum()))(alpha)
    alpha64 = np.asarray(alpha, dtype=np.float64)
    step = 1e-4
    derivatives = []
    for index in range(3):
        offset = np.eye(3)[index] * step
        upper = _convolution_reference(media, alpha64 + offset, max_lag=3, normalize=True, axis=1)
        lower = _convolution_reference(media, alpha64 - offset, max_lag=3, normalize=True, axis=1)
        derivatives.append((upper - lower) / (2 * step))

    expected_tangent = np.tensordot(np.asarray(direction), derivatives, axes=1)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(gradient, np.sum(derivatives, axis=(1, 2, 3)), rtol=3e-6, atol=1e-6)


def test_geometric_adstock_composes_with_model_parameters() -> None:
    data = {"media": jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]), "target": jnp.array([0.5, 1.5, 2.0])}

    def log_density(data, alpha):
        prediction = geometric_adstock(data["media"], alpha, max_lag=2).sum(axis=-1)
        return mmmjax.normal(data["target"], prediction, 1.0)

    specification = mmmjax.Model({"alpha": mmmjax.Interval(0.0, 1.0, shape=(2,))}, log_density)
    value, gradient = jax.jit(jax.value_and_grad(specification.log_density))({"alpha": jnp.zeros(2)}, data)
    prediction = _convolution_reference(data["media"], np.full(2, 0.5), max_lag=2, normalize=True).sum(axis=-1)
    normal_log_density = (-0.5 * (np.asarray(data["target"]) - prediction) ** 2 - 0.5 * np.log(2 * np.pi)).sum()
    # At a zero unconstrained position, each unit-interval transform has derivative 1/4
    expected = normal_log_density + 2 * np.log(0.25)

    np.testing.assert_allclose(value, expected, rtol=1e-6, atol=0)
    assert gradient["alpha"].shape == (2,)
    assert np.isfinite(gradient["alpha"]).all()


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
def test_geometric_adstock_derivatives_match_independent_polynomials(alpha, normalize) -> None:
    media = jnp.array([2.0, 0.0, 1.0, 3.0, 0.5], dtype=jnp.float32)
    max_lag = 3
    # Each lag contributes every input except those too late to reach an observed output
    numerator = Polynomial([np.asarray(media, dtype=np.float64)[: media.size - lag].sum() for lag in range(4)])
    denominator = Polynomial(np.ones(4) if normalize else [1.0])
    n, d = numerator(alpha), denominator(alpha)
    dn, dd = numerator.deriv()(alpha), denominator.deriv()(alpha)
    expected_gradient = dn / d - n * dd / d**2
    expected_hessian = numerator.deriv(2)(alpha) / d - 2 * dn * dd / d**2
    expected_hessian += 2 * n * dd**2 / d**3 - n * denominator.deriv(2)(alpha) / d**2

    def total(retention):
        return geometric_adstock(media, retention, max_lag=max_lag, normalize=normalize).sum()

    value, gradient = jax.jit(jax.value_and_grad(total))(jnp.float32(alpha))
    hessian = jax.jit(jax.grad(jax.grad(total)))(jnp.float32(alpha))

    np.testing.assert_allclose(value, n / d, rtol=2e-6, atol=0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=5e-6, atol=2e-6)


def test_geometric_adstock_media_jacobian_has_only_causal_lag_weights() -> None:
    # A lower-triangular band also detects accidental cross-time leakage from centered convolution
    expected = np.array([[1, 0, 0, 0], [0.5, 1, 0, 0], [0.25, 0.5, 1, 0], [0, 0.25, 0.5, 1]]) / 1.75
    function = partial(geometric_adstock, alpha=0.5, max_lag=2)

    forward = jax.jit(jax.jacfwd(function))(jnp.ones(4))
    reverse = jax.jit(jax.jacrev(function))(jnp.ones(4))

    np.testing.assert_allclose(forward, expected, rtol=1e-7, atol=0)
    np.testing.assert_allclose(reverse, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize("max_lag", [0, 3])
def test_geometric_adstock_invalid_retention_only_affects_its_series(max_lag) -> None:
    media = jnp.ones((5, 7))
    alpha = jnp.array([0.0, 0.5, 1.0, -0.1, 1.1, jnp.inf, jnp.nan])

    result = jax.jit(partial(geometric_adstock, max_lag=max_lag))(media, alpha)

    expected = _convolution_reference(media[:, :3], alpha[:3], max_lag=max_lag, normalize=True)
    np.testing.assert_allclose(result[:, :3], expected, rtol=2e-6, atol=0)
    assert np.isnan(result[:, 3:]).all()


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float16, jnp.bfloat16, jnp.float32])
def test_adstock_uses_at_least_float32(adstock, dtype) -> None:
    result = adstock(jnp.ones((4, 2), dtype=dtype), dtype(1), max_lag=2)

    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, np.array([[1, 1], [2, 2], [3, 3], [3, 3]]) / 3, rtol=1e-7, atol=0)


def test_adstock_respects_default_precision_and_typed_float32(adstock) -> None:
    default = adstock([1, 2, 3], 0.5, max_lag=2)
    typed = adstock(jnp.ones(3, dtype=jnp.float32), 0.5, max_lag=2)

    assert default.dtype == jnp.asarray(0.5).dtype
    assert typed.dtype == jnp.float32


@pytest.mark.parametrize("shape,axis", [((0,), 0), ((0, 3), 0), ((3, 0), 0), ((2, 0, 3), 1)])
def test_adstock_preserves_empty_shapes(adstock, shape, axis) -> None:
    result = jax.jit(partial(adstock, max_lag=2, axis=axis))(jnp.zeros(shape), 0.5)

    assert result.shape == shape


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"max_lag": -1}, ValueError, "max_lag must be nonnegative"),
        ({"max_lag": 1.5}, TypeError, "max_lag must be a static nonnegative integer"),
        ({"max_lag": True}, TypeError, "max_lag must be a static nonnegative integer"),
        ({"max_lag": 2, "axis": 1}, ValueError, "axis 1 is out of range"),
        ({"max_lag": 2, "axis": -2}, ValueError, "axis -2 is out of range"),
        ({"max_lag": 2, "axis": 0.0}, TypeError, "axis must be a static integer"),
        ({"max_lag": 2, "axis": False}, TypeError, "axis must be a static integer"),
        ({"max_lag": 2, "normalize": 1}, TypeError, "normalize must be a static boolean"),
    ],
)
def test_adstock_rejects_invalid_configuration(adstock, options, error, message) -> None:
    with pytest.raises(error, match=message):
        adstock(jnp.ones(4), 0.5, **options)


@pytest.mark.parametrize("name", ["max_lag", "axis", "normalize"])
def test_adstock_requires_static_configuration_under_jit(adstock, name) -> None:
    options = {"max_lag": 2, "axis": 0, "normalize": True}
    with pytest.raises(TypeError, match=rf"{name} must be a static"):
        jax.jit(lambda value: adstock(jnp.ones(4), 0.5, **(options | {name: value})))(options[name])


def test_adstock_requires_a_time_dimension(adstock) -> None:
    with pytest.raises(ValueError, match="media must have at least one dimension for time"):
        adstock(2.0, 0.5, max_lag=2)


def test_adstock_rejects_incompatible_retention_shape(adstock) -> None:
    with pytest.raises(ValueError, match=r"alpha shape \(4,\).*non-time media shape \(2, 3\)"):
        adstock(jnp.ones((2, 7, 3)), jnp.ones(4), max_lag=2, axis=1)


@pytest.mark.parametrize("name", ["media", "alpha"])
@pytest.mark.parametrize("value", ["invalid", 1.0j])
def test_adstock_requires_real_numeric_inputs(adstock, name, value) -> None:
    arguments = {"media": jnp.ones(4), "alpha": 0.5}
    arguments[name] = value
    with pytest.raises(TypeError, match=rf"{name} must.*real numeric"):
        adstock(**arguments, max_lag=2)


def test_delayed_adstock_is_exported() -> None:
    assert mmmjax.delayed_adstock is delayed_adstock
    assert "delayed_adstock" in mmmjax.__all__


def test_delayed_adstock_includes_delay_in_dtype_promotion() -> None:
    # A Python float introduces the default float dtype even when media and retention are integers
    result = delayed_adstock(jnp.ones(3, dtype=jnp.int32), jnp.int32(1), 0.0, max_lag=2)

    assert result.dtype == jnp.asarray(0.0).dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "alpha,theta,max_lag", [(0.4, 0.0, 0), (0.2, 0.0, 4), (0.6, 1.3, 4), (0.3, 2.5, 12), (1.0, 4.0, 4)]
)
def test_delayed_adstock_matches_independent_lag_sum(alpha, theta, max_lag, normalize, dtype) -> None:
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    media = jnp.array([0.0, 2.0, -1.0, 0.5, 3.0, 0.0, 0.0], dtype=dtype)
    retention, delay = jnp.asarray(alpha, dtype=dtype), jnp.asarray(theta, dtype=dtype)
    expected = _delayed_reference(media, retention, delay, max_lag=max_lag, normalize=normalize)
    function = partial(delayed_adstock, max_lag=max_lag, normalize=normalize)

    for result in (function(media, retention, delay), jax.jit(function)(media, retention, delay)):
        assert result.shape == media.shape
        assert result.dtype == media.dtype
        np.testing.assert_allclose(result, expected, rtol=3e-6 if dtype == jnp.float32 else 3e-14, atol=1e-15)


@pytest.mark.parametrize("normalize", [False, True])
def test_delayed_adstock_impulse_peaks_after_exposure(normalize) -> None:
    media = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # With alpha=1/2 and theta=2, the squared distances give exponents 4, 1, 0, 1, 4
    expected = np.array([0, 1 / 16, 1 / 2, 1, 1 / 2, 1 / 16, 0])
    if normalize:
        expected /= 17 / 8

    result = delayed_adstock(media, 0.5, 2.0, max_lag=4, normalize=normalize)

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


def test_delayed_adstock_normalizes_the_full_window_for_short_series() -> None:
    result = delayed_adstock(jnp.ones(2), 0.5, 2.0, max_lag=4)

    np.testing.assert_allclose(result, [1 / 34, 9 / 34], rtol=1e-7, atol=0)


def test_delayed_adstock_accepts_history_by_prepending_and_slicing() -> None:
    media = np.array([1.0, 4.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 5.0])
    expected = _delayed_reference(media, 0.6, 1.2, max_lag=3, normalize=True)[5:]

    result = delayed_adstock(media[2:], 0.6, 1.2, max_lag=3)[3:]

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("alpha_shape,theta_shape", [((), ()), ((3,), (2, 1)), ((2, 3), (3,))])
def test_delayed_adstock_broadcasts_parameters_independently(axis, alpha_shape, theta_shape) -> None:
    media = np.moveaxis(np.arange(42, dtype=np.float32).reshape(2, 3, 7) / 10, -1, axis)
    alpha = np.linspace(0.1, 0.9, int(np.prod(alpha_shape)), dtype=np.float32).reshape(alpha_shape)
    theta = np.linspace(0.0, 3.0, int(np.prod(theta_shape)), dtype=np.float32).reshape(theta_shape)
    expected = _delayed_reference(media, alpha, theta, max_lag=3, normalize=True, axis=axis)

    result = jax.jit(partial(delayed_adstock, max_lag=3, axis=axis))(media, alpha, theta)

    assert result.shape == media.shape
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_delayed_adstock_vectorizes_over_parameter_draws() -> None:
    media = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3) / 10
    alpha = jnp.linspace(0.1, 0.9, 12).reshape(4, 3)
    theta = jnp.linspace(0.0, 2.0, 8).reshape(4, 2, 1)
    expected = np.stack(
        [_delayed_reference(media, a, t, max_lag=2, normalize=True) for a, t in zip(alpha, theta, strict=True)]
    )
    function = jax.vmap(partial(delayed_adstock, max_lag=2), in_axes=(None, 0, 0))

    result = jax.jit(function)(media, alpha, theta)

    assert result.shape == (4, 5, 2, 3)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_delayed_adstock_composes_with_batched_model_parameters(normalize) -> None:
    data = {
        "media": jnp.asarray(np.random.default_rng(23).uniform(size=(5, 2, 3))),
        "target": jnp.linspace(0.3, 2.0, 10).reshape(5, 2),
    }
    position = {"alpha": jnp.array([-0.8, 0.2, 1.1]), "theta": jnp.array([[-0.5], [0.7]])}

    def predict(data, alpha, theta):
        return mmmjax.delayed_adstock(data["media"], alpha, theta, max_lag=3, normalize=normalize).sum(axis=-1)

    def log_density(data, alpha, theta):
        return mmmjax.normal(data["target"], predict(data, alpha, theta), 0.7)

    def generate(key, data, alpha, theta):
        return {"prediction": predict(data, alpha, theta)}

    specification = mmmjax.Model(
        {
            "alpha": mmmjax.Interval(0.0, 1.0, shape=(3,)),
            "theta": mmmjax.Interval(0.0, 3.0, shape=(2, 1)),
        },
        log_density,
        generate,
    )

    def reference(unconstrained):
        # Check the whole path independently, including interval transforms and their Jacobians
        alpha_position = np.asarray(unconstrained["alpha"], dtype=np.float64)
        theta_position = np.asarray(unconstrained["theta"], dtype=np.float64)
        alpha = 1 / (1 + np.exp(-alpha_position))
        theta = 3 / (1 + np.exp(-theta_position))
        prediction = _delayed_reference(data["media"], alpha, theta, max_lag=3, normalize=normalize).sum(axis=-1)
        residual = (np.asarray(data["target"]) - prediction) / 0.7
        density = (-0.5 * residual**2 - np.log(0.7) - 0.5 * np.log(2 * np.pi)).sum()
        density += (-np.logaddexp(0, -alpha_position) - np.logaddexp(0, alpha_position)).sum()
        density += (np.log(3) - np.logaddexp(0, -theta_position) - np.logaddexp(0, theta_position)).sum()
        return density, prediction

    value, gradient = jax.jit(jax.value_and_grad(specification.log_density))(position, data)
    generated = jax.jit(specification.generate)(jax.random.key(0), specification.constrain(position), data)
    expected_density, expected_prediction = reference(position)

    np.testing.assert_allclose(value, expected_density, rtol=3e-6, atol=1e-6)
    assert set(generated) == {"prediction"}
    assert generated["prediction"].shape == (5, 2)
    np.testing.assert_allclose(generated["prediction"], expected_prediction, rtol=3e-6, atol=0)
    assert set(gradient) == set(position)
    for name, values in position.items():
        values64 = np.asarray(values, dtype=np.float64)
        expected_gradient = np.empty_like(values64)
        for index in np.ndindex(values64.shape):
            offset = np.zeros_like(values64)
            offset[index] = 1e-4
            upper = reference(position | {name: values64 + offset})[0]
            lower = reference(position | {name: values64 - offset})[0]
            expected_gradient[index] = (upper - lower) / 2e-4
        assert gradient[name].shape == values.shape
        np.testing.assert_allclose(gradient[name], expected_gradient, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("alpha,theta", [(0.45, 1.3), (0.2, 0.0), (1.0, 1.5)])
def test_delayed_adstock_derivatives_match_numpy_finite_differences(alpha, theta, normalize) -> None:
    media = np.array([2.0, 0.0, 1.0, 3.0, 0.5])
    parameters = np.array([alpha, theta])

    def reference(values):
        return _delayed_reference(media, *values, max_lag=3, normalize=normalize).sum()

    def total(values):
        return delayed_adstock(media, values[0], values[1], max_lag=3, normalize=normalize).sum()

    offsets = np.eye(2) * 1e-4
    expected_gradient = np.array([(reference(parameters + dx) - reference(parameters - dx)) / 2e-4 for dx in offsets])
    expected_hessian = np.array(
        [
            [
                (
                    reference(parameters + dx + dy)
                    - reference(parameters + dx - dy)
                    - reference(parameters - dx + dy)
                    + reference(parameters - dx - dy)
                )
                / 4e-8
                for dy in offsets
            ]
            for dx in offsets
        ]
    )

    for gradient in (jax.jit(jax.jacfwd(total))(parameters), jax.jit(jax.grad(total))(parameters)):
        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=2e-6)
    hessian = jax.jit(jax.hessian(total))(parameters)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=2e-4, atol=2e-5)


def test_delayed_adstock_media_jacobian_has_only_causal_lag_weights() -> None:
    expected = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1]]) / 4
    function = partial(delayed_adstock, alpha=0.5, theta=1.0, max_lag=2)

    for derivative in (jax.jacfwd(function), jax.jacrev(function)):
        np.testing.assert_allclose(jax.jit(derivative)(jnp.ones(4)), expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_delayed_adstock_invalid_parameters_only_affect_their_series(normalize) -> None:
    alpha = jnp.array([0.5, 1.0, 0.0, -0.1, 1.1, jnp.inf, jnp.nan, 0.5, 0.5, 0.5, 0.5])
    theta = jnp.array([1.5, 3.0, 1.5, 0.0, 0.0, 0.0, 0.0, -0.1, 3.1, jnp.inf, jnp.nan])
    media = jnp.ones((5, 11))

    result = jax.jit(partial(delayed_adstock, max_lag=3, normalize=normalize))(media, alpha, theta)

    expected = _delayed_reference(media[:, :2], alpha[:2], theta[:2], max_lag=3, normalize=normalize)
    np.testing.assert_allclose(result[:, :2], expected, rtol=2e-6, atol=0)
    assert np.isnan(result[:, 2:]).all()


def test_delayed_adstock_rejects_incompatible_delay_shape() -> None:
    with pytest.raises(ValueError, match=r"theta shape \(4,\).*non-time media shape \(2, 3\)"):
        delayed_adstock(jnp.ones((2, 7, 3)), 0.5, jnp.ones(4), max_lag=2, axis=1)


@pytest.mark.parametrize("theta", ["invalid", 1.0j])
def test_delayed_adstock_requires_real_numeric_delay(theta) -> None:
    with pytest.raises(TypeError, match=r"theta must.*real numeric"):
        delayed_adstock(jnp.ones(4), 0.5, theta, max_lag=2)


def test_weibull_pdf_adstock_is_exported() -> None:
    assert mmmjax.weibull_pdf_adstock is weibull_pdf_adstock
    assert "weibull_pdf_adstock" in mmmjax.__all__


def test_weibull_cdf_adstock_is_exported() -> None:
    assert mmmjax.weibull_cdf_adstock is weibull_cdf_adstock
    assert "weibull_cdf_adstock" in mmmjax.__all__


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("shape,scale,max_lag", [(0.5, 0.3, 4), (1.0, 2.0, 3), (2.5, 4.0, 12), (5.0, 3.0, 4)])
def test_weibull_adstock_matches_scipy_lag_sum(weibull, shape, scale, max_lag, normalize, dtype) -> None:
    adstock, reference = weibull
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    media = jnp.array([0.0, 2.0, -1.0, 0.5, 3.0, 0.0, 0.0], dtype=dtype)
    shape, scale = jnp.asarray(shape, dtype=dtype), jnp.asarray(scale, dtype=dtype)
    expected = reference(media, shape, scale, max_lag=max_lag, normalize=normalize)
    function = partial(adstock, max_lag=max_lag, normalize=normalize)

    for result in (function(media, shape, scale), jax.jit(function)(media, shape, scale)):
        assert result.shape == media.shape
        assert result.dtype == dtype
        np.testing.assert_allclose(result, expected, rtol=4e-6 if dtype == jnp.float32 else 4e-14, atol=1e-15)


@pytest.mark.parametrize("normalize", [False, True])
def test_weibull_pdf_adstock_impulse_uses_rescaled_weights_and_causal_padding(normalize) -> None:
    # Shape one and scale 1/log(2) give densities proportional to 1/2, 1/4, 1/8, 1/16
    # Subtracting the minimum and dividing by the range gives 1, 3/7, 1/7, 0
    media = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    expected = np.array([0.0, 1.0, 3 / 7, 1 / 7, 0.0, 0.0])
    if normalize:
        expected /= 11 / 7

    result = weibull_pdf_adstock(media, 1.0, 1 / np.log(2), max_lag=3, normalize=normalize)

    np.testing.assert_allclose(result, expected, rtol=4e-7, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_weibull_cdf_adstock_impulse_uses_cumulative_survival_and_causal_padding(normalize) -> None:
    # Shape one and scale 1/log(2) give survival probabilities 1/2, 1/4, 1/8
    # Their cumulative products distinguish these weights from individual survival values
    media = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    expected = np.array([0.0, 1.0, 1 / 2, 1 / 8, 1 / 64, 0.0])
    if normalize:
        expected /= 105 / 64

    function = partial(weibull_cdf_adstock, max_lag=3, normalize=normalize)
    for result in (function(media, 1.0, 1 / np.log(2)), jax.jit(function)(media, 1.0, 1 / np.log(2))):
        np.testing.assert_allclose(result, expected, rtol=4e-7, atol=0)


def test_weibull_cdf_adstock_retains_small_survival_weights() -> None:
    # At shape one, cumulative survival is exp(-sum(lags)/scale)
    # Subtracting the CDF from one would lose these representable float32 weights
    media = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    shape, scale = jnp.float32(1), jnp.float32(0.05)
    expected = np.exp(-np.array([0.0, 1.0, 3.0]) / float(scale))
    function = partial(weibull_cdf_adstock, max_lag=2, normalize=False)

    for result in (function(media, shape, scale), jax.jit(function)(media, shape, scale)):
        np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_weibull_adstock_broadcasts_parameters_independently(weibull, axis) -> None:
    adstock, reference = weibull
    media = np.moveaxis(np.arange(42, dtype=np.float32).reshape(2, 3, 7) / 10, -1, axis)
    shape = np.array([[0.7], [2.5]], dtype=np.float32)
    scale = np.array([1.5, 3.0, 4.0], dtype=np.float32)
    expected = reference(media, shape, scale, max_lag=4, normalize=True, axis=axis)

    result = jax.jit(partial(adstock, max_lag=4, axis=axis))(media, shape, scale)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=1e-7)


def test_weibull_pdf_adstock_media_jacobian_has_only_causal_lag_weights() -> None:
    expected = np.array([[7, 0, 0, 0], [3, 7, 0, 0], [1, 3, 7, 0], [0, 1, 3, 7]]) / 11
    function = partial(weibull_pdf_adstock, shape=1.0, scale=1 / np.log(2), max_lag=3)

    for derivative in (jax.jacfwd(function), jax.jacrev(function)):
        np.testing.assert_allclose(jax.jit(derivative)(jnp.ones(4)), expected, rtol=4e-7, atol=0)


def test_weibull_cdf_adstock_media_jacobian_has_only_causal_lag_weights() -> None:
    expected = np.array([[64, 0, 0, 0], [32, 64, 0, 0], [8, 32, 64, 0], [1, 8, 32, 64]]) / 105
    function = partial(weibull_cdf_adstock, shape=1.0, scale=1 / np.log(2), max_lag=3)

    for derivative in (jax.jacfwd(function), jax.jacrev(function)):
        np.testing.assert_allclose(jax.jit(derivative)(jnp.ones(4)), expected, rtol=4e-7, atol=0)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float16, jnp.bfloat16, jnp.float32])
def test_weibull_adstock_promotes_low_precision_inputs(weibull, dtype) -> None:
    adstock, reference = weibull
    media = jnp.ones(4, dtype=dtype)
    shape, scale = dtype(2), dtype(3)
    expected = reference(media.astype(jnp.float32), 2.0, 3.0, max_lag=4, normalize=True)

    result = adstock(media, shape, scale, max_lag=4)

    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)


def test_weibull_adstock_vectorizes_over_parameter_draws(weibull) -> None:
    adstock, reference = weibull
    media = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3) / 10
    shape = jnp.linspace(0.5, 3.0, 12).reshape(4, 3)
    scale = jnp.linspace(1.0, 4.0, 8).reshape(4, 2, 1)
    expected = np.stack([reference(media, k, s, max_lag=4, normalize=True) for k, s in zip(shape, scale, strict=True)])
    function = jax.vmap(partial(adstock, max_lag=4), in_axes=(None, 0, 0))

    result = jax.jit(function)(media, shape, scale)

    assert result.shape == (4, 5, 2, 3)
    np.testing.assert_allclose(result, expected, rtol=4e-6, atol=1e-7)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("shape,scale", [(0.7, 2.3), (1.0, 1.7), (2.5, 3.2)])
def test_weibull_adstock_derivatives_match_scipy_finite_differences(weibull, shape, scale, normalize) -> None:
    adstock, reference_adstock = weibull
    media = np.array([2.0, 0.0, 1.0, 3.0, 0.5])
    parameters = np.array([shape, scale])

    def reference(values):
        return reference_adstock(media, *values, max_lag=4, normalize=normalize).sum()

    def total(values):
        return adstock(media, values[0], values[1], max_lag=4, normalize=normalize).sum()

    # Keep the finite-difference reference separate from JAX and its autodiff rules
    offsets = np.eye(2) * 1e-4
    expected_gradient = np.array([(reference(parameters + dx) - reference(parameters - dx)) / 2e-4 for dx in offsets])
    expected_hessian = np.array(
        [
            [
                (
                    reference(parameters + dx + dy)
                    - reference(parameters + dx - dy)
                    - reference(parameters - dx + dy)
                    + reference(parameters - dx - dy)
                )
                / 4e-8
                for dy in offsets
            ]
            for dx in offsets
        ]
    )

    for gradient in (jax.jit(jax.jacfwd(total))(parameters), jax.jit(jax.grad(total))(parameters)):
        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=2e-6)
    hessian = jax.jit(jax.hessian(total))(parameters)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("max_lag,scale", [(0, 2.0), (4, 0.001)])
@pytest.mark.parametrize("normalize", [False, True])
def test_weibull_adstock_handles_single_period_and_concentrated_weights(weibull, max_lag, scale, normalize) -> None:
    adstock, _ = weibull
    media = jnp.array([2.0, 0.0, 1.0, 3.0])
    function = partial(adstock, max_lag=max_lag, normalize=normalize)

    result = jax.jit(function)(media, 1.0, scale)
    gradient = jax.jit(jax.grad(lambda k, s: function(media, k, s).sum(), argnums=(0, 1)))(1.0, scale)

    np.testing.assert_array_equal(result, media)
    np.testing.assert_array_equal(gradient, [0.0, 0.0])


@pytest.mark.parametrize("normalize", [False, True])
def test_weibull_pdf_adstock_flat_kernel_has_undefined_rescaling(normalize) -> None:
    media = jnp.array([2.0, 0.0, 1.0, 3.0])
    # The sampled densities round to the same value, making their min/max range zero
    with np.errstate(invalid="ignore"):
        expected = _weibull_pdf_reference(media, 1.0, 1e20, max_lag=4, normalize=normalize)
    function = partial(weibull_pdf_adstock, max_lag=4, normalize=normalize)

    assert np.isnan(expected).all()
    for result in (function(media, 1.0, 1e20), jax.jit(function)(media, 1.0, 1e20)):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("max_lag", [0, 4])
def test_weibull_adstock_invalid_parameters_only_affect_their_series(weibull, max_lag) -> None:
    adstock, reference = weibull
    shape = jnp.array([0.5, 2.5, 0.0, -1.0, jnp.inf, jnp.nan, 2.5, 2.5, 2.5, 2.5])
    scale = jnp.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, jnp.inf, jnp.nan])
    media = jnp.ones((5, 10))
    function = partial(adstock, max_lag=max_lag)

    result = jax.jit(function)(media, shape, scale)
    gradient = jax.jit(jax.grad(lambda k, s: function(media, k, s)[:, :2].sum(), argnums=(0, 1)))(shape, scale)

    expected = reference(media[:, :2], shape[:2], scale[:2], max_lag=max_lag, normalize=True)
    np.testing.assert_allclose(result[:, :2], expected, rtol=3e-6, atol=0)
    assert np.isnan(result[:, 2:]).all()
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(np.asarray(gradient)[:, 2:], 0)


@pytest.mark.parametrize("name", ["shape", "scale"])
def test_weibull_adstock_rejects_incompatible_parameter_shapes(weibull, name) -> None:
    adstock, _ = weibull
    parameters = {"shape": 2.0, "scale": 3.0, name: jnp.ones(4)}
    with pytest.raises(ValueError, match=rf"{name} shape \(4,\).*non-time media shape \(2, 3\)"):
        adstock(jnp.ones((2, 7, 3)), **parameters, max_lag=4, axis=1)


@pytest.mark.parametrize("name", ["shape", "scale"])
@pytest.mark.parametrize("value", ["invalid", 1.0j])
def test_weibull_adstock_requires_real_numeric_parameters(weibull, name, value) -> None:
    adstock, _ = weibull
    parameters = {"shape": 2.0, "scale": 3.0, name: value}
    with pytest.raises(TypeError, match=rf"{name} must.*real numeric"):
        adstock(jnp.ones(4), **parameters, max_lag=4)


@pytest.fixture(params=[geometric_adstock, partial(delayed_adstock, theta=0)], ids=["geometric", "delayed"])
def adstock(request):
    return request.param


@pytest.fixture(params=["pdf", "cdf"])
def weibull(request):
    if request.param == "pdf":
        return weibull_pdf_adstock, _weibull_pdf_reference
    return weibull_cdf_adstock, _weibull_cdf_reference


def _weibull_cdf_reference(media, shape, scale, *, max_lag, normalize, axis=0):
    media = np.moveaxis(np.asarray(media, dtype=np.float64), axis, -1)
    shape = np.broadcast_to(np.asarray(shape, dtype=np.float64), media.shape[:-1])
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), media.shape[:-1])
    periods = np.arange(max_lag) + 1
    survival = stats.weibull_min.sf(periods, c=shape[..., None], scale=scale[..., None])
    weights = np.cumprod(np.concatenate((np.ones((*shape.shape, 1)), survival), axis=-1), axis=-1)
    if normalize:
        weights /= weights.sum(axis=-1, keepdims=True)
    result = np.zeros_like(media)
    for lag in range(min(max_lag + 1, media.shape[-1])):
        result[..., lag:] += media[..., : media.shape[-1] - lag] * weights[..., lag, None]
    return np.moveaxis(result, -1, axis)


def _weibull_pdf_reference(media, shape, scale, *, max_lag, normalize, axis=0):
    media = np.moveaxis(np.asarray(media, dtype=np.float64), axis, -1)
    shape = np.broadcast_to(np.asarray(shape, dtype=np.float64), media.shape[:-1])
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), media.shape[:-1])
    periods = np.arange(max_lag + 1) + 1
    if max_lag == 0:
        weights = np.ones((*shape.shape, 1))
    else:
        weights = stats.weibull_min.pdf(periods, c=shape[..., None], scale=scale[..., None])
        minimum = weights.min(axis=-1, keepdims=True)
        width = weights.max(axis=-1, keepdims=True) - minimum
        weights = (weights - minimum) / width
    if normalize:
        weights /= weights.sum(axis=-1, keepdims=True)
    result = np.zeros_like(media)
    for lag in range(min(max_lag + 1, media.shape[-1])):
        result[..., lag:] += media[..., : media.shape[-1] - lag] * weights[..., lag, None]
    return np.moveaxis(result, -1, axis)


def _delayed_reference(media, alpha, theta, *, max_lag, normalize, axis=0):
    media = np.moveaxis(np.asarray(media, dtype=np.float64), axis, -1)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), media.shape[:-1])
    theta = np.broadcast_to(np.asarray(theta, dtype=np.float64), media.shape[:-1])
    weights = alpha[..., None] ** (np.arange(max_lag + 1) - theta[..., None]) ** 2
    if normalize:
        weights /= weights.sum(axis=-1, keepdims=True)
    # Explicit lagged sums check convolution orientation and padding independently of JAX's convolution
    result = np.zeros_like(media)
    for lag in range(min(max_lag + 1, media.shape[-1])):
        result[..., lag:] += media[..., : media.shape[-1] - lag] * weights[..., lag, None]
    return np.moveaxis(result, -1, axis)


def _convolution_reference(media, alpha, *, max_lag, normalize, axis=0):
    media = np.moveaxis(np.asarray(media, dtype=np.float64), axis, -1)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), media.shape[:-1])
    result = np.empty_like(media)
    for index in np.ndindex(media.shape[:-1]):
        weights = alpha[index] ** np.arange(max_lag + 1)
        if normalize:
            weights /= weights.sum()
        result[index] = np.convolve(media[index], weights, mode="full")[: media.shape[-1]]
    return np.moveaxis(result, -1, axis)
