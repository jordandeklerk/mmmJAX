"""Tests for finite geometric carryover."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial

import mmmjax
from mmmjax.adstock import geometric_adstock


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
def test_geometric_adstock_uses_at_least_float32(dtype) -> None:
    result = geometric_adstock(jnp.ones((4, 2), dtype=dtype), dtype(1), max_lag=2)

    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result, np.array([[1, 1], [2, 2], [3, 3], [3, 3]]) / 3, rtol=1e-7, atol=0)


def test_geometric_adstock_respects_default_precision_and_typed_float32() -> None:
    default = geometric_adstock([1, 2, 3], 0.5, max_lag=2)
    typed = geometric_adstock(jnp.ones(3, dtype=jnp.float32), 0.5, max_lag=2)

    assert default.dtype == jnp.asarray(0.5).dtype
    assert typed.dtype == jnp.float32


@pytest.mark.parametrize("shape,axis", [((0,), 0), ((0, 3), 0), ((3, 0), 0), ((2, 0, 3), 1)])
def test_geometric_adstock_preserves_empty_shapes(shape, axis) -> None:
    result = jax.jit(partial(geometric_adstock, max_lag=2, axis=axis))(jnp.zeros(shape), 0.5)

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
def test_geometric_adstock_rejects_invalid_configuration(options, error, message) -> None:
    with pytest.raises(error, match=message):
        geometric_adstock(jnp.ones(4), 0.5, **options)


@pytest.mark.parametrize("name", ["max_lag", "axis", "normalize"])
def test_geometric_adstock_requires_static_configuration_under_jit(name) -> None:
    options = {"max_lag": 2, "axis": 0, "normalize": True}
    with pytest.raises(TypeError, match=rf"{name} must be a static"):
        jax.jit(lambda value: geometric_adstock(jnp.ones(4), 0.5, **(options | {name: value})))(options[name])


def test_geometric_adstock_requires_a_time_dimension() -> None:
    with pytest.raises(ValueError, match="media must have at least one dimension for time"):
        geometric_adstock(2.0, 0.5, max_lag=2)


def test_geometric_adstock_rejects_incompatible_retention_shape() -> None:
    with pytest.raises(ValueError, match=r"alpha shape \(4,\).*non-time media shape \(2, 3\)"):
        geometric_adstock(jnp.ones((2, 7, 3)), jnp.ones(4), max_lag=2, axis=1)


@pytest.mark.parametrize("name", ["media", "alpha"])
@pytest.mark.parametrize("value", ["invalid", 1.0j])
def test_geometric_adstock_requires_real_numeric_inputs(name, value) -> None:
    arguments = {"media": jnp.ones(4), "alpha": 0.5}
    arguments[name] = value
    with pytest.raises(TypeError, match=rf"{name} must.*real numeric"):
        geometric_adstock(**arguments, max_lag=2)


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
