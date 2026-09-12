"""Tests for HSGP features, spectral weights, and prepared model effects."""

from dataclasses import FrozenInstanceError, replace
from datetime import date
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scipy.integrate import quad

import mmmjax
from mmmjax import (
    FourierSeasonality,
    HSGPEffect,
    MediaEffect,
    Model,
    Positive,
    Real,
    generate_quantities,
    normal,
    prepare_data,
    sample_prior,
)
from mmmjax._results import _collect_results
from mmmjax.hsgp import HSGPConfig, hsgp_basis, hsgp_weights, prepare_hsgp

_COVARIANCES = ("expquad", "matern32", "matern52")


def _basis_reference(time, center, boundary, n_basis):
    center, boundary = np.float64(center), np.float64(boundary)
    frequencies = np.pi * np.arange(1, n_basis + 1, dtype=np.float64) / (2 * boundary)
    angles = (np.asarray(time, dtype=np.float64) - center + boundary)[:, None] * frequencies
    return np.sin(angles) / np.sqrt(boundary), frequencies


def _unit_covariance(distance, length_scale, covariance):
    scaled_distance = np.abs(np.asarray(distance, dtype=np.float64)) / np.float64(length_scale)
    if covariance == "expquad":
        return np.exp(-0.5 * scaled_distance**2)
    if covariance == "matern32":
        scaled_distance *= np.sqrt(3)
        return (1 + scaled_distance) * np.exp(-scaled_distance)
    scaled_distance *= np.sqrt(5)
    return (1 + scaled_distance + scaled_distance**2 / 3) * np.exp(-scaled_distance)


def _weights_from_covariance(frequencies, length_scale, amplitude, covariance):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    length_scale, amplitude = np.float64(length_scale), np.float64(amplitude)
    # The even covariance's cosine transform independently checks PSD constants
    # and the convention that frequencies are angular frequencies.
    spectral_density = [
        2
        * quad(
            lambda distance: _unit_covariance(distance, length_scale, covariance),
            0,
            np.inf,
            weight="cos",
            wvar=abs(float(frequency)),
            epsabs=1e-11,
            epsrel=1e-11,
        )[0]
        for frequency in frequencies
    ]
    return amplitude * np.sqrt(spectral_density)


def _data(times, *, groups=(), outcome=True, media_history=None):
    labels = tuple(times)
    group_count = len(groups) if groups else 1
    columns = {"time": [label for label in labels for _ in range(group_count)]}
    if groups:
        columns["region"] = list(groups) * len(labels)
    if outcome:
        columns["sales"] = (0.5 + np.arange(len(labels) * group_count) / 10).tolist()
    columns["video"] = np.ones(len(labels) * group_count)
    return prepare_data(
        pl.DataFrame(columns),
        time="time",
        groups=["region"] if groups else [],
        outcome="sales" if outcome else None,
        media=["video"],
        media_history=media_history,
        frequency=None,
    )


def _matern52_curve(prepared, coefficients, length_scale, amplitude):
    frequencies = np.asarray(prepared.frequencies, dtype=np.float64)
    length_scale = np.float64(length_scale)
    amplitude = np.float64(amplitude)
    weights = (
        amplitude
        * np.sqrt((400 * np.sqrt(5.0) / 3) * length_scale)
        / (5 + np.square(length_scale * frequencies)) ** 1.5
    )
    features = np.asarray(prepared.features, dtype=np.float64) - np.asarray(prepared.basis_mean, dtype=np.float64)
    return features @ (weights * np.asarray(coefficients, dtype=np.float64))


def _values(prepared, coefficients=(0.3, -0.7, 1.1), length_scale=2.5, amplitude=0.8):
    name = prepared.specification.name
    return {
        f"{name}_coefficients": jnp.asarray(coefficients),
        f"{name}_length_scale": jnp.asarray(length_scale),
        f"{name}_amplitude": jnp.asarray(amplitude),
    }


def _normal_reference(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sum(-0.5 * values**2 - 0.5 * np.log(2 * np.pi))


def test_hsgp_helpers_are_exported():
    assert mmmjax.HSGPConfig is HSGPConfig
    assert mmmjax.hsgp_basis is hsgp_basis
    assert mmmjax.hsgp_weights is hsgp_weights
    assert mmmjax.prepare_hsgp is prepare_hsgp
    assert {"HSGPConfig", "hsgp_basis", "hsgp_weights", "prepare_hsgp"} <= set(mmmjax.__all__)


@pytest.mark.parametrize(
    "covariance,padded_boundary,padded_count,span_count",
    [("expquad", 6.4, 22, 84), ("matern32", 9.0, 61, 164), ("matern52", 8.2, 43, 127)],
)
def test_prepare_matches_hand_calculated_padding_and_resolution(covariance, padded_boundary, padded_count, span_count):
    padded = prepare_hsgp((2, 10), length_scale_range=(0.5, 2), covariance=covariance)
    span_dominated = prepare_hsgp((-20, 20), length_scale_range=(0.5, 1), covariance=covariance)

    assert isinstance(padded, HSGPConfig)
    assert padded.center == 6.0
    assert padded.boundary == pytest.approx(padded_boundary)
    assert padded.n_basis == padded_count
    assert padded.covariance == covariance
    assert span_dominated.center == 0.0
    assert span_dominated.boundary == 24.0
    assert span_dominated.n_basis == span_count


def test_prepare_preserves_overrides_and_recalculates_resolution_for_manual_domain():
    automatic = prepare_hsgp((2, 10), length_scale_range=(0.5, 2))
    wider = prepare_hsgp((2, 10), length_scale_range=(0.5, 2), boundary=np.float64(12))
    fixed_count = prepare_hsgp((2, 10), length_scale_range=(0.5, 2), n_basis=17)
    both = prepare_hsgp((2, 10), length_scale_range=(0.5, 2), boundary=12, n_basis=17)

    assert automatic.covariance == "matern52"
    assert wider.boundary == both.boundary == 12.0
    assert wider.n_basis == 63
    assert fixed_count.boundary == automatic.boundary
    assert fixed_count.n_basis == both.n_basis == 17


def test_prepare_keeps_at_least_one_basis_function_for_a_small_manual_domain():
    config = prepare_hsgp((0, 2), length_scale_range=(3, 4), boundary=1.1)

    assert config.boundary == 1.1
    assert config.n_basis == 1
    assert config.basis([0.0, 2.0])[0].shape == (2, 1)


def test_prepared_configuration_is_immutable():
    config = prepare_hsgp((0, 8), length_scale_range=(0.5, 2))

    for attribute, value in (("center", 1.0), ("boundary", 20.0), ("n_basis", 99), ("covariance", "expquad")):
        with pytest.raises(FrozenInstanceError):
            setattr(config, attribute, value)


def test_preparation_and_weighted_curves_preserve_time_reference_and_units():
    time = np.array([-3.0, -1.5, 0.0, 2.75, 5.0], dtype=np.float32)
    for covariance in _COVARIANCES:
        config = prepare_hsgp((-3, 5), length_scale_range=(0.4, 1.1), covariance=covariance)
        shifted = prepare_hsgp((3.25, 11.25), length_scale_range=(0.4, 1.1), covariance=covariance)
        daily = prepare_hsgp((-21, 35), length_scale_range=(2.8, 7.7), covariance=covariance)
        basis, frequencies = config.basis(time)
        shifted_basis, shifted_frequencies = shifted.basis(time + 6.25)
        daily_basis, daily_frequencies = daily.basis(time * 7)
        coefficients = jnp.linspace(-0.3, 0.7, config.n_basis)
        weekly_weights = config.weights(frequencies, length_scale=0.7, amplitude=1.3)
        daily_weights = daily.weights(daily_frequencies, length_scale=4.9, amplitude=1.3)

        assert config.center == 1.0
        assert shifted.center == 7.25
        assert daily.center == 7.0
        assert config.boundary == shifted.boundary
        assert daily.boundary == pytest.approx(7 * config.boundary)
        assert config.n_basis == shifted.n_basis == daily.n_basis
        np.testing.assert_allclose(shifted_basis, basis, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(shifted_frequencies, frequencies)
        np.testing.assert_allclose(
            daily_basis @ (daily_weights * coefficients),
            basis @ (weekly_weights * coefficients),
            rtol=5e-6,
            atol=3e-6,
        )


def test_prepared_methods_reuse_training_and_forecast_basis_under_jit():
    config = prepare_hsgp((-2, 4), length_scale_range=(0.5, 1.5), covariance="matern32", boundary=6, n_basis=13)
    time = jnp.array([-2.0, -1.25, 0.0, 1.5, 2.5, 4.0])
    basis_function = jax.jit(config.basis)
    complete, frequencies = basis_function(time)
    training, training_frequencies = basis_function(time[:4])
    forecast, forecast_frequencies = basis_function(time[4:])
    expected_basis, expected_frequencies = _basis_reference(time, 1.0, 6.0, 13)
    weights_function = jax.jit(config.weights)

    np.testing.assert_array_equal(training, complete[:4])
    np.testing.assert_array_equal(forecast, complete[4:])
    np.testing.assert_array_equal(training_frequencies, frequencies)
    np.testing.assert_array_equal(forecast_frequencies, frequencies)
    np.testing.assert_allclose(complete, expected_basis, rtol=8e-6, atol=3e-6)
    np.testing.assert_allclose(frequencies, expected_frequencies, rtol=2e-6)
    for length_scale, amplitude in ((0.7, 1.3), (2.0, 0.5)):
        weights = weights_function(frequencies, length_scale=length_scale, amplitude=amplitude)
        expected_weights = _weights_from_covariance(frequencies, length_scale, amplitude, "matern32")
        np.testing.assert_allclose(weights, expected_weights, rtol=5e-6, atol=2e-6)


def test_prepared_methods_keep_time_and_covariance_parameter_gradients():
    config = prepare_hsgp((-2, 4), length_scale_range=(0.5, 1.5), covariance="expquad", boundary=6, n_basis=7)
    arguments = jnp.array([0.37, 0.75, 1.3], dtype=jnp.float32)
    coefficients = jnp.linspace(-0.5, 0.7, config.n_basis)

    def response(values):
        basis, frequencies = config.basis(values[:1])
        weights = config.weights(frequencies, length_scale=values[1], amplitude=values[2])
        return basis[0] @ (weights * coefficients)

    def reference(values):
        basis, frequencies = _basis_reference(values[:1], config.center, config.boundary, config.n_basis)
        weights = _weights_from_covariance(frequencies, values[1], values[2], config.covariance)
        return basis[0] @ (weights * np.asarray(coefficients, dtype=np.float64))

    arguments64 = np.asarray(arguments, dtype=np.float64)
    perturbations = np.eye(3) * 1e-4
    expected_gradient = np.array(
        [(reference(arguments64 + delta) - reference(arguments64 - delta)) / 2e-4 for delta in perturbations]
    )
    value, gradient = jax.jit(jax.value_and_grad(response))(arguments)

    np.testing.assert_allclose(value, reference(arguments64), rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=8e-6, atol=3e-6)


@pytest.mark.parametrize("covariance", _COVARIANCES)
def test_prepared_resolution_approximates_covariance_across_length_scale_range(covariance):
    time = np.linspace(-1, 1, 13, dtype=np.float32)
    config = prepare_hsgp((-1, 1), length_scale_range=(0.3, 1.3), covariance=covariance, boundary=9)
    basis, frequencies = config.basis(time)
    distances = time.astype(np.float64)[:, None] - time.astype(np.float64)[None, :]

    # The sizing heuristic leaves its largest spectral truncation error at
    # the shortest scale. The generous domain isolates that source of error.
    for length_scale, tolerance in ((0.3, 0.025), (0.7, 0.0025), (1.3, 0.0005)):
        weights = config.weights(frequencies, length_scale=length_scale, amplitude=1.3)
        weighted_basis = np.asarray(basis, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
        expected = 1.3**2 * _unit_covariance(distances, length_scale, covariance)

        np.testing.assert_allclose(weighted_basis @ weighted_basis.T, expected, rtol=0, atol=tolerance)


def test_more_basis_functions_cannot_correct_insufficient_domain_padding():
    time = np.linspace(-1, 1, 13, dtype=np.float32)
    config = prepare_hsgp((-1, 1), length_scale_range=(0.1, 0.2))
    more_modes = prepare_hsgp((-1, 1), length_scale_range=(0.1, 0.2), n_basis=4 * config.n_basis)
    wider = prepare_hsgp((-1, 1), length_scale_range=(0.1, 0.2), boundary=3, n_basis=more_modes.n_basis)
    distances = time.astype(np.float64)[:, None] - time.astype(np.float64)[None, :]
    expected = _unit_covariance(distances, 0.2, "matern52")
    errors = []

    for settings in (config, more_modes, wider):
        basis, frequencies = settings.basis(time)
        weights = settings.weights(frequencies, length_scale=0.2)
        weighted_basis = np.asarray(basis, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
        errors.append(np.max(np.abs(weighted_basis @ weighted_basis.T - expected)))

    assert config.boundary == more_modes.boundary == 1.2
    assert errors[0] > 0.13
    assert errors[1] > 0.13
    assert abs(errors[0] - errors[1]) < 0.001
    assert errors[2] < 1e-4


@pytest.mark.parametrize("argument", ["time_range", "length_scale_range"])
def test_prepare_names_nonreal_ranges_in_errors(argument):
    for value in ([True, False], [1j, 2j], ["1", "2"], None, np.array([1, 2], dtype=object)):
        arguments = {"time_range": (0, 4), "length_scale_range": (0.5, 2)}
        arguments[argument] = value
        with pytest.raises(TypeError, match=argument):
            prepare_hsgp(**arguments)


@pytest.mark.parametrize("argument", ["time_range", "length_scale_range"])
def test_prepare_requires_two_finite_increasing_endpoints(argument):
    invalid = [1, [], [1], [[1, 2]], [1, 2, 3], [1, 1], [2, 1], [np.nan, 2], [1, np.inf], [-np.inf, 2]]
    if argument == "length_scale_range":
        invalid.extend(([0, 2], [-1, 2]))
    for value in invalid:
        arguments = {"time_range": (0, 4), "length_scale_range": (0.5, 2)}
        arguments[argument] = value
        with pytest.raises(ValueError, match=argument):
            prepare_hsgp(**arguments)


def test_prepare_validates_manual_domain_and_basis_count():
    arguments = {"time_range": (0, 4), "length_scale_range": (0.5, 2)}
    for boundary in (True, [3], 3j, "3"):
        with pytest.raises(TypeError, match="boundary"):
            prepare_hsgp(**arguments, boundary=boundary)
    for boundary in (0, 1, 2, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="boundary"):
            prepare_hsgp(**arguments, boundary=boundary)
    for n_basis in (True, 1.5, np.int64(2)):
        with pytest.raises(TypeError, match="n_basis"):
            prepare_hsgp(**arguments, n_basis=n_basis)
    for n_basis in (0, -1):
        with pytest.raises(ValueError, match="n_basis"):
            prepare_hsgp(**arguments, n_basis=n_basis)


def test_prepare_rejects_unknown_covariance_names():
    for covariance in ("matern", "Matern52", "", None):
        with pytest.raises(ValueError, match="covariance"):
            prepare_hsgp((0, 4), length_scale_range=(0.5, 2), covariance=covariance)


def test_preparation_explains_that_dynamic_ranges_are_not_supported_under_jit():
    function = jax.jit(lambda time_range: prepare_hsgp(time_range, length_scale_range=(0.5, 2)).boundary)

    with pytest.raises(TypeError, match=r"time_range.*outside jax.jit"):
        function(jnp.array([0.0, 4.0]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_basis_matches_eigenfunctions_on_irregular_positions(dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    time = jnp.asarray([-3.0, -1.75, -0.125, 0.0, 2.3], dtype=dtype)
    center, boundary = dtype(-0.25), dtype(3.0)
    expected_basis, expected_frequencies = _basis_reference(time, center, boundary, 5)
    function = partial(hsgp_basis, n_basis=5)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14

    for basis, frequencies in (
        function(time, center=center, boundary=boundary),
        jax.jit(function)(time, center=center, boundary=boundary),
    ):
        assert basis.shape == (5, 5)
        assert frequencies.shape == (5,)
        assert basis.dtype == frequencies.dtype == dtype
        np.testing.assert_allclose(basis, expected_basis, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(frequencies, expected_frequencies, rtol=tolerance, atol=tolerance)


def test_basis_includes_both_zero_boundaries_and_has_no_constant_mode():
    basis, frequencies = hsgp_basis([-2.0, 1.0, 4.0], center=1.0, boundary=3.0, n_basis=4)

    np.testing.assert_allclose(np.asarray(basis)[[0, 2]], 0, rtol=0, atol=5e-7)
    np.testing.assert_allclose(basis[1], np.array([1, 0, -1, 0]) / np.sqrt(3), rtol=0, atol=5e-7)
    assert (frequencies > 0).all()
    assert (np.diff(frequencies) > 0).all()


def test_basis_is_orthonormal_over_the_complete_domain():
    time = np.linspace(-1.0, 7.0, 4097, dtype=np.float32)
    basis, _ = hsgp_basis(time, center=3.0, boundary=4.0, n_basis=9)
    basis64 = np.asarray(basis, dtype=np.float64)
    products = basis64[:, :, None] * basis64[:, None, :]
    gram = np.trapezoid(products, time.astype(np.float64), axis=0)

    np.testing.assert_allclose(gram, np.eye(9), rtol=0, atol=1e-6)


def test_basis_reuses_fixed_domain_for_training_and_prediction_and_common_shifts():
    time = np.array([-2.0, -0.5, 0.0, 1.25, 3.0, 4.0], dtype=np.float32)
    function = partial(hsgp_basis, center=1.0, boundary=4.0, n_basis=5)
    complete, frequencies = function(time)
    training, training_frequencies = function(time[:3])
    prediction, prediction_frequencies = function(time[3:])
    shifted, shifted_frequencies = hsgp_basis(time + 11, center=12.0, boundary=4.0, n_basis=5)

    np.testing.assert_array_equal(training, complete[:3])
    np.testing.assert_array_equal(prediction, complete[3:])
    np.testing.assert_array_equal(training_frequencies, frequencies)
    np.testing.assert_array_equal(prediction_frequencies, frequencies)
    np.testing.assert_array_equal(shifted_frequencies, frequencies)
    np.testing.assert_allclose(shifted, complete, rtol=0, atol=2e-6)
    assert not np.allclose(prediction[0], complete[0])


@pytest.mark.parametrize("time", [[], [0.25]])
def test_basis_accepts_empty_and_singleton_time(time):
    basis, frequencies = jax.jit(partial(hsgp_basis, n_basis=1))(jnp.asarray(time), center=0.0, boundary=2.0)
    expected_basis, expected_frequencies = _basis_reference(time, 0.0, 2.0, 1)

    assert basis.shape == (len(time), 1)
    np.testing.assert_allclose(basis, expected_basis, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(frequencies, expected_frequencies, rtol=2e-6)


def test_basis_jit_accepts_dynamic_time_center_and_boundary():
    function = jax.jit(hsgp_basis, static_argnames=("n_basis",))
    for time, center, boundary in [([-1.0, 0.3, 1.0], 0.0, 2.0), ([2.0, 3.5, 4.0], 3.0, 1.5)]:
        basis, frequencies = function(jnp.asarray(time), center=center, boundary=boundary, n_basis=3)
        expected_basis, expected_frequencies = _basis_reference(time, center, boundary, 3)
        np.testing.assert_allclose(basis, expected_basis, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(frequencies, expected_frequencies, rtol=2e-6)


def test_basis_time_center_and_boundary_gradients_match_analytic_derivatives():
    arguments = jnp.array([0.37, -0.2, 2.7], dtype=jnp.float32)
    coefficients = jnp.array([0.3, -0.5, 1.2, 0.7], dtype=jnp.float32)
    time, center, boundary = np.asarray(arguments, dtype=np.float64)
    basis, frequencies = _basis_reference([time], center, boundary, 4)
    angles = frequencies * (time - center + boundary)
    time_derivative = np.cos(angles) * frequencies / np.sqrt(boundary)
    boundary_derivative = -basis[0] / (2 * boundary) - time_derivative * (time - center) / boundary
    expected = np.array([time_derivative, -time_derivative, boundary_derivative]) @ np.asarray(coefficients)

    def response(values):
        features, _ = hsgp_basis(values[:1], center=values[1], boundary=values[2], n_basis=4)
        return features[0] @ coefficients

    gradient = jax.jit(jax.grad(response))(arguments)

    np.testing.assert_allclose(gradient, expected, rtol=5e-6, atol=2e-6)


@pytest.mark.parametrize("covariance", _COVARIANCES)
def test_weights_match_numerical_cosine_transform_of_stationary_covariance(covariance):
    frequencies = jnp.array([-8.0, -2.3, -0.5, 0.0, 0.5, 2.3, 8.0])
    expected = _weights_from_covariance(np.asarray(frequencies), 0.7, 1.3, covariance)
    function = partial(hsgp_weights, covariance=covariance)

    for result in (
        function(frequencies, length_scale=0.7, amplitude=1.3),
        jax.jit(function)(frequencies, length_scale=0.7, amplitude=1.3),
    ):
        assert result.shape == (7,)
        np.testing.assert_allclose(result, expected, rtol=5e-6, atol=2e-7)
        np.testing.assert_array_equal(result[:3], result[-1:-4:-1])


@pytest.mark.parametrize("covariance", _COVARIANCES)
def test_weighted_basis_reconstructs_stationary_covariance_in_domain_interior(covariance):
    time = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
    basis, frequencies = hsgp_basis(time, center=0.0, boundary=5.0, n_basis=256)
    weights = hsgp_weights(frequencies, length_scale=0.7, amplitude=1.3, covariance=covariance)
    weighted_basis = np.asarray(basis, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
    expected = 1.3**2 * _unit_covariance(time[:, None] - time[None, :], 0.7, covariance)

    # The interior avoids the finite domain's imposed zero boundary values.
    # The Matérn 3/2 tail is the largest truncation error at 256 basis terms.
    np.testing.assert_allclose(weighted_basis @ weighted_basis.T, expected, rtol=0, atol=5e-5)


@pytest.mark.parametrize("covariance", _COVARIANCES)
def test_weights_gradients_match_finite_differences_of_covariance_transform(covariance):
    arguments = jnp.array([0.3, 1.8, 4.5, 0.75, 1.3], dtype=jnp.float32)
    coefficients = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float32)

    def response(values):
        return (
            hsgp_weights(values[:3], length_scale=values[3], amplitude=values[4], covariance=covariance) @ coefficients
        )

    def reference(values):
        return _weights_from_covariance(values[:3], values[3], values[4], covariance) @ np.asarray(coefficients)

    arguments64 = np.asarray(arguments, dtype=np.float64)
    perturbations = np.eye(len(arguments)) * 1e-4
    expected = np.array(
        [(reference(arguments64 + delta) - reference(arguments64 - delta)) / 2e-4 for delta in perturbations]
    )
    gradient = jax.jit(jax.grad(response))(arguments)

    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("covariance", _COVARIANCES)
def test_zero_amplitude_is_valid_and_has_finite_correct_gradients(covariance):
    frequencies = jnp.array([0.0, 0.5, 2.0])

    def response(length_scale, amplitude):
        return hsgp_weights(frequencies, length_scale=length_scale, amplitude=amplitude, covariance=covariance).sum()

    result = hsgp_weights(frequencies, length_scale=0.7, amplitude=0.0, covariance=covariance)
    gradient = jax.jit(jax.grad(response, argnums=(0, 1)))(0.7, 0.0)
    expected_amplitude_gradient = _weights_from_covariance(frequencies, 0.7, 1.0, covariance).sum()

    np.testing.assert_array_equal(result, np.zeros(3))
    np.testing.assert_allclose(gradient, [0.0, expected_amplitude_gradient], rtol=4e-6, atol=2e-6)


def test_expquad_underflow_has_finite_zero_values_and_gradients():
    frequencies = jnp.array([100.0, 1000.0, 10_000.0])

    def response(values):
        return hsgp_weights(values[:3], length_scale=values[3], amplitude=values[4], covariance="expquad").sum()

    arguments = jnp.concatenate((frequencies, jnp.array([1.0, 1.3])))
    value, gradient = jax.jit(jax.value_and_grad(response))(arguments)

    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, np.zeros(5))


def test_shared_basis_supports_group_and_channel_parameter_broadcasting():
    time = jnp.array([-1.0, -0.3, 0.5, 1.0, 1.7])
    basis, frequencies = hsgp_basis(time, center=0.5, boundary=3.0, n_basis=4)
    length_scale = jnp.array([[0.5], [1.2]])
    amplitude = jnp.array([0.0, 0.7, 1.3])
    coefficients = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 24 - 0.5
    weights = jax.jit(hsgp_weights)(frequencies, length_scale=length_scale, amplitude=amplitude)
    response = jax.jit(lambda features, scale, z: jnp.einsum("tm,gcm->tgc", features, scale * z))(
        basis, weights, coefficients
    )
    expected_weights = np.stack(
        [
            np.stack([_weights_from_covariance(frequencies, ell, amp, "matern52") for amp in amplitude])
            for ell in np.asarray(length_scale).ravel()
        ]
    )
    expected_basis, _ = _basis_reference(time, 0.5, 3.0, 4)
    expected = np.einsum("tm,gcm->tgc", expected_basis, expected_weights * np.asarray(coefficients))
    group_weights = hsgp_weights(frequencies, length_scale=length_scale[:, 0])

    assert basis.shape == (5, 4)
    assert weights.shape == (2, 3, 4)
    assert group_weights.shape == (2, 4)
    assert response.shape == (5, 2, 3)
    np.testing.assert_allclose(weights, expected_weights, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(response, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_array_equal(response[:, :, 0], np.zeros((5, 2)))


def test_weights_vmap_matches_broadcasting_with_dynamic_parameters():
    frequencies = jnp.array([0.0, 0.3, 1.7])
    scales = jnp.array([0.3, 0.7, 1.8])
    amplitudes = jnp.array([0.5, 1.3, 2.0])
    function = jax.jit(
        jax.vmap(
            lambda omega, ell, amp: hsgp_weights(omega, length_scale=ell, amplitude=amp),
            in_axes=(None, 0, 0),
        )
    )

    result = function(frequencies, scales, amplitudes)
    expected = hsgp_weights(frequencies, length_scale=scales, amplitude=amplitudes)

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_weighted_curves_are_invariant_to_common_time_units():
    time = jnp.array([-0.5, 0.0, 0.75, 2.0])
    coefficients = jnp.array([0.3, -0.5, 1.1, 0.7])
    weekly_basis, weekly_frequencies = hsgp_basis(time, center=0.5, boundary=3.0, n_basis=4)
    daily_basis, daily_frequencies = hsgp_basis(time * 7, center=3.5, boundary=21.0, n_basis=4)

    for covariance in _COVARIANCES:
        weekly_weights = hsgp_weights(weekly_frequencies, length_scale=0.7, amplitude=1.3, covariance=covariance)
        daily_weights = hsgp_weights(daily_frequencies, length_scale=4.9, amplitude=1.3, covariance=covariance)
        np.testing.assert_allclose(
            weekly_basis @ (weekly_weights * coefficients),
            daily_basis @ (daily_weights * coefficients),
            rtol=4e-6,
            atol=2e-6,
        )


def test_weights_accepts_empty_frequencies_with_parameter_batch_dimensions():
    result = jax.jit(hsgp_weights)(jnp.empty(0), length_scale=jnp.ones((2, 1)), amplitude=jnp.ones(3))

    assert result.shape == (2, 3, 0)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float16, jnp.bfloat16, jnp.float32])
def test_hsgp_uses_at_least_float32(dtype):
    basis, frequencies = hsgp_basis(jnp.asarray([-1, 0, 1], dtype=dtype), center=dtype(0), boundary=dtype(2), n_basis=2)
    weights = hsgp_weights(jnp.asarray([0, 1, 2], dtype=dtype), length_scale=dtype(1), amplitude=dtype(2))

    assert basis.dtype == frequencies.dtype == weights.dtype == jnp.float32
    np.testing.assert_allclose(basis, _basis_reference([-1, 0, 1], 0, 2, 2)[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(weights, _weights_from_covariance([0, 1, 2], 1, 2, "matern52"), rtol=3e-6)


def test_hsgp_preserves_common_float64_dtype_when_enabled():
    if not jax.config.x64_enabled:
        pytest.skip("JAX 64-bit mode is disabled")
    basis, frequencies = hsgp_basis(
        jnp.array([-1, 0, 1], dtype=jnp.float32), center=0.0, boundary=jnp.float64(2), n_basis=3
    )
    weights = hsgp_weights(frequencies.astype(jnp.float32), length_scale=jnp.float64(0.7), amplitude=1.3)
    expected_basis, expected_frequencies = _basis_reference([-1, 0, 1], 0, 2, 3)
    expected_weights = _weights_from_covariance(np.asarray(frequencies, dtype=np.float32), 0.7, 1.3, "matern52")

    assert basis.dtype == frequencies.dtype == weights.dtype == jnp.float64
    np.testing.assert_allclose(basis, expected_basis, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(frequencies, expected_frequencies, rtol=2e-14)
    np.testing.assert_allclose(weights, expected_weights, rtol=2e-12, atol=2e-12)


def test_basis_converts_large_integer_positions_before_narrowing():
    time = np.array([0, 3_000_000_000, 6_000_000_000], dtype=np.int64)
    basis, frequencies = hsgp_basis(time, center=3_000_000_000, boundary=6_000_000_000, n_basis=2)
    expected_basis, expected_frequencies = _basis_reference(time, 3_000_000_000, 6_000_000_000, 2)

    np.testing.assert_allclose(basis, expected_basis, rtol=3e-6, atol=1e-11)
    np.testing.assert_allclose(frequencies, expected_frequencies, rtol=3e-6)


def test_basis_masks_only_nonfinite_and_outside_domain_rows():
    time = jnp.array([-2.01, -2.0, -0.5, 2.0, 2.01, jnp.nan, jnp.inf, -jnp.inf])
    function = partial(hsgp_basis, center=0.0, boundary=2.0, n_basis=3)

    for basis, frequencies in (function(time), jax.jit(function)(time)):
        expected_basis, expected_frequencies = _basis_reference(np.asarray(time)[[1, 2, 3]], 0.0, 2.0, 3)
        np.testing.assert_allclose(np.asarray(basis)[[1, 2, 3]], expected_basis, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(frequencies, expected_frequencies, rtol=2e-6)
        assert np.isnan(np.asarray(basis)[[0, 4, 5, 6, 7]]).all()


@pytest.mark.parametrize("boundary", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_boundary_masks_basis_and_frequencies(boundary):
    function = partial(hsgp_basis, n_basis=3)
    for basis, frequencies in (
        function([0.0, 1.0], center=0.0, boundary=boundary),
        jax.jit(function)(jnp.array([0.0, 1.0]), center=0.0, boundary=boundary),
    ):
        assert basis.shape == (2, 3)
        assert frequencies.shape == (3,)
        assert np.isnan(basis).all()
        assert np.isnan(frequencies).all()


def test_invalid_center_masks_basis_and_keeps_valid_domain_frequencies():
    function = jax.jit(partial(hsgp_basis, n_basis=3))
    for center in (np.nan, np.inf, -np.inf):
        basis, frequencies = function(jnp.array([0.0, 1.0]), center=center, boundary=2.0)
        assert np.isnan(basis).all()
        np.testing.assert_allclose(frequencies, _basis_reference([], 0.0, 2.0, 3)[1], rtol=2e-6)


def test_invalid_weight_parameters_mask_only_affected_broadcast_batches():
    length_scale = jnp.array([0.7, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])[:, None]
    amplitude = jnp.array([1.3, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
    frequencies = jnp.array([0.0, 0.5, 2.0])
    function = partial(hsgp_weights, covariance="matern32")

    for result in (
        function(frequencies, length_scale=length_scale, amplitude=amplitude),
        jax.jit(function)(frequencies, length_scale=length_scale, amplitude=amplitude),
    ):
        assert result.shape == (6, 6, 3)
        np.testing.assert_allclose(result[0, 0], _weights_from_covariance(frequencies, 0.7, 1.3, "matern32"), rtol=3e-6)
        np.testing.assert_array_equal(result[0, 1], np.zeros(3))
        assert np.isnan(result[1:]).all()
        assert np.isnan(result[0, 2:]).all()


def test_nonfinite_frequencies_mask_columns_even_at_zero_amplitude():
    frequencies = jnp.array([-1.0, jnp.nan, 0.0, jnp.inf, -jnp.inf, 2.0])
    function = partial(hsgp_weights, length_scale=jnp.array([0.7, 1.0]), amplitude=jnp.array([1.3, 0.0]))

    for result in (function(frequencies), jax.jit(function)(frequencies)):
        assert result.shape == (2, 6)
        assert np.isfinite(np.asarray(result)[:, [0, 2, 5]]).all()
        assert np.isnan(np.asarray(result)[:, [1, 3, 4]]).all()
        np.testing.assert_array_equal(np.asarray(result)[1, [0, 2, 5]], np.zeros(3))


@pytest.mark.parametrize("argument", ["time", "center", "boundary", "frequencies", "length_scale", "amplitude"])
def test_hsgp_names_nonreal_inputs_in_errors(argument):
    basis_arguments = {"time": [0.0, 1.0], "center": 0.0, "boundary": 2.0, "n_basis": 2}
    weight_arguments = {"frequencies": [0.0, 1.0], "length_scale": 0.7, "amplitude": 1.3}
    arguments = basis_arguments if argument in basis_arguments else weight_arguments
    function = hsgp_basis if argument in basis_arguments else hsgp_weights

    for value in (True, 1 + 2j, "1", None, np.array([1], dtype=object)):
        arguments[argument] = value
        with pytest.raises(TypeError, match=argument):
            function(**arguments)


def test_hsgp_names_invalid_input_dimensions_in_errors():
    for argument, values in (
        ("time", [1.0, [[0.0], [1.0]]]),
        ("center", [[0.0], [[0.0]]]),
        ("boundary", [[2.0], [[2.0]]]),
    ):
        for value in values:
            arguments = {"time": [0.0, 1.0], "center": 0.0, "boundary": 2.0, "n_basis": 2}
            arguments[argument] = value
            with pytest.raises(ValueError, match=argument):
                hsgp_basis(**arguments)
    for frequencies in (1.0, [[0.0], [1.0]]):
        with pytest.raises(ValueError, match="frequencies"):
            hsgp_weights(frequencies, length_scale=0.7)


@pytest.mark.parametrize("n_basis", [True, 1.5, np.int64(2)])
def test_basis_requires_python_integer_basis_count(n_basis):
    with pytest.raises(TypeError, match="n_basis"):
        hsgp_basis([0.0], center=0.0, boundary=2.0, n_basis=n_basis)


@pytest.mark.parametrize("n_basis", [0, -1])
def test_basis_requires_positive_basis_count(n_basis):
    with pytest.raises(ValueError, match="n_basis"):
        hsgp_basis([0.0], center=0.0, boundary=2.0, n_basis=n_basis)


def test_basis_explains_that_jit_basis_count_must_be_static():
    with pytest.raises(TypeError, match="n_basis"):
        jax.jit(hsgp_basis)(jnp.array([0.0]), center=0.0, boundary=2.0, n_basis=2)


def test_weights_reject_unknown_covariance_names():
    for covariance in ("matern", "Matern52", "", None):
        with pytest.raises(ValueError, match="covariance"):
            hsgp_weights([0.0, 1.0], length_scale=0.7, covariance=covariance)


def test_weights_reject_incompatible_parameter_batch_shapes():
    with pytest.raises(ValueError, match=r"broadcast|shape|length_scale|amplitude"):
        hsgp_weights([0.0, 1.0], length_scale=jnp.ones(2), amplitude=jnp.ones(3))


def test_hsgp_effect_is_exported_immutable_and_declares_actual_parameter_names():
    effect = HSGPEffect(length_scale_range=(1.0, 4.0), n_basis=3)
    prepared = effect._prepare(_data([10.0, 12.0, 15.0]))

    assert mmmjax.HSGPEffect is HSGPEffect
    assert "HSGPEffect" in mmmjax.__all__
    assert effect.name == "baseline"
    assert effect.covariance == "matern52"
    assert effect.boundary is None
    assert effect.n_basis == 3
    assert effect.demean is True
    assert list(prepared.parameters) == [
        "baseline_coefficients",
        "baseline_length_scale",
        "baseline_amplitude",
    ]
    assert isinstance(prepared.parameters["baseline_coefficients"], Real)
    assert prepared.parameters["baseline_coefficients"].shape == (3,)
    assert isinstance(prepared.parameters["baseline_length_scale"], Positive)
    assert isinstance(prepared.parameters["baseline_amplitude"], Positive)
    for attribute, value in (("name", "trend"), ("covariance", "expquad"), ("demean", False)):
        with pytest.raises(FrozenInstanceError):
            setattr(effect, attribute, value)


def test_numeric_basis_and_spectral_weighted_curve_match_independent_formula():
    prepared = HSGPEffect(length_scale_range=(1.0, 4.0), boundary=8.0, n_basis=3, demean=False)._prepare(
        _data([15.0, 10.0, 12.0])
    )
    coefficients = np.array([0.3, -0.7, 1.1])
    positions = np.array([0.0, 2.0, 5.0])
    center = 2.5
    modes = np.arange(1, 4)
    frequencies = np.pi * modes / 16.0
    expected_basis = np.sin((positions - center + 8.0)[:, None] * frequencies) / np.sqrt(8.0)
    expected = _matern52_curve(prepared, coefficients, 2.5, 0.8)

    assert prepared.origin == 10.0
    np.testing.assert_allclose(prepared.features, expected_basis, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(prepared.frequencies, frequencies, rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(prepared.basis_mean, np.zeros(3))
    np.testing.assert_allclose(prepared.apply(_values(prepared, coefficients)), expected, rtol=5e-6, atol=3e-6)


def test_demeaning_uses_frozen_training_mean_for_subsets_and_forecasts():
    prepared = HSGPEffect(length_scale_range=(1.0, 4.0), boundary=10.0, n_basis=3)._prepare(_data([0.0, 2.0, 5.0, 7.0]))
    parameters = _values(prepared)
    expected_mean = np.asarray(prepared.features).mean(axis=0)
    subset = prepared.for_data(_data([2.0, 7.0], outcome=False))
    future = subset.for_data(_data([8.0, 9.0], outcome=False))

    np.testing.assert_allclose(prepared.basis_mean, expected_mean, rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(subset.basis_mean, prepared.basis_mean)
    np.testing.assert_array_equal(future.basis_mean, prepared.basis_mean)
    assert subset.config == future.config == prepared.config
    assert subset.origin == future.origin == prepared.origin
    np.testing.assert_allclose(prepared.apply(parameters).mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        subset.apply(parameters), np.asarray(prepared.apply(parameters))[[1, 3]], rtol=5e-6, atol=3e-6
    )
    np.testing.assert_allclose(
        subset.apply(parameters), _matern52_curve(subset, [0.3, -0.7, 1.1], 2.5, 0.8), rtol=5e-6, atol=3e-6
    )
    np.testing.assert_allclose(
        future.apply(parameters), _matern52_curve(future, [0.3, -0.7, 1.1], 2.5, 0.8), rtol=5e-6, atol=3e-6
    )


def test_calendar_time_uses_elapsed_days_and_ignores_media_history():
    history = pl.DataFrame({"time": [date(2024, 2, 26), date(2024, 2, 27)], "video": [1.0, 1.0]})
    labels = [date(2024, 2, 28), date(2024, 3, 1), date(2024, 3, 4)]
    with_history = HSGPEffect(length_scale_range=(1.0, 5.0), boundary=10.0, n_basis=2)._prepare(
        _data(labels, media_history=history)
    )
    without_history = HSGPEffect(length_scale_range=(1.0, 5.0), boundary=10.0, n_basis=2)._prepare(_data(labels))
    frequencies = np.pi * np.arange(1, 3) / 20.0
    expected = np.sin((np.array([0.0, 2.0, 5.0]) - 2.5 + 10.0)[:, None] * frequencies) / np.sqrt(10.0)

    assert len(with_history.features) == 3
    np.testing.assert_allclose(with_history.features, expected, rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(with_history.features, without_history.features)


def test_shared_curve_broadcasts_over_groups_and_retains_group_metadata():
    data = _data([0.0, 1.0, 3.0], groups=("west", "east", "central"))
    prepared = HSGPEffect(length_scale_range=(0.5, 3.0), boundary=8.0, n_basis=3)._prepare(data)
    parameters = _values(prepared)
    curve = _matern52_curve(prepared, [0.3, -0.7, 1.1], 2.5, 0.8)
    result = jax.jit(lambda component, values: component.apply(values))(prepared, parameters)

    assert prepared.group_columns == ("region",)
    assert prepared.group_values == (("west",), ("east",), ("central",))
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, np.broadcast_to(curve[:, None], result.shape), rtol=5e-6, atol=3e-6)


def test_forecasts_reuse_training_domain_and_reject_incompatible_or_outside_times():
    prepared = HSGPEffect(length_scale_range=(1.0, 4.0), boundary=6.0, n_basis=3)._prepare(_data([10.0, 12.0, 14.0]))
    forecast = prepared.for_data(_data([15.0, 16.0], outcome=False))

    assert forecast.config == prepared.config
    assert forecast.origin == 10.0
    np.testing.assert_array_equal(forecast.basis_mean, prepared.basis_mean)
    with pytest.raises(ValueError, match=r"domain|boundary|outside"):
        prepared.for_data(_data([21.0], outcome=False))
    with pytest.raises((TypeError, ValueError), match=r"time|calendar|numeric"):
        prepared.for_data(_data(["2026-01-01"], outcome=False))
    with pytest.raises(ValueError, match=r"time|column|group"):
        prepared.for_data(replace(_data([15.0], outcome=False), time_column="date"))


def test_configuration_and_apply_validation_is_focused_on_component_contract():
    for options in (
        {"length_scale_range": (0.0, 2.0)},
        {"length_scale_range": (2.0, 1.0)},
        {"length_scale_range": (1.0, np.inf)},
        {"length_scale_range": (1.0, 2.0), "covariance": "periodic"},
        {"length_scale_range": (1.0, 2.0), "boundary": -1.0},
        {"length_scale_range": (1.0, 2.0), "n_basis": 0},
        {"length_scale_range": (1.0, 2.0), "name": "not-valid"},
    ):
        with pytest.raises((TypeError, ValueError)):
            HSGPEffect(**options)
    with pytest.raises(TypeError, match="demean"):
        HSGPEffect(length_scale_range=(1.0, 2.0), demean=1)
    with pytest.raises(TypeError, match=r"data|PreparedData"):
        HSGPEffect(length_scale_range=(1.0, 2.0))._prepare({"time": [0, 1]})

    prepared = HSGPEffect(length_scale_range=(1.0, 2.0), boundary=4.0, n_basis=2)._prepare(_data([0, 1]))
    valid = _values(prepared, coefficients=(0.2, -0.3))
    for changed in (
        {**valid, "baseline_coefficients": jnp.ones(3)},
        {**valid, "baseline_length_scale": jnp.ones(1)},
        {key: value for key, value in valid.items() if key != "baseline_amplitude"},
    ):
        with pytest.raises((TypeError, ValueError, KeyError), match=r"parameter|coefficient|shape|baseline|key"):
            prepared.apply(changed)

    np.testing.assert_array_equal(prepared.apply({**valid, "intercept": 1.0}), prepared.apply(valid))


def test_apply_supports_jit_grad_and_vmap_without_changing_the_mapping_contract():
    prepared = HSGPEffect(length_scale_range=(1.0, 4.0), boundary=8.0, n_basis=3)._prepare(_data([0.0, 1.0, 3.0]))
    parameters = _values(prepared)
    evaluate = jax.jit(lambda values: prepared.apply(values))
    observation_weights = jnp.array([0.2, -0.3, 0.7])
    gradient = jax.jit(jax.grad(lambda values: observation_weights @ prepared.apply(values)))(parameters)
    batched = jax.vmap(prepared.apply)(
        {
            "baseline_coefficients": jnp.stack(
                (parameters["baseline_coefficients"], -parameters["baseline_coefficients"])
            ),
            "baseline_length_scale": jnp.array([2.5, 2.0]),
            "baseline_amplitude": jnp.array([0.8, 0.4]),
        }
    )

    np.testing.assert_allclose(
        evaluate(parameters), _matern52_curve(prepared, [0.3, -0.7, 1.1], 2.5, 0.8), rtol=5e-6, atol=3e-6
    )
    assert set(gradient) == set(parameters)
    assert all(np.isfinite(value).all() for value in gradient.values())
    np.testing.assert_allclose(
        gradient["baseline_amplitude"],
        (observation_weights @ evaluate(parameters)) / parameters["baseline_amplitude"],
        rtol=5e-6,
        atol=3e-6,
    )
    step = 1e-3
    length_derivative = (
        np.asarray(observation_weights)
        @ (
            _matern52_curve(prepared, [0.3, -0.7, 1.1], 2.5 + step, 0.8)
            - _matern52_curve(prepared, [0.3, -0.7, 1.1], 2.5 - step, 0.8)
        )
        / (2 * step)
    )
    np.testing.assert_allclose(gradient["baseline_length_scale"], length_derivative, rtol=2e-5, atol=3e-6)
    assert batched.shape == (2, 3)


def test_model_density_has_only_explicit_priors_likelihood_and_positive_jacobians():
    data = _data([0.0, 1.0, 3.0])
    specification = HSGPEffect(length_scale_range=(1.0, 4.0), boundary=8.0, n_basis=3)

    def log_density(outcome, baseline, baseline_coefficients, baseline_length_scale, baseline_amplitude):
        return (
            normal(baseline_coefficients, 0.0, 1.0)
            + normal(baseline_length_scale, 0.0, 1.0)
            + normal(baseline_amplitude, 0.0, 1.0)
            + normal(outcome, baseline, 1.0)
        )

    model = Model({}, log_density, data=data, components=[specification])
    constrained = _values(model.data.components[0])
    position = model.unconstrain(constrained)
    actual = jax.jit(model.log_density)(position, model.data)
    curve = np.asarray(model.data.components[0].apply(constrained))
    outcome = np.asarray(data.arrays["outcome"])
    expected = (
        _normal_reference(constrained["baseline_coefficients"])
        + _normal_reference(constrained["baseline_length_scale"])
        + _normal_reference(constrained["baseline_amplitude"])
        + _normal_reference(outcome - curve)
        + np.log(float(constrained["baseline_length_scale"]))
        + np.log(float(constrained["baseline_amplitude"]))
    )

    assert set(model.parameters) == {
        "baseline_coefficients",
        "baseline_length_scale",
        "baseline_amplitude",
    }
    np.testing.assert_allclose(actual, expected, rtol=6e-6, atol=4e-6)


def test_prior_and_scenario_generation_preserve_parameter_and_effect_axes():
    training = _data([0.0, 1.0, 3.0], groups=("west", "east"))

    def generate(key, baseline, baseline_coefficients):
        del key
        return {"prediction": baseline, "coefficient_copy": baseline_coefficients}

    def prior(key):
        coefficient_key, length_key, amplitude_key = jax.random.split(key, 3)
        return {
            "baseline_coefficients": jax.random.normal(coefficient_key, (3,)),
            "baseline_length_scale": jnp.exp(jax.random.normal(length_key)),
            "baseline_amplitude": jnp.exp(jax.random.normal(amplitude_key)),
        }

    model = Model(
        {},
        lambda outcome, baseline: normal(outcome, baseline, 1.0),
        generate,
        prior=prior,
        data=training,
        components=[HSGPEffect(length_scale_range=(0.5, 4.0), boundary=8.0, n_basis=3)],
        predictive=("prediction",),
    )
    draws = sample_prior(model, draws=3, seed=7)

    assert draws["prior"]["baseline_coefficients"].dims == ("chain", "draw", "baseline_basis")
    np.testing.assert_array_equal(draws["prior"]["baseline_basis"], [1, 2, 3])
    assert draws["prior"]["baseline_length_scale"].dims == ("chain", "draw")
    assert draws["prior_generated_quantities"]["coefficient_copy"].dims == (
        "chain",
        "draw",
        "baseline_basis",
    )
    assert draws["prior_predictive"]["prediction"].dims == ("chain", "draw", "time", "group")
    np.testing.assert_array_equal(draws["prior_predictive"]["group"], ["west", "east"])

    posterior = _collect_results(
        {
            name: draws["prior"][name].values
            for name in ("baseline_coefficients", "baseline_length_scale", "baseline_amplitude")
        },
        data=training,
        dims={"baseline_coefficients": ("baseline_basis",)},
        coords={"baseline_basis": [1, 2, 3]},
    )
    scenario = _data([4.0, 5.0], groups=("east", "west"), outcome=False)
    generated = generate_quantities(model, posterior, new_data=scenario, seed=11)
    assert generated["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time", "group")
    np.testing.assert_array_equal(generated["posterior_predictive"]["group"], ["west", "east"])
    assert generated["posterior_predictive"]["prediction"].shape == (1, 3, 2, 2)


def test_hsgp_composes_with_media_seasonality_and_user_parameters():
    data = _data([0.0, 1.0, 2.0, 3.0])

    def transformed(intercept, baseline, annual, paid_media_total):
        return {"mean": intercept + baseline + annual + paid_media_total}

    def density(outcome, mean, baseline_coefficients):
        return normal(outcome, mean, 1.0) + normal(baseline_coefficients, 0.0, 1.0)

    def generate(key, mean, baseline, annual, paid_media_total):
        return {"mean": mean, "baseline": baseline, "annual": annual, "media": paid_media_total}

    model = Model(
        {"intercept": Real()},
        density,
        generate,
        transformed_parameters=transformed,
        data=data,
        components=[
            HSGPEffect(length_scale_range=(1.0, 4.0), n_basis=3),
            FourierSeasonality(period=7, order=1, name="annual"),
            MediaEffect(max_lag=1),
        ],
    )
    position = model.initialize_random(jax.random.key(8))
    constrained = model.constrain(position)
    value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
    original = jax.jit(model.generate)(jax.random.key(1), constrained, model.data)
    changed = model.generate(jax.random.key(1), {**constrained, "intercept": constrained["intercept"] + 1}, model.data)

    assert np.isfinite(value)
    assert all(np.isfinite(value).all() for value in gradient.values())
    np.testing.assert_allclose(
        original["mean"],
        constrained["intercept"] + original["baseline"] + original["annual"] + original["media"],
        rtol=4e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(changed["mean"] - original["mean"], 1.0, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(changed["baseline"], original["baseline"], rtol=4e-6, atol=3e-6)


def test_prior_uses_automatically_prepared_basis_shape_after_model_construction():
    def prior(key):
        return {
            "trend_coefficients": jax.random.normal(key, model.parameters["trend_coefficients"].shape),
            "trend_length_scale": 2.0,
            "trend_amplitude": 0.5,
        }

    model = Model(
        {},
        lambda trend: jnp.sum(trend),
        lambda key, trend: {"curve": trend},
        prior=prior,
        data=_data([0, 1, 2, 3]),
        components=[HSGPEffect(length_scale_range=(1.0, 4.0), name="trend")],
    )
    results = sample_prior(model, draws=2)

    assert results["prior"]["trend_coefficients"].shape == (1, 2, *model.parameters["trend_coefficients"].shape)
    assert results["prior_generated_quantities"]["curve"].dims == ("chain", "draw", "time")
    np.testing.assert_allclose(results["prior_generated_quantities"]["curve"].mean("time"), 0.0, atol=1e-7)
