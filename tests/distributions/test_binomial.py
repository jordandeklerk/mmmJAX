"""Tests for Binomial distribution functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betainc
from jax.scipy.stats import binom as jax_binomial_distribution
from scipy import special, stats

from mmmjax import (
    bernoulli_logit_logpmf,
    bernoulli_logpmf,
    binomial,
    binomial_logcdf,
    binomial_logit,
    binomial_logit_logcdf,
    binomial_logit_logpmf,
    binomial_logit_logsf,
    binomial_logit_rng,
    binomial_logpmf,
    binomial_logsf,
    binomial_rng,
)


def test_binomial_logpmf_matches_scipy_across_support_and_broadcasting() -> None:
    values = np.array([[-1.0], [0.0], [1.0], [2.0], [5.0], [0.5], [np.nan]], dtype=np.float32)
    trials = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    probabilities = np.array([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
    expected = stats.binom.logpmf(
        values.astype(np.float64),
        trials.astype(np.float64),
        probabilities.astype(np.float64),
    )

    result = binomial_logpmf(values, trials, probabilities)

    assert result.shape == (7, 4)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-6, equal_nan=True)


def test_binomial_logpmf_matches_jax_on_valid_integer_support() -> None:
    values = jnp.array([[0], [1], [2]])
    trials = jnp.array([2, 3, 5])
    probabilities = jnp.array([0.0, 0.3, 0.8])
    expected = jax_binomial_distribution.logpmf(values, trials, probabilities)

    result = binomial_logpmf(values, trials, probabilities)
    compiled = jax.jit(binomial_logpmf)(values, trials, probabilities)

    assert jnp.allclose(result, expected, rtol=3e-6, atol=2e-6)
    assert jnp.allclose(compiled, result, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("value", "trials", "probability", "expected"),
    [
        (0, 0, 0.0, 0.0),
        (0, 0, 0.4, 0.0),
        (0, 0, 1.0, 0.0),
        (0, 5, 0.0, 0.0),
        (1, 5, 0.0, -jnp.inf),
        (4, 5, 1.0, -jnp.inf),
        (5, 5, 1.0, 0.0),
        (2, 4, 0.5, -0.9808292530117262),
    ],
)
def test_binomial_logpmf_matches_known_values(
    value: int,
    trials: int,
    probability: float,
    expected: float,
) -> None:
    result = binomial_logpmf(value, trials, probability)

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(("trials", "probability"), [(0, 0.3), (1, 0.8), (5, 0.2), (20, 0.65)])
def test_binomial_probability_mass_normalizes(trials: int, probability: float) -> None:
    values = jnp.arange(trials + 1)

    total_probability = jnp.sum(jnp.exp(binomial_logpmf(values, trials, probability)))

    assert jnp.allclose(total_probability, 1.0, rtol=2e-6, atol=2e-6)


def test_binomial_with_one_trial_matches_bernoulli() -> None:
    values = jnp.array([0, 1])
    probabilities = jnp.array([0.2, 0.8])

    assert jnp.allclose(binomial_logpmf(values, 1, probabilities), bernoulli_logpmf(values, probabilities))


def test_binomial_logpmf_is_symmetric_under_success_failure_exchange() -> None:
    values = jnp.array([0, 1, 3, 7])
    trials = jnp.array([7, 7, 7, 7])
    probabilities = jnp.array([0.1, 0.3, 0.6, 0.9])

    result = binomial_logpmf(values, trials, probabilities)
    reflected = binomial_logpmf(trials - values, trials, 1 - probabilities)

    assert jnp.allclose(result, reflected, rtol=3e-6, atol=2e-6)


def test_binomial_logpmf_requires_exact_integer_values() -> None:
    values = jnp.array([-jnp.inf, -1.0, -0.0, 0.5, 2.0, 2.5, 5.0, 6.0, jnp.inf, jnp.nan])

    result = binomial_logpmf(values, 5, 0.4)

    assert jnp.all(jnp.isneginf(result[jnp.array([0, 1, 3, 5, 7, 8])]))
    assert jnp.all(jnp.isfinite(result[jnp.array([2, 4, 6])]))
    assert jnp.isnan(result[9])


def test_binomial_logpmf_rejects_invalid_trials_before_support() -> None:
    trials = jnp.array([-jnp.inf, -1.0, 2.5, jnp.inf, jnp.nan])

    result = binomial_logpmf(10, trials, 0.4)

    assert jnp.all(jnp.isnan(result))


def test_binomial_logpmf_rejects_invalid_probability_before_support() -> None:
    probabilities = jnp.array([-jnp.inf, -0.1, 1.1, jnp.inf, jnp.nan])

    result = binomial_logpmf(10, 5, probabilities)

    assert jnp.all(jnp.isnan(result))


def test_binomial_sums_broadcast_log_masses() -> None:
    values = jnp.array([0, 2, 4])
    trials = jnp.asarray(4)
    probability = jnp.asarray(0.3)
    expected = np.sum(stats.binom.logpmf(np.asarray(values), 4, 0.3))

    result = binomial(values, trials, probability)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_binomial_empty_batch_returns_scalar_zero() -> None:
    values = jnp.empty((0,), dtype=jnp.int32)

    assert binomial(values, 5, 0.4) == 0
    assert jax.jit(binomial)(values, -1, -0.1) == 0


def test_binomial_logpmf_remains_accurate_for_large_trial_counts() -> None:
    values = np.array([1, 5_000], dtype=np.int32)
    trials = np.array([1_000_000, 10_000], dtype=np.int32)
    probabilities = np.array([0.1, 0.5], dtype=np.float32)
    expected = stats.binom.logpmf(values, trials, probabilities.astype(np.float64))

    result = binomial_logpmf(values, trials, probabilities)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=6e-3)


def test_binomial_logpmf_avoids_large_count_cancellation_near_the_mode() -> None:
    values = np.array([10_000_000, 50_000_000, 500_000_000], dtype=np.int32)
    trials = np.array([20_000_000, 100_000_000, 1_000_000_000], dtype=np.int32)
    expected = stats.binom.logpmf(values, trials, 0.5)

    probability_result = binomial_logpmf(values, trials, jnp.float32(0.5))
    logit_result = binomial_logit_logpmf(values, trials, jnp.float32(0.0))

    np.testing.assert_allclose(probability_result, expected, rtol=1e-6, atol=5e-6)
    np.testing.assert_allclose(logit_result, expected, rtol=1e-6, atol=5e-6)
    assert jnp.all(probability_result <= 0)
    assert jnp.all(logit_result <= 0)


def test_binomial_logpmf_avoids_large_count_cancellation_near_skewed_modes() -> None:
    values = np.array([100, 99_999_900], dtype=np.int32)
    trials = np.int32(100_000_000)
    probabilities = np.array([1e-6, 1 - 1e-6], dtype=np.float32)
    expected_probability = stats.binom.logpmf(values, trials, probabilities.astype(np.float64))

    logits = np.asarray(special.logit(probabilities.astype(np.float64)), dtype=np.float32)
    expected_logit = stats.binom.logpmf(values, trials, special.expit(logits.astype(np.float64)))

    probability_result = binomial_logpmf(values, trials, probabilities)
    logit_result = binomial_logit_logpmf(values, trials, logits)

    np.testing.assert_allclose(probability_result, expected_probability, rtol=1e-6, atol=5e-6)
    np.testing.assert_allclose(logit_result, expected_logit, rtol=1e-6, atol=5e-6)


@pytest.mark.parametrize(("value", "probability"), [(12, np.float32(1e-9)), (10, np.float32(1e-8))])
def test_binomial_logpmf_matches_small_count_identity_at_int32_limit(
    value: int,
    probability: np.float32,
) -> None:
    trials = 2_000_000_000
    log_coefficient = np.sum(np.log(trials - np.arange(value, dtype=np.float64))) - special.gammaln(value + 1)
    expected_probability = (
        log_coefficient + value * np.log(float(probability)) + (trials - value) * np.log1p(-float(probability))
    )

    logits = np.float32(special.logit(float(probability)))
    expected_logit = (
        log_coefficient
        + value * special.log_expit(float(logits))
        + (trials - value) * special.log_expit(-float(logits))
    )

    probability_result = binomial_logpmf(value, trials, probability)
    logit_result = binomial_logit_logpmf(value, trials, logits)

    np.testing.assert_allclose(probability_result, expected_probability, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(logit_result, expected_logit, rtol=1e-6, atol=2e-6)


def test_binomial_logpmf_is_stable_across_rare_side_switch() -> None:
    values = np.array([[4_999_999], [5_000_000], [5_000_001]], dtype=np.int32)
    trials = np.int32(10_000_000)
    probabilities = np.array(
        [
            np.nextafter(np.float32(0.5), np.float32(0)),
            np.float32(0.5),
            np.nextafter(np.float32(0.5), np.float32(1)),
        ]
    )
    expected = stats.binom.logpmf(values, trials, probabilities.astype(np.float64))
    logits = np.array(
        [
            np.nextafter(np.float32(0), np.float32(-1)),
            np.float32(0),
            np.nextafter(np.float32(0), np.float32(1)),
        ]
    )
    expected_logit = stats.binom.logpmf(values, trials, special.expit(logits.astype(np.float64)))

    result = binomial_logpmf(values, trials, probabilities)
    logit_result = binomial_logit_logpmf(values, trials, logits)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(logit_result, expected_logit, rtol=1e-6, atol=1e-6)


def test_binomial_logpmf_preserves_adjacent_integer_counts_before_float_conversion() -> None:
    trials = jnp.int32(16_777_217)
    value = jnp.int32(16_777_216)

    assert jnp.isneginf(binomial_logpmf(value, trials, 1.0))
    assert jnp.isneginf(binomial_logit_logpmf(value, trials, jnp.inf))


def test_binomial_logpmf_preserves_adjacent_counts_across_mixed_dtypes() -> None:
    trials = jnp.int32(16_777_217)
    value = jnp.float32(16_777_216)

    assert jnp.isneginf(binomial_logpmf(value, trials, 1.0))
    assert jnp.isneginf(binomial_logit_logpmf(value, trials, jnp.inf))


def test_binomial_logpmf_rejects_float_count_at_signed_integer_limit() -> None:
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    count_bits = 64 if jax.config.x64_enabled else 32
    trials = dtype(2 ** (count_bits - 1))
    value = jnp.nextafter(trials, dtype(0))

    assert jnp.isnan(binomial_logpmf(value, trials, 0.5))
    assert jnp.isnan(binomial_logit_logpmf(value, trials, 0.0))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_binomial_logpmf_matches_scipy_for_large_central_count_in_float64() -> None:
    value = jnp.int64(500_000)
    trials = jnp.int64(1_000_000)
    probability = jnp.float64(0.5)
    expected = stats.binom.logpmf(500_000, 1_000_000, 0.5)

    result = binomial_logpmf(value, trials, probability)

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_binomial_probability_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 2.0, 5.0])
    trials = jnp.array([5.0, 5.0, 5.0])
    probabilities = jnp.array([0.2, 0.4, 0.8])
    expected_gradient = jnp.diag(values / probabilities - (trials - values) / (1 - probabilities))
    expected_hessian = jnp.diag(-values / jnp.square(probabilities) - (trials - values) / jnp.square(1 - probabilities))

    def evaluate(current_probabilities):
        return binomial_logpmf(values, trials, current_probabilities)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(probabilities)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(probabilities)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(probabilities)

    assert jnp.allclose(forward_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(reverse_gradient, expected_gradient, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(hessian, expected_hessian, rtol=2e-6, atol=2e-6)


def test_binomial_endpoint_gradients_do_not_use_impossible_log_branches() -> None:
    values = jnp.array([0.0, 0.0, 5.0])
    trials = jnp.array([5.0, 0.0, 5.0])
    probabilities = jnp.array([0.0, 0.3, 1.0])
    expected = jnp.diag(jnp.array([-5.0, 0.0, 5.0]))

    def evaluate(current_probabilities):
        return binomial_logpmf(values, trials, current_probabilities)

    assert jnp.array_equal(jax.jit(jax.jacfwd(evaluate))(probabilities), expected)
    assert jnp.array_equal(jax.jit(jax.jacrev(evaluate))(probabilities), expected)


@pytest.mark.parametrize(
    ("values", "trials", "probability", "expected_gradient", "expected_second_derivative"),
    [
        (jnp.array([0, 0]), jnp.array([2, 3]), 0.0, -5.0, -5.0),
        (jnp.array([2, 3]), jnp.array([2, 3]), 1.0, 5.0, -5.0),
        (jnp.array([0, 0]), jnp.array([0, 0]), 0.4, 0.0, 0.0),
    ],
)
def test_binomial_shared_probability_endpoint_derivatives(
    values,
    trials,
    probability: float,
    expected_gradient: float,
    expected_second_derivative: float,
) -> None:
    def evaluate(current_probability):
        return binomial(values, trials, current_probability)

    assert jnp.allclose(jax.grad(evaluate)(probability), expected_gradient)
    assert jnp.allclose(jax.grad(jax.grad(evaluate))(probability), expected_second_derivative)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_binomial_unsupported_values_have_zero_probability_derivative(probability: float) -> None:
    def evaluate(current_probability):
        return binomial_logpmf(6, 5, current_probability)

    assert jnp.isneginf(evaluate(probability))
    assert jax.jacfwd(evaluate)(probability) == 0
    assert jax.jacrev(evaluate)(probability) == 0


@pytest.mark.parametrize(
    ("function", "reference"),
    [(binomial_logcdf, stats.binom.logcdf), (binomial_logsf, stats.binom.logsf)],
)
def test_binomial_tails_match_scipy_across_support_and_broadcasting(function, reference) -> None:
    values = np.array([-np.inf, -0.2, 0, 0.9, 1, 3.7, 10, np.inf, np.nan])[:, None, None]
    trials = np.array([0, 1, 5, 10])[None, :, None]
    probabilities = np.array([0, 0.01, 0.4, 0.9, 1], dtype=np.float32)[None, None, :]
    expected = reference(values, trials, probabilities.astype(np.float64))

    result = function(values, trials, probabilities)
    compiled = jax.jit(function)(values, trials, probabilities)

    assert result.shape == (9, 4, 5)
    np.testing.assert_allclose(result, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(compiled, expected, rtol=5e-6, atol=2e-6)


@pytest.mark.parametrize("function", [binomial_logcdf, binomial_logsf])
@pytest.mark.parametrize("invalid", [-np.inf, -1.0, 1.5, np.inf, np.nan])
def test_binomial_tails_reject_invalid_parameters_before_support(function, invalid) -> None:
    values = jnp.array([-jnp.inf, -1, 0, 1, 10, jnp.inf])

    assert jnp.all(jnp.isnan(jax.jit(function)(values, invalid, 0.3)))
    assert jnp.all(jnp.isnan(jax.jit(function)(values, 5, invalid)))


@pytest.mark.parametrize("trials", [0, 1, 7, 25])
@pytest.mark.parametrize("probability", [0.01, 0.35, 0.9])
def test_binomial_tails_match_independent_probability_sums(trials: int, probability: float) -> None:
    probability = float(jnp.asarray(probability))
    values = np.arange(-1, trials + 1)
    # Summing the finite Binomial PMF checks both tails without reusing incomplete Beta
    masses = [math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k) for k in range(trials + 1)]
    with np.errstate(divide="ignore"):
        expected_cdf = np.log([math.fsum(masses[: k + 1]) for k in values])
        expected_sf = np.log([math.fsum(masses[k + 1 :]) for k in values])

    tolerance = 2e-12 if jax.config.x64_enabled else 5e-6
    np.testing.assert_allclose(
        binomial_logcdf(values, trials, probability), expected_cdf, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(binomial_logsf(values, trials, probability), expected_sf, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("function", "reference", "sign"),
    [(binomial_logcdf, stats.binom.logcdf, -1), (binomial_logsf, stats.binom.logsf, 1)],
)
def test_binomial_tail_probability_derivatives_match_analytic_identities(function, reference, sign) -> None:
    counts = jnp.array([0, 0, 1, 2, 2, 2, 9, 40])
    trials = jnp.array([1, 5, 5, 7, 7, 7, 10, 100])
    probabilities = jnp.array([0.3, 0.6, 0.2, 0.4 - 1e-5, 0.4, 0.4 + 1e-5, 0.8, 0.45])
    p = np.asarray(probabilities, dtype=np.float64)
    k, n = np.asarray(counts), np.asarray(trials)
    expected_log = reference(k, n, p)
    # The derivative of a Binomial tail is +/- n times a Binomial(n-1, p) mass
    expected_gradient = sign * np.exp(np.log(n) + stats.binom.logpmf(k, n - 1, p) - expected_log)
    expected_curvature = expected_gradient * (k / p - (n - k - 1) / (1 - p)) - expected_gradient**2

    forward = jax.jit(jax.vmap(jax.jacfwd(function, argnums=2)))(counts, trials, probabilities)
    reverse = jax.jit(jax.vmap(jax.grad(function, argnums=2)))(counts, trials, probabilities)
    curvature = jax.jit(jax.vmap(jax.grad(jax.grad(function, argnums=2), argnums=2)))(counts, trials, probabilities)
    vectorized = jax.jit(jax.vmap(function))(counts, trials, probabilities)

    gradient_tolerance = 2e-11 if jax.config.x64_enabled else 3e-5
    curvature_tolerance = 2e-10 if jax.config.x64_enabled else 1e-4
    np.testing.assert_allclose(vectorized, expected_log, rtol=gradient_tolerance, atol=gradient_tolerance)
    np.testing.assert_allclose(forward, expected_gradient, rtol=gradient_tolerance, atol=gradient_tolerance)
    np.testing.assert_allclose(reverse, expected_gradient, rtol=gradient_tolerance, atol=gradient_tolerance)
    np.testing.assert_allclose(curvature, expected_curvature, rtol=curvature_tolerance, atol=curvature_tolerance)


@pytest.mark.parametrize(
    ("function", "counts", "probability", "expected_gradient"),
    [
        (binomial_logcdf, [0, 1, 4, 5], 0.0, [-5, 0, 0, 0]),
        (binomial_logsf, [0, 1, 4, -1], 1.0, [0, 0, 5, 0]),
    ],
)
def test_binomial_finite_tail_endpoints_have_correct_probability_gradients(
    function, counts, probability, expected_gradient
) -> None:
    counts = jnp.asarray(counts)

    def evaluate(current_probability):
        return function(counts, 5, current_probability)

    np.testing.assert_array_equal(evaluate(probability), 0)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(evaluate))(probability), expected_gradient, atol=1e-6)
    np.testing.assert_allclose(jax.jit(jax.jacrev(evaluate))(probability), expected_gradient, atol=1e-6)


@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("function", [binomial_logcdf, binomial_logsf])
def test_binomial_tails_outside_support_have_zero_probability_gradients(function, probability) -> None:
    def evaluate(current_probability):
        return function(jnp.array([-jnp.inf, -0.1, 5, jnp.inf]), 5, current_probability)

    np.testing.assert_array_equal(jax.jit(jax.jacfwd(evaluate))(probability), 0)
    np.testing.assert_array_equal(jax.jit(jax.jacrev(evaluate))(probability), 0)
    assert jax.grad(lambda p: function(0, 0, p))(probability) == 0


def test_binomial_tails_are_monotonic_and_complementary() -> None:
    values = jnp.arange(-1, 22)
    logcdf = binomial_logcdf(values, 20, 0.3)
    logsf = binomial_logsf(values, 20, 0.3)

    assert jnp.all(logcdf[1:] >= logcdf[:-1])
    assert jnp.all(logsf[1:] <= logsf[:-1])
    np.testing.assert_allclose(jnp.logaddexp(logcdf, logsf), 0, atol=2e-7)


def test_binomial_logcdf_preserves_small_success_probabilities() -> None:
    probabilities = jnp.array([1e-8, 1e-10, 1e-12])
    p = np.asarray(probabilities, dtype=np.float64)
    # SciPy's survival function retains the tiny mass that its CDF can round away
    expected = np.log1p(-stats.binom.sf(1, 100, p))

    result = jax.jit(binomial_logcdf)(1, 100, probabilities)

    assert jnp.all(result < 0)
    np.testing.assert_allclose(result, jnp.log1p(-betainc(2, 99, probabilities)), rtol=3e-6, atol=0)
    # The native float32 Beta normalizer limits relative accuracy for these tiny masses
    tolerance = 2e-12 if jax.config.x64_enabled else 1e-4
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize(
    ("function", "reference", "count", "probability"),
    [(binomial_logcdf, stats.binom.logcdf, 0, 0.9), (binomial_logsf, stats.binom.logsf, 99, 0.1)],
)
def test_binomial_shape_one_tails_preserve_log_probabilities(function, reference, count, probability) -> None:
    p = jnp.float32(probability)
    expected = reference(count, 100, float(p))

    result = jax.jit(function)(count, 100, p)

    assert jnp.isfinite(result)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize(
    ("function", "reference", "count", "probability"),
    [(binomial_logcdf, stats.binom.logcdf, 10, 0.9), (binomial_logsf, stats.binom.logsf, 89, 0.1)],
)
def test_binomial_tails_recover_underflowed_beta_values_and_gradients(function, reference, count, probability) -> None:
    p = jnp.asarray(probability)
    expected = reference(count, 100, float(p))
    sign = -1 if function is binomial_logcdf else 1
    expected_gradient = sign * np.exp(np.log(100) + stats.binom.logpmf(count, 99, float(p)) - expected)
    expected_curvature = expected_gradient * (count / float(p) - (99 - count) / (1 - float(p))) - expected_gradient**2

    result = jax.jit(function)(count, 100, p)
    forward = jax.jit(jax.jacfwd(function, argnums=2))(count, 100, p)
    reverse = jax.jit(jax.grad(function, argnums=2))(count, 100, p)
    curvature = jax.jit(jax.grad(jax.grad(function, argnums=2), argnums=2))(count, 100, p)

    assert jnp.isfinite(result)
    tolerance = 2e-11 if jax.config.x64_enabled else 5e-5
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)
    np.testing.assert_allclose(forward, expected_gradient, rtol=tolerance, atol=0)
    np.testing.assert_allclose(reverse, expected_gradient, rtol=tolerance, atol=0)
    np.testing.assert_allclose(curvature, expected_curvature, rtol=tolerance, atol=0)


@pytest.mark.parametrize(
    ("function", "reference", "probability"),
    [(binomial_logcdf, stats.binom.logcdf, 0.99), (binomial_logsf, stats.binom.logsf, 0.01)],
)
def test_binomial_tails_stay_monotonic_across_beta_underflow(function, reference, probability) -> None:
    values = jnp.arange(26)
    p = jnp.asarray(probability)
    expected = reference(np.asarray(values), 25, float(p))

    result = jax.jit(function)(values, 25, p)

    tolerance = 2e-12 if jax.config.x64_enabled else 1e-5
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
    if function is binomial_logcdf:
        assert jnp.all(result[1:] >= result[:-1])
    else:
        assert jnp.all(result[1:] <= result[:-1])


@pytest.mark.parametrize("function", [binomial_logcdf, binomial_logsf, binomial_logit_logcdf, binomial_logit_logsf])
def test_binomial_tails_preserve_empty_shapes(function) -> None:
    result = jax.jit(function)(jnp.empty((0, 3)), 10, jnp.array([0.1, 0.3, 0.9]))

    assert result.shape == (0, 3)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
@pytest.mark.parametrize("function", [binomial_logcdf, binomial_logsf, binomial_logit_logcdf, binomial_logit_logsf])
def test_binomial_tail_counts_do_not_control_probability_dtype(function) -> None:
    counts = jnp.array([0, 1, 4], dtype=jnp.int64)
    trials = jnp.asarray(5, dtype=jnp.int64)

    assert jax.jit(function)(counts, trials, jnp.float32(0.3)).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize("function", [binomial_logcdf, binomial_logsf, binomial_logit_logcdf, binomial_logit_logsf])
@pytest.mark.parametrize("argument_index", [0, 1, 2])
def test_binomial_tails_reject_nonreal_arguments(function, argument_index) -> None:
    arguments = [0, 5, 0.3]
    arguments[argument_index] = 1j
    parameter_name = "logits" if function in (binomial_logit_logcdf, binomial_logit_logsf) else "probability"
    name = ("value", "trials", parameter_name)[argument_index]

    with pytest.raises(TypeError, match=rf"argument '{name}' must have a real numeric dtype"):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "reference"),
    [(binomial_logit_logcdf, stats.binom.logcdf), (binomial_logit_logsf, stats.binom.logsf)],
)
def test_binomial_logit_tails_match_scipy_across_support_and_broadcasting(function, reference) -> None:
    values = np.array([-np.inf, -0.2, 0, 0.9, 1, 3.7, 10, np.inf, np.nan])[:, None, None]
    trials = np.array([0, 1, 5, 10])[None, :, None]
    logits = np.array([-np.inf, -5, -1, 0, 1, 5, np.inf, np.nan])[None, None, :]
    expected = reference(values, trials, special.expit(logits))
    # Our parameter validation comes before support checks, including at the upper boundary
    expected = np.where(np.isnan(logits), np.nan, expected)

    result = function(values, trials, logits)
    compiled = jax.jit(function)(values, trials, logits)

    assert result.shape == (9, 4, 8)
    np.testing.assert_allclose(result, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(compiled, expected, rtol=5e-6, atol=2e-6)


@pytest.mark.parametrize("function", [binomial_logit_logcdf, binomial_logit_logsf])
def test_binomial_logit_tails_reject_invalid_parameters_before_support(function) -> None:
    values = jnp.array([-jnp.inf, -1, 0, 1, 10, jnp.inf])[:, None]
    trials = jnp.array([-jnp.inf, -1, 1.5, jnp.inf, jnp.nan])

    assert jnp.all(jnp.isnan(jax.jit(function)(values, trials, 0.3)))
    assert jnp.all(jnp.isnan(jax.jit(function)(values, 5, jnp.nan)))


@pytest.mark.parametrize("function", [binomial_logit_logcdf, binomial_logit_logsf])
def test_binomial_logit_tails_match_independent_log_mass_sums_and_derivatives(function) -> None:
    trials = 25
    counts = np.array([0, 1, 5, 12, 24])
    logits = np.array([-3.0, 0, 3, 20, 100, 1000])
    if function is binomial_logit_logsf:
        logits = -logits
    thresholds, log_odds = np.broadcast_arrays(counts[:, None], logits)
    expected_log = np.empty_like(log_odds)
    expected_gradient = np.empty_like(log_odds)
    expected_curvature = np.empty_like(log_odds)

    for index in np.ndindex(log_odds.shape):
        k, eta = thresholds[index], log_odds[index]
        outcomes = np.arange(k + 1) if function is binomial_logit_logcdf else np.arange(k + 1, trials + 1)
        # A finite log-space PMF sum stays independent of incomplete Beta and its fallback
        log_masses = np.array([math.log(math.comb(trials, int(x))) for x in outcomes])
        log_masses -= outcomes * np.logaddexp(0, -eta) + (trials - outcomes) * np.logaddexp(0, eta)
        expected_log[index] = special.logsumexp(log_masses)
        weights = np.exp(log_masses - expected_log[index])
        mean = np.sum(weights * outcomes)
        variance = np.sum(weights * (outcomes - mean) ** 2)
        # Logit derivatives follow from the conditional mean and variance within the tail
        p, q = special.expit(eta), special.expit(-eta)
        expected_gradient[index] = mean - trials * p
        expected_curvature[index] = variance - trials * p * q

    scalar_values = jnp.asarray(thresholds.ravel())
    scalar_logits = jnp.asarray(log_odds.ravel())
    result = jax.jit(function)(thresholds, trials, log_odds)

    def evaluate(k, eta):
        return function(k, trials, eta)

    forward = jax.jit(jax.vmap(jax.jacfwd(evaluate, argnums=1)))(scalar_values, scalar_logits)
    reverse = jax.jit(jax.vmap(jax.grad(evaluate, argnums=1)))(scalar_values, scalar_logits)
    curvature = jax.jit(jax.vmap(jax.grad(jax.grad(evaluate, argnums=1), argnums=1)))(scalar_values, scalar_logits)

    tolerance = 5e-11 if jax.config.x64_enabled else 5e-5
    assert jnp.all(jnp.isfinite(result))
    np.testing.assert_allclose(result, expected_log, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(forward.reshape(log_odds.shape), expected_gradient, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(reverse.reshape(log_odds.shape), expected_gradient, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(curvature.reshape(log_odds.shape), expected_curvature, rtol=tolerance, atol=tolerance)


def test_binomial_logit_tails_preserve_near_certain_probabilities() -> None:
    logits = jnp.array([20.0, 25, 30])
    rare_probability = special.expit(-np.asarray(logits, dtype=np.float64))
    # Compare the tiny departure from one, which a rounded sigmoid input would erase
    expected = np.log1p(-stats.binom.sf(1, 10, rare_probability))

    logcdf = jax.jit(binomial_logit_logcdf)(1, 10, -logits)
    logsf = jax.jit(binomial_logit_logsf)(8, 10, logits)

    assert jnp.all(logcdf < 0)
    assert jnp.all(logsf < 0)
    tolerance = 2e-12 if jax.config.x64_enabled else 1e-5
    np.testing.assert_allclose(logcdf, expected, rtol=tolerance, atol=0)
    np.testing.assert_allclose(logsf, expected, rtol=tolerance, atol=0)


def test_binomial_logit_tails_are_monotonic_complementary_and_symmetric() -> None:
    values = jnp.arange(-1, 27)[:, None]
    logits = jnp.array([-1000.0, -25, -3, 0, 3, 25, 1000])
    logcdf = jax.jit(binomial_logit_logcdf)(values, 25, logits)
    logsf = jax.jit(binomial_logit_logsf)(values, 25, logits)

    assert jnp.all(logcdf[1:] >= logcdf[:-1])
    assert jnp.all(logsf[1:] <= logsf[:-1])
    np.testing.assert_allclose(jnp.logaddexp(logcdf, logsf), 0, atol=2e-7)
    # Native float32 incomplete Beta has rounding error even at its symmetric midpoint
    tolerance = 2e-12 if jax.config.x64_enabled else 3e-5
    np.testing.assert_allclose(logcdf, binomial_logit_logsf(24 - values, 25, -logits), rtol=tolerance, atol=0)


@pytest.mark.parametrize("function", [binomial_logit_logcdf, binomial_logit_logsf])
def test_binomial_logit_tails_outside_support_have_zero_gradients(function) -> None:
    logits = jnp.array([-jnp.inf, -1000, 0, 1000, jnp.inf])

    def evaluate(eta):
        return function(jnp.array([-jnp.inf, -0.1, 5, jnp.inf]), 5, eta)

    np.testing.assert_array_equal(jax.jit(jax.vmap(jax.jacfwd(evaluate)))(logits), 0)
    np.testing.assert_array_equal(jax.jit(jax.vmap(jax.jacrev(evaluate)))(logits), 0)
    np.testing.assert_array_equal(jax.jit(jax.vmap(jax.grad(lambda eta: function(0, 0, eta))))(logits), 0)


def test_binomial_logit_logpmf_matches_scipy() -> None:
    values = np.array([[0], [1], [3], [5]], dtype=np.int32)
    trials = np.array([1, 3, 5], dtype=np.int32)
    logits = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    expected = stats.binom.logpmf(values, trials, special.expit(logits.astype(np.float64)))

    result = binomial_logit_logpmf(values, trials, logits)

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_binomial_logit_logpmf_handles_support_and_infinite_limits() -> None:
    values = jnp.array([0, 5, 0, 5, -1, 6])
    logits = jnp.array([-jnp.inf, -jnp.inf, jnp.inf, jnp.inf, 0.0, 0.0])

    result = binomial_logit_logpmf(values, 5, logits)

    assert result[0] == 0
    assert jnp.isneginf(result[1])
    assert jnp.isneginf(result[2])
    assert result[3] == 0
    assert jnp.isneginf(result[4])
    assert jnp.isneginf(result[5])


def test_binomial_logit_logpmf_matches_scipy_log_expit_in_finite_tails() -> None:
    values = np.array([[0], [2], [5]], dtype=np.float64)
    trials = 5.0
    logits = np.array([-1000.0, -20.0, 20.0, 1000.0], dtype=np.float64)
    log_coefficient = special.gammaln(trials + 1) - special.gammaln(values + 1) - special.gammaln(trials - values + 1)
    expected = log_coefficient + values * special.log_expit(logits) + (trials - values) * special.log_expit(-logits)

    result = binomial_logit_logpmf(values.astype(np.float32), int(trials), logits.astype(np.float32))

    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-5)


def test_binomial_logit_with_one_trial_matches_bernoulli_logit() -> None:
    values = jnp.array([0, 1])
    logits = jnp.array([-20.0, 20.0])

    assert jnp.allclose(binomial_logit_logpmf(values, 1, logits), bernoulli_logit_logpmf(values, logits))


def test_binomial_logit_logpmf_rejects_invalid_parameters() -> None:
    invalid_trials = binomial_logit_logpmf(0, jnp.array([-1.0, 2.5, jnp.inf, jnp.nan]), 0.5)
    invalid_logits = binomial_logit_logpmf(0, 5, jnp.nan)

    assert jnp.all(jnp.isnan(invalid_trials))
    assert jnp.isnan(invalid_logits)


def test_binomial_logit_derivatives_match_closed_form() -> None:
    values = jnp.array([0.0, 2.0, 5.0])
    trials = jnp.array([5.0, 5.0, 5.0])
    logits = jnp.array([-3.0, 0.5, 2.0])
    probabilities = jax.nn.sigmoid(logits)
    expected_gradient = jnp.diag(values - trials * probabilities)
    expected_hessian = jnp.diag(-trials * probabilities * (1 - probabilities))

    def evaluate(current_logits):
        return binomial_logit_logpmf(values, trials, current_logits)

    forward_gradient = jax.jit(jax.jacfwd(evaluate))(logits)
    reverse_gradient = jax.jit(jax.jacrev(evaluate))(logits)
    hessian = jax.jit(jax.jacfwd(jax.jacrev(lambda current: jnp.sum(evaluate(current)))))(logits)

    assert jnp.allclose(forward_gradient, expected_gradient)
    assert jnp.allclose(reverse_gradient, expected_gradient)
    assert jnp.allclose(hessian, expected_hessian)


def test_binomial_logit_tail_gradients_match_limiting_values() -> None:
    values = jnp.array([0.0, 5.0, 0.0, 5.0])
    logits = jnp.array([-jnp.inf, -jnp.inf, jnp.inf, jnp.inf])
    expected = jnp.diag(jnp.array([0.0, 5.0, -5.0, 0.0]))

    def evaluate(current_logits):
        return binomial_logit_logpmf(values, 5, current_logits)

    assert jnp.array_equal(jax.jacfwd(evaluate)(logits), expected)
    assert jnp.array_equal(jax.jacrev(evaluate)(logits), expected)


def test_binomial_logit_sums_log_masses() -> None:
    values = jnp.array([0, 2, 5])
    trials = jnp.array([5, 5, 5])
    logits = jnp.array([-2.0, 0.5, 3.0])
    expected = np.sum(stats.binom.logpmf(np.asarray(values), np.asarray(trials), special.expit(np.asarray(logits))))

    result = binomial_logit(values, trials, logits)

    assert result.shape == ()
    assert jnp.allclose(result, expected)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="JAX 64-bit mode is disabled")
def test_binomial_counts_do_not_control_parameter_dtype() -> None:
    values = jnp.array([0, 1], dtype=jnp.int64)
    trials = jnp.array([5, 5], dtype=jnp.int64)

    assert binomial_logpmf(values, trials, jnp.float32(0.4)).dtype == jnp.dtype(jnp.float32)
    assert binomial_logit_logpmf(values, trials, jnp.float32(0.2)).dtype == jnp.dtype(jnp.float32)


@pytest.mark.parametrize(
    ("function", "arguments", "argument_name"),
    [
        (binomial_logpmf, (0.0 + 0.0j, 5, 0.5), "value"),
        (binomial_logpmf, (0, 5.0 + 0.0j, 0.5), "trials"),
        (binomial_logpmf, (0, 5, 0.5 + 0.0j), "probability"),
        (binomial_logit_logpmf, (0, 5, 0.5 + 0.0j), "logits"),
    ],
)
def test_binomial_functions_reject_complex_arguments(function, arguments, argument_name: str) -> None:
    with pytest.raises(TypeError, match=rf"argument '{argument_name}' must have a real numeric dtype, got complex"):
        function(*arguments)


def test_binomial_rng_matches_jax_and_uses_integer_output() -> None:
    key = jax.random.key(42)
    trials = jnp.array([3, 8])
    probabilities = jnp.array([0.2, 0.8], dtype=jnp.float32)
    expected = jax.random.binomial(
        key,
        trials.astype(probabilities.dtype),
        probabilities,
        shape=(4, 2),
        dtype=probabilities.dtype,
    ).astype(jnp.int32)

    result = binomial_rng(key, trials, probabilities, sample_shape=(4,))

    assert result.shape == (4, 2)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert jnp.array_equal(result, expected)


def test_binomial_rng_handles_deterministic_parameters() -> None:
    trials = jnp.array([0, 5, 5])
    probabilities = jnp.array([0.7, 0.0, 1.0])

    result = binomial_rng(jax.random.key(0), trials, probabilities, sample_shape=(32,))

    assert jnp.all(result[:, 0] == 0)
    assert jnp.all(result[:, 1] == 0)
    assert jnp.all(result[:, 2] == 5)


def test_binomial_rng_preserves_the_largest_exact_float32_trial_count() -> None:
    trials = 16_777_216

    probability_result = binomial_rng(jax.random.key(0), trials, jnp.float32(1.0))
    logit_result = binomial_logit_rng(jax.random.key(0), trials, jnp.float32(jnp.inf))

    assert probability_result == trials
    assert logit_result == trials


def test_binomial_rng_matches_expected_moments() -> None:
    trials = jnp.array([5.0, 20.0])
    probabilities = jnp.array([0.2, 0.7])

    samples = binomial_rng(jax.random.key(7), trials, probabilities, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples, axis=0), trials * probabilities, rtol=0, atol=0.03)
    assert jnp.allclose(jnp.var(samples, axis=0), trials * probabilities * (1 - probabilities), rtol=0.03, atol=0.03)


def test_binomial_logit_rng_uses_stable_rare_event_probabilities() -> None:
    key = jax.random.key(5)
    trials = jnp.array([20.0, 20.0, 20.0])
    logits = jnp.array([-17.0, 0.0, 17.0])
    rare_probability = jnp.exp(jax.nn.log_sigmoid(-jnp.abs(logits)))
    rare_outcomes = jax.random.binomial(
        key,
        trials,
        rare_probability,
        shape=(8, 3),
        dtype=logits.dtype,
    )
    expected = jnp.where(logits > 0, trials - rare_outcomes, rare_outcomes).astype(jnp.int32)

    result = binomial_logit_rng(key, trials, logits, sample_shape=(8,))

    assert jnp.array_equal(result, expected)


def test_binomial_logit_rng_preserves_rare_failures_beyond_float32_sigmoid() -> None:
    trials = 1_000_000
    logits = jnp.float32(17.0)
    expected_failure_probability = special.expit(-17.0)

    samples = binomial_logit_rng(jax.random.key(13), trials, logits, sample_shape=(50_000,))
    failures = trials - samples

    assert jax.nn.sigmoid(logits) == 1
    assert jnp.any(failures > 0)
    assert jnp.allclose(jnp.mean(failures), trials * expected_failure_probability, rtol=0, atol=0.004)


def test_binomial_logit_rng_handles_infinite_limits() -> None:
    trials = jnp.array([0, 5, 5])
    logits = jnp.array([jnp.inf, -jnp.inf, jnp.inf])

    result = binomial_logit_rng(jax.random.key(0), trials, logits, sample_shape=(32,))

    assert jnp.all(result[:, 0] == 0)
    assert jnp.all(result[:, 1] == 0)
    assert jnp.all(result[:, 2] == 5)


def test_binomial_logit_rng_matches_expected_means() -> None:
    trials = jnp.array([5.0, 20.0, 50.0])
    logits = jnp.array([-2.0, 0.0, 2.0])

    samples = binomial_logit_rng(jax.random.key(11), trials, logits, sample_shape=(50_000,))

    assert jnp.allclose(jnp.mean(samples, axis=0), trials * jax.nn.sigmoid(logits), rtol=0, atol=0.04)


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (binomial_rng, (jnp.array([5, 10]), jnp.array([0.2, 0.8]))),
        (binomial_logit_rng, (jnp.array([5, 10]), jnp.array([-2.0, 2.0]))),
    ],
)
def test_binomial_rngs_can_be_jitted(function, parameters) -> None:
    key = jax.random.key(21)

    result = jax.jit(function)(key, *parameters)
    expected = function(key, *parameters)

    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (binomial_rng, (5, 0.4)),
        (binomial_logit_rng, (5, -0.5)),
    ],
)
def test_binomial_rngs_can_be_vectorized_over_keys(function, parameters) -> None:
    keys = jax.random.split(jax.random.key(22), 4)

    result = jax.vmap(lambda key: function(key, *parameters))(keys)
    expected = jnp.stack([function(key, *parameters) for key in keys])

    assert jnp.array_equal(result, expected)
