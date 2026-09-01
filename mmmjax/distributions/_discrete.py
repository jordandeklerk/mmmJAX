"""Shared utilities for discrete distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from mmmjax.distributions._utils import (
    _gamma_shape_normalizer,
    _weighted_log_ratio_deviance,
)


def _prepare_nonnegative_count(
    value: jax.Array,
    *,
    dtype: DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    if value.dtype == jnp.dtype(jnp.bool_) or jnp.issubdtype(value.dtype, jnp.integer):
        supported = value >= 0
    else:
        supported = jnp.isfinite(value) & (value >= 0) & (value == jnp.floor(value))

    # Support is checked before conversion so parameter dtype cannot round fractional counts onto the support
    safe_value = jnp.where(supported, value, jnp.zeros_like(value))
    count = jnp.asarray(safe_value, dtype=dtype)
    return count, supported


def _binomial_interior_log_mass(
    successes: jax.Array,
    failures: jax.Array,
    trials: jax.Array,
    success_probability: jax.Array,
    failure_probability: jax.Array,
    log_success_probability: jax.Array,
    log_failure_probability: jax.Array,
) -> jax.Array:
    # Loader's deviance decomposition avoids large-count cancellation in the direct log PMF
    log_trials = jnp.log(trials)
    success_is_rare = success_probability <= failure_probability
    rare_probability = jnp.where(success_is_rare, success_probability, failure_probability)
    # Starting from the small expected count preserves its deviation in float32
    rare_expected_count = trials * rare_probability
    rare_count = jnp.where(success_is_rare, successes, failures)
    rare_deviation = rare_expected_count - rare_count
    success_deviation = jnp.where(success_is_rare, rare_deviation, -rare_deviation)
    expected_successes = jnp.where(success_is_rare, rare_expected_count, trials - rare_expected_count)
    expected_failures = jnp.where(success_is_rare, trials - rare_expected_count, rare_expected_count)

    success_contribution = _binomial_deviance_contribution(
        successes,
        expected_successes,
        success_deviation,
        log_trials + log_success_probability - jnp.log(successes),
        use_expected_ratio=success_is_rare,
    )
    failure_contribution = _binomial_deviance_contribution(
        failures,
        expected_failures,
        -success_deviation,
        log_trials + log_failure_probability - jnp.log(failures),
        use_expected_ratio=~success_is_rare,
    )

    # The deviance form avoids subtracting terms that each grow with the trial count
    log_normalizer = (
        _gamma_shape_normalizer(successes)
        + _gamma_shape_normalizer(failures)
        - _gamma_shape_normalizer(trials)
        + log_trials
        - jnp.log(successes)
        - jnp.log(failures)
    )
    return log_normalizer + success_contribution + failure_contribution


def _binomial_deviance_contribution(
    count: jax.Array,
    expected_count: jax.Array,
    linear_deviation: jax.Array,
    raw_log_ratio: jax.Array,
    *,
    use_expected_ratio: jax.Array,
) -> jax.Array:
    ratio_deviation = linear_deviation / count
    # The rare ratio is direct while log1p retains the common side's small difference
    safe_ratio_deviation = jnp.where(use_expected_ratio, jnp.zeros_like(ratio_deviation), ratio_deviation)
    deviation_log_ratio = jnp.log1p(safe_ratio_deviation)

    expected_ratio = jnp.where(use_expected_ratio, expected_count / count, jnp.ones_like(count))
    valid_expected_ratio = jnp.isfinite(expected_ratio) & (expected_ratio > 0)
    safe_expected_ratio = jnp.where(valid_expected_ratio, expected_ratio, jnp.ones_like(expected_ratio))
    expected_log_ratio = jnp.log(safe_expected_ratio)

    stable_log_ratio = jnp.where(use_expected_ratio, expected_log_ratio, deviation_log_ratio)
    valid_stable_log_ratio = jnp.where(use_expected_ratio, valid_expected_ratio, jnp.isfinite(deviation_log_ratio))
    log_ratio = jnp.where(valid_stable_log_ratio, stable_log_ratio, raw_log_ratio)
    stable_contribution = _weighted_log_ratio_deviance(
        count,
        log_ratio,
        linear_deviation,
    )

    # Preserve the exact derivative when representable and use the stable path through overflow
    direct_contribution = count * raw_log_ratio - linear_deviation
    use_direct_derivative = jnp.isfinite(direct_contribution) & jnp.isfinite(stable_contribution)
    differentiable_contribution = jnp.where(
        use_direct_derivative,
        direct_contribution,
        stable_contribution,
    )
    derivative_correction = jnp.where(
        use_direct_derivative,
        stable_contribution - direct_contribution,
        jnp.zeros_like(stable_contribution),
    )
    return differentiable_contribution + jax.lax.stop_gradient(derivative_correction)
