"""Binomial distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from mmmjax.distributions._discrete import _binomial_interior_log_mass
from mmmjax.distributions._utils import (
    _as_real_array,
    _promote_inexact,
    _random_shape,
)


def binomial_logpmf(
    value: ArrayLike,
    trials: ArrayLike,
    probability: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Binomial log probability mass elementwise.

    For successes :math:`k \in \{0, \ldots, n\}`, trials :math:`n`, and
    success probability :math:`p \in [0, 1]`, the log probability mass is

    .. math::

        \log p(k \mid n, p)
        = \log {n \choose k}
          + k\log(p)
          + (n-k)\log(1-p).

    Parameters
    ----------
    value
        Numbers of successes at which to evaluate the probability mass.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the integer support produce ``-inf``. An
        invalid trial count or probability produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    trials_array = _as_real_array("trials", trials)
    (probability_array,) = _promote_inexact(("probability", probability))

    successes, failures, trials_float, supported, valid_trials = _prepare_binomial_counts(
        value_array,
        trials_array,
        dtype=probability_array.dtype,
    )

    valid_probability = jnp.isfinite(probability_array) & (probability_array >= 0) & (probability_array <= 1)
    safe_probability = jnp.where(valid_probability, probability_array, jnp.full_like(probability_array, 0.5))

    success_probability = jnp.where(successes > 0, safe_probability, jnp.ones_like(safe_probability))
    failure_probability = jnp.where(failures > 0, safe_probability, jnp.zeros_like(safe_probability))
    boundary_log_mass = successes * jnp.log(success_probability) + failures * jnp.log1p(-failure_probability)

    use_interior_mass = (successes > 0) & (failures > 0) & (safe_probability > 0) & (safe_probability < 1)
    interior_successes = jnp.where(use_interior_mass, successes, jnp.ones_like(successes))
    interior_failures = jnp.where(use_interior_mass, failures, jnp.ones_like(failures))
    interior_trials = jnp.where(use_interior_mass, trials_float, jnp.full_like(trials_float, 2))
    interior_probability = jnp.where(use_interior_mass, safe_probability, jnp.full_like(safe_probability, 0.5))
    interior_log_mass = _binomial_interior_log_mass(
        interior_successes,
        interior_failures,
        interior_trials,
        interior_probability,
        1 - interior_probability,
        jnp.log(interior_probability),
        jnp.log1p(-interior_probability),
    )
    supported_log_mass = jnp.where(use_interior_mass, interior_log_mass, boundary_log_mass)

    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_trials & valid_probability, log_mass, jnp.nan)


def binomial(
    value: ArrayLike,
    trials: ArrayLike,
    probability: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Binomial log probability masses.

    Parameters
    ----------
    value
        Numbers of successes at which to evaluate the probability mass.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(binomial_logpmf(value, trials, probability))


def binomial_rng(
    key: jax.Array,
    trials: ArrayLike,
    probability: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw Binomial outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability
        Success probabilities in the closed interval from zero to one. The
        caller must provide valid probabilities because invalid values do not
        have a defined sampling result.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + broadcast_shape``.
    """
    trials_array = _as_real_array("trials", trials)
    (probability_array,) = _promote_inexact(("probability", probability))
    output_shape = _random_shape(sample_shape, trials_array, probability_array)

    samples = jax.random.binomial(
        key,
        jnp.asarray(trials_array, dtype=probability_array.dtype),
        probability_array,
        shape=output_shape,
        dtype=probability_array.dtype,
    )
    return samples.astype(jnp.int32)


def binomial_logit_logpmf(
    value: ArrayLike,
    trials: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    r"""Evaluate the logit-parameterized Binomial log probability mass.

    For successes :math:`k \in \{0, \ldots, n\}`, trials :math:`n`, and log
    odds :math:`\eta`, the log probability mass is

    .. math::

        \log p(k \mid n, \eta)
        = \log {n \choose k}
          + k\log\!\left(\operatorname{logit}^{-1}(\eta)\right)
          + (n-k)\log\!\left(\operatorname{logit}^{-1}(-\eta)\right).

    Parameters
    ----------
    value
        Numbers of successes at which to evaluate the probability mass.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits
        Log odds of success.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the integer support produce ``-inf``. An
        invalid trial count or a ``nan`` logit produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    trials_array = _as_real_array("trials", trials)
    (logits_array,) = _promote_inexact(("logits", logits))

    successes, failures, trials_float, supported, valid_trials = _prepare_binomial_counts(
        value_array,
        trials_array,
        dtype=logits_array.dtype,
    )

    valid_logits = ~jnp.isnan(logits_array)
    safe_logits = jnp.where(valid_logits, logits_array, jnp.zeros_like(logits_array))
    success_logits = jnp.where(successes > 0, safe_logits, jnp.zeros_like(safe_logits))
    failure_logits = jnp.where(failures > 0, -safe_logits, jnp.zeros_like(safe_logits))
    boundary_log_mass = successes * jax.nn.log_sigmoid(success_logits) + failures * jax.nn.log_sigmoid(failure_logits)

    use_interior_mass = (successes > 0) & (failures > 0) & jnp.isfinite(safe_logits)
    interior_successes = jnp.where(use_interior_mass, successes, jnp.ones_like(successes))
    interior_failures = jnp.where(use_interior_mass, failures, jnp.ones_like(failures))
    interior_trials = jnp.where(use_interior_mass, trials_float, jnp.full_like(trials_float, 2))
    interior_logits = jnp.where(use_interior_mass, safe_logits, jnp.zeros_like(safe_logits))
    log_success_probability = jax.nn.log_sigmoid(interior_logits)
    log_failure_probability = jax.nn.log_sigmoid(-interior_logits)
    interior_log_mass = _binomial_interior_log_mass(
        interior_successes,
        interior_failures,
        interior_trials,
        jnp.exp(log_success_probability),
        jnp.exp(log_failure_probability),
        log_success_probability,
        log_failure_probability,
    )
    supported_log_mass = jnp.where(use_interior_mass, interior_log_mass, boundary_log_mass)

    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_trials & valid_logits, log_mass, jnp.nan)


def binomial_logit(
    value: ArrayLike,
    trials: ArrayLike,
    logits: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of logit-parameterized Binomial log masses.

    Parameters
    ----------
    value
        Numbers of successes at which to evaluate the probability mass.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits
        Log odds of success.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(binomial_logit_logpmf(value, trials, logits))


def binomial_logit_rng(
    key: jax.Array,
    trials: ArrayLike,
    logits: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw logit-parameterized Binomial outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    trials
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits
        Log odds of success. The caller must not provide ``nan`` because it
        does not have a defined sampling result.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + broadcast_shape``.
    """
    trials_array = _as_real_array("trials", trials)
    (logits_array,) = _promote_inexact(("logits", logits))
    output_shape = _random_shape(sample_shape, trials_array, logits_array)

    trials_float = jnp.asarray(trials_array, dtype=logits_array.dtype)
    rare_probability = jnp.exp(jax.nn.log_sigmoid(-jnp.abs(logits_array)))
    rare_outcomes = jax.random.binomial(
        key,
        trials_float,
        rare_probability,
        shape=output_shape,
        dtype=logits_array.dtype,
    )
    outcomes = jnp.where(logits_array > 0, trials_float - rare_outcomes, rare_outcomes)
    return outcomes.astype(jnp.int32)


def _prepare_binomial_counts(
    value: jax.Array,
    trials: jax.Array,
    *,
    dtype: DTypeLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    count_dtype = jax.dtypes.canonicalize_dtype(jnp.int64)
    count_bits = jax.dtypes.itemsize_bits(count_dtype)
    count_limit = 2 ** (count_bits - 1)
    integer_value = jnp.asarray(value, dtype=count_dtype)
    integer_trials = jnp.asarray(trials, dtype=count_dtype)

    if jnp.issubdtype(value.dtype, jnp.floating):
        value_range_dtype = jnp.promote_types(value.dtype, jnp.float32)
        value_in_range = jnp.asarray(value, dtype=value_range_dtype) < jnp.asarray(
            count_limit,
            dtype=value_range_dtype,
        )
    else:
        value_in_range = jnp.ones_like(value, dtype=jnp.bool_)
    if jnp.issubdtype(trials.dtype, jnp.floating):
        trials_range_dtype = jnp.promote_types(trials.dtype, jnp.float32)
        trials_in_range = jnp.asarray(trials, dtype=trials_range_dtype) < jnp.asarray(
            count_limit,
            dtype=trials_range_dtype,
        )
    else:
        trials_in_range = jnp.ones_like(trials, dtype=jnp.bool_)

    exact_value = value_in_range & jnp.isfinite(value) & (value == jnp.asarray(integer_value, dtype=value.dtype))
    exact_trials = trials_in_range & jnp.isfinite(trials) & (trials == jnp.asarray(integer_trials, dtype=trials.dtype))
    valid_trials = exact_trials & (integer_trials >= 0)
    supported = exact_value & (integer_value >= 0) & valid_trials & (integer_value <= integer_trials)

    # Count arithmetic stays integer so mixed input dtypes cannot erase adjacent large counts
    safe_value = jnp.where(supported, integer_value, jnp.zeros_like(integer_value))
    safe_trials = jnp.where(supported, integer_trials, jnp.zeros_like(integer_trials))
    safe_failures = safe_trials - safe_value

    successes = jnp.asarray(safe_value, dtype=dtype)
    failures = jnp.asarray(safe_failures, dtype=dtype)
    trials_float = jnp.asarray(safe_trials, dtype=dtype)

    return successes, failures, trials_float, supported, valid_trials
