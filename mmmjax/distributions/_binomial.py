"""Binomial distribution functions."""

from typing import cast

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc
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
    value : array_like
        Numbers of successes at which to evaluate the probability mass.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability : array_like
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the integer support produce ``-inf``. An
        invalid trial count or probability produces ``nan``.

    Examples
    --------
    Evaluate one log probability mass per value. Each count records successes
    out of ten trials.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logpmf
           ...: counts = jnp.array([1, 3, 5])
           ...: binomial_logpmf(counts, trials=10, probability=0.3)
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
    value : array_like
        Numbers of successes at which to evaluate the probability mass.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability : array_like
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.

    Examples
    --------
    Sum the log probability masses of independent observations into a single log
    likelihood. Each count records successes out of ten trials.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial
           ...: counts = jnp.array([1, 3, 5])
           ...: binomial(counts, trials=10, probability=0.3)
    """
    return jnp.sum(binomial_logpmf(value, trials, probability))


def binomial_logcdf(value: ArrayLike, trials: ArrayLike, probability: ArrayLike) -> jax.Array:
    r"""Evaluate the Binomial log cumulative distribution function elementwise.

    For a threshold :math:`0 \leq x < n`, a nonnegative integer number of
    trials :math:`n`, and success probability :math:`p \in [0, 1]`,

    .. math::

        \log P(X \leq x)
        = \log I_{1-p}(n - \lfloor x \rfloor, \lfloor x \rfloor + 1),

    where :math:`I` is the regularized incomplete Beta function.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the cumulative probability.
        Fractional thresholds are rounded down to the nearest integer.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability : array_like
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Negative thresholds produce ``-inf`` and thresholds at
        or above the trial count produce zero. Invalid parameters or
        ``nan`` thresholds produce ``nan``.

    Notes
    -----
    Large trial counts can lose accuracy in float32. Enable JAX 64-bit mode
    when additional precision is needed.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logcdf
           ...: thresholds = jnp.array([2, 4, 6])
           ...: log_prob = binomial_logcdf(
           ...:     thresholds,
           ...:     trials=10,
           ...:     probability=0.3,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _binomial_log_probability(value, trials, probability, upper_tail=False)


def binomial_logsf(value: ArrayLike, trials: ArrayLike, probability: ArrayLike) -> jax.Array:
    r"""Evaluate the Binomial log survival function elementwise.

    For a threshold :math:`0 \leq x < n`, a nonnegative integer number of
    trials :math:`n`, and success probability :math:`p \in [0, 1]`,

    .. math::

        \log P(X > x)
        = \log I_p(\lfloor x \rfloor + 1, n - \lfloor x \rfloor),

    where :math:`I` is the regularized incomplete Beta function. This avoids
    obtaining a small survival probability by subtracting the CDF from one.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the survival probability.
        Fractional thresholds are rounded down to the nearest integer.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability : array_like
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the
        arguments. Negative thresholds produce zero and thresholds at or
        above the trial count produce ``-inf``. Invalid parameters or
        ``nan`` thresholds produce ``nan``.

    Notes
    -----
    Large trial counts can lose accuracy in float32. Enable JAX 64-bit mode
    when additional precision is needed.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logsf
           ...: thresholds = jnp.array([2, 4, 6])
           ...: log_prob = binomial_logsf(
           ...:     thresholds,
           ...:     trials=10,
           ...:     probability=0.3,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _binomial_log_probability(value, trials, probability, upper_tail=True)


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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    probability : array_like
        Success probabilities in the closed interval from zero to one. The
        caller must provide valid probabilities because invalid values do not
        have a defined sampling result.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + broadcast_shape``.

    Examples
    --------
    Draw five outcomes using a key to make the draw reproducible.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import binomial_rng
           ...: key = random.key(0)
           ...: binomial_rng(key, trials=10, probability=0.3, sample_shape=(5,))
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
    value : array_like
        Numbers of successes at which to evaluate the probability mass.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits : array_like
        Log odds of success.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the integer support produce ``-inf``. An
        invalid trial count or a ``nan`` logit produces ``nan``.

    Examples
    --------
    Evaluate one log probability mass per value. Specify ten trials and the log
    odds of success.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logit_logpmf
           ...: counts = jnp.array([1, 3, 5])
           ...: binomial_logit_logpmf(counts, trials=10, logits=-0.8)
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
    value : array_like
        Numbers of successes at which to evaluate the probability mass.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits : array_like
        Log odds of success.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.

    Examples
    --------
    Sum the log probability masses of independent observations into a single log
    likelihood. Specify ten trials and the log odds of success.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logit
           ...: counts = jnp.array([1, 3, 5])
           ...: binomial_logit(counts, trials=10, logits=-0.8)
    """
    return jnp.sum(binomial_logit_logpmf(value, trials, logits))


def binomial_logit_logcdf(value: ArrayLike, trials: ArrayLike, logits: ArrayLike) -> jax.Array:
    r"""Evaluate the logit-parameterized Binomial log CDF elementwise.

    For log odds :math:`\eta`, let :math:`p = \operatorname{sigmoid}(\eta)`.
    With :math:`0 \leq x < n`, the log cumulative probability is

    .. math::

        \log P(X \leq x)
        = \log I_{1-p}(n - \lfloor x \rfloor, \lfloor x \rfloor + 1),

    where :math:`I` is the regularized incomplete Beta function. Log odds
    are retained in tail calculations to avoid rounding rare probabilities
    away when converting to the probability scale.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the cumulative probability.
        Fractional thresholds are rounded down to the nearest integer.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits : array_like
        Log odds of success. Negative and positive infinity correspond to
        success probabilities of zero and one, respectively.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Negative thresholds produce ``-inf`` and thresholds at
        or above the trial count produce zero. Invalid parameters or
        ``nan`` thresholds produce ``nan``.

    Notes
    -----
    Large trial counts can lose accuracy in float32. Enable JAX 64-bit mode
    when additional precision is needed.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logit_logcdf
           ...: thresholds = jnp.array([2, 4, 6])
           ...: log_prob = binomial_logit_logcdf(
           ...:     thresholds,
           ...:     trials=10,
           ...:     logits=-0.8,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _binomial_log_probability(value, trials, logits, upper_tail=False, logit=True)


def binomial_logit_logsf(value: ArrayLike, trials: ArrayLike, logits: ArrayLike) -> jax.Array:
    r"""Evaluate the logit-parameterized Binomial log survival function elementwise.

    For log odds :math:`\eta`, let :math:`p = \operatorname{sigmoid}(\eta)`.
    With :math:`0 \leq x < n`, the log survival probability is

    .. math::

        \log P(X > x)
        = \log I_p(\lfloor x \rfloor + 1, n - \lfloor x \rfloor),

    where :math:`I` is the regularized incomplete Beta function. This
    computes the strict upper tail without subtracting the CDF from one.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the survival probability.
        Fractional thresholds are rounded down to the nearest integer.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits : array_like
        Log odds of success. Negative and positive infinity correspond to
        success probabilities of zero and one, respectively.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the
        arguments. Negative thresholds produce zero and thresholds at or
        above the trial count produce ``-inf``. Invalid parameters or
        ``nan`` thresholds produce ``nan``.

    Notes
    -----
    Large trial counts can lose accuracy in float32. Enable JAX 64-bit mode
    when additional precision is needed.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import binomial_logit_logsf
           ...: thresholds = jnp.array([2, 4, 6])
           ...: log_prob = binomial_logit_logsf(
           ...:     thresholds,
           ...:     trials=10,
           ...:     logits=-0.8,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _binomial_log_probability(value, trials, logits, upper_tail=True, logit=True)


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
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    trials : array_like
        Numbers of independent trials. Values must be finite nonnegative
        integers.
    logits : array_like
        Log odds of success. The caller must not provide ``nan`` because it
        does not have a defined sampling result.
    sample_shape : tuple of int, default ()
        Independent sample dimensions prepended to the broadcast parameter
        shape. The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + broadcast_shape``.

    Examples
    --------
    Draw five outcomes using a key to make the draw reproducible.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import binomial_logit_rng
           ...: key = random.key(0)
           ...: binomial_logit_rng(
           ...:     key,
           ...:     trials=10,
           ...:     logits=-0.8,
           ...:     sample_shape=(5,),
           ...: )
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


def _binomial_log_probability(
    value: ArrayLike,
    trials: ArrayLike,
    parameter: ArrayLike,
    *,
    upper_tail: bool,
    logit: bool = False,
) -> jax.Array:
    value_array = _as_real_array("value", value)
    trials_array = _as_real_array("trials", trials)
    (parameter_array,) = _promote_inexact(("logits" if logit else "probability", parameter))
    # Keep integer thresholds exact before the shared count validation and subtraction
    threshold = jnp.floor(value_array) if jnp.issubdtype(value_array.dtype, jnp.floating) else value_array
    successes, failures, _, supported, valid_trials = _prepare_binomial_counts(
        threshold, trials_array, dtype=parameter_array.dtype
    )

    if logit:
        valid_parameter = ~jnp.isnan(parameter_array)
    else:
        valid_parameter = jnp.isfinite(parameter_array) & (parameter_array >= 0) & (parameter_array <= 1)
    interior = supported & (failures > 0) & valid_parameter
    safe_parameter = jnp.where(interior, parameter_array, 0.0 if logit else 0.5)
    if logit:
        success_probability = jnp.exp(jax.nn.log_sigmoid(safe_parameter))
        failure_probability = jnp.exp(jax.nn.log_sigmoid(-safe_parameter))
    else:
        success_probability = safe_parameter
        failure_probability = 1 - safe_parameter
    success_shape = jnp.where(interior, successes + 1, 2.0)
    failure_shape = jnp.where(interior, failures, 2.0)

    # Use the same symmetry criterion as JAX's incomplete Beta calculation
    # Keeping small p unreflected avoids losing it when 1 - p rounds to one
    reflected = success_probability > (success_shape + 1) / (success_shape + failure_shape + 2)
    first_shape = jnp.where(reflected, failure_shape, success_shape)
    second_shape = jnp.where(reflected, success_shape, failure_shape)
    argument = jnp.where(reflected, failure_probability, success_probability)
    direct_tail = reflected != upper_tail

    # Shape-one identities retain log probabilities without an exp/log round trip
    # They also avoid NaN endpoint derivatives in JAX for these Beta shapes
    complement_power = (first_shape == 1) & ((second_shape != 1) | ~direct_tail)
    direct_power = second_shape == 1
    invert_power = complement_power == direct_tail
    power_supported = ~invert_power | ((argument > 0) & (argument < 1))
    if logit:
        tail_logits = jnp.where(reflected, -safe_parameter, safe_parameter)
        # Select the inputs first so log-sigmoid and its derivative are evaluated once
        power_logits = jnp.where(complement_power, -tail_logits, tail_logits)
        power_shape = jnp.where(complement_power, second_shape, first_shape)
        log_power = power_shape * jax.nn.log_sigmoid(power_logits)
    else:
        complement_argument = jnp.where(complement_power & power_supported, argument, 0.5)
        direct_argument = jnp.where(direct_power & power_supported, argument, 0.5)
        log_power = jnp.where(
            complement_power,
            second_shape * jnp.log1p(-complement_argument),
            first_shape * jnp.log(direct_argument),
        )

    # Below this scale, squaring a tail probability can underflow in second derivatives
    minimum_tail = jnp.sqrt(jnp.finfo(argument.dtype).tiny)
    use_power = (complement_power | direct_power) & power_supported & (~invert_power | (-log_power >= minimum_tail))
    inverse_argument = jnp.where(use_power & invert_power, -log_power, 1.0)
    power_log_probability = jnp.where(invert_power, jax.nn.log1mexp(inverse_argument), log_power)

    # Safe inactive inputs keep boundary calculations out of unrelated gradients
    rounded_argument = (argument == 0) if logit else jnp.zeros_like(argument, dtype=jnp.bool_)
    skip_beta = use_power | rounded_argument
    beta_probability = betainc(
        jnp.where(skip_beta, 2.0, first_shape),
        jnp.where(skip_beta, 2.0, second_shape),
        jnp.where(skip_beta, 0.5, argument),
    )
    beta_probability = jnp.where(rounded_argument, 0.0, beta_probability)
    can_sum = jnp.isfinite(safe_parameter) if logit else (argument > 0) & (argument < 1)
    needs_sum = direct_tail & ~use_power & interior & can_sum & (beta_probability < minimum_tail)
    direct_probability = jnp.where(direct_tail & ~needs_sum, beta_probability, 1.0)
    complement_probability = jnp.where(direct_tail, 0.0, beta_probability)
    log_probability = jnp.where(direct_tail, jnp.log(direct_probability), jnp.log1p(-complement_probability))
    log_probability = jnp.where(use_power, power_log_probability, log_probability)

    def sum_small_tail(_: None) -> jax.Array:
        first_count = jnp.where(needs_sum, first_shape, 1.0)
        trials = jnp.where(needs_sum, first_shape + second_shape - 1, 1.0)
        if logit:
            logits = jnp.where(needs_sum, tail_logits, 0.0)
            log_mass = binomial_logit_logpmf(first_count, trials, logits)
        else:
            probability = jnp.where(needs_sum, argument, 0.5)
            logits = jnp.log(probability) - jnp.log1p(-probability)
            log_mass = binomial_logpmf(first_count, trials, probability)
        relative_sum = _binomial_tail_sum(first_count, trials, logits)
        recovered = log_mass + jnp.log(relative_sum)
        return jnp.where(needs_sum, recovered, log_probability)

    log_probability = cast(
        jax.Array,
        jax.lax.cond(jnp.any(needs_sum), sum_small_tail, lambda _: log_probability, operand=None),
    )

    below_support = 0.0 if upper_tail else -jnp.inf
    above_support = -jnp.inf if upper_tail else 0.0
    log_probability = jnp.where(value_array < 0, below_support, log_probability)
    log_probability = jnp.where(value_array >= trials_array, above_support, log_probability)
    return jnp.where(valid_trials & valid_parameter & ~jnp.isnan(value_array), log_probability, jnp.nan)


@jax.custom_jvp
def _binomial_tail_sum(first_count: jax.Array, trials: jax.Array, logits: jax.Array) -> jax.Array:
    # Boost also sums neighboring Binomial masses when evaluating incomplete Beta
    # Factoring out the first log mass keeps the terms representable after Beta underflows
    first_count, trials, logits = jnp.broadcast_arrays(first_count, trials, logits)
    odds = jnp.exp(logits)
    epsilon = jnp.finfo(logits.dtype).eps

    def has_terms(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        count, term, total = state
        return jnp.any((count < trials) & (term > epsilon * total))

    def add_term(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        count, term, total = state
        active = (count < trials) & (term > epsilon * total)
        next_term = jnp.where(active, term * ((trials - count) / (count + 1)) * odds, 0.0)
        return count + active.astype(count.dtype), next_term, total + next_term

    _, _, relative_sum = jax.lax.while_loop(
        has_terms, add_term, (first_count, jnp.ones_like(logits), jnp.ones_like(logits))
    )
    return relative_sum


@_binomial_tail_sum.defjvp
def _binomial_tail_sum_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    first_count, trials, logits = primals
    _, _, logits_tangent = tangents
    relative_sum = _binomial_tail_sum(first_count, trials, logits)
    # Counts are discrete; this identity avoids differentiating the dynamic loop
    # Working with the factored sum also avoids subtracting nearly equal log masses
    # Differentiating in log odds avoids dividing by an underflowed probability
    derivative = first_count * jax.nn.sigmoid(-logits) * (1 - relative_sum) + relative_sum * (
        (trials - first_count) * jax.nn.sigmoid(logits)
    )
    return relative_sum, derivative * logits_tangent


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
