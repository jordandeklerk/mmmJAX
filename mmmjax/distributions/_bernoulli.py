"""Bernoulli distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _as_real_array, _promote_inexact, _random_shape


def bernoulli_logpmf(value: ArrayLike, probability: ArrayLike) -> jax.Array:
    r"""Evaluate the Bernoulli log probability mass elementwise.

    For outcome :math:`k \in \{0, 1\}` and success probability
    :math:`p \in [0, 1]`, the log probability mass is

    .. math::

        \log p(k \mid p)
        = \begin{cases}
            \log(1 - p), & k = 0, \\
            \log(p), & k = 1, \\
            -\infty, & \text{otherwise}.
          \end{cases}

    Parameters
    ----------
    value
        Binary outcomes at which to evaluate the probability mass.
    probability
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the binary support produce ``-inf``. A
        nonfinite probability or one outside ``[0, 1]`` produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    (probability_array,) = _promote_inexact(("probability", probability))

    is_failure = value_array == 0
    is_success = value_array == 1
    valid_probability = jnp.isfinite(probability_array) & (probability_array >= 0) & (probability_array <= 1)
    safe_probability = jnp.where(valid_probability, probability_array, jnp.full_like(probability_array, 0.5))

    success_probability = jnp.where(is_success, safe_probability, jnp.ones_like(safe_probability))
    failure_probability = jnp.where(is_failure, safe_probability, jnp.zeros_like(safe_probability))
    supported_log_mass = jnp.where(
        is_success,
        jnp.log(success_probability),
        jnp.log1p(-failure_probability),
    )

    log_mass = jnp.where(is_failure | is_success, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_probability, log_mass, jnp.nan)


def bernoulli(value: ArrayLike, probability: ArrayLike) -> jax.Array:
    """Return the scalar sum of Bernoulli log probability masses.

    Parameters
    ----------
    value
        Binary outcomes at which to evaluate the probability mass.
    probability
        Success probabilities in the closed interval from zero to one.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(bernoulli_logpmf(value, probability))


def bernoulli_rng(
    key: jax.Array,
    probability: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw Bernoulli outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    probability
        Success probabilities in the closed interval from zero to one. The
        caller must provide valid probabilities because invalid values do not
        have a defined sampling result.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + probability.shape``.
    """
    (probability_array,) = _promote_inexact(("probability", probability))
    output_shape = _random_shape(sample_shape, probability_array)

    samples = jax.random.bernoulli(
        key,
        probability_array,
        shape=output_shape,
        mode="high",
    )
    return samples.astype(jnp.int32)


def bernoulli_logit_logpmf(value: ArrayLike, logits: ArrayLike) -> jax.Array:
    r"""Evaluate the logit-parameterized Bernoulli log probability mass.

    For outcome :math:`k \in \{0, 1\}` and log odds :math:`\eta`, the log
    probability mass is

    .. math::

        \log p(k \mid \eta)
        = \log\!\left[\operatorname{logit}^{-1}
          \!\left((2k - 1)\eta\right)\right].

    Parameters
    ----------
    value
        Binary outcomes at which to evaluate the probability mass.
    logits
        Log odds of success.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the binary support produce ``-inf`` and a
        ``nan`` logit produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    (logits_array,) = _promote_inexact(("logits", logits))

    is_failure = value_array == 0
    is_success = value_array == 1
    signed_logits = jnp.where(is_success, logits_array, -logits_array)
    supported_log_mass = jax.nn.log_sigmoid(signed_logits)

    log_mass = jnp.where(is_failure | is_success, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(jnp.isnan(logits_array), jnp.nan, log_mass)


def bernoulli_logit(value: ArrayLike, logits: ArrayLike) -> jax.Array:
    """Return the scalar sum of logit-parameterized Bernoulli log masses.

    Parameters
    ----------
    value
        Binary outcomes at which to evaluate the probability mass.
    logits
        Log odds of success.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(bernoulli_logit_logpmf(value, logits))


def bernoulli_logit_rng(
    key: jax.Array,
    logits: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw logit-parameterized Bernoulli outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    logits
        Log odds of success. The caller must not provide ``nan`` because it
        does not have a defined sampling result.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + logits.shape``.
    """
    (logits_array,) = _promote_inexact(("logits", logits))
    output_shape = _random_shape(sample_shape, logits_array)

    # Sampling from logits keeps rare outcomes that sigmoid can round away in float32
    categorical_logits = jnp.stack(
        (jnp.zeros_like(logits_array), logits_array),
        axis=-1,
    )
    samples = jax.random.categorical(
        key,
        categorical_logits,
        shape=output_shape,
        mode="high",
    )
    return samples.astype(jnp.int32)
