"""Poisson distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.typing import ArrayLike, DTypeLike

from mmmjax.distributions._utils import (
    _as_real_array,
    _gamma_shape_normalizer,
    _promote_inexact,
    _random_shape,
    _stable_log_ratio,
    _weighted_log_ratio_deviance,
)


def poisson_logpmf(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    r"""Evaluate the Poisson log probability mass elementwise.

    For count :math:`k \in \{0, 1, \ldots\}` and rate
    :math:`\lambda \geq 0`, the log probability mass is

    .. math::

        \log p(k \mid \lambda)
        = k\log(\lambda) - \lambda - \log\Gamma(k + 1),

    where :math:`0\log(0)` is defined as zero.

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass. With float32 rates,
        counts above 16,777,216 must be paired with float64 rates in JAX
        64-bit mode when adjacent integer values must remain distinguishable.
    rate
        Finite nonnegative rate parameter, equal to both the mean and
        variance.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the nonnegative integer support produce
        ``-inf``. A negative or nonfinite rate produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    (rate_array,) = _promote_inexact(("rate", rate))
    count, supported = _prepare_poisson_count(value_array, dtype=rate_array.dtype)

    valid_rate = jnp.isfinite(rate_array) & (rate_array >= 0)
    safe_rate = jnp.where(valid_rate, rate_array, jnp.ones_like(rate_array))
    positive_count = count > 0

    direct_rate = jnp.where(positive_count, safe_rate, jnp.ones_like(safe_rate))
    direct_log_mass = count * jnp.log(direct_rate) - safe_rate

    use_interior_mass = positive_count & (safe_rate > 0)
    interior_count = jnp.where(use_interior_mass, count, jnp.ones_like(count))
    interior_rate = jnp.where(use_interior_mass, safe_rate, jnp.ones_like(safe_rate))
    interior_log_mass = _poisson_rate_interior_log_mass(
        interior_count,
        interior_rate,
    )

    supported_log_mass = jnp.where(use_interior_mass, interior_log_mass, direct_log_mass)

    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_rate, log_mass, jnp.nan)


def poisson(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    """Return the scalar sum of Poisson log probability masses.

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass. With float32 rates,
        counts above 16,777,216 must be paired with float64 rates in JAX
        64-bit mode when adjacent integer values must remain distinguishable.
    rate
        Finite nonnegative rate parameter, equal to both the mean and
        variance.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(poisson_logpmf(value, rate))


def poisson_rng(
    key: jax.Array,
    rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw Poisson outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    rate
        Nonnegative rate parameter. The caller must provide finite rates whose
        sampled outcomes fit in ``int32`` because invalid or larger values do
        not have a defined result.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + rate.shape``.
    """
    (rate_array,) = _promote_inexact(("rate", rate))
    output_shape = _random_shape(sample_shape, rate_array)
    return jax.random.poisson(
        key,
        rate_array,
        shape=output_shape,
        dtype=jnp.int32,
    )


def poisson_log_logpmf(value: ArrayLike, log_rate: ArrayLike) -> jax.Array:
    r"""Evaluate the log-rate Poisson log probability mass elementwise.

    For count :math:`k \in \{0, 1, \ldots\}` and log rate
    :math:`\eta \in \mathbb{R}`, the log probability mass is

    .. math::

        \log p(k \mid \eta)
        = k\eta - \exp(\eta) - \log\Gamma(k + 1).

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass. With float32 log
        rates, counts above 16,777,216 must be paired with float64 log rates
        in JAX 64-bit mode when adjacent integer values must remain
        distinguishable.
    log_rate
        Logarithm of the Poisson rate. Negative infinity represents a
        degenerate distribution at zero.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the nonnegative integer support produce
        ``-inf``. A ``nan`` log rate produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    (log_rate_array,) = _promote_inexact(("log_rate", log_rate))
    count, supported = _prepare_poisson_count(value_array, dtype=log_rate_array.dtype)

    valid_log_rate = ~jnp.isnan(log_rate_array)
    safe_log_rate = jnp.where(valid_log_rate, log_rate_array, jnp.zeros_like(log_rate_array))
    positive_count = count > 0

    use_interior_mass = positive_count & jnp.isfinite(safe_log_rate)
    direct_log_rate = jnp.where(use_interior_mass, jnp.zeros_like(safe_log_rate), safe_log_rate)
    direct_rate = jnp.exp(direct_log_rate)
    linear_log_rate = jnp.where(positive_count & ~jnp.isposinf(direct_log_rate), direct_log_rate, 0)
    direct_log_mass = count * linear_log_rate - direct_rate

    interior_count = jnp.where(use_interior_mass, count, jnp.ones_like(count))
    interior_log_rate = jnp.where(use_interior_mass, safe_log_rate, jnp.zeros_like(safe_log_rate))
    interior_log_mass = _poisson_log_interior_log_mass(
        interior_count,
        interior_log_rate,
    )

    supported_log_mass = jnp.where(use_interior_mass, interior_log_mass, direct_log_mass)

    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_log_rate, log_mass, jnp.nan)


def poisson_log(value: ArrayLike, log_rate: ArrayLike) -> jax.Array:
    """Return the scalar sum of log-rate Poisson log probability masses.

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass. With float32 log
        rates, counts above 16,777,216 must be paired with float64 log rates
        in JAX 64-bit mode when adjacent integer values must remain
        distinguishable.
    log_rate
        Logarithm of the Poisson rate.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(poisson_log_logpmf(value, log_rate))


def poisson_log_rng(
    key: jax.Array,
    log_rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw log-rate Poisson outcomes using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    log_rate
        Logarithm of the Poisson rate. The caller must provide values whose
        exponentiated rates are finite and whose sampled outcomes fit in
        ``int32``.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Integer outcomes with shape ``sample_shape + log_rate.shape``.
    """
    (log_rate_array,) = _promote_inexact(("log_rate", log_rate))
    return poisson_rng(
        key,
        jnp.exp(log_rate_array),
        sample_shape=sample_shape,
    )


def _prepare_poisson_count(
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


@jax.custom_jvp
def _poisson_rate_interior_log_mass(
    count: jax.Array,
    rate: jax.Array,
) -> jax.Array:
    raw_log_ratio = jnp.log(rate) - jnp.log(count)
    log_ratio, _, _ = _stable_log_ratio(rate, count, raw_log_ratio)
    return _poisson_stable_log_mass(count, log_ratio, rate - count)


@_poisson_rate_interior_log_mass.defjvp
def _poisson_rate_interior_log_mass_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    count, rate = primals
    count_tangent, rate_tangent = tangents
    log_mass = _poisson_rate_interior_log_mass(count, rate)

    count_derivative = jnp.log(rate) - digamma(count + 1)
    rate_derivative = count / rate - 1
    log_mass_tangent = count_derivative * count_tangent + rate_derivative * rate_tangent
    return log_mass, log_mass_tangent


@jax.custom_jvp
def _poisson_log_interior_log_mass(
    count: jax.Array,
    log_rate: jax.Array,
) -> jax.Array:
    log_ratio, relative_rate_difference = _poisson_log_ratio(count, log_rate)
    linear_deviation = count * relative_rate_difference
    log_mass = _poisson_stable_log_mass(count, log_ratio, linear_deviation)

    # This equivalent form remains finite when the rate itself exceeds the dtype range
    log_rate_deviance = count * (log_ratio - relative_rate_difference)
    log_rate_log_mass = _gamma_shape_normalizer(count) - jnp.log(count) + log_rate_deviance
    return jnp.where(log_ratio > 2, log_rate_log_mass, log_mass)


@_poisson_log_interior_log_mass.defjvp
def _poisson_log_interior_log_mass_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    count, log_rate = primals
    count_tangent, log_rate_tangent = tangents
    log_mass = _poisson_log_interior_log_mass(count, log_rate)

    count_derivative = log_rate - digamma(count + 1)
    _, relative_rate_difference = _poisson_log_ratio(count, log_rate)
    log_rate_derivative = -count * relative_rate_difference
    log_mass_tangent = count_derivative * count_tangent + log_rate_derivative * log_rate_tangent
    return log_mass, log_mass_tangent


def _poisson_log_ratio(
    count: jax.Array,
    log_rate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    raw_log_ratio = log_rate - jnp.log(count)
    rate = jnp.exp(log_rate)
    stable_log_ratio, ratio_deviation, valid_ratio = _stable_log_ratio(
        rate,
        count,
        raw_log_ratio,
    )

    log_ratio = jnp.where(valid_ratio, stable_log_ratio, raw_log_ratio)
    relative_rate_difference = jnp.where(valid_ratio, ratio_deviation, jnp.expm1(raw_log_ratio))
    return log_ratio, relative_rate_difference


def _poisson_stable_log_mass(
    count: jax.Array,
    log_ratio: jax.Array,
    linear_deviation: jax.Array,
) -> jax.Array:
    # Loader's deviance decomposition avoids large-count cancellation near the mode
    return (
        _gamma_shape_normalizer(count)
        - jnp.log(count)
        + _weighted_log_ratio_deviance(count, log_ratio, linear_deviation)
    )
