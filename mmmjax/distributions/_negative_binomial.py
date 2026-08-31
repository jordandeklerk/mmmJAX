"""Negative Binomial distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._discrete import (
    _binomial_interior_log_mass,
    _prepare_nonnegative_count,
)
from mmmjax.distributions._utils import (
    _as_real_array,
    _gamma_shape_log_derivative,
    _promote_inexact,
    _weighted_log_ratio_deviance,
)


def negative_binomial_logpmf(
    value: ArrayLike,
    mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Negative Binomial log probability mass elementwise.

    For count :math:`k \in \{0, 1, \ldots\}`, mean :math:`\mu > 0`, and
    concentration :math:`\phi > 0`, the log probability mass is

    .. math::

        \log p(k \mid \mu, \phi)
        = \log\Gamma(k + \phi)
          - \log\Gamma(\phi)
          - \log\Gamma(k + 1)
          + \phi\log\left(\frac{\phi}{\mu + \phi}\right)
          + k\log\left(\frac{\mu}{\mu + \phi}\right).

    This parameterization has variance
    :math:`\mu + \mu^2 / \phi` and approaches a Poisson distribution as
    :math:`\phi` increases.

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass.
    mean
        Finite positive mean parameter.
    concentration
        Finite positive concentration parameter. Larger values reduce
        overdispersion relative to a Poisson distribution.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the nonnegative integer support produce
        ``-inf``. A nonpositive or nonfinite parameter produces ``nan``.
    """
    value_array = _as_real_array("value", value)
    mean_array, concentration_array = _promote_inexact(
        ("mean", mean),
        ("concentration", concentration),
    )
    count, supported = _prepare_nonnegative_count(value_array, dtype=mean_array.dtype)

    valid_mean = jnp.isfinite(mean_array) & (mean_array > 0)
    valid_concentration = jnp.isfinite(concentration_array) & (concentration_array > 0)
    valid_parameters = valid_mean & valid_concentration
    safe_mean = jnp.where(valid_mean, mean_array, jnp.ones_like(mean_array))
    safe_concentration = jnp.where(
        valid_concentration,
        concentration_array,
        jnp.ones_like(concentration_array),
    )

    supported_log_mass = _negative_binomial_log_mass(
        count,
        safe_mean,
        safe_concentration,
    )
    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_parameters, log_mass, jnp.nan)


def negative_binomial(
    value: ArrayLike,
    mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Negative Binomial log probability masses.

    Parameters
    ----------
    value
        Counts at which to evaluate the probability mass.
    mean
        Finite positive mean parameter.
    concentration
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.
    """
    return jnp.sum(negative_binomial_logpmf(value, mean, concentration))


@jax.custom_jvp
def _negative_binomial_log_mass(
    count: jax.Array,
    mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    log_mean = jnp.log(mean)
    log_concentration = jnp.log(concentration)
    log_count_probability = jax.nn.log_sigmoid(log_mean - log_concentration)
    log_concentration_probability = jax.nn.log_sigmoid(log_concentration - log_mean)

    positive_count = count > 0
    interior_count = jnp.where(positive_count, count, jnp.ones_like(count))
    total_count = interior_count + concentration

    # This generalized Binomial identity reuses Loader's stable large-count calculation
    interior_log_mass = _binomial_interior_log_mass(
        interior_count,
        concentration,
        total_count,
        jnp.exp(log_count_probability),
        jnp.exp(log_concentration_probability),
        log_count_probability,
        log_concentration_probability,
    )
    interior_log_mass += _log_concentration_fraction(interior_count, concentration)

    zero_count_log_mass = concentration * log_concentration_probability
    return jnp.where(positive_count, interior_log_mass, zero_count_log_mass)


@_negative_binomial_log_mass.defjvp
def _negative_binomial_log_mass_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    count, mean, concentration = primals
    count_tangent, mean_tangent, concentration_tangent = tangents
    log_mass = _negative_binomial_log_mass(count, mean, concentration)

    log_mean = jnp.log(mean)
    log_concentration = jnp.log(concentration)
    log_parameter_sum = jnp.logaddexp(log_mean, log_concentration)
    log_count_probability = jax.nn.log_sigmoid(log_mean - log_concentration)
    log_concentration_probability = jax.nn.log_sigmoid(log_concentration - log_mean)

    positive_count = count > 0
    safe_count = jnp.where(positive_count, count, jnp.ones_like(count))
    log_count = jnp.log(safe_count)
    total_count = count + concentration
    log_total_count = jnp.logaddexp(
        jnp.where(positive_count, log_count, -jnp.inf),
        log_concentration,
    )

    count_plus_one = count + 1
    count_derivative = (
        log_total_count
        - jnp.log(count_plus_one)
        - _gamma_shape_log_derivative(total_count)
        + _gamma_shape_log_derivative(count_plus_one)
        + log_count_probability
    )

    concentration_probability = jnp.exp(log_concentration_probability)
    log_count_mean_ratio = log_count - log_mean
    mean_derivative = jnp.where(
        positive_count,
        jnp.where(
            log_count_mean_ratio > 0,
            jnp.exp(log_concentration_probability + log_count_mean_ratio) - concentration_probability,
            concentration_probability * jnp.expm1(log_count_mean_ratio),
        ),
        -concentration_probability,
    )

    scale = jnp.maximum(jnp.maximum(count, mean), concentration)
    ratio_difference = ((count / scale) - (mean / scale)) / ((mean / scale) + (concentration / scale))
    raw_log_ratio = log_total_count - log_parameter_sum
    stable_log_ratio = jnp.where(
        jnp.isfinite(ratio_difference) & (ratio_difference > -1),
        jnp.log1p(ratio_difference),
        raw_log_ratio,
    )
    concentration_derivative = (
        _weighted_log_ratio_deviance(
            jnp.ones_like(concentration),
            stable_log_ratio,
            ratio_difference,
        )
        + _gamma_shape_log_derivative(concentration)
        - _gamma_shape_log_derivative(total_count)
    )

    log_mass_tangent = (
        count_derivative * count_tangent
        + mean_derivative * mean_tangent
        + concentration_derivative * concentration_tangent
    )
    return log_mass, log_mass_tangent


def _log_concentration_fraction(
    count: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    count_is_smaller = count <= concentration
    smaller = jnp.where(count_is_smaller, count, concentration)
    larger = jnp.where(count_is_smaller, concentration, count)
    ratio = smaller / larger

    return jnp.where(
        count_is_smaller,
        -jnp.log1p(ratio),
        jnp.log(concentration) - jnp.log(count) - jnp.log1p(ratio),
    )
