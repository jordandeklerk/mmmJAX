"""Negative Binomial distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.typing import ArrayLike

from mmmjax.distributions._beta_cdf import _log_betainc
from mmmjax.distributions._discrete import (
    _binomial_interior_log_mass,
    _prepare_nonnegative_count,
)
from mmmjax.distributions._utils import (
    _as_real_array,
    _gamma_shape_log_derivative,
    _promote_inexact,
    _random_shape,
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
    value : array_like
        Counts at which to evaluate the probability mass. Values must be
        nonnegative integers.
    mean : array_like
        Finite positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter. Larger values reduce
        overdispersion relative to a Poisson distribution.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the nonnegative integer support produce
        ``-inf``. A nonpositive or nonfinite parameter produces ``nan``.

    Examples
    --------
    Evaluate one log probability mass per value. Specify the mean count and
    concentration.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_logpmf
           ...: counts = jnp.array([0, 2, 5])
           ...: negative_binomial_logpmf(counts, mean=4.0, concentration=2.0)
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
    value : array_like
        Counts at which to evaluate the probability mass. Values must be
        nonnegative integers.
    mean : array_like
        Finite positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.

    Examples
    --------
    Sum the log probability masses of independent observations into a single log
    likelihood. Specify the mean count and concentration.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial
           ...: counts = jnp.array([0, 2, 5])
           ...: negative_binomial(counts, mean=4.0, concentration=2.0)
    """
    return jnp.sum(negative_binomial_logpmf(value, mean, concentration))


def negative_binomial_logcdf(
    value: ArrayLike,
    mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Negative Binomial log cumulative probability elementwise.

    For threshold :math:`x \geq 0`, mean :math:`\mu > 0`, and concentration
    :math:`\phi > 0`, the cumulative probability is

    .. math::

        \log P(X \leq x)
        = \log I_{\phi/(\phi + \mu)}(\phi, \lfloor x \rfloor + 1),

    where :math:`I_z(a, b)` is the regularized incomplete Beta function.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the cumulative probability.
        Fractional thresholds are rounded down to the nearest integer.
    mean : array_like
        Finite positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Negative thresholds produce ``-inf`` and positive
        infinity produces zero. An invalid parameter or ``nan`` threshold
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_logcdf
           ...: thresholds = jnp.array([1, 4, 8])
           ...: log_prob = negative_binomial_logcdf(
           ...:     thresholds,
           ...:     mean=4.0,
           ...:     concentration=2.0,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _negative_binomial_log_probability(value, mean, concentration, upper_tail=False)


def negative_binomial_logsf(
    value: ArrayLike,
    mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Negative Binomial log survival probability elementwise.

    For threshold :math:`x \geq 0`, mean :math:`\mu > 0`, and concentration
    :math:`\phi > 0`, the survival probability is

    .. math::

        \log P(X > x)
        = \log I_{\mu/(\phi + \mu)}(\lfloor x \rfloor + 1, \phi),

    where :math:`I_z(a, b)` is the regularized incomplete Beta function.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the survival probability.
        Fractional thresholds are rounded down to the nearest integer.
    mean : array_like
        Finite positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the
        arguments. Negative thresholds produce zero and positive infinity
        produces ``-inf``. An invalid parameter or ``nan`` threshold
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_logsf
           ...: thresholds = jnp.array([1, 4, 8])
           ...: log_prob = negative_binomial_logsf(
           ...:     thresholds,
           ...:     mean=4.0,
           ...:     concentration=2.0,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _negative_binomial_log_probability(value, mean, concentration, upper_tail=True)


def negative_binomial_rng(
    key: jax.Array,
    mean: ArrayLike,
    concentration: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw Negative Binomial outcomes using a JAX random key.

    Parameters
    ----------
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    mean : array_like
        Finite positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.
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
           ...: from mmmjax import negative_binomial_rng
           ...: key = random.key(0)
           ...: negative_binomial_rng(
           ...:     key,
           ...:     mean=4.0,
           ...:     concentration=2.0,
           ...:     sample_shape=(5,),
           ...: )
    """
    mean_array, concentration_array = _promote_inexact(
        ("mean", mean),
        ("concentration", concentration),
    )
    output_shape = _random_shape(sample_shape, mean_array, concentration_array)
    return _negative_binomial_log_mean_rng(
        key,
        jnp.log(mean_array),
        concentration_array,
        output_shape=output_shape,
    )


def negative_binomial_log_logpmf(
    value: ArrayLike,
    log_mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the log-mean Negative Binomial log probability mass.

    For count :math:`k \in \{0, 1, \ldots\}`, log mean
    :math:`\eta \in \mathbb{R}`, and concentration :math:`\phi > 0`, the log
    probability mass is

    .. math::

        \log p(k \mid \eta, \phi)
        = \log\Gamma(k + \phi)
          - \log\Gamma(\phi)
          - \log\Gamma(k + 1)
          + k\eta
          + \phi\log(\phi)
          - (k + \phi)\log\left(\exp(\eta) + \phi\right).

    The implied mean is :math:`\exp(\eta)` and the variance is
    :math:`\exp(\eta) + \exp(2\eta) / \phi`.

    Parameters
    ----------
    value : array_like
        Counts at which to evaluate the probability mass. Values must be
        nonnegative integers.
    log_mean : array_like
        Finite logarithm of the Negative Binomial mean.
    concentration : array_like
        Finite positive concentration parameter. Larger values reduce
        overdispersion relative to a Poisson distribution.

    Returns
    -------
    jax.Array
        Normalized log probability masses with the broadcast shape of the
        arguments. Values outside the nonnegative integer support produce
        ``-inf``. A nonfinite log mean or a nonpositive or nonfinite
        concentration produces ``nan``.

    Examples
    --------
    Evaluate one log probability mass per value. Supply the logarithm of the
    mean count and a concentration.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_log_logpmf
           ...: counts = jnp.array([0, 2, 5])
           ...: negative_binomial_log_logpmf(
           ...:     counts,
           ...:     log_mean=jnp.log(4.0),
           ...:     concentration=2.0,
           ...: )
    """
    value_array = _as_real_array("value", value)
    log_mean_array, concentration_array = _promote_inexact(
        ("log_mean", log_mean),
        ("concentration", concentration),
    )
    count, supported = _prepare_nonnegative_count(value_array, dtype=log_mean_array.dtype)

    valid_log_mean = jnp.isfinite(log_mean_array)
    valid_concentration = jnp.isfinite(concentration_array) & (concentration_array > 0)
    valid_parameters = valid_log_mean & valid_concentration
    safe_log_mean = jnp.where(valid_log_mean, log_mean_array, jnp.zeros_like(log_mean_array))
    safe_concentration = jnp.where(
        valid_concentration,
        concentration_array,
        jnp.ones_like(concentration_array),
    )

    supported_log_mass = _negative_binomial_log_mean_log_mass(
        count,
        safe_log_mean,
        safe_concentration,
    )
    log_mass = jnp.where(supported, supported_log_mass, -jnp.inf)
    log_mass = jnp.where(jnp.isnan(value_array), jnp.nan, log_mass)
    return jnp.where(valid_parameters, log_mass, jnp.nan)


def negative_binomial_log(
    value: ArrayLike,
    log_mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of log-mean Negative Binomial log masses.

    Parameters
    ----------
    value : array_like
        Counts at which to evaluate the probability mass. Values must be
        nonnegative integers.
    log_mean : array_like
        Finite logarithm of the Negative Binomial mean.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Complete normalized log probability mass summed across every
        dimension of the broadcast result.

    Examples
    --------
    Sum the log probability masses of independent observations into a single log
    likelihood. Supply the logarithm of the mean count and a concentration.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_log
           ...: counts = jnp.array([0, 2, 5])
           ...: negative_binomial_log(
           ...:     counts,
           ...:     log_mean=jnp.log(4.0),
           ...:     concentration=2.0,
           ...: )
    """
    return jnp.sum(negative_binomial_log_logpmf(value, log_mean, concentration))


def negative_binomial_log_logcdf(
    value: ArrayLike,
    log_mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Negative Binomial log CDF with a log-mean parameter.

    For threshold :math:`x \geq 0`, log mean :math:`\eta \in \mathbb{R}`,
    and concentration :math:`\phi > 0`, the cumulative probability is

    .. math::

        \log P(X \leq x)
        = \log I_{\phi/(\phi + e^\eta)}(\phi, \lfloor x \rfloor + 1),

    where :math:`I_z(a, b)` is the regularized incomplete Beta function.
    The calculation uses the log mean without first exponentiating it.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the cumulative probability.
        Fractional thresholds are rounded down to the nearest integer.
    log_mean : array_like
        Finite logarithm of the positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Negative thresholds produce ``-inf`` and positive
        infinity produces zero. An invalid parameter or ``nan`` threshold
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X \leq x)` at three thresholds, then convert the
    results to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_log_logcdf
           ...: thresholds = jnp.array([1, 4, 8])
           ...: log_prob = negative_binomial_log_logcdf(
           ...:     thresholds,
           ...:     log_mean=jnp.log(4.0),
           ...:     concentration=2.0,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _negative_binomial_log_probability(value, log_mean, concentration, upper_tail=False, log_mean=True)


def negative_binomial_log_logsf(
    value: ArrayLike,
    log_mean: ArrayLike,
    concentration: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Negative Binomial log survival with a log-mean parameter.

    For threshold :math:`x \geq 0`, log mean :math:`\eta \in \mathbb{R}`,
    and concentration :math:`\phi > 0`, the survival probability is

    .. math::

        \log P(X > x)
        = \log I_{e^\eta/(\phi + e^\eta)}(\lfloor x \rfloor + 1, \phi),

    where :math:`I_z(a, b)` is the regularized incomplete Beta function.
    The calculation uses the log mean without first exponentiating it.

    Parameters
    ----------
    value : array_like
        Thresholds at which to evaluate the survival probability.
        Fractional thresholds are rounded down to the nearest integer.
    log_mean : array_like
        Finite logarithm of the positive mean parameter.
    concentration : array_like
        Finite positive concentration parameter.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the
        arguments. Negative thresholds produce zero and positive infinity
        produces ``-inf``. An invalid parameter or ``nan`` threshold
        produces ``nan``.

    Examples
    --------
    Evaluate :math:`\log P(X > x)` at three thresholds, then convert the results
    to probabilities.

    .. ipython::

        In [1]: import jax.numpy as jnp
           ...: from mmmjax import negative_binomial_log_logsf
           ...: thresholds = jnp.array([1, 4, 8])
           ...: log_prob = negative_binomial_log_logsf(
           ...:     thresholds,
           ...:     log_mean=jnp.log(4.0),
           ...:     concentration=2.0,
           ...: )
           ...: jnp.exp(log_prob)
    """
    return _negative_binomial_log_probability(value, log_mean, concentration, upper_tail=True, log_mean=True)


def negative_binomial_log_rng(
    key: jax.Array,
    log_mean: ArrayLike,
    concentration: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw log-mean Negative Binomial outcomes using a JAX random key.

    Parameters
    ----------
    key : jax.Array
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    log_mean : array_like
        Finite logarithm of the Negative Binomial mean.
    concentration : array_like
        Finite positive concentration parameter.
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

        In [1]: import jax.numpy as jnp
           ...: from jax import random
           ...: from mmmjax import negative_binomial_log_rng
           ...: key = random.key(0)
           ...: negative_binomial_log_rng(
           ...:     key,
           ...:     log_mean=jnp.log(4.0),
           ...:     concentration=2.0,
           ...:     sample_shape=(5,),
           ...: )
    """
    log_mean_array, concentration_array = _promote_inexact(
        ("log_mean", log_mean),
        ("concentration", concentration),
    )
    output_shape = _random_shape(sample_shape, log_mean_array, concentration_array)
    return _negative_binomial_log_mean_rng(
        key,
        log_mean_array,
        concentration_array,
        output_shape=output_shape,
    )


def _negative_binomial_log_probability(
    value: ArrayLike,
    mean: ArrayLike,
    concentration: ArrayLike,
    *,
    upper_tail: bool,
    log_mean: bool = False,
) -> jax.Array:
    value_array = _as_real_array("value", value)
    mean_array, concentration_array = _promote_inexact(
        ("log_mean" if log_mean else "mean", mean), ("concentration", concentration)
    )
    valid_mean = jnp.isfinite(mean_array)
    if not log_mean:
        valid_mean &= mean_array > 0
    valid_parameters = valid_mean & jnp.isfinite(concentration_array) & (concentration_array > 0)
    finite_threshold = jnp.isfinite(value_array) & (value_array >= 0)
    count = jnp.floor(jnp.where(finite_threshold, value_array, 0)).astype(mean_array.dtype)
    evaluate = finite_threshold & valid_parameters
    zero_count = (count == 0) & (not log_mean)

    if not log_mean:
        # For zero counts the CDF is exactly the zero-count mass, including its concentration derivative
        zero_mean = jnp.where(evaluate & zero_count, mean_array, 1.0)
        zero_concentration = jnp.where(evaluate & zero_count, concentration_array, 1.0)
        zero_logcdf = _negative_binomial_log_mass(jnp.zeros_like(count), zero_mean, zero_concentration)
        zero_log_probability = jax.nn.log1mexp(-zero_logcdf) if upper_tail else zero_logcdf

    safe_mean = jnp.where(evaluate & ~zero_count, mean_array, 1.0)
    safe_concentration = jnp.where(evaluate & ~zero_count, concentration_array, 1.0)
    count_shape = jnp.where(evaluate & ~zero_count, count + 1, 2.0)
    # Compute the smaller probability directly and its complement by subtraction
    # This avoids overflowing the parameter sum or rounding both probabilities independently
    if log_mean:
        log_odds = safe_mean - jnp.log(safe_concentration)
        mean_is_smaller = log_odds <= 0
        ratio = jnp.exp(jnp.where(mean_is_smaller, log_odds, -log_odds))
    else:
        mean_is_smaller = safe_mean <= safe_concentration
        smaller = jnp.where(mean_is_smaller, safe_mean, safe_concentration)
        larger = jnp.where(mean_is_smaller, safe_concentration, safe_mean)
        ratio = smaller / larger
    small_probability = ratio / (1 + ratio)
    probability = jnp.where(mean_is_smaller, 1 - small_probability, small_probability)
    complement = jnp.where(mean_is_smaller, small_probability, 1 - small_probability)

    reflected = probability > (safe_concentration + 1) / (safe_concentration + count_shape + 2)
    first_shape = jnp.where(reflected, count_shape, safe_concentration)
    second_shape = jnp.where(reflected, safe_concentration, count_shape)
    argument = jnp.where(reflected, complement, probability)
    interior = (argument > 0) & (argument < 1)
    beta_value = jax.nn.log_sigmoid(jnp.where(reflected, log_odds, -log_odds)) if log_mean else argument
    log_beta = _log_betainc(
        jnp.where(interior, first_shape, 2.0),
        jnp.where(interior, second_shape, 2.0),
        jnp.where(interior, beta_value, jnp.log(0.25) if log_mean else 0.25),
        log_mean,
    )
    log_beta = jnp.where(argument == 0, -jnp.inf, jnp.where(argument == 1, 0.0, log_beta))

    if log_mean:
        # At an underflowed Beta argument the continued fraction tends to one
        # Express its prefactor through the NB mass to retain the log probability and its derivatives
        underflow = argument == 0
        tail_count = jnp.where(underflow, jnp.where(reflected, count + 1, count), 1.0)
        tail_mean = jnp.where(underflow, safe_mean, 0.0)
        tail_concentration = jnp.where(underflow, safe_concentration, 1.0)
        log_prefactor = _negative_binomial_log_mean_log_mass(tail_count, tail_mean, tail_concentration)
        lower_adjustment = jax.nn.log_sigmoid(tail_mean - jnp.log(tail_concentration))
        lower_adjustment -= _log_concentration_fraction(tail_count, tail_concentration)
        log_prefactor += jnp.where(reflected, 0.0, lower_adjustment)
        log_beta = jnp.where(underflow, log_prefactor, log_beta)

    # Computing the opposite tail avoids subtracting a CDF rounded to one
    direct_tail = reflected == upper_tail
    complement_log_beta = jnp.where(direct_tail, -1.0, log_beta)
    log_probability = jnp.where(direct_tail, log_beta, jax.nn.log1mexp(-complement_log_beta))
    if not log_mean:
        log_probability = jnp.where(zero_count, zero_log_probability, log_probability)
    log_probability = jnp.where(value_array < 0, 0.0 if upper_tail else -jnp.inf, log_probability)
    log_probability = jnp.where(jnp.isposinf(value_array), -jnp.inf if upper_tail else 0.0, log_probability)
    return jnp.where(valid_parameters & ~jnp.isnan(value_array), log_probability, jnp.nan)


@jax.custom_jvp
def _negative_binomial_log_mass(
    count: jax.Array,
    mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    return _stable_negative_binomial_log_mass(count, jnp.log(mean), concentration)


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
    concentration_derivative = _negative_binomial_concentration_derivative(
        count,
        concentration,
        log_count_probability,
        log_concentration_probability,
        raw_log_ratio,
        ratio_difference,
    )

    log_mass_tangent = (
        count_derivative * count_tangent
        + mean_derivative * mean_tangent
        + concentration_derivative * concentration_tangent
    )
    return log_mass, log_mass_tangent


@jax.custom_jvp
def _negative_binomial_log_mean_log_mass(
    count: jax.Array,
    log_mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
    return _stable_negative_binomial_log_mass(count, log_mean, concentration)


@_negative_binomial_log_mean_log_mass.defjvp
def _negative_binomial_log_mean_log_mass_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    count, log_mean, concentration = primals
    count_tangent, log_mean_tangent, concentration_tangent = tangents
    log_mass = _negative_binomial_log_mean_log_mass(count, log_mean, concentration)

    log_concentration = jnp.log(concentration)
    log_parameter_sum = jnp.logaddexp(log_mean, log_concentration)
    log_count_probability = jax.nn.log_sigmoid(log_mean - log_concentration)
    log_concentration_probability = jax.nn.log_sigmoid(log_concentration - log_mean)
    count_probability = jnp.exp(log_count_probability)
    concentration_probability = jnp.exp(log_concentration_probability)

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

    log_count_mean_ratio = log_count - log_mean
    count_above_mean = positive_count & (log_count_mean_ratio > 0)
    positive_log_ratio = jnp.where(count_above_mean, log_count_mean_ratio, jnp.zeros_like(log_count_mean_ratio))
    negative_log_ratio = jnp.where(
        positive_count & ~count_above_mean,
        log_count_mean_ratio,
        jnp.zeros_like(log_count_mean_ratio),
    )

    # Log products recover finite derivatives when a tail probability flushes to zero
    direct_weighted_count = count * concentration_probability
    use_log_weighted_count = count_above_mean & (concentration_probability == 0)
    log_weighted_count = jnp.exp(
        jnp.where(
            use_log_weighted_count,
            log_count + log_concentration_probability,
            jnp.zeros_like(log_count),
        )
    )
    weighted_count = jnp.where(use_log_weighted_count, log_weighted_count, direct_weighted_count)

    direct_weighted_mean = concentration * count_probability
    use_log_weighted_mean = ~count_above_mean & (count_probability == 0)
    log_weighted_mean = jnp.exp(
        jnp.where(
            use_log_weighted_mean,
            log_concentration + log_count_probability,
            jnp.zeros_like(log_concentration),
        )
    )
    weighted_mean = jnp.where(use_log_weighted_mean, log_weighted_mean, direct_weighted_mean)

    positive_log_mean_derivative = weighted_count * -jnp.expm1(-positive_log_ratio)
    negative_log_mean_derivative = weighted_mean * jnp.expm1(negative_log_ratio)
    log_mean_derivative = jnp.where(
        positive_count,
        jnp.where(count_above_mean, positive_log_mean_derivative, negative_log_mean_derivative),
        -weighted_mean,
    )

    positive_ratio_log = jnp.where(
        count_above_mean,
        log_count - log_parameter_sum,
        jnp.zeros_like(log_count),
    )
    positive_ratio_difference = jnp.exp(positive_ratio_log) * -jnp.expm1(-positive_log_ratio)
    negative_ratio_difference = count_probability * jnp.expm1(negative_log_ratio)
    ratio_difference = jnp.where(
        positive_count,
        jnp.where(count_above_mean, positive_ratio_difference, negative_ratio_difference),
        -count_probability,
    )

    raw_log_ratio = log_concentration_probability - _log_concentration_fraction(count, concentration)
    concentration_derivative = _negative_binomial_concentration_derivative(
        count,
        concentration,
        log_count_probability,
        log_concentration_probability,
        raw_log_ratio,
        ratio_difference,
    )

    log_mass_tangent = (
        count_derivative * count_tangent
        + log_mean_derivative * log_mean_tangent
        + concentration_derivative * concentration_tangent
    )
    return log_mass, log_mass_tangent


def _stable_negative_binomial_log_mass(
    count: jax.Array,
    log_mean: jax.Array,
    concentration: jax.Array,
) -> jax.Array:
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
    probability_underflowed = log_concentration_probability == 0
    # Recover the first-order limit when the log probability rounds to zero before scaling
    recovered_log_mass = -jnp.exp(
        jnp.where(
            probability_underflowed,
            log_mean,
            jnp.zeros_like(log_mean),
        )
    )
    zero_count_log_mass = jnp.where(
        probability_underflowed,
        recovered_log_mass,
        zero_count_log_mass,
    )
    return jnp.where(positive_count, interior_log_mass, zero_count_log_mass)


def _negative_binomial_log_mean_rng(
    key: jax.Array,
    log_mean: jax.Array,
    concentration: jax.Array,
    *,
    output_shape: tuple[int, ...],
) -> jax.Array:
    gamma_key, poisson_key = jax.random.split(key)
    log_unit_rate = jax.random.loggamma(
        gamma_key,
        concentration,
        shape=output_shape,
        dtype=concentration.dtype,
    )
    # Scaling in log space keeps the Gamma-Poisson mixture stable in the tails
    latent_rate = jnp.exp(log_unit_rate + log_mean - jnp.log(concentration))
    return jax.random.poisson(
        poisson_key,
        latent_rate,
        shape=output_shape,
        dtype=jnp.int32,
    )


def _negative_binomial_concentration_derivative(
    count: jax.Array,
    concentration: jax.Array,
    log_count_probability: jax.Array,
    log_concentration_probability: jax.Array,
    raw_log_ratio: jax.Array,
    ratio_difference: jax.Array,
) -> jax.Array:
    use_ratio_difference = jnp.isfinite(ratio_difference) & (jnp.abs(ratio_difference) <= 0.5)
    # log1p preserves ratios near one while the log form retains far-tail detail
    stable_log_ratio = jnp.where(
        use_ratio_difference,
        jnp.log1p(jnp.where(use_ratio_difference, ratio_difference, jnp.zeros_like(ratio_difference))),
        raw_log_ratio,
    )

    total_count = count + concentration
    concentration_derivative = _weighted_log_ratio_deviance(
        jnp.ones_like(concentration),
        stable_log_ratio,
        ratio_difference,
    ) + (_gamma_shape_log_derivative(concentration) - _gamma_shape_log_derivative(total_count))

    positive_count = count > 0
    safe_count = jnp.where(positive_count, count, jnp.ones_like(count))
    log_count = jnp.log(safe_count)
    count_probability = jnp.exp(log_count_probability)
    # Shifting digamma past its pole avoids cancelling 1 / concentration terms
    small_concentration_derivative = (
        digamma(total_count)
        - digamma(concentration + 1)
        + log_concentration_probability
        + count_probability
        - jnp.expm1(log_count + log_concentration_probability) / concentration
    )
    small_concentration_derivative = jnp.where(
        positive_count,
        small_concentration_derivative,
        log_concentration_probability + count_probability,
    )
    return jnp.where(
        concentration < 1,
        small_concentration_derivative,
        concentration_derivative,
    )


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
