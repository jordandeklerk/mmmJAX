"""Uniform distribution functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def uniform_logpdf(
    value: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Uniform log density elementwise.

    For value :math:`x \in \mathbb{R}` and finite bounds :math:`a < b`, the
    log density is

    .. math::

        \log p(x \mid a, b)
        = \begin{cases}
            -\log(b - a), & a \le x \le b, \\
            -\infty, & \text{otherwise}.
          \end{cases}

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    lower
        Finite lower bounds.
    upper
        Finite upper bounds greater than ``lower``.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values outside the bounds produce ``-inf``. Nonfinite bounds or bounds
        where ``lower >= upper`` produce ``nan``. A ``nan`` value also produces
        ``nan``.
    """
    value_array, lower_array, upper_array = _promote_inexact(
        ("value", value),
        ("lower", lower),
        ("upper", upper),
    )

    valid_bounds = jnp.isfinite(lower_array) & jnp.isfinite(upper_array) & (lower_array < upper_array)
    log_width = _log_difference(upper_array, lower_array, valid_bounds)

    outside_support = (value_array < lower_array) | (value_array > upper_array)
    supported_log_density = jnp.where(outside_support, -jnp.inf, -log_width)
    supported_log_density = jnp.where(jnp.isnan(value_array), jnp.nan, supported_log_density)
    return jnp.where(valid_bounds, supported_log_density, jnp.nan)


def uniform(
    value: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Uniform log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    lower
        Finite lower bounds.
    upper
        Finite upper bounds greater than ``lower``.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_densities = uniform_logpdf(value, lower, upper)
    log_density = jnp.sum(log_densities)

    # Only empty results need a separate check because no element can carry nan into the sum
    if log_densities.size:
        return log_density

    lower_array, upper_array = _promote_inexact(("lower", lower), ("upper", upper))
    finite_bounds = jnp.all(jnp.isfinite(lower_array)) & jnp.all(jnp.isfinite(upper_array))
    valid_bounds = finite_bounds & jnp.all(lower_array < upper_array)
    return jnp.where(valid_bounds, log_density, jnp.nan)


def uniform_logcdf(
    value: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Uniform log cumulative distribution function elementwise.

    For finite bounds :math:`a < b`, the log cumulative probability is

    .. math::

        \log F(x \mid a, b)
        = \begin{cases}
            -\infty, & x \leq a, \\
            \log(x - a) - \log(b - a), & a < x < b, \\
            0, & x \geq b.
          \end{cases}

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    lower
        Finite lower bounds.
    upper
        Finite upper bounds greater than ``lower``.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. Nonfinite bounds or bounds where ``lower >= upper`` produce
        ``nan``. A ``nan`` value also produces ``nan``.
    """
    value_array, lower_array, upper_array = _promote_inexact(
        ("value", value),
        ("lower", lower),
        ("upper", upper),
    )

    valid_bounds = jnp.isfinite(lower_array) & jnp.isfinite(upper_array) & (lower_array < upper_array)
    inside_support = (
        valid_bounds & jnp.isfinite(value_array) & (value_array > lower_array) & (value_array < upper_array)
    )

    log_left_distance = _log_difference(value_array, lower_array, inside_support)
    log_right_distance = _log_difference(upper_array, value_array, inside_support)
    interior_log_cdf = jax.nn.log_sigmoid(log_left_distance - log_right_distance)

    boundary_log_cdf = jnp.where(value_array <= lower_array, -jnp.inf, 0)
    boundary_log_cdf = jnp.where(jnp.isnan(value_array), jnp.nan, boundary_log_cdf)
    supported_log_cdf = jnp.where(inside_support, interior_log_cdf, boundary_log_cdf)
    return jnp.where(valid_bounds, supported_log_cdf, jnp.nan)


def uniform_logsf(
    value: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Uniform log survival function elementwise.

    For finite bounds :math:`a < b`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid a, b)
        = \begin{cases}
            0, & x \leq a, \\
            \log(b - x) - \log(b - a), & a < x < b, \\
            -\infty, & x \geq b.
          \end{cases}

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    lower
        Finite lower bounds.
    upper
        Finite upper bounds greater than ``lower``.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        Nonfinite bounds or bounds where ``lower >= upper`` produce ``nan``. A
        ``nan`` value also produces ``nan``.
    """
    value_array, lower_array, upper_array = _promote_inexact(
        ("value", value),
        ("lower", lower),
        ("upper", upper),
    )

    valid_bounds = jnp.isfinite(lower_array) & jnp.isfinite(upper_array) & (lower_array < upper_array)
    inside_support = (
        valid_bounds & jnp.isfinite(value_array) & (value_array > lower_array) & (value_array < upper_array)
    )

    log_left_distance = _log_difference(value_array, lower_array, inside_support)
    log_right_distance = _log_difference(upper_array, value_array, inside_support)
    interior_log_survival = jax.nn.log_sigmoid(log_right_distance - log_left_distance)

    boundary_log_survival = jnp.where(value_array <= lower_array, 0, -jnp.inf)
    boundary_log_survival = jnp.where(jnp.isnan(value_array), jnp.nan, boundary_log_survival)
    supported_log_survival = jnp.where(inside_support, interior_log_survival, boundary_log_survival)
    return jnp.where(valid_bounds, supported_log_survival, jnp.nan)


def uniform_rng(
    key: jax.Array,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Uniform distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    lower
        Finite lower bounds.
    upper
        Finite upper bounds greater than ``lower``.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. The
        lower bound is inclusive and the upper bound is exclusive. Invalid
        bounds produce ``nan``.
    """
    lower_array, upper_array = _promote_inexact(("lower", lower), ("upper", upper))
    output_shape = _random_shape(sample_shape, lower_array, upper_array)

    valid_bounds = jnp.isfinite(lower_array) & jnp.isfinite(upper_array) & (lower_array < upper_array)
    safe_lower = jnp.where(valid_bounds, lower_array, jnp.zeros_like(lower_array))
    safe_upper = jnp.where(valid_bounds, upper_array, jnp.ones_like(upper_array))

    unit_samples = jax.random.uniform(key, shape=output_shape, dtype=lower_array.dtype)
    crosses_zero = (safe_lower < 0) & (safe_upper > 0)

    direct_lower = jnp.where(crosses_zero, jnp.zeros_like(safe_lower), safe_lower)
    direct_upper = jnp.where(crosses_zero, jnp.ones_like(safe_upper), safe_upper)
    direct_samples = direct_lower + unit_samples * (direct_upper - direct_lower)

    cross_zero_lower = jnp.where(crosses_zero, safe_lower, jnp.zeros_like(safe_lower))
    cross_zero_upper = jnp.where(crosses_zero, safe_upper, jnp.ones_like(safe_upper))
    # Convex interpolation stays finite when valid bounds span the dtype range
    cross_zero_samples = (1 - unit_samples) * cross_zero_lower + unit_samples * cross_zero_upper
    samples = jnp.where(crosses_zero, cross_zero_samples, direct_samples)
    samples = jnp.maximum(samples, safe_lower)
    # Keep the rounding guard out of pathwise gradients through the bounds
    upper_limit = jnp.nextafter(
        jax.lax.stop_gradient(safe_upper),
        jax.lax.stop_gradient(safe_lower),
    )
    samples = jnp.minimum(samples, upper_limit)

    return jnp.where(valid_bounds, samples, jnp.nan)


def _log_difference(
    upper: jax.Array,
    lower: jax.Array,
    valid_interval: jax.Array,
) -> jax.Array:
    crosses_zero = valid_interval & (lower < 0) & (upper > 0)

    # Splitting intervals that cross zero avoids subtracting opposite dtype extremes
    direct_lower = jnp.where(valid_interval & ~crosses_zero, lower, 0)
    direct_upper = jnp.where(valid_interval & ~crosses_zero, upper, 1)
    direct_log_difference = jnp.log(direct_upper - direct_lower)

    negative_lower = jnp.where(crosses_zero, -lower, 1)
    positive_upper = jnp.where(crosses_zero, upper, 1)
    cross_zero_log_difference = jnp.logaddexp(jnp.log(negative_lower), jnp.log(positive_upper))
    return jnp.where(crosses_zero, cross_zero_log_difference, direct_log_difference)
