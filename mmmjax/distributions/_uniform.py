"""Uniform distribution functions."""

from typing import cast

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
    return jnp.sum(uniform_logpdf(value, lower, upper))


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

    interior_log_cdf = _uniform_interior_log_probability(
        value_array,
        lower_array,
        upper_array,
        inside_support,
        survival=False,
    )

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

    interior_log_survival = _uniform_interior_log_probability(
        value_array,
        lower_array,
        upper_array,
        inside_support,
        survival=True,
    )

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

    unit_samples = jax.random.uniform(key, shape=output_shape, dtype=lower_array.dtype)
    crosses_zero = (lower_array < 0) & (upper_array > 0)

    direct_lower = jnp.where(crosses_zero, jnp.zeros_like(lower_array), lower_array)
    direct_upper = jnp.where(crosses_zero, jnp.ones_like(upper_array), upper_array)
    direct_samples = direct_lower + unit_samples * (direct_upper - direct_lower)

    cross_zero_lower = jnp.where(crosses_zero, lower_array, jnp.zeros_like(lower_array))
    cross_zero_upper = jnp.where(crosses_zero, upper_array, jnp.ones_like(upper_array))
    # Convex interpolation stays finite when valid bounds span the dtype range
    cross_zero_samples = (1 - unit_samples) * cross_zero_lower + unit_samples * cross_zero_upper
    samples = jnp.where(crosses_zero, cross_zero_samples, direct_samples)
    samples = jnp.maximum(samples, lower_array)
    # Keep the rounding guard out of pathwise gradients through the bounds
    upper_limit = jnp.nextafter(
        jax.lax.stop_gradient(upper_array),
        jax.lax.stop_gradient(lower_array),
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


def _uniform_interior_log_probability(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    inside_support: jax.Array,
    *,
    survival: bool,
) -> jax.Array:
    width = upper - lower

    def finite_width_probability(_: None) -> jax.Array:
        safe_value = jnp.where(inside_support, value, jnp.zeros_like(value))
        safe_lower = jnp.where(inside_support, lower, -jnp.ones_like(lower))
        safe_upper = jnp.where(inside_support, upper, jnp.ones_like(upper))
        safe_width = jnp.where(inside_support, width, 2)

        left_distance = safe_value - safe_lower
        right_distance = safe_upper - safe_value
        direct_distance = right_distance if survival else left_distance
        complement_distance = left_distance if survival else right_distance

        use_direct_probability = direct_distance <= complement_distance
        safe_direct_distance = jnp.where(use_direct_probability, direct_distance, jnp.ones_like(direct_distance))
        safe_complement_distance = jnp.where(
            use_direct_probability,
            jnp.zeros_like(complement_distance),
            complement_distance,
        )
        direct_log_probability = jnp.log(safe_direct_distance) - jnp.log(safe_width)
        complement_log_probability = jnp.log1p(-safe_complement_distance / safe_width)
        return jnp.where(
            use_direct_probability,
            direct_log_probability,
            complement_log_probability,
        )

    def log_space_probability(_: None) -> jax.Array:
        log_left_distance = _log_difference(value, lower, inside_support)
        log_right_distance = _log_difference(upper, value, inside_support)
        log_odds = log_right_distance - log_left_distance if survival else log_left_distance - log_right_distance
        return jax.nn.log_sigmoid(log_odds)

    # Ordinary finite widths use fewer operations while extreme intervals keep the log-space path
    return cast(
        jax.Array,
        jax.lax.cond(
            jnp.all(jnp.isfinite(width)),
            finite_width_probability,
            log_space_probability,
            operand=None,
        ),
    )
