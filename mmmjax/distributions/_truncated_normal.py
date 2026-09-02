"""Truncated Normal distribution functions."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, log_ndtr
from jax.typing import ArrayLike

from mmmjax.distributions._normal import _normal_logpdf_kernel, _standardize
from mmmjax.distributions._utils import _promote_inexact


def truncated_normal_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Truncated Normal log density elementwise.

    For value :math:`a \leq x \leq b`, location
    :math:`\mu \in \mathbb{R}`, and scale :math:`\sigma > 0`, the log density
    is

    .. math::

        \log p(x \mid \mu, \sigma, a, b)
        = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2
          - \log(\sigma)
          - \frac{1}{2}\log(2\pi)
          - \log\left[
              \Phi\left(\frac{b - \mu}{\sigma}\right)
              - \Phi\left(\frac{a - \mu}{\sigma}\right)
            \right],

    where :math:`a < b` and :math:`\Phi` is the standard Normal cumulative
    distribution function. The density is zero outside the truncation bounds.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the underlying Normal distribution.
    scale
        Positive standard deviation of the underlying Normal distribution.
    lower
        Lower truncation bound. May be ``-inf``.
    upper
        Upper truncation bound. May be ``inf``.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values outside the bounds produce ``-inf``. Invalid distribution
        parameters produce ``nan``.
    """
    value_array, location_array, scale_array, lower_array, upper_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
        ("lower", lower),
        ("upper", upper),
    )

    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    valid_bounds = ~jnp.isnan(lower_array) & ~jnp.isnan(upper_array) & (lower_array < upper_array)
    valid_parameters = valid_location & valid_scale & valid_bounds

    safe_location = jnp.where(valid_location, location_array, jnp.zeros_like(location_array))
    safe_scale = jnp.where(valid_scale, scale_array, jnp.ones_like(scale_array))
    safe_lower = jnp.where(valid_bounds, lower_array, -jnp.ones_like(lower_array))
    safe_upper = jnp.where(valid_bounds, upper_array, jnp.ones_like(upper_array))

    standardized_lower = _standardize_bound(safe_lower, safe_location, safe_scale)
    standardized_upper = _standardize_bound(safe_upper, safe_location, safe_scale)
    finite_bounds = jnp.isfinite(safe_lower) & jnp.isfinite(safe_upper)
    width_lower = jnp.where(finite_bounds, safe_lower, jnp.zeros_like(safe_lower))
    width_upper = jnp.where(finite_bounds, safe_upper, jnp.ones_like(safe_upper))
    standardized_width = _standardize(width_upper, width_lower, safe_scale)
    standardized_width = jnp.where(finite_bounds, standardized_width, jnp.inf)
    log_normalizing_mass = _normal_log_mass(
        standardized_lower,
        standardized_upper,
        standardized_width,
    )
    log_density = _normal_logpdf_kernel(value_array, safe_location, safe_scale) - log_normalizing_mass

    outside_support = (value_array < safe_lower) | (value_array > safe_upper)
    supported_log_density = jnp.where(outside_support, -jnp.inf, log_density)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def truncated_normal(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Truncated Normal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the underlying Normal distribution.
    scale
        Positive standard deviation of the underlying Normal distribution.
    lower
        Lower truncation bound. May be ``-inf``.
    upper
        Upper truncation bound. May be ``inf``.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    return jnp.sum(truncated_normal_logpdf(value, location, scale, lower, upper))


def _standardize_bound(
    bound: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    finite_bound = jnp.isfinite(bound)
    safe_bound = jnp.where(finite_bound, bound, location)
    standardized = _standardize(safe_bound, location, scale)

    # Keeping infinite bounds constant avoids undefined inf times zero terms in gradients
    return jnp.where(finite_bound, standardized, bound)


@jax.custom_jvp
def _normal_log_mass(
    lower: jax.Array,
    upper: jax.Array,
    width: jax.Array,
) -> jax.Array:
    central_interval = (lower <= 0) & (upper >= 0)
    right_interval = lower > 0

    # Reflect the right tail so log_ndtr only sees stable left-tail probabilities
    tail_lower = jnp.where(right_interval, -upper, lower)
    tail_upper = jnp.where(right_interval, -lower, upper)
    safe_tail_lower = jnp.where(central_interval, -jnp.ones_like(tail_lower), tail_lower)
    safe_tail_upper = jnp.where(central_interval, jnp.zeros_like(tail_upper), tail_upper)
    lower_logcdf = log_ndtr(safe_tail_lower)
    upper_logcdf = log_ndtr(safe_tail_upper)
    tail_log_mass = upper_logcdf + jnp.log(-jnp.expm1(lower_logcdf - upper_logcdf))

    # Use a midpoint series when nearby log CDFs cannot retain enough difference
    finite_interval = jnp.isfinite(lower) & jnp.isfinite(upper) & jnp.isfinite(width) & (width > 0)
    safe_width = jnp.where(finite_interval, width, jnp.ones_like(width))
    midpoint = 0.5 * jnp.where(finite_interval, lower, jnp.zeros_like(lower)) + 0.5 * jnp.where(
        finite_interval,
        upper,
        jnp.zeros_like(upper),
    )
    half_width = 0.5 * safe_width
    midpoint_span = midpoint * half_width
    squared_half_width = jnp.square(half_width)
    squared_midpoint_span = jnp.square(midpoint_span)
    fourth_order = (
        jnp.square(squared_midpoint_span)
        - 6 * squared_midpoint_span * squared_half_width
        + 3 * jnp.square(squared_half_width)
    ) / 120
    sixth_order = (
        squared_midpoint_span**3
        - 15 * jnp.square(squared_midpoint_span) * squared_half_width
        + 45 * squared_midpoint_span * jnp.square(squared_half_width)
        - 15 * squared_half_width**3
    ) / 5040
    interval_correction = 1 + (squared_midpoint_span - squared_half_width) / 6 + fourth_order + sixth_order
    narrow_log_mass = (
        _normal_logpdf_kernel(midpoint, jnp.zeros_like(midpoint), jnp.ones_like(midpoint))
        + jnp.log(safe_width)
        + jnp.log(interval_correction)
    )
    expansion_size = jnp.abs(midpoint_span) + 0.5 * squared_half_width
    # The first omitted even term is eighth order, so this cutoff tracks machine precision
    expansion_limit = jnp.asarray(np.finfo(lower.dtype).eps ** (1 / 8), dtype=lower.dtype)
    narrow_interval = ~central_interval & finite_interval & (expansion_size <= expansion_limit)
    same_side_log_mass = jnp.where(narrow_interval, narrow_log_mass, tail_log_mass)

    # Across zero, the interval mass is an erf sum with no subtraction
    sqrt_two = jnp.sqrt(jnp.asarray(2, dtype=lower.dtype))
    central_lower = jnp.where(central_interval, -lower, jnp.ones_like(lower))
    central_upper = jnp.where(central_interval, upper, jnp.ones_like(upper))
    central_mass = erf(central_lower / sqrt_two) + erf(central_upper / sqrt_two)
    central_log_mass = jnp.log(central_mass) - jnp.log(jnp.asarray(2, dtype=lower.dtype))

    return jnp.where(central_interval, central_log_mass, same_side_log_mass)


@_normal_log_mass.defjvp
def _normal_log_mass_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    lower, upper, width = primals
    lower_tangent, upper_tangent, _ = tangents
    log_mass = _normal_log_mass(lower, upper, width)

    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    safe_lower = jnp.where(finite_lower, lower, jnp.zeros_like(lower))
    safe_upper = jnp.where(finite_upper, upper, jnp.zeros_like(upper))
    zero = jnp.zeros_like(log_mass)
    one = jnp.ones_like(log_mass)

    lower_logpdf = _normal_logpdf_kernel(safe_lower, zero, one)
    upper_logpdf = _normal_logpdf_kernel(safe_upper, zero, one)
    lower_weight = jnp.where(finite_lower, jnp.exp(lower_logpdf - log_mass), zero)
    upper_weight = jnp.where(finite_upper, jnp.exp(upper_logpdf - log_mass), zero)

    log_mass_tangent = -lower_weight * lower_tangent + upper_weight * upper_tangent
    return log_mass, log_mass_tangent
