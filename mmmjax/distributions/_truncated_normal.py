"""Truncated Normal distribution functions."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import truncnorm as jax_truncnorm
from jax.typing import ArrayLike

from mmmjax.distributions._normal import (
    _normal_logcdf_kernel,
    _normal_logpdf_kernel,
    _normal_logsf_kernel,
    _standardize,
)
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

    outside_support = (value_array < safe_lower) | (value_array > safe_upper)
    safe_value = jnp.where(outside_support, safe_location, value_array)

    # Keep infinite bounds out of JAX's two-sided calculation so their gradients stay zero
    finite_lower = jnp.isfinite(safe_lower)
    finite_upper = jnp.isfinite(safe_upper)
    two_sided = finite_lower & finite_upper

    two_sided_lower = jnp.where(two_sided, safe_lower, -jnp.ones_like(safe_lower))
    two_sided_upper = jnp.where(two_sided, safe_upper, jnp.ones_like(safe_upper))
    standardized_lower = _standardize(two_sided_lower, safe_location, safe_scale)
    standardized_upper = _standardize(two_sided_upper, safe_location, safe_scale)
    two_sided_log_density = jax_truncnorm.logpdf(
        safe_value,
        standardized_lower,
        standardized_upper,
        loc=safe_location,
        scale=safe_scale,
    )

    normal_log_density = _normal_logpdf_kernel(safe_value, safe_location, safe_scale)
    lower_log_density = normal_log_density - _normal_logsf_kernel(safe_lower, safe_location, safe_scale)
    upper_log_density = normal_log_density - _normal_logcdf_kernel(safe_upper, safe_location, safe_scale)
    log_density = jnp.where(
        two_sided,
        two_sided_log_density,
        jnp.where(
            finite_lower,
            lower_log_density,
            jnp.where(finite_upper, upper_log_density, normal_log_density),
        ),
    )

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
