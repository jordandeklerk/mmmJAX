"""Cauchy distribution functions."""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.distributions._utils import _promote_inexact, _random_shape


def cauchy_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Cauchy log density elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \mu, \sigma)
        = -\log(\pi) - \log(\sigma)
          - \log\left[1 + \left(\frac{x - \mu}{\sigma}\right)^2\right],
        \qquad \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments. A
        nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )

    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)

    residual = value_array - location_array
    subtraction_overflowed = jnp.isinf(residual) & jnp.isfinite(value_array)
    central = ~subtraction_overflowed & (jnp.abs(residual) <= scale_array)

    direct_tail = ~central & ~subtraction_overflowed
    direct_residual = jnp.where(direct_tail, jnp.abs(residual), jnp.ones_like(residual))
    direct_log_residual = jnp.log(direct_residual)

    # Halving both parts keeps a finite subtraction in range and only changes its log by log(2)
    half = jnp.asarray(0.5, dtype=value_array.dtype)
    overflow_value = jnp.where(subtraction_overflowed, value_array, jnp.zeros_like(value_array))
    overflow_location = jnp.where(subtraction_overflowed, location_array, jnp.zeros_like(location_array))
    halved_residual = half * overflow_value - half * overflow_location
    safe_halved_residual = jnp.where(
        subtraction_overflowed,
        jnp.abs(halved_residual),
        jnp.ones_like(halved_residual),
    )
    overflow_log_residual = jnp.log(safe_halved_residual) + jnp.log(jnp.asarray(2, dtype=value_array.dtype))

    log_residual = jnp.where(
        subtraction_overflowed,
        overflow_log_residual,
        direct_log_residual,
    )

    log_scale = jnp.log(scale_array)

    central_residual = jnp.where(central, residual, jnp.zeros_like(residual))
    central_scale = jnp.where(central, scale_array, jnp.ones_like(scale_array))
    standardized = _standardize(central_residual, central_scale)
    central_kernel = jnp.log1p(jnp.square(standardized))

    # This is SciPy's reciprocal tail form written without first forming a large ratio
    tail_log_residual = jnp.where(central, log_scale, log_residual)
    tail_log_ratio = log_scale - tail_log_residual
    inverse_standardized = jnp.exp(tail_log_ratio)
    tail_kernel = 2 * (tail_log_residual - log_scale) + jnp.log1p(jnp.square(inverse_standardized))

    log_pi = jnp.asarray(math.log(math.pi), dtype=value_array.dtype)
    log_density = -log_pi - log_scale - jnp.where(central, central_kernel, tail_kernel)
    return jnp.where(valid_location & valid_scale, log_density, jnp.nan)


def cauchy(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Cauchy log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    return jnp.sum(cauchy_logpdf(value, location, scale))


def cauchy_logcdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Cauchy log cumulative distribution function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log cumulative probability is

    .. math::

        \log F(x \mid \mu, \sigma)
        = \log\left[
            \frac{1}{2}
            + \frac{1}{\pi}\arctan\left(\frac{x - \mu}{\sigma}\right)
          \right].

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log cumulative probabilities with the broadcast shape of the
        arguments. A nonfinite location or a nonpositive or nonfinite scale
        produces ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )
    return _cauchy_logcdf_kernel(value_array, location_array, scale_array)


def cauchy_logsf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Cauchy log survival function elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \mu, \sigma)
        = \log\left[
            \frac{1}{2}
            - \frac{1}{\pi}\arctan\left(\frac{x - \mu}{\sigma}\right)
          \right].

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
    location
        Location of the distribution.
    scale
        Positive scale parameter.

    Returns
    -------
    jax.Array
        Log survival probabilities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )
    return _cauchy_logcdf_kernel(-value_array, -location_array, scale_array)


def cauchy_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Cauchy distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Location of the distribution.
    scale
        Positive scale parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    location_array, scale_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
    )
    output_shape = _random_shape(sample_shape, location_array, scale_array)

    standard_samples = jax.random.cauchy(key, shape=output_shape, dtype=location_array.dtype)
    samples = location_array + scale_array * standard_samples

    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)


def _cauchy_logcdf_kernel(
    value: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    valid_parameters = jnp.isfinite(location) & jnp.isfinite(scale) & (scale > 0)
    infinite_value = jnp.isinf(value)
    evaluate_probability = valid_parameters & ~infinite_value
    safe_value = jnp.where(evaluate_probability, value, jnp.zeros_like(value))
    safe_location = jnp.where(evaluate_probability, location, jnp.zeros_like(location))
    safe_scale = jnp.where(evaluate_probability, scale, jnp.ones_like(scale))

    residual = safe_value - safe_location
    subtraction_overflowed = jnp.isinf(residual) & jnp.isfinite(value) & jnp.isfinite(location)

    # Halving the operands recovers the magnitude when the direct subtraction overflows
    half = jnp.asarray(0.5, dtype=value.dtype)
    halved_residual = half * safe_value - half * safe_location
    safe_residual = jnp.where(
        subtraction_overflowed,
        jnp.ones_like(residual),
        residual,
    )
    absolute_residual = jnp.abs(safe_residual)
    near_location = ~subtraction_overflowed & (absolute_residual <= safe_scale)

    near_residual = jnp.where(
        near_location,
        absolute_residual,
        jnp.zeros_like(absolute_residual),
    )
    near_scale = jnp.where(near_location, safe_scale, jnp.ones_like(safe_scale))
    standardized_distance = _standardize(near_residual, near_scale)
    log_pi = jnp.asarray(math.log(math.pi), dtype=value.dtype)
    near_log_tail = jnp.log(jnp.arctan2(jnp.ones_like(standardized_distance), standardized_distance)) - log_pi

    far_log_residual = jnp.log(
        jnp.where(
            near_location | subtraction_overflowed,
            jnp.ones_like(absolute_residual),
            absolute_residual,
        )
    )
    overflow_log_residual = jnp.log(
        jnp.where(
            subtraction_overflowed,
            jnp.abs(halved_residual),
            jnp.ones_like(halved_residual),
        )
    ) + jnp.asarray(math.log(2), dtype=value.dtype)
    far_log_residual = jnp.where(
        subtraction_overflowed,
        overflow_log_residual,
        far_log_residual,
    )

    far_scale = jnp.where(near_location, jnp.ones_like(safe_scale), safe_scale)
    log_scale_ratio = jnp.log(far_scale) - far_log_residual
    scale_ratio = jnp.exp(log_scale_ratio)
    precision = jnp.asarray(jnp.finfo(scale_ratio.dtype).eps, dtype=scale_ratio.dtype)
    needs_correction = scale_ratio >= jnp.sqrt(precision)
    safe_ratio = jnp.where(needs_correction, scale_ratio, jnp.ones_like(scale_ratio))

    # Below sqrt(eps), the correction and its derivatives are below working precision
    correction = jnp.where(
        needs_correction,
        jnp.log(jnp.arctan(safe_ratio) / safe_ratio),
        jnp.zeros_like(scale_ratio),
    )
    far_log_tail = log_scale_ratio + correction - log_pi
    log_tail = jnp.where(near_location, near_log_tail, far_log_tail)

    lower_tail = residual < 0
    log_probability = jnp.where(
        lower_tail,
        log_tail,
        jnp.log1p(-jnp.exp(log_tail)),
    )

    endpoint_probability = jnp.where(value > 0, jnp.zeros_like(value), -jnp.inf)
    supported_log_probability = jnp.where(infinite_value, endpoint_probability, log_probability)
    return jnp.where(valid_parameters, supported_log_probability, jnp.nan)


@jax.custom_jvp
def _standardize(residual: jax.Array, scale: jax.Array) -> jax.Array:
    return residual / scale


@_standardize.defjvp
def _standardize_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    residual, scale = primals
    residual_tangent, scale_tangent = tangents
    standardized = _standardize(residual, scale)

    # The quotient rule avoids forming scale squared in transformed gradients
    standardized_tangent = (residual_tangent - standardized * scale_tangent) / scale
    return standardized, standardized_tangent
