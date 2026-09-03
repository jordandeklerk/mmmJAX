"""Truncated Normal distribution functions."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, ndtr, ndtri
from jax.scipy.stats import truncnorm as jax_truncnorm
from jax.typing import ArrayLike

from mmmjax.distributions._normal import (
    _normal_logcdf_kernel,
    _normal_logpdf_kernel,
    _normal_logsf_kernel,
    _standardize,
)
from mmmjax.distributions._utils import _promote_inexact, _random_shape


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

    two_sided_active = valid_parameters & two_sided & ~outside_support & ~jnp.isnan(value_array)
    two_sided_value = jnp.where(two_sided_active, safe_value, jnp.zeros_like(safe_value))
    two_sided_location = jnp.where(two_sided_active, safe_location, jnp.zeros_like(safe_location))
    two_sided_scale = jnp.where(two_sided_active, safe_scale, jnp.ones_like(safe_scale))
    two_sided_lower = jnp.where(two_sided_active, safe_lower, -jnp.ones_like(safe_lower))
    two_sided_upper = jnp.where(two_sided_active, safe_upper, jnp.ones_like(safe_upper))
    standardized_value = _standardize(two_sided_value, two_sided_location, two_sided_scale)
    standardized_lower = _standardize(two_sided_lower, two_sided_location, two_sided_scale)
    standardized_upper = _standardize(two_sided_upper, two_sided_location, two_sided_scale)
    two_sided_log_density = _standard_logpdf(
        standardized_value,
        standardized_lower,
        standardized_upper,
    ) - jnp.log(two_sided_scale)

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
    supported_log_density = jnp.where(jnp.isnan(value_array), jnp.nan, supported_log_density)
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


def truncated_normal_logcdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Truncated Normal log cumulative distribution function elementwise.

    For value :math:`a \leq x \leq b`, location
    :math:`\mu \in \mathbb{R}`, scale :math:`\sigma > 0`, and standardized
    values :math:`z=(x-\mu)/\sigma`, :math:`\alpha=(a-\mu)/\sigma`, and
    :math:`\beta=(b-\mu)/\sigma`, the log cumulative probability is

    .. math::

        \log F(x \mid \mu, \sigma, a, b)
        = \log\left[
            \frac{\Phi(z)-\Phi(\alpha)}
                 {\Phi(\beta)-\Phi(\alpha)}
          \right],

    where :math:`a < b` and :math:`\Phi` is the standard Normal cumulative
    distribution function.

    Parameters
    ----------
    value
        Values at which to evaluate the cumulative probability.
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
        Log cumulative probabilities with the broadcast shape of the
        arguments. Invalid distribution parameters produce ``nan``.
    """
    return _truncated_normal_log_probability(
        value,
        location,
        scale,
        lower,
        upper,
        upper_tail=False,
    )


def truncated_normal_logsf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Truncated Normal log survival function elementwise.

    For value :math:`a \leq x \leq b`, location
    :math:`\mu \in \mathbb{R}`, scale :math:`\sigma > 0`, and standardized
    values :math:`z=(x-\mu)/\sigma`, :math:`\alpha=(a-\mu)/\sigma`, and
    :math:`\beta=(b-\mu)/\sigma`, the log survival probability is

    .. math::

        \log \overline{F}(x \mid \mu, \sigma, a, b)
        = \log\left[
            \frac{\Phi(\beta)-\Phi(z)}
                 {\Phi(\beta)-\Phi(\alpha)}
          \right],

    where :math:`a < b` and :math:`\Phi` is the standard Normal cumulative
    distribution function.

    Parameters
    ----------
    value
        Values at which to evaluate the survival probability.
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
        Log survival probabilities with the broadcast shape of the arguments.
        Invalid distribution parameters produce ``nan``.
    """
    return _truncated_normal_log_probability(
        value,
        location,
        scale,
        lower,
        upper,
        upper_tail=True,
    )


def truncated_normal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Truncated Normal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Location of the underlying Normal distribution.
    scale
        Positive standard deviation of the underlying Normal distribution.
    lower
        Lower truncation bound. May be ``-inf``.
    upper
        Upper truncation bound. May be ``inf``.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. Invalid
        distribution parameters produce ``nan``.
    """
    location_array, scale_array, lower_array, upper_array = _promote_inexact(
        ("location", location),
        ("scale", scale),
        ("lower", lower),
        ("upper", upper),
    )
    output_shape = _random_shape(
        sample_shape,
        location_array,
        scale_array,
        lower_array,
        upper_array,
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
    log_mass = _normal_log_mass(standardized_lower, standardized_upper)

    minimum_probability = jnp.asarray(np.finfo(location_array.dtype).tiny, dtype=location_array.dtype)
    unit_samples = jax.random.uniform(
        key,
        shape=output_shape,
        dtype=location_array.dtype,
        minval=minimum_probability,
    )
    left_log_probability = jnp.logaddexp(
        log_ndtr(standardized_lower),
        jnp.log(unit_samples) + log_mass,
    )
    right_log_probability = jnp.logaddexp(
        log_ndtr(-standardized_upper),
        jnp.log1p(-unit_samples) + log_mass,
    )
    use_left_probability = standardized_lower < 0
    log_probability = jnp.where(use_left_probability, left_log_probability, right_log_probability)
    reflected_samples = _inverse_normal_logcdf(log_probability)
    standardized_samples = jnp.where(use_left_probability, reflected_samples, -reflected_samples)
    samples = safe_location + safe_scale * standardized_samples

    # Rounding in the inverse transform must not move finite draws onto a bound
    lower_limit = jnp.nextafter(
        jax.lax.stop_gradient(safe_lower),
        jax.lax.stop_gradient(safe_upper),
    )
    upper_limit = jnp.nextafter(
        jax.lax.stop_gradient(safe_upper),
        jax.lax.stop_gradient(safe_lower),
    )
    samples = jnp.where(samples <= safe_lower, lower_limit, samples)
    samples = jnp.where(samples >= safe_upper, upper_limit, samples)
    return jnp.where(valid_parameters, samples, jnp.nan)


def _truncated_normal_log_probability(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    upper_tail: bool,
) -> jax.Array:
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

    interior = (value_array > safe_lower) & (value_array < safe_upper) & ~jnp.isnan(value_array)

    active = valid_parameters & interior
    active_value = jnp.where(active, value_array, jnp.zeros_like(value_array))
    active_location = jnp.where(active, safe_location, jnp.zeros_like(safe_location))
    active_scale = jnp.where(active, safe_scale, jnp.ones_like(safe_scale))
    active_lower = jnp.where(active, safe_lower, -jnp.ones_like(safe_lower))
    active_upper = jnp.where(active, safe_upper, jnp.ones_like(safe_upper))
    standardized_value = _standardize(active_value, active_location, active_scale)
    standardized_lower = _standardize_bound(active_lower, active_location, active_scale)
    standardized_upper = _standardize_bound(active_upper, active_location, active_scale)
    if upper_tail:
        interior_log_probability = _standard_logsf(
            standardized_value,
            standardized_lower,
            standardized_upper,
        )
    else:
        interior_log_probability = _standard_logcdf(
            standardized_value,
            standardized_lower,
            standardized_upper,
        )

    untruncated = ~jnp.isfinite(safe_lower) & ~jnp.isfinite(safe_upper)
    normal_log_probability = (
        _normal_logsf_kernel(value_array, safe_location, safe_scale)
        if upper_tail
        else _normal_logcdf_kernel(value_array, safe_location, safe_scale)
    )
    interior_log_probability = jnp.where(
        untruncated,
        normal_log_probability,
        interior_log_probability,
    )

    below_probability = jnp.zeros_like(value_array) if upper_tail else jnp.full_like(value_array, -jnp.inf)
    above_probability = jnp.full_like(value_array, -jnp.inf) if upper_tail else jnp.zeros_like(value_array)
    supported_log_probability = jnp.where(
        value_array <= safe_lower,
        below_probability,
        jnp.where(value_array >= safe_upper, above_probability, interior_log_probability),
    )
    supported_log_probability = jnp.where(jnp.isnan(value_array), jnp.nan, supported_log_probability)
    return jnp.where(valid_parameters, supported_log_probability, jnp.nan)


# Public JAX tail values are stable, but its autodiff rule can return nonfinite gradients
@jax.custom_jvp
def _standard_logpdf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    return cast(jax.Array, jax_truncnorm.logpdf(value, lower, upper))


@_standard_logpdf.defjvp
def _standard_logpdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, lower, upper = primals
    value_tangent, lower_tangent, upper_tangent = tangents
    log_density = _standard_logpdf(value, lower, upper)

    # Endpoint density ratios are the analytic derivatives of the normalizing mass
    squared_value = jnp.square(value)
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    safe_lower = jnp.where(finite_lower, lower, jnp.zeros_like(lower))
    safe_upper = jnp.where(finite_upper, upper, jnp.zeros_like(upper))
    lower_weight = jnp.where(
        finite_lower,
        jnp.exp(log_density + 0.5 * (squared_value - jnp.square(safe_lower))),
        jnp.zeros_like(lower),
    )
    upper_weight = jnp.where(
        finite_upper,
        jnp.exp(log_density + 0.5 * (squared_value - jnp.square(safe_upper))),
        jnp.zeros_like(upper),
    )
    log_density_tangent = -value * value_tangent + lower_weight * lower_tangent - upper_weight * upper_tangent
    return log_density, log_density_tangent


@jax.custom_jvp
def _standard_logcdf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    return cast(jax.Array, jax_truncnorm.logcdf(value, lower, upper))


@_standard_logcdf.defjvp
def _standard_logcdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    value, lower, upper = primals
    value_tangent, lower_tangent, upper_tangent = tangents
    log_probability = _standard_logcdf(value, lower, upper)
    log_survival = _standard_logsf(value, lower, upper)
    log_density = _standard_logpdf(value, lower, upper)

    squared_value = jnp.square(value)
    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    safe_lower = jnp.where(finite_lower, lower, jnp.zeros_like(lower))
    safe_upper = jnp.where(finite_upper, upper, jnp.zeros_like(upper))
    lower_log_weight = jnp.where(
        finite_lower,
        log_density + 0.5 * (squared_value - jnp.square(safe_lower)),
        -jnp.inf,
    )
    upper_log_weight = jnp.where(
        finite_upper,
        log_density + 0.5 * (squared_value - jnp.square(safe_upper)),
        -jnp.inf,
    )
    value_derivative = jnp.exp(log_density - log_probability)
    lower_derivative = -jnp.exp(lower_log_weight + log_survival - log_probability)
    upper_derivative = -jnp.exp(upper_log_weight)
    probability_tangent = (
        value_derivative * value_tangent + lower_derivative * lower_tangent + upper_derivative * upper_tangent
    )
    return log_probability, probability_tangent


def _standard_logsf(
    value: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    return _standard_logcdf(-value, -upper, -lower)


# Infinite bounds represent fixed endpoints, so their location and scale tangents must stay zero
@jax.custom_jvp
def _standardize_bound(
    bound: jax.Array,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return _standardize(bound, location, scale)


@_standardize_bound.defjvp
def _standardize_bound_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    bound, location, scale = primals
    bound_tangent, location_tangent, scale_tangent = tangents
    standardized = _standardize_bound(bound, location, scale)
    finite_bound = jnp.isfinite(bound)
    finite_standardized = jnp.where(finite_bound, standardized, jnp.zeros_like(standardized))
    standardized_tangent = (bound_tangent - location_tangent) / scale - finite_standardized * (scale_tangent / scale)
    return standardized, jnp.where(finite_bound, standardized_tangent, jnp.zeros_like(standardized_tangent))


def _normal_log_mass(lower: jax.Array, upper: jax.Array) -> jax.Array:
    left_tail = upper <= 0
    right_tail = lower > 0
    central = ~(left_tail | right_tail)

    # Reflect right-tail intervals because log_ndtr is most accurate on the left
    tail_lower = jnp.where(right_tail, -upper, lower)
    tail_upper = jnp.where(right_tail, -lower, upper)
    upper_logcdf = log_ndtr(tail_upper)
    lower_logcdf = log_ndtr(tail_lower)
    tail_mass = upper_logcdf + jax.nn.log1mexp(upper_logcdf - lower_logcdf)

    # Keep inactive same-side tails away from log1p(-1) so their gradients stay finite
    central_lower = jnp.where(central, lower, -jnp.ones_like(lower))
    central_upper = jnp.where(central, upper, jnp.ones_like(upper))
    central_mass = jnp.log1p(-ndtr(central_lower) - ndtr(-central_upper))
    return jnp.where(central, central_mass, tail_mass)


def _inverse_normal_logcdf(log_probability: jax.Array) -> jax.Array:
    finite_probability = jnp.isfinite(log_probability)
    safe_log_probability = jnp.where(finite_probability, log_probability, -jnp.ones_like(log_probability))

    ordinary_value = ndtri(jnp.exp(safe_log_probability))
    near_one_value = -ndtri(-jnp.expm1(safe_log_probability))

    # Cephes switches approximation when sqrt(-2 log(p)) reaches 8, or log(p) reaches -32
    far_tail = safe_log_probability < -32
    tail_log_probability = jnp.where(far_tail, safe_log_probability, -32)
    maximum = jnp.asarray(np.finfo(log_probability.dtype).max, dtype=log_probability.dtype)
    direct_root = tail_log_probability >= -maximum / 2
    direct_argument = jnp.where(direct_root, tail_log_probability, -jnp.ones_like(tail_log_probability))
    large_argument = jnp.where(direct_root, -jnp.ones_like(tail_log_probability), tail_log_probability)
    root = jnp.where(
        direct_root,
        jnp.sqrt(-2 * direct_argument),
        jnp.sqrt(jnp.asarray(2, dtype=log_probability.dtype)) * jnp.sqrt(-large_argument),
    )
    leading_term = root - jnp.log(root) / root
    reciprocal_root = 1 / root

    # These are the far-tail Cephes coefficients used by SciPy and JAX's ndtri
    numerator_coefficients = jnp.asarray(
        [
            3.23774891776946035970e0,
            6.91522889068984211695e0,
            3.93881025292474443415e0,
            1.33303460815807542389e0,
            2.01485389549179081538e-1,
            1.23716634817820021358e-2,
            3.01581553508235416007e-4,
            2.65806974686737550832e-6,
            6.23974539184983293730e-9,
        ],
        dtype=log_probability.dtype,
    )
    denominator_coefficients = jnp.asarray(
        [
            1.0,
            6.02427039364742014255e0,
            3.67983563856160859403e0,
            1.37702099489081330271e0,
            2.16236993594496635890e-1,
            1.34204006088543189037e-2,
            3.28014464682127739104e-4,
            2.89247864745380683936e-6,
            6.79019408009981274425e-9,
        ],
        dtype=log_probability.dtype,
    )
    correction = reciprocal_root * (
        jnp.polyval(numerator_coefficients, reciprocal_root) / jnp.polyval(denominator_coefficients, reciprocal_root)
    )
    tail_value = correction - leading_term

    near_one_boundary = jnp.log1p(-jnp.exp(jnp.asarray(-2, dtype=log_probability.dtype)))
    value = jnp.where(
        far_tail,
        tail_value,
        jnp.where(safe_log_probability > near_one_boundary, near_one_value, ordinary_value),
    )
    value = jnp.where(jnp.isneginf(log_probability), -jnp.inf, value)
    return jnp.where(log_probability <= 0, value, jnp.nan)
