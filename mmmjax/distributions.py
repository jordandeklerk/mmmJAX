"""Probability functions for transparent JAX model definitions."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jax.scipy.stats import beta as beta_distribution
from jax.typing import ArrayLike

__all__ = [
    "beta",
    "beta_logpdf",
    "beta_rng",
    "exponential",
    "exponential_logpdf",
    "exponential_rng",
    "gamma",
    "gamma_logpdf",
    "gamma_rng",
    "half_normal",
    "half_normal_logpdf",
    "half_normal_rng",
    "lognormal",
    "lognormal_logpdf",
    "lognormal_rng",
    "normal",
    "normal_logpdf",
    "normal_rng",
    "student_t",
    "student_t_logpdf",
    "student_t_rng",
]


def normal_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Normal log density elementwise.

    For value :math:`x \in \mathbb{R}`, location :math:`\mu \in \mathbb{R}`,
    and scale :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \mu, \sigma)
        = -\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2
          - \log(\sigma)
          - \frac{1}{2}\log(2\pi),
        \qquad \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        A nonfinite location or a nonpositive or nonfinite scale produces
        ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )
    # Keep the scale out of the square so extreme values stay finite
    standardized = (value_array - location_array) / scale_array
    half_log_two_pi = jnp.asarray(math.log(2 * math.pi) / 2, dtype=value_array.dtype)
    log_density = -0.5 * jnp.square(standardized) - jnp.log(scale_array) - half_log_two_pi
    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, log_density, jnp.nan)


def normal(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Normal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(normal_logpdf(value, location, scale))
    location_array = jnp.asarray(location)
    scale_array = jnp.asarray(scale)
    valid_parameters = jnp.all(jnp.isfinite(location_array)) & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_parameters, log_density, jnp.nan)


def normal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Normal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Location of the distribution.
    scale
        Positive standard deviation of the distribution.
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
    shape = _random_shape(sample_shape, location_array, scale_array)
    standard_normal = jax.random.normal(key, shape=shape, dtype=location_array.dtype)
    samples = location_array + scale_array * standard_normal
    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, samples, jnp.nan)


def half_normal_logpdf(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    r"""Evaluate the HalfNormal log density elementwise.

    For value :math:`x \geq 0` and scale :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \sigma)
        = \frac{1}{2}\log\left(\frac{2}{\pi}\right)
          - \log(\sigma)
          - \frac{1}{2}\left(\frac{x}{\sigma}\right)^2,
        \qquad x \geq 0,\; \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values below zero produce ``-inf`` and a nonpositive or nonfinite scale
        produces ``nan``.
    """
    value_array, scale_array = _promote_inexact(("value", value), ("scale", scale))
    log_two = jnp.asarray(math.log(2), dtype=value_array.dtype)
    log_density = normal_logpdf(value_array, 0, scale_array) + log_two
    supported_log_density = jnp.where(value_array < 0, -jnp.inf, log_density)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_scale, supported_log_density, jnp.nan)


def half_normal(value: ArrayLike, scale: ArrayLike) -> jax.Array:
    """Return the scalar sum of HalfNormal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(half_normal_logpdf(value, scale))
    scale_array = jnp.asarray(scale)
    valid_scale = jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_scale, log_density, jnp.nan)


def half_normal_rng(
    key: jax.Array,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a HalfNormal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    scale
        Positive standard deviation of the underlying zero-centered Normal
        distribution.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + scale.shape``. A
        nonpositive or nonfinite scale produces ``nan``.
    """
    return jnp.abs(normal_rng(key, 0, scale, sample_shape=sample_shape))


def lognormal_logpdf(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the LogNormal log density elementwise.

    For value :math:`x > 0`, log-scale location :math:`\mu \in \mathbb{R}`,
    and log-scale standard deviation :math:`\sigma > 0`, the log density is

    .. math::

        \log p(x \mid \mu, \sigma)
        = -\frac{1}{2}\left(\frac{\log(x) - \mu}{\sigma}\right)^2
          - \log(\sigma)
          - \log(x)
          - \frac{1}{2}\log(2\pi),
        \qquad x > 0,\; \sigma > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values at or below zero produce ``-inf``. A nonfinite location or a
        nonpositive or nonfinite scale produces ``nan``.
    """
    value_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("location", location),
        ("scale", scale),
    )
    outside_support = value_array <= 0
    # Avoid an indeterminate expression at zero without changing NaN inputs
    safe_value = jnp.where(outside_support, jnp.ones_like(value_array), value_array)
    log_value = jnp.log(safe_value)
    log_density = normal_logpdf(log_value, location_array, scale_array) - log_value
    supported_log_density = jnp.where(outside_support, -jnp.inf, log_density)
    valid_parameters = jnp.isfinite(location_array) & jnp.isfinite(scale_array) & (scale_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def lognormal(
    value: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of LogNormal log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(lognormal_logpdf(value, location, scale))
    location_array = jnp.asarray(location)
    scale_array = jnp.asarray(scale)
    valid_parameters = jnp.all(jnp.isfinite(location_array)) & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    return jnp.where(valid_parameters, log_density, jnp.nan)


def lognormal_rng(
    key: jax.Array,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a LogNormal distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    location
        Mean of the underlying Normal distribution for ``log(value)``.
    scale
        Positive standard deviation of the underlying Normal distribution for
        ``log(value)``.
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
    return jnp.exp(normal_rng(key, location, scale, sample_shape=sample_shape))


def exponential_logpdf(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    r"""Evaluate the Exponential log density elementwise.

    For value :math:`x \in \mathbb{R}` and rate :math:`\lambda > 0`, the log
    density is

    .. math::

        \log p(x \mid \lambda)
        = \begin{cases}
            \log(\lambda) - \lambda x, & x \ge 0, \\
            -\infty, & x < 0,
          \end{cases}
        \qquad \lambda > 0.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values below zero produce ``-inf`` and a nonpositive or nonfinite rate
        produces ``nan``.
    """
    value_array, rate_array = _promote_inexact(("value", value), ("rate", rate))
    log_density = jnp.log(rate_array) - rate_array * value_array
    supported_log_density = jnp.where(value_array < 0, -jnp.inf, log_density)
    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, supported_log_density, jnp.nan)


def exponential(value: ArrayLike, rate: ArrayLike) -> jax.Array:
    """Return the scalar sum of Exponential log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(exponential_logpdf(value, rate))
    rate_array = jnp.asarray(rate)
    valid_rate = jnp.all(jnp.isfinite(rate_array) & (rate_array > 0))
    return jnp.where(valid_rate, log_density, jnp.nan)


def exponential_rng(
    key: jax.Array,
    rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from an Exponential distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    rate
        Positive rate parameter, equal to the inverse scale.
    sample_shape
        Independent sample dimensions prepended to the parameter shape. The
        tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + rate.shape``. A nonpositive
        or nonfinite rate produces ``nan``.
    """
    (rate_array,) = _promote_inexact(("rate", rate))
    shape = _random_shape(sample_shape, rate_array)
    standard_exponential = jax.random.exponential(key, shape=shape, dtype=rate_array.dtype)
    samples = standard_exponential / rate_array
    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_rate, samples, jnp.nan)


def gamma_logpdf(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Gamma log density elementwise.

    For value :math:`x > 0`, shape :math:`\alpha > 0`, and rate
    :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = \alpha\log(\beta)
          - \log\Gamma(\alpha)
          + (\alpha - 1)\log(x)
          - \beta x.

    At :math:`x = 0`, the log density is :math:`+\infty` when
    :math:`\alpha < 1`, :math:`\log(\beta)` when :math:`\alpha = 1`, and
    :math:`-\infty` when :math:`\alpha > 1`.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Negative values and positive infinity produce ``-inf``. A nonpositive
        or nonfinite shape or rate produces ``nan``.
    """
    value_array, shape_array, rate_array = _promote_inexact(
        ("value", value),
        ("shape", shape),
        ("rate", rate),
    )
    at_boundary = value_array == 0
    outside_support = value_array < 0
    positive_infinity = jnp.isposinf(value_array)
    # Keep boundary and unsupported values out of undefined logarithms
    safe_value = jnp.where(
        at_boundary | outside_support | positive_infinity,
        jnp.ones_like(value_array),
        value_array,
    )
    interior_log_density = (
        shape_array * jnp.log(rate_array)
        - gammaln(shape_array)
        + xlogy(shape_array - 1, safe_value)
        - rate_array * safe_value
    )
    boundary_log_density = jnp.where(
        shape_array < 1,
        jnp.inf,
        jnp.where(shape_array == 1, jnp.log(rate_array), -jnp.inf),
    )
    supported_log_density = jnp.where(at_boundary, boundary_log_density, interior_log_density)
    supported_log_density = jnp.where(
        outside_support | positive_infinity,
        -jnp.inf,
        supported_log_density,
    )
    valid_parameters = jnp.isfinite(shape_array) & (shape_array > 0) & jnp.isfinite(rate_array) & (rate_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def gamma(
    value: ArrayLike,
    shape: ArrayLike,
    rate: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Gamma log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(gamma_logpdf(value, shape, rate))
    shape_array = jnp.asarray(shape)
    rate_array = jnp.asarray(rate)
    valid_parameters = jnp.all(jnp.isfinite(shape_array) & (shape_array > 0)) & jnp.all(
        jnp.isfinite(rate_array) & (rate_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def gamma_rng(
    key: jax.Array,
    shape: ArrayLike,
    rate: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Gamma distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    shape
        Positive shape parameter.
    rate
        Positive rate parameter, equal to the inverse scale.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape or rate produces ``nan``.
    """
    shape_array, rate_array = _promote_inexact(("shape", shape), ("rate", rate))
    output_shape = _random_shape(sample_shape, shape_array, rate_array)
    valid_shape = jnp.isfinite(shape_array) & (shape_array > 0)
    valid_rate = jnp.isfinite(rate_array) & (rate_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_shape = jnp.where(valid_shape, shape_array, jnp.ones_like(shape_array))
    safe_rate = jnp.where(valid_rate, rate_array, jnp.ones_like(rate_array))
    # Scale in log space so tiny unit-rate draws can be rescaled before they underflow
    log_unit_rate_samples = jax.random.loggamma(
        key,
        safe_shape,
        shape=output_shape,
        dtype=shape_array.dtype,
    )
    samples = jnp.exp(log_unit_rate_samples - jnp.log(safe_rate))
    return jnp.where(valid_shape & valid_rate, samples, jnp.nan)


def beta_logpdf(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Beta log density elementwise.

    For value :math:`x \in (0, 1)` and shape parameters :math:`\alpha > 0`
    and :math:`\beta > 0`, the log density is

    .. math::

        \log p(x \mid \alpha, \beta)
        = (\alpha - 1)\log(x)
          + (\beta - 1)\log(1 - x)
          - \log\mathrm{B}(\alpha, \beta).

    At :math:`x = 0`, the log density is :math:`+\infty` when
    :math:`\alpha < 1`, :math:`\log(\beta)` when :math:`\alpha = 1`, and
    :math:`-\infty` when :math:`\alpha > 1`. At :math:`x = 1`, the log
    density is :math:`+\infty` when :math:`\beta < 1`, :math:`\log(\alpha)`
    when :math:`\beta = 1`, and :math:`-\infty` when :math:`\beta > 1`.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments.
        Values outside ``[0, 1]`` produce ``-inf``. A nonpositive or
        nonfinite shape parameter produces ``nan``.
    """
    value_array, alpha_array, beta_array = _promote_inexact(
        ("value", value),
        ("alpha", alpha),
        ("beta", beta),
    )
    at_lower_boundary = value_array == 0
    at_upper_boundary = value_array == 1
    outside_support = (value_array < 0) | (value_array > 1) | jnp.isinf(value_array)
    # Keep endpoints and unsupported values out of undefined logarithms
    safe_value = jnp.where(
        at_lower_boundary | at_upper_boundary | outside_support,
        jnp.full_like(value_array, 0.5),
        value_array,
    )
    interior_log_density = beta_distribution.logpdf(safe_value, alpha_array, beta_array)
    lower_boundary_log_density = jnp.where(
        alpha_array < 1,
        jnp.inf,
        jnp.where(alpha_array == 1, jnp.log(beta_array), -jnp.inf),
    )
    upper_boundary_log_density = jnp.where(
        beta_array < 1,
        jnp.inf,
        jnp.where(beta_array == 1, jnp.log(alpha_array), -jnp.inf),
    )
    supported_log_density = jnp.where(
        at_lower_boundary,
        lower_boundary_log_density,
        jnp.where(at_upper_boundary, upper_boundary_log_density, interior_log_density),
    )
    supported_log_density = jnp.where(outside_support, -jnp.inf, supported_log_density)
    valid_parameters = jnp.isfinite(alpha_array) & (alpha_array > 0) & jnp.isfinite(beta_array) & (beta_array > 0)
    return jnp.where(valid_parameters, supported_log_density, jnp.nan)


def beta(
    value: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Beta log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.

    Returns
    -------
    jax.Array
        Complete normalized log density, including constants, summed across
        every dimension of the broadcast result.
    """
    log_density = jnp.sum(beta_logpdf(value, alpha, beta))
    alpha_array = jnp.asarray(alpha)
    beta_array = jnp.asarray(beta)
    valid_parameters = jnp.all(jnp.isfinite(alpha_array) & (alpha_array > 0)) & jnp.all(
        jnp.isfinite(beta_array) & (beta_array > 0)
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def beta_rng(
    key: jax.Array,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Beta distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    alpha
        Positive first shape parameter.
    beta
        Positive second shape parameter.
    sample_shape
        Independent sample dimensions prepended to the broadcast parameter shape.
        The tuple must be static when the function is JIT-compiled.

    Returns
    -------
    jax.Array
        Random variates with shape ``sample_shape + broadcast_shape``. A
        nonpositive or nonfinite shape parameter produces ``nan``.
    """
    alpha_array, beta_array = _promote_inexact(("alpha", alpha), ("beta", beta))
    output_shape = _random_shape(sample_shape, alpha_array, beta_array)
    valid_alpha = jnp.isfinite(alpha_array) & (alpha_array > 0)
    valid_beta = jnp.isfinite(beta_array) & (beta_array > 0)
    # Keep invalid shapes out of JAX's rejection sampler
    safe_alpha = jnp.where(valid_alpha, alpha_array, jnp.ones_like(alpha_array))
    safe_beta = jnp.where(valid_beta, beta_array, jnp.ones_like(beta_array))
    samples = jax.random.beta(
        key,
        safe_alpha,
        safe_beta,
        shape=output_shape,
        dtype=alpha_array.dtype,
    )
    return jnp.where(valid_alpha & valid_beta, samples, jnp.nan)


def student_t_logpdf(
    value: ArrayLike,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    r"""Evaluate the Student-t log density elementwise.

    For value :math:`x \in \mathbb{R}`, degrees of freedom :math:`\nu > 0`,
    location :math:`\mu \in \mathbb{R}`, and scale :math:`\sigma > 0`, the
    log density is

    .. math::

        \log p(x \mid \nu, \mu, \sigma)
        = \log\Gamma\left(\frac{\nu + 1}{2}\right)
          - \log\Gamma\left(\frac{\nu}{2}\right)
          - \frac{1}{2}\log(\nu\pi)
          - \log(\sigma)
          - \frac{\nu + 1}{2}
            \log\left(1 + \frac{1}{\nu}
            \left(\frac{x - \mu}{\sigma}\right)^2\right).

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    degrees_of_freedom
        Positive degrees of freedom.
    location
        Location of the distribution.
    scale
        Positive scale parameter. This is not the standard deviation except
        in the Normal limit.

    Returns
    -------
    jax.Array
        Normalized log densities with the broadcast shape of the arguments. A
        nonpositive or nonfinite degrees of freedom or scale, or a nonfinite
        location, produces ``nan``.
    """
    value_array, degrees_array, location_array, scale_array = _promote_inexact(
        ("value", value),
        ("degrees_of_freedom", degrees_of_freedom),
        ("location", location),
        ("scale", scale),
    )

    valid_degrees = jnp.isfinite(degrees_array) & (degrees_array > 0)
    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)

    safe_degrees = jnp.where(valid_degrees, degrees_array, jnp.ones_like(degrees_array))
    safe_location = jnp.where(valid_location, location_array, jnp.zeros_like(location_array))
    safe_scale = jnp.where(valid_scale, scale_array, jnp.ones_like(scale_array))
    log_scale = jnp.log(safe_scale)

    one = jnp.ones((), dtype=value_array.dtype)
    log_two = jnp.asarray(math.log(2), dtype=value_array.dtype)

    residual = value_array - safe_location
    at_location = value_array == safe_location
    scaled_residual_region = jnp.isfinite(value_array) & valid_location & ~at_location

    scaled_value = jnp.where(scaled_residual_region, value_array, jnp.ones_like(value_array))
    scaled_location = jnp.where(scaled_residual_region, safe_location, jnp.zeros_like(safe_location))
    residual_magnitude = jnp.maximum(jnp.abs(scaled_value), jnp.abs(scaled_location))
    _, residual_exponent = jnp.frexp(residual_magnitude)
    residual_scale = jnp.ldexp(
        jnp.ones_like(residual_magnitude),
        residual_exponent - 1,
    )
    scaled_residual = scaled_value / residual_scale - scaled_location / residual_scale

    direct_residual = jnp.where(
        scaled_residual_region | at_location,
        jnp.ones_like(residual),
        residual,
    )
    residual_for_log = jnp.where(scaled_residual_region, scaled_residual, direct_residual)
    log_scale_adjustment = (residual_exponent - 1).astype(value_array.dtype) * log_two
    log_absolute_residual = jnp.log(jnp.abs(residual_for_log)) + jnp.where(
        scaled_residual_region,
        log_scale_adjustment,
        jnp.zeros_like(log_scale_adjustment),
    )

    epsilon = jnp.spacing(one)
    tail_weight = (safe_degrees + 1) / 2
    log_squared_ratio = 2 * (log_absolute_residual - log_scale) - jnp.log(safe_degrees)
    log_squared_ratio = jnp.where(at_location, -jnp.inf, log_squared_ratio)
    small_ratio_region = log_squared_ratio < jnp.log(epsilon)

    # The leading expansion keeps the Normal limit when 1 / nu rounds to zero
    small_noncentral_region = small_ratio_region & ~at_location
    noncentral_log_squared_standardized = jnp.where(
        small_noncentral_region,
        2 * (log_absolute_residual - log_scale),
        jnp.zeros_like(log_squared_ratio),
    )
    noncentral_squared_standardized = jnp.exp(noncentral_log_squared_standardized)
    noncentral_squared_ratio = jnp.exp(
        jnp.where(
            small_noncentral_region,
            log_squared_ratio,
            jnp.zeros_like(log_squared_ratio),
        )
    )

    center_residual = jnp.where(at_location, residual, jnp.zeros_like(residual))
    center_scale = jnp.where(
        at_location,
        jax.lax.stop_gradient(safe_scale),
        jnp.ones_like(safe_scale),
    )
    center_degrees = jnp.where(
        at_location,
        jax.lax.stop_gradient(safe_degrees),
        jnp.ones_like(safe_degrees),
    )
    center_standardized = center_residual / center_scale
    center_squared_standardized = jnp.square(center_standardized)
    center_squared_ratio = center_squared_standardized / center_degrees

    squared_standardized = jnp.where(
        at_location,
        center_squared_standardized,
        noncentral_squared_standardized,
    )
    squared_ratio = jnp.where(at_location, center_squared_ratio, noncentral_squared_ratio)
    small_ratio_kernel = 0.5 * (squared_standardized + squared_ratio)

    log_kernel_ratio = jnp.where(
        small_ratio_region,
        jnp.zeros_like(log_squared_ratio),
        log_squared_ratio,
    )
    log_kernel = tail_weight * jnp.logaddexp(jnp.zeros_like(log_kernel_ratio), log_kernel_ratio)

    density_kernel = jnp.where(small_ratio_region, small_ratio_kernel, log_kernel)

    # These cutoffs keep the first omitted asymptotic term below the dtype precision
    dtype_bits = jax.dtypes.itemsize_bits(value_array.dtype)
    asymptotic_threshold = jnp.asarray(
        64 if dtype_bits == 64 else 8,
        dtype=value_array.dtype,
    )
    small_degrees_region = safe_degrees < 1
    asymptotic_region = safe_degrees >= asymptotic_threshold
    gamma_normalizer_degrees = jnp.where(asymptotic_region, jnp.ones_like(safe_degrees), safe_degrees)

    half = jnp.asarray(0.5, dtype=value_array.dtype)
    log_pi = jnp.asarray(math.log(math.pi), dtype=value_array.dtype)
    half_log_two_pi = jnp.asarray(math.log(2 * math.pi) / 2, dtype=value_array.dtype)

    gamma_numerator = jnp.where(
        small_degrees_region,
        1 + gamma_normalizer_degrees / 2,
        gamma_normalizer_degrees / 2,
    )
    gamma_ratio = gammaln(gamma_numerator) - gammaln((gamma_normalizer_degrees + 1) / 2)
    gamma_normalizer = gamma_ratio + jnp.where(
        small_degrees_region,
        -half * jnp.log(gamma_normalizer_degrees) + log_two + half * log_pi,
        half * (jnp.log(gamma_normalizer_degrees) + log_pi),
    )

    # The asymptotic form avoids subtracting nearly equal log-Gamma values
    asymptotic_degrees = jnp.where(
        asymptotic_region,
        safe_degrees,
        jnp.full_like(safe_degrees, asymptotic_threshold),
    )
    inverse_degrees = 1 / asymptotic_degrees
    squared_inverse_degrees = jnp.square(inverse_degrees)
    asymptotic_normalizer = half_log_two_pi + inverse_degrees * (
        0.25
        + squared_inverse_degrees
        * (-1 / 24 + squared_inverse_degrees * (1 / 20 + squared_inverse_degrees * (-17 / 112)))
    )

    degrees_normalizer = jnp.where(asymptotic_region, asymptotic_normalizer, gamma_normalizer)

    log_density = -log_scale - degrees_normalizer - density_kernel
    valid_parameters = valid_degrees & valid_location & valid_scale
    return jnp.where(valid_parameters, log_density, jnp.nan)


def student_t(
    value: ArrayLike,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
) -> jax.Array:
    """Return the scalar sum of Student-t log densities.

    Parameters
    ----------
    value
        Values at which to evaluate the density.
    degrees_of_freedom
        Positive degrees of freedom.
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
    log_density = jnp.sum(student_t_logpdf(value, degrees_of_freedom, location, scale))
    degrees_array = jnp.asarray(degrees_of_freedom)
    location_array = jnp.asarray(location)
    scale_array = jnp.asarray(scale)
    valid_parameters = (
        jnp.all(jnp.isfinite(degrees_array) & (degrees_array > 0))
        & jnp.all(jnp.isfinite(location_array))
        & jnp.all(jnp.isfinite(scale_array) & (scale_array > 0))
    )
    return jnp.where(valid_parameters, log_density, jnp.nan)


def student_t_rng(
    key: jax.Array,
    degrees_of_freedom: ArrayLike,
    location: ArrayLike,
    scale: ArrayLike,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from a Student-t distribution using a JAX random key.

    Parameters
    ----------
    key
        JAX random key controlling the draw. Reusing a key repeats the same
        sample. Use ``jax.random.split`` to create keys for new random
        operations.
    degrees_of_freedom
        Positive degrees of freedom.
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
        nonpositive or nonfinite degrees of freedom or scale, or a nonfinite
        location, produces ``nan``.
    """
    degrees_array, location_array, scale_array = _promote_inexact(
        ("degrees_of_freedom", degrees_of_freedom),
        ("location", location),
        ("scale", scale),
    )
    output_shape = _random_shape(sample_shape, degrees_array, location_array, scale_array)
    valid_degrees = jnp.isfinite(degrees_array) & (degrees_array > 0)
    valid_location = jnp.isfinite(location_array)
    valid_scale = jnp.isfinite(scale_array) & (scale_array > 0)
    safe_degrees = jnp.where(valid_degrees, degrees_array, jnp.ones_like(degrees_array))
    safe_location = jnp.where(valid_location, location_array, jnp.zeros_like(location_array))
    safe_scale = jnp.where(valid_scale, scale_array, jnp.ones_like(scale_array))

    normal_key, gamma_key = jax.random.split(key)
    standard_normal = jax.random.normal(normal_key, shape=output_shape, dtype=degrees_array.dtype)
    log_unit_gamma = jax.random.loggamma(
        gamma_key,
        safe_degrees / 2,
        shape=output_shape,
        dtype=degrees_array.dtype,
    )
    nonzero_normal = standard_normal != 0
    safe_absolute_normal = jnp.where(nonzero_normal, jnp.abs(standard_normal), jnp.ones_like(standard_normal))
    log_magnitude = (
        jnp.log(safe_scale)
        + jnp.log(safe_absolute_normal)
        + 0.5 * (jnp.log(safe_degrees) - math.log(2) - log_unit_gamma)
    )
    centered_samples = jnp.where(
        nonzero_normal,
        jnp.copysign(jnp.exp(log_magnitude), standard_normal),
        jnp.zeros_like(standard_normal),
    )
    samples = safe_location + centered_samples
    valid_parameters = valid_degrees & valid_location & valid_scale
    return jnp.where(valid_parameters, samples, jnp.nan)


def _random_shape(sample_shape: tuple[int, ...], *parameters: jax.Array) -> tuple[int, ...]:
    if not isinstance(sample_shape, tuple):
        raise TypeError(f"sample_shape must be a tuple of nonnegative integers, got {type(sample_shape).__name__}")
    invalid_dimension = next(
        (
            (index, size)
            for index, size in enumerate(sample_shape)
            if isinstance(size, bool) or not isinstance(size, int)
        ),
        None,
    )
    if invalid_dimension is not None:
        invalid_index, invalid_size = invalid_dimension
        raise TypeError(
            f"sample_shape[{invalid_index}] must be a nonnegative integer, "
            f"got {invalid_size!r} of type {type(invalid_size).__name__}"
        )
    negative_dimension = next(
        ((index, size) for index, size in enumerate(sample_shape) if size < 0),
        None,
    )
    if negative_dimension is not None:
        negative_index, negative_size = negative_dimension
        raise ValueError(
            f"sample_shape[{negative_index}] must be nonnegative, got {negative_size} in sample_shape {sample_shape}"
        )
    parameter_shapes = tuple(parameter.shape for parameter in parameters)
    try:
        batch_shape = jnp.broadcast_shapes(*parameter_shapes)
    except ValueError as exc:
        raise ValueError(f"distribution parameter shapes cannot be broadcast together: {parameter_shapes}") from exc
    return sample_shape + batch_shape


def _promote_inexact(
    *arguments: tuple[str, ArrayLike],
) -> tuple[jax.Array, ...]:
    values: list[ArrayLike] = []
    for name, value in arguments:
        try:
            argument_dtype = jnp.result_type(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"distribution argument {name!r} must be real numeric and array-like, got {type(value).__name__}"
            ) from exc
        is_real_numeric = (
            argument_dtype == jnp.dtype(jnp.bool_)
            or jnp.issubdtype(argument_dtype, jnp.integer)
            or jnp.issubdtype(argument_dtype, jnp.floating)
        )
        if not is_real_numeric:
            raise TypeError(f"distribution argument {name!r} must have a real numeric dtype, got {argument_dtype}")
        values.append(value)

    dtype = jnp.result_type(*values)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    # Use float32 or better so the distribution tails have enough detail
    if jax.dtypes.itemsize_bits(dtype) < 32:
        dtype = jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return tuple(jnp.asarray(value, dtype=dtype) for value in values)
