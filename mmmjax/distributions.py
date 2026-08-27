"""Probability functions for transparent JAX model definitions."""

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "exponential",
    "exponential_logpdf",
    "exponential_rng",
    "lognormal",
    "lognormal_logpdf",
    "lognormal_rng",
    "normal",
    "normal_logpdf",
    "normal_rng",
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
