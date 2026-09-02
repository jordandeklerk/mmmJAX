"""Shared distribution implementation utilities."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jax.typing import ArrayLike


def _as_real_array(name: str, value: ArrayLike) -> jax.Array:
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"distribution argument {name!r} must be real numeric and array-like, got {type(value).__name__}"
        ) from exc

    is_real_numeric = (
        array.dtype == jnp.dtype(jnp.bool_)
        or jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.floating)
    )
    if not is_real_numeric:
        raise TypeError(f"distribution argument {name!r} must have a real numeric dtype, got {array.dtype}")

    return array


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
    values = [_as_real_array(name, value) for name, value in arguments]

    dtype = jnp.result_type(*values)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    # Use float32 or better so the distribution tails have enough detail
    if jax.dtypes.itemsize_bits(dtype) < 32:
        dtype = jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return tuple(jnp.asarray(value, dtype=dtype) for value in values)


def _is_valid_simplex(value: jax.Array) -> jax.Array:
    """Return whether each final-axis vector is a finite probability simplex."""
    tolerance = jnp.asarray(
        1e-8 if jax.dtypes.itemsize_bits(value.dtype) == 64 else 1e-6,
        dtype=value.dtype,
    )
    return jnp.all(jnp.isfinite(value) & (value >= 0), axis=-1) & (jnp.abs(jnp.sum(value, axis=-1) - 1) <= tolerance)


def _gamma_shape_normalizer(shape: jax.Array) -> jax.Array:
    dtype_bits = jax.dtypes.itemsize_bits(shape.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=shape.dtype)
    asymptotic_region = shape >= asymptotic_threshold

    exact_shape = jnp.where(asymptotic_region, jnp.ones_like(shape), shape)
    exact_normalizer = exact_shape * jnp.log(exact_shape) - exact_shape - gammaln(exact_shape)

    asymptotic_shape = jnp.where(asymptotic_region, shape, asymptotic_threshold)
    inverse_shape = 1 / asymptotic_shape
    asymptotic_normalizer = _asymptotic_gamma_shape_normalizer(
        jnp.log(asymptotic_shape),
        inverse_shape,
    )
    return jnp.where(asymptotic_region, asymptotic_normalizer, exact_normalizer)


def _gamma_shape_log_derivative(shape: jax.Array) -> jax.Array:
    dtype_bits = jax.dtypes.itemsize_bits(shape.dtype)
    asymptotic_threshold = jnp.asarray(64 if dtype_bits == 64 else 8, dtype=shape.dtype)
    asymptotic_region = shape >= asymptotic_threshold

    exact_shape = jnp.where(asymptotic_region, jnp.ones_like(shape), shape)
    exact_derivative = jnp.log(exact_shape) - digamma(exact_shape)

    asymptotic_shape = jnp.where(asymptotic_region, shape, asymptotic_threshold)
    inverse_shape = 1 / asymptotic_shape
    asymptotic_derivative = _asymptotic_gamma_shape_log_derivative(inverse_shape)
    return jnp.where(asymptotic_region, asymptotic_derivative, exact_derivative)


def _asymptotic_gamma_shape_normalizer(
    log_shape: jax.Array,
    inverse_shape: jax.Array,
) -> jax.Array:
    squared_inverse_shape = jnp.square(inverse_shape)
    stirling_correction = inverse_shape * (
        1 / 12
        + squared_inverse_shape * (-1 / 360 + squared_inverse_shape * (1 / 1260 + squared_inverse_shape * (-1 / 1680)))
    )
    return 0.5 * (log_shape - jnp.asarray(math.log(2 * math.pi), dtype=log_shape.dtype)) - stirling_correction


def _asymptotic_gamma_shape_log_derivative(inverse_shape: jax.Array) -> jax.Array:
    squared_inverse_shape = jnp.square(inverse_shape)
    return inverse_shape * (
        0.5
        + inverse_shape
        * (
            1 / 12
            + squared_inverse_shape
            * (-1 / 120 + squared_inverse_shape * (1 / 252 + squared_inverse_shape * (-1 / 240)))
        )
    )


def _stable_log_ratio(
    numerator: jax.Array,
    denominator: jax.Array,
    raw_log_ratio: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ratio = numerator / denominator
    ratio_deviation = (numerator - denominator) / denominator
    valid_ratio = (
        jnp.isfinite(denominator)
        & (denominator > 0)
        & jnp.isfinite(ratio)
        & (ratio > 0)
        & jnp.isfinite(ratio_deviation)
    )
    quotient_log_ratio = jnp.where(valid_ratio, jnp.log(ratio), raw_log_ratio)

    # Tapering the correction keeps last-bit differences from becoming jumps at large shapes
    blend_distance = jnp.where(valid_ratio, jnp.abs(ratio_deviation), jnp.ones_like(ratio_deviation))
    blend_position = jnp.clip((0.5 - blend_distance) / 0.25, min=0, max=1)
    blend_weight = jnp.square(blend_position) * (3 - 2 * blend_position)
    blend_quotient_log_ratio = jnp.where(blend_weight > 0, quotient_log_ratio, jnp.zeros_like(quotient_log_ratio))
    blend_ratio_deviation = jnp.where(blend_weight > 0, ratio_deviation, jnp.zeros_like(ratio_deviation))
    blended_log_ratio = blend_quotient_log_ratio + blend_weight * (
        jnp.log1p(blend_ratio_deviation) - blend_quotient_log_ratio
    )
    stable_log_ratio = jnp.where(blend_weight > 0, blended_log_ratio, quotient_log_ratio)

    # The raw expression supplies the exact analytical derivative to higher-order AD
    log_ratio = raw_log_ratio + jax.lax.stop_gradient(stable_log_ratio - raw_log_ratio)
    return log_ratio, ratio_deviation, valid_ratio


def _log_ratio_deviance_series(argument: jax.Array) -> jax.Array:
    coefficient = jnp.asarray(-1 / math.factorial(18), dtype=argument.dtype)
    for order in range(17, 1, -1):
        coefficient = jnp.asarray(-1 / math.factorial(order), dtype=argument.dtype) + argument * coefficient
    return jnp.square(argument) * coefficient


def _weighted_log_ratio_deviance(
    shape: jax.Array,
    log_ratio: jax.Array,
    linear_deviation: jax.Array,
) -> jax.Array:
    series_region = (jnp.abs(log_ratio) <= 1) & jnp.isfinite(linear_deviation)
    series_argument = jnp.where(series_region, log_ratio, jnp.zeros_like(log_ratio))
    series_shape = jnp.where(series_region, shape, jnp.ones_like(shape))
    series_contribution = series_shape * _log_ratio_deviance_series(series_argument)

    # expm1 is stable outside the cancellation region and cannot overflow on the lower tail
    exponential_region = ~series_region & (log_ratio <= 2)
    exponential_argument = jnp.where(exponential_region, log_ratio, jnp.zeros_like(log_ratio))
    exponential_shape = jnp.where(exponential_region, shape, jnp.ones_like(shape))
    exponential_contribution = exponential_shape * (exponential_argument - jnp.expm1(exponential_argument))

    # The direct linear ratio keeps the far upper tail finite at the dtype limit
    direct_contribution = shape * log_ratio - linear_deviation
    return jnp.where(
        series_region,
        series_contribution,
        jnp.where(exponential_region, exponential_contribution, direct_contribution),
    )
