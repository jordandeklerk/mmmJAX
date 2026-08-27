"""Shared distribution implementation utilities."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


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
