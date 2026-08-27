"""Parameter declarations and their inference-space mappings."""

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

__all__ = ["Parameterization", "Positive", "Real"]


@runtime_checkable
class Parameterization(Protocol):
    """Contract for mapping unconstrained positions to model parameters."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter in model space."""
        ...

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        ...

    @property
    def dtype(self) -> DTypeLike:
        """Floating-point dtype of the parameter and its position."""
        ...

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Map an unconstrained position into model space."""
        ...

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Map a parameter from model space into inference space."""
        ...

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the scalar log-density adjustment for ``position``."""
        ...

    def initialize(self, key: jax.Array) -> jax.Array:
        """Generate an initial position using a caller-managed PRNG key."""
        ...


@dataclass(frozen=True, slots=True)
class Real:
    """Unconstrained real parameter with the identity parameterization."""

    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        """Normalize metadata so equal specifications share JIT cache keys."""
        _validate_shape(self.shape)
        object.__setattr__(self, "dtype", _canonicalize_dtype(self.dtype))

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        return self.shape

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Apply the identity map from inference space to model space."""
        return _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Apply the identity map from model space to inference space."""
        return _as_array(
            parameter,
            name="parameter",
            shape=self.shape,
            dtype=self.dtype,
        )

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the zero adjustment for the identity parameterization."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        return jnp.zeros((), dtype=position.dtype)

    def initialize(self, key: jax.Array) -> jax.Array:
        """Draw an unconstrained initial position uniformly from ``[-2, 2)``."""
        return _initialize(key, shape=self.position_shape, dtype=self.dtype)


@dataclass(frozen=True, slots=True)
class Positive:
    """Positive parameter using an exponential parameterization.

    ``unconstrain`` assumes its input is strictly positive. Support validation
    belongs at the eager model boundary so this numerical kernel remains safe
    to use with traced JAX arrays.
    """

    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        """Normalize metadata so equal specifications share JIT cache keys."""
        _validate_shape(self.shape)
        object.__setattr__(self, "dtype", _canonicalize_dtype(self.dtype))

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        return self.shape

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Map an unconstrained position to the positive reals."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        return jnp.exp(position)

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Map a positive parameter to unconstrained inference space."""
        parameter = _as_array(
            parameter,
            name="parameter",
            shape=self.shape,
            dtype=self.dtype,
        )
        return jnp.log(parameter)

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the log absolute Jacobian determinant of ``exp``."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        return jnp.sum(position)

    def initialize(self, key: jax.Array) -> jax.Array:
        """Draw an unconstrained initial position uniformly from ``[-2, 2)``."""
        return _initialize(key, shape=self.position_shape, dtype=self.dtype)


def _validate_shape(shape: tuple[int, ...]) -> None:
    if not isinstance(shape, tuple):
        raise TypeError(f"shape must be a tuple of positive integers, got {type(shape).__name__}")
    invalid_dimension = next(
        ((index, size) for index, size in enumerate(shape) if isinstance(size, bool) or not isinstance(size, int)),
        None,
    )
    if invalid_dimension is not None:
        invalid_index, invalid_size = invalid_dimension
        raise TypeError(
            f"shape[{invalid_index}] must be a positive integer, "
            f"got {invalid_size!r} of type {type(invalid_size).__name__}"
        )
    nonpositive_dimension = next(
        ((index, size) for index, size in enumerate(shape) if size <= 0),
        None,
    )
    if nonpositive_dimension is not None:
        nonpositive_index, nonpositive_size = nonpositive_dimension
        raise ValueError(f"shape[{nonpositive_index}] must be positive, got {nonpositive_size} in shape {shape}")


def _canonicalize_dtype(dtype: DTypeLike) -> DTypeLike:
    try:
        canonical_dtype = jnp.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"dtype must be a floating-point dtype, got {dtype!r}") from exc
    if not jnp.issubdtype(canonical_dtype, jnp.floating):
        raise TypeError(f"dtype must be a floating-point dtype, got {canonical_dtype}")
    return cast(DTypeLike, jax.dtypes.canonicalize_dtype(canonical_dtype))


def _as_array(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: DTypeLike,
) -> jax.Array:
    try:
        array = jnp.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be array-like and convertible to dtype {jnp.dtype(dtype)}, got {type(value).__name__}"
        ) from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _initialize(
    key: jax.Array,
    *,
    shape: tuple[int, ...],
    dtype: DTypeLike,
) -> jax.Array:
    # Match Stan's default range so initialization is familiar to Stan users
    return jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=-2.0,
        maxval=2.0,
    )
