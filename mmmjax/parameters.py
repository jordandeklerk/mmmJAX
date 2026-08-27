"""Parameter declarations and their inference-space mappings."""

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

__all__ = [
    "Interval",
    "LowerBound",
    "Parameterization",
    "Positive",
    "Real",
    "UpperBound",
]


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
    to use with traced JAX arrays. Finite-precision arithmetic can round
    constrained values to zero for extreme negative positions.
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


@dataclass(frozen=True, slots=True)
class LowerBound:
    r"""Parameter constrained to be greater than a finite lower bound.

    For an unconstrained position :math:`z` and lower bound :math:`a`, the
    parameter is :math:`a + \exp(z)`.

    ``unconstrain`` assumes its input is greater than ``lower``. Support
    validation belongs at the eager model boundary so this numerical kernel
    remains safe to use with traced JAX arrays. Finite-precision arithmetic can
    round constrained values to the boundary for extreme negative positions.
    """

    lower: float
    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        """Normalize metadata so equal specifications share JIT cache keys."""
        _validate_shape(self.shape)
        dtype = _canonicalize_dtype(self.dtype)
        lower = _canonicalize_bound(self.lower, name="lower", dtype=dtype)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "dtype", dtype)

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        return self.shape

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Apply the lower-bound transform to an unconstrained position."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        lower = jnp.asarray(self.lower, dtype=self.dtype)
        return lower + jnp.exp(position)

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Map a lower-bounded parameter to unconstrained inference space."""
        parameter = _as_array(
            parameter,
            name="parameter",
            shape=self.shape,
            dtype=self.dtype,
        )
        lower = jnp.asarray(self.lower, dtype=self.dtype)
        return jnp.log(parameter - lower)

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the log absolute Jacobian determinant of the transform."""
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


@dataclass(frozen=True, slots=True)
class UpperBound:
    r"""Parameter constrained to be less than a finite upper bound.

    For an unconstrained position :math:`z` and upper bound :math:`b`, the
    parameter is :math:`b - \exp(z)`.

    ``unconstrain`` assumes its input is less than ``upper``. Support
    validation belongs at the eager model boundary so this numerical kernel
    remains safe to use with traced JAX arrays. Finite-precision arithmetic can
    round constrained values to the boundary for extreme negative positions.
    """

    upper: float
    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        """Normalize metadata so equal specifications share JIT cache keys."""
        _validate_shape(self.shape)
        dtype = _canonicalize_dtype(self.dtype)
        upper = _canonicalize_bound(self.upper, name="upper", dtype=dtype)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "dtype", dtype)

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        return self.shape

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Apply the upper-bound transform to an unconstrained position."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        upper = jnp.asarray(self.upper, dtype=self.dtype)
        return upper - jnp.exp(position)

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Map an upper-bounded parameter to unconstrained inference space."""
        parameter = _as_array(
            parameter,
            name="parameter",
            shape=self.shape,
            dtype=self.dtype,
        )
        upper = jnp.asarray(self.upper, dtype=self.dtype)
        return jnp.log(upper - parameter)

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the log absolute Jacobian determinant of the transform."""
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


@dataclass(frozen=True, slots=True)
class Interval:
    r"""Parameter constrained to an open interval with finite bounds.

    For an unconstrained position :math:`z`, lower bound :math:`a`, and upper
    bound :math:`b`, the parameter is
    :math:`a + (b - a)\operatorname{sigmoid}(z)`.

    ``unconstrain`` assumes its input is inside the interval. Finite-precision
    arithmetic can round constrained values to a boundary for extreme
    positions.
    """

    lower: float
    upper: float
    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        """Normalize and validate the finite interval metadata."""
        _validate_shape(self.shape)
        dtype = _canonicalize_dtype(self.dtype)
        lower = _canonicalize_bound(self.lower, name="lower", dtype=dtype)
        upper = _canonicalize_bound(self.upper, name="upper", dtype=dtype)
        if lower >= upper:
            raise ValueError(
                f"lower must be less than upper after conversion to {jnp.dtype(dtype)}, "
                f"got lower={lower} and upper={upper}"
            )
        width = jnp.asarray(upper, dtype=dtype) - jnp.asarray(lower, dtype=dtype)
        if not bool(jnp.isfinite(width)):
            raise ValueError(
                f"upper - lower must be finite in dtype {jnp.dtype(dtype)}, got lower={lower} and upper={upper}"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "dtype", dtype)

    @property
    def position_shape(self) -> tuple[int, ...]:
        """Shape of the parameter in unconstrained inference space."""
        return self.shape

    def constrain(self, position: ArrayLike) -> jax.Array:
        """Map an unconstrained position into the open interval."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        lower = jnp.asarray(self.lower, dtype=self.dtype)
        width = jnp.asarray(self.upper - self.lower, dtype=self.dtype)
        # Keep positive-tail gradients from disappearing when sigmoid rounds to one
        unit = jax.lax.logistic(position, accuracy=jax.lax.AccuracyMode.HIGHEST)
        return lower + width * unit

    def unconstrain(self, parameter: ArrayLike) -> jax.Array:
        """Map an interval-constrained parameter to inference space."""
        parameter = _as_array(
            parameter,
            name="parameter",
            shape=self.shape,
            dtype=self.dtype,
        )
        lower = jnp.asarray(self.lower, dtype=self.dtype)
        upper = jnp.asarray(self.upper, dtype=self.dtype)
        return jnp.log(parameter - lower) - jnp.log(upper - parameter)

    def log_density_adjustment(self, position: ArrayLike) -> jax.Array:
        """Return the scalar log absolute Jacobian determinant."""
        position = _as_array(
            position,
            name="position",
            shape=self.position_shape,
            dtype=self.dtype,
        )
        width = jnp.asarray(self.upper - self.lower, dtype=self.dtype)
        adjustment = jnp.log(width) + jax.nn.log_sigmoid(position) + jax.nn.log_sigmoid(-position)
        return jnp.sum(adjustment)

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


def _canonicalize_bound(value: object, *, name: str, dtype: DTypeLike) -> float:
    try:
        value_dtype = jnp.result_type(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar, got {type(value).__name__}") from exc
    is_real_numeric = jnp.issubdtype(value_dtype, jnp.integer) or jnp.issubdtype(value_dtype, jnp.floating)
    if value_dtype == jnp.dtype(jnp.bool_) or not is_real_numeric:
        raise TypeError(f"{name} must be a finite real scalar, got dtype {value_dtype}")

    try:
        array = jnp.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar, got {type(value).__name__}") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    if not bool(jnp.isfinite(array)):
        raise ValueError(f"{name} must be finite after conversion to {jnp.dtype(dtype)}, got {array.item()!r}")
    return float(array.item())


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
