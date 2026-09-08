"""Reusable centering and scaling for model inputs."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["Scaling", "fit_scaling"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class Scaling:
    r"""Store an offset and scale for repeated transformations.

    For an offset :math:`c` and a positive scale :math:`s`, transformation
    and inversion use

    .. math::

        z = \frac{x - c}{s}, \qquad x = z s + c.

    Use :func:`fit_scaling` to estimate these values from training data.
    Applying them to new data does not update them. Direct construction
    does not validate the supplied values.

    Attributes
    ----------
    offset : jax.Array
        Values subtracted before scaling. Reduced axes have length one
        so the offsets broadcast across new observations.
    scale : jax.Array
        Positive divisors with the same shape as ``offset``. Constant
        training series use a divisor of one.

    Notes
    -----
    This object can be passed to JAX transformations with its arrays as
    dynamic inputs. Its fields cannot be reassigned.
    """

    offset: jax.Array
    scale: jax.Array

    def transform(self, values: ArrayLike) -> jax.Array:
        """Center and scale values using the stored training statistics.

        Parameters
        ----------
        values : array_like
            Real-valued inputs. Stored statistics must broadcast to their
            shape without adding axes or enlarging any dimension. Use
            the same feature and group ordering as the training data.

        Returns
        -------
        jax.Array
            Transformed values with the same shape as ``values``. Inputs
            and statistics are promoted to a common floating-point dtype.

        Examples
        --------
        Apply training statistics to a new observation.

        .. ipython::

            In [1]: from mmmjax import fit_scaling
               ...: scaling = fit_scaling([2.0, 4.0, 6.0])
               ...: scaling.transform([8.0])
        """
        values_array = self._prepare_values(values)
        return (values_array - self.offset) / self.scale

    def inverse_transform(self, values: ArrayLike) -> jax.Array:
        """Return scaled values to their original units.

        Parameters
        ----------
        values : array_like
            Real-valued scaled inputs or predictions. Stored statistics
            must broadcast to their shape without adding axes or enlarging
            any dimension. Keep the training feature and group ordering.

        Returns
        -------
        jax.Array
            Values in the original units, with the same shape as ``values``.
            Inputs and statistics use a common floating-point dtype.

        Examples
        --------
        Convert predictions from the scaled units back to the original units.

        .. ipython::

            In [1]: from mmmjax import fit_scaling
               ...: scaling = fit_scaling([2.0, 4.0, 6.0])
               ...: scaling.inverse_transform([0.0, 1.0])
        """
        values_array = self._prepare_values(values)
        return values_array * self.scale + self.offset

    def _prepare_values(self, values: ArrayLike) -> jax.Array:
        """Check numeric inputs and broadcasting for either transformation."""
        try:
            values_array = jnp.asarray(values)
        except (TypeError, ValueError) as error:
            raise TypeError("values must be a real numeric array or sequence") from error
        if not (
            jnp.issubdtype(values_array.dtype, jnp.floating)
            or jnp.issubdtype(values_array.dtype, jnp.integer)
            or values_array.dtype == jnp.bool_
        ):
            raise TypeError(f"values must have a real numeric dtype, got {values_array.dtype}")
        try:
            result_shape = jnp.broadcast_shapes(values_array.shape, self.offset.shape, self.scale.shape)
        except ValueError as error:
            raise ValueError(
                f"values with shape {values_array.shape} do not match scaling statistics with shapes "
                f"{self.offset.shape} and {self.scale.shape}. Keep the training feature and group axes"
            ) from error
        if result_shape != values_array.shape:
            raise ValueError(
                f"scaling statistics would expand values from shape {values_array.shape} to {result_shape}. "
                "Keep the training feature and group axes, including axes of length one"
            )
        dtype = jnp.result_type(values_array, self.offset, self.scale)
        return jnp.asarray(values_array, dtype=dtype)


def fit_scaling(
    values: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = 0,
    center: bool = True,
    scale: bool = True,
) -> Scaling:
    """Estimate reusable centering and scaling from training observations.

    By default, subtract the training mean and divide by the training
    standard deviation. Statistics use all observations along the selected
    axes, with no degrees-of-freedom correction. Constant training series
    use a scale of one, preserving differences in future observations.

    Parameters
    ----------
    values : array_like
        Finite real-valued training observations with at least one dimension
        and no empty axes. Boolean inputs are treated as zeros and ones.
        The original values are not modified.
    axis : int or tuple of int or None, default 0
        Axes over which to estimate statistics. For values shaped
        ``(time, group, feature)``, use ``0`` for separate group-feature
        statistics or ``(0, 1)`` to pool observations across groups.
        ``None`` uses all axes. Negative axes are accepted.
    center : bool, default True
        Whether to subtract the training mean. Set to ``False`` to use
        an offset of zero.
    scale : bool, default True
        Whether to divide by the training standard deviation. Set to
        ``False`` to use a divisor of one.

    Returns
    -------
    Scaling
        Stored transformation containing

        - **offset** : Training mean, or zeros when centering is disabled
        - **scale** : Training standard deviation, with ones for constant
          series or when scaling is disabled

        Reduced axes have length one. Arrays use at least float32 precision,
        with float64 available when JAX 64-bit mode is enabled. Fitting runs
        outside JAX transformations. The returned object's ``transform`` and
        ``inverse_transform`` methods support JIT, gradients, and vectorization.

    Examples
    --------
    Fit separate statistics for two columns, then reuse them on later
    observations without changing the training scales.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import fit_scaling
           ...: training = pl.DataFrame({
           ...:     "temperature": [10.0, 12.0, 14.0],
           ...:     "price": [8.0, 10.0, 12.0],
           ...: })
           ...: scaling = fit_scaling(training.to_numpy())
           ...: future = pl.DataFrame({"temperature": [16.0], "price": [9.0]})
           ...: scaled = scaling.transform(future.to_numpy())
           ...: scaling.inverse_transform(scaled)
    """
    if not isinstance(center, bool):
        raise TypeError(f"center must be True or False, got {center!r}")
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be True or False, got {scale!r}")

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError("values must be a real numeric array or sequence") from error
    if array.dtype.kind not in "biuf":
        raise TypeError(f"values must have a real numeric dtype, got {array.dtype}")
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"values must have at least one dimension and no empty axes, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinite entries. Resolve them before fitting scaling")

    if axis is None:
        axes = tuple(range(array.ndim))
    else:
        requested_axes = axis if isinstance(axis, tuple) else (axis,)
        if not requested_axes or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in requested_axes
        ):
            raise TypeError("axis must be an integer, a nonempty tuple of integers, or None")
        if any(item < -array.ndim or item >= array.ndim for item in requested_axes):
            raise ValueError(f"axis {axis!r} is out of bounds for values with {array.ndim} dimensions")
        axes = tuple(int(item) % array.ndim for item in requested_axes)
        if len(set(axes)) != len(axes):
            raise ValueError(f"axis {axis!r} selects an axis more than once. Select each axis only once")

    dtype = array.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64 if dtype.itemsize == 8 else np.float32)
    dtype = jax.dtypes.canonicalize_dtype(np.promote_types(dtype, np.float32))
    statistic_shape = tuple(1 if index in axes else size for index, size in enumerate(array.shape))

    # Estimate once on the host with a wider accumulator, then store the chosen JAX precision
    with np.errstate(over="ignore", invalid="ignore"):
        offset = np.mean(array, axis=axes, dtype=np.float64, keepdims=True) if center else np.zeros(statistic_shape)
        deviation = np.std(array, axis=axes, dtype=np.float64, keepdims=True) if scale else np.ones(statistic_shape)
        if scale:
            minimum = np.min(array, axis=axes, keepdims=True)
            constant = minimum == np.max(array, axis=axes, keepdims=True)
            # Even identical decimal values can have a tiny computed standard deviation
            deviation = np.where(constant | (deviation == 0), 1.0, deviation)
            if center:
                offset = np.where(constant, minimum, offset)
        scale_values = deviation.astype(dtype)
        offset = offset.astype(dtype)
    if not np.isfinite(offset).all() or not np.isfinite(scale_values).all() or np.any(scale_values <= 0):
        raise ValueError(
            f"values do not produce a finite offset and positive scale in {dtype}. "
            "Rescale the training values or use float64 inputs with JAX 64-bit mode"
        )

    return Scaling(offset=jnp.asarray(offset), scale=jnp.asarray(scale_values))
