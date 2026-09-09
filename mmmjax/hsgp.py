"""Hilbert-space Gaussian process primitives for smooth time-varying effects."""

from typing import Literal

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["hsgp_basis", "hsgp_weights"]


def hsgp_basis(
    time: ArrayLike,
    *,
    center: ArrayLike,
    boundary: ArrayLike,
    n_basis: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Build a fixed time basis for a Hilbert-space Gaussian process.

    For time :math:`t`, center :math:`c`, and domain half-width :math:`L`,
    basis functions and their angular frequencies are

    .. math::

        \omega_j = \frac{\pi j}{2L}, \qquad
        \phi_j(t) = \frac{1}{\sqrt{L}}
        \sin\left(\omega_j(t-c+L)\right), \quad j=1,\ldots,m.

    The basis can be shared across groups and channels. Combine it with
    :func:`hsgp_weights` and model coefficients to obtain time-varying curves.
    This function does not choose priors or estimate a curve from observations.

    Parameters
    ----------
    time : array_like
        One-dimensional, finite numeric time positions. For dated inputs,
        measure elapsed time from a fixed reference date and retain that
        reference for prediction.
    center : array_like
        Finite scalar subtracted from the time positions. The midpoint of
        the training time range is a useful choice. Reuse this value for
        prediction rather than recentering each new dataset.
    boundary : array_like
        Positive, finite half-width of the approximation domain in the same
        units as ``time``. Both training and intended prediction dates should
        lie well inside ``center - boundary`` and ``center + boundary``.
        The basis vanishes at the endpoints. Keep this value fixed after
        defining the model.
    n_basis : int
        Positive number of basis functions. More functions allow the
        approximation to capture shorter-scale changes. A wider domain
        generally needs more functions for the same resolution. Keep this
        argument static when using ``jax.jit``.

    Returns
    -------
    basis : jax.Array
        Matrix with shape ``(len(time), n_basis)`` and increasing frequency
        along the final axis. Nonfinite or out-of-domain time positions
        produce ``nan`` rows. An invalid center or boundary gives an invalid
        basis. Values use a common floating dtype of at least float32.
    frequencies : jax.Array
        Angular frequencies with shape ``(n_basis,)`` in inverse time units.
        Pass these to :func:`hsgp_weights`. An invalid boundary gives ``nan``
        frequencies.

    Examples
    --------
    Prepare a shared basis for weekly observations. The domain also leaves
    room for future weeks without changing the basis definition.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import hsgp_basis
           ...: frame = pl.DataFrame({
           ...:     "week": [0, 1, 2, 3],
           ...:     "sales": [100.0, 120.0, 110.0, 130.0],
           ...: })
           ...: basis, frequencies = hsgp_basis(
           ...:     frame["week"].to_numpy(),
           ...:     center=1.5, boundary=10.0, n_basis=20,
           ...: )
           ...: basis.shape, frequencies.shape
    """
    if isinstance(n_basis, bool) or not isinstance(n_basis, int):
        raise TypeError("n_basis must be a positive Python integer and stay fixed during JIT compilation")
    if n_basis <= 0:
        raise ValueError(f"n_basis must be at least 1, got {n_basis}")

    time_array, center_array, boundary_array = _prepare_hsgp_inputs(
        ("time", time), ("center", center), ("boundary", boundary)
    )
    if time_array.ndim != 1:
        raise ValueError(f"time must be one-dimensional, got shape {time_array.shape}")
    if center_array.ndim != 0:
        raise ValueError(f"center must be a scalar time position, got shape {center_array.shape}")
    if boundary_array.ndim != 0:
        raise ValueError(f"boundary must be a scalar domain half-width, got shape {boundary_array.shape}")

    valid_boundary = jnp.isfinite(boundary_array) & (boundary_array > 0)
    safe_boundary = jnp.where(valid_boundary, boundary_array, 1.0)
    centered_time = time_array - center_array
    valid_time = (
        jnp.isfinite(time_array)
        & jnp.isfinite(center_array)
        & valid_boundary
        & (jnp.abs(centered_time) <= safe_boundary)
    )
    safe_time = jnp.where(valid_time, centered_time, 0.0)

    modes = jnp.arange(1, n_basis + 1, dtype=time_array.dtype)
    frequencies = (jnp.pi / 2) * modes / safe_boundary
    angles = (safe_time / safe_boundary + 1)[:, None] * ((jnp.pi / 2) * modes)
    basis = jnp.sin(angles) / jnp.sqrt(safe_boundary)
    return jnp.where(valid_time[:, None], basis, jnp.nan), jnp.where(valid_boundary, frequencies, jnp.nan)


def hsgp_weights(
    frequencies: ArrayLike,
    *,
    length_scale: ArrayLike,
    amplitude: ArrayLike = 1.0,
    covariance: Literal["expquad", "matern32", "matern52"] = "matern52",
) -> jax.Array:
    r"""Compute coefficient standard deviations for an HSGP approximation.

    For angular frequency :math:`\omega`, length scale :math:`\ell`, and
    amplitude :math:`\eta`, the weights are :math:`\sqrt{S(\omega)}` with

    .. math::

        S(\omega) = \eta^2
        \begin{cases}
          \sqrt{2\pi}\ell\exp\left(-\ell^2\omega^2/2\right)
              & \text{expquad}, \\
          12\sqrt{3}\ell\left(3+\ell^2\omega^2\right)^{-2}
              & \text{matern32}, \\
          \frac{400\sqrt{5}}{3}\ell\left(5+\ell^2\omega^2\right)^{-3}
              & \text{matern52}.
        \end{cases}

    With standard-normal model coefficients ``z``, evaluate a curve as
    ``basis @ (weights * z)``. Alternatively, put zero-mean Normal priors
    with these standard deviations directly on the basis coefficients.

    Parameters
    ----------
    frequencies : array_like
        One-dimensional, finite angular frequencies returned by
        :func:`hsgp_basis`. Zero and negative frequencies are also accepted.
    length_scale : array_like
        Positive, finite scale controlling how quickly the curve changes,
        measured in the same time units used to build the basis. Larger
        values favor slower changes. Use a scalar for a shared scale or
        an array such as ``(group, channel)`` for separate scales.
    amplitude : array_like, default 1.0
        Nonnegative, finite standard deviation of the underlying stationary
        process. Zero disables variation. Its shape must broadcast with
        ``length_scale``. The target marginal variance is ``amplitude**2``.
        Approximation quality also depends on the domain and basis count.
    covariance : {"expquad", "matern32", "matern52"}, default "matern52"
        Covariance family. Matérn 3/2 allows rougher curves than Matérn 5/2.
        The squared-exponential family ``expquad`` produces very smooth
        curves. Keep this argument static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Nonnegative weights with shape ``batch_shape + (len(frequencies),)``.
        The batch shape is the broadcast shape of ``length_scale`` and
        ``amplitude``. Invalid numeric inputs give ``nan`` in affected
        positions. Values use a common floating dtype of at least float32.
        Batched coefficients can share one basis through a contraction such
        as ``jnp.einsum("tm,gcm->tgc", basis, weights * z)``.

    Examples
    --------
    Draw a smooth curve from standard-normal coefficients. These are prior
    draws, not estimates fitted to observed outcomes.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import hsgp_basis, hsgp_weights, normal_rng
           ...: basis, frequencies = hsgp_basis(
           ...:     [0.0, 1.0, 2.0, 3.0],
           ...:     center=1.5, boundary=10.0, n_basis=20,
           ...: )
           ...: weights = hsgp_weights(
           ...:     frequencies, length_scale=3.0, amplitude=0.5,
           ...: )
           ...: z = normal_rng(random.key(0), 0.0, 1.0, sample_shape=(20,))
           ...: basis @ (weights * z)
    """
    if covariance not in ("expquad", "matern32", "matern52"):
        raise ValueError(f"covariance must be 'expquad', 'matern32', or 'matern52', got {covariance!r}")
    frequency_array, length_array, amplitude_array = _prepare_hsgp_inputs(
        ("frequencies", frequencies), ("length_scale", length_scale), ("amplitude", amplitude)
    )
    if frequency_array.ndim != 1:
        raise ValueError(f"frequencies must be one-dimensional, got shape {frequency_array.shape}")
    try:
        length_array, amplitude_array = jnp.broadcast_arrays(length_array, amplitude_array)
    except ValueError as error:
        raise ValueError(
            f"length_scale with shape {length_array.shape} and amplitude with shape "
            f"{amplitude_array.shape} must broadcast together"
        ) from error

    valid_parameters = (
        jnp.isfinite(length_array) & (length_array > 0) & jnp.isfinite(amplitude_array) & (amplitude_array >= 0)
    )
    valid_frequencies = jnp.isfinite(frequency_array)
    safe_length = jnp.where(valid_parameters, length_array, 1.0)[..., None]
    safe_amplitude = jnp.where(valid_parameters, amplitude_array, 1.0)[..., None]
    safe_frequency = jnp.where(valid_frequencies, frequency_array, 0.0)
    squared_frequency = jnp.square(safe_length * safe_frequency)

    # Evaluate the square root analytically so a vanishing spectrum does not
    # introduce a derivative through sqrt(0).
    if covariance == "expquad":
        unit_weights = jnp.sqrt(jnp.sqrt(2 * jnp.pi) * safe_length) * jnp.exp(-0.25 * squared_frequency)
    elif covariance == "matern32":
        unit_weights = jnp.sqrt(12 * jnp.sqrt(3.0) * safe_length) / (3 + squared_frequency)
    else:
        unit_weights = jnp.sqrt((400 * jnp.sqrt(5.0) / 3) * safe_length) / (5 + squared_frequency) ** 1.5

    return jnp.where(valid_parameters[..., None] & valid_frequencies, safe_amplitude * unit_weights, jnp.nan)


def _prepare_hsgp_inputs(*arguments: tuple[str, ArrayLike]) -> list[jax.Array]:
    """Convert basis or covariance inputs to a common real floating dtype."""
    leaves = []
    for name, value in arguments:
        value_leaves = jax.tree_util.tree_leaves(value)
        try:
            argument_dtype = jnp.result_type(*value_leaves)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be real numeric and array-like") from error
        if value is None or not (
            jnp.issubdtype(argument_dtype, jnp.floating) or jnp.issubdtype(argument_dtype, jnp.integer)
        ):
            raise TypeError(f"{name} must have a real numeric dtype, got {argument_dtype}")
        leaves.extend(value_leaves)

    dtype = jnp.result_type(*leaves)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
    arrays = []
    for name, value in arguments:
        try:
            arrays.append(jnp.asarray(value, dtype=dtype))
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be real numeric and array-like") from error
    return arrays
