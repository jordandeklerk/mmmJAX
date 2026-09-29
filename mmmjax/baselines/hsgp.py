"""Hilbert-space Gaussian process primitives for smooth time-varying effects."""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["HSGPApproximation", "hsgp_basis", "hsgp_weights", "prepare_hsgp"]


@dataclass(frozen=True, slots=True)
class HSGPApproximation:
    """Retain one HSGP approximation for training and prediction.

    :func:`prepare_hsgp` builds this object before the model is defined. It
    holds the domain, basis count, and covariance family that training and
    prediction share.

    Calling :meth:`basis` on new time positions reuses the same domain, basis
    count, and column centering, so a forecast basis lines up column for
    column with the training basis. Calling :meth:`weights` reuses the
    frequencies and covariance family, and the length scales and amplitudes
    can change from call to call.

    Attributes
    ----------
    center : float
        Midpoint of the time range used to prepare the approximation.
    boundary : float
        Domain half-width in the same units as the time positions.
    n_basis : int
        Number of basis functions retained in the approximation.
    covariance : str
        Covariance family used to choose the settings and compute weights.
    length_scale_range : tuple of float or None
        Length scales the settings were prepared for.
    column_means : tuple of float or None
        Basis column means over the preparation positions. :meth:`basis`
        subtracts them when the approximation was prepared with centering.
    """

    center: float
    boundary: float
    n_basis: int
    covariance: Literal["expquad", "matern32", "matern52"]
    length_scale_range: tuple[float, float] | None = None
    column_means: tuple[float, ...] | None = None

    @property
    def frequencies(self) -> jax.Array:
        """Return the angular frequencies shared by every basis evaluation."""
        _, frequencies = hsgp_basis(
            jnp.asarray([self.center]), center=self.center, boundary=self.boundary, n_basis=self.n_basis
        )
        return frequencies

    def basis(self, time: ArrayLike) -> jax.Array:
        """Evaluate the stored basis without changing its time reference.

        The center, domain, and basis count come from preparation, so new
        positions get the same columns as the training basis. When the
        approximation was prepared with ``center_columns``, each column also
        has its mean over the training positions subtracted.

        Parameters
        ----------
        time : array_like
            One-dimensional numeric time positions in the same units and
            from the same reference date used during preparation.

        Returns
        -------
        jax.Array
            Matrix with shape ``(len(time), n_basis)``. Nonfinite positions
            or positions outside the domain give ``nan`` rows.
        """
        basis, _ = hsgp_basis(time, center=self.center, boundary=self.boundary, n_basis=self.n_basis)
        if self.column_means is not None:
            basis = basis - jnp.asarray(self.column_means, dtype=basis.dtype)
        return basis

    def weights(
        self,
        *,
        length_scale: ArrayLike,
        amplitude: ArrayLike = 1.0,
    ) -> jax.Array:
        """Compute coefficient weights using the stored frequencies and covariance.

        The weights match :func:`hsgp_weights` for the prepared frequencies
        and covariance family. The length scale range given to
        ``prepare_hsgp`` only sized the approximation. It neither limits
        these length scales nor sets their prior.

        Parameters
        ----------
        length_scale : array_like
            Positive, finite length scales in the same units as the time
            positions.
        amplitude : array_like, default 1.0
            Nonnegative, finite process standard deviations. Their shape
            must broadcast with ``length_scale``.

        Returns
        -------
        jax.Array
            Coefficient standard deviations with shape
            ``batch_shape + (n_basis,)``. ``batch_shape`` is the broadcast
            shape of ``length_scale`` and ``amplitude``. Invalid numeric
            inputs give ``nan`` in affected positions.
        """
        return hsgp_weights(
            self.frequencies, length_scale=length_scale, amplitude=amplitude, covariance=self.covariance
        )


def prepare_hsgp(
    time_range: ArrayLike,
    *,
    length_scale_range: ArrayLike,
    covariance: Literal["expquad", "matern32", "matern52"] = "matern52",
    boundary: float | None = None,
    n_basis: int | None = None,
    center_columns: bool = False,
) -> HSGPApproximation:
    """Choose a fixed domain and basis count for a time-varying process.

    The longest expected length scale sets how far the domain reaches past
    the time range, and the shortest sets how many basis functions it takes
    to follow faster changes. The recommended half-width is never less than
    1.2 times half the time span.

    These recommendations for each covariance family follow
    Riutort-Mayol et al. (2022) and can understate variation near the ends of
    the time range. Refitting with a wider ``boundary`` and a larger
    ``n_basis`` shows whether results depend on the approximation. More basis
    functions alone cannot make up for too little padding.

    The approximation is prepared once on the host, outside ``jax.jit``,
    before the model is defined. It fits no curve and sets no priors, and the
    length scale range places no limit on the model's length scales.

    Later predictions reuse the approximation and must measure time from the
    same origin. With ``center_columns``, any curve built on the basis
    averages zero over the observed positions, so an intercept keeps the
    level.

    Parameters
    ----------
    time_range : array_like
        Two finite numeric endpoints in increasing order that cover the
        training periods and planned forecasts, as in ``(0, 116)`` for weeks
        0 through 104 and forecasts through week 116. Observed positions such
        as ``data.time_positions`` reduce to their smallest and largest values.
    length_scale_range : array_like
        Two positive, finite endpoints in increasing order, in the units of
        ``time_range``. The range should hold most of the prior probability
        of the model's length scales.
    covariance : {"expquad", "matern32", "matern52"}, default "matern52"
        Covariance family.
    boundary : float, optional
        Domain half-width in the units of ``time_range``. It must be finite
        and greater than half the time span. Defaults to the recommended
        half-width.
    n_basis : int, optional
        Positive number of basis functions. Defaults to the recommended count
        for the domain half-width and the shortest length scale.
    center_columns : bool, default False
        Subtract each basis column's mean over the observed positions from
        every later evaluation. Requires observed positions in ``time_range``
        rather than two endpoints.

    Returns
    -------
    HSGPApproximation
        Reusable approximation with the following fields.

        - **center** — Time range midpoint
        - **boundary** — Padded domain half-width
        - **n_basis** — Basis count
        - **covariance** — Covariance family for coefficient weights

    Examples
    --------
    Start with observations at four points in time and a domain that
    leaves room for four forecast weeks.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import prepare_hsgp

        In [2]: frame = pl.DataFrame({
           ...:     "week": [0, 4, 8, 12],
           ...:     "sales": [100.0, 120.0, 110.0, 130.0],
           ...: })

        In [3]: approximation = prepare_hsgp((0, 16), length_scale_range=(2, 8))

    The basis for the training weeks and the spectral weights for a chosen
    length scale come from the same approximation.

    .. ipython::

        In [4]: basis = approximation.basis(frame["week"].to_numpy())
           ...: weights = approximation.weights(length_scale=4.0)

    Future weeks reuse the definition, so their basis has the same number of
    columns and pairs with the same weights.

    .. ipython::

        In [5]: future_basis = approximation.basis([13.0, 14.0, 15.0, 16.0])
           ...: basis.shape, weights.shape, future_basis.shape
    """
    if covariance not in ("expquad", "matern32", "matern52"):
        raise ValueError(f"covariance must be 'expquad', 'matern32', or 'matern52', got {covariance!r}")

    ranges = []
    for name, value in (("time_range", time_range), ("length_scale_range", length_scale_range)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{name} must contain two real numeric endpoints. Run preparation outside jax.jit"
            ) from error
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric endpoints, got dtype {array.dtype}")
        if name == "time_range" and array.ndim == 1 and array.size > 2:
            # Observed positions stand in for their own extent.
            array = np.array([array.min(), array.max()])
        if array.shape != (2,):
            raise ValueError(f"{name} must contain at least two positions or two endpoints, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite endpoints, got {value!r}")
        lower, upper = float(array[0]), float(array[1])
        if name == "length_scale_range" and lower <= 0:
            raise ValueError(f"length_scale_range must contain positive endpoints, got {value!r}")
        if lower >= upper:
            raise ValueError(f"{name} must have its lower endpoint less than its upper endpoint, got {value!r}")
        ranges.append((lower, upper))

    (start, end), (shortest, longest) = ranges
    center = (start + end) / 2
    half_span = (end - start) / 2

    # One-dimensional sizing recommendations from Riutort-Mayol et al.
    # Domain padding controls boundary effects and the basis count controls
    # how much of the high-frequency spectrum is retained.
    if covariance == "expquad":
        padding, resolution = 3.2, 1.75
    elif covariance == "matern32":
        padding, resolution = 4.5, 3.42
    else:
        padding, resolution = 4.1, 2.65
    if boundary is None:
        domain_width = max(padding * longest, 1.2 * half_span)
    else:
        if isinstance(boundary, (bool, np.bool_)) or not isinstance(boundary, (int, float, np.integer, np.floating)):
            raise TypeError("boundary must be a real scalar domain half-width")
        domain_width = float(boundary)
        if not np.isfinite(domain_width) or domain_width <= half_span:
            raise ValueError(f"boundary must be finite and greater than the time range half-span of {half_span}")

    basis_count = resolution * domain_width / shortest
    if not np.all(np.isfinite([center, domain_width, basis_count])):
        raise ValueError(
            "The HSGP settings exceed numeric limits. Check the time range, domain width, and length scales"
        )
    if n_basis is None:
        n_basis = max(1, int(basis_count))
    elif isinstance(n_basis, bool) or not isinstance(n_basis, int):
        raise TypeError("n_basis must be a positive Python integer")
    if n_basis <= 0:
        raise ValueError(f"n_basis must be at least 1, got {n_basis}")

    column_means = None
    if center_columns:
        positions = np.asarray(time_range, dtype=np.float64)
        if positions.ndim != 1 or positions.size < 3:
            raise ValueError("center_columns requires the observed positions rather than two endpoints")
        training_basis, _ = hsgp_basis(positions, center=center, boundary=domain_width, n_basis=n_basis)
        column_means = tuple(float(value) for value in np.asarray(training_basis).mean(axis=0))
    return HSGPApproximation(
        center=center,
        boundary=domain_width,
        n_basis=n_basis,
        covariance=covariance,
        length_scale_range=(shortest, longest),
        column_means=column_means,
    )


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

    The basis depends only on time, so one basis serves every group and
    channel. The frequencies go to :func:`hsgp_weights`, and multiplying the
    basis by those weights and by model coefficients gives a time-varying
    curve. The function chooses no priors and fits no curve to observations.

    Every basis function vanishes at ``center - boundary`` and
    ``center + boundary``, so training and forecast positions should lie well
    inside that interval. More functions follow shorter-scale changes, and a
    wider domain generally needs more of them for the same resolution.

    Forecast calls must reuse the training center, boundary, and basis count
    and measure dated inputs from the same reference date. Their basis then
    lines up with the training basis instead of being recentered on the new
    data.

    Parameters
    ----------
    time : array_like
        One-dimensional, finite numeric time positions, such as elapsed time
        from a fixed reference date.
    center : array_like
        Finite scalar subtracted from the time positions, such as the
        midpoint of the training time range.
    boundary : array_like
        Positive, finite half-width of the approximation domain in the same
        units as ``time``.
    n_basis : int
        Positive number of basis functions. Keep this argument static when
        using ``jax.jit``.

    Returns
    -------
    basis : jax.Array
        Shape ``(len(time), n_basis)`` in increasing frequency order. Floating
        dtype is at least float32. Nonfinite or out-of-domain times give
        ``nan`` rows. An invalid center or boundary makes every row ``nan``.
    frequencies : jax.Array
        Angular frequencies shaped ``(n_basis,)`` in inverse time units. An
        invalid boundary gives ``nan`` frequencies.

    Examples
    --------
    Start with four weekly observations.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import hsgp_basis

        In [2]: frame = pl.DataFrame({
           ...:     "week": [0, 1, 2, 3],
           ...:     "sales": [100.0, 120.0, 110.0, 130.0],
           ...: })

    Center the basis on the observed weeks with a boundary wide enough to
    leave room for future weeks. The basis has one column per frequency.

    .. ipython::

        In [3]: basis, frequencies = hsgp_basis(
           ...:     frame["week"].to_numpy(),
           ...:     center=1.5, boundary=10.0, n_basis=20,
           ...: )

        In [4]: basis.shape, frequencies.shape
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

    With standard-normal model coefficients ``z``, the curve is
    ``basis @ (weights * z)``. Zero-mean Normal priors with these standard
    deviations placed directly on the basis coefficients give the same prior
    over curves.

    Larger length scales favor slower changes. Matérn 3/2 allows rougher
    curves than Matérn 5/2, and the squared-exponential family ``expquad``
    produces very smooth curves.

    The amplitude is the standard deviation of the underlying process, so
    zero disables variation. The target marginal variance is
    ``amplitude**2``, and how closely the approximation matches it depends on
    the domain and basis count.

    Parameters
    ----------
    frequencies : array_like
        One-dimensional, finite angular frequencies returned by
        :func:`hsgp_basis`. Zero and negative frequencies are also accepted.
    length_scale : array_like
        Positive, finite length scale in the time units used to build the
        basis. A scalar gives one shared scale and an array such as
        ``(group, channel)`` gives separate scales.
    amplitude : array_like, default 1.0
        Nonnegative, finite standard deviation of the underlying stationary
        process. Its shape must broadcast with ``length_scale``.
    covariance : {"expquad", "matern32", "matern52"}, default "matern52"
        Covariance family. Keep this argument static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Nonnegative weights shaped ``batch_shape + (len(frequencies),)``.
        ``batch_shape`` is the broadcast shape of ``length_scale`` and
        ``amplitude``. Floating dtype is at least float32. Invalid numeric
        inputs give ``nan`` in affected positions.

    Examples
    --------
    Build a basis over four points in time.

    .. ipython::

        In [1]: from jax import random
           ...: from mmmjax import hsgp_basis, hsgp_weights, normal_rng

        In [2]: basis, frequencies = hsgp_basis(
           ...:     [0.0, 1.0, 2.0, 3.0],
           ...:     center=1.5, boundary=10.0, n_basis=20,
           ...: )

    The weights scale each frequency according to the length scale and
    amplitude of the process.

    .. ipython::

        In [3]: weights = hsgp_weights(
           ...:     frequencies, length_scale=3.0, amplitude=0.5,
           ...: )

    Standard-normal coefficients give one draw of a smooth curve. These are
    prior draws, not estimates fitted to observed outcomes.

    .. ipython::

        In [4]: z = normal_rng(random.key(0), 0.0, 1.0, sample_shape=(20,))
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
        value_leaves = jax.tree.leaves(value)
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
