"""Hilbert-space Gaussian process primitives for smooth time-varying effects."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from keyword import iskeyword
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from mmmjax.data import PreparedData, _time_positions
from mmmjax.parameters import Parameterization, Positive, Real

__all__ = ["HSGPConfig", "HSGPEffect", "hsgp_basis", "hsgp_weights", "prepare_hsgp"]


@dataclass(frozen=True, slots=True)
class HSGPConfig:
    """Retain one HSGP approximation for training and prediction.

    Use :func:`prepare_hsgp` to choose these settings before defining a
    model. Calling :meth:`basis` for new time positions reuses the same
    domain and basis count. Calling :meth:`weights` reuses the covariance
    family while allowing length scales and amplitudes to vary.
    Direct construction does not validate the supplied settings.

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
    """

    center: float
    boundary: float
    n_basis: int
    covariance: Literal["expquad", "matern32", "matern52"]

    def basis(self, time: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Evaluate the stored basis without changing its time reference.

        Parameters
        ----------
        time : array_like
            One-dimensional numeric time positions in the same units and
            from the same reference date used during preparation.

        Returns
        -------
        basis : jax.Array
            Matrix with shape ``(len(time), n_basis)``. Nonfinite positions
            or positions outside the domain give ``nan`` rows.
        frequencies : jax.Array
            Angular frequencies with shape ``(n_basis,)``. Pass these to
            :meth:`weights` to compute coefficient standard deviations.

        Examples
        --------
        .. ipython::

            In [1]: from mmmjax import prepare_hsgp
               ...: config = prepare_hsgp((0, 16), length_scale_range=(2, 8))
               ...: basis, frequencies = config.basis([0.0, 1.0, 2.0])
               ...: basis.shape
        """
        return hsgp_basis(time, center=self.center, boundary=self.boundary, n_basis=self.n_basis)

    def weights(
        self,
        frequencies: ArrayLike,
        *,
        length_scale: ArrayLike,
        amplitude: ArrayLike = 1.0,
    ) -> jax.Array:
        """Compute coefficient weights using the stored covariance family.

        Parameters
        ----------
        frequencies : array_like
            One-dimensional angular frequencies returned by :meth:`basis`.
        length_scale : array_like
            Positive, finite length scales in the same units as the time
            positions. The preparation range guides approximation sizing
            and does not constrain these values or specify a prior.
        amplitude : array_like, default 1.0
            Nonnegative, finite process standard deviations. Their shape
            must broadcast with ``length_scale``.

        Returns
        -------
        jax.Array
            Coefficient standard deviations with shape
            ``batch_shape + (len(frequencies),)``. Invalid numeric inputs
            give ``nan`` in affected positions, as in :func:`hsgp_weights`.

        Examples
        --------
        .. ipython::

            In [1]: from mmmjax import prepare_hsgp
               ...: config = prepare_hsgp((0, 16), length_scale_range=(2, 8))
               ...: basis, frequencies = config.basis([0.0, 1.0, 2.0])
               ...: weights = config.weights(frequencies, length_scale=4.0)
               ...: weights.shape
        """
        return hsgp_weights(frequencies, length_scale=length_scale, amplitude=amplitude, covariance=self.covariance)


@dataclass(frozen=True, slots=True, eq=False)
class HSGPEffect:
    """Configure a smooth time-varying contribution for model composition.

    Model preparation fixes the approximation domain and basis from the
    observed time positions. The same basis definition and training mean are
    retained for prediction. The coefficient block is unscaled, so a standard
    Normal prior gives the usual HSGP construction after the component applies
    its spectral weights. No priors are assigned automatically.

    Parameters
    ----------
    length_scale_range : tuple of float
        Shortest and longest expected length scales in increasing order. Use
        days for calendar dates and the original units for numeric times. The
        range sizes the approximation and does not constrain the learned
        length scale.
    name : str, default "baseline"
        Name of the time-varying contribution. Named callbacks request its
        parameters through ``<name>_coefficients``, ``<name>_length_scale``,
        and ``<name>_amplitude``.
    covariance : {"expquad", "matern32", "matern52"}, default "matern52"
        Covariance family used to weight the basis coefficients.
    boundary : float, optional
        Fixed domain half-width in time units. When omitted, preparation
        chooses a value from the observed span and expected length scales.
    n_basis : int, optional
        Positive number of basis functions. When omitted, preparation chooses
        a count from the domain and shortest expected length scale.
    demean : bool, default True
        Subtract the training feature mean so the curve has no average
        contribution over the training observations.
    channel_specific : bool, default False
        Give each media channel its own curve, with a shared length scale
        and amplitude. Channel labels come from prepared data. Coefficients
        have shape ``(channel, basis)`` and curves have a final channel axis.
        Curves remain shared across groups. This does not assign priors or
        transform the curves into positive multipliers.

    Examples
    --------
    Configure a smooth baseline whose priors will be written in the model
    callback.

    .. ipython::

        In [1]: from mmmjax import HSGPEffect
           ...: baseline = HSGPEffect(length_scale_range=(14.0, 90.0))
           ...: baseline.name
    """

    length_scale_range: tuple[float, float]
    name: str = "baseline"
    covariance: Literal["expquad", "matern32", "matern52"] = "matern52"
    boundary: float | None = None
    n_basis: int | None = None
    demean: bool = True
    channel_specific: bool = False

    def __post_init__(self) -> None:
        """Validate choices that remain fixed after model preparation."""
        if not isinstance(self.length_scale_range, tuple) or len(self.length_scale_range) != 2:
            raise TypeError("length_scale_range must be a tuple containing two real numbers")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            for value in self.length_scale_range
        ):
            raise TypeError("length_scale_range must contain two real numbers")
        shortest, longest = (float(value) for value in self.length_scale_range)
        if not np.isfinite(shortest) or not np.isfinite(longest):
            raise ValueError("length_scale_range must contain finite values")
        if shortest <= 0 or shortest >= longest:
            raise ValueError("length_scale_range must contain positive values in increasing order")
        object.__setattr__(self, "length_scale_range", (shortest, longest))

        if not isinstance(self.name, str):
            raise TypeError("name must be a string naming the time-varying contribution")
        if not self.name.isidentifier() or iskeyword(self.name):
            raise ValueError("name must be a valid non-keyword Python identifier")
        if self.covariance not in ("expquad", "matern32", "matern52"):
            raise ValueError("covariance must be 'expquad', 'matern32', or 'matern52'")
        if self.boundary is not None:
            if isinstance(self.boundary, (bool, np.bool_)) or not isinstance(
                self.boundary, (int, float, np.integer, np.floating)
            ):
                raise TypeError("boundary must be a real scalar domain half-width")
            boundary = float(self.boundary)
            if not np.isfinite(boundary) or boundary <= 0:
                raise ValueError("boundary must be finite and positive")
            object.__setattr__(self, "boundary", boundary)
        if self.n_basis is not None:
            if isinstance(self.n_basis, bool) or not isinstance(self.n_basis, int):
                raise TypeError("n_basis must be a positive Python integer")
            if self.n_basis <= 0:
                raise ValueError("n_basis must be at least 1")
        if not isinstance(self.demean, bool):
            raise TypeError("demean must be True or False")
        if not isinstance(self.channel_specific, bool):
            raise TypeError("channel_specific must be True or False")

    def _prepare(self, data: PreparedData, *, reference: "_PreparedHSGP | None" = None) -> "_PreparedHSGP":
        """Prepare fixed basis state outside JAX transformations."""
        if not isinstance(data, PreparedData):
            raise TypeError("HSGP effects require PreparedData from prepare_data")
        if self.channel_specific and not data.channels:
            raise ValueError("Channel-specific HSGP effects require media channels selected in prepare_data")
        channels = data.channels if self.channel_specific else ()

        if reference is not None and (
            self is not reference.specification
            or data.time_column != reference.time_column
            or data.group_columns != reference.group_columns
        ):
            raise ValueError("HSGP prediction inputs must retain the training time and group columns")
        if reference is not None and set(data.group_values) != set(reference.group_values):
            raise ValueError("HSGP prediction inputs must contain the same group labels as training")
        if reference is not None and channels != reference.channels:
            raise ValueError("HSGP prediction inputs must retain the training channel labels and ordering")

        positions, origin = _time_positions(data.time_values, origin=None if reference is None else reference.origin)
        if reference is None:
            if len(positions) < 2 or float(np.min(positions)) == float(np.max(positions)):
                raise ValueError("HSGP effects require at least two distinct observation times")
            config = prepare_hsgp(
                np.asarray((np.min(positions), np.max(positions))),
                length_scale_range=np.asarray(self.length_scale_range),
                covariance=self.covariance,
                boundary=self.boundary,
                n_basis=self.n_basis,
            )
            features, frequencies = config.basis(positions)
            basis_mean = jnp.mean(features, axis=0) if self.demean else jnp.zeros(features.shape[1], features.dtype)
        else:
            config = reference.config
            outside_domain = np.abs(positions - config.center) > config.boundary
            if np.any(outside_domain):
                raise ValueError(
                    "HSGP prediction times fall outside the training domain. "
                    "Use a wider boundary when configuring the training model"
                )
            features, frequencies = config.basis(jnp.asarray(positions, dtype=reference.features.dtype))
            basis_mean = reference.basis_mean

        return _PreparedHSGP(
            features=features,
            frequencies=frequencies,
            basis_mean=basis_mean,
            specification=self,
            config=config,
            origin=origin,
            time_column=data.time_column,
            group_columns=data.group_columns,
            group_values=data.group_values if reference is None else reference.group_values,
            channels=channels,
        )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("features", "frequencies", "basis_mean"),
    meta_fields=("specification", "config", "origin", "time_column", "group_columns", "group_values", "channels"),
)
@dataclass(frozen=True, slots=True, eq=False)
class _PreparedHSGP:
    """Carry one fixed HSGP approximation into model callbacks."""

    features: jax.Array
    frequencies: jax.Array
    basis_mean: jax.Array
    specification: HSGPEffect
    config: HSGPConfig
    origin: float | datetime
    time_column: str
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]
    channels: tuple[str, ...]

    @property
    def _coefficient_shape(self) -> tuple[int, ...]:
        """Retain a single parameter block for shared or channel-specific curves."""
        basis_shape = (self.config.n_basis,)
        return (len(self.channels), *basis_shape) if self.specification.channel_specific else basis_shape

    @property
    def parameters(self) -> dict[str, Parameterization]:
        """Declare curve parameters without assigning their priors."""
        name = self.specification.name
        dtype = self.features.dtype
        return {
            f"{name}_coefficients": Real(shape=self._coefficient_shape, dtype=dtype),
            f"{name}_length_scale": Positive(dtype=dtype),
            f"{name}_amplitude": Positive(dtype=dtype),
        }

    def apply(self, parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Return the smooth contribution for each prepared observation."""
        if not isinstance(parameters, Mapping):
            raise TypeError("HSGP parameters must be a mapping of declared names to arrays")

        name = self.specification.name
        shapes = {
            f"{name}_coefficients": self._coefficient_shape,
            f"{name}_length_scale": (),
            f"{name}_amplitude": (),
        }
        values = {}
        for parameter_name, shape in shapes.items():
            if parameter_name not in parameters:
                raise ValueError(f"Missing HSGP parameter {parameter_name!r}")
            try:
                value = jnp.asarray(parameters[parameter_name])
            except (TypeError, ValueError) as error:
                raise TypeError(f"HSGP parameter {parameter_name!r} must be real numeric and array-like") from error
            if not (jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(value.dtype, jnp.integer)):
                raise TypeError(f"HSGP parameter {parameter_name!r} must have a real numeric dtype")
            if value.shape != shape:
                raise ValueError(f"HSGP parameter {parameter_name!r} must have shape {shape}, got shape {value.shape}")
            values[parameter_name] = jnp.asarray(value, dtype=jnp.result_type(value, self.features))

        coefficients = values[f"{name}_coefficients"]
        weights = self.config.weights(
            self.frequencies,
            length_scale=values[f"{name}_length_scale"],
            amplitude=values[f"{name}_amplitude"],
        )
        contribution = (self.features - self.basis_mean) @ (weights * coefficients).T
        if self.group_columns:
            shape = (self.features.shape[0], len(self.group_values), *contribution.shape[1:])
            return jnp.broadcast_to(contribution[:, None, ...], shape)
        return contribution

    def for_data(self, data: PreparedData) -> "_PreparedHSGP":
        """Prepare predictions with the training domain and feature mean."""
        return self.specification._prepare(data, reference=self)


def prepare_hsgp(
    time_range: ArrayLike,
    *,
    length_scale_range: ArrayLike,
    covariance: Literal["expquad", "matern32", "matern52"] = "matern52",
    boundary: float | None = None,
    n_basis: int | None = None,
) -> HSGPConfig:
    """Choose a fixed domain and basis count for a time-varying process.

    Prepare the approximation once before defining the model, outside
    ``jax.jit``. The longest expected length scale determines how far the
    domain extends beyond the observations. The shortest determines how
    many basis functions are needed to represent faster changes.

    These heuristic recommendations can understate variation near the
    ends of the time range. Check that model results remain stable with a
    wider domain and more basis functions. More functions alone cannot
    correct insufficient domain padding. Preparation does not fit a curve
    or define any priors.

    Parameters
    ----------
    time_range : array_like
        Two finite numeric endpoints in increasing order, covering both
        training observations and planned forecasts. For example, use
        ``(0, 116)`` for observations in weeks 0 through 104 and predictions
        through week 116. Keep the same time origin for later predictions.
    length_scale_range : array_like
        Two positive, finite endpoints in increasing order, in the same
        units as ``time_range``. Choose a range covering the length scales
        you expect the model to use, such as most of the prior probability.
        This range guides sizing and does not constrain model parameters.
    covariance : {"expquad", "matern32", "matern52"}, default "matern52"
        Covariance family. The returned settings retain this choice for
        subsequent weight calculations.
    boundary : float, optional
        Domain half-width in time units. When omitted, use the recommended
        padding. Supply a larger value to check boundary sensitivity. It
        must be finite and greater than half the supplied time span.
        The recommended basis count is recalculated for this domain.
    n_basis : int, optional
        Positive number of basis functions. When omitted, choose a count
        using the domain half-width and shortest expected length scale.
        Supply a larger value to check sensitivity to the approximation.

    Returns
    -------
    HSGPConfig
        Reusable settings containing

        - **center** — Midpoint of the supplied time range.
        - **boundary** — Half-width of the padded approximation domain.
        - **n_basis** — Number of basis functions.
        - **covariance** — Covariance family for coefficient weights.

        Use the object's ``basis`` and ``weights`` methods in the model.
        Reuse it for predictions instead of preparing new settings.

    Examples
    --------
    Prepare a weekly time basis with room for four forecast weeks.
    Training and future observations share the same basis definition.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import prepare_hsgp
           ...: frame = pl.DataFrame({
           ...:     "week": [0, 4, 8, 12],
           ...:     "sales": [100.0, 120.0, 110.0, 130.0],
           ...: })
           ...: config = prepare_hsgp((0, 16), length_scale_range=(2, 8))
           ...: basis, frequencies = config.basis(frame["week"].to_numpy())
           ...: weights = config.weights(frequencies, length_scale=4.0)
           ...: future_basis, _ = config.basis([13.0, 14.0, 15.0, 16.0])
           ...: basis.shape, future_basis.shape
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
        if array.shape != (2,):
            raise ValueError(f"{name} must contain two endpoints, got shape {array.shape}")
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

    # One-dimensional sizing recommendations from Ruitort-Mayol et al.
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

    return HSGPConfig(center=center, boundary=domain_width, n_basis=n_basis, covariance=covariance)


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
