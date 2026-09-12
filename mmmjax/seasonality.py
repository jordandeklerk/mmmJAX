"""Seasonal components and features for recurring patterns over time."""

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from keyword import iskeyword

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from mmmjax.data import PreparedData, _time_positions
from mmmjax.parameters import Real

__all__ = ["FourierSeasonality", "fourier_features"]


@dataclass(frozen=True, slots=True, eq=False)
class FourierSeasonality:
    """Specify a repeating seasonal contribution for model composition.

    Choose the cycle and its flexibility. Model preparation converts dates,
    builds Fourier features, and declares coefficients. The time reference
    is retained for prediction. This configuration does not fit a model.

    Write coefficient priors in the model's ``log_density`` callback.
    Request ``<name>_coefficients`` for the coefficient array and ``<name>``
    for the seasonal effect.

    Parameters
    ----------
    period : {"yearly", "monthly", "weekly"} or float, default "yearly"
        Length of the repeating cycle. Named periods require calendar
        dates and represent fixed durations of 365.25, 365.25 / 12, and
        7 days. A numeric period uses days for dated inputs or the original
        time units for numeric observation labels.
    order : int, default 2
        Positive number of harmonics. Higher orders allow more detailed
        seasonal patterns. Each harmonic adds a sine and a cosine
        coefficient. No intercept is included.
    name : str, default "seasonality"
        Coefficient parameter name and seasonal effect name during model
        composition. Named callbacks request raw coefficients through
        ``<name>_coefficients``. Use distinct names for seasonal components.
    group_specific_coefficients : bool, default False
        Whether each prepared group has its own coefficients. The default
        shares one curve across all groups. This does not add hierarchical
        pooling.

    Examples
    --------
    Configure an annual pattern whose coefficient prior will be written in
    the model callback using ``annual_coefficients``. No dates, feature
    matrices, or coefficient shapes need to be supplied here.

    .. ipython::

        In [1]: from mmmjax import FourierSeasonality
           ...: annual = FourierSeasonality(
           ...:     period="yearly", order=3, name="annual",
           ...: )
           ...: annual.period, annual.order
    """

    period: str | float = "yearly"
    order: int = 2
    name: str = "seasonality"
    group_specific_coefficients: bool = False

    def __post_init__(self) -> None:
        """Validate the static choices before preparing a component."""
        if isinstance(self.period, str):
            if self.period not in ("yearly", "monthly", "weekly"):
                raise ValueError("period must be 'yearly', 'monthly', 'weekly', or a positive numeric cycle length")
        else:
            if isinstance(self.period, (bool, np.bool_)) or not isinstance(
                self.period, (int, float, np.integer, np.floating)
            ):
                raise TypeError("period must be a named cycle or a real scalar cycle length")
            if not np.isfinite(self.period) or self.period <= 0:
                raise ValueError(f"period must be finite and positive, got {self.period!r}")
            object.__setattr__(self, "period", float(self.period))
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise TypeError("order must be a positive Python integer")
        if self.order <= 0:
            raise ValueError(f"order must be at least 1, got {self.order}")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string naming the component's coefficients")
        if not self.name.isidentifier() or iskeyword(self.name):
            raise ValueError(f"name must be a valid non-keyword Python identifier, got {self.name!r}")
        if not isinstance(self.group_specific_coefficients, bool):
            raise TypeError("group_specific_coefficients must be True or False")

    def _prepare(self, data: PreparedData, *, reference: "_PreparedFourier | None" = None) -> "_PreparedFourier":
        """Prepare numeric state for composition outside JAX transformations."""
        if not isinstance(data, PreparedData):
            raise TypeError("seasonality requires PreparedData. Use prepare_data with the observation dataframe")
        if self.group_specific_coefficients and not data.group_columns:
            raise ValueError("group_specific_coefficients=True requires grouped data. Select groups in prepare_data")
        if reference is not None and (
            data.time_column != reference.time_column or data.group_columns != reference.group_columns
        ):
            raise ValueError("Prediction time and group columns must match those used for training")

        positions, origin = _time_positions(data.time_values, origin=None if reference is None else reference.origin)
        if isinstance(self.period, str):
            if not isinstance(origin, datetime):
                raise ValueError(
                    "Named seasonal periods require calendar dates. "
                    "Use a numeric period in the units of the time column"
                )
            period = {"yearly": 365.25, "monthly": 365.25 / 12, "weekly": 7.0}[self.period]
        else:
            period = self.period

        group_values = data.group_values
        group_indices = list(range(len(group_values)))
        if reference is not None and self.group_specific_coefficients:
            training_groups = reference.group_values
            if set(group_values) != set(training_groups):
                raise ValueError("Group-specific seasonality requires the same groups in training and prediction")
            # Keep coefficients in training order while matching the new
            # observations' order, without rearranging any other model input.
            group_indices = [training_groups.index(group) for group in group_values]
            group_values = training_groups

        return _PreparedFourier(
            features=fourier_features(
                positions if reference is None else jnp.asarray(positions, dtype=reference.features.dtype),
                period=period,
                order=self.order,
            ),
            group_indices=jnp.asarray(group_indices, dtype=jnp.int32),
            specification=self,
            origin=origin,
            time_column=data.time_column,
            group_columns=data.group_columns,
            group_values=group_values,
        )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("features", "group_indices"),
    meta_fields=("specification", "origin", "time_column", "group_columns", "group_values"),
)
@dataclass(frozen=True, slots=True, eq=False)
class _PreparedFourier:
    """Carry numerical features and a fixed time reference into model callbacks."""

    features: jax.Array
    group_indices: jax.Array
    specification: FourierSeasonality
    origin: float | datetime
    time_column: str
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]

    @property
    def parameters(self) -> dict[str, Real]:
        """Declare the named coefficient block without assigning a prior."""
        return {self.specification.name: Real(shape=self._coefficient_shape, dtype=self.features.dtype)}

    @property
    def _coefficient_shape(self) -> tuple[int, ...]:
        shape = (self.features.shape[1],)
        return (*shape, len(self.group_values)) if self.specification.group_specific_coefficients else shape

    def apply(self, coefficients: ArrayLike) -> jax.Array:
        """Return the seasonal contribution in the prepared observation order."""
        values = self._coefficients(coefficients)
        if self.specification.group_specific_coefficients:
            return self.features @ values[:, self.group_indices]
        contribution = self.features @ values
        if self.group_columns:
            return jnp.broadcast_to(contribution[:, None], (self.features.shape[0], len(self.group_values)))
        return contribution

    def for_data(self, data: PreparedData) -> "_PreparedFourier":
        """Prepare predictions without changing the training time reference or coefficients."""
        return self.specification._prepare(data, reference=self)

    def _coefficients(self, coefficients: ArrayLike) -> jax.Array:
        """Validate the coefficient block for the seasonal contribution."""
        try:
            values = jnp.asarray(coefficients)
        except (TypeError, ValueError) as error:
            raise TypeError("Seasonality coefficients must be real numeric and array-like") from error
        if not (jnp.issubdtype(values.dtype, jnp.floating) or jnp.issubdtype(values.dtype, jnp.integer)):
            raise TypeError("Seasonality coefficients must have a real numeric dtype")
        if values.shape != self._coefficient_shape:
            raise ValueError(
                f"Seasonality coefficients must have shape {self._coefficient_shape}, got shape {values.shape}"
            )
        return jnp.asarray(values, dtype=jnp.result_type(values, self.features))


def fourier_features(time: ArrayLike, *, period: ArrayLike, order: int) -> jax.Array:
    r"""Build sine and cosine features for a repeating seasonal pattern.

    For time positions :math:`t`, cycle length :math:`P > 0`, and harmonics
    :math:`k = 1, \ldots, K`, the features are

    .. math::

        \sin\left(\frac{2\pi kt}{P}\right), \qquad
        \cos\left(\frac{2\pi kt}{P}\right).

    Multiply the features by model coefficients to obtain a seasonal
    contribution. Coefficients and their priors are specified separately.

    Parameters
    ----------
    time : array_like
        One-dimensional, finite numeric positions for the modeling periods.
        For dated observations, use elapsed time from a fixed reference date.
        Keep that reference date and the time units unchanged for prediction.
        Negative and fractional positions are accepted.
    period : array_like
        Positive, finite scalar giving the cycle length in the same units as
        ``time``. For example, use 7 for weekly seasonality when time is in
        days, or 365.25 for an annual cycle measured in days.
    order : int
        Positive number of harmonics. Each adds a sine and a cosine term.
        Higher orders allow more detailed patterns within a cycle. Keep
        this argument static when using ``jax.jit``.

    Returns
    -------
    jax.Array
        Features with shape ``(len(time), 2 * order)``. Sine terms come first
        in increasing harmonic order, followed by cosine terms in the same
        order. No intercept column is included. Values use a common floating
        dtype of at least float32. Invalid numeric inputs produce ``nan`` in
        the affected rows.

        A coefficient vector of shape ``(2 * order,)`` gives one shared
        seasonal curve through ``features @ coefficients``. Coefficients
        shaped ``(2 * order, group)`` give separate group curves without
        duplicating the features.

    Examples
    --------
    Build annual features from weekly dates and combine them with illustrative
    coefficients. Reuse ``origin`` when preparing later dates for prediction.

    .. ipython::

        In [1]: from datetime import date
           ...: import jax.numpy as jnp
           ...: import numpy as np
           ...: import polars as pl
           ...: from mmmjax import fourier_features
           ...: frame = pl.DataFrame({
           ...:     "week": [date(2026, 1, 1), date(2026, 1, 8),
           ...:              date(2026, 1, 15)],
           ...:     "sales": [100.0, 120.0, 110.0],
           ...: })
           ...: origin = date(2026, 1, 1)
           ...: days = (frame["week"] - origin).dt.total_days().to_numpy()
           ...: features = fourier_features(days, period=365.25, order=2)
           ...: coefficients = jnp.array([0.2, -0.1, 0.3, 0.1])
           ...: seasonal = features @ coefficients
           ...: frame.with_columns(
           ...:     pl.Series("seasonality", np.asarray(seasonal)),
           ...: )
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be a positive Python integer and stay fixed during JIT compilation")
    if order <= 0:
        raise ValueError(f"order must be at least 1, got {order}")

    leaves = []
    for name, value in (("time", time), ("period", period)):
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
    time_array = jnp.asarray(time, dtype=dtype)
    period_array = jnp.asarray(period, dtype=dtype)
    if time_array.ndim != 1:
        raise ValueError(f"time must be one-dimensional, got shape {time_array.shape}")
    if period_array.ndim != 0:
        raise ValueError(f"period must be a scalar cycle length, got shape {period_array.shape}")

    valid = jnp.isfinite(time_array) & jnp.isfinite(period_array) & (period_array > 0)
    safe_time = jnp.where(valid, time_array, 0.0)
    safe_period = jnp.where(jnp.isfinite(period_array) & (period_array > 0), period_array, 1.0)
    harmonics = jnp.arange(1, order + 1, dtype=dtype)
    angles = (2 * jnp.pi * (safe_time / safe_period))[:, None] * harmonics
    features = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    return jnp.where(valid[:, None], features, jnp.nan)
