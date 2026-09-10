"""Composition of carryover and saturation for media exposures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from keyword import iskeyword

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from mmmjax.adstock import delayed_adstock, geometric_adstock, weibull_cdf_adstock, weibull_pdf_adstock
from mmmjax.data import PreparedData
from mmmjax.parameters import Interval, Parameterization, Positive
from mmmjax.saturation import hill_saturation, log_saturation, logistic_saturation, root_saturation

__all__ = ["MediaEffect", "media_response", "reach_frequency_response"]


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class MediaEffect:
    """Configure channel contributions with a choice of carryover and response curve.

    Add this configuration to ``Model`` components. Each channel has a
    positive contribution coefficient and the selected curve parameters,
    with shapes inferred from the prepared media columns.
    For custom transformations or parameterizations, declare parameters on
    ``Model`` and call :func:`media_response` from ``transformed_parameters``.

    Write parameter priors in the model's ``log_density`` callback.
    Exposures are not scaled here. Choose coefficient priors in contribution
    units and half-saturation priors in the prepared exposure units.

    Exposure gradients at zero use a zero-gradient convention for Hill
    slopes and root exponents below one. Evaluate marginal responses at
    positive exposures for these curves.

    Parameters
    ----------
    max_lag : int
        Nonnegative number of previous periods to include alongside the
        current period. Observations must be regularly spaced. Supply earlier
        exposures through ``media_history`` when preparing data.
    name : str, default "paid_media"
        Name of the weighted per-channel array. Named callbacks request
        ``<name>_total`` for its channel sum or parameter names such as
        ``paid_media_retention`` for their values. Keep the name distinct
        from data roles and other components.
    adstock : callable, default geometric_adstock
        Choose :func:`geometric_adstock`, :func:`delayed_adstock`,
        :func:`weibull_pdf_adstock`, or :func:`weibull_cdf_adstock` directly.
        Geometric uses ``retention``, delayed adds ``delay``, and Weibull
        uses ``adstock_shape`` and ``adstock_scale``. Delayed adstock fixes
        delay at zero when ``max_lag=0``.
    saturation : callable, default hill_saturation
        Choose :func:`hill_saturation`, :func:`logistic_saturation`,
        :func:`root_saturation`, or :func:`log_saturation` directly.
        Hill uses ``half_saturation`` and ``slope``, logistic uses
        ``half_saturation``, root uses ``exponent``, and log has no curve
        parameters.
    normalize : bool, default True
        Divide the adstock weights by their sum over the full lag window.
    adstock_first : bool, default True
        Apply carryover before saturation. Set to ``False`` to reverse
        their order. Earlier history is retained through both operations.
    group_specific_coefficients : bool, default False
        Give each group its own channel coefficients while sharing curve
        parameters across groups. This does not add hierarchical pooling.
    """

    max_lag: int
    name: str = "paid_media"
    adstock: Callable[..., jax.Array] = geometric_adstock
    saturation: Callable[..., jax.Array] = hill_saturation
    normalize: bool = True
    adstock_first: bool = True
    group_specific_coefficients: bool = False

    def __post_init__(self) -> None:
        """Validate configuration before preparing channel parameters."""
        if isinstance(self.max_lag, bool) or not isinstance(self.max_lag, int):
            raise TypeError("max_lag must be a nonnegative Python integer")
        if self.max_lag < 0:
            raise ValueError("max_lag must be nonnegative")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string naming the media contribution")
        if not self.name.isidentifier() or iskeyword(self.name):
            raise ValueError("name must be a valid non-keyword Python identifier")
        for field in ("normalize", "adstock_first", "group_specific_coefficients"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be True or False")
        for field, supported in (
            ("adstock", (geometric_adstock, delayed_adstock, weibull_pdf_adstock, weibull_cdf_adstock)),
            ("saturation", (hill_saturation, logistic_saturation, root_saturation, log_saturation)),
        ):
            selected = getattr(self, field)
            if not callable(selected):
                raise TypeError(f"{field} must be a supported transformation function")
            if not any(selected is function for function in supported):
                choices = ", ".join(function.__name__ for function in supported)
                raise ValueError(f"Choose {field} directly from {choices}. Use media_response for custom functions")

    @property
    def _parameter_roles(self) -> tuple[str, ...]:
        """Identify the learned parameters for the selected transformations."""
        roles: tuple[str, ...] = ("coefficient",)
        if self.adstock is geometric_adstock:
            roles += ("retention",)
        elif self.adstock is delayed_adstock:
            roles += ("retention", "delay") if self.max_lag > 0 else ("retention",)
        else:
            roles += ("adstock_shape", "adstock_scale")
        if self.saturation is hill_saturation:
            roles += ("half_saturation", "slope")
        elif self.saturation is logistic_saturation:
            roles += ("half_saturation",)
        elif self.saturation is root_saturation:
            roles += ("exponent",)
        return roles

    def _prepare(self, data: PreparedData, *, reference: "_PreparedMedia | None" = None) -> "_PreparedMedia":
        """Retain channel identities and shapes without capturing exposure arrays."""
        if not isinstance(data, PreparedData):
            raise TypeError("Media effects require PreparedData. Use prepare_data with the exposure dataframe")
        if "media" not in data.arrays or not data.channels:
            raise ValueError("Media effects require exposure columns selected with media in prepare_data")
        if self.group_specific_coefficients and not data.group_columns:
            raise ValueError("group_specific_coefficients=True requires grouped data. Select groups in prepare_data")
        if reference is not None and (
            self is not reference.specification
            or data.time_column != reference.time_column
            or data.columns["media"] != reference.media_columns
            or data.channels != reference.channels
            or data.group_columns != reference.group_columns
            or data.group_values != reference.group_values
        ):
            raise ValueError("Media prediction inputs must retain the training time column, channel and group ordering")

        n_periods = len(data.time_values)
        n_media_periods = len(data.media_time_values)
        if n_periods == 0 or n_media_periods < n_periods or data.media_time_values[-n_periods:] != data.time_values:
            raise ValueError("Media periods must include the modeling periods after any earlier exposure history")
        group_shape = (len(data.group_values),) if data.group_columns else ()
        media_shape = (n_media_periods, *group_shape, len(data.channels))
        if data.arrays["media"].shape != media_shape:
            raise ValueError(
                f"Prepared media must have shape {media_shape} to match its time, group and channel labels"
            )

        return _PreparedMedia(
            specification=self,
            n_periods=n_periods,
            media_shape=media_shape,
            time_column=data.time_column,
            media_columns=data.columns["media"],
            channels=data.channels,
            group_columns=data.group_columns,
            group_values=data.group_values,
            dtype=np.dtype(jax.dtypes.canonicalize_dtype(float)) if reference is None else reference.dtype,
        )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=(),
    meta_fields=(
        "specification",
        "n_periods",
        "media_shape",
        "time_column",
        "media_columns",
        "channels",
        "group_columns",
        "group_values",
        "dtype",
    ),
)
@dataclass(frozen=True, slots=True, eq=False)
class _PreparedMedia:
    """Keep static media layout separate from dynamic exposures and parameters."""

    specification: MediaEffect
    n_periods: int
    media_shape: tuple[int, ...]
    time_column: str
    media_columns: tuple[str, ...]
    channels: tuple[str, ...]
    group_columns: tuple[str, ...]
    group_values: tuple[tuple[object, ...], ...]
    dtype: np.dtype[np.floating]

    @property
    def _parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        channel_shape = (len(self.channels),)
        coefficient_shape = (
            (len(self.group_values), *channel_shape)
            if self.specification.group_specific_coefficients
            else channel_shape
        )
        return {
            role: coefficient_shape if role == "coefficient" else channel_shape
            for role in self.specification._parameter_roles
        }

    @property
    def parameters(self) -> dict[str, Parameterization]:
        """Declare constraints for each channel parameter block without assigning priors."""
        parameters: dict[str, Parameterization] = {}
        for role, shape in self._parameter_shapes.items():
            name = f"{self.specification.name}_{role}"
            if role in ("retention", "exponent"):
                parameters[name] = Interval(0.0, 1.0, shape=shape, dtype=self.dtype)
            elif role == "delay":
                parameters[name] = Interval(0.0, float(self.specification.max_lag), shape=shape, dtype=self.dtype)
            else:
                parameters[name] = Positive(shape=shape, dtype=self.dtype)
        return parameters

    def apply(self, media: ArrayLike, parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Return weighted channel responses over the modeling periods."""
        if np.shape(media) != self.media_shape:
            raise ValueError(f"Media inputs must have the prepared shape {self.media_shape}, got {np.shape(media)}")
        values = self._parameter_values(parameters)

        def carryover(exposures: ArrayLike) -> jax.Array:
            if self.specification.adstock is geometric_adstock:
                arguments = {"alpha": values["retention"]}
            elif self.specification.adstock is delayed_adstock:
                arguments = {
                    "alpha": values["retention"],
                    "theta": values.get("delay", jnp.zeros((), dtype=self.dtype)),
                }
            else:
                arguments = {"shape": values["adstock_shape"], "scale": values["adstock_scale"]}
            return self.specification.adstock(
                exposures,
                **arguments,
                max_lag=self.specification.max_lag,
                normalize=self.specification.normalize,
            )

        def saturate(exposures: ArrayLike) -> jax.Array:
            if self.specification.saturation is logistic_saturation:
                return logistic_saturation(exposures, values["half_saturation"])
            if self.specification.saturation is root_saturation:
                return root_saturation(exposures, values["exponent"])
            if self.specification.saturation is log_saturation:
                return log_saturation(exposures)
            exposures = jnp.asarray(exposures)
            zero_origin = (exposures == 0) & (values["slope"] < 1)
            # Sublinear curves have no finite exposure derivative at the origin.
            # Use zero there so fixed-zero histories do not contaminate parameter gradients.
            response = hill_saturation(
                jnp.where(zero_origin, 1.0, exposures), values["half_saturation"], values["slope"]
            )
            # Multiplication retains NaNs from invalid parameters instead of hiding them.
            return jnp.where(zero_origin, response * 0.0, response)

        response = media_response(
            media,
            adstock=carryover,
            saturation=saturate,
            n_periods=self.n_periods,
            adstock_first=self.specification.adstock_first,
        )
        return response * values["coefficient"]

    def for_data(self, data: PreparedData) -> "_PreparedMedia":
        """Prepare aligned observations with the training parameter identities and precision."""
        return self.specification._prepare(data, reference=self)

    def _parameter_values(self, parameters: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
        """Validate parameter block shapes without imposing extra numerical support checks."""
        if not isinstance(parameters, Mapping):
            raise TypeError("Media parameters must be a mapping of declared names to arrays")
        values = {}
        for role, shape in self._parameter_shapes.items():
            name = f"{self.specification.name}_{role}"
            if name not in parameters:
                raise ValueError(f"Missing media parameter {name!r}")
            try:
                value = jnp.asarray(parameters[name])
            except (TypeError, ValueError) as error:
                raise TypeError(f"Media parameter {name!r} must be real numeric and array-like") from error
            if not (jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(value.dtype, jnp.integer)):
                raise TypeError(f"Media parameter {name!r} must have a real numeric dtype")
            if value.shape != shape:
                raise ValueError(f"Media parameter {name!r} must have shape {shape}, got shape {value.shape}")
            values[role] = jnp.asarray(value, dtype=jnp.result_type(value, self.dtype))
        return values


def media_response(
    media: ArrayLike,
    *,
    adstock: Callable[[ArrayLike], ArrayLike],
    saturation: Callable[[ArrayLike], ArrayLike],
    n_periods: int | None = None,
    adstock_first: bool = True,
) -> jax.Array:
    """Apply carryover and saturation, retaining the requested modeling periods.

    Apply both transformations to the full exposure history before selecting
    modeling periods. Scaling and coefficient multiplication remain separate.

    Parameters
    ----------
    media : array_like
        Exposure values with time on the first axis. Prepared inputs use
        ``(time, channel)`` or ``(time, group, channel)``. Include any earlier
        exposure history needed by the carryover transformation.
    adstock : callable
        Function applying carryover along axis zero while preserving shape.
        Supply parameters with a function or ``functools.partial``.
    saturation : callable
        Shape-preserving response curve with parameters supplied as for
        ``adstock``. For learned parameters, define both callbacks inside
        ``transformed_parameters`` or the log density using current values.
    n_periods : int, optional
        Number of final periods to return, such as ``len(data.time_values)``.
        Must be positive and not exceed the exposure window. If omitted,
        return all periods, including history.
    adstock_first : bool, default True
        Apply carryover before saturation, or reverse the order with ``False``.
        Callbacks, this flag, and ``n_periods`` must stay fixed when compiling.

    Returns
    -------
    jax.Array
        Media responses over the selected periods with all other axes
        unchanged. Precision and numeric domain behavior follow the supplied
        transformations. Channel contributions are not summed.

    Examples
    --------
    Include earlier impressions when calculating carryover, then return
    one response for each modeling period.

    .. ipython::

        In [1]: from functools import partial
           ...: import numpy as np
           ...: import polars as pl
           ...: import mmmjax as mm
           ...: frame = pl.DataFrame({
           ...:     "week": [2, 3, 4],
           ...:     "sales": [120.0, 140.0, 110.0],
           ...:     "video": [200.0, 400.0, 100.0],
           ...: })
           ...: history = pl.DataFrame({"week": [1], "video": [100.0]})
           ...: data = mm.prepare_data(
           ...:     frame, time="week", outcome="sales", media=["video"],
           ...:     media_history=history,
           ...: )
           ...: response = mm.media_response(
           ...:     data.arrays["media"],
           ...:     adstock=partial(mm.geometric_adstock, alpha=0.5, max_lag=2),
           ...:     saturation=partial(
           ...:         mm.hill_saturation, half_saturation=200.0, slope=1.0,
           ...:     ),
           ...:     n_periods=len(data.time_values),
           ...: )
           ...: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response[:, 0])),
           ...: )
    """
    if not callable(adstock):
        raise TypeError("adstock must be a function that preserves the media shape")
    if not callable(saturation):
        raise TypeError("saturation must be a function that preserves the media shape")
    if not isinstance(adstock_first, bool):
        raise TypeError("adstock_first must be a static boolean")

    media_shape = np.shape(media)
    if not media_shape:
        raise ValueError("media must have at least one dimension with time on the first axis")
    if n_periods is not None:
        if isinstance(n_periods, bool) or not isinstance(n_periods, int):
            raise TypeError("n_periods must be a static positive integer or None")
        if not 0 < n_periods <= media_shape[0]:
            raise ValueError(
                f"n_periods must be between 1 and the {media_shape[0]} supplied exposure periods, got {n_periods}. "
                "Use the number of modeling periods without counting earlier history"
            )
    else:
        n_periods = media_shape[0]

    transformations = (("adstock", adstock), ("saturation", saturation))
    if not adstock_first:
        transformations = transformations[::-1]

    response = media
    for name, transform in transformations:
        response = transform(response)
        response_shape = np.shape(response)
        if response_shape != media_shape:
            raise ValueError(
                f"{name} must preserve media shape {media_shape}, got {response_shape}. "
                "Keep all exposure periods and retain the group and channel axes"
            )

    # Keep historical exposures available to both transformations before selecting modeling periods
    return jnp.asarray(response)[media_shape[0] - n_periods :]


def reach_frequency_response(
    reach: ArrayLike,
    frequency: ArrayLike,
    *,
    adstock: Callable[[ArrayLike], ArrayLike],
    saturation: Callable[[ArrayLike], ArrayLike],
    n_periods: int | None = None,
) -> jax.Array:
    r"""Apply saturation to frequency, weight by reach, then apply carryover.

    For reach :math:`r` and frequency :math:`f`, the response is

    .. math::

        \operatorname{Adstock}\bigl(r \odot S(f)\bigr),

    where :math:`S` is the supplied saturation function and
    :math:`\odot` denotes elementwise multiplication. Earlier reach and
    frequency contribute to carryover before the modeling periods are
    selected. The same calculation supports paid and organic channels.

    Parameters
    ----------
    reach : array_like
        Finite, nonnegative audience reached, including earlier history.
        Prepared shapes are ``(time, channel)`` or ``(time, group, channel)``.
        Supply raw reach or reach transformed using fitted scales.
    frequency : array_like
        Finite, nonnegative average exposures per person, matching the shape
        and order of ``reach``. Keep its original units when scaling inputs.
        Use :func:`prepare_data` to validate and order reach and frequency.
    adstock : callable
        Function applying carryover along axis zero without changing the
        input shape. It receives reach multiplied by saturated frequency.
    saturation : callable
        Shape-preserving response curve with thresholds in frequency units.
        Supply parameters with a function or ``functools.partial``. For
        learned parameters, define callbacks inside ``transformed_parameters``
        or the log density using current values.
    n_periods : int, optional
        Number of final modeling periods to return, such as ``len(data.time_values)``.
        Must be positive and not exceed the exposure window. If omitted,
        return all periods. Callbacks and ``n_periods`` must stay fixed when compiling.

    Returns
    -------
    jax.Array
        Reach-weighted responses over the selected periods, with other
        axes unchanged. Reach and saturated frequency are multiplied using
        a common floating-point dtype of at least float32. Scaling, model
        coefficients, and channel summation remain separate.

    Examples
    --------
    Use earlier audience measurements to calculate responses for later
    periods, including one with no new exposure.

    .. ipython::

        In [1]: from functools import partial
           ...: import numpy as np
           ...: import polars as pl
           ...: import mmmjax as mm
           ...: frame = pl.DataFrame({
           ...:     "week": [2, 3, 4],
           ...:     "video_reach": [1_000, 1_500, 0],
           ...:     "video_frequency": [2.0, 3.0, 0.0],
           ...: })
           ...: history = pl.DataFrame({
           ...:     "week": [1], "video_reach": [800],
           ...:     "video_frequency": [1.5],
           ...: })
           ...: data = mm.prepare_data(
           ...:     frame, time="week", reach=["video_reach"],
           ...:     media_frequency=["video_frequency"], media_history=history,
           ...: )
           ...: response = mm.reach_frequency_response(
           ...:     data.arrays["reach"], data.arrays["media_frequency"],
           ...:     adstock=partial(mm.geometric_adstock, alpha=0.5, max_lag=2),
           ...:     saturation=partial(
           ...:         mm.hill_saturation, half_saturation=2.0, slope=1.0,
           ...:     ),
           ...:     n_periods=len(data.time_values),
           ...: )
           ...: frame.with_columns(
           ...:     pl.Series("video_response", np.asarray(response[:, 0])),
           ...: )
    """
    reach_shape, frequency_shape = np.shape(reach), np.shape(frequency)
    if not reach_shape:
        raise ValueError("reach must have at least one dimension with time on the first axis")
    if reach_shape != frequency_shape:
        raise ValueError(
            f"reach shape {reach_shape} and frequency shape {frequency_shape} must match. "
            "Supply the same periods, groups and channels, including earlier history"
        )
    if not callable(adstock):
        raise TypeError("adstock must be a function that preserves the reach and frequency shape")

    reach_leaves = jax.tree_util.tree_leaves(reach)
    try:
        reach_dtype = jnp.result_type(*reach_leaves)
    except (TypeError, ValueError) as error:
        raise TypeError("reach must be real numeric and array-like") from error
    if not (
        jnp.issubdtype(reach_dtype, jnp.floating)
        or jnp.issubdtype(reach_dtype, jnp.integer)
        or reach_dtype == jnp.bool_
    ):
        raise TypeError(f"reach must have a real numeric dtype, got {reach_dtype}")

    def carry_response(saturated_frequency: ArrayLike) -> ArrayLike:
        # Convert integer reach directly to floating point before multiplying
        dtype = jnp.result_type(*reach_leaves, *jax.tree_util.tree_leaves(saturated_frequency))
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
        dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
        weighted = jnp.asarray(reach, dtype=dtype) * jnp.asarray(saturated_frequency, dtype=dtype)
        return adstock(weighted)

    return media_response(
        frequency,
        adstock=carry_response,
        saturation=saturation,
        n_periods=n_periods,
        adstock_first=False,
    )
