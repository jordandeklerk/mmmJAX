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
from mmmjax.distributions import beta, half_normal, uniform
from mmmjax.parameters import Interval, Parameterization, Positive
from mmmjax.saturation import hill_saturation, log_saturation, logistic_saturation, root_saturation

__all__ = ["MediaEffect", "media_response", "reach_frequency_response"]


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class MediaEffect:
    """Configure channel contributions with a choice of carryover and response curve.

    Each channel has a positive contribution coefficient and the parameters
    required by the selected transformations. Preparation infers their shapes
    from the media columns. Transformations use the current parameter values
    at each model evaluation.

    Include this configuration in ``Model`` components. Its named callback
    input contains weighted contributions for each channel. Request
    ``paid_media_total`` for their sum at each period and group, or
    ``paid_media`` to retain individual channels. Both are JAX arrays.
    The model declares parameters and adds their priors automatically.
    Set ``automatic_priors=False`` to write priors directly in the named
    ``Model.log_density`` callback, requesting constrained parameter arrays such as
    ``paid_media_coefficient`` and ``paid_media_retention``.

    This configuration does not scale exposures. Set coefficient priors on
    the scale where contributions enter the model, and half-saturation priors
    in the prepared exposure units. Their defaults are intended for scaled
    data and are not suitable for every dataset.

    Exposure gradients at zero use a zero-gradient convention for Hill
    slopes and root exponents below one. Evaluate marginal responses at
    positive exposures for these curves.

    Parameters
    ----------
    max_lag : int
        Number of previous observation periods to include in carryover.
        The current period is also included. Must be nonnegative. Supply
        regularly spaced media observations and include earlier exposures
        through ``media_history`` when preparing data.
    name : str, default "paid_media"
        Name of the per-channel contribution. Named callbacks can also
        request ``<name>_total``. Parameter names use this prefix, such as
        ``paid_media_retention``. Use a name distinct from data roles and
        other components.
    adstock : callable, default geometric_adstock
        Choose :func:`geometric_adstock`, :func:`delayed_adstock`,
        :func:`weibull_pdf_adstock`, or :func:`weibull_cdf_adstock` directly.
        Geometric uses ``retention``, delayed adds ``delay``, and Weibull
        uses ``adstock_shape`` and ``adstock_scale``. With ``max_lag=0``,
        delayed adstock fixes the delay at zero rather than sampling it.
    saturation : callable, default hill_saturation
        Choose :func:`hill_saturation`, :func:`logistic_saturation`,
        :func:`root_saturation`, or :func:`log_saturation` directly.
        Hill uses ``half_saturation`` and ``slope``, logistic uses
        ``half_saturation``, root uses ``exponent``, and log has no curve
        parameters. Only priors for selected parameters may be supplied.
    normalize : bool, default True
        Divide the adstock weights by their sum over the full lag window.
    adstock_first : bool, default True
        Apply carryover before saturation. Set to ``False`` to reverse
        their order. Earlier history is retained through both operations.
    group_specific : bool, default False
        Give each group its own channel contribution coefficients. Adstock
        and saturation parameters remain shared across groups. Coefficients
        receive independent priors by default, without hierarchical pooling.
    coefficient_prior : callable, optional
        Prior for positive channel contribution coefficients. The
        default is HalfNormal with scale 1.5. A supplied function receives
        the coefficient array and returns a scalar log density or matching
        elementwise log densities. The same convention applies to all priors.
    retention_prior : callable, optional
        Prior for each channel's retention rate between zero and one.
        The default is Beta with shape parameters 1 and 3.
    delay_prior : callable, optional
        Prior for delayed adstock's peak in observation periods. The
        default is Uniform between zero and ``max_lag``.
    adstock_shape_prior : callable, optional
        Prior for positive Weibull shapes. The default is HalfNormal
        with scale 1.5.
    adstock_scale_prior : callable, optional
        Prior for positive Weibull scales in observation periods. The
        default is HalfNormal with scale 1.5.
    half_saturation_prior : callable, optional
        Prior for positive half-saturation points in the prepared exposure
        units. The default is HalfNormal with scale 1.5. Override it when
        the exposure scale calls for different thresholds.
    slope_prior : callable, optional
        Prior for positive Hill slopes. The default is HalfNormal with
        scale 1.5.
    exponent_prior : callable, optional
        Prior for root exponents between zero and one. The default is
        Uniform over this interval.
    automatic_priors : bool, default True
        Add the component's prior terms to the model log density. Set to
        ``False`` when the model callback supplies these priors. Custom
        ``*_prior`` callbacks cannot be combined with ``False``. Parameter
        constraints and media contributions remain the same.
    """

    max_lag: int
    name: str = "paid_media"
    adstock: Callable[..., jax.Array] = geometric_adstock
    saturation: Callable[..., jax.Array] = hill_saturation
    normalize: bool = True
    adstock_first: bool = True
    group_specific: bool = False
    coefficient_prior: Callable[[jax.Array], ArrayLike] | None = None
    retention_prior: Callable[[jax.Array], ArrayLike] | None = None
    delay_prior: Callable[[jax.Array], ArrayLike] | None = None
    adstock_shape_prior: Callable[[jax.Array], ArrayLike] | None = None
    adstock_scale_prior: Callable[[jax.Array], ArrayLike] | None = None
    half_saturation_prior: Callable[[jax.Array], ArrayLike] | None = None
    slope_prior: Callable[[jax.Array], ArrayLike] | None = None
    exponent_prior: Callable[[jax.Array], ArrayLike] | None = None
    automatic_priors: bool = True

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
        for field in ("normalize", "adstock_first", "group_specific", "automatic_priors"):
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
        for role in (
            "coefficient",
            "retention",
            "delay",
            "adstock_shape",
            "adstock_scale",
            "half_saturation",
            "slope",
            "exponent",
        ):
            field = f"{role}_prior"
            prior = getattr(self, field)
            if prior is not None and not callable(prior):
                raise TypeError(f"{field} must be a callable accepting the parameter array")
            if prior is not None and not self.automatic_priors:
                raise ValueError(f"{field} cannot be supplied when automatic_priors=False")
            if prior is not None and role not in self._parameter_roles:
                raise ValueError(f"{field} is not used by the selected transformations and max_lag")

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
        if self.group_specific and not data.group_columns:
            raise ValueError("group_specific=True requires grouped data. Select groups in prepare_data")
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
            (len(self.group_values), *channel_shape) if self.specification.group_specific else channel_shape
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

    def log_prior(self, parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Add one prior per parameter block, independent of observation count."""
        values = self._parameter_values(parameters)
        total = jnp.zeros((), dtype=self.dtype)
        if not self.specification.automatic_priors:
            return total
        for role, value in values.items():
            prior = getattr(self.specification, f"{role}_prior")
            if prior is None:
                if role == "retention":
                    result = beta(value, alpha=1.0, beta=3.0)
                elif role == "delay":
                    result = uniform(value, lower=0.0, upper=float(self.specification.max_lag))
                elif role == "exponent":
                    result = uniform(value, lower=0.0, upper=1.0)
                else:
                    result = half_normal(value, scale=1.5)
            else:
                result = prior(value)
            try:
                density = jnp.asarray(result)
            except (TypeError, ValueError) as error:
                raise TypeError(f"The {role} prior must return real numeric log densities") from error
            if not (jnp.issubdtype(density.dtype, jnp.floating) or jnp.issubdtype(density.dtype, jnp.integer)):
                raise TypeError(f"The {role} prior must return real numeric log densities")
            if density.shape not in ((), value.shape):
                raise ValueError(
                    f"The {role} prior must return a scalar or shape {value.shape}, got shape {density.shape}"
                )
            total = total + jnp.sum(density)
        return total

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

    Both transformations receive the full exposure window, including any
    earlier history. Historical periods are removed from the response only
    after both transformations have been applied. Scaling and multiplication
    by model coefficients remain separate steps.

    Parameters
    ----------
    media : array_like
        Exposure values with time on the first axis. Prepared inputs use
        ``(time, channel)`` or ``(time, group, channel)``. Include any earlier
        exposure history needed by the carryover transformation.
    adstock : callable
        Function accepting exposure values and applying carryover along
        axis zero. It must preserve the input shape. Supply its parameters
        with a function or ``functools.partial``.
    saturation : callable
        Function accepting exposure values and applying a response curve
        without changing the input shape. Channel or group-channel
        parameters can be supplied in the same way as for ``adstock``.
        For learned parameters, define the callbacks inside the model's
        log-density function using its current parameter values.
    n_periods : int, optional
        Number of final periods to return. Use ``len(data.time_values)``
        for data returned by :func:`prepare_data`, including prediction
        data without outcomes. Must be positive and cannot exceed the
        supplied number of exposure periods. If omitted, all periods are
        returned, including history.
    adstock_first : bool, default True
        Apply carryover before saturation. Set to ``False`` to apply
        saturation before carryover. Callbacks, transformation order, and
        ``n_periods`` must be fixed when compiling a model function.

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
        Finite, nonnegative audience reached, with time on the first axis.
        Prepared inputs use ``(time, channel)`` or ``(time, group, channel)``.
        Supply either raw reach or reach transformed using training scales,
        including any earlier history.
    frequency : array_like
        Finite, nonnegative average exposures per person reached. Must
        match the shape and ordering of ``reach``, including history.
        Keep frequency in its original units when using fitted data scaling.
        Use :func:`prepare_data` to validate and order both inputs.
    adstock : callable
        Function applying carryover along axis zero without changing the
        input shape. It receives reach multiplied by saturated frequency.
    saturation : callable
        Function applying a response curve to frequency without changing
        its shape. Express thresholds such as ``half_saturation`` in
        frequency units, not reach or impression units. Supply parameters
        with a function or ``functools.partial``. For learned parameters,
        define the callbacks inside the model's log-density function using
        its current values.
    n_periods : int, optional
        Number of final modeling periods to return. Use
        ``len(data.time_values)`` for prepared data. Must be positive and
        cannot exceed the supplied exposure periods. If omitted, all
        periods are returned, including history. Callbacks and ``n_periods``
        must be fixed when compiling a model function.

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
