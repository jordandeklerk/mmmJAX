"""Distribution-owned bindings for reusable densities and prior sampling."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tensorflow_probability.substrates.jax import distributions as tfd

__all__ = ["custom_distribution"]

type _Distribution = Callable[..., jax.Array]


@dataclass(frozen=True, slots=True)
class _DistributionSpec:
    """Keep a function's numerical operations and parameter-axis metadata together."""

    density: _Distribution
    logpdf: _Distribution
    rng: _Distribution
    parameter_events: tuple[tuple[str, int], ...]
    event_ndims: int
    dimension_parameter: str | None = None


def custom_distribution(
    logpdf: _Distribution,
    rng: _Distribution,
    *,
    name: str | None = None,
    event_ndims: int = 0,
    parameter_event_ndims: Mapping[str, int] | None = None,
) -> _Distribution:
    """Make a distribution written as plain functions usable like the built-in families.

    Write the pointwise log density as ``logpdf(value, *settings)`` and the
    draw function as ``rng(key, *settings, sample_shape=())``. These signatures
    follow the conventions of the built-in families. Both functions must be
    pure JAX code whose settings broadcast against the values.

    The returned function sums the log density over every axis, so it adds to
    the target in ``log_density`` the way ``normal`` or ``gamma`` does.
    ``Prior`` accepts it with fixed settings for prior draws and pointwise
    log-prior terms.

    Parameters
    ----------
    logpdf : callable
        Pointwise log density or log mass. Its first argument is the value
        and the remaining arguments name the distribution settings.
    rng : callable
        Random draws. Its first argument is a JAX random key, the settings
        follow with the same names as in ``logpdf``, and a keyword argument
        ``sample_shape`` with a default of ``()`` prepends independent
        sample axes to the broadcast setting shape.
    name : str, optional
        Name of the returned function. Defaults to the ``logpdf`` name
        without its ``_logpdf`` or ``_logpmf`` suffix.
    event_ndims : int, default 0
        Trailing value axes that form one event. Use zero for scalar
        families and one for vector families such as a Dirichlet. A vector
        family needs at least one setting listed in ``parameter_event_ndims``
        so the event size is known.
    parameter_event_ndims : mapping of str to int, optional
        Trailing event axes of any setting that is a vector per batch entry,
        such as ``{"concentration": 1}``. Unlisted settings are scalars.

    Returns
    -------
    callable
        Summed log density with the ``logpdf`` signature. It carries the
        metadata that ``Prior`` needs.

    Examples
    --------
    A half-Cauchy prior for a scale parameter is written from its pointwise
    log density and a draw function.

    .. ipython::

        In [1]: import jax
           ...: import jax.numpy as jnp
           ...: from mmmjax import Prior, custom_distribution
           ...:
           ...: def half_cauchy_logpdf(value, scale):
           ...:     standardized = value / scale
           ...:     density = jnp.log(2.0 / jnp.pi) - jnp.log(scale) - jnp.log1p(standardized**2)
           ...:     return jnp.where(value < 0, -jnp.inf, density)
           ...:
           ...: def half_cauchy_rng(key, scale, *, sample_shape=()):
           ...:     shape = sample_shape + jnp.shape(jnp.asarray(scale))
           ...:     return jnp.abs(scale * jax.random.cauchy(key, shape))
           ...:
           ...: half_cauchy = custom_distribution(half_cauchy_logpdf, half_cauchy_rng)

    The returned function sums the log density, and a prior binds its setting
    for draws and pointwise terms.

    .. ipython::

        In [2]: half_cauchy(jnp.array([0.5, 2.0]), 1.0)

        In [3]: prior = Prior(half_cauchy, scale=1.0)
           ...: prior.sample(jax.random.key(0), sample_shape=(3,))
    """
    if not callable(logpdf) or not callable(rng):
        raise TypeError("custom_distribution requires callable logpdf and rng functions")
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")
    settings = _setting_names(logpdf, "logpdf", leading="value")
    draw_settings = _setting_names(rng, "rng", leading="key", optional=("sample_shape",))
    if settings != draw_settings:
        raise TypeError(
            f"logpdf settings {list(settings)} and rng settings {list(draw_settings)} must use the same names "
            "in the same order"
        )
    sample_shape = signature(rng).parameters.get("sample_shape")
    if sample_shape is None or sample_shape.default != ():
        raise TypeError("rng must accept a sample_shape keyword argument with a default of ()")
    if isinstance(event_ndims, bool) or not isinstance(event_ndims, int) or event_ndims not in (0, 1):
        raise ValueError("event_ndims must be 0 for scalar values or 1 for vector events")
    ranks = {} if parameter_event_ndims is None else dict(parameter_event_ndims)
    unknown = sorted(set(ranks) - set(settings))
    if unknown:
        raise ValueError(f"parameter_event_ndims names settings {unknown} that logpdf does not take")
    for setting, rank in ranks.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"parameter_event_ndims for {setting!r} must be a nonnegative integer")
    if event_ndims == 1 and not any(rank >= 1 for rank in ranks.values()):
        raise ValueError(
            "A vector distribution needs at least one setting with an event axis in parameter_event_ndims "
            "so the event size is known"
        )
    parameter_events = tuple((setting, ranks.get(setting, 0)) for setting in settings)
    density_name = _density_name(logpdf) if name is None else name

    def density(*arguments: ArrayLike, **keywords: ArrayLike) -> jax.Array:
        pointwise = logpdf(*arguments, **keywords)
        total = jnp.sum(pointwise)
        return total

    density.__name__ = density_name
    density.__qualname__ = density_name
    density.__doc__ = f"Summed log density of the user-defined {density_name} distribution."
    density.__dict__["__signature__"] = signature(logpdf)
    _register(density, logpdf, rng, parameter_events, event_ndims)
    return density


def _setting_names(
    function: _Distribution, label: str, *, leading: str, optional: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Read the distribution settings a user function names after its leading argument."""
    try:
        parameters = list(signature(function).parameters.values())
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a function with an inspectable signature") from error
    positional = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    if not parameters or parameters[0].kind not in positional:
        raise TypeError(f"{label} must take the {leading} as its first positional argument")
    names = []
    for parameter in parameters[1:]:
        if parameter.name in optional:
            continue
        if parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            raise TypeError(f"{label} must name every distribution setting as an argument that accepts keywords")
        names.append(parameter.name)
    settings = tuple(names)
    return settings


def _density_name(logpdf: _Distribution) -> str:
    """Name the summed density after the pointwise function without its suffix."""
    base = getattr(logpdf, "__name__", "distribution")
    for suffix in ("_logpdf", "_logpmf"):
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _register(
    density: _Distribution,
    logpdf: _Distribution,
    rng: _Distribution,
    parameter_events: tuple[tuple[str, int], ...],
    event_ndims: int,
    dimension_parameter: str | None = None,
) -> None:
    """Store the binding on the summed density so Prior can read it back."""
    binding = _DistributionSpec(density, logpdf, rng, parameter_events, event_ndims, dimension_parameter)
    density.__dict__["_mmmjax_distribution"] = binding


def _bind_distribution(
    density: _Distribution,
    logpdf: _Distribution,
    rng: _Distribution,
    backend: type[tfd.Distribution],
    *,
    event_ndims: int = 0,
    dimension_parameter: str | None = None,
    **parameters: str,
) -> None:
    """Attach numerical functions and read event ranks through backend argument names."""
    properties = backend.parameter_properties()
    parameter_events = []
    for name, backend_name in parameters.items():
        ndims = properties[backend_name].event_ndims
        if not isinstance(ndims, int) or ndims < 0:
            raise TypeError(f"Distribution parameter {name!r} requires a fixed nonnegative event rank")
        parameter_events.append((name, ndims))
    _register(density, logpdf, rng, tuple(parameter_events), event_ndims, dimension_parameter)


def _get_distribution_spec(distribution: _Distribution) -> _DistributionSpec:
    """Read a binding from its original distribution function."""
    binding = getattr(distribution, "_mmmjax_distribution", None)
    if not isinstance(binding, _DistributionSpec) or binding.density is not distribution:
        raise TypeError("Prior requires a distribution function exported by mmmjax or returned by custom_distribution")
    return binding
