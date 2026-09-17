"""Distribution-owned bindings for reusable densities and prior sampling."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
from tensorflow_probability.substrates.jax import distributions as tfd

_Distribution = Callable[..., jax.Array]


@dataclass(frozen=True, slots=True)
class _DistributionSpec:
    """Keep a function's numerical operations and parameter-axis metadata together."""

    density: _Distribution
    logpdf: _Distribution
    rng: _Distribution
    parameter_events: tuple[tuple[str, int], ...]
    event_ndims: int
    dimension_parameter: str | None = None


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

    binding = _DistributionSpec(density, logpdf, rng, tuple(parameter_events), event_ndims, dimension_parameter)
    density.__dict__["_mmmjax_distribution"] = binding


def _get_distribution_spec(distribution: _Distribution) -> _DistributionSpec:
    """Read a binding from its original distribution function."""
    binding = getattr(distribution, "_mmmjax_distribution", None)
    if not isinstance(binding, _DistributionSpec) or binding.density is not distribution:
        raise TypeError("Prior requires a registered mmmjax scalar distribution function")
    return binding
