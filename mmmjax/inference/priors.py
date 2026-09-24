"""Reusable probability terms and independent prior draws."""

from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from mmmjax.distributions._distribution import _Distribution, _DistributionSpec, _get_distribution_spec
from mmmjax.distributions._utils import _as_real_array, _random_shape

__all__ = ["Prior"]


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class Prior:
    """Bind fixed distribution settings for density evaluation and sampling.

    Calling a prior evaluates its summed log density. Include that call
    explicitly in the model density. Pass the same objects by parameter name
    to ``sample_prior`` for independent prior draws, and use ``logpdf`` for
    pointwise log-prior terms in ``generated_quantities``.

    Parameters
    ----------
    distribution : callable
        A distribution function exported by ``mmmjax``, such as ``normal``,
        ``lognormal``, or ``dirichlet``, or one returned by
        ``custom_distribution`` for a distribution written as plain functions.
    **parameters : array_like
        All named distribution settings. Construction copies them in the
        precision in effect at that moment, so enable JAX 64-bit mode before
        building priors as with any other array. Multinomial priors also
        require ``trials`` and reject counts with another total. LKJ priors
        accept optional ``dimension`` for direct ``sample`` calls. A model's
        parameter declaration supplies it otherwise.

    Examples
    --------
    Bind the settings of a Normal prior once.

    .. ipython::

        In [1]: import jax
           ...: import jax.numpy as jnp
           ...: from mmmjax import Prior, normal

        In [2]: prior = Prior(normal, location=0.0, scale=1.0)
           ...: values = jnp.array([-0.5, 0.0, 0.5])

    Calling the prior returns the summed log density to add to a model
    density, and ``logpdf`` returns one term per value.

    .. ipython::

        In [3]: prior(values), prior.logpdf(values)

    Draws put the sample shape first.

    .. ipython::

        In [4]: prior.sample(jax.random.key(0), sample_shape=(4,))
    """

    _spec: _DistributionSpec
    _parameters: tuple[tuple[str, jax.Array], ...]
    _batch_shape: tuple[int, ...]
    _event_shape: tuple[int, ...] | None
    _dimension: int | None

    def __init__(self, distribution: _Distribution, **parameters: ArrayLike) -> None:
        """Copy settings for a registered distribution and validate their shapes."""
        spec = _get_distribution_spec(distribution)

        dimension = None
        if spec.dimension_parameter is not None and spec.dimension_parameter in parameters:
            supplied_dimension = parameters.pop(spec.dimension_parameter)
            if isinstance(supplied_dimension, bool) or not isinstance(supplied_dimension, int):
                raise TypeError("dimension must be a positive integer")
            if supplied_dimension < 1:
                raise ValueError("dimension must be positive")
            dimension = supplied_dimension

        required = {name for name, _ in spec.parameter_events}
        missing = required - parameters.keys()
        unknown = parameters.keys() - required
        distribution_name = getattr(distribution, "__name__", "distribution")
        if missing:
            raise TypeError(f"Missing parameters for {distribution_name} are {sorted(missing)}")
        if unknown:
            raise TypeError(f"Unknown parameters for {distribution_name} are {sorted(unknown)}")

        copied = {
            name: jnp.array(_as_real_array(name, parameters[name]), copy=True) for name, _ in spec.parameter_events
        }
        batch_shapes = []
        event_sizes = set()
        for name, ndims in spec.parameter_events:
            shape = copied[name].shape
            if ndims:
                if len(shape) < ndims or any(size == 0 for size in shape[-ndims:]):
                    raise ValueError(f"Prior parameter {name!r} must include {ndims} nonempty event dimensions")
                if ndims == 2 and shape[-2] != shape[-1]:
                    raise ValueError(f"Prior parameter {name!r} must end in a square matrix")
                event_sizes.add(shape[-1])
                batch_shapes.append(shape[:-ndims])
            else:
                batch_shapes.append(shape)
        if spec.event_ndims == 1 and len(event_sizes) != 1:
            raise ValueError("Prior distribution parameters must have matching event sizes")
        try:
            batch_shape = jnp.broadcast_shapes(*batch_shapes)
        except ValueError as exc:
            raise ValueError(f"Prior distribution parameter batch shapes {batch_shapes} cannot broadcast") from exc
        event_shape: tuple[int, ...] | None = ()
        if spec.event_ndims == 1:
            event_shape = (event_sizes.pop(),)
        elif spec.event_ndims == 2:
            event_shape = None if dimension is None else (dimension, dimension)

        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_parameters", tuple(copied.items()))
        object.__setattr__(self, "_batch_shape", batch_shape)
        object.__setattr__(self, "_event_shape", event_shape)
        object.__setattr__(self, "_dimension", dimension)

    @property
    def event_ndims(self) -> int:
        """Return the number of value axes reduced by ``logpdf``.

        Returns
        -------
        int
            Zero for scalar distributions, one for vector events, and two
            for LKJ Cholesky factors.
        """
        return self._spec.event_ndims

    def __call__(self, value: ArrayLike) -> jax.Array:
        """Return the scalar log density summed over all batch dimensions.

        Parameters
        ----------
        value : array_like
            Values compatible with the bound distribution settings.

        Returns
        -------
        jax.Array
            Summed log density or probability mass, including constants.
        """
        result = jnp.sum(self.logpdf(value))
        return result

    def logpdf(self, value: ArrayLike) -> jax.Array:
        """Return log densities or masses summed over distribution event axes only.

        Parameters
        ----------
        value : array_like
            Values compatible with the bound distribution settings.

        Returns
        -------
        jax.Array
            Normalized log densities or masses with the broadcast batch shape.
            Scalar distributions retain every value axis. Vector and matrix
            distributions reduce their event axes.
        """
        self._validate_value_event(value)
        result = self._spec.logpdf(value, **dict(self._parameters))
        return result

    def sample(self, key: jax.Array, *, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw with sample dimensions prepended to the bound batch and event shape.

        Parameters
        ----------
        key : jax.Array
            Random key for this draw. Use a fresh key for independent draws.
        sample_shape : tuple of int, default ()
            Independent sample dimensions. Keep this argument static when
            using ``jax.jit``.

        Returns
        -------
        jax.Array
            Draws with shape ``sample_shape + batch_shape + event_shape``.
        """
        parameters: dict[str, ArrayLike] = dict(self._parameters)
        if self._spec.dimension_parameter is not None:
            if self._dimension is None:
                raise ValueError(
                    f"Direct prior sampling requires {self._spec.dimension_parameter}. "
                    "Model infers it from the declaration"
                )
            parameters[self._spec.dimension_parameter] = self._dimension
        result = self._spec.rng(key, sample_shape=sample_shape, **parameters)
        return result

    def _validate_value_event(self, value: ArrayLike) -> None:
        """Check an explicitly bound matrix dimension before evaluating a density."""
        if self._dimension is None:
            return
        # jnp.shape deprecates lists and np.shape cannot read a list of tracers
        value_shape = _as_real_array("value", value).shape
        if value_shape[-2:] != self._event_shape:
            raise ValueError(f"value must end in event shape {self._event_shape}, got shape {value_shape}")

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        """Check that the prior can generate the complete declared value shape."""
        _random_shape(shape)
        if len(shape) < self.event_ndims:
            raise ValueError(f"Prior requires {self.event_ndims} event dimensions, got declared shape {shape}")
        event_shape = shape[-self.event_ndims :] if self.event_ndims else ()
        if self.event_ndims == 2 and (event_shape[0] != event_shape[1] or event_shape[0] < 1):
            raise ValueError(f"Matrix prior requires a nonempty square event, got declared shape {shape}")
        if self._event_shape is not None and event_shape != self._event_shape:
            raise ValueError(f"Prior event shape {self._event_shape} does not match declared shape {shape}")
        batch_shape = shape[: -self.event_ndims] if self.event_ndims else shape
        try:
            broadcast = jnp.broadcast_shapes(self._batch_shape, batch_shape)
        except ValueError as exc:
            raise ValueError(
                f"Prior batch shape {self._batch_shape} cannot broadcast to declared shape {shape}"
            ) from exc
        if broadcast != batch_shape:
            raise ValueError(f"Prior batch shape {self._batch_shape} cannot broadcast to declared shape {shape}")

    def _sample(self, key: jax.Array, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
        """Draw exactly the declared shape with any existing batch axes."""
        self._validate_shape(shape)
        batch_shape = shape[: -self.event_ndims] if self.event_ndims else shape
        ndims_by_name = dict(self._spec.parameter_events)
        parameters: dict[str, ArrayLike] = {}
        parameter_dtype = dtype if jnp.issubdtype(jnp.dtype(dtype), jnp.floating) else None

        for name, value in self._parameters:
            ndims = ndims_by_name[name]
            event_shape = value.shape[-ndims:] if ndims else ()
            parameters[name] = jnp.broadcast_to(jnp.asarray(value, dtype=parameter_dtype), batch_shape + event_shape)

        if self._spec.dimension_parameter is not None:
            parameters[self._spec.dimension_parameter] = shape[-1]

        samples = self._spec.rng(key, **parameters)
        samples = jnp.asarray(samples, dtype=dtype)
        return samples


def _validate_prior_sampler(prior: object) -> None:
    """Check a one-key callback without executing it or inferring its density."""
    if isinstance(prior, Prior) or isinstance(getattr(prior, "__self__", None), Prior):
        raise TypeError(
            "prior must be a prior-draw function supporting prior(key) and returning a parameter mapping. "
            "Pass Prior objects to sample_prior by parameter name instead of a density or sampling method"
        )
    if isinstance(prior, (str, bytes, Sequence)) and not callable(prior):
        raise TypeError(
            "prior requires Prior definitions or a prior-draw function, not output names. "
            "Return log-prior terms under log_prior in generated_quantities"
        )
    if not callable(prior):
        raise TypeError("prior must be a mapping of parameter names to Prior objects or a prior-draw function")

    binding = getattr(prior, "_mmmjax_distribution", None)
    if isinstance(binding, _DistributionSpec) and binding.density is prior:
        raise TypeError(
            "prior received a log-density function rather than a prior-draw function. "
            "Use Prior with its distribution settings in the Model prior mapping, "
            "or supply prior(key) returning a mapping of parameter names to constrained draws"
        )

    try:
        prior_signature = signature(prior)
    except (TypeError, ValueError):
        # Opaque callables are still checked when their draws are evaluated.
        return

    try:
        prior_signature.bind(object())
    except TypeError as exc:
        raise TypeError(
            "prior must support prior(key) with one positional random key and no other required inputs. "
            "Return a mapping of parameter names to constrained draws, not a log density"
        ) from exc
