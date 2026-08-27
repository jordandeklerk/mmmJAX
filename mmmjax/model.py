"""Model composition for transparent JAX probability models."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter as SignatureParameter
from inspect import signature
from keyword import iskeyword
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.parameters import Parameterization

__all__ = ["Model"]

LogDensity: TypeAlias = Callable[..., ArrayLike]
Generate: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
ParameterValues: TypeAlias = Mapping[str, ArrayLike]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Model:
    """Compose parameter declarations with density and generation functions.

    Parameters
    ----------
    parameters : mapping of str to Parameterization
        Named declarations for every model parameter
    log_density : callable
        Scalar log density in the constrained model space. It receives data
        followed by either every named model parameter or ``**parameters``
    generate : callable, optional
        Generated-quantities function. It receives a JAX random key, data, and
        either the named model parameters it needs or ``**parameters``
    """

    _parameterizations: tuple[tuple[str, Parameterization], ...]
    _log_density: LogDensity
    _generate: Generate | None
    _generate_parameter_names: tuple[str, ...] | None

    def __init__(
        self,
        parameters: Mapping[str, Parameterization],
        log_density: LogDensity,
        generate: Generate | None = None,
    ) -> None:
        """Create a model from named parameter declarations and plain functions."""
        parameterizations = _prepare_parameterizations(parameters)
        parameter_names = tuple(name for name, _ in parameterizations)
        _validate_log_density_signature(log_density, parameter_names)
        generate_parameter_names = None
        if generate is not None:
            generate_parameter_names = _validate_generate_signature(generate, parameter_names)

        object.__setattr__(self, "_parameterizations", parameterizations)
        object.__setattr__(self, "_log_density", log_density)
        object.__setattr__(self, "_generate", generate)
        object.__setattr__(self, "_generate_parameter_names", generate_parameter_names)

    @property
    def parameters(self) -> dict[str, Parameterization]:
        """Return a copy of the named parameter declarations."""
        return dict(self._parameterizations)

    def constrain(self, position: ParameterValues) -> dict[str, jax.Array]:
        """Map a complete unconstrained position into model space."""
        _validate_value_names(position, self._parameterizations, name="position")
        return {name: parameterization.constrain(position[name]) for name, parameterization in self._parameterizations}

    def unconstrain(self, parameters: ParameterValues) -> dict[str, jax.Array]:
        """Map a complete set of model parameters into inference space."""
        _validate_value_names(parameters, self._parameterizations, name="parameters")
        return {
            name: parameterization.unconstrain(parameters[name]) for name, parameterization in self._parameterizations
        }

    def initialize_random(self, key: jax.Array) -> dict[str, jax.Array]:
        """Draw an unconstrained initial position from one JAX random key."""
        keys = jax.random.split(key, len(self._parameterizations))
        return {
            name: parameterization.initialize(parameter_key)
            for (name, parameterization), parameter_key in zip(self._parameterizations, keys, strict=True)
        }

    def log_density(self, position: ParameterValues, data: object) -> jax.Array:
        r"""Evaluate the adjusted scalar log density in inference space.

        For an unconstrained position :math:`z` and parameter mapping
        :math:`\theta = T(z)`, the returned density is

        .. math::

            \log p_z(z) = \log p_\theta(T(z))
            + \sum_k A_k(z_k),

        where :math:`A_k` is the log-density adjustment supplied by each
        parameterization

        ``data`` may be any JAX-compatible PyTree. Passing it explicitly keeps
        the same compiled model reusable across datasets with matching shapes
        and dtypes
        """
        parameters = self.constrain(position)
        density = _as_scalar(self._log_density(data, **parameters), name="log_density")

        for name, parameterization in self._parameterizations:
            adjustment = _as_scalar(
                parameterization.log_density_adjustment(position[name]),
                name=f"log-density adjustment for {name!r}",
            )
            density = density + adjustment

        return density

    def generate(
        self,
        key: jax.Array,
        parameters: ParameterValues,
        data: object,
    ) -> dict[str, jax.Array]:
        """Evaluate generated quantities from constrained model parameters."""
        if self._generate is None:
            raise RuntimeError("generated quantities are unavailable because this model has no generate callback")

        _validate_value_names(parameters, self._parameterizations, name="parameters")
        if self._generate_parameter_names is None:
            callback_parameters = dict(parameters)
        else:
            callback_parameters = {name: parameters[name] for name in self._generate_parameter_names}
        generated = self._generate(key, data, **callback_parameters)
        if not isinstance(generated, Mapping):
            raise TypeError(
                f"generate must return a mapping from quantity names to values, got {type(generated).__name__}"
            )

        for name in generated:
            _validate_name(name, label="generated quantity")

        quantities: dict[str, jax.Array] = {}
        for name, value in sorted(generated.items()):
            try:
                quantities[name] = jnp.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"generated quantity {name!r} must be array-like, got {type(value).__name__}") from exc
        return quantities


def _prepare_parameterizations(
    parameters: Mapping[str, Parameterization],
) -> tuple[tuple[str, Parameterization], ...]:
    if not isinstance(parameters, Mapping):
        raise TypeError(
            f"parameters must be a mapping from names to Parameterization objects, got {type(parameters).__name__}"
        )

    for name in parameters:
        _validate_name(name, label="parameter")
        if not isinstance(parameters[name], Parameterization):
            raise TypeError(
                f"parameter {name!r} must implement Parameterization, got {type(parameters[name]).__name__}"
            )
    return tuple(sorted(parameters.items()))


def _validate_name(name: object, *, label: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{label} name {name!r} must be a string, got {type(name).__name__}")
    if not name.isidentifier() or iskeyword(name):
        raise ValueError(f"{label} name {name!r} must be a valid non-keyword Python identifier")


def _validate_log_density_signature(
    function: Callable[..., object],
    expected_names: tuple[str, ...],
) -> None:
    actual_names = _model_parameter_names(
        function,
        expected_names,
        name="log_density",
        leading_arguments=("data",),
    )
    if actual_names is None:
        return

    missing = sorted(set(expected_names) - set(actual_names))
    unexpected = sorted(set(actual_names) - set(expected_names))
    if missing or unexpected:
        details = _name_mismatch_details(missing, unexpected)
        raise ValueError(f"log_density signature does not match the declared model parameters: {details}")


def _validate_generate_signature(
    function: Callable[..., object],
    expected_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    actual_names = _model_parameter_names(
        function,
        expected_names,
        name="generate",
        leading_arguments=("key", "data"),
    )
    if actual_names is None:
        return None

    unexpected = sorted(set(actual_names) - set(expected_names))
    if unexpected:
        details = _name_mismatch_details([], unexpected)
        raise ValueError(f"generate requests undeclared model parameters: {details}")

    requested_names = set(actual_names)
    return tuple(name for name in expected_names if name in requested_names)


def _model_parameter_names(
    function: Callable[..., object],
    expected_names: tuple[str, ...],
    *,
    name: str,
    leading_arguments: tuple[str, ...],
) -> tuple[str, ...] | None:
    if not callable(function):
        raise TypeError(f"{name} must be callable, got {type(function).__name__}")

    try:
        callable_parameters = list(signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an inspectable Python signature, got {type(function).__name__}") from exc

    callback_example = "log_density(data, ...)" if name == "log_density" else "generate(key, data, ...)"
    for index, argument in enumerate(leading_arguments):
        position_error = (
            f"{name} must accept {argument} as positional argument {index + 1}; "
            f"expected a signature like {callback_example}"
        )
        if not callable_parameters:
            raise TypeError(position_error)
        leading_parameter = callable_parameters.pop(0)
        if leading_parameter.kind not in (
            SignatureParameter.POSITIONAL_ONLY,
            SignatureParameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(position_error)
        if leading_parameter.name in expected_names:
            raise TypeError(
                f"{name} places declared model parameter {leading_parameter.name!r} "
                f"where {argument} is required; expected a signature like "
                f"{callback_example}"
            )

    variadic_keywords = [
        parameter for parameter in callable_parameters if parameter.kind is SignatureParameter.VAR_KEYWORD
    ]
    if variadic_keywords:
        if len(callable_parameters) != 1:
            explicit_names = [
                parameter.name
                for parameter in callable_parameters
                if parameter.kind is not SignatureParameter.VAR_KEYWORD
            ]
            raise TypeError(
                f"{name} cannot combine named model parameters {explicit_names} "
                f"with **{variadic_keywords[0].name}; use named parameters or "
                "one **parameters argument, not both"
            )
        return None

    unsupported = [
        parameter.name
        for parameter in callable_parameters
        if parameter.kind
        not in (
            SignatureParameter.POSITIONAL_OR_KEYWORD,
            SignatureParameter.KEYWORD_ONLY,
        )
    ]
    if unsupported:
        raise TypeError(
            f"{name} has unsupported model parameter arguments {unsupported}; "
            "declare them as named arguments or use **parameters"
        )

    return tuple(parameter.name for parameter in callable_parameters)


def _validate_value_names(
    values: ParameterValues,
    parameterizations: tuple[tuple[str, Parameterization], ...],
    *,
    name: str,
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping from parameter names to values, got {type(values).__name__}")
    invalid_names = [value_name for value_name in values if not isinstance(value_name, str)]
    if invalid_names:
        invalid_name = invalid_names[0]
        raise TypeError(
            f"{name} contains non-string parameter name {invalid_name!r} of type {type(invalid_name).__name__}"
        )

    expected = {parameter_name for parameter_name, _ in parameterizations}
    actual = set(values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = _name_mismatch_details(missing, unexpected)
        raise ValueError(f"{name} does not match the model parameters: {details}")


def _name_mismatch_details(missing: list[str], unexpected: list[str]) -> str:
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    return "; ".join(details)


def _as_scalar(value: ArrayLike, *, name: str) -> jax.Array:
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must return an array-like floating-point scalar, got {type(value).__name__}") from exc
    if array.shape != ():
        raise ValueError(f"{name} must return a scalar, got shape {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must return a real floating-point value, got dtype {array.dtype}")
    return array
