"""Resolve program block inputs by name from prepared data and parameters."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from inspect import Parameter as SignatureParameter
from inspect import signature
from keyword import iskeyword
from numbers import Integral
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mmmjax.data import Reference
from mmmjax.scaling import Scaling

type _Source = Literal[
    "data", "parameter", "transformed", "reference", "transformed_data", "constant", "builtin", "variable"
]
type _InputBindings = tuple[_Binding, ...]


class _Binding(NamedTuple):
    """Tie one callback argument to the kind of source that supplies it."""

    argument: str
    source: _Source


class _Origin(NamedTuple):
    """Identify the physical source behind a declared or derived input name."""

    kind: _Source
    name: str


class _FrozenMapping[V](Mapping[str, V]):
    """Hashable read-only mapping for static PyTree metadata with constant-time lookups."""

    __slots__ = ("_hash", "_items")

    def __init__(self, items: Mapping[str, V] | Iterable[tuple[str, V]] = ()) -> None:
        self._items: dict[str, V] = dict(items)
        self._hash: int | None = None

    def __getitem__(self, key: str) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._items.items()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenMapping):
            return self._items == other._items
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._items!r})"


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class _ModelData:
    """Keep observation arrays in a model-owned dynamic PyTree.

    Array-valued fields are PyTree children so the bundle passes through JAX
    transformations. Every other field is static metadata, so it must be
    hashable and comparable for the JIT cache.
    """

    values: dict[str, jax.Array]
    owner: object = field(default_factory=object, metadata={"static": True})
    reference: Reference | None = None
    outcome_scaling: Scaling | None = None
    outcome_group_scale: bool = field(default=False, metadata={"static": True})
    n_periods: int = field(default=0, metadata={"static": True})
    reserved_names: frozenset[str] = field(default=frozenset(), metadata={"static": True})
    variable_sources: _FrozenMapping[_Origin] | None = field(default=None, metadata={"static": True})
    transformed_values: dict[str, jax.Array] = field(default_factory=dict)
    static_values: _FrozenMapping[int | bool] = field(default_factory=_FrozenMapping, metadata={"static": True})
    transformed_sources: _FrozenMapping[_Origin] = field(default_factory=_FrozenMapping, metadata={"static": True})
    constants: _FrozenMapping[object] = field(default_factory=_FrozenMapping, metadata={"static": True})

    @property
    def fixed_names(self) -> set[str]:
        """Name every value returned by ``transformed_data``."""
        return set(self.transformed_values) | set(self.static_values)


def _validate_name(name: object, *, label: str) -> None:
    """Require a name usable as a Python keyword argument."""
    if not isinstance(name, str):
        raise TypeError(f"{label} name {name!r} must be a string, got {type(name).__name__}")
    if not name.isidentifier() or iskeyword(name):
        raise ValueError(f"{label} name {name!r} must be a valid non-keyword Python identifier")


def _data_sources(data: _ModelData) -> dict[_Source, set[str]]:
    """List physical data sources separately from user-chosen function names."""
    sources: dict[_Source, set[str]] = {
        "data": set(data.values),
        "reference": {"reference"} if data.reference is not None else set(),
        "builtin": {"n_periods"},
        "constant": set(data.constants),
    }
    if data.outcome_scaling is not None:
        sources["builtin"].add("outcome_scaling")
    return sources


def _resolve_data_variables(declarations: Mapping[str, str], data: _ModelData) -> _FrozenMapping[_Origin]:
    """Validate declarations against prepared and auxiliary data sources."""
    sources = _data_sources(data)
    resolved: dict[str, _Origin] = {}
    for name, source_name in declarations.items():
        matches = [kind for kind, names in sources.items() if source_name in names]
        if not matches:
            raise ValueError(
                f"Data variable {name!r} refers to unavailable source {source_name!r}. "
                "Select it in prepare_data or supply it through inputs"
            )
        if len(matches) != 1:
            raise ValueError(f"Source {source_name!r} for data variable {name!r} is ambiguous")
        resolved[name] = _Origin(matches[0], source_name)
    return _FrozenMapping(resolved)


def _callback_data_names(data: _ModelData) -> set[str]:
    """Identify data names visible to the model's program blocks."""
    fixed = data.fixed_names
    if data.variable_sources is not None:
        return set(data.variable_sources) | fixed | set(data.constants)
    return set().union(*_data_sources(data).values(), data.reserved_names, fixed)


def _input_source(name: str, source: _Source, data: _ModelData) -> _Origin:
    """Resolve a declared argument to its physical source for evaluation and labels."""
    if source != "variable":
        return _Origin(source, name)
    assert data.variable_sources is not None
    try:
        return data.variable_sources[name]
    except KeyError:
        raise ValueError(f"Unknown declared data variable {name!r}") from None


def _metadata_source(name: str, source: _Source, data: _ModelData) -> _Origin:
    """Retain labels for unchanged inputs returned by transformed_data."""
    origin = _input_source(name, source, data)
    if origin.kind == "transformed_data":
        return data.transformed_sources.get(origin.name, origin)
    return origin


def _concrete_array(array: jax.Array) -> np.ndarray[Any, Any] | None:
    """Return the host values of an array or None while it is being traced."""
    try:
        return np.asarray(array)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


def _compute_transformed_data(
    function: Callable[..., object],
    bindings: _InputBindings,
    data: _ModelData,
    parameter_names: tuple[str, ...],
    *,
    expected_names: set[str] | None = None,
) -> _ModelData:
    """Evaluate and validate data-only calculations outside the parameter loop."""
    base = replace(data, transformed_values={}, static_values=_FrozenMapping(), transformed_sources=_FrozenMapping())
    arguments = _callback_inputs(bindings, base, {}, {}, name="transformed_data")
    outputs = function(**arguments)
    if not isinstance(outputs, Mapping):
        raise TypeError("transformed_data must return a mapping from names to fixed values")

    reserved = _callback_data_names(base) | set(parameter_names)
    arrays: dict[str, jax.Array] = {}
    static: dict[str, int | bool] = {}
    origins: dict[str, _Origin] = {}
    for name, value in outputs.items():
        _validate_name(name, label="transformed data")
        if name in reserved:
            raise ValueError(f"Transformed data {name!r} conflicts with an input or parameter")
        if isinstance(value, (bool, np.bool_)):
            static[name] = bool(value)
        elif isinstance(value, Integral):
            static[name] = int(value)
        else:
            arrays[name] = _transformed_array(name, value)

        inherited = {
            _input_source(argument, source, base) for argument, source in bindings if value is arguments[argument]
        }
        # Members of the reference namespace keep their training identity.
        if base.reference is not None:
            inherited.update(
                _Origin("reference", member) for member, array in base.reference.values.items() if value is array
            )
        if len(inherited) == 1:
            origins[name] = inherited.pop()

    if expected_names is not None and set(outputs) != expected_names:
        raise ValueError("transformed_data must return the same names for each dataset")
    return replace(
        data,
        transformed_values=arrays,
        static_values=_FrozenMapping(static),
        transformed_sources=_FrozenMapping(origins),
    )


def _transformed_array(name: str, value: object) -> jax.Array:
    """Convert one transformed_data output to a finite real array."""
    if value is None or callable(value):
        raise TypeError(f"Transformed data {name!r} must contain real numeric values")
    if not isinstance(value, jax.Array):
        try:
            host_value = np.asarray(value)
        except (TypeError, ValueError):
            host_value = None
        if host_value is not None and host_value.dtype == object:
            # NumPy widens ints beyond int64 to object dtype, which JAX would reject with OverflowError
            raise TypeError(f"Transformed data {name!r} must contain real numeric values")
        if host_value is not None and np.issubdtype(host_value.dtype, np.integer):
            canonical_dtype = jax.dtypes.canonicalize_dtype(host_value.dtype)
            if canonical_dtype != host_value.dtype:
                limits = np.iinfo(canonical_dtype)
                if np.any(host_value < limits.min) or np.any(host_value > limits.max):
                    raise ValueError(
                        f"Transformed data {name!r} contains integers outside the JAX dtype range. "
                        "Return floating-point values or enable 64-bit mode"
                    )
    try:
        array = jnp.array(value, copy=not isinstance(value, jax.Array))
    except (TypeError, ValueError) as error:
        raise TypeError(f"Transformed data {name!r} must contain real numeric values") from error
    if array.dtype.kind not in "biuf":
        raise TypeError(f"Transformed data {name!r} must contain real numeric values")
    concrete = _concrete_array(array)
    if concrete is not None and not np.isfinite(concrete).all():
        raise ValueError(f"Transformed data {name!r} must contain finite values")
    return array


def _bind_inputs(
    function: Callable[..., object],
    parameter_names: tuple[str, ...],
    data: _ModelData,
    *,
    name: str,
    leading_key: bool = False,
    allow_parameters: bool = True,
    require_all_parameters: bool = False,
    upstream_parameters: tuple[str, ...] = (),
    has_transformed: bool = False,
) -> _InputBindings:
    """Resolve ordinary or keyword-only callback inputs by name during preparation.

    Parameters
    ----------
    function : callable
        The program block whose signature names its inputs.
    parameter_names : tuple of str
        Declared parameter names available when ``allow_parameters`` is set.
    data : _ModelData
        Prepared inputs whose names the block may request.
    name : str
        Block name used in error messages.
    leading_key : bool, optional
        Whether the first positional argument receives a random key rather than an input.
    allow_parameters : bool, optional
        Whether sampled parameters may be requested. Data-only blocks disallow them.
    require_all_parameters : bool, optional
        Whether every declared parameter must be requested here or upstream.
    upstream_parameters : tuple of str, optional
        Parameters already consumed by ``transformed_parameters``.
    has_transformed : bool, optional
        Whether unknown names may resolve to ``transformed_parameters`` outputs.
    """
    if not callable(function):
        raise TypeError(f"{name} must be callable, got {type(function).__name__}")
    try:
        arguments = list(signature(function).parameters.values())
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose an inspectable Python signature") from error

    key_argument = None
    if leading_key:
        if not arguments or arguments[0].kind not in (
            SignatureParameter.POSITIONAL_ONLY,
            SignatureParameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(f"{name} must accept a random key as its first positional argument")
        key_argument = arguments.pop(0)
    if any(
        argument.kind not in (SignatureParameter.POSITIONAL_OR_KEYWORD, SignatureParameter.KEYWORD_ONLY)
        for argument in arguments
    ):
        raise TypeError(
            f"Inputs for {name} must be named arguments without positional-only parameters, *args or **kwargs"
        )

    sources: dict[_Source, set[str]]
    if data.variable_sources is None:
        sources = _data_sources(data)
    else:
        sources = {"variable": set(data.variable_sources), "constant": set(data.constants)}
    sources["transformed_data"] = data.fixed_names
    if allow_parameters:
        sources["parameter"] = set(parameter_names)
    if key_argument is not None and (
        key_argument.name in data.reserved_names or any(key_argument.name in names for names in sources.values())
    ):
        raise TypeError(f"{name} places input {key_argument.name!r} where the random key is required")

    declared = data.variable_sources is not None
    bindings: list[_Binding] = []
    for argument in arguments:
        matches = [source for source, names in sources.items() if argument.name in names]
        if len(matches) > 1:
            raise ValueError(
                f"{name} input {argument.name!r} is ambiguous across {matches}. "
                "Use distinct names for data roles and parameters"
            )
        if matches:
            bindings.append(_Binding(argument.name, matches[0]))
            continue

        available = sorted(set().union(*sources.values()))
        if not allow_parameters:
            raise ValueError(
                _unknown_input_message(name, argument.name, available, declared=declared)
                + f" Sampled parameters are not available in {name}."
            )
        if argument.name in data.reserved_names:
            raise ValueError(
                f"{name} requests {argument.name!r}, which requires 'outcome' in the original prepared data"
            )
        if has_transformed:
            bindings.append(_Binding(argument.name, "transformed"))
            continue
        if not declared and argument.name in ("data", "effects"):
            raise TypeError(
                f"{name} no longer receives data or effects bundles in prepared models. "
                "Request individual inputs by name, such as outcome or a declared parameter"
            )
        raise ValueError(_unknown_input_message(name, argument.name, available, declared=declared))

    if require_all_parameters:
        requested = {argument.name for argument in arguments}
        missing = sorted(set(parameter_names) - set(upstream_parameters) - requested)
        if missing:
            callbacks = f"{name} or transformed_parameters" if has_transformed else name
            raise ValueError(f"{callbacks} must request every declared parameter. Missing parameters {missing}")
    return tuple(bindings)


def _unknown_input_message(block: str, argument: str, available: list[str], *, declared: bool) -> str:
    """Explain an unbound argument with the names that would have bound."""
    guidance = (
        "Declare its source in Data variables or declare it as a parameter"
        if declared
        else "Use a selected data role, declared constant, or declared parameter"
    )
    return f"{block} requests unknown input {argument!r}. {guidance}. Available inputs are {', '.join(available)}."


def _callback_inputs(
    bindings: _InputBindings,
    data: _ModelData,
    effects: Mapping[str, jax.Array],
    parameters: Mapping[str, object],
    *,
    name: str,
) -> dict[str, object]:
    """Supply current inputs, fixed references, and explicit unit conversions."""
    arguments: dict[str, object] = {}
    for argument, source in bindings:
        origin = _input_source(argument, source, data)
        try:
            arguments[argument] = _lookup_input(origin, data, effects, parameters)
        except LookupError:
            declared = data.variable_sources is not None
            raise ValueError(_missing_input_message(name, argument, origin, declared=declared)) from None
    return arguments


def _lookup_input(
    origin: _Origin,
    data: _ModelData,
    effects: Mapping[str, jax.Array],
    parameters: Mapping[str, object],
) -> object:
    """Read one resolved input from its container or raise LookupError when it is absent."""
    kind, source_name = origin
    match kind:
        case "builtin":
            if source_name == "n_periods":
                return data.n_periods
            if data.outcome_scaling is None:
                raise LookupError(source_name)
            return data.outcome_scaling
        case "reference":
            if data.reference is None:
                raise LookupError(source_name)
            return data.reference
        case "transformed_data":
            if source_name in data.transformed_values:
                return data.transformed_values[source_name]
            return data.static_values[source_name]
        case "data":
            return data.values[source_name]
        case "parameter":
            return parameters[source_name]
        case "transformed":
            return effects[source_name]
        case "constant":
            return data.constants[source_name]
    raise ValueError(f"Unknown input source {kind!r} for {source_name!r}")


def _missing_input_message(name: str, argument: str, origin: _Origin, *, declared: bool) -> str:
    """Explain an input the prepared data or transformed_parameters did not supply."""
    if origin.kind != "transformed":
        return f"{name} requires input {argument!r}. Include it when preparing data for this evaluation"
    if declared:
        return (
            f"{name} requires input {argument!r}. "
            "Return it from transformed_parameters or declare its source in Data variables"
        )
    if argument in ("data", "effects"):
        return (
            f"{name} requires transformed quantity {argument!r}. "
            "Prepared callbacks receive individual inputs by name, not data or effects bundles"
        )
    return f"{name} requires transformed quantity {argument!r}. Return it from transformed_parameters"


def _validate_log_density_signature(
    function: Callable[..., object],
    expected_names: tuple[str, ...],
) -> None:
    """Require a data-first density to name exactly the declared parameters."""
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
    """Return the declared parameters a data-first generator requests.

    The result is None when the generator accepts ``**parameters``.
    """
    actual_names = _model_parameter_names(
        function,
        expected_names,
        name="generated_quantities",
        leading_arguments=("key", "data"),
    )
    if actual_names is None:
        return None

    unexpected = sorted(set(actual_names) - set(expected_names))
    if unexpected:
        details = _name_mismatch_details([], unexpected)
        raise ValueError(f"generated_quantities requests undeclared model parameters. {details}")

    requested_names = set(actual_names)
    return tuple(name for name in expected_names if name in requested_names)


def _model_parameter_names(
    function: Callable[..., object],
    expected_names: tuple[str, ...],
    *,
    name: str,
    leading_arguments: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Read the named parameters after the leading positional arguments of a data-first block."""
    if not callable(function):
        raise TypeError(f"{name} must be callable, got {type(function).__name__}")

    try:
        callable_parameters = list(signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an inspectable Python signature, got {type(function).__name__}") from exc

    callback_example = f"{name}({', '.join(leading_arguments)}, ...)"
    for index, argument in enumerate(leading_arguments):
        position_error = (
            f"{name} must accept {argument} as positional argument {index + 1}. "
            f"Expected a signature like {callback_example}"
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
                f"where {argument} is required. Expected a signature like "
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


def _name_mismatch_details(missing: list[str], unexpected: list[str]) -> str:
    """Describe which names were absent and which were unexpected."""
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    return ". ".join(details)
