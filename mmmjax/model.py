"""Model composition for transparent JAX probability models."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter as SignatureParameter
from inspect import signature
from keyword import iskeyword
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from mmmjax.data import PreparedData, _DataLayout
from mmmjax.media import MediaEffect, _PreparedMedia
from mmmjax.parameters import Parameterization
from mmmjax.seasonality import FourierSeasonality, _PreparedFourier

__all__ = ["Model"]

LogDensity: TypeAlias = Callable[..., ArrayLike]
Generate: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
TransformedParameters: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
ParameterValues: TypeAlias = Mapping[str, ArrayLike]
_InputBindings: TypeAlias = tuple[tuple[str, str], ...]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class _ModelData:
    """Keep observation arrays and prepared components in one dynamic PyTree."""

    values: dict[str, jax.Array]
    components: tuple[_PreparedFourier | _PreparedMedia, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Model:
    """Compose parameter declarations with density and generation functions.

    Prepared models support keyword-only inputs named for data roles, parameters,
    components, or transformed quantities. Positional callbacks receive
    ``data, effects`` for prepared models or ``data`` otherwise, followed
    by declared parameters.

    Parameters
    ----------
    parameters : mapping of str to Parameterization
        Named parameter declarations and constraints. Do not include
        parameters declared automatically by components.
    log_density : callable
        Scalar log density for constrained parameters. Every user-declared
        parameter must be requested here or by ``transformed_parameters``.
        Component priors and constraint adjustments are added automatically.
    generate : callable, optional
        Function returning a mapping of names to array-like generated quantities.
        Receives a JAX random key first, followed by the model inputs it needs.
    data : PreparedData, optional
        Training observations from ``prepare_data``, required with
        ``components``. Callback inputs use data roles such as ``outcome``,
        not source column names.
    components : sequence of FourierSeasonality or MediaEffect, optional
        Named effects with automatic parameters and priors. Media effects
        retain per-channel contributions. Component names must be unique and
        distinct from user parameters. Use ``components=[]`` without effects.
    transformed_parameters : callable, optional
        Pure JAX-compatible function returning a mapping of names to derived
        array-like quantities shared by both callbacks. Requires prepared data
        and keyword-only named inputs. Output names must not shadow data roles,
        parameters, or components. Outputs are not sampled parameters.
        The whole function runs for generation, so all its requested inputs
        are needed for prediction.
    """

    _parameterizations: tuple[tuple[str, Parameterization], ...]
    _log_density: LogDensity
    _generate: Generate | None
    _transformed_parameters: TransformedParameters | None
    _transform_inputs: _InputBindings
    _generate_parameter_names: tuple[str, ...] | None
    _callback_parameter_names: tuple[str, ...]
    _density_inputs: _InputBindings | None
    _generation_inputs: _InputBindings | None
    _data: _ModelData | None
    _layout: _DataLayout | None
    _time_column: str | None
    _dtype: DTypeLike

    def __init__(
        self,
        parameters: Mapping[str, Parameterization],
        log_density: LogDensity,
        generate: Generate | None = None,
        *,
        data: PreparedData | None = None,
        components: Sequence[FourierSeasonality | MediaEffect] | None = None,
        transformed_parameters: TransformedParameters | None = None,
    ) -> None:
        """Create a model from named parameter declarations and plain functions."""
        parameterizations = _prepare_parameterizations(parameters)
        parameter_names = tuple(name for name, _ in parameterizations)
        prepared_data = None
        if components is None:
            if data is not None:
                raise ValueError("Constructor data requires components. Use an empty sequence for no effects")
        else:
            if not isinstance(data, PreparedData):
                raise TypeError("Components require PreparedData. Use prepare_data with the observation dataframe")
            if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
                raise TypeError("components must be a sequence of FourierSeasonality or MediaEffect configurations")
            declarations = dict(parameterizations)
            specifications = tuple(components)
            names = set(parameter_names)
            for specification in specifications:
                if not isinstance(specification, (FourierSeasonality, MediaEffect)):
                    raise TypeError("Each component must be a FourierSeasonality or MediaEffect configuration")
                if specification.name in names:
                    raise ValueError(
                        f"Component name {specification.name!r} conflicts with another component or parameter"
                    )
                names.add(specification.name)

            prepared_components = tuple(specification._prepare(data) for specification in specifications)
            component_names = {specification.name for specification in specifications}
            for component in prepared_components:
                component_parameters = component.parameters
                conflicts = set(component_parameters) & set(declarations)
                conflicts |= set(component_parameters) & (component_names - {component.specification.name})
                if conflicts:
                    raise ValueError(
                        f"Component parameter names {sorted(conflicts)} conflict with other parameters or components"
                    )
                declarations.update(component_parameters)
            parameterizations = _prepare_parameterizations(declarations)
            prepared_data = _ModelData(data._to_jax(), prepared_components)

        transform_inputs: _InputBindings = ()
        if transformed_parameters is not None:
            if prepared_data is None:
                raise ValueError("transformed_parameters requires prepared data and components, which may be empty")
            bindings = _bind_inputs(
                transformed_parameters, parameter_names, prepared_data, name="transformed_parameters"
            )
            if bindings is None:
                raise TypeError("transformed_parameters must be callable with explicit keyword-only inputs")
            transform_inputs = bindings

        upstream_parameters = tuple(name for name, source in transform_inputs if source == "parameter")
        density_inputs = (
            None
            if prepared_data is None
            else _bind_inputs(
                log_density,
                parameter_names,
                prepared_data,
                name="log_density",
                upstream_parameters=upstream_parameters,
                has_transformed=transformed_parameters is not None,
            )
        )
        if density_inputs is None:
            if transformed_parameters is not None:
                raise TypeError("log_density must use explicit keyword-only inputs with transformed_parameters")
            _validate_log_density_signature(log_density, parameter_names, has_components=components is not None)
        generate_parameter_names = None
        generation_inputs = None
        if generate is not None:
            generation_inputs = (
                None
                if prepared_data is None
                else _bind_inputs(
                    generate,
                    parameter_names,
                    prepared_data,
                    name="generate",
                    has_transformed=transformed_parameters is not None,
                )
            )
            if generation_inputs is None:
                if transformed_parameters is not None:
                    raise TypeError(
                        "generate must use a positional random key and keyword-only inputs with transformed_parameters"
                    )
                generate_parameter_names = _validate_generate_signature(
                    generate, parameter_names, has_components=components is not None
                )

        object.__setattr__(self, "_parameterizations", parameterizations)
        object.__setattr__(self, "_log_density", log_density)
        object.__setattr__(self, "_generate", generate)
        object.__setattr__(self, "_transformed_parameters", transformed_parameters)
        object.__setattr__(self, "_transform_inputs", transform_inputs)
        object.__setattr__(self, "_generate_parameter_names", generate_parameter_names)
        object.__setattr__(self, "_callback_parameter_names", parameter_names)
        object.__setattr__(self, "_density_inputs", density_inputs)
        object.__setattr__(self, "_generation_inputs", generation_inputs)
        object.__setattr__(self, "_data", prepared_data)
        object.__setattr__(self, "_layout", None if data is None else data._layout())
        object.__setattr__(self, "_time_column", None if data is None else data.time_column)
        object.__setattr__(self, "_dtype", jax.dtypes.canonicalize_dtype(float))

    @property
    def parameters(self) -> dict[str, Parameterization]:
        """Return a copy of the named parameter declarations."""
        return dict(self._parameterizations)

    @property
    def data(self) -> object:
        """Return prepared training inputs for density and generation calls.

        Pass this object as the ``data`` argument to ``log_density`` or
        ``generate``, including inside JAX transformations. Observation
        arrays are copied during construction, so later edits to the
        original dataframe or prepared data do not change the model inputs.

        Returns
        -------
        object
            JAX-compatible input bundle containing observation arrays and
            prepared component features. Callbacks receive only the arrays
            and evaluated effects, without needing to unpack this bundle.
            Available when ``components`` was supplied at construction.
        """
        if self._data is None:
            raise RuntimeError("This model has no prepared data. Pass your data directly when evaluating it")
        return _ModelData(dict(self._data.values), self._data.components)

    def prepare_data(self, data: PreparedData) -> object:
        """Prepare new observations using the model's training configuration.

        Use this before evaluating a model with new observations or making
        predictions. It retains the training seasonal phase, parameter
        declarations, and priors without changing the model's stored data.
        Preparation does not apply scaling. Reuse any fitted scaling before
        passing the data here, just as for the training observations.

        Parameters
        ----------
        data : PreparedData
            New observations returned by ``prepare_data``. Use the training
            time and group column names and the same set of groups. Supplied
            inputs must use the same source columns and channel assignments
            as training, but their order may differ. Inputs may be omitted
            when neither the callback being evaluated, a configured
            component, nor ``transformed_parameters`` requires them.
            This allows prediction without outcomes.
            Models with MediaEffect require media exposures, with any needed
            carryover history supplied through ``media_history``. Training
            exposures are not prepended automatically. A seasonal-only
            prediction needs just dates and any group labels.

        Returns
        -------
        object
            JAX-compatible inputs for ``log_density`` or ``generate`` with
            arrays and component features in training group and channel
            order. The time axis follows the new observations. This bundle
            is independent of the model's training inputs and later edits
            to the supplied data. Create it outside JAX transformations.
        """
        if self._data is None or self._layout is None:
            raise RuntimeError("This model has no prepared training data. Pass your data directly when evaluating it")
        if not isinstance(data, PreparedData):
            raise TypeError("Model data must be PreparedData. Use prepare_data with the observation dataframe")
        if data.time_column != self._time_column:
            raise ValueError(f"The time column must match the training column {self._time_column!r}")

        aligned = data._align_to(self._layout)
        values = aligned._to_jax(dtype=self._dtype)
        components = tuple(component.for_data(aligned) for component in self._data.components)
        return _ModelData(values, components)

    def constrain(self, position: ParameterValues) -> dict[str, jax.Array]:
        """Map a complete unconstrained position into model space.

        Parameters
        ----------
        position : mapping of str to array_like
            Unconstrained values for every declared parameter, with each
            value matching its declaration's ``position_shape``.

        Returns
        -------
        dict of str to jax.Array
            Dictionary mapping every declared parameter name to its
            constrained array with the declaration's ``shape``.
        """
        _validate_value_names(position, self._parameterizations, name="position")
        return {name: parameterization.constrain(position[name]) for name, parameterization in self._parameterizations}

    def unconstrain(self, parameters: ParameterValues) -> dict[str, jax.Array]:
        """Map a complete set of model parameters into inference space.

        Parameters
        ----------
        parameters : mapping of str to array_like
            Constrained values for every declared parameter, with each
            value matching its declaration's ``shape`` and constraints.

        Returns
        -------
        dict of str to jax.Array
            Dictionary mapping every declared parameter name to its
            unconstrained array with the declaration's ``position_shape``.
        """
        _validate_value_names(parameters, self._parameterizations, name="parameters")
        return {
            name: parameterization.unconstrain(parameters[name]) for name, parameterization in self._parameterizations
        }

    def initialize_random(self, key: jax.Array) -> dict[str, jax.Array]:
        """Draw an unconstrained initial position from one JAX random key.

        Parameters
        ----------
        key : jax.Array
            JAX random key, split internally across parameter declarations.
            Use a fresh key for each independent initialization.

        Returns
        -------
        dict of str to jax.Array
            Dictionary mapping every declared parameter name to its
            unconstrained initial array with the declaration's ``position_shape``.
        """
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
        parameterization. With components, :math:`p_\theta` includes their
        parameter priors as well as the callback's density.

        For models without prepared data, ``data`` may be any JAX-compatible PyTree.
        Passing it explicitly keeps the same compiled model reusable across
        datasets with matching shapes and dtypes. Prepared models
        use ``model.data`` for training inputs or ``model.prepare_data``
        for new observations.

        Parameters
        ----------
        position : mapping of str to array_like
            Unconstrained values for every declared parameter, with each
            value matching its declaration's ``position_shape``.
        data : object
            For a prepared model, pass ``model.data`` or the result of
            ``model.prepare_data``. Named callbacks receive the requested
            observation arrays, contributions, and parameters directly.
            Otherwise, this is passed as the first argument to the callback.
            Use a JAX-compatible PyTree when applying JAX transformations.

        Returns
        -------
        jax.Array
            Scalar model log density including component priors and
            parameterization adjustments.
        """
        parameters = self.constrain(position)
        if self._data is None:
            density = _as_scalar(self._log_density(data, **parameters), name="log_density")
        else:
            inputs = self._component_data(data)
            effects = self._evaluate_quantities(inputs, parameters)
            if self._density_inputs is None:
                callback_parameters = {name: parameters[name] for name in self._callback_parameter_names}
                result = self._log_density(inputs.values, effects, **callback_parameters)
            else:
                arguments = _callback_inputs(self._density_inputs, inputs, effects, parameters, name="log_density")
                result = self._log_density(**arguments)
            density = _as_scalar(result, name="log_density")
            for component in inputs.components:
                prior = (
                    component.log_prior(parameters)
                    if isinstance(component, _PreparedMedia)
                    else component.log_prior(parameters[component.specification.name])
                )
                density = density + prior

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
        """Evaluate generated quantities from constrained model parameters.

        Parameters
        ----------
        key : jax.Array
            JAX random key passed to the generation callback. The callback
            must split it when drawing multiple independent samples.
        parameters : mapping of str to array_like
            Constrained values for every declared parameter. Only the names
            requested by the generation callback are passed to it.
        data : object
            For a prepared model, pass ``model.data`` or the result of
            ``model.prepare_data``. Named callbacks receive only the inputs
            they request. Otherwise, this is passed as the second argument
            to the callback. Use a JAX-compatible PyTree when applying JAX
            transformations.

        Returns
        -------
        dict of str to jax.Array
            Dictionary mapping the names returned by the generation callback
            to JAX arrays. The callback determines the keys and array shapes.
        """
        if self._generate is None:
            raise RuntimeError("generated quantities are unavailable because this model has no generate callback")

        _validate_value_names(parameters, self._parameterizations, name="parameters")
        if self._generate_parameter_names is None:
            callback_parameters = (
                dict(parameters)
                if self._data is None
                else {name: parameters[name] for name in self._callback_parameter_names}
            )
        else:
            callback_parameters = {name: parameters[name] for name in self._generate_parameter_names}
        if self._data is None:
            generated = self._generate(key, data, **callback_parameters)
        else:
            inputs = self._component_data(data)
            effects = self._evaluate_quantities(inputs, parameters)
            if self._generation_inputs is None:
                generated = self._generate(key, inputs.values, effects, **callback_parameters)
            else:
                arguments = _callback_inputs(self._generation_inputs, inputs, effects, parameters, name="generate")
                generated = self._generate(key, **arguments)
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

    def _evaluate_quantities(self, inputs: _ModelData, parameters: ParameterValues) -> dict[str, jax.Array]:
        """Evaluate components and one shared transformation using the current inputs."""
        effects = _component_effects(inputs, parameters)
        if self._transformed_parameters is None:
            return effects

        arguments = _callback_inputs(self._transform_inputs, inputs, effects, parameters, name="transformed_parameters")
        transformed = self._transformed_parameters(**arguments)
        if not isinstance(transformed, Mapping):
            raise TypeError("transformed_parameters must return a mapping from quantity names to array-like values")

        # Retain training names even when prediction omits their observation arrays.
        assert self._data is not None
        reserved = set(self._data.values) | set(effects) | set(parameters)
        for name in transformed:
            _validate_name(name, label="transformed quantity")
            if name in reserved:
                raise ValueError(f"Transformed quantity {name!r} conflicts with a data role, component, or parameter")
        for name, value in transformed.items():
            try:
                effects[name] = jnp.asarray(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Transformed quantity {name!r} must be array-like, got {type(value).__name__}"
                ) from error
        return effects

    def _component_data(self, data: object) -> _ModelData:
        """Keep component inputs tied to the model's parameter and prior definitions."""
        if not isinstance(data, _ModelData):
            raise TypeError(
                "Models with components require prepared model inputs. "
                "Pass model.data or the result of model.prepare_data"
            )
        if self._data is None or len(data.components) != len(self._data.components):
            raise ValueError("The prepared inputs must use this model's component configurations and training labels")
        for component, reference in zip(data.components, self._data.components, strict=True):
            matches = (
                component.specification is reference.specification
                and component.time_column == reference.time_column
                and component.group_columns == reference.group_columns
                and component.group_values == reference.group_values
            )
            if isinstance(component, _PreparedMedia):
                matches = (
                    matches
                    and isinstance(reference, _PreparedMedia)
                    and component.media_columns == reference.media_columns
                    and component.channels == reference.channels
                    and component.dtype == reference.dtype
                )
            else:
                matches = matches and isinstance(reference, _PreparedFourier) and component.origin == reference.origin
            if not matches:
                raise ValueError(
                    "The prepared inputs must use this model's component configurations and training labels"
                )
        return data


def _component_effects(inputs: _ModelData, parameters: ParameterValues) -> dict[str, jax.Array]:
    """Evaluate component contributions with the current observations and constrained parameters."""
    effects = {}
    for component in inputs.components:
        name = component.specification.name
        if isinstance(component, _PreparedMedia):
            if "media" not in inputs.values:
                raise ValueError("Media effects require media exposures in the prepared model inputs")
            effects[name] = component.apply(inputs.values["media"], parameters)
        else:
            effects[name] = component.apply(parameters[name])
    return effects


def _bind_inputs(
    function: Callable[..., object],
    parameter_names: tuple[str, ...],
    data: _ModelData,
    *,
    name: str,
    upstream_parameters: tuple[str, ...] = (),
    has_transformed: bool = False,
) -> _InputBindings | None:
    """Resolve keyword-only inputs once while leaving positional callbacks unchanged."""
    if not callable(function):
        return None
    try:
        arguments = list(signature(function).parameters.values())
    except (TypeError, ValueError):
        return None

    key_argument = None
    if name == "generate":
        if not arguments or arguments[0].kind not in (
            SignatureParameter.POSITIONAL_ONLY,
            SignatureParameter.POSITIONAL_OR_KEYWORD,
        ):
            return None
        key_argument = arguments.pop(0)
    if any(
        argument.kind in (SignatureParameter.POSITIONAL_ONLY, SignatureParameter.POSITIONAL_OR_KEYWORD)
        for argument in arguments
    ):
        return None
    if any(argument.kind is not SignatureParameter.KEYWORD_ONLY for argument in arguments):
        raise TypeError(f"Named inputs for {name} must be explicit keyword-only arguments without *args or **kwargs")

    sources = {
        "data": set(data.values),
        "effect": {component.specification.name for component in data.components},
        "parameter": set(parameter_names),
    }
    if key_argument is not None and any(key_argument.name in names for names in sources.values()):
        raise TypeError(f"generate places input {key_argument.name!r} where the random key is required")

    bindings = []
    for argument in arguments:
        matches = [source for source, names in sources.items() if argument.name in names]
        if not matches:
            if has_transformed:
                bindings.append((argument.name, "transformed"))
                continue
            raise ValueError(
                f"{name} requests unknown input {argument.name!r}. "
                "Use a selected data role, component name, or declared parameter"
            )
        if len(matches) > 1:
            raise ValueError(
                f"{name} input {argument.name!r} is ambiguous across {matches}. "
                "Use distinct names for data roles, components, and parameters"
            )
        bindings.append((argument.name, matches[0]))

    if name == "log_density":
        missing = sorted(set(parameter_names) - set(upstream_parameters) - {argument.name for argument in arguments})
        if missing:
            callbacks = "log_density or transformed_parameters" if has_transformed else "log_density"
            raise ValueError(f"{callbacks} must request every declared parameter. Missing parameters {missing}")
    return tuple(bindings)


def _callback_inputs(
    bindings: _InputBindings,
    data: _ModelData,
    effects: Mapping[str, jax.Array],
    parameters: ParameterValues,
    *,
    name: str,
) -> dict[str, ArrayLike]:
    """Supply the requested numerical inputs from the current model evaluation."""
    sources: dict[str, ParameterValues] = {
        "data": data.values,
        "effect": effects,
        "parameter": parameters,
        "transformed": effects,
    }
    arguments = {}
    for argument, source in bindings:
        if argument not in sources[source]:
            if source == "transformed":
                raise ValueError(
                    f"{name} requires transformed quantity {argument!r}. Return it from transformed_parameters"
                )
            raise ValueError(f"{name} requires input {argument!r}. Include it when preparing data for this evaluation")
        arguments[argument] = sources[source][argument]
    return arguments


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
    *,
    has_components: bool = False,
) -> None:
    actual_names = _model_parameter_names(
        function,
        expected_names,
        name="log_density",
        leading_arguments=("data", "effects") if has_components else ("data",),
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
    *,
    has_components: bool = False,
) -> tuple[str, ...] | None:
    actual_names = _model_parameter_names(
        function,
        expected_names,
        name="generate",
        leading_arguments=("key", "data", "effects") if has_components else ("key", "data"),
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
