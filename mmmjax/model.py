"""Model composition for transparent JAX probability models."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter as SignatureParameter
from inspect import signature
from keyword import iskeyword
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from mmmjax.data import PreparedData
from mmmjax.parameters import Parameterization
from mmmjax.seasonality import FourierSeasonality, _PreparedFourier

__all__ = ["Model"]

LogDensity: TypeAlias = Callable[..., ArrayLike]
Generate: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
ParameterValues: TypeAlias = Mapping[str, ArrayLike]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class _ModelData:
    """Keep observation arrays and prepared components in one dynamic PyTree."""

    values: dict[str, jax.Array]
    components: tuple[_PreparedFourier, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Model:
    """Compose parameter declarations with density and generation functions.

    Parameters
    ----------
    parameters : mapping of str to Parameterization
        Named declarations for the parameters used by the callbacks.
        Component coefficients are declared automatically and must not
        also appear in this mapping.
    log_density : callable
        Scalar log density in the constrained model space. It receives data
        followed by either every declared callback parameter or
        ``**parameters``. With components, it receives ``data, effects``
        before those parameters. Component priors are added automatically
        and should not be included in this callback.
    generate : callable, optional
        Generated-quantities function. It receives a JAX random key, data, and
        either the named model parameters it needs or ``**parameters``.
        With components, it receives ``key, data, effects`` before the
        callback parameters. Component coefficients are not passed directly
        to either callback.
        If omitted, generated quantities are unavailable.
    data : PreparedData, optional
        Training observations from ``prepare_data``. Required when
        ``components`` is supplied. Callbacks receive its numeric arrays
        keyed by data role, such as ``"outcome"`` or ``"media"``.
    components : sequence of FourierSeasonality, optional
        Seasonal contributions to prepare against the training observations.
        Each component adds a coefficient parameter and an entry in the
        callback's ``effects`` mapping under its name. The callback decides
        how to combine these effects in the model. An empty sequence passes
        an empty effects mapping. Omitting this argument keeps the callback
        signature without effects and lets callers supply their own data.

    Examples
    --------
    Add a yearly seasonal curve to a Normal model. The model prepares the
    seasonal features and assigns the coefficient prior. The callback
    specifies the likelihood and the intercept prior.

    .. ipython::

        In [1]: import jax
           ...: import polars as pl
           ...: from mmmjax import (
           ...:     FourierSeasonality, Model, Real, normal, prepare_data,
           ...: )
           ...: frame = pl.DataFrame({
           ...:     "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
           ...:     "sales": [1.0, 1.4, 1.2],
           ...: })
           ...: observations = prepare_data(
           ...:     frame, time="date", outcome="sales",
           ...: )
           ...: def density(data, effects, intercept):
           ...:     mean = intercept + effects["annual"]
           ...:     return (
           ...:         normal(data["outcome"], mean, 0.5)
           ...:         + normal(intercept, 0.0, 2.0)
           ...:     )
           ...: model = Model(
           ...:     parameters={"intercept": Real()},
           ...:     log_density=density,
           ...:     data=observations,
           ...:     components=[FourierSeasonality(name="annual", order=2)],
           ...: )
           ...: position = model.initialize_random(jax.random.key(0))
           ...: value = jax.jit(model.log_density)(position, model.data)
           ...: value.shape
    """

    _parameterizations: tuple[tuple[str, Parameterization], ...]
    _log_density: LogDensity
    _generate: Generate | None
    _generate_parameter_names: tuple[str, ...] | None
    _callback_parameter_names: tuple[str, ...]
    _data: _ModelData | None

    def __init__(
        self,
        parameters: Mapping[str, Parameterization],
        log_density: LogDensity,
        generate: Generate | None = None,
        *,
        data: PreparedData | None = None,
        components: Sequence[FourierSeasonality] | None = None,
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
                raise TypeError("components must be a sequence of FourierSeasonality configurations")
            declarations = dict(parameterizations)
            specifications = tuple(components)
            names = set(parameter_names)
            for specification in specifications:
                if not isinstance(specification, FourierSeasonality):
                    raise TypeError("Each component must be a FourierSeasonality configuration")
                if specification.name in names:
                    raise ValueError(
                        f"Component name {specification.name!r} conflicts with another component or parameter"
                    )
                names.add(specification.name)

            prepared_components = tuple(specification._prepare(data) for specification in specifications)
            for component in prepared_components:
                declarations.update(component.parameters)
            parameterizations = _prepare_parameterizations(declarations)
            prepared_data = _ModelData(data._to_jax(), prepared_components)

        _validate_log_density_signature(log_density, parameter_names, has_components=components is not None)
        generate_parameter_names = None
        if generate is not None:
            generate_parameter_names = _validate_generate_signature(
                generate, parameter_names, has_components=components is not None
            )

        object.__setattr__(self, "_parameterizations", parameterizations)
        object.__setattr__(self, "_log_density", log_density)
        object.__setattr__(self, "_generate", generate)
        object.__setattr__(self, "_generate_parameter_names", generate_parameter_names)
        object.__setattr__(self, "_callback_parameter_names", parameter_names)
        object.__setattr__(self, "_data", prepared_data)

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
        coefficient priors as well as the callback's density.

        Without components, ``data`` may be any JAX-compatible PyTree.
        Passing it explicitly keeps the same compiled model reusable across
        datasets with matching shapes and dtypes. Models with components
        use the prepared inputs from ``model.data``.

        Parameters
        ----------
        position : mapping of str to array_like
            Unconstrained values for every declared parameter, with each
            value matching its declaration's ``position_shape``.
        data : object
            Data passed as the first argument to the log-density callback.
            Use a JAX-compatible PyTree when applying JAX transformations.
            For a model with components, pass ``model.data``. The callback
            receives numeric observation arrays and named effects.

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
            effects = {
                component.specification.name: component.apply(parameters[component.specification.name])
                for component in inputs.components
            }
            callback_parameters = {name: parameters[name] for name in self._callback_parameter_names}
            density = _as_scalar(self._log_density(inputs.values, effects, **callback_parameters), name="log_density")
            for component in inputs.components:
                density = density + component.log_prior(parameters[component.specification.name])

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
            Data passed as the second argument to the generation callback.
            Use a JAX-compatible PyTree when applying JAX transformations.
            For a model with components, pass ``model.data``. The callback
            receives numeric observation arrays and named effects.

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
            effects = {
                component.specification.name: component.apply(parameters[component.specification.name])
                for component in inputs.components
            }
            generated = self._generate(key, inputs.values, effects, **callback_parameters)
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

    def _component_data(self, data: object) -> _ModelData:
        """Keep component inputs tied to the model's coefficient and prior definitions."""
        if not isinstance(data, _ModelData):
            raise TypeError("Models with components require prepared model inputs. Pass model.data")
        if (
            self._data is None
            or len(data.components) != len(self._data.components)
            or any(
                component.specification is not reference.specification
                or (
                    component.origin,
                    component.time_column,
                    component.group_columns,
                    component.group_values,
                )
                != (
                    reference.origin,
                    reference.time_column,
                    reference.group_columns,
                    reference.group_values,
                )
                for component, reference in zip(data.components, self._data.components, strict=True)
            )
        ):
            raise ValueError("The prepared inputs must use this model's component configurations and training labels")
        return data


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
