"""Model composition for transparent JAX probability models."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from inspect import Parameter as SignatureParameter
from inspect import signature
from keyword import iskeyword
from numbers import Integral
from typing import Literal, TypeAlias, get_args

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray

from mmmjax._results import _coordinates, _dimensions, _name, _prepared_coordinates, _same_labels
from mmmjax.data import (
    Data,
    PreparedData,
    _data_dimensions,
    _DataLayout,
    _day_of_year,
    _prepare_model_frame,
    _time_positions,
    _TimeInput,
)
from mmmjax.parameters import (
    CorrelationCholesky,
    Interval,
    LowerBound,
    Parameterization,
    Positive,
    Real,
    Simplex,
    UpperBound,
    _as_array,
)
from mmmjax.priors import Prior, _validate_prior_sampler
from mmmjax.scaling import DataScaling, Scaling, fit_data_scaling

__all__ = ["Model"]

LogDensity: TypeAlias = Callable[..., ArrayLike]
GeneratedQuantities: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
PriorSampler: TypeAlias = Callable[[jax.Array], Mapping[str, ArrayLike]]
TransformedParameters: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
TransformedData: TypeAlias = Callable[..., Mapping[str, ArrayLike]]
ParameterValues: TypeAlias = Mapping[str, ArrayLike]
_CallbackValue: TypeAlias = ArrayLike | Callable[[ArrayLike], jax.Array]
_InputBindings: TypeAlias = tuple[tuple[str, str], ...]
_DataVariables: TypeAlias = tuple[tuple[str, str, str], ...]
_BuiltinParameter: TypeAlias = Real | Positive | LowerBound | UpperBound | Interval | Simplex | CorrelationCholesky
_ResultGroup: TypeAlias = Literal["predictive", "log_likelihood", "log_prior"]
_OutputGroup: TypeAlias = Literal["generated"] | _ResultGroup
_OutputKey: TypeAlias = tuple[_OutputGroup, str]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, eq=False)
class _ModelData:
    """Keep observation arrays in a model-owned dynamic PyTree."""

    values: dict[str, jax.Array]
    owner: object = field(default_factory=object, metadata={"static": True})
    reference_values: dict[str, jax.Array] = field(default_factory=dict)
    outcome_scaling: Scaling | None = None
    outcome_group_scale: bool = field(default=False, metadata={"static": True})
    n_periods: int = field(default=0, metadata={"static": True})
    reference_n_periods: int = field(default=0, metadata={"static": True})
    reserved_names: tuple[str, ...] = field(default=(), metadata={"static": True})
    variable_sources: _DataVariables | None = field(default=None, metadata={"static": True})
    transformed_values: dict[str, jax.Array] = field(default_factory=dict)
    static_values: tuple[tuple[str, int | bool], ...] = field(default=(), metadata={"static": True})
    transformed_sources: _DataVariables = field(default=(), metadata={"static": True})
    constants: tuple[tuple[str, object], ...] = field(default=(), metadata={"static": True})


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Model:
    """Compose parameter declarations, log density, and generated quantities.

    A model evaluates its program blocks in order, from ``data`` and
    ``transformed_data`` through ``parameters``, ``transformed_parameters``,
    and ``log_density`` to ``generated_quantities``. Each function requests the
    inputs it needs by argument name from data inputs, the outputs of earlier
    blocks, and parameters. Models without prepared data receive the data as
    a leading positional argument instead.

    Prepared data supplies the selected role names, such as ``outcome`` and
    ``media``, together with elapsed ``time`` and ``media_time``, the calendar
    ``day_of_year`` and ``media_day_of_year``,
    ``n_periods``, the outcome conversions ``outcome_scale``,
    ``outcome_offset``, and ``unscale_outcome``, and the training inputs
    ``reference_<role>`` and ``reference_n_periods`` when evaluating new
    data. ``Data`` variables replace these names with declared ones.

    Inspect derived quantities with ``evaluate`` and the constrained log
    density with ``log_prob`` before fitting. Samplers call ``log_density``
    on unconstrained positions and add parameterization adjustments.

    Parameters
    ----------
    parameters : mapping of str to Parameterization
        All named parameter declarations and constraints. Use declaration
        dimensions, such as ``Real(dims="control")``, to infer shapes and
        result labels from prepared data or ``coords``.
    log_density : callable
        Scalar log density for constrained parameters. Every user-declared
        parameter must be requested here or by ``transformed_parameters``.
        Only parameterization adjustments are added automatically.
    generated_quantities : callable, optional
        Function returning named reporting quantities or simulated observations.
        Outputs do not contribute to the log density.
        Receives a JAX random key first, followed by the model inputs it needs.
        Entries under the keys ``predictive``, ``log_likelihood``, and
        ``log_prior`` are mappings of named outputs stored in the matching
        result groups, and all other entries are ordinary generated quantities.
        Predictive draws and pointwise log likelihoods matching the outcome
        shape inherit its observation labels. Pointwise log likelihoods exclude
        priors and adjustments. Mapped prior definitions already record
        ``log_prior_<parameter>`` terms, so return only additional prior
        factors, each once and without constraint adjustments.
    save : sequence of str, default ()
        Transformed quantities to retain in results,
        such as ``("mu", "paid_media", "paid_media_total")``. These are evaluated
        for each draw without a ``generated_quantities`` callback. Names must
        differ from its returned outputs. Requires prepared data.
    prior : mapping of str to Prior or callable, optional
        Reusable prior definitions for every declared parameter. The mapping
        supplies independent prior draws and records named log-prior terms
        at posterior draws. Include these priors explicitly in ``log_density``.
        For dependent or custom draws, supply a JAX-compatible ``prior(key)``
        returning all constrained parameters with their declared shapes.
        A custom sampler does not automatically record log-prior terms.
    data : Data or PreparedData, optional
        Fixed model inputs and their declarations. Use ``Data`` for
        custom names, auxiliary observations, constants, or scaling. Plain
        prepared data uses standard role names without fitting new
        transformations.
    transformed_data : callable, optional
        Data-only calculations returning named fixed values. Runs once at
        construction and again for changed data, not per parameter draw.
        Inputs must be declared data variables and outputs must have distinct
        names from inputs and parameters. Return finite real arrays or scalars.
        Python integers remain static for array shapes. Requires prepared data.
        Use pure JAX-compatible operations for response analysis and optimization.
    transformed_parameters : callable, optional
        Pure JAX-compatible function returning a mapping of names to derived
        arrays shared by the density and generated-quantities callbacks.
        Evaluated for density and output calculations. Requires prepared data
        and named inputs. Names must not shadow declared inputs or parameters.
        Outputs are not sampled parameters. Use ``save`` to retain them.
        All requested inputs are needed for output evaluation.
    dims : mapping of str to sequence of str, optional
        Named axes for constrained parameter arrays, excluding chain and draw.
        Use for custom parameterizations or explicit-shape declarations.
        Must agree with any dimensions set on a declaration.
    coords : mapping of str to array_like, optional
        One-dimensional labels for named axes. Labels are copied at construction.
    generated_dims : mapping of str to sequence of str, optional
        Axis labels for saved or generated arrays, excluding chain and draw.
        Overrides labels inherited from unchanged data or parameter
        inputs and from observation-shaped predictive and log-likelihood
        outputs. A name shared by several result groups receives the same axes.
    """

    _parameterizations: tuple[tuple[str, Parameterization], ...]
    _log_density: LogDensity
    _generate: GeneratedQuantities | None
    _prior: PriorSampler | None
    _priors: tuple[tuple[str, Prior], ...]
    _transformed_parameters: TransformedParameters | None
    _transformed_data: TransformedData | None
    _transformed_data_inputs: _InputBindings
    _transform_inputs: _InputBindings
    _saved_inputs: _InputBindings
    _generate_parameter_names: tuple[str, ...] | None
    _density_inputs: _InputBindings
    _generation_inputs: _InputBindings
    _data: _ModelData | None
    _layout: _DataLayout | None
    _input_dims: dict[str, tuple[str, ...]]
    _input_coords: dict[str, NDArray[np.generic]]
    _time_column: str | None
    _frequency: str | None
    _time_values: tuple[object, ...]
    _media_time_values: tuple[object, ...]
    _time_inputs: tuple[str, ...]
    _time_origin: float | datetime | None
    _scaling: DataScaling | None
    _dtype: DTypeLike
    _result_dims: dict[str, tuple[str, ...]]
    _result_coords: dict[str, NDArray[np.generic]]
    _generated_dims: dict[str, tuple[str, ...]]

    def __init__(
        self,
        parameters: Mapping[str, Parameterization],
        log_density: LogDensity,
        generated_quantities: GeneratedQuantities | None = None,
        *,
        save: Sequence[str] = (),
        prior: Mapping[str, Prior] | PriorSampler | None = None,
        data: Data | PreparedData | None = None,
        transformed_data: TransformedData | None = None,
        transformed_parameters: TransformedParameters | None = None,
        dims: Mapping[str, Sequence[str]] | None = None,
        coords: Mapping[str, object] | None = None,
        generated_dims: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Create a model from named parameter declarations and plain functions."""
        if prior is not None and not isinstance(prior, Mapping):
            _validate_prior_sampler(prior)

        parameterizations = _prepare_parameterizations(parameters)
        parameter_names = tuple(name for name, _ in parameterizations)
        variables: Mapping[str, str] | None = None
        inputs: xr.Dataset | None = None
        constants: dict[str, object] = {}
        scaling: DataScaling | Literal["auto"] | None = None
        if isinstance(data, Data):
            data, variables, inputs, constants, scaling = data._snapshot()
        conflicts = sorted(set(constants) & set(parameter_names))
        if conflicts:
            raise ValueError(f"Constants {conflicts} conflict with declared parameters")
        variable_declarations = _data_variable_declarations(variables, parameter_names, tuple(constants))
        prepared_data = None
        fitted_scaling = None
        time_inputs: tuple[str, ...] = ()
        time_origin = None
        if data is None:
            if transformed_data is not None:
                raise ValueError("transformed_data requires prepared data")
        else:
            if not isinstance(data, PreparedData):
                raise TypeError("data must be Data or PreparedData. Use prepare_data with the observation dataframe")
            if scaling == "auto":
                fitted_scaling = fit_data_scaling(data)
            elif isinstance(scaling, DataScaling):
                fitted_scaling = scaling
            else:
                fitted_scaling = data._scaling
            if fitted_scaling is not None:
                if data._scaling is fitted_scaling:
                    data = data._align_to(fitted_scaling._layout)
                else:
                    data = fitted_scaling.transform(data)
            requested = (
                set(variable_declarations.values())
                if variable_declarations is not None
                else _requested_inputs(log_density, transformed_parameters, generated_quantities)
                | _requested_inputs(transformed_data, None, None)
            )
            time_inputs = tuple(
                name for name in get_args(_TimeInput) if name in requested or f"reference_{name}" in requested
            )
            values = data._to_jax()
            if time_inputs:
                _, time_origin = _time_positions(data.time_values)
                values.update(_model_time_inputs(data, time_inputs, time_origin))
            reference_values = {
                f"reference_{role}": jnp.array(value, copy=True)
                for role, value in values.items()
                if f"reference_{role}" in requested
            }
            reserved = {f"reference_{role}" for role in (*_data_dimensions(data), *get_args(_TimeInput))}
            reserved.update(("outcome_scale", "outcome_offset", "unscale_outcome", "n_periods", "reference_n_periods"))
            conflicts = reserved & set(parameter_names) if variable_declarations is None else set()
            if conflicts:
                raise ValueError(f"Parameter names {sorted(conflicts)} conflict with model-supplied inputs")

            outcome_scaling = None
            if "outcome" in values:
                outcome_scaling = Scaling(offset=jnp.asarray(0.0), scale=jnp.asarray(1.0))
                if fitted_scaling is not None:
                    outcome_scaling = fitted_scaling.transformations.get("outcome", outcome_scaling)
            population_outcome = (
                fitted_scaling is not None
                and "outcome" in fitted_scaling._population_roles
                and bool(data.group_columns)
            )
            prepared_data = _ModelData(
                values,
                reference_values=reference_values,
                outcome_scaling=outcome_scaling,
                outcome_group_scale=population_outcome,
                n_periods=len(data.time_values),
                reference_n_periods=len(data.time_values),
                reserved_names=tuple(sorted(reserved)) if variable_declarations is None else (),
                constants=tuple(constants.items()),
            )

        result_dims = _dimensions(dims)
        result_coords = _coordinates(coords)
        axis_coordinates = result_coords.copy()
        input_dims: dict[str, tuple[str, ...]] = {}
        input_coords: dict[str, NDArray[np.generic]] = {}
        if data is not None:
            prepared_coords, _ = _prepared_coordinates(data)
            for axis, labels in prepared_coords.items():
                if axis in axis_coordinates and not _same_labels(axis_coordinates[axis], labels):
                    raise ValueError(f"Coordinate {axis!r} conflicts with prepared data labels")
                axis_coordinates[axis] = labels
            reserved_parameters = parameter_names if variable_declarations is None else ()
            input_values, input_dims, input_coords = _prepare_inputs(
                inputs, data, reserved_parameters, axis_coordinates, tuple(constants)
            )
            axis_coordinates.update(input_coords)
            assert prepared_data is not None
            conflicts = set(prepared_data.reserved_names) & (set(input_values) | set(axis_coordinates) | set(constants))
            if conflicts:
                raise ValueError(f"Input or coordinate names {sorted(conflicts)} conflict with model-supplied inputs")
            prepared_data.values.update(input_values)
            if variable_declarations is not None:
                prepared_data = replace(
                    prepared_data,
                    variable_sources=_resolve_data_variables(variable_declarations, prepared_data),
                )
        parameterizations = _resolve_parameter_dimensions(parameterizations, result_dims, axis_coordinates)

        transformed_data_inputs: _InputBindings = ()
        if transformed_data is not None:
            assert prepared_data is not None
            transformed_data_inputs = _bind_inputs(transformed_data, (), prepared_data, name="transformed_data")
            prepared_data = _compute_transformed_data(
                transformed_data, transformed_data_inputs, prepared_data, parameter_names
            )

        prior_definitions: tuple[tuple[str, Prior], ...] = ()
        if isinstance(prior, Mapping):
            _validate_value_names(prior, parameterizations, name="prior")
            for name, declaration in parameterizations:
                if not isinstance(prior[name], Prior):
                    raise TypeError(
                        f"Prior definition for {name!r} must be a Prior object with fixed distribution settings. "
                        "Use a prior-draw function for dependent draws. "
                        "Return additional log-prior terms under log_prior in generated_quantities"
                    )
                try:
                    prior[name]._validate_shape(declaration.shape)
                except ValueError as exc:
                    raise ValueError(f"Invalid prior shape for parameter {name!r}. {exc}") from exc
            prior_definitions = tuple((name, prior[name]) for name in parameter_names)

        transform_inputs: _InputBindings = ()
        if transformed_parameters is not None:
            if prepared_data is None:
                raise ValueError("transformed_parameters requires prepared data")
            transform_inputs = _bind_inputs(
                transformed_parameters, parameter_names, prepared_data, name="transformed_parameters"
            )

        upstream_parameters = tuple(name for name, source in transform_inputs if source == "parameter")
        density_inputs: _InputBindings = ()
        if prepared_data is None:
            _validate_log_density_signature(log_density, parameter_names)
        else:
            density_inputs = _bind_inputs(
                log_density,
                parameter_names,
                prepared_data,
                name="log_density",
                upstream_parameters=upstream_parameters,
                has_transformed=transformed_parameters is not None,
            )
        generate_parameter_names = None
        generation_inputs: _InputBindings = ()
        if generated_quantities is not None:
            if prepared_data is None:
                generate_parameter_names = _validate_generate_signature(generated_quantities, parameter_names)
            else:
                generation_inputs = _bind_inputs(
                    generated_quantities,
                    parameter_names,
                    prepared_data,
                    name="generated_quantities",
                    has_transformed=transformed_parameters is not None,
                )

        saved_names = _result_names(save, name="save")
        saved_inputs: list[tuple[str, str]] = []
        if saved_names:
            if prepared_data is None:
                raise ValueError("save requires prepared data")
            reserved = _callback_data_names(prepared_data) | set(parameter_names)
            for name in saved_names:
                _validate_name(name, label="saved quantity")
                if name in reserved or transformed_parameters is None:
                    raise ValueError(f"save must select a transformed quantity, got {name!r}")
                saved_inputs.append((name, "transformed"))

        output_dims = _dimensions(generated_dims)
        if generated_quantities is None and not saved_inputs and not prior_definitions and output_dims:
            raise ValueError("Generated result metadata requires a generated_quantities callback or saved quantities")
        declarations = dict(parameterizations)
        dimension_sizes = {axis: len(labels) for axis, labels in axis_coordinates.items()}
        for name, axes in result_dims.items():
            if name not in declarations:
                raise ValueError(f"dims refers to undeclared parameter {name!r}")
            shape = declarations[name].shape
            if len(axes) != len(shape):
                raise ValueError(f"Dimensions for parameter {name!r} must match its constrained shape {shape}")
            for axis, size in zip(axes, shape, strict=True):
                if axis in dimension_sizes and dimension_sizes[axis] != size:
                    raise ValueError(f"Dimension {axis!r} must have length {size} for parameter {name!r}")
                dimension_sizes[axis] = size

        object.__setattr__(self, "_parameterizations", parameterizations)
        object.__setattr__(self, "_log_density", log_density)
        object.__setattr__(self, "_generate", generated_quantities)
        object.__setattr__(self, "_priors", prior_definitions)
        object.__setattr__(self, "_prior", self._draw_prior if isinstance(prior, Mapping) else prior)
        object.__setattr__(self, "_transformed_parameters", transformed_parameters)
        object.__setattr__(self, "_transformed_data", transformed_data)
        object.__setattr__(self, "_transformed_data_inputs", transformed_data_inputs)
        object.__setattr__(self, "_transform_inputs", transform_inputs)
        object.__setattr__(self, "_saved_inputs", tuple(saved_inputs))
        object.__setattr__(self, "_generate_parameter_names", generate_parameter_names)
        object.__setattr__(self, "_density_inputs", density_inputs)
        object.__setattr__(self, "_generation_inputs", generation_inputs)
        object.__setattr__(self, "_data", prepared_data)
        object.__setattr__(self, "_layout", None if data is None else data._layout())
        object.__setattr__(self, "_input_dims", input_dims)
        object.__setattr__(self, "_input_coords", input_coords)
        object.__setattr__(self, "_time_column", None if data is None else data.time_column)
        object.__setattr__(self, "_frequency", None if data is None else data.frequency)
        object.__setattr__(self, "_time_values", () if data is None else tuple(data.time_values))
        object.__setattr__(self, "_media_time_values", () if data is None else tuple(data.media_time_values))
        object.__setattr__(self, "_time_inputs", time_inputs)
        object.__setattr__(self, "_time_origin", time_origin)
        object.__setattr__(self, "_scaling", fitted_scaling)
        object.__setattr__(self, "_dtype", jax.dtypes.canonicalize_dtype(float))
        object.__setattr__(self, "_result_dims", result_dims)
        object.__setattr__(self, "_result_coords", result_coords)
        object.__setattr__(self, "_generated_dims", output_dims)

    @property
    def parameters(self) -> dict[str, Parameterization]:
        """Return a copy of the named parameter declarations."""
        return dict(self._parameterizations)

    @property
    def data_variables(self) -> dict[str, str]:
        """Return the available model input names and their declared sources.

        Returns
        -------
        dict of str to str
            A copy of the input declarations. Without explicit declarations,
            the standard names currently supplied map to themselves, including
            ``time`` and ``reference_`` inputs only when a function requests
            them. Models without prepared data return an empty mapping.
        """
        if self._data is None:
            return {}
        constants = {name: name for name, _ in self._data.constants}
        if self._data.variable_sources is not None:
            declared = {name: source_name for name, _, source_name in self._data.variable_sources}
            return declared | constants
        available = set().union(*_data_sources(self._data).values())
        return {name: name for name in sorted(available)}

    @property
    def _has_generated_quantities(self) -> bool:
        """Indicate whether evaluation has saved, generated, or prior outputs."""
        return self._generate is not None or bool(self._saved_inputs) or bool(self._priors)

    def _draw_prior(self, key: jax.Array) -> dict[str, jax.Array]:
        """Draw independent prior definitions in declared parameter shapes."""
        keys = jax.random.split(key, len(self._priors))
        declarations = self.parameters
        values = {}
        for (name, prior), draw_key in zip(self._priors, keys, strict=True):
            declaration = declarations[name]
            values[name] = prior._sample(draw_key, declaration.shape, declaration.dtype)
        return values

    @property
    def scaling(self) -> DataScaling | None:
        """Return the fitted input transformations or None for unscaled inputs.

        Returns
        -------
        DataScaling or None
            Fitted transformations available by role through
            ``transformations``. Use the outcome transformation to restore
            predicted levels to their original units. Multiply contributions
            by its scale only, without adding the outcome offset.
        """
        return self._scaling

    @property
    def data(self) -> object:
        """Return prepared training inputs for density and generation calls.

        Pass to ``log_density`` or ``generate_quantities``, including under JIT.
        Arrays are copied at construction, independent of later source edits.

        Returns
        -------
        object
            JAX-compatible observation arrays. Callbacks
            receive their requested inputs without unpacking this bundle.
            Requires prepared ``data`` at construction.
        """
        if self._data is None:
            raise RuntimeError("This model has no prepared data. Pass your data directly when evaluating it")
        return replace(
            self._data,
            values=dict(self._data.values),
            reference_values=dict(self._data.reference_values),
            transformed_values=dict(self._data.transformed_values),
        )

    def prepare_data(self, data: object) -> object:
        """Prepare new observations using the model's training configuration.

        Reuse fitted scaling, the time origin, and parameter declarations
        without changing the model's stored data. Call outside JAX transformations.
        Auxiliary ``Data`` inputs retain their original values and labels.

        Parameters
        ----------
        data : dataframe-like or PreparedData
            Observations using the original source columns and groups.
            Dataframes reuse the model's selections and observation spacing.
            Prepared inputs may be raw or use this model's fitted scaling.
            Omit inputs only when no evaluated callback needs them.
            For explicit media history, use ``prepare_data(media_history=...)``.

        Returns
        -------
        object
            JAX-compatible inputs for ``log_density`` or ``generate_quantities`` in the
            fitted group and channel order, covering the supplied periods.
            Independent of stored model data and later source edits.
        """
        return self._prepare_data(data)[0]

    def _prepare_data(self, data: object) -> tuple[_ModelData, PreparedData]:
        """Keep aligned observation labels alongside the evaluated model inputs."""
        if self._data is None or self._layout is None:
            raise RuntimeError("This model has no prepared training data. Pass your data directly when evaluating it")
        if not isinstance(data, PreparedData):
            assert self._time_column is not None
            data = _prepare_model_frame(data, self._layout, time=self._time_column, frequency=self._frequency)
        if data.time_column != self._time_column:
            raise ValueError(f"The time column must match the training column {self._time_column!r}")
        if (
            self._frequency is not None
            and data.frequency is not None
            and self._frequency != data.frequency
            and bool({"media", "organic_media", "reach", "organic_reach"} & self._data.values.keys())
        ):
            raise ValueError(
                f"Media inputs must retain the model's {self._frequency} observation spacing. "
                "Changing frequency changes the meaning of the lag parameters"
            )

        aligned = data._align_to(self._layout)
        if self._scaling is not None:
            if aligned._scaling is not self._scaling:
                aligned = self._scaling.transform(aligned)
        elif aligned._scaling is not None:
            raise ValueError("This model uses unscaled inputs. Supply data in the original units")
        values = aligned._to_jax(dtype=self._dtype)
        if self._time_origin is not None:
            values.update(_model_time_inputs(aligned, self._time_inputs, self._time_origin, dtype=self._dtype))
        values.update({name: self._data.values[name] for name in self._input_dims})
        prepared = replace(
            self._data,
            values=values,
            reference_values=dict(self._data.reference_values),
            n_periods=len(aligned.time_values),
        )
        return self._refresh_transformed_data(prepared), aligned

    def _refresh_transformed_data(self, inputs: _ModelData) -> _ModelData:
        """Recompute fixed calculations when the supplied observations change."""
        if self._transformed_data is None:
            return inputs
        assert self._data is not None
        expected_names = set(self._data.transformed_values) | dict(self._data.static_values).keys()
        return _compute_transformed_data(
            self._transformed_data,
            self._transformed_data_inputs,
            inputs,
            tuple(self.parameters),
            expected_names=expected_names,
        )

    def _replace_data_values(self, inputs: _ModelData, values: dict[str, jax.Array]) -> _ModelData:
        """Prepare differentiable scenario inputs before evaluating parameter draws."""
        return self._refresh_transformed_data(replace(inputs, values=values))

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

    def evaluate(self, parameters: ParameterValues, data: object = None) -> dict[str, jax.Array]:
        """Inspect deterministic model quantities at chosen parameter values.

        Evaluate transformed parameters without sampling or
        calling the density or generation functions. Supports JIT, automatic
        differentiation, and batching through ``jax.vmap``.

        Parameters
        ----------
        parameters : mapping of str to array_like
            Constrained values for every declared parameter, matching its
            shape and constraints. Values use the declaration's dtype.
        data : object, optional
            Prepared model inputs from ``model.prepare_data``. Defaults to
            stored training inputs. Prepare new data outside JAX transformations.

        Returns
        -------
        dict of str to jax.Array
            All quantities returned by ``transformed_parameters``, regardless
            of ``save``. Returns an empty dictionary when no transformation
            function is supplied.
        """
        values = self._constrained_values(parameters)
        if self._data is None:
            return {}

        inputs = self._validated_data(self._data if data is None else data)
        return self._evaluate_quantities(inputs, values)

    def log_prob(self, parameters: ParameterValues, data: object = None) -> jax.Array:
        """Evaluate the scalar log density at constrained parameter values.

        Includes the priors and likelihood written in the density callback,
        without parameterization adjustments. This need not be a normalized
        probability density. Supports JIT, gradients, and ``jax.vmap``.

        Parameters
        ----------
        parameters : mapping of str to array_like
            Constrained values for every declared parameter, matching its
            shape and constraints. Values use the declaration's dtype.
        data : object, optional
            Defaults to stored training inputs for prepared models. For new
            observations, pass ``model.prepare_data`` output. Otherwise, pass
            the JAX-compatible data expected by the data-first callback.

        Returns
        -------
        jax.Array
            Scalar log density in model space, without constraint Jacobians
            or other parameterization adjustments.
        """
        values = self._constrained_values(parameters)
        inputs = self._data if data is None and self._data is not None else data
        return self._constrained_log_density(values, inputs)

    def _constrained_values(self, parameters: ParameterValues) -> dict[str, jax.Array]:
        """Check declared names and shapes without transforming model-space values."""
        _validate_value_names(parameters, self._parameterizations, name="parameters")
        return {
            name: _as_array(
                parameters[name],
                name=f"parameter {name!r}",
                shape=parameterization.shape,
                dtype=parameterization.dtype,
            )
            for name, parameterization in self._parameterizations
        }

    def _constrained_log_density(self, parameters: ParameterValues, data: object) -> jax.Array:
        """Share the model-space density between inspection and inference."""
        if self._data is None:
            return _as_scalar(self._log_density(data, **parameters), name="log_density")

        inputs = self._validated_data(data)
        effects = self._evaluate_quantities(inputs, parameters)
        arguments = _callback_inputs(self._density_inputs, inputs, effects, parameters, name="log_density")
        return _as_scalar(self._log_density(**arguments), name="log_density")

    def log_density(self, position: ParameterValues, data: object) -> jax.Array:
        r"""Evaluate the adjusted scalar log density in inference space.

        For an unconstrained position :math:`z` and parameter mapping
        :math:`\theta = T(z)`, the returned density is

        .. math::

            \log p_z(z) = \log p_\theta(T(z))
            + \sum_k A_k(z_k),

        where :math:`A_k` is the log-density adjustment supplied by each
        parameterization. The callback supplies all priors and likelihood
        terms in :math:`p_\theta`.

        Parameters
        ----------
        position : mapping of str to array_like
            Unconstrained values for every declared parameter, with each
            value matching its declaration's ``position_shape``.
        data : object
            For a prepared model, pass ``model.data`` or the result of
            ``model.prepare_data``. Otherwise, pass a JAX-compatible PyTree
            received as the callback's first argument.

        Returns
        -------
        jax.Array
            Scalar callback log density plus parameterization adjustments.
        """
        parameters = self.constrain(position)
        density = self._constrained_log_density(parameters, data)

        for name, parameterization in self._parameterizations:
            adjustment = _as_scalar(
                parameterization.log_density_adjustment(position[name]),
                name=f"log-density adjustment for {name!r}",
            )
            density = density + adjustment

        return density

    def generate_quantities(
        self,
        key: jax.Array,
        parameters: ParameterValues,
        data: object,
    ) -> dict[str, jax.Array | dict[str, jax.Array]]:
        """Evaluate saved and generated quantities from constrained model parameters.

        Parameters
        ----------
        key : jax.Array
            JAX random key passed to the ``generated_quantities`` callback. The callback
            must split it when drawing multiple independent samples.
        parameters : mapping of str to array_like
            Constrained values for every declared parameter. Only the names
            requested by the ``generated_quantities`` callback are passed to it.
        data : object
            For a prepared model, pass ``model.data`` or the result of
            ``model.prepare_data``. Otherwise, pass a JAX-compatible PyTree
            received as the callback's second argument.

        Returns
        -------
        dict of str to jax.Array or dict of str to jax.Array
            Saved transformed quantities and ordinary callback outputs by name.
            Outputs returned under ``predictive``, ``log_likelihood``, and
            ``log_prior``, together with mapped log-prior terms, appear as
            mappings under those keys when present.
        """
        quantities = self._generate_with_inputs(key, parameters, data)[0]
        outputs: dict[str, jax.Array | dict[str, jax.Array]] = {
            name: value for (group, name), value in quantities.items() if group == "generated"
        }
        for group in get_args(_ResultGroup):
            grouped = {name: value for (kind, name), value in quantities.items() if kind == group}
            if grouped:
                outputs[group] = grouped
        return outputs

    def _generate_with_inputs(
        self,
        key: jax.Array,
        parameters: ParameterValues,
        data: object,
    ) -> tuple[dict[_OutputKey, jax.Array], dict[str, ArrayLike]]:
        """Return outputs keyed by result group and name, with the callback inputs for labeling."""
        if not self._has_generated_quantities:
            raise RuntimeError(
                "Generated quantities are unavailable because this model has no generated_quantities callback, "
                "save selection, or mapped prior definitions"
            )

        constrained = self._constrained_values(parameters)
        saved: dict[str, ArrayLike] = {}
        arguments: dict[str, _CallbackValue]
        if self._data is None:
            names = parameters if self._generate_parameter_names is None else self._generate_parameter_names
            arguments = {name: constrained[name] for name in names}
            generated = {} if self._generate is None else self._generate(key, data, **arguments)
        else:
            inputs = self._validated_data(data)
            effects = self._evaluate_quantities(inputs, constrained)
            bindings = tuple(dict((*self._generation_inputs, *self._saved_inputs)).items())
            arguments = _callback_inputs(bindings, inputs, effects, constrained, name="generated_quantities")
            saved = {name: effects[name] for name, _ in self._saved_inputs}
            callback_arguments = {name: arguments[name] for name, _ in self._generation_inputs}
            generated = {} if self._generate is None else self._generate(key, **callback_arguments)
        if not isinstance(generated, Mapping):
            raise TypeError(
                "generated_quantities must return a mapping from quantity names to values, "
                f"got {type(generated).__name__}"
            )

        outputs: dict[_OutputKey, ArrayLike] = {}
        result_groups: tuple[_ResultGroup, ...] = get_args(_ResultGroup)
        for group in result_groups:
            if group not in generated:
                continue
            grouped = generated[group]
            if not isinstance(grouped, Mapping):
                raise TypeError(
                    f"generated_quantities must return a mapping of named outputs under {group!r}, "
                    f"got {type(grouped).__name__}"
                )
            for output_name, output in grouped.items():
                self._validate_generated_name(output_name, label=f"{group} quantity")
                outputs[(group, output_name)] = output
        for name, value in generated.items():
            if name in result_groups:
                continue
            self._validate_generated_name(name, label="generated quantity")
            outputs[("generated", name)] = value

        conflicts = {name for name in saved if ("generated", name) in outputs}
        if conflicts:
            raise ValueError(
                f"Saved quantities {sorted(conflicts)} are also returned by generated_quantities. "
                "Choose one place to retain them"
            )
        outputs.update({("generated", name): value for name, value in saved.items()})

        for name, prior in self._priors:
            output_name = f"log_prior_{name}"
            if ("log_prior", output_name) in outputs or ("generated", output_name) in outputs:
                raise ValueError(f"Generated quantity {output_name!r} conflicts with a mapped prior output")
            outputs[("log_prior", output_name)] = prior.logpdf(constrained[name])

        quantities: dict[_OutputKey, jax.Array] = {}
        for output_key, value in sorted(outputs.items()):
            try:
                quantities[output_key] = jnp.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"generated quantity {output_key[1]!r} must be array-like, got {type(value).__name__}"
                ) from exc
        numeric_arguments = {name: value for name, value in arguments.items() if not callable(value)}
        return quantities, numeric_arguments

    def _validate_generated_name(self, name: object, *, label: str) -> None:
        """Reject output names that would shadow model-supplied inputs."""
        _validate_name(name, label=label)
        if self._data is not None and name in self._data.reserved_names:
            raise ValueError(f"Generated quantity {name!r} conflicts with a model-supplied input")

    def _evaluate_quantities(self, inputs: _ModelData, parameters: ParameterValues) -> dict[str, jax.Array]:
        """Evaluate the shared deterministic calculations with current inputs."""
        effects: dict[str, jax.Array] = {}
        if self._transformed_parameters is None:
            return effects

        arguments = _callback_inputs(self._transform_inputs, inputs, effects, parameters, name="transformed_parameters")
        transformed = self._transformed_parameters(**arguments)
        if not isinstance(transformed, Mapping):
            raise TypeError("transformed_parameters must return a mapping from quantity names to array-like values")

        # Retain training names even when prediction omits their observation arrays.
        assert self._data is not None
        reserved = _callback_data_names(self._data) | set(parameters)
        for name in transformed:
            _validate_name(name, label="transformed quantity")
            if name in reserved:
                raise ValueError(f"Transformed quantity {name!r} conflicts with a data role or parameter")
        for name, value in transformed.items():
            try:
                effects[name] = jnp.asarray(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Transformed quantity {name!r} must be array-like, got {type(value).__name__}"
                ) from error
        return effects

    def _validated_data(self, data: object) -> _ModelData:
        """Keep prepared inputs tied to this model and its training labels."""
        if not isinstance(data, _ModelData):
            raise TypeError("Prepared models require model.data or the result of model.prepare_data")
        if self._data is None or data.owner is not self._data.owner:
            raise ValueError("The prepared inputs must belong to this model and use its training labels")
        return data


def _prepare_inputs(
    inputs: xr.Dataset | None,
    data: PreparedData,
    parameter_names: tuple[str, ...],
    coordinates: Mapping[str, NDArray[np.generic]],
    constant_names: tuple[str, ...] = (),
) -> tuple[dict[str, jax.Array], dict[str, tuple[str, ...]], dict[str, NDArray[np.generic]]]:
    """Validate fixed labeled inputs without aligning or rescaling their values."""
    if inputs is None:
        return {}, {}, {}
    if not isinstance(inputs, xr.Dataset):
        raise TypeError("inputs must be an xarray.Dataset with named numeric variables")

    dimensions = _dimensions(
        {_name(name): tuple(_name(axis) for axis in value.dims) for name, value in inputs.data_vars.items()}
    )
    if set(get_args(_TimeInput)) & inputs.sizes.keys():
        raise ValueError("Additional inputs remain fixed across scenarios. Use axes other than time or media_time")
    if {"chain", "draw", "sample", "pred_id"} & inputs.sizes.keys():
        raise ValueError("Additional inputs must not contain sample dimensions")
    _, group_labels = _prepared_coordinates(data)
    role_names = set(_data_dimensions(data)) | set(group_labels)
    conflicts = role_names & inputs.sizes.keys()
    if conflicts:
        raise ValueError(f"Input dimensions {sorted(conflicts)} conflict with prepared data variables")
    for name, coordinate in inputs.coords.items():
        if coordinate.dims != (name,):
            raise ValueError(f"Input coordinate {name!r} must label only its own dimension")

    input_coords = _coordinates(
        {
            _name(axis): inputs.coords[axis].values if axis in inputs.coords else np.arange(size)
            for axis, size in inputs.sizes.items()
        }
    )
    for axis, labels in input_coords.items():
        if not inputs.get_index(axis).is_unique:
            raise ValueError(f"Input coordinate {axis!r} must have unique labels")
        if axis in coordinates and not _same_labels(coordinates[axis], labels):
            raise ValueError(f"Input coordinate {axis!r} must match the model labels and ordering")

    reserved = role_names | set(get_args(_TimeInput)) | set(parameter_names) | set(coordinates) | set(input_coords)
    reserved |= set(constant_names)
    values = {}
    for name in dimensions:
        _validate_name(name, label="input")
        if name in reserved:
            raise ValueError(f"Input {name!r} conflicts with a data role, parameter, or coordinate")
        array = np.array(inputs[name].values, copy=True)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"Input {name!r} must contain finite real numbers or booleans")
        dtype = jax.dtypes.canonicalize_dtype(array.dtype.newbyteorder("="))
        if array.dtype.kind in "iu":
            limits = np.iinfo(dtype)
            if np.any(array < limits.min) or np.any(array > limits.max):
                raise ValueError(f"Input {name!r} contains integers outside the JAX dtype range. Enable 64-bit mode")
        with np.errstate(over="ignore"):
            array = array.astype(dtype)
        if not np.isfinite(array).all():
            raise ValueError(
                f"Input {name!r} is not finite at the current JAX precision. Rescale it or enable 64-bit mode"
            )
        values[name] = jnp.asarray(array)
    return values, dimensions, input_coords


def _requested_inputs(
    density: Callable[..., object] | None,
    transformed: TransformedParameters | None,
    generated_quantities: GeneratedQuantities | None,
) -> set[str]:
    """Inspect input names without evaluating user functions."""
    names: set[str] = set()
    for function, skip in ((density, 0), (transformed, 0), (generated_quantities, 1)):
        if function is None:
            continue
        try:
            names.update(list(signature(function).parameters)[skip:])
        except (TypeError, ValueError):
            # Callback validation supplies the function-specific error.
            continue
    return names


def _model_time_inputs(
    data: PreparedData,
    names: tuple[str, ...],
    origin: float | datetime,
    *,
    dtype: DTypeLike = float,
) -> dict[str, jax.Array]:
    """Supply numeric positions without selecting a seasonal or time-varying model."""
    inputs = {}
    for name in names:
        labels = data.media_time_values if name.startswith("media_") else data.time_values
        if not labels:
            continue
        if name.endswith("day_of_year"):
            positions = _day_of_year(labels)
        else:
            positions, _ = _time_positions(labels, origin=origin)
        inputs[name] = jnp.asarray(positions, dtype=dtype)
    return inputs


def _result_names(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    """Copy distinct generated output names without evaluating their callback."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of generated output names")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty string names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate output names")
    return tuple(values)


def _data_variable_declarations(
    declarations: Mapping[str, str] | None,
    parameter_names: tuple[str, ...],
    constant_names: tuple[str, ...] = (),
) -> dict[str, str] | None:
    """Copy explicit callback names without changing the underlying data layout."""
    if declarations is None:
        return None

    # Data validates the names and sources before the model receives them.
    for name in declarations:
        if name in parameter_names:
            raise ValueError(f"Data variable {name!r} conflicts with a declared parameter")
        if name in constant_names:
            raise ValueError(f"Data variable {name!r} conflicts with a constant")

    return dict(declarations)


def _data_sources(data: _ModelData) -> dict[str, set[str]]:
    """List physical data sources separately from user-chosen function names."""
    sources = {
        "data": set(data.values),
        "reference": set(data.reference_values),
        "builtin": {"n_periods", "reference_n_periods"},
        "constant": {name for name, _ in data.constants},
    }
    if data.outcome_scaling is not None:
        sources["builtin"].update(("outcome_scale", "outcome_offset", "unscale_outcome"))
    return sources


def _resolve_data_variables(declarations: Mapping[str, str], data: _ModelData) -> _DataVariables:
    """Validate declarations against prepared and auxiliary data sources."""
    sources = _data_sources(data)
    resolved = []
    for name, source_name in declarations.items():
        matches = [kind for kind, names in sources.items() if source_name in names]
        if not matches:
            raise ValueError(
                f"Data variable {name!r} refers to unavailable source {source_name!r}. "
                "Select it in prepare_data or supply it through inputs"
            )
        if len(matches) != 1:
            raise ValueError(f"Source {source_name!r} for data variable {name!r} is ambiguous")
        resolved.append((name, matches[0], source_name))
    return tuple(resolved)


def _callback_data_names(data: _ModelData) -> set[str]:
    """Identify data names visible to the model's program blocks."""
    fixed = set(data.transformed_values) | dict(data.static_values).keys()
    if data.variable_sources is not None:
        return {name for name, _, _ in data.variable_sources} | fixed | {name for name, _ in data.constants}
    return set().union(*_data_sources(data).values(), data.reserved_names, fixed)


def _input_source(name: str, source: str, data: _ModelData) -> tuple[str, str]:
    """Resolve a declared argument to its physical source for evaluation and labels."""
    if source != "variable":
        return source, name
    assert data.variable_sources is not None
    for argument, kind, source_name in data.variable_sources:
        if name == argument:
            return kind, source_name
    raise ValueError(f"Unknown declared data variable {name!r}")


def _metadata_source(name: str, source: str, data: _ModelData) -> tuple[str, str]:
    """Retain labels for unchanged inputs returned by transformed_data."""
    source, name = _input_source(name, source, data)
    if source == "transformed_data":
        for output, kind, source_name in data.transformed_sources:
            if name == output:
                return kind, source_name
    return source, name


def _compute_transformed_data(
    function: TransformedData,
    bindings: _InputBindings,
    data: _ModelData,
    parameter_names: tuple[str, ...],
    *,
    expected_names: set[str] | None = None,
) -> _ModelData:
    """Evaluate and validate data-only calculations outside the parameter loop."""
    base = replace(data, transformed_values={}, static_values=(), transformed_sources=())
    arguments = _callback_inputs(bindings, base, {}, {}, name="transformed_data")
    outputs = function(**arguments)
    if not isinstance(outputs, Mapping):
        raise TypeError("transformed_data must return a mapping from names to fixed values")

    reserved = _callback_data_names(base) | set(parameter_names)
    arrays = {}
    static = []
    origins = []
    for name, value in outputs.items():
        _validate_name(name, label="transformed data")
        if name in reserved:
            raise ValueError(f"Transformed data {name!r} conflicts with an input or parameter")
        if isinstance(value, (bool, np.bool_)):
            static.append((name, bool(value)))
        elif isinstance(value, Integral):
            static.append((name, int(value)))
        else:
            if value is None or callable(value):
                raise TypeError(f"Transformed data {name!r} must contain real numeric values")
            try:
                array = jnp.array(value, copy=not isinstance(value, jax.Array))
            except (TypeError, ValueError) as error:
                raise TypeError(f"Transformed data {name!r} must contain real numeric values") from error
            if array.dtype.kind not in "biuf":
                raise TypeError(f"Transformed data {name!r} must contain real numeric values")
            if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
                raise ValueError(f"Transformed data {name!r} must contain finite values")
            arrays[name] = array

        inherited = {
            _input_source(argument, source, base) for argument, source in bindings if value is arguments[argument]
        }
        if len(inherited) == 1:
            source, source_name = inherited.pop()
            origins.append((name, source, source_name))

    if expected_names is not None and set(outputs) != expected_names:
        raise ValueError("transformed_data must return the same names for each dataset")
    return replace(
        data,
        transformed_values=arrays,
        static_values=tuple(sorted(static)),
        transformed_sources=tuple(sorted(origins)),
    )


def _bind_inputs(
    function: Callable[..., object],
    parameter_names: tuple[str, ...],
    data: _ModelData,
    *,
    name: str,
    upstream_parameters: tuple[str, ...] = (),
    has_transformed: bool = False,
) -> _InputBindings:
    """Resolve ordinary or keyword-only callback inputs by name during preparation."""
    if not callable(function):
        raise TypeError(f"{name} must be callable, got {type(function).__name__}")
    try:
        arguments = list(signature(function).parameters.values())
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose an inspectable Python signature") from error

    key_argument = None
    if name == "generated_quantities":
        if not arguments or arguments[0].kind not in (
            SignatureParameter.POSITIONAL_ONLY,
            SignatureParameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("generated_quantities must accept a random key as its first positional argument")
        key_argument = arguments.pop(0)
    if any(
        argument.kind not in (SignatureParameter.POSITIONAL_OR_KEYWORD, SignatureParameter.KEYWORD_ONLY)
        for argument in arguments
    ):
        raise TypeError(
            f"Inputs for {name} must be named arguments without positional-only parameters, *args or **kwargs"
        )

    sources = (
        _data_sources(data)
        if data.variable_sources is None
        else {
            "variable": {variable for variable, _, _ in data.variable_sources},
            "constant": {name for name, _ in data.constants},
        }
    )
    sources["transformed_data"] = set(data.transformed_values) | dict(data.static_values).keys()
    sources["parameter"] = set(parameter_names)
    if key_argument is not None and (
        key_argument.name in data.reserved_names or any(key_argument.name in names for names in sources.values())
    ):
        raise TypeError(f"generated_quantities places input {key_argument.name!r} where the random key is required")

    bindings = []
    for argument in arguments:
        matches = [source for source, names in sources.items() if argument.name in names]
        if not matches:
            if name == "transformed_data":
                raise ValueError(
                    f"transformed_data requests unknown data variable {argument.name!r}. "
                    "Declare its source in Data variables. Sampled parameters are not available here"
                )
            if argument.name in data.reserved_names:
                role = argument.name.removeprefix("reference_") if argument.name.startswith("reference_") else "outcome"
                raise ValueError(
                    f"{name} requests {argument.name!r}, which requires {role!r} in the original prepared data"
                )
            if has_transformed:
                bindings.append((argument.name, "transformed"))
                continue
            available = sorted(set().union(*sources.values()))
            if data.variable_sources is not None:
                raise ValueError(
                    f"{name} requests unknown input {argument.name!r}. "
                    "Declare its source in Data variables or declare it as a parameter. "
                    f"Available inputs are {available}"
                )
            if argument.name in ("data", "effects"):
                raise TypeError(
                    f"{name} no longer receives data or effects bundles in prepared models. "
                    "Request individual inputs by name, such as outcome or a declared parameter"
                )
            raise ValueError(
                f"{name} requests unknown input {argument.name!r}. Use a selected data role or declared parameter. "
                f"Available inputs are {available}, with time, media_time, day_of_year, media_day_of_year, "
                "and reference_ inputs on request"
            )
        if len(matches) > 1:
            raise ValueError(
                f"{name} input {argument.name!r} is ambiguous across {matches}. "
                "Use distinct names for data roles and parameters"
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
) -> dict[str, _CallbackValue]:
    """Supply current inputs, fixed references, and explicit unit conversions."""
    sources: dict[str, ParameterValues] = {
        "data": data.values,
        "parameter": parameters,
        "transformed": effects,
        "reference": data.reference_values,
        "transformed_data": {**data.transformed_values, **dict(data.static_values)},
        "constant": dict(data.constants),
    }
    arguments: dict[str, _CallbackValue] = {}
    for argument, source in bindings:
        source, source_name = _input_source(argument, source, data)
        if source == "builtin":
            if source_name == "n_periods":
                arguments[argument] = data.n_periods
            elif source_name == "reference_n_periods":
                arguments[argument] = data.reference_n_periods
            else:
                assert data.outcome_scaling is not None
                if source_name == "unscale_outcome":
                    arguments[argument] = data.outcome_scaling.inverse_transform
                else:
                    value = (
                        data.outcome_scaling.offset if source_name == "outcome_offset" else data.outcome_scaling.scale
                    )
                    arguments[argument] = value.reshape(-1) if data.outcome_group_scale else value.reshape(())
            continue
        if source_name not in sources[source]:
            if source == "transformed":
                if data.variable_sources is not None:
                    raise ValueError(
                        f"{name} requires input {argument!r}. "
                        "Return it from transformed_parameters or declare its source in Data variables"
                    )
                if argument in ("data", "effects"):
                    raise ValueError(
                        f"{name} requires transformed quantity {argument!r}. "
                        "Prepared callbacks receive individual inputs by name, not data or effects bundles"
                    )
                raise ValueError(
                    f"{name} requires transformed quantity {argument!r}. Return it from transformed_parameters"
                )
            raise ValueError(f"{name} requires input {argument!r}. Include it when preparing data for this evaluation")
        arguments[argument] = sources[source][source_name]
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
        # Pending built-in dimensions are resolved before numerical protocol properties are used.
        if not isinstance(parameters[name], _BuiltinParameter) and not isinstance(parameters[name], Parameterization):
            raise TypeError(
                f"parameter {name!r} must implement Parameterization, got {type(parameters[name]).__name__}"
            )
    return tuple(sorted(parameters.items()))


def _resolve_parameter_dimensions(
    parameterizations: tuple[tuple[str, Parameterization], ...],
    dimensions: dict[str, tuple[str, ...]],
    coordinates: Mapping[str, NDArray[np.generic]],
) -> tuple[tuple[str, Parameterization], ...]:
    """Resolve named built-in shapes without changing reusable declarations."""
    resolved = []
    for name, parameter in parameterizations:
        if isinstance(parameter, _BuiltinParameter) and parameter.dims:
            axes = tuple(parameter.dims)
            if name in dimensions and dimensions[name] != axes:
                raise ValueError(
                    f"Model dimensions for parameter {name!r} conflict with its declared dimensions {axes}"
                )
            shape = parameter.shape
            if not shape:
                missing = set(axes) - coordinates.keys()
                if missing:
                    raise ValueError(
                        f"Unknown dimensions {sorted(missing)} for parameter {name!r}. "
                        "Use prepared data axes or supply their labels in coords"
                    )
                shape = tuple(len(coordinates[axis]) for axis in axes)
            parameter = replace(parameter, shape=shape)
            dimensions[name] = axes
        resolved.append((name, parameter))
    return tuple(resolved)


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
    values: Mapping[str, object],
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
        raise ValueError(f"{name} does not match the model parameters. {details}")


def _name_mismatch_details(missing: list[str], unexpected: list[str]) -> str:
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    return ". ".join(details)


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
