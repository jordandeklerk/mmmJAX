"""Model composition for transparent JAX probability models."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, override

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray

from mmmjax._binding import (
    _bind_inputs,
    _callback_data_names,
    _callback_inputs,
    _compute_transformed_data,
    _data_sources,
    _FrozenMapping,
    _InputBindings,
    _ModelData,
    _name_mismatch_details,
    _resolve_data_variables,
    _validate_generate_signature,
    _validate_log_density_signature,
    _validate_name,
)
from mmmjax._results import _coordinates, _dimensions, _name, _prepared_coordinates, _same_labels
from mmmjax.data import (
    Data,
    PreparedData,
    Reference,
    _calendar_dates,
    _data_dimensions,
    _DataLayout,
    _day_of_year,
    _prepare_model_frame,
    _time_input_names,
    _time_positions,
    _TimeInput,
    _validate_calendar_anchor,
    _validate_calendar_contiguity,
)
from mmmjax.parameters import Parameterization, _as_array, _Dimensioned
from mmmjax.scaling import DataScaling, Scaling, fit_data_scaling

__all__ = ["Model"]

type LogDensity = Callable[..., ArrayLike]
type GeneratedQuantities = Callable[..., Mapping[str, ArrayLike]]
type PriorSampler = Callable[[jax.Array], Mapping[str, ArrayLike]]
type TransformedParameters = Callable[..., Mapping[str, ArrayLike]]
type TransformedData = Callable[..., Mapping[str, ArrayLike]]
type ParameterValues = Mapping[str, ArrayLike]
type _Parameterizations = tuple[tuple[str, Parameterization], ...]
type _Coordinates = dict[str, NDArray[np.generic]]
type _OutputGroup = Literal["generated"] | _ResultGroup
type _OutputKey = tuple[_OutputGroup, str]


class _ResultGroup(StrEnum):
    """Result groups a ``generated_quantities`` callback can return outputs under."""

    PREDICTIVE = "predictive"
    LOG_LIKELIHOOD = "log_likelihood"
    LOG_PRIOR = "log_prior"


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class _Blocks(ABC):
    """Hold the program blocks and decide how each one receives its inputs.

    Prepared models resolve inputs by argument name once at construction.
    Models without prepared data pass the caller's bundle positionally.
    Both strategies share the same evaluation interface so the model never
    branches on which one it holds.
    """

    log_density: LogDensity
    generate: GeneratedQuantities | None = None
    transformed_parameters: TransformedParameters | None = None
    transformed_data: TransformedData | None = None
    transformed_data_inputs: _InputBindings = ()
    transform_inputs: _InputBindings = ()
    density_inputs: _InputBindings = ()
    generation_inputs: _InputBindings = ()

    @property
    def bindings(self) -> _InputBindings:
        """Concatenate the resolved inputs of every block for result labeling."""
        return (*self.transformed_data_inputs, *self.transform_inputs, *self.density_inputs, *self.generation_inputs)

    @abstractmethod
    def evaluate_transformed(self, parameters: Mapping[str, jax.Array], data: object) -> dict[str, jax.Array]:
        """Evaluate ``transformed_parameters`` at constrained values."""

    @abstractmethod
    def evaluate_density(self, parameters: Mapping[str, jax.Array], data: object) -> jax.Array:
        """Evaluate the scalar model-space log density."""

    @abstractmethod
    def evaluate_generated(
        self, key: jax.Array, parameters: Mapping[str, jax.Array], data: object
    ) -> tuple[object, dict[str, object]]:
        """Call ``generated_quantities`` and return its raw output with the inputs it received."""


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class _RawBlocks(_Blocks):
    """Pass the caller's data bundle as the leading positional argument."""

    generate_parameter_names: tuple[str, ...] | None = None

    @override
    def evaluate_transformed(self, parameters: Mapping[str, jax.Array], data: object) -> dict[str, jax.Array]:
        return {}

    @override
    def evaluate_density(self, parameters: Mapping[str, jax.Array], data: object) -> jax.Array:
        return _as_scalar(self.log_density(data, **parameters), name="log_density")

    @override
    def evaluate_generated(
        self, key: jax.Array, parameters: Mapping[str, jax.Array], data: object
    ) -> tuple[object, dict[str, object]]:
        assert self.generate is not None
        names = parameters if self.generate_parameter_names is None else self.generate_parameter_names
        arguments: dict[str, object] = {name: parameters[name] for name in names}
        return self.generate(key, data, **arguments), arguments


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class _PreparedBlocks(_Blocks):
    """Supply named inputs from a prepared data bundle owned by the model."""

    training: _ModelData

    def validated_data(self, data: object) -> _ModelData:
        """Keep prepared inputs tied to this model and its training labels."""
        if not isinstance(data, _ModelData):
            raise TypeError("Prepared models require model.data or the result of model.prepare_data")
        if data.owner is not self.training.owner:
            raise ValueError("The prepared inputs must belong to this model and use its training labels")
        return data

    @override
    def evaluate_transformed(self, parameters: Mapping[str, jax.Array], data: object) -> dict[str, jax.Array]:
        return self._transformed(parameters, self.validated_data(data))

    @override
    def evaluate_density(self, parameters: Mapping[str, jax.Array], data: object) -> jax.Array:
        inputs = self.validated_data(data)
        effects = self._transformed(parameters, inputs)
        arguments = _callback_inputs(self.density_inputs, inputs, effects, parameters, name="log_density")
        return _as_scalar(self.log_density(**arguments), name="log_density")

    @override
    def evaluate_generated(
        self, key: jax.Array, parameters: Mapping[str, jax.Array], data: object
    ) -> tuple[object, dict[str, object]]:
        assert self.generate is not None
        inputs = self.validated_data(data)
        effects = self._transformed(parameters, inputs)
        arguments = _callback_inputs(self.generation_inputs, inputs, effects, parameters, name="generated_quantities")
        return self.generate(key, **arguments), arguments

    def _transformed(self, parameters: Mapping[str, jax.Array], inputs: _ModelData) -> dict[str, jax.Array]:
        """Evaluate the shared deterministic calculations with current inputs."""
        effects: dict[str, jax.Array] = {}
        if self.transformed_parameters is None:
            return effects

        arguments = _callback_inputs(self.transform_inputs, inputs, effects, parameters, name="transformed_parameters")
        transformed = self.transformed_parameters(**arguments)
        if not isinstance(transformed, Mapping):
            raise TypeError("transformed_parameters must return a mapping from quantity names to array-like values")

        # Retain training names even when prediction omits their observation arrays.
        reserved = _callback_data_names(self.training) | set(parameters)
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


@dataclass(frozen=True, slots=True, eq=False)
class _Training:
    """Record the layout and calendar of the fitted observations along with their scaling."""

    layout: _DataLayout
    time_column: str | None
    frequency: str | None
    time_values: tuple[object, ...]
    media_time_values: tuple[object, ...]
    time_inputs: tuple[str, ...]
    time_origin: float | datetime | None
    scaling: DataScaling | None
    input_dims: dict[str, tuple[str, ...]]
    input_coords: _Coordinates


@dataclass(frozen=True, slots=True, eq=False)
class _Declarations:
    """Carry the ``Data`` declarations that accompany prepared observations."""

    variables: dict[str, str] | None = None
    inputs: xr.Dataset | None = None
    constants: dict[str, object] | None = None
    scaling: DataScaling | Literal["auto"] | None = None

    @property
    def constant_values(self) -> dict[str, object]:
        """Return the declared constants or an empty mapping when there are none."""
        return {} if self.constants is None else self.constants


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class Model:
    """Compose parameter declarations and program blocks into one model.

    A model evaluates its program blocks in order, from ``data`` and
    ``transformed_data`` through ``parameters``, ``transformed_parameters``,
    and ``log_density`` to ``generated_quantities``. Each function requests the
    inputs it needs by argument name from data inputs, the outputs of earlier
    blocks, and parameters. Models without prepared data receive the data as
    a leading positional argument instead.

    Prepared data supplies the selected role names, such as ``outcome`` and
    ``media``, together with elapsed ``time`` and ``media_time``, the calendar
    ``day_of_year`` and ``media_day_of_year``, the period count ``n_periods``,
    the fitted outcome transform ``outcome_scaling`` with its ``scale``,
    ``offset``, and ``inverse_transform``, and the ``reference`` namespace
    holding every training input, such as ``reference.spend``, for evaluating
    new data. ``Data`` variables replace these names with declared ones.

    Inspect derived quantities with ``evaluate`` and the constrained log
    density with ``log_prob`` before fitting. Samplers call ``log_density``
    on unconstrained positions and add parameterization adjustments.

    Parameters
    ----------
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
    parameters : mapping of str to Parameterization
        Required. All named parameter declarations and constraints. Use declaration
        dimensions, such as ``Real(dims="control")``, to infer shapes and
        result labels from prepared data or ``coords``.
    transformed_parameters : callable, optional
        Pure JAX-compatible function returning a mapping of names to derived
        arrays shared by the density and generated-quantities callbacks.
        Evaluated for density and output calculations. Requires prepared data
        and named inputs. Names must not shadow declared inputs or parameters.
        Outputs are not sampled parameters. Return them from
        ``generated_quantities`` to retain them in results.
        All requested inputs are needed for output evaluation.
    log_density : callable
        Required. Scalar log density for constrained parameters. Every user-declared
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
        priors and adjustments. Log-prior terms are prior factors, each once
        and without constraint adjustments.
    dims : mapping of str to str or sequence of str, optional
        Named axes for constrained parameter arrays, excluding chain and draw.
        A single string names one axis.
        Use for custom parameterizations or explicit-shape declarations.
        Must agree with any dimensions set on a declaration.
    coords : mapping of str to array_like, optional
        One-dimensional labels for named axes. Labels are copied at construction.
    generated_dims : mapping of str to str or sequence of str, optional
        Axis labels for saved or generated arrays, excluding chain and draw.
        A single string names one axis.
        Overrides labels inherited from unchanged data or parameter
        inputs and from observation-shaped predictive and log-likelihood
        outputs. A name shared by several result groups receives the same axes.
    """

    _parameterizations: _Parameterizations
    _blocks: _Blocks
    _data: _ModelData | None
    _training: _Training | None
    _dtype: DTypeLike
    _result_dims: dict[str, tuple[str, ...]]
    _result_coords: _Coordinates
    _generated_dims: dict[str, tuple[str, ...]]

    def __init__(
        self,
        data: Data | PreparedData | None = None,
        transformed_data: TransformedData | None = None,
        parameters: Mapping[str, Parameterization] | None = None,
        transformed_parameters: TransformedParameters | None = None,
        log_density: LogDensity | None = None,
        generated_quantities: GeneratedQuantities | None = None,
        *,
        dims: Mapping[str, str | Sequence[str]] | None = None,
        coords: Mapping[str, object] | None = None,
        generated_dims: Mapping[str, str | Sequence[str]] | None = None,
    ) -> None:
        """Create a model from its program blocks in evaluation order."""
        required = {"parameters": parameters, "log_density": log_density}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise TypeError(f"Model requires {' and '.join(missing)}")
        assert parameters is not None and log_density is not None
        parameterizations = _prepare_parameterizations(parameters)
        parameter_names = tuple(name for name, _ in parameterizations)
        prepared, declarations = _unpack_data(data, parameter_names)

        result_dims = _dimensions(dims)
        result_coords = _coordinates(coords)
        axis_coordinates = result_coords.copy()
        model_data: _ModelData | None = None
        training: _Training | None = None
        if prepared is not None:
            model_data, training, axis_coordinates = _prepare_training(
                prepared, declarations, parameter_names, axis_coordinates
            )
        elif transformed_data is not None:
            raise ValueError("transformed_data requires prepared data")
        parameterizations = _resolve_parameter_dimensions(parameterizations, result_dims, axis_coordinates)

        blocks, model_data = _bind_blocks(
            model_data,
            parameter_names,
            log_density=log_density,
            generated_quantities=generated_quantities,
            transformed_parameters=transformed_parameters,
            transformed_data=transformed_data,
        )

        output_dims = _dimensions(generated_dims)
        if generated_quantities is None and output_dims:
            raise ValueError("Generated result metadata requires a generated_quantities callback")
        _validate_result_dims(result_dims, parameterizations, axis_coordinates)

        object.__setattr__(self, "_parameterizations", parameterizations)
        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_data", model_data)
        object.__setattr__(self, "_training", training)
        object.__setattr__(self, "_dtype", jax.dtypes.canonicalize_dtype(float))
        object.__setattr__(self, "_result_dims", result_dims)
        object.__setattr__(self, "_result_coords", result_coords)
        object.__setattr__(self, "_generated_dims", output_dims)

    def __repr__(self) -> str:
        """Summarize the declared parameters and blocks without printing data arrays."""
        blocks = {
            "transformed_data": self._blocks.transformed_data,
            "transformed_parameters": self._blocks.transformed_parameters,
            "log_density": self._blocks.log_density,
            "generated_quantities": self._blocks.generate,
        }
        present = [name for name, block in blocks.items() if block is not None]
        data = "prepared" if self._data is not None else "none"
        return f"Model(parameters={list(self.parameters)}, blocks={present}, data={data})"

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
            A copy of the input declarations, with constants mapped to their
            own names. Without explicit declarations, every standard name maps
            to itself. Models without prepared data return an empty mapping.
        """
        if self._data is None:
            return {}
        constants = {name: name for name in self._data.constants}
        if self._data.variable_sources is not None:
            declared = {name: origin.name for name, origin in self._data.variable_sources.items()}
            return declared | constants
        available = set().union(*_data_sources(self._data).values())
        return {name: name for name in sorted(available)}

    @property
    def _has_generated_quantities(self) -> bool:
        """Indicate whether evaluation has generated outputs."""
        return self._blocks.generate is not None

    @property
    def scaling(self) -> DataScaling | None:
        """Return the fitted input transformations or None for unscaled inputs.

        Returns
        -------
        DataScaling or None
            Fitted transformations available by role through
            ``transformations``. Use the outcome transformation to restore
            predicted levels to their original units. Multiply contributions
            by its scale without adding the outcome offset.
        """
        return None if self._training is None else self._training.scaling

    @property
    def data(self) -> object:
        """Return prepared training inputs for density and generation calls.

        Pass to ``log_density`` or ``generate_quantities``, including under JIT.
        Arrays are copied at construction, independent of later source edits.

        Returns
        -------
        object
            JAX-compatible observation arrays. Callbacks
            receive their requested inputs one by one rather than this bundle.
            Requires prepared ``data`` at construction.
        """
        if self._data is None:
            raise RuntimeError("This model has no prepared data. Pass your data directly when evaluating it")
        return replace(
            self._data,
            values=dict(self._data.values),
            reference=_copy_reference(self._data.reference),
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
            Dataframes reuse the model's selections, and every scenario
            follows the calendar of the training periods without gaps.
            These calendar checks apply when the training data carries an
            inferred or declared frequency. Training prepared with
            ``frequency=None`` leaves scenario dates unchecked.
            Prepared inputs may be raw or use this model's fitted scaling.
            Omit inputs only when no evaluated callback needs them.
            For explicit media history, use ``prepare_data(media_history=...)``.

        Returns
        -------
        object
            JAX-compatible inputs for ``log_density`` or ``generate_quantities``
            that cover the supplied periods in the fitted group and channel
            order. Independent of stored model data and later source edits.
        """
        return self._prepare_data(data)[0]

    def _prepare_data(self, data: object) -> tuple[_ModelData, PreparedData]:
        """Keep aligned observation labels alongside the evaluated model inputs."""
        if self._data is None or self._training is None:
            raise RuntimeError("This model has no prepared training data. Pass your data directly when evaluating it")
        training = self._training
        if not isinstance(data, PreparedData):
            assert training.time_column is not None
            data = _prepare_model_frame(data, training.layout, time=training.time_column)
        if data.time_column != training.time_column:
            raise ValueError(f"The time column must match the training column {training.time_column!r}")
        if (
            training.frequency is not None
            and data.frequency is not None
            and training.frequency != data.frequency
            and bool({"media", "organic_media", "reach", "organic_reach"} & self._data.values.keys())
        ):
            raise ValueError(
                f"Media inputs must retain the model's {training.frequency} observation spacing. "
                "Changing frequency changes the meaning of the lag parameters"
            )
        if training.frequency is not None and all(isinstance(label, (str, date)) for label in data.time_values):
            # Scenarios never re-anchor the calendar, so labels stay on the training grid without gaps
            column = data.time_column
            frequency = training.frequency
            earliest = (training.media_time_values or training.time_values)[0]
            anchor = _calendar_dates([earliest], time=column, frequency=frequency)[0]
            _validate_calendar_anchor(data.time_values, anchor=anchor, time=column, frequency=frequency)
            _validate_calendar_contiguity(data.time_values, anchor=anchor, time=column, frequency=frequency)
            if data.media_time_values and data.media_time_values != data.time_values:
                _validate_calendar_anchor(data.media_time_values, anchor=anchor, time=column, frequency=frequency)
                _validate_calendar_contiguity(data.media_time_values, anchor=anchor, time=column, frequency=frequency)

        aligned = data._align_to(training.layout)
        if training.scaling is not None:
            if aligned._scaling is not training.scaling:
                aligned = training.scaling.transform(aligned)
        elif aligned._scaling is not None:
            raise ValueError("This model uses unscaled inputs. Supply data in the original units")
        values = aligned._to_jax(dtype=self._dtype)
        if training.time_origin is not None:
            values.update(_model_time_inputs(aligned, training.time_inputs, training.time_origin, dtype=self._dtype))
        values.update({name: self._data.values[name] for name in training.input_dims})
        prepared = replace(
            self._data,
            values=values,
            reference=_copy_reference(self._data.reference),
            n_periods=len(aligned.time_values),
        )
        return self._refresh_transformed_data(prepared), aligned

    def _refresh_transformed_data(self, inputs: _ModelData) -> _ModelData:
        """Recompute fixed calculations when the supplied observations change."""
        if self._blocks.transformed_data is None:
            return inputs
        assert self._data is not None
        return _compute_transformed_data(
            self._blocks.transformed_data,
            self._blocks.transformed_data_inputs,
            inputs,
            tuple(self.parameters),
            expected_names=self._data.fixed_names,
        )

    def _replace_data_values(self, inputs: _ModelData, values: dict[str, jax.Array]) -> _ModelData:
        """Prepare differentiable scenario inputs before evaluating parameter draws."""
        return self._refresh_transformed_data(replace(inputs, values=values))

    def constrain(self, position: ParameterValues) -> dict[str, jax.Array]:
        """Map a complete unconstrained position into model space.

        Parameters
        ----------
        position : mapping of str to array_like
            Unconstrained values for every declared parameter. Each value
            matches its declaration's ``position_shape``.

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
            Constrained values for every declared parameter. Each value
            matches its declaration's ``shape`` and constraints.

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
            Constrained values for every declared parameter. Each value
            matches its declaration's shape and constraints. Values use the
            declaration's dtype.
        data : object, optional
            Prepared model inputs from ``model.prepare_data``. Defaults to
            stored training inputs. Prepare new data outside JAX transformations.

        Returns
        -------
        dict of str to jax.Array
            All quantities returned by ``transformed_parameters``. Returns an
            empty dictionary when no transformation function is supplied.
        """
        values = self._constrained_values(parameters)
        return self._blocks.evaluate_transformed(values, self._default_data(data))

    def log_prob(self, parameters: ParameterValues, data: object = None) -> jax.Array:
        """Evaluate the scalar log density at constrained parameter values.

        Includes the priors and likelihood written in the density callback
        but no parameterization adjustments. This need not be a normalized
        probability density. Supports JIT, gradients, and ``jax.vmap``.

        Parameters
        ----------
        parameters : mapping of str to array_like
            Constrained values for every declared parameter. Each value
            matches its declaration's shape and constraints. Values use the
            declaration's dtype.
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
        return self._blocks.evaluate_density(values, self._default_data(data))

    def _default_data(self, data: object) -> object:
        """Fall back to the stored training inputs when a prepared model receives none."""
        return self._data if data is None and self._data is not None else data

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
            Unconstrained values for every declared parameter. Each value
            matches its declaration's ``position_shape``.
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
        density = self._blocks.evaluate_density(parameters, data)

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
            ``log_prior`` appear as mappings under those keys when present.
        """
        quantities = self._generate_with_inputs(key, parameters, data)[0]
        outputs: dict[str, jax.Array | dict[str, jax.Array]] = {
            name: value for (group, name), value in quantities.items() if group == "generated"
        }
        for group in _ResultGroup:
            grouped = {name: value for (kind, name), value in quantities.items() if kind == group}
            if grouped:
                outputs[group.value] = grouped
        return outputs

    def _generate_with_inputs(
        self,
        key: jax.Array,
        parameters: ParameterValues,
        data: object,
    ) -> tuple[dict[_OutputKey, jax.Array], dict[str, object]]:
        """Return outputs keyed by result group and name along with the callback inputs used for labeling."""
        if not self._has_generated_quantities:
            raise RuntimeError(
                "Generated quantities are unavailable because this model has no generated_quantities callback"
            )
        constrained = self._constrained_values(parameters)
        # Data-first callbacks taking **parameters see the caller's ordering.
        ordered = {name: constrained[name] for name in parameters}
        generated, arguments = self._blocks.evaluate_generated(key, ordered, data)
        reserved = frozenset() if self._data is None else self._data.reserved_names
        return _collect_outputs(generated, reserved), arguments


def _collect_outputs(generated: object, reserved: frozenset[str]) -> dict[_OutputKey, jax.Array]:
    """Validate and flatten callback outputs into arrays keyed by result group and name."""
    if not isinstance(generated, Mapping):
        raise TypeError(
            f"generated_quantities must return a mapping from quantity names to values, got {type(generated).__name__}"
        )

    outputs: dict[_OutputKey, object] = {}
    for group in _ResultGroup:
        if group not in generated:
            continue
        grouped = generated[group]
        if not isinstance(grouped, Mapping):
            raise TypeError(
                f"generated_quantities must return a mapping of named outputs under {group.value!r}, "
                f"got {type(grouped).__name__}"
            )
        for output_name, output in grouped.items():
            _validate_output_name(output_name, reserved, label=f"{group} quantity")
            outputs[(group, output_name)] = output
    for name, value in generated.items():
        if name in _ResultGroup:
            continue
        _validate_output_name(name, reserved, label="generated quantity")
        outputs[("generated", name)] = value

    quantities: dict[_OutputKey, jax.Array] = {}
    for output_key, value in sorted(outputs.items()):
        try:
            quantities[output_key] = jnp.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"generated quantity {output_key[1]!r} must be array-like, got {type(value).__name__}"
            ) from exc
    return quantities


def _validate_output_name(name: object, reserved: frozenset[str], *, label: str) -> None:
    """Reject output names that would shadow model-supplied inputs."""
    _validate_name(name, label=label)
    if name in reserved:
        raise ValueError(f"Generated quantity {name!r} conflicts with a model-supplied input")


def _unpack_data(
    data: Data | PreparedData | None, parameter_names: tuple[str, ...]
) -> tuple[PreparedData | None, _Declarations]:
    """Separate prepared observations from the declarations a ``Data`` wrapper adds."""
    declarations = _Declarations()
    if isinstance(data, Data):
        data, variables, inputs, constants, scaling = data._snapshot()
        constant_conflicts = sorted(set(constants) & set(parameter_names))
        if constant_conflicts:
            raise ValueError(f"Constants {constant_conflicts} conflict with declared parameters")
        variable_declarations = _data_variable_declarations(variables, parameter_names, tuple(constants))
        declarations = _Declarations(variable_declarations, inputs, constants, scaling)
    if data is None:
        return None, declarations
    if not isinstance(data, PreparedData):
        raise TypeError("data must be Data or PreparedData. Use prepare_data with the observation dataframe")
    return data, declarations


def _fit_scaling(
    data: PreparedData, scaling: DataScaling | Literal["auto"] | None
) -> tuple[PreparedData, DataScaling | None]:
    """Apply the requested scaling once and return the observations in fitted units."""
    fitted: DataScaling | None
    if scaling == "auto":
        fitted = fit_data_scaling(data)
    elif isinstance(scaling, DataScaling):
        fitted = scaling
    else:
        fitted = data._scaling
    if fitted is None:
        return data, None
    if data._scaling is fitted:
        return data._align_to(fitted._layout), fitted
    return fitted.transform(data), fitted


def _outcome_scaling(data: PreparedData, fitted: DataScaling | None, has_outcome: bool) -> tuple[Scaling | None, bool]:
    """Expose the outcome transform to blocks.

    An outcome scaled by population across regions gets one factor per region. Any other outcome gets scalars.
    """
    population_outcome = fitted is not None and "outcome" in fitted._population_roles and bool(data.group_columns)
    if not has_outcome:
        return None, population_outcome
    transform = Scaling(offset=jnp.asarray(0.0), scale=jnp.asarray(1.0))
    if fitted is not None:
        transform = fitted.transformations.get("outcome", transform)
    factor_shape = (-1,) if population_outcome else ()
    outcome_scaling = Scaling(
        offset=transform.offset.reshape(factor_shape), scale=transform.scale.reshape(factor_shape)
    )
    return outcome_scaling, population_outcome


def _prepare_training(
    data: PreparedData,
    declarations: _Declarations,
    parameter_names: tuple[str, ...],
    coordinates: _Coordinates,
) -> tuple[_ModelData, _Training, _Coordinates]:
    """Build the model-owned data bundle with its training record and merged axis labels."""
    constants = declarations.constant_values
    declared = declarations.variables is not None
    data, fitted_scaling = _fit_scaling(data, declarations.scaling)
    values = data._to_jax()
    time_inputs = _time_input_names(data)
    time_origin = None
    if time_inputs:
        _, time_origin = _time_positions(data.time_values)
        values.update(_model_time_inputs(data, time_inputs, time_origin))
    # Copies keep the training references distinct from the current inputs,
    # so unchanged outputs can be labeled by which one they came from. The
    # namespace uses the names the blocks see, so a declared mapping renames
    # the training arrays exactly as it renames the current inputs.
    if declarations.variables is None:
        reference_values = {role: jnp.array(value, copy=True) for role, value in values.items()}
    else:
        reference_values = {
            name: jnp.array(values[source], copy=True)
            for name, source in declarations.variables.items()
            if source in values
        }
    reference = Reference(values=reference_values, n_periods=len(data.time_values))
    reserved = frozenset() if declared else frozenset({"n_periods", "outcome_scaling", "reference"})
    conflicts = reserved & set(parameter_names)
    if conflicts:
        raise ValueError(f"Parameter names {sorted(conflicts)} conflict with model-supplied inputs")
    outcome_scaling, population_outcome = _outcome_scaling(data, fitted_scaling, "outcome" in values)

    axis_coordinates = coordinates.copy()
    prepared_coords, _ = _prepared_coordinates(data)
    for axis, labels in prepared_coords.items():
        if axis in axis_coordinates and not _same_labels(axis_coordinates[axis], labels):
            raise ValueError(f"Coordinate {axis!r} conflicts with prepared data labels")
        axis_coordinates[axis] = labels
    reserved_parameters = () if declared else parameter_names
    input_values, input_dims, input_coords = _prepare_inputs(
        declarations.inputs, data, reserved_parameters, axis_coordinates, tuple(constants)
    )
    axis_coordinates.update(input_coords)
    conflicts = reserved & (set(input_values) | set(axis_coordinates) | set(constants))
    if conflicts:
        raise ValueError(f"Input or coordinate names {sorted(conflicts)} conflict with model-supplied inputs")
    values.update(input_values)

    model_data = _ModelData(
        values,
        reference=reference,
        outcome_scaling=outcome_scaling,
        outcome_group_scale=population_outcome,
        n_periods=len(data.time_values),
        reserved_names=reserved,
        constants=_FrozenMapping(constants),
    )
    if declarations.variables is not None:
        model_data = replace(model_data, variable_sources=_resolve_data_variables(declarations.variables, model_data))
    training = _Training(
        layout=data._layout(),
        time_column=data.time_column,
        frequency=data.frequency,
        time_values=tuple(data.time_values),
        media_time_values=tuple(data.media_time_values),
        time_inputs=time_inputs,
        time_origin=time_origin,
        scaling=fitted_scaling,
        input_dims=input_dims,
        input_coords=input_coords,
    )
    return model_data, training, axis_coordinates


def _bind_blocks(
    data: _ModelData | None,
    parameter_names: tuple[str, ...],
    *,
    log_density: LogDensity,
    generated_quantities: GeneratedQuantities | None,
    transformed_parameters: TransformedParameters | None,
    transformed_data: TransformedData | None,
) -> tuple[_Blocks, _ModelData | None]:
    """Resolve every block's inputs once and evaluate ``transformed_data`` on the training inputs."""
    if data is None:
        if transformed_parameters is not None:
            raise ValueError("transformed_parameters requires prepared data")
        _validate_log_density_signature(log_density, parameter_names)
        generate_parameter_names = None
        if generated_quantities is not None:
            generate_parameter_names = _validate_generate_signature(generated_quantities, parameter_names)
        raw = _RawBlocks(
            log_density=log_density,
            generate=generated_quantities,
            generate_parameter_names=generate_parameter_names,
        )
        return raw, None

    transformed_data_inputs: _InputBindings = ()
    if transformed_data is not None:
        transformed_data_inputs = _bind_inputs(
            transformed_data, (), data, name="transformed_data", allow_parameters=False
        )
        data = _compute_transformed_data(transformed_data, transformed_data_inputs, data, parameter_names)

    has_transformed = transformed_parameters is not None
    transform_inputs: _InputBindings = ()
    if transformed_parameters is not None:
        transform_inputs = _bind_inputs(transformed_parameters, parameter_names, data, name="transformed_parameters")
    upstream_parameters = tuple(name for name, source in transform_inputs if source == "parameter")
    density_inputs = _bind_inputs(
        log_density,
        parameter_names,
        data,
        name="log_density",
        require_all_parameters=True,
        upstream_parameters=upstream_parameters,
        has_transformed=has_transformed,
    )
    generation_inputs: _InputBindings = ()
    if generated_quantities is not None:
        generation_inputs = _bind_inputs(
            generated_quantities,
            parameter_names,
            data,
            name="generated_quantities",
            leading_key=True,
            has_transformed=has_transformed,
        )
    prepared = _PreparedBlocks(
        log_density=log_density,
        generate=generated_quantities,
        transformed_parameters=transformed_parameters,
        transformed_data=transformed_data,
        transformed_data_inputs=transformed_data_inputs,
        transform_inputs=transform_inputs,
        density_inputs=density_inputs,
        generation_inputs=generation_inputs,
        training=data,
    )
    return prepared, data


def _validate_result_dims(
    dimensions: Mapping[str, tuple[str, ...]],
    parameterizations: _Parameterizations,
    coordinates: _Coordinates,
) -> None:
    """Require explicit parameter axes to match declared shapes and known axis lengths."""
    declarations = dict(parameterizations)
    dimension_sizes = {axis: len(labels) for axis, labels in coordinates.items()}
    for name, axes in dimensions.items():
        if name not in declarations:
            raise ValueError(f"dims refers to undeclared parameter {name!r}")
        shape = declarations[name].shape
        if len(axes) != len(shape):
            raise ValueError(f"Dimensions for parameter {name!r} must match its constrained shape {shape}")
        for axis, size in zip(axes, shape, strict=True):
            if axis in dimension_sizes and dimension_sizes[axis] != size:
                raise ValueError(f"Dimension {axis!r} must have length {size} for parameter {name!r}")
            dimension_sizes[axis] = size


def _prepare_inputs(
    inputs: xr.Dataset | None,
    data: PreparedData,
    parameter_names: tuple[str, ...],
    coordinates: Mapping[str, NDArray[np.generic]],
    constant_names: tuple[str, ...] = (),
) -> tuple[dict[str, jax.Array], dict[str, tuple[str, ...]], _Coordinates]:
    """Validate fixed labeled inputs without aligning or rescaling their values."""
    if inputs is None:
        return {}, {}, {}
    if not isinstance(inputs, xr.Dataset):
        raise TypeError("inputs must be an xarray.Dataset with named numeric variables")

    dimensions = _dimensions(
        {_name(name): tuple(_name(axis) for axis in value.dims) for name, value in inputs.data_vars.items()}
    )
    if {name.value for name in _TimeInput} & inputs.sizes.keys():
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

    reserved = role_names | {name.value for name in _TimeInput} | set(parameter_names) | set(coordinates)
    reserved |= set(input_coords)
    reserved |= set(constant_names)
    values = {}
    for name in dimensions:
        _validate_name(name, label="input")
        if name in reserved:
            raise ValueError(f"Input {name!r} conflicts with a data role, parameter, or coordinate")
        values[name] = _input_array(name, inputs[name].values)
    return values, dimensions, input_coords


def _input_array(name: str, values: object) -> jax.Array:
    """Convert one auxiliary input to a finite JAX array at the current precision."""
    array = np.array(values, copy=True)
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
        raise ValueError(f"Input {name!r} is not finite at the current JAX precision. Rescale it or enable 64-bit mode")
    return jnp.asarray(array)


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


def _copy_reference(reference: Reference | None) -> Reference | None:
    """Give each bundle its own reference mapping so edits never reach the model."""
    if reference is None:
        return None
    return replace(reference, values=dict(reference.values))


def _prepare_parameterizations(parameters: Mapping[str, Parameterization]) -> _Parameterizations:
    """Validate parameter names and declarations.

    Sorting the pairs by name fixes their evaluation order.
    """
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


def _resolve_parameter_dimensions(
    parameterizations: _Parameterizations,
    dimensions: dict[str, tuple[str, ...]],
    coordinates: Mapping[str, NDArray[np.generic]],
) -> _Parameterizations:
    """Resolve named built-in shapes without changing reusable declarations."""
    resolved = []
    for name, parameter in parameterizations:
        if isinstance(parameter, _Dimensioned) and parameter.dims:
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


def _validate_value_names(
    values: Mapping[str, object],
    parameterizations: _Parameterizations,
    *,
    name: str,
) -> None:
    """Require a value mapping to name exactly the declared parameters."""
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


def _as_scalar(value: ArrayLike, *, name: str) -> jax.Array:
    """Require a block to return a real floating-point scalar."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must return an array-like floating-point scalar, got {type(value).__name__}") from exc
    if array.shape != ():
        raise ValueError(f"{name} must return a scalar, got shape {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must return a real floating-point value, got dtype {array.dtype}")
    return array
