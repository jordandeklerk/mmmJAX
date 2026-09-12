"""Prior and posterior sampling with labeled model results."""

import warnings
from collections.abc import Mapping
from dataclasses import replace
from importlib.metadata import version
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax._nuts import _sample_nuts
from mmmjax._results import _collect_results, _data_dimensions, _prepared_groups, _same_labels
from mmmjax.data import PreparedData
from mmmjax.media import _PreparedMedia
from mmmjax.model import Model, Prior, _component_parameter_inputs
from mmmjax.seasonality import _PreparedFourier

__all__ = ["generate_quantities", "sample", "sample_prior"]


def sample(
    model: Model,
    *,
    data: object = None,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    initial_values: Mapping[str, ArrayLike] | None = None,
    generate: bool = True,
) -> xr.DataTree:
    """Sample a model with NUTS and return labeled posterior results.

    Warmup tunes each chain's step size and, when long enough, its diagonal
    mass matrix. Chains run sequentially. Warmup draws are discarded, and generated
    quantities are evaluated for every retained draw when available.

    Parameters
    ----------
    model : Model
        Model defining parameter constraints, priors, and likelihood.
    data : object, optional
        Inputs for a model without prepared data. Prepared models use their
        stored observations and fitted scaling automatically.
    draws : int, default 1000
        Retained draws per chain, excluding warmup.
    warmup : int, default 1000
        Adaptation steps per chain.
    chains : int, default 4
        Number of independently adapted chains.
    seed : int, default 0
        Random seed for initialization, sampling, and generated quantities.
    target_accept : float, default 0.8
        Target acceptance probability during adaptation, between zero and one.
    max_tree_depth : int, default 10
        Maximum trajectory expansion depth for each NUTS step.
    initial_values : mapping of str to array_like, optional
        Complete constrained parameter values used to start every chain.
        Otherwise, each chain starts from a random unconstrained position.
    generate : bool, default True
        Evaluate the model's generated quantities when a callback is available.

    Returns
    -------
    xarray.DataTree
        Host-side results with chain and draw dimensions.

        - **posterior** contains constrained parameter draws.
        - **sample_stats** contains sampler diagnostics, including divergences
          and the unconstrained log density in ``lp``.
        - **posterior_predictive** and **log_likelihood** contain generated
          outputs selected by the model. Other outputs are stored in
          **generated_quantities**.
        - **observed_data** and **constant_data** contain prepared model inputs
          in their evaluated units, including any fitted scaling.

        Known component and data axes retain their labels, including unchanged
        generation inputs. Observation-shaped predictive and likelihood outputs
        inherit outcome labels. Use ``dims``, ``generated_dims``, and ``coords``
        on the model for custom axes. Inspect diagnostics before interpreting results.

    Examples
    --------
    With a constructed ``Model``, sample and inspect its labeled draws.

    .. code-block:: python

        from mmmjax import sample

        results = sample(model, seed=42)
        results["posterior"]
        results["sample_stats"]
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    for name, value in (("draws", draws), ("warmup", warmup), ("chains", chains), ("max_tree_depth", max_tree_depth)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(target_accept, bool) or not isinstance(target_accept, Real) or not 0 < target_accept < 1:
        raise ValueError("target_accept must be between zero and one")
    if not isinstance(generate, bool):
        raise TypeError("generate must be a bool")
    if model._data is not None and data is not None:
        raise ValueError("Prepared models use their stored data. Omit data when sampling")
    if not any(np.prod(parameter.position_shape) > 0 for parameter in model.parameters.values()):
        raise ValueError("Sampling requires at least one free parameter")

    inputs = model.data if model._data is not None else data
    initialization_key, sampling_key, generation_key, preview_key = jax.random.split(jax.random.key(int(seed)), 4)
    if initial_values is None:
        positions = jax.vmap(model.initialize_random)(jax.random.split(initialization_key, chains))
    else:
        initial = model.unconstrain(initial_values)
        restored = model.constrain(initial)
        for name, value in restored.items():
            supplied = np.asarray(initial_values[name])
            tolerance = 32 * np.finfo(value.dtype).eps
            if not np.isfinite(supplied).all() or not np.allclose(supplied, value, rtol=tolerance, atol=0):
                raise ValueError(f"initial_values for {name!r} must satisfy its parameter constraints")
        positions = {name: jnp.broadcast_to(value, (chains, *value.shape)) for name, value in initial.items()}

    def logdensity(position: dict[str, jax.Array]) -> jax.Array:
        return model.log_density(position, inputs)

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(logdensity)))(positions)
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves((positions, values, gradients))):
        raise ValueError(
            "Initial positions must have finite log density and gradients. "
            "Check the model or supply valid initial_values"
        )

    prepared = _result_data(model)
    dimensions, coordinates = _parameter_metadata(model)

    if set(coordinates) & {"chain", "draw"}:
        raise ValueError("Sampling assigns chain and draw coordinates. Supply only model axes in coords")

    initial_parameters = model.constrain({name: value[0] for name, value in positions.items()})
    outputs: dict[str, jax.Array] = {}
    output_dimensions: dict[str, tuple[str, ...]] = {}

    if generate and model._generate:
        outputs, arguments = model._generate_with_inputs(preview_key, initial_parameters, inputs)
        output_dimensions = _output_dimensions(model, outputs, arguments, dimensions, prepared)

    def collect(
        posterior: dict[str, jax.Array],
        generated: dict[str, jax.Array],
        stats: dict[str, jax.Array] | None = None,
    ) -> xr.DataTree:
        return _collect_results(
            posterior,
            data=prepared,
            posterior_predictive={name: value for name, value in generated.items() if name in model._predictive_names},
            log_likelihood={name: value for name, value in generated.items() if name in model._likelihood_names},
            generated_quantities={
                name: value
                for name, value in generated.items()
                if name not in (*model._predictive_names, *model._likelihood_names)
            },
            sample_stats=stats,
            dims=dimensions,
            generated_dims=output_dimensions,
            coords=coordinates,
        )

    # Validate output axes and labels before adapting any chains.
    collect(
        {name: value[None, None] for name, value in initial_parameters.items()},
        {name: value[None, None] for name, value in outputs.items()},
    )
    positions, stats = _sample_nuts(
        logdensity,
        positions,
        jax.random.split(sampling_key, chains),
        draws=draws,
        warmup=warmup,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
    )
    if not np.isfinite(np.asarray(stats["lp"])).all():
        raise RuntimeError("Sampling produced nonfinite log densities. Check the model and initial_values")

    posterior = jax.jit(jax.vmap(jax.vmap(model.constrain)))(positions)
    generated = {}

    if generate and model._generate is not None:
        keys = jax.random.split(generation_key, (chains, draws))
        generated = jax.jit(jax.vmap(jax.vmap(lambda key, parameters: model.generate(key, parameters, inputs))))(
            keys, posterior
        )

    results = collect(posterior, generated, stats)
    results.attrs.update(
        inference_library="blackjax",
        inference_library_version=version("blackjax"),
        inference_method="nuts",
        warmup_steps=warmup,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        seed=int(seed),
        data_scale="model" if model.scaling is not None else "original",
    )
    divergences = int(np.asarray(stats["diverging"]).sum())
    if divergences:
        warnings.warn(
            f"Sampling encountered {divergences} divergent transitions after warmup. "
            "Inspect the model parameterization and consider a higher target_accept",
            RuntimeWarning,
            stacklevel=2,
        )
    limited = int(np.asarray(stats["reached_max_treedepth"]).sum())
    if limited:
        warnings.warn(
            f"Sampling reached max_tree_depth on {limited} draws. "
            "Inspect mixing and consider increasing max_tree_depth",
            RuntimeWarning,
            stacklevel=2,
        )
    return results


def sample_prior(
    model: Model,
    prior: Prior | None = None,
    *,
    data: object = None,
    draws: int = 500,
    seed: int = 0,
    generate: bool = True,
) -> xr.DataTree:
    """Draw explicit priors and inspect their implied outcomes before fitting.

    Use the model's prior-draw function and reuse its transformed parameters
    and generation callback. No log density or posterior sampler is evaluated.
    Keep the sampling distributions consistent with the priors in ``log_density``.

    Parameters
    ----------
    model : Model
        Model supplying parameter declarations and generated quantities.
    prior : callable, optional
        Override the prior-draw function attached to ``Model`` for this call.
        The function ``prior(key)`` must return one constrained draw per
        declared parameter. Required if the model has no prior-draw function.
    data : object, optional
        Inputs for a model without prepared data. Prepared models use their
        stored observations and fitted scaling automatically.
    draws : int, default 500
        Number of independent prior draws.
    seed : int, default 0
        Random seed for parameters and generated quantities.
    generate : bool, default True
        Evaluate generated quantities when the model has a generation callback.

    Returns
    -------
    xarray.DataTree
        Labeled results with one chain axis and ``draws`` draws.

        - **prior** contains constrained parameter draws.
        - **prior_predictive** contains outputs selected by the model's
          ``predictive`` argument.
        - **prior_generated_quantities** contains other generated outputs.
        - **observed_data** and **constant_data** contain prepared model inputs
          in their evaluated units, including fitted scaling.

        Outputs selected as log likelihoods are omitted. The chain axis is for
        result compatibility, not an MCMC chain. Without generation, only prior
        draws and available model inputs are returned.

    Examples
    --------
    Define a Normal observation model with known noise scale. Write the prior
    and likelihood explicitly, sharing the prior mean and scale with the
    prior-draw function.

    .. ipython::

        In [1]: from mmmjax import Model, Real, normal, normal_rng
           ...: from mmmjax import sample_prior
           ...: prior_mean = 0.0
           ...: prior_scale = 2.0
           ...: observation_scale = 1.0
           ...: parameters = {"location": Real()}

        In [2]: def log_density(data, location):
           ...:     target = normal(
           ...:         location, location=prior_mean, scale=prior_scale
           ...:     )
           ...:     target += normal(
           ...:         data, location=location, scale=observation_scale
           ...:     )
           ...:     return target

        In [3]: def prior(key):
           ...:     location = normal_rng(
           ...:         key, location=prior_mean, scale=prior_scale
           ...:     )
           ...:     return {"location": location}

        In [4]: def generate(key, data, location):
           ...:     outcome = normal_rng(
           ...:         key, location=location, scale=observation_scale
           ...:     )
           ...:     return {"outcome": outcome}

        In [5]: model = Model(
           ...:     parameters=parameters,
           ...:     log_density=log_density,
           ...:     prior=prior,
           ...:     generate=generate,
           ...:     predictive=("outcome",),
           ...: )

        In [6]: results = sample_prior(model, draws=100, seed=42)
           ...: results
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    if prior is None:
        prior = model._prior
    if prior is None:
        raise ValueError("Provide a prior-draw function on Model or pass prior to sample_prior")
    if not callable(prior):
        raise TypeError("prior must be a JAX-compatible function accepting a random key")
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(generate, bool):
        raise TypeError("generate must be a bool")
    if model._data is not None and data is not None:
        raise ValueError("Prepared models use their stored data. Omit data when sampling")
    if not model.parameters:
        raise ValueError("Prior sampling requires at least one model parameter")

    dimensions, coordinates = _parameter_metadata(model)
    if set(coordinates) & {"chain", "draw"}:
        raise ValueError("Sampling assigns chain and draw coordinates. Supply only model axes in coords")
    inputs = model.data if model._data is not None else data
    prepared = _result_data(model)
    prior_key, generation_key, preview_key = jax.random.split(jax.random.key(int(seed)), 3)
    parameters = _prior_draws(model, prior, prior_key, int(draws))
    generated: dict[str, jax.Array] = {}
    output_dimensions: dict[str, tuple[str, ...]] = {}

    if generate and model._generate is not None:
        initial = {name: value[0] for name, value in parameters.items()}
        outputs, arguments = model._generate_with_inputs(preview_key, initial, inputs)
        output_dimensions = _output_dimensions(model, outputs, arguments, dimensions, prepared)
        keys = jax.random.split(generation_key, draws)
        generated = jax.jit(jax.vmap(lambda key, values: model.generate(key, values, inputs)))(keys, parameters)

    results = _collect_results(
        {name: value[None] for name, value in parameters.items()},
        data=prepared,
        posterior_predictive={
            name: value[None] for name, value in generated.items() if name in model._predictive_names
        },
        generated_quantities={
            name: value[None]
            for name, value in generated.items()
            if name not in (*model._predictive_names, *model._likelihood_names)
        },
        dims=dimensions,
        generated_dims=output_dimensions,
        coords=coordinates,
        sample_group="prior",
    )
    results.attrs.update(
        sampling_method="prior",
        seed=int(seed),
        data_scale="model" if model.scaling is not None else "original",
    )
    return results


def _prior_draws(
    model: Model,
    prior: Prior,
    key: jax.Array,
    draws: int,
) -> dict[str, jax.Array]:
    """Draw and validate constrained parameters without evaluating a density."""

    def draw_parameters(draw_key: jax.Array) -> dict[str, jax.Array]:
        values = prior(draw_key)
        if not isinstance(values, Mapping):
            raise TypeError("prior must return a mapping of parameter names to constrained draws")
        if set(values) != set(model.parameters):
            raise ValueError("Prior parameter names must match all model declarations, including component parameters")

        parameters = {}
        for name, declaration in model.parameters.items():
            value = jnp.asarray(values[name])
            if value.shape != declaration.shape:
                raise ValueError(f"Prior draw shape for {name!r} must match its declared shape {declaration.shape}")
            if not (jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(value.dtype, jnp.integer)):
                raise TypeError(f"Prior draws for {name!r} must be real numbers")
            parameters[name] = value.astype(declaration.dtype)
        return parameters

    def valid_support(parameters: dict[str, jax.Array]) -> dict[str, jax.Array]:
        valid = {}
        for name, declaration in model.parameters.items():
            value = parameters[name]
            position = declaration.unconstrain(value)
            restored = declaration.constrain(position)
            tolerance = 32 * jnp.finfo(restored.dtype).eps
            valid[name] = jnp.all(jnp.isfinite(position)) & jnp.all(
                jnp.isclose(restored, value, rtol=tolerance, atol=0)
            )
        return valid

    parameters: dict[str, jax.Array] = jax.jit(jax.vmap(draw_parameters))(jax.random.split(key, draws))
    validity = jax.jit(jax.vmap(valid_support))(parameters)
    for name, values in parameters.items():
        if not np.isfinite(np.asarray(values)).all():
            raise ValueError(f"Prior draws for {name!r} must be finite real numbers")
        if not np.asarray(validity[name]).all():
            raise ValueError(f"Prior draws for {name!r} must satisfy its parameter constraints")
    return parameters


def generate_quantities(
    model: Model,
    results: xr.DataTree,
    *,
    new_data: object = None,
    seed: int = 0,
) -> xr.DataTree:
    """Evaluate generated quantities from existing posterior draws without refitting.

    The model's components, transformed parameters, and generation callback
    run for every draw. Priors, likelihood evaluation in ``log_density``, and
    sampling are not rerun. Scenario calculations remain defined by the model.

    Parameters
    ----------
    model : Model
        Model with a generation callback and the fitted parameter declarations.
    results : xarray.DataTree
        Results containing constrained posterior draws with the model's parameter
        names, shapes, and axis labels. Draws may be sliced or thinned.
    new_data : dataframe-like, PreparedData, or object, optional
        Scenario observations using the original source columns. Dataframes
        reuse the model's column selections, labels, and fitted scaling.
        Omit to evaluate stored observations. Earlier exposures are not added
        automatically. Include them through ``prepare_data(media_history=...)``.
        Other models receive this input directly. Omit outcomes only when no
        evaluated callback needs them.
    seed : int, default 0
        Random seed for generated quantities, with an independent key per draw.

    Returns
    -------
    xarray.DataTree
        A new result tree. The original results and model are unchanged.

        - **posterior** contains the reused draws and sample labels.
        - **posterior_predictive**, **log_likelihood**, and
          **generated_quantities** contain newly evaluated outputs, classified
          by the model's result settings.
        - **observed_data** and **constant_data** contain the evaluated inputs
          in model units. Original sampler diagnostics are not copied.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    if model._generate is None:
        raise ValueError("The model must define a generation callback")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    dimensions, coordinates = _parameter_metadata(model)
    posterior, coordinates = _posterior_draws(model, results, dimensions, coordinates)
    prepared = _result_data(model)
    inputs = model.data if model._data is not None else new_data
    if model._data is not None and new_data is not None:
        inputs, aligned = model._prepare_data(new_data)
        prepared = replace(aligned, arrays={name: np.array(value, copy=True) for name, value in inputs.values.items()})

    generation_key, preview_key = jax.random.split(jax.random.key(int(seed)))
    initial = {name: value[0, 0] for name, value in posterior.items()}
    outputs, arguments = model._generate_with_inputs(preview_key, initial, inputs)
    output_dimensions = _output_dimensions(model, outputs, arguments, dimensions, prepared)
    chains, draws = next(iter(posterior.values())).shape[:2]
    keys = jax.random.split(generation_key, (chains, draws))
    generated = jax.jit(jax.vmap(jax.vmap(lambda key, parameters: model.generate(key, parameters, inputs))))(
        keys, posterior
    )
    evaluated = _collect_results(
        posterior,
        data=prepared,
        posterior_predictive={name: value for name, value in generated.items() if name in model._predictive_names},
        log_likelihood={name: value for name, value in generated.items() if name in model._likelihood_names},
        generated_quantities={
            name: value
            for name, value in generated.items()
            if name not in (*model._predictive_names, *model._likelihood_names)
        },
        dims=dimensions,
        generated_dims=output_dimensions,
        coords=coordinates,
    )
    evaluated.attrs.update(generation_seed=int(seed), data_scale="model" if model.scaling is not None else "original")
    return evaluated


def _posterior_draws(
    model: Model,
    results: xr.DataTree,
    dimensions: dict[str, tuple[str, ...]],
    coordinates: dict[str, NDArray[np.generic]],
) -> tuple[dict[str, jax.Array], dict[str, NDArray[np.generic]]]:
    """Validate labeled constrained draws before passing them to model callbacks."""
    if not isinstance(results, xr.DataTree):
        raise TypeError("results must be an xarray.DataTree containing posterior draws")
    if "posterior" not in results.children:
        raise ValueError("results must contain a posterior group")
    dataset = results["posterior"].to_dataset()
    if not dataset.data_vars or set(dataset.data_vars) != set(model.parameters):
        raise ValueError("Posterior parameter names must match the model declarations")
    for axis in ("chain", "draw"):
        if dataset.sizes.get(axis, 0) == 0:
            raise ValueError(f"posterior must contain at least one {axis}")
        if axis in coordinates:
            raise ValueError("Posterior draws supply chain and draw coordinates. Supply only model axes in coords")

    expected_coordinates = coordinates.copy()
    training = _result_data(model)
    if training is not None:
        _, _, training_coordinates, auxiliary = _prepared_groups(training)
        for axis, labels in training_coordinates.items():
            if axis in expected_coordinates and not _same_labels(expected_coordinates[axis], labels):
                raise ValueError(f"Coordinate {axis!r} conflicts with prepared data labels")
            expected_coordinates[axis] = labels
        if any("group" in axes for axes in dimensions.values()):
            for name, (axis, labels) in auxiliary.items():
                if (
                    name not in dataset.coords
                    or dataset.coords[name].dims != (axis,)
                    or not _same_labels(dataset.coords[name].values, labels)
                ):
                    raise ValueError(f"Posterior coordinate {name!r} must match the model group labels and ordering")

    posterior = {}
    for name, parameter in model.parameters.items():
        value = dataset[name]
        axes = ("chain", "draw", *dimensions[name])
        if len(value.dims) != len(axes) or set(value.dims) != set(axes):
            raise ValueError(f"Posterior dimensions for {name!r} must match the model axes {axes}")
        value = value.transpose(*axes)
        if value.shape[2:] != parameter.shape:
            raise ValueError(f"Posterior shape for {name!r} must match its constrained parameter shape")
        for axis, size in zip(dimensions[name], parameter.shape, strict=True):
            labels = expected_coordinates.get(axis, np.arange(size))
            if axis not in value.coords or not _same_labels(value.coords[axis].values, labels):
                raise ValueError(f"Posterior coordinate {axis!r} must match the model labels and ordering")
            coordinates[axis] = labels.copy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Posterior draws for {name!r} must be finite real numbers")
        if jax.dtypes.canonicalize_dtype(array.dtype) != array.dtype:
            raise ValueError("Enable JAX 64-bit mode to evaluate these posterior draws without losing precision")
        posterior[name] = jnp.asarray(array)

    for axis in ("chain", "draw"):
        coordinates[axis] = np.array(
            dataset.coords[axis].values if axis in dataset.coords else np.arange(dataset.sizes[axis])
        )
    return posterior, coordinates


def _parameter_metadata(model: Model) -> tuple[dict[str, tuple[str, ...]], dict[str, NDArray[np.generic]]]:
    """Label known component axes without guessing the meaning of custom shapes."""
    dimensions: dict[str, tuple[str, ...]] = {}
    coordinates = {name: labels.copy() for name, labels in model._result_coords.items()}
    if model._data is not None:
        for component in model._data.components:
            if isinstance(component, _PreparedMedia):
                for name, parameter in component.parameters.items():
                    dimensions[name] = ("group", "channel") if len(parameter.shape) == 2 else ("channel",)
            elif isinstance(component, _PreparedFourier):
                name = component.specification.name
                axis = f"{name}_mode"
                dimensions[name] = (axis, "group") if component.specification.group_specific_coefficients else (axis,)
                order = component.specification.order
                coordinates.setdefault(
                    axis, np.array([f"{kind}_{index}" for kind in ("sin", "cos") for index in range(1, order + 1)])
                )
    dimensions.update(model._result_dims)
    for name, parameter in model.parameters.items():
        dimensions.setdefault(name, tuple(f"{name}_dim_{index}" for index in range(len(parameter.shape))))
    return dimensions, coordinates


def _output_dimensions(
    model: Model,
    outputs: dict[str, jax.Array],
    arguments: dict[str, ArrayLike],
    parameter_dimensions: dict[str, tuple[str, ...]],
    prepared: PreparedData | None,
) -> dict[str, tuple[str, ...]]:
    """Label known generation inputs and declared observations without shape guessing."""
    requested = set(model._generated_dims) | set(model._predictive_names) | set(model._likelihood_names)
    missing = requested - outputs.keys()
    if missing:
        raise ValueError(f"Generated result metadata refers to missing outputs {sorted(missing)}")
    input_dimensions: dict[str, tuple[str, ...]] = {}
    observation_axes: tuple[str, ...] = ()
    outcome_shape: tuple[int, ...] | None = None
    if prepared is not None:
        role_dimensions = _data_dimensions(prepared)
        observation_axes = role_dimensions["outcome"]
        outcome_shape = (len(prepared.time_values),)
        if prepared.group_columns:
            outcome_shape += (len(prepared.group_values),)
        assert model._data is not None
        effect_dimensions = {
            component.specification.name: (
                (*observation_axes, "channel") if isinstance(component, _PreparedMedia) else observation_axes
            )
            for component in model._data.components
        }
        aliases = _component_parameter_inputs(model._data.components)
        for name, source in model._generation_inputs:
            if source == "data":
                input_dimensions[name] = role_dimensions[name]
            elif source == "effect":
                input_dimensions[name] = effect_dimensions[name]
            elif source == "media_total":
                input_dimensions[name] = observation_axes
            elif source == "component_parameter":
                input_dimensions[name] = parameter_dimensions[aliases[name]]
            elif source == "parameter":
                input_dimensions[name] = parameter_dimensions[name]
    else:
        input_dimensions.update(parameter_dimensions)

    observation_names = set(model._predictive_names) | set(model._likelihood_names)
    dimensions = {}
    for name, value in outputs.items():
        if name in model._generated_dims:
            dimensions[name] = model._generated_dims[name]
            continue
        if name in observation_names and value.shape == outcome_shape:
            dimensions[name] = observation_axes
            continue
        # Equal values or equal shapes do not establish a shared axis or ordering.
        inherited = {axes for argument, axes in input_dimensions.items() if value is arguments.get(argument)}
        if len(inherited) == 1:
            dimensions[name] = inherited.pop()
        else:
            dimensions[name] = tuple(f"{name}_dim_{index}" for index in range(value.ndim))
    return dimensions


def _result_data(model: Model) -> PreparedData | None:
    """Snapshot the exact evaluated inputs with their original observation labels."""
    if model._data is None or model._layout is None:
        return None
    layout = model._layout
    return PreparedData(
        arrays={name: np.array(value, copy=True) for name, value in model._data.values.items()},
        time_column=model._time_column or "time",
        time_values=model._time_values,
        media_time_values=model._media_time_values,
        group_columns=layout.group_columns,
        group_values=layout.group_values,
        columns=layout.columns.copy(),
        channels=layout.channels,
        organic_channels=layout.organic_channels,
        rf_channels=layout.rf_channels,
        organic_rf_channels=layout.organic_rf_channels,
        frequency=model._frequency,
    )
