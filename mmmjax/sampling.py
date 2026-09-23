"""Prior and posterior sampling with labeled model results."""

import warnings
from collections.abc import Callable, Hashable, Mapping
from dataclasses import replace
from importlib.metadata import version
from numbers import Integral, Real
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax._binding import _metadata_source
from mmmjax._nuts import _NUTSContinuation, _sample_nuts
from mmmjax._results import _collect_results, _coordinates, _prepared_groups, _same_labels
from mmmjax.data import PreparedData, Reference, _data_dimensions, _reference_dimensions
from mmmjax.model import (
    Model,
    PriorSampler,
    _OutputGroup,
    _OutputKey,
    _ResultGroup,
    _validate_value_names,
)
from mmmjax.priors import Prior, _validate_prior_sampler
from mmmjax.scaling import Scaling

__all__ = ["continue_sampling", "generate_quantities", "sample", "sample_prior"]


def sample(
    model: Model,
    *,
    data: object = None,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    chain_method: str = "sequential",
    seed: int = 0,
    target_accept: float = 0.8,
    mass_matrix: str = "diagonal",
    max_tree_depth: int = 10,
    initial_values: Mapping[str, ArrayLike] | None = None,
    generate: bool = True,
    chunk_size: int = 100,
    batch_size: int = 64,
    progress: bool = True,
) -> xr.DataTree:
    """Sample a model with NUTS and return labeled posterior results.

    Warmup tunes each chain's step size and, when long enough, its selected
    mass matrix. Warmup draws are discarded, and generated quantities are
    evaluated for every retained draw when available. The results carry the
    final sampler state, so :func:`continue_sampling` can add draws later
    without repeating warmup, including after saving the results to disk.

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
    chain_method : {"sequential", "vectorized", "parallel"}, default "sequential"
        Run chains one at a time, together on one device, or one per local
        JAX device. Parallel sampling requires at least ``chains`` devices.
        Vectorized chains use more memory and can wait for the longest
        trajectory, so they are not always faster.
    seed : int, default 0
        Random seed for initialization, sampling, and generated quantities.
    target_accept : float, default 0.8
        Target acceptance probability during adaptation, between zero and one.
    mass_matrix : {"diagonal", "dense"}, default "diagonal"
        Adapt individual unconstrained parameter scales, or also their
        correlations with a full dense matrix. Dense adaptation uses more
        computation, with matrix storage growing quadratically in the
        number of unconstrained parameter values.
    max_tree_depth : int, default 10
        Maximum trajectory expansion depth for each NUTS step.
    initial_values : mapping of str to array_like, optional
        Complete constrained parameter values used to start every chain.
        Otherwise, each chain starts from a random unconstrained position.
    generate : bool, default True
        Evaluate mapped log-prior terms and outputs from the
        ``generated_quantities`` callback.
    chunk_size : int, default 100
        Maximum retained draws per chain in a sampling chunk before transfer
        to host memory. Smaller chunks reduce device memory use without
        restarting warmup or thinning draws.
    batch_size : int, default 64
        Maximum draws evaluated together across chains when converting
        parameters and generating quantities. Smaller batches reduce working
        memory. This does not change NUTS or the number of retained draws.
    progress : bool, default True
        Show warmup and sampling progress bars. Sampling bars restart for each
        chunk. Parallel counters reflect individual devices, not completion
        of all chains.

    Returns
    -------
    xarray.DataTree
        Results with chain and draw dimensions that must fit in host memory.

        - **posterior** contains constrained parameter draws.
        - **sample_stats** contains diagnostics including divergences and
          unconstrained log density ``lp``.
        - **posterior_predictive** contains outputs returned under ``predictive``.
        - **log_likelihood** and **log_prior** contain outputs returned under
          the corresponding keys by ``generated_quantities``.
        - **generated_quantities** contains other generated outputs.
        - **observed_data** and **constant_data** contain inputs in evaluated
          units, including fitted scaling. Auxiliary ``Data`` uses **constant_data**.
        - **sampling_state** stores chain positions, tuning, and random streams
          for continued sampling.

        Declared axes retain labels. Predictive and likelihood outputs matching
        observations inherit outcome labels. Custom axes use model ``dims``,
        ``generated_dims``, and ``coords``. Inspect diagnostics before interpretation.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    _validate_batch_size(batch_size)
    for name, value in (
        ("draws", draws),
        ("warmup", warmup),
        ("chains", chains),
        ("max_tree_depth", max_tree_depth),
        ("chunk_size", chunk_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(chain_method, str) or chain_method not in ("sequential", "vectorized", "parallel"):
        raise ValueError("chain_method must be 'sequential', 'vectorized', or 'parallel'")
    if not isinstance(mass_matrix, str) or mass_matrix not in ("diagonal", "dense"):
        raise ValueError("mass_matrix must be 'diagonal' or 'dense'")
    if chain_method == "parallel" and chains > jax.local_device_count():
        raise ValueError(
            f"Parallel sampling requested {chains} chains but only {jax.local_device_count()} local JAX devices "
            "are available. Use fewer chains or choose 'sequential' or 'vectorized'"
        )

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(target_accept, bool) or not isinstance(target_accept, Real) or not 0 < target_accept < 1:
        raise ValueError("target_accept must be between zero and one")
    if not isinstance(generate, bool):
        raise TypeError("generate must be a bool")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a bool")
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
    outputs, output_dimensions = _generation_layout(
        model, inputs, initial_parameters, preview_key, dimensions, prepared, generate=generate
    )
    # Validate output axes and labels before adapting any chains.
    _collect_sampling_results(
        model,
        {name: value[None, None] for name, value in initial_parameters.items()},
        {name: value[None, None] for name, value in outputs.items()},
        None,
        output_dimensions,
    )
    (unconstrained, stats), continuation = _sample_nuts(
        logdensity,
        positions,
        jax.random.split(sampling_key, chains),
        draws=draws,
        warmup=warmup,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        chain_method=chain_method,
        mass_matrix=mass_matrix,
        chunk_size=chunk_size,
        progress=progress,
    )
    if not np.isfinite(np.asarray(stats["lp"])).all():
        raise RuntimeError("Sampling produced nonfinite log densities. Check the model and initial_values")

    results = _run_results(
        model,
        inputs,
        unconstrained,
        stats,
        chains=chains,
        draws=draws,
        start=0,
        generation_key=generation_key,
        generate=generate,
        output_dimensions=output_dimensions,
        batch_size=batch_size,
    )
    results.attrs.update(
        inference_library="blackjax",
        inference_library_version=version("blackjax"),
        inference_method="nuts",
        chain_method=chain_method,
        mass_matrix=mass_matrix,
        warmup_steps=warmup,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        seed=int(seed),
        data_scale="model" if model.scaling is not None else "original",
    )
    _warn_sampling(stats)
    return _with_state(results, continuation, generation_key, generate)


def continue_sampling(
    model: Model,
    results: xr.DataTree,
    *,
    data: object = None,
    draws: int = 1000,
    chunk_size: int = 100,
    batch_size: int = 64,
    progress: bool = True,
) -> xr.DataTree:
    """Add posterior draws to sampled results without repeating warmup.

    Resumes every chain from the final position, tuning, and random stream
    stored in the ``sampling_state`` group, so the combined draws match a
    single longer run. The supplied results are not modified. For a
    different model or dataset, start a new run with :func:`sample`.

    Parameters
    ----------
    model : Model
        The model that produced ``results``. Its parameter names and shapes
        must match the stored draws.
    results : xarray.DataTree
        Results returned by :func:`sample` or this function, including the
        ``sampling_state`` group. Results restored from disk are accepted.
    data : object, optional
        Inputs for a model without prepared data, matching the original run.
        Prepared models use their stored observations automatically.
    draws : int, default 1000
        Additional retained draws per chain.
    chunk_size : int, default 100
        Maximum additional draws per chain in each device sampling chunk.
    batch_size : int, default 64
        Maximum draws evaluated together when generating results.
    progress : bool, default True
        Show sampling progress bars for the additional draws.

    Returns
    -------
    xarray.DataTree
        Original and additional draws in every sampled group, with continuous
        draw numbering, unchanged observation labels, and an advanced
        ``sampling_state`` for further continuation.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    _validate_batch_size(batch_size)
    for name, value in (("draws", draws), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(progress, bool):
        raise TypeError("progress must be a bool")
    if model._data is not None and data is not None:
        raise ValueError("Prepared models use their stored data. Omit data when continuing")
    continuation, generation_key, generate = _restore_continuation(results, model)

    settings = results.attrs
    missing = [name for name in ("chain_method", "warmup_steps", "target_accept", "max_tree_depth", "mass_matrix")]
    missing = [name for name in missing if name not in settings]
    if missing:
        raise ValueError(
            f"results are missing the sampling settings {missing}. Continue from results returned by sample"
        )
    chain_method = str(settings["chain_method"])
    chains = int(continuation.step_size.shape[0])
    if chain_method == "parallel" and chains > jax.local_device_count():
        raise ValueError("Continuing parallel sampling requires the original number of local JAX devices")
    completed = continuation.completed_draws
    stop = completed + int(draws)
    if stop > np.iinfo(np.uint32).max:
        raise ValueError("The continued run exceeds the supported number of draws")

    inputs = model.data if model._data is not None else data

    def logdensity(position: dict[str, jax.Array]) -> jax.Array:
        return model.log_density(position, inputs)

    prepared = _result_data(model)
    dimensions, _ = _parameter_metadata(model)
    initial_parameters = model.constrain({name: value[0] for name, value in continuation.positions.items()})
    _, output_dimensions = _generation_layout(
        model, inputs, initial_parameters, generation_key, dimensions, prepared, generate=generate
    )
    (unconstrained, stats), resumed = _sample_nuts(
        logdensity,
        continuation.positions,
        continuation.sampling_keys,
        draws=int(draws),
        warmup=int(settings["warmup_steps"]),
        target_accept=float(settings["target_accept"]),
        max_tree_depth=int(settings["max_tree_depth"]),
        chain_method=chain_method,
        mass_matrix=str(settings["mass_matrix"]),
        chunk_size=int(chunk_size),
        progress=progress,
        continuation=continuation,
    )
    if not np.isfinite(np.asarray(stats["lp"])).all():
        raise RuntimeError("Sampling produced nonfinite log densities. The supplied results are unchanged")

    additional = _run_results(
        model,
        inputs,
        unconstrained,
        stats,
        chains=chains,
        draws=int(draws),
        start=completed,
        generation_key=generation_key,
        generate=generate,
        output_dimensions=output_dimensions,
        batch_size=batch_size,
    )
    combined = _concatenate_draws(results, additional, start=completed, stop=stop)
    _warn_sampling(stats)
    return _with_state(combined, resumed, generation_key, generate)


def _generation_layout(
    model: Model,
    inputs: object,
    parameters: dict[str, jax.Array],
    key: jax.Array,
    dimensions: dict[str, tuple[str, ...]],
    prepared: PreparedData | None,
    *,
    generate: bool,
) -> tuple[dict[_OutputKey, jax.Array], dict[str, tuple[str, ...]]]:
    """Preview one generated draw to fix output names and axis labels before sampling."""
    if not (generate and model._has_generated_quantities):
        return {}, {}
    outputs, arguments = model._generate_with_inputs(key, parameters, inputs)
    return outputs, _output_dimensions(model, outputs, arguments, dimensions, prepared)


def _run_results(
    model: Model,
    inputs: object,
    unconstrained: Mapping[str, NDArray[np.generic]],
    stats: Mapping[str, NDArray[np.generic]],
    *,
    chains: int,
    draws: int,
    start: int,
    generation_key: jax.Array,
    generate: bool,
    output_dimensions: dict[str, tuple[str, ...]],
    batch_size: int,
) -> xr.DataTree:
    """Constrain new draws, generate their quantities, and label them as result groups."""
    posterior = _evaluate_draws(model.constrain, unconstrained, sample_shape=(chains, draws), batch_size=batch_size)
    generated: dict[_OutputKey, NDArray[np.generic]] = {}
    if generate and model._has_generated_quantities:
        keys = _generation_keys(generation_key, chains, start, start + draws)
        generated = _evaluate_draws(
            lambda key, parameters: model._generate_with_inputs(key, parameters, inputs)[0],
            keys,
            posterior,
            sample_shape=(chains, draws),
            batch_size=batch_size,
        )
    return _collect_sampling_results(model, posterior, generated, stats, output_dimensions)


def _concatenate_draws(previous: xr.DataTree, additional: xr.DataTree, *, start: int, stop: int) -> xr.DataTree:
    """Append new draws to every sampled group without sharing memory with the inputs."""
    combined = {}
    for name, original in previous.children.items():
        if name == "sampling_state":
            continue
        earlier = original.to_dataset()
        current = additional[name].to_dataset()
        if "draw" in earlier.dims:
            current = current.assign_coords(draw=np.arange(start, stop))
            joined = xr.concat(
                (earlier, current),
                dim="draw",
                data_vars="minimal",
                coords="minimal",
                compat="equals",
                join="exact",
                combine_attrs="identical",
            )
            # Concatenation can share unchanged labels with the previous results.
            combined[name] = joined.assign_coords(
                {axis: coord.copy(deep=True) for axis, coord in joined.coords.items()}
            )
        else:
            combined[name] = earlier.copy(deep=True)
    results = xr.DataTree.from_dict(combined, name=previous.name)
    results.attrs = dict(previous.attrs)
    return results


def _with_state(
    results: xr.DataTree, continuation: _NUTSContinuation, generation_key: jax.Array, generate: bool
) -> xr.DataTree:
    """Store the resumable sampler state beside the results as writable numpy arrays."""
    groups = {name: node.to_dataset() for name, node in results.children.items() if name != "sampling_state"}
    mass_matrix = np.array(continuation.inverse_mass_matrix, copy=True)
    # netCDF stores dimensions beside child groups. Reusing the position group's name makes results unwritable.
    mass_dims = ("chain", "unconstrained") if mass_matrix.ndim == 2 else ("chain", "unconstrained", "unconstrained_")
    groups["sampling_state"] = xr.Dataset(
        {
            "logdensity": ("chain", np.array(continuation.logdensity, copy=True)),
            "step_size": ("chain", np.array(continuation.step_size, copy=True)),
            "inverse_mass_matrix": (mass_dims, mass_matrix),
            "sampling_key": (
                ("chain", "key_data"),
                np.array(jax.random.key_data(continuation.sampling_keys), copy=True),
            ),
            "generation_key": ("key_data", np.array(jax.random.key_data(generation_key), copy=True)),
        },
        attrs={
            "completed_draws": int(continuation.completed_draws),
            "generate": int(generate),
            "precision": "float64" if jax.config.values["jax_enable_x64"] else "float32",
        },
    )
    groups["sampling_state/position"] = _position_dataset(continuation.positions)
    groups["sampling_state/gradient"] = _position_dataset(continuation.gradients)
    tree = xr.DataTree.from_dict(groups, name=results.name)
    tree.attrs = dict(results.attrs)
    return tree


def _position_dataset(values: Mapping[str, jax.Array]) -> xr.Dataset:
    """Label per-chain unconstrained arrays with parameter-specific axis names."""
    variables = {}
    for name, value in values.items():
        array = np.array(value, copy=True)
        axes = ("chain", *(f"{name}_position_{index}" for index in range(array.ndim - 1)))
        variables[name] = xr.Variable(axes, array)
    return xr.Dataset(variables)


def _restore_continuation(results: object, model: Model) -> tuple[_NUTSContinuation, jax.Array, bool]:
    """Rebuild the sampler continuation stored with results and check that it belongs to the model."""
    if (
        not isinstance(results, xr.DataTree)
        or "sampling_state" not in results.children
        or "posterior" not in results.children
    ):
        raise TypeError(
            "results must be the DataTree returned by sample or continue_sampling with its sampling_state group"
        )
    state = results["sampling_state"]
    if "position" not in state.children or "gradient" not in state.children:
        raise ValueError("The sampling_state group is incomplete. Continue from results returned by sample")
    precision = "float64" if jax.config.values["jax_enable_x64"] else "float32"
    if state.attrs.get("precision") != precision:
        raise ValueError("Continue sampling with the same JAX precision setting as the original run")

    stored = state.to_dataset()
    positions = {str(name): jnp.asarray(value.values) for name, value in state["position"].to_dataset().items()}
    gradients = {str(name): jnp.asarray(value.values) for name, value in state["gradient"].to_dataset().items()}
    declared = model.parameters
    if set(positions) != set(declared) or set(gradients) != set(declared):
        raise ValueError("model does not declare the parameters sampled in results. Continue with the original model")
    for name, parameter in declared.items():
        if positions[name].shape[1:] != tuple(parameter.position_shape):
            raise ValueError(f"Parameter {name!r} has a different shape in results than in the model")

    continuation = _NUTSContinuation(
        positions=positions,
        logdensity=jnp.asarray(stored["logdensity"].values),
        gradients=gradients,
        step_size=jnp.asarray(stored["step_size"].values),
        inverse_mass_matrix=jnp.asarray(stored["inverse_mass_matrix"].values),
        sampling_keys=_restore_keys(stored["sampling_key"].values),
        completed_draws=int(state.attrs["completed_draws"]),
    )
    generation_key = _restore_keys(stored["generation_key"].values)
    return continuation, generation_key, bool(int(state.attrs["generate"]))


def _restore_keys(words: NDArray[np.generic]) -> jax.Array:
    """Rebuild typed random keys from stored key words of any integer dtype."""
    return jnp.asarray(jax.random.wrap_key_data(jnp.asarray(np.asarray(words, dtype=np.uint32))))


def _generation_keys(key: jax.Array, chains: int, start: int, stop: int) -> jax.Array:
    """Assign random draws by chain and absolute draw index."""
    indices = jnp.arange(start, stop, dtype=jnp.uint32)
    chain_keys = jax.random.split(key, chains)
    return jax.vmap(lambda chain_key: jax.vmap(lambda index: jax.random.fold_in(chain_key, index))(indices))(chain_keys)


def _grouped_outputs[Value](generated: Mapping[_OutputKey, Value]) -> dict[_OutputGroup, dict[str, Value]]:
    """Separate result groups from the flat callback outputs."""
    groups: dict[_OutputGroup, dict[str, Value]] = {"generated": {}}
    groups.update({group: {} for group in _ResultGroup})
    for (group, name), value in generated.items():
        groups[group][name] = value
    return groups


def _collect_sampling_results(
    model: Model,
    posterior: Mapping[str, ArrayLike],
    generated: Mapping[_OutputKey, ArrayLike],
    stats: Mapping[str, ArrayLike] | None,
    output_dimensions: dict[str, tuple[str, ...]],
) -> xr.DataTree:
    """Use the same result groups and labels for initial and continued draws."""
    dimensions, coordinates = _parameter_metadata(model)
    groups = _grouped_outputs(generated)
    return _collect_results(
        posterior,
        data=_result_data(model),
        inputs=_result_inputs(model),
        posterior_predictive=groups[_ResultGroup.PREDICTIVE],
        log_likelihood=groups[_ResultGroup.LOG_LIKELIHOOD],
        log_prior=groups[_ResultGroup.LOG_PRIOR],
        generated_quantities=groups["generated"],
        sample_stats=stats,
        dims=dimensions,
        generated_dims=output_dimensions,
        coords=coordinates,
        copy_draws=False,
    )


def _warn_sampling(stats: Mapping[str, ArrayLike]) -> None:
    """Report diagnostics for newly sampled transitions."""
    divergences = int(np.asarray(stats["diverging"]).sum())
    if divergences:
        warnings.warn(
            f"Sampling encountered {divergences} divergent transitions after warmup. "
            "Inspect the model parameterization and consider a higher target_accept",
            RuntimeWarning,
            stacklevel=3,
        )
    limited = int(np.asarray(stats["reached_max_treedepth"]).sum())
    if limited:
        warnings.warn(
            f"Sampling reached max_tree_depth on {limited} draws. "
            "Inspect mixing and consider increasing max_tree_depth",
            RuntimeWarning,
            stacklevel=3,
        )


def sample_prior(
    model: Model,
    prior: Mapping[str, Prior] | PriorSampler,
    *,
    data: object = None,
    draws: int = 500,
    seed: int = 0,
    generate: bool = True,
    batch_size: int = 64,
) -> xr.DataTree:
    """Draw explicit priors and inspect their implied outcomes before fitting.

    Draw every parameter from the supplied priors and reuse the model's
    transformed parameters and ``generated_quantities`` callback.
    The ``log_density`` callback and posterior sampler are not evaluated.
    Keep the sampling distributions consistent with the priors in ``log_density``.

    Parameters
    ----------
    model : Model
        Model supplying parameter declarations and generated quantities.
    prior : mapping of str to Prior or callable
        ``Prior`` objects by parameter name for independent draws in the
        declared shapes, or a JAX-compatible function ``prior(key)`` returning
        one constrained draw per declared parameter for dependent draws.
    data : object, optional
        Inputs for a model without prepared data. Prepared models use their
        stored observations and fitted scaling automatically.
    draws : int, default 500
        Number of independent prior draws.
    seed : int, default 0
        Random seed for parameters and generated quantities.
    generate : bool, default True
        Evaluate outputs from the ``generated_quantities`` callback.
    batch_size : int, default 64
        Maximum prior draws evaluated together, including generated quantities.
        Smaller batches reduce working memory without reducing the draw count.

    Returns
    -------
    xarray.DataTree
        Labeled results with one chain and ``draws`` draws that must fit in
        host memory. The chain axis serves result compatibility, not MCMC.

        - **prior** contains constrained parameter draws.
        - **prior_predictive** contains callback outputs under ``predictive``.
        - **prior_generated_quantities** contains other generated outputs.
        - **observed_data** and **constant_data** contain inputs in evaluated
          units, including fitted scaling. Auxiliary ``Data`` uses **constant_data**.

        Log-likelihood and log-prior outputs are omitted. Without generation,
        only prior draws and available inputs are returned.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    _validate_batch_size(batch_size)
    if isinstance(prior, Mapping):
        prior = _prior_mapping_sampler(model, prior)
    else:
        _validate_prior_sampler(prior)
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
    parameters = _prior_draws(model, prior, prior_key, int(draws), batch_size=batch_size)
    generated: dict[_OutputKey, NDArray[np.generic]] = {}
    output_dimensions: dict[str, tuple[str, ...]] = {}

    if generate and model._has_generated_quantities:
        initial = {name: value[0] for name, value in parameters.items()}
        outputs, arguments = model._generate_with_inputs(preview_key, initial, inputs)
        output_dimensions = _output_dimensions(model, outputs, arguments, dimensions, prepared)
        keys = jax.random.split(generation_key, draws)
        generated = _evaluate_draws(
            lambda key, values: model._generate_with_inputs(key, values, inputs)[0],
            keys,
            parameters,
            sample_shape=(draws,),
            batch_size=batch_size,
        )

    groups = _grouped_outputs(generated)
    results = _collect_results(
        {name: value[None] for name, value in parameters.items()},
        data=prepared,
        inputs=_result_inputs(model),
        posterior_predictive={name: value[None] for name, value in groups[_ResultGroup.PREDICTIVE].items()},
        generated_quantities={name: value[None] for name, value in groups["generated"].items()},
        dims=dimensions,
        generated_dims=output_dimensions,
        coords=coordinates,
        sample_group="prior",
        copy_draws=False,
    )
    results.attrs.update(
        sampling_method="prior",
        seed=int(seed),
        data_scale="model" if model.scaling is not None else "original",
    )
    return results


def _prior_mapping_sampler(model: Model, priors: Mapping[str, object]) -> PriorSampler:
    """Turn Prior objects keyed by parameter name into an independent prior-draw function."""
    _validate_value_names(priors, model._parameterizations, name="prior")
    definitions = []
    for name, declaration in model.parameters.items():
        definition = priors[name]
        if not isinstance(definition, Prior):
            raise TypeError(
                f"Prior definition for {name!r} must be a Prior object with fixed distribution settings. "
                "Use a prior-draw function for dependent draws"
            )
        try:
            definition._validate_shape(declaration.shape)
        except ValueError as exc:
            raise ValueError(f"Invalid prior shape for parameter {name!r}. {exc}") from exc
        definitions.append((name, definition, declaration))

    def draw(key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(key, len(definitions))
        return {
            name: definition._sample(draw_key, declaration.shape, declaration.dtype)
            for (name, definition, declaration), draw_key in zip(definitions, keys, strict=True)
        }

    return draw


def _prior_draws(
    model: Model,
    prior: PriorSampler,
    key: jax.Array,
    draws: int,
    *,
    batch_size: int,
) -> dict[str, NDArray[np.generic]]:
    """Draw and validate constrained parameters without evaluating a density."""

    def draw_parameters(draw_key: jax.Array) -> dict[str, jax.Array]:
        values = prior(draw_key)
        if not isinstance(values, Mapping):
            raise TypeError(
                "prior(key) must return a mapping of parameter names to constrained draws, not a log density. "
                "Return custom log-prior terms under log_prior in generated_quantities"
            )
        _validate_value_names(values, model._parameterizations, name="Prior draws")

        parameters = {}
        for name, declaration in model.parameters.items():
            try:
                value = jnp.asarray(values[name])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Prior draw for {name!r} must be a real array-like value. "
                    "Return constrained draws from prior(key). "
                    "Pass Prior objects to sample_prior by parameter name, "
                    "or return log-prior terms under log_prior in generated_quantities"
                ) from exc
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

    parameters = _evaluate_draws(
        draw_parameters, jax.random.split(key, draws), sample_shape=(draws,), batch_size=batch_size
    )
    validity = _evaluate_draws(valid_support, parameters, sample_shape=(draws,), batch_size=batch_size)
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
    batch_size: int = 64,
) -> xr.DataTree:
    """Evaluate generated quantities from existing posterior draws without refitting.

    Evaluate mapped log-prior terms and any ``generated_quantities``
    callback for every draw. The ``log_density``
    callback and sampling are not rerun. Scenario calculations remain
    defined by the model.

    Parameters
    ----------
    model : Model
        Model with mapped priors or a ``generated_quantities`` callback and
        the fitted parameter declarations.
    results : xarray.DataTree
        Results containing constrained posterior draws with the model's parameter
        names, shapes, and axis labels. Draws may be sliced or thinned.
    new_data : dataframe-like, PreparedData, or object, optional
        Scenario observations using the original source columns. Dataframes
        reuse the model's column selections, labels, and fitted scaling.
        Omit to evaluate stored observations. Earlier exposures are not added
        automatically. Include them through ``prepare_data(media_history=...)``.
        Other models receive this input directly. Omit outcomes only when no
        evaluated callback needs them. Auxiliary ``Data`` inputs remain
        fixed across scenarios.
    seed : int, default 0
        Random seed for generated quantities, with an independent key per draw.
    batch_size : int, default 64
        Maximum posterior draws evaluated together across chains. Smaller
        batches reduce working memory without changing the selected draws.

    Returns
    -------
    xarray.DataTree
        New results that must fit in host memory, leaving inputs unchanged.

        - **posterior** retains the draws and sample labels.
        - **posterior_predictive**, **log_likelihood**, **log_prior**, and
          **generated_quantities** contain newly evaluated model outputs.
        - **observed_data** and **constant_data** contain inputs in model units.

        Original sampler diagnostics are omitted.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    _validate_batch_size(batch_size)
    if not model._has_generated_quantities:
        raise ValueError("The model must define a generated_quantities callback or map priors")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    dimensions, coordinates = _parameter_metadata(model)
    posterior, coordinates = _parameter_draws(model, results, dimensions, coordinates)
    # The new result tree owns its draws without modifying the supplied results.
    posterior = {name: value.copy() for name, value in posterior.items()}
    prepared = _result_data(model)
    inputs = model.data if model._data is not None else new_data
    if model._data is not None and new_data is not None:
        inputs, aligned = model._prepare_data(new_data)
        prepared = replace(aligned, arrays={name: np.array(inputs.values[name], copy=True) for name in aligned.arrays})

    generation_key, preview_key = jax.random.split(jax.random.key(int(seed)))
    initial = {name: value[0, 0] for name, value in posterior.items()}
    outputs, arguments = model._generate_with_inputs(preview_key, initial, inputs)
    output_dimensions = _output_dimensions(model, outputs, arguments, dimensions, prepared)
    chains, draws = next(iter(posterior.values())).shape[:2]
    keys = jax.random.split(generation_key, (chains, draws))
    generated = _evaluate_draws(
        lambda key, parameters: model._generate_with_inputs(key, parameters, inputs)[0],
        keys,
        posterior,
        sample_shape=(chains, draws),
        batch_size=batch_size,
    )
    groups = _grouped_outputs(generated)
    evaluated = _collect_results(
        posterior,
        data=prepared,
        inputs=_result_inputs(model),
        posterior_predictive=groups[_ResultGroup.PREDICTIVE],
        log_likelihood=groups[_ResultGroup.LOG_LIKELIHOOD],
        log_prior=groups[_ResultGroup.LOG_PRIOR],
        generated_quantities=groups["generated"],
        dims=dimensions,
        generated_dims=output_dimensions,
        coords=coordinates,
        copy_draws=False,
    )
    evaluated.attrs.update(generation_seed=int(seed), data_scale="model" if model.scaling is not None else "original")
    return evaluated


def _parameter_draws(
    model: Model,
    results: xr.DataTree,
    dimensions: dict[str, tuple[str, ...]],
    coordinates: dict[str, NDArray[np.generic]],
    *,
    group: Literal["prior", "posterior"] = "posterior",
) -> tuple[dict[str, NDArray[np.generic]], dict[str, NDArray[np.generic]]]:
    """Validate labeled constrained draws from the selected result group."""
    if not isinstance(group, str) or group not in ("prior", "posterior"):
        raise ValueError("group must be 'prior' or 'posterior'")
    label = group.capitalize()
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray.DataTree containing {group} draws")
    if group not in results.children:
        raise ValueError(f"results must contain a {group} group")
    dataset = results[group].to_dataset()
    if not dataset.data_vars or set(dataset.data_vars) != set(model.parameters):
        raise ValueError(f"{label} parameter names must match the model declarations")
    for axis in ("chain", "draw"):
        if dataset.sizes.get(axis, 0) == 0:
            raise ValueError(f"{group} must contain at least one {axis}")
        if axis in coordinates:
            raise ValueError(f"{label} draws supply chain and draw coordinates. Supply only model axes in coords")

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
                    raise ValueError(f"{label} coordinate {name!r} must match the model group labels and ordering")

    samples = {}
    for name, parameter in model.parameters.items():
        value = dataset[name]
        axes = ("chain", "draw", *dimensions[name])
        if len(value.dims) != len(axes) or set(value.dims) != set(axes):
            raise ValueError(f"{label} dimensions for {name!r} must match the model axes {axes}")
        value = value.transpose(*axes)
        if value.shape[2:] != parameter.shape:
            raise ValueError(f"{label} shape for {name!r} must match its constrained parameter shape")
        for axis, size in zip(dimensions[name], parameter.shape, strict=True):
            labels = expected_coordinates.get(axis, np.arange(size))
            if axis not in value.coords or not _same_labels(value.coords[axis].values, labels):
                raise ValueError(f"{label} coordinate {axis!r} must match the model labels and ordering")
            coordinates[axis] = labels.copy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{label} draws for {name!r} must be finite real numbers")
        if jax.dtypes.canonicalize_dtype(array.dtype) != array.dtype:
            raise ValueError(f"Enable JAX 64-bit mode to evaluate these {group} draws without losing precision")
        samples[name] = array

    for axis in ("chain", "draw"):
        coordinates[axis] = np.array(
            dataset.coords[axis].values if axis in dataset.coords else np.arange(dataset.sizes[axis])
        )
    return samples, coordinates


def _validate_batch_size(batch_size: int) -> None:
    """Reject invalid batch sizes before sampling or evaluating callbacks."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _evaluate_draws[Key: Hashable](
    function: Callable[..., Mapping[Key, jax.Array]],
    *arguments: jax.Array | Mapping[str, jax.Array | NDArray[np.generic]],
    sample_shape: tuple[int, ...],
    batch_size: int,
) -> dict[Key, NDArray[np.generic]]:
    """Evaluate flattened draw batches into preallocated host-side results."""
    total = int(np.prod(sample_shape))
    flattened = jax.tree.map(lambda value: value.reshape((total, *value.shape[len(sample_shape) :])), arguments)
    evaluate = jax.jit(jax.vmap(function))
    buffers: dict[Key, NDArray[np.generic]] = {}

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = jax.tree.map(lambda value, start=start, stop=stop: value[start:stop], flattened)
        outputs = jax.device_get(evaluate(*batch))

        for name in outputs:
            if name not in buffers:
                buffers[name] = np.empty((total, *outputs[name].shape[1:]), dtype=outputs[name].dtype)
            buffers[name][start:stop] = outputs[name]

        # Release batch buffers before the next device evaluation.
        del batch, outputs

    return {name: value.reshape((*sample_shape, *value.shape[1:])) for name, value in buffers.items()}


def _parameter_metadata(model: Model) -> tuple[dict[str, tuple[str, ...]], dict[str, NDArray[np.generic]]]:
    """Use declared parameter axes without guessing the meaning of custom shapes."""
    dimensions: dict[str, tuple[str, ...]] = {}
    coordinates: dict[str, NDArray[np.generic]] = {}
    training = model._training
    if training is not None:
        coordinates.update({name: labels.copy() for name, labels in training.input_coords.items()})
    coordinates.update({name: labels.copy() for name, labels in model._result_coords.items()})
    if (
        model._data is not None
        and training is not None
        and any(
            _metadata_source(name, source, model._data).kind == "reference" for name, source in model._blocks.bindings
        )
    ):
        reference_coordinates = _coordinates(
            {"reference_time": training.time_values, "reference_media_time": training.media_time_values}
        )
        for axis, labels in reference_coordinates.items():
            if axis in coordinates and not _same_labels(coordinates[axis], labels):
                raise ValueError(f"Coordinate {axis!r} conflicts with the training reference labels")
            coordinates[axis] = labels
    dimensions.update(model._result_dims)
    for name, parameter in model.parameters.items():
        dimensions.setdefault(name, tuple(f"{name}_dim_{index}" for index in range(len(parameter.shape))))
    return dimensions, coordinates


def _output_dimensions(
    model: Model,
    outputs: Mapping[_OutputKey, jax.Array],
    arguments: Mapping[str, object],
    parameter_dimensions: dict[str, tuple[str, ...]],
    prepared: PreparedData | None,
) -> dict[str, tuple[str, ...]]:
    """Label known callback inputs and declared observations without shape guessing."""
    missing = set(model._generated_dims) - {name for _, name in outputs}
    if missing:
        raise ValueError(f"Generated result metadata refers to missing outputs {sorted(missing)}")
    input_dimensions: dict[str, tuple[str, ...]] = {}
    reference_inputs: set[str] = set()
    reference_namespaces: set[str] = set()
    scaling_inputs: set[str] = set()
    auxiliary_inputs: set[str] = set()
    observation_axes: tuple[str, ...] = ()
    outcome_shape: tuple[int, ...] | None = None
    role_dimensions: dict[str, tuple[str, ...]] = {}
    reference_dimensions: dict[str, tuple[str, ...]] = {}
    grouped_scale = False
    # A declared mapping names reference members after the block inputs, so the
    # role that carries each member's axes is recovered through the declarations.
    declared_roles: dict[str, str] = {}
    if model._data is not None and model._data.variable_sources is not None:
        declared_roles = {name: origin.name for name, origin in model._data.variable_sources.items()}

    def reference_axes(member: str) -> tuple[str, ...]:
        # Training arrays keep their own time axes even when new data lacks the role.
        role = declared_roles.get(member, member)
        return tuple(
            f"reference_{axis}" if axis in ("time", "media_time") else axis for axis in reference_dimensions[role]
        )

    if prepared is not None:
        assert model._data is not None
        observation_axes = _data_dimensions(prepared)["outcome"]
        reference_dimensions = _reference_dimensions(prepared)
        outcome_shape = (len(prepared.time_values),)
        if prepared.group_columns:
            outcome_shape += (len(prepared.group_values),)
        role_dimensions = {
            name: spec.axes for name, spec in prepared.model_inputs.items() if spec.source in ("data", "time")
        }
        assert model._training is not None
        auxiliary_dims = model._training.input_dims
        role_dimensions.update(auxiliary_dims)
        grouped_scale = model._data.outcome_group_scale
        for name, source in model._blocks.generation_inputs:
            source, source_name = _metadata_source(name, source, model._data)
            if source == "data":
                input_dimensions[name] = role_dimensions[source_name]
                if source_name in auxiliary_dims:
                    auxiliary_inputs.add(name)
            elif source == "reference" and source_name == "reference":
                reference_namespaces.add(name)
            elif source == "reference":
                # A transformed data output that passed a training array through.
                input_dimensions[name] = reference_axes(source_name)
                reference_inputs.add(name)
            elif source == "builtin" and source_name == "outcome_scaling":
                scaling_inputs.add(name)
            elif source == "builtin":
                input_dimensions[name] = ()
            elif source == "parameter":
                input_dimensions[name] = parameter_dimensions[name]
    else:
        input_dimensions.update(parameter_dimensions)

    dimensions: dict[str, tuple[str, ...]] = {}
    for (group, name), value in outputs.items():
        fallback = tuple(f"{name}_dim_{index}" for index in range(value.ndim))
        inherited = {axes for argument, axes in input_dimensions.items() if value is arguments.get(argument)}
        input_axes = {input_dimensions[argument] for argument in auxiliary_inputs if value is arguments.get(argument)}
        # Unchanged members of the reference namespace keep their training axes, and the
        # outcome transform's factors keep the group axis under population scaling.
        member_axes: set[tuple[str, ...]] = set()
        for argument in reference_namespaces:
            namespace = arguments.get(argument)
            if isinstance(namespace, Reference):
                member_axes.update(
                    reference_axes(member) for member, array in namespace.values.items() if value is array
                )
        for argument in scaling_inputs:
            transform = arguments.get(argument)
            if isinstance(transform, Scaling) and (value is transform.scale or value is transform.offset):
                member_axes.add(("group",) if grouped_scale else ())
        if name in model._generated_dims:
            axes = model._generated_dims[name]
        elif member_axes or any(value is arguments.get(argument) for argument in reference_inputs):
            candidates = inherited | member_axes
            axes = candidates.pop() if len(candidates) == 1 else fallback
        elif len(input_axes) == 1:
            axes = input_axes.pop()
        elif group in (_ResultGroup.PREDICTIVE, _ResultGroup.LOG_LIKELIHOOD) and value.shape == outcome_shape:
            axes = observation_axes
        elif len(inherited) == 1:
            # Equal values or equal shapes do not establish a shared axis or ordering.
            axes = inherited.pop()
        else:
            axes = fallback
        if dimensions.setdefault(name, axes) != axes:
            raise ValueError(f"Generated output {name!r} has different axes across result groups. Use generated_dims")
    return dimensions


def _result_data(model: Model) -> PreparedData | None:
    """Snapshot the exact evaluated inputs with their original observation labels."""
    if model._data is None or model._training is None:
        return None
    training = model._training
    layout = training.layout
    return PreparedData(
        arrays={
            name: np.array(value, copy=True)
            for name, value in model._data.values.items()
            if name not in training.time_inputs and name not in training.input_dims
        },
        time_column=training.time_column or "time",
        time_values=training.time_values,
        media_time_values=training.media_time_values,
        group_columns=layout.group_columns,
        group_values=layout.group_values,
        columns=layout.columns.copy(),
        channels=layout.channels,
        organic_channels=layout.organic_channels,
        rf_channels=layout.rf_channels,
        organic_rf_channels=layout.organic_rf_channels,
        frequency=training.frequency,
    )


def _result_inputs(model: Model) -> xr.Dataset | None:
    """Snapshot auxiliary inputs in their evaluated dtype and declared ordering."""
    training = model._training
    if training is None or not training.input_dims:
        return None
    assert model._data is not None
    return xr.Dataset(
        {
            name: xr.Variable(axes, np.array(model._data.values[name], copy=True))
            for name, axes in training.input_dims.items()
        },
        coords={name: labels.copy() for name, labels in training.input_coords.items()},
    )
