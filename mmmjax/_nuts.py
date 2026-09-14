"""NUTS sampling for unconstrained model positions."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeAlias, cast

import blackjax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from numpy.typing import NDArray

_Position: TypeAlias = dict[str, jax.Array]
_DeviceSamples: TypeAlias = tuple[_Position, dict[str, jax.Array]]
_Samples: TypeAlias = tuple[dict[str, NDArray[np.generic]], dict[str, NDArray[np.generic]]]
_State: TypeAlias = blackjax.mcmc.hmc.HMCState
_Parameters: TypeAlias = dict[str, jax.Array]
_Adapted: TypeAlias = tuple[_State, _Parameters, jax.Array]


def _sample_nuts(
    logdensity: Callable[[_Position], jax.Array],
    initial_positions: _Position,
    keys: jax.Array,
    *,
    draws: int,
    warmup: int,
    target_accept: float,
    max_tree_depth: int,
    chain_method: str,
    mass_matrix: str,
    chunk_size: int,
    progress: bool = True,
) -> _Samples:
    """Adapt independently, then copy bounded sampling chunks into host arrays."""
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=mass_matrix == "diagonal",
        target_acceptance_rate=target_accept,
        max_num_doublings=max_tree_depth,
        adaptation_info_fn=_discard_adaptation,
    )
    run_warmup = cast(
        Callable[..., tuple[tuple[blackjax.mcmc.hmc.HMCState, dict[str, jax.Array | int]], object]],
        adaptation.run,
    )

    def adapt_chain(position: _Position, key: jax.Array) -> _Adapted:
        warmup_key, sampling_key = jax.random.split(key)
        (state, parameters), _ = run_warmup(warmup_key, position, num_steps=warmup)
        # Only adapted arrays cross the compilation boundary. The tree-depth
        # limit remains a static Python value when constructing the sampler.
        adapted = {name: jnp.asarray(parameters[name]) for name in ("step_size", "inverse_mass_matrix")}
        return state, adapted, jax.random.split(sampling_key, draws)

    def sample_chunk(state: _State, parameters: _Parameters, step_keys: jax.Array) -> tuple[_State, _DeviceSamples]:
        sampler = blackjax.nuts(logdensity, **parameters, max_num_doublings=max_tree_depth)

        def step(state: _State, key: jax.Array) -> tuple[_State, _DeviceSamples]:
            state, info = sampler.step(key, state)
            statistics = {
                "lp": jnp.asarray(state.logdensity),
                "diverging": jnp.asarray(info.is_divergent),
                "acceptance_rate": jnp.asarray(info.acceptance_rate),
                "energy": jnp.asarray(info.energy),
                "tree_depth": jnp.asarray(info.num_trajectory_expansions),
                "n_steps": jnp.asarray(info.num_integration_steps),
                "reached_max_treedepth": jnp.asarray(info.num_trajectory_expansions >= max_tree_depth),
                "step_size": jnp.asarray(parameters["step_size"]),
            }
            return state, (cast(_Position, state.position), statistics)

        return jax.lax.scan(step, state, step_keys)

    chains = keys.shape[0]
    samples: _Samples | None = None

    def transfer(chunk: _DeviceSamples, start: int, stop: int, chain: int | None = None) -> None:
        nonlocal samples
        host_chunk = jax.device_get(chunk)
        if chain is not None:
            host_chunk = jax.tree.map(lambda value: value[None], host_chunk)
        if samples is None:
            samples = jax.tree.map(
                lambda value: np.empty((chains, draws, *value.shape[2:]), dtype=value.dtype), host_chunk
            )
        chain_slice = slice(None) if chain is None else slice(chain, chain + 1)
        for destination, value in zip(jax.tree.leaves(samples), jax.tree.leaves(host_chunk), strict=True):
            destination[chain_slice, start:stop] = value

    def run_batched(
        adapt: Callable[[_Position, jax.Array], _Adapted],
        sample: Callable[[_State, _Parameters, jax.Array], tuple[_State, _DeviceSamples]],
        positions: _Position,
        chain_keys: jax.Array,
    ) -> None:
        counter = " per device" if chain_method == "parallel" else ""
        with _progress(progress, f"Warmup{counter} ({chains} chains)"):
            state, parameters, step_keys = jax.tree.map(
                lambda value: value.block_until_ready(), adapt(positions, chain_keys)
            )
        for start in range(0, draws, chunk_size):
            stop = min(start + chunk_size, draws)
            with _progress(progress, f"Sampling{counter} ({chains} chains) draws {start + 1}-{stop}/{draws}"):
                state, chunk = sample(state, parameters, step_keys[:, start:stop])
                transfer(chunk, start, stop)
                del chunk

    if chain_method == "vectorized":
        run_batched(jax.jit(jax.vmap(adapt_chain)), jax.jit(jax.vmap(sample_chunk)), initial_positions, keys)

    elif chain_method == "parallel":
        mesh = jax.make_mesh(
            (chains,),
            ("chain",),
            axis_types=(AxisType.Auto,),
            devices=jax.local_devices()[:chains],
        )

        def adapt_device(positions: _Position, keys: jax.Array) -> _Adapted:
            position = jax.tree.map(lambda value: value[0], positions)
            adapted = adapt_chain(position, keys[0])
            return cast(_Adapted, jax.tree.map(lambda value: value[None], adapted))

        def sample_device(
            state: _State, parameters: _Parameters, step_keys: jax.Array
        ) -> tuple[_State, _DeviceSamples]:
            state, parameters, step_keys = jax.tree.map(lambda value: value[0], (state, parameters, step_keys))
            result = sample_chunk(state, parameters, step_keys)
            return cast(tuple[_State, _DeviceSamples], jax.tree.map(lambda value: value[None], result))

        chain_spec = PartitionSpec("chain")  # type: ignore[no-untyped-call]
        with jax.set_mesh(mesh):
            initial_positions, keys = jax.device_put((initial_positions, keys), NamedSharding(mesh, chain_spec))
            adapt_parallel = jax.jit(
                jax.shard_map(
                    adapt_device,
                    mesh=mesh,
                    in_specs=chain_spec,
                    out_specs=chain_spec,
                    # Chain-local control flow mixes constant and varying state.
                    # Every output is sharded, with no replication assertions.
                    check_vma=False,
                )
            )
            sample_parallel = jax.jit(
                jax.shard_map(
                    sample_device,
                    mesh=mesh,
                    in_specs=chain_spec,
                    out_specs=chain_spec,
                    check_vma=False,
                )
            )
            run_batched(adapt_parallel, sample_parallel, initial_positions, keys)

    else:
        adapt = jax.jit(adapt_chain)
        sample = jax.jit(sample_chunk)
        for chain in range(chains):
            position = {name: values[chain] for name, values in initial_positions.items()}
            with _progress(progress, f"Warmup chain {chain + 1}/{chains}"):
                state, parameters, step_keys = jax.tree.map(
                    lambda value: value.block_until_ready(), adapt(position, keys[chain])
                )
            for start in range(0, draws, chunk_size):
                stop = min(start + chunk_size, draws)
                with _progress(progress, f"Sampling chain {chain + 1}/{chains} draws {start + 1}-{stop}/{draws}"):
                    state, chunk = sample(state, parameters, step_keys[start:stop])
                    transfer(chunk, start, stop, chain)
                    del chunk
            del state, parameters, step_keys

    assert samples is not None
    return samples


@contextmanager
def _progress(enabled: bool, label: str) -> Iterator[None]:
    """Keep scan callbacks within their display context, including on errors."""
    if not enabled:
        yield
        return

    with blackjax.progress_bar(label=label):
        try:
            yield
        finally:
            jax.effects_barrier()  # type: ignore[no-untyped-call]


def _discard_adaptation(_state: object, _info: object, _adaptation_state: object) -> tuple[()]:
    """Exclude intermediate warmup states from the retained output."""
    return ()
