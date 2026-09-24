"""NUTS sampling for unconstrained model positions."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import blackjax  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from numpy.typing import NDArray

type _Position = dict[str, jax.Array]
type _DeviceSamples = tuple[_Position, dict[str, jax.Array]]
type _Samples = tuple[dict[str, NDArray[np.generic]], dict[str, NDArray[np.generic]]]
type _State = blackjax.mcmc.hmc.HMCState
type _Parameters = dict[str, jax.Array]
type _Adapted = tuple[_State, _Parameters, jax.Array]


@dataclass(frozen=True)
class _NUTSContinuation:
    """Retain each chain's final sampler state and key stream as plain arrays.

    Every field is an array or an integer so the continuation can be stored
    alongside results and restored without live Python objects.
    """

    positions: _Position
    logdensity: jax.Array
    gradients: _Position
    step_size: jax.Array
    inverse_mass_matrix: jax.Array
    sampling_keys: jax.Array
    completed_draws: int

    def state(self) -> _State:
        """Rebuild the sampler state blackjax expects for the next step."""
        build_state = cast(Callable[..., _State], blackjax.mcmc.hmc.HMCState)
        return build_state(self.positions, self.logdensity, self.gradients)


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
    continuation: _NUTSContinuation | None = None,
) -> tuple[_Samples, _NUTSContinuation]:
    """Adapt or continue chains and copy bounded sampling chunks into host arrays.

    Returns the host-side draws and statistics together with the final chain
    positions and tuning, from which sampling can resume without warmup.
    """
    adapt_chain: Callable[[_Position, jax.Array], _Adapted] | None = None
    retained: _Adapted | None = None
    completed_draws = 0

    if continuation is None:
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

        def adapt_position(position: _Position, key: jax.Array) -> _Adapted:
            warmup_key, sampling_key = jax.random.split(key)
            (state, parameters), _ = run_warmup(warmup_key, position, num_steps=warmup)
            # Only adapted arrays cross the compilation boundary. The tree-depth
            # limit remains a static Python value when constructing the sampler.
            adapted = {name: jnp.asarray(parameters[name]) for name in ("step_size", "inverse_mass_matrix")}
            return state, adapted, sampling_key

        adapt_chain = adapt_position
    else:
        retained = (
            continuation.state(),
            {"step_size": continuation.step_size, "inverse_mass_matrix": continuation.inverse_mass_matrix},
            continuation.sampling_keys,
        )
        completed_draws = continuation.completed_draws

    def draw_keys(sampling_key: jax.Array, start: int, stop: int) -> jax.Array:
        indices = jnp.arange(completed_draws + start, completed_draws + stop, dtype=jnp.uint32)
        return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(sampling_key, indices)

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

    chains = keys.shape[0] if continuation is None else continuation.sampling_keys.shape[0]
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
        adapt: Callable[[_Position, jax.Array], _Adapted] | None,
        sample: Callable[[_State, _Parameters, jax.Array], tuple[_State, _DeviceSamples]],
        positions: _Position,
        chain_keys: jax.Array,
        retained: _Adapted | None,
    ) -> _Adapted:
        counter = " per device" if chain_method == "parallel" else ""
        if retained is None:
            assert adapt is not None
            with _progress(progress, f"Warmup{counter} ({chains} chains)"):
                state, parameters, sampling_keys = jax.tree.map(
                    lambda value: value.block_until_ready(), adapt(positions, chain_keys)
                )
        else:
            state, parameters, sampling_keys = retained

        for start in range(0, draws, chunk_size):
            stop = min(start + chunk_size, draws)
            step_keys = jax.vmap(draw_keys, in_axes=(0, None, None))(sampling_keys, start, stop)
            with _progress(progress, f"Sampling{counter} ({chains} chains) draws {start + 1}-{stop}/{draws}"):
                state, chunk = sample(state, parameters, step_keys)
                transfer(chunk, start, stop)
                del chunk
        return state, parameters, sampling_keys

    if chain_method == "vectorized":
        adapt_batched = None if adapt_chain is None else jax.jit(jax.vmap(adapt_chain))
        final = run_batched(adapt_batched, jax.jit(jax.vmap(sample_chunk)), initial_positions, keys, retained)

    elif chain_method == "parallel":
        mesh = jax.make_mesh(
            (chains,),
            ("chain",),
            axis_types=(AxisType.Auto,),
            devices=jax.local_devices()[:chains],
        )

        def adapt_device(positions: _Position, keys: jax.Array) -> _Adapted:
            assert adapt_chain is not None
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
            sharding = NamedSharding(mesh, chain_spec)
            adapt_parallel = None
            if retained is None:
                initial_positions, keys = jax.device_put((initial_positions, keys), sharding)
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
            else:
                retained = jax.device_put(retained, sharding)

            sample_parallel = jax.jit(
                jax.shard_map(
                    sample_device,
                    mesh=mesh,
                    in_specs=chain_spec,
                    out_specs=chain_spec,
                    check_vma=False,
                )
            )
            final = run_batched(adapt_parallel, sample_parallel, initial_positions, keys, retained)

    else:
        adapt = None if adapt_chain is None else jax.jit(adapt_chain)
        sample = jax.jit(sample_chunk)
        final_chains: list[_Adapted] = []
        for chain in range(chains):
            if retained is None:
                assert adapt is not None
                position = {name: values[chain] for name, values in initial_positions.items()}
                with _progress(progress, f"Warmup chain {chain + 1}/{chains}"):
                    state, parameters, sampling_key = jax.tree.map(
                        lambda value: value.block_until_ready(), adapt(position, keys[chain])
                    )
            else:
                state, parameters, sampling_key = jax.tree.map(lambda value, chain=chain: value[chain], retained)

            for start in range(0, draws, chunk_size):
                stop = min(start + chunk_size, draws)
                step_keys = draw_keys(sampling_key, start, stop)
                with _progress(progress, f"Sampling chain {chain + 1}/{chains} draws {start + 1}-{stop}/{draws}"):
                    state, chunk = sample(state, parameters, step_keys)
                    transfer(chunk, start, stop, chain)
                    del chunk
            final_chains.append((state, parameters, sampling_key))

        final = jax.tree.map(lambda *values: jnp.stack(values), *final_chains)

    assert samples is not None
    state, parameters, sampling_keys = final
    resumable = _NUTSContinuation(
        positions=cast(_Position, state.position),
        logdensity=jnp.asarray(state.logdensity),
        gradients=cast(_Position, state.logdensity_grad),
        step_size=parameters["step_size"],
        inverse_mass_matrix=parameters["inverse_mass_matrix"],
        sampling_keys=sampling_keys,
        completed_draws=completed_draws + draws,
    )
    return samples, resumable


@contextmanager
def _progress(enabled: bool, label: str) -> Iterator[None]:
    """Keep scan callbacks within their display context even when an error occurs."""
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
