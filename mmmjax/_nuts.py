"""NUTS sampling for unconstrained model positions."""

from collections.abc import Callable
from typing import TypeAlias, cast

import blackjax.util  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

_Position: TypeAlias = dict[str, jax.Array]
_Samples: TypeAlias = tuple[_Position, dict[str, jax.Array]]


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
) -> _Samples:
    """Adapt each chain independently and retain positions and diagnostics."""
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

    @jax.jit
    def run_chain(position: _Position, key: jax.Array) -> _Samples:
        warmup_key, sampling_key = jax.random.split(key)
        (state, parameters), _ = run_warmup(warmup_key, position, num_steps=warmup)
        sampler = blackjax.nuts(logdensity, **parameters)

        def retain(state: blackjax.mcmc.hmc.HMCState, info: blackjax.mcmc.nuts.NUTSInfo) -> _Samples:
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
            return cast(_Position, state.position), statistics

        _, history = blackjax.util.run_inference_algorithm(
            rng_key=sampling_key,
            inference_algorithm=sampler,
            num_steps=draws,
            initial_state=state,
            transform=retain,
        )
        return cast(_Samples, history)

    if chain_method == "vectorized":
        return cast(_Samples, jax.jit(jax.vmap(run_chain))(initial_positions, keys))

    if chain_method == "parallel":
        mesh = jax.make_mesh(
            (keys.shape[0],),
            ("chain",),
            axis_types=(AxisType.Auto,),
            devices=jax.local_devices()[: keys.shape[0]],
        )

        def run_device(positions: _Position, keys: jax.Array) -> _Samples:
            position = jax.tree.map(lambda value: value[0], positions)
            samples = run_chain(position, keys[0])
            return cast(_Samples, jax.tree.map(lambda value: value[None], samples))

        chain_spec = PartitionSpec("chain")  # type: ignore[no-untyped-call]
        with jax.set_mesh(mesh):
            initial_positions, keys = jax.device_put((initial_positions, keys), NamedSharding(mesh, chain_spec))
            run_parallel = jax.jit(
                jax.shard_map(
                    run_device,
                    mesh=mesh,
                    in_specs=chain_spec,
                    out_specs=chain_spec,
                    # Chain-local control flow mixes constant and varying state.
                    # Every output is sharded, with no replication assertions.
                    check_vma=False,
                )
            )
            return cast(_Samples, run_parallel(initial_positions, keys))

    chains = [
        run_chain({name: values[index] for name, values in initial_positions.items()}, keys[index])
        for index in range(keys.shape[0])
    ]
    return cast(_Samples, jax.tree.map(lambda *values: jnp.stack(values), *chains))


def _discard_adaptation(_state: object, _info: object, _adaptation_state: object) -> tuple[()]:
    """Exclude intermediate warmup states from the retained output."""
    return ()
