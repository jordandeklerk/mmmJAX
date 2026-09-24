"""Tests for sampling JAX models and collecting constrained, labeled draws."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from textwrap import dedent

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mmmjax
import mmmjax._nuts as nuts
import mmmjax.sampling as sampling
from mmmjax import (
    CorrelationCholesky,
    Data,
    Interval,
    Model,
    Positive,
    Prior,
    Real,
    Simplex,
    beta,
    continue_sampling,
    dirichlet,
    fit_data_scaling,
    fourier_features,
    geometric_adstock,
    half_normal,
    hill_saturation,
    lkj_cholesky,
    lognormal,
    normal,
    normal_logpdf,
    normal_rng,
    prepare_data,
    sample,
)


@pytest.fixture
def normal_model():
    return Model(parameters={"location": Real()}, log_density=lambda data, location: normal(location, data, 1.0))


@pytest.fixture
def nuts_calls(monkeypatch):
    calls = []

    def stationary_draws(
        logdensity,
        initial_positions,
        keys,
        *,
        draws,
        warmup,
        target_accept,
        max_tree_depth,
        chain_method,
        mass_matrix,
        chunk_size,
        progress,
        continuation=None,
    ):
        calls.append(
            {
                "positions": jax.tree.map(np.array, initial_positions),
                "keys": np.asarray(jax.random.key_data(keys)),
                "draws": draws,
                "warmup": warmup,
                "target_accept": target_accept,
                "max_tree_depth": max_tree_depth,
                "chain_method": chain_method,
                "mass_matrix": mass_matrix,
                "chunk_size": chunk_size,
                "progress": progress,
            }
        )
        positions = jax.tree.map(
            lambda value: np.broadcast_to(value[:, None], (value.shape[0], draws, *value.shape[1:])).copy(),
            initial_positions,
        )
        chains = next(iter(initial_positions.values())).shape[0]
        statistics = {
            "diverging": np.zeros((chains, draws), dtype=bool),
            "reached_max_treedepth": np.zeros((chains, draws), dtype=bool),
            "lp": np.asarray(jax.vmap(jax.vmap(logdensity))(positions)).copy(),
        }
        return (positions, statistics), _stub_continuation(positions, statistics, keys)

    monkeypatch.setattr(sampling, "_sample_nuts", stationary_draws)
    return calls


def _stub_continuation(positions, statistics, keys):
    """Build the resumable state a sampler stub would hand back after its last draw."""
    final = {name: jnp.asarray(value[:, -1]) for name, value in positions.items()}
    chains = keys.shape[0]
    size = sum(int(np.prod(value.shape[1:])) for value in final.values())
    return nuts._NUTSContinuation(
        positions=final,
        logdensity=jnp.asarray(statistics["lp"][:, -1]),
        gradients={name: jnp.zeros_like(value) for name, value in final.items()},
        step_size=jnp.ones(chains),
        inverse_mass_matrix=jnp.ones((chains, size)),
        sampling_keys=keys,
        completed_draws=int(next(iter(positions.values())).shape[1]),
    )


@pytest.fixture
def adaptation_metrics(monkeypatch):
    captured = {"diagonal": [], "matrices": [], "steps": []}
    window_adaptation = nuts.blackjax.window_adaptation

    def record_adaptation(*args, **kwargs):
        captured["diagonal"].append(kwargs["is_mass_matrix_diagonal"])
        adaptation = window_adaptation(*args, **kwargs)

        def run(*args, **kwargs):
            adapted, information = adaptation.run(*args, **kwargs)

            def record_result(matrix):
                captured["matrices"].append(np.array(matrix))
                captured["steps"].append(kwargs["num_steps"])

            jax.debug.callback(record_result, adapted[1]["inverse_mass_matrix"])
            return adapted, information

        return adaptation._replace(run=run)

    monkeypatch.setattr(nuts.blackjax, "window_adaptation", record_adaptation)
    return captured


@pytest.fixture
def progress_contexts(monkeypatch):
    contexts = []
    progress_bar = nuts.blackjax.progress_bar

    @contextmanager
    def record_progress(*args, **kwargs):
        original_scan = jax.lax.scan
        try:
            with progress_bar(*args, **kwargs) as state:
                contexts.append(state)
                yield state
                assert not state.closed
                assert state.n_steps > 0
                assert state.current_step == state.n_steps - 1
        finally:
            assert jax.lax.scan is original_scan

    monkeypatch.setattr(nuts.blackjax, "progress_bar", record_progress)
    return contexts


def _prepared_model():
    data = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [100.0, 150.0, 200.0], "price": [5.0, 4.0, 3.0]}),
        time="week",
        outcome="sales",
        controls=["price"],
    )

    def density(outcome, intercept):
        return normal(outcome, intercept, 10.0) + normal(intercept, 150.0, 50.0)

    def generate(key, outcome, intercept):
        mean = jnp.full_like(outcome, intercept)
        return {
            "predictive": {"prediction": normal_rng(key, mean, 10.0)},
            "log_likelihood": {"pointwise": normal_logpdf(outcome, mean, 10.0)},
            "mean": mean,
        }

    model = Model(
        parameters={"intercept": Real()},
        log_density=density,
        generated_quantities=generate,
        data=data,
        generated_dims={"mean": ("time",)},
    )
    return model, data


def test_public_sampling_replaces_manual_result_collection():
    assert "sample" in mmmjax.__all__
    assert "collect_results" not in mmmjax.__all__
    assert not hasattr(mmmjax, "collect_results")


@pytest.mark.parametrize(
    "chain_method,mass_matrix", [("sequential", "diagonal"), ("vectorized", "diagonal"), ("sequential", "dense")]
)
def test_normal_nuts_produces_reproducible_independent_chains(normal_model, chain_method, mass_matrix):
    options = {
        "data": 0.0,
        "draws": 200,
        "warmup": 150,
        "chains": 2,
        "seed": 19,
        "target_accept": 0.8,
        "chain_method": chain_method,
        "mass_matrix": mass_matrix,
    }
    first = sample(normal_model, **options)
    second = sample(normal_model, **options)
    assert isinstance(first, xr.DataTree)
    posterior = first["posterior"]["location"]
    assert posterior.dims == ("chain", "draw")
    assert posterior.shape == (2, 200)
    assert np.isfinite(posterior).all()
    assert abs(float(posterior.mean())) < 0.3
    assert 0.65 < float(posterior.std()) < 1.4
    assert not np.array_equal(posterior.values[0], posterior.values[1])
    xr.testing.assert_equal(first["posterior"].to_dataset(), second["posterior"].to_dataset())
    assert first["sample_stats"]["diverging"].dtype == np.bool_
    assert first["sample_stats"]["diverging"].shape == (2, 200)
    assert first.attrs["chain_method"] == chain_method
    assert first.attrs["mass_matrix"] == mass_matrix
    assert first["sampling_state"].attrs["completed_draws"] == 200
    statistics = first["sample_stats"]
    assert set(statistics.data_vars) == {
        "lp",
        "diverging",
        "acceptance_rate",
        "energy",
        "tree_depth",
        "n_steps",
        "reached_max_treedepth",
        "step_size",
    }
    for variable in statistics.data_vars.values():
        assert variable.dims == ("chain", "draw")
        assert variable.shape == (2, 200)
        assert np.isfinite(variable).all()
    assert np.all((statistics["acceptance_rate"] >= 0) & (statistics["acceptance_rate"] <= 1))
    assert np.all(statistics["step_size"] > 0)
    assert np.unique(statistics["step_size"].values[:, 0]).size == 2
    for group in first.children.values():
        for variable in group.data_vars.values():
            assert isinstance(variable.data, np.ndarray)


def test_dense_adaptation_learns_correlation_across_parameter_names(adaptation_metrics):
    location = np.array([-0.5, 0.7])
    covariance = np.array([[1.0, 0.85], [0.85, 1.5]])
    conditional_scale = np.sqrt(covariance[1, 1] - covariance[0, 1] ** 2)

    def density(data, first, second):
        conditional_mean = location[1] + covariance[0, 1] * (first - location[0])
        return normal(first, location[0], 1.0) + normal(second, conditional_mean, conditional_scale)

    model = Model(parameters={"first": Real(), "second": Real()}, log_density=density)
    result = sample(model, draws=400, warmup=250, chains=2, seed=23, chain_method="vectorized", mass_matrix="dense")
    values = np.stack([result["posterior"][name].values for name in ("first", "second")], axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(values.mean(axis=0), location, atol=0.2)
    np.testing.assert_allclose(np.cov(values.T), covariance, rtol=0.25, atol=0.1)
    assert result.attrs["mass_matrix"] == "dense"

    jax.effects_barrier()
    assert adaptation_metrics["diagonal"] == [False]
    matrices = adaptation_metrics["matrices"]
    assert len(matrices) == 2
    for matrix in matrices:
        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(matrix) > 0)
        assert matrix[0, 1] > 0.1
    assert not np.allclose(matrices[0], matrices[1])


@pytest.mark.parametrize("named_axes", [False, True])
def test_nuts_allows_parameters_named_after_sampler_diagnostics(named_axes):
    model = Model(
        parameters={"energy": Real(shape=(2,))},
        log_density=lambda data, energy: normal(energy, 0.0, 1.0),
        dims={"energy": ("feature",)} if named_axes else None,
        coords={"feature": ["first", "second"]} if named_axes else None,
    )
    results = sample(model, draws=10, warmup=50, chains=1, seed=19, generate=False)
    parameter_axis = "feature" if named_axes else "energy_dim_0"
    assert results["posterior"]["energy"].dims == ("chain", "draw", parameter_axis)
    assert results["posterior"]["energy"].shape == (1, 10, 2)
    assert results["sample_stats"]["energy"].dims == ("chain", "draw")
    assert results["sample_stats"]["energy"].shape == (1, 10)
    assert parameter_axis not in results["sample_stats"].coords
    assert np.isfinite(results["posterior"]["energy"]).all()
    assert np.isfinite(results["sample_stats"]["energy"]).all()
    if named_axes:
        np.testing.assert_array_equal(results["posterior"]["feature"], ["first", "second"])


@pytest.mark.parametrize(
    "chain_method,mass_matrix", [("sequential", "diagonal"), ("vectorized", "diagonal"), ("vectorized", "dense")]
)
def test_nuts_returns_positive_parameters_and_full_simplex_events(chain_method, mass_matrix, adaptation_metrics):
    model = Model(
        parameters={"scale": Positive(), "weights": Simplex((3,))},
        log_density=lambda data, scale, weights: (
            half_normal(scale, 1.0) + dirichlet(weights, jnp.array([2.0, 3.0, 4.0]))
        ),
        dims={"weights": ("category",)},
        coords={"category": ["a", "b", "c"]},
    )
    result = sample(
        model,
        draws=40,
        warmup=80,
        chains=1,
        seed=5,
        initial_values={"scale": 1.0, "weights": np.array([0.2, 0.3, 0.5])},
        chain_method=chain_method,
        mass_matrix=mass_matrix,
    )
    assert result["posterior"]["scale"].shape == (1, 40)
    assert (result["posterior"]["scale"].values > 0).all()
    assert result["posterior"]["weights"].shape == (1, 40, 3)
    assert (result["posterior"]["weights"].values > 0).all()
    np.testing.assert_allclose(result["posterior"]["weights"].sum("category"), 1.0, rtol=2e-6)
    np.testing.assert_array_equal(result["posterior"]["category"], ["a", "b", "c"])
    assert result.attrs["chain_method"] == chain_method
    assert result.attrs["mass_matrix"] == mass_matrix
    jax.effects_barrier()
    dimension = sum(int(np.prod(parameter.position_shape)) for parameter in model.parameters.values())
    assert dimension == 3
    assert adaptation_metrics["diagonal"] == [mass_matrix == "diagonal"]
    assert len(adaptation_metrics["matrices"]) == 1
    assert adaptation_metrics["matrices"][0].shape == (
        (dimension,) if mass_matrix == "diagonal" else (dimension, dimension)
    )


def test_default_initialization_and_keys_are_independent_per_chain(normal_model, nuts_calls):
    sample(normal_model, data=0.0, draws=3, warmup=5, chains=3, seed=12)
    call = nuts_calls[0]
    assert call["positions"]["location"].shape == (3,)
    assert len(np.unique(call["positions"]["location"])) == 3
    assert len(np.unique(call["keys"], axis=0)) == 3
    assert call["chain_method"] == "sequential"
    assert call["mass_matrix"] == "diagonal"
    assert call["chunk_size"] == 100
    assert call["progress"] is True


@pytest.mark.parametrize("chunk_size,progress", [(1, False), (3, True), (20, False)])
def test_chunk_and_progress_options_preserve_keys_initialization_and_result_groups(nuts_calls, chunk_size, progress):
    model, _ = _prepared_model()
    options = {"draws": 7, "warmup": 5, "chains": 2, "seed": 12, "batch_size": 2}
    expected = sample(model, **options)
    result = sample(model, **options, chunk_size=chunk_size, progress=progress)

    assert nuts_calls[1]["chunk_size"] == chunk_size
    assert nuts_calls[1]["progress"] is progress
    np.testing.assert_array_equal(nuts_calls[0]["keys"], nuts_calls[1]["keys"])
    for name in nuts_calls[0]["positions"]:
        np.testing.assert_array_equal(nuts_calls[0]["positions"][name], nuts_calls[1]["positions"][name])
    xr.testing.assert_identical(result, expected)
    for group in result.children.values():
        for variable in group.data_vars.values():
            assert isinstance(variable.data, np.ndarray)


@pytest.mark.parametrize("progress", [None, 0, 1, "true", [], {}, np.bool_(True)])
def test_progress_requires_a_bool_before_initialization(normal_model, nuts_calls, monkeypatch, progress):
    def forbidden_initialize(self, key):
        raise AssertionError("Progress validation must run before parameter initialization")

    monkeypatch.setattr(Model, "initialize_random", forbidden_initialize)
    with pytest.raises(TypeError, match="progress"):
        sample(normal_model, data=0.0, draws=2, warmup=3, chains=1, progress=progress)
    assert not nuts_calls


@pytest.mark.parametrize(
    "chain_method,mass_matrix", [("sequential", "diagonal"), ("vectorized", "diagonal"), ("vectorized", "dense")]
)
def test_seeded_nuts_is_unchanged_by_chunk_boundaries_and_progress(normal_model, chain_method, mass_matrix):
    options = {
        "data": 0.0,
        "draws": 7,
        "warmup": 60,
        "chains": 2,
        "seed": 19,
        "chain_method": chain_method,
        "mass_matrix": mass_matrix,
    }
    expected = sample(normal_model, **options, chunk_size=20, progress=False)
    for chunk_size, progress in ((1, True), (3, False)):
        result = sample(normal_model, **options, chunk_size=chunk_size, progress=progress)
        xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("chain_method", ["sequential", "vectorized"])
@pytest.mark.parametrize("progress", [False, True])
def test_nuts_chunks_preserve_draw_keys_warmup_counts_and_bounded_host_transfers(
    monkeypatch, adaptation_metrics, progress_contexts, capsys, chain_method, progress
):
    draw_keys = []
    chunk_shapes = []
    make_sampler = nuts.blackjax.nuts.differentiable
    device_get = jax.device_get

    def record_sampler(*args, **kwargs):
        sampler = make_sampler(*args, **kwargs)

        def step(key, state):
            jax.debug.callback(lambda words: draw_keys.append(tuple(np.asarray(words))), jax.random.key_data(key))
            return sampler.step(key, state)

        return sampler._replace(step=step)

    def record_transfer(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict) and "lp" in value[1]:
            chunk_shapes.append(value[1]["lp"].shape)
            if progress:
                assert not progress_contexts[-1].closed
        return device_get(value)

    monkeypatch.setattr(nuts.blackjax.nuts, "differentiable", record_sampler)
    monkeypatch.setattr(nuts.jax, "device_get", record_transfer)
    keys = jax.random.split(jax.random.key(19), 2)
    history, continuation = nuts._sample_nuts(
        lambda position: normal(position["location"], 0.0, 1.0),
        {"location": jnp.array([-0.1, 0.1])},
        keys,
        draws=7,
        warmup=40,
        target_accept=0.8,
        max_tree_depth=6,
        chain_method=chain_method,
        mass_matrix="diagonal",
        chunk_size=3,
        progress=progress,
    )
    jax.effects_barrier()

    assert adaptation_metrics["steps"] == [40, 40]
    expected_keys = jax.vmap(
        lambda key: jax.vmap(lambda draw: jax.random.fold_in(jax.random.split(key)[1], draw))(jnp.arange(7))
    )(keys)
    expected_words = np.asarray(jax.random.key_data(expected_keys)).reshape(-1, 2)
    assert sorted(draw_keys) == sorted(tuple(words) for words in expected_words)
    expected_shapes = [(3,), (3,), (1,)] * 2 if chain_method == "sequential" else [(2, 3), (2, 3), (2, 1)]
    assert chunk_shapes == expected_shapes
    for value in jax.tree.leaves(history):
        assert isinstance(value, np.ndarray)
        assert value.shape == (2, 7)
        assert value.flags.writeable
    assert continuation.completed_draws == 7
    assert continuation.step_size.shape == (2,)
    np.testing.assert_array_equal(
        jax.random.key_data(continuation.sampling_keys),
        jax.random.key_data(jax.vmap(lambda key: jax.random.split(key)[1])(keys)),
    )
    if progress:
        expected_steps = [40, 3, 3, 1] * (2 if chain_method == "sequential" else 1)
        assert [state.n_steps for state in progress_contexts] == expected_steps
        assert all(state.closed for state in progress_contexts)
    else:
        assert not progress_contexts
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err == ""


def test_native_progress_restores_scan_after_transfer_failure_and_repeated_calls(
    normal_model, progress_contexts, monkeypatch
):
    original_scan = jax.lax.scan
    device_get = jax.device_get
    options = {"data": 0.0, "draws": 7, "warmup": 40, "chains": 1, "chunk_size": 3, "seed": 19}

    def fail_transfer(value):
        result = device_get(value)
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict) and "lp" in value[1]:
            raise RuntimeError("Simulated chunk transfer failure")
        return result

    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(nuts.jax, "device_get", fail_transfer)
        with pytest.raises(RuntimeError, match="Simulated chunk transfer failure"):
            sample(normal_model, **options)
    assert jax.lax.scan is original_scan
    assert len(progress_contexts) == 2
    assert all(state.closed for state in progress_contexts)

    first = sample(normal_model, **options)
    repeated = sample(normal_model, **options)
    xr.testing.assert_identical(first, repeated)
    assert jax.lax.scan is original_scan
    assert len(progress_contexts) == 10
    assert all(state.closed for state in progress_contexts)


@pytest.mark.parametrize("chain_method", ["sequential", "vectorized", "parallel"])
def test_chain_methods_forward_options_without_changing_initialization_or_random_keys(nuts_calls, chain_method):
    model, _ = _prepared_model()
    chains = min(3, jax.local_device_count()) if chain_method == "parallel" else 3
    options = {"draws": 3, "warmup": 5, "chains": chains, "seed": 12, "batch_size": 2}
    expected = sample(model, **options)
    result = sample(model, **options, chain_method=chain_method)

    assert nuts_calls[1]["chain_method"] == chain_method
    assert result.attrs["chain_method"] == chain_method
    np.testing.assert_array_equal(nuts_calls[0]["keys"], nuts_calls[1]["keys"])
    for name in nuts_calls[0]["positions"]:
        np.testing.assert_array_equal(nuts_calls[0]["positions"][name], nuts_calls[1]["positions"][name])
    for group in expected.children:
        xr.testing.assert_identical(result[group], expected[group])


@pytest.mark.parametrize("mass_matrix", ["diagonal", "dense"])
def test_mass_matrix_options_preserve_keys_generation_and_labels(nuts_calls, mass_matrix):
    model, _ = _prepared_model()
    options = {"draws": 3, "warmup": 5, "chains": 2, "seed": 12, "batch_size": 2}
    expected = sample(model, **options)
    result = sample(model, **options, mass_matrix=mass_matrix)

    assert nuts_calls[0]["mass_matrix"] == "diagonal"
    assert nuts_calls[1]["mass_matrix"] == mass_matrix
    assert result.attrs["mass_matrix"] == mass_matrix
    np.testing.assert_array_equal(nuts_calls[0]["keys"], nuts_calls[1]["keys"])
    for name in nuts_calls[0]["positions"]:
        np.testing.assert_array_equal(nuts_calls[0]["positions"][name], nuts_calls[1]["positions"][name])
    for group in expected.children:
        xr.testing.assert_identical(result[group], expected[group])


@pytest.mark.parametrize("mass_matrix", [None, True, 1, [], {}, "", "Dense", "block"])
def test_invalid_mass_matrix_options_are_rejected_before_initialization(
    normal_model, nuts_calls, monkeypatch, mass_matrix
):
    def forbidden_initialize(self, key):
        raise AssertionError("Mass matrix validation must run before parameter initialization")

    monkeypatch.setattr(Model, "initialize_random", forbidden_initialize)
    with pytest.raises(ValueError, match="mass_matrix"):
        sample(normal_model, data=0.0, draws=2, warmup=3, chains=1, mass_matrix=mass_matrix)
    assert not nuts_calls


@pytest.mark.parametrize("chain_method", [None, True, 1, [], {}, "", "automatic", "Parallel"])
def test_invalid_chain_methods_are_rejected_before_sampling(normal_model, nuts_calls, chain_method):
    with pytest.raises(ValueError, match="chain_method"):
        sample(normal_model, data=0.0, draws=2, warmup=3, chains=1, chain_method=chain_method)
    assert not nuts_calls


def test_parallel_sampling_requires_enough_devices_before_evaluating_model(monkeypatch, nuts_calls):
    def forbidden_density(data, location):
        raise AssertionError("Device validation must run before model evaluation")

    def forbidden_initialize(self, key):
        raise AssertionError("Device validation must run before parameter initialization")

    model = Model(parameters={"location": Real()}, log_density=forbidden_density)
    devices = jax.local_devices()
    configuration = {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "JAX_NUM_CPU_DEVICES", "XLA_FLAGS")}
    monkeypatch.setattr(Model, "initialize_random", forbidden_initialize)

    with pytest.raises(ValueError, match=r"(?i)device"):
        sample(model, draws=2, warmup=3, chains=len(devices) + 1, chain_method="parallel")

    assert not nuts_calls
    assert jax.local_devices() == devices
    assert {name: os.environ.get(name) for name in configuration} == configuration


def test_parallel_nuts_on_two_cpu_devices_preserves_targets_generation_and_labels():
    script = dedent(
        """
        from contextlib import contextmanager

        import blackjax
        import jax
        import jax.numpy as jnp
        import numpy as np
        import xarray as xr
        import mmmjax.sampling as sampling
        from mmmjax import Model, Real, Simplex, continue_sampling, normal, sample

        assert jax.local_device_count() == 2
        assert all(device.platform == "cpu" for device in jax.local_devices())
        run_nuts = sampling._sample_nuts
        device_get = jax.device_get
        progress_bar = blackjax.progress_bar
        progress_contexts = []
        parallel_chunks = []
        active_method = None

        @contextmanager
        def check_progress(*args, **kwargs):
            original_scan = jax.lax.scan
            try:
                with progress_bar(*args, **kwargs) as state:
                    progress_contexts.append(state)
                    yield state
                    assert not state.closed
                    assert state.n_steps > 0
                    assert state.current_step == state.n_steps - 1
                    if active_method == "parallel":
                        assert "per device" in state.label.replace("-", " ").lower()
            finally:
                assert jax.lax.scan is original_scan

        def check_transfer(value):
            if (
                active_method == "parallel" and isinstance(value, tuple) and len(value) == 2
                and isinstance(value[1], dict) and "lp" in value[1]
            ):
                chains = value[1]["lp"].shape[0]
                for leaf in jax.tree.leaves(value):
                    assert leaf.sharding.device_set == set(jax.local_devices()[:chains])
                parallel_chunks.append(value[1]["lp"].shape)
            return device_get(value)

        def check_execution(*args, **kwargs):
            global active_method
            active_method = kwargs["chain_method"]
            start = len(parallel_chunks)
            context_start = len(progress_contexts)
            history, continuation = run_nuts(*args, **kwargs)
            active_method = None
            for value in jax.tree.leaves(history):
                assert isinstance(value, np.ndarray)
                assert value.flags.writeable
            assert continuation.completed_draws == kwargs["draws"]
            states = progress_contexts[context_start:]
            if kwargs["progress"]:
                steps = [kwargs["warmup"]] + [
                    min(kwargs["chunk_size"], kwargs["draws"] - offset)
                    for offset in range(0, kwargs["draws"], kwargs["chunk_size"])
                ]
                if kwargs["chain_method"] == "sequential":
                    steps *= next(iter(args[1].values())).shape[0]
                assert [state.n_steps for state in states] == steps
                assert all(state.closed for state in states)
            else:
                assert not states
            if kwargs["chain_method"] == "parallel":
                chains = next(iter(args[1].values())).shape[0]
                assert parallel_chunks[start:] == [
                    (chains, min(kwargs["chunk_size"], kwargs["draws"] - offset))
                    for offset in range(0, kwargs["draws"], kwargs["chunk_size"])
                ]
            return history, continuation

        jax.device_get = check_transfer
        blackjax.progress_bar = check_progress
        sampling._sample_nuts = check_execution
        observed_location = jax.device_put(jnp.array(0.0), jax.local_devices()[1])
        model = Model(
            parameters={"location": Real()},
            log_density=lambda data, location: normal(location, data, 1.0),
            generated_quantities=lambda key, data, location: {
                "predictive": {"prediction": location + 0.2 * jax.random.normal(key)},
                "location_copy": location,
            },
        )
        options = dict(data=observed_location, draws=160, warmup=120, chains=2, seed=19, batch_size=17, chunk_size=37)
        generation_key = jax.random.split(jax.random.key(19), 4)[2]
        keys = jax.vmap(lambda key: jax.vmap(lambda draw: jax.random.fold_in(key, draw))(jnp.arange(160)))(
            jax.random.split(generation_key, 2)
        )
        noise = jax.jit(jax.vmap(jax.vmap(jax.random.normal)))(keys)

        parallel = None
        for method in ("parallel", "vectorized", "sequential"):
            result = sample(model, **options, chain_method=method)
            assert result.attrs["chain_method"] == method
            assert set(result.children) == {
                "posterior", "sample_stats", "posterior_predictive", "generated_quantities", "sampling_state"
            }
            location = result["posterior"]["location"]
            assert location.dims == ("chain", "draw")
            assert location.shape == (2, 160)
            assert abs(float(location.mean())) < 0.4
            assert 0.55 < float(location.std()) < 1.5
            assert not np.array_equal(location.values[0], location.values[1])
            assert result["sampling_state"].attrs["completed_draws"] == 160
            for name, group in result.children.items():
                # The stored state keeps its own axes, but its leaves make the same storage promises.
                sampled = name != "sampling_state"
                if sampled:
                    np.testing.assert_array_equal(group.coords["chain"], [0, 1])
                    np.testing.assert_array_equal(group.coords["draw"], np.arange(160))
                for value in group.data_vars.values():
                    assert value.shape == (2, 160) or not sampled
                    assert isinstance(value.data, np.ndarray)
                    assert value.data.flags.writeable
                    assert np.isfinite(value).all()
            expected_lp = -0.5 * (location.values**2 + np.log(2 * np.pi))
            np.testing.assert_allclose(result["sample_stats"]["lp"], expected_lp, rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(result["generated_quantities"]["location_copy"], location)
            np.testing.assert_allclose(
                result["posterior_predictive"]["prediction"],
                location.values + 0.2 * np.asarray(noise),
                rtol=2e-5, atol=2e-5,
            )
            if method == "parallel":
                parallel = result

        repeated = sample(model, **(options | {"chunk_size": 200}), chain_method="parallel", progress=False)
        xr.testing.assert_identical(parallel, repeated)
        single_options = options | {
            "draws": 8,
            "warmup": 60,
            "chains": 1,
            "chunk_size": 3,
            "initial_values": {"location": jax.device_put(jnp.array(0.25), jax.local_devices()[1])},
        }
        single = sample(model, **single_options, chain_method="parallel", generate=False)
        assert single["posterior"]["location"].shape == (1, 8)
        assert set(single.children) == {"posterior", "sample_stats", "sampling_state"}
        np.testing.assert_allclose(
            single["sample_stats"]["lp"],
            -0.5 * (single["posterior"]["location"].values**2 + np.log(2 * np.pi)),
            rtol=2e-5, atol=2e-5,
        )
        original_mesh = jax.sharding.get_abstract_mesh()
        outer_mesh = jax.make_mesh(
            (2,), ("user_axis",), axis_types=(jax.sharding.AxisType.Explicit,),
        )
        nested_options = single_options | {"data": 0.0, "initial_values": {"location": 0.25}}
        with jax.set_mesh(outer_mesh):
            caller_mesh = jax.sharding.get_abstract_mesh()
            nested = sample(model, **nested_options, chain_method="parallel")
            assert jax.sharding.get_abstract_mesh() == caller_mesh
        assert jax.sharding.get_abstract_mesh() == original_mesh
        xr.testing.assert_allclose(nested["posterior"], single["posterior"])
        assert nested["posterior_predictive"]["prediction"].shape == (1, 8)
        np.testing.assert_array_equal(
            nested["generated_quantities"]["location_copy"], nested["posterior"]["location"],
        )
        simplex_model = Model(
            parameters={"location": Real(), "weights": Simplex((1,))},
            log_density=lambda data, location, weights: normal(location, 0.0, 1.0) + 0.0 * weights.sum(),
        )
        simplex = sample(
            simplex_model, chains=2, draws=8, warmup=40, seed=8, chain_method="parallel",
            initial_values={"location": 0.0, "weights": jnp.ones(1)}, generate=False, chunk_size=3,
        )
        assert simplex["posterior"]["weights"].shape == (2, 8, 1)
        np.testing.assert_array_equal(simplex["posterior"]["weights"], np.ones((2, 8, 1)))
        assert np.isfinite(simplex["posterior"]["location"]).all()
        assert np.isfinite(simplex["sample_stats"]["lp"]).all()
        dense_model = Model(
            parameters={"intercept": Real(), "coefficients": Real((2,))},
            log_density=lambda data, intercept, coefficients: (
                normal(intercept, 0.0, 1.0) + normal(coefficients, intercept, 1.0)
            ),
            generated_quantities=lambda key, data, intercept, coefficients: {"response": intercept + coefficients},
        )
        dense = sample(
            dense_model, chains=2, draws=12, warmup=100, seed=17,
            chain_method="parallel", mass_matrix="dense", batch_size=5, chunk_size=5,
        )
        assert dense.attrs["mass_matrix"] == "dense"
        assert dense["posterior"]["coefficients"].shape == (2, 12, 2)
        intercept = dense["posterior"]["intercept"].values
        coefficients = dense["posterior"]["coefficients"].values
        expected_lp = -0.5 * (
            intercept**2 + ((coefficients - intercept[..., None])**2).sum(-1) + 3 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(dense["sample_stats"]["lp"], expected_lp, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            dense["generated_quantities"]["response"], intercept[..., None] + coefficients,
            rtol=2e-5, atol=2e-5,
        )

        sampling._sample_nuts = run_nuts
        continuation_options = dict(
            data=observed_location, warmup=60, chains=2, seed=31, chain_method="parallel",
            mass_matrix="dense", chunk_size=3, batch_size=3, progress=False,
        )
        full = sample(model, draws=9, **continuation_options)
        first = sample(model, draws=4, **continuation_options)
        context_start = len(progress_contexts)
        transfer_start = len(parallel_chunks)
        active_method = "parallel"
        continued = continue_sampling(
            model, first, data=observed_location, draws=5, chunk_size=3, batch_size=3, progress=True
        )
        active_method = None
        xr.testing.assert_identical(continued, full)
        assert first["sampling_state"].attrs["completed_draws"] == 4
        assert continued["sampling_state"].attrs["completed_draws"] == 9
        assert first["posterior"].sizes["draw"] == 4
        assert [state.n_steps for state in progress_contexts[context_start:]] == [3, 2]
        assert parallel_chunks[transfer_start:] == [(2, 3), (2, 2)]
        assert jax.sharding.get_abstract_mesh() == original_mesh
        assert jax.local_device_count() == 2
        print("Two-device chain checks passed")
        """
    )
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="2")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, env=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Two-device chain checks passed" in result.stdout


def test_explicit_constrained_initial_values_are_replicated_and_sampler_options_forwarded(normal_model, nuts_calls):
    initial = {"location": np.array(1.5)}
    result = sample(
        normal_model,
        data=3.0,
        draws=4,
        warmup=7,
        chains=3,
        seed=11,
        target_accept=0.9,
        max_tree_depth=6,
        initial_values=initial,
    )
    np.testing.assert_array_equal(nuts_calls[0]["positions"]["location"], [1.5, 1.5, 1.5])
    assert nuts_calls[0]["draws"] == 4
    assert nuts_calls[0]["warmup"] == 7
    assert nuts_calls[0]["target_accept"] == 0.9
    assert nuts_calls[0]["max_tree_depth"] == 6
    np.testing.assert_array_equal(result["posterior"]["location"], np.full((3, 4), 1.5))
    np.testing.assert_allclose(result["sample_stats"]["lp"], normal(1.5, 3.0, 1.0), rtol=1e-6)
    assert initial["location"] == 1.5


def test_prepared_data_and_selected_generated_outputs_are_collected_automatically(nuts_calls):
    model, data = _prepared_model()
    result = sample(model, draws=4, warmup=5, chains=2, seed=7, initial_values={"intercept": 150.0})
    assert set(result.children) == {
        "posterior",
        "sample_stats",
        "posterior_predictive",
        "log_likelihood",
        "generated_quantities",
        "observed_data",
        "constant_data",
        "sampling_state",
    }
    for group, name in (
        ("posterior_predictive", "prediction"),
        ("log_likelihood", "pointwise"),
        ("generated_quantities", "mean"),
    ):
        assert result[group][name].dims == ("chain", "draw", "time")
        assert result[group][name].shape == (2, 4, 3)
        np.testing.assert_array_equal(result[group]["time"], [10, 11, 12])
    np.testing.assert_array_equal(result["observed_data"]["outcome"], data.arrays["outcome"])
    np.testing.assert_array_equal(result["constant_data"]["controls"], data.arrays["controls"])
    np.testing.assert_array_equal(result["generated_quantities"]["mean"], np.full((2, 4, 3), 150.0))
    expected = np.broadcast_to(normal_logpdf(data.arrays["outcome"], 150.0, 10.0), (2, 4, 3))
    np.testing.assert_allclose(result["log_likelihood"]["pointwise"], expected, rtol=1e-6)
    predictions = result["posterior_predictive"]["prediction"].values
    assert not np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[:, 0], predictions[:, 1])


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_sampling_collects_log_prior_without_sampling_jacobian(nuts_calls, batch_size):
    prior = Prior(lognormal, location=0.0, scale=1.0)

    def density(data, scale):
        return prior(scale) + normal(data, scale, 0.5)

    def generate(key, data, scale):
        return {"log_likelihood": {"observation": normal(data, scale, 0.5)}, "log_prior": {"lp_scale": prior(scale)}}

    model = Model(parameters={"scale": Positive()}, log_density=density, generated_quantities=generate)
    result = sample(model, data=1.5, draws=3, warmup=1, chains=2, initial_values={"scale": 2.0}, batch_size=batch_size)
    scale = result["posterior"]["scale"].values
    expected_prior = -0.5 * np.log(scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    expected_likelihood = -0.5 * ((1.5 - scale) / 0.5) ** 2 - np.log(0.5) - 0.5 * np.log(2 * np.pi)
    assert set(result.children) == {"posterior", "sample_stats", "log_likelihood", "log_prior", "sampling_state"}
    prior_name = "lp_scale"
    assert result["log_prior"][prior_name].dims == ("chain", "draw")
    np.testing.assert_allclose(result["log_prior"][prior_name], expected_prior, rtol=2e-6)
    np.testing.assert_allclose(result["log_likelihood"]["observation"], expected_likelihood, rtol=2e-6)
    np.testing.assert_allclose(
        result["sample_stats"]["lp"], expected_prior + expected_likelihood + np.log(scale), rtol=2e-6
    )
    assert not np.allclose(result["log_prior"][prior_name], result["sample_stats"]["lp"])
    assert len(nuts_calls) == 1


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_sampling_batches_preserve_seeded_draws_generation_and_labels(nuts_calls, batch_size):
    model, _ = _prepared_model()
    options = {"draws": 5, "warmup": 3, "chains": 2, "seed": 17}
    expected = sample(model, **options, batch_size=64)
    result = sample(model, **options, batch_size=batch_size)

    xr.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(nuts_calls[0]["keys"], nuts_calls[1]["keys"])
    np.testing.assert_array_equal(nuts_calls[0]["positions"]["intercept"], nuts_calls[1]["positions"]["intercept"])
    for group in ("posterior", "sample_stats", "generated_quantities", "posterior_predictive", "log_likelihood"):
        for value in result[group].data_vars.values():
            assert value.values.flags.writeable

    generation_key = jax.random.split(jax.random.key(17), 4)[2]
    keys = jax.vmap(lambda key: jax.vmap(lambda draw: jax.random.fold_in(key, draw))(jnp.arange(5)))(
        jax.random.split(generation_key, 2)
    )
    posterior = {"intercept": jnp.asarray(result["posterior"]["intercept"].values)}
    reference = jax.jit(jax.vmap(jax.vmap(lambda key, values: model.generate_quantities(key, values, model.data))))(
        keys, posterior
    )
    direct = {
        "prediction": reference["predictive"]["prediction"],
        "pointwise": reference["log_likelihood"]["pointwise"],
        "mean": reference["mean"],
    }
    for group, name in (
        ("posterior_predictive", "prediction"),
        ("log_likelihood", "pointwise"),
        ("generated_quantities", "mean"),
    ):
        np.testing.assert_allclose(result[group][name], direct[name], rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(result[group]["time"], [10, 11, 12])
        assert result[group][name].dims == ("chain", "draw", "time")
        assert isinstance(result[group][name].data, np.ndarray)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_sampling_batches_keep_constrained_event_shapes_and_exact_draw_keys(nuts_calls, batch_size):
    model = Model(
        parameters={"scale": Positive(), "weights": Simplex((3,))},
        log_density=lambda data, scale, weights: half_normal(scale, 1.0) + dirichlet(weights, jnp.ones(3)),
        generated_quantities=lambda key, data, scale: {
            "key_words": jax.random.key_data(key),
            "squared_scale": scale**2,
        },
    )
    result = sample(model, draws=5, warmup=3, chains=2, seed=11, batch_size=batch_size)
    positions = jax.tree.map(
        lambda value: jnp.repeat(jnp.asarray(value)[:, None], 5, axis=1), nuts_calls[0]["positions"]
    )
    reference = jax.jit(jax.vmap(jax.vmap(model.constrain)))(positions)

    for name, values in reference.items():
        np.testing.assert_allclose(result["posterior"][name], values, rtol=2e-6, atol=2e-6)
        assert isinstance(result["posterior"][name].data, np.ndarray)
    assert result["posterior"]["weights"].shape == (2, 5, 3)
    np.testing.assert_allclose(result["posterior"]["weights"].sum("weights_dim_0"), 1.0, rtol=2e-6)

    generation_key = jax.random.split(jax.random.key(11), 4)[2]
    keys = jax.vmap(lambda key: jax.vmap(lambda draw: jax.random.fold_in(key, draw))(jnp.arange(5)))(
        jax.random.split(generation_key, 2)
    )
    np.testing.assert_array_equal(result["generated_quantities"]["key_words"], jax.random.key_data(keys))
    assert result["generated_quantities"]["key_words"].dtype == np.uint32


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_evaluate_draws_only_transfers_bounded_batches_and_preserves_event_axes(monkeypatch, batch_size):
    arguments = {
        "values": np.arange(30, dtype=np.float32).reshape(2, 5, 3),
        "empty": np.empty((2, 5, 0), dtype=np.float32),
    }
    original = jax.tree.map(np.copy, arguments)
    calls = []
    compile_function = jax.jit

    def record_compile(function):
        compiled = compile_function(function)

        def evaluate(batch):
            assert all(isinstance(value, np.ndarray) for value in batch.values())
            size = batch["values"].shape[0]
            assert 0 < size <= batch_size
            assert batch["empty"].shape == (size, 0)
            calls.append(size)
            return compiled(batch)

        return evaluate

    monkeypatch.setattr(sampling.jax, "jit", record_compile)
    result = sampling._evaluate_draws(
        lambda values: {"values": values["values"] * 2, "empty": values["empty"], "positive": values["values"] > 0},
        arguments,
        sample_shape=(2, 5),
        batch_size=batch_size,
    )

    assert calls == [min(batch_size, 10 - start) for start in range(0, 10, batch_size)]
    np.testing.assert_array_equal(result["values"], original["values"] * 2)
    np.testing.assert_array_equal(result["positive"], original["values"] > 0)
    assert result["values"].dtype == np.float32
    assert result["positive"].dtype == np.bool_
    assert result["empty"].shape == (2, 5, 0)
    for name in ("values", "empty"):
        assert isinstance(result[name], np.ndarray)
        assert not np.shares_memory(result[name], arguments[name])
        np.testing.assert_array_equal(arguments[name], original[name])
    result["values"][...] = -1
    np.testing.assert_array_equal(arguments["values"], original["values"])


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_generation_can_be_disabled_without_executing_the_callback(nuts_calls, batch_size):
    def forbidden_generate(key, data, location):
        raise AssertionError("Generation was disabled")

    model = Model(
        parameters={"location": Real()},
        log_density=lambda data, location: normal(location, 0.0, 1.0),
        generated_quantities=forbidden_generate,
    )
    result = sample(model, draws=3, warmup=3, chains=2, generate=False, batch_size=batch_size)
    assert set(result.children) == {"posterior", "sample_stats", "sampling_state"}


@pytest.mark.parametrize("generate", [False, True])
def test_auxiliary_inputs_keep_evaluated_values_and_experiment_labels(nuts_calls, generate):
    _, data = _prepared_model()
    inputs = xr.Dataset(
        {
            "lift": ("experiment", np.array([0.1, 0.2, 0.3], dtype=np.float64)),
            "uncertainty": ("experiment", [0.5, 0.25, 0.125]),
            "baseline": 2.0,
        },
        coords={"experiment": ["north", "south", "national"]},
    )
    model = Model(
        parameters={"effect": Real(dims="experiment")},
        log_density=lambda lift, expected_lift, uncertainty, effect: (
            normal(lift, expected_lift, uncertainty) + normal(effect, 0.0, 1.0)
        ),
        generated_quantities=lambda key, lift, expected_lift, uncertainty: {
            "predictive": {"lift_copy": lift},
            "log_likelihood": {"pointwise": normal_logpdf(lift, expected_lift, uncertainty)},
            "expected_lift": expected_lift,
        },
        data=Data(data, inputs=inputs, scaling=fit_data_scaling(data, scale_outcome=True)),
        transformed_parameters=lambda effect, baseline: {"expected_lift": effect * baseline},
        generated_dims={"expected_lift": ("experiment",), "pointwise": ("experiment",)},
    )
    effect = np.array([1.0, 2.0, 3.0])
    result = sample(model, draws=2, warmup=1, chains=1, initial_values={"effect": effect}, generate=generate)

    assert result["posterior"]["effect"].dims == ("chain", "draw", "experiment")
    np.testing.assert_array_equal(result["posterior"]["experiment"], inputs.experiment)
    assert set(result["observed_data"].data_vars) == {"outcome"}
    np.testing.assert_allclose(result["observed_data"]["outcome"], model.data.values["outcome"])
    for name in inputs.data_vars:
        actual = result["constant_data"][name]
        assert actual.dims == inputs[name].dims
        assert actual.dtype == model.data.values[name].dtype
        np.testing.assert_array_equal(actual, model.data.values[name])
    np.testing.assert_array_equal(result["constant_data"]["time"], data.time_values)
    if generate:
        assert result["posterior_predictive"]["lift_copy"].dims == ("chain", "draw", "experiment")
        assert "time" not in result["posterior_predictive"].dims
        np.testing.assert_allclose(result["generated_quantities"]["expected_lift"], [[effect * 2, effect * 2]])
        expected = normal_logpdf(model.data.values["lift"], effect * 2, model.data.values["uncertainty"])
        np.testing.assert_allclose(result["log_likelihood"]["pointwise"], [[expected, expected]])
    else:
        assert "generated_quantities" not in result.children
        assert "posterior_predictive" not in result.children


def test_unspecified_vector_axes_receive_parameter_specific_names(nuts_calls):
    model = Model(
        parameters={"coefficient": Real((2,))}, log_density=lambda data, coefficient: normal(coefficient, 0.0, 1.0)
    )
    result = sample(model, draws=3, warmup=5, chains=1)
    variable = result["posterior"]["coefficient"]
    assert variable.dims[:2] == ("chain", "draw")
    assert variable.dims[2].startswith("coefficient")
    assert variable.shape == (1, 3, 2)


def test_generated_parameter_aliases_keep_explicit_and_neutral_axes(nuts_calls):
    model = Model(
        parameters={"coefficient": Real((2,)), "other": Real((2,))},
        log_density=lambda data, coefficient, other: normal(coefficient, 0.0, 1.0) + normal(other, 0.0, 1.0),
        generated_quantities=lambda key, data, coefficient, other: {
            "copy": coefficient,
            "other_copy": other,
            "sum": coefficient + other,
        },
        dims={"coefficient": ("feature",)},
        coords={"feature": ["first", "second"]},
    )
    result = sample(model, draws=2, warmup=3, chains=1)["generated_quantities"]
    assert result["copy"].dims == ("chain", "draw", "feature")
    assert result["other_copy"].dims == ("chain", "draw", "other_dim_0")
    assert result["sum"].dims == ("chain", "draw", "sum_dim_0")
    np.testing.assert_array_equal(result["feature"], ["first", "second"])


def test_custom_generation_does_not_infer_axes_from_names_or_equal_lengths(nuts_calls):
    data = prepare_data(
        pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0], "a": [5.0, 6.0], "b": [7.0, 8.0]}),
        time="week",
        outcome="sales",
        controls=["a", "b"],
        treatments=["a", "b"],
    )

    def generate(key, outcome, controls, treatments, coefficient):
        return {
            "controls_copy": controls,
            "treatments_copy": treatments,
            "controls": controls.T,
            "outcome": outcome[::-1],
            "mean": coefficient + outcome,
            "predictive": {"prediction": jnp.array([1.0, 2.0, 3.0])},
        }

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=lambda coefficient: normal(coefficient, 0.0, 1.0),
        generated_quantities=generate,
        data=data,
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    generated = result["generated_quantities"]
    assert result["posterior"]["coefficient"].dims == ("chain", "draw", "coefficient_dim_0")
    assert generated["controls_copy"].dims == ("chain", "draw", "time", "control")
    assert generated["treatments_copy"].dims == ("chain", "draw", "time", "treatment")
    assert generated["controls"].dims == ("chain", "draw", "controls_dim_0", "controls_dim_1")
    assert generated["outcome"].dims == ("chain", "draw", "outcome_dim_0")
    assert generated["mean"].dims == ("chain", "draw", "mean_dim_0")
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "prediction_dim_0")


def test_generated_dimension_overrides_take_precedence(nuts_calls):
    data = prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales")
    model = Model(
        parameters={"intercept": Real()},
        log_density=lambda intercept: normal(intercept, 0.0, 1.0),
        generated_quantities=lambda key, outcome, intercept: {
            "copy": outcome,
            "predictive": {"prediction": outcome + intercept},
        },
        data=data,
        generated_dims={"copy": ("custom",), "prediction": ("custom",)},
        coords={"custom": ["early", "late"]},
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    assert result["generated_quantities"]["copy"].dims == ("chain", "draw", "custom")
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "custom")
    np.testing.assert_array_equal(result["posterior_predictive"]["custom"], ["early", "late"])


def test_predictive_outputs_use_declared_observation_order(nuts_calls):
    data = prepare_data(pd.DataFrame({"week": [1, 2], "sales": [3.0, 4.0]}), time="week", outcome="sales")
    model = Model(
        parameters={"levels": Real((2,))},
        log_density=lambda levels: normal(levels, 0.0, 1.0),
        generated_quantities=lambda key, levels: {"predictive": {"prediction": levels}, "copy": levels},
        data=data,
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time")
    assert result["generated_quantities"]["copy"].dims == ("chain", "draw", "levels_dim_0")


def test_missing_labeled_generated_output_is_reported_before_sampling(nuts_calls):
    model = Model(
        parameters={"location": Real()},
        log_density=lambda data, location: normal(location, 0.0, 1.0),
        generated_quantities=lambda key, data, location: {"mean": location},
        generated_dims={"prediction": ("scenario",)},
    )
    with pytest.raises(ValueError, match="prediction"):
        sample(model, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize("group_specific", [False, True])
def test_explicit_media_and_fourier_parameters_retain_declared_axes(nuts_calls, group_specific):
    data = prepare_data(
        pd.DataFrame(
            {
                "week": [1, 1, 2, 2, 3, 3],
                "region": ["west", "east"] * 3,
                "sales": [10.0, 15.0, 12.0, 18.0, 13.0, 17.0],
                "video": [100.0, 200.0, 120.0, 160.0, 180.0, 140.0],
                "search": [60.0, 80.0, 90.0, 70.0, 50.0, 90.0],
            }
        ),
        time="week",
        groups=["region"],
        outcome="sales",
        media=["video", "search"],
        channels=["Video", "Search"],
        media_history=pd.DataFrame(
            {"week": [0, 0], "region": ["west", "east"], "video": [80.0, 120.0], "search": [40.0, 50.0]}
        ),
    )

    def density(
        outcome,
        paid_total,
        annual,
        paid_coefficient,
        paid_retention,
        paid_half_saturation,
        paid_slope,
        annual_coefficients,
    ):
        target = normal(outcome, paid_total + annual, 10.0)
        target += half_normal(paid_coefficient, 1.0)
        target += beta(paid_retention, 2.0, 3.0)
        target += lognormal(paid_half_saturation, 0.0, 0.5)
        target += lognormal(paid_slope, 0.0, 0.5)
        target += normal(annual_coefficients, 0.0, 0.5)
        return target

    def generate(key, outcome, media, paid, paid_total, annual, annual_coefficients, paid_retention):
        mean = paid_total + annual
        return {
            "predictive": {"prediction": normal_rng(key, mean, 10.0)},
            "log_likelihood": {"pointwise": normal_logpdf(outcome, mean, 10.0)},
            "contribution": paid,
            "total": paid_total,
            "seasonal": annual,
            "weights": annual_coefficients,
            "retention": paid_retention,
            "exposure": media,
        }

    def transformed(
        media,
        time,
        paid_coefficient,
        paid_retention,
        paid_half_saturation,
        paid_slope,
        annual_coefficients,
    ):
        carried = geometric_adstock(media, alpha=paid_retention, max_lag=1)
        response = hill_saturation(carried, half_saturation=paid_half_saturation, slope=paid_slope)[-time.shape[0] :]
        paid = response * paid_coefficient
        annual = fourier_features(time, period=52, order=2) @ annual_coefficients
        if not group_specific:
            annual = jnp.broadcast_to(annual[:, None], paid.shape[:2])
        return {"paid": paid, "paid_total": paid.sum(-1), "annual": annual}

    coefficient_axes = ("group", "channel") if group_specific else ("channel",)
    annual_axes = ("annual_mode", "group") if group_specific else ("annual_mode",)
    model = Model(
        parameters={
            "paid_coefficient": Positive(dims=coefficient_axes),
            "paid_retention": Interval(0.0, 1.0, dims="channel"),
            "paid_half_saturation": Positive(dims="channel"),
            "paid_slope": Positive(dims="channel"),
            "annual_coefficients": Real(dims=annual_axes),
        },
        log_density=density,
        generated_quantities=generate,
        data=data,
        transformed_parameters=transformed,
        coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
        generated_dims={
            "contribution": ("time", "group", "channel"),
            "total": ("time", "group"),
            "seasonal": ("time", "group"),
        },
    )
    result = sample(model, draws=2, warmup=3, chains=1)
    posterior = result["posterior"]
    coefficient_axes = ("group", "channel") if group_specific else ("channel",)
    assert posterior["paid_coefficient"].dims == ("chain", "draw", *coefficient_axes)
    for name in ("paid_retention", "paid_half_saturation", "paid_slope"):
        assert posterior[name].dims == ("chain", "draw", "channel")
    np.testing.assert_array_equal(posterior["channel"], ["Video", "Search"])
    modes = posterior["annual_coefficients"].dims[2]
    np.testing.assert_array_equal(posterior[modes], ["sin_1", "sin_2", "cos_1", "cos_2"])
    annual_axes = (modes, "group") if group_specific else (modes,)
    assert posterior["annual_coefficients"].dims == ("chain", "draw", *annual_axes)
    if group_specific:
        np.testing.assert_array_equal(posterior["group"], ["west", "east"])
    generated = result["generated_quantities"]
    assert generated["contribution"].dims == ("chain", "draw", "time", "group", "channel")
    for name in ("total", "seasonal"):
        assert generated[name].dims == ("chain", "draw", "time", "group")
    assert generated["weights"].dims == ("chain", "draw", *annual_axes)
    assert generated["retention"].dims == ("chain", "draw", "channel")
    assert generated["exposure"].dims == ("chain", "draw", "media_time", "group", "channel")
    np.testing.assert_array_equal(generated["media_time"], [0, 1, 2, 3])
    np.testing.assert_array_equal(generated["time"], [1, 2, 3])
    assert result["posterior_predictive"]["prediction"].dims == ("chain", "draw", "time", "group")
    assert result["log_likelihood"]["pointwise"].dims == ("chain", "draw", "time", "group")
    np.testing.assert_array_equal(generated["channel"], ["Video", "Search"])


def test_collected_data_uses_model_scaling_and_does_not_rescale_draws_or_generated_values(nuts_calls):
    data = prepare_data(
        pd.DataFrame({"time": [1, 2, 3], "sales": [100.0, 200.0, 300.0], "price": [10.0, 20.0, 40.0]}),
        time="time",
        outcome="sales",
        controls=["price"],
    )
    scaling = fit_data_scaling(data, scale_outcome=True)

    def density(outcome, controls, location):
        return normal(outcome, location + controls[..., 0], 1.0) + normal(location, 0.0, 1.0)

    def generate(key, outcome, location):
        return {"mean": jnp.full_like(outcome, location)}

    model = Model(
        parameters={"location": Real()},
        log_density=density,
        generated_quantities=generate,
        data=Data(data, scaling=scaling),
        generated_dims={"mean": ("time",)},
    )
    expected_outcome = np.array(model.data.values["outcome"])
    expected_controls = np.array(model.data.values["controls"])
    assert not np.allclose(expected_outcome, data.arrays["outcome"])
    assert not np.allclose(expected_controls, data.arrays["controls"])
    data.arrays["outcome"][...] = -999.0
    data.arrays["controls"][...] = -999.0
    results = sample(model, draws=2, warmup=3, chains=1, initial_values={"location": 1.25})
    np.testing.assert_array_equal(results["observed_data"]["outcome"], expected_outcome)
    np.testing.assert_array_equal(results["constant_data"]["controls"], expected_controls)
    np.testing.assert_array_equal(results["posterior"]["location"], np.full((1, 2), 1.25))
    np.testing.assert_array_equal(results["generated_quantities"]["mean"], np.full((1, 2, 3), 1.25))
    assert results.attrs["data_scale"] == "model"


@pytest.mark.parametrize(
    "statistic,message",
    [("diverging", "1 divergent transition"), ("reached_max_treedepth", "max_tree_depth on 1 draw")],
)
def test_problematic_sampler_diagnostics_warn_without_dropping_draws(
    normal_model, nuts_calls, monkeypatch, statistic, message
):
    backend = sampling._sample_nuts

    def flagged_draws(*args, **kwargs):
        (positions, statistics), continuation = backend(*args, **kwargs)
        statistics[statistic][0, 0] = True
        return (positions, statistics), continuation

    monkeypatch.setattr(sampling, "_sample_nuts", flagged_draws)
    with pytest.warns(RuntimeWarning, match=message):
        results = sample(normal_model, data=0.0, draws=3, warmup=4, chains=2)
    assert results["posterior"]["location"].shape == (2, 3)
    assert int(results["sample_stats"][statistic].sum()) == 1


def test_nonfinite_sampled_log_density_raises_instead_of_returning_results(normal_model, nuts_calls, monkeypatch):
    backend = sampling._sample_nuts

    def invalid_draws(*args, **kwargs):
        (positions, statistics), continuation = backend(*args, **kwargs)
        statistics["lp"][0, 0] = -np.inf
        return (positions, statistics), continuation

    monkeypatch.setattr(sampling, "_sample_nuts", invalid_draws)
    with pytest.raises(RuntimeError, match="nonfinite log densities"):
        sample(normal_model, data=0.0, draws=3, warmup=4, chains=1)


@pytest.mark.parametrize("option", ["draws", "warmup", "chains", "max_tree_depth", "batch_size", "chunk_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, "2", np.nan, np.bool_(True)])
def test_counts_require_positive_integers_before_sampling(normal_model, nuts_calls, monkeypatch, option, value):
    def forbidden_initialize(self, key):
        raise AssertionError("Count validation must run before parameter initialization")

    monkeypatch.setattr(Model, "initialize_random", forbidden_initialize)
    arguments = {"draws": 2, "warmup": 3, "chains": 1, option: value}
    with pytest.raises(ValueError, match=option):
        sample(normal_model, data=0.0, **arguments)
    assert not nuts_calls


@pytest.mark.parametrize("value", [0.0, 1.0, np.nan, np.inf, True, "0.8"])
def test_acceptance_target_must_be_a_probability(normal_model, nuts_calls, value):
    with pytest.raises(ValueError, match="target_accept"):
        sample(normal_model, data=0.0, target_accept=value, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_seed_requires_a_nonnegative_integer(normal_model, nuts_calls, value):
    with pytest.raises(ValueError, match="seed"):
        sample(normal_model, data=0.0, seed=value, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize(
    "initial", [{}, {"unexpected": 0.0}, {"location": [0.0, 1.0]}, {"location": np.nan}, {"location": np.inf}]
)
def test_initial_values_must_be_complete_finite_constrained_parameters(normal_model, nuts_calls, initial):
    with pytest.raises((ValueError, TypeError)):
        sample(normal_model, data=0.0, initial_values=initial, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_initial_positive_parameters_must_be_in_the_interior(nuts_calls, value):
    model = Model(parameters={"scale": Positive()}, log_density=lambda data, scale: half_normal(scale, 1.0))
    with pytest.raises(ValueError):
        sample(model, initial_values={"scale": value}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def test_nuts_samples_correlation_factors_with_labeled_matrix_axes():
    model = Model(
        parameters={"factor": CorrelationCholesky(dims=("effect", "effect_to"))},
        log_density=lambda data, factor: lkj_cholesky(factor, 2.0),
        coords={"effect": ["search", "video"], "effect_to": ["search", "video"]},
    )
    results = sample(model, draws=20, warmup=60, chains=1, seed=8, initial_values={"factor": jnp.eye(2)})
    factors = results["posterior"]["factor"]
    assert factors.dims == ("chain", "draw", "effect", "effect_to")
    assert factors.shape == (1, 20, 2, 2)
    np.testing.assert_array_equal(factors.effect, ["search", "video"])
    np.testing.assert_allclose(np.sum(factors.values**2, axis=-1), 1, atol=3e-6)
    assert np.all(np.isfinite(results["sample_stats"]["lp"]))
    assert np.unique(factors.values[..., 1, 0]).size > 1


@pytest.mark.parametrize(
    "factor",
    [
        [[1.0, 0.1], [0.0, 1.0]],
        [[1.0, 0.0], [0.5, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    ],
)
def test_initial_correlation_factors_cannot_be_projected_into_support(nuts_calls, factor):
    model = Model(
        parameters={"factor": CorrelationCholesky((2, 2))}, log_density=lambda data, factor: lkj_cholesky(factor, 1.0)
    )
    with pytest.raises(ValueError, match="factor"):
        sample(model, initial_values={"factor": factor}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize(
    "weights", [[1.0, 1.0, 1.0], [0.2, 0.3, 0.4], [np.nan, 0.5, 0.5], [np.inf, 0.5, 0.5], [0.3], [np.nan], [np.inf]]
)
def test_initial_simplex_values_cannot_be_renormalized_or_discarded(nuts_calls, weights):
    model = Model(
        parameters={"weights": Simplex((len(weights),)), "location": Real()},
        log_density=lambda data, weights, location: normal(weights, 0.0, 1.0) + normal(location, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="weights"):
        sample(model, initial_values={"weights": weights, "location": 0.0}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def test_valid_single_category_simplex_is_kept_with_other_free_parameters(nuts_calls):
    model = Model(
        parameters={"weights": Simplex((1,)), "location": Real()},
        log_density=lambda data, weights, location: normal(weights, 0.0, 1.0) + normal(location, 0.0, 1.0),
    )
    results = sample(model, initial_values={"weights": [1.0], "location": 0.0}, draws=2, warmup=3, chains=1)
    assert nuts_calls[0]["positions"]["weights"].shape == (1, 0)
    np.testing.assert_array_equal(results["posterior"]["weights"], np.ones((1, 2, 1)))


@pytest.mark.parametrize("failure", ["density", "gradient"])
def test_nonfinite_initial_density_or_gradient_is_rejected_before_sampling(nuts_calls, failure):
    def density(data, location):
        return jnp.log(-jnp.square(location) - 1.0) if failure == "density" else -jnp.sqrt(jnp.abs(location))

    model = Model(parameters={"location": Real()}, log_density=density)
    with pytest.raises(ValueError, match=r"finite|density|gradient"):
        sample(model, initial_values={"location": 0.0}, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def test_prepared_models_do_not_accept_separate_sampling_data(nuts_calls):
    model, data = _prepared_model()
    with pytest.raises(ValueError, match="data"):
        sample(model, data=data, draws=2, warmup=3, chains=1)
    assert not nuts_calls


def _saved_model(**options):
    _, data = _prepared_model()

    def transformed(controls, intercept):
        return {"mu": intercept + controls[:, 0], "unused": intercept**2}

    def density(outcome, mu, intercept):
        return normal(outcome, mu, 10.0) + normal(intercept, 0.0, 1.0)

    settings = {"generated_quantities": lambda key, mu: {"mu": mu}, "generated_dims": {"mu": ("time",)}} | options
    return Model(
        parameters={"intercept": Real()},
        log_density=density,
        data=data,
        transformed_parameters=transformed,
        **settings,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 64])
def test_sampling_collects_returned_transformed_quantities(nuts_calls, batch_size):
    model = _saved_model()
    options = {"draws": 3, "warmup": 4, "chains": 2, "seed": 9, "initial_values": {"intercept": 2.0}}
    results = sample(model, **options, batch_size=batch_size)
    disabled = sample(model, **options, generate=False, batch_size=batch_size)

    assert set(results["posterior"].data_vars) == {"intercept"}
    assert set(results["generated_quantities"].data_vars) == {"mu"}
    assert results["generated_quantities"]["mu"].dims == ("chain", "draw", "time")
    np.testing.assert_allclose(results["generated_quantities"]["mu"], np.broadcast_to([7.0, 6.0, 5.0], (2, 3, 3)))
    np.testing.assert_array_equal(results["generated_quantities"].coords["time"], [10, 11, 12])
    xr.testing.assert_identical(results["posterior"], disabled["posterior"])
    xr.testing.assert_identical(results["sample_stats"], disabled["sample_stats"])
    assert "generated_quantities" not in disabled
    assert len(nuts_calls) == 2


@pytest.mark.parametrize("problem", ["missing", "dimensions"])
def test_generated_quantities_are_validated_before_chain_adaptation(nuts_calls, problem):
    if problem == "missing":
        model = _saved_model(generated_quantities=lambda key, unknown: {"unknown": unknown})
        message = "unknown"
    else:
        model = _saved_model(generated_dims={"mu": ()})
        message = "dims matching"

    with pytest.raises(ValueError, match=message):
        sample(model, draws=2, warmup=3, chains=1)
    assert not nuts_calls


@pytest.mark.parametrize(
    "chain_method,mass_matrix",
    [("sequential", "diagonal"), ("vectorized", "diagonal"), ("sequential", "dense"), ("vectorized", "dense")],
)
def test_continuation_matches_one_run_without_reinitializing_or_adapting(monkeypatch, chain_method, mass_matrix):
    model = Model(
        parameters={"location": Real((2,))},
        log_density=lambda data, location: normal(location[0], 0.0, 1.0) + normal(location[1], 0.5 * location[0], 0.8),
    )
    options = {
        "warmup": 60,
        "chains": 2,
        "seed": 91,
        "target_accept": 0.85,
        "chain_method": chain_method,
        "mass_matrix": mass_matrix,
        "initial_values": {"location": np.zeros(2)},
        "generate": False,
        "progress": False,
    }
    full = sample(model, draws=9, chunk_size=9, **options)
    first = sample(model, draws=4, chunk_size=3, **options)
    first_snapshot = first.copy(deep=True)

    def forbidden_restart(*args, **kwargs):
        raise AssertionError("Continuation must reuse the adapted chain state")

    monkeypatch.setattr(nuts.blackjax, "window_adaptation", forbidden_restart)
    monkeypatch.setattr(Model, "initialize_random", forbidden_restart)
    second = continue_sampling(model, first, draws=3, chunk_size=2, batch_size=2, progress=False)
    final = continue_sampling(model, second, draws=2, chunk_size=1, batch_size=1, progress=False)

    xr.testing.assert_identical(final, full)
    xr.testing.assert_identical(first, first_snapshot)
    xr.testing.assert_identical(second["posterior"], full["posterior"].isel(draw=slice(0, 7)))
    completed = tuple(results["sampling_state"].attrs["completed_draws"] for results in (first, second, final))
    assert completed == (4, 7, 9)
    assert final["sampling_state"]["step_size"].sizes["chain"] == 2
    assert final["sampling_state/position"]["location"].shape == (2, 2)
    np.testing.assert_array_equal(final["posterior"].coords["draw"], np.arange(9))
    for group in final.children.values():
        for value in group.data_vars.values():
            assert isinstance(value.data, np.ndarray)
            assert value.data.flags.writeable


def test_continuation_preserves_generated_streams_saved_quantities_and_prepared_labels():
    data = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [1.0, 2.0, 3.0], "price": [0.1, 0.2, 0.3]}),
        time="week",
        outcome="sales",
        controls=["price"],
    )

    def transformed(controls, location):
        return {"mu": location + controls[:, 0]}

    def density(outcome, mu, location):
        return normal(outcome, mu, 1.0) + normal(location, 0.0, 2.0)

    def generate(key, outcome, mu, location):
        return {
            "predictive": {"prediction": normal_rng(key, mu, 1.0)},
            "log_likelihood": {"pointwise": normal_logpdf(outcome, mu, 1.0)},
            "log_prior": {"lp_location": normal(location, 0.0, 2.0)},
            "key_words": jax.random.key_data(key),
            "mu": mu,
        }

    model = Model(
        parameters={"location": Real()},
        log_density=density,
        generated_quantities=generate,
        data=data,
        transformed_parameters=transformed,
        generated_dims={"mu": ("time",), "key_words": ("word",)},
        coords={"word": ["first", "second"]},
    )
    options = {"warmup": 60, "chains": 2, "seed": 17, "progress": False, "batch_size": 3}
    full = sample(model, draws=7, **options)
    first = sample(model, draws=3, **options)
    combined = continue_sampling(model, first, draws=4, batch_size=3, progress=False)

    xr.testing.assert_identical(combined, full)
    assert combined["sampling_state"].attrs["completed_draws"] == 7
    assert combined.attrs == first.attrs
    assert combined.attrs["warmup_steps"] == 60
    assert combined.attrs["seed"] == 17
    assert combined["log_prior"]["lp_location"].dims == ("chain", "draw")
    np.testing.assert_array_equal(combined["log_prior"]["draw"], np.arange(7))
    np.testing.assert_allclose(
        combined["log_prior"]["lp_location"],
        normal_logpdf(combined["posterior"]["location"].values, 0.0, 2.0),
        rtol=2e-6,
    )
    for group in ("posterior_predictive", "log_likelihood", "generated_quantities"):
        np.testing.assert_array_equal(combined[group].coords["draw"], np.arange(7))
        np.testing.assert_array_equal(combined[group].coords["time"], [10, 11, 12])
    assert combined["generated_quantities"]["mu"].dims == ("chain", "draw", "time")
    assert combined["generated_quantities"]["key_words"].dims == ("chain", "draw", "word")
    words = combined["generated_quantities"]["key_words"].values.reshape(-1, 2)
    assert np.unique(words, axis=0).shape[0] == 14
    for group in ("observed_data", "constant_data"):
        xr.testing.assert_identical(combined[group], first[group])
        assert "draw" not in combined[group].dims


def test_continuation_preserves_prepared_inputs_in_wrapped_models():
    data = prepare_data(
        pd.DataFrame({"week": [10, 11, 12], "sales": [1.0, 2.0, 3.0]}),
        time="week",
        outcome="sales",
    )
    prepared_model = Model(
        parameters={"location": Real()},
        log_density=lambda outcome, location: normal(outcome, location, 1.0) + normal(location, 0.0, 2.0),
        generated_quantities=lambda key, outcome, location: {
            "predictive": {"prediction": normal_rng(key, location, 1.0, sample_shape=outcome.shape)}
        },
        data=data,
    )
    model = Model(
        parameters=prepared_model.parameters,
        log_density=lambda data, location: prepared_model.log_prob({"location": location}, data=data),
        generated_quantities=lambda key, data, location: prepared_model.generate_quantities(
            key, {"location": location}, data
        ),
    )
    options = {"data": prepared_model.data, "warmup": 60, "chains": 1, "seed": 29, "progress": False}

    full = sample(model, draws=7, **options)
    first = sample(model, draws=3, **options)
    continued = continue_sampling(model, first, data=prepared_model.data, draws=4, batch_size=2, progress=False)

    xr.testing.assert_identical(continued, full)
    xr.testing.assert_identical(first["posterior"], full["posterior"].isel(draw=slice(0, 3)))
    assert first["sampling_state"].attrs["completed_draws"] == 3
    assert continued["sampling_state"].attrs["completed_draws"] == 7


def test_continuation_leaves_supplied_results_unchanged_and_repeats_from_them():
    inputs = {"center": np.array([0.0, 0.5])}
    model = Model(
        parameters={"location": Real((2,))},
        log_density=lambda data, location: normal(location, data["center"], 1.0),
        generated_quantities=lambda key, data, location: {
            "center": data["center"],
            "noise": jax.random.normal(key, (2,)),
        },
    )
    options = {"data": inputs, "warmup": 60, "chains": 1, "seed": 23, "progress": False}
    full = sample(model, draws=7, **options)
    first = sample(model, draws=3, **options)
    first_snapshot = first.copy(deep=True)

    continued = continue_sampling(model, first, data=inputs, draws=4, chunk_size=2, progress=False)
    xr.testing.assert_identical(continued, full)
    xr.testing.assert_identical(first, first_snapshot)
    continued["posterior"]["location"].values[:] = 99.0
    continued["sampling_state"]["step_size"].values[:] = 99.0
    continued.attrs["seed"] = 99
    xr.testing.assert_identical(first, first_snapshot)

    repeated = continue_sampling(model, first, data=inputs, draws=4, chunk_size=3, progress=False)
    xr.testing.assert_identical(repeated, full)
    extended = continue_sampling(model, full, data=inputs, draws=1, progress=False)
    xr.testing.assert_identical(extended["posterior"].isel(draw=slice(0, 7)), full["posterior"])
    assert extended["sampling_state"].attrs["completed_draws"] == 8


def test_continuation_resumes_from_results_rebuilt_from_plain_data(normal_model):
    options = {"data": 0.0, "warmup": 60, "chains": 2, "seed": 5, "progress": False}
    full = sample(normal_model, draws=6, **options)
    first = sample(normal_model, draws=2, **options)

    # Everything needed to continue survives a trip through plain Python data, as a file would store it.
    serialized = json.loads(json.dumps({path: node.to_dict() for path, node in first.to_dict().items()}))
    restored = xr.DataTree.from_dict({path: xr.Dataset.from_dict(node) for path, node in serialized.items()})
    assert restored["sampling_state"]["sampling_key"].dtype != first["sampling_state"]["sampling_key"].dtype

    continued = continue_sampling(normal_model, restored, data=0.0, draws=4, progress=False)
    xr.testing.assert_identical(continued["posterior"], full["posterior"])
    xr.testing.assert_identical(continued["sample_stats"], full["sample_stats"])
    assert continued["sampling_state"].attrs["completed_draws"] == 6


@pytest.mark.parametrize("mass_matrix", ["diagonal", "dense"])
def test_result_dimensions_never_share_a_name_with_a_child_group(normal_model, mass_matrix):
    # netCDF writes dimensions beside child groups. A shared name makes results impossible to save.
    options = {"data": 0.0, "draws": 2, "warmup": 20, "chains": 2, "seed": 5, "progress": False}
    results = sample(normal_model, mass_matrix=mass_matrix, **options)
    for node in results.subtree:
        assert not set(node.dims) & set(node.children), node.path


def test_continuation_coordinate_edits_do_not_change_other_results():
    data = prepare_data(
        pd.DataFrame(
            {
                "week": [10, 11, 10, 11],
                "country": ["US", "US", "CA", "CA"],
                "market": ["east", "east", "west", "west"],
                "sales": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        time="week",
        outcome="sales",
        groups=["country", "market"],
    )
    model = Model(
        parameters={"location": Real(dims="group")},
        log_density=lambda outcome, location: normal(outcome, location, 1.0) + normal(location, 0.0, 2.0),
        generated_quantities=lambda key, outcome, location: {
            "predictive": {"prediction": normal_rng(key, jnp.broadcast_to(location, outcome.shape), 1.0)}
        },
        data=data,
    )
    options = {"warmup": 60, "chains": 2, "seed": 27, "progress": False}
    full = sample(model, draws=7, **options)
    first = sample(model, draws=2, **options)
    first_snapshot = first.copy(deep=True)
    combined = continue_sampling(model, first, draws=3, progress=False)
    combined_snapshot = combined.copy(deep=True)

    for name, group in combined.children.items():
        for coordinate, replacement in (("chain", 99), ("group", 99), ("group_market", "edit")):
            if coordinate in group.coords:
                values = group.coords[coordinate].values
                if coordinate in first[name].coords:
                    assert not np.shares_memory(values, first[name].coords[coordinate].values)
                # Some pandas versions expose index arrays through read-only views.
                if coordinate == "group_market":
                    assert values.flags.writeable
                if values.flags.writeable:
                    values[0] = replacement

    repeated = continue_sampling(model, first, draws=3, progress=False)
    extended = continue_sampling(model, combined_snapshot, draws=2, progress=False)
    xr.testing.assert_identical(first, first_snapshot)
    xr.testing.assert_identical(repeated, combined_snapshot)
    xr.testing.assert_identical(extended, full)
    assert first["sampling_state"].attrs["completed_draws"] == 2
    assert combined["sampling_state"].attrs["completed_draws"] == 5


def test_continuation_keeps_generation_disabled_and_only_shows_retained_progress(progress_contexts):
    def forbidden_generate(key, data, location):
        raise AssertionError("Disabled generation must remain disabled during continuation")

    model = Model(
        parameters={"location": Real()},
        log_density=lambda data, location: normal(location, 0.0, 1.0),
        generated_quantities=forbidden_generate,
    )
    first = sample(model, draws=2, warmup=60, chains=2, seed=13, generate=False, progress=False)
    result = continue_sampling(model, first, draws=3, chunk_size=2, progress=True)
    assert set(result.children) == {"posterior", "sample_stats", "sampling_state"}
    assert result["sampling_state"].attrs["generate"] == 0
    assert [context.n_steps for context in progress_contexts] == [2, 1, 2, 1]
    assert all(context.closed for context in progress_contexts)


def test_continuation_rejects_invalid_results_counts_and_sampling_changes_before_execution(normal_model, monkeypatch):
    result = sample(normal_model, data=0.0, draws=2, warmup=60, chains=1, seed=11, progress=False)

    def forbidden_sampling(*args, **kwargs):
        raise AssertionError("Continuation validation must run before sampling")

    monkeypatch.setattr(nuts.blackjax, "nuts", forbidden_sampling)
    with jax.enable_x64(not jax.config.x64_enabled), pytest.raises(ValueError, match="precision"):
        continue_sampling(normal_model, result, data=0.0, draws=2, progress=False)
    stripped = xr.DataTree.from_dict(
        {name: node.to_dataset() for name, node in result.children.items() if name != "sampling_state"}
    )
    for invalid in (None, normal_model, object(), result["posterior"].to_dataset(), stripped):
        with pytest.raises(TypeError, match="results"):
            continue_sampling(normal_model, invalid, data=0.0, draws=2, progress=False)
    with pytest.raises(TypeError, match="model"):
        continue_sampling(None, result, data=0.0, draws=2, progress=False)
    renamed = Model(parameters={"scale": Real()}, log_density=lambda data, scale: normal(scale, data, 1.0))
    with pytest.raises(ValueError, match="parameters"):
        continue_sampling(renamed, result, data=0.0, draws=2, progress=False)
    reshaped = Model(
        parameters={"location": Real((2,))}, log_density=lambda data, location: normal(location, data, 1.0)
    )
    with pytest.raises(ValueError, match="shape"):
        continue_sampling(reshaped, result, data=0.0, draws=2, progress=False)
    for name in ("draws", "chunk_size", "batch_size"):
        for value in (0, -1, True, 1.5, None, "2", np.nan, np.bool_(True)):
            with pytest.raises(ValueError, match=name):
                continue_sampling(
                    normal_model, result, **({"data": 0.0, "draws": 2, "progress": False} | {name: value})
                )
    for value in (None, 0, 1, "true", [], {}, np.bool_(True)):
        with pytest.raises(TypeError, match="progress"):
            continue_sampling(normal_model, result, data=0.0, draws=2, progress=value)
    for name in (
        "seed",
        "warmup",
        "target_accept",
        "mass_matrix",
        "max_tree_depth",
        "chains",
        "chain_method",
        "generate",
        "initial_values",
        "return_state",
    ):
        with pytest.raises(TypeError, match=name):
            continue_sampling(normal_model, result, data=0.0, draws=2, progress=False, **{name: None})


def test_prepared_models_reject_data_when_continuing():
    model, _ = _prepared_model()
    result = sample(model, draws=2, warmup=60, chains=1, seed=3, progress=False)
    with pytest.raises(ValueError, match="Omit data"):
        continue_sampling(model, result, data=model.data, draws=2, progress=False)
