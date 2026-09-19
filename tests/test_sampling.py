"""Tests for sampling JAX models and collecting constrained, labeled draws."""

import json
from contextlib import contextmanager

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
    Model,
    Real,
    continue_sampling,
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
