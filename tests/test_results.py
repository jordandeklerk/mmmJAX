"""Private assembly of labeled posterior arrays and model inputs."""

from copy import deepcopy
from datetime import date
from io import BytesIO

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmmjax import Model, Positive, Simplex, normal, normal_rng, prepare_data
from mmmjax._results import _collect_results


@pytest.fixture
def posterior():
    return {
        "intercept": np.arange(6, dtype=float).reshape(2, 3),
        "coefficient": np.arange(12, dtype=float).reshape(2, 3, 2),
    }


def _prepared_data(grouping):
    if grouping == "single":
        columns, labels = ["region"], [("west",), ("east",)]
    elif grouping == "multiple":
        columns, labels = ["region", "store"], [("west", "retail"), ("east", "online")]
    else:
        columns, labels = [], [()]
    rows = []
    for period in range(4):
        for group, values in enumerate(labels):
            rows.append(
                {
                    "week": period,
                    **dict(zip(columns, values, strict=True)),
                    "sales": 100.0 + period + group,
                    "unit_revenue": 2.0,
                    "population": 1000.0 + group,
                    "video": 10.0 + period + group,
                    "search": 15.0 + period + group,
                    "video_cost": 1.0 + period,
                    "search_cost": 2.0 + period,
                    "email": 3.0 + period,
                    "audience": 100.0 + period,
                    "views": 2.0,
                    "reach_cost": 3.0 + period,
                    "organic_audience": 50.0 + period,
                    "organic_views": 1.0,
                    "temperature": 20.0 + period,
                    "price": 5.0 - period,
                }
            )
    frame = pd.DataFrame(rows)
    exposures = ["video", "search", "email", "audience", "views", "organic_audience", "organic_views"]
    return prepare_data(
        frame.loc[frame["week"] > 0],
        time="week",
        groups=columns,
        outcome="sales",
        revenue_per_outcome="unit_revenue",
        population="population",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Video", "Search"],
        organic_media=["email"],
        organic_channels=["Email"],
        reach=["audience"],
        media_frequency=["views"],
        rf_spend=["reach_cost"],
        rf_channels=["Streaming"],
        organic_reach=["organic_audience"],
        organic_frequency=["organic_views"],
        organic_rf_channels=["Newsletter"],
        controls=["temperature"],
        treatments=["price"],
        media_history=frame.loc[frame["week"] == 0, ["week", *columns, *exposures]],
    )


def test_collection_labels_all_result_groups(posterior):
    data = _prepared_data("national")
    predictions = np.arange(18).reshape(2, 3, 3)
    divergences = np.array([[False, True, False], [False, False, False]])
    results = _collect_results(
        posterior,
        data=data,
        posterior_predictive={"outcome": predictions},
        log_likelihood={"outcome": -predictions.astype(float)},
        generated_quantities={"mean": predictions.astype(float)},
        sample_stats={"diverging": divergences, "tree_depth": np.full((2, 3), 4), "warmup_steps": 200},
        dims={"coefficient": ("channel",)},
        generated_dims={"outcome": ("time",), "mean": ("time",)},
        coords={"chain": ["first", "second"], "draw": [10, 20, 30]},
    )
    assert isinstance(results, xr.DataTree)
    assert set(results.children) == {
        "posterior",
        "posterior_predictive",
        "log_likelihood",
        "generated_quantities",
        "sample_stats",
        "observed_data",
        "constant_data",
    }
    assert results["posterior"]["intercept"].dims == ("chain", "draw")
    assert results["posterior"]["coefficient"].dims == ("chain", "draw", "channel")
    for name in ("posterior_predictive", "log_likelihood"):
        assert results[name]["outcome"].dims == ("chain", "draw", "time")
    assert results["generated_quantities"]["mean"].dims == ("chain", "draw", "time")
    assert results["observed_data"]["outcome"].dims == ("time",)
    assert results["constant_data"]["media"].dims == ("media_time", "channel")
    assert results["constant_data"]["population"].dims == ()
    assert results["sample_stats"]["diverging"].dtype == np.bool_
    assert np.issubdtype(results["sample_stats"]["tree_depth"].dtype, np.integer)
    assert results["sample_stats"]["warmup_steps"].dims == ()
    for name in ("posterior", "posterior_predictive", "log_likelihood", "generated_quantities", "sample_stats"):
        np.testing.assert_array_equal(results[name]["chain"], ["first", "second"])
        np.testing.assert_array_equal(results[name]["draw"], [10, 20, 30])
        assert results[name].attrs["sample_dims"] == ["chain", "draw"]
    np.testing.assert_array_equal(results["posterior"]["coefficient"], posterior["coefficient"])
    np.testing.assert_array_equal(results["posterior_predictive"]["outcome"], predictions)
    np.testing.assert_array_equal(results["generated_quantities"]["mean"], predictions)


def test_default_coordinates_are_integer_labels_not_inferred_data_axes(posterior):
    results = _collect_results(posterior, dims={"coefficient": ("feature",)})
    assert set(results.children) == {"posterior"}
    for dimension, labels in {"chain": [0, 1], "draw": [0, 1, 2], "feature": [0, 1]}.items():
        np.testing.assert_array_equal(results["posterior"][dimension], labels)


def test_collection_preserves_constrained_simplex_values_and_jax_array_dtype():
    simplex = jnp.array([[[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]], dtype=jnp.float32)
    results = _collect_results({"weights": simplex}, dims={"weights": ("category",)})
    weights = results["posterior"]["weights"]
    assert weights.shape == (1, 2, 3)
    assert weights.dtype == np.float32
    assert isinstance(weights.data, np.ndarray)
    np.testing.assert_array_equal(weights, np.asarray(simplex))


def test_nonfinite_log_likelihood_and_diagnostics_are_preserved():
    values = np.array([[-np.inf, np.nan, -1.0]])
    results = _collect_results(
        {"intercept": np.zeros((1, 3))},
        log_likelihood={"joint_observation": values},
        sample_stats={"energy": values},
    )
    np.testing.assert_array_equal(results["log_likelihood"]["joint_observation"], values)
    np.testing.assert_array_equal(results["sample_stats"]["energy"], values)


@pytest.mark.parametrize("grouping", ["national", "single", "multiple"])
def test_prepared_data_supplies_every_role_with_distinct_time_and_feature_labels(grouping):
    data = _prepared_data(grouping)
    original = {name: values.copy() for name, values in data.arrays.items()}
    results = _collect_results({"intercept": np.zeros((2, 3))}, data=data)
    observed = results["observed_data"]
    constant = results["constant_data"]
    group_dims = () if grouping == "national" else ("group",)
    expected_dims = {
        "outcome": ("time", *group_dims),
        "revenue_per_outcome": ("time", *group_dims),
        "population": group_dims,
        "media": ("media_time", *group_dims, "channel"),
        "spend": ("time", *group_dims, "channel"),
        "organic_media": ("media_time", *group_dims, "organic_channel"),
        "reach": ("media_time", *group_dims, "rf_channel"),
        "media_frequency": ("media_time", *group_dims, "rf_channel"),
        "rf_spend": ("time", *group_dims, "rf_channel"),
        "organic_reach": ("media_time", *group_dims, "organic_rf_channel"),
        "organic_frequency": ("media_time", *group_dims, "organic_rf_channel"),
        "controls": ("time", *group_dims, "control"),
        "treatments": ("time", *group_dims, "treatment"),
    }
    assert set(observed.data_vars) == {"outcome"}
    assert set(constant.data_vars) == set(data.arrays) - {"outcome"}
    for role, values in original.items():
        variable = observed[role] if role == "outcome" else constant[role]
        assert variable.dims == expected_dims[role]
        np.testing.assert_array_equal(variable, values)
        np.testing.assert_array_equal(data.arrays[role], values)
    for dimension, labels in {
        "time": [1, 2, 3],
        "media_time": [0, 1, 2, 3],
        "channel": ["Video", "Search"],
        "organic_channel": ["Email"],
        "rf_channel": ["Streaming"],
        "organic_rf_channel": ["Newsletter"],
        "control": ["temperature"],
        "treatment": ["price"],
    }.items():
        np.testing.assert_array_equal(constant[dimension], labels)
    if grouping == "single":
        np.testing.assert_array_equal(observed["group"], ["west", "east"])
    elif grouping == "multiple":
        np.testing.assert_array_equal(observed["group"], [0, 1])
        np.testing.assert_array_equal(observed["group_region"], ["west", "east"])
        np.testing.assert_array_equal(observed["group_store"], ["retail", "online"])
    else:
        assert "group" not in observed.dims


def test_prepared_labels_can_describe_posterior_parameters_without_size_inference():
    data = _prepared_data("single")
    values = np.zeros((1, 2, 2, 2))
    results = _collect_results({"coefficient": values}, data=data, dims={"coefficient": ("group", "channel")})
    np.testing.assert_array_equal(results["posterior"]["group"], ["west", "east"])
    np.testing.assert_array_equal(results["posterior"]["channel"], ["Video", "Search"])
    with pytest.raises(ValueError, match="coefficient"):
        _collect_results({"coefficient": values}, data=data)


def test_group_aggregated_likelihood_keeps_its_explicit_observation_axes():
    data = _prepared_data("single")
    results = _collect_results(
        {"intercept": np.zeros((1, 2))},
        data=data,
        log_likelihood={"outcome": np.full((1, 2, 2), -3.0)},
        generated_dims={"outcome": ("group",)},
    )
    assert results["log_likelihood"]["outcome"].dims == ("chain", "draw", "group")
    assert results["observed_data"]["outcome"].dims == ("time", "group")


def test_same_name_in_posterior_and_generated_quantities_has_separate_dimensions():
    coefficient = np.arange(24).reshape(2, 3, 4)
    effect = np.arange(18).reshape(2, 3, 3)
    results = _collect_results(
        {"annual": coefficient},
        data=_prepared_data("national"),
        generated_quantities={"annual": effect},
        dims={"annual": ("annual_mode",)},
        generated_dims={"annual": ("time",)},
        coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
    )
    assert results["posterior"]["annual"].dims == ("chain", "draw", "annual_mode")
    assert results["generated_quantities"]["annual"].dims == ("chain", "draw", "time")
    np.testing.assert_array_equal(results["posterior"]["annual"], coefficient)
    np.testing.assert_array_equal(results["generated_quantities"]["annual"], effect)


def test_collection_takes_snapshots_without_mutating_inputs_or_metadata(posterior):
    data = _prepared_data("national")
    dims = {"coefficient": ["channel"]}
    coords = {"channel": ["Video", "Search"]}
    original_dims, original_coords = deepcopy(dims), deepcopy(coords)
    original_coefficient = posterior["coefficient"].copy()
    original_outcome = data.arrays["outcome"].copy()
    predictions = np.zeros((2, 3, 3))
    results = _collect_results(
        posterior,
        data=data,
        posterior_predictive={"outcome": predictions},
        dims=dims,
        generated_dims={"outcome": ("time",)},
        coords=coords,
    )
    assert dims == original_dims
    assert coords == original_coords
    np.testing.assert_array_equal(data.arrays["outcome"], original_outcome)
    posterior["coefficient"][...] = -100
    predictions[...] = 50
    coords["channel"][0] = "changed"
    data.arrays["outcome"][:] = -999
    np.testing.assert_array_equal(results["posterior"]["coefficient"], original_coefficient)
    np.testing.assert_array_equal(results["posterior"]["channel"], ["Video", "Search"])
    np.testing.assert_array_equal(results["posterior_predictive"]["outcome"], 0)
    np.testing.assert_array_equal(results["observed_data"]["outcome"], original_outcome)


@pytest.mark.parametrize("name", ["channel", "chain"])
def test_mapping_variable_names_cannot_be_promoted_to_dimension_coordinates(name):
    with pytest.raises(ValueError, match=name):
        _collect_results(
            {name: np.ones((1, 2)), "coefficient": np.ones((1, 2, 3))},
            dims={"coefficient": ("channel",)},
        )


def test_constant_diagnostics_do_not_claim_sample_dimensions():
    results = _collect_results({"intercept": np.zeros((2, 3))}, sample_stats={"warmup_steps": 100})
    assert results["posterior"].attrs["sample_dims"] == ["chain", "draw"]
    assert results["sample_stats"].attrs["sample_dims"] == []
    assert results["sample_stats"]["warmup_steps"].dims == ()


def test_prepared_python_dates_become_serializable_datetime_coordinates():
    dates = [date(2025, 1, 6), date(2025, 1, 13), date(2025, 1, 20)]
    data = prepare_data(pd.DataFrame({"week": dates, "sales": [100.0, 200.0, 150.0]}), time="week", outcome="sales")
    observed = _collect_results({"intercept": np.zeros((1, 2))}, data=data)["observed_data"].to_dataset()
    assert np.issubdtype(observed["time"].dtype, np.datetime64)
    np.testing.assert_array_equal(
        observed["time"].values.astype("datetime64[D]"), np.asarray(dates, dtype="datetime64[D]")
    )
    serialized = observed.to_netcdf(engine="scipy")
    with xr.open_dataset(BytesIO(bytes(serialized)), engine="scipy") as restored:
        xr.testing.assert_allclose(restored, observed)
    assert data.time_values == tuple(dates)


def test_national_data_does_not_claim_an_unrelated_posterior_group_dimension():
    results = _collect_results(
        {"coefficient": np.ones((1, 2, 4))},
        data=_prepared_data("national"),
        dims={"coefficient": ("group",)},
    )
    np.testing.assert_array_equal(results["posterior"]["group"], [0, 1, 2, 3])
    assert "group" not in results["observed_data"].dims


def test_model_constrained_draws_and_generated_quantities_keep_values_and_shapes():
    data = prepare_data(
        pd.DataFrame(
            {
                "time": [1, 2, 3],
                "sales": [100.0, 200.0, 150.0],
                "a": [90.0, 190.0, 140.0],
                "b": [110.0, 210.0, 160.0],
                "c": [100.0, 200.0, 150.0],
            }
        ),
        time="time",
        outcome="sales",
        controls=["a", "b", "c"],
    )

    def density(outcome, controls, weights, sigma):
        return normal(outcome, controls @ weights, sigma)

    def generate(key, controls, weights, sigma):
        return {"outcome": normal_rng(key, controls @ weights, sigma)}

    model = Model({"weights": Simplex((3,)), "sigma": Positive()}, density, generate, data=data, components=[])
    positions = {
        "weights": jnp.linspace(-0.4, 0.5, 12).reshape(2, 3, 2),
        "sigma": jnp.full((2, 3), jnp.log(2.0)),
    }
    constrained = jax.vmap(jax.vmap(model.constrain))(positions)
    keys = jax.random.split(jax.random.key(3), (2, 3))
    generated = jax.vmap(jax.vmap(lambda key, parameters: model.generate(key, parameters, model.data)))(
        keys, constrained
    )
    results = _collect_results(
        constrained,
        data=data,
        posterior_predictive=generated,
        dims={"weights": ("control",)},
        generated_dims={"outcome": ("time",)},
    )
    assert positions["weights"].shape == (2, 3, 2)
    assert results["posterior"]["weights"].shape == (2, 3, 3)
    for name, values in constrained.items():
        np.testing.assert_array_equal(results["posterior"][name], np.asarray(values))
    np.testing.assert_array_equal(results["posterior_predictive"]["outcome"], np.asarray(generated["outcome"]))
    np.testing.assert_array_equal(results["observed_data"]["outcome"], data.arrays["outcome"])
    np.testing.assert_array_equal(results["constant_data"]["controls"], data.arrays["controls"])


def test_posterior_requires_at_least_one_variable():
    with pytest.raises(ValueError, match="posterior"):
        _collect_results({})


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_posterior_requires_nonempty_sample_axes(shape):
    with pytest.raises(ValueError, match=r"chain|draw"):
        _collect_results({"intercept": np.zeros(shape)})


@pytest.mark.parametrize("group", ["posterior", "posterior_predictive", "log_likelihood", "generated_quantities"])
@pytest.mark.parametrize("values", [1.0, np.ones(3)])
def test_sampled_mappings_require_explicit_chain_and_draw_axes(group, values):
    arguments = {"posterior": {"intercept": np.zeros((1, 3))}, group: {"invalid": values}}
    with pytest.raises(ValueError, match=r"invalid|chain|draw"):
        _collect_results(**arguments)


@pytest.mark.parametrize("group", ["posterior_predictive", "log_likelihood", "generated_quantities", "sample_stats"])
@pytest.mark.parametrize("shape", [(1, 3), (2, 4)])
def test_sampled_groups_reject_mismatched_chain_or_draw_counts(group, shape):
    with pytest.raises(ValueError, match=r"chain|draw"):
        _collect_results({"intercept": np.zeros((2, 3))}, **{group: {"value": np.zeros(shape)}})


@pytest.mark.parametrize("group", ["posterior_predictive", "log_likelihood", "generated_quantities"])
def test_generated_mapping_arrays_require_named_intrinsic_dimensions(group):
    with pytest.raises(ValueError, match="custom"):
        _collect_results({"intercept": np.ones((1, 2))}, **{group: {"custom": np.ones((1, 2, 3))}})


@pytest.mark.parametrize("labels", [["one"], [["one", "two"]]])
def test_coordinate_lengths_and_rank_must_match_dimensions(posterior, labels):
    with pytest.raises(ValueError, match="channel"):
        _collect_results(posterior, dims={"coefficient": ("channel",)}, coords={"channel": labels})


@pytest.mark.parametrize("dimension,labels", [("time", [3, 2, 1]), ("channel", ["Search", "Video"])])
def test_explicit_coordinates_cannot_relabel_prepared_data(dimension, labels):
    with pytest.raises(ValueError, match=dimension):
        _collect_results({"intercept": np.ones((1, 2))}, data=_prepared_data("single"), coords={dimension: labels})


@pytest.mark.parametrize(
    "group", ["posterior", "posterior_predictive", "log_likelihood", "generated_quantities", "sample_stats"]
)
def test_sampled_group_inputs_are_unlabeled_array_mappings(group):
    dataset = xr.Dataset({"intercept": (("chain", "draw"), np.zeros((1, 2)))})
    arguments = {"posterior": {"intercept": np.zeros((1, 2))}, group: dataset}
    with pytest.raises(TypeError, match="mapping"):
        _collect_results(**arguments)
    arguments[group] = {"intercept": dataset["intercept"]}
    with pytest.raises(TypeError, match="unlabeled arrays"):
        _collect_results(**arguments)


def test_prepared_data_type_is_validated():
    with pytest.raises(TypeError, match="PreparedData"):
        _collect_results({"intercept": np.zeros((1, 2))}, data={})
