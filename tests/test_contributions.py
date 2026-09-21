"""Tests for baseline and channel contribution decomposition."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr

from mmmjax import Data, Model, Real, contributions, fit_data_scaling, prepare_data
from mmmjax._results import _collect_results


def _data(*, grouped=False):
    rows = []
    exposures = [(4.0, 2.0, 1.0), (1.0, 2.0, 3.0), (3.0, 4.0, 2.0), (5.0, 1.0, 4.0)]
    for week, (video, search, email) in enumerate(exposures):
        for group in range(2 if grouped else 1):
            scale = group + 1
            rows.append(
                {
                    "week": week,
                    "region": ("east", "west")[group],
                    "video": video * scale,
                    "search": search * scale,
                    "email": email * scale,
                    "video_cost": video * scale / 2,
                    "search_cost": search * scale / 4,
                    "price": 10.0 + week + group,
                    "promotion": float(week % 2),
                    "control": 2.0 + week + group,
                    "population": 100.0 + 200 * group,
                    "outcome": 50.0 + 10 * week + 20 * group,
                }
            )
    frame = pl.DataFrame(rows)
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region"] if grouped else (),
        outcome="outcome",
        population="population",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Online video", "Paid search"],
        organic_media=["email"],
        organic_channels=["Email"],
        treatments=["price", "promotion"],
        controls=["control"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _response(values, coefficient):
    carried = values["media"][1:] + 0.5 * values["media"][:-1]
    organic = values["organic_media"][1:] + 0.25 * values["organic_media"][:-1]
    response = (
        3.0
        + values["controls"][..., 0]
        + (carried @ coefficient) ** 2
        + 0.7 * carried[..., 0] * carried[..., 1]
        + 1.5 * organic[..., 0]
        + 2.0 * values["treatments"][..., 0]
        - 0.5 * values["treatments"][..., 1]
    )
    return response


def _model(data, *, scaling=None):
    def transformed(media, organic_media, treatments, controls, coefficient):
        values = {"media": media, "organic_media": organic_media, "treatments": treatments, "controls": controls}
        expected = _response(values, coefficient)
        return {"expected_users": expected, "exposure": media}

    def density(expected_users):
        raise AssertionError("Contributions must not evaluate the log density")

    def generated(key, expected_users):
        raise AssertionError("Contributions must not generate observations")

    return Model(
        parameters={"coefficient": Real((2,))},
        log_density=density,
        generated_quantities=generated,
        data=Data(data, scaling=scaling),
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )


def _results(data, *, group="posterior"):
    return _collect_results(
        {
            "coefficient": np.array(
                [[[0.5, 1.0], [1.0, 0.5], [2.0, 1.5]], [[0.8, 1.1], [1.5, 1.8], [2.5, 2.2]]], dtype=np.float32
            )
        },
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [4, 8], "draw": [10, 20, 30]},
        sample_group=group,
    )


def _labels(data):
    return (*data.channels, *data.organic_channels, *data.columns["treatments"])


def _baselines(data, rule="min", **overrides):
    treatments = data.arrays["treatments"]
    reduce = np.min if rule == "min" else np.max
    levels = {name: float(reduce(treatments[..., index])) for index, name in enumerate(data.columns["treatments"])}
    levels.update(overrides)
    return levels


def _restore_outcome_units(response, scaling):
    """Apply the fitted outcome transform in double precision for the closed form."""
    if scaling is None or "outcome" not in scaling.transformations:
        return response
    outcome = scaling.transformations["outcome"]
    scale = np.asarray(outcome.scale, dtype=np.float64)
    offset = np.asarray(outcome.offset, dtype=np.float64)
    restored = response * scale + offset
    return restored


def _expected(
    data,
    coefficient,
    *,
    removed=(),
    baselines=None,
    periods=None,
    response_periods=None,
    scaling=None,
    by=(),
):
    arrays = {name: value.copy() for name, value in data.arrays.items()}
    changed = (
        np.array([data.time_values.index(t) for t in periods])
        if periods is not None
        else np.arange(len(data.time_values))
    )
    measured = (
        np.array(sorted(data.time_values.index(t) for t in response_periods))
        if response_periods is not None
        else np.arange(len(data.time_values))
    )
    for label in removed:
        if label in data.channels:
            arrays["media"][1 + changed, ..., data.channels.index(label)] = 0.0
        elif label in data.organic_channels:
            arrays["organic_media"][1 + changed, ..., data.organic_channels.index(label)] = 0.0
        else:
            index = data.columns["treatments"].index(label)
            arrays["treatments"][changed, ..., index] = baselines[label]
    scenario = replace(data, arrays=arrays)
    values = (scaling.transform(scenario) if scaling is not None else scenario).arrays
    response = _response(values, coefficient)
    response = _restore_outcome_units(response, scaling)
    axes = ("time", "group") if data.group_columns else ("time",)
    kept = tuple(index for index, axis in enumerate(axes) if axis not in by)
    totals = response[measured].sum(axis=kept)
    return totals


def _exposure(data, label, periods=None):
    changed = (
        np.array([data.time_values.index(t) for t in periods])
        if periods is not None
        else np.arange(len(data.time_values))
    )
    if label in data.channels:
        return data.arrays["media"][1 + changed, ..., data.channels.index(label)].sum()
    if label in data.organic_channels:
        return data.arrays["organic_media"][1 + changed, ..., data.organic_channels.index(label)].sum()
    return np.nan


def _assert_closed_form(result, data, results, *, group="posterior", baselines, **options):
    by = result["reference_response"].dims[2:]
    everything = _labels(data)
    for chain in range(result.sizes["chain"]):
        for draw in range(result.sizes["draw"]):
            coefficient = results[group]["coefficient"].values[chain, draw]
            reference = _expected(data, coefficient, by=by, **options)
            baseline = _expected(data, coefficient, removed=everything, baselines=baselines, by=by, **options)
            np.testing.assert_allclose(result["reference_response"][chain, draw], reference, rtol=3e-6)
            np.testing.assert_allclose(result["baseline_response"][chain, draw], baseline, rtol=3e-6)
            np.testing.assert_allclose(
                result["joint_incremental_response"][chain, draw], reference - baseline, rtol=3e-5, atol=3e-5
            )
            np.testing.assert_allclose(
                result["baseline_share"][chain, draw], baseline.sum() / reference.sum(), rtol=3e-5, atol=3e-5
            )
            for index, label in enumerate(result.channel.values):
                alone = _expected(data, coefficient, removed=(label,), baselines=baselines, by=by, **options)
                increment = reference - alone
                exposure = _exposure(data, label, options.get("periods"))
                selected = result.isel(chain=chain, draw=draw, channel=index)
                np.testing.assert_allclose(selected["incremental_response"], increment, rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(
                    selected["contribution_share"], increment.sum() / reference.sum(), rtol=3e-5, atol=3e-5
                )
                if np.isnan(exposure):
                    assert np.isnan(selected["effectiveness"])
                    assert result["exposure"].values[index] != result["exposure"].values[index]
                else:
                    np.testing.assert_allclose(result["exposure"].values[index], exposure)
                    np.testing.assert_allclose(
                        selected["effectiveness"], increment.sum() / exposure, rtol=3e-5, atol=3e-5
                    )


@pytest.mark.parametrize("grouped", [False, True])
def test_contributions_match_closed_form_for_every_draw(grouped):
    data = _data(grouped=grouped)
    model = _model(data)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users")

    assert list(result.channel.values) == list(_labels(data))
    assert list(result["channel_type"].values) == ["media", "media", "organic_media", "treatment", "treatment"]
    assert result["reference_response"].dims == ("chain", "draw")
    assert result["incremental_response"].dims == ("chain", "draw", "channel")
    assert list(result["period"].values) == [1, 2, 3]
    assert list(result["response_period"].values) == [1, 2, 3]
    assert result.attrs["group"] == "posterior"
    assert result.attrs["treatment_baselines"] == "min"
    baselines = _baselines(data)
    np.testing.assert_allclose(result["treatment_baseline"].values[3:], [baselines["price"], baselines["promotion"]])
    assert np.isnan(result["treatment_baseline"].values[:3]).all()
    _assert_closed_form(result, data, results, baselines=baselines)


def test_contributions_select_channels_in_requested_order_and_keep_the_full_baseline():
    data = _data()
    model = _model(data)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users", channels=["promotion", "Email"])
    full = contributions(model, results, quantity="expected_users")

    assert list(result.channel.values) == ["promotion", "Email"]
    assert list(result["channel_type"].values) == ["treatment", "organic_media"]
    np.testing.assert_allclose(result["baseline_response"], full["baseline_response"])
    np.testing.assert_allclose(
        result["incremental_response"], full["incremental_response"].sel(channel=["promotion", "Email"])
    )
    _assert_closed_form(result, data, results, baselines=_baselines(data))


@pytest.mark.parametrize(
    "rule, expected_attr, overrides",
    [
        ("max", "max", {}),
        ({"price": 12.0}, "custom", {"price": 12.0}),
        ({"price": "max", "promotion": 0.5}, "custom", {"price": "max", "promotion": 0.5}),
    ],
)
def test_contributions_apply_treatment_baseline_rules(rule, expected_attr, overrides):
    data = _data()
    model = _model(data)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users", treatment_baselines=rule)

    base = "max" if rule == "max" else "min"
    baselines = _baselines(data, base)
    for name, level in overrides.items():
        baselines[name] = _baselines(data, level)[name] if isinstance(level, str) else level
    assert result.attrs["treatment_baselines"] == expected_attr
    np.testing.assert_allclose(result["treatment_baseline"].values[3:], [baselines["price"], baselines["promotion"]])
    _assert_closed_form(result, data, results, baselines=baselines)


@pytest.mark.parametrize(
    "rule, message",
    [
        ("median", "must be 'min', 'max', or a mapping"),
        (0.0, "must be 'min', 'max', or a mapping"),
        ({"discount": 1.0}, "unknown treatments"),
        ({"price": "mean"}, "must be 'min', 'max', or a finite number"),
        ({"price": True}, "must be 'min', 'max', or a finite number"),
        ({"price": float("nan")}, "must be 'min', 'max', or a finite number"),
    ],
)
def test_contributions_reject_invalid_treatment_baselines(rule, message):
    data = _data()
    with pytest.raises(ValueError, match=message):
        contributions(_model(data), _results(data), quantity="expected_users", treatment_baselines=rule)


def test_contributions_require_an_input_family_to_remove():
    frame = pl.DataFrame({"week": [1, 2, 3], "outcome": [1.0, 2.0, 3.0], "control": [0.5, 0.25, 0.75]})
    data = prepare_data(frame, time="week", outcome="outcome", controls=["control"])

    def transformed(controls, coefficient):
        expected = 1.0 + coefficient * controls[..., 0]
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 1.0]], dtype=np.float32)}, data=data)
    with pytest.raises(ValueError, match="requires media, reach and frequency, organic media"):
        contributions(model, results, quantity="expected_users")


def test_contributions_reject_treatment_baselines_without_treatments():
    frame = pl.DataFrame({"week": [1, 2, 3], "outcome": [1.0, 2.0, 3.0], "video": [1.0, 2.0, 3.0]})
    data = prepare_data(frame, time="week", outcome="outcome", media=["video"])

    def transformed(media, coefficient):
        expected = 1.0 + coefficient * media[..., 0]
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 1.0]], dtype=np.float32)}, data=data)
    with pytest.raises(ValueError, match="unknown treatments"):
        contributions(model, results, quantity="expected_users", treatment_baselines={"price": 1.0})


def test_contributions_reject_labels_shared_across_families():
    frame = pl.DataFrame(
        {
            "week": [1, 2, 3],
            "outcome": [1.0, 2.0, 3.0],
            "video": [1.0, 2.0, 3.0],
            "email": [2.0, 1.0, 2.0],
        }
    )
    data = prepare_data(
        frame,
        time="week",
        outcome="outcome",
        media=["video"],
        channels=["Email"],
        organic_media=["email"],
        organic_channels=["Email"],
    )

    def transformed(media, organic_media, coefficient):
        expected = coefficient * (media[..., 0] + organic_media[..., 0])
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 1.0]], dtype=np.float32)}, data=data)
    with pytest.raises(ValueError, match="unique across input families"):
        contributions(model, results, quantity="expected_users")


@pytest.mark.parametrize("channels", [[], "Email", ["Email", "Email"], ["Radio"], [1], ["Email", None]])
def test_contributions_reject_invalid_channel_selections(channels):
    data = _data()
    with pytest.raises(ValueError, match="channels must"):
        contributions(_model(data), _results(data), quantity="expected_users", channels=channels)


def test_contributions_remove_inputs_in_selected_periods_and_count_later_carryover():
    data = _data()
    model = _model(data)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users", periods=[1], response_periods=[2, 3])

    assert list(result["period"].values) == [1]
    assert list(result["response_period"].values) == [2, 3]
    np.testing.assert_allclose(result["exposure"].values[0], _exposure(data, "Online video", [1]))
    _assert_closed_form(result, data, results, baselines=_baselines(data), periods=[1], response_periods=[2, 3])


@pytest.mark.parametrize("by", ["time", "group", ("time", "group")])
def test_contributions_breakdowns_preserve_global_removals_and_total_shares(by):
    data = _data(grouped=True)
    model = _model(data)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users", by=by)
    totals = contributions(model, results, quantity="expected_users")

    retained = (by,) if isinstance(by, str) else tuple(by)
    assert result["incremental_response"].dims == ("chain", "draw", *retained, "channel")
    assert result["baseline_response"].dims == ("chain", "draw", *retained)
    assert result["contribution_share"].dims == ("chain", "draw", "channel")
    if "time" in retained:
        assert list(result["time"].values) == [1, 2, 3]
    if "group" in retained:
        assert list(result["group"].values) == ["east", "west"]
    observation_axes = tuple(name for name in retained)
    xr.testing.assert_allclose(
        result["incremental_response"].sum(observation_axes), totals["incremental_response"], rtol=3e-5, atol=3e-5
    )
    xr.testing.assert_allclose(result["contribution_share"], totals["contribution_share"], rtol=3e-5, atol=3e-5)
    _assert_closed_form(result, data, results, baselines=_baselines(data))


def test_contributions_reject_group_breakdown_for_national_data_and_unknown_axes():
    data = _data()
    with pytest.raises(ValueError, match="requires grouped data"):
        contributions(_model(data), _results(data), quantity="expected_users", by="group")
    with pytest.raises(ValueError, match="by must contain only time or group"):
        contributions(_model(data), _results(data), quantity="expected_users", by="channel")


def test_contributions_evaluate_new_data_with_training_baselines_and_scenario_exposures():
    data = _data()
    model = _model(data)
    results = _results(data)
    frame = pl.DataFrame(
        {
            "week": [4, 5, 6],
            "region": ["east", "east", "east"],
            "video": [2.0, 6.0, 1.0],
            "search": [3.0, 1.0, 5.0],
            "email": [4.0, 2.0, 2.0],
            "video_cost": [1.0, 3.0, 0.5],
            "search_cost": [0.75, 0.25, 1.25],
            "price": [15.0, 16.0, 17.0],
            "promotion": [1.0, 1.0, 1.0],
            "control": [1.0, 2.0, 3.0],
            "population": [100.0, 100.0, 100.0],
            "outcome": [80.0, 90.0, 100.0],
        }
    )
    scenario = prepare_data(
        frame.filter(pl.col("week") > 4),
        time="week",
        outcome="outcome",
        population="population",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        channels=["Online video", "Paid search"],
        organic_media=["email"],
        organic_channels=["Email"],
        treatments=["price", "promotion"],
        controls=["control"],
        media_history=frame.filter(pl.col("week") == 4),
    )

    result = contributions(model, results, quantity="expected_users", new_data=scenario)

    assert list(result["period"].values) == [5, 6]
    training_levels = _baselines(data)
    np.testing.assert_allclose(
        result["treatment_baseline"].values[3:], [training_levels["price"], training_levels["promotion"]]
    )
    later = frame.filter(pl.col("week") > 4)
    np.testing.assert_allclose(result["exposure"].values[0], later["video"].sum())
    np.testing.assert_allclose(result["exposure"].values[2], later["email"].sum())
    _assert_closed_form(result, scenario, results, baselines=training_levels)


@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_contributions_select_prior_or_posterior_draws(group):
    data = _data()
    model = _model(data)
    results = _results(data, group=group)

    result = contributions(model, results, quantity="expected_users", group=group)

    assert result.attrs["group"] == group
    _assert_closed_form(result, data, results, group=group, baselines=_baselines(data))
    other = "posterior" if group == "prior" else "prior"
    with pytest.raises(ValueError):
        contributions(model, results, quantity="expected_users", group=other)


def test_contributions_batches_produce_the_same_values():
    data = _data(grouped=True)
    model = _model(data)
    results = _results(data)

    single = contributions(model, results, quantity="expected_users", batch_size=1)
    default = contributions(model, results, quantity="expected_users")

    # Different batch sizes change how XLA groups reductions across parameter draws,
    # so per-channel increments only agree up to ordinary float32 reduction noise.
    xr.testing.assert_allclose(single, default, rtol=1e-4, atol=1e-4)


def test_contributions_reuse_fitted_scaling_with_population_adjustment():
    data = _data(grouped=True)
    scaling = fit_data_scaling(
        data, media_method="median", adjust_population=True, scale_treatments=True, scale_controls=True
    )
    model = _model(data, scaling=scaling)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users", treatment_baselines={"price": 12.0})

    baselines = _baselines(data, "min", price=12.0)
    np.testing.assert_allclose(result["exposure"].values[0], _exposure(data, "Online video"))
    _assert_closed_form(result, data, results, baselines=baselines, scaling=scaling)


def test_contributions_restore_original_outcome_units_with_fitted_outcome_scaling():
    data = _data(grouped=True)
    scaling = fit_data_scaling(data, media_method="median", scale_outcome="population", adjust_population=True)
    model = _model(data, scaling=scaling)
    results = _results(data)

    result = contributions(model, results, quantity="expected_users")
    plain = contributions(_model(data), results, quantity="expected_users")

    assert result.attrs["response_units"] == "original outcome units"
    assert not np.allclose(result["reference_response"], plain["reference_response"])
    _assert_closed_form(result, data, results, baselines=_baselines(data), scaling=scaling)


def _rf_data():
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3],
            "audience": [3.0, 4.0, 6.0, 8.0],
            "frequency": [1.0, 2.0, 3.0, 4.0],
            "newsletter_reach": [2.0, 2.0, 4.0, 2.0],
            "newsletter_frequency": [1.0, 1.5, 1.0, 2.0],
            "outcome": [10.0, 20.0, 30.0, 40.0],
        }
    )
    return prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        outcome="outcome",
        reach=["audience"],
        media_frequency=["frequency"],
        rf_channels=["Video"],
        organic_reach=["newsletter_reach"],
        organic_frequency=["newsletter_frequency"],
        organic_rf_channels=["Newsletter"],
        media_history=frame.filter(pl.col("week") == 0),
    )


def _rf_response(values, coefficient):
    paid = (
        values["reach"][:, 0] * values["media_frequency"][:, 0] / (1.0 + values["media_frequency"][:, 0])
        + 0.1 * values["media_frequency"][:, 0]
    )
    organic = values["organic_reach"][:, 0] * jnp.log1p(values["organic_frequency"][:, 0])
    carried = paid[1:] + 0.5 * paid[:-1]
    response = 3.0 + coefficient**2 * carried + coefficient * organic[1:]
    return response


def test_contributions_remove_reach_and_measure_impressions_for_rf_families():
    data = _rf_data()

    def transformed(reach, media_frequency, organic_reach, organic_frequency, coefficient):
        values = {
            "reach": reach,
            "media_frequency": media_frequency,
            "organic_reach": organic_reach,
            "organic_frequency": organic_frequency,
        }
        expected = _rf_response(values, coefficient)
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results(
        {"coefficient": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)},
        coords={"chain": [4, 8], "draw": [10, 30]},
    )

    result = contributions(model, results, quantity="expected_users", periods=[1, 2])

    assert list(result.channel.values) == ["Video", "Newsletter"]
    assert list(result["channel_type"].values) == ["reach_frequency", "organic_reach_frequency"]
    impressions = data.arrays["reach"][1:3, 0] * data.arrays["media_frequency"][1:3, 0]
    organic_impressions = data.arrays["organic_reach"][1:3, 0] * data.arrays["organic_frequency"][1:3, 0]
    np.testing.assert_allclose(result["exposure"].values, [impressions.sum(), organic_impressions.sum()])
    assert np.isnan(result["treatment_baseline"].values).all()

    for chain in range(2):
        for draw in range(2):
            coefficient = results["posterior"]["coefficient"].values[chain, draw]
            reference = np.asarray(_rf_response(data.arrays, coefficient)).sum()
            removed = {name: value.copy() for name, value in data.arrays.items()}
            removed["reach"][1:3, 0] = 0.0
            without_video = np.asarray(_rf_response(removed, coefficient)).sum()
            removed = {name: value.copy() for name, value in data.arrays.items()}
            removed["organic_reach"][1:3, 0] = 0.0
            without_newsletter = np.asarray(_rf_response(removed, coefficient)).sum()
            removed["reach"][1:3, 0] = 0.0
            baseline = np.asarray(_rf_response(removed, coefficient)).sum()
            selected = result.isel(chain=chain, draw=draw)
            np.testing.assert_allclose(selected["reference_response"], reference, rtol=3e-6)
            np.testing.assert_allclose(selected["baseline_response"], baseline, rtol=3e-6)
            np.testing.assert_allclose(
                selected["incremental_response"], [reference - without_video, reference - without_newsletter], rtol=3e-5
            )
            np.testing.assert_allclose(
                selected["effectiveness"],
                [
                    (reference - without_video) / impressions.sum(),
                    (reference - without_newsletter) / organic_impressions.sum(),
                ],
                rtol=3e-5,
            )


def test_contributions_promote_integer_exposures_when_removing_them():
    frame = pl.DataFrame(
        {
            "week": [1, 2, 3],
            "outcome": [1.0, 2.0, 3.0],
            "video": [4, 0, 2],
            "promotion": [0, 1, 0],
        }
    )
    data = prepare_data(frame, time="week", outcome="outcome", media=["video"], treatments=["promotion"])

    def transformed(media, treatments, coefficient):
        expected = 1.0 + coefficient * media[..., 0] + 0.5 * treatments[..., 0]
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 2.0]], dtype=np.float32)}, data=data)

    result = contributions(model, results, quantity="expected_users")

    for draw, coefficient in enumerate([0.5, 2.0]):
        selected = result.isel(chain=0, draw=draw)
        np.testing.assert_allclose(selected["incremental_response"], [coefficient * 6.0, 0.5 * 1.0], rtol=1e-6)
        np.testing.assert_allclose(selected["baseline_response"], 3.0, rtol=1e-6)


@pytest.mark.parametrize("quantity", ["exposure", "missing"])
def test_contributions_require_an_observation_shaped_transformed_quantity(quantity):
    data = _data()
    with pytest.raises(ValueError):
        contributions(_model(data), _results(data), quantity=quantity)


def test_contributions_require_transformed_parameters():
    data = _data()
    model = Model(parameters={"coefficient": Real((2,))}, log_density=lambda coefficient: 0.0, data=Data(data))
    with pytest.raises(ValueError, match="transformed_parameters"):
        contributions(model, _results(data), quantity="expected_users")


def test_contributions_name_the_removed_inputs_when_a_normalization_breaks():
    data = _data()

    def transformed(media, organic_media, treatments, controls, coefficient):
        # Normalizing by the scenario's own exposures divides by zero once a channel is removed.
        scale = media[1:].sum(axis=0)
        expected = 1.0 + (media[1:] / scale) @ coefficient + organic_media[1:, 0] + treatments[..., 0]
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    with pytest.raises(ValueError, match=r"Removing Online video, Paid search, the baseline produced.*reference"):
        contributions(model, _results(data), quantity="expected_users")


def test_contributions_report_a_nonfinite_reference_response():
    data = _data()

    def transformed(media, coefficient):
        expected = media[1:, 0] * coefficient[0] * jnp.nan
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    with pytest.raises(ValueError, match="reference response is nonfinite"):
        contributions(model, _results(data), quantity="expected_users")


def test_contributions_leave_unchanged_inputs_with_their_original_dtype():
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3],
            "video": [4.0, 1.0, 3.0, 5.0],
            "video_cost": [2.0, 0.5, 1.5, 2.5],
            "segment": [0, 1, 0, 1],
            "outcome": [10.0, 20.0, 30.0, 40.0],
        }
    )
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        outcome="outcome",
        media=["video"],
        spend=["video_cost"],
        channels=["Online video"],
        treatments=["segment"],
        media_history=frame.filter(pl.col("week") == 0),
    )

    # An integer treatment used as an index fails whenever a scenario promotes it to floating point.
    def transformed(media, treatments, coefficient):
        return {"expected_users": 1.0 + coefficient[treatments[:, 0]] * media[1:, 0]}

    model = Model(
        parameters={"coefficient": Real((2,))},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
        dims={"coefficient": ("level",)},
        coords={"level": [0, 1]},
    )
    draws = np.array([[[0.5, 1.0], [1.0, 0.5]]], dtype=np.float32)
    results = _collect_results(
        {"coefficient": draws}, data=data, dims={"coefficient": ("level",)}, coords={"level": [0, 1]}
    )

    selected = contributions(model, results, quantity="expected_users", channels=["Online video"])
    everything = contributions(model, results, quantity="expected_users")

    assert list(selected.channel.values) == ["Online video"]
    assert list(everything.channel.values) == ["Online video", "segment"]
    segment = np.asarray(data.arrays["treatments"][:, 0], dtype=np.intp)
    exposure = np.asarray(data.arrays["media"][1:, 0], dtype=np.float64)
    periods = len(data.time_values)
    for chain, draw in np.ndindex(draws.shape[:2]):
        coefficient = np.asarray(draws[chain, draw], dtype=np.float64)
        reference = float((1.0 + coefficient[segment] * exposure).sum())
        np.testing.assert_allclose(selected["reference_response"][chain, draw], reference, rtol=2e-6)
        np.testing.assert_allclose(selected["baseline_response"][chain, draw], periods, rtol=2e-6)
        np.testing.assert_allclose(selected["incremental_response"][chain, draw, 0], reference - periods, rtol=2e-6)
        treatment_lift = float((coefficient[segment] * exposure).sum() - (coefficient[0] * exposure).sum())
        np.testing.assert_allclose(everything["incremental_response"][chain, draw, 1], treatment_lift, rtol=2e-6)


def test_contributions_keep_boolean_treatments_boolean():
    frame = pl.DataFrame(
        {
            "week": [1, 2, 3],
            "video": [4.0, 1.0, 3.0],
            "promotion": [True, False, True],
            "outcome": [10.0, 20.0, 30.0],
        }
    )
    data = prepare_data(frame, time="week", outcome="outcome", media=["video"], treatments=["promotion"])

    # Negating a boolean treatment fails whenever a scenario promotes it to floating point.
    def transformed(media, treatments, coefficient):
        quiet = ~treatments[:, 0]
        expected = 1.0 + coefficient * media[:, 0] + 2.0 * quiet
        return {"expected_users": expected}

    model = Model(
        parameters={"coefficient": Real()},
        log_density=lambda expected_users: 0.0,
        data=Data(data),
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[0.5, 2.0]], dtype=np.float32)}, data=data)

    result = contributions(model, results, quantity="expected_users")

    # The minimum promotion level is False, so removing the treatment switches on the quiet effect in
    # the two promoted weeks, and the baseline keeps only the intercept and that effect.
    np.testing.assert_allclose(result["treatment_baseline"].values, [np.nan, 0.0])
    for draw, coefficient in enumerate([0.5, 2.0]):
        selected = result.isel(chain=0, draw=draw)
        np.testing.assert_allclose(selected["incremental_response"], [coefficient * 8.0, -4.0], rtol=1e-6)
        np.testing.assert_allclose(selected["baseline_response"], 9.0, rtol=1e-6)
