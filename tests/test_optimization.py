"""Tests for fixed-budget allocation with posterior response uncertainty."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr
from scipy.optimize import OptimizeResult

import mmmjax
import mmmjax.optimization as optimization
from mmmjax import MediaEffect, Model, Real, fit_data_scaling, optimize_budget, prepare_data
from mmmjax._results import _collect_results


def _problem(*, spend_unit=1.0, transformed=None):
    reference = np.array([6.0, 9.0, 5.0])
    exposure = np.array([0.4, 0.6])[:, None] * reference
    frame = pl.DataFrame(
        {
            "week": [1, 2],
            "video": exposure[:, 0],
            "search": exposure[:, 1],
            "email": exposure[:, 2],
            "video_cost": exposure[:, 0] * spend_unit,
            "search_cost": exposure[:, 1] * spend_unit,
            "email_cost": exposure[:, 2] * spend_unit,
        }
    )
    data = prepare_data(
        frame,
        time="week",
        media=["video", "search", "email"],
        spend=["video_cost", "search_cost", "email_cost"],
    )
    curvature = np.array([[2.0, 0.6, 0.2], [0.6, 1.5, 0.3], [0.2, 0.3, 1.0]])

    def quadratic(media, coefficient):
        difference = media.sum(axis=0) - coefficient**2
        response = 1000.0 - 0.5 * difference @ jnp.asarray(curvature) @ difference
        return {"expected": jnp.full(media.shape[0], response / media.shape[0])}

    def density(expected):
        raise AssertionError("Budget allocation must not evaluate the log density")

    def generated(key, expected):
        raise AssertionError("Budget allocation must not generate observations")

    model = Model(
        {"coefficient": Real((3,))},
        density,
        generated,
        data=data,
        components=[],
        transformed_parameters=quadratic if transformed is None else transformed,
        dims={"coefficient": ("channel",)},
    )
    coefficients = np.array(
        [[[3.0, 4.0, 2.0], [2.0, 3.0, 3.0]], [[4.0, 2.0, 3.0], [3.0, 3.0, 2.0]]],
        dtype=np.float32,
    )
    results = _collect_results(
        {"coefficient": coefficients},
        data=data,
        dims={"coefficient": ("channel",)},
        coords={"chain": [3, 9], "draw": [10, 30]},
    )
    return model, results, curvature


def _optimize(model, results, **kwargs):
    options = {
        "budget": 20.0,
        "quantity": "expected",
        "spend_to_media": "proportional",
        "bounds": (0.0, 40.0),
    }
    options.update(kwargs)
    return optimize_budget(model, results, **options)


def _response(results, curvature, allocation):
    difference = np.asarray(allocation) - results["posterior"]["coefficient"].values.astype(float) ** 2
    return 1000.0 - 0.5 * np.einsum("...i,ij,...j->...", difference, curvature, difference)


@pytest.mark.parametrize("options", [{}, {"budget": None}])
def test_optimize_budget_defaults_to_reference_total(options):
    model, results, _ = _problem()
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        spend_to_media="proportional",
        bounds=(0.0, 40.0),
        **options,
    )
    explicit = _optimize(model, results, budget=20.0)
    xr.testing.assert_identical(allocation, explicit)


@pytest.mark.parametrize("use_new_data", [False, True])
def test_optimize_budget_infers_total_only_from_selected_channels_and_spending_periods(use_new_data):
    model, results, _ = _problem()
    frame = pl.DataFrame(
        {
            "week": [1, 2],
            "video": [4.0, 8.0],
            "search": [10.0, 20.0],
            "email": [6.0, 12.0],
            "video_cost": [2.0, 4.0],
            "search_cost": [5.0, 10.0],
            "email_cost": [3.0, 6.0],
        }
    )
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        spend_to_media="proportional",
        bounds=(0.0, 40.0),
        channels=["search", "video"],
        spend_periods=[2],
        response_periods=[1, 2],
        new_data=frame if use_new_data else None,
    )
    expected = 14.0 if use_new_data else 9.0
    assert allocation.attrs["budget"] == pytest.approx(expected)
    assert allocation.attrs["reference_budget"] == allocation.attrs["budget"]
    np.testing.assert_allclose(allocation["spend"].sum("channel"), expected, rtol=2e-6)
    np.testing.assert_allclose(allocation["initial_spend"].sum(), expected, rtol=2e-6)


def test_optimize_budget_does_not_adjust_inferred_total_to_infeasible_bounds():
    model, results, _ = _problem()
    with pytest.raises(ValueError, match="budget must lie between"):
        _optimize(model, results, budget=None, bounds=(0.0, 2.0))


@pytest.mark.parametrize("budget", [20.0, 35.0])
@pytest.mark.parametrize("spend_unit", [1.0, 1_000_000.0])
def test_optimize_budget_maximizes_joint_posterior_response(budget, spend_unit):
    model, results, curvature = _problem(spend_unit=spend_unit)
    original_results = results.copy(deep=True)
    original_inputs = {name: np.asarray(value).copy() for name, value in model.data.values.items()}

    allocation = _optimize(model, results, budget=budget * spend_unit, bounds=(0.0, 40.0 * spend_unit))

    target = (results["posterior"]["coefficient"].values.astype(float) ** 2).mean(axis=(0, 1))
    direction = np.linalg.solve(curvature, np.ones(3))
    expected = target - direction * (target.sum() - budget) / direction.sum()
    optimized = allocation["spend"].sel(allocation="optimized").values / spend_unit
    np.testing.assert_allclose(optimized, expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(optimized.sum(), budget, rtol=2e-6)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="reference"), np.array([6, 9, 5]) * spend_unit)

    assert "optimize_budget" in mmmjax.__all__
    assert allocation["spend"].dims == ("allocation", "channel")
    assert allocation["response"].dims == ("chain", "draw", "allocation")
    assert allocation["response_change"].dims == ("chain", "draw")
    assert allocation["initial_spend"].dims == ("channel",)
    np.testing.assert_array_equal(allocation.allocation, ["reference", "optimized"])
    np.testing.assert_array_equal(allocation.channel, ["video", "search", "email"])
    np.testing.assert_array_equal(allocation.chain, [3, 9])
    np.testing.assert_array_equal(allocation.draw, [10, 30])
    np.testing.assert_array_equal(allocation.spend_period, [1, 2])
    np.testing.assert_array_equal(allocation.response_period, [1, 2])
    reference_response = _response(results, curvature, [6, 9, 5])
    optimized_response = _response(results, curvature, optimized)
    np.testing.assert_allclose(allocation["response"].sel(allocation="reference"), reference_response, atol=1e-4)
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), optimized_response, atol=1e-4)
    np.testing.assert_allclose(allocation["response_change"], optimized_response - reference_response, atol=2e-4)
    assert allocation.attrs["budget"] == budget * spend_unit
    assert allocation.attrs["reference_budget"] == pytest.approx(20.0 * spend_unit)
    assert allocation.attrs["solver"] == "SLSQP"
    assert allocation.attrs["success"]
    assert isinstance(allocation.attrs["iterations"], int)
    xr.testing.assert_identical(results, original_results)
    for name, original in original_inputs.items():
        np.testing.assert_array_equal(model.data.values[name], original)


def test_optimize_budget_enforces_channel_bounds():
    model, results, _ = _problem()
    allocation = _optimize(
        model,
        results,
        bounds={"email": (0.0, 20.0), "video": (10.0, 12.0), "search": (0.0, 6.0)},
    )
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [10.0, 6.0, 4.0], atol=2e-4)
    np.testing.assert_array_equal(allocation["lower_bound"], [10, 0, 0])
    np.testing.assert_array_equal(allocation["upper_bound"], [12, 6, 20])
    initial = allocation["initial_spend"].values
    np.testing.assert_allclose(initial.sum(), 20.0)
    assert np.all(initial >= allocation["lower_bound"].values)
    assert np.all(initial <= allocation["upper_bound"].values)


def test_optimize_budget_preserves_selection_order_and_fixed_channels():
    model, results, curvature = _problem()
    allocation = _optimize(
        model,
        results,
        budget=15.0,
        channels=["search", "video"],
        bounds={"video": (0.0, 15.0), "search": (0.0, 15.0)},
        initial_spend={"video": 10.0, "search": 5.0},
        batch_size=3,
    )
    np.testing.assert_array_equal(allocation.channel, ["search", "video"])
    np.testing.assert_array_equal(allocation["initial_spend"], [5.0, 10.0])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [7.13043478, 7.86956522], atol=2e-3)
    optimized = allocation["spend"].sel(allocation="optimized").values
    expected = _response(results, curvature, [optimized[1], optimized[0], 5.0])
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), expected, atol=1e-4)
    assert allocation.attrs["reference_budget"] == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"budget": 12.0, "channels": ["search"]}, [12.0]),
        ({"bounds": {"video": (3.0, 3.0), "search": (7.0, 7.0), "email": (10.0, 10.0)}}, [3.0, 7.0, 10.0]),
        ({"bounds": {"video": (4.0, 4.0), "search": (0.0, 20.0), "email": (6.0, 6.0)}}, [4.0, 10.0, 6.0]),
        ({"bounds": (0.0, 20 / 3)}, [20 / 3, 20 / 3, 20 / 3]),
        ({"bounds": (20 / 3, 40.0)}, [20 / 3, 20 / 3, 20 / 3]),
        (
            {
                "budget": 1074.35,
                "channels": ["video", "search"],
                "bounds": {"video": (505.61, 505.61), "search": (568.74, 568.74)},
            },
            [505.61, 568.74],
        ),
    ],
)
def test_optimize_budget_evaluates_unique_feasible_allocations_without_a_solver(monkeypatch, options, expected):
    model, results, _ = _problem()

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("A uniquely determined allocation does not require optimization")

    monkeypatch.setattr(optimization, "minimize", unexpected_solver)
    allocation = _optimize(model, results, **options)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), expected)
    assert allocation.attrs["iterations"] == 0
    assert np.all(np.isfinite(allocation["response"].values))


@pytest.mark.parametrize("budget", [0.0, -1.0, np.inf, np.nan, True, [20.0]])
def test_optimize_budget_rejects_invalid_total_budgets(budget):
    model, results, _ = _problem()
    with pytest.raises((TypeError, ValueError)):
        _optimize(model, results, budget=budget)


@pytest.mark.parametrize(
    "bounds",
    [
        (-1.0, 40.0),
        (10.0, 5.0),
        (0.0, np.inf),
        (np.nan, 40.0),
        (True, 40.0),
        (0.0,),
        (0.0, 20.0, 40.0),
        {"video": (0.0, 40.0)},
        {"video": (0.0, 40.0), "search": (0.0, 40.0), "unknown": (0.0, 40.0)},
        {"video": (0.0, 40.0), "search": (0.0, 40.0), "email": (0.0, 40.0), "extra": (0.0, 40.0)},
        (10.0, 40.0),
        (0.0, 5.0),
    ],
)
def test_optimize_budget_rejects_invalid_or_infeasible_bounds(bounds):
    model, results, _ = _problem()
    with pytest.raises((TypeError, ValueError)):
        _optimize(model, results, bounds=bounds)


@pytest.mark.parametrize(
    "initial_spend",
    [
        {"video": 6.0, "search": 9.0},
        {"video": 6.0, "search": 9.0, "email": 5.0, "extra": 0.0},
        {"video": 6.0, "search": 9.0, "email": 4.0},
        {"video": -1.0, "search": 16.0, "email": 5.0},
        {"video": np.nan, "search": 9.0, "email": 5.0},
        {"video": np.inf, "search": 9.0, "email": 5.0},
        {"video": True, "search": 14.0, "email": 5.0},
        [6.0, 9.0, 5.0],
    ],
)
def test_optimize_budget_rejects_invalid_initial_allocations(initial_spend):
    model, results, _ = _problem()
    with pytest.raises((TypeError, ValueError)):
        _optimize(model, results, initial_spend=initial_spend)


def test_optimize_budget_rejects_initial_allocations_outside_channel_bounds():
    model, results, _ = _problem()
    with pytest.raises(ValueError):
        _optimize(
            model,
            results,
            bounds={"video": (0.0, 5.0), "search": (0.0, 20.0), "email": (0.0, 20.0)},
            initial_spend={"video": 6.0, "search": 9.0, "email": 5.0},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("maxiter", 0), ("maxiter", 1.5), ("maxiter", True), ("tolerance", 0.0), ("tolerance", np.nan)],
)
def test_optimize_budget_rejects_invalid_solver_controls(name, value):
    model, results, _ = _problem()
    with pytest.raises((TypeError, ValueError)):
        _optimize(model, results, **{name: value})


def test_optimize_budget_reports_solver_failure(monkeypatch):
    model, results, _ = _problem()

    def failed_solver(fun, x0, **kwargs):
        return OptimizeResult(x=x0, success=False, message="Iteration limit reached", nit=1)

    monkeypatch.setattr(optimization, "minimize", failed_solver)
    with pytest.raises(RuntimeError, match="Iteration limit"):
        _optimize(model, results)


def test_optimize_budget_checks_final_feasibility_even_when_solver_reports_success(monkeypatch):
    model, results, _ = _problem()

    def infeasible_solver(fun, x0, **kwargs):
        return OptimizeResult(x=np.zeros_like(x0), success=True, message="Finished", nit=1, nfev=1)

    monkeypatch.setattr(optimization, "minimize", infeasible_solver)
    with pytest.raises(RuntimeError):
        _optimize(model, results)


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_optimize_budget_rejects_nonfinite_objectives_and_gradients(bad_gradient):
    def transformed(media, coefficient):
        value = jnp.sqrt(media.sum() - 20.0) if bad_gradient else jnp.asarray(jnp.nan)
        return {"expected": jnp.full(media.shape[0], value + 0.0 * coefficient.sum())}

    model, results, _ = _problem(transformed=transformed)
    with pytest.raises(ValueError):
        _optimize(model, results)


def test_optimize_budget_ignores_large_terms_constant_under_the_budget_constraint():
    frame = pl.DataFrame({"week": [1], "video": [0.5], "search": [0.5]})
    data = prepare_data(frame, time="week", media=["video", "search"], spend=["video", "search"])

    def transformed(media, spend, coefficient):
        response = 1000.0 * spend.sum(axis=-1) + coefficient * media[..., 0] - media[..., 0] ** 2
        return {"expected": response}

    model = Model(
        {"coefficient": Real()},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
    )
    results = _collect_results({"coefficient": np.array([[2.0]], dtype=np.float32)}, data=data)
    allocation = _optimize(model, results, budget=1.0, bounds=(0.0, 1.0))
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [1.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(allocation["response_change"], 0.25, atol=2e-4)


def test_optimize_budget_uses_separate_spending_and_carryover_measurement_periods():
    frame = pl.DataFrame(
        {
            "week": [0, 1, 2, 3],
            "video": [8.0, 4.0, 2.0, 5.0],
            "search": [10.0, 6.0, 3.0, 7.0],
            "video_cost": [8.0, 4.0, 2.0, 5.0],
            "search_cost": [10.0, 6.0, 3.0, 7.0],
        }
    )
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        media_history=frame.filter(pl.col("week") == 0),
    )

    def transformed(media, coefficient):
        carried = media[1:] + media[:-1] * jnp.array([0.5, 0.25])
        return {"expected": jnp.log1p(carried) @ coefficient}

    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
        dims={"coefficient": ("channel",)},
    )
    coefficient = np.array([[[1.0, 3.0], [3.0, 5.0]]], dtype=np.float32)
    results = _collect_results({"coefficient": coefficient}, data=data, dims={"coefficient": ("channel",)})
    allocation = _optimize(model, results, budget=10.0, bounds=(0.0, 10.0), spend_periods=[1], response_periods=[2])
    optimized = allocation["spend"].sel(allocation="optimized").values
    np.testing.assert_allclose(optimized, [14 / 3, 16 / 3], rtol=2e-3, atol=1e-3)
    np.testing.assert_array_equal(allocation.spend_period, [1])
    np.testing.assert_array_equal(allocation.response_period, [2])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="reference"), [4, 6])
    for name, spend in [("reference", np.array([4.0, 6.0])), ("optimized", optimized)]:
        expected = np.einsum("i,...i->...", np.log1p(np.array([2.0, 3.0]) + [0.5, 0.25] * spend), coefficient)
        np.testing.assert_allclose(allocation["response"].sel(allocation=name), expected, rtol=2e-6)


@pytest.mark.parametrize("infer_budget", [False, True])
def test_optimize_budget_recomputes_media_components_with_fitted_group_scaling(infer_budget):
    rows = []
    exposures = np.array([[8.0, 9.0], [1.0, 7.0], [4.0, 2.0], [0.0, 6.0], [8.0, 1.0]])
    for week, exposure in enumerate(exposures):
        for group, population, multiplier in [("east", 100.0, 1.0), ("west", 250.0, 2.0)]:
            rows.append(
                {
                    "week": week,
                    "region": group,
                    "population": population,
                    "video": exposure[0] * multiplier,
                    "search": exposure[1] * multiplier,
                    "video_cost": exposure[0] * multiplier / 2,
                    "search_cost": exposure[1] * multiplier / 3,
                }
            )
    frame = pl.DataFrame(rows)
    data = prepare_data(
        frame.filter(pl.col("week") > 0),
        time="week",
        groups=["region"],
        population="population",
        media=["video", "search"],
        spend=["video_cost", "search_cost"],
        media_history=frame.filter(pl.col("week") == 0),
    )
    scaling = fit_data_scaling(data, adjust_population=True)
    model = Model(
        {},
        lambda expected: expected.sum(),
        data=data,
        components=[MediaEffect(max_lag=1)],
        transformed_parameters=lambda paid_media_total: {"expected": paid_media_total},
        scaling=scaling,
    )
    parameters = {
        "paid_media_coefficient": [[0.8, 1.2], [1.0, 0.9], [1.2, 1.4], [0.9, 1.1]],
        "paid_media_retention": [[0.3, 0.6], [0.2, 0.4], [0.4, 0.7], [0.25, 0.5]],
        "paid_media_half_saturation": [[0.8, 1.2], [1.0, 0.9], [0.7, 1.1], [0.9, 1.0]],
        "paid_media_slope": np.ones((4, 2)),
    }
    posterior = {name: np.asarray(value, dtype=np.float32).reshape(2, 2, 2) for name, value in parameters.items()}
    results = _collect_results(posterior, data=data, dims=dict.fromkeys(parameters, ("channel",)))
    reference = data.arrays["spend"].sum(axis=(0, 1))
    budget = float(reference.sum())
    bounds = (0.15 * budget, 0.85 * budget)
    allocation = _optimize(model, results, budget=None if infer_budget else budget, bounds=bounds, batch_size=3)
    assert allocation.attrs["budget"] == pytest.approx(budget)
    optimized = allocation["spend"].sel(allocation="optimized").values

    flattened = {name: jnp.asarray(value.reshape(4, 2)) for name, value in posterior.items()}
    evaluate = jax.jit(
        lambda inputs: jax.vmap(lambda draw: model._evaluate_quantities(inputs, draw)["expected"].sum())(flattened)
    )

    def direct_response(spend):
        arrays = {name: value.copy() for name, value in data.arrays.items()}
        multiplier = np.asarray(spend) / reference
        arrays["spend"] *= multiplier
        arrays["media"][1:] *= multiplier
        inputs, _ = model._prepare_data(replace(data, arrays=arrays))
        return np.asarray(evaluate(inputs)).reshape(2, 2)

    grid = np.linspace(*bounds, 129)
    grid_means = np.array([direct_response([video, budget - video]).mean() for video in grid])
    expected = direct_response(optimized)
    expected_reference = direct_response(reference)
    grid_step = grid[1] - grid[0]
    assert abs(optimized[0] - grid[np.argmax(grid_means)]) <= 2 * grid_step
    assert expected.mean() >= grid_means.max() - 2e-5
    assert expected.mean() >= expected_reference.mean() - 2e-5
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), expected, rtol=2e-6)
    np.testing.assert_allclose(allocation["response"].sel(allocation="reference"), expected_reference, rtol=2e-6)
    np.testing.assert_allclose(allocation["response_change"], expected - expected_reference, atol=3e-6)
