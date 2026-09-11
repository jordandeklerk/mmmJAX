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
from mmmjax import (
    MediaEffect,
    Model,
    Real,
    fit_data_scaling,
    hill_saturation,
    media_metrics,
    optimize_budget,
    prepare_data,
    root_saturation,
)
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


def test_optimize_budget_defaults_to_reference_total_and_proportional_media():
    model, results, _ = _problem()
    allocation = optimize_budget(model, results, quantity="expected")
    explicit_none = optimize_budget(
        model,
        results,
        quantity="expected",
        budget=None,
        bounds=None,
        spend_constraint_lower=None,
        spend_constraint_upper=None,
    )
    explicit = _optimize(model, results, budget=20.0, bounds=(0.0, 20.0))

    xr.testing.assert_identical(allocation, explicit)
    xr.testing.assert_identical(explicit_none, explicit)
    np.testing.assert_array_equal(allocation["lower_bound"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(allocation["upper_bound"], [20.0, 20.0, 20.0])


def test_optimize_budget_uses_overridden_total_for_default_bounds():
    model, results, _ = _problem()
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        budget=35.0,
        channels=["search", "video"],
        spend_periods=[2],
    )

    np.testing.assert_array_equal(allocation.channel, ["search", "video"])
    np.testing.assert_array_equal(allocation["lower_bound"], [0.0, 0.0])
    np.testing.assert_array_equal(allocation["upper_bound"], [35.0, 35.0])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized").sum(), 35.0, rtol=2e-6)
    assert allocation.attrs["budget"] == 35.0
    assert allocation.attrs["reference_budget"] == pytest.approx(9.0)


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
    np.testing.assert_array_equal(allocation["lower_bound"], [0.0, 0.0])
    np.testing.assert_allclose(allocation["upper_bound"], [expected, expected], rtol=2e-6)


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
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        bounds={"email": (0.0, 20.0), "video": (10.0, 12.0), "search": (0.0, 6.0)},
    )
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [10.0, 6.0, 4.0], atol=2e-4)
    np.testing.assert_array_equal(allocation["lower_bound"], [10, 0, 0])
    np.testing.assert_array_equal(allocation["upper_bound"], [12, 6, 20])
    initial = allocation["initial_spend"].values
    np.testing.assert_allclose(initial.sum(), 20.0)
    assert np.all(initial >= allocation["lower_bound"].values)
    assert np.all(initial <= allocation["upper_bound"].values)


def test_optimize_budget_preserves_total_with_fifty_percent_reference_bounds():
    model, results, _ = _problem(spend_unit=1_000.0)
    reference = {"video": 6_000.0, "search": 9_000.0, "email": 5_000.0}
    bounds = {channel: (0.5 * spend, 1.5 * spend) for channel, spend in reference.items()}

    allocation = optimize_budget(model, results, quantity="expected", bounds=bounds)
    relative = optimize_budget(
        model,
        results,
        quantity="expected",
        spend_constraint_lower=0.5,
        spend_constraint_upper=0.5,
    )

    xr.testing.assert_identical(relative, allocation)
    np.testing.assert_array_equal(allocation["lower_bound"], [3_000.0, 4_500.0, 2_500.0])
    np.testing.assert_array_equal(allocation["upper_bound"], [9_000.0, 13_500.0, 7_500.0])
    optimized = allocation["spend"].sel(allocation="optimized").values
    np.testing.assert_allclose(optimized.sum(), sum(reference.values()), rtol=2e-6)
    assert np.all(optimized >= allocation["lower_bound"].values)
    assert np.all(optimized <= allocation["upper_bound"].values)
    assert allocation.attrs["budget"] == pytest.approx(allocation.attrs["reference_budget"])
    assert float(allocation["response_change"].mean()) > 0


def test_optimize_budget_applies_asymmetric_constraint_vectors_in_channel_order():
    model, results, _ = _problem()
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        spend_constraint_lower=[0.0, 0.5, 1.0],
        spend_constraint_upper=[1.0, 0.0, 3.0],
    )

    np.testing.assert_array_equal(allocation.channel, ["video", "search", "email"])
    np.testing.assert_array_equal(allocation["lower_bound"], [6.0, 4.5, 0.0])
    np.testing.assert_array_equal(allocation["upper_bound"], [12.0, 9.0, 20.0])
    optimized = allocation["spend"].sel(allocation="optimized").values
    np.testing.assert_allclose(optimized.sum(), 20.0, rtol=2e-6)
    assert np.all(optimized >= allocation["lower_bound"].values)
    assert np.all(optimized <= allocation["upper_bound"].values)


@pytest.mark.parametrize("use_new_data", [False, True])
def test_optimize_budget_centers_relative_bounds_on_selected_reference_shares(use_new_data):
    model, results, _ = _problem()
    frame = pl.DataFrame(
        {
            "week": [1, 2],
            "video": [4.0, 8.0],
            "search": [4.0, 20.0],
            "email": [6.0, 12.0],
            "video_cost": [2.0, 4.0],
            "search_cost": [2.0, 10.0],
            "email_cost": [3.0, 6.0],
        }
    )
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        budget=35.0,
        channels=["search", "video"],
        spend_periods=[2],
        response_periods=[1, 2],
        new_data=frame if use_new_data else None,
        spend_constraint_lower=[0.2, 0.6],
        spend_constraint_upper=[0.1, 0.8],
    )

    reference = np.array([10.0, 4.0] if use_new_data else [5.4, 3.6])
    center = 35.0 * reference / reference.sum()
    np.testing.assert_array_equal(allocation.channel, ["search", "video"])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="reference"), reference)
    np.testing.assert_allclose(allocation["lower_bound"], [0.8, 0.4] * center)
    np.testing.assert_allclose(allocation["upper_bound"], [1.1, 1.8] * center)
    np.testing.assert_allclose(allocation["initial_spend"], center)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized").sum(), 35.0, rtol=2e-6)
    assert allocation.attrs["reference_budget"] == pytest.approx(reference.sum())


@pytest.mark.parametrize(
    ("constraints", "lower", "upper"),
    [
        ({"spend_constraint_lower": 0.5}, [3.0, 4.5, 2.5], [20.0, 20.0, 20.0]),
        ({"spend_constraint_upper": 1.5}, [0.0, 0.0, 0.0], [15.0, 22.5, 12.5]),
    ],
)
def test_optimize_budget_retains_default_bound_on_unspecified_constraint_side(constraints, lower, upper):
    model, results, _ = _problem()
    allocation = optimize_budget(model, results, quantity="expected", **constraints)

    np.testing.assert_array_equal(allocation["lower_bound"], lower)
    np.testing.assert_array_equal(allocation["upper_bound"], upper)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized").sum(), 20.0, rtol=2e-6)


@pytest.mark.parametrize("budget", [None, 35.0])
def test_optimize_budget_zero_relative_constraints_fix_reference_shares(monkeypatch, budget):
    model, results, _ = _problem()

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("Zero relative constraints determine the allocation")

    monkeypatch.setattr(optimization, "minimize", unexpected_solver)
    allocation = optimize_budget(
        model,
        results,
        quantity="expected",
        budget=budget,
        spend_constraint_lower=0.0,
        spend_constraint_upper=0.0,
    )

    center = np.array([6.0, 9.0, 5.0]) * ((20.0 if budget is None else budget) / 20.0)
    np.testing.assert_allclose(allocation["lower_bound"], center)
    np.testing.assert_allclose(allocation["upper_bound"], center)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), center)
    assert allocation.attrs["iterations"] == 0


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
    ("name", "value"),
    [
        ("spend_constraint_lower", -0.1),
        ("spend_constraint_upper", -0.1),
        ("spend_constraint_lower", 1.1),
        ("spend_constraint_lower", np.nan),
        ("spend_constraint_upper", np.inf),
        ("spend_constraint_lower", True),
        ("spend_constraint_upper", True),
        ("spend_constraint_lower", [0.5]),
        ("spend_constraint_upper", [[0.5, 0.5, 0.5]]),
        ("spend_constraint_upper", [0.5, np.nan, 0.5]),
        ("spend_constraint_lower", [0.5, True, 0.5]),
        ("spend_constraint_upper", [0.5, -0.1, 0.5]),
        ("spend_constraint_lower", [0.5, 1.1, 0.5]),
    ],
)
def test_optimize_budget_rejects_invalid_relative_constraints(name, value):
    model, results, _ = _problem()
    with pytest.raises(ValueError, match=name):
        optimize_budget(model, results, quantity="expected", **{name: value})


def test_optimize_budget_rejects_nonfinite_bounds_from_finite_relative_constraints():
    model, results, _ = _problem()
    with pytest.raises(ValueError, match="nonfinite bounds"):
        optimize_budget(model, results, quantity="expected", spend_constraint_upper=np.finfo(float).max)


@pytest.mark.parametrize("name", ["spend_constraint_lower", "spend_constraint_upper"])
def test_optimize_budget_requires_one_constraint_per_selected_channel(name):
    model, results, _ = _problem()
    with pytest.raises(ValueError, match=name):
        optimize_budget(
            model,
            results,
            quantity="expected",
            channels=["search", "video"],
            **{name: [0.5, 0.5, 0.5]},
        )


@pytest.mark.parametrize(
    "constraints",
    [
        {"spend_constraint_lower": 0.5},
        {"spend_constraint_upper": 0.5},
        {"spend_constraint_lower": 0.0, "spend_constraint_upper": 0.0},
    ],
)
def test_optimize_budget_rejects_combined_monetary_and_relative_bounds(constraints):
    model, results, _ = _problem()
    with pytest.raises(ValueError, match="bounds"):
        optimize_budget(model, results, quantity="expected", bounds=(0.0, 20.0), **constraints)


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


@pytest.mark.parametrize("saturation", [hill_saturation, root_saturation], ids=["hill", "root"])
@pytest.mark.parametrize("initial", [None, {"video": 0.0, "search": 1.0}, {"video": 1.0, "search": 0.0}])
def test_optimize_budget_recovers_concave_optimum_when_solver_reaches_zero(saturation, initial):
    frame = pl.DataFrame({"week": [1], "video": [0.5], "search": [0.5]})
    data = prepare_data(frame, time="week", media=["video", "search"], spend=["video", "search"])
    model = Model(
        {},
        lambda expected: expected.sum(),
        data=data,
        components=[MediaEffect(max_lag=0, saturation=saturation)],
        transformed_parameters=lambda paid_media_total: {"expected": paid_media_total},
    )
    parameters = {"paid_media_coefficient": [1.0, 3.0], "paid_media_retention": [0.0, 0.0]}
    if saturation is hill_saturation:
        parameters.update(
            paid_media_coefficient=[1.0, 6.0],
            paid_media_half_saturation=[1.0, 1.0],
            paid_media_slope=[0.5, 0.5],
        )
    else:
        parameters["paid_media_exponent"] = [0.5, 0.5]
    posterior = {name: jnp.array([[value]]) for name, value in parameters.items()}
    results = _collect_results(posterior, data=data, dims=dict.fromkeys(parameters, ("channel",)))

    allocation = optimize_budget(model, results, quantity="expected", initial_spend=initial)

    # Evaluate the closed-form concave curves independently on a fine allocation grid.
    grid = np.linspace(0.0, 1.0, 10_001)
    roots = np.sqrt(np.stack((grid, 1 - grid), axis=-1))
    responses = roots / (1 + roots) if saturation is hill_saturation else roots
    expected = responses @ np.asarray(parameters["paid_media_coefficient"])
    optimized = allocation["spend"].sel(allocation="optimized").values
    np.testing.assert_allclose(optimized, [grid[expected.argmax()], 1 - grid[expected.argmax()]], atol=2e-3)
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), expected.max(), atol=1e-5)
    np.testing.assert_array_equal(allocation["lower_bound"], [0.0, 0.0])
    np.testing.assert_allclose(optimized.sum(), 1.0, atol=1e-8)
    assert allocation.attrs["success"]


def test_optimize_budget_reallocates_integer_exposures_and_preserves_a_zero_optimum():
    frame = pl.DataFrame({"week": [1], "video": [1], "search": [1], "video_cost": [1.0], "search_cost": [1.0]})
    data = prepare_data(frame, time="week", media=["video", "search"], spend=["video_cost", "search_cost"])
    model = Model(
        {"coefficient": Real((2,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=lambda media, coefficient: {"expected": media @ coefficient},
        dims={"coefficient": ("channel",)},
    )
    results = _collect_results(
        {"coefficient": jnp.array([[[1.0, 3.0]]])}, data=data, dims={"coefficient": ("channel",)}
    )

    allocation = optimize_budget(model, results, quantity="expected")

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [0.0, 2.0], atol=1e-8)
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), 6.0, atol=1e-6)
    np.testing.assert_allclose(allocation["response_change"], 2.0, atol=1e-6)


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
    allocation = _optimize(
        model,
        results,
        budget=10.0,
        bounds=(0.0, 10.0),
        spend_periods=[1],
        response_periods=[2],
        include_metrics=True,
        incremental_increase=0.25,
    )
    optimized = allocation["spend"].sel(allocation="optimized").values
    np.testing.assert_allclose(optimized, [14 / 3, 16 / 3], rtol=2e-3, atol=1e-3)
    np.testing.assert_array_equal(allocation.spend_period, [1])
    np.testing.assert_array_equal(allocation.response_period, [2])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="reference"), [4, 6])
    for name, spend in [("reference", np.array([4.0, 6.0])), ("optimized", optimized)]:
        expected = np.einsum("i,...i->...", np.log1p(np.array([2.0, 3.0]) + [0.5, 0.25] * spend), coefficient)
        np.testing.assert_allclose(allocation["response"].sel(allocation=name), expected, rtol=2e-6)

        for index, channel in enumerate(["video", "search"]):
            metrics = allocation.sel(allocation=name, channel=channel)
            zero = spend.copy()
            zero[index] = 0.0
            increased = spend.copy()
            increased[index] += float(metrics["incremental_spend"])
            removed_response = np.einsum(
                "i,...i->...", np.log1p(np.array([2.0, 3.0]) + [0.5, 0.25] * zero), coefficient
            )
            increased_response = np.einsum(
                "i,...i->...", np.log1p(np.array([2.0, 3.0]) + [0.5, 0.25] * increased), coefficient
            )
            np.testing.assert_allclose(metrics["incremental_response"], expected - removed_response, atol=2e-6)
            np.testing.assert_allclose(metrics["marginal_response"], increased_response - expected, atol=2e-6)
            np.testing.assert_allclose(metrics["roi"], (expected - removed_response) / spend[index], atol=2e-6)
            np.testing.assert_allclose(
                metrics["marginal_roi"],
                (increased_response - expected) / float(metrics["incremental_spend"]),
                atol=2e-6,
            )


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
    allocation = _optimize(
        model,
        results,
        budget=None if infer_budget else budget,
        bounds=bounds,
        batch_size=3,
        include_metrics=True,
        incremental_increase=0.25,
    )
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

    for label, spend, expected_response in [
        ("reference", reference, expected_reference),
        ("optimized", optimized, expected),
    ]:
        for index, channel in enumerate(data.channels):
            metrics = allocation.sel(allocation=label, channel=channel)
            zero = spend.copy()
            zero[index] = 0.0
            increased = spend.copy()
            increased[index] += float(metrics["incremental_spend"])
            removed = expected_response - direct_response(zero)
            marginal = direct_response(increased) - expected_response

            np.testing.assert_allclose(metrics["incremental_response"], removed, atol=5e-6)
            np.testing.assert_allclose(metrics["marginal_response"], marginal, atol=5e-6)
            np.testing.assert_allclose(metrics["roi"], removed / spend[index], atol=2e-6)
            np.testing.assert_allclose(
                metrics["marginal_roi"], marginal / float(metrics["incremental_spend"]), atol=2e-6
            )


@pytest.mark.parametrize("channels", [None, ["email", "video"]])
@pytest.mark.parametrize("exposure_per_spend", [1.0, 2.0])
def test_optimize_budget_reports_paired_channel_metrics_at_each_joint_allocation(channels, exposure_per_spend):
    model, results, curvature = _problem()
    conversion = "proportional" if exposure_per_spend == 1.0 else lambda spend: exposure_per_spend * spend
    plain = _optimize(model, results, channels=channels, spend_to_media=conversion, batch_size=3)
    reported = _optimize(
        model,
        results,
        channels=channels,
        spend_to_media=conversion,
        include_metrics=True,
        incremental_increase=0.2,
        batch_size=3,
    )

    xr.testing.assert_equal(reported.drop_vars(set(reported.data_vars) - set(plain.data_vars)), plain)
    assert reported.attrs["incremental_increase"] == 0.2
    assert {name: value for name, value in reported.attrs.items() if name != "incremental_increase"} == plain.attrs
    assert reported["incremental_spend"].dims == ("allocation", "channel")
    names = ["video", "search", "email"]
    selected = names if channels is None else channels
    indices = [names.index(name) for name in selected]
    np.testing.assert_array_equal(reported.channel, selected)
    np.testing.assert_array_equal(reported.chain, [3, 9])
    np.testing.assert_array_equal(reported.draw, [10, 30])
    target = results["posterior"]["coefficient"].values.astype(float) ** 2
    reference_metrics = media_metrics(
        model,
        results,
        quantity="expected",
        channels=channels,
        spend_to_media=conversion,
        incremental_increase=0.2,
        batch_size=3,
    )
    for name in ["incremental_response", "roi", "marginal_response", "marginal_roi", "incremental_spend"]:
        xr.testing.assert_allclose(reported[name].sel(allocation="reference", drop=True), reference_metrics[name])

    for label in ["reference", "optimized"]:
        full_spend = np.array([6.0, 9.0, 5.0])
        spend = reported["spend"].sel(allocation=label).values
        full_spend[indices] = spend
        increase = reported["incremental_spend"].sel(allocation=label).values
        np.testing.assert_allclose(increase, 0.2 * spend, rtol=2e-6)

        # Closed-form differences retain cross-channel interactions without
        # evaluating the model or subtracting rounded response totals.
        gradient = (
            -exposure_per_spend
            * np.einsum("ij,...j->...i", curvature, exposure_per_spend * full_spend - target)[..., indices]
        )
        diagonal = exposure_per_spend**2 * np.diag(curvature)[indices]
        incremental = spend * gradient + 0.5 * spend**2 * diagonal
        marginal = increase * gradient - 0.5 * increase**2 * diagonal
        expected = {
            "incremental_response": incremental,
            "roi": incremental / spend,
            "marginal_response": marginal,
            "marginal_roi": marginal / increase,
        }

        for name, values in expected.items():
            assert reported[name].dims == ("chain", "draw", "allocation", "channel")
            np.testing.assert_allclose(reported[name].sel(allocation=label), values, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(
            reported["response"].sel(allocation=label),
            _response(results, curvature, full_spend * exposure_per_spend),
            atol=2e-4,
        )


def test_optimize_budget_reports_undefined_returns_for_zero_spend_without_rejecting_allocation():
    model, results, _ = _problem(transformed=lambda media, coefficient: {"expected": media @ coefficient**2})
    reported = _optimize(
        model,
        results,
        bounds={"video": (0.0, 0.0), "search": (20.0, 20.0), "email": (0.0, 0.0)},
        include_metrics=True,
    )

    optimized = reported.sel(allocation="optimized")
    np.testing.assert_array_equal(optimized["spend"], [0.0, 20.0, 0.0])
    inactive = optimized.sel(channel=["video", "email"])
    for name in ["incremental_response", "marginal_response", "incremental_spend"]:
        np.testing.assert_array_equal(inactive[name], np.zeros(inactive[name].shape))
    for name in ["roi", "marginal_roi"]:
        assert np.isnan(inactive[name]).all()
        np.testing.assert_allclose(
            optimized[name].sel(channel="search"),
            results["posterior"]["coefficient"].sel(channel="search") ** 2,
            rtol=2e-5,
        )
        assert np.isfinite(reported[name].sel(allocation="reference")).all()


def test_optimize_budget_does_not_evaluate_channel_metrics_unless_requested(monkeypatch):
    def unexpected_metrics(*args, **kwargs):
        raise AssertionError("Channel metrics should only run when requested")

    monkeypatch.setattr(optimization, "_allocation_metrics", unexpected_metrics)
    model, results, _ = _problem()
    default = _optimize(model, results, spend_constraint_lower=0.0, spend_constraint_upper=0.0, bounds=None)
    explicit = _optimize(
        model,
        results,
        include_metrics=False,
        spend_constraint_lower=0.0,
        spend_constraint_upper=0.0,
        bounds=None,
    )

    xr.testing.assert_identical(default, explicit)
    assert "roi" not in default
    assert "incremental_increase" not in default.attrs


@pytest.mark.parametrize("include_metrics", [None, 0, 1, "yes"])
def test_optimize_budget_rejects_invalid_include_metrics(include_metrics):
    model, results, _ = _problem()
    with pytest.raises((TypeError, ValueError), match="include_metrics"):
        _optimize(model, results, include_metrics=include_metrics)


@pytest.mark.parametrize("increase", [0.0, -0.1, np.nan, np.inf, True, "0.01"])
def test_optimize_budget_rejects_invalid_metric_increases(increase):
    model, results, _ = _problem()
    with pytest.raises(ValueError, match="incremental_increase"):
        _optimize(model, results, include_metrics=True, incremental_increase=increase)


def test_optimize_budget_rejects_unrepresentable_positive_spend_increases():
    model, results, _ = _problem()
    with pytest.raises(ValueError, match="too small for the model precision"):
        _optimize(
            model,
            results,
            include_metrics=True,
            incremental_increase=1e-12,
            spend_constraint_lower=0.0,
            spend_constraint_upper=0.0,
            bounds=None,
        )
