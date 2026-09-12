"""Tests for joint channel spending constraints in budget optimization."""

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import xarray as xr
from scipy.optimize import OptimizeResult

import mmmjax.optimization as optimization
from mmmjax import Model, Real, SpendConstraint, optimize_budget, prepare_data, root_saturation
from mmmjax._results import _collect_results


def _problem(*, spend_unit=1.0, fail_evaluation=False):
    channels = ["video", "search", "email"]
    exposure = np.array([0.4, 0.6])[:, None] * np.array([6.0, 9.0, 5.0])
    frame = pl.DataFrame(
        {
            "week": [1, 2],
            **{name: exposure[:, i] for i, name in enumerate(channels)},
            **{f"{name}_cost": exposure[:, i] * spend_unit for i, name in enumerate(channels)},
        }
    )
    data = prepare_data(
        frame,
        time="week",
        media=channels,
        spend=[f"{name}_cost" for name in channels],
    )

    def transformed(media, target):
        if fail_evaluation:
            raise AssertionError("Infeasible constraints must be rejected before model evaluation")

        difference = media.sum(axis=0) - target
        response = 100.0 - 0.5 * jnp.sum(difference**2)
        return {"expected": jnp.full(media.shape[0], response / media.shape[0])}

    model = Model(
        {"target": Real((3,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
        dims={"target": ("channel",)},
    )
    targets = np.array(
        [[[11.0, 6.0, 2.0], [13.0, 6.0, 2.0]], [[12.0, 4.0, 2.0], [12.0, 8.0, 2.0]]],
        dtype=np.float32,
    )
    results = _collect_results({"target": targets}, data=data, dims={"target": ("channel",)})
    return model, results


def _optimize(model, results, **kwargs):
    return optimize_budget(model, results, quantity="expected", **kwargs)


def _response(results, allocation):
    difference = np.asarray(allocation) - results["posterior"]["target"].values.astype(float)
    return 100.0 - 0.5 * np.sum(difference**2, axis=-1)


def _four_channel_problem(*, fail_evaluation=False, transformed=None):
    channels = ["video", "streaming", "search", "email"]
    frame = pl.DataFrame({"week": [1, 2], **{name: [0.125, 0.125] for name in channels}})
    data = prepare_data(frame, time="week", media=channels, spend=channels)

    def quadratic(media, target):
        if fail_evaluation:
            raise AssertionError("Infeasible constraints must be rejected before model evaluation")

        response = -0.5 * jnp.sum((media.sum(axis=0) - target) ** 2)
        return {"expected": jnp.full(media.shape[0], response / media.shape[0])}

    model = Model(
        {"target": Real((4,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=quadratic if transformed is None else transformed,
        dims={"target": ("channel",)},
    )
    targets = np.full((1, 1, 4), 0.25, dtype=np.float32)
    results = _collect_results({"target": targets}, data=data, dims={"target": ("channel",)})
    return model, results


def test_spend_constraint_copies_channels_and_is_immutable():
    channels = ["video", "search"]
    constraint = SpendConstraint("awareness", channels, upper=12.0)
    channels.append("email")

    assert tuple(constraint.channels) == ("video", "search")
    with pytest.raises(FrozenInstanceError):
        constraint.upper = 15.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": " "},
        {"name": 1},
        {"channels": "video"},
        {"channels": []},
        {"channels": ["video", "video"]},
        {"channels": [""]},
        {"channels": [1]},
        {"lower": -1.0},
        {"lower": np.nan},
        {"lower": np.inf},
        {"lower": True},
        {"lower": [0.0]},
        {"upper": -1.0},
        {"upper": np.nan},
        {"upper": np.inf},
        {"upper": True},
        {"lower": 3.0, "upper": 2.0},
        {"units": "percent"},
        {"units": "share", "lower": 1.1},
        {"units": "share", "upper": 1.1},
    ],
)
def test_spend_constraint_rejects_invalid_declarations(kwargs):
    arguments = {"name": "awareness", "channels": ["video", "search"], **kwargs}

    with pytest.raises((TypeError, ValueError)):
        SpendConstraint(**arguments)


def test_optimize_budget_enforces_group_cap_and_reports_both_allocations():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["search", "video"], upper=12.0)],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [9.0, 3.0, 8.0], atol=5e-4)
    np.testing.assert_allclose(
        allocation["response"].sel(allocation="optimized"), _response(results, [9, 3, 8]), atol=1e-3
    )
    assert allocation["constraint_spend"].dims == ("allocation", "constraint")
    assert allocation["constraint_lower_bound"].dims == ("constraint",)
    assert allocation["constraint_upper_bound"].dims == ("constraint",)
    assert allocation["constraint_satisfied"].dims == ("allocation", "constraint")
    assert allocation["constraint_channels"].dims == ("constraint", "channel")
    assert allocation["constraint"].values.tolist() == ["awareness"]
    np.testing.assert_allclose(allocation["constraint_spend"], [[15.0], [12.0]], atol=2e-5)
    np.testing.assert_array_equal(allocation["constraint_lower_bound"], [0.0])
    np.testing.assert_array_equal(allocation["constraint_upper_bound"], [12.0])
    np.testing.assert_array_equal(allocation["constraint_satisfied"], [[False], [True]])
    np.testing.assert_array_equal(allocation["constraint_channels"], [[True, True, False]])


@pytest.mark.parametrize("spend_unit", [0.01, 1000.0])
def test_optimize_budget_group_shares_match_spend_limits_and_currency_scaling(spend_unit):
    model, results = _problem(spend_unit=spend_unit)
    shares = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=0.6, units="share")],
    )
    spending = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=12.0 * spend_unit)],
    )

    np.testing.assert_allclose(shares["spend"] / spend_unit, [[6, 9, 5], [9, 3, 8]], atol=8e-4)
    np.testing.assert_allclose(shares["spend"], spending["spend"], rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(shares["constraint_upper_bound"], 12.0 * spend_unit)
    np.testing.assert_allclose(shares["response"], spending["response"], atol=1e-3)


def test_optimize_budget_scales_group_shares_with_an_overridden_budget():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        budget=30.0,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=0.6, units="share")],
    )

    np.testing.assert_allclose(allocation["constraint_upper_bound"], 18.0)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [12.0, 6.0, 12.0], atol=8e-4)


def test_optimize_budget_applies_a_group_minimum_in_budget_share_units():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("lower_funnel", ["search", "email"], lower=0.6, units="share")],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [8.0, 8.0, 4.0], atol=5e-4)
    np.testing.assert_allclose(allocation["constraint_lower_bound"], 12.0)
    np.testing.assert_allclose(allocation["constraint_upper_bound"], 20.0)


def test_optimize_budget_preserves_explicit_group_limits_above_the_new_budget_in_reports():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        budget=10.0,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=100.0)],
    )

    np.testing.assert_allclose(allocation["constraint_upper_bound"], 100.0)
    np.testing.assert_allclose(allocation["constraint_spend"].sel(allocation="reference"), 15.0)
    np.testing.assert_array_equal(allocation["constraint_satisfied"], [[True], [True]])


def test_optimize_budget_enforces_overlapping_groups_in_declared_order():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[
            SpendConstraint("lower_funnel", ["search", "email"], lower=12.0),
            SpendConstraint("awareness", ["video", "search"], upper=12.0),
        ],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [8.0, 4.0, 8.0], atol=6e-4)
    assert allocation["constraint"].values.tolist() == ["lower_funnel", "awareness"]
    np.testing.assert_array_equal(allocation["constraint_channels"], [[False, True, True], [True, True, False]])
    np.testing.assert_allclose(allocation["constraint_lower_bound"], [12.0, 0.0])
    np.testing.assert_allclose(allocation["constraint_upper_bound"], [20.0, 12.0])
    assert allocation["constraint_satisfied"].sel(allocation="optimized").values.all()


@pytest.mark.parametrize("redundant", [False, True])
def test_optimize_budget_respects_group_equalities_including_redundant_rules(redundant):
    model, results = _problem()
    constraints = [SpendConstraint("awareness", ["video", "search"], lower=12.0, upper=12.0)]
    if redundant:
        constraints.extend(
            [
                SpendConstraint("awareness_duplicate", ["search", "video"], lower=12.0, upper=12.0),
                SpendConstraint("all", ["video", "search", "email"], lower=20.0, upper=20.0),
                SpendConstraint("email", ["email"], lower=8.0, upper=8.0),
            ]
        )

    allocation = _optimize(model, results, constraints=constraints, tolerance=1e-8)

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [9.0, 3.0, 8.0], atol=5e-4)
    assert allocation["constraint_satisfied"].sel(allocation="optimized").values.all()


def test_optimize_budget_group_equalities_can_determine_a_unique_allocation():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[
            SpendConstraint("awareness", ["video", "search"], lower=12.0, upper=12.0),
            SpendConstraint("lower_funnel", ["search", "email"], lower=11.0, upper=11.0),
        ],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [9.0, 3.0, 8.0], atol=3e-5)
    np.testing.assert_allclose(allocation["utility"], allocation["response"].mean(("chain", "draw")), atol=1e-4)


def test_optimize_budget_group_inequalities_can_determine_a_unique_allocation():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[
            SpendConstraint("awareness", ["video", "search"], upper=12.0),
            SpendConstraint("video", ["video"], lower=9.0),
            SpendConstraint("search", ["search"], lower=3.0),
            SpendConstraint("email", ["email"], lower=8.0),
        ],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [9.0, 3.0, 8.0], atol=3e-5)


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"bounds": {"video": (0, 7), "search": (0, 20), "email": (0, 20)}}, [7, 7, 6]),
        ({"spend_constraint_lower": 0.5, "spend_constraint_upper": 0.5}, [9, 5, 6]),
    ],
)
def test_optimize_budget_combines_group_rules_with_individual_limits(options, expected):
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=14.0)],
        **options,
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), expected, atol=7e-4)


def test_optimize_budget_group_rules_follow_selected_channel_order():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        channels=["search", "video"],
        constraints=[SpendConstraint("video_limit", ["video"], upper=8.0)],
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [7.0, 8.0], atol=5e-4)
    np.testing.assert_array_equal(allocation["constraint_channels"], [[False, True]])
    np.testing.assert_allclose(
        allocation["response"].sel(allocation="optimized"), _response(results, [8, 7, 5]), atol=1e-3
    )


def test_optimize_budget_repairs_the_implicit_start_to_satisfy_joint_constraints():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=12.0)],
    )
    start = allocation["initial_spend"].values

    assert np.all(start >= 0)
    assert start[:2].sum() <= 12.0 + 1e-5
    np.testing.assert_allclose(start.sum(), 20.0, atol=1e-5)


def test_optimize_budget_uses_linear_feasibility_for_overlapping_group_caps(monkeypatch):
    model, results = _four_channel_problem()
    feasibility_calls = []
    linear_solver = optimization.linprog

    def record_feasibility(*args, **kwargs):
        result = linear_solver(*args, **kwargs)
        feasibility_calls.append(result)
        return result

    monkeypatch.setattr(optimization, "linprog", record_feasibility)
    allocation = _optimize(
        model,
        results,
        constraints=[
            SpendConstraint("awareness", ["video", "streaming"], upper=0.4),
            SpendConstraint("online", ["streaming", "search"], upper=0.4),
        ],
    )

    assert feasibility_calls
    assert all(result.success for result in feasibility_calls)
    start = allocation["initial_spend"].values
    assert start[:2].sum() <= 0.4 + 1e-8
    assert start[1:3].sum() <= 0.4 + 1e-8
    np.testing.assert_allclose(start.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [0.25, 0.15, 0.25, 0.35], atol=2e-4)


@pytest.mark.parametrize(
    "constraints",
    [
        [
            SpendConstraint("awareness", ["video", "streaming"], upper=0.4),
            SpendConstraint("lower_funnel", ["search", "email"], upper=0.4),
        ],
        [
            SpendConstraint("awareness", ["video", "streaming"], lower=0.5, upper=0.5),
            SpendConstraint("conflicting", ["video", "streaming"], lower=0.500001, upper=0.500001),
        ],
    ],
    ids=["disjoint_caps", "conflicting_equalities"],
)
def test_optimize_budget_linear_feasibility_rejects_infeasible_groups_before_evaluation(monkeypatch, constraints):
    model, results = _four_channel_problem(fail_evaluation=True)
    feasibility_calls = []
    linear_solver = optimization.linprog

    def record_feasibility(*args, **kwargs):
        result = linear_solver(*args, **kwargs)
        feasibility_calls.append(result)
        return result

    monkeypatch.setattr(optimization, "linprog", record_feasibility)

    with pytest.raises(ValueError):
        _optimize(model, results, constraints=constraints)

    assert feasibility_calls
    assert feasibility_calls[-1].status == 2


def test_optimize_budget_allows_zero_spend_groups_with_fractional_response_curves():
    def transformed(media, target):
        return {"expected": jnp.sum(root_saturation(media, exponent=0.5) * target, axis=-1)}

    model, results = _four_channel_problem(transformed=transformed)
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "streaming"], upper=0.0)],
        include_metrics=True,
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [0.0, 0.0, 0.5, 0.5], atol=3e-5)
    np.testing.assert_allclose(allocation["response"].sel(allocation="optimized"), 0.5, atol=1e-6)
    assert np.isfinite(allocation["response"].values).all()
    assert np.isnan(allocation["roi"].sel(allocation="optimized", channel=["video", "streaming"]).values).all()


def test_optimize_budget_preserves_a_feasible_explicit_start():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=12.0)],
        initial_spend={"video": 5.0, "search": 6.0, "email": 9.0},
    )

    np.testing.assert_allclose(allocation["initial_spend"], [5.0, 6.0, 9.0])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [9.0, 3.0, 8.0], atol=7e-4)


def test_optimize_budget_accepts_an_explicit_start_on_a_derived_bound():
    model, results = _problem()
    allocation = _optimize(
        model,
        results,
        budget=20.0,
        channels=["video", "search"],
        constraints=[SpendConstraint("video", ["video"], upper=14.0)],
        initial_spend={"video": 14.0, "search": 6.0},
    )

    np.testing.assert_allclose(allocation["initial_spend"], [14.0, 6.0])
    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), [13.0, 7.0], atol=7e-4)


@pytest.mark.parametrize(
    "constraints, options",
    [
        ([SpendConstraint("awareness", ["video", "search"], lower=21.0)], {}),
        (
            [
                SpendConstraint("awareness", ["video", "search"], upper=12.0),
                SpendConstraint("email", ["email"], upper=7.0),
            ],
            {},
        ),
        (
            [
                SpendConstraint("awareness", ["video", "search"], lower=12.0, upper=12.0),
                SpendConstraint("duplicate", ["video", "search"], lower=13.0, upper=13.0),
            ],
            {},
        ),
        (
            [SpendConstraint("awareness", ["video", "search"], upper=12.0)],
            {"bounds": {"video": (0, 20), "search": (0, 20), "email": (0, 7)}},
        ),
        (
            [SpendConstraint("awareness", ["video", "search"], upper=12.0)],
            {"spend_constraint_upper": 0.5},
        ),
        (
            [SpendConstraint("awareness", ["video", "search"], upper=12.0)],
            {"initial_spend": {"video": 6.0, "search": 9.0, "email": 5.0}},
        ),
    ],
)
def test_optimize_budget_rejects_joint_infeasibility_before_model_evaluation(constraints, options):
    model, results = _problem(fail_evaluation=True)

    with pytest.raises(ValueError):
        _optimize(model, results, constraints=constraints, **options)


@pytest.mark.parametrize(
    "constraints",
    [
        "awareness",
        {"awareness": ["video", "search"]},
        [object()],
        [SpendConstraint("unknown", ["missing"])],
        [SpendConstraint("repeated", ["video"]), SpendConstraint("repeated", ["search"])],
    ],
)
def test_optimize_budget_rejects_invalid_constraint_collections(constraints):
    model, results = _problem(fail_evaluation=True)

    with pytest.raises((TypeError, ValueError)):
        _optimize(model, results, constraints=constraints)


def test_optimize_budget_rejects_groups_containing_unselected_channels():
    model, results = _problem(fail_evaluation=True)

    with pytest.raises(ValueError):
        _optimize(
            model,
            results,
            channels=["video", "search"],
            constraints=[SpendConstraint("all", ["video", "search", "email"], upper=20.0)],
        )


def test_optimize_budget_empty_constraints_preserve_existing_output():
    model, results = _problem()
    omitted = _optimize(model, results)
    empty = _optimize(model, results, constraints=[])

    xr.testing.assert_identical(omitted, empty)
    assert "constraint" not in omitted.dims
    assert not any(name.startswith("constraint_") for name in omitted.data_vars)


def test_optimize_budget_checks_group_feasibility_after_solver_success(monkeypatch):
    model, results = _problem()

    def infeasible_solver(fun, x0, **kwargs):
        return OptimizeResult(x=np.array([0.6, 0.3, 0.1]), success=True, message="Finished", nit=1, nfev=1)

    monkeypatch.setattr(optimization, "minimize", infeasible_solver)

    with pytest.raises(RuntimeError):
        _optimize(
            model,
            results,
            constraints=[SpendConstraint("awareness", ["video", "search"], upper=12.0)],
        )


def test_optimize_budget_scales_gradients_with_a_constant_group_response():
    channels = ["video", "streaming", "search", "email"]
    frame = pl.DataFrame(
        {
            "week": [1, 2],
            **{name: [amount / 2, amount / 2] for name, amount in zip(channels, [0.2, 0.2, 0.3, 0.3], strict=True)},
        }
    )
    data = prepare_data(frame, time="week", media=channels, spend=channels)

    def transformed(media, target):
        totals = media.sum(axis=0)
        return {"expected": jnp.stack([1000.0 * totals[:2].sum(), -0.5 * jnp.sum((totals - target) ** 2)])}

    model = Model(
        {"target": Real((4,))},
        lambda expected: expected.sum(),
        data=data,
        components=[],
        transformed_parameters=transformed,
        dims={"target": ("channel",)},
    )
    targets = np.array([[[0.1, 0.3, 0.2, 0.4]]], dtype=np.float32)
    results = _collect_results({"target": targets}, data=data, dims={"target": ("channel",)})
    allocation = _optimize(
        model,
        results,
        budget=1.0,
        constraints=[SpendConstraint("awareness", channels[:2], lower=0.4, upper=0.4)],
        initial_spend=dict(zip(channels, [0.2, 0.2, 0.3, 0.3], strict=True)),
    )

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), targets[0, 0], atol=2e-3)


def test_optimize_budget_combines_group_constraints_with_utility_and_channel_metrics():
    model, results = _problem()
    penalty = 0.2

    def utility(response):
        return jnp.mean(response) - penalty * jnp.var(response)

    allocation = _optimize(
        model,
        results,
        constraints=[SpendConstraint("awareness", ["video", "search"], upper=12.0)],
        utility_function=utility,
        include_metrics=True,
    )

    # The quadratic response makes this utility quadratic in spending too.
    # Solve its linear optimality equations with the active group cap.
    targets = results["posterior"]["target"].values.reshape(-1, 3).astype(float)
    centered = targets - targets.mean(axis=0)
    constants = -0.5 * np.sum(targets**2, axis=1)
    covariance = centered.T @ centered / len(targets)
    mixed = centered.T @ (constants - constants.mean()) / len(targets)
    curvature = np.eye(3) + 2.0 * penalty * covariance
    linear = targets.mean(axis=0) - 2.0 * penalty * mixed
    equations = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    system = np.block([[curvature, equations.T], [equations, np.zeros((2, 2))]])
    expected = np.linalg.solve(system, np.concatenate((linear, [20.0, 12.0])))[:3]

    np.testing.assert_allclose(allocation["spend"].sel(allocation="optimized"), expected, atol=7e-4)
    assert not np.allclose(expected, [9.0, 3.0, 8.0])
    assert allocation["roi"].dims == ("chain", "draw", "allocation", "channel")
    for label in ["reference", "optimized"]:
        spending = allocation["spend"].sel(allocation=label).values
        response = _response(results, spending)
        np.testing.assert_allclose(
            allocation["utility"].sel(allocation=label), response.mean() - penalty * response.var()
        )

        for i, channel in enumerate(allocation["channel"].values):
            removed = spending.copy()
            removed[i] = 0.0
            expected_roi = (response - _response(results, removed)) / spending[i]
            np.testing.assert_allclose(
                allocation["roi"].sel(allocation=label, channel=channel), expected_roi, atol=2e-4
            )
