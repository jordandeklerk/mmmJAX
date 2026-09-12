"""Fixed-budget allocation using posterior expected responses."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from numbers import Integral, Real
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, approx_fprime, linprog, minimize

from mmmjax.model import Model
from mmmjax.response import _allocation_metrics, _prepare_response

__all__ = ["SpendConstraint", "optimize_budget"]


@dataclass(frozen=True)
class SpendConstraint:
    """Limit combined spending across a named set of channels.

    Parameters
    ----------
    name : str
        Unique label for this constraint in the allocation report.
    channels : sequence of str
        Channel names included in the combined spending total. All must
        be selected for optimization. Constraints may share channels.
    lower : float, default 0.0
        Minimum combined spending. Set equal to ``upper`` to fix the total.
    upper : float, optional
        Maximum combined spending. Defaults to the entire selected budget.
    units : {"spend", "share"}, default "spend"
        Use original spending units or fractions of the selected budget.
        With ``"share"``, ``upper=0.3`` caps the group at 30% of that budget.

    Examples
    --------
    Cap combined social spending while allowing redistribution between channels.

    .. ipython::

        In [1]: from mmmjax import SpendConstraint
           ...: social = SpendConstraint(
           ...:     "social", ["Meta", "TikTok"], upper=0.3, units="share"
           ...: )
           ...: social.channels
    """

    name: str
    channels: Sequence[str]
    _: KW_ONLY
    lower: float = 0.0
    upper: float | None = None
    units: Literal["spend", "share"] = "spend"

    def __post_init__(self) -> None:
        """Validate limits and retain an immutable copy of channel names."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A spending constraint needs a nonempty name")
        if isinstance(self.channels, str) or not isinstance(self.channels, Sequence) or not self.channels:
            raise ValueError("Constraint channels must be a nonempty sequence of names")
        if any(not isinstance(name, str) or not name.strip() for name in self.channels):
            raise ValueError("Constraint channels must contain nonempty names")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Constraint channels must not contain duplicates")
        if self.units not in ("spend", "share"):
            raise ValueError("Constraint units must be spend or share")

        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if value is None and name == "upper":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("Constraint limits must be finite nonnegative numbers")
            if self.units == "share" and value > 1:
                raise ValueError("Constraint shares must lie between zero and one")
        if self.upper is not None and self.lower > self.upper:
            raise ValueError("The constraint lower limit must not exceed its upper limit")

        object.__setattr__(self, "channels", tuple(self.channels))


def optimize_budget(
    model: Model,
    results: xr.DataTree,
    *,
    budget: float | None = None,
    quantity: str,
    utility_function: Callable[[jax.Array], ArrayLike] | None = None,
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike] = "proportional",
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    spend_constraint_lower: float | Sequence[float] | None = None,
    spend_constraint_upper: float | Sequence[float] | None = None,
    constraints: Sequence[SpendConstraint] | None = None,
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    initial_spend: Mapping[str, float] | None = None,
    include_metrics: bool = False,
    incremental_increase: float = 0.01,
    batch_size: int = 64,
    maxiter: int = 200,
    tolerance: float = 1e-6,
) -> xr.Dataset:
    """Allocate a fixed budget using posterior responses and a chosen objective.

    Optimize selected channels jointly using the full model. Retain their
    reference spending proportions across selected periods and groups.
    By default, redistribute the same total spending across those channels.
    Other spending and earlier media history stay fixed. The result is a
    local constrained solution and can depend on the starting allocation.

    Parameters
    ----------
    model : Model
        Prepared model with paired media and spend inputs.
    results : xarray.DataTree
        Results containing the model's constrained posterior draws.
    budget : float, optional
        Positive total spending for the selected channels and spending
        periods, in original spend units. Defaults to their total reference
        spending, using ``new_data`` when supplied.
    quantity : str
        Key in the mapping returned by ``transformed_parameters``, such as
        ``"expected_revenue"``. Its value must contain one expected response
        per observation in the desired reporting units. It is recomputed for
        each allocation and posterior draw and need not be stored in ``results``.
        Responses are totaled over selected measurement periods and groups.
    utility_function : callable, optional
        Differentiable JAX function receiving total expected responses with
        shape ``(chain, draw)`` and returning a floating-point scalar to
        maximize. Defaults to the posterior mean. Custom functions can
        penalize uncertainty. Inputs are absolute responses in the quantity's
        units, not changes from the reference allocation.
    spend_to_media : {"proportional"} or callable, default "proportional"
        By default, exposure scales with spending at each period and group,
        retaining reference exposure per unit spend. A differentiable JAX
        callable instead receives raw spending in model channel order and
        returns exposures of the same shape.
    bounds : tuple of float or mapping of str to tuple of float, optional
        Finite nonnegative lower and upper spending limits in original units.
        Supply one pair for all selected channels or one pair per channel name.
        Cannot be combined with percentage constraints. Without either form,
        each channel may receive zero through the total budget.
    spend_constraint_lower : float or sequence of float, optional
        Allowed fractional decrease from each channel's share of ``budget``
        using reference spending proportions. Use ``0.5`` for a 50% decrease
        across all channels or one value per selected channel in ``channels``
        order. Values must be between zero and one. Omit for a zero lower limit.
    spend_constraint_upper : float or sequence of float, optional
        Allowed fractional increase from the same reference-proportioned
        budget. Use ``0.5`` for a 50% increase or one value per selected channel.
        Values must be finite and nonnegative. Omit for an upper limit equal
        to the total budget. Lists use data channel order if ``channels`` is omitted.
    constraints : sequence of SpendConstraint, optional
        Limits on combined spending across named channel groups. These apply
        alongside the total budget and individual channel limits. Overlapping
        groups are allowed. Omit for individual limits only.
    channels : sequence of str, optional
        Paid-media channels to optimize. Defaults to all. Each needs positive
        reference spending to define its allocation across periods and groups.
    new_data : dataframe-like or PreparedData, optional
        Reference observations. Omit to use stored observations. The model's
        fitted scales are reused.
    spend_periods : sequence, optional
        Time labels whose spending changes. Defaults to all supplied periods.
    response_periods : sequence, optional
        Time labels whose responses count. Defaults to all supplied periods.
        Include later dates to measure carryover. No periods are added.
    initial_spend : mapping of str to float, optional
        Feasible starting spend for each selected channel. Defaults to
        reference proportions adjusted to the budget, bounds, and constraints.
    include_metrics : bool, default False
        Also report channel incremental response, ROI, and marginal ROI at
        both allocations. Requires additional evaluations after optimization.
    incremental_increase : float, default 0.01
        Positive fractional spend increase for marginal ROI when
        ``include_metrics=True``. The default measures a 1% increase.
    batch_size : int, default 64
        Maximum posterior draws evaluated together.
    maxiter : int, default 200
        Maximum solver iterations.
    tolerance : float, default 1e-6
        Solver stopping tolerance for budget shares and the scaled objective.

    Returns
    -------
    xarray.Dataset
        Labeled allocations and responses, retaining posterior uncertainty.

        - **spend** contains reference and optimized channel budgets.
        - **response** contains their total responses for every chain and draw.
        - **response_change** contains paired optimized-minus-reference responses.
        - **utility** contains the objective value for each allocation.
        - **lower_bound**, **upper_bound**, and **initial_spend** record constraints
          and the starting allocation.
        - **spend_period** and **response_period** record selected dates.

        When group constraints are supplied, **constraint_spend** and
        **constraint_satisfied** describe both allocations. **constraint_lower_bound**
        and **constraint_upper_bound** use original spend units, and
        **constraint_channels** identifies each group's channels. The reference
        allocation need not satisfy the new limits.

        With ``include_metrics=True``, **incremental_response** measures
        response lost by removing a channel's spending, and **roi** divides
        it by that spending. **marginal_response** and **marginal_roi** measure
        an increase using **incremental_spend**. Response and ROI arrays retain
        chain, draw, allocation, and channel axes. Ratios use the quantity's units
        per unit spend and are undefined (``NaN``) at zero spending.
        Channel removal and increase scenarios hold other channels at the
        allocation being evaluated and are not restricted by optimization
        bounds. Channel effects need not add up when channels interact.

        A single allocation maximizes the chosen utility across posterior
        draws. If the reference total differs from ``budget``, the
        comparison also reflects the change in total spending. Responses
        retain the quantity's units and receive no inverse scaling.

    Raises
    ------
    ValueError
        Inputs are invalid, constraints are infeasible, or model evaluation
        produces invalid responses, utility values, or gradients.
    RuntimeError
        The solver fails to converge or returns an infeasible allocation.
    """
    tolerance = _positive_number(tolerance, "tolerance")
    if tolerance >= 1:
        raise ValueError("tolerance must be smaller than one")
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if bounds is not None and (spend_constraint_lower is not None or spend_constraint_upper is not None):
        raise ValueError("Use either bounds or spend_constraint_lower and spend_constraint_upper, not both")
    if not isinstance(include_metrics, bool):
        raise ValueError("include_metrics must be a boolean")
    if utility_function is not None and not callable(utility_function):
        raise ValueError("utility_function must be callable")
    if include_metrics:
        incremental_increase = _positive_number(incremental_increase, "incremental_increase")

    context = _prepare_response(
        model,
        results,
        quantity=quantity,
        spend_to_media=spend_to_media,
        channels=channels,
        new_data=new_data,
        spend_periods=spend_periods,
        response_periods=response_periods,
        batch_size=batch_size,
    )

    reference = np.asarray(context.reference_spend, dtype=np.float64)[context.indices]
    if not np.isfinite(reference).all():
        raise ValueError("Reference spending is too large for the model precision. Change spending units")

    budget = _positive_number(float(reference.sum()) if budget is None else budget, "budget")
    labels = context.coords["channel"].tolist()
    limits = _spending_bounds((0.0, budget) if bounds is None else bounds, labels)

    center = budget * (reference / reference.sum())
    if spend_constraint_lower is not None:
        decrease = _spend_constraint(spend_constraint_lower, len(labels), "spend_constraint_lower")
        if np.any(decrease > 1):
            raise ValueError("spend_constraint_lower must be between zero and one")
        limits[:, 0] = (1 - decrease) * center

    if spend_constraint_upper is not None:
        increase = _spend_constraint(spend_constraint_upper, len(labels), "spend_constraint_upper")
        with np.errstate(over="ignore"):
            limits[:, 1] = (1 + increase) * center
        if not np.isfinite(limits[:, 1]).all():
            raise ValueError(
                "Spending constraints produce nonfinite bounds. Reduce the upper constraints or spending units"
            )

    # Shares keep the constraints independent of the currency and budget size.
    if np.any(limits[:, 0] > budget):
        raise ValueError("The budget must lie between the sums of the lower and upper bounds")
    lower = limits[:, 0] / budget
    upper = np.minimum(limits[:, 1], budget) / budget
    if lower.sum() > 1 + 1e-12 or upper.sum() < 1 - 1e-12:
        raise ValueError("The budget must lie between the sums of the lower and upper bounds")

    specifications, group_matrix, group_limits = _resolve_constraints(constraints, labels, budget)
    start = _initial_allocation(initial_spend, labels, reference, budget, lower, upper)
    if specifications:
        lower, upper = _tighten_bounds(lower, upper, group_matrix, group_limits)
        if initial_spend is None:
            start = _initial_allocation(None, labels, reference, budget, lower, upper)

    if specifications and not _constraints_satisfied(start, group_matrix, group_limits).all():
        if initial_spend is not None:
            raise ValueError("initial_spend must satisfy all group constraints")
        start = _feasible_group_allocation(start, lower, upper, group_matrix, group_limits)

    indices = jnp.asarray(context.indices)
    baseline = context.reference_spend
    initial = baseline.at[indices].set(jnp.asarray(start * budget, dtype=baseline.dtype))
    if not np.isfinite(np.asarray(initial)).all():
        raise ValueError("The budget is too large for the model precision. Change spending units")

    def utility(responses: jax.Array) -> jax.Array:
        value = jnp.asarray(jnp.mean(responses) if utility_function is None else utility_function(responses))
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("utility_function must return a floating-point scalar")

        # Custom reductions must not hide invalid draws by dropping or masking them.
        return jnp.where(jnp.all(jnp.isfinite(responses)), value, jnp.nan)

    free = np.flatnonzero(upper > lower)
    optimized = start.copy()
    iterations = 0
    evaluations = 0

    linear_constraints, equality_rank = _solver_constraints(start, free, group_matrix, group_limits)

    if len(free) > equality_rank and lower.sum() < 1 - 1e-12 and upper.sum() > 1 + 1e-12:
        free_indices = jnp.asarray(free)
        template = jnp.asarray(start, dtype=baseline.dtype)

        def loss(shares: jax.Array, reference_shares: jax.Array) -> jax.Array:
            selected = template.at[free_indices].set(shares)
            allocation = baseline.at[indices].set(selected * budget)
            comparison = template.at[free_indices].set(reference_shares)
            reference_allocation = baseline.at[indices].set(comparison * budget)

            if utility_function is None:
                _, change = context.evaluator.paired_evaluation(allocation, reference_allocation)
                return -jnp.mean(change)

            # Nonlinear utilities score absolute responses. Subtracting a scalar
            # reference score shifts the objective without changing its optimum.
            return utility(context.evaluator(reference_allocation)) - utility(context.evaluator(allocation))

        compiled = jax.jit(jax.value_and_grad(loss))
        difference = jax.jit(loss)
        initial_shares = jnp.asarray(start[free], dtype=baseline.dtype)
        difference_step = np.sqrt(np.finfo(baseline.dtype).eps)

        def evaluate(shares: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            current = jnp.asarray(shares, dtype=baseline.dtype)
            value, gradient = compiled(current, initial_shares)
            value_host = float(value)
            gradient_host = np.array(gradient, dtype=np.float64, copy=True)

            # Fractional response curves can have unbounded or masked derivatives
            # near zero. Measure one-sided changes without imposing a spend floor.
            near_zero = np.flatnonzero(shares <= difference_step)
            if near_zero.size:
                steps = np.minimum(difference_step, upper[free][near_zero] - shares[near_zero])
                backward = steps == 0
                steps[backward] = -np.minimum(
                    difference_step, shares[near_zero][backward] - lower[free][near_zero][backward]
                )

                def changed_loss(values: NDArray[np.float64]) -> float:
                    candidate = shares.copy()
                    candidate[near_zero] = values
                    return float(difference(jnp.asarray(candidate, dtype=baseline.dtype), current))

                gradient_host[near_zero] = approx_fprime(shares[near_zero], changed_loss, epsilon=steps)

            if not np.isfinite(value_host) or not np.isfinite(gradient_host).all():
                raise ValueError(
                    "The response, utility, or gradient is nonfinite. "
                    "Check the model, utility_function, conversion, posterior draws, and spending bounds"
                )

            return value_host, gradient_host

        # Group rules can move the feasible start onto zero-spend boundaries.
        # Apply the one-sided derivative fallback there before scaling.
        scale_point = _initial_allocation(None, labels, reference, budget, lower, upper)[free]
        if specifications:
            _, initial_gradient = evaluate(start[free])
        else:
            _, scale_gradient = compiled(jnp.asarray(scale_point, dtype=baseline.dtype), initial_shares)
            initial_gradient = np.asarray(scale_gradient, dtype=np.float64)
        if not np.isfinite(initial_gradient).all():
            raise ValueError("The objective gradient is nonfinite at the reference-proportioned allocation")

        # Fixed budget and group totals make their affine effects constant.
        # Remove those gradients before scaling sensitivity to reallocation.
        affine_gradient: float | NDArray[np.float64] = float(initial_gradient.mean())
        if specifications:
            equality_matrix = np.asarray(linear_constraints[0].A)
            affine_gradient = equality_matrix.T @ np.linalg.lstsq(equality_matrix.T, initial_gradient, rcond=None)[0]
        scale = float(np.max(np.abs(initial_gradient - affine_gradient))) or 1.0

        def objective(shares: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            value, gradient = evaluate(shares)
            value -= float(np.sum(affine_gradient * (shares - start[free])))
            return value / scale, (gradient - affine_gradient) / scale

        result = minimize(
            objective,
            start[free],
            method="SLSQP",
            jac=True,
            bounds=list(zip(lower[free], upper[free], strict=True)),
            constraints=linear_constraints,
            options={"maxiter": int(maxiter), "ftol": tolerance},
        )
        if not result.success:
            raise RuntimeError(f"Budget optimization did not converge. {result.message}")

        optimized[free] = result.x
        iterations = int(result.nit)
        evaluations = int(result.nfev)

    # Check feasibility independently of the solver's convergence flag.
    if (
        not np.isfinite(optimized).all()
        or abs(optimized.sum() - 1) > 1e-8
        or np.any(optimized < lower - 1e-8)
        or np.any(optimized > upper + 1e-8)
        or not _constraints_satisfied(optimized, group_matrix, group_limits).all()
    ):
        raise RuntimeError(
            "Budget optimization returned an allocation outside the budget, bounds, or group constraints"
        )

    allocation = baseline.at[indices].set(jnp.asarray(optimized * budget, dtype=baseline.dtype))
    evaluate_final = jax.jit(lambda current: context.evaluator.paired_evaluation(current, baseline))
    reference_response, _ = evaluate_final(baseline)
    response, change = evaluate_final(allocation)
    reference_response, response, change = map(np.asarray, (reference_response, response, change))
    if not all(np.isfinite(value).all() for value in (reference_response, response, change)):
        raise ValueError("An allocation produced invalid media or a nonfinite response. Check the model and conversion")

    evaluate_utility = jax.jit(utility)
    utilities = np.asarray([evaluate_utility(jnp.asarray(value)) for value in (reference_response, response)])
    if not np.isfinite(utilities).all():
        raise ValueError("An allocation produced a nonfinite utility. Check utility_function and the response units")

    report = xr.Dataset(
        {
            "spend": (("allocation", "channel"), np.stack((reference, optimized * budget))),
            "response": (("chain", "draw", "allocation"), np.stack((reference_response, response), axis=-1)),
            "response_change": (("chain", "draw"), change),
            "utility": ("allocation", utilities),
            "lower_bound": ("channel", limits[:, 0]),
            "upper_bound": ("channel", limits[:, 1]),
            "initial_spend": ("channel", start * budget),
        },
        coords={**context.coords, "allocation": ["reference", "optimized"]},
        attrs={
            **context.attrs,
            "budget": budget,
            "reference_budget": float(reference.sum()),
            "solver": "SLSQP",
            "success": True,
            "iterations": iterations,
            "function_evaluations": evaluations,
            "objective": "posterior mean response" if utility_function is None else "custom utility",
            "tolerance": tolerance,
        },
    )
    if specifications:
        shares = np.stack((reference / budget, optimized))
        report = report.assign_coords(constraint=[spec.name for spec in specifications])
        report["constraint_spend"] = (("allocation", "constraint"), (shares @ group_matrix.T) * budget)
        report["constraint_lower_bound"] = ("constraint", group_limits[:, 0] * budget)
        report["constraint_upper_bound"] = ("constraint", group_limits[:, 1] * budget)
        report["constraint_channels"] = (("constraint", "channel"), group_matrix.astype(bool))
        report["constraint_satisfied"] = (
            ("allocation", "constraint"),
            _constraints_satisfied(shares, group_matrix, group_limits),
        )

    if include_metrics:
        metrics = _allocation_metrics(
            context,
            np.stack((np.asarray(baseline), np.asarray(allocation))),
            allocation_labels=["reference", "optimized"],
            incremental_increase=incremental_increase,
        )
        for name in ("incremental_response", "roi", "marginal_response", "marginal_roi", "incremental_spend"):
            report[name] = metrics[name]
        report.attrs["incremental_increase"] = incremental_increase

    return report


def _positive_number(value: float, name: str) -> float:
    """Validate a finite positive scalar without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _resolve_constraints(
    constraints: Sequence[SpendConstraint] | None, labels: list[str], budget: float
) -> tuple[tuple[SpendConstraint, ...], NDArray[np.float64], NDArray[np.float64]]:
    """Resolve named groups to indicator rows and budget-share limits."""
    if constraints is None:
        constraints = ()
    if isinstance(constraints, str) or not isinstance(constraints, Sequence):
        raise ValueError("constraints must be a sequence of SpendConstraint objects")
    specifications = tuple(constraints)
    if any(not isinstance(spec, SpendConstraint) for spec in specifications):
        raise ValueError("constraints must contain only SpendConstraint objects")
    if len({spec.name for spec in specifications}) != len(specifications):
        raise ValueError("Spending constraint names must be unique")

    matrix = np.zeros((len(specifications), len(labels)), dtype=np.float64)
    limits = np.zeros((len(specifications), 2), dtype=np.float64)
    for index, spec in enumerate(specifications):
        if set(spec.channels) - set(labels):
            raise ValueError(f"Constraint {spec.name!r} contains channels not selected for optimization")
        matrix[index] = [name in spec.channels for name in labels]
        divisor = budget if spec.units == "spend" else 1.0
        limits[index, 0] = spec.lower / divisor
        limits[index, 1] = 1.0 if spec.upper is None else spec.upper / divisor

    if not np.isfinite(limits).all() or np.any(limits[:, 0] > np.minimum(limits[:, 1], 1.0)):
        raise ValueError("Group constraints cannot be satisfied with the selected budget")
    return specifications, matrix, limits


def _constraints_satisfied(
    shares: NDArray[np.float64], matrix: NDArray[np.float64], limits: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Check each group in normalized units independently of solver status."""
    values = shares @ matrix.T
    return np.isfinite(values) & (values >= limits[:, 0] - 1e-8) & (values <= limits[:, 1] + 1e-8)


def _tighten_bounds(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    matrix: NDArray[np.float64],
    limits: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate group and total limits, including channels forced to zero."""
    lower, upper = lower.copy(), upper.copy()
    rows = np.vstack((np.ones(len(lower)), matrix)).astype(bool)
    targets = np.vstack(([1.0, 1.0], limits))

    # Propagation is only preprocessing. Limit repeated tightening and leave
    # unresolved combinations to the joint linear feasibility problem.
    for _ in range(16):
        previous_lower, previous_upper = lower.copy(), upper.copy()
        for row, (minimum, maximum) in zip(rows, targets, strict=True):
            low, high = lower[row], upper[row]
            if low.sum() > maximum + 1e-12 or high.sum() < minimum - 1e-12:
                raise ValueError("The budget, channel bounds, and group constraints cannot be satisfied together")

            lower[row] = np.maximum(low, minimum - (high.sum() - high))
            upper[row] = np.minimum(high, maximum - (low.sum() - low))

        if np.any(lower > upper + 1e-12):
            raise ValueError("The budget, channel bounds, and group constraints cannot be satisfied together")
        # Collapse intervals determined to a single value up to arithmetic roundoff.
        fixed = upper - lower <= 1e-12
        lower[fixed] = upper[fixed] = (lower[fixed] + upper[fixed]) / 2
        lower, upper = np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)

        if max(np.max(lower - previous_lower), np.max(previous_upper - upper)) <= 1e-12:
            return lower, upper

    return lower, upper


def _feasible_group_allocation(
    start: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    matrix: NDArray[np.float64],
    limits: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Find a jointly feasible allocation minimizing absolute changes from the start."""
    size = len(start)
    identity = np.eye(size)
    zeros = np.zeros_like(matrix)
    inequalities = np.vstack(
        (
            np.hstack((matrix, zeros)),
            np.hstack((-matrix, zeros)),
            np.hstack((identity, -identity)),
            np.hstack((-identity, -identity)),
        )
    )
    targets = np.concatenate((limits[:, 1], -limits[:, 0], start, -start))
    result = linprog(
        np.concatenate((np.zeros(size), np.ones(size))),
        A_ub=inequalities,
        b_ub=targets,
        A_eq=np.concatenate((np.ones(size), np.zeros(size)))[None, :],
        b_eq=[1.0],
        bounds=[*zip(lower, upper, strict=True), *[(0.0, None)] * size],
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if result.status == 2:
        raise ValueError("The budget, channel bounds, and group constraints cannot be satisfied together")
    if not result.success:
        raise RuntimeError(f"Could not find a feasible starting allocation. {result.message}")

    feasible = np.asarray(result.x[:size], dtype=np.float64)
    if (
        not np.isfinite(feasible).all()
        or abs(feasible.sum() - 1) > 1e-8
        or np.any(feasible < lower - 1e-8)
        or np.any(feasible > upper + 1e-8)
        or not _constraints_satisfied(feasible, matrix, limits).all()
    ):
        raise RuntimeError("The starting allocation does not satisfy all spending constraints")
    return feasible


def _solver_constraints(
    start: NDArray[np.float64],
    free: NDArray[np.intp],
    matrix: NDArray[np.float64],
    limits: NDArray[np.float64],
) -> tuple[list[LinearConstraint], int]:
    """Substitute fixed channels and remove redundant equality rows."""
    if not len(free):
        return [], 0
    rows = np.vstack((np.ones(len(start)), matrix))
    targets = np.vstack(([1.0, 1.0], limits))
    fixed_values = start.copy()
    fixed_values[free] = 0
    targets = targets - (rows @ fixed_values)[:, None]
    rows = rows[:, free]

    equal = targets[:, 0] == targets[:, 1]
    independent: list[int] = []
    for index in np.flatnonzero(equal).tolist():
        if np.linalg.matrix_rank(rows[[*independent, index]]) > len(independent):
            independent.append(index)
    equality = rows[independent]
    result = [LinearConstraint(equality, targets[independent, 0], targets[independent, 1])]
    inequalities = ~equal & np.any(rows != 0, axis=1)
    if np.any(inequalities):
        result.append(LinearConstraint(rows[inequalities], targets[inequalities, 0], targets[inequalities, 1]))
    return result, len(independent)


def _spending_bounds(
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]], labels: list[str]
) -> NDArray[np.float64]:
    """Resolve explicit bounds in the selected channel order."""
    if isinstance(bounds, Mapping):
        if set(bounds) != set(labels):
            raise ValueError("bounds must contain exactly the selected channel names")
        values = [bounds[name] for name in labels]
    else:
        values = [bounds] * len(labels)

    try:
        array = np.asarray(values)
        contains_bool = any(isinstance(value, (bool, np.bool_)) for value in np.asarray(values, dtype=object).flat)
    except (TypeError, ValueError) as error:
        raise ValueError("bounds must supply a lower and upper spending limit for each channel") from error
    if array.shape != (len(labels), 2) or array.dtype.kind not in "fiu" or contains_bool:
        raise ValueError("bounds must supply a pair of real spending limits for each channel")

    limits = array.astype(np.float64)
    if not np.isfinite(limits).all() or np.any(limits < 0) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Spending bounds must be finite and nonnegative with lower limits no larger than upper limits")

    return limits


def _spend_constraint(value: float | Sequence[float], size: int, name: str) -> NDArray[np.float64]:
    """Validate a fractional change for all channels or one per selected channel."""
    try:
        array = np.asarray(value)
        contains_bool = any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number or one number per selected channel") from error

    if array.ndim > 1 or (array.ndim == 1 and array.size != size):
        raise ValueError(f"{name} must be a number or one number per selected channel")
    if array.dtype.kind not in "fiu" or contains_bool or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative numbers")

    return np.broadcast_to(array.astype(np.float64), (size,)).copy()


def _initial_allocation(
    supplied: Mapping[str, float] | None,
    labels: list[str],
    reference: NDArray[np.float64],
    budget: float,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a feasible starting allocation in budget shares."""
    start: NDArray[np.float64]
    if supplied is not None:
        if not isinstance(supplied, Mapping) or set(supplied) != set(labels):
            raise ValueError("initial_spend must contain exactly the selected channel names")

        values = np.asarray([supplied[name] for name in labels])
        if (
            values.shape != (len(labels),)
            or values.dtype.kind not in "fiu"
            or any(isinstance(value, (bool, np.bool_)) for value in supplied.values())
        ):
            raise ValueError("initial_spend must contain real spending amounts")

        start = values.astype(np.float64) / budget
        if (
            not np.isfinite(start).all()
            or abs(start.sum() - 1) > 1e-8
            or np.any(start < lower)
            or np.any(start > upper)
        ):
            raise ValueError("initial_spend must satisfy the budget and all spending bounds")

        return start

    start = np.clip(reference / reference.sum(), lower, upper)
    remaining = 1 - start.sum()
    capacity = upper - start if remaining > 0 else start - lower
    if capacity.sum() > 0:
        start = start + remaining * capacity / capacity.sum()

    return start
