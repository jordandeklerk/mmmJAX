"""Fixed-budget allocation using posterior expected responses."""

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.optimize import approx_fprime, minimize

from mmmjax.model import Model
from mmmjax.response import _prepare_response

__all__ = ["optimize_budget"]


def optimize_budget(
    model: Model,
    results: xr.DataTree,
    *,
    budget: float | None = None,
    quantity: str,
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike] = "proportional",
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    spend_constraint_lower: float | Sequence[float] | None = None,
    spend_constraint_upper: float | Sequence[float] | None = None,
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    initial_spend: Mapping[str, float] | None = None,
    batch_size: int = 64,
    maxiter: int = 200,
    tolerance: float = 1e-6,
) -> xr.Dataset:
    """Allocate a fixed budget to maximize the posterior mean response.

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
        The objective maximizes the posterior mean of the total over selected
        measurement periods and groups.
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
        reference proportions adjusted to the budget and bounds.
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
        - **lower_bound**, **upper_bound**, and **initial_spend** record constraints
          and the starting allocation.
        - **spend_period** and **response_period** record selected dates.

        A single allocation maximizes the posterior mean, not each draw
        separately. If the reference total differs from ``budget``, the
        comparison also reflects the change in total spending. Responses
        retain the quantity's units and receive no inverse scaling.

    Raises
    ------
    ValueError
        Inputs are invalid, constraints are infeasible, or model evaluation
        produces nonfinite responses or gradients.
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

    start = _initial_allocation(initial_spend, labels, reference, budget, lower, upper)
    indices = jnp.asarray(context.indices)
    baseline = context.reference_spend
    initial = baseline.at[indices].set(jnp.asarray(start * budget, dtype=baseline.dtype))
    if not np.isfinite(np.asarray(initial)).all():
        raise ValueError("The budget is too large for the model precision. Change spending units")

    free = np.flatnonzero(upper > lower)
    optimized = start.copy()
    iterations = 0
    evaluations = 0

    if len(free) > 1 and lower.sum() < 1 - 1e-12 and upper.sum() > 1 + 1e-12:
        free_indices = jnp.asarray(free)
        template = jnp.asarray(start, dtype=baseline.dtype)

        def loss(shares: jax.Array, reference_shares: jax.Array) -> jax.Array:
            selected = template.at[free_indices].set(shares)
            allocation = baseline.at[indices].set(selected * budget)
            comparison = template.at[free_indices].set(reference_shares)
            reference_allocation = baseline.at[indices].set(comparison * budget)
            _, change = context.evaluator.paired_evaluation(allocation, reference_allocation)
            return -jnp.mean(change)

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
                    "The response or its gradient is nonfinite. "
                    "Check the conversion, transformed quantity, posterior draws, and spending bounds"
                )

            return value_host, gradient_host

        # Scale at the reference-proportioned feasible allocation, not at a
        # user-supplied zero-spend starting point with singular derivatives.
        scale_point = _initial_allocation(None, labels, reference, budget, lower, upper)[free]
        _, scale_gradient = compiled(jnp.asarray(scale_point, dtype=baseline.dtype), initial_shares)
        initial_gradient = np.asarray(scale_gradient, dtype=np.float64)
        if not np.isfinite(initial_gradient).all():
            raise ValueError("The response gradient is nonfinite at the reference-proportioned allocation")

        # A common derivative changes only the total budget, which is fixed.
        # Remove that affine term before scaling by sensitivity to reallocation.
        common_gradient = float(initial_gradient.mean())
        scale = float(np.max(np.abs(initial_gradient - common_gradient))) or 1.0

        def objective(shares: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            value, gradient = evaluate(shares)
            value -= common_gradient * (shares.sum() - start[free].sum())
            return value / scale, (gradient - common_gradient) / scale

        fixed_total = float(start.sum() - start[free].sum())
        result = minimize(
            objective,
            start[free],
            method="SLSQP",
            jac=True,
            bounds=list(zip(lower[free], upper[free], strict=True)),
            constraints={
                "type": "eq",
                "fun": lambda shares: np.sum(shares) + fixed_total - 1,
                "jac": lambda shares: np.ones_like(shares),
            },
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
    ):
        raise RuntimeError("Budget optimization returned an allocation outside the budget or spending bounds")

    allocation = baseline.at[indices].set(jnp.asarray(optimized * budget, dtype=baseline.dtype))
    evaluate_final = jax.jit(lambda current: context.evaluator.paired_evaluation(current, baseline))
    reference_response, _ = evaluate_final(baseline)
    response, change = evaluate_final(allocation)
    reference_response, response, change = map(np.asarray, (reference_response, response, change))
    if not all(np.isfinite(value).all() for value in (reference_response, response, change)):
        raise ValueError("An allocation produced invalid media or a nonfinite response. Check the model and conversion")

    return xr.Dataset(
        {
            "spend": (("allocation", "channel"), np.stack((reference, optimized * budget))),
            "response": (("chain", "draw", "allocation"), np.stack((reference_response, response), axis=-1)),
            "response_change": (("chain", "draw"), change),
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
            "objective": "posterior mean response",
            "tolerance": tolerance,
        },
    )


def _positive_number(value: float, name: str) -> float:
    """Validate a finite positive scalar without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


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
