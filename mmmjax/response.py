"""Prior and posterior media responses for spending and frequency scenarios."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax._results import _coordinates, _prepared_coordinates
from mmmjax.data import PreparedData
from mmmjax.model import Model, _ModelData
from mmmjax.sampling import _parameter_draws, _parameter_metadata, _result_data

__all__ = ["frequency_curves", "media_metrics", "response_curves"]

_ReachFrequencyConversion: TypeAlias = (
    Literal["reach", "frequency"] | Callable[[jax.Array], tuple[ArrayLike, ArrayLike]]
)


def response_curves(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    multipliers: Sequence[float] | ArrayLike,
    group: Literal["prior", "posterior"] = "posterior",
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike] = "proportional",
    spend_to_rf: _ReachFrequencyConversion = "reach",
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    by: str | Sequence[str] | None = None,
    batch_size: int = 64,
) -> xr.Dataset:
    """Evaluate paid-media spending curves using prior or posterior draws.

    Vary one channel's spending at a time, retaining its allocation across
    selected periods and groups. Spending and exposures outside those periods
    stay fixed. Evaluate the full model for every draw, summing responses
    by default or retaining time and group breakdowns. Include later
    measurement dates to count carryover. No observations are added automatically.

    Parameters
    ----------
    model : Model
        Prepared model with media and spend, reach/frequency and their spend,
        or both. Input families without spending stay fixed.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        Key in the mapping returned by ``transformed_parameters``, such as
        ``"expected_revenue"``. Its value must contain one expected response
        per observation in the desired reporting units. It is recomputed for
        each scenario and draw and need not be stored in ``results``.
        No inverse scaling is applied to this output.
    multipliers : array_like
        Distinct nonnegative spending multipliers. One retains reference
        spending and zero removes the channel's spending during ``spend_periods``.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Choose ``"prior"`` with results from
        ``sample_prior`` to inspect responses implied by the priors.
        No sampling is performed. Compare groups using separate calls.
    spend_to_media : {"proportional"} or callable, default "proportional"
        By default, exposure scales with spend at each period and group,
        retaining reference exposure per unit spend. A JAX-compatible function
        instead receives ordinary media spend in original units and its channel
        order, then returns nonnegative exposures of the same shape. Only
        exposures during ``spend_periods`` are replaced. Earlier history is
        excluded from the conversion.
    spend_to_rf : {"reach", "frequency"} or callable, default "reach"
        Scale reach with spending while keeping frequency fixed, or choose
        ``"frequency"`` to scale frequency at fixed reach. Both assume constant
        cost per impression. A JAX-compatible callable receives raw RF spending
        in ``rf_channels`` order and returns ``(reach, frequency)`` in original
        units, each with the same shape. Only spending periods are changed.
    channels : sequence of str, optional
        Channel labels to evaluate, in the desired order. Defaults to all
        paid channels with spending, ordinary media first and RF second.
        Names must be unique across both families. Each selected channel needs
        positive reference spending.
    new_data : dataframe-like or PreparedData, optional
        Reference observations using the model's columns and fitted scales.
        Omit to use stored observations. Supply ``PreparedData`` with
        ``media_history`` to include earlier exposures for new observations.
    spend_periods : sequence, optional
        Time labels whose spending changes. Defaults to all supplied modeling
        periods. Channel budgets cover only these dates, not earlier history.
    response_periods : sequence, optional
        Time labels whose outcomes count. Defaults to all supplied modeling
        periods. May include dates after spending ends to measure carryover.
    by : str or sequence of str, optional
        Retain ``"time"``, ``"group"``, or both in responses. Group labels
        come from ``prepare_data`` and require grouped data. Omit for totals.
        Retained axes follow time then group order.
    batch_size : int, default 64
        Maximum parameter draws evaluated together. Scenarios run sequentially.

    Returns
    -------
    xarray.Dataset
        Labeled curves preserving chain and draw coordinates. The ``group``
        attribute records which parameter draws were used.

        - **response** contains responses by channel and multiplier.
        - **incremental_response** subtracts the same channel's response
          with zero spending during ``spend_periods`` and other inputs fixed.
        - **spend** contains candidate totals during the spending periods
          in original spend units.
        - **reference_spend** contains the corresponding reference totals.
        - **reference_response** evaluates reference spending under the
          selected spend-to-media conversion.
        - **spend_period** and **response_period** record the selected dates.
        - **channel_type** identifies ordinary media and reach/frequency channels.

        With ``by``, response fields retain the requested observation axes.
        Spending remains totaled over the selected periods and groups.
        Breakdowns show where and when each overall spending change has an
        effect, not independent spending changes within each period or group.

        Incremental curves need not add up when channels interact. Zero
        spending in selected periods can retain carryover from earlier exposures.
        These are model-based interventions, not additional causal evidence.
    """
    grid = np.asarray(multipliers)
    if grid.ndim != 1 or grid.size == 0 or grid.dtype.kind not in "fiu":
        raise ValueError("multipliers must be a nonempty one-dimensional sequence of real numbers")
    if not np.isfinite(grid).all() or np.any(grid < 0) or len(np.unique(grid)) != len(grid):
        raise ValueError("multipliers must be distinct finite nonnegative numbers")

    context = _prepare_response(
        model,
        results,
        quantity=quantity,
        group=group,
        spend_to_media=spend_to_media,
        spend_to_rf=spend_to_rf,
        channels=channels,
        new_data=new_data,
        spend_periods=spend_periods,
        response_periods=response_periods,
        batch_size=batch_size,
    )
    retained, retain_axes, response_coords = _response_breakdown(context, by)
    totals = context.reference_spend
    indices = context.indices
    selected_count = len(indices)

    allocation = np.broadcast_to(np.asarray(totals), (1 + selected_count * len(grid), len(totals))).copy()
    comparison = allocation.copy()

    for index, channel_index in enumerate(indices):
        start = 1 + index * len(grid)
        allocation[start : start + len(grid), channel_index] *= grid
        comparison[start : start + len(grid), channel_index] = 0

    if not np.isfinite(allocation).all():
        raise ValueError(
            "Candidate spending is too large for the model precision. Reduce the multipliers or change units"
        )

    responses, increments = _evaluate_response_pairs(context, allocation, comparison, retain_axes=retain_axes)
    reference = responses[0]
    shape = (selected_count, len(grid), *reference.shape)
    response = np.moveaxis(responses[1:].reshape(shape), (0, 1), (-2, -1))
    incremental = np.moveaxis(increments[1:].reshape(shape), (0, 1), (-2, -1))
    curve_axes = ("chain", "draw", *retained, "channel", "multiplier")

    return xr.Dataset(
        {
            "response": (curve_axes, response),
            "incremental_response": (curve_axes, incremental),
            "spend": (
                ("channel", "multiplier"),
                allocation[1:].reshape(selected_count, len(grid), -1)[np.arange(selected_count), :, indices],
            ),
            "reference_spend": ("channel", np.asarray(totals)[indices]),
            "reference_response": (("chain", "draw", *retained), reference),
        },
        coords={
            **context.coords,
            **response_coords,
            "channel_type": ("channel", context.channel_types),
            "multiplier": grid,
        },
        attrs=context.attrs,
    )


def frequency_curves(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    frequencies: Sequence[float] | ArrayLike,
    group: Literal["prior", "posterior"] = "posterior",
    channels: Sequence[str] | None = None,
    new_data: object = None,
    periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    batch_size: int = 64,
) -> xr.Dataset:
    r"""Compare advertising frequencies while keeping spending and impressions fixed.

    Change one RF channel at a time. For candidate frequency :math:`f`,
    reach becomes :math:`r' = r f_{\mathrm{reference}} / f`. Other channels,
    earlier history, and cells without impressions stay unchanged. This assumes
    constant cost per impression. Choose frequencies feasible for your audience.

    Select each channel's best tested frequency using mean response across
    the selected parameter draws. These choices are conditional on the other channels'
    reference inputs, not a jointly optimized plan when channels interact.

    Parameters
    ----------
    model : Model
        Prepared model with reach, media frequency, and paired RF spending.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        Key returned by ``transformed_parameters`` with one expected response
        per observation in reporting units. Recomputed for each scenario.
    frequencies : array_like
        Distinct finite positive average exposures per reached person to test.
        These are levels, not multipliers. Each applies across the selected
        periods and groups wherever impressions are positive.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Choose ``"prior"`` with results from
        ``sample_prior`` to inspect responses implied by the priors.
        No sampling is performed. Compare groups using separate calls.
    channels : sequence of str, optional
        RF channel labels in the desired order. Defaults to all RF channels.
        Each needs positive spending and impressions during ``periods``.
    new_data : dataframe-like or PreparedData, optional
        Reference observations. Omit to use stored observations. Fitted scales
        are reused. Supply ``PreparedData`` to include earlier media history.
    periods : sequence, optional
        Time labels whose frequency changes. Defaults to all modeling periods.
    response_periods : sequence, optional
        Time labels whose responses count, summed across groups. Defaults to
        all modeling periods. Include later supplied dates to measure carryover.
    batch_size : int, default 64
        Maximum parameter draws evaluated together. Scenarios run sequentially.

    Returns
    -------
    xarray.Dataset
        Labeled frequency comparisons retaining draw-level uncertainty. The
        ``group`` attribute records which parameter draws were used.

        - **response** contains total responses by chain, draw, channel, and frequency.
        - **response_change** contains paired changes from the reference inputs.
        - **reference_response** contains the full reference response for each draw.
        - **reference_spend** records fixed spending for the selected periods.
        - **best_frequency** maximizes mean change across the selected draws
          within the supplied grid. Exact ties select the first supplied value.
        - **frequency_period** and **response_period** record the selected dates.

        Responses retain the quantity's units without inverse scaling. No
        audience cap is inferred, and the model is not refitted.
    """
    grid = np.asarray(frequencies)
    if grid.ndim != 1 or grid.size == 0 or grid.dtype.kind not in "fiu":
        raise ValueError("frequencies must be a nonempty one-dimensional sequence of real numbers")
    if not np.isfinite(grid).all() or np.any(grid <= 0) or len(np.unique(grid)) != len(grid):
        raise ValueError("frequencies must be distinct finite positive numbers")

    inputs, prepared, samples, coordinates = _response_inputs(
        model, results, quantity, new_data, batch_size, group=group
    )
    if not {"reach", "media_frequency", "rf_spend"}.issubset(prepared.arrays):
        raise ValueError("Frequency evaluation requires reach, media_frequency, and paired rf_spend columns")

    # Scenario arrays become floating point, including fixed cells and earlier history.
    for role in ("reach", "media_frequency"):
        original = np.asarray(inputs.values[role])
        if np.issubdtype(original.dtype, np.integer):
            converted = original.astype(model._dtype)
            upper_limit = 2 ** (np.iinfo(original.dtype).bits - (original.dtype.kind == "i"))
            if np.any(converted >= upper_limit) or not np.array_equal(converted.astype(original.dtype), original):
                raise ValueError(
                    f"Integer {role} inputs lose precision in frequency scenarios. "
                    "Rescale them or enable JAX 64-bit mode before constructing the model"
                )

    labels = prepared.rf_channels
    selected = labels if channels is None else channels
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence) or not selected:
        raise ValueError("channels must be a nonempty sequence of RF channel labels")
    if any(not isinstance(name, str) or name not in labels for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("channels must contain distinct labels from the model's RF channels")

    indices = np.array([labels.index(name) for name in selected], dtype=np.intp)

    time_labels = _coordinates({"time": prepared.time_values})["time"]
    period_indices = _period_indices(time_labels, periods, name="periods")
    response_indices = jnp.asarray(_period_indices(time_labels, response_periods, name="response_periods"))
    n_periods = len(time_labels)

    spend = jnp.asarray(prepared.arrays["rf_spend"], dtype=model._dtype)
    reach = jnp.asarray(prepared.arrays["reach"][-n_periods:], dtype=model._dtype)
    frequency = jnp.asarray(prepared.arrays["media_frequency"][-n_periods:], dtype=model._dtype)
    impressions = reach * frequency

    period_mask = np.zeros(n_periods, dtype=bool)
    period_mask[period_indices] = True
    mask = jnp.asarray(period_mask.reshape((-1,) + (1,) * (spend.ndim - 1)))
    changed_impressions = np.asarray(impressions)[period_indices][..., indices]

    if not np.isfinite(changed_impressions).all():
        raise ValueError("Impressions are too large for the model precision. Change exposure units")
    if np.any(~np.any(changed_impressions > 0, axis=tuple(range(spend.ndim - 1)))):
        raise ValueError("Selected channels need positive impressions during periods")

    totals = np.asarray(jnp.sum(jnp.where(mask, spend, 0), axis=tuple(range(spend.ndim - 1))))[indices]
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Selected channels need finite positive reference spending during periods")

    candidate_grid = np.asarray(jnp.asarray(grid, dtype=model._dtype))
    if (
        not np.isfinite(candidate_grid).all()
        or np.any(candidate_grid <= 0)
        or len(np.unique(candidate_grid)) != len(grid)
    ):
        raise ValueError("frequencies must remain distinct finite positive values in the model precision")

    transformations = {} if model.scaling is None else model.scaling.transformations

    def evaluate(candidate: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        channel, level = candidate
        changed = mask & (jnp.arange(len(labels)) == channel) & (impressions > 0)
        values = dict(inputs.values)
        valid = jnp.asarray(True)

        for role, raw in (("reach", impressions / level), ("media_frequency", jnp.full_like(frequency, level))):
            valid = valid & jnp.all(jnp.where(changed, jnp.isfinite(raw) & (raw > 0), True))
            transformed = transformations[role].transform(raw) if role in transformations else raw
            valid = valid & jnp.all(jnp.where(changed, jnp.isfinite(transformed), True))
            current = jnp.where(changed, transformed, inputs.values[role][-n_periods:])
            values[role] = jnp.concatenate((inputs.values[role][:-n_periods], current), axis=0)

        response, change = _sample_response(
            model,
            samples,
            replace(inputs, values=values),
            inputs,
            quantity=quantity,
            observation_shape=spend.shape[:-1],
            response_indices=response_indices,
            batch_size=batch_size,
        )
        return jnp.where(valid, response, jnp.nan), jnp.where(valid, change, jnp.nan)

    def evaluate_grid(candidates: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        reference, _ = _sample_response(
            model,
            samples,
            inputs,
            None,
            quantity=quantity,
            observation_shape=spend.shape[:-1],
            response_indices=response_indices,
            batch_size=batch_size,
        )
        responses, changes = cast(tuple[jax.Array, jax.Array], jax.lax.map(evaluate, candidates))
        return reference, responses, changes

    scenario_channels = jnp.asarray(np.repeat(indices, len(grid)))
    scenario_levels = jnp.asarray(np.tile(candidate_grid, len(indices)))
    reference, responses, changes = map(np.asarray, jax.jit(evaluate_grid)((scenario_channels, scenario_levels)))
    if not all(np.isfinite(value).all() for value in (reference, responses, changes)):
        raise ValueError("A frequency scenario produced invalid exposures or a nonfinite response")

    shape = (len(indices), len(grid), *responses.shape[1:])
    response = responses.reshape(shape).transpose(2, 3, 0, 1)
    change = changes.reshape(shape).transpose(2, 3, 0, 1)
    best = np.argmax(change.mean(axis=(0, 1), dtype=np.float64), axis=-1)
    axes = ("chain", "draw", "channel", "frequency")

    return xr.Dataset(
        {
            "response": (axes, response),
            "response_change": (axes, change),
            "reference_response": (("chain", "draw"), reference),
            "reference_spend": ("channel", totals),
            "best_frequency": ("channel", grid[best]),
        },
        coords={
            "chain": coordinates["chain"],
            "draw": coordinates["draw"],
            "channel": np.asarray(selected),
            "channel_type": ("channel", ["reach_frequency"] * len(indices)),
            "frequency": grid,
            "frequency_period": time_labels[period_indices],
            "response_period": time_labels[np.asarray(response_indices)],
        },
        attrs={
            "quantity": quantity,
            "group": group,
            "intervention": "fixed impressions and spending",
            "history": "fixed",
            "selection": f"highest {group} mean response among supplied frequencies with other channels fixed",
            "response_units": "as returned by the transformed quantity",
        },
    )


def media_metrics(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    group: Literal["prior", "posterior"] = "posterior",
    incremental_increase: float = 0.01,
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike] = "proportional",
    spend_to_rf: _ReachFrequencyConversion = "reach",
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    by: str | Sequence[str] | None = None,
    batch_size: int = 64,
) -> xr.Dataset:
    """Calculate incremental response, ROI, and marginal ROI by paid-media channel.

    Compare reference spending with removing or increasing one channel's
    spending at a time. Evaluate the full model for each selected parameter draw,
    keeping other spending and earlier history fixed. Channel effects need
    not add up when channels interact.

    Parameters
    ----------
    model : Model
        Prepared model with media and spend, reach/frequency and their spend,
        or both. Input families without spending stay fixed.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        Key returned by ``transformed_parameters`` with one expected response
        per observation in reporting units, such as ``"expected_revenue"``.
        Values are recomputed for each scenario without inverse scaling.
        Revenue gives monetary returns. Other outcomes give outcome per unit spend.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Choose ``"prior"`` with results from
        ``sample_prior`` to inspect returns implied by the priors.
        No sampling is performed. Compare groups using separate calls.
    incremental_increase : float, default 0.01
        Positive fractional spend increase used for marginal ROI. The default
        measures return on a 1% increase, not an exact derivative.
    spend_to_media : {"proportional"} or callable, default "proportional"
        Scale exposures with spending at their reference ratios. Alternatively,
        supply a JAX-compatible function mapping ordinary media spending in
        its channel order to nonnegative exposures of the same shape.
    spend_to_rf : {"reach", "frequency"} or callable, default "reach"
        Scale reach at fixed frequency, or frequency at fixed reach, assuming
        constant cost per impression. A JAX-compatible callable instead maps
        raw RF spending in ``rf_channels`` order to ``(reach, frequency)`` in
        original units, each with the same shape. Marginal ROI follows this
        selected spending change.
    channels : sequence of str, optional
        Paid channels to report, in the desired order. Defaults to all channels
        with spending, ordinary media first and RF second. Names must be unique.
        Each needs positive reference spending during ``spend_periods``.
    new_data : dataframe-like or PreparedData, optional
        Reference observations. Omit to use stored observations. Fitted scales
        are reused. Earlier exposures can be supplied through ``PreparedData``.
    spend_periods : sequence, optional
        Time labels whose spending changes and enters the return denominators.
        Defaults to all supplied modeling periods.
    response_periods : sequence, optional
        Time labels whose responses count.
        Defaults to all supplied periods. Include later dates to count carryover.
    by : str or sequence of str, optional
        Retain ``"time"``, ``"group"``, or both in response effects. Groups use
        the labels from ``prepare_data`` and require grouped data. Omit to
        sum over both. Retained axes follow time then group order.
    batch_size : int, default 64
        Maximum parameter draws evaluated together. Scenarios run sequentially.

    Returns
    -------
    xarray.Dataset
        Channel metrics retaining chain and draw coordinates. The ``group``
        attribute records which parameter draws were used.

        - **incremental_response** is reference response minus response with
          that channel's spending removed during ``spend_periods``.
        - **roi** divides total incremental response by **reference_spend**. It does
          not subtract spending from the numerator to calculate profit.
        - **marginal_response** is increased-spend response minus reference response.
        - **marginal_roi** divides total marginal response by **incremental_spend**,
          the additional spending used for the comparison.
        - **reference_response** contains the full response at reference spending.
        - **spend_period** and **response_period** record the selected dates.
        - **channel_type** identifies ordinary media and reach/frequency channels.

        With ``by``, response fields retain the requested observation axes.
        ROI, marginal ROI, and spending still aggregate all selected periods
        and groups. Sum effects within each draw before computing intervals.
        Breakdowns report where and when the same intervention changes outcomes,
        not separate interventions for each period or group.

        Zero spending can retain carryover from earlier exposures. Returns
        reflect the model and intervention assumptions, not new causal evidence.
    """
    if (
        isinstance(incremental_increase, bool)
        or not isinstance(incremental_increase, Real)
        or not np.isfinite(incremental_increase)
        or incremental_increase <= 0
    ):
        raise ValueError("incremental_increase must be a finite positive number")

    context = _prepare_response(
        model,
        results,
        quantity=quantity,
        group=group,
        spend_to_media=spend_to_media,
        spend_to_rf=spend_to_rf,
        channels=channels,
        new_data=new_data,
        spend_periods=spend_periods,
        response_periods=response_periods,
        batch_size=batch_size,
    )
    metrics = _allocation_metrics(
        context,
        np.asarray(context.reference_spend)[None, :],
        allocation_labels=["reference"],
        incremental_increase=float(incremental_increase),
        by=by,
    )

    return metrics.sel(allocation="reference", drop=True).rename(
        {"spend": "reference_spend", "response": "reference_response"}
    )


@dataclass(frozen=True)
class _ResponseContext:
    """Prepared draw evaluation and labels shared by spending analyses."""

    evaluator: "_BudgetResponse"
    reference_spend: jax.Array
    indices: NDArray[np.intp]
    channel_types: NDArray[np.str_]
    coords: dict[str, NDArray[np.generic]]
    response_coords: dict[str, NDArray[np.generic] | tuple[str, NDArray[np.generic]]]
    attrs: dict[str, str]


def _response_breakdown(
    context: _ResponseContext,
    by: str | Sequence[str] | None,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    dict[str, NDArray[np.generic] | tuple[str, NDArray[np.generic]]],
]:
    """Resolve retained observation axes and their prepared data labels."""
    requested = () if by is None else (by,) if isinstance(by, str) else by
    if not isinstance(requested, Sequence) or any(
        not isinstance(name, str) or name not in ("time", "group") for name in requested
    ):
        raise ValueError("by must contain only time or group dimension names")
    if len(set(requested)) != len(requested):
        raise ValueError("by must contain distinct dimension names")
    if "group" in requested and "group" not in context.response_coords:
        raise ValueError("Retaining group requires grouped data from prepare_data")

    retained = tuple(name for name in ("time", "group") if name in requested)
    retain_axes = tuple(("time", "group").index(name) for name in retained)
    response_coords = {name: context.response_coords[name] for name in retained}
    if "group" in retained:
        response_coords.update(
            {name: value for name, value in context.response_coords.items() if isinstance(value, tuple)}
        )

    return retained, retain_axes, response_coords


def _allocation_metrics(
    context: _ResponseContext,
    totals: NDArray[np.float32 | np.float64],
    *,
    allocation_labels: Sequence[str],
    incremental_increase: float,
    by: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Evaluate channel interventions around each joint spending allocation."""
    retained, retain_axes, response_coords = _response_breakdown(context, by)
    indices = context.indices
    count = len(indices)

    allocation = np.broadcast_to(totals[:, None, :], (len(totals), 1 + 2 * count, totals.shape[1])).copy()
    comparison = allocation.copy()
    rows = np.arange(count)
    comparison[:, 1 + rows, indices] = 0

    with np.errstate(over="ignore", invalid="ignore"):
        allocation[:, 1 + count + rows, indices] *= 1 + incremental_increase

    if not np.isfinite(allocation).all():
        raise ValueError("Candidate spending is too large for the model precision. Reduce incremental_increase")

    spend = totals[:, indices]
    positive = spend > 0
    incremental_spend = allocation[:, 1 + count + rows, indices] - spend

    if np.any(positive & (incremental_spend <= 0)):
        raise ValueError("incremental_increase is too small for the model precision. Use a larger increase")

    responses, differences = _evaluate_response_pairs(
        context,
        allocation.reshape(-1, totals.shape[1]),
        comparison.reshape(-1, totals.shape[1]),
        retain_axes=retain_axes,
    )

    shape = (len(totals), 1 + 2 * count, *responses.shape[1:])
    responses, differences = responses.reshape(shape), differences.reshape(shape)

    # Place allocation after chain and draw, and channels after retained observation axes.
    incremental = np.moveaxis(differences[:, 1 : 1 + count], (0, 1), (2, -1))
    marginal = np.moveaxis(differences[:, 1 + count :], (0, 1), (2, -1))
    response = np.moveaxis(responses[:, 0], 0, 2)

    # Removing or proportionally increasing zero spending is unchanged, but
    # neither ratio is defined. Preserve missing values rather than zero returns.
    with np.errstate(over="ignore", invalid="ignore"):
        observation_axes = tuple(range(3, 3 + len(retained)))
        total_incremental = incremental.sum(axis=observation_axes)
        total_marginal = marginal.sum(axis=observation_axes)
        roi = np.divide(total_incremental, spend, out=np.full_like(total_incremental, np.nan), where=positive)
        marginal_roi = np.divide(
            total_marginal, incremental_spend, out=np.full_like(total_marginal, np.nan), where=positive
        )

    if not np.all(np.isfinite(roi) | ~positive) or not np.all(np.isfinite(marginal_roi) | ~positive):
        raise ValueError("Channel returns are nonfinite. Check the response and spending units")

    axes = ("chain", "draw", "allocation", *retained, "channel")
    ratio_axes = ("chain", "draw", "allocation", "channel")

    return xr.Dataset(
        {
            "incremental_response": (axes, incremental),
            "roi": (ratio_axes, roi),
            "marginal_response": (axes, marginal),
            "marginal_roi": (ratio_axes, marginal_roi),
            "spend": (("allocation", "channel"), spend),
            "incremental_spend": (("allocation", "channel"), incremental_spend),
            "response": (("chain", "draw", "allocation", *retained), response),
        },
        coords={
            **context.coords,
            **response_coords,
            "channel_type": ("channel", context.channel_types),
            "allocation": list(allocation_labels),
        },
        attrs={**context.attrs, "incremental_increase": incremental_increase},
    )


def _evaluate_response_pairs(
    context: _ResponseContext,
    allocation: NDArray[np.generic],
    comparison: NDArray[np.generic],
    *,
    retain_axes: tuple[int, ...] = (),
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """Evaluate paired scenarios sequentially while batching parameter draws."""
    evaluate = jax.jit(
        lambda budgets: jax.lax.map(
            lambda pair: context.evaluator.paired_evaluation(*pair, retain_axes=retain_axes), budgets
        )
    )

    responses, differences = map(np.asarray, evaluate((jnp.asarray(allocation), jnp.asarray(comparison))))
    if not np.isfinite(responses).all() or not np.isfinite(differences).all():
        raise ValueError(
            "A scenario produced invalid media or a nonfinite response. "
            "Check the conversion, transformed quantity, and parameter draws"
        )

    return responses, differences


def _response_inputs(
    model: Model,
    results: xr.DataTree,
    quantity: str,
    new_data: object,
    batch_size: int,
    *,
    group: Literal["prior", "posterior"] = "posterior",
) -> tuple[_ModelData, PreparedData, dict[str, jax.Array], dict[str, NDArray[np.generic]]]:
    """Recover raw reference data while retaining fitted inputs and draw labels."""
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    if model._data is None:
        raise ValueError("Response evaluation requires a model with prepared data")
    if not isinstance(quantity, str) or not quantity:
        raise ValueError("quantity must name an observation-shaped transformed output")
    if model._transformed_parameters is None:
        raise ValueError("The model must define transformed_parameters for its expected response")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    dimensions, coordinates = _parameter_metadata(model)
    samples, coordinates = _parameter_draws(model, results, dimensions, coordinates, group=group)
    inputs = model._data
    prepared = _result_data(model)
    assert prepared is not None

    if new_data is not None:
        inputs, aligned = model._prepare_data(new_data)
        prepared = replace(aligned, arrays={name: np.array(inputs.values[name], copy=True) for name in aligned.arrays})

    # Recover original input units once. Candidate evaluation reuses the fitted factors.
    if model.scaling is not None:
        prepared = model.scaling.inverse_transform(prepared)

    return inputs, prepared, {name: jnp.asarray(value) for name, value in samples.items()}, coordinates


def _prepare_response(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    group: Literal["prior", "posterior"] = "posterior",
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike],
    spend_to_rf: _ReachFrequencyConversion = "reach",
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    batch_size: int = 64,
) -> _ResponseContext:
    """Prepare fixed model inputs and labels before numerical budget evaluation."""
    inputs, prepared, samples, coordinates = _response_inputs(
        model, results, quantity, new_data, batch_size, group=group
    )
    if not callable(spend_to_media) and not (isinstance(spend_to_media, str) and spend_to_media == "proportional"):
        raise ValueError("spend_to_media must be 'proportional' or a JAX-compatible callable")
    if not callable(spend_to_rf) and not (isinstance(spend_to_rf, str) and spend_to_rf in ("reach", "frequency")):
        raise ValueError("spend_to_rf must be 'reach', 'frequency', or a JAX-compatible callable")

    has_media = {"media", "spend"}.issubset(inputs.values)
    has_rf = {"reach", "media_frequency", "rf_spend"}.issubset(inputs.values)
    if not (has_media or has_rf):
        raise ValueError(
            "Response evaluation requires paired media and spend or reach, frequency, and rf_spend columns"
        )

    prepared_coords, group_coords = _prepared_coordinates(prepared)
    time_labels = prepared_coords["time"]
    spend_indices = _period_indices(time_labels, spend_periods, name="spend_periods")
    response_indices = _period_indices(time_labels, response_periods, name="response_periods")
    response_coords: dict[str, NDArray[np.generic] | tuple[str, NDArray[np.generic]]] = {
        "time": time_labels[response_indices]
    }
    if "group" in prepared_coords:
        response_coords["group"] = prepared_coords["group"]
        response_coords.update(group_coords)

    spend_mask = np.zeros(len(time_labels), dtype=bool)
    spend_mask[spend_indices] = True

    media_labels = prepared.channels if has_media else ()
    rf_labels = prepared.rf_channels if has_rf else ()
    labels = media_labels + rf_labels
    if len(set(labels)) != len(labels):
        raise ValueError(
            "Paid channel names must be unique across media and reach/frequency. Rename overlapping channels"
        )
    selected = labels if channels is None else channels
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence) or not selected:
        raise ValueError("channels must be a nonempty sequence of channel labels")
    if any(not isinstance(name, str) or name not in labels for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("channels must contain distinct labels from the model's paid-media channels")
    indices = np.array([labels.index(name) for name in selected], dtype=np.intp)

    # Default conversions affect selected channels. Custom mappings cover their full input family.
    conversion_mask = np.zeros(len(labels), dtype=bool)
    conversion_mask[indices] = True

    spend = jnp.concatenate(
        [
            jnp.asarray(prepared.arrays[name], dtype=model._dtype)
            for name, included in (("spend", has_media), ("rf_spend", has_rf))
            if included
        ],
        axis=-1,
    )
    observation_shape = spend.shape[:-1]
    periods = spend.shape[0]

    mask = jnp.asarray(spend_mask.reshape((-1,) + (1,) * (spend.ndim - 1)))
    selected_spend = jnp.where(mask, spend, 0)
    totals = jnp.sum(selected_spend, axis=tuple(range(spend.ndim - 1)))
    if np.any(np.asarray(totals)[indices] <= 0):
        raise ValueError(
            "Selected channels need positive reference spending during spend_periods to define their allocation"
        )

    convert: Callable[[jax.Array], ArrayLike] | None = None
    if has_media:
        media_spend = spend[..., : len(media_labels)]
        media = jnp.asarray(prepared.arrays["media"][-periods:], dtype=model._dtype)
        if isinstance(spend_to_media, str):
            changed = np.asarray(mask) & conversion_mask[: len(media_labels)]
            if np.any(changed & (np.asarray(media_spend) == 0) & (np.asarray(media) > 0)):
                raise ValueError("Positive media with zero spend needs an explicit spend_to_media function")

            def convert_media(candidate_spend: jax.Array) -> jax.Array:
                return media * (candidate_spend / jnp.where(media_spend > 0, media_spend, 1))

            convert = convert_media
        else:
            convert = spend_to_media
            conversion_mask[: len(media_labels)] = True

    convert_reach_frequency: Callable[[jax.Array], tuple[ArrayLike, ArrayLike]] | None = None
    if has_rf:
        rf_spend = spend[..., len(media_labels) :]
        reach = jnp.asarray(prepared.arrays["reach"][-periods:], dtype=model._dtype)
        frequency = jnp.asarray(prepared.arrays["media_frequency"][-periods:], dtype=model._dtype)
        if isinstance(spend_to_rf, str):
            positive_exposure = (np.asarray(reach) > 0) & (np.asarray(frequency) > 0)
            changed = np.asarray(mask) & conversion_mask[len(media_labels) :]
            if np.any(changed & (np.asarray(rf_spend) == 0) & positive_exposure):
                raise ValueError("Positive reach and frequency with zero spend need an explicit spend_to_rf function")

            def convert_rf(candidate_spend: jax.Array) -> tuple[jax.Array, jax.Array]:
                ratio = candidate_spend / jnp.where(rf_spend > 0, rf_spend, 1)
                # Change one exposure dimension so impressions scale once with spend.
                if spend_to_rf == "reach":
                    return reach * ratio, frequency
                return reach, frequency * ratio

            convert_reach_frequency = convert_rf
        else:
            convert_reach_frequency = spend_to_rf
            conversion_mask[len(media_labels) :] = True

    evaluator = _BudgetResponse(
        model=model,
        inputs=inputs,
        samples=samples,
        quantity=quantity,
        spend_weights=selected_spend / jnp.where(totals > 0, totals, 1),
        convert=convert,
        convert_reach_frequency=convert_reach_frequency,
        observation_shape=observation_shape,
        batch_size=int(batch_size),
        spend_mask=jnp.asarray(spend_mask),
        reference_spend=spend,
        response_indices=jnp.asarray(response_indices),
        conversion_mask=jnp.asarray(conversion_mask),
    )

    return _ResponseContext(
        evaluator=evaluator,
        reference_spend=totals,
        indices=indices,
        channel_types=np.asarray(["media"] * len(media_labels) + ["reach_frequency"] * len(rf_labels))[indices],
        coords={
            "chain": coordinates["chain"],
            "draw": coordinates["draw"],
            "channel": np.asarray(selected),
            "spend_period": time_labels[spend_indices],
            "response_period": time_labels[response_indices],
        },
        response_coords=response_coords,
        attrs={
            "quantity": quantity,
            "group": group,
            "spend_to_media": "proportional" if isinstance(spend_to_media, str) else "custom",
            "spend_to_rf": (spend_to_rf if isinstance(spend_to_rf, str) else "custom"),
            "allocation": "reference spending proportions across selected spending periods and groups",
            "history": "fixed",
            "response_window": "selected supplied modeling periods only",
            "response_units": "as returned by the transformed quantity",
        },
    )


def _sample_response(
    model: Model,
    samples: dict[str, jax.Array],
    inputs: _ModelData,
    reference_inputs: _ModelData | None,
    *,
    quantity: str,
    observation_shape: tuple[int, ...],
    response_indices: jax.Array | None,
    batch_size: int,
    retain_axes: tuple[int, ...] = (),
) -> tuple[jax.Array, jax.Array]:
    """Evaluate paired observation responses in bounded parameter batches."""
    reduction_axes = tuple(axis for axis in range(len(observation_shape)) if axis not in retain_axes)

    def evaluate_quantity(data: _ModelData, parameters: dict[str, jax.Array]) -> jax.Array:
        quantities = model._evaluate_quantities(data, parameters)
        if quantity not in quantities:
            raise ValueError(f"Transformed quantity {quantity!r} is not available")

        value = quantities[quantity]
        if value.shape != observation_shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("The response quantity must be floating-point with the observation shape")

        return value if response_indices is None else value[response_indices]

    def response(parameters: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value = evaluate_quantity(inputs, parameters)
        difference = jnp.zeros_like(value)
        if reference_inputs is not None:
            # Subtract per observation before a large baseline can hide changes in the sum.
            difference = value - evaluate_quantity(reference_inputs, parameters)

        return jnp.sum(value, axis=reduction_axes), jnp.sum(difference, axis=reduction_axes)

    chains, draws = next(iter(samples.values())).shape[:2]
    flattened = {name: value.reshape((-1, *value.shape[2:])) for name, value in samples.items()}
    totals, differences = cast(
        tuple[jax.Array, jax.Array],
        jax.lax.map(response, flattened, batch_size=min(int(batch_size), chains * draws)),
    )

    return (
        totals.reshape(chains, draws, *totals.shape[1:]),
        differences.reshape(chains, draws, *differences.shape[1:]),
    )


def _period_indices(labels: NDArray[np.generic], selected: Sequence[object] | None, *, name: str) -> NDArray[np.intp]:
    """Resolve existing time labels without changing the model's evaluation axis."""
    if selected is None:
        return np.arange(len(labels), dtype=np.intp)

    requested = np.asarray(selected)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError(f"{name} must be a nonempty sequence of observation time labels")
    if any(isinstance(value, (bool, np.bool_)) for value in selected):
        raise ValueError(f"{name} must contain time labels, not a boolean mask")

    index = pd.Index(labels)
    if isinstance(index, pd.DatetimeIndex):
        if requested.dtype.kind in "biufc":
            raise ValueError(f"{name} must contain dates matching the observation time labels")
        try:
            requested = pd.DatetimeIndex(requested).to_numpy()
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain valid observation dates") from error

    positions = index.get_indexer(pd.Index(requested))
    if np.any(positions < 0):
        raise ValueError(f"{name} contains unknown observation labels {requested[positions < 0].tolist()}")
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"{name} must contain distinct observation time labels")

    return np.sort(positions).astype(np.intp)


@dataclass(frozen=True)
class _BudgetResponse:
    """Evaluate joint channel allocations without host-side data preparation."""

    model: Model
    inputs: _ModelData
    samples: dict[str, jax.Array]
    quantity: str
    spend_weights: jax.Array
    convert: Callable[[jax.Array], ArrayLike] | None
    observation_shape: tuple[int, ...]
    batch_size: int
    spend_mask: jax.Array | None = None
    reference_spend: jax.Array | None = None
    response_indices: jax.Array | None = None
    convert_reach_frequency: Callable[[jax.Array], tuple[ArrayLike, ArrayLike]] | None = None
    conversion_mask: jax.Array | None = None

    def __call__(self, budgets: jax.Array) -> jax.Array:
        """Return one total response per parameter draw for a joint allocation."""
        return self.paired_evaluation(budgets)[0]

    def _scenario_inputs(self, budgets: jax.Array) -> tuple[_ModelData, jax.Array]:
        """Replace paid exposures and spending while preserving other model inputs."""
        spend = self.spend_weights * budgets
        mask = jnp.asarray(True)
        if self.spend_mask is not None:
            assert self.reference_spend is not None
            mask = self.spend_mask.reshape((-1,) + (1,) * (spend.ndim - 1))
            spend = jnp.where(mask, spend, self.reference_spend)

        values = dict(self.inputs.values)
        scaling = self.model.scaling
        transformations = {} if scaling is None else scaling.transformations
        periods = self.observation_shape[0]
        n_media = 0 if self.convert is None else values["media"].shape[-1]
        valid = jnp.asarray(True)

        def update(name: str, value: ArrayLike, shape: tuple[int, ...], conversion: str) -> None:
            nonlocal valid
            current = jnp.asarray(value)
            if current.shape != shape or not (
                jnp.issubdtype(current.dtype, jnp.floating) or jnp.issubdtype(current.dtype, jnp.integer)
            ):
                raise ValueError(f"{conversion} must return real {name} values with the current spend shape")

            current_mask = mask
            if self.conversion_mask is not None:
                channels = (
                    self.conversion_mask[:n_media] if name in ("media", "spend") else self.conversion_mask[n_media:]
                )
                current_mask = mask & channels

            current = current.astype(self.model._dtype)
            valid = valid & jnp.all(jnp.where(current_mask, jnp.isfinite(current) & (current >= 0), True))
            if name in transformations:
                current = transformations[name].transform(current)

            # Preserve exact inputs for unselected channels, excluded periods, and history.
            if name in ("spend", "rf_spend"):
                values[name] = jnp.where(current_mask, current, values[name])
            else:
                current = jnp.where(current_mask, current, values[name][-periods:])
                values[name] = jnp.concatenate((values[name][:-periods], current), axis=0)

        if self.convert is not None:
            media_spend = spend[..., :n_media]
            update("media", self.convert(media_spend), media_spend.shape, "spend_to_media")
            update("spend", media_spend, media_spend.shape, "spend_to_media")

        if self.convert_reach_frequency is not None:
            rf_spend = spend[..., n_media:]
            exposures = self.convert_reach_frequency(rf_spend)
            if not isinstance(exposures, (tuple, list)) or len(exposures) != 2:
                raise ValueError("spend_to_rf must return a pair of reach and frequency arrays")
            reach, frequency = exposures
            update("reach", reach, rf_spend.shape, "spend_to_rf")
            update("media_frequency", frequency, rf_spend.shape, "spend_to_rf")
            update("rf_spend", rf_spend, rf_spend.shape, "spend_to_rf")

        return replace(self.inputs, values=values), valid

    def paired_evaluation(
        self,
        budgets: jax.Array,
        reference: jax.Array | None = None,
        *,
        retain_axes: tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate paired differences while retaining selected observation axes."""
        inputs, valid = self._scenario_inputs(budgets)
        reference_inputs = None
        if reference is not None:
            reference_inputs, reference_valid = self._scenario_inputs(reference)
            valid = valid & reference_valid

        totals, differences = _sample_response(
            self.model,
            self.samples,
            inputs,
            reference_inputs,
            quantity=self.quantity,
            observation_shape=self.observation_shape,
            response_indices=self.response_indices,
            batch_size=self.batch_size,
            retain_axes=retain_axes,
        )

        return jnp.where(valid, totals, jnp.nan), jnp.where(valid, differences, jnp.nan)
