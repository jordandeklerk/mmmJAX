"""Posterior response curves for explicit spending scenarios."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax._results import _coordinates
from mmmjax.model import Model, _ModelData
from mmmjax.sampling import _parameter_metadata, _posterior_draws, _result_data

__all__ = ["response_curves"]


def response_curves(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    multipliers: Sequence[float] | ArrayLike,
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike],
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    batch_size: int = 64,
) -> xr.Dataset:
    """Evaluate paid-media spending curves using existing posterior draws.

    Vary one channel's spending at a time, retaining its allocation across
    selected periods and groups. Spending and exposures outside those periods
    stay fixed. Evaluate the full model for every draw, then sum responses
    over the measurement periods and groups. Include later measurement dates
    to count carryover. No observations are added automatically.

    Parameters
    ----------
    model : Model
        Prepared model with paired media and spend inputs. Reach/frequency
        inputs, if present, stay fixed.
    results : xarray.DataTree
        Results containing the model's constrained posterior draws.
    quantity : str
        Key in the mapping returned by ``transformed_parameters``, such as
        ``"expected_revenue"``. Its value must contain one expected response
        per observation in the desired reporting units. It is recomputed for
        each scenario and posterior draw and need not be stored in ``results``.
        No inverse scaling is applied to this output.
    multipliers : array_like
        Distinct nonnegative spending multipliers. One retains reference
        spending and zero removes the channel's spending during ``spend_periods``.
    spend_to_media : {"proportional"} or callable
        Use ``"proportional"`` to scale exposure with spend at each period
        and group. A JAX-compatible function instead receives current spend
        in original units and model channel order, then returns nonnegative
        exposures of the same shape. Only exposures during ``spend_periods``
        are replaced. Earlier history is excluded from the conversion.
    channels : sequence of str, optional
        Channel labels to evaluate, in the desired order. Defaults to all
        paid-media channels. Each needs positive reference spending.
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
    batch_size : int, default 64
        Maximum posterior draws evaluated together. Scenarios run sequentially.

    Returns
    -------
    xarray.Dataset
        Labeled curves preserving chain and draw coordinates.

        - **response** contains total responses by channel and multiplier.
        - **incremental_response** subtracts the same channel's response
          with zero spending during ``spend_periods`` and other inputs fixed.
        - **spend** contains candidate totals during the spending periods
          in original spend units.
        - **reference_spend** contains the corresponding reference totals.
        - **reference_response** evaluates reference spending under the
          selected spend-to-media conversion.
        - **spend_period** and **response_period** record the selected dates.

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
        spend_to_media=spend_to_media,
        channels=channels,
        new_data=new_data,
        spend_periods=spend_periods,
        response_periods=response_periods,
        batch_size=batch_size,
    )
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

    evaluate = jax.jit(lambda budgets: jax.lax.map(lambda pair: context.evaluator.paired_evaluation(*pair), budgets))
    responses, increments = map(np.asarray, evaluate((jnp.asarray(allocation), jnp.asarray(comparison))))
    if not np.isfinite(responses).all() or not np.isfinite(increments).all():
        raise ValueError(
            "A scenario produced invalid media or a nonfinite response. "
            "Check the conversion, transformed quantity, and posterior draws"
        )
    reference = responses[0]
    response = responses[1:].reshape(selected_count, len(grid), *reference.shape).transpose(2, 3, 0, 1)
    incremental = increments[1:].reshape(selected_count, len(grid), *reference.shape).transpose(2, 3, 0, 1)
    curve_axes = ("chain", "draw", "channel", "multiplier")
    return xr.Dataset(
        {
            "response": (curve_axes, response),
            "incremental_response": (curve_axes, incremental),
            "spend": (
                ("channel", "multiplier"),
                allocation[1:].reshape(selected_count, len(grid), -1)[np.arange(selected_count), :, indices],
            ),
            "reference_spend": ("channel", np.asarray(totals)[indices]),
            "reference_response": (("chain", "draw"), reference),
        },
        coords={**context.coords, "multiplier": grid},
        attrs=context.attrs,
    )


@dataclass(frozen=True)
class _ResponseContext:
    """Prepared posterior evaluation and labels shared by spending analyses."""

    evaluator: "_BudgetResponse"
    reference_spend: jax.Array
    indices: NDArray[np.intp]
    coords: dict[str, NDArray[np.generic]]
    attrs: dict[str, str]


def _prepare_response(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    spend_to_media: Literal["proportional"] | Callable[[jax.Array], ArrayLike],
    channels: Sequence[str] | None = None,
    new_data: object = None,
    spend_periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    batch_size: int = 64,
) -> _ResponseContext:
    """Prepare fixed model inputs and labels before numerical budget evaluation."""
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
    if not callable(spend_to_media) and not (isinstance(spend_to_media, str) and spend_to_media == "proportional"):
        raise ValueError("spend_to_media must be 'proportional' or a JAX-compatible callable")

    dimensions, coordinates = _parameter_metadata(model)
    posterior, coordinates = _posterior_draws(model, results, dimensions, coordinates)
    inputs = model._data
    prepared = _result_data(model)
    assert prepared is not None
    if new_data is not None:
        inputs, aligned = model._prepare_data(new_data)
        prepared = replace(aligned, arrays={name: np.array(value, copy=True) for name, value in inputs.values.items()})
    if not {"media", "spend"}.issubset(inputs.values):
        raise ValueError("Response evaluation requires paired media and spend columns")
    time_labels = _coordinates({"time": prepared.time_values})["time"]
    spend_indices = _period_indices(time_labels, spend_periods, name="spend_periods")
    response_indices = _period_indices(time_labels, response_periods, name="response_periods")
    spend_mask = np.zeros(len(time_labels), dtype=bool)
    spend_mask[spend_indices] = True
    labels = prepared.channels
    selected = labels if channels is None else channels
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence) or not selected:
        raise ValueError("channels must be a nonempty sequence of channel labels")
    if any(not isinstance(name, str) or name not in labels for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("channels must contain distinct labels from the model's paid-media channels")
    indices = np.array([labels.index(name) for name in selected], dtype=np.intp)

    # Recover original input units once. Candidate evaluation reuses the fitted factors.
    if model.scaling is not None:
        prepared = model.scaling.inverse_transform(prepared)
    spend = jnp.asarray(prepared.arrays["spend"], dtype=model._dtype)
    media = jnp.asarray(prepared.arrays["media"], dtype=model._dtype)
    observation_shape = spend.shape[:-1]
    periods = spend.shape[0]
    mask = jnp.asarray(spend_mask.reshape((-1,) + (1,) * (spend.ndim - 1)))
    selected_spend = jnp.where(mask, spend, 0)
    totals = jnp.sum(selected_spend, axis=tuple(range(spend.ndim - 1)))
    if np.any(np.asarray(totals)[indices] <= 0):
        raise ValueError(
            "Selected channels need positive reference spending during spend_periods to define their allocation"
        )
    if isinstance(spend_to_media, str):
        if np.any(np.asarray(mask) & (np.asarray(spend) == 0) & (np.asarray(media[-periods:]) > 0)):
            raise ValueError("Positive media with zero spend needs an explicit spend_to_media function")

        def convert(candidate_spend: jax.Array) -> jax.Array:
            return media[-periods:] * (candidate_spend / jnp.where(spend > 0, spend, 1))

    else:
        convert = spend_to_media

    evaluator = _BudgetResponse(
        model=model,
        inputs=inputs,
        posterior=posterior,
        quantity=quantity,
        spend_weights=selected_spend / jnp.where(totals > 0, totals, 1),
        convert=convert,
        observation_shape=observation_shape,
        batch_size=int(batch_size),
        spend_mask=jnp.asarray(spend_mask),
        reference_spend=spend,
        response_indices=jnp.asarray(response_indices),
    )
    return _ResponseContext(
        evaluator=evaluator,
        reference_spend=totals,
        indices=indices,
        coords={
            "chain": coordinates["chain"],
            "draw": coordinates["draw"],
            "channel": np.asarray(selected),
            "spend_period": time_labels[spend_indices],
            "response_period": time_labels[response_indices],
        },
        attrs={
            "quantity": quantity,
            "spend_to_media": "proportional" if isinstance(spend_to_media, str) else "custom",
            "allocation": "reference spending proportions across selected spending periods and groups",
            "history": "fixed",
            "response_window": "selected supplied modeling periods only",
            "response_units": "as returned by the transformed quantity",
        },
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
    positions = index.get_indexer(requested)
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
    posterior: dict[str, jax.Array]
    quantity: str
    spend_weights: jax.Array
    convert: Callable[[jax.Array], ArrayLike]
    observation_shape: tuple[int, ...]
    batch_size: int
    spend_mask: jax.Array | None = None
    reference_spend: jax.Array | None = None
    response_indices: jax.Array | None = None

    def __call__(self, budgets: jax.Array) -> jax.Array:
        """Return one total response per posterior draw for a joint allocation."""
        return self.paired_evaluation(budgets)[0]

    def _scenario_inputs(self, budgets: jax.Array) -> tuple[_ModelData, jax.Array]:
        """Replace current media and spend while preserving other model inputs."""
        spend = self.spend_weights * budgets
        mask = jnp.asarray(True)
        if self.spend_mask is not None:
            assert self.reference_spend is not None
            mask = self.spend_mask.reshape((-1,) + (1,) * (spend.ndim - 1))
            spend = jnp.where(mask, spend, self.reference_spend)
        media = jnp.asarray(self.convert(spend))
        if media.shape != spend.shape or not (
            jnp.issubdtype(media.dtype, jnp.floating) or jnp.issubdtype(media.dtype, jnp.integer)
        ):
            raise ValueError("spend_to_media must return real media values with the current spend shape")
        media = media.astype(self.inputs.values["media"].dtype)
        valid = jnp.all(jnp.where(mask, jnp.isfinite(media) & (media >= 0), True))
        values = dict(self.inputs.values)
        scaling = self.model.scaling
        transformations = {} if scaling is None else scaling.transformations
        if "media" in transformations:
            media = transformations["media"].transform(media)
        if "spend" in transformations:
            spend = transformations["spend"].transform(spend)
        # Retain exact stored values outside the intervention, including fitted scaling.
        periods = self.observation_shape[0]
        media = jnp.where(mask, media, values["media"][-periods:])
        spend = jnp.where(mask, spend, values["spend"])
        values["media"] = jnp.concatenate((values["media"][:-periods], media), axis=0)
        values["spend"] = spend
        return replace(self.inputs, values=values), valid

    def _quantity(self, inputs: _ModelData, parameters: dict[str, jax.Array]) -> jax.Array:
        """Select the observation-shaped deterministic response before aggregation."""
        quantities = self.model._evaluate_quantities(inputs, parameters)
        if self.quantity not in quantities:
            raise ValueError(f"Transformed quantity {self.quantity!r} is not available")
        value = quantities[self.quantity]
        if value.shape != self.observation_shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("The response quantity must be floating-point with the observation shape")
        return value if self.response_indices is None else value[self.response_indices]

    def paired_evaluation(self, budgets: jax.Array, reference: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        """Sum paired observation differences before a large baseline can hide them."""
        inputs, valid = self._scenario_inputs(budgets)
        reference_inputs = None
        if reference is not None:
            reference_inputs, reference_valid = self._scenario_inputs(reference)
            valid = valid & reference_valid

        def response(parameters: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
            value = self._quantity(inputs, parameters)
            difference = jnp.zeros((), dtype=value.dtype)
            if reference_inputs is not None:
                difference = jnp.sum(value - self._quantity(reference_inputs, parameters))
            return jnp.where(valid, jnp.sum(value), jnp.nan), jnp.where(valid, difference, jnp.nan)

        chains, draws = next(iter(self.posterior.values())).shape[:2]
        flattened = {name: value.reshape((-1, *value.shape[2:])) for name, value in self.posterior.items()}
        totals, differences = cast(
            tuple[jax.Array, jax.Array],
            jax.lax.map(response, flattened, batch_size=min(self.batch_size, chains * draws)),
        )
        return totals.reshape(chains, draws), differences.reshape(chains, draws)
