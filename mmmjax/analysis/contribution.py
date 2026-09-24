"""Baseline and channel contributions to the expected response from prior or posterior draws."""

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from numbers import Real
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import DTypeLike
from numpy.typing import NDArray

from mmmjax.analysis.response import (
    _period_indices,
    _response_breakdown,
    _response_coordinates,
    _response_inputs,
    _sample_response,
)
from mmmjax.data._results import _prepared_coordinates
from mmmjax.data.prepare import PreparedData
from mmmjax.inference.sampling import _result_data
from mmmjax.model.model import Model

__all__ = ["contributions"]

type _BaselineRule = Literal["min", "max"]
type _TreatmentBaselines = _BaselineRule | Mapping[str, _BaselineRule | float]
type _Paired = tuple[jax.Array, jax.Array]


class _Family(StrEnum):
    """Input families whose members the analysis removes one at a time."""

    MEDIA = "media"
    REACH_FREQUENCY = "reach_frequency"
    ORGANIC_MEDIA = "organic_media"
    ORGANIC_REACH_FREQUENCY = "organic_reach_frequency"
    TREATMENT = "treatment"


def contributions(
    model: Model,
    results: xr.DataTree,
    *,
    quantity: str,
    group: Literal["prior", "posterior"] = "posterior",
    channels: Sequence[str] | None = None,
    treatment_baselines: _TreatmentBaselines = "min",
    new_data: object = None,
    periods: Sequence[object] | None = None,
    response_periods: Sequence[object] | None = None,
    by: str | Sequence[str] | None = None,
    batch_size: int = 64,
) -> xr.Dataset:
    """Decompose the expected response into a baseline and channel contributions.

    Remove one input at a time and compare the result with the reference
    response. Remove every input at once to measure the baseline. Paid and
    organic exposures are set to zero during ``periods``, only reach is set to
    zero for reach and frequency families while frequency stays fixed, and
    treatments are set to baseline levels. Controls, seasonality, earlier
    history, and every other input stay fixed. Each scenario rewrites only the
    input it removes. The baseline necessarily rewrites every removable input
    at once. The full model is evaluated for each selected parameter draw and
    differences are taken within draws, so the returned draws support credible
    intervals.

    Parameters
    ----------
    model : Model
        Prepared model with media, reach and frequency, organic media, organic
        reach and frequency, or treatment inputs. Spending is not required.
    results : xarray.DataTree
        Results containing the model's constrained draws in the selected group.
    quantity : str
        Key returned by ``transformed_parameters`` holding the expected outcome
        for every observation on the scale the likelihood uses, such as
        ``"mu"``. It is recomputed for each scenario and draw, and the fitted
        outcome scaling restores original outcome units before anything is
        summed, so results are in the units of the outcome column. Any
        normalization by exposures or spending inside the blocks should read
        the training arrays from ``reference`` so that a scenario cannot zero it.
    group : {"prior", "posterior"}, default "posterior"
        Parameter draws to use. Choose ``"prior"`` with results from
        ``sample_prior`` to inspect contributions implied by the priors.
        No sampling is performed. Compare groups using separate calls.
    channels : sequence of str, optional
        Labels to report, in the desired order. Defaults to every member of
        every family present. The default order is paid media, reach and
        frequency, organic media, organic reach and frequency, and then
        treatments by their column names. Labels must be unique across
        families.
    treatment_baselines : {"min", "max"} or mapping, default "min"
        Counterfactual level for each treatment in original data units. A word
        applies the minimum or maximum observed in the training data to every
        treatment. A mapping from treatment name to a number or to one of those
        words sets treatments individually, and unnamed treatments use the
        minimum.
    new_data : dataframe-like or PreparedData, optional
        Reference observations using the model's columns and fitted scales.
        Omit to use stored observations. Supply ``PreparedData`` with
        ``media_history`` to include earlier exposures for new observations.
    periods : sequence, optional
        Time labels during which inputs are removed. Defaults to all supplied
        modeling periods. Earlier history is never changed.
    response_periods : sequence, optional
        Time labels whose responses count. Defaults to all supplied modeling
        periods. Include later dates to count carryover.
    by : str or sequence of str, optional
        Retain ``"time"``, ``"group"``, or both in response fields. Group
        labels come from ``prepare_data`` and require grouped data. Omit for
        totals. Retained axes follow time then group order.
    batch_size : int, default 64
        Maximum parameter draws evaluated together. Scenarios run sequentially.

    Returns
    -------
    xarray.Dataset
        Responses in original outcome units with chain and draw labels and a
        ``group`` attribute identifying the parameter draws.

        - **reference_response** — Full model response
        - **baseline_response** — Response with all exposures removed and
          treatments at baseline levels during ``periods``. Earlier history is
          unchanged
        - **joint_incremental_response** — Reference minus baseline response
        - **incremental_response** — Response lost by removing each input alone
        - **contribution_share**, **baseline_share** — Each increment and the
          baseline, respectively, divided by the reference response
        - **exposure**, **effectiveness** — Total exposure during ``periods``
          and increment per exposure. RF channels use reach times frequency.
          Both are missing for treatments
        - **treatment_baseline** — Treatment counterfactual levels
        - **channel_type**, **period**, **response_period** — Input families
          and selected dates

        Responses retain chain, draw, and optional ``by`` axes. Shares and
        effectiveness use totals across periods and groups within each draw.
        Interacting channel effects need not sum to the joint effect.
    """
    inputs, prepared, samples, coordinates = _response_inputs(
        model, results, quantity, new_data, batch_size, group=group
    )
    families = tuple(family for family in _Family if all(role in prepared.arrays for role in _exposure_roles(family)))
    if not families:
        raise ValueError(
            "Contribution analysis requires media, reach and frequency, organic media, "
            "organic reach and frequency, or treatment inputs"
        )

    members = [
        (family, index, label) for family in families for index, label in enumerate(_family_labels(family, prepared))
    ]
    names = [label for _, _, label in members]
    if len(set(names)) != len(names):
        raise ValueError("Channel and treatment names must be unique across input families. Rename overlapping labels")
    selected = names if channels is None else channels
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence) or not selected:
        raise ValueError("channels must be a nonempty sequence of channel or treatment labels")
    if any(not isinstance(name, str) or name not in names for name in selected) or len(set(selected)) != len(selected):
        raise ValueError(
            "channels must contain distinct labels from the model's media, RF, organic, and treatment inputs"
        )
    chosen = [members[names.index(name)] for name in selected]

    prepared_coords, _ = _prepared_coordinates(prepared)
    time_labels = prepared_coords["time"]
    period_indices = _period_indices(time_labels, periods, name="periods")
    response_indices = _period_indices(time_labels, response_periods, name="response_periods")
    response_coords = _response_coordinates(prepared, response_indices)
    retained, retain_axes, breakdown_coords = _response_breakdown(response_coords, by)
    n_periods = len(time_labels)
    group_shape = (len(prepared.group_values),) if prepared.group_columns else ()
    observation_shape = (n_periods, *group_shape)

    treatment_labels = _family_labels(_Family.TREATMENT, prepared)
    training = _training_data(model) if treatment_labels else None
    baselines = _treatment_baselines(treatment_baselines, training, treatment_labels)

    dtype = model._dtype
    transformations = {} if model.scaling is None else model.scaling.transformations
    period_mask = np.zeros(n_periods, dtype=bool)
    period_mask[period_indices] = True
    roles = [_removed_role(family) for family in families]
    removed: dict[str, jax.Array] = {}
    masks: dict[str, jax.Array] = {}
    for family, role in zip(families, roles, strict=True):
        current = inputs.values[role]
        recent = np.asarray(prepared.arrays[role], dtype=np.float64)[-n_periods:]
        if family is _Family.TREATMENT:
            assert baselines is not None
            replacement = np.broadcast_to(baselines, recent.shape)
        else:
            replacement = np.zeros_like(recent)
        role_dtype: DTypeLike
        if role in transformations:
            role_dtype = dtype
            scaled = transformations[role].transform(jnp.asarray(replacement, dtype=role_dtype))
            replaced = scaled.astype(role_dtype)
        else:
            role_dtype = _counterfactual_dtype(current, replacement, dtype)
            replaced = jnp.asarray(replacement, dtype=role_dtype)
        history_rows = current.shape[0] - n_periods
        removed[role] = jnp.concatenate((current[:history_rows].astype(role_dtype), replaced), axis=0)
        role_mask = np.concatenate((np.zeros(history_rows, dtype=bool), period_mask))
        masks[role] = jnp.asarray(role_mask.reshape((-1,) + (1,) * (current.ndim - 1)))

    response_positions = jnp.asarray(response_indices)
    selected_members: dict[str, list[int]] = {role: [] for role in roles}
    selected_positions: dict[str, list[int]] = {role: [] for role in roles}
    for position, (family, index, _) in enumerate(chosen):
        role = _removed_role(family)
        selected_members[role].append(index)
        selected_positions[role].append(position)
    intervened = [role for role in roles if selected_members[role]]

    def replaced_values(role: str, select: jax.Array) -> dict[str, jax.Array]:
        # Only the intervened role is rewritten, so every other input keeps its original dtype.
        current = inputs.values[role]
        values = dict(inputs.values)
        values[role] = jnp.where(select, removed[role], current.astype(removed[role].dtype))
        return values

    def evaluate_scenario(values: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        totals, differences = _sample_response(
            model,
            samples,
            model._replace_data_values(inputs, values),
            inputs,
            quantity=quantity,
            observation_shape=observation_shape,
            response_indices=response_positions,
            batch_size=batch_size,
            retain_axes=retain_axes,
        )
        return totals, differences

    def evaluate_members(role: str, members: jax.Array) -> tuple[jax.Array, jax.Array]:
        positions = jnp.arange(inputs.values[role].shape[-1])

        def evaluate(member: jax.Array) -> tuple[jax.Array, jax.Array]:
            return evaluate_scenario(replaced_values(role, masks[role] & (positions == member)))

        return cast(tuple[jax.Array, jax.Array], jax.lax.map(evaluate, members))

    def evaluate_all(selections: dict[str, jax.Array]) -> tuple[jax.Array, dict[str, _Paired], _Paired]:
        reference, _ = _sample_response(
            model,
            samples,
            inputs,
            None,
            quantity=quantity,
            observation_shape=observation_shape,
            response_indices=response_positions,
            batch_size=batch_size,
            retain_axes=retain_axes,
        )
        scenarios = {role: evaluate_members(role, selections[role]) for role in intervened}
        # The baseline removes every member of every family at once, so it rewrites every role.
        baseline_values = dict(inputs.values)
        for role in roles:
            baseline_values[role] = jnp.where(
                masks[role], removed[role], inputs.values[role].astype(removed[role].dtype)
            )
        return reference, scenarios, evaluate_scenario(baseline_values)

    selections = {role: jnp.asarray(selected_members[role], dtype=np.int32) for role in intervened}
    reference, scenarios, baseline_pair = jax.jit(evaluate_all)(selections)
    reference = np.asarray(reference)
    baseline_total, baseline_difference = (np.asarray(value) for value in baseline_pair)
    ordered_totals: list[NDArray[np.floating]] = [np.empty(())] * len(chosen)
    ordered_differences: list[NDArray[np.floating]] = [np.empty(())] * len(chosen)
    for role, (role_totals, role_differences) in scenarios.items():
        for slot, position in enumerate(selected_positions[role]):
            ordered_totals[position] = np.asarray(role_totals[slot])
            ordered_differences[position] = np.asarray(role_differences[slot])
    totals = np.stack(ordered_totals)
    differences = np.stack(ordered_differences)
    if not np.isfinite(reference).all():
        raise ValueError(
            "The reference response is nonfinite. Check that the transformed quantity and the parameter "
            "draws are finite"
        )
    nonfinite = [label for (_, _, label), value in zip(chosen, totals, strict=True) if not np.isfinite(value).all()]
    if not np.isfinite(baseline_total).all():
        nonfinite.append("the baseline")
    if nonfinite or not np.isfinite(differences).all() or not np.isfinite(baseline_difference).all():
        failing = ", ".join(nonfinite) if nonfinite else "an input"
        raise ValueError(
            f"Removing {failing} produced a nonfinite response. A transformed parameter is probably normalized by "
            "exposures or spending that the scenario sets to zero. Compute such normalizations from the training "
            "arrays in the reference namespace so that removing an input leaves them unchanged"
        )

    count = len(chosen)
    # Differences are scenario minus reference, so increments and the joint effect flip the sign.
    incremental = np.moveaxis(-differences, 0, -1)
    baseline = baseline_total
    joint = -baseline_difference
    observation_axes = tuple(range(2, 2 + len(retained)))
    total_reference = reference.sum(axis=observation_axes)
    total_baseline = baseline.sum(axis=observation_axes)
    total_incremental = incremental.sum(axis=observation_axes)
    exposure = _exposures(chosen, prepared, period_indices, n_periods)
    treatment_baseline = np.full(count, np.nan)
    for position, (family, index, _) in enumerate(chosen):
        if family is _Family.TREATMENT:
            assert baselines is not None
            treatment_baseline[position] = baselines[index]

    with np.errstate(invalid="ignore", divide="ignore"):
        defined = total_reference != 0
        contribution_share = np.divide(
            total_incremental,
            total_reference[..., None],
            out=np.full_like(total_incremental, np.nan),
            where=defined[..., None],
        )
        baseline_share = np.divide(
            total_baseline, total_reference, out=np.full_like(total_baseline, np.nan), where=defined
        )
        measurable = np.isfinite(exposure) & (exposure > 0)
        effectiveness = np.divide(
            total_incremental, exposure, out=np.full_like(total_incremental, np.nan), where=measurable
        )

    axes = ("chain", "draw", *retained)
    channel_axes = (*axes, "channel")
    channel_types = np.asarray([family.value for family, _, _ in chosen])
    baseline_rule = treatment_baselines if isinstance(treatment_baselines, str) else "custom"
    dataset = xr.Dataset(
        {
            "reference_response": (axes, reference),
            "baseline_response": (axes, baseline),
            "joint_incremental_response": (axes, joint),
            "incremental_response": (channel_axes, incremental),
            "contribution_share": (("chain", "draw", "channel"), contribution_share),
            "baseline_share": (("chain", "draw"), baseline_share),
            "effectiveness": (("chain", "draw", "channel"), effectiveness),
            "exposure": ("channel", exposure),
            "treatment_baseline": ("channel", treatment_baseline),
        },
        coords={
            "chain": coordinates["chain"],
            "draw": coordinates["draw"],
            **breakdown_coords,
            "channel": np.asarray(selected),
            "channel_type": ("channel", channel_types),
            "period": time_labels[period_indices],
            "response_period": time_labels[response_indices],
        },
        attrs={
            "quantity": quantity,
            "group": group,
            "intervention": "exposures or reach set to zero and treatments set to baseline levels during periods",
            "treatment_baselines": baseline_rule,
            "history": "fixed",
            "response_units": "original outcome units",
        },
    )
    return dataset


def _exposure_roles(family: _Family) -> tuple[str, ...]:
    """Name the prepared roles whose product measures the family's exposure."""
    match family:
        case _Family.MEDIA:
            return ("media",)
        case _Family.REACH_FREQUENCY:
            return ("reach", "media_frequency")
        case _Family.ORGANIC_MEDIA:
            return ("organic_media",)
        case _Family.ORGANIC_REACH_FREQUENCY:
            return ("organic_reach", "organic_frequency")
        case _Family.TREATMENT:
            return ("treatments",)


def _family_labels(family: _Family, prepared: PreparedData) -> tuple[str, ...]:
    """List the member labels of a family in prepared order."""
    match family:
        case _Family.MEDIA:
            return tuple(prepared.channels)
        case _Family.REACH_FREQUENCY:
            return tuple(prepared.rf_channels)
        case _Family.ORGANIC_MEDIA:
            return tuple(prepared.organic_channels)
        case _Family.ORGANIC_REACH_FREQUENCY:
            return tuple(prepared.organic_rf_channels)
        case _Family.TREATMENT:
            return tuple(prepared.columns.get("treatments", ()))


def _training_data(model: Model) -> PreparedData:
    """Recover the training observations in original units for baseline statistics."""
    training = _result_data(model)
    assert training is not None
    if model.scaling is None:
        return training
    original = model.scaling.inverse_transform(training)
    return original


def _treatment_baselines(
    rule: _TreatmentBaselines, training: PreparedData | None, labels: tuple[str, ...]
) -> NDArray[np.float64] | None:
    """Resolve one counterfactual level per treatment in original units."""
    if isinstance(rule, str):
        if rule not in ("min", "max"):
            raise ValueError("treatment_baselines must be 'min', 'max', or a mapping from treatment names to levels")
        word: _BaselineRule = "max" if rule == "max" else "min"
        settings: dict[str, _BaselineRule | float] = {name: word for name in labels}
    elif isinstance(rule, Mapping):
        unknown = [name for name in rule if name not in labels]
        if unknown:
            raise ValueError(f"treatment_baselines names unknown treatments {unknown}")
        settings = {name: rule.get(name, "min") for name in labels}
    else:
        raise ValueError("treatment_baselines must be 'min', 'max', or a mapping from treatment names to levels")
    if not labels:
        return None

    assert training is not None
    observed = np.asarray(training.arrays["treatments"], dtype=np.float64)
    levels = np.empty(len(labels))
    for index, name in enumerate(labels):
        setting = settings[name]
        if isinstance(setting, str) and setting == "min":
            levels[index] = observed[..., index].min()
        elif isinstance(setting, str) and setting == "max":
            levels[index] = observed[..., index].max()
        elif isinstance(setting, bool) or not isinstance(setting, Real) or not np.isfinite(setting):
            raise ValueError(f"treatment_baselines level for {name!r} must be 'min', 'max', or a finite number")
        else:
            levels[index] = float(setting)
    return levels


def _removed_role(family: _Family) -> str:
    """Name the role that changes when a member of the family is removed."""
    roles = _exposure_roles(family)
    role = roles[0]
    return role


def _counterfactual_dtype(current: jax.Array, replacement: NDArray[np.generic], model_dtype: DTypeLike) -> DTypeLike:
    """Keep a boolean or integer input's dtype when its counterfactual level is exactly representable in it.

    A model that indexes with an integer input, or negates a boolean one, still
    receives that dtype from the scenarios that rewrite it. Those include the
    baseline because it rewrites every removable input.
    """
    values = np.asarray(replacement, dtype=np.float64)
    if jnp.issubdtype(current.dtype, jnp.bool_):
        binary = bool(np.all((values == 0) | (values == 1)))
        boolean_dtype = current.dtype if binary else model_dtype
        return boolean_dtype
    if not jnp.issubdtype(current.dtype, jnp.integer):
        return model_dtype
    integer_dtype = cast(np.dtype[np.integer], np.dtype(current.dtype))
    limits = np.iinfo(integer_dtype)
    integral = bool(np.all(values == np.rint(values)))
    within = bool(np.all(values >= limits.min) and np.all(values <= limits.max))
    exact_dtype = current.dtype if integral and within else model_dtype
    return exact_dtype


def _exposures(
    chosen: Sequence[tuple[_Family, int, str]],
    prepared: PreparedData,
    period_indices: NDArray[np.intp],
    n_periods: int,
) -> NDArray[np.float64]:
    """Sum original-unit exposures during the removal periods and leave treatments undefined."""
    exposure = np.full(len(chosen), np.nan)
    for position, (family, index, _) in enumerate(chosen):
        if family is _Family.TREATMENT:
            continue
        factors = [
            np.asarray(prepared.arrays[role], dtype=np.float64)[-n_periods:][period_indices][..., index]
            for role in _exposure_roles(family)
        ]
        # A family always names at least one role, so this never falls back to the empty-product literal.
        units = cast(NDArray[np.float64], math.prod(factors))
        exposure[position] = units.sum()
    return exposure
