"""Posterior summaries and labels shared by the plotnine plots."""

import math
import numbers
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence

import pandas as pd
import xarray as xr


def _require_dataset(value: object, name: str, variables: Sequence[str]) -> xr.Dataset:
    """Check that an analysis result is a Dataset holding the variables a plot reads."""
    if not isinstance(value, xr.Dataset):
        raise TypeError(f"{name} must be an xarray Dataset, got {type(value).__name__}")
    missing = [variable for variable in variables if variable not in value.data_vars]
    if missing:
        raise ValueError(f"{name} is missing {', '.join(repr(variable) for variable in missing)}")
    return value


def _require_draws(values: xr.DataArray, name: str) -> None:
    """Check that a variable keeps the chain and draw axes a summary reduces."""
    missing = [dim for dim in ("chain", "draw") if dim not in values.dims]
    if missing:
        raise ValueError(f"{name} must have chain and draw axes, got dims {values.dims}")


def _ci_prob(ci_prob: float | None) -> float:
    """Take the interval probability from ArviZ's settings unless one is given."""
    # ArviZ adds over a second to import mmmjax, so it loads on the first plot instead.
    from arviz_base import rcParams

    probability = rcParams["stats.ci_prob"] if ci_prob is None else ci_prob
    if isinstance(probability, bool) or not isinstance(probability, numbers.Real):
        raise ValueError(f"ci_prob must be between 0 and 1, got {probability!r}")
    resolved = float(probability)
    if not 0 < resolved < 1:
        raise ValueError(f"ci_prob must be between 0 and 1, got {probability!r}")
    return resolved


def _summarize(values: xr.DataArray, probability: float, point_estimate: str | None = None) -> pd.DataFrame:
    """Reduce chain and draw to a point estimate and credible interval in long form."""
    from arviz_base import rcParams
    from arviz_stats.summary import mean, median, mode
    from arviz_stats.visualization import eti, hdi

    sample_dims = ["chain", "draw"]
    # ArviZ rounds point estimates for display unless told otherwise, which would flatten plotted curves.
    match point_estimate or rcParams["stats.point_estimate"]:
        case "median":
            estimate = median(values, dim=sample_dims, round_to="none")
        case "mode":
            estimate = mode(values, dim=sample_dims, round_to="none")
        case _:
            estimate = mean(values, dim=sample_dims, round_to="none")
    interval_function = hdi if rcParams["stats.ci_kind"] == "hdi" else eti
    interval = interval_function(values, prob=probability, dim=sample_dims)
    summary = xr.Dataset(
        {
            "estimate": estimate,
            "lower": interval.sel(ci_bound="lower", drop=True),
            "upper": interval.sel(ci_bound="upper", drop=True),
        }
    )
    if not estimate.dims:
        frame = _scalar_frame(summary)
        return frame
    frame = summary.reset_coords(drop=True).to_dataframe().reset_index()
    # Label axes such as channel or allocation keep their result order instead of sorting in legends and panels.
    for dim in estimate.dims:
        if values[dim].dtype.kind in "OUS":
            frame = _ordered(frame, str(dim), values[dim].values)
    return frame


def _scalar_frame(summary: xr.Dataset) -> pd.DataFrame:
    """Give a summary without labeled axes one row."""
    frame = pd.DataFrame({name: [float(value)] for name, value in summary.data_vars.items()})
    return frame


def _ordered(frame: pd.DataFrame, column: str, labels: Iterable[Hashable]) -> pd.DataFrame:
    """Keep a column's categories as text in the order of its coordinate labels rather than sorted."""
    categories = list(dict.fromkeys(str(label) for label in labels))
    ordered = frame.assign(**{column: pd.Categorical(frame[column].astype(str), categories=categories)})
    return ordered


def _percent(probability: float) -> str:
    """Format an interval probability for a legend or subtitle."""
    whole = math.isclose(probability * 100, round(probability * 100))
    text = f"{probability:.0%}" if whole else f"{probability:.1%}"
    return text


def _label(name: str) -> str:
    """Turn a result variable name into axis text."""
    words = name.replace("_", " ").replace("roi", "ROI")
    text = words[:1].upper() + words[1:]
    return text


def _outcome_words(text: str, dataset: xr.Dataset) -> str:
    """Name the outcome in place of the word response when the analysis recorded its column."""
    outcome = dataset.attrs.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        return text
    name = outcome.replace("_", " ")
    capitalized = name[:1].upper() + name[1:]
    words = [capitalized if word == "Response" else name if word == "response" else word for word in text.split(" ")]
    renamed = " ".join(words)
    return renamed


def _distinct_shortened(names: Sequence[str], limit: int) -> list[str]:
    """Shorten labels past the limit and lengthen the limit until no two shortened labels match."""
    longest = max((len(name) for name in names), default=0)
    shortened = [_shorten(name, limit) for name in names]
    while len(set(shortened)) < len(set(names)) and limit < longest:
        limit += 4
        shortened = [_shorten(name, limit) for name in names]
    return shortened


def _select_panels(
    values: xr.DataArray, coords: Mapping[str, object] | None, n_groups: int | None, measure: str
) -> tuple[dict[str, list[object]], str]:
    """Choose the labels the panel axes keep from coords or from the largest groups when there are too many."""
    if coords is not None and not isinstance(coords, Mapping):
        raise TypeError(f"coords must map axis names to labels, got {type(coords).__name__}")
    invalid = isinstance(n_groups, bool) or not isinstance(n_groups, numbers.Integral) or n_groups < 1
    if n_groups is not None and invalid:
        raise ValueError(f"n_groups must be a positive integer or None, got {n_groups!r}")
    selection = {
        str(dim): list(labels) if isinstance(labels, (list, tuple)) else [labels]
        for dim, labels in (coords or {}).items()
    }
    unknown = [dim for dim in selection if dim not in values.dims]
    if unknown:
        axes = [str(dim) for dim in values.dims if dim not in ("chain", "draw")]
        raise ValueError(f"coords has no axis {', '.join(repr(dim) for dim in unknown)}. Its axes are {axes}")
    try:
        chosen = values.sel(selection)
    except KeyError as error:
        raise ValueError(f"coords asks for labels the results lack: {error}") from None
    if "group" in selection or "group" not in chosen.dims or n_groups is None or chosen.sizes["group"] <= n_groups:
        return selection, ""
    # Sizes come from the point estimates over every other axis, so the busiest groups keep their panels.
    averaged = abs(chosen.mean(("chain", "draw")))
    sizes = averaged.sum([dim for dim in averaged.dims if dim != "group"])
    ranked = sorted(range(sizes.size), key=lambda index: -float(sizes[index]))[:n_groups]
    largest = {chosen["group"].values[index] for index in ranked}
    selection["group"] = [label for label in chosen["group"].values if label in largest]
    total = chosen.sizes["group"]
    note = f"Showing the {n_groups} of {total} groups with the largest {measure}. Pass coords to choose others."
    return selection, note


def _restrict[Result: (xr.DataArray, xr.Dataset)](values: Result, selection: Mapping[str, list[object]]) -> Result:
    """Apply a panel selection to the axes a result has and skip the rest."""
    restricted = values.sel({dim: labels for dim, labels in selection.items() if dim in values.dims})
    return restricted


def _shorten(label: str, limit: int) -> str:
    """Replace the middle of a label longer than the limit with an ellipsis."""
    if len(label) <= limit:
        return label
    # Keeping both ends leaves the parts that usually tell similar channel names apart.
    head = (limit - 1) // 2
    tail = limit - 1 - head
    shortened = f"{label[:head]}\u2026{label[len(label) - tail :]}"
    return shortened


def _wrap(label: str, width: int = 16) -> str:
    """Break a long label after its underscores or spaces so no line runs past the width."""
    pieces = re.findall(r"[^_\s-]+[_\s-]*|[_\s-]+", label)
    lines = [""]
    for piece in pieces:
        if lines[-1] and len(lines[-1]) + len(piece.rstrip()) > width:
            lines.append("")
        lines[-1] += piece
    wrapped = "\n".join(line.rstrip() for line in lines)
    return wrapped


def _facets(dims: Sequence[Hashable], shown: Sequence[Hashable]) -> list[str]:
    """List the axes a plot does not draw so they can become panels."""
    remaining = [str(dim) for dim in dims if dim not in shown and dim not in ("chain", "draw")]
    return remaining


def _pick_channels(
    labels: Iterable[Hashable],
    sizes: Iterable[float] | None,
    requested: Sequence[str] | None,
    limit: int,
    measure: str | None,
) -> tuple[list[str], str]:
    """Keep the requested channels or as many of the largest as a plot can show and note what was left out."""
    names = [str(label) for label in labels]
    if requested is not None:
        if isinstance(requested, str):
            raise TypeError("channels must be a sequence of names, not a single string")
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(f"channels has no {', '.join(repr(name) for name in missing)}")
        chosen = [name for name in names if name in set(requested)]
        return chosen, ""
    if len(names) <= limit:
        return names, ""
    if sizes is None or measure is None:
        chosen = names[:limit]
        note = f"Showing the first {limit} of {len(names)} channels. Pass channels to choose others."
        return chosen, note
    magnitudes = [abs(float(size)) for size in sizes]
    ranked = sorted(range(len(names)), key=lambda index: -magnitudes[index])
    kept = set(ranked[:limit])
    chosen = [name for index, name in enumerate(names) if index in kept]
    note = f"Showing the {limit} of {len(names)} channels with the largest {measure}. Pass channels to choose others."
    return chosen, note


def _plan_spend(plan: object) -> xr.DataArray:
    """Check a budget plan and read its spending by allocation and channel."""
    dataset = _require_dataset(plan, "plan", ["spend", "response_change"])
    allocations = [str(label) for label in dataset["allocation"].values] if "allocation" in dataset.coords else []
    if "reference" not in allocations or "optimized" not in allocations:
        raise ValueError(f"plan must hold 'reference' and 'optimized' allocations, got {allocations}")
    spend = dataset["spend"]
    return spend
