"""Posterior summaries and labels shared by the plotnine plots."""

import math
import numbers
import re
from collections.abc import Hashable, Iterable, Sequence

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


def _summarize(values: xr.DataArray, probability: float) -> pd.DataFrame:
    """Reduce chain and draw to ArviZ's point estimate and credible interval in long form."""
    from arviz_base import rcParams
    from arviz_stats.summary import mean, median, mode
    from arviz_stats.visualization import eti, hdi

    sample_dims = ["chain", "draw"]
    # ArviZ rounds point estimates for display unless told otherwise, which would flatten plotted curves.
    match rcParams["stats.point_estimate"]:
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
