"""Descriptive checks for marketing data before model specification."""

from numbers import Real

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from mmmjax._results import _prepared_coordinates
from mmmjax.data import PreparedData

__all__ = ["check_data"]


def check_data(
    data: PreparedData,
    *,
    max_zero_fraction: float = 0.8,
    correlation_threshold: float = 0.9,
) -> xr.DataTree:
    """Inspect marketing inputs for limited variation and overlapping signals.

    Check raw modeling periods without changing the data. Earlier media history
    is counted separately and excluded from the calculations. Flags identify
    patterns to review, not invalid inputs or proof of causal identification.
    Correlations describe raw inputs, not their adstock or saturation transforms.

    Parameters
    ----------
    data : PreparedData
        Inputs from ``prepare_data``, before scaling. Numeric validity and
        observation alignment are checked during preparation.
    max_zero_fraction : float, default 0.8
        Flag a series as sparse when at least this fraction of its periods
        are zero. Must be greater than zero and at most one.
    correlation_threshold : float, default 0.9
        Flag absolute Pearson correlations at or above this value.
        Must be greater than zero and at most one.

    Returns
    -------
    xarray.DataTree
        Labeled report containing three groups.

        - **coverage** records modeling dates, history dates, groups, and cadence.
        - **series** contains minimum, maximum, mean, population standard
          deviation, nonzero periods, zero fraction, longest zero run, and
          ``constant`` and ``sparse`` flags by feature and optional group.
        - **pairs** contains predictor correlations and ``high_correlation``
          flags. For grouped data, ``within_group_correlation`` removes each
          group's temporal mean and has its own flag. Constant inputs give NaN.

        Features are identified by role and source column. Predictor pairs
        exclude outcomes and spend. Exposure pairs also report jointly and
        exclusively active observations, activity overlap as intersection over
        union, and ``matching_activity`` for identical nonzero patterns with
        temporal on/off variation. These describe observations, not campaign IDs.
        Other pairs have NaN overlap and counts of -1. All-inactive pairs have
        NaN overlap. Constant or always-active pairs are not flagged as matching.

        Fixed frequency or sparse promotions may be intentional. No data are
        dropped, and no minimum history or overall readiness score is imposed.

    Examples
    --------
    Inspect exposure columns before choosing their model specification.

    .. ipython::

        In [1]: import polars as pl
           ...: from mmmjax import check_data, prepare_data
           ...: frame = pl.DataFrame({
           ...:     "week": [1, 2, 3, 4, 5],
           ...:     "sales": [100, 120, 110, 130, 105],
           ...:     "video": [0, 200, 0, 400, 0],
           ...:     "social": [0, 100, 0, 200, 0],
           ...: })
           ...: data = prepare_data(
           ...:     frame, time="week", outcome="sales",
           ...:     media=["video", "social"],
           ...: )
           ...: report = check_data(data)
           ...: report["pairs"].to_dataset().to_dataframe()
    """
    if not isinstance(data, PreparedData):
        raise TypeError("data must be PreparedData returned by prepare_data")
    if data._scaling is not None:
        raise ValueError("check_data requires unscaled inputs. Use the original data returned by prepare_data")

    for name, value in (("max_zero_fraction", max_zero_fraction), ("correlation_threshold", correlation_threshold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number greater than zero and at most one")
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be greater than zero and at most one")

    values, features, roles, columns, channels = _readiness_values(data)
    coordinates, auxiliary = _prepared_coordinates(data)
    n_periods, n_groups, _ = values.shape
    coverage = xr.Dataset(
        coords={
            **{name: labels for name, labels in coordinates.items() if name in ("time", "media_time", "group")},
            **auxiliary,
        },
        attrs={
            "n_periods": n_periods,
            "n_groups": n_groups,
            "n_history_periods": max(0, len(data.media_time_values) - n_periods),
            "frequency": data.frequency if data.frequency is not None else "unknown",
            "time_column": data.time_column,
        },
    )

    series = _series_checks(values, max_zero_fraction)
    if not data.group_columns:
        series = series.isel(group=0, drop=True)
    else:
        series = series.assign_coords({"group": coordinates["group"], **auxiliary})
    series = series.assign_coords(
        feature=np.asarray(features, dtype=str),
        role=("feature", np.asarray(roles, dtype=str)),
        column=("feature", np.asarray(columns, dtype=str)),
        channel=("feature", np.asarray(channels, dtype=str)),
    )

    selected = np.array([role not in ("outcome", "spend", "rf_spend") for role in roles], dtype=bool)
    pairs = _pair_checks(
        values[:, :, selected],
        np.asarray(features, dtype=str)[selected],
        np.asarray(roles, dtype=str)[selected],
        bool(data.group_columns),
        correlation_threshold,
    )

    return xr.DataTree.from_dict(
        {"coverage": coverage, "series": series, "pairs": pairs},
        name="data_checks",
    )


def _readiness_values(
    data: PreparedData,
) -> tuple[NDArray[np.float64], list[str], list[str], list[str], list[str]]:
    """Collect unscaled modeling observations in time, group, feature order."""
    n_periods = len(data.time_values)
    n_groups = len(data.group_values) if data.group_columns else 1
    if n_periods == 0 or n_groups == 0:
        raise ValueError("data must contain modeling periods and observations. Use prepare_data")

    exposure_roles = ("media", "organic_media", "reach", "media_frequency", "organic_reach", "organic_frequency")
    channel_labels = {
        "media": data.channels,
        "spend": data.channels,
        "organic_media": data.organic_channels,
        "reach": data.rf_channels,
        "media_frequency": data.rf_channels,
        "rf_spend": data.rf_channels,
        "organic_reach": data.organic_rf_channels,
        "organic_frequency": data.organic_rf_channels,
    }
    blocks: list[NDArray[np.float64]] = []
    features: list[str] = []
    roles: list[str] = []
    columns: list[str] = []
    channels: list[str] = []

    for role in ("outcome", *exposure_roles, "spend", "rf_spend", "controls", "treatments"):
        if role not in data.arrays:
            continue

        names = data.columns.get(role, ())
        if not names or len(set(names)) != len(names) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"Input {role!r} has invalid column labels. Use prepare_data")
        period_count = len(data.media_time_values) if role in exposure_roles else n_periods
        observation_shape = (period_count, n_groups) if data.group_columns else (period_count,)
        expected_shape = observation_shape if role == "outcome" else (*observation_shape, len(names))
        array = np.asarray(data.arrays[role])
        if array.shape != expected_shape or period_count < n_periods or (role == "outcome" and len(names) != 1):
            raise ValueError(f"Input {role!r} has a shape inconsistent with its labels. Use prepare_data")
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"Input {role!r} must contain finite real values. Use prepare_data")
        if role in channel_labels and len(channel_labels[role]) != len(names):
            raise ValueError(f"Input {role!r} has inconsistent channel labels. Use prepare_data")

        # Exposure blocks include earlier history, unlike outcome and spend.
        blocks.append(array[-n_periods:].astype(np.float64).reshape(n_periods, n_groups, len(names)))
        features.extend(f"{role}.{name}" for name in names)
        roles.extend([role] * len(names))
        columns.extend(names)
        channels.extend(channel_labels.get(role, ("",) * len(names)))

    values = np.concatenate(blocks, axis=-1) if blocks else np.empty((n_periods, n_groups, 0), dtype=np.float64)
    return values, features, roles, columns, channels


def _series_checks(values: NDArray[np.float64], max_zero_fraction: float) -> xr.Dataset:
    """Measure each series along time without pooling geographic variation."""
    zero = values == 0
    zero_fraction = zero.mean(axis=0)
    current_run = np.zeros(values.shape[1:], dtype=np.int64)
    longest_run = np.zeros_like(current_run)
    for row in zero:
        current_run = np.where(row, current_run + 1, 0)
        longest_run = np.maximum(longest_run, current_run)

    # Rescale before reductions to avoid overflow when units are large.
    magnitude = np.max(np.abs(values), axis=0)
    scaled = values / np.where(magnitude == 0, 1, magnitude)
    centered = scaled - scaled[:1]
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    statistics = {
        "minimum": minimum,
        "maximum": maximum,
        "mean": scaled.mean(axis=0) * magnitude,
        "std": centered.std(axis=0) * magnitude,
        "nonzero_periods": (~zero).sum(axis=0),
        "zero_fraction": zero_fraction,
        "longest_zero_run": longest_run,
        "constant": minimum == maximum,
        "sparse": zero_fraction >= max_zero_fraction,
    }
    result = xr.Dataset(
        {name: (("feature", "group"), array.T) for name, array in statistics.items()},
        attrs={"max_zero_fraction": max_zero_fraction, "period_scope": "modeling periods only"},
    )
    result["std"].attrs["description"] = "Temporal standard deviation with ddof zero"
    result["longest_zero_run"].attrs["description"] = "Longest consecutive zero run in observation periods"
    return result


def _pair_checks(
    values: NDArray[np.float64],
    features: NDArray[np.str_],
    roles: NDArray[np.str_],
    grouped: bool,
    threshold: float,
) -> xr.Dataset:
    """Compare raw predictors and separate temporal from geographic alignment."""
    first, second = np.triu_indices(values.shape[-1], k=1)
    flattened = values.reshape(values.shape[0] * values.shape[1], values.shape[-1])
    correlation = _correlations(flattened)[first, second]
    result = xr.Dataset(
        {
            "correlation": ("pair", correlation),
            "high_correlation": ("pair", np.abs(correlation) >= threshold),
        },
        coords={
            "pair": np.arange(len(first)),
            "feature_a": ("pair", features[first]),
            "feature_b": ("pair", features[second]),
        },
        attrs={"correlation_threshold": threshold, "period_scope": "modeling periods only"},
    )
    result["correlation"].attrs["description"] = "Pearson correlation pooled over modeling observations"

    if grouped:
        magnitude = np.max(np.abs(values), axis=(0, 1), keepdims=True)
        normalized = values / np.where(magnitude == 0, 1, magnitude)
        centered = normalized - normalized[:1]
        centered -= centered.mean(axis=0, keepdims=True)
        within = _correlations(centered.reshape(flattened.shape))[first, second]
        result["within_group_correlation"] = ("pair", within)
        result["high_within_group_correlation"] = ("pair", np.abs(within) >= threshold)
        result["within_group_correlation"].attrs["description"] = (
            "Pearson correlation after removing each group's temporal mean. No time effects are removed"
        )

    activity_roles = np.isin(roles, ("media", "organic_media", "reach", "organic_reach"))
    exposure_pair = activity_roles[first] & activity_roles[second]
    active = (flattened != 0).astype(np.int64)
    count = active.sum(axis=0)
    joint = (active.T @ active)[first, second]
    union = count[first] + count[second] - joint
    exclusive = union - joint
    overlap = np.full(len(first), np.nan)
    np.divide(joint, union, out=overlap, where=exposure_pair & (union > 0))

    # Identical always-on patterns, or fixed differences between groups, do
    # not identify shared campaign timing. Require temporal on/off variation.
    temporal_activity = np.any((values != 0) != (values[:1] != 0), axis=(0, 1))
    matching = exposure_pair & (exclusive == 0) & temporal_activity[first] & temporal_activity[second]
    result["activity_overlap"] = ("pair", overlap)
    result["matching_activity"] = ("pair", matching)
    result["joint_active_observations"] = ("pair", np.where(exposure_pair, joint, -1))
    result["exclusive_active_observations"] = ("pair", np.where(exposure_pair, exclusive, -1))
    result["matching_activity"].attrs["description"] = "Matching on/off patterns with some temporal variation"
    return result


def _correlations(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Pearson correlations with undefined constant columns left as NaN."""
    magnitude = np.max(np.abs(values), axis=0)
    normalized = values / np.where(magnitude == 0, 1, magnitude)
    centered = normalized - normalized[:1]
    centered -= centered.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered**2, axis=0))
    unit = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
    result: NDArray[np.float64] = np.clip(unit.T @ unit, -1, 1)
    # Preserve the inclusive threshold at one despite normalization roundoff.
    at_boundary = np.abs(result) >= 1 - 8 * np.finfo(np.float64).eps
    result[at_boundary] = np.sign(result[at_boundary])
    result[norm == 0, :] = np.nan
    result[:, norm == 0] = np.nan
    return result
