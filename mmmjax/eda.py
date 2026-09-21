"""Exploratory data analysis for marketing inputs before model specification."""

from numbers import Real
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from mmmjax._results import _prepared_coordinates
from mmmjax.data import PreparedData

__all__ = ["check_data", "check_prior"]


def check_prior(
    draws: xr.DataArray,
    *,
    lower: float | xr.DataArray | None = None,
    upper: float | xr.DataArray | None = None,
) -> xr.Dataset:
    """Inspect prior probability outside a range meaningful to the modeler.

    Use existing prior predictions, parameters, or calculated quantities such
    as channel ROI. Limits must use the same units as the selected quantity.
    No sampling, model changes, or automatic pass/fail decisions are performed.

    Parameters
    ----------
    draws : xarray.DataArray
        Prior draws with a ``draw`` axis and an optional ``chain`` axis.
        Other axes are retained. Aggregate within each draw first when checking
        a total rather than individual periods, geographies, or channels.
    lower, upper : float or xarray.DataArray, optional
        Inclusive plausible limits. Supply at least one. Labeled limits may
        vary across the retained axes and must match their coordinate labels.
        Omit a limit to leave that side unbounded.

    Returns
    -------
    xarray.Dataset
        Checks over retained axes with the following fields.

        - **finite_draws**, **nonfinite_fraction** — Finite count and nonfinite fraction
        - **probability_below**, **probability_above** — Probability beyond each
          supplied limit
        - **probability_outside** — Probability outside either limit
        - **lower**, **upper** — Supplied limits in evaluated order

        Probabilities condition on finite draws and are NaN where none are finite.
    """
    if not isinstance(draws, xr.DataArray):
        raise TypeError("draws must be an xarray.DataArray selected from prior results")
    if "draw" not in draws.dims:
        raise ValueError("draws must include a draw dimension")
    sample_dims = [dim for dim in ("chain", "draw") if dim in draws.dims]
    if any(draws.sizes[dim] == 0 for dim in sample_dims):
        raise ValueError("draws must contain at least one draw in each sampling dimension")
    if draws.dtype.kind not in "biuf":
        raise TypeError("draws must contain real numeric values")
    if lower is None and upper is None:
        raise ValueError("Supply lower or upper to define a plausible range")

    # A sampling-free template keeps geographic and channel labels attached
    # while checking limits, without allocating another copy of every draw.
    template = draws.isel({dim: 0 for dim in sample_dims}, drop=True)
    bounds = {
        name: _prior_bound(value, template, name)
        for name, value in (("lower", lower), ("upper", upper))
        if value is not None
    }
    if "lower" in bounds and "upper" in bounds and bool((bounds["lower"] > bounds["upper"]).any()):
        raise ValueError("lower must not exceed upper at any location")

    finite = cast(xr.DataArray, np.isfinite(draws))
    count = finite.sum(dim=sample_dims)
    denominator = count.where(count > 0)
    variables = {
        "finite_draws": count,
        "nonfinite_fraction": (~finite).mean(dim=sample_dims),
    }
    outside = xr.zeros_like(finite)
    for name, bound in bounds.items():
        exceeds = finite & ((draws < bound) if name == "lower" else (draws > bound))
        tail = "below" if name == "lower" else "above"
        variables[f"probability_{tail}"] = exceeds.sum(dim=sample_dims) / denominator
        variables[name] = bound
        outside = outside | exceeds

    variables["probability_outside"] = outside.sum(dim=sample_dims) / denominator
    if set(variables) & (set(template.coords) | set(template.dims)):
        raise ValueError(
            "Draw coordinates must not use names reserved for the reported checks. Rename those coordinates"
        )

    result = xr.Dataset(
        variables,
        attrs={
            "quantity": str(draws.name) if draws.name is not None else "unnamed",
            "sample_count": int(np.prod([draws.sizes[dim] for dim in sample_dims])),
            "probability_scope": "finite draws only",
        },
    )
    if "units" in draws.attrs:
        result.attrs["units"] = draws.attrs["units"]
    return result.copy(deep=True)


def _prior_bound(value: float | xr.DataArray, template: xr.DataArray, name: str) -> xr.DataArray:
    """Align nonsampling limits without dropping or introducing observations."""
    if isinstance(value, xr.DataArray):
        if value.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric limits")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite real limits")
        if any(dim not in template.dims for dim in value.dims):
            raise ValueError(f"{name} dimensions must be retained quantity axes, not sampling axes")

        # Bounds may be supplied in a different label order. Require exactly
        # the same labels so alignment cannot silently remove observations.
        for dim in value.dims:
            if value.sizes[dim] != template.sizes[dim]:
                raise ValueError(f"{name} must match the quantity's {dim!r} size and labels")
            if dim not in value.coords or dim not in template.coords:
                raise ValueError(f"{name} and the quantity must both label the {dim!r} axis")
            index, target = value.get_index(dim), template.get_index(dim)
            if not index.is_unique or not target.is_unique or np.any(index.get_indexer(target) < 0):
                raise ValueError(f"{name} must match the quantity's {dim!r} labels without duplicates")
            value = value.isel({dim: index.get_indexer(target)})

        # Retain the quantity's auxiliary labels, not incidental coordinates
        # attached to a limit array by a previous calculation.
        bound = xr.DataArray(value.data, dims=value.dims, coords={dim: template.coords[dim] for dim in value.dims})
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number or labeled xarray.DataArray")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite. Omit it to leave that side unbounded")
        bound = xr.DataArray(float(value))

    return bound.broadcast_like(template).transpose(*template.dims)


def check_data(
    data: PreparedData,
    *,
    max_zero_fraction: float = 0.8,
    correlation_threshold: float = 0.9,
) -> xr.DataTree:
    """Inspect marketing inputs for measurement issues and overlapping signals.

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
        Five report groups with features labeled by role and source column.

        - **coverage** — Modeling and history dates, groups, and cadence
        - **series** — Minimum, maximum, mean, population standard deviation,
          nonzero periods, zero fraction, longest zero run, quartiles, and
          ``constant``/``sparse`` flags by feature and optional group.
          ``outlier_periods`` counts values beyond quartiles plus or minus
          1.5 interquartile ranges. ``std_without_outliers`` excludes these values,
          and ``outlier_driven_variation`` flags variation lost entirely without them.
        - **pairs** — Predictor correlations and ``high_correlation`` flags,
          excluding outcomes and spend. ``within_group_correlation`` removes
          group temporal means and has its own flag. Constant inputs give NaN.
          Exposure pairs include jointly and exclusively active counts, activity
          overlap as intersection over union, and ``matching_activity`` for
          identical nonzero patterns with temporal on/off variation. These count
          observations, not campaigns. Other pairs have NaN overlap and -1 counts.
          All-inactive pairs have NaN overlap. Constant or always-active pairs
          are not flagged as matching.
        - **spend** — Dated spend/exposure mismatches and cost-per-exposure
          outliers. Reach-frequency exposure is reach times frequency. Costs are
          NaN without exposure. Fences use quartiles plus or minus 1.5 interquartile
          ranges across time within each channel and group.
        - **predictors** — ``vif`` variance inflation factors, with NaN for constant
          predictors and infinity for linear dependence. Grouped inputs also
          report unadjusted variation explained by group and time indicators,
          separately and together.
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

    values, features, roles, columns, channels = _data_values(data)
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

    predictors = _predictor_checks(
        values[:, :, selected], np.asarray(features, dtype=str)[selected], bool(data.group_columns)
    )
    predictors = predictors.assign_coords(
        role=("feature", np.asarray(roles, dtype=str)[selected]),
        column=("feature", np.asarray(columns, dtype=str)[selected]),
        channel=("feature", np.asarray(channels, dtype=str)[selected]),
    )

    spend = _spend_checks(values, features, roles, columns, channels)
    spend = spend.assign_coords(time=coordinates["time"])
    if not data.group_columns:
        spend = spend.isel(group=0, drop=True)
    else:
        spend = spend.assign_coords({"group": coordinates["group"], **auxiliary})

    return xr.DataTree.from_dict(
        {"coverage": coverage, "series": series, "pairs": pairs, "spend": spend, "predictors": predictors},
        name="data_checks",
    )


def _spend_checks(
    values: NDArray[np.float64],
    features: list[str],
    roles: list[str],
    columns: list[str],
    channels: list[str],
) -> xr.Dataset:
    """Compare paired spending and exposure over each group's modeling periods."""
    role_labels = np.asarray(roles, dtype=str)
    spend_indices: list[int] = []
    exposure_indices: list[int] = []
    frequency_columns: list[str] = []
    exposure_blocks: list[NDArray[np.float64]] = []

    for spend_role, exposure_role in (("spend", "media"), ("rf_spend", "reach")):
        selected = np.flatnonzero(role_labels == spend_role)
        if selected.size == 0:
            continue

        exposures = np.flatnonzero(role_labels == exposure_role)
        exposure = values[:, :, exposures]
        if spend_role == "rf_spend":
            frequencies = np.flatnonzero(role_labels == "media_frequency")
            exposure = exposure * values[:, :, frequencies]
            frequency_columns.extend(columns[index] for index in frequencies)
        else:
            frequency_columns.extend([""] * len(selected))

        spend_indices.extend(selected.tolist())
        exposure_indices.extend(exposures.tolist())
        exposure_blocks.append(exposure)

    spend = values[:, :, spend_indices]
    exposure = np.concatenate(exposure_blocks, axis=-1) if exposure_blocks else np.empty_like(spend)
    has_exposure = exposure > 0
    cost = np.divide(spend, exposure, out=np.full_like(spend, np.nan), where=has_exposure)
    lower = np.full(spend.shape[1:], np.nan)
    upper = np.full_like(lower, np.nan)

    # Only observations with exposure define a cost. All-inactive channels
    # have no reference costs, rather than artificial zero-cost observations.
    for group in range(spend.shape[1]):
        for feature in range(spend.shape[2]):
            eligible = cost[has_exposure[:, group, feature], group, feature]
            if eligible.size:
                q1, q3 = np.quantile(eligible, [0.25, 0.75])
                width = 1.5 * (q3 - q1)
                lower[group, feature] = q1 - width
                upper[group, feature] = q3 + width

    observations = {
        "spend_without_exposure": (spend > 0) & ~has_exposure,
        "exposure_without_spend": has_exposure & (spend == 0),
        "cost_per_exposure": cost,
        "cost_outlier": (cost < lower) | (cost > upper),
    }
    summaries = {
        "cost_observations": has_exposure.sum(axis=0),
        "cost_lower_fence": lower,
        "cost_upper_fence": upper,
    }
    result = xr.Dataset(
        {
            **{name: (("time", "group", "feature"), array) for name, array in observations.items()},
            **{name: (("feature", "group"), array.T) for name, array in summaries.items()},
        },
        coords={
            "feature": np.asarray(features, dtype=str)[spend_indices],
            "role": ("feature", role_labels[spend_indices]),
            "column": ("feature", np.asarray(columns, dtype=str)[spend_indices]),
            "channel": ("feature", np.asarray(channels, dtype=str)[spend_indices]),
            "exposure_column": ("feature", np.asarray(columns, dtype=str)[exposure_indices]),
            "frequency_column": ("feature", np.asarray(frequency_columns, dtype=str)),
        },
        attrs={"outlier_iqr_multiplier": 1.5, "period_scope": "modeling periods only"},
    )
    result["cost_per_exposure"].attrs["description"] = (
        "Spend per selected media unit or reach times frequency. NaN without exposure"
    )
    result["cost_outlier"].attrs["description"] = (
        "Cost outside IQR fences calculated over modeling periods within each channel and group"
    )
    result["spend_without_exposure"].attrs["description"] = "Positive spend with zero exposure"
    result["exposure_without_spend"].attrs["description"] = "Positive exposure with zero spend"
    return result


def _data_values(
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

    lower_quartile, upper_quartile = np.quantile(values, [0.25, 0.75], axis=0)
    interquartile_range = upper_quartile - lower_quartile
    retained = (values >= lower_quartile - 1.5 * interquartile_range) & (
        values <= upper_quartile + 1.5 * interquartile_range
    )
    retained_minimum = np.min(scaled, axis=0, where=retained, initial=np.inf)
    retained_maximum = np.max(scaled, axis=0, where=retained, initial=-np.inf)

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
        "lower_quartile": lower_quartile,
        "upper_quartile": upper_quartile,
        "interquartile_range": interquartile_range,
        "outlier_periods": (~retained).sum(axis=0),
        "std_without_outliers": centered.std(axis=0, where=retained) * magnitude,
        "outlier_driven_variation": (minimum != maximum) & (retained_minimum == retained_maximum),
    }
    result = xr.Dataset(
        {name: (("feature", "group"), array.T) for name, array in statistics.items()},
        attrs={
            "max_zero_fraction": max_zero_fraction,
            "outlier_iqr_multiplier": 1.5,
            "period_scope": "modeling periods only",
        },
    )
    result["std"].attrs["description"] = "Temporal standard deviation with ddof zero"
    result["longest_zero_run"].attrs["description"] = "Longest consecutive zero run in observation periods"
    result["outlier_periods"].attrs["description"] = "Periods strictly outside the quartiles plus or minus 1.5 IQR"
    result["std_without_outliers"].attrs["description"] = (
        "Temporal standard deviation with ddof zero after excluding outliers"
    )
    result["outlier_driven_variation"].attrs["description"] = (
        "Varying series that becomes constant after excluding outliers"
    )
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


def _predictor_checks(
    values: NDArray[np.float64],
    features: NDArray[np.str_],
    grouped: bool,
) -> xr.Dataset:
    """Measure multivariate dependence and additive geographic and time patterns."""
    n_periods, n_groups, n_features = values.shape
    magnitude = np.max(np.abs(values), axis=(0, 1), keepdims=True)
    normalized = values / np.where(magnitude == 0, 1, magnitude)
    centered = normalized - normalized[:1, :1]
    centered -= centered.mean(axis=(0, 1), keepdims=True)
    total_variation = np.sum(centered**2, axis=(0, 1))
    variable = total_variation > 0

    # Unit column norms make the rank tolerance independent of measurement units.
    flattened = centered.reshape(n_periods * n_groups, n_features)
    unit = np.divide(flattened, np.sqrt(total_variation), out=np.zeros_like(flattened), where=variable)
    vif = np.full(n_features, np.nan)
    if np.any(variable):
        vif[variable] = _variance_inflation(unit[:, variable])

    result = xr.Dataset(
        {"vif": ("feature", vif)},
        coords={"feature": features},
        attrs={
            "period_scope": "modeling periods only",
            "predictor_scope": "raw predictors pooled over modeling observations",
            "n_observations": n_periods * n_groups,
            "n_predictors": n_features,
        },
    )
    result["vif"].attrs["description"] = (
        "Variance inflation from regressing each predictor on the others with an intercept. "
        "Constant predictors are NaN and linearly dependent predictors are infinite"
    )

    if grouped:
        group_mean = centered.mean(axis=0, keepdims=True)
        time_mean = centered.mean(axis=1, keepdims=True)
        residuals = {
            "group_r_squared": (centered - group_mean, "group indicators"),
            "time_r_squared": (centered - time_mean, "time indicators"),
            "group_time_r_squared": (centered - group_mean - time_mean, "additive group and time indicators"),
        }
        for name, (residual, description) in residuals.items():
            fraction = np.full(n_features, np.nan)
            np.divide(np.sum(residual**2, axis=(0, 1)), total_variation, out=fraction, where=variable)
            result[name] = ("feature", np.clip(1 - fraction, 0, 1))
            result[name].attrs["description"] = (
                f"Unadjusted fraction of variation explained by {description}. Constant predictors are NaN"
            )

    return result


def _variance_inflation(unit: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute VIFs from centered unit-norm columns, including singular designs."""
    # Retain all right singular vectors without building an observation-sized
    # square matrix when there are more observations than predictors.
    _, singular_values, right = np.linalg.svd(unit, full_matrices=unit.shape[0] < unit.shape[1])
    tolerance = np.finfo(np.float64).eps * max(unit.shape)
    rank = np.count_nonzero(singular_values > tolerance * singular_values[0])
    vif: NDArray[np.float64] = np.sum((right[:rank] / singular_values[:rank, None]) ** 2, axis=0)

    # A pseudoinverse alone gives misleading finite values for dependencies.
    # Only columns involved in a null-space direction have infinite VIF.
    dependent = np.linalg.norm(right[rank:], axis=0) > tolerance
    vif[dependent] = np.inf
    return np.maximum(vif, 1)
