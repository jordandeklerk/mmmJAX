"""Labeled collections of model draws and evaluated quantities."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from numpy.typing import NDArray

from mmmjax.data import PreparedData

_Group: TypeAlias = Mapping[str, ArrayLike]
_Coordinates: TypeAlias = dict[str, NDArray[np.generic]]


def _collect_results(
    posterior: _Group,
    *,
    data: PreparedData | None = None,
    posterior_predictive: _Group | None = None,
    log_likelihood: _Group | None = None,
    sample_stats: _Group | None = None,
    generated_quantities: _Group | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    generated_dims: Mapping[str, Sequence[str]] | None = None,
    coords: Mapping[str, object] | None = None,
    sample_group: Literal["posterior", "prior"] = "posterior",
) -> xr.DataTree:
    """Copy constrained draws, diagnostics, and model inputs into labeled groups."""
    if sample_group not in ("posterior", "prior"):
        raise ValueError("sample_group must be posterior or prior")
    if sample_group == "prior" and (log_likelihood is not None or sample_stats is not None):
        raise ValueError("Prior draws must not contain posterior likelihoods or sampler diagnostics")
    predictive_group = f"{sample_group}_predictive"
    generated_group = "prior_generated_quantities" if sample_group == "prior" else "generated_quantities"

    dimensions = _dimensions(dims)
    output_dimensions = dimensions if generated_dims is None else _dimensions(generated_dims)
    coordinates = _coordinates(coords)
    auxiliary: dict[str, tuple[str, NDArray[np.generic]]] = {}
    observed_data: xr.Dataset | None = None
    constant_data: xr.Dataset | None = None
    if data is not None:
        if not isinstance(data, PreparedData):
            raise TypeError("data must be PreparedData returned by prepare_data")
        observed_data, constant_data, prepared_coords, auxiliary = _prepared_groups(data)
        for name, labels in prepared_coords.items():
            if name in coordinates and not _same_labels(coordinates[name], labels):
                raise ValueError(f"Coordinate {name!r} conflicts with prepared data labels")
            coordinates[name] = labels

    parameter_dataset = _dataset(posterior, sample_group, dimensions, coordinates)
    if not parameter_dataset.data_vars:
        raise ValueError(f"{sample_group} must contain at least one variable")
    for axis in ("chain", "draw"):
        if parameter_dataset.sizes[axis] == 0:
            raise ValueError(f"{sample_group} must contain at least one {axis}")
        # Array mappings in other groups inherit the parameter draws' sample labels.
        labels = parameter_dataset.coords[axis].values
        if axis in coordinates and not _same_labels(coordinates[axis], labels):
            raise ValueError(f"Coordinate {axis!r} conflicts with {sample_group} sample labels")
        coordinates[axis] = labels.copy()

    groups: dict[str, xr.Dataset] = {sample_group: parameter_dataset}
    for name, values in (
        (predictive_group, posterior_predictive),
        ("log_likelihood", log_likelihood),
        ("sample_stats", sample_stats),
        (generated_group, generated_quantities),
        ("observed_data", observed_data),
        ("constant_data", constant_data),
    ):
        if values is None:
            continue
        # Diagnostics do not inherit model parameter axes. Prepared datasets already have labels.
        axes = output_dimensions if name in (predictive_group, "log_likelihood", generated_group) else {}
        dataset = _dataset(values, name, axes, coordinates)
        if not dataset.data_vars:
            continue
        if name in (predictive_group, "log_likelihood", generated_group, "sample_stats"):
            for axis in ("chain", "draw"):
                if axis in dataset.dims and not _same_labels(dataset.coords[axis].values, coordinates[axis]):
                    raise ValueError(f"{name} must match {sample_group} {axis} coordinates")
        groups[name] = dataset

    for dataset in groups.values():
        # Auxiliary group labels only apply to the original ordered group axis.
        if auxiliary and "group" in dataset.dims and _same_labels(dataset.coords["group"].values, coordinates["group"]):
            for name, (axis, group_labels) in auxiliary.items():
                if name in dataset.variables:
                    raise ValueError(f"Variable {name!r} conflicts with a prepared group label")
                dataset.coords[name] = (axis, group_labels.copy())

    # Keep coordinates on sibling groups so their observation axes stay independent.
    tree = xr.DataTree.from_dict(groups, name="results")
    tree.attrs["creation_library"] = "mmmjax"
    return tree


def _dimensions(dims: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    """Normalize explicitly named variable axes."""
    if dims is None:
        return {}
    if not isinstance(dims, Mapping):
        raise TypeError("dims must map variable names to sequences of dimension names")
    result = {}
    for name, axes in dims.items():
        _name(name)
        if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence):
            raise TypeError(f"Dimensions for {name!r} must be a sequence of names, not a string")
        for axis in axes:
            _name(axis)
        if len(set(axes)) != len(axes):
            raise ValueError(f"Dimensions for {name!r} must not repeat")
        if set(axes) & {"chain", "draw", "sample", "pred_id"}:
            raise ValueError(f"Dimensions for {name!r} must omit reserved sample dimensions")
        result[name] = tuple(axes)
    return result


def _name(name: object) -> str:
    """Reject unnamed axes and variables."""
    if not isinstance(name, str) or not name:
        raise ValueError("Variable and dimension names must be nonempty strings")
    return name


def _same_labels(left: NDArray[np.generic], right: NDArray[np.generic]) -> bool:
    """Compare coordinate values without unrelated auxiliary coordinates."""
    return xr.DataArray(left).equals(xr.DataArray(right))


def _coordinates(coords: Mapping[str, object] | None) -> _Coordinates:
    """Copy coordinate vectors without assigning meaning from their lengths."""
    if coords is None:
        return {}
    if not isinstance(coords, Mapping):
        raise TypeError("coords must map dimension names to one-dimensional labels")
    result = {}
    for name, labels in coords.items():
        _name(name)
        values = np.array(labels, copy=True)
        if values.ndim != 1:
            raise ValueError(f"Coordinate {name!r} must be one-dimensional")
        if values.dtype == object and values.size and all(isinstance(value, date) for value in values):
            if any(isinstance(value, datetime) and value.utcoffset() is not None for value in values):
                raise ValueError(f"Coordinate {name!r} must use timezone-naive dates or datetimes")
            values = values.astype("datetime64[us]")
        result[name] = values
    return result


def _dataset(
    values: _Group | xr.Dataset,
    group: str,
    dimensions: dict[str, tuple[str, ...]],
    coordinates: _Coordinates,
) -> xr.Dataset:
    """Create one group without aligning its arrays or dropping labeled axes."""
    sampled = group in (
        "posterior",
        "posterior_predictive",
        "log_likelihood",
        "generated_quantities",
        "sample_stats",
        "prior",
        "prior_predictive",
        "prior_generated_quantities",
    )
    if isinstance(values, xr.Dataset):
        if sampled:
            raise TypeError(f"{group} must be a mapping of names to unlabeled arrays")
        dataset = values.copy(deep=True).as_numpy()
    elif isinstance(values, Mapping):
        variables: dict[str, xr.Variable] = {}
        for raw_name, value in values.items():
            name = _name(raw_name)
            if isinstance(value, xr.DataArray):
                raise TypeError("Result mappings must contain unlabeled arrays")
            array = np.array(value, copy=True)
            axes = dimensions.get(name, ())
            has_samples = sampled and not (group == "sample_stats" and array.ndim == 0)
            sample_axes = ("chain", "draw") if has_samples else ()
            if array.ndim < len(sample_axes):
                raise ValueError(f"{group} variable {name!r} requires leading chain and draw axes")
            if array.ndim != len(sample_axes) + len(axes):
                raise ValueError(f"Provide dims matching the non-sample axes of {group} variable {name!r}")
            variables[name] = xr.Variable((*sample_axes, *axes), array)
        conflicts = set(variables) & {axis for variable in variables.values() for axis in variable.dims}
        if conflicts:
            raise ValueError(f"Variables {sorted(conflicts)} must not share dimension names")
        dataset = xr.Dataset(variables)
    else:
        raise TypeError(f"{group} must be a mapping of names to unlabeled arrays")

    for raw_name, variable in dataset.data_vars.items():
        name = _name(raw_name)
        if name in dataset.dims:
            raise ValueError(f"Variable {name!r} must not share a dimension name")
        if len(set(variable.dims)) != len(variable.dims):
            raise ValueError(f"Dimensions for {name!r} must not repeat")
        if sampled and group != "sample_stats" and not {"chain", "draw"}.issubset(variable.dims):
            raise ValueError(f"{group} variable {name!r} requires chain and draw axes")
        if not sampled and set(variable.dims) & {"chain", "draw", "sample", "pred_id"}:
            raise ValueError(f"{group} must not contain sample dimensions")
    for raw_axis, size in dataset.sizes.items():
        axis = _name(raw_axis)
        if axis not in dataset.coords:
            labels = coordinates.get(axis, np.arange(size))
            if len(labels) != size:
                raise ValueError(f"Coordinate {axis!r} has the wrong length for {group}")
            dataset.coords[axis] = labels.copy()
        elif dataset.coords[axis].dims != (axis,):
            raise ValueError(f"Coordinate {axis!r} must label only its own dimension")
    if sampled:
        dataset.attrs["sample_dims"] = [axis for axis in ("chain", "draw") if axis in dataset.dims]
    return dataset


def _prepared_groups(
    data: PreparedData,
) -> tuple[xr.Dataset, xr.Dataset, _Coordinates, dict[str, tuple[str, NDArray[np.generic]]]]:
    """Label prepared observations without extending their window to exposure history."""
    coordinates, auxiliary = _prepared_coordinates(data)
    role_dims = _data_dimensions(data)
    observed: dict[str, xr.Variable] = {}
    constant: dict[str, xr.Variable] = {}
    for name, array in data.arrays.items():
        target = observed if name == "outcome" else constant
        target[name] = xr.Variable(role_dims[name], np.array(array, copy=True))
    return xr.Dataset(observed), xr.Dataset(constant), coordinates, auxiliary


def _prepared_coordinates(
    data: PreparedData,
) -> tuple[_Coordinates, dict[str, tuple[str, NDArray[np.generic]]]]:
    """Share prepared axis labels between parameter declarations and results."""
    labels: dict[str, object] = {"time": data.time_values}
    if data.media_time_values:
        labels["media_time"] = data.media_time_values
    for name, values in (
        ("channel", data.channels),
        ("organic_channel", data.organic_channels),
        ("rf_channel", data.rf_channels),
        ("organic_rf_channel", data.organic_rf_channels),
        ("control", data.columns.get("controls", ())),
        ("treatment", data.columns.get("treatments", ())),
    ):
        if values:
            labels[name] = values
    auxiliary = {}
    if data.group_columns:
        labels["group"] = (
            [values[0] for values in data.group_values]
            if len(data.group_columns) == 1
            else np.arange(len(data.group_values))
        )
        if len(data.group_columns) > 1:
            for index, column in enumerate(data.group_columns):
                auxiliary[f"group_{column}"] = ("group", np.array([values[index] for values in data.group_values]))

    return _coordinates(labels), auxiliary


def _data_dimensions(data: PreparedData) -> dict[str, tuple[str, ...]]:
    """Name each prepared role's axes independently of their lengths."""
    group_axes = ("group",) if data.group_columns else ()
    observation_axes = ("time", *group_axes)
    exposure_axes = ("media_time", *group_axes)
    return {
        "outcome": observation_axes,
        "revenue_per_outcome": observation_axes,
        "media": (*exposure_axes, "channel"),
        "organic_media": (*exposure_axes, "organic_channel"),
        "reach": (*exposure_axes, "rf_channel"),
        "media_frequency": (*exposure_axes, "rf_channel"),
        "organic_reach": (*exposure_axes, "organic_rf_channel"),
        "organic_frequency": (*exposure_axes, "organic_rf_channel"),
        "spend": (*observation_axes, "channel"),
        "rf_spend": (*observation_axes, "rf_channel"),
        "controls": (*observation_axes, "control"),
        "treatments": (*observation_axes, "treatment"),
        "population": group_axes,
    }
