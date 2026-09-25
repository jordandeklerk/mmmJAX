"""Plots that check a model's fit and its sampler's convergence."""

from collections.abc import Hashable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr

from mmmjax.model.model import Model
from mmmjax.plotting._layers import _channel_axis, _facet, _scales
from mmmjax.plotting._summary import (
    _ci_prob,
    _label,
    _ordered,
    _percent,
    _require_dataset,
    _require_draws,
    _summarize,
    _wrap,
)
from mmmjax.plotting.theme import _colors, _matplotlib_style, theme_mmmjax

if TYPE_CHECKING:
    from arviz_plots.plot_collection import PlotCollection

__all__ = [
    "plot_fit",
    "plot_ppc_dist",
    "plot_prior_posterior",
    "plot_rank",
    "plot_residuals",
    "plot_rhat",
    "plot_trace_dist",
]


def plot_fit(
    model: Model,
    results: xr.DataTree,
    *,
    effects: xr.Dataset | None = None,
    group: Literal["prior", "posterior"] = "posterior",
    by: str | Sequence[str] | None = None,
    n_groups: int | None = 3,
    var_name: str = "outcome",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot observed outcomes against the model's predictions over time.

    The colored line and band show the point estimate and credible interval
    of the predictive draws, and the dark line shows the observations, both
    in the outcome's original units. Groups are summed into one total per
    draw unless ``by`` keeps them apart, and then the most populous groups
    get their own panels. The subtitle, or each panel's title, gives R
    squared and the weighted mean absolute percentage error of the point
    estimate along with the share of periods whose observation falls inside
    the interval. With ``effects``, a third line shows the point estimate of
    the baseline, the expected outcome with media removed and treatments at
    their baseline levels. The gap between it and the predictions is what
    those inputs added.

    Parameters
    ----------
    model : Model
        Model that produced ``results``.
    results : xarray.DataTree
        Output of ``sample``, ``sample_prior``, or ``generate_quantities``
        with predictive draws and ``observed_data``.
    effects : xarray.Dataset, optional
        Output of ``contributions`` for the same draws that keeps time and
        every axis in ``by``. Omit to leave out the baseline.
    group : {"prior", "posterior"}, default "posterior"
        Predictive draws to plot, from ``prior_predictive`` or
        ``posterior_predictive``.
    by : str or sequence of str, optional
        Observation axes to keep as panels, such as ``"group"``. Omit to sum
        every axis except time.
    n_groups : int or None, default 3
        Number of groups to show when ``by`` keeps ``group``. None shows every
        group.
    var_name : str, default "outcome"
        Variable in both the predictive group and ``observed_data``.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Observed and predicted outcomes by period.
    """
    predicted, observed = _predictive_pair(model, results, group, var_name)
    _require_count(n_groups, "n_groups")
    probability = _ci_prob(ci_prob)
    predicted, observed, time, kept = _observation_panels(predicted, observed, results, by, n_groups)
    band = _summarize(predicted, probability)
    points = observed.reset_coords(drop=True).to_dataframe(name="observed").reset_index()
    for dim in kept:
        points = _ordered(points, dim, observed[dim].values)
    matched = band.merge(points, on=[time, *kept])
    subtitle = ""
    if kept:
        titles = {keys: _panel_title(keys, _fit_text(panel, probability)) for keys, panel in _panels(matched, kept)}
        matched = matched.assign(panel=[titles[_key(row)] for row in matched[kept].itertuples(index=False)])
        matched = _ordered(matched, "panel", list(titles.values()))
    else:
        subtitle = _fit_text(matched, probability)
    predicted_label = f"Predicted, {_percent(probability)} interval"
    panels = ["panel"] if kept else []
    series = [
        matched[[time, *panels, "observed"]].rename(columns={"observed": "estimate"}).assign(series="Observed"),
        matched[[time, *panels, "estimate"]].assign(series=predicted_label),
    ]
    if effects is not None:
        baseline = _baseline(effects, predicted, group, time, kept, probability)
        # Joining on the fit's own rows gives the baseline the same periods and panel titles.
        joined = matched[[time, *kept, *panels]].merge(baseline, on=[time, *kept])
        series.append(joined[[time, *panels, "estimate"]].assign(series="Baseline"))
    lines = pd.concat(series, ignore_index=True)
    lines = _ordered(lines, "series", ["Observed", predicted_label, "Baseline"][: len(series)])
    color, baseline_color = _colors(2)
    time_label, outcome_label = _axis_labels(model, var_name)

    plot: pn.ggplot = (
        pn.ggplot(lines, pn.aes(time, "estimate", color="series"))
        + pn.geom_ribbon(
            pn.aes(x=time, ymin="lower", ymax="upper"), data=matched, fill=color, alpha=0.25, inherit_aes=False
        )
        + pn.geom_line(size=0.8)
        + pn.scale_color_manual(values={"Observed": "#262626", predicted_label: color, "Baseline": baseline_color})
        + pn.labs(x=time_label, y=outcome_label, color="", subtitle=subtitle)
        + _scales(lines, time)
        + _facet(panels, stacked=True)
        + theme_mmmjax()
    )
    return plot


def plot_ppc_dist(
    model: Model,
    results: xr.DataTree,
    *,
    group: Literal["prior", "posterior"] = "posterior",
    var_name: str = "outcome",
    **kwargs: Any,
) -> "PlotCollection":
    """Compare the distribution of the observations with predictive draws.

    Draws ArviZ's ``plot_ppc_dist`` after returning a scaled outcome to its
    original units, so the curves read in the outcome's own units. Each curve
    pools every period and group, and ArviZ leaves the observed curve off a
    prior check unless ``visuals`` asks for it.

    Parameters
    ----------
    model : Model
        Model that produced ``results``.
    results : xarray.DataTree
        Output of ``sample``, ``sample_prior``, or ``generate_quantities``
        with predictive draws and ``observed_data``.
    group : {"prior", "posterior"}, default "posterior"
        Predictive draws to plot, from ``prior_predictive`` or
        ``posterior_predictive``.
    var_name : str, default "outcome"
        Variable in both the predictive group and ``observed_data``.
    **kwargs
        Further keywords for ``arviz_plots.plot_ppc_dist``, such as
        ``num_samples``, ``visuals``, or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_legend`` extend.
    """
    predicted, observed = _predictive_pair(model, results, group, var_name)
    from arviz_plots import plot_ppc_dist as arviz_plot_ppc_dist

    predictive = f"{group}_predictive"
    tree = xr.DataTree.from_dict(
        {predictive: xr.Dataset({var_name: predicted}), "observed_data": xr.Dataset({var_name: observed})}
    )
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_ppc_dist(
            tree, group=predictive, var_names=[var_name], **_arviz_options(kwargs)
        )
    return collection


def plot_prior_posterior(
    results: xr.DataTree,
    prior: xr.DataTree,
    *,
    var_names: Sequence[str] | None = None,
    n_groups: int | None = 3,
    n_periods: int | None = 3,
    clip_tails: bool = True,
    **kwargs: Any,
) -> "PlotCollection":
    """Overlay each parameter's prior and posterior distributions.

    Draws ArviZ's ``plot_prior_posterior`` from the posterior of a fit and the
    prior of ``sample_prior``, so the two runs need not be combined first. A
    posterior that looks like its prior shows that the data said little
    about the parameter. Parameters on a ``group`` axis show the groups with
    the largest population, or the first groups when the data has no
    population, and parameters on a ``time`` axis show the first periods.
    ArviZ's ``coords`` picks other labels instead. Extreme draws are pulled in
    so that a heavy-tailed prior cannot squash the posterior into a spike.
    When the parameters still need more panels than ArviZ's
    ``plot.max_subplots`` setting allows, the plot keeps the 12 elements
    whose posterior moved furthest from the prior and says so in its title.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with a ``posterior`` group.
    prior : xarray.DataTree
        Output of ``sample_prior`` with a ``prior`` group.
    var_names : sequence of str, optional
        Parameters to plot. Defaults to every parameter in both groups.
    n_groups : int or None, default 3
        Number of groups to show for parameters on a ``group`` axis. None
        shows every group.
    n_periods : int or None, default 3
        Number of leading periods to show for parameters on a ``time`` axis.
        None shows every period.
    clip_tails : bool, default True
        Limit each panel's draws to a range that covers the 1st to 99th
        percentiles of both the prior and the posterior, widened by a fifth.
    **kwargs
        Further keywords for ``arviz_plots.plot_prior_posterior``, such as
        ``coords``, ``kind``, or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_legend`` extend.
    """
    _require_group(results, "results", "posterior")
    _require_group(prior, "prior", "prior")
    _require_count(n_groups, "n_groups")
    _require_count(n_periods, "n_periods")
    if not isinstance(clip_tails, bool):
        raise TypeError(f"clip_tails must be a bool, got {type(clip_tails).__name__}")
    posterior = results["posterior"].to_dataset()
    draws = prior["prior"].to_dataset()
    shared = [str(name) for name in posterior.data_vars if name in draws.data_vars]
    names = _variables(shared, var_names, "prior and posterior")
    posterior, draws = posterior[names], draws[names]
    coords = dict(kwargs.pop("coords", None) or {})
    if n_groups is not None and "group" in posterior.dims and "group" not in coords:
        largest = _largest_groups((results, prior), list(posterior["group"].values), n_groups)
        posterior, draws = posterior.sel(group=largest), draws.sel(group=largest)
    if n_periods is not None and "time" in posterior.dims and "time" not in coords:
        posterior, draws = posterior.isel(time=slice(0, n_periods)), draws.isel(time=slice(0, n_periods))
    if clip_tails:
        posterior, draws = _clip_tails(posterior, draws)
    title = ""
    limit = _subplot_limit()
    count = _panel_count(posterior)
    if not coords and count > limit:
        chosen = _largest_shifts(posterior, draws, 12)
        posterior, draws = _flatten(posterior, chosen), _flatten(draws, chosen)
        names = [str(name) for name in posterior.data_vars]
        title = f"The {len(chosen)} of {count} parameters whose posterior moved furthest from the prior"
    # ArviZ stacks the prior and posterior along an axis named group, which our own group axis would collide with.
    if "group" in posterior.dims:
        posterior, draws = posterior.rename(group="series"), draws.rename(group="series")
        coords = {"series" if dim == "group" else dim: labels for dim, labels in coords.items()}
    combined = xr.DataTree.from_dict({"posterior": posterior, "prior": draws})
    from arviz_plots import plot_prior_posterior as arviz_plot_prior_posterior

    options = {**kwargs, "coords": coords} if coords else kwargs
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_prior_posterior(combined, var_names=names, **_arviz_options(options))
        if title:
            collection.add_title(title)
    return collection


def plot_rank(
    results: xr.DataTree,
    *,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> "PlotCollection":
    """Draw rank plots that check whether the chains agree.

    Draws ArviZ's ``plot_rank`` on thinned draws, which removes most
    autocorrelation from the test, with a legend of chains. After
    convergence every chain's line stays within its envelope, and ArviZ marks
    stretches that leave it. When the parameters need more panels than
    ArviZ's ``plot.max_subplots`` setting allows, the plot keeps the 12
    elements with the highest R-hat, the ones most likely to have mixed
    poorly, and says so in its title.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with a ``posterior`` group.
    var_names : sequence of str, optional
        Parameters to plot. Defaults to every posterior variable.
    **kwargs
        Further keywords for ``arviz_plots.plot_rank``, such as ``coords``,
        ``thin``, or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_title`` extend.
    """
    data, names, title = _convergence_panels(results, var_names, kwargs, _subplot_limit(), 12)
    from arviz_plots import plot_rank as arviz_plot_rank

    options = {"thin": True, **kwargs}
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_rank(data, var_names=names, **_arviz_options(options))
        # ArviZ colors chains only when the results label them, as sample's results do.
        if ("chain",) in collection.aes_dims.values():
            collection.add_legend("chain")
        if title:
            collection.add_title(title)
    return collection


def plot_residuals(
    model: Model,
    results: xr.DataTree,
    *,
    group: Literal["prior", "posterior"] = "posterior",
    by: str | Sequence[str] | None = None,
    n_groups: int | None = 3,
    var_name: str = "outcome",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot how far the observations fall from the predictive draws over time.

    Each draw's residual is the observation minus the prediction, in the
    outcome's original units. The line follows the point estimate across
    draws and the band its credible interval. A model that captures the
    data's structure leaves a line that wanders around zero without trend or
    seasonal pattern. Groups are summed into one total per draw unless
    ``by`` keeps them apart, and then the most populous groups get their own
    panels.

    Parameters
    ----------
    model : Model
        Model that produced ``results``.
    results : xarray.DataTree
        Output of ``sample``, ``sample_prior``, or ``generate_quantities``
        with predictive draws and ``observed_data``.
    group : {"prior", "posterior"}, default "posterior"
        Predictive draws to compare with, from ``prior_predictive`` or
        ``posterior_predictive``.
    by : str or sequence of str, optional
        Observation axes to keep as panels, such as ``"group"``. Omit to sum
        every axis except time.
    n_groups : int or None, default 3
        Number of groups to show when ``by`` keeps ``group``. None shows every
        group.
    var_name : str, default "outcome"
        Variable in both the predictive group and ``observed_data``.
    ci_prob : float, optional
        Probability of the credible interval. Defaults to ArviZ's
        ``stats.ci_prob`` setting.

    Returns
    -------
    plotnine.ggplot
        Residuals by period.
    """
    predicted, observed = _predictive_pair(model, results, group, var_name)
    _require_count(n_groups, "n_groups")
    probability = _ci_prob(ci_prob)
    predicted, observed, time, kept = _observation_panels(predicted, observed, results, by, n_groups)
    frame = _summarize(observed - predicted, probability)
    color = _colors(1)[0]
    time_label, outcome_label = _axis_labels(model, var_name)

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes(time, "estimate"))
        + pn.geom_hline(yintercept=0, linetype="dashed", color="#262626", size=0.6)
        + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), fill=color, alpha=0.25)
        + pn.geom_line(color=color, size=0.8)
        + pn.labs(x=time_label, y=f"{outcome_label} residual, {_percent(probability)} interval")
        + _scales(frame, time)
        + _facet(kept, stacked=True)
        + theme_mmmjax()
    )
    return plot


def plot_rhat(results: xr.DataTree, *, var_names: Sequence[str] | None = None) -> pn.ggplot:
    """Show the R-hat of every posterior parameter to check that the chains agree.

    Computes ArviZ's rank-normalized split R-hat for every element of every
    parameter. Each parameter gets a box of its elements' values with a point
    for each, so a model with hundreds of elements still fits one plot, and
    a dashed line marks ArviZ's recommended limit of 1.01. Values near one
    mean the chains agree, and the subtitle counts the values past the limit.
    Parameters without a finite R-hat, such as constants, are left out.
    ``plot_rank`` shows the elements with the highest values.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with a ``posterior`` group.
    var_names : sequence of str, optional
        Parameters to plot. Defaults to every posterior variable.

    Returns
    -------
    plotnine.ggplot
        R-hat values by parameter.
    """
    _require_group(results, "results", "posterior")
    names = _variables(_names(results, "posterior"), var_names, "posterior")
    from arviz_stats.sampling_diagnostics import rhat

    # Constant draws have no R-hat, and the rows they leave missing are dropped below.
    with np.errstate(divide="ignore", invalid="ignore"):
        diagnostics = rhat(results["posterior"].to_dataset()[names])
    values = pd.concat(
        [
            pd.DataFrame({"parameter": name, "rhat": np.asarray(diagnostics[name], dtype=float).ravel()})
            for name in names
        ],
        ignore_index=True,
    )
    finite = values[np.isfinite(values["rhat"])]
    if finite.empty:
        raise ValueError("results has no parameter with a finite R-hat. R-hat needs draws that vary")
    frame = _ordered(finite, "parameter", list(dict.fromkeys(finite["parameter"])))
    above = int((frame["rhat"] > 1.01).sum())
    subtitle = (
        f"{above} of {len(frame)} R-hat values are above 1.01"
        if above
        else f"All {len(frame)} R-hat values are at or below 1.01"
    )
    color = _colors(1)[0]
    names = list(frame["parameter"].cat.categories)
    texts, axis = _channel_axis(names)

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes("parameter", "rhat"))
        + pn.geom_hline(yintercept=1.01, linetype="dashed", color="#8c8c8c", size=0.6)
        + pn.geom_boxplot(outlier_shape="", width=0.5, color="#545454", fill="white")
        # A fixed seed keeps the jittered points in place from one drawing to the next.
        + pn.geom_point(position=pn.position_jitter(width=0.15, height=0, random_state=0), color=color, alpha=0.6)
        + pn.scale_x_discrete(labels=dict(zip(names, texts, strict=True)))
        + pn.labs(x="", y="R-hat", subtitle=subtitle)
        + theme_mmmjax()
        + axis
    )
    return plot


def plot_trace_dist(
    results: xr.DataTree,
    *,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> "PlotCollection":
    """Draw each parameter's posterior density beside its draws in sampling order.

    Draws ArviZ's ``plot_trace_dist``, where each chain gets its own line
    style. Converged chains overlap in the densities and show flat, well mixed
    traces without drift. When the parameters have more elements than half
    of ArviZ's ``plot.max_subplots`` setting, the plot keeps the six
    elements with the highest R-hat, each in its own row, and says so in its
    title.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with a ``posterior`` group.
    var_names : sequence of str, optional
        Parameters to plot. Defaults to every posterior variable.
    **kwargs
        Further keywords for ``arviz_plots.plot_trace_dist``, such as
        ``coords``, ``aes``, or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_legend`` extend.
    """
    # Each trace row takes a density panel and a trace panel.
    data, names, title = _convergence_panels(results, var_names, kwargs, _subplot_limit() // 2, 6)
    from arviz_plots import plot_trace_dist as arviz_plot_trace_dist

    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_trace_dist(data, var_names=names, **_arviz_options(kwargs))
        if title:
            collection.add_title(title)
    return collection


def _predictive_pair(
    model: Model,
    results: xr.DataTree,
    group: str,
    var_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Read predictive draws and observations of one variable in the outcome's original units."""
    if not isinstance(model, Model):
        raise TypeError("model must be a Model")
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"results must be an xarray DataTree, got {type(results).__name__}")
    if group not in ("prior", "posterior"):
        raise ValueError(f"group must be 'prior' or 'posterior', got {group!r}")
    if not isinstance(var_name, str):
        raise TypeError(f"var_name must be a string, got {type(var_name).__name__}")
    predictive = f"{group}_predictive"
    _require_group(results, "results", predictive)
    _require_group(results, "results", "observed_data")
    draws = results[predictive].to_dataset()
    observations = results["observed_data"].to_dataset()
    if var_name not in draws.data_vars or var_name not in observations.data_vars:
        raise ValueError(f"var_name must name a variable in both {predictive} and observed_data, got {var_name!r}")
    predicted = draws[var_name]
    observed = observations[var_name]
    _require_draws(predicted, f"results[{predictive!r}][{var_name!r}]")
    scaling = model._data.outcome_scaling if model._data is not None else None
    # Only the prepared outcome carries the outcome transform, and results record whether it was applied.
    if var_name != "outcome" or results.attrs.get("data_scale") != "model" or scaling is None:
        return predicted, observed
    # Population scaling keeps one factor per group, which broadcasts along the last axis.
    predicted = predicted.transpose(..., "group") if "group" in predicted.dims else predicted
    observed = observed.transpose(..., "group") if "group" in observed.dims else observed
    restored_predicted = predicted.copy(data=np.asarray(scaling.inverse_transform(predicted.values)))
    restored_observed = observed.copy(data=np.asarray(scaling.inverse_transform(observed.values)))
    return restored_predicted, restored_observed


def _require_group(results: object, name: str, group: str) -> None:
    """Check that a DataTree holds the group a plot reads."""
    if not isinstance(results, xr.DataTree):
        raise TypeError(f"{name} must be an xarray DataTree, got {type(results).__name__}")
    if group not in results.children:
        raise ValueError(f"{name} has no {group} group")


def _require_count(value: object, name: str) -> None:
    """Check a panel count that None leaves unlimited."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def _observation_panels(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    results: xr.DataTree,
    by: str | Sequence[str] | None,
    n_groups: int | None,
) -> tuple[xr.DataArray, xr.DataArray, str, list[str]]:
    """Sum the observation axes that by leaves out and keep the most populous groups."""
    time = "time" if "time" in observed.dims else str(observed.dims[0])
    if by is not None and not isinstance(by, str) and not isinstance(by, Sequence):
        raise TypeError(f"by must be an axis name or a sequence of them, got {type(by).__name__}")
    kept = [] if by is None else [by] if isinstance(by, str) else [str(dim) for dim in by]
    invalid = [dim for dim in kept if dim == time or dim not in observed.dims]
    if invalid:
        raise ValueError(f"by must name observation axes other than {time}, got {invalid[0]!r}")
    summed = [dim for dim in observed.dims if dim != time and dim not in kept]
    # Summing each draw keeps the uncertainty of the total rather than adding up interval bounds.
    predicted = predicted.sum(summed) if summed else predicted
    observed = observed.sum(summed) if summed else observed
    if "group" in kept and n_groups is not None:
        largest = _largest_groups((results,), list(observed["group"].values), n_groups)
        predicted, observed = predicted.sel(group=largest), observed.sel(group=largest)
    return predicted, observed, time, kept


def _largest_groups(trees: Sequence[xr.DataTree], labels: list[Hashable], count: int) -> list[Hashable]:
    """Rank groups by recorded population and fall back to their order without one."""
    for tree in trees:
        if "constant_data" not in tree.children:
            continue
        inputs = tree["constant_data"].to_dataset()
        if "population" in inputs.data_vars and inputs["population"].dims == ("group",):
            population = inputs["population"].sel(group=labels).values
            order = np.argsort(-population, kind="stable")
            ranked = [labels[index] for index in order[:count]]
            return ranked
    first = labels[:count]
    return first


def _panels(frame: pd.DataFrame, kept: list[str]) -> Iterator[tuple[tuple[str, ...], pd.DataFrame]]:
    """Split a frame by the axes kept as panels in their plotted order."""
    for keys, panel in frame.groupby(kept, observed=True, sort=True):
        labels = keys if isinstance(keys, tuple) else (keys,)
        yield tuple(str(label) for label in labels), panel


def _key(row: tuple[object, ...]) -> tuple[str, ...]:
    """Read one row's panel labels as the key that _panels produces."""
    key = tuple(str(value) for value in row)
    return key


def _panel_title(keys: tuple[str, ...], text: str) -> str:
    """Put a panel's labels in front of its fit summary."""
    title = f"{' '.join(keys)}   {text}"
    return title


def _fit_text(frame: pd.DataFrame, probability: float) -> str:
    """Summarize how closely the point estimate tracks the observations and how often the interval covers them."""
    observed = frame["observed"].to_numpy(dtype=float)
    residual = observed - frame["estimate"].to_numpy(dtype=float)
    spread = float(np.sum((observed - observed.mean()) ** 2))
    magnitude = float(np.sum(np.abs(observed)))
    r_squared = 1.0 - float(np.sum(residual**2)) / spread if spread > 0 else float("nan")
    weighted_error = float(np.sum(np.abs(residual))) / magnitude if magnitude > 0 else float("nan")
    inside = float(np.mean((observed >= frame["lower"].to_numpy()) & (observed <= frame["upper"].to_numpy())))
    text = (
        f"R² {r_squared:.2f}, wMAPE {weighted_error:.1%}, "
        f"{inside:.0%} of periods inside the {_percent(probability)} interval"
    )
    return text


def _baseline(
    effects: object,
    predicted: xr.DataArray,
    group: str,
    time: str,
    kept: list[str],
    probability: float,
) -> pd.DataFrame:
    """Sum each draw's baseline over the axes the fit leaves out and summarize it by period."""
    effects = _require_dataset(effects, "effects", ["baseline_response"])
    baseline = effects["baseline_response"]
    _require_draws(baseline, "effects['baseline_response']")
    drawn = effects.attrs.get("group", group)
    if drawn != group:
        raise ValueError(f"effects must use the {group} draws that group selects, got {drawn!r}")
    missing = [dim for dim in (time, *kept) if dim not in baseline.dims]
    if missing:
        axes = repr((time, *kept)) if kept else repr(time)
        raise ValueError(f"effects must keep the {missing[0]!r} axis. Pass by={axes} to contributions")
    summed = [dim for dim in baseline.dims if dim not in ("chain", "draw", time, *kept)]
    baseline = baseline.sum(summed) if summed else baseline
    if "group" in kept:
        baseline = baseline.sel(group=predicted["group"].values)
    frame = _summarize(baseline, probability)[[time, *kept, "estimate"]]
    return frame


def _axis_labels(model: Model, var_name: str) -> tuple[str, str]:
    """Name the axes after the source columns of prepared data when the model has them."""
    training = model._training
    if training is None:
        return "Time", _label(var_name)
    time_label = _label(training.time_column) if training.time_column else "Time"
    columns = training.layout.columns.get(var_name, ())
    outcome_label = _label(columns[0]) if len(columns) == 1 else _label(var_name)
    return time_label, outcome_label


def _arviz_options(options: dict[str, Any]) -> dict[str, Any]:
    """Give ArviZ the figure size of the plotnine plots and wrapped panel titles unless the caller sets them."""
    figure = {"figsize": (12, 7), **options.get("figure_kwargs", {})}
    sized = {"labeller": _wrapping_labeller(), **options, "figure_kwargs": figure}
    return sized


def _wrapping_labeller() -> Any:
    """Build an ArviZ labeller that wraps long variable and coordinate names in panel titles."""
    from arviz_base.labels import BaseLabeller

    # ArviZ loads on the first plot, so the labeller class is built here rather than at import.
    class _WrappingLabeller(BaseLabeller):
        def dim_coord_to_str(self, dim: Hashable, coord_val: Any, coord_idx: int | Sequence[int]) -> str:
            text = _wrap(str(coord_val), 28)
            return text

        def var_name_to_str(self, var_name: str | None) -> str | None:
            text = var_name if var_name is None else _wrap(str(var_name), 28)
            return text

    labeller = _WrappingLabeller()
    return labeller


def _variables(available: list[str], requested: Sequence[str] | None, group: str) -> list[str]:
    """Resolve the variables a diagnostic draws and reject unknown names."""
    if requested is None:
        return available
    if isinstance(requested, str):
        raise TypeError("var_names must be a sequence of names, not a single string")
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"var_names has no {group} variable {', '.join(repr(name) for name in missing)}")
    names = list(requested)
    return names


def _clip_tails(posterior: xr.Dataset, prior: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Pull each panel's extreme draws in so both distributions keep their shape on a shared axis."""
    clipped_posterior = posterior.copy()
    clipped_prior = prior.copy()
    for name in posterior.data_vars:
        event_dims = [dim for dim in posterior[name].dims if dim not in ("chain", "draw")]
        # Each distribution sets its own range, so the many posterior draws cannot crowd out the prior's body.
        ranges = [
            np.nanquantile(draws[name].transpose("chain", "draw", *event_dims).values, [0.01, 0.99], axis=(0, 1))
            for draws in (posterior, prior)
        ]
        low = np.minimum(ranges[0][0], ranges[1][0])
        high = np.maximum(ranges[0][1], ranges[1][1])
        margin = 0.2 * (high - low)
        lower = xr.DataArray(low - margin, dims=event_dims)
        upper = xr.DataArray(high + margin, dims=event_dims)
        clipped_posterior[name] = posterior[name].clip(lower, upper)
        clipped_prior[name] = prior[name].clip(lower, upper)
    return clipped_posterior, clipped_prior


def _subplot_limit() -> int:
    """Read the most panels ArviZ will draw in one figure."""
    from arviz_base import rcParams

    limit = int(rcParams["plot.max_subplots"])
    return limit


def _panel_count(dataset: xr.Dataset) -> int:
    """Count the elements outside chain and draw that each need a panel."""
    count = sum(
        int(np.prod([dataset[name].sizes[dim] for dim in dataset[name].dims if dim not in ("chain", "draw")]))
        for name in dataset.data_vars
    )
    return count


def _largest_shifts(posterior: xr.Dataset, prior: xr.Dataset, count: int) -> list[tuple[str, dict[str, int]]]:
    """Rank elements by how far the posterior mean moved from the prior mean in prior standard deviations."""
    scored = []
    for name, index in _elements(posterior):
        after = posterior[name].isel(index)
        before = prior[name].isel(index)
        spread = float(before.std())
        shift = abs(float(after.mean()) - float(before.mean())) / spread if spread > 0 else float("inf")
        scored.append((shift, name, index))
    ranked = [(name, index) for _, name, index in sorted(scored, key=lambda item: -item[0])[:count]]
    return ranked


def _elements(dataset: xr.Dataset) -> Iterator[tuple[str, dict[str, int]]]:
    """List every element of every variable by its position along each axis."""
    for name in dataset.data_vars:
        dims = [str(dim) for dim in dataset[name].dims if dim not in ("chain", "draw")]
        for position in np.ndindex(*[dataset[name].sizes[dim] for dim in dims]):
            yield str(name), dict(zip(dims, position, strict=True))


def _flatten(dataset: xr.Dataset, chosen: list[tuple[str, dict[str, int]]]) -> xr.Dataset:
    """Turn chosen elements into scalar variables named after their labels so each gets one panel."""
    variables = {}
    for name, index in chosen:
        element = dataset[name].isel(index)
        labels = ", ".join(str(element[dim].values) for dim in index)
        variables[f"{name}[{labels}]" if labels else name] = element.reset_coords(drop=True)
    flattened = xr.Dataset(variables)
    return flattened


def _names(results: xr.DataTree, group: str) -> list[str]:
    """List the variables of one result group."""
    names = [str(name) for name in results[group].to_dataset().data_vars]
    return names


def _convergence_panels(
    results: xr.DataTree,
    var_names: Sequence[str] | None,
    options: dict[str, Any],
    limit: int,
    keep: int,
) -> tuple[xr.DataTree, list[str], str]:
    """Keep the worst-mixing elements when the parameters need more panels than the limit."""
    _require_group(results, "results", "posterior")
    names = _variables(_names(results, "posterior"), var_names, "posterior")
    posterior = results["posterior"].to_dataset()[names]
    count = _panel_count(posterior)
    if "coords" in options or count <= limit:
        return results, names, ""
    from arviz_stats.sampling_diagnostics import rhat

    diagnostics = rhat(posterior)
    scored = [(float(diagnostics[name].isel(index)), name, index) for name, index in _elements(posterior)]
    # Missing R-hat values sort last, since they flag nothing.
    ranked = sorted(scored, key=lambda item: -item[0] if np.isfinite(item[0]) else np.inf)
    chosen = [(name, index) for _, name, index in ranked[:keep]]
    flattened = _flatten(posterior, chosen)
    data = xr.DataTree.from_dict({"posterior": flattened})
    title = f"The {len(chosen)} of {count} parameters with the highest R-hat"
    return data, [str(name) for name in flattened.data_vars], title
