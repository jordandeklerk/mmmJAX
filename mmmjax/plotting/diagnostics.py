"""Plots that check a model's fit and its sampler's convergence."""

import numbers
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotnine as pn
import xarray as xr
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from mmmjax.analysis.contribution import contributions
from mmmjax.inference.priors import Prior
from mmmjax.inference.sensitivity import _sensitivity_tree
from mmmjax.model.model import Model
from mmmjax.plotting._layers import _compact, _facet, _scales
from mmmjax.plotting._summary import (
    _ci_prob,
    _distinct_shortened,
    _label,
    _ordered,
    _percent,
    _require_draws,
    _restrict,
    _select_panels,
    _summarize,
    _wrap,
)
from mmmjax.plotting.theme import _colors, _matplotlib_style, theme_mmmjax

if TYPE_CHECKING:
    from arviz_plots.plot_collection import PlotCollection

__all__ = [
    "plot_fit",
    "plot_ppc_dist",
    "plot_ppc_tstat",
    "plot_prior_posterior",
    "plot_psense",
    "plot_rank",
    "plot_residuals",
    "plot_rhat",
    "plot_trace_dist",
]

type _Statistic = Literal["mean", "median", "std", "min", "max", "autocorrelation", "residual_autocorrelation"]


def plot_fit(
    model: Model,
    results: xr.DataTree,
    *,
    show_baseline: bool = False,
    quantity: str | None = None,
    group: Literal["prior", "posterior"] = "posterior",
    by: str | Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    var_name: str = "outcome",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot observed outcomes against the model's predictions over time.

    The colored line and band show the point estimate and credible interval of
    the predictive draws, and the dark line shows the observations, both in the
    outcome's original units. The subtitle, or each panel's title, gives R
    squared and the weighted mean absolute percentage error of the point
    estimate along with the share of periods whose observation falls inside the
    interval.

    Groups are summed into one total per draw unless ``by`` keeps them apart,
    and then the most populous groups get their own panels.

    With ``show_baseline``, a third line shows the point estimate of the
    baseline, the expected outcome with media removed and treatments at their
    baseline levels, from ``contributions`` on the same draws. The gap between
    it and the predictions is what those inputs added.

    Parameters
    ----------
    model : Model
        Model that produced ``results``.
    results : xarray.DataTree
        Output of ``sample``, ``sample_prior``, or ``generate_quantities``
        with predictive draws and ``observed_data``.
    show_baseline : bool, default False
        Add the baseline. Requires ``quantity``.
    quantity : str, optional
        Key returned by ``transformed_parameters`` holding the expected
        outcome, such as ``"mu"``. Required with ``show_baseline``.
    group : {"prior", "posterior"}, default "posterior"
        Predictive draws to plot, from ``prior_predictive`` or
        ``posterior_predictive``.
    by : str or sequence of str, optional
        Observation axes to keep as panels, such as ``"group"``. Omit to sum
        every axis except time.
    coords : mapping of str to sequence, optional
        Labels to keep on the observation axes before anything is summed, as
        in ``{"group": ["north", "south"]}``. Groups chosen here replace the
        ``n_groups`` choice.
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
    _require_baseline_quantity(show_baseline, quantity)
    probability = _ci_prob(ci_prob)
    axes = [str(dim) for dim in observed.dims]
    predicted, observed, time, kept = _observation_panels(predicted, observed, results, by, coords, n_groups)
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
    if show_baseline:
        # Every observation axis stays in the baseline so that coords picks its labels before it is summed.
        effects = contributions(model, results, quantity=str(quantity), group=group, by=axes)
        baseline = _align_with_fit(effects["baseline_response"], predicted, time, kept, coords)
        summary = _summarize(baseline, probability)[[time, *kept, "estimate"]]
        # Joining on the fit's own rows gives the baseline the same periods and panel titles.
        joined = matched[[time, *kept, *panels]].merge(summary, on=[time, *kept])
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


def plot_ppc_tstat(
    model: Model,
    results: xr.DataTree,
    *,
    statistics: Sequence[_Statistic | float | Callable[[NDArray[np.float64]], float]] | None = None,
    quantity: str | None = None,
    group: Literal["prior", "posterior"] = "posterior",
    by: str | Sequence[str] | None = None,
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    var_name: str = "outcome",
    **kwargs: Any,
) -> "PlotCollection":
    """Compare statistics of the observed series with the same statistics of predictive draws.

    Each panel shows the distribution of one statistic across the predictive
    draws in the outcome's original units, and a dot marks its value for the
    observations. The p in the title is the share of draws whose statistic is
    at least the observed one. A share near zero or one means the model
    rarely reproduces that feature of the data.

    Residual autocorrelation measures how strongly each period's residual
    follows the one before. It takes the residuals from each draw's own
    expected outcome, which ``contributions`` computes from ``quantity``, so
    the observed value changes from draw to draw and appears as a black curve
    in place of the dot. A model that misses seasonality, a trend, or a driver
    of demand leaves residuals that run in streaks, and this check catches
    them when checks that pool the periods do not.

    Groups are summed into one series per draw unless ``by`` keeps them apart,
    and then the most populous groups get a row of panels each. The title
    names how many groups were left out.

    Parameters
    ----------
    model : Model
        Model that produced ``results``.
    results : xarray.DataTree
        Output of ``sample``, ``sample_prior``, or ``generate_quantities``
        with predictive draws and ``observed_data``.
    statistics : sequence, optional
        Statistics to compare, each a name from ``"mean"``, ``"median"``,
        ``"std"``, ``"min"``, ``"max"``, ``"autocorrelation"``, and
        ``"residual_autocorrelation"``, a quantile such as ``0.9``, or a
        function that takes one series in time order and returns a number.
        Defaults to ``("residual_autocorrelation", "std", "max")`` with
        ``quantity`` and to ``("autocorrelation", "std", "max")`` without it.
    quantity : str, optional
        Key returned by ``transformed_parameters`` holding the expected
        outcome, such as ``"mu"``. Required for residual autocorrelation.
    group : {"prior", "posterior"}, default "posterior"
        Predictive draws to compare with, from ``prior_predictive`` or
        ``posterior_predictive``.
    by : str or sequence of str, optional
        Observation axes to keep as rows of panels, such as ``"group"``. Omit
        to sum every axis except time.
    coords : mapping of str to sequence, optional
        Labels to keep on the observation axes before anything is summed, as
        in ``{"group": ["north", "south"]}``. Groups chosen here replace the
        ``n_groups`` choice.
    n_groups : int or None, default 3
        Number of groups to show when ``by`` keeps ``group``. None shows every
        group.
    var_name : str, default "outcome"
        Variable in both the predictive group and ``observed_data``.
    **kwargs
        Further keywords for ``arviz_plots.plot_ppc_tstat``, such as
        ``visuals`` or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_title`` extend.
    """
    predicted, observed = _predictive_pair(model, results, group, var_name)
    _require_count(n_groups, "n_groups")
    if quantity is not None and not isinstance(quantity, str):
        raise TypeError(f"quantity must be a string, got {type(quantity).__name__}")
    chosen = _statistics(statistics, quantity)
    axes = [str(dim) for dim in observed.dims]
    available = observed.sizes.get("group", 0)
    predicted, observed, time, kept = _observation_panels(predicted, observed, results, by, coords, n_groups)
    series = predicted.transpose("chain", "draw", *kept, time).values.astype(np.float64)
    actual = observed.transpose(*kept, time).values.astype(np.float64)
    expected = None
    if quantity is not None:
        # Every observation axis stays in the expected outcome so that coords picks its labels before it is summed.
        effects = contributions(model, results, quantity=quantity, group=group, by=axes)
        aligned = _align_with_fit(effects["reference_response"], predicted, time, kept, coords)
        expected = aligned.transpose("chain", "draw", *kept, time).values.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        computed = [
            (
                label,
                _apply_statistic(statistic, series, expected, label),
                _apply_statistic(statistic, actual, expected, label),
            )
            for label, statistic in chosen
        ]
    replicated = {}
    observations = {}
    realized = {}
    titles = {}
    # Each group's statistics fill one row, so the panels run group by group.
    for position in np.ndindex(*[observed.sizes[dim] for dim in kept]):
        keys = tuple(str(observed[dim].values[index]) for dim, index in zip(kept, position, strict=True))
        for label, drawn, seen in computed:
            name = _panel_title(keys, label) if keys else label
            values = drawn[(slice(None), slice(None), *position)]
            # A statistic of residuals has one observed value per draw, and each draw is compared with its own.
            per_draw = seen.ndim == drawn.ndim
            actual_values = seen[(slice(None), slice(None), *position)] if per_draw else seen[position]
            share = float(np.mean(values >= actual_values))
            replicated[name] = (("chain", "draw"), values)
            observations[name] = ((), float(np.median(actual_values)))
            if per_draw:
                realized[name] = np.ravel(actual_values)
            titles[name] = f"{name}, p = {share:.2f}"
    tree = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(replicated),
            "observed_data": xr.Dataset(observations),
        }
    )
    from arviz_plots import plot_ppc_tstat as arviz_plot_ppc_tstat

    options = {"col_wrap": len(chosen), **kwargs}
    rows = -(-len(titles) // int(options["col_wrap"]))
    # A row of density panels reads best at about three and a half inches, and one row keeps a little more.
    height = max(4.5, 3.5 * rows)
    shown = observed.sizes.get("group", 0)
    with mpl.rc_context(_matplotlib_style()), warnings.catch_warnings():
        # ArviZ reads an observed statistic of zero or one as a binary outcome, which these never are.
        warnings.filterwarnings("ignore", message=".*look binary", category=UserWarning)
        # The statistics are already one number per draw, so the mean ArviZ applies leaves them unchanged.
        collection: PlotCollection = arviz_plot_ppc_tstat(tree, t_stat="mean", **_arviz_options(options, height=height))
        _title_panels(collection, titles, realized)
        if "group" in kept and shown < available:
            collection.add_title(f"Showing {shown} of {available} groups. Pass coords to choose others.")
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
    posterior that looks like its prior shows that the data said little about
    the parameter.

    Parameters on a ``group`` axis show the groups with the largest population,
    or the first groups when the data has no population, and parameters on a
    ``time`` axis show the first periods. ArviZ's ``coords`` picks other labels
    instead.

    Extreme draws are pulled in so that a heavy-tailed prior cannot squash the
    posterior into a spike. The figure grows taller with its rows of panels
    unless ``figure_kwargs`` sets its size.

    When the parameters still need more panels than ArviZ's
    ``plot.max_subplots`` setting allows, the plot keeps the 12 elements the
    data narrowed least and says so in its title. Narrowing is measured by
    where the posterior draws fall among the prior draws, so it reads the same
    on a log or logit scale as on the parameter's own.

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
    title = ""
    limit = _subplot_limit()
    count = _panel_count(posterior)
    if not coords and count > limit:
        chosen = _least_narrowed(posterior, draws, 12)
        posterior, draws = _flatten(posterior, chosen), _flatten(draws, chosen)
        names = [str(name) for name in posterior.data_vars]
        title = f"The {len(chosen)} of {count} parameters the data narrowed least"
    if clip_tails:
        posterior, draws = _clip_tails(posterior, draws)
    # ArviZ stacks the prior and posterior along an axis named group, which our own group axis would collide with.
    if "group" in posterior.dims:
        posterior, draws = posterior.rename(group="series"), draws.rename(group="series")
        coords = {"series" if dim == "group" else dim: labels for dim, labels in coords.items()}
    combined = xr.DataTree.from_dict({"posterior": posterior, "prior": draws})
    from arviz_plots import plot_prior_posterior as arviz_plot_prior_posterior

    options = {**kwargs, "coords": coords} if coords else kwargs
    height = _grid_height(_panel_count(posterior), options)
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_prior_posterior(
            combined, var_names=names, **_arviz_options(options, height=height)
        )
        if title:
            collection.add_title(title)
    return collection


def plot_psense(
    results: xr.DataTree,
    *,
    priors: Mapping[str, Prior] | None = None,
    metrics: xr.Dataset | None = None,
    var_names: Sequence[str] | None = None,
    kind: Literal["dist", "quantities"] = "dist",
    **kwargs: Any,
) -> "PlotCollection":
    """Show how each posterior quantity moves when the priors or the likelihood are scaled.

    Raises the priors or the likelihood to a power of 0.8 and 1.25 and
    reweights the posterior draws without refitting, as ``psense_summary``
    does. A quantity that moves when the prior is scaled depends on the prior,
    and one that moves the opposite way under the likelihood points to a
    prior that disagrees with the data.

    With ``kind="dist"``, each row shows an element's distribution under
    each power beside a point estimate and credible interval, for the priors
    on the left and the likelihood on the right. Long tails are cut from view
    so the curves keep their shape, while the intervals use every draw. With
    ``kind="quantities"``, each panel follows one summary, such as the mean,
    across the powers, and dashed lines mark two Monte Carlo standard errors
    around its unscaled value.

    When the elements need more panels than ArviZ's ``plot.max_subplots``
    setting allows, the plot keeps the six most sensitive to the priors and
    says so in its title. The figure grows taller with its rows unless
    ``figure_kwargs`` sets its size.

    Parameters
    ----------
    results : xarray.DataTree
        Output of ``sample`` with ``posterior`` and ``log_likelihood`` groups.
    priors : mapping of str to Prior, optional
        The ``Prior`` objects that ``log_density`` uses, keyed by parameter
        name. Every posterior parameter needs one. Omit to read the
        ``log_prior`` group of ``results``.
    metrics : xarray.Dataset, optional
        Output of ``media_metrics`` or another dataset of values for the
        posterior draws of ``results``. Omit to plot the parameters.
    var_names : sequence of str, optional
        Parameters, or variables of ``metrics``, to plot. Defaults to every
        one with chain and draw axes.
    kind : {"dist", "quantities"}, default "dist"
        Draw distributions or summaries against the power.
    **kwargs
        Further keywords for ``arviz_plots.plot_psense_dist`` or
        ``arviz_plots.plot_psense_quantities``, such as ``coords``,
        ``quantities``, or ``figure_kwargs``.

    Returns
    -------
    arviz_plots.PlotCollection
        ArviZ's plot, which methods such as ``add_title`` extend.
    """
    if kind not in ("dist", "quantities"):
        raise ValueError(f"kind must be 'dist' or 'quantities', got {kind!r}")
    tree, names = _sensitivity_tree(results, priors, metrics, var_names)
    everything = tree["posterior"].to_dataset()
    draws = everything[names]
    quantities = kwargs.get("quantities") or ("mean", "sd")
    columns = 2 if kind == "dist" else 1 if isinstance(quantities, str) else len(quantities)
    count = _panel_count(draws)
    title = ""
    if "coords" not in kwargs and count * columns > _subplot_limit():
        chosen = _most_sensitive(tree, names, 6)
        draws = everything = _flatten(draws, chosen)
        names = [str(name) for name in draws.data_vars]
        noun = "parameters" if metrics is None else "metrics"
        title = f"The {len(chosen)} of {count} {noun} most sensitive to the priors"
    # ArviZ turns a posterior of one variable into an array it cannot plot, so a hidden copy keeps it a dataset.
    if len(everything.data_vars) == 1:
        everything = everything.assign({f"{names[0]} copy": everything[names[0]]})
    data = xr.DataTree.from_dict(
        {
            "posterior": everything,
            "log_prior": tree["log_prior"].to_dataset(),
            "log_likelihood": tree["log_likelihood"].to_dataset(),
        }
    )
    from arviz_plots import plot_psense_dist, plot_psense_quantities

    # A row needs about 1.8 inches for its panels, tick labels, and a title of up to three lines.
    height = 7.0 if "coords" in kwargs else max(7.0, 1.8 * _panel_count(draws))
    draw = plot_psense_dist if kind == "dist" else plot_psense_quantities
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = draw(data, var_names=names, **_arviz_options(kwargs, height=height))
        if kind == "dist":
            _limit_tails(collection, draws, kwargs.get("coords") or {})
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

    Draws ArviZ's ``plot_rank`` on thinned draws and adds a legend of chains.
    Thinning removes most autocorrelation from the test. After convergence
    every chain's line stays within its envelope, and ArviZ marks stretches
    that leave it.

    When the parameters need more panels than ArviZ's ``plot.max_subplots``
    setting allows, the plot keeps the 12 elements with the highest R-hat, the
    ones most likely to have mixed poorly, and says so in its title. The figure
    grows taller with its rows of panels unless ``figure_kwargs`` sets its
    size.

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
    height = _grid_height(_panel_count(data["posterior"].to_dataset()[names]), options)
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_rank(data, var_names=names, **_arviz_options(options, height=height))
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
    coords: Mapping[str, object] | None = None,
    n_groups: int | None = 3,
    var_name: str = "outcome",
    ci_prob: float | None = None,
) -> pn.ggplot:
    """Plot how far the observations fall from the predictive draws over time.

    Each draw's residual is the observation minus the prediction, in the
    outcome's original units. The line follows the point estimate across draws
    and the band its credible interval. A model that captures the data's
    structure leaves a line that wanders around zero without trend or seasonal
    pattern.

    Groups are summed into one total per draw unless ``by`` keeps them apart,
    and then the most populous groups get their own panels.

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
    coords : mapping of str to sequence, optional
        Labels to keep on the observation axes before anything is summed, as
        in ``{"group": ["north", "south"]}``. Groups chosen here replace the
        ``n_groups`` choice.
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
    predicted, observed, time, kept = _observation_panels(predicted, observed, results, by, coords, n_groups)
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
    for each, so a model with hundreds of elements still fits one plot.
    Parameters without a finite R-hat, such as constants, are left out.

    Parameters run down the side, and a dotted line marks ArviZ's recommended
    limit of 1.01. Values near one mean the chains agree. Values past the limit
    turn orange, and the subtitle counts them. ``plot_rank`` shows the elements
    with the highest values.

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
    flagged = frame["rhat"] > 1.01
    above = int(flagged.sum())
    subtitle = (
        f"{above} of {len(frame)} R-hat values are above 1.01"
        if above
        else f"All {len(frame)} R-hat values are at or below 1.01"
    )
    frame = frame.assign(status=np.where(flagged, "Above 1.01", "At or below 1.01"))
    blue, orange = _colors(2)
    names = list(frame["parameter"].cat.categories)
    texts = _distinct_shortened(names, 27)

    plot: pn.ggplot = (
        pn.ggplot(frame, pn.aes("parameter", "rhat"))
        # The first parameter sits at the top once the axes turn.
        + pn.scale_x_discrete(limits=names[::-1], labels=dict(zip(names, texts, strict=True)))
        + pn.geom_hline(yintercept=1.01, linetype="dotted", color="#8c8c8c", size=0.8)
        + pn.annotate("text", x=len(names) + 0.45, y=1.01, label=" 1.01", ha="left", va="center", color="#262626")
        + pn.geom_boxplot(outlier_shape="", width=0.55, color="#545454", fill="#e9eafc", size=0.5)
        # A fixed seed keeps the jittered points in place from one drawing to the next.
        + pn.geom_point(
            pn.aes(fill="status"),
            position=pn.position_jitter(width=0.18, height=0, random_state=0),
            color="white",
            stroke=0.4,
            size=2.6,
            alpha=0.9,
        )
        + pn.scale_fill_manual(values={"At or below 1.01": blue, "Above 1.01": orange})
        + pn.guides(fill="none")
        # The limit stays in view when every value sits well below it.
        + pn.expand_limits(y=1.012)
        + pn.coord_flip()
        + pn.labs(x="", y="R-hat", subtitle=subtitle)
        + theme_mmmjax()
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
    traces without drift.

    When the parameters have more elements than half of ArviZ's
    ``plot.max_subplots`` setting, the plot keeps the six elements with the
    highest R-hat, each in its own row, and says so in its title. Each row's
    name sits under its density, and the figure grows taller with its rows
    unless ``figure_kwargs`` sets its size.

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

    # A row needs about 1.8 inches for its density, tick labels, and a name of two lines.
    height = max(7.0, 1.8 * len(names))
    with mpl.rc_context(_matplotlib_style()):
        collection: PlotCollection = arviz_plot_trace_dist(
            data, var_names=names, **_arviz_options(kwargs, height=height)
        )
        _drop_trace_titles(collection)
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


def _require_baseline_quantity(show_baseline: object, quantity: object) -> None:
    """Check that the baseline has the expected outcome it is computed from."""
    if not isinstance(show_baseline, bool):
        raise TypeError(f"show_baseline must be a bool, got {type(show_baseline).__name__}")
    if quantity is not None and not isinstance(quantity, str):
        raise TypeError(f"quantity must be a string, got {type(quantity).__name__}")
    if show_baseline and quantity is None:
        raise ValueError("quantity must name the expected outcome, such as 'mu', to show the baseline")
    if not show_baseline and quantity is not None:
        raise ValueError("quantity is only used with show_baseline=True")


def _observation_panels(
    predicted: xr.DataArray,
    observed: xr.DataArray,
    results: xr.DataTree,
    by: str | Sequence[str] | None,
    coords: Mapping[str, object] | None,
    n_groups: int | None,
) -> tuple[xr.DataArray, xr.DataArray, str, list[str]]:
    """Keep the requested labels and sum the axes that by leaves out before keeping the most populous groups."""
    selection, _ = _select_panels(observed, coords, None, "")
    predicted, observed = _restrict(predicted, selection), _restrict(observed, selection)
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
    if "group" in kept and "group" not in selection and n_groups is not None:
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


def _align_with_fit(
    values: xr.DataArray,
    predicted: xr.DataArray,
    time: str,
    kept: list[str],
    coords: Mapping[str, object] | None,
) -> xr.DataArray:
    """Keep the labels coords picks and sum each draw over the axes the fit leaves out."""
    selection, _ = _select_panels(values, coords, None, "")
    restricted = _restrict(values, selection)
    summed = [dim for dim in restricted.dims if dim not in ("chain", "draw", time, *kept)]
    aligned = restricted.sum(summed) if summed else restricted
    if "group" in kept:
        aligned = aligned.sel(group=predicted["group"].values)
    return aligned


def _axis_labels(model: Model, var_name: str) -> tuple[str, str]:
    """Name the axes after the source columns of prepared data when the model has them."""
    training = model._training
    if training is None:
        return "Time", _label(var_name)
    time_label = _label(training.time_column) if training.time_column else "Time"
    columns = training.layout.columns.get(var_name, ())
    outcome_label = _label(columns[0]) if len(columns) == 1 else _label(var_name)
    return time_label, outcome_label


def _arviz_options(options: dict[str, Any], *, height: float = 7.0) -> dict[str, Any]:
    """Give ArviZ the figure size of the plotnine plots and wrapped panel titles unless the caller sets them."""
    figure = {"figsize": (12, height), **options.get("figure_kwargs", {})}
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
            text = var_name if var_name is None else _element_label(str(var_name))
            return text

    labeller = _WrappingLabeller()
    return labeller


def _element_label(name: str) -> str:
    """Put a long element's labels on a line of their own so rotated axis titles fit their row."""
    head, bracket, labels = name.partition("[")
    label = f"{_wrap(head, 28)}\n{_wrap(bracket + labels, 28)}" if bracket and len(name) > 16 else _wrap(name, 28)
    return label


def _statistics(
    statistics: object,
    quantity: str | None,
) -> list[tuple[str, str | float | Callable[[NDArray[np.float64]], float]]]:
    """Name each requested statistic and check that it can be computed."""
    first = "autocorrelation" if quantity is None else "residual_autocorrelation"
    requested = (first, "std", "max") if statistics is None else statistics
    if isinstance(requested, str) or not isinstance(requested, Sequence):
        raise TypeError(
            f"statistics must be a sequence of names, quantiles, or functions, got {type(requested).__name__}"
        )
    if not requested:
        raise ValueError("statistics must hold at least one statistic")
    words = {
        "mean": "Mean",
        "median": "Median",
        "std": "Standard deviation",
        "min": "Minimum",
        "max": "Maximum",
        "autocorrelation": "Autocorrelation",
        "residual_autocorrelation": "Residual autocorrelation",
    }
    chosen: list[tuple[str, str | float | Callable[[NDArray[np.float64]], float]]] = []
    for position, statistic in enumerate(requested, start=1):
        if isinstance(statistic, str):
            if statistic not in words:
                raise ValueError(
                    "statistics must name 'mean', 'median', 'std', 'min', 'max', 'autocorrelation', "
                    f"or 'residual_autocorrelation', got {statistic!r}"
                )
            label = words[statistic]
            setting: str | float | Callable[[NDArray[np.float64]], float] = statistic
        elif isinstance(statistic, numbers.Real) and not isinstance(statistic, (bool, np.bool_)):
            if not 0.0 < float(statistic) < 1.0:
                raise ValueError(f"statistics quantiles must lie between 0 and 1, got {statistic!r}")
            label = f"{_ordinal(100.0 * float(statistic))} percentile"
            setting = float(statistic)
        elif callable(statistic):
            name = str(getattr(statistic, "__name__", ""))
            # An anonymous function has no name to show, so its place in the list stands in for one.
            label = f"Statistic {position}" if not name or name == "<lambda>" else _label(name)
            setting = statistic
        else:
            raise TypeError(f"statistics must hold names, quantiles, or functions, got {type(statistic).__name__}")
        if label in [named for named, _ in chosen]:
            raise ValueError(f"statistics must be distinct, got {label!r} twice")
        chosen.append((label, setting))
    if quantity is None and any(setting == "residual_autocorrelation" for _, setting in chosen):
        raise ValueError("quantity must name the expected outcome, such as 'mu', for residual autocorrelation")
    if quantity is not None and all(setting != "residual_autocorrelation" for _, setting in chosen):
        raise ValueError("quantity is only used with residual autocorrelation")
    return chosen


def _ordinal(percent: float) -> str:
    """Write a percentile as an ordinal such as 90th or 2.5th."""
    number = f"{percent:.4g}"
    whole = float(number).is_integer()
    last = int(float(number)) % 10 if whole and int(float(number)) % 100 not in (11, 12, 13) else 0
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(last, "th")
    text = f"{number}{suffix}"
    return text


def _apply_statistic(
    statistic: str | float | Callable[[NDArray[np.float64]], float],
    series: NDArray[np.float64],
    expected: NDArray[np.float64] | None,
    label: str,
) -> NDArray[np.float64]:
    """Compute one statistic of every series along the last axis."""
    match statistic:
        case "mean":
            values = series.mean(axis=-1)
        case "median":
            values = np.median(series, axis=-1)
        case "std":
            values = series.std(axis=-1)
        case "min":
            values = series.min(axis=-1)
        case "max":
            values = series.max(axis=-1)
        case "autocorrelation":
            values = _lag_one(series)
        case "residual_autocorrelation":
            # Each draw's residuals come from its own expected outcome, so the observed series gets one per draw.
            values = _lag_one(series - cast(NDArray[np.float64], expected))
        case float():
            values = np.quantile(series, statistic, axis=-1)
        case _:
            function = cast(Callable[[NDArray[np.float64]], float], statistic)
            values = np.apply_along_axis(function, -1, series)
            if np.shape(values) != series.shape[:-1]:
                raise ValueError(
                    f"statistics functions must return one number per series, got shape {np.shape(values)} "
                    f"from {label!r}"
                )
    computed = np.asarray(values, dtype=np.float64)
    return computed


def _lag_one(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Measure how strongly each value follows the one before it."""
    centered = series - series.mean(axis=-1, keepdims=True)
    covariance = np.sum(centered[..., 1:] * centered[..., :-1], axis=-1)
    variance = np.sum(centered**2, axis=-1)
    correlation: NDArray[np.float64] = covariance / variance
    return correlation


def _title_panels(
    collection: "PlotCollection", titles: Mapping[str, str], realized: Mapping[str, NDArray[np.float64]]
) -> None:
    """Title each panel with its share of draws and draw observed values that vary by draw as a curve."""
    from arviz_stats.base.array import array_stats

    plots = collection.viz["plot"].to_dataset()
    dots = collection.viz["observed_tstat"].to_dataset()
    lines = collection.viz["dist"].to_dataset()
    formatter = FuncFormatter(lambda value, _: _compact([value])[0])
    for name in plots.data_vars:
        axis = plots[name].item()
        axis.set_title(titles[str(name)])
        axis.xaxis.set_major_formatter(formatter)
        if str(name) not in realized:
            continue
        dot = dots[name].item()
        grid, density, _ = array_stats.kde(realized[str(name)])
        axis.plot(grid, density, color=dot.get_facecolor()[0], linewidth=lines[name].item().get_linewidth())
        dot.remove()
        axis.relim()
        axis.autoscale_view()


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


def _least_narrowed(posterior: xr.Dataset, prior: xr.Dataset, count: int) -> list[tuple[str, dict[str, int]]]:
    """Rank elements by how little the data narrowed them on the prior's quantile scale."""
    scored = []
    for name, index in _elements(posterior):
        after = np.ravel(posterior[name].isel(index).values)
        before = np.sort(np.ravel(prior[name].isel(index).values))
        # A draw's place among the prior draws is the same on any increasing scale, such as log or logit.
        places = (np.searchsorted(before, after, side="left") + np.searchsorted(before, after, side="right")) / (
            2 * before.size
        )
        # Places spread like the prior's own have variance 1/12, so an unchanged posterior scores zero.
        narrowing = 1.0 - 12.0 * float(np.var(places))
        scored.append((narrowing, name, index))
    ranked = [(name, index) for _, name, index in sorted(scored, key=lambda item: item[0])[:count]]
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


def _grid_height(panels: int, options: dict[str, Any]) -> float:
    """Give each row of an ArviZ grid three inches so its titles and tick labels keep their room."""
    # Chosen coordinates change the panel count in ways only ArviZ resolves, so they keep the default height.
    if "coords" in options:
        return 7.0
    columns = int(options.get("col_wrap", 4))
    rows = -(-panels // columns)
    height = max(7.0, 3.0 * rows)
    return height


def _most_sensitive(tree: xr.DataTree, names: list[str], count: int) -> list[tuple[str, dict[str, int]]]:
    """Rank elements by how far scaling every prior moves their distribution."""
    from arviz_stats.psense import psense

    sensitivity = psense(tree, var_names=names, group="prior")
    scored = [(float(sensitivity[name].isel(index)), name, index) for name, index in _elements(sensitivity[names])]
    # Missing values sort last, since they flag nothing.
    ranked = sorted(scored, key=lambda item: -item[0] if np.isfinite(item[0]) else np.inf)
    chosen = [(name, index) for _, name, index in ranked[:count]]
    return chosen


def _limit_tails(collection: "PlotCollection", draws: xr.Dataset, coords: Mapping[str, object]) -> None:
    """Cut each density panel's view to the bulk of its draws so a long tail cannot squash the curves."""
    plots = collection.viz["plot"].to_dataset()
    for name in plots.data_vars:
        # The same labels ArviZ drew, where a single label drops its axis as it does in the panels.
        values = draws[str(name)].sel({dim: labels for dim, labels in coords.items() if dim in draws[str(name)].dims})
        low = values.quantile(0.01, dim=("chain", "draw")).drop_vars("quantile")
        high = values.quantile(0.99, dim=("chain", "draw")).drop_vars("quantile")
        margin = 0.2 * (high - low)
        # A limit never reaches past the draws, so only a long tail leaves the view.
        lowest = values.min(("chain", "draw"))
        highest = values.max(("chain", "draw"))
        lower = (low - margin).where(low - margin > lowest, lowest)
        upper = (high + margin).where(high + margin < highest, highest)
        axes = plots[name]
        shown = {dim: axes[dim].values for dim in lower.dims if dim in axes.dims}
        left = lower.sel(shown).broadcast_like(axes).transpose(*axes.dims)
        right = upper.sel(shown).broadcast_like(axes).transpose(*axes.dims)
        for axis, start, stop in zip(np.ravel(axes.values), np.ravel(left.values), np.ravel(right.values), strict=True):
            axis.set_xlim(float(start), float(stop))


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
    groups = {"posterior": flattened}
    # ArviZ marks divergent draws from sample_stats, which the trimmed tree would otherwise lose.
    if "sample_stats" in results.children:
        groups["sample_stats"] = results["sample_stats"].to_dataset()
    data = xr.DataTree.from_dict(groups)
    title = f"The {len(chosen)} of {count} parameters with the highest R-hat"
    return data, [str(name) for name in flattened.data_vars], title


def _drop_trace_titles(collection: "PlotCollection") -> None:
    """Clear the axis title beside each trace, which repeats the name under its density."""
    plots = collection.viz["plot"].to_dataset()
    for name in plots.data_vars:
        for axis in np.ravel(plots[name].sel(column="trace").values):
            axis.set_ylabel("")
