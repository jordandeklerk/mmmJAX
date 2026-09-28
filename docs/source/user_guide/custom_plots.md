---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Customizing plots

Every plot mmmJAX draws is an ordinary plotnine `ggplot` or ArviZ
`PlotCollection`, and every analysis output is an xarray Dataset with a value
for each draw. This page recolors, annotates, reorders, and combines the plots
from [Plotting](plotting), and builds one new plot from an analysis output, all
on the ten-channel brand from [A first model](first_model).

The examples cover the main ideas, not everything the two libraries can do.
The [plotnine](https://plotnine.org/) and [ArviZ](https://python.arviz.org/)
documentation cover the rest.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_results

results = first_model_results(model)

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = ".33"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["figure.dpi"] = 100
```

## Colors

A plotnine plot colors its marks through scales, and `+` with a new scale
replaces the plot's own. The ROI bars take their fill and outline from two
scales keyed by the kind of bar.

```{code-cell} ipython3
import plotnine as pn

returns = mj.media_metrics(model, results, quantity="mu")
roi = mj.plot_media_metrics(returns)
coral = {"Channel": "#fb5639", "Other channels": "#a6a6a6"}
roi + pn.scale_fill_manual(values=coral) + pn.scale_color_manual(values=coral)
```

`"Other channels"` is the pooled bar that appears when `channels` leaves some
out. `roi` itself keeps its own colors, so one plot can start several
variants.

:::{admonition} Finding a plot's keys
:class: tip

A comparison of several results keys its colors by each label you pass
instead, and each plot keeps the data it draws in `data`. The column a scale
maps, such as `kind` for these bars or `series` for
{func}`~mmmjax.plot_fit`, holds the keys a new scale needs.
:::

## Annotations

`annotate` places a mark at positions you give, and a plot's own `data`
supplies them, since it holds each point estimate and interval the plot draws.

```{code-cell} ipython3
best = roi.data.loc[roi.data["estimate"].idxmax()]
note = pn.annotate(
    "text",
    x=best["channel"],
    y=best["upper"] + 1.4,
    label="Highest mean return",
    color="#c10c90",
    size=11,
)
roi + note
```

YouTube has the highest mean return at 4.06, and the note sits above its
interval, whose top is 6.84. On a plot over time, `x` takes a date.

```{code-cell} ipython3
import pandas as pd

residuals = mj.plot_residuals(model, results)
worst = residuals.data.loc[residuals.data["estimate"].abs().idxmax()]
label = f"Largest miss, week of {worst['time']:%B %-d, %Y}"
point = pn.annotate("point", x=worst["time"], y=worst["estimate"], color="#c10c90", size=3)
text = pn.annotate(
    "text",
    x=worst["time"] + pd.Timedelta(days=12),
    y=worst["estimate"],
    label=label,
    ha="left",
    color="#c10c90",
    size=11,
)
residuals + point + text
```

The largest residual comes in the week of March 27, 2023, at about \$52,600.

A layer can also bring data of its own. The source data records the weeks
Linear TV was on air, and a gray band over each one shows whether the model
misses something while TV runs.

```{code-cell} ipython3
import numpy as np

weeks = pd.to_datetime(brand.frame["week"])
on_air = brand.frame["linear_tv_spend"] > 0
half_week = pd.Timedelta(days=3.5)
flights = pd.DataFrame({"start": weeks - half_week, "end": weeks + half_week})
shading = pn.geom_rect(
    pn.aes(xmin="start", xmax="end"),
    data=flights[on_air],
    ymin=-np.inf,
    ymax=np.inf,
    fill="#8c8c8c",
    alpha=0.15,
    inherit_aes=False,
)
residuals + shading
```

Each band covers one week of TV spending, and neighboring weeks merge into
the flights. Infinite bounds stretch each band over the whole panel.

:::{admonition} Layers with their own data
:class: tip

`inherit_aes=False` keeps a layer that brings its own data from looking for
the plot's own columns.
:::

The residuals inside the bands center on zero like the ones outside, so the
model shows no steady miss while TV runs. The two largest misses, in March
2023, do fall inside a flight.

## Order, titles, and size

The bars run from the most spending to the least. A new `x` scale with
`limits` puts them in any order, here by mean return.

```{code-cell} ipython3
order = roi.data.sort_values("estimate", ascending=False)["channel"].astype(str).tolist()
(
    roi
    + pn.scale_x_discrete(limits=order)
    + pn.labs(title="Return on ad spend, highest mean first")
    + pn.theme(figure_size=(12, 4))
)
```

`labs` sets any title, subtitle, caption, or axis label. `theme` changes any
setting of {func}`~mmmjax.theme_mmmjax`, from the figure's size here to fonts,
legends, and tick labels.

## Combining plots

plotnine stacks plots with `/` and sets them side by side with `|`.

```{code-cell} ipython3
fit = mj.plot_fit(model, results) + pn.labs(x="")
residuals = mj.plot_residuals(model, results) + pn.labs(y="Residual")
fit / residuals
```

The fit sits above its residuals on the same weeks, so a week where the black
line leaves the band lines up with the residual it leaves behind. The panels
align across the two plots, `labs(x="")` drops the axis title the upper one
would repeat, and a shorter `y` title fits the lower one's half height.

## Your own plots

Every analysis output holds a value for each draw, so a plot the library
doesn't draw is often a few lines of xarray away, and
{func}`~mmmjax.theme_mmmjax` gives it the same look.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
pair = effects["incremental_response"].sel(channel=["Linear TV", "Generic search"])
quantiles = pair.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
names = {0.05: "lower", 0.5: "median", 0.95: "upper"}
band = quantiles.to_series().unstack("quantile").rename(columns=names).reset_index()
colors = {"Linear TV": "#2a2eec", "Generic search": "#fa7c17"}
(
    pn.ggplot(band, pn.aes("time", "median", color="channel", fill="channel"))
    + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), alpha=0.2, color="none")
    + pn.geom_line()
    + pn.scale_color_manual(values=colors)
    + pn.scale_fill_manual(values=colors)
    + pn.scale_x_datetime(date_labels="%b %Y")
    + pn.labs(
        x="Week",
        y="Incremental revenue",
        color="Channel, 90% interval",
        fill="Channel, 90% interval",
    )
    + mj.theme_mmmjax()
)
```

`quantile` reduces the draws to each week's median and 90 percent interval,
and `unstack` gives each quantile its own column. Linear TV's contribution
comes in flights with wide bands at their peaks, while Generic search's runs
every week. The stacked areas of {func}`~mmmjax.plot_contributions` leave that
uncertainty out.

## ArviZ plots

The diagnostics pass any extra keyword to the [ArviZ](https://python.arviz.org/)
function they wrap and return its `PlotCollection`.

```{code-cell} ipython3
pc = mj.plot_trace_dist(
    results,
    var_names=["retention"],
    coords={"channel": ["Meta", "YouTube", "Streaming"]},
    compact=False,
    aes={"color": ["channel"]},
    figure_kwargs={"figsize": (12, 7)},
)
pc.add_legend("channel")
pc.add_title("Retention by channel")
plt.show()
```

`coords` keeps three of the ten channels, `compact=False` gives each its own
row, `aes` colors each one, and the collection's own methods add the legend
and the title. Meta's retention stays as spread out as its prior, while
YouTube's leans toward shorter carryover, as the adstock plot on
[Plotting](plotting) showed.

ArviZ's own functions read the analysis outputs as well, since their draws sit
on the same chain and draw axes as a fit's.

```{code-cell} ipython3
import arviz as az

az.plot_forest(returns[["roi", "marginal_roi"]], combined=True, figure_kwargs={"figsize": (12, 7)})
plt.show()
```

Each row places one channel's return or marginal return, with the point at its
mean, the thick line over the middle half of the draws, and the thin line over
89 percent of them. Every channel's marginal return sits left of its average
return, and Snapchat's thin lines reach furthest in both, since its budget is
the smallest.

## Intervals and point estimates

ArviZ's settings decide the intervals and point estimates of every plot, and
`rc_context` changes them for a block of code.

```{code-cell} ipython3
with az.rc_context({"stats.ci_prob": 0.9, "stats.point_estimate": "median"}):
    median_roi = mj.plot_media_metrics(returns)
median_roi
```

The bars now mark each channel's median return, 2.99 for Meta instead of its
mean of 3.49, and the error bars cover 90 percent. Every median sits below its
mean, since the returns have long right tails.

:::{admonition} When the settings apply
:class: tip

A plot reads the settings when it's made, so it keeps them after the block
ends. Assigning to `az.rcParams` changes them for the rest of a session.
:::

## Saving

`save` writes a plotnine plot in any format matplotlib supports, and a
`PlotCollection` has `savefig`.

```{code-cell} ipython3
:tags: [skip-execution]

roi.save("roi.png", dpi=200)
pc.savefig("retention.png")
```

A bar chart with many channels saves at its full width, so SVG or PDF keeps
every bar sharp. For anything plotnine can't express, `draw` returns the
matplotlib figure behind a plot.
