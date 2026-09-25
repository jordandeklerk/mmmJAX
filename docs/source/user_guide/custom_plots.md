---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Customizing plots

Every plot mmmJAX draws is an ordinary plotnine `ggplot` or ArviZ
`PlotCollection`, so the tools of those two libraries change it. This page
adds layers and themes to the plots from [Plotting](plotting), builds new plots
from the analysis outputs, and adjusts the ArviZ diagnostics. The examples use
the ten-channel brand from that page.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/brand_model.py
from prerun import brand_results

results = brand_results(model)

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
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
```

## Adding layers

A [plotnine](https://plotnine.org/) plot is a stack of layers, and `+` adds
another without changing the plot it starts from.

```{code-cell} ipython3
import plotnine as pn

returns = mj.media_metrics(model, results, quantity="mu")
roi = mj.plot_media_metrics(returns)
target = pn.geom_hline(yintercept=4.0, linetype="dotted", color="#c10c90", size=0.8)
roi + target + pn.labs(title="ROI against a target of 4")
```

The dotted line marks a target return of 4 next to the dashed break-even line
the plot draws itself, and `labs` adds a title. Only streaming and Snapchat
clear the target. `roi` still holds the plot without them, so one plot can
start several variants.

## Layout and theme

Facets, coordinates, and themes change how a plot is laid out without touching
what it draws.

```{code-cell} ipython3
import numpy as np

curves = mj.response_curves(model, results, quantity="mu", multipliers=np.linspace(0.0, 2.0, 21))
panels = mj.plot_response_curves(curves)
shared = pn.scale_x_continuous(breaks=[0, 500_000, 1_000_000], labels=["0", "500K", "1M"])
panels + pn.facet_wrap("channel", ncol=5, scales="free_y") + shared
```

`facet_wrap` replaces the plot's own panels with two rows of five, and
`scales="free_y"` gives them one spending axis, whose breaks
`scale_x_continuous` thins to fit the narrow panels. Each curve keeps its own
height, but the size of each budget now shows at a glance, from Meta's curve
running across its panel to Snapchat's in a corner. The plots start from
{func}`~mmmjax.theme_mmmjax`, and a `theme` added afterward changes any of
its settings.

```{code-cell} ipython3
mj.plot_fit(model, results) + pn.theme(figure_size=(12, 5), legend_position="bottom")
```

`figure_size` is in inches and sets a shorter figure, and the legend moves
under the plot. A complete theme such as `pn.theme_bw()` replaces the whole
look instead.

## Your own plots

Every analysis output is an xarray Dataset with a value for each draw, and
`to_dataframe` hands those draws to plotnine.

```{code-cell} ipython3
draws = returns["roi"].to_dataframe().reset_index()
(
    pn.ggplot(draws, pn.aes("roi"))
    + pn.geom_density(fill="#2a2eec", alpha=0.5, color="none")
    + pn.facet_wrap("~channel", ncol=5)
    + pn.coord_cartesian(xlim=(0, 15))
    + pn.labs(x="ROI", y="Density")
    + mj.theme_mmmjax()
)
```

Each panel holds one channel's draws of its return, and `theme_mmmjax` gives
the plot the same look as the rest. `coord_cartesian` zooms in on returns up to
15 without dropping any draws, since Snapchat's long tail would otherwise
stretch every panel's axis. A summary works the same way once xarray reduces
the draws.

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
comes in flights with wide bands at their peaks, while generic search's runs
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
    aes={"color": ["channel"]},
    figure_kwargs={"figsize": (12, 5)},
)
pc.add_legend("channel")
pc.add_title("Retention by channel")
plt.show()
```

`coords` keeps three of the ten channels, `aes` colors each one,
`figure_kwargs` sets the figure's size, and the collection's own methods add
the legend and the title. Meta's retention stays as spread out as its prior,
while YouTube's leans toward shorter carryover, as the adstock plot on
[Plotting](plotting) showed. ArviZ's other plots read the results directly, as
the forest plot on [Changing the model](changing) does.

## Intervals and point estimates

ArviZ's settings decide the intervals and point estimates of every plot, and
`rc_context` changes them for a block of code.

```{code-cell} ipython3
import arviz as az

with az.rc_context({"stats.ci_prob": 0.9, "stats.point_estimate": "median"}):
    median_roi = mj.plot_media_metrics(returns)
median_roi
```

The bars now mark each channel's median return, 3.26 for Meta instead of its
mean of 3.87, and the error bars cover 90 percent. Every median sits below its
mean, since the returns have long right tails. The plot reads the settings when
it is made, so it keeps them after the block ends. Assigning to `az.rcParams`
changes them for the rest of a session.

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
