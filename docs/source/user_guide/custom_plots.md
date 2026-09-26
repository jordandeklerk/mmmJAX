---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Customizing plots

Every plot mmmJAX draws is an ordinary plotnine `ggplot` or ArviZ
`PlotCollection`, and every analysis output is an xarray Dataset with a value
for each draw. This page adds to the plots from [Plotting](plotting), combines
them, and builds new plots from the same outputs, all on the ten-channel brand
from that page.

The examples introduce the main ideas and are far from everything the two
libraries can do. The [plotnine](https://plotnine.org/) and
[ArviZ](https://python.arviz.org/) documentation cover the rest.

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

A plotnine plot is a stack of layers, and `+` adds another without changing
the plot it starts from.

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
start several variants. A `theme` adds the same way and changes any setting of
{func}`~mmmjax.theme_mmmjax`.

A layer can also bring data of its own. The source frame records the weeks
linear TV was on air, and a gray band for each one asks whether the model
misses something while it runs.

```{code-cell} ipython3
import numpy as np
import pandas as pd

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
mj.plot_residuals(model, results) + shading
```

Each band covers one of the 50 weeks with TV spending, which fall in 12
flights. `inherit_aes=False` keeps the layer from looking for the plot's own
columns, and infinite bounds stretch each band over the whole panel. The
residuals inside the bands wander around zero like the ones outside, so the
model accounts for the weeks TV runs.

## Reusing what a plot computed

Each plot keeps the frame it draws in `data`, with the point estimates and
interval bounds already worked out, so a different chart can start from it.

```{code-cell} ipython3
frame = roi.data
order = frame.sort_values("estimate")["channel"].astype(str).tolist()
(
    pn.ggplot(frame, pn.aes("channel", "estimate", ymin="lower", ymax="upper"))
    + pn.geom_hline(yintercept=1.0, linetype="dashed", color="#8c8c8c")
    + pn.geom_pointrange(color="#2a2eec")
    + pn.scale_x_discrete(limits=order)
    + pn.coord_flip()
    + pn.labs(x="", y="ROI, 89% interval")
    + mj.theme_mmmjax()
)
```

These are the returns and intervals of the bars above, sorted by return
instead of spending, with Snapchat at the top and influencer at the bottom.
`limits` fixes the order of the channels, and `coord_flip` turns the chart so
the names read across.

## Combining plots

plotnine stacks plots with `/` and sets them side by side with `|`.

```{code-cell} ipython3
fit = mj.plot_fit(model, results) + pn.labs(x="")
fit / mj.plot_residuals(model, results)
```

The fit sits above its residuals on the same weeks, so a week where the black
line leaves the band lines up with the residual it leaves behind. The panels
align across the two plots, and `labs(x="")` drops the axis title the upper
one would repeat.

## Your own plots

Every analysis output holds a value for each draw, so a question the plots
leave open is often a line of xarray away.

```{code-cell} ipython3
paying = (returns["marginal_roi"] > 1).mean(("chain", "draw"))
shares = paying.to_dataframe(name="probability").reset_index()
(
    pn.ggplot(shares, pn.aes("reorder(channel, probability)", "probability"))
    + pn.geom_col(fill="#2a2eec", alpha=0.8, width=0.7)
    + pn.coord_flip()
    + pn.scale_y_continuous(labels=lambda values: [f"{value:.0%}" for value in values], limits=(0, 1))
    + pn.labs(x="", y="Probability that the next dollar returns more than a dollar")
    + mj.theme_mmmjax()
)
```

The share of draws whose marginal return is above one is the probability that
one more dollar pays for itself. Streaming and Snapchat clear it in 94 percent
of the draws, while generic search is close to a coin flip at 52.5 percent,
and the plan on [Plotting](plotting) cuts its budget.

A summary over time works the same way once xarray reduces the draws.

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

The draws of {func}`~mmmjax.response_curves` give the return on the next
dollar at every level of spending, not only at today's.

```{code-cell} ipython3
curves = mj.response_curves(model, results, quantity="mu")
gained = curves["incremental_response"].diff("multiplier")
spent = curves["spend"].diff("multiplier")
steps = (gained / spent).sel(multiplier=slice(0.5, 2.0))
quantiles = steps.quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
margins = quantiles.to_series().unstack("quantile").rename(columns=names).reset_index()
(
    pn.ggplot(margins, pn.aes("multiplier", "median"))
    + pn.geom_hline(yintercept=1.0, linetype="dashed", color="#8c8c8c")
    + pn.geom_vline(xintercept=1.0, linetype="dotted", color="#8c8c8c")
    + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), fill="#2a2eec", alpha=0.2)
    + pn.geom_line(color="#2a2eec")
    + pn.facet_wrap("channel", ncol=5)
    + pn.scale_x_continuous(breaks=[1.0, 2.0], labels=["100%", "200%"])
    + pn.labs(x="Spending as a share of today's", y="Revenue from the next dollar, 90% interval")
    + mj.theme_mmmjax()
)
```

Each step of a tenth of today's spending divides the revenue it adds by the
dollars it costs, draw by draw, before the quantiles summarize the draws. The
dotted line marks today's spending and the dashed line break-even. Generic
search's median falls below a dollar with a tenth more spending, while Snapchat
and streaming still return more than a dollar at twice today's budgets.

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

`coords` keeps three of the ten channels, `aes` colors each one, and the
collection's own methods add the legend and the title. Meta's retention stays
as spread out as its prior, while YouTube's leans toward shorter carryover, as
the adstock plot on [Plotting](plotting) showed.

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
