---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Customizing plots

Every plot mmmJAX draws is an ordinary plotnine `ggplot` or ArviZ
`PlotCollection`. That means you can change a plotnine plot by adding a scale,
label, or theme with `+`, as `+ pn.labs(title="Returns")` retitles one, and
`save` writes it to a file. For anything past these basics, the
[plotnine](https://plotnine.org/) and [ArviZ](https://python.arviz.org/) docs
cover customization in full.

Because every analysis output is an xarray Dataset with a value for each draw,
you can also build figures the built-in plots don't draw. The three below each
start from one mmmJAX output for the ten-channel brand from
[A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_plan, first_model_results

results = first_model_results(model)
```

## Every return in full

{func}`~mmmjax.plot_media_metrics` reduces each channel's return on ad spend
to a mean and an interval. To see the shape of the whole posterior instead, you
can build a ridgeline from the draws that {func}`~mmmjax.media_metrics`
returns.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import gaussian_kde

draws = returns["roi"].stack(sample=("chain", "draw"))
order = draws.mean("sample").to_series().sort_values().index.tolist()
grid = np.union1d(np.linspace(0, float(draws.quantile(0.995)), 400), [1.0])
ridges = []
for position, channel in enumerate(order):
    values = draws.sel(channel=channel).values
    density = gaussian_kde(values)(grid)
    ridges.append(
        pd.DataFrame(
            {
                "channel": channel,
                "roi": grid,
                "base": position,
                "top": position + 1.6 * density / density.max(),
                "side": np.where(grid < 1, "below", "above"),
                "share": f"{(values > 1).mean():.0%} above 1",
            }
        )
    )
ridges = pd.concat(ridges)
# Lower ridges draw later, so each one covers the foot of the ridge above it.
layers = [channel + side for channel in reversed(order) for side in ("below", "above")]
ridges["layer"] = pd.Categorical(ridges["channel"] + ridges["side"], categories=layers)
labels = ridges.drop_duplicates("channel")
(
    pn.ggplot(ridges, pn.aes("roi"))
    + pn.geom_ribbon(
        pn.aes(ymin="base", ymax="top", fill="side", group="layer"),
        color="#2a2eec",
        size=0.5,
        outline_type="upper",
    )
    + pn.geom_vline(xintercept=1, linetype="dashed", color="#8c8c8c")
    + pn.annotate("text", x=1.08, y=len(order) + 0.4, label="Break-even", ha="left", size=9, color="#4d4d4d")
    + pn.geom_text(
        pn.aes(x=float(grid.max()), y="base + 0.25", label="share"),
        data=labels,
        ha="right",
        size=9,
        color="#4d4d4d",
    )
    + pn.scale_fill_manual(values={"above": "#d2d3fb", "below": "#f4b7aa"})
    + pn.scale_y_continuous(
        breaks=list(range(len(order))), labels=order, minor_breaks=[], expand=(0.02, 0, 0.12, 0)
    )
    + pn.guides(fill="none")
    + pn.labs(x="Return on ad spend", y="")
    + mj.theme_mmmjax()
    + pn.theme(figure_size=(12, 6))
)
```

In the ridgeline, every return has a long right tail of draws well above its
mean. The coral foot of each ridge holds the draws below break-even, and its
label gives the share above it. No channel has more than 5 percent of its draws
down there, and YouTube and Streaming pay for themselves in 99 percent of
theirs.

## Where the optimizer moves money

The [budget optimizer](budgets) moves money until every channel's next dollar
returns the same. The left panel follows the revenue each channel's next
dollar brings, from today's spending to the optimized plan, and the right
panel shows how much budget the plan moves.

```{code-cell} ipython3
:tags: [skip-execution]

plan = mj.optimize_budget(model, results, quantity="mu", include_metrics=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

plan = first_model_plan(model, results)
```

```{code-cell} ipython3
:tags: [hide-input]

marginal = plan["marginal_roi"].mean(("chain", "draw")).to_pandas().T
moved = plan["spend"].sel(allocation="optimized") - plan["spend"].sel(allocation="reference")
moves = marginal.assign(change=moved.to_series()).reset_index().sort_values("reference")
moves["direction"] = np.where(moves["change"] >= 0, "Gains budget", "Loses budget")
moves["label"] = [
    f"+${value / 1e3:,.0f}K" if value >= 0 else f"−${-value / 1e3:,.0f}K" for value in moves["change"]
]
common = moves["optimized"].mean()

# Each layer's data names the panel it draws in.
panels = ["Revenue from the next dollar", "Budget moved"]
returns_panel = moves.assign(panel=panels[0])
offset = np.sign(moves["change"]) * 6e3
budget_panel = moves.assign(panel=panels[1], zero=0.0, text_x=moves["change"] + offset)
splits = moves.melt(
    id_vars="channel", value_vars=["reference", "optimized"], var_name="split", value_name="marginal"
)
names = {"reference": "Current spending", "optimized": "Optimized plan"}
splits = splits.assign(panel=panels[0], split=splits["split"].map(names))
common_line = pd.DataFrame({"panel": [panels[0]], "x": [common]})
zero_line = pd.DataFrame({"panel": [panels[1]], "x": [0.0]})
for frame in (returns_panel, budget_panel, splits, common_line, zero_line):
    frame["panel"] = pd.Categorical(frame["panel"], categories=panels)


def dollars(values):
    # The panels share one x scale, so returns read in cents and budgets in thousands.
    labels = []
    for value in values:
        if value == 0:
            labels.append("$0")
        elif abs(value) < 10:
            labels.append(f"${value:.2f}")
        else:
            labels.append(f"{'+' if value > 0 else '−'}${abs(value) / 1e3:,.0f}K")
    return labels


gains = budget_panel[budget_panel["change"] >= 0]
losses = budget_panel[budget_panel["change"] < 0]
(
    pn.ggplot(mapping=pn.aes(y="channel"))
    + pn.geom_vline(pn.aes(xintercept="x"), data=common_line, linetype="dashed", color="#8c8c8c")
    + pn.geom_vline(pn.aes(xintercept="x"), data=zero_line, color="#8c8c8c")
    + pn.geom_segment(
        pn.aes(x="reference", xend="optimized", yend="channel", color="direction"),
        data=returns_panel,
        size=1.2,
    )
    + pn.geom_point(pn.aes(x="marginal", fill="split"), data=splits, size=3.6, color="#262626", stroke=0.8)
    + pn.geom_segment(
        pn.aes(x="zero", xend="change", yend="channel", color="direction"), data=budget_panel, size=7
    )
    + pn.geom_text(pn.aes(x="text_x", label="label", color="direction"), data=gains, ha="left", size=9)
    + pn.geom_text(pn.aes(x="text_x", label="label", color="direction"), data=losses, ha="right", size=9)
    + pn.facet_wrap("panel", scales="free_x", ncol=2)
    + pn.scale_y_discrete(limits=moves["channel"].tolist())
    + pn.scale_x_continuous(labels=dollars, expand=(0.12, 0))
    + pn.scale_color_manual(values={"Gains budget": "#2a2eec", "Loses budget": "#d9432a"})
    + pn.scale_fill_manual(values={"Current spending": "white", "Optimized plan": "#262626"})
    + pn.labs(
        x="",
        y="",
        color="",
        fill="",
        title=f"The optimizer moves budget until every channel's next dollar returns about ${common:.2f}",
    )
    + mj.theme_mmmjax()
    + pn.theme(figure_size=(12, 6), legend_position="top", subplots_adjust={"wspace": 0.08})
)
```

The channels whose next dollar returns the most gain budget, and as they grow,
the return on their next dollar falls. The rest give budget up, so theirs
rises until all ten meet at about \$1.67 on the dashed line. Streaming gains
the most, \$184,000, and Generic search gives up the most, \$145,000.

## When each channel earns

{func}`~mmmjax.contributions` with `by="time"` gives each channel's
incremental revenue in every week and draw. A heatmap of the weekly means,
each scaled to the channel's best week, shows when each channel earns.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
```

```{code-cell} ipython3
:tags: [hide-input]

paid = effects["incremental_response"].sel(channel=effects["channel_type"] == "media")
weekly = paid.mean(("chain", "draw"))
calendar = weekly.to_dataframe(name="revenue").reset_index()
calendar["share"] = calendar["revenue"] / calendar.groupby("channel")["revenue"].transform("max")
ranking = weekly.sum("time").to_series().sort_values().index.tolist()
(
    pn.ggplot(calendar, pn.aes("time", "channel", fill="share"))
    + pn.geom_tile(width=7, height=0.9)
    + pn.scale_y_discrete(limits=ranking)
    + pn.scale_x_datetime(date_labels="%b %Y", expand=(0, 0))
    + pn.scale_fill_gradient(
        low="#f3f7f5", high="#074230", labels=lambda values: [f"{value:.0%}" for value in values]
    )
    + pn.labs(x="", y="", fill="Share of the\nchannel's best week")
    + mj.theme_mmmjax()
    + pn.theme(figure_size=(12, 5))
)
```

Meta, Display, and both search channels spend every week and earn every week.
Streaming, Linear TV, TikTok, and Influencer run flights on one shared
calendar, which Snapchat joins in July 2022, and each flight's revenue fades
over the weeks after it ends as carryover runs out. YouTube runs flights too,
but on a schedule of its own.

The shared calendar makes its channels hard to tell apart, so
[Media effects](media_effects) finds the total they bring firmer than its split
among them.
