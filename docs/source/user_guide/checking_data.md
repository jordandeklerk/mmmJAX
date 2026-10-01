---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Checking the data

{func}`~mmmjax.check_data` looks for problems that no model can fix, so read
its report before you write any blocks. This page runs it on the data that
[Data and scaling](data.md) prepares.

```{code-cell} ipython3
:tags: [remove-cell]

import arviz as az
import matplotlib.pyplot as plt
import mmmjax as mj
import pandas as pd

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

brand = mj.simulate_data(seed=7, groups=None)
channels = {
    "meta": "Meta",
    "tiktok": "TikTok",
    "snapchat": "Snapchat",
    "youtube": "YouTube",
    "streaming": "Streaming",
    "linear_tv": "Linear TV",
    "branded_search": "Branded search",
    "generic_search": "Generic search",
    "influencer": "Influencer",
    "display": "Display",
}
data = mj.prepare_data(
    brand.frame,
    time="week",
    outcome="revenue",
    media=[f"{name}_impressions" for name in channels],
    spend=[f"{name}_spend" for name in channels],
    channels=list(channels.values()),
    organic_media=["email_sends"],
    organic_channels=["Email"],
    controls=["demand", "holiday"],
    treatments=["price", "promotion"],
)
```

## Correlated inputs

The report's `pairs` group correlates every pair of inputs and flags a pair as
highly correlated when the correlation passes 0.9 in either direction.

```{code-cell} ipython3
checks = mj.check_data(data)
list(checks.children)
```

The report holds five groups, and the heatmap below draws each pair from
`pairs` once. Clustering orders the inputs so the ones that move together sit
side by side. Only a flagged pair shows its correlation, and a black outline
marks a pair whose `matching_activity` says the two are active in exactly the
same weeks.

```{code-cell} ipython3
:tags: [hide-input]

import plotnine as pn
from scipy.cluster.hierarchy import leaves_list, linkage

pairs = checks["pairs"].to_dataset().to_dataframe()
threshold = checks["pairs"].attrs["correlation_threshold"]
inputs = checks["predictors"].to_dataset().to_dataframe()
kinds = {
    "media": "",
    "organic_media": " (organic)",
    "controls": " (control)",
    "treatments": " (treatment)",
}
names = {row.Index: (row.channel or row.column.capitalize()) + kinds[row.role] for row in inputs.itertuples()}

# The pairs hold each correlation once and skip the diagonal, so mirror them and fill in ones.
mirrored = pairs.rename(columns={"feature_a": "feature_b", "feature_b": "feature_a"})
matrix = pd.concat([pairs, mirrored]).pivot(index="feature_a", columns="feature_b", values="correlation")
matrix = matrix.fillna(1.0)
# Clustering places inputs that move together side by side.
order = [names[feature] for feature in matrix.index[leaves_list(linkage(matrix, method="average"))]]

# Each pair sits below the diagonal, with the input that comes first in the order across the bottom.
first = pairs["feature_a"].map(names)
second = pairs["feature_b"].map(names)
earlier = first.map(order.index) < second.map(order.index)
cells = pairs.assign(across=first.where(earlier, second), down=second.where(earlier, first))
cells["label"] = cells["correlation"].map("{:.2f}".format).where(cells["high_correlation"], "")
matched = cells[cells["matching_activity"]].assign(outline="Active in exactly the same weeks")

(
    pn.ggplot(cells, pn.aes("across", "down"))
    + pn.geom_tile(pn.aes(fill="correlation"), color="white", size=1)
    + pn.geom_text(pn.aes(label="label"), color="white", size=9)
    + pn.geom_tile(pn.aes(color="outline"), data=matched, fill="none", size=1.2)
    + pn.scale_fill_cmap("RdBu_r", limits=(-1, 1))
    + pn.scale_color_manual(values=["black"])
    + pn.scale_x_discrete(limits=order[:-1])
    + pn.scale_y_discrete(limits=order[:0:-1])
    + pn.coord_equal()
    + pn.labs(
        x="",
        y="",
        fill="Correlation",
        color="",
        title=f"Correlation between inputs, labeled beyond ±{threshold}",
    )
    + mj.theme_mmmjax()
    + pn.theme(
        figure_size=(12, 7),
        axis_line=pn.element_blank(),
        axis_ticks=pn.element_blank(),
        axis_text_x=pn.element_text(rotation=45, ha="right"),
    )
)
```

The shared campaign calendar from [The example data](example_data) shows up
right away as a block of dark red. TikTok, Streaming, Linear TV, and Influencer
correlate at 0.98 or 0.99, and every pair among them is outlined, since the
four channels are on air in exactly the same weeks. Email joins them at 0.98
or 0.99, and its outlines tell you the newsletter goes out in exactly the
flight weeks.

Meta, Generic search, and Display run all year, but because they rise in those
same weeks, their labeled pairs still land between 0.91 and 0.97. YouTube and
Branded search, the two channels on calendars of their own, stay pale against
the flight channels. Snapchat shares the calendar too, but the half year before
its launch keeps it below the threshold.

In the top-left corner, price and promotion correlate at -0.92, because the
brand cuts its price in promotion weeks. The brand also runs a promotion in
every campaign week, so the promotion column turns red and the price column
blue beside the flight channels and Email, though neither reaches the
threshold. Because of that overlap, the model has to adjust for the treatments,
or it would hand the promotions' lift to the flight channels.

## Variance inflation

The `predictors` group gives each input's variance inflation factor, a number
that grows as the other inputs predict it better. From the `series` group, the
table adds the share of weeks each input is zero.

```{code-cell} ipython3
predictors = checks["predictors"].to_dataset().to_dataframe()
series = checks["series"].to_dataset().to_dataframe()
predictors[["vif"]].join(series["zero_fraction"]).round({"vif": 1, "zero_fraction": 2})
```

In the table, the factor reaches 120.3 for Email, 105.9 for TikTok, and 87.5
for Linear TV, so the other inputs predict those three almost perfectly. The
matched channels are all off in 0.68 of the weeks. YouTube and Snapchat are
off air about as often, in 0.62 and 0.74 of the weeks, but YouTube's factor of
1.4 says the data can tell its effect apart from the rest. The factors for
price and promotion land in between, at 11.5 and 19.8.

:::{admonition} Priors decide the split
:class: warning

The matched channels are never active apart, so the data sees the combined
lift of a flight and can't say how to split it among them. Any model you fit to
this data leaves that split to its priors.
:::

That's one reason the model in [A first model](first_model) states its media
priors as returns on spend and email's as a share of revenue, quantities you
can hold beliefs about.
[Returns against the truth](first_model.md#returns-against-the-truth), at the
end of that page, shows what the split costs.

A lift test on one of the matched channels can settle part of the split by
centering that channel's ROI prior, as
[Introduction to MMM](../getting_started/intro_to_mmm.md#calibration-with-experiments)
describes. Regional data can too, wherever a region runs a channel off the
shared calendar, as [Geo-level models](geo.md#calendars-by-region) finds.
