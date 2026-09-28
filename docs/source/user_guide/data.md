---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data and scaling

Every model starts from prepared, scaled data, and the choices you make there
decide what your blocks receive. This page prepares, checks, and scales the
simulated data from [The example data](example_data) with
{func}`~mmmjax.prepare_data` and {func}`~mmmjax.fit_data_scaling`, the same
two calls [A first model](first_model) makes before it writes its blocks.

```{code-cell} ipython3
:tags: [remove-cell]

import arviz as az
import matplotlib.pyplot as plt
import mmmjax as mj

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
```

## Supplied names

{func}`~mmmjax.prepare_data` selects the columns a model uses and gives each
one a role. Here the roles are the outcome, the paid media and their spend,
organic media, controls, and treatments, and `channels` and `organic_channels`
label the two kinds of media. Each role becomes a supplied name, one a block
asks for exactly as spelled, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
explains. {attr}`~mmmjax.PreparedData.model_inputs` lists them together with
the inputs mmmJAX adds.

```{code-cell} ipython3
import pandas as pd

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
data.model_inputs["media"]
```

`model_inputs` maps each supplied name to its kind, its axes, and where it
comes from, as the entry for `media` shows. The tabs below list every name this
data offers and what it holds, grouped by where each comes from. Constants and extra inputs you pass to
{class}`~mmmjax.Data` go under names of your own.

::::{tab-set}

:::{tab-item} Data roles

The columns `prepare_data` selects, one array for each role.

| Supplied name | What it holds | Axes |
| --- | --- | --- |
| `outcome` | The weekly revenue | `time` |
| `media` | The paid channels' impressions | `media_time`, `channel` |
| `organic_media` | The email sends | `media_time`, `organic_channel` |
| `spend` | The paid channels' spend in dollars | `time`, `channel` |
| `controls` | Demand and the holiday flag | `time`, `control` |
| `treatments` | Price and promotion | `time`, `treatment` |

:::

:::{tab-item} Calendar

Positions in time that mmmJAX works out from the dates.

| Supplied name | What it holds | Axes |
| --- | --- | --- |
| `time` | Days since the first training week, which new data counts from too | `time` |
| `media_time` | The same count for the media's weeks | `media_time` |
| `day_of_year` | Each week's day in the calendar year | `time` |
| `media_day_of_year` | The same for the media's weeks | `media_time` |

:::

:::{tab-item} Model inputs

What mmmJAX adds from the prepared data and its scaling.

| Supplied name | What it holds | Axes |
| --- | --- | --- |
| `n_periods` | The number of modeled weeks, as a Python integer | none |
| `outcome_scaling` | The outcome's fitted transform, with `scale` and `inverse_transform` | none |
| `reference` | The training arrays under these same names | none |

:::

::::

The media's weeks match the modeled weeks unless `media_history` adds earlier
ones, as a note in [A first model](first_model) explains. `reference` holds
arrays such as `reference.media` and `reference.spend`, along with
`reference.n_periods`, and keeps them in every scenario, as
[Scenarios](scenarios) explains.

Every role keyword of {func}`~mmmjax.prepare_data` becomes a supplied name
spelled the same way, so reach and frequency data adds `reach`,
`media_frequency`, and `rf_spend`, and `model_inputs` lists whatever your own
data selects. Your own names can't reuse any of them.

A block receives each array after scaling, so `media` and the rest arrive on
the scales [Scaling](#scaling) describes, while `spend` stays in dollars.
`generated_quantities` also gets a random key as its first argument. The key
arrives by position, so its name is yours, and the guide calls it `key`.

The data's axis names are supplied names too. A parameter declared with
`dims="channel"` gets one value per paid channel, labeled with the channel
names, and `organic_channel`, `control`, and `treatment` work the same way.

:::{admonition} Data axes and your own axes
:class: important

The axes above keep mmmJAX's spelling, as do `rf_channel` and
`organic_rf_channel` with reach and frequency data and `group` for grouped
data, as [Groups](#groups) shows. Their lengths and labels come from the data.
An axis of your own can have any name, but you give its length, as in
`mj.Real(4, dims="harmonic")`, or its labels through `coords` on
{class}`~mmmjax.Model`. With neither, the model fails when it's built. A
parameter can also leave out `dims`, as the `mj.Real(4)` that holds four
seasonal coefficients in [A first model](first_model) does.
:::

## Controls, treatments, and organic media

Beyond the paid channels, the data holds three kinds of input. The role you
give each one decides how mmmJAX scales it and what the analyses do with it,
while your blocks decide how the model uses it.

Demand and the holidays are controls, and here both are confounders, things
that move revenue and also move how much the brand advertises. The brand
buys more media when demand is high and in the holiday weeks, as
[The example data](example_data) shows, so a model without them would credit
the media with lifts that demand and the holidays produced. The model adjusts
for each control, but a control's coefficient has no causal reading. No
analysis reports a contribution for it, and its effect stays in the baseline.

:::{admonition} Don't control for what the ads move
:class: warning

Leave out anything the ads themselves move, such as site visits. Adjusting for
it hides part of the effect you want to measure.
:::

Price and promotion are non-media treatments, inputs the brand sets for itself
the way it sets a budget. The model adjusts for a treatment just as it does
for a control. That matters here because the brand promotes during its
flights. The difference is in what you get back.
{func}`~mmmjax.contributions` reports a contribution for each treatment, the
revenue that would be lost if the treatment sat at a baseline level instead
of its observed values. A price of zero means nothing, so the baseline is the
lowest level in the training data unless `treatment_baselines` sets another.

```{code-cell} ipython3
lowest = data.arrays["treatments"].min(axis=0)
pd.Series(lowest, index=data.columns["treatments"], name="lowest").round(2)
```

The baseline for price is \$17.13, the lowest price the brand charged, and
for promotion it is zero, a week without one. Price's contribution is then
what the brand's prices above \$17.13 earned or cost it, and it comes out
negative when higher prices lose revenue. Choose the treatment role for an
input you could change and want an answer about, and the control role for one
you only need to hold fixed.

Email is organic media. Its sends carry over and saturate like a paid
channel's impressions, so the model gives them the same carryover and Hill
curve. The newsletter costs nothing, so there is no spend to put a return on,
and email gets a contribution but no ROI.
[A first model](first_model.md#emails-share-of-revenue) puts its prior on
email's share of revenue instead.

## Arrays

Prepared arrays put time first, then the group when the data has one, then
the channel or column.

```{code-cell} ipython3
{name: array.shape for name, array in data.arrays.items()}
```

Each role keeps its own array. Paid media and spend have one column per paid
channel, organic media one for email, and the controls and treatments two
each. The outcome has one value per week, so a block can combine it with any
of them directly.

## Any data frame

{func}`~mmmjax.prepare_data` reads your data through
[narwhals](https://narwhals-dev.github.io/narwhals/), so it can stay in the
library it already lives in. pandas, polars, and PyArrow all work
directly, as does any other eager data frame narwhals supports, and nothing is
converted to pandas along the way.

```{code-cell} ipython3
import polars as pl
import pyarrow as pa

frames = {
    "polars": pl.from_pandas(brand.frame),
    "pyarrow": pa.Table.from_pandas(brand.frame),
}
for library, frame in frames.items():
    prepared = mj.prepare_data(
        frame,
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
    matches = (prepared.arrays["media"] == data.arrays["media"]).all()
    print(library, bool(matches))
```

Both give the same media array as the pandas version above. The same goes for
every place mmmJAX takes new data, such as the scenarios in
[Scenarios](scenarios).

## Checking the data

{func}`~mmmjax.check_data` looks for problems that no model can fix, so read
its report before you write any blocks. Its `pairs` group correlates every
pair of inputs and flags a pair as highly correlated when the correlation
passes 0.9 in either direction.

```{code-cell} ipython3
checks = mj.check_data(data)
list(checks.children)
```

The heatmap below draws each pair once from that group, with each input
labeled by its role unless it is a paid channel. Clustering orders the inputs
so the ones that move together sit side by side. A flagged pair carries its
correlation as a label, and an outline marks a pair whose `matching_activity`
says the two are active in exactly the same weeks.

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
or 0.99 as well. As organic media it gets the same activity check as a paid
channel, and its outlines say the newsletter goes out in exactly the flight
weeks.

Meta, Generic search, and Display run all year and rise in those weeks. Their
labeled pairs run from 0.91 to 0.97, and Generic search falls just short of
the threshold with Linear TV, Influencer, and Email. YouTube and Branded
search, the two channels on calendars of their own, stay pale against the
flight channels. Snapchat shares the calendar too, but the half year before
its launch keeps it below the threshold.

Price and promotion correlate at -0.92, because the brand cuts its price in
promotion weeks. The brand also runs a promotion in every campaign week, so
beside the flight channels and Email the promotion column turns red and the
price column blue, though neither reaches the threshold. Because of that overlap, the
model has to adjust for the treatments. Without them it would hand the
promotions' lift to the flight channels.

The `predictors` group gives each input's variance inflation factor, a number
that grows as the other inputs predict it better. The `series` group gives the
share of weeks each input is zero.

```{code-cell} ipython3
predictors = checks["predictors"].to_dataset().to_dataframe()
series = checks["series"].to_dataset().to_dataframe()
predictors[["vif"]].join(series["zero_fraction"]).round({"vif": 1, "zero_fraction": 2})
```

The factor reaches 120.3 for Email, 105.9 for TikTok, and 87.5 for Linear TV,
and the matched channels are all off in 0.68 of the weeks.

:::{admonition} Priors decide the split
:class: important

The matched channels are never active apart, so the data sees the combined
lift of a flight and can't say how to split it among them. A model fit to this
data leaves that split to its priors.
:::

That's one reason the model in [A first model](first_model) states its media
priors as returns on spend and email's as a share of revenue, quantities you
can hold beliefs about. [Recovering the truth](recovery) shows what the split
costs.

YouTube runs on a calendar of its own, and its factor of 1.4 says the data can
tell its effect apart from the rest. It is off air in 0.62 of the weeks, and
Snapchat in 0.74, since it launches later than the others. The factors for
price and promotion are 11.5 and 19.8, because the price falls in promotion
weeks and the promotions follow the flights.

## Scaling

{func}`~mmmjax.fit_data_scaling` fits a transform for each role it scales, and
{class}`~mmmjax.Data` applies them before any block runs.

```{code-cell} ipython3
import numpy as np

scaling = mj.fit_data_scaling(data, scale_outcome=True)
media_transform = scaling.transformations["media"]
organic_transform = scaling.transformations["organic_media"]
medians = np.concatenate([media_transform.scale[0], organic_transform.scale[0]])
pd.Series(medians, index=[*data.channels, *data.organic_channels], name="median").round().astype(int)
```

Each channel's exposure is divided by its median $m_c$ over the weeks with any
exposure, so the model sees $x_{tc} = z_{tc} / m_c$ for raw impressions
$z_{tc}$. A typical week of Linear TV becomes one, and a week without Linear
TV stays at zero. The medians run from 51,098 impressions for Branded search
to 454,068 for TikTok, and dividing by them puts every channel on a scale of
about one.

Email's sends $n_{to}$ are divided by their own median of 46,854 in the same
way, so $x_{to} = n_{to} / m_o$. Organic media carries over and saturates
on the same terms as paid media, so it gets the same scale. That's what lets
email share the paid channels' carryover and saturation priors in
[A first model](first_model).

```{code-cell} ipython3
transforms = [scaling.transformations[role] for role in ("outcome", "controls", "treatments")]
offsets = np.concatenate([np.ravel(transform.offset) for transform in transforms], dtype=float)
scales = np.concatenate([np.ravel(transform.scale) for transform in transforms], dtype=float)
names = ["revenue", *data.columns["controls"], *data.columns["treatments"]]
standardized = pd.DataFrame({"offset": offsets, "scale": scales}, index=names)
standardized.round(2)
```

Revenue becomes $y_t = (R_t - \bar{R}) / s_R$ with its mean $\bar{R}$ and
standard deviation $s_R$, the offset and scale above. The controls and
treatments are standardized the same way, as
$p_{tj} = (q_{tj} - \bar{q}_j) / s_j$ and
$g_{ti} = (w_{ti} - \bar{w}_i) / s_i$, the promotion flag included.

A treatment's coefficient then measures the change in standardized revenue per
standard deviation of the treatment, \$1.67 of price or 0.49 on the promotion
flag. The analyses don't report on that scale.
{func}`~mmmjax.contributions` sets each treatment to its baseline in the
data's own units, \$17.13 for price, and returns revenue in dollars.

:::{admonition} The outcome keeps its units by default
:class: tip

Revenue is standardized here only because the call asks for
`scale_outcome=True`. By default the outcome keeps its own units, which is
what a count likelihood needs.
:::

Spend is never scaled, because the model's ROI prior is revenue per dollar.
[A first model](first_model.md#return-on-investment) shows how a return
becomes a coefficient.

## Groups

Passing `groups` to {func}`~mmmjax.prepare_data` adds a group axis after time
to every observation array. {func}`~mmmjax.simulate_data` makes three regions
unless told otherwise.

```{code-cell} ipython3
regional = mj.simulate_data(seed=7)
regional_data = mj.prepare_data(
    regional.frame,
    time="week",
    groups=["region"],
    outcome="revenue",
    media=[f"{name}_impressions" for name in channels],
    spend=[f"{name}_spend" for name in channels],
    channels=list(channels.values()),
    organic_media=["email_sends"],
    organic_channels=["Email"],
    controls=["demand", "holiday"],
    treatments=["price", "promotion"],
)
regional_data.arrays["media"].shape, regional_data.arrays["outcome"].shape
```

Every region has to cover the same weeks. A model can then give each region
its own values by declaring a parameter with `dims=("group", "channel")`.
[A first model](first_model) declares `roi`, `retention`, and
`half_saturation` with `dims="channel"` for one value per channel, and a
regional model adds the group axis the same way.
