---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data and scaling

Every model starts from prepared, scaled data, and the choices you make there
decide what your blocks receive. Here you'll prepare and scale the simulated
data from [The example data](example_data) with
{func}`~mmmjax.prepare_data` and {func}`~mmmjax.fit_data_scaling`, the same two
calls [A first model](first_model) makes before it writes its blocks.

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

{func}`~mmmjax.prepare_data` selects the columns a model uses, gives each one a
role, and labels the paid and organic media with the names in `channels` and
`organic_channels`. Each role becomes a supplied name that a block asks for by
its exact spelling, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
explains. {attr}`~mmmjax.PreparedData.model_inputs` lists them together with
the inputs mmmJAX adds, so it's the place to check a name's spelling and axes.

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
comes from, so the entry above tells you `media` is an array over `media_time`
and `channel` that comes from the data. The tabs below group every name this
data offers by its source and say what each one holds.

::::{tab-set}

:::{tab-item} Data roles

The columns `prepare_data` selects reach your blocks as one array for each
role, under the supplied names below.

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

The calendar names are positions in time that mmmJAX works out from the dates.
The media's weeks match the modeled weeks unless `media_history` adds earlier
ones, as a note in [A first model](first_model) explains.

| Supplied name | What it holds | Axes |
| --- | --- | --- |
| `time` | Days since the first training week, which new data counts from too | `time` |
| `media_time` | The same count for the media's weeks | `media_time` |
| `day_of_year` | Each week's day in the calendar year | `time` |
| `media_day_of_year` | The same for the media's weeks | `media_time` |

:::

:::{tab-item} Model inputs

mmmJAX adds these names from the prepared data and its scaling. `reference`
holds arrays such as `reference.media` and `reference.spend`, along with
`reference.n_periods`, and keeps them in every scenario, as
[Scenarios](scenarios.md#predictions-and-definitions) explains.

| Supplied name | What it holds | Axes |
| --- | --- | --- |
| `n_periods` | The number of modeled weeks, as a Python integer | none |
| `outcome_scaling` | The outcome's transform, with `scale` and `inverse_transform` | none |
| `reference` | The training arrays under these same names | none |

:::

::::

Every role keyword of {func}`~mmmjax.prepare_data` becomes a supplied name
spelled the same way, so reach and frequency data adds `reach`,
`media_frequency`, and `rf_spend`. Constants and extra inputs you pass to
{class}`~mmmjax.Data` go under names of your own, as long as none of them
reuses a supplied name.

If you'd rather call a supplied input something else,
{class}`~mmmjax.Data` can rename it. Its `variables` argument maps names you
choose to supplied ones, as in
`mj.Data(data, variables={"impressions": "media", ...})`, and blocks then see
only the names it declares.

A block receives each array after scaling, so `media` and the rest arrive on
the scales [Scaling](#scaling) describes, while `spend` stays in dollars.

`generated_quantities` also gets a random key as its first argument, and
because that key arrives by position, you can name it whatever you like, though
the guide calls it `key`.

### Axes

The data's axis names count as supplied names too, so when you declare a
parameter with `dims="channel"`, it gets one value per paid channel and takes
the channel names as its labels. The axes `organic_channel`, `control`, and
`treatment` work the same way and give a parameter one labeled value per
organic channel, control, or treatment.

:::{admonition} Data axes and your own axes
:class: note

The axes above keep mmmJAX's spelling, and so do `rf_channel` and
`organic_rf_channel` with reach and frequency data and `group` with grouped
data, as [Groups](#groups) shows. An axis of your own can have any name, but
you give its length, as in `mj.Real(4, dims="harmonic")`, or its labels
through `coords` on {class}`~mmmjax.Model`. If you give neither, the model
fails as soon as you build it, before sampling starts. A parameter can also
leave out `dims`, as the `mj.Real(4)` that holds four seasonal coefficients in
[A first model](first_model) does.
:::

## Controls, treatments, and organic media

Beyond the paid channels, the data holds three kinds of input. The role you
give each one decides how mmmJAX scales it and what the analyses do with it,
while your blocks decide how the model uses it.

### Controls

Demand and the holidays are controls, and here both are confounders, things
that move revenue and also move how much the brand advertises. The brand buys
more media when demand is high and in the holiday weeks, as
[The example data](example_data) shows, so a model without them would credit
the media with lifts that demand and the holidays produced.

The model adjusts for each control, but a control's coefficient has no causal
reading. That's why no analysis reports a contribution for a control, and its
effect stays in the baseline.

:::{admonition} Don't control for what the ads move
:class: danger

Leave out anything the ads themselves move, such as site visits, because
adjusting for it hides part of the effect you want to measure.
:::

### Treatments

Price and promotion are non-media treatments, inputs the brand sets for itself
the way it sets a budget. The model adjusts for a treatment the same way it
does for a control.

What you get back is different, though, because {func}`~mmmjax.contributions`
reports a contribution for each treatment, the revenue that would be lost if
the treatment sat at a baseline level instead of its observed values. A price
of zero wouldn't mean anything, so the baseline is the lowest level in the
training data unless you set another with `treatment_baselines`.

```{code-cell} ipython3
lowest = data.arrays["treatments"].min(axis=0)
pd.Series(lowest, index=data.columns["treatments"], name="lowest").round(2)
```

That puts the price baseline at \$17.13, the lowest price the brand charged, and
the promotion baseline at zero, a week without one. Price's contribution is then
whatever charging more than \$17.13 earned or cost the brand, and it comes out
negative when the higher prices lose revenue. The level you pick shapes the
answer, too, and [Treatment baselines](media_effects.md#treatment-baselines) on
Media effects shows the price's median contribution flipping sign when it's
measured from the highest price instead.

:::{admonition} Treatment or control
:class: tip

Choose the treatment role for an input you could change and want an answer
about, and the control role for one you only need to hold fixed.
:::

### Organic media

Email comes in as organic media, and because its sends carry over and saturate
like a paid channel's impressions, the model gives them the same carryover and
Hill curve. The newsletter costs nothing, so there's no spend to put a return
on, and email gets a contribution but no ROI. For the same reason,
[A first model](first_model.md#emails-share-of-revenue) puts its prior on
email's share of revenue instead.

## Arrays

A prepared array puts time first, the group next when the data has one, and
the channel or column last.

```{code-cell} ipython3
{name: array.shape for name, array in data.arrays.items()}
```

## Groups

Passing `groups` to {func}`~mmmjax.prepare_data` adds a group axis after time
to every observation array. The cell below calls {func}`~mmmjax.simulate_data`
without `groups=None`, so it makes its default three regions.

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

In both shapes, the three regions come right after the 156 weeks, and every
region has to cover those same weeks. You can then give each region its own
values by declaring a parameter with `dims=("group", "channel")`.

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

Both lines print `True`, so polars and PyArrow give the same media array as
`data`, the pandas version above. The same goes for every place mmmJAX takes
new data, such as the scenarios in [Scenarios](scenarios).

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
$z_{tc}$. The medians above run from 51,098 impressions for Branded search to
454,068 for TikTok, and dividing by them makes a typical week on air one for
every channel while a week off air stays at zero.

That shared unit is what lets one half-saturation prior describe every channel,
as [Priors](priors.md#carryover-and-saturation) shows, and email's sends,
divided by their own median of 46,854, can share it too.

The median leaves out the weeks off air on purpose, since `media_method="mean"`
would count them and
put Linear TV's typical flight week well above one. A mean would also let the
flight weeks pull up the scale of a channel that never goes dark, so Meta's
typical week would fall below one. If you choose another unit, restate the
half-saturation prior in it.

```{code-cell} ipython3
transforms = [scaling.transformations[role] for role in ("outcome", "controls", "treatments")]
offsets = np.concatenate([np.ravel(transform.offset) for transform in transforms], dtype=float)
scales = np.concatenate([np.ravel(transform.scale) for transform in transforms], dtype=float)
names = ["revenue", *data.columns["controls"], *data.columns["treatments"]]
standardized = pd.DataFrame({"offset": offsets, "scale": scales}, index=names)
standardized.round(2)
```

Revenue becomes $y_t = (R_t - \bar{R}) / s_R$, and the offset and scale in the
table are its mean $\bar{R}$ and standard deviation $s_R$. The controls and
treatments are standardized the same way, and that includes the promotion flag
even though it only takes the values zero and one.

A treatment's coefficient then measures the change in standardized revenue per
standard deviation of the treatment, \$1.67 of price or 0.49 on the promotion
flag. You won't see that scale in the analyses, though, because
{func}`~mmmjax.contributions` sets each treatment to its baseline in the
data's own units, \$17.13 for price, and returns revenue in dollars.

Your priors do see that scale, and because each control and treatment is
measured in its own standard deviations, [A first model](first_model.md#priors)
can give both controls one prior and both treatments another.
[Price and promotion](priors.md#price-and-promotion) on Priors turns the shared
treatment prior back into dollars for each.

:::{admonition} The outcome keeps its units by default
:class: note

Revenue is standardized here because [A first model](first_model.md#priors) asks
for `scale_outcome=True`, so that its baseline and control priors speak in
standard deviations of weekly revenue and would fit a brand of any size. A
normal likelihood accepts the negative values this gives every below-average
week, but a log-normal or count likelihood can't, as
[The edges of the support](distributions.md#the-edges-of-the-support) warns, so
by default the outcome keeps its own units.
:::

Spend is never scaled, whether by `fit_data_scaling` or by
[a scaling of your own](#scaling-of-your-own), because the model's ROI prior
is revenue per dollar, and [A first model](first_model.md#return-on-investment)
shows how that return becomes a coefficient.

## Scaling of your own

When a role needs other statistics than {func}`~mmmjax.fit_data_scaling`
fits, you can set them yourself. Pass {class}`~mmmjax.Data` a mapping from
role names to {class}`~mmmjax.Scaling` objects as `scaling`. It scales each
role you name before any block runs, and any role you leave out keeps its
original units. The mapping can name `outcome`, `media`, `organic_media`,
`reach`, `organic_reach`, `controls`, and `treatments`, as long as the data
selects them.

Like the transforms `fit_data_scaling` fits, a `Scaling` subtracts an offset
and divides by a scale. Each statistic holds one value, one per column, or one
per group and column, but never one per week, because new data covers other
weeks. The exposure roles also need a zero offset, because the analyses remove
a channel by setting its exposure to zero.

The cell below divides each channel's impressions by its busiest training week
and revenue by its training mean. Revenue keeps a zero offset instead of the
mean that `scale_outcome=True` would subtract, so every week stays positive for
a likelihood such as the log-normal. Blocks receive the outcome's scaling as
`outcome_scaling`, so the ROI calibration in
[A first model](first_model.md#return-on-investment) works unchanged.

```{code-cell} ipython3
peaks = data.arrays["media"].max(axis=0)
custom = mj.Data(
    data,
    scaling={
        "media": mj.Scaling(offset=0.0, scale=peaks),
        "outcome": mj.Scaling(offset=0.0, scale=data.arrays["outcome"].mean()),
    },
)
scaled = custom.scaling.transform(data)

tiktok = data.channels.index("TikTok")
weeks = pd.DataFrame(
    {
        "impressions": data.arrays["media"][:, tiktok].round().astype(int),
        "model units": scaled.arrays["media"][:, tiktok],
    },
    index=pd.DatetimeIndex(data.time_values, name="week"),
)
weeks.loc["2022-10-10":"2022-11-21"].round(2)
```

`custom.scaling.transform` gives the arrays that a model built on `custom`
passes to its blocks. You can see in TikTok's autumn flight that the weeks off
air stay at zero and the busiest training week becomes one.

:::{admonition} Blocks see scaled values
:class: warning

`reference.media` holds the training weeks after this scaling too, so a
statistic a block computes from `reference` is in scaled units. Priors on
anything that reads the media, such as a half-saturation point, describe the
scaled values as well, here in units of each channel's busiest week.
:::

### New data and scenarios

New data and scenarios are scaled with these same statistics, so you can call
`custom.scaling.transform` yourself to see what a model would receive for a
plan that doubles every channel's impressions.

```{code-cell} ipython3
impressions = [f"{name}_impressions" for name in channels]
doubled = brand.frame.assign(**{column: 2 * brand.frame[column] for column in impressions})
plan = mj.prepare_data(doubled, time="week", media=impressions, channels=list(channels.values()))

weeks["doubled plan"] = custom.scaling.transform(plan).arrays["media"][:, tiktok]
weeks.loc["2022-10-10":"2022-11-21"].round(2)
```

In the doubled plan, the busiest week reaches two, because its impressions are
still divided by the training peak. If you took the statistics from the plan
itself, it would come out with the training values again, and the model would
never see the extra impressions.

### Nonlinear transforms

Since a `Scaling` only subtracts and divides, a nonlinear transform such as a
log of the impressions belongs in the model's `transformed_data` block instead.
If you read its statistics from `reference`, they stay at their training values
in every scenario, as they would in a declared scaling.

```{code-cell} ipython3
import jax.numpy as jnp


def transformed_data(media, reference):
    # The training peak scales every scenario, so a plan with more impressions passes one
    log_media = jnp.log1p(media) / jnp.log1p(reference.media.max(axis=0))
    return {"log_media": log_media}
```

### Mixing with fitted scaling

You can keep the fitted statistics for the roles you leave alone. If you spread
`scaling.transformations` into the mapping and name `media` after it, only the
paid media switch to the peak scaling.

```{code-cell} ipython3
mixed = mj.Data(data, scaling={**scaling.transformations, "media": mj.Scaling(offset=0.0, scale=peaks)})
list(mixed.scaling.transformations)
```

All five roles are still there, and email, revenue, the controls, and the
treatments keep what `fit_data_scaling` fitted.

A mapping keeps no record of the training population, though, so nothing checks
new data against it. When the fitted statistics adjust for population, as they
do for the regions in [Geo-level models](geo), it's up to you to keep the
training population in every scenario.
