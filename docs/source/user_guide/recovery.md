---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Recovering the truth

The checks on [Sampling and diagnostics](sampling) compare the model with the
data, and a model can pass them while crediting revenue to the wrong cause.
Real data never records the right answer, but `brand.truth` does. This page
checks the ten-channel brand from [A first model](first_model) against it,
from the paid channels to Email, the price, and the promotions. The model
departs from the simulation in the ways [The example data](example_data)
lists, and the brand runs several channels, its promotions, its price cuts,
and its email sends on one campaign calendar. So the page also asks what the
data can separate and what it leaves to the model's assumptions.

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
plt.rcParams["date.converter"] = "concise"
```

## Returns on spend

```{code-cell} ipython3
names = list(channels.values())
truth = brand.truth.assign_coords(channel=names + ["Email"], paid_channel=names)
returns = mj.media_metrics(model, results, quantity="mu")
true_returns = truth[["roi"]].rename(paid_channel="channel").expand_dims(chain=[0], draw=[0])
mj.plot_media_metrics({"Model": returns, "Truth": true_returns})
```

`assign_coords` gives the simulation's channels the model's names. The truth
also records Email, which has no return because it has no spend.

:::{admonition} Plotting a known truth
:class: tip

`expand_dims` gives the true returns the chain and draw axes of a fit.
{func}`~mmmjax.plot_media_metrics` then treats them as a result with a single
draw and plots them next to the model's, the way [Plotting](plotting) compares
labeled results.
:::

Each blue bar is the model's mean return with its 89 percent interval, and the
orange bar beside it is the true return. All ten intervals hold the truth,
though YouTube's true return of \$6.59 sits near the top of its interval.

The means lean low. Eight of the ten sit below the truth, by as much as 39
percent for TikTok and 38 percent for YouTube, while Streaming's \$4.04 and
Generic search's \$3.07 sit above their true \$3.48 and \$2.99. The model sets
each channel's return through its `roi` parameter, and that prior's median
sits below every true return in the plot, as [Priors](priors) shows. Where the
data says little, the estimates settle low.

```{code-cell} ipython3
best = returns["roi"].idxmax("channel")
best.to_series().value_counts(normalize=True).round(3)
```

`idxmax` names the channel with the highest return in each draw, so counting
the names gives each channel's chance of being the best. YouTube, the true
best, wins only 0.118 of the draws. Streaming takes 0.170 and Snapchat 0.153,
though Streaming's true return is the second lowest of the ten. Comparing the
channels within each draw keeps every estimate's uncertainty in the answer,
where comparing the means would throw it away. Here the draws show that the
data can't settle which channel pays best.

## Week by week

A return can come out right for the wrong reasons, so the next check follows
each channel through the weeks. With `by="time"`,
{func}`~mmmjax.contributions` keeps every week's contribution for every draw.

```{code-cell} ipython3
weekly_effects = mj.contributions(model, results, quantity="mu", by="time")
weekly = weekly_effects["incremental_response"]
dict(weekly.sizes)
```

Each figure below sets a channel's weekly median and 90 percent interval
against the simulation's true contribution, drawn as a dashed line.

```{code-cell} ipython3
:tags: [hide-input]

quantiles = [0.05, 0.5, 0.95]
band = weekly.quantile(quantiles, dim=("chain", "draw"))


def plot_weekly(channel):
    estimate = band.sel(channel=channel)
    fig, axis = plt.subplots(layout="constrained")
    lower, upper = estimate.sel(quantile=0.05), estimate.sel(quantile=0.95)
    axis.fill_between(estimate["time"], lower, upper, alpha=0.3, label="90% interval")
    axis.plot(estimate["time"], estimate.sel(quantile=0.5), linewidth=1, label="Posterior median")
    true = truth["contribution"].sel(channel=channel)
    axis.plot(truth["time"], true, color="black", linestyle="--", linewidth=1, label="Truth")
    axis.set_title(f"What {channel} adds to weekly revenue")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


plot_weekly("Linear TV")
```

Linear TV shares its campaign calendar with TikTok, Streaming, and
Influencer, as [Data and scaling](data.md) shows, so its flights air
alongside theirs. The model follows each flight as it rises and fades, and
the wide band holds the true contribution, but the median sits below the
truth at every peak. The data can't say how much of a shared flight's lift
belongs to Linear TV.

```{code-cell} ipython3
:tags: [hide-input]

plot_weekly("YouTube")
```

YouTube runs on a calendar of its own, so the data can time its effect, and
the band rises and falls with each flight. At the peak of most flights the
truth reaches the top of the band or climbs above it, just as YouTube's true
return sits near the top of its interval in the first plot. The data times
the flights well, but their size leans on the ROI prior.

The model also treats the weeks before January 2022 as having no exposure, as
the Media history box on [A first model](first_model) warns. Carryover reaches
back eight weeks, so the missing weeks touch only the first eight weeks of the
data, and they can't cost a channel more than all it added in those weeks.

```{code-cell} ipython3
true_weekly = truth["contribution"].sel(channel=names)
first_weeks = true_weekly.isel(time=slice(0, 8)).sum("time") / true_weekly.sum("time")
first_weeks.to_series().round(3)
```

Each value is the share of a channel's simulated revenue that came in the
first eight weeks, and that share caps what the missing history can cost it.
YouTube's 7.5 percent is far less than the 38 percent its return falls short
by, and no channel's share passes 10.2 percent, so the missing history
explains little of the gap. When you know what aired before your first week,
`media_history` gives the model those weeks.

## Email

Email's sends go out in the flight weeks too, so the data sees them alongside
TikTok, Streaming, Linear TV, and Influencer.

```{code-cell} ipython3
:tags: [hide-input]

plot_weekly("Email")
```

The model sees email's lift in every flight, but it puts the median well above
the truth. After each flight the truth drops to zero, while the model's band
fades over several weeks. Two quantities sit behind that picture, email's
share of revenue and its retention rate. The simulation records both, so the
table below sets the model's quantiles beside them.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

email = results["posterior"].sel(organic_channel="Email")
share = email["organic_share"].quantile(quantiles).values
retention = email["organic_retention"].quantile(quantiles).values
true_share = truth["contribution"].sel(channel="Email").sum() / truth["expected_revenue"].sum()
true_retention = truth["retention"].sel(channel="Email")
table = pd.DataFrame(
    {
        "organic_share": [*share, true_share.item()],
        "organic_retention": [*retention, true_retention.item()],
    },
    index=[*quantiles, "truth"],
)
table.round(3)
```

The median share, 1.4 percent, is about twice the true 0.7 percent, and the
interval from 0.3 to 3.4 percent spans most of what the
$\operatorname{Beta}(2, 98)$ prior allows. The median retention of 0.524,
against a true 0.1, sits near the middle of its $\operatorname{Beta}(2, 2)$
prior. That's why the model's email keeps selling for weeks after each send,
while the true email fades within a week. Email never goes out apart from the
flights, so the data can't tell its lift from theirs, and both answers mostly
repeat their priors.

:::{admonition} Email's share rests on its prior
:class: warning

The share prior is what keeps email small, so give it the care
[Priors](priors) describes.
:::

## Price and promotions

The price and the promotions are treatments, so
{func}`~mmmjax.contributions` reports what each one earned. Those answers need
the same check as the channels'. Promotions run in the campaign weeks, cut the
price, and come with email sends.

```{code-cell} ipython3
drivers = brand.frame[["promotion", "price", "email_sends", "linear_tv_impressions"]]
drivers.corr()["promotion"].round(2)
```

Promotion correlates with price at -0.92 and with email sends and Linear TV's
impressions at 0.79, so the data sees a campaign week as one bundle. The list
price also climbs 4 percent a year, as [The example data](example_data)
describes, so the price rises with the trend.

The table below puts each treatment coefficient in the simulation's units,
dollars of weekly revenue per dollar of price or per promotion week, by
multiplying it by the standard deviation of revenue and dividing by that of
the treatment. The simulation's treatment effects are straight lines,
so a fitted slope recovers each true one. `prior_sd` puts the prior's standard
deviation of 0.25 in the same units.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

revenue_scale = scaling.transformations["outcome"].scale.item()
treatment_scale = scaling.transformations["treatments"].scale[0]
per_unit = results["posterior"]["treatment_coefficient"] * revenue_scale / treatment_scale

treatments = ["price", "promotion"]
table = per_unit.quantile(quantiles, dim=("chain", "draw")).T.to_pandas()
table["prior_sd"] = 0.25 * revenue_scale / treatment_scale
table["truth"] = [np.polyfit(brand.frame[name], truth[f"{name}_effect"], 1)[0] for name in treatments]
table.round(-2)
```

Both intervals hold the truth, but neither says much. The price's interval
runs from a loss of \$14,000 of weekly revenue for each dollar added to the
price to a gain of \$7,600. The true effect is a loss of \$3,300, and the
median, a loss of \$3,000, lands close to it. A promotion week is worth
\$25,400 in the median against a true \$13,300, and its interval runs from a
loss of \$11,300 to a gain of \$60,300. Set against prior standard deviations
of \$9,200 and \$31,300, each interval covers much of what the prior allows.
The prior does most of the work of keeping the two effects in bounds.

The treatment role doesn't cause this. A control and a treatment enter the
model as the same linear term, and what the fit learns about either depends
only on the data and the prior. The role changes what the analyses report. As
controls, the price and the promotions would only adjust the other estimates,
and their loose coefficients would stay out of every answer. As treatments,
their effects are answers of their own, so their uncertainty reaches you
through {func}`~mmmjax.contributions` and its plots.

:::{admonition} Treatment priors need more thought
:class: important

Because a treatment's effect is an answer of its own, its prior deserves more
thought than a control's. [A first model](first_model) gives the price and the
promotions a tighter one than demand and the holidays.
:::

What the data does pin down is the lift of the whole bundle, the extra revenue
of a promotion week over any other week.

```{code-cell} ipython3
:tags: [hide-input]

promotion = brand.frame["promotion"].to_numpy() == 1


def lift(weekly_revenue):
    during = weekly_revenue.isel(time=promotion).mean("time")
    otherwise = weekly_revenue.isel(time=~promotion).mean("time")
    return during - otherwise


kind = weekly_effects["channel_type"]
true_media = truth["contribution"].sum("channel")
true_treatments = truth["price_effect"] + truth["promotion_effect"]
true_baseline = truth["expected_revenue"] - true_media - true_treatments
parts = {
    "baseline": (weekly_effects["baseline_response"], true_baseline),
    "media": (weekly.sel(channel=kind != "treatment").sum("channel"), true_media),
    "treatments": (weekly.sel(channel=kind == "treatment").sum("channel"), true_treatments),
    "total": (weekly_effects["reference_response"], truth["expected_revenue"]),
}
rows = {}
for name, (fitted, actual) in parts.items():
    rows[name] = [*lift(fitted).quantile(quantiles).values, lift(actual).item()]
pd.DataFrame(rows, index=quantiles + ["truth"]).T.round(-2)
```

`lift` subtracts the mean of the other weeks from the mean of the promotion
weeks, and the table applies it to four series.

- `baseline` is the revenue the model expects with every channel removed and
  the treatments at their lowest levels, so its lift comes from the trend, the
  seasons, and the controls.
- `media` sums the ten paid channels and Email.
- `treatments` sums the price and the promotions.
- `total` is the model's expected revenue.

The truth's parts come from `brand.truth` the same way.

The model gives the treatments \$34,700 of a promotion week's lift where the
truth has \$23,800, and the media \$46,000 where the truth has \$59,900, above
its whole interval. Together it finds \$89,100 against a true \$94,000. That
interval, from \$84,100 to \$94,300, is far narrower than either part's and
just holds the truth. The draws that give the treatments more give the media
less, so the data measures what a campaign week does as a whole and leaves the
split among its parts to the model's assumptions.

## Linear TV's curve

The model fixes the slope of every Hill curve at one, while the simulation
gives each channel its own.

```{code-cell} ipython3
truth["slope"].sel(channel=names).to_series()
```

The true slopes run from 0.9 for Branded search to 1.3 for Snapchat and
Linear TV. `true_response` runs the simulation's own adstock and Hill curve
with the true settings. The first eight rows of the exposure hold the weeks
before the data, and the function scales every channel's impressions after
them by `multiplier`, the way {func}`~mmmjax.response_curves` scales spending.
It returns the revenue each channel adds over the modeled weeks, and a dotted
line draws Linear TV's true curve over the model's.

```{code-cell} ipython3
:tags: [hide-input]

import plotnine as pn

population = truth["population"].item()


def true_response(multiplier):
    exposure = truth["exposure"].values.copy()
    exposure[8:] *= multiplier
    carried = mj.geometric_adstock(exposure / population, alpha=truth["retention"].values, max_lag=8)
    saturated = mj.hill_saturation(carried, truth["half_saturation"].values, truth["slope"].values)
    revenue = truth["coefficient"] * saturated[8:].sum(axis=0)
    return revenue


curves = mj.response_curves(model, results, quantity="mu")
tv = curves["spend"].sel(channel="Linear TV").to_dataframe()
tv["truth"] = [true_response(multiplier).sel(channel="Linear TV").item() for multiplier in tv.index]
true_line = pn.geom_line(pn.aes("spend", "truth"), data=tv, linetype="dotted", size=1, inherit_aes=False)
mj.plot_response_curves(curves, channels=["Linear TV"]) + true_line
```

The dotted line is the true curve, and the light blue line and band are the
model's mean and 89 percent interval, with a point at today's spending. The
band holds the true curve at every level of spending, but it's wide, so the
data leaves the curve loose. The model's curve also has the wrong shape. It
rises a little faster than the truth at first, drops below it well short of
today's spending, and falls further behind past it.

A slope of one rises fastest at the first impression, while the true slope of
1.3 starts slowly and then climbs past it. Linear TV runs in flights, high on
its curve. That's where the model falls short, and it's why Linear TV's return
comes in 34 percent low in the first plot.

## Marginal returns

Budgets turn on marginal returns, which follow the slope of the curve rather
than its height. {func}`~mmmjax.media_metrics` finds each one from 1 percent
more impressions on one channel in every modeled week, and `incremental_spend`
records what that 1 percent costs. The simulation's channels don't affect each
other, so raising them all at once in `true_response` gives each channel's
true gain.

```{code-cell} ipython3
:tags: [hide-input]

true_gain = (true_response(1.01) - true_response(1.0)).sel(channel=names)
true_marginal = true_gain / returns["incremental_spend"]
true_margins = true_marginal.expand_dims(chain=[0], draw=[0]).to_dataset(name="marginal_roi")
mj.plot_media_metrics({"Model": returns, "Truth": true_margins}, metric="marginal_roi")
```

In the simulation the next dollar brings the most on YouTube, \$4.17, and on
Linear TV, \$2.95, and the least on Generic search, \$1.34. Every channel's
truth but YouTube's falls inside its interval, Linear TV's only just. The
model's highest means belong to Snapchat and Streaming, at \$2.40 and \$2.39,
where the truth has \$2.68 and \$1.96. So the plan that
[Budget optimization](budgets) makes is right to cut Generic search, which is
the lowest in the truth as well. It's wrong to move the most money into
Streaming, whose true marginal return ranks seventh of the ten.

## What recovery shows

Every comparison here sets one fit of one simulated dataset against the
process that made it. One interval that holds the truth is a single
observation. To show that 90 percent intervals hold it nine times in ten, you'd
need many simulated datasets, each fitted and checked.

The brand simulation already has much of what revenue has in practice, with a
drifting baseline, seasons that change strength, shifts in demand, holidays,
and price cuts and promotions timed with the campaigns. The model absorbs some
of this through its trend, Fourier terms, controls, and treatments. The data
can't separate the channels on the shared calendar, the email sends that go
out with them, or the price cuts and promotions that run alongside. Where it
can't separate two causes, the model's answers rest on the priors and its
structure instead. On real data, experiments such as lift tests come closest
to a recorded truth.
