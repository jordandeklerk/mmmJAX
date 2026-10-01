---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Recovering the truth

The checks on [Sampling and diagnostics](sampling) compare the model with the
data, but a model can pass them all and still credit revenue to the wrong
cause. [A first model](first_model.md#returns-against-the-truth) set the fit's
returns beside the true ones, and every interval held the truth even though
eight of the ten means came in low. This page checks the fit's other answers
against `brand.truth`, from the paid channels' weekly contributions through
Email, the price, and the promotions to the curves a budget plan relies on.

Across these checks, the data pins down what a campaign week adds as a whole
and when each flight lifts revenue, but it can't say how to split that lift
among the channels, Email, and the treatments. The brand runs several
channels, its promotions, its price cuts, and its email sends on one campaign
calendar, so the split falls to the priors and the model's structure.

The model also departs from the simulation in the ways
[The example data](example_data.md#where-the-first-model-differs) lists, and
two of them get checks of their own below.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_curves, first_model_results

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

## Which channel pays best

Each draw holds a return for all ten channels, so the draws let you rank them
and see how sure that ranking is. A budget ranks channels by the next dollar
instead, and [Marginal returns](#marginal-returns) checks that ranking later.

```{code-cell} ipython3
names = list(channels.values())
truth = brand.truth.assign_coords(channel=names + ["Email"], paid_channel=names)
returns = mj.media_metrics(model, results, quantity="mu")
best = returns["roi"].idxmax("channel")
best.to_series().value_counts(normalize=True).round(3)
```

`idxmax` names the channel with the highest return in each draw, so counting
the names gives each channel's chance of being the best. Comparing the means
would throw away every estimate's uncertainty, but comparing the channels
within each draw keeps it in the answer.

YouTube, which pays best in the simulation, wins only 0.118 of the draws.
Streaming and Snapchat win more often, at 0.170 and 0.153, even though
Streaming's true return is the second lowest of the ten. No channel wins even
a fifth of the draws, so the data can't settle which one pays best.

## Week by week

A return can come out right for the wrong reasons, so the next check follows
the paid channels through the weeks. Passing `by="time"` makes
{func}`~mmmjax.contributions` keep every week's contribution for every draw
instead of summing the weeks.

```{code-cell} ipython3
weekly_effects = mj.contributions(model, results, quantity="mu", by="time")
weekly = weekly_effects["incremental_response"]
dict(weekly.sizes)
```

Each figure below shows the model's weekly median and 90 percent interval, and
its dashed line is the simulation's true contribution. On
[Media effects](media_effects.md#all-together) the paid channels' true total
over the three years sat near the top of the model's interval. The first figure
adds up all ten of them so you can follow that total week by week.

```{code-cell} ipython3
:tags: [hide-input]

quantiles = [0.05, 0.5, 0.95]


def plot_weekly(shown, title):
    estimate = weekly.sel(channel=shown).sum("channel").quantile(quantiles, dim=("chain", "draw"))
    fig, axis = plt.subplots(layout="constrained")
    lower, upper = estimate.sel(quantile=0.05), estimate.sel(quantile=0.95)
    axis.fill_between(estimate["time"], lower, upper, alpha=0.3, label="90% interval")
    axis.plot(estimate["time"], estimate.sel(quantile=0.5), linewidth=1, label="Posterior median")
    true = truth["contribution"].sel(channel=shown).sum("channel")
    axis.plot(truth["time"], true, color="black", linestyle="--", linewidth=1, label="Truth")
    axis.set_title(title)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


plot_weekly(names, "What the ten paid channels add to weekly revenue")
```

In the plot, the model follows every flight, but at the peaks the truth rises
to the band's upper edge or past it. To see how often that happens, count the
weeks on each side of the band.

```{code-cell} ipython3
paid_band = weekly.sel(channel=names).sum("channel").quantile([0.05, 0.95], dim=("chain", "draw"))
true_paid = truth["contribution"].sel(channel=names).sum("channel")
above = (true_paid > paid_band.sel(quantile=0.95)).values
below = (true_paid < paid_band.sel(quantile=0.05)).values
tv_flights = truth["exposure"].sel(channel="Linear TV").values[8:] > 0
int(above.sum()), int(below.sum()), int((above & tv_flights).sum()), int(tv_flights.sum())
```

The truth sits above the band in 52 of the 156 weeks and never drops below it.
Of those 52 weeks, 47 fall among Linear TV's 50 flight weeks, when the
channels that share its calendar are on air together. The promotions, with
their price cuts, and the email sends run in those same weeks, and the
sections below show the model handing part of the paid channels' lift to the
promotions and to Email.

```{code-cell} ipython3
:tags: [hide-input]

plot_weekly(["Linear TV"], "What Linear TV adds to weekly revenue")
```

For Linear TV, the model follows each flight as it rises and fades, and the
wide band holds the true contribution, but the median sits below the truth at
every peak. Linear TV shares its campaign calendar with TikTok, Streaming, and
Influencer, as [Checking the data](checking_data) shows, and with all four on
air together the data can't say how much of a shared flight's lift belongs to
Linear TV.

```{code-cell} ipython3
:tags: [hide-input]

plot_weekly(["YouTube"], "What YouTube adds to weekly revenue")
```

YouTube runs on a calendar of its own, and its band rises and falls with each
flight. At the peak of most flights, though, the truth reaches the top of the
band or climbs above it, much as YouTube's true return sits near the top of
its ROI interval. So the data gets the flights' timing right, but their size
leans on the ROI prior.

### Missing media history

The model treats the weeks before January 2022 as having no exposure, as the
Media history box on [A first model](first_model) warns. Carryover reaches
back eight weeks, so the missing history can only affect the first eight weeks
of the data, and it can't cost a channel more than all it added in those
weeks.

```{code-cell} ipython3
true_weekly = truth["contribution"].sel(channel=names)
first_weeks = true_weekly.isel(time=slice(0, 8)).sum("time") / true_weekly.sum("time")
first_weeks.to_series().round(3)
```

Each value is the share of a channel's simulated revenue that came in those
first eight weeks. None passes 10.2 percent, and YouTube's 7.5 percent is far
less than the 38 percent its return falls short by, so the missing history
explains little of the shortfall.

## Email

Email's sends go out in the flight weeks too, so the data sees them alongside
TikTok, Streaming, Linear TV, and Influencer.

```{code-cell} ipython3
:tags: [hide-input]

plot_weekly(["Email"], "What Email adds to weekly revenue")
```

The model picks up email's lift in every flight, but it puts the median well
above the truth. And where the truth drops to zero after each flight, the
model's band fades over several weeks.

Two parameters sit behind that picture, email's share of revenue and its
retention rate. The simulation records both, so the table below sets the
model's quantiles beside them.

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

In the table, the median share of 1.4 percent is about twice the true
0.7 percent, and its interval, from 0.3 to 3.4 percent, spans most of what the
$\operatorname{Beta}(2, 98)$ prior allows. The median retention of 0.524,
against a true 0.1, sits near the middle of its $\operatorname{Beta}(2, 2)$
prior.

Because email never goes out apart from the flights, the data can't tell its
lift from theirs, and both answers mostly repeat their priors.

:::{admonition} Email's share rests on its prior
:class: warning

The share prior is what keeps email small, so give it the care
[Priors](priors) describes.
:::

## Price and promotions

Because the price and the promotions are treatments,
{func}`~mmmjax.contributions` reports what each one earned, and those answers
need the same check as the channels'. Promotions run in the campaign weeks,
cut the price, and come with email sends, so start by looking at how closely
these inputs move together.

```{code-cell} ipython3
drivers = brand.frame[["promotion", "price", "email_sends", "linear_tv_impressions"]]
drivers.corr()["promotion"].round(2)
```

Promotion correlates with price at -0.92 and with email sends and Linear TV's
impressions at 0.79, so as far as the data can tell, a campaign week is one
bundle. The list price also climbs 4 percent a year, as
[The example data](example_data) describes, so the price rises with the trend.

### Effects per unit

The table below puts each treatment coefficient in the simulation's units,
dollars of weekly revenue per dollar of price or per promotion week. Because
the simulation's treatment effects are straight lines, a fitted slope recovers
each true one. The `prior_sd` column puts the prior's standard deviation of
0.25 in the same units.

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

Both intervals hold the truth, but each is too wide to say much about its
effect. For each dollar added to the price, the interval runs from a loss of
\$14,000 of weekly revenue to a gain of \$7,600, and the median, a loss of
\$3,000, lands close to the true loss of \$3,300. A promotion week is worth
\$25,400 in the median against a true \$13,300, and its interval runs from a
loss of \$11,300 to a gain of \$60,300.

Set against prior standard deviations of \$9,200 and \$31,300, each interval
covers much of what the prior allows, so it's the prior that does most of the
work of keeping the two effects in bounds.

Making the price and the promotions treatments doesn't loosen these
intervals. A control and a treatment enter the model as the same linear term,
and what the fit learns about either depends only on the data and the prior.
What the role changes instead is where that uncertainty shows up in your
results. As controls, the price and the promotions would only adjust the other
estimates, and their loose coefficients would stay out of every answer. As
treatments, their effects are answers of their own, so their uncertainty
reaches you through {func}`~mmmjax.contributions` and its plots.

:::{admonition} Treatment priors need more thought
:class: note

A treatment's prior deserves more thought than a control's, because its
effect is an answer of its own. [A first model](first_model) gives the price
and the promotions a tighter prior than demand and the holidays.
:::

### A promotion week's lift

What the data does pin down is the lift of the whole bundle, the extra revenue
of a promotion week over any other week. The table below measures that lift in
four series and sets each one beside its truth.

- `baseline` is the revenue the model expects with every channel removed and
  the treatments at their lowest levels, so its lift comes from the trend, the
  seasons, and the controls.
- `media` sums the ten paid channels and Email.
- `treatments` sums the price and the promotions.
- `total` is the model's expected revenue.

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

The model gives the treatments \$34,700 of a promotion week's lift where the
truth has \$23,800, and the media \$46,000 where the truth has \$59,900, above
the media's whole interval. For the full bundle, though, it finds \$89,100
against a true \$94,000, and that interval, from \$84,100 to \$94,300, is far
narrower than either part's and barely holds the truth.

The draws that give the treatments more give the media less, so the data
measures what a campaign week does as a whole and leaves the split among its
parts to the model's assumptions.

## Linear TV's curve

The model fixes the slope of every Hill curve at one, but the simulation gives
each channel a slope of its own.

```{code-cell} ipython3
truth["slope"].sel(channel=names).to_series()
```

The true slopes run from 0.9 for Branded search to 1.3 for Snapchat and
Linear TV, so Linear TV is one of the two channels furthest from the model's
fixed slope. `brand.contributions(multiplier)` reruns the simulation's own
adstock and Hill curves with every channel's media in the modeled weeks scaled
by `multiplier`, the way {func}`~mmmjax.response_curves` scales spending. Add
up Linear TV's weeks at each multiplier and you trace out its true curve.

```{code-cell} ipython3
:tags: [skip-execution]

curves = mj.response_curves(model, results, quantity="mu")
```

```{code-cell} ipython3
:tags: [remove-cell]

curves = first_model_curves(model, results)
```

```{code-cell} ipython3
import plotnine as pn

tv = curves["spend"].sel(channel="Linear TV").to_dataframe()
tv["truth"] = [
    brand.contributions(multiplier).sel(channel="linear_tv").sum().item() for multiplier in tv.index
]
true_line = pn.geom_line(pn.aes("spend", "truth"), data=tv, linetype="dotted", size=1, inherit_aes=False)
mj.plot_response_curves(curves, channels=["Linear TV"]) + true_line
```

The dotted line is the true curve, the light blue line and band are the
model's mean and 89 percent interval, and the point marks today's spending.
The band holds the true curve at every level of spending, but it's wide, so
the data leaves the curve loose.

Follow the two lines up from zero and you'll find that the model's curve also
has the wrong shape. It starts out rising a little faster than the truth,
drops below it well short of today's spending, and falls further behind past
it.

That difference in shape comes from the slope, because a slope of one rises
fastest at the first impression, while the true slope of 1.3 starts slowly and
then climbs past it. Linear TV runs in flights, high on its curve, and that's
where the model falls short. Because the model's curve bends too early, Linear
TV's next dollar falls further below its truth than its return does, as
[Marginal returns](#marginal-returns) shows. The return itself comes in a
third low for the reasons [Priors](priors.md#why-the-returns-lean-low) gives.

## Marginal returns

Budgets turn on marginal returns, and a marginal return follows the slope of
the curve rather than its height. {func}`~mmmjax.media_metrics` finds each one
from 1 percent more impressions on one channel in every modeled week, and
`incremental_spend` records what that 1 percent costs.

Because the simulation's channels don't affect each other, you can raise them
all at once with `brand.contributions` and still get each channel's true gain.
As on [A first model](first_model.md#returns-against-the-truth), `expand_dims`
gives the true values a single draw so they plot beside the model's.

```{code-cell} ipython3
added = (brand.contributions(1.01) - brand.contributions()).sum("time")
true_gain = added.assign_coords(channel=names + ["Email"]).sel(channel=names)
true_marginal = true_gain / returns["incremental_spend"]
true_margins = true_marginal.expand_dims(chain=[0], draw=[0]).to_dataset(name="marginal_roi")
mj.plot_media_metrics({"Model": returns, "Truth": true_margins}, metric="marginal_roi")
```

The orange bars are the truth, and they put the next dollar's biggest return
on YouTube, \$4.17, and Linear TV, \$2.95, and its smallest on Generic search,
\$1.34. Every channel's truth but YouTube's falls inside its interval, though
only narrowly for Linear TV. The model's highest means belong to Snapchat and
Streaming instead, at \$2.40 and \$2.39, against true values of \$2.68 and
\$1.96.

Both rankings put Generic search last, but they part ways on Streaming. Its
marginal return ranks second among the model's means and seventh of the ten
in the truth. [Budget optimization](budgets) moves money by the model's
marginal returns, so it checks what that misranking costs a plan.

## What recovery shows

Every comparison here sets one fit of one simulated dataset against the
process that made it, so each interval that holds the truth is a single
observation. To show that 90 percent intervals hold it nine times in ten,
you'd need to fit and check many simulated datasets.

With real data you have no truth to check against, but the same problems come
up, and these steps guard against them.

- Run {func}`~mmmjax.check_data` before fitting to find the inputs that move
  together.
- Give careful priors to what the data can't separate, since those priors
  carry the answer.
- Calibrate with experiments such as lift tests, the closest thing real data
  has to a recorded truth.
- Fit regional data where regions run channels off the shared calendar.
- Keep budget plans near the spending the data has seen.

[Changing the model](changing) and [Geo-level models](geo) check their
variants against truths of their own.
