---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Recovering the truth

The checks on [Sampling and diagnostics](sampling) compare the model with the
data, and a model can pass them while crediting revenue to the wrong cause.
Real data never records the right answer, but `example.truth` does, so this
page checks the model from [A first model](first_model) against it. Besides
how noise enters, the model departs from the true process in two ways. It
fixes TV's Hill slope at one where the true slope is 1.3, and it treats the
weeks before the data as having no exposure, although TV aired in two of them
and search in all eight.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["date.converter"] = "concise"
```

## Totals and returns

```{code-cell} ipython3
import pandas as pd

truth = example.truth.assign_coords(channel=["TV", "Search"], paid_channel=["TV", "Search"])
effects = mj.contributions(model, results, quantity="mu")
returns = mj.media_metrics(model, results, quantity="mu")
quantiles = [0.05, 0.5, 0.95]

revenue = effects["incremental_response"].quantile(quantiles, dim=("chain", "draw")).T.to_pandas()
revenue["truth"] = truth["contribution"].sum("time").to_series()
roi = returns["roi"].quantile(quantiles, dim=("chain", "draw")).T.to_pandas()
roi["truth"] = truth["roi"].to_series()
pd.concat({"revenue": revenue.round(-3), "roi": roi.round(2)}, axis=1)
```

`assign_coords` gives the simulation's channels the model's names. Each row
puts the 5th, 50th, and 95th percentiles of the posterior next to the true
value, and every true value lands inside its 90 percent interval. TV's revenue
is pinned down to between \$1.17 million and \$1.46 million around a true
\$1.36 million, while search's interval is about five times as wide. Both
medians fall below the truth, TV's by 3.5 percent and search's by 11 percent.

```{code-cell} ipython3
search_ahead = returns["roi"].sel(channel="Search") > returns["roi"].sel(channel="TV")
round(float(search_ahead.mean()), 2)
```

Search does return more per dollar, \$4.52 against \$3.71, and the model
leans the same way, but it gives that order only a 68 percent chance. Medians
of 4.03 and 3.58 look like a clear ranking, and the draws show that the data
can't settle it. Comparing the channels within each draw keeps both
estimates' uncertainty in the answer, which comparing medians throws away.

## Week by week

Totals can come out right for the wrong reasons, so the next check follows
each channel through the weeks, with the simulation's true contribution as a
dashed line.

```{code-cell} ipython3
weekly = mj.contributions(model, results, quantity="mu", by="time")["incremental_response"]
band = weekly.quantile(quantiles, dim=("chain", "draw"))


def plot_weekly(channel):
    estimate = band.sel(channel=channel)
    fig, axis = plt.subplots(layout="constrained")
    axis.fill_between(
        estimate["time"], estimate.sel(quantile=0.05), estimate.sel(quantile=0.95), alpha=0.3, label="90% interval"
    )
    axis.plot(estimate["time"], estimate.sel(quantile=0.5), linewidth=1, label="Posterior median")
    true = truth["contribution"].sel(channel=channel)
    axis.plot(truth["time"], true, color="black", linestyle="--", linewidth=1, label="Truth")
    axis.set_title(f"What {channel} adds to weekly revenue")
    axis.legend(frameon=False)
    plt.show()


plot_weekly("TV")
```

The model follows every TV flight as it rises and fades, and after the first
eight weeks its median runs about 5 percent below the truth in a typical
flight week. The clear miss is January 2022, where the truth starts at about
\$18,600 and the median at \$11,600. TV aired in the last two weeks of
December 2021, and this model treats those weeks as having no exposure, as
the Media history box on [A first model](first_model) warns. Running the true
process again with those weeks set to zero shows how much that costs.

```{code-cell} ipython3
population = truth["population"].item()


def true_contribution(impressions):
    carried = mj.geometric_adstock(impressions / population, alpha=truth["retention"].values, max_lag=8)
    saturated = mj.hill_saturation(carried, truth["half_saturation"].values, truth["slope"].values)
    return saturated[8:] * truth["coefficient"].values


impressions = truth["exposure"].values
no_history = impressions.copy()
no_history[:8] = 0.0
first_weeks = {
    "truth": true_contribution(impressions)[:8, 0].sum(),
    "truth with no history": true_contribution(no_history)[:8, 0].sum(),
    "model": weekly.sel(channel="TV").isel(time=slice(0, 8)).sum("time").median(),
}
{name: round(float(value)) for name, value in first_weeks.items()}
```

The record starts eight weeks before the data, and `true_contribution` runs
the true adstock and Hill curve over it, matching `truth["contribution"]` to
within a fraction of a cent. Over the first eight weeks TV added \$90,422.
Without the earlier weeks the same process gives \$67,121, close to the
model's \$67,264, so the January miss comes from the missing history and not
from the fit. When you know what aired before your first week,
`media_history` gives the model those weeks.

```{code-cell} ipython3
plot_weekly("Search")
```

Every week of search falls inside its band, but the median sits below the
truth in all 156 weeks, by about 9 percent in a typical week, and the band is
about as wide as the estimate itself. The model has the timing of search's
effect right and is unsure of its size. Search runs every week, so the data
never shows revenue without it.

```{code-cell} ipython3
import xarray as xr

intercept = results["posterior"]["intercept"]
xr.corr(effects["incremental_response"], intercept, dim=("chain", "draw")).to_series().round(2)
```

Across draws, search's three-year revenue moves almost exactly against the
intercept, while TV's barely does. A higher baseline with less search fits the
data about as well as a lower baseline with more, so the data alone can't
choose, and the intercept's prior does much of the choosing. It is centered
where media adds nothing to average revenue, so it leans toward the higher
baseline and less search. TV is off the air in about two thirds of the weeks,
and those weeks show revenue with its effect faded or gone.

## Parameters and curves

Multiplying the price coefficient by the standard deviation of revenue and
dividing by that of price turns it into dollars of weekly revenue per dollar
of price, the units the simulation uses.

```{code-cell} ipython3
revenue_scale = scaling.transformations["outcome"].scale.item()
price_scale = scaling.transformations["controls"].scale.item()
price_effect = results["posterior"]["control_coefficient"].sel(control="price") * revenue_scale / price_scale
price_effect.quantile(quantiles).values.round(), -0.015 * population
```

Each dollar added to the price costs about \$3,378 of weekly revenue in the
middle of the posterior, and the interval holds the true \$3,332, the
simulation's 1.5 cents per person across 222,147 people. TV's retention came
back as 0.69 against a true 0.70 on [Sampling and diagnostics](sampling). TV's
curve parameters need one more step. With $u$ carried impressions per person,
$m$ TV's median nonzero week, and $P$ the population, the model's curve is

$$
s_r \beta\, \frac{u}{u + \kappa m / P},
$$

so $s_r \beta$ is the most TV can add in a week and $\kappa m / P$ is where it
reaches half of that.

```{code-cell} ipython3
tv = results["posterior"].sel(channel="TV")
median_week = scaling.transformations["media"].scale[0, 0].item()
most = tv["coefficient"] * revenue_scale
half = tv["half_saturation"] * median_week / population
print(most.quantile(quantiles).values.round(-2), round(truth["coefficient"].sel(channel="TV").item(), -2))
print(half.quantile(quantiles).values.round(2), truth["half_saturation"].sel(channel="TV").item())
print(round(float(xr.corr(most, half)), 2))
```

The most TV can add comes out at about \$45,000 a week against a true
\$33,500, and its half-saturation point at 1.85 carried impressions per person
against a true 1.0. Both intervals leave out the truth. The two move together
across draws, since a higher ceiling reached more slowly draws nearly the same
curve over the data. With its slope fixed at one, the model can't draw TV's
S-shaped start. The closest curve it can draw has a higher ceiling still,
reached more slowly, and the prior on the coefficient holds these values
between that curve and the truth.

```{code-cell} ipython3
import numpy as np

exposure = xr.DataArray(np.linspace(0.0, 3.0, 301), dims="exposure")
curve = (most * exposure / (exposure + half)).quantile(quantiles, dim=("chain", "draw"))
settings = truth.sel(channel="TV")
true_curve = settings["coefficient"].item() * mj.hill_saturation(
    exposure.values, settings["half_saturation"].item(), settings["slope"].item()
)
carried = mj.geometric_adstock(impressions / population, alpha=truth["retention"].values, max_lag=8)

fig, axis = plt.subplots(layout="constrained")
axis.fill_between(exposure, curve.sel(quantile=0.05), curve.sel(quantile=0.95), alpha=0.3, label="90% interval")
axis.plot(exposure, curve.sel(quantile=0.5), label="Posterior median")
axis.plot(exposure, true_curve, color="black", linestyle="--", label="Truth")
axis.axvline(carried[8:, 0].max(), color="gray", linewidth=1, label="Largest week in the data")
axis.set_title("TV's response curve")
axis.set_xlabel("Carried impressions per person")
axis.set_ylabel("Revenue added per week")
axis.legend(frameon=False)
plt.show()
```

From about 0.2 carried impressions per person up to the largest week, at
1.57, the band holds the true curve, and from 0.5 to that week the median
runs 1 to 6 percent below it. Below 0.2 the whole band sits above the truth,
with the median 44 percent too high at 0.1, because a slope of one rises from
the first impression. Those low exposures are the fading weeks after each
flight, where TV adds little. Contributions, returns, and budgets all run on
the curve, so being right where most revenue is made matters more than
matching parameters.

## Marginal returns

Budgets turn on marginal returns, which follow the slope of the curve rather
than its height. `true_contribution` gives the true ones the way
{func}`~mmmjax.media_metrics` computes them, from 1 percent more impressions
on one channel in every modeled week.

```{code-cell} ipython3
spend = truth["spend"].values[8:].sum(axis=0)
base = true_contribution(impressions).sum(axis=0)
true_marginal = []
for index in range(2):
    more = impressions.copy()
    more[8:, index] *= 1.01
    gain = true_contribution(more).sum(axis=0)[index] - base[index]
    true_marginal.append(float(gain / (0.01 * spend[index])))

marginal = returns["marginal_roi"].quantile(quantiles, dim=("chain", "draw")).T.to_pandas()
marginal["truth"] = true_marginal
marginal.round(2)
```

In the simulation one more dollar brings \$2.71 on TV and \$2.20 on search.
TV's interval holds its truth, but search's stops at \$2.16, so the model
understates what more search would bring. TV still comes out ahead, so moving
money from search to TV at the current split points the right way, though each
dollar moved gains about 51 cents in the truth where the model counts 73. The
plan that [Budget optimization](budgets) makes next still gains revenue in the
truth, but it moves more money to TV than the truth's best split would.

## What recovery shows

Every comparison here sets one fit of one simulated dataset against the
process that made it. A true value inside a 90 percent interval is a single
observation, and showing that the intervals hold the truth nine times in ten
takes many simulated datasets, each fitted and checked.

The simple setting also builds revenue from the same pieces this model
has, with no season, no shifts in demand, and spending that ignores sales.
Revenue in practice has all of these, and a model can match such data closely
while crediting the wrong cause. A simulation shows where a model's answers
rest on the data and where they rest on its assumptions, as search's do here.
On real data, experiments such as lift tests come closest to a recorded truth.
