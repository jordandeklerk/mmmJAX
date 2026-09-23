---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# The example data

Every page in this guide fits the same simulated data, and because the data
is simulated, the true effects are known. {func}`~mmmjax.simulate_data`
generates weekly marketing data for a fictional consumer brand and keeps a
record of how it was made. The guide uses its simple setting, where revenue
comes from a constant baseline, a price effect, and two media channels. Those
are exactly the pieces of the model in [A first model](first_model), so every
estimate the guide makes can be checked against the truth.

```{code-cell} ipython3
:tags: [remove-cell]

import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["date.converter"] = "concise"
```

## The simulated frame

```{code-cell} ipython3
import mmmjax as mj

example = mj.simulate_data(seed=7, groups=None, complexity="simple", noise_scale=0.02)
example.frame.head().round(2)
```

The frame holds three years of weeks, from January 2022 to December 2024,
with impressions and spend for linear TV and generic search, the price, and
revenue. `groups=None` asks for one national series instead of several
regions, and `noise_scale=0.02` keeps the week-to-week noise at 2 percent of
expected revenue. The frame looks like data from a real brand, and the
record of how it was made sits beside it in `example.channels` and
`example.truth`.

## How revenue is made

Revenue in week $t$ is a constant baseline $b$ plus the price effect $\pi_t$
and each channel's contribution, scaled by a noise factor $\eta_t$,

$$
r_t = \eta_t \Big( b + \pi_t + \sum_{c} \beta_c \,
\frac{\bar{x}_{tc}^{\,s_c}}{\bar{x}_{tc}^{\,s_c} + \kappa_c^{\,s_c}} \Big),
$$

where $\bar{x}_{tc}$ is the channel's impressions per person after carryover
at retention $\rho_c$, $\kappa_c$ is its half-saturation point, $s_c$ its
slope, $\beta_c$ the most it can add in a week, and $\eta_t$ a mean-one
lognormal factor with a standard deviation of 2 percent. The channel table
holds each channel's settings.

```{code-cell} ipython3
example.channels[["channel", "activity", "retention", "half_saturation", "slope"]]
```

TV runs in flights with long gaps between them, and search stays on every
week. Each week's impressions carry forward at the channel's retention rate,
so a week of TV keeps working for several weeks, while search's effect is
mostly gone the week after. Plotting the impressions above what each driver
adds to revenue shows both.

```{code-cell} ipython3
import pandas as pd

truth = example.truth
weeks = pd.to_datetime(truth["time"].values)
media_weeks = pd.to_datetime(truth["media_time"].values)
labels = {"linear_tv": "TV", "generic_search": "Search"}

fig, (impressions, effects) = plt.subplots(2, 1, sharex=True, layout="constrained")
for channel, label in labels.items():
    impressions.plot(media_weeks, truth["exposure"].sel(channel=channel), linewidth=1, label=label)
    effects.plot(weeks, truth["contribution"].sel(channel=channel), linewidth=1, label=label)
effects.plot(weeks, truth["price_effect"], linewidth=1, label="Price")
impressions.set_title("Weekly impressions")
effects.set_title("What each driver adds to weekly revenue")
impressions.legend(frameon=False)
effects.legend(frameon=False)
plt.show()
```

Each TV flight lifts revenue and then fades over the following weeks, so TV
adds nothing between flights and up to about \$21,500 in its best weeks.
Search adds a steadier \$12,300 a week on average and rises with its own
campaign bursts. Price moves revenue by up to about \$14,000 either way, on
the same scale as the channels, which is why the model needs it as a control.
Adding the three to a constant baseline of about \$181,000 a week gives the
expected revenue.

```{code-cell} ipython3
fig, axis = plt.subplots(layout="constrained")
axis.plot(weeks, truth["revenue"], color="black", linewidth=1, label="Observed revenue")
axis.plot(weeks, truth["expected_revenue"], color="tab:orange", linewidth=1.5, label="Expected revenue")
axis.set_title("Weekly revenue")
axis.legend(frameon=False)
plt.show()
```

The noise is about 2 percent of expected revenue, a few thousand dollars in a
typical week, so most of the movement in revenue comes from price and the two
channels. Before either channel adds anything, its carried impressions,
measured per person, pass through a Hill curve that reaches half its maximum
at `half_saturation`.

```{code-cell} ipython3
import numpy as np

exposure = np.linspace(0.0, 3.0, 200)
fig, axis = plt.subplots(layout="constrained")
for channel, label in labels.items():
    settings = truth.sel(channel=channel)
    curve = mj.hill_saturation(exposure, settings["half_saturation"].item(), settings["slope"].item())
    axis.plot(exposure, curve, label=label)
axis.set_title("Response curves")
axis.set_xlabel("Carried impressions per person")
axis.legend(frameon=False)
plt.show()
```

Search saturates sooner, reaching half its maximum at 0.6 impressions per
person against TV's 1.0. TV's slope of 1.3 gives its curve a slightly
S-shaped start, while search's slope of one makes it concave from the first
impression.

## The truth to recover

```{code-cell} ipython3
shares = truth["contribution"].sum("time") / truth["expected_revenue"].sum()
pd.DataFrame({"contribution_share": shares.to_series(), "roi": truth["roi"].to_series()}).round(3)
```

Over the three years, TV produced about 4.3 percent of revenue and search
about 6.1 percent, and each dollar returned about \$3.70 on TV and \$4.50 on
search. [A first model](first_model) and [Media effects and
budgets](media_effects) compare their estimates with these values. The first
model fixes every Hill slope at one, so TV's slightly S-shaped curve is the
one place its structure differs from the truth, and [Changing the
model](changing) shows what happens when the slope is set free.
