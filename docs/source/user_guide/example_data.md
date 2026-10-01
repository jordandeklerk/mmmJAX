---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# The example data

Every model in this guide fits the same simulated data, so you know the true
effects and can check each model against them.
{func}`~mmmjax.simulate_data` generates weekly marketing data for a fictional
consumer brand and keeps a record of how it was made.

The brand advertises on ten paid channels and sends an email newsletter. Its
revenue also moves with demand, the seasons, holidays, its own prices and
promotions, and a baseline that drifts over time. With that much going on, the
data looks a lot like a real brand's.

```{code-cell} ipython3
:tags: [remove-cell]

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

## The simulated data

```{code-cell} ipython3
:tags: [wide-table]

import mmmjax as mj

brand = mj.simulate_data(seed=7, groups=None)
brand.frame.head().round(2)
```

:::{admonition} One national series
:class: note

{func}`~mmmjax.simulate_data` makes three regions unless told otherwise, so
the call above passes `groups=None` to get one national series instead. All
three regions come back later in the guide, when [Geo-level models](geo) fits
them together.
:::

The data runs weekly for three years, from January 2022 to December 2024.
Each paid channel has an impressions column and a spend column, and email has
only its sends, since the newsletter costs nothing to run. The record of how the
data was made sits beside it in `brand.channels` and `brand.truth`.

:::{admonition} Blocks ask for roles, not columns
:class: important

When you write a model's blocks, you never ask for a column such as `price` or
`linear_tv_impressions`. Instead, [Data and scaling](data.md#supplied-names)
gives each column a role, and a block asks for the role by its supplied name,
such as `treatments` or `media`.
:::

## How the data is made

Revenue in week $t$ is

$$
\begin{aligned}
r_t &= \Big( b_t + w_t + P\, \big[0.25\, (D_t - 1) - 0.015\, (q_t - 20) + 0.06\, m_t + 0.12\, H_t\big] \\
&\qquad + \sum_{c} \beta_c \operatorname{Hill}\big(x_{tc};\, \kappa_c, s_c\big) \Big)\, \eta_t, \\[6pt]
x_{tc} &= \frac{\sum_{\ell=0}^{8} \rho_c^{\ell}\, z_{t-\ell,c}}{P \sum_{\ell=0}^{8} \rho_c^{\ell}},
\qquad
\operatorname{Hill}(x;\, \kappa, s) = \frac{x^{s}}{x^{s} + \kappa^{s}}.
\end{aligned}
$$

The first line holds the baseline $b_t$, the season $w_t$, demand $D_t$, the
price $q_t$, promotions $m_t$, and holidays $H_t$, each scaled by the
population $P$. Here $t$ counts weeks from zero at the start of the data. The
noise $\eta_t$ multiplies the whole sum, and it's lognormal with mean one and a
standard deviation of 5 percent.

The media sum runs over the ten paid channels and email. Each channel's
exposures per person carry over eight weeks at retention $\rho_c$ and then pass
through a Hill curve with half saturation $\kappa_c$ and slope $s_c$. Since that
curve never passes one, $\beta_c$ is the most the channel can add in a week.

### The baseline and other drivers

The terms outside the media follow the calendar and slow random drift.

$$
\begin{aligned}
b_t &= P\, u \exp\big(0.025\, t / 52 + 0.08\, g_t + 0.04\, g'_t\big), \\
w_t &= P\, \big[\big(0.08 + 0.02 \sin(2\pi t / 156)\big) \sin \omega_t + 0.04 \cos 2\omega_t\big], \\
D_t &= \exp\big(0.2\, \xi_t + 0.12 \sin \omega_t\big),
\qquad
\xi_t = 0.8\, \xi_{t-1} + 0.6\, \varepsilon_t, \\
H_t &= \exp\big(-\tfrac{1}{2} \big((d_t - 330) / 10\big)^2\big) + \exp\big(-\tfrac{1}{2} \big((d_t - 355) / 6\big)^2\big), \\
m_t &= \mathbf{1}\big[f_t > 0 \text{ or } H_t > 0.5\big],
\qquad
q_t = 20\, \big(1 + 0.04\, (t + 8) / 52\big) (1 - 0.15\, m_t).
\end{aligned}
$$

Here $d_t$ is the day of the year and $\omega_t = 2\pi d_t / 365.25$. The
baseline grows 2.5 percent a year from a level
$u \sim \operatorname{Uniform}(0.8, 1.2)$ and drifts with $g_t$ and $g'_t$,
two noise series smoothed over about a quarter. The season's amplitude rises
and falls once over the three years.

Demand adds a yearly swing to $\xi_t$, a persistent AR(1) series driven by
standard normal draws $\varepsilon_t$. The holidays peak in late November and
again shortly before Christmas. The brand runs promotions during those peaks
and during its flights $f_t$, and each one cuts the list price by 15 percent.

```{code-cell} ipython3
:tags: [hide-input]

truth = brand.truth
drivers = {
    "seasonality": "Seasons",
    "demand_effect": "Demand",
    "price_effect": "Price",
    "promotion_effect": "Promotions",
    "holiday_effect": "Holidays",
}

fig, (baseline, effects) = plt.subplots(2, 1, sharex=True, figsize=(12, 7), layout="constrained")
baseline.plot(truth["time"], truth["baseline"], color="black", linewidth=1)
for name, label in drivers.items():
    effects.plot(truth["time"], truth[name], linewidth=1, label=label)
baseline.set_title("Baseline revenue")
effects.set_title("What the other drivers add to weekly revenue")
effects.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

terms = truth[["baseline", *drivers]].to_dataframe()
terms.agg(["min", "max"]).T.round(-2)
```

In the top panel, the baseline's drift outweighs its growth, so it climbs
through 2022 and slides over 2023 and 2024. Across those three years, the
table puts its weekly level between \$218,900 and \$302,600.

Of the other drivers in the bottom panel, demand moves revenue the most, from
\$23,700 below normal to \$37,600 above. The holiday spikes at the end of each
year add as much as \$27,100. A promotion adds a flat \$13,300 on top of what
its lower price adds.

### The media

Each channel's activity $a_{tc}$ becomes execution $e_{tc}$, execution becomes
spend $S_{tc}$, and spend buys impressions $z_{tc}$.

$$
\begin{aligned}
a_{tc} &= \alpha_c + f^{c}_t, \\
e_{tc} &= a_{tc}\, \lambda_c\, D_t^{0.6}\, (1 + 0.4\, H_t) \exp\big(0.16\, \xi^{\text{budget}}_t + 0.08\, \xi^{\text{exec}}_{tc}\big), \\
S_{tc} &= \phi_c\, P\, e_{tc},
\qquad
\beta_c = \theta_c\, P\, \epsilon_c, \\
z_{tc} &= \frac{1000\, S_{tc}}{k_c \exp\big(0.08\, \xi^{\text{cpm}}_{tc} + 0.25\, H_t + 0.03\, (t + 8) / 52\big)}.
\end{aligned}
$$

The brand's calendar $f_t$ runs a flight of three to six weeks about every 13
weeks, at a strength between 0.8 and 1.4, and is zero between them. Each
channel's calendar $f^c_t$ copies it with probability 0.7 and is otherwise
drawn the same way on its own. Meta, the two search channels, and Display have
$\alpha_c = 0.5$, so their activity never drops below that level. Every other
channel has $\alpha_c = 0$ and is off between flights.

Execution follows demand and holidays, so the brand spends more when revenue
would be up anyway. Search execution also scales with $D_t^{0.7}$, since people
search more when demand is high.

Execution drifts too, with a brand-wide budget and weekly noise, and each $\xi$
is its own AR(1) series. The draws $\lambda_c$ and $\epsilon_c$ jitter each
channel's budget and effect by about 15 percent. Email has no spend, so it goes
straight from execution to $0.15\, P\, e_{tc}$ sends a week.

The simulation starts eight weeks before the data, so carryover is already
under way in January 2022, and `brand.media_history` holds those eight weeks.

The last six columns of the channel table below hold each channel's spend per
person $\phi_c$, cost per thousand impressions $k_c$, retention $\rho_c$, half
saturation $\kappa_c$, slope $s_c$, and coefficient per person $\theta_c$.

```{code-cell} ipython3
:tags: [hide-input]

columns = [
    "activity",
    "spend_per_person",
    "cpm",
    "retention",
    "half_saturation",
    "slope",
    "coefficient_per_person",
]
brand.channels.set_index("channel")[columns]
```

Linear TV airs in flights on the brand's calendar and goes dark in between, and
the other channels fall into four groups.

- TikTok, Streaming, Influencer, and email share Linear TV's calendar, and
  Snapchat joins them after its launch in July 2022.
- Meta, Generic search, and Display stay on every week and rise during the
  flights.
- Branded search also stays on every week but rises on a calendar of its own.
- YouTube runs flights of its own.

Channels that move together are hard to tell apart, and
[Checking the data](checking_data) measures how closely the shared calendar
ties them.

### What each channel adds

A channel's contribution follows its exposure with a lag, since carryover keeps
each week's impressions working for several weeks.

```{code-cell} ipython3
:tags: [hide-input]

names = {
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
    "email": "Email",
}
shown = ["linear_tv", "youtube", "generic_search"]

fig, axis = plt.subplots(layout="constrained")
for channel in shown:
    contribution = truth["contribution"].sel(channel=channel)
    axis.plot(truth["time"], contribution, linewidth=1, label=names[channel])
axis.set_title("What each channel adds to weekly revenue")
axis.legend(frameon=False)
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

truth["contribution"].sel(channel=shown).to_pandas().agg(["min", "mean", "max"]).round(-2)
```

Linear TV, in blue, adds up to \$27,400 in a flight week and \$11,100 in an
average week, and after each flight it tails off over several weeks. YouTube,
in orange, reaches \$20,100 in flights that run on a calendar of its own.
Generic search never goes dark, so it never adds less than \$3,700, and it
peaks at \$16,500 in the brand's flights.

Before a channel adds anything, its carried exposure per person passes through
its Hill curve.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

carried = np.linspace(0.0, 3.0, 200)

fig, axis = plt.subplots(layout="constrained")
for channel in truth["paid_channel"].values:
    settings = truth.sel(channel=channel)
    half_saturation, slope = settings["half_saturation"].item(), settings["slope"].item()
    curve = mj.hill_saturation(carried, half_saturation, slope)
    axis.plot(carried, curve, label=names[channel])
axis.set_title("Saturation curves")
axis.set_xlabel("Carried impressions per person")
axis.set_ylabel("Share of the maximum effect")
axis.legend(frameon=False, ncols=2)
plt.show()
```

Branded search, in yellow, saturates first and reaches half its maximum by 0.3
impressions per person, while Meta, in blue, is the slowest and needs 1.2.

Slopes above one, such as Linear TV's and Snapchat's 1.3, give a slightly
S-shaped start. Branded search's 0.9 makes its curve concave from the first
impression, so each impression adds less than the one before.

TikTok's curve hides under Streaming's, since both have a half saturation of 0.8
and a slope of 1.1. Email's curve, left out of the figure, reaches half its
maximum at 0.08 sends per person, well below a newsletter week's sends.

### Revenue

In the plot below, the noise scatters observed revenue around its expected
value.

```{code-cell} ipython3
:tags: [hide-input]

fig, axis = plt.subplots(layout="constrained")
axis.plot(truth["time"], truth["revenue"], color="black", linewidth=1, label="Observed revenue")
expected = truth["expected_revenue"]
axis.plot(truth["time"], expected, color="tab:orange", linewidth=1.5, label="Expected revenue")
axis.set_title("Weekly revenue")
axis.legend(frameon=False)
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

relative_noise = truth["noise"] / truth["expected_revenue"]
round(float(truth["noise"].std()), -2), round(float(relative_noise.std()), 3)
```

The noise came out at 4.8 percent of expected revenue, about \$17,100 in a
typical week. That's more than Linear TV, YouTube, or Generic search adds in an
average week, so a model has to find their effects under it.

## The truth to recover

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

shares = truth["contribution"].sum("time") / truth["expected_revenue"].sum()
roi = truth["roi"].rename(paid_channel="channel").reindex(channel=truth["channel"].values)
truths = pd.DataFrame({"contribution_share": shares.to_series(), "roi": roi.to_series()})
truths.round({"contribution_share": 3, "roi": 2})
```

Meta produced the most revenue over the three years, 4.3 percent of it, while
Snapchat produced the least at 0.6 percent. Returns run from \$6.59 per dollar
on YouTube, the best of the ten, down to \$2.99 on Generic search. Email's ROI
shows up as NaN because the newsletter has no spend.

[A first model](first_model.md#returns-against-the-truth) sets the fitted
returns beside these values, and [Recovering the truth](recovery) checks the
model's other estimates against the rest of this record.

## Where the first model differs

The model you'll fit in [A first model](first_model) matches parts of this
process exactly. Demand and holidays move expected revenue in straight lines,
as the model's controls do, and so do price and promotions, the model's
non-media treatments. Its media curve uses the same carryover over eight lagged
weeks and the same Hill form. Email gets that curve too, as organic media, and
its true slope of one is the slope the model fixes.

Still, the model departs from the simulation in four ways that are worth
knowing before you compare its estimates with the truth.

- It fixes every Hill slope at one, where the paid channels' true slopes run
  from 0.9 to 1.3, and [Recovering the truth](recovery.md#linear-tvs-curve)
  shows what that costs Linear TV.
- It treats the weeks before January 2022 as having no exposure, so it misses
  the carryover from the always-on channels and from a YouTube flight already
  under way when the data begins.
  [Recovering the truth](recovery.md#missing-media-history) finds that this
  explains little of where the returns fall short.
- Its quadratic trend and two yearly Fourier pairs can only approximate a
  baseline that drifts without a fixed shape and a seasonal wave whose strength
  changes from year to year, and [Scenarios](scenarios.md#a-forecast) shows the
  quadratic's bend carrying on past the data.
- Its noise adds to revenue, where the simulation's multiplies it.

When [Changing the model](changing) adds a media effect that varies over time,
it warns that gaps like these can leak into that effect.
