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
plt.rcParams["axes.grid"] = False
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

## How the data is made

Revenue in week $t$ is a constant baseline, a price effect, and each channel's
contribution, multiplied by noise with mean one,

$$
\begin{gathered}
r_t = \Big( b - 0.015\, P\, (q_t - 20)
+ \sum_{c} \beta_c \operatorname{Hill}\Big(\operatorname{Adstock}\big(\{z_{t-\ell,c} / P\}_{\ell=0}^{8};\, \rho_c\big);\, \kappa_c, s_c\Big) \Big)\, \eta_t, \\[6pt]
\operatorname{Adstock}\big(\{u_{t-\ell}\}_{\ell=0}^{8};\, \rho\big)
= \frac{\sum_{\ell=0}^{8} \rho^{\ell}\, u_{t-\ell}}{\sum_{\ell=0}^{8} \rho^{\ell}},
\qquad
\operatorname{Hill}(u;\, \kappa, s) = \frac{u^{s}}{u^{s} + \kappa^{s}}.
\end{gathered}
$$

Adstock carries each week's impressions per person forward at the channel's
retention $\rho_c$, the Hill curve reaches half its maximum at $\kappa_c$, and
$\beta_c$ is the most channel $c$ can add in a week. Each dollar the price
rises above \$20 costs 1.5 cents of revenue per person. The simulation draws
its random quantities once.

$$
\begin{aligned}
P &\sim \operatorname{UniformInteger}(100{,}000,\ 500{,}000) \\
u &\sim \operatorname{Uniform}(0.8, 1.2), \quad b = P\, u \\
\epsilon_c &\sim \operatorname{LogNormal}(0, 0.15), \quad \beta_c = \theta_c\, P\, \epsilon_c \\
\xi_t &= 0.8\, \xi_{t-1} + 0.6\, \varepsilon_t, \quad \varepsilon_t \sim \operatorname{Normal}(0, 1), \quad q_t = 20 \exp(0.08\, \xi_t) \\
\log \eta_t &\sim \operatorname{Normal}\big(-\tfrac{1}{2}\varsigma^2, \varsigma\big), \quad \varsigma^2 = \log\big(1 + 0.02^2\big)
\end{aligned}
$$

The price wanders slowly around \$20, and the noise has mean one and a
standard deviation of 2 percent. The channel table holds each channel's fixed
settings, including its coefficient per person $\theta_c$.

```{code-cell} ipython3
columns = [
    "activity",
    "spend_per_person",
    "cpm",
    "retention",
    "half_saturation",
    "slope",
    "coefficient_per_person",
]
example.channels.set_index("channel")[columns]
```

Impressions come from a flight calendar. Flights are planned every 13 weeks,
start anywhere from two weeks early to four weeks late, last three to six
weeks, and run at a
strength between 0.8 and 1.4, and each channel shares the brand's calendar
with probability 0.7 or draws its own. TV is on only during its flights, and
search runs at a base level all year and rises during its flights. The
budget, the channel's share of it, and weekly execution noise turn this
activity $a_{tc}$ into spend, and a drifting cost per thousand impressions
turns spend into impressions,

$$
S_{tc} = \phi_c\, P\, a_{tc}\, \lambda_c \exp\big(0.16\, \xi^{\text{budget}}_t + 0.08\, \xi^{\text{exec}}_{tc}\big),
\qquad
z_{tc} = \frac{1000\, S_{tc}}{k_c \exp\big(0.08\, \xi^{\text{cpm}}_{tc} + 0.03\, t / 52\big)},
$$

where $\phi_c$ is the spend per person, $k_c$ the base cost per thousand
impressions, $\lambda_c \sim \operatorname{LogNormal}(0, 0.15)$ the channel's
share of the budget, and each $\xi$ its own AR(1) series like the one behind
price. The simulation starts eight weeks before the first week in the frame, so
carryover is already under way in January 2022. With seed 7 the population is 222,147,
the baseline is about \$180,800 a week, and the most each channel can add in a
week is about \$33,500 for TV and \$26,200 for search.

TV runs in flights with long gaps between them, and search stays on every
week. Each week's impressions carry forward at the channel's retention rate,
so a week of TV keeps working for several weeks, while search's effect is
mostly gone the week after. Plotting the impressions above what each driver
adds to revenue shows both.

```{code-cell} ipython3
truth = example.truth
labels = {"linear_tv": "TV", "generic_search": "Search"}

fig, (impressions, effects) = plt.subplots(2, 1, sharex=True, layout="constrained")
for channel, label in labels.items():
    impressions.plot(example.frame["week"], example.frame[f"{channel}_impressions"], linewidth=1, label=label)
    effects.plot(truth["time"], truth["contribution"].sel(channel=channel), linewidth=1, label=label)
effects.plot(truth["time"], truth["price_effect"], linewidth=1, label="Price")
impressions.set_title("Weekly impressions")
effects.set_title("What each driver adds to weekly revenue")
impressions.legend(frameon=False)
effects.legend(frameon=False)
plt.show()
```

Each TV flight lifts revenue and then fades over the following weeks, so TV adds little or nothing by the time the next
flight starts and up to about \$21,500 in its best weeks.
Search adds a steadier \$12,300 a week on average and rises with its own
campaign bursts. Price moves revenue by as much as \$14,000 down and \$11,000 up, on the same
scale as the channels, which is why the model needs it as a control.
Adding the three to a constant baseline of about \$181,000 a week gives the
expected revenue.

```{code-cell} ipython3
fig, axis = plt.subplots(layout="constrained")
axis.plot(truth["time"], truth["revenue"], color="black", linewidth=1, label="Observed revenue")
axis.plot(truth["time"], truth["expected_revenue"], color="tab:orange", linewidth=1.5, label="Expected revenue")
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
import pandas as pd

shares = truth["contribution"].sum("time") / truth["expected_revenue"].sum()
pd.DataFrame({"contribution_share": shares.to_series(), "roi": truth["roi"].to_series()}).round(3)
```

Over the three years, TV produced about 4.3 percent of revenue and search
about 6.1 percent, and each dollar returned about \$3.70 on TV and \$4.50 on
search. [Recovering the truth](recovery) compares the first model's estimates with
these values and with the rest of this record. The first model departs from
the simulation in three places. It fixes every Hill slope at one, which misses
TV's slightly S-shaped start. It treats the weeks before the data as having no
exposure, which misses the carryover from a TV flight at the end of December
2021. Its noise also adds to revenue where the simulation's multiplies it,
which matters little at 2 percent. [Changing the model](changing) shows what
happens when the slope is set free.
