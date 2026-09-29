---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Geo-level models

The national data behind [A first model](first_model) runs several channels on
one campaign calendar. The fit can't tell how a flight's lift splits among
them, so it leaves the split to the ROI prior, as
[Recovering the truth](recovery) shows. Many brands also keep their data by
region, and a region that runs a channel off the brand's calendar shows that
channel's effect apart from the rest. This page fits the ten-channel brand by
region, letting each channel's return vary by region around a shared one, and
asks what the regional data buys.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_prior_results, first_model_results, stored

results = first_model_results(model)
prior_results = first_model_prior_results(model, priors)

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
```

:::{admonition} A separate draw
:class: note

The regional data comes from the same simulator, seed, and channel settings as
the national series, but it's a draw of its own and not the national series
split three ways. Its true returns differ from those on
[A first model](first_model), so this page checks each fit against its own
truth.
:::

## The regional data

{func}`~mmmjax.simulate_data` makes three regions unless `groups=None` asks for
one series. Preparing it works as it did nationally, with two additions.
`groups` adds the region axis that [Data and scaling](data.md#groups)
describes, and `population` records each region's size for the scaling below.

```{code-cell} ipython3
import pandas as pd

regional = mj.simulate_data(seed=7)
geo_data = mj.prepare_data(
    regional.frame,
    time="week",
    groups=["region"],
    outcome="revenue",
    population="population",
    media=[f"{name}_impressions" for name in channels],
    spend=[f"{name}_spend" for name in channels],
    channels=list(channels.values()),
    organic_media=["email_sends"],
    organic_channels=["Email"],
    controls=["demand", "holiday"],
    treatments=["price", "promotion"],
)
regions = [label for (label,) in geo_data.group_values]
pd.Series(geo_data.arrays["population"], index=regions, name="population")
```

As [The example data](example_data) describes, a channel usually copies the
brand's campaign calendar and otherwise runs flights of its own. The regional
data makes that draw once per region, so each region gets its own calendar for
every channel. The promotions follow the brand's calendar everywhere, so a
channel's correlation with the promotion weeks shows which calendar it follows.
The table below gives that correlation for five of the paid channels that air
in flights, for Email, and for the price, in each region and in the national
series from [A first model](first_model).

```{code-cell} ipython3
:tags: [hide-input]

flighted = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "streaming": "Streaming",
    "linear_tv": "Linear TV",
    "influencer": "Influencer",
}
columns = [f"{name}_impressions" for name in flighted] + ["email_sends", "price"]


def with_promotions(frame):
    return frame[columns].corrwith(frame["promotion"])


calendars = regional.frame.groupby("region").apply(with_promotions).T
calendars.insert(0, "national", with_promotions(brand.frame))
calendars.index = [*flighted.values(), "Email", "Price"]
calendars.round(2)
```

Nationally, TikTok, Streaming, Linear TV, Influencer, and Email correlate with
the promotions at 0.78 or 0.79, since all five follow the shared calendar that
[Data and scaling](data.md) found. YouTube runs on a calendar of its own, at
0.02. Each region breaks the tie for some channels. TikTok and Influencer leave
the brand's calendar in the south, Streaming in the north, and Linear TV and
Email in the west, where their correlations fall to between 0.08 and 0.31. Those
regions show a channel's lift in weeks when the others are quiet, and a model
that shares what it learns across regions can use that everywhere. YouTube goes
the other way in this draw and follows the brand's calendar in all three
regions.

The price correlates with the promotions at -0.92 in every region, as it does
nationally, because the brand cuts its price in every promotion week
everywhere. No region breaks that tie.

## Scaling by population

The south has 419,144 people and the west 181,559, and a region's impressions
and revenue grow with its size. `adjust_population=True` divides each
region's exposure by its population before taking each channel's median, and
`scale_outcome="population"` standardizes revenue per person.

```{code-cell} ipython3
geo_scaling = mj.fit_data_scaling(geo_data, scale_outcome="population", adjust_population=True)
revenue_scale = geo_scaling.transformations["outcome"].scale
pd.Series(revenue_scale[0], index=regions, name="dollars per unit").round().astype(int)
```

Each value is the standard deviation of weekly revenue per person, pooled over
the weeks and the regions, times the region's population. One unit of scaled
revenue is \$119,004 a week in the south and \$51,549 in the west. Once
exposure and revenue are both per person, a half-saturation point or a
coefficient means the same thing in every region, and that's what lets the
regions share them. The simulation works per person too, as
[The example data](example_data) shows.

## The model

The regional model keeps the blocks of [A first model](first_model) and makes
three changes.

- Each region gets its own intercept, because the regions differ in baseline
  revenue per person, and a shared intercept would push that difference into
  the channels.
- Each channel's return varies by region around a shared return, because
  audiences and prices differ from one region to the next, much as
  [Sun et al. (2017)](https://research.google.com/pubs/archive/46000.pdf) let
  each region's coefficients vary around shared values.
- Exposure and revenue are scaled by population, as above, so every other
  parameter can stay shared.

Carryover and saturation stay shared across regions. Sun et al. expected
region-level curves to make the model hard to identify, and three years of
weekly data from three regions leave little to pin down 30 of them. The trend,
the season, the controls, the treatments, Email, and the noise stay shared as
well. Region $g$ runs over north, south, and west, with population $P_g$. The
specification below covers what changes, and everything else stays as in
[A first model](first_model).

### Data transformations

$$
\begin{gathered}
x_{tgc} = \frac{z_{tgc}}{P_g m_c}, \qquad
x_{tgo} = \frac{n_{tgo}}{P_g m_o}, \qquad
y_{tg} = \frac{R_{tg} / P_g - \bar{R}}{s_R}, \\
p_{tgj} = \frac{q_{tgj} - \bar{q}_j}{s_j}, \qquad
\tilde{w}_{tgi} = \frac{w_{tgi} - \bar{w}_i}{s_i}
\end{gathered}
$$

The median $m_c$ is now taken over impressions per person in every region and
week with any exposure. Here $\bar{R}$ and $s_R$ are the mean and standard
deviation of weekly revenue per person over all weeks and regions. The controls
and the treatments pool the weeks and the regions the same way. The scaled
treatment is written $\tilde{w}_{tgi}$ here, since $g$ now counts regions. The
trend $\tau_t$ and the day of the year $d_t$ are the same in every region.
Channel $c$'s spending $S_c = \sum_t \sum_g v_{tgc}$ now totals every region.

### Model equation

$$
\begin{aligned}
y_{tg} &= \alpha_g + \delta_1 \tau_t + \delta_2 \tau_t^2
+ \sum_{k=1}^{2} \Big( a_k \sin\frac{2\pi k d_t}{365.25} + b_k \cos\frac{2\pi k d_t}{365.25} \Big) \\
&\quad + \sum_{c=1}^{10} \beta_{gc} h_{tgc} + \sum_{o} \lambda_o h_{tgo}
+ \sum_{j=1}^{2} \gamma_j p_{tgj} + \sum_{i=1}^{2} \theta_i \tilde{w}_{tgi} + \varepsilon_{tg}, \\
\varepsilon_{tg} &\sim \operatorname{Normal}(0, \sigma), \\
h_{tgc} &= \operatorname{HillAdstock}\big(\{x_{t-\ell,gc}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big), \\
h_{tgo} &= \operatorname{HillAdstock}\big(\{x_{t-\ell,go}\}_{\ell=0}^{8};\, \rho_o, \kappa_o\big)
\end{aligned}
$$

The intercept $\alpha_g$ and the channel coefficients $\beta_{gc}$ carry the
region. Every other term has the meaning it has in the first model.

### Regional returns

A unit of $y_{tg}$ is $s_R P_g$ dollars in region $g$. The model samples a
shared return $r_c$ for each channel and a standard normal offset $u_{gc}$ for
each region, scales the offsets by a spread $\eta_c$, and sets

$$
\beta_{gc} = \frac{r_c S_c\, e^{\eta_c u_{gc}}}
{\sum_{g'} s_R P_{g'}\, e^{\eta_c u_{g'c}} \sum_t h_{tg'c}},
$$

with $h_{tgc}$ computed from the training exposure. The regions' contributions
then add up to $\sum_g s_R P_g \beta_{gc} \sum_t h_{tgc} = r_c S_c$, so $r_c$
keeps its meaning from the first model as the return on every dollar the
brand spent on channel $c$. The factor $e^{\eta_c u_{gc}}$ sets how the
regions' coefficients compare. A region's own return also depends on how far
its spending pushes it along the saturation curve.

Email's coefficient stays shared, and its share $\phi_o$ now covers the
revenue of every region,

$$
\lambda_o = \frac{\phi_o \sum_t \sum_g R_{tg}}{\sum_g s_R P_g \sum_t h_{tgo}}.
$$

### Priors

The region intercepts replace the single intercept, and the spread and the
offsets are new.

$$
\begin{aligned}
\alpha_g &\sim \operatorname{Normal}(0, 1) \\
\eta_c &\sim \operatorname{HalfNormal}(0.2) \\
u_{gc} &\sim \operatorname{Normal}(0, 1)
\end{aligned}
$$

The other priors are those of [A first model](first_model). A spread of 0.2
lets a region's coefficient sit about 20 percent above or below the center,
since $e^{0.2}$ is about 1.2. The half-normal favors smaller spreads over
larger ones.

:::{admonition} Write the offsets noncentered
:class: tip

Drawing each region's log coefficient around the center with standard
deviation $\eta_c$ would make the offsets shrink with $\eta_c$. As $\eta_c$
nears zero, the posterior then narrows into a funnel the sampler can enter only
with steps too small for the rest of the model, and that shows up as
divergences. Standard normal offsets multiplied by $\eta_c$, as here, give the
sampler the same geometry whatever the spread.
:::

## The code

The region index $g$ becomes the `group` axis that `groups` added to the data,
and each new symbol takes a name in the code.

| Symbol | Name in the code | Where the name comes from |
| --- | --- | --- |
| $\alpha_g$ | `intercept` | Declared in `parameters`, now with the `group` axis |
| $\eta_c$ | `roi_spread` | Declared in `parameters` as positive, with the `channel` axis |
| $u_{gc}$ | `roi_offset` | Declared in `parameters`, with the `group` and `channel` axes |
| $\eta_c u_{gc}$ | `deviations` | Local to `transformed_parameters` |
| $\beta_{gc}$ | `coefficient` | Local to `transformed_parameters` |
| $s_R P_g$ | `outcome_scaling.scale` | Supplied by mmmJAX, now one value per region |

`transformed_parameters` passes `deviations` to {func}`~mmmjax.roi_coefficient`,
which returns `coefficient` with one value for each region and channel. The
supplied `outcome_scaling` already holds $s_R P_g$ for each region, so no block
needs the population $P_g$ itself.

`geo_parameters` starts from the first model's declarations, redeclares
`intercept`, and adds the spread and the offsets.

```{code-cell} ipython3
geo_parameters = parameters | {
    # One intercept for each region, in place of the first model's single intercept.
    "intercept": mj.Real(dims="group"),
    # How far each channel's return varies by region, and a standard normal offset per region and channel.
    "roi_spread": mj.Positive(dims="channel"),
    "roi_offset": mj.Real(dims=("group", "channel")),
}
```

:::{admonition} Groups add a supplied axis
:class: important

Passing `groups` to `prepare_data` gives the data a `group` axis over the
regions, so `dims="group"` works here and fails on the national data. The
supplied names stay the same, but `media`, `outcome`, and the other
observation arrays gain that axis after time, while `time` and `day_of_year`
keep only their time axis.
:::

`transformed_parameters` changes in a few lines.

```{code-cell} ipython3
import jax.numpy as jnp


def geo_transformed_parameters(
    media,
    organic_media,
    controls,
    treatments,
    reference,
    outcome_scaling,
    annual,
    trend,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    roi_spread,
    roi_offset,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
):
    # Paid media's coefficients read the training exposure. reference holds the training data
    # in every scenario, so the coefficients keep their fitted values when an analysis changes media.
    trained = hill_adstock(reference.media, retention, half_saturation)

    # A region's deviation is its standard normal offset times the channel's spread. This
    # noncentered form gives the sampler the same geometry whatever the spread.
    deviations = roi_spread * roi_offset

    # The deviations set how the regions' coefficients compare, and roi_coefficient sets their
    # level so the regions' contributions add up to roi times the channel's total spending.
    coefficient = mj.roi_coefficient(
        roi, trained, reference.spend, outcome_scale=outcome_scaling.scale, deviations=deviations
    )

    # Email's coefficient is shared, and its share covers the revenue of every region. The share
    # is of dollars, since standardized revenue sums to zero over the training weeks and regions.
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )

    # Every region shares the trend and the season, and adds its own intercept because the
    # regions differ in baseline revenue per person.
    shared = growth * trend + curvature * trend**2 + annual @ annual_coefficients
    baseline = intercept + shared[:, None]

    # The paid coefficients differ by region, so the media effect uses einsum in place of @.
    # "tgc,gc->tg" lists each input's axes, then the output's, with t for week, g for region,
    # and c for channel. c is absent from the output, so the channels are summed over.
    saturated = hill_adstock(media, retention, half_saturation)
    media_effect = jnp.einsum("tgc,gc->tg", saturated, coefficient)

    # The other effects work as in the first model, and every effect reads the inputs the
    # model is given, so a scenario that changes media, sends, or prices changes it.
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient

    # Expected revenue in each week and region, which the likelihood and every analysis read.
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}
```

{func}`~mmmjax.roi_coefficient` applies the formula for $\beta_{gc}$ above, and
its default `effects="lognormal"` keeps every coefficient positive. With the
region axis, `hill_adstock` returns weeks by regions by channels. `jnp.einsum`
multiplies it by the region-by-channel coefficients and sums over channels,
which `@` can't express because each region has coefficients of its own.
`shared` holds the trend and the season, and `[:, None]` spreads it over the
regions before the intercepts are added. Email, the controls, and the
treatments work unchanged.

:::{admonition} One scale per region
:class: warning

Outside the model, `geo_scaling.transformations["outcome"].scale` keeps a
leading time axis of length one, as the cell above shows. Inside a block, pass
`outcome_scaling.scale` to both calibration functions as it is, since taking
its first element would hand every region the north's scale.
:::

The density adds a prior for the spread and one for the offsets to the first
model's thirteen.

```{code-cell} ipython3
def geo_log_density(
    outcome,
    mu,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    roi_spread,
    roi_offset,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    sigma,
):
    # The baseline's priors describe standardized revenue, and every region's intercept gets
    # the same Normal(0, 1). The curvature's scale is smaller because trend**2 outgrows trend
    # after the first year.
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)

    # Each channel's shared return, and the spread and offsets that let it vary by region.
    # A spread of 0.2 lets a region's coefficient sit about 20 percent above or below the
    # center, and the half-normal favors smaller spreads.
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.half_normal(roi_spread, 0.2)
    target += mj.normal(roi_offset, 0.0, 1.0)

    # Carryover and saturation stay shared, since three years of weekly data from three
    # regions leave little to pin down a curve for each region and channel.
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)

    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)

    # The treatments get a tighter prior than the controls. The price climbs with the trend,
    # and a wider prior would let the fit credit price with growth.
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)

    # The noise scale's prior and the likelihood, normal noise around the expected revenue.
    # One noise scale serves every region.
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


geo_model = mj.Model(
    parameters=geo_parameters,
    data=mj.Data(geo_data, scaling=geo_scaling),
    transformed_data=transformed_data,
    transformed_parameters=geo_transformed_parameters,
    log_density=geo_log_density,
    generated_quantities=generated_quantities,
)
```

`mj.normal(intercept, 0.0, 1.0)` sums the log density over the three
intercepts, as every distribution function sums over its first argument. The
first model's `transformed_data` and `generated_quantities` work as they are.
`transformed_data` depends only on the dates, and `generated_quantities` draws
and scores `mu` whatever its shape.

## Fitting

The fit uses the same settings as the first model.

```{code-cell} ipython3
:tags: [skip-execution]

geo = mj.sample(geo_model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

geo = stored(
    "geo",
    lambda: mj.sample(geo_model, draws=1000, warmup=1000, chains=4, seed=7),
    groups=["posterior", "sample_stats"],
)
```

A badly parameterized hierarchy shows itself in divergences, so check them
first.

```{code-cell} ipython3
int(geo["sample_stats"]["diverging"].sum())
```

One of the 4,000 transitions diverged. A funnel would crowd its divergences
where a spread nears zero, so the output below gives where each spread at
the divergent draw falls within its own posterior.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

spreads = geo["posterior"]["roi_spread"]
chain, draw = np.argwhere(geo["sample_stats"]["diverging"].values)[0]
divergent = spreads.isel(chain=chain, draw=draw, drop=True)
(spreads < divergent).mean(("chain", "draw")).to_series().round(2)
```

They fall between the 14th and 86th percentiles, none in the lowest tenth, so
this divergence points to no funnel. The summary covers the parameters the
rest of the page reads.

```{code-cell} ipython3
print(
    az.summary(
        geo,
        var_names=["intercept", "roi", "roi_spread", "organic_share", "treatment_coefficient", "sigma"],
    )
)
```

Every `r_hat` is 1.00, and the lowest bulk ESS is 1,188, for the north's
intercept. {func}`~mmmjax.plot_rank` then keeps the 12 elements with the
highest `r_hat` in the whole model.

```{code-cell} ipython3
mj.plot_rank(geo)
plt.show()
```

Snapchat's spread fails the test with a p that rounds to 0.00. Black dots mark
the stretch below its 20th percentile, where chain 1 holds too few draws and
chain 2 too many. Every other panel's p sits above the 1 percent level, the
lowest at 0.02 for YouTube's half-saturation, and none of the ten returns is
among the twelve, so the comparisons below can use these draws. Before you
report from a fit like this one, run it longer, as
[Sampling and diagnostics](sampling.md#more-draws) does, and check that
divergences stay rare and every panel passes.

## Returns

{func}`~mmmjax.media_metrics` measures each channel's return in both fits.
`by="group"` keeps each region's response, which the last section reads, and
the returns themselves still pool every region.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
geo_returns = mj.media_metrics(geo_model, geo, quantity="mu", by="group")
mj.plot_media_metrics({"National": returns, "Regional": geo_returns})
```

The table below sets each fit's 89 percent interval beside its own truth.

```{code-cell} ipython3
:tags: [hide-input]

def interval(fit_returns, label):
    bounds = fit_returns["roi"].quantile([0.055, 0.945], dim=("chain", "draw")).T.to_pandas()
    bounds.columns = [f"{label}_low", f"{label}_high"]
    return bounds


regional_spend = geo_data.arrays["spend"].sum(axis=0)
regional_added = regional.truth["contribution"].sel(channel=list(channels)).sum("time")
comparison = pd.concat([interval(returns, "national"), interval(geo_returns, "regional")], axis=1)
comparison.insert(2, "national_truth", brand.truth["roi"].values)
comparison["regional_truth"] = regional_added.sum("group").values / regional_spend.sum(axis=0)
comparison.round(2)
```

The regional intervals are shorter for six of the ten channels. Streaming's
and Linear TV's shrink to less than half their national length, and
Streaming's regional mean of \$3.42 sits on its true \$3.42. YouTube's
interval grows to about twice its national length, since YouTube follows the
brand's calendar in every region of this draw and the data rarely sees it
alone.

The national fit holds all ten of its truths, and the regional fit holds nine.
It misses only Linear TV, whose interval runs from 1.48 to 3.61, below its
true 3.82. TikTok's interval, from 5.98 to 13.29, holds its true 6.51, though
its mean of \$9.53 sits well above it. Both channels share the brand's
calendar in the north and leave it in one other region, so the regional data
holds weeks of each on its own. Even so, the fit gives TikTok too much and
Linear TV too little.

The national means lean low, with eight of the ten below their truth where
the ROI prior pulls them, as [Recovering the truth](recovery) found. The
regional means split evenly. Meta, TikTok, YouTube, and Branded search sit
above their truth, Streaming sits on it, and the other five fall below.

Linear TV's regional interval is about half as long as its national one and
still misses the truth. The model differs from the simulation in the ways
[The example data](example_data) lists, such as the Hill slopes the simulation
varies and the model fixes at one. Each region adds weeks to learn from, and
more data makes a fit more confident in whatever answer those differences
favor.

:::{admonition} Narrower is not always closer
:class: warning

Check a regional fit against what you know, such as an experiment, as
carefully as a national one.
:::

## Price, promotions, and Email

[Priors](priors) found that the national fit repeats Email's prior and can't
say whether the price's effect is negative. The two models state the same priors
for Email's share and the treatment coefficients, so the prior draws from
[A first model](first_model) serve the regional fit too.

```{code-cell} ipython3
mj.plot_prior_posterior(geo, prior_results, var_names=["organic_share", "treatment_coefficient"])
plt.show()
```

Email's orange curve is now much narrower than its blue prior, around the mean
of 0.0109 in the summary above. The price's curve moves left, toward higher
prices losing revenue, and the promotion's settles just above zero. Both are
narrower than their priors but still wide. {func}`~mmmjax.contributions` turns
these into shares of revenue. The table below sets their 5th, 50th, and 95th
percentiles, in percent, beside the truth of each simulation. The simulation's
price term takes \$0.015 of revenue per person for each dollar of price.

```{code-cell} ipython3
:tags: [hide-input]

def true_shares(simulation, lowest_price):
    truth = simulation.truth
    frame = simulation.frame
    price_change = -0.015 * frame["population"] * (frame["price"] - lowest_price)
    added = [
        truth["contribution"].sel(channel="email").sum().item(),
        price_change.sum(),
        truth["promotion_effect"].sum().item(),
    ]
    return 100 * np.array(added) / truth["expected_revenue"].sum().item()


fits = {"national": (model, results, brand), "regional": (geo_model, geo, regional)}
shares = {}
for label, (fit_model, fit, simulation) in fits.items():
    effects = mj.contributions(fit_model, fit, quantity="mu", channels=["Email", "price", "promotion"])
    share = 100 * effects["contribution_share"]
    table = share.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).T.to_pandas()
    lowest_price = effects["treatment_baseline"].sel(channel="price").item()
    table["truth"] = true_shares(simulation, lowest_price)
    shares[label] = table
pd.concat(shares).round(2)
```

Email's interval shrinks from 0.32 to 3.44 percent in the national fit to 0.33
to 2.05 percent in the regional one, and its median of 1.03 percent sits
closer to the true 0.77. The west sends Email off the brand's calendar, so
the data sees its lift apart from the flights there.

The national fit's price interval, from -11.56 to 6.22 percent, holds the true
-2.73 but can't say whether the higher prices cost revenue. The regional one,
from -15.67 to 3.29 percent around a true -3.10, can't say either, and its
median of -6.04 puts the loss at about twice the truth. The promotions go the
other way. Their national interval, from -1.33 to 7.09 percent, holds the true
1.56, and the regional one runs from -3.92 to 5.03 percent around a true 1.81,
with its median of 0.60 well below it.

Regional data separates two inputs only where some region moves one without
the other. The price and the promotions move together in every region, so the
regional fit still can't tell a promotion week's lift from its price cut. It
gives the price cut more of that lift than the simulation did and the flag
less.

## Pooling toward the shared return

A region's return is its own incremental revenue over its own spending. To
get it, divide the regional responses that `by="group"` kept by each region's
spending. {func}`~mmmjax.plot_media_metrics` draws a panel for each region, and
`channels` picks TikTok and Linear TV from above, along with Snapchat and
YouTube.

```{code-cell} ipython3
:tags: [hide-input]

import xarray as xr

region_axes = {"group": geo_returns["group"], "channel": geo_returns["channel"]}
spend_by_region = xr.DataArray(regional_spend, coords=region_axes)
region_returns = xr.Dataset({"roi": geo_returns["incremental_response"] / spend_by_region})
true_region = regional.truth[["roi"]].rename(paid_channel="channel")
true_region = true_region.assign_coords(channel=list(channels.values())).expand_dims(chain=[0], draw=[0])
shown = ["TikTok", "Snapchat", "YouTube", "Linear TV"]
mj.plot_media_metrics({"Model": region_returns, "Truth": true_region}, channels=shown, n_groups=None)
```

The true returns differ a lot between regions. Snapchat's is \$8.75 in the
west and \$4.44 in the south, and TikTok's runs from \$4.99 in the west to
\$7.09 in the south. The model's regional means differ far less. Snapchat's run
from \$3.79 to \$5.47 around its shared \$4.16, and TikTok's from \$9.08 to
\$10.22 around its shared \$9.53. Nine of the twelve intervals hold their
truth. The three that miss are Linear TV's in the south and the west, below
the truth, and TikTok's in the west, where its true \$4.99 falls below the
interval.

The spread $\eta_c$ decides how far the regions may part. The table below sets
its posterior percentiles beside its prior's.

```{code-cell} ipython3
:tags: [hide-input]

import jax

percentiles = [0.055, 0.5, 0.945]
spread = geo["posterior"]["roi_spread"].quantile(percentiles, dim=("chain", "draw")).T.to_pandas()
prior_spread = mj.half_normal_rng(jax.random.key(0), 0.2, sample_shape=(4000,))
spread.loc["prior"] = np.quantile(prior_spread, percentiles)
spread.round(3)
```

Every channel's percentiles sit close to the prior's 0.014, 0.135, and 0.390, so
the data says almost nothing about how far the regions differ. Three regions
give each channel only three offsets to learn a spread from, which leaves the
spread to the prior, and the model pools each region's return close to the
shared one. That pooling is what lets the regional data sharpen
most of the brand-wide returns without splitting the fit three ways.

A brand with dozens of regions would learn the spread from the data. With few
regions, one spread shared by every channel gives the data thirty offsets to
learn it from instead of three.
