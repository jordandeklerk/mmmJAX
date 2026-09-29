---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Changing the model

The model is a set of functions you wrote, so changing it means editing those
functions, and nothing else in the workflow has to move. This page makes one
change to the ten-channel brand model from [A first model](first_model).

That model gives each paid channel one coefficient for all 156 weeks, so an ad
works as well in the last week as in the first. Real effects drift. Creative
wears out, competitors come and go, and audiences change, and a constant
coefficient then reports an average over weeks that differ
([Ng, Wang, and Dai, 2021](https://arxiv.org/abs/2106.03322)). The change here
lets the media effect move smoothly from week to week.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import stored

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

## A brand whose ads wear out

The simulated brand's media effects stay constant, so this page builds a copy
whose effects fade. Its ten paid channels lose effect over the three years,
slowly at first and last and fastest in 2023, from one and a half times their
simulated effect in the first week to half of it in the last,

$$
\text{wear}_t = 1 + 0.5 \cos\frac{\pi t}{155}, \qquad t = 0, \dots, 155,
$$

and each week's revenue is rebuilt around that with the same noise.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import xarray as xr

truth = brand.truth
weeks = np.arange(truth.sizes["time"])
wear = xr.DataArray(1.0 + 0.5 * np.cos(np.pi * weeks / weeks[-1]), coords={"time": truth["time"]})
paid = truth["contribution"].sel(channel=list(channels))
expected = truth["expected_revenue"] + ((wear - 1.0) * paid).sum("channel")
# The simulation's noise multiplies expected revenue, so each week keeps its draw.
revenue = expected * truth["revenue"] / truth["expected_revenue"]
worn = brand.frame.assign(revenue=revenue.values)
```

The data is then prepared exactly as for the first model.

```{code-cell} ipython3
worn_data = mj.prepare_data(
    worn,
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
worn_scaling = mj.fit_data_scaling(worn_data, scale_outcome=True)
```

## A multiplier on the media effect

A multiplier $\zeta_t$ scales the paid channels' effect in week $t$, and a
Gaussian process over the weeks lets it follow any smooth path. The
Hilbert-space approximation writes the process as a fixed set of sine
functions times coefficients, so the sampler works with ordinary parameters
([Riutort-Mayol et al., 2022](https://arxiv.org/abs/2004.11408)). Everything
else stays as in [A first model](first_model),

$$
\begin{aligned}
\mu_t &= \cdots + \zeta_t \sum_{c=1}^{10} \beta_c h_{tc} + \cdots, \qquad
\beta_c = \frac{r_c S_c}{s_R \sum_t \zeta_t h_{tc}}, \\
\log \zeta_t &= \sum_{m=1}^{M} \big(\psi_m(e_t) - \bar{\psi}_m\big) \sqrt{\mathcal{S}(\omega_m)}\, z_m, \qquad
\mathcal{S}(\omega) = \eta^2 \sqrt{2\pi}\, \ell \exp\big(-\ell^2 \omega^2 / 2\big), \\
z_m &\sim \operatorname{Normal}(0, 1), \qquad
\ell \sim \operatorname{LogNormal}(5.5, 0.5), \qquad
\eta \sim \operatorname{HalfNormal}(0.25),
\end{aligned}
$$

where $e_t$ counts the days from the first training week to week $t$, as in
the first model, $\psi_m$ is the $m$th of $M$ basis functions, $\bar{\psi}_m$
its mean over the training weeks, $\omega_m$ its frequency, and $\mathcal{S}$
the spectral density of the squared exponential covariance, `expquad` in the
code. The length scale $\ell$ is in days, and its prior puts the median at
about 245 days, so the effect turns over seasons rather than weeks. The
amplitude $\eta$ sets how far $\log \zeta_t$ swings, and under its prior most
paths stay within a factor of two from their lowest week to their highest.

The model works on $\log \zeta_t$, so the multiplier stays positive. The
multiplier also enters the sum in $\beta_c$, so $r_c$ stays the channel's return
over the training weeks. Doubling every $\zeta_t$ then halves every $\beta_c$
and leaves the revenue unchanged, so the data can't pin down the multiplier's
level. Subtracting $\bar{\psi}_m$ fixes it by making $\log \zeta_t$ average zero
over the training weeks, so $\zeta_t$ reads as the effect relative to that
average. One multiplier serves all ten paid channels, and Email keeps a
constant coefficient.

The tables below sort the new symbols by where each name comes from, and
highlight each symbol in a formula that uses it.

::::{tab-set}

:::{tab-item} Supplied by mmmJAX

`reference.time` gives the days of the training weeks, which set the
multiplier that $\beta_c$ divides by, and `time` gives those of the weeks
being evaluated.

| Symbol | In the math | Name in the code |
| --- | --- | --- |
| $e_t$ | $\log \zeta_t = \sum_{m} \big(\psi_m(\hl{e_t}) - \bar{\psi}_m\big) \sqrt{\mathcal{S}(\omega_m)}\, z_m$ | `time`, and `reference.time` for the training weeks |

:::

:::{tab-item} Declared in parameters

Each unknown is one of your names. The basis functions have no data axis, so
`mj.Real(approximation.n_basis)` gives the coefficients a plain shape.

| Symbol | In the math | Name in the code |
| --- | --- | --- |
| $z_m$ | $\log \zeta_t = \sum_{m} \big(\psi_m(e_t) - \bar{\psi}_m\big) \sqrt{\mathcal{S}(\omega_m)}\, \hl{z_m}$ | `multiplier_coefficients` |
| $\ell$ | $\mathcal{S}(\omega) = \eta^2 \sqrt{2\pi}\, \hl{\ell} \exp\big(-\ell^2 \omega^2 / 2\big)$ | `length_scale` |
| $\eta$ | $\mathcal{S}(\omega) = \hl{\eta}^2 \sqrt{2\pi}\, \ell \exp\big(-\ell^2 \omega^2 / 2\big)$ | `amplitude` |

:::

:::{tab-item} Computed in the blocks

`trained` now holds the training exposure already multiplied by $\zeta_t$, so
the first model's `roi_coefficient` call sets $\beta_c$ from the new sum.

| Symbol | In the math | Name in the code | Where it's computed |
| --- | --- | --- | --- |
| $\zeta_t$ | $\mu_t = \cdots + \hl{\zeta_t} \sum_{c} \beta_c h_{tc} + \cdots$ | `multiplier` | Returned by `transformed_parameters` |
| $\zeta_t$ in the training weeks | $\beta_c = \dfrac{r_c S_c}{s_R \sum_t \hl{\zeta_t} h_{tc}}$ | `training_multiplier` | Local to `transformed_parameters` |
| $\zeta_t h_{tc}$ in the training weeks | $\beta_c = \dfrac{r_c S_c}{s_R \sum_t \hl{\zeta_t h_{tc}}}$ | `trained` | Local to `transformed_parameters` |
| $\psi_m(e_t) - \bar{\psi}_m$ | $\log \zeta_t = \sum_{m} \hl{\big(\psi_m(e_t) - \bar{\psi}_m\big)} \sqrt{\mathcal{S}(\omega_m)}\, z_m$ | `multiplier_basis`, and `training_basis` for the training weeks | Returned by `transformed_data` |
| $\sqrt{\mathcal{S}(\omega_m)}$ | $\log \zeta_t = \sum_{m} \big(\psi_m(e_t) - \bar{\psi}_m\big) \hl{\sqrt{\mathcal{S}(\omega_m)}}\, z_m$ | `weights` | Local to `transformed_parameters` |
| $M$, $\omega_m$, $\psi_m$, and $\bar{\psi}_m$ | $\log \zeta_t = \sum_{m=1}^{\hl{M}} \big(\hl{\psi_m}(e_t) - \hl{\bar{\psi}_m}\big) \sqrt{\mathcal{S}(\hl{\omega_m})}\, z_m$ | `approximation` | Built by `prepare_hsgp` before the model |

:::

::::

{func}`~mmmjax.prepare_hsgp` builds that object once, outside the model, from
the training weeks.

```{code-cell} ipython3
approximation = mj.prepare_hsgp(
    worn_data.time_positions,
    length_scale_range=(90.0, 730.0),
    covariance="expquad",
    center_columns=True,
)
approximation.center, approximation.boundary, approximation.n_basis
```

`time_positions` holds the days since the first week, and
`length_scale_range` names the length scales the multiplier should be able to
use, from 90 days to two years. The basis centers on day 542.5, the middle of
the training weeks, and reaches 2,336 days to either side, since a process
with a long length scale needs room past the data or it bends toward zero at
the edges. Its 45 functions follow the sizing rule for the shortest length
scale in the range, and `center_columns=True` subtracts each function's mean
over the training weeks, the $\bar{\psi}_m$ above.

```{code-cell} ipython3
import jax.numpy as jnp

# The first model's parameters, plus the process behind the multiplier.
varying_parameters = parameters | {
    # One standard-normal coefficient per basis function.
    "multiplier_coefficients": mj.Real(approximation.n_basis),
    # How many days the effect takes to change, and how far it swings.
    "length_scale": mj.Positive(),
    "amplitude": mj.Positive(),
}


def varying_transformed_data(day_of_year, time, reference):
    # The season and the trend, as in the first model.
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    trend = time / 365.25

    # The basis for the weeks being evaluated, and for the training weeks, whose multiplier
    # enters the ROI calibration.
    multiplier_basis = approximation.basis(time)
    training_basis = approximation.basis(reference.time)

    return {
        "annual": annual,
        "trend": trend,
        "multiplier_basis": multiplier_basis,
        "training_basis": training_basis,
    }


def varying_transformed_parameters(
    media,
    organic_media,
    controls,
    treatments,
    reference,
    outcome_scaling,
    annual,
    trend,
    multiplier_basis,
    training_basis,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    multiplier_coefficients,
    length_scale,
    amplitude,
):
    # The weights turn the fixed basis into a Gaussian process with the sampled length scale
    # and amplitude, and the exponential keeps the multiplier positive.
    weights = approximation.weights(length_scale=length_scale, amplitude=amplitude)
    multiplier = jnp.exp(multiplier_basis @ (weights * multiplier_coefficients))
    training_multiplier = jnp.exp(training_basis @ (weights * multiplier_coefficients))

    # The training weeks' exposure carries their multiplier, so roi stays the return on the
    # training weeks' spend.
    trained = hill_adstock(reference.media, retention, half_saturation) * training_multiplier[:, None]
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)

    # Email keeps a constant coefficient, as in the first model.
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )

    # The baseline follows the trend and the season.
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients

    # The multiplier scales the paid channels' effect week by week, the one change to the
    # first model.
    media_effect = multiplier * (hill_adstock(media, retention, half_saturation) @ coefficient)
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient

    # Expected revenue, and the multiplier for the generated quantities to record.
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu, "multiplier": multiplier}


def varying_log_density(
    outcome,
    mu,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    sigma,
    multiplier_coefficients,
    length_scale,
    amplitude,
):
    # The first model's priors and likelihood.
    target = log_density(
        outcome,
        mu,
        intercept,
        growth,
        curvature,
        annual_coefficients,
        roi,
        retention,
        half_saturation,
        organic_share,
        organic_retention,
        organic_half_saturation,
        control_coefficient,
        treatment_coefficient,
        sigma,
    )

    # The multiplier's priors. The length scale's median is about 245 days, so the effect
    # turns over seasons rather than weeks.
    target += mj.normal(multiplier_coefficients, 0.0, 1.0)
    target += mj.lognormal(length_scale, 5.5, 0.5)
    target += mj.half_normal(amplitude, 0.25)
    return target


def varying_generated_quantities(key, outcome, mu, multiplier, sigma):
    # Simulated revenue for every week, and each observed week's log likelihood.
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)

    # multiplier is one of your names, so it lands in the results under that key.
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
        "multiplier": multiplier,
    }


varying_model = mj.Model(
    parameters=varying_parameters,
    data=mj.Data(worn_data, scaling=worn_scaling),
    transformed_data=varying_transformed_data,
    transformed_parameters=varying_transformed_parameters,
    log_density=varying_log_density,
    generated_quantities=varying_generated_quantities,
    generated_dims={"multiplier": "time"},
)
```

The basis depends only on the dates, so it goes in `transformed_data`. The
multiplier for the weeks being evaluated comes from `time`, while the one in
$\beta_c$ comes from `reference.time`, the training weeks. That's the split
between predictions and definitions that [Scenarios](scenarios) describes.

The weights depend on the length scale and the amplitude, so they go in
`transformed_parameters`. `varying_log_density` adds the three new priors to
the first model's density. `varying_generated_quantities` returns the
multiplier and `generated_dims` labels it by week, so every draw of it lands
in the results.

:::{admonition} Plan on recent weeks
:class: tip

Analyses such as {func}`~mmmjax.optimize_budget` rerun the blocks on the
training weeks, so each week keeps its fitted multiplier, and a plan over all
three years counts the stronger early weeks too. Pass recent weeks as
`spend_periods` and `response_periods` for a plan that reflects today's
effect. Weeks past the data get the process's forecast, which drifts back
toward the training average and widens, so a plan for next year expects the
ads to recover.
:::

## Fitting both versions

The first model, unchanged, fits the same data, so the two versions differ
only in the multiplier.

```{code-cell} ipython3
constant_model = mj.Model(
    parameters=parameters,
    data=mj.Data(worn_data, scaling=worn_scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
settings = {"draws": 1000, "warmup": 1000, "chains": 4, "seed": 7}
```

```{code-cell} ipython3
:tags: [skip-execution]

constant = mj.sample(constant_model, **settings)
varying = mj.sample(varying_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

constant = stored(
    "worn_constant",
    lambda: mj.sample(constant_model, **settings),
    groups=["posterior"],
)
varying = stored(
    "worn_varying",
    lambda: mj.sample(varying_model, **settings),
    groups=["posterior", "sample_stats", "generated_quantities"],
)
```

Check a new kind of parameter before you trust its fit.

```{code-cell} ipython3
print(az.summary(varying, var_names=["length_scale", "amplitude"]))
```

```{code-cell} ipython3
int(varying["sample_stats"]["diverging"].sum())
```

The length scale settles near 269 days, with an 89 percent interval from 180
to 380. The amplitude settles near 0.449, in its prior's upper tail, since the
simulated wear swings by a factor of three. Both have an `r_hat` of 1.00, and
the sampler reports no divergences.

## What the multiplier finds

The model pins the average of $\log \zeta_t$ at zero, so the plot divides the
true wear by its geometric mean to put it on the same footing.

```{code-cell} ipython3
:tags: [hide-input]

import plotnine as pn

true_multiplier = wear / np.exp(np.log(wear).mean())
draws = varying["generated_quantities"]["multiplier"]
band = draws.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).to_series().unstack("quantile")
band.columns = ["lower", "median", "upper"]
band = band.assign(truth=true_multiplier.to_series()).reset_index()
(
    pn.ggplot(band, pn.aes("time"))
    + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), fill="#2a2eec", alpha=0.2)
    + pn.geom_line(pn.aes(y="median"), color="#2a2eec", size=1)
    + pn.geom_line(pn.aes(y="truth"), linetype="dashed", size=0.8)
    + pn.geom_hline(yintercept=1, color="#8c8c8c", size=0.5)
    + pn.scale_x_datetime(date_labels="%b %Y")
    + pn.labs(x="", y="Media multiplier, median and 90% interval")
    + mj.theme_mmmjax()
)
```

The blue line and band are the fitted multiplier and its 90 percent interval,
the dashed line is the true wear, and the gray line marks one, the training
average. The fit finds the overall decline, from about one and a half early in
2022 to about a half in 2024, and its band holds the truth at both ends. It
gets the timing wrong in between. It keeps rising until late 2022, and the
truth runs below its band from the autumn of 2022 to mid-2023 and above it
from December 2023 to mid-2024.

:::{admonition} A multiplier follows anything that changes over time
:class: warning

The multiplier can't tell a change in the ads from any other change over time
that the rest of the model misses, such as a baseline that drifts away from
the quadratic trend. Check its path against what you know, such as creative
launches or experiments, before you read it as the ads' effect.
:::

## What a constant coefficient misses

{func}`~mmmjax.contributions` gives each version's weekly revenue from paid
media. The truth is each channel's simulated contribution times the wear.

```{code-cell} ipython3
constant_effects = mj.contributions(constant_model, constant, quantity="mu", by="time")
varying_effects = mj.contributions(varying_model, varying, quantity="mu", by="time")
```

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

paid_names = list(channels.values())
true_weekly = (wear * paid).sum("channel").rename("truth").to_series().reset_index()
versions = {"Constant coefficient": constant_effects, "Time-varying": varying_effects}
weekly = []
for name, effects in versions.items():
    total = effects["incremental_response"].sel(channel=paid_names).sum("channel")
    quantiles = total.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).to_series().unstack("quantile")
    quantiles.columns = ["lower", "median", "upper"]
    weekly.append(quantiles.assign(version=name).reset_index())
weekly = pd.concat(weekly, ignore_index=True)
colors = {"Constant coefficient": "#fa7c17", "Time-varying": "#2a2eec"}
(
    pn.ggplot(weekly, pn.aes("time"))
    + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper", fill="version"), alpha=0.2)
    + pn.geom_line(pn.aes(y="median", color="version"))
    + pn.geom_line(pn.aes(y="truth"), data=true_weekly, linetype="dashed")
    + pn.scale_color_manual(values=colors)
    + pn.scale_fill_manual(values=colors)
    + pn.scale_x_datetime(date_labels="%b %Y")
    + pn.scale_y_continuous(labels=lambda values: [f"${value / 1e3:,.0f}K" for value in values])
    + pn.labs(x="", y="Weekly revenue from paid media", color="", fill="")
    + mj.theme_mmmjax()
    + pn.theme(legend_position="top")
)
```

Under the constant coefficient, paid media's revenue runs too low in the 2022
flights and too high in the 2024 ones. The time-varying version comes closer
but still runs low, most in the early 2022 flights and from late 2023 on,
where its multiplier falls below the true wear. The bands are 90 percent
intervals.

Summed by year, each version's median revenue from paid media sits next to
the truth, in millions of dollars.

```{code-cell} ipython3
:tags: [hide-input]

yearly = {}
for name, effects in versions.items():
    total = effects["incremental_response"].sel(channel=paid_names).sum("channel")
    yearly[name] = total.groupby("time.year").sum().median(("chain", "draw")).to_series()
yearly["Truth"] = (wear * paid).sum("channel").groupby("time.year").sum().to_series()
(pd.DataFrame(yearly) / 1e6).round(2)
```

The constant coefficient gives every week the same return, so its yearly
credit follows the media alone. It credits paid media with \$3.96 million in
2022, when the truth is \$6.53 million, and with \$3.27 million in 2024
against \$2.31 million, about 40 percent too much. The time-varying version
comes closer in every year, though it too falls short in 2024, at
\$1.64 million. A plan built on the constant model would count on 2024's ads
to work as well as 2022's.
