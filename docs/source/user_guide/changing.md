---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Changing the model

The model is a set of functions you wrote, so changing it means editing those
functions. Nothing else in the workflow has to move. This page makes three
changes to the ten-channel brand model from [A first model](first_model), to
its response curve, its likelihood, and its trend, and then compares the four
versions.

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
```

Each version below is fitted with the same settings as the first model.

```{code-cell} ipython3
settings = {"draws": 1000, "warmup": 1000, "chains": 4, "seed": 7}
```

## The response curve

The first model fixes the Hill curve's slope at one, which makes every
channel's response concave from the first impression. Letting the slope $s$ in
$\operatorname{Hill}(u;\, \kappa, s) = u^{s} / (u^{s} + \kappa^{s})$ rise above
one allows an S-shaped curve, where the first impressions do little until
exposure builds. The new model gives each channel its own slope $s_c \geq 1$,
so the transformed exposure becomes

$$
\begin{aligned}
h_{tc} &= \operatorname{Hill}\Big(\operatorname{Adstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c\big);\, \kappa_c, s_c\Big), \\
s_c - 1 &\sim \operatorname{HalfNormal}(1),
\end{aligned}
$$

with the model equation, the ROI coefficient, and the other priors of [A first
model](first_model). The prior keeps each slope at one or above and lets the
data move it up.

The slope is the only new symbol, and it becomes `slope`, one of your names,
declared in `parameters` with a lower bound of one to match $s_c \geq 1$. Its
prior becomes one more `target +=` line in `log_density`.

`slope` is one of your names, declared in `parameters`, so every block that
uses it asks for it by that name.

```{code-cell} ipython3
shaped_parameters = parameters | {"slope": mj.LowerBound(1.0, dims="channel")}


def shaped_hill_adstock(media, retention, half_saturation, slope):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=slope)
    return saturated


def shaped_transformed_parameters(
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
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    slope,
):
    trained = shaped_hill_adstock(reference.media, retention, half_saturation, slope)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = shaped_hill_adstock(media, retention, half_saturation, slope) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}


def shaped_log_density(
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
    slope,
):
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
    target += mj.half_normal(slope - 1.0, 1.0)
    return target


shaped_model = mj.Model(
    parameters=shaped_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=shaped_transformed_parameters,
    log_density=shaped_log_density,
    generated_quantities=generated_quantities,
)
```

`shaped_hill_adstock` is a plain helper, not a block, so it takes the slope as
an ordinary fourth argument. Both calls pass it, because the ROI coefficient
reads the same curve on the training exposure. Email's two calls still use
`hill_adstock`, so its curve keeps a slope of one. Blocks are ordinary Python
functions, so the new density calls the first model's density and adds one
prior term.

:::{admonition} Keep the slope at one or above
:class: warning

A slope below one makes the curve infinitely steep at zero exposure. A flighted
channel spends weeks at zero, where the sampler's gradient wouldn't be finite,
so the declaration bounds the slope at one, even though that rules out Branded
search's true 0.9.
:::

```{code-cell} ipython3
:tags: [skip-execution]

shaped = mj.sample(shaped_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import stored

shaped = stored("shaped", lambda: mj.sample(shaped_model, **settings), groups=["posterior", "log_likelihood"])
```

The cell below puts each channel's 5th, 50th, and 95th slope percentiles next
to the slope the simulation used.

```{code-cell} ipython3
slopes = shaped["posterior"]["slope"].quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).T.to_pandas()
slopes["truth"] = brand.truth["slope"].sel(channel=list(channels)).values
slopes.round(2)
```

For comparison, draws from the prior alone give these three percentiles.

```{code-cell} ipython3
import jax
import numpy as np

prior_slope = 1.0 + mj.half_normal_rng(jax.random.key(0), 1.0, sample_shape=(4000,))
np.quantile(prior_slope, [0.05, 0.5, 0.95]).round(2)
```

The posterior barely moves from the prior. Every interval starts just above
the bound, and every median falls between 1.26 and 1.78, where the prior's is
1.68. Linear TV and Streaming lean furthest toward one, with medians of 1.26
and 1.31 against true slopes of 1.3 and 1.1. Meta, Display, and the two search
channels run every week, so the data never shows the low end of their curves,
where the slope matters most. Their medians stay near the prior's, even though
their true slopes are 1.2, 1.0, 0.9, and 1.0.

{func}`~mmmjax.plot_saturation` shows what these slopes do to each channel's
curve. It passes every draw to `mj.hill_saturation`, whose `half_saturation`
and `slope` arguments take the posterior's variables of the same names.

```{code-cell} ipython3
mj.plot_saturation(shaped, mj.hill_saturation, max_input=3.0)
```

Each panel shows how much of its greatest effect a channel reaches at each
level of carried media. Meta, Display, and the two search channels start flat
and bend upward, the S shape of slopes that stayed near the prior's. Linear TV
and Streaming climb from the first impression, much as every curve does when
the slope is fixed at one.

## The likelihood

A Student-t likelihood has heavier tails than the normal, so a few unusual
weeks pull less on the fit. Its degrees of freedom $\nu$ set how heavy the
tails are, and the tails approach the normal's as $\nu$ grows. A gamma prior
with shape two and rate 0.1 leaves room for heavy tails and nearly normal ones
alike. Only the noise changes,

$$
\varepsilon_t \sim \operatorname{StudentT}(\nu, 0, \sigma), \qquad \nu \sim \operatorname{Gamma}(2, 0.1),
$$

where the gamma distribution takes a shape and a rate, and the model equation
and every other prior stay as in [A first model](first_model). Since $y_t$ is
$\mu_t$ plus that noise, the likelihood becomes $y_t \sim
\operatorname{StudentT}(\nu, \mu_t, \sigma)$. The degrees of freedom are the
only new symbol, and $\nu$ becomes `degrees_of_freedom`, one of your names,
declared in `parameters` as positive.

The first model's density ends in the normal likelihood, so the new density
can't call it. It repeats the first model's thirteen prior terms and then adds
one `target +=` line for the gamma prior and one for the Student-t likelihood.

:::{admonition} Change the generated quantities too
:class: warning

The generated quantities simulate from the likelihood and score it, so their
two lines switch to the Student-t as well. Left on the normal, they'd hand the
comparison below the wrong pointwise log likelihood. The keys `"predictive"`
and `"log_likelihood"` stay as they are, since they're supplied names that tell
mmmJAX where to store the draws.
:::

```{code-cell} ipython3
student_parameters = parameters | {"degrees_of_freedom": mj.Positive()}


def student_log_density(
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
    degrees_of_freedom,
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)
    target += mj.half_normal(sigma, 1.0)
    target += mj.gamma(degrees_of_freedom, 2.0, 0.1)
    target += mj.student_t(outcome, degrees_of_freedom, mu, sigma)
    return target


def student_generated_quantities(key, outcome, mu, sigma, degrees_of_freedom):
    prediction = mj.student_t_rng(key, degrees_of_freedom, mu, sigma)
    pointwise = mj.student_t_logpdf(outcome, degrees_of_freedom, mu, sigma)
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
    }


student_model = mj.Model(
    parameters=student_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=student_log_density,
    generated_quantities=student_generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

student = mj.sample(student_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

student = stored("student", lambda: mj.sample(student_model, **settings), groups=["posterior", "log_likelihood"])
```

```{code-cell} ipython3
student["posterior"]["degrees_of_freedom"].quantile([0.05, 0.5, 0.95]).to_series().round(1)
```

The degrees of freedom have a median of 18.5 and a 90 percent interval from
7.2 to 48.3, much like the prior's spread, so the data neither asks for heavy
tails nor rules them out. The simulation's noise has no heavy tails, which
leaves no outlying weeks for the Student-t to discount.

## The trend

The first model's trend is a quadratic in time, which can bend only once. The
brand's baseline drifts up and down without a fixed shape, and [Sampling and
diagnostics](sampling) found residuals that run in streaks the model doesn't
reproduce. A Gaussian process lets the trend follow any smooth path. Its
length scale $\ell$ sets how quickly the path can turn, and its amplitude
$\eta$ sets how far it can move. The Hilbert-space approximation (HSGP) writes
the process as a fixed set of sine functions times coefficients, so the
sampler works with ordinary parameters. The trend $f_t$ takes the place of
$\delta_1 \tau_t + \delta_2 \tau_t^2$,

$$
\begin{aligned}
y_t &= \alpha + f_t + \sum_{k=1}^{2} \Big( a_k \sin\frac{2\pi k d_t}{365.25} + b_k \cos\frac{2\pi k d_t}{365.25} \Big) \\
&\quad + \sum_{c=1}^{10} \beta_c h_{tc} + \sum_{o} \lambda_o h_{to}
+ \sum_{j=1}^{2} \gamma_j p_{tj} + \sum_{i=1}^{2} \theta_i g_{ti} + \varepsilon_t, \\
f_t &= \sum_{m=1}^{M} \phi_m(e_t) \sqrt{S(\omega_m)}\, z_m, \qquad
S(\omega) = \eta^2 \sqrt{2\pi}\, \ell \exp\big(-\ell^2 \omega^2 / 2\big), \\
z_m &\sim \operatorname{Normal}(0, 1), \qquad
\ell \sim \operatorname{LogNormal}(5, 0.5), \qquad
\eta \sim \operatorname{HalfNormal}(0.5),
\end{aligned}
$$

where $e_t$ counts the days from the first week to week $t$, $\phi_m$ is the
$m$th of $M$ basis functions, $\omega_m$ is its frequency, and $S$ is the
spectral density of the squared exponential covariance. Everything else stays
as in [A first model](first_model). The length scale is in days and the
amplitude in standard deviations of revenue, so the prior on $\ell$ favors
paths that turn over a few months and the prior on $\eta$ keeps the trend's
swings within revenue's usual spread.

The quadratic trend's $\delta_1$, $\delta_2$, and $\tau_t$ leave the model, so
`growth` and `curvature` drop out of `hsgp_parameters` and
`hsgp_transformed_data` no longer returns a `trend`. Each new symbol takes a
name in the code.

| Symbol | Name in the code | Where the name comes from |
| --- | --- | --- |
| $z_m$ | `trend_coefficients` | Declared in `parameters`, one entry per basis function |
| $\ell$ | `length_scale` | Declared in `parameters` as positive |
| $\eta$ | `amplitude` | Declared in `parameters` as positive |
| $\phi_m(e_t)$ | The columns of `trend_basis` | Returned by `transformed_data`, which computes them from `time` |
| $\sqrt{S(\omega_m)}$ | `weights` | Local to `transformed_parameters` |
| $f_t$ | `trend` | Local to `transformed_parameters`, which adds it into `baseline` |
| $M$, $\omega_m$, and $\phi_m$ | `approximation` | An object you build before the model |

{func}`~mmmjax.prepare_hsgp` builds that object once, outside the model, from
the training weeks.

```{code-cell} ipython3
approximation = mj.prepare_hsgp(
    data.time_positions,
    length_scale_range=(60.0, 400.0),
    covariance="expquad",
    center_columns=True,
)
approximation.center, approximation.boundary, approximation.n_basis
```

`data.time_positions` holds the days since the first week, and
`length_scale_range` names the length scales the trend should be able to use,
from 60 to 400 days. The basis centers on day 542.5, the middle of the
training weeks, and reaches 1,280 days to either side. A process with a long
length scale needs that room past the data, or it bends toward zero at the
edges. The basis's 37 functions are enough for the shortest length scale in
the range. `center_columns=True` subtracts each function's mean over the
training weeks, so the trend averages zero there and the intercept keeps the
level.

```{code-cell} ipython3
hsgp_parameters = {
    "intercept": mj.Real(),
    "trend_coefficients": mj.Real(approximation.n_basis),
    "length_scale": mj.Positive(),
    "amplitude": mj.Positive(),
    "annual_coefficients": mj.Real(4),
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_half_saturation": mj.Positive(dims="organic_channel"),
    "control_coefficient": mj.Real(dims="control"),
    "treatment_coefficient": mj.Real(dims="treatment"),
    "sigma": mj.Positive(),
}


def hsgp_transformed_data(day_of_year, time):
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    trend_basis = approximation.basis(time)
    return {"annual": annual, "trend_basis": trend_basis}


def hsgp_transformed_parameters(
    media,
    organic_media,
    controls,
    treatments,
    reference,
    outcome_scaling,
    annual,
    trend_basis,
    intercept,
    trend_coefficients,
    length_scale,
    amplitude,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
):
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    weights = approximation.weights(length_scale=length_scale, amplitude=amplitude)
    trend = trend_basis @ (weights * trend_coefficients)
    baseline = intercept + trend + annual @ annual_coefficients
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}


def hsgp_log_density(
    outcome,
    mu,
    intercept,
    trend_coefficients,
    length_scale,
    amplitude,
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
):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(trend_coefficients, 0.0, 1.0)
    target += mj.lognormal(length_scale, 5.0, 0.5)
    target += mj.half_normal(amplitude, 0.5)
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target


hsgp_model = mj.Model(
    parameters=hsgp_parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=hsgp_transformed_data,
    transformed_parameters=hsgp_transformed_parameters,
    log_density=hsgp_log_density,
    generated_quantities=generated_quantities,
)
```

The basis depends only on the dates, so it goes in `transformed_data`, which
runs once for each dataset. It reads `time`, so a scenario's weeks get their
own rows, while the center, the boundary, and the column means stay those of
the training weeks. That's the split between predictions and definitions
that [Scenarios](scenarios) describes. The weights depend on the length scale
and the amplitude, so they go in `transformed_parameters`. The coefficients
$z_m$ have standard normal priors and the weights scale them, which keeps the
posterior's geometry simple for the sampler. The basis functions have no data
axis, so `mj.Real(approximation.n_basis)` gives the coefficients a plain shape
of 37. An axis name of your own would work too, as
[Data and scaling](data.md#supplied-names) explains.

```{code-cell} ipython3
:tags: [skip-execution]

hsgp = mj.sample(hsgp_model, **settings)
```

```{code-cell} ipython3
:tags: [remove-cell]

hsgp = stored(
    "hsgp",
    lambda: mj.sample(hsgp_model, **settings),
    groups=["posterior", "log_likelihood", "posterior_predictive", "observed_data"],
)
```

Check a new kind of parameter before you trust its fit.

```{code-cell} ipython3
import arviz as az

print(az.summary(hsgp, var_names=["length_scale", "amplitude"]))
```

The length scale settles near 217 days, with an 89 percent interval from 130
to 330, and the amplitude near 0.497. Both have an `r_hat` of 1.00, with a bulk
ESS of 1,388 for the length scale and 2,014 for the amplitude.

The new trend was meant to absorb the streaks in the first model's residuals,
and {func}`~mmmjax.plot_ppc_tstat` checks them the way [Sampling and
diagnostics](sampling) did.

```{code-cell} ipython3
mj.plot_ppc_tstat(hsgp_model, hsgp, quantity="mu")
plt.show()
```

The black curve of observed residual autocorrelation now overlaps the blue
curve of the replicated draws, with a p of 0.35, so the streaks the first
model left no longer stand out. The standard deviation and the largest week
stay near the middle of their draws, with p values of 0.52 and 0.72.

## Comparing the versions

Leave-one-out cross-validation estimates how well each version predicts a
week it hasn't seen. It works from the pointwise log likelihood that every
version's generated quantities return.

```{code-cell} ipython3
versions = {"first": results, "free slope": shaped, "student-t": student, "HSGP trend": hsgp}
comparison = az.compare(versions, round_to=1)
print(comparison[["elpd", "elpd_diff", "dse"]])
```

`elpd` is the expected log predictive density, where higher is better, and
`dse` is the standard error of each version's difference from the best one.
The HSGP trend comes out ahead of the first model by 14.3, more than twice
the 5.8 standard error of that difference. The Student-t and free-slope
versions trail the HSGP trend by 14.2 and 16.7, one just ahead of the first
model and one behind it, so neither changes the fit much. That matches how the
data was made, with noise that has no heavy tails, slopes close to one, and a
baseline that drifts in a way no quadratic can follow. None of this says
which model is true, and a version that predicts better need not credit the
channels better.

{func}`~mmmjax.media_metrics` computes each version's returns, and
{func}`~mmmjax.plot_media_metrics` sets them side by side for the ten paid
channels. Email has no spending, so it has a
contribution but no ROI, and the plot leaves it out.

```{code-cell} ipython3
models = {
    "first": model,
    "free slope": shaped_model,
    "student-t": student_model,
    "HSGP trend": hsgp_model,
}
returns = {name: mj.media_metrics(models[name], fit, quantity="mu") for name, fit in versions.items()}
mj.plot_media_metrics(returns)
```

Given a mapping of labeled results, the plot gives each version its own color
and labels each bar with its mean ROI. With four versions it grows wide enough
to scroll sideways. The Student-t version stays near the first model on
every channel, and freeing the slope lowers nine of the ten means, Streaming's
the most. The HSGP trend moves the returns the most. It lifts Linear TV from
2.79 to 3.77 and YouTube from 4.06 to 4.77, and it cuts Snapchat from 3.98 to
3.13 along with its long upper tail. The simulation keeps the true returns in
`brand.truth`.

```{code-cell} ipython3
true_roi = brand.truth["roi"].sel(paid_channel=["linear_tv", "streaming", "youtube", "snapchat"])
true_roi.to_series().round(2)
```

No version gets every channel right. The HSGP trend brings Linear TV and
YouTube closer to their true 4.20 and 6.59, although YouTube stays well short.
It moves Streaming from 4.04 to 3.13, past its true 3.48, and takes Snapchat
further below its true 5.09. The version that predicts best moves
revenue among the flighted channels without bringing each one closer to the
truth.

Snapchat, Streaming, and Linear TV share one campaign calendar. As
[Data and scaling](data.md) shows, the data can't split revenue between
channels whose exposure moves together, so a new baseline can shift credit
among them.

The baseline trades credit with the treatments too. The price climbs with the
trend, as [A first model](first_model) shows, so the two compete for the same
rise in revenue. {func}`~mmmjax.contributions` reports each treatment's share of
revenue against its lowest observed level. The cell below prints the 5th,
50th, and 95th percentiles of the two shares, in percent, for the first model
and the HSGP trend.

```{code-cell} ipython3
import pandas as pd

treatments = ["price", "promotion"]
treatment_shares = {}
for name in ["first", "HSGP trend"]:
    decomposition = mj.contributions(models[name], versions[name], quantity="mu", channels=treatments)
    share = 100 * decomposition["contribution_share"]
    treatment_shares[name] = share.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).T.to_pandas()
pd.concat(treatment_shares).round(1)
```

The first model's median gives the price -2.5 percent of revenue, so at the
median the higher prices cost revenue, as they did in the simulation. With the
HSGP trend taking up the growth, the price's median falls further, to -3.3
percent, and the promotions' median falls from 3.0 to 2.6 percent. Both price intervals still
run from a loss to a gain, so neither version settles what the price did.

A better fit is a reason to keep the HSGP trend, but the channel answers still
lean on the ROI prior, which [Priors](priors) examines.
