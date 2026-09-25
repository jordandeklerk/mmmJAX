---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Plotting

mmmJAX draws the plots a marketing mix analysis keeps coming back to, from
checking the sampler to deciding where the next dollar should go. Each
function reads the results of a fit or the output of an analysis function, so
no plot refits the model. Plots change as channels are added, so this
page fits a brand with ten of them.

The functions are there for convenience, and a plot can start anywhere else.
Results are ordinary xarray trees and every analysis output is an ordinary
xarray Dataset, so ArviZ, plotnine, matplotlib, or any other plotting library
can draw from them directly. [Customizing plots](custom_plots) changes what
these functions draw and builds new plots from the same outputs.

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
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
```

## A ten-channel brand

{func}`~mmmjax.simulate_data` with its default settings gives three years of
weekly revenue for a brand with ten paid channels, from Meta and TikTok to
linear TV and display. Demand, price, promotions, holidays, and owned email
also move revenue, and the model takes all five as controls.

The model extends [A first model](first_model). Its baseline gains a quadratic
trend in $\tau_t$, the time since the first week as a share of the whole
window, and the Fourier terms of [Changing the model](changing). Each
channel's coefficient follows from its return on investment $r_c$, which is
easier to hold a belief about than a coefficient on scaled media, so the model
equation becomes

$$
\begin{aligned}
y_t &= \alpha + \delta_1 \tau_t + \delta_2 \tau_t^2
+ \sum_{k=1}^{2} \Big( a_k \sin\frac{2\pi k d_t}{365.25} + b_k \cos\frac{2\pi k d_t}{365.25} \Big) \\
&\quad + \sum_{c=1}^{10} \beta_c h_{tc} + \sum_{j=1}^{5} \gamma_j p_{tj} + \varepsilon_t, \\
h_{tc} &= \operatorname{HillAdstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big), \qquad
\beta_c = \frac{r_c S_c}{s_r \sum_{t} h_{tc}}, \\
r_c &\sim \operatorname{LogNormal}(1, 0.6), \qquad
\delta_1, \delta_2 \sim \operatorname{Normal}(0, 1), \qquad
a_k, b_k \sim \operatorname{Normal}(0, 0.5),
\end{aligned}
$$

with everything else as in [A first model](first_model). The five
standardized controls are $p_{tj}$, $S_c$ is channel $c$'s spending over the
training weeks, and $s_r$ is the revenue scale, so $\beta_c$ makes the
channel's contribution over those weeks $r_c S_c$ dollars. The blocks compute
$\beta_c$ with {func}`~mmmjax.roi_coefficient`.

```{code-cell} ipython3
:tags: [hide-input]

import mmmjax as mj

brand = mj.simulate_data(seed=7, groups=None)
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
    controls=["demand", "price", "promotion", "holiday", "email_sends"],
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)

parameters = {
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    "annual_coefficients": mj.Real(dims="annual_mode"),
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "control_coefficient": mj.Real(dims="control"),
    "sigma": mj.Positive(),
}

priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "growth": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "curvature": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "annual_coefficients": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "roi": mj.Prior(mj.lognormal, location=1.0, scale=0.6),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "sigma": mj.Prior(mj.half_normal, scale=1.0),
}


def transformed_data(day_of_year, time, reference):
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)
    trend = time / reference.time.max()
    return {"annual": annual, "trend": trend}


def hill_adstock(media, retention, half_saturation):
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated


def transformed_parameters(
    media,
    controls,
    annual,
    trend,
    reference,
    outcome_scaling,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    control_coefficient,
):
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    mu = baseline + media_effect + controls @ control_coefficient
    return {"mu": mu}


def log_density(
    outcome,
    mu,
    intercept,
    growth,
    curvature,
    annual_coefficients,
    roi,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
):
    target = priors["intercept"](intercept)
    target += priors["growth"](growth)
    target += priors["curvature"](curvature)
    target += priors["annual_coefficients"](annual_coefficients)
    target += priors["roi"](roi)
    target += priors["retention"](retention)
    target += priors["half_saturation"](half_saturation)
    target += priors["control_coefficient"](control_coefficient)
    target += priors["sigma"](sigma)
    target += mj.normal(outcome, mu, sigma)
    return target


def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
    }


model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
    coords={"annual_mode": ["sin_1", "sin_2", "cos_1", "cos_2"]},
)
```

The fit uses the first model's sampler settings, and
{func}`~mmmjax.sample_prior` draws from the same priors for the comparisons
below.

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import brand_results, stored

results = brand_results(model)
prior_results = stored("brand_prior", lambda: mj.sample_prior(model, priors, draws=500, seed=0), groups=["prior"])
```

## Convergence

{func}`~mmmjax.plot_rhat` draws every parameter's R-hat at once.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Parameters run down the side, and each point is one element of a parameter,
such as one channel's return. R-hat compares the spread between the chains
with the spread within them, so a value near one means every chain found the
same distribution. A value past ArviZ's limit of 1.01 would sit right of the
dotted line and turn orange, and it couldn't be trusted until a longer run or a
better parameterization brought it back. The subtitle confirms that all 43
values of the brand fit sit below the limit, and `var_names` limits the plot to
some parameters when a model has many.

{func}`~mmmjax.plot_rank` looks at the chains more closely.

```{code-cell} ipython3
mj.plot_rank(results)
plt.show()
```

Each panel follows one element, and each line one chain's ranks among all the
draws. Chains that explore the same distribution spread their ranks evenly, so
their lines stay near zero, and a line that bows away shows a chain that
lingered in one part of the distribution. The p-value above each panel tests
for it, as [Sampling and diagnostics](sampling) explains. The brand has more
parameters than ArviZ draws in one figure, so the plot keeps the 12 with the
highest R-hat, the likeliest to have mixed poorly, and says so in its title.
Chain 1 bows away furthest, for Snapchat's return.

{func}`~mmmjax.plot_trace_dist` makes the same choice with six rows.

```{code-cell} ipython3
mj.plot_trace_dist(results)
plt.show()
```

Each row pairs an element's density on the left with its draws in sampling
order on the right, with a line style for each chain. Healthy chains overlap in
the densities and look like flat, even bands on the right, while a trend, a
jump, or one chain sitting apart from the rest signals trouble. Even these six,
the worst by R-hat, look healthy.

## Fit

{func}`~mmmjax.plot_ppc_dist` compares the distribution of the observed
revenue with that of the posterior predictive draws.

```{code-cell} ipython3
mj.plot_ppc_dist(model, results)
plt.show()
```

Each blue curve is the distribution of one draw's predicted weeks and the
black curve that of the observed weeks. A model that reproduces the data keeps
the black curve inside the bundle of blue ones, as it does here down to the
second hump of busy weeks. A black curve outside the bundle, such as a longer
tail, would point to a feature of the data the model misses. With the output of
{func}`~mmmjax.sample_prior` and `group="prior"`, the same plot checks the
priors before any fit, as [Priors](priors) shows.

{func}`~mmmjax.plot_fit` keeps the weeks in order.

```{code-cell} ipython3
mj.plot_fit(model, results)
```

The black line is observed revenue, and the blue line and band are the mean
and 89 percent interval of the predictive draws, all in dollars because the
plot undoes the outcome scaling first. Weeks where the black line leaves the
band are ones the model can't explain, and a run of them would point to
something missing, such as an unmodeled promotion. The subtitle gives an $R^2$
of 0.91, a weighted mean absolute percentage error of 4.1 percent, and the
share of weeks inside the band, 92 percent here. A calibrated model keeps that
share close to the interval's probability.

:::{admonition} Intervals and point estimates
:class: tip

Every plot follows ArviZ's settings, 89 percent intervals around the mean
unless changed. `ci_prob` changes the interval for one call, and
[Customizing plots](custom_plots) shows how to change both for a block of
code.
:::

`show_baseline=True` adds the baseline, the revenue the model expects with
every channel removed, which {func}`~mmmjax.contributions` computes from the
same draws. `quantity` names the expected revenue the blocks return.

```{code-cell} ipython3
mj.plot_fit(model, results, show_baseline=True, quantity="mu")
```

The gap between the orange baseline and the blue predictions is what the ten
channels add each week. It opens widest during the flights, and the baseline
itself carries the trend and the seasons.

{func}`~mmmjax.plot_residuals` shows what the predictions leave over.

```{code-cell} ipython3
mj.plot_residuals(model, results)
```

Each week's residual is the observed revenue minus the prediction, and the
band is its 89 percent interval across draws. A good fit leaves residuals that
wander around zero with no pattern, while a trend, a repeating season, or a
long stretch above or below zero points to structure the model misses. Here
the line stays around zero without a trend or a season.

{func}`~mmmjax.plot_ppc_tstat` turns checks like these into numbers.

```{code-cell} ipython3
mj.plot_ppc_tstat(model, results, quantity="mu")
plt.show()
```

Each panel shows one statistic computed on every predictive draw, and the
black dot marks its value for the observed revenue. The p in each title is the
share of draws whose statistic reaches the observed one, so a value near zero
or one flags a feature the model rarely reproduces. The standard deviation and
the largest week sit near the middle of their draws.

Residual autocorrelation measures how closely each week's residual follows the
last, with the residuals taken from each draw's expected revenue, which
`quantity` names. The observed residuals then change from draw to draw, so a
black curve takes the place of the dot. The replicated values center on zero,
as independent noise should, while the observed ones center near 0.2, and the
title's p rounds to zero. The residuals run in short streaks the model doesn't
reproduce, a sign that something carrying over from week to week is missing,
and a pattern too faint to see in the residual plot above. `statistics` takes
other names, quantiles such as `0.9`, or any function from a series of weekly
revenue to a number.

## Priors and posteriors

{func}`~mmmjax.plot_prior_posterior` overlays each parameter's prior on its
posterior.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results)
plt.show()
```

Each panel draws a parameter's prior in blue and its posterior in orange. The
further the posterior moves from the prior, or the narrower it gets, the more
the data taught the model about that parameter, and a posterior that looks
like its prior leaves the prior to carry the answer. The brand has more
parameters than one figure holds, so the plot keeps the 12 of 43 that the data
narrowed least, and `var_names` picks others.

Every one of the twelve is a media parameter whose posterior sits on its
prior. Branded search appears three times, with its retention, half-saturation
point, and return, so what the model says about that channel comes mostly from
the priors.

{func}`~mmmjax.psense_summary` measures how much each posterior depends on the
priors and how much on the data.

```{code-cell} ipython3
:tags: [wide-table]

mj.psense_summary(results, priors=priors, var_names=["roi"])
```

Raising the priors or the likelihood to a power a little above or below one
reweights the draws without refitting, and each value measures how far that
moves a channel's return. `priors` is the mapping the model's `log_density`
uses, since the results hold no log prior of their own. The prior column
passes ArviZ's threshold of 0.05 for all ten channels, and for nine the
likelihood column does too, which ArviZ reads as a possible conflict between
prior and data. The columns after the diagnosis scale one prior at a time, and
for every channel the ROI prior moves the return most. Values near the
threshold can be noise, so the ranking says more than the labels.

{func}`~mmmjax.plot_psense` shows which way each posterior moves.

```{code-cell} ipython3
mj.plot_psense(results, priors=priors)
plt.show()
```

Each row draws an element's distribution with the priors or the likelihood
raised to 0.8, 1, and 1.25, the priors on the left and the likelihood on the
right, and the lines below give the point estimate and 89 percent interval at
each power. The plot keeps the six elements most sensitive to the priors, all
returns and half-saturation points. For Display's and Snapchat's returns, a
stronger prior pulls the interval down while a stronger likelihood pushes it
up, the mark of a prior that sits below what the data suggest. The upper end
of Display's interval falls from about 10 to about 7 as the prior strengthens.

## Contributions

{func}`~mmmjax.plot_contributions` breaks revenue down into the baseline and
each channel from the output of {func}`~mmmjax.contributions`, summing its
weeks first.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
mj.plot_contributions(effects)
```

Each bar starts where the one before it ends, so the bars walk from the
baseline through every channel to all of the revenue, and each label gives a
share and its total in dollars. The baseline holds 79.9 percent, and Meta, the
largest channel, holds 3.9 percent, about \$2.14 million over the three years.
The order ranks what each channel delivered. A large budget can buy a large
share at a middling return, so the returns below put the channels in another
order. The shares
are ratios of posterior means, so they add up, and a last gray bar would carry
any difference left by channels whose effects interact.

`by="time"` keeps the weeks instead.

```{code-cell} ipython3
mj.plot_contributions(effects, by="time")
```

The gray area is the baseline, each channel stacks on top of it, and the black
line is total revenue. The axis starts near the lowest weekly baseline, so the
channels stay visible above a much larger baseline. Linear TV and streaming
arrive in flights, while generic search and display run every week.
`include_baseline=False` drops the baseline and stacks the channels under a
line that traces what removing all ten at once would cost.

## Returns

The return plots read the output of {func}`~mmmjax.media_metrics`, and
{func}`~mmmjax.plot_media_metrics` draws the return on investment by default.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

Each bar is a channel's mean return on a dollar and the error bar its 89
percent interval. The dashed line marks break-even, where a channel returns
exactly what it costs, so a bar above it has paid for itself. Channels run from
the most spending to the least, from Meta at \$3.87 to Snapchat at \$6.69.
Snapchat's return is the highest, but its interval is also by far the widest,
since the smallest budget leaves the model the least to learn from.

:::{admonition} Comparing results
:class: tip

A mapping of labeled results, such as
`{"Prior": prior_returns, "Posterior": returns}`, draws each channel's bars
side by side in one color per result. `prior_returns` comes from
{func}`~mmmjax.media_metrics` with `group="prior"` on the output of
{func}`~mmmjax.sample_prior`.
:::

`metric` switches the bars to any other metric with draws.

```{code-cell} ipython3
mj.plot_media_metrics(returns, metric="marginal_roi")
```

The marginal return is the return on one more percent of spending. The
average return says how well past spending paid off, and the marginal return
says what the next dollar would earn, which is what a budget decision needs.
Here it runs from \$1.23 for generic search to \$3.96 for Snapchat, below every
channel's average return, since each channel's curve flattens as it spends. A
channel with a high average return but a low marginal one has already spent
far into its saturation.

{func}`~mmmjax.plot_psense` with `metrics` asks how much of that ranking comes
from the priors, and `kind="quantities"` follows summaries across the powers.

```{code-cell} ipython3
mj.plot_psense(
    results,
    priors=priors,
    metrics=returns,
    var_names=["marginal_roi"],
    kind="quantities",
    coords={"channel": ["Snapchat", "Streaming", "Generic search"]},
)
plt.show()
```

Each panel follows the mean or standard deviation of a channel's marginal
return as the priors, in blue, or the likelihood, in orange, are raised to a
power, and the dashed lines mark two Monte Carlo standard errors around the
unscaled value. Snapchat's mean falls from \$4.73 to \$3.37 as the priors
strengthen and rises to \$4.27 as the likelihood does, so the priors hold its
lead back and the data widen it. Streaming's mean barely moves with the
priors, and generic search stays the lowest of the three at every power.

{func}`~mmmjax.plot_roi_bubbles` sets each channel's average return against
its marginal one.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

Each bubble places a channel at its average return across and its marginal
return up, and its area is proportional to the channel's spending. The dashed
lines mark break-even on both axes. A channel to the right has returned more
than it cost so far, and a channel above would still return more than its next
dollar costs. Every channel here sits above and to the right, so what matters
is how far up each one sits. Streaming and Snapchat sit highest, while generic
search, the second largest budget, has the lowest marginal return. Moving
money from low bubbles to high ones is what the budget below does.

`metric="effectiveness"` puts the incremental revenue per impression on the
vertical axis instead.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns, metric="effectiveness")
```

Effectiveness asks how much each impression does and the return asks how much
each dollar does, so the two differ by what an impression costs. A channel high
on both works well and buys its impressions at a fair price, while one with a
middling return and low effectiveness buys cheap impressions that each do
little. Streaming and branded search earn the most on each impression, and
display pairs a middling return with the lowest effectiveness. The horizontal
line is gone because only a return has a break-even point.

{func}`~mmmjax.plot_spend_vs_contribution` compares each channel's share of
the spending with its share of the incremental revenue.

```{code-cell} ipython3
mj.plot_spend_vs_contribution(returns)
```

Each channel gets two bars side by side, a hatched one for its share of the
spending and a solid one for its share of the incremental revenue, with its
return above. When the solid bar stands taller, the channel earns more than
its share, which means a return above the average of all ten. When it falls
short, the channel takes more of the budget than it gives back. Streaming's
solid bar stands well above its hatched one, while generic search's falls
short, and the budget below moves more money between these two than between
any others.

## Media transformations

{func}`~mmmjax.plot_adstock` passes one unit of exposure through the model's
own adstock function for every draw and gives each channel a panel, and
`prior` adds the draws of {func}`~mmmjax.sample_prior` for comparison.

```{code-cell} ipython3
mj.plot_adstock(
    results,
    mj.geometric_adstock,
    max_lag=8,
    parameters={"alpha": "retention"},
    prior=prior_results,
)
```

:::{admonition} How the parameters reach the function
:class: tip

The function receives each draw of a parameter through the argument of the
same name, the way the model's blocks receive their inputs. `parameters` maps
an argument to a parameter with another name, as `alpha` to the model's
`retention` here, or fixes it at a number, as the slope below.
:::

Each panel shows the share of an exposure's effect that lands in each later
week, with the posterior in blue and the prior in orange. A curve that drops
fast means the effect is spent within a week or two, and a slow decline means
it carries over for weeks. For most channels the posterior band sits on the
prior's, so the data say little about how long their effects last. YouTube's
posterior moves the most, toward effects that fade faster than its prior
expects. Where the data barely move a prior, the prior carries the answer and
deserves the most thought. Without `prior`, `combine=True` draws the first five
channels together in one panel so their decay compares directly.

{func}`~mmmjax.plot_saturation` does the same for the saturation curve.

```{code-cell} ipython3
mj.plot_saturation(
    results,
    mj.hill_saturation,
    max_input=3.0,
    parameters={"slope": 1.0},
    prior=prior_results,
)
```

Each panel shows how much of its greatest effect a channel reaches at each
level of carried media. A curve that bends early saturates quickly, so extra
exposure adds little, and a curve that keeps climbing still has room.
`max_input` is in the units the function receives, here carried media on the
scale the model fits, where a typical active week sits near one. Linear TV's
and streaming's posterior curves climb furthest above their priors, and most of
the others stay close to theirs.

{func}`~mmmjax.plot_response_curves` shows what the curves mean in dollars.

```{code-cell} ipython3
curves = mj.response_curves(model, results, quantity="mu")
mj.plot_response_curves(curves, combine=True)
```

Each curve follows the incremental revenue as a channel's spending runs from
zero to twice its current level, the range {func}`~mmmjax.response_curves`
covers unless `multipliers` sets another. The point marks the current spending,
and the dashed stretch beyond it extrapolates past what the data observed. The
steepness at the point is the marginal return, what the next dollar would add.
`combine=True` draws the five channels with the most spending in one panel so
their curves compare on common axes, and without it each channel gets a panel
of its own, as in the budget plot below. Meta and streaming follow nearly the
same curve, but Meta spends further along it, where each extra dollar adds
less.

## Budgets

The budget plots read the output of {func}`~mmmjax.optimize_budget`. Limits
of 30 percent either way keep each channel near the spending the data
observed, and `include_metrics=True` records the incremental revenue each
channel brings under both splits. {func}`~mmmjax.plot_budget_response` walks
from the historical revenue to the optimized one.

```{code-cell} ipython3
plan = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_constraint_lower=0.3,
    spend_constraint_upper=0.3,
    include_metrics=True,
)
mj.plot_budget_response(plan)
```

The first bar is the incremental revenue of the historical split and the
last is that of the optimized one. Each bar between is one channel's change,
cuts first in pale red with diagonal lines and gains after in light green.
Cutting generic search costs \$198,000 and streaming's increase adds
\$253,000, the largest moves in each direction. The subtitle summarizes the
change as a gain of \$192,000 with an 89 percent interval from a loss of
\$148,000 to a gain of \$503,000, and the plan gains in 84 percent of the
draws. An interval that reaches into losses means the plan could lose revenue,
and the share of draws that gain says how likely it is to come out ahead.

{func}`~mmmjax.plot_budget_spend` shows the moves themselves.

```{code-cell} ipython3
mj.plot_budget_spend(plan)
```

Each bar is a channel's optimized spending minus its historical spending, in
the same colors. Generic search gives up \$137,000 and streaming gains
\$94,100, which takes streaming to its upper limit.

Passing the plan to {func}`~mmmjax.plot_response_curves` marks both splits
on each channel's curve.

```{code-cell} ipython3
mj.plot_response_curves(curves, plan=plan)
```

Each channel gets a panel with its own axes, a circle at the historical
spending, and a triangle at the optimized one, and the line turns dashed
outside the plan's limits. The optimizer moves spending from the flat
stretches of the curves to the steep ones until the next dollar earns about
the same everywhere or a channel reaches its limit, which is why streaming's
triangle sits at the end of its solid stretch.

## Larger models

The plots keep working as a model grows. The figures below draw made-up
results for a brand with 60 channels, so their code is left out.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import xarray as xr

rng = np.random.default_rng(3)
platforms = [
    "Meta",
    "TikTok",
    "Snapchat",
    "YouTube",
    "Pinterest",
    "Reddit",
    "LinkedIn",
    "X",
    "Google search",
    "Bing search",
    "Amazon",
    "Display",
    "Streaming TV",
    "Linear TV",
    "Podcasts",
]
tactics = ["brand", "prospecting", "retargeting", "promotions"]
names = [f"{platform} {tactic}" for platform in platforms for tactic in tactics]
typical = rng.lognormal(1.1, 0.35, size=60)
large_returns = xr.Dataset(
    {
        "roi": (("chain", "draw", "channel"), typical * rng.lognormal(0.0, 0.3, size=(4, 1000, 60))),
        "reference_spend": (("channel",), rng.lognormal(11.5, 0.9, size=60)),
    },
    coords={"chain": np.arange(4), "draw": np.arange(1000), "channel": names},
    attrs={"outcome": "revenue"},
)
mj.plot_media_metrics(large_returns)
```

Bar charts show every channel and widen once they pass about a dozen. A
notebook shows the wide chart at full size in a box that scrolls sideways, as
this page does. Names longer than 27 characters are shortened so every label
takes the same room.

```{code-cell} ipython3
:tags: [remove-input]

shapes = {
    "intercept": {},
    "growth": {},
    "curvature": {},
    "sigma": {},
    "annual_coefficients": {"annual_mode": 4},
    "holiday_coefficients": {"holiday": 6},
    "control_coefficient": {"control": 8},
    "roi": {"channel": 60},
    "retention": {"channel": 60},
    "half_saturation": {"channel": 60},
    "slope": {"channel": 60},
    "roi_location": {"platform": 15},
    "roi_scale": {"platform": 15},
    "changepoint_effects": {"changepoint": 10},
    "noise_scale": {},
}
variables = {}
for name, axes in shapes.items():
    draws = rng.normal(size=(4, 1000, *axes.values()))
    if name == "roi":
        # Two chains sit apart from the others for two channels, so their R-hat passes the limit.
        draws[1::2, :, [5, 41]] += 0.4
    variables[name] = (("chain", "draw", *axes), draws)
large_results = xr.DataTree.from_dict({"posterior": xr.Dataset(variables)})
mj.plot_rhat(large_results)
```

{func}`~mmmjax.plot_rhat` gives each of the 15 parameters one row however many
elements it holds, so the 60 channels' returns share a row, as do their
retention rates, half-saturation points, and slopes. Two returns were made to
mix poorly here, and they stand out in orange past the line.

Response curves keep the ten channels with the most spending, and the adstock
and saturation plots keep the first ten. Each says in its caption what it left
out and takes `channels` to pick others. The rank, trace, and
prior and posterior plots keep the elements that most need a look, as the
brand's own plots above do. A model fitted by group gives each of its largest
groups a panel, and `coords` and `n_groups` choose which groups and how many.
Channels bought by reach and frequency add
{func}`~mmmjax.plot_frequency_curves`, which draws the output of
{func}`~mmmjax.frequency_curves`.
