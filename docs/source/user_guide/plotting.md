---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Plotting

mmmJAX draws the plots a marketing mix analysis keeps coming back to, from
checking the sampler to deciding where the next dollar should go. Each
function reads the results of a fit or the output of an analysis function, so a
plot never runs the model again. Plots change as channels are added, so this
page fits a brand with ten of them, and [Customizing plots](custom_plots) shows
how to change what the plots draw.

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

Parameters run down the side, and each point is one of a parameter's
elements, such as one channel's return. R-hat compares the spread between the
four chains with the spread within them, so a value near one means every chain
found the same distribution. A value past ArviZ's limit of 1.01 would sit
right of the dotted line and turn orange, and the subtitle confirms that all
43 sit below it. An element past the line can't be trusted until a longer run
or a better parameterization brings it back. {func}`~mmmjax.plot_rank` looks at the
chains more closely.

```{code-cell} ipython3
mj.plot_rank(results)
plt.show()
```

Each panel follows one element, and each line one chain's ranks among all the
draws. Chains that explore the same distribution spread their ranks evenly, so
their lines stay near zero. A line that bows away, as chain 1 does for
Snapchat's return, shows a chain that lingered in one part of the distribution.
The p-value above each panel tests that, as [Sampling and diagnostics](sampling)
explains. The brand has more parameters than ArviZ draws in one figure, so the
plot keeps the 12 with the highest R-hat, the likeliest to have mixed poorly,
and says so in its title. {func}`~mmmjax.plot_trace_dist` makes the same
choice with six rows.

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
tail, would point to a feature of the data the model misses.
{func}`~mmmjax.plot_fit` keeps the weeks in order.

```{code-cell} ipython3
mj.plot_fit(model, results)
```

The black line is observed revenue, and the blue line and band are the mean
and 89 percent interval of the predictive draws, all in dollars because the
plot undoes the outcome scaling first. The subtitle gives an $R^2$ of 0.91, a
weighted mean absolute percentage error of 4.1 percent, and the share of weeks
inside the band, 92 percent here. A calibrated model keeps that share close to
the interval's probability. Weeks where the black line leaves the band are
ones the model can't explain, and a run of them points to something missing,
such as an unmodeled promotion.

Intervals and point estimates follow ArviZ's settings, 89 percent intervals
around the mean unless changed, and every plot with an interval takes
`ci_prob` to change it for one call. Passing the output of
{func}`~mmmjax.contributions` with its weeks kept adds the baseline, the
revenue the model expects with every channel removed.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
mj.plot_fit(model, results, effects=effects)
```

The gap between the orange baseline and the blue predictions is what the ten
channels add each week. It opens widest during the flights, and the baseline
itself carries the trend and the seasons. {func}`~mmmjax.plot_residuals` shows
what the predictions leave over.

```{code-cell} ipython3
mj.plot_residuals(model, results)
```

Each week's residual is the observed revenue minus the prediction, and the
band is its 89 percent interval across draws. A good fit leaves residuals that
wander around zero with no pattern. A trend, a repeating season, or a long
stretch above or below zero would each point to structure the model misses.
Here the line stays around zero without a trend or a season.

## Priors and posteriors

{func}`~mmmjax.plot_prior_posterior` overlays each parameter's prior on its
posterior.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results)
plt.show()
```

Each panel draws a parameter's prior in blue and its posterior in orange. The
further the posterior moves from the prior, or the narrower it gets, the more
the data taught the model about that parameter. A posterior that looks like its
prior means the data said little, so the prior carries the answer. The plot
keeps the 12 of the 43 parameters whose posterior moved furthest. The trend's
curvature moved furthest, followed by Snapchat's return and the trend's
growth. The media parameters that made the list, such as linear TV's
half-saturation point and YouTube's retention, moved much less, and the
adstock and saturation plots below show this channel by channel.

## Contributions

{func}`~mmmjax.plot_contributions` breaks revenue down into the baseline and
each channel. The weeks in `effects` are summed first.

```{code-cell} ipython3
mj.plot_contributions(effects)
```

Each bar starts where the one before it ends, so the bars walk from the
baseline through every channel to all of the revenue, and each label gives a
share and its total in dollars. The baseline holds 79.9 percent, and Meta, the
largest channel, holds 3.9 percent, about \$2.14 million over the three years.
The order ranks what each channel delivered, not what it returns on a dollar,
since a large budget can buy a large share at a middling return. The shares
are ratios of posterior means, so they add up, and a last gray bar would carry
any difference left by channels whose effects interact. `by="time"` keeps the
weeks instead.

```{code-cell} ipython3
mj.plot_contributions(effects, by="time")
```

The gray area is the baseline, each channel stacks on top of it, and the black
line is total revenue. The axis starts near the lowest weekly baseline, so the
channels stay visible above a much larger baseline. The layers show when each
channel works. Linear TV and streaming arrive in flights, while generic search
and display run every week. `include_baseline=False` drops the baseline and
stacks the channels under a line that traces what removing all ten at once
would cost.

## Returns

The return plots read the output of {func}`~mmmjax.media_metrics`.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

{func}`~mmmjax.plot_media_metrics` draws the return on investment by default.
Each bar is a channel's mean return on a dollar and the error bar its 89
percent interval. The dashed line marks break-even, where a channel returns
exactly what it costs, so a bar above it has paid for itself. Channels run from
the most spending to the least, from Meta at \$3.87 to Snapchat at \$6.69. Read
the intervals as closely as the bars. Snapchat's return is the highest, but its
interval is also by far the widest, since the smallest budget leaves the model
the least to learn from. `metric` switches the bars to any other metric with
draws.

```{code-cell} ipython3
mj.plot_media_metrics(returns, metric="marginal_roi")
```

The marginal return is the return on one more percent of spending. The
average return says how well past spending paid off, and the marginal return
says what the next dollar would earn, which is what a budget decision needs.
It runs from \$1.23 for generic search to \$3.96 for Snapchat, below every
channel's average return, since each channel's curve flattens as it spends. A
channel with a high average return but a low marginal one has already spent
far into its saturation.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

{func}`~mmmjax.plot_roi_bubbles` places each channel at its average return
across and its marginal return up, and each bubble's area is proportional to
the channel's spending. The dashed lines mark break-even on both axes. A
channel to the right has returned more than it cost so far, and a channel above
would still return more than its next dollar costs. Every channel here sits
above and to the right, so what matters is how far up each one sits. Streaming
and Snapchat sit highest, while generic search, the second largest budget,
has the lowest marginal return. Moving money from low bubbles to high ones is
what the budget below does.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns, metric="effectiveness")
```

`metric="effectiveness"` puts the incremental revenue per impression on the
vertical axis. Effectiveness asks how much each impression does and the return
asks how much each dollar does, so the two differ by what an impression
costs. A channel high on both works well and buys its impressions at a fair
price. A channel with a middling return and low effectiveness, such as display,
buys cheap impressions that each do little, while one with high effectiveness
and a low return would have strong impressions at too high a price. Streaming
and branded search earn the most on each impression. The horizontal line is
gone because only a return has a break-even point.

```{code-cell} ipython3
mj.plot_spend_vs_contribution(returns)
```

{func}`~mmmjax.plot_spend_vs_contribution` gives each channel a wide hatched
frame for its share of the spending and a narrow solid bar inside it for its
share of the incremental revenue, with its return above. When the solid bar
rises above its frame, the channel earns more than its share, which means a
return above the average of all ten. When it stops short, the channel takes
more of the budget than it gives back. Streaming's solid bar rises well above
its frame, while generic search's stops short, and the budget below moves more
money between these two than between any others.

## Media transformations

{func}`~mmmjax.plot_adstock` passes one unit of exposure through the model's
own adstock function for every draw, and `prior` adds the draws of
{func}`~mmmjax.sample_prior` for comparison.

```{code-cell} ipython3
mj.plot_adstock(
    results,
    mj.geometric_adstock,
    max_lag=8,
    parameters={"alpha": "retention"},
    prior=prior_results,
)
```

The function receives each draw of a parameter through the argument of the
same name, the way the model's blocks receive their inputs, and `parameters`
maps the function's `alpha` to the model's `retention` since the two names
differ. Each panel shows the share of an exposure's effect that lands in each
later week, with the posterior in blue and the prior in orange. A curve that
drops fast means the effect is spent within a week or two, and a slow decline
means it carries over for weeks. For most channels the posterior band sits on
the prior's, so the data say little about how long their effects last.
YouTube's posterior moves the most, toward effects that fade faster than its
prior expects. Where the data barely move a prior, the prior carries the answer
and deserves the most thought. {func}`~mmmjax.plot_saturation` does the same
for the saturation curve.

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
`half_saturation` reaches the function by name, and `parameters` fixes the
slope the model holds at one. `max_input` is in the units the function
receives, here carried media on the scale the model fits, where a typical
active week sits near one. Linear TV's and streaming's posterior curves climb
furthest above their priors, and most of the others stay close to theirs.
{func}`~mmmjax.plot_response_curves` shows what the curves mean in dollars.

```{code-cell} ipython3
import numpy as np

curves = mj.response_curves(model, results, quantity="mu", multipliers=np.linspace(0.0, 2.0, 21))
mj.plot_response_curves(curves, combine=True)
```

Each curve follows the incremental revenue as a channel's spending runs from
zero to twice its current level. The point marks the current spending, and the
dashed stretch beyond it extrapolates past what the data observed. The
steepness at the point is the marginal return, what the next dollar would add.
`combine=True` draws the five channels with the most spending in one panel so
their curves compare on common axes. Meta and streaming follow nearly the same
curve, but Meta spends further along it, where each extra dollar adds less.
Without `combine`, each channel gets a panel of its own, as in the budget plot
below.

## Budgets

The budget plots read the output of {func}`~mmmjax.optimize_budget`. Limits
of 30 percent either way keep each channel near the spending the data
observed, and `include_metrics=True` records the incremental revenue each
channel brings under both splits.

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

The first bar is the incremental revenue of the current split and the last is
that of the optimized one, and each bar between is one channel's change, cuts
first. Cutting generic search costs \$198,000 and streaming's increase adds
\$253,000, the largest moves in each direction. The subtitle summarizes the
change as a gain of \$192,000 with an 89 percent interval from a loss of
\$148,000 to a gain of \$503,000, and the plan gains in 84 percent of the
draws. Read the subtitle before the bars. An interval that reaches into losses
makes the plan a good bet rather than a sure one, and the share of draws that
gain says how good.

```{code-cell} ipython3
mj.plot_budget_spend(plan)
```

{func}`~mmmjax.plot_budget_spend` shows the moves themselves, cuts in orange
and increases in blue. Generic search gives up \$137,000 and streaming gains
\$94,100, which takes streaming to its upper limit. Passing the plan to
{func}`~mmmjax.plot_response_curves` marks both splits on each channel's curve.

```{code-cell} ipython3
mj.plot_response_curves(curves, plan=plan)
```

Each channel gets a panel with its own axes, a circle at the current spending,
and a triangle at the optimized one. The line turns dashed outside the plan's
limits, so streaming's triangle sits at the end of its solid stretch. The
optimizer moves spending from the flat stretches of the curves to the steep
ones until the next dollar earns about the same everywhere or a channel
reaches its limit.

These plots grow with the model. Bar charts show every channel and widen once
they pass about a dozen, and a notebook shows a wide chart at full size in a
box that scrolls sideways. Curve plots keep the ten channels with the most
spending, say in the caption what they left out, and take `channels` to pick
others. A channel keeps its color in every plot, whichever others a plot
shows. A model fitted by group gives each of its largest groups a panel, and
`coords` and `n_groups` choose which groups and how many. Channels bought by
reach and frequency add {func}`~mmmjax.plot_frequency_curves`, which draws the
output of {func}`~mmmjax.frequency_curves`. The simulated brand buys every
channel by the impression, so it has none to show.
