---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Plotting

mmmJAX draws the plots a marketing mix analysis needs for convergence, fit,
priors, contributions, returns, media transformations, and budgets. Each
function reads the results of a fit or the output of an analysis function, so
no plot refits the model. Apart from the last section, every plot below comes
from the ten-channel brand that [A first model](first_model) fits.

Results are ordinary xarray DataTree objects and every analysis output is an
ordinary xarray Dataset, so ArviZ, plotnine, matplotlib, or any other plotting
library can draw from them directly. When the built-in plots don't show what
you need, [Customizing plots](custom_plots) builds three figures of its own
from the same outputs.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_curves, first_model_limited_plan, first_model_prior_results, first_model_results

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

## Convergence

Start with {func}`~mmmjax.plot_rhat`, since it draws every parameter's R-hat
at once.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Parameters run down the side, and each point is one element of a parameter,
such as one channel's return. A value past ArviZ's limit of 1.01 would land
right of the dotted line in orange, but the subtitle confirms that all 45
values of the brand's fit sit at or below it. [Sampling and
diagnostics](sampling) explains what R-hat measures, and `var_names` limits the
plot to the parameters you name.

For a closer look at the chains themselves, turn to {func}`~mmmjax.plot_rank`.

```{code-cell} ipython3
mj.plot_rank(results)
plt.show()
```

A rank plot gives each element a panel and each chain a line that traces its
ranks among all the draws. When a line bows away, its chain lingered in one
part of the distribution.

The brand has more parameters than ArviZ draws in one figure, so the plot keeps
the 12 with the highest R-hat. Among them, chain 3 on Snapchat's retention
rate, in the bottom-right panel, bows away furthest, though its R-hat stays
within the limit.

{func}`~mmmjax.plot_trace_dist` picks its elements the same way, though it
keeps only six rows.

```{code-cell} ipython3
mj.plot_trace_dist(results)
plt.show()
```

The rows pair an element's density on the left with its draws in sampling
order on the right, and each chain has its own line style. Even for these six,
the worst by R-hat, the chains mix well, since their densities overlap and
their draws run as flat, even bands.

## Fit

{func}`~mmmjax.plot_ppc_dist` compares the distribution of the observed
revenue with that of the posterior predictive draws.

```{code-cell} ipython3
mj.plot_ppc_dist(model, results)
plt.show()
```

The blue curves each trace the distribution of one draw's predicted weeks, and
the black curve of the observed weeks stays inside their bundle, second hump of
busy weeks included. So the observed revenue looks like one more draw from the
model.

If you pass the output of {func}`~mmmjax.sample_prior` with `group="prior"`
instead, the same plot checks the priors before any fit, as [Priors](priors)
shows.

{func}`~mmmjax.plot_fit` keeps the weeks in order, so you can follow the fit
through time.

```{code-cell} ipython3
mj.plot_fit(model, results)
```

The black line is observed revenue, and the blue line and band are the mean
and 89 percent interval of the predictive draws. Because the plot undoes the
outcome scaling, both lines and the band are in dollars.

In the subtitle, an $R^2$ of 0.91 and a weighted mean absolute percentage error
of 4.1 percent say the mean tracks revenue closely, and 93 percent of weeks
fall inside the band. [Sampling and diagnostics](sampling) looks at the weeks
outside it and why their timing matters.

`show_baseline=True` adds the baseline, the revenue the model expects with the
paid channels and Email removed and with price and promotion at their baseline
levels. To get it, the plot runs {func}`~mmmjax.contributions` on the same
draws.

:::{admonition} `quantity` takes one of your names
:class: important

`mu` is the key the brand's `transformed_parameters` returns, so it's one of
your names rather than one mmmJAX supplies, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
explains. If your model returns its expected revenue under another key, pass
that key here and in every analysis below.
:::

```{code-cell} ipython3
mj.plot_fit(model, results, show_baseline=True, quantity="mu")
```

The gap between the orange baseline and the blue predictions is what media
and the two treatments add each week. It opens widest during the flights and
promotions, while the baseline carries the trend, the seasons, and the two
controls.

:::{admonition} Intervals and point estimates
:class: tip

ArviZ's settings give every plot 89 percent intervals around the mean unless
you change them. `ci_prob` changes the interval for one call, and ArviZ's
`rc_context`, as in
`az.rc_context({"stats.ci_prob": 0.9, "stats.point_estimate": "median"})`,
changes both for a block of code.
:::

### Residuals and test statistics

{func}`~mmmjax.plot_residuals` shows what's left over after the predictions,
so you can look for patterns the model missed.

```{code-cell} ipython3
mj.plot_residuals(model, results)
```

A week's residual is the observed revenue minus the prediction, and the band
is its 89 percent interval across draws. The line wanders around zero without a
trend or a season, since the model accounts for both. In all but one week of
February and March 2023, though, it sits above zero, so the predictions there
ran below the observed revenue.

Rather than reading checks like these by eye, you can put numbers on them with
{func}`~mmmjax.plot_ppc_tstat`.

```{code-cell} ipython3
mj.plot_ppc_tstat(model, results, quantity="mu")
plt.show()
```

The plot computes one statistic per panel on every predictive draw, and the
black dot marks its value for the observed revenue. The p in each title is the
share of draws that reach it. For the standard deviation and the largest week,
the observed value lands near the middle of the draws, so the model reproduces
both.

Unlike those two, residual autocorrelation takes its residuals from each
draw's expected revenue, which `quantity` names, so the observed values show
up as a black curve instead of a dot. That curve sits well to the right of the
blue one, so its p rounds to zero. [Sampling and diagnostics](sampling) traces
that back to streaks in the residuals.

Beyond these three, `statistics` takes other names, quantiles such as `0.9`,
or any function that turns a series of weekly revenue into a number.

## Priors and posteriors

Before you compare priors with posteriors, you need draws from the priors
alone, and {func}`~mmmjax.sample_prior` makes them from the `priors` mapping.

```{code-cell} ipython3
:tags: [skip-execution]

prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

prior_results = first_model_prior_results(model, priors)
```

{func}`~mmmjax.plot_prior_posterior` takes both results and overlays each
parameter's prior on its posterior.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results)
plt.show()
```

The plot draws each parameter's prior in blue and its posterior in orange, and
it keeps the 12 of 45 that the data narrowed least. Most of their posteriors
sit on their priors, so the priors carry most of what the model says about how
these channels carry over and saturate. In the panels for Snapchat's return and
Branded search's retention rate, though, the orange posterior leans further
right than the blue prior. [Priors](priors) goes through what the data did
change for the returns and treatments.

### Prior sensitivity

{func}`~mmmjax.psense_summary` measures how much each posterior depends on the
priors and how much on the data.

```{code-cell} ipython3
:tags: [wide-table]

mj.psense_summary(results, priors=priors, var_names=["roi"])
```

Raising the priors or the likelihood to a power a little above or below one
reweights the draws without refitting, and the prior and likelihood columns
measure how far that moves a channel's return.

Reading down the table, eight channels run high in both columns, and the
diagnosis points to a possible conflict between prior and data. For TikTok and
Generic search only the prior runs high, so the diagnosis points to a strong
prior with a weak likelihood. The columns after the diagnosis scale each prior
on its own, and the ROI prior moves every return the most.

The table says how far each posterior moves, and {func}`~mmmjax.plot_psense`
shows which way.

```{code-cell} ipython3
mj.plot_psense(results, priors=priors)
plt.show()
```

The left panel of each row draws an element's distribution with the priors
raised to the powers in the legend, and the right panel does the same with the
likelihood. The lines under the curves give the point estimate and 89 percent
interval at each power. For Display's and Snapchat's returns a stronger prior
pulls the interval down while a stronger likelihood pushes it up. That's what
you see when a prior sits below what the data suggests.

## Contributions

{func}`~mmmjax.plot_contributions` sums the weeks in the output of
{func}`~mmmjax.contributions` and breaks revenue down into the baseline, each
channel, and each treatment.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
mj.plot_contributions(effects)
```

Because each bar starts where the one before it ends, the bars walk from the
baseline through every input to all of the revenue. The label on each bar
gives its share and its total in dollars, both posterior means, and
[Media effects](media_effects) explains how they're computed.

The baseline holds 79.8 percent of revenue, and Meta, the largest paid
channel, brings in 3.5 percent, about \$1.93 million over the three years.
Promotion adds 3.0 percent, about \$1.61 million, while price's orange bar
takes away 2.5 percent, about \$1.36 million, because it compares each week
with the lowest price the brand charged.

Passing `by="time"` keeps the weeks instead of summing them, so the plot shows
when each input contributes.

```{code-cell} ipython3
mj.plot_contributions(effects, by="time")
```

The light gray area is the baseline and the black line is total revenue. The
ten largest inputs stack on top of the baseline, and the other three share the
darker gray area. Linear TV and Streaming arrive in flights and promotion
swells in the weeks the brand runs one, while Generic search and Display run
every week.

If you'd rather see the inputs alone, `include_baseline=False` drops the
baseline and stacks them under a line that traces what removing them all at
once would cost.

:::{admonition} The axis starts above zero
:class: note

The vertical axis starts near the lowest weekly baseline rather than at zero,
so the inputs stay visible.
:::

## Returns

The return plots read the output of {func}`~mmmjax.media_metrics`, and since a
return needs spending, that output covers only the ten paid channels.
{func}`~mmmjax.plot_media_metrics` draws the return on investment unless you ask
for another metric.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

The bars give each channel's mean return on a dollar with its 89 percent
interval, and the dashed line marks break-even. The channels run from the most
spending to the least, from Meta at \$3.49 to Snapchat at \$3.98. Snapchat's
interval is the widest, because the smallest budget leaves the model the least
to learn from. If you want the formula behind each return,
[Media effects](media_effects) writes it out.

:::{admonition} Comparing results
:class: tip

If you pass a mapping of labeled results, such as
`{"Prior": prior_returns, "Posterior": returns}`, the plot draws each channel's
bars side by side in one color per result. To get `prior_returns`, run
{func}`~mmmjax.media_metrics` with `group="prior"` on the output of
{func}`~mmmjax.sample_prior`.
:::

The `metric` argument switches the bars to any other metric that has draws.

```{code-cell} ipython3
mj.plot_media_metrics(returns, metric="marginal_roi")
```

The average return says how well past spending paid off, and the marginal
return says what the next dollar would earn. Here the marginal return runs
from \$1.18 for Generic search to \$2.40 for Snapchat. Each channel's sits
below its average return, because its curve flattens as it spends and each new
dollar earns less than the last.

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

These panels follow the mean or standard deviation of a channel's marginal
return as the priors (blue) or the likelihood (orange) are raised to a power.
The dashed lines mark two Monte Carlo standard errors around the unscaled
value, so a change between them could be noise in the draws. Snapchat's mean
falls as the priors strengthen and rises as the likelihood does, so the priors
pull it down. Generic search, though, stays the lowest of the three at every
power, so the priors don't set its place at the bottom.

### Average and marginal returns

{func}`~mmmjax.plot_roi_bubbles` lets you weigh each channel's average return
against its marginal one.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

The horizontal axis holds each channel's average return and the vertical axis
its marginal return, and each bubble's area is proportional to the channel's
spending. Every channel sits above and to the right of the dashed break-even
lines, so on average and at the margin, each one brings in more than it costs.
Streaming and Snapchat sit highest, while Generic search, the second largest
budget, has the lowest marginal return.

`metric="effectiveness"` puts the incremental revenue per impression on the
vertical axis instead.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns, metric="effectiveness")
```

Effectiveness asks how much each impression does and the return asks how much
each dollar does, so the two differ by what an impression costs. Branded
search and Streaming earn the most on each impression, while Display pairs a
middling return with the lowest effectiveness, so its impressions must come
cheap.

### Shares of spend and revenue

{func}`~mmmjax.plot_spend_vs_contribution` compares each channel's share of
the spending with its share of the incremental revenue.

```{code-cell} ipython3
mj.plot_spend_vs_contribution(returns)
```

When a channel's solid bar is taller than its hatched one, its return is above
the average of all ten. Streaming's and YouTube's solid bars stand well above
their hatched ones, while Generic search's and Linear TV's fall short. The
budget below still cuts Generic search the most and Linear TV hardly at all,
because it follows the marginal returns above rather than these averages.

## Media transformations

{func}`~mmmjax.plot_adstock` passes one unit of exposure through the model's
own adstock function for every draw and gives each channel a panel. The
`prior` argument adds the draws of {func}`~mmmjax.sample_prior` so you can
compare them with the posterior.

```{code-cell} ipython3
mj.plot_adstock(
    results,
    mj.geometric_adstock,
    max_lag=8,
    parameters={"alpha": "retention"},
    prior=prior_results,
)
```

In each panel, the blue posterior and the orange prior show the share of an
exposure's effect that lands in each later week. For most channels the
posterior sits on the prior, so the data says little about how long their
effects last. YouTube's posterior does move toward effects that fade faster,
while Branded search's moves toward effects that last longer.

:::{admonition} How the parameters reach the function
:class: note

The function receives each draw of a parameter through the argument of the
same name, the way the model's blocks receive their inputs. `parameters` maps
an argument to a parameter you named differently, as `alpha` to the brand's
`retention` here, or fixes it at a number, as the slope below. Swap in
`organic_retention` for `retention` and the same call draws Email's carryover.
:::

{func}`~mmmjax.plot_saturation` does the same for the model's second
transformation, the saturation curve.

```{code-cell} ipython3
mj.plot_saturation(
    results,
    mj.hill_saturation,
    max_input=3.0,
    parameters={"slope": 1.0},
    prior=prior_results,
)
```

The panels show how much of its greatest effect a channel reaches at each level
of carried media. `max_input` is in the units the function receives, here
carried media on the scale the model fits.

While most channels' posterior curves stay close to their priors, Linear TV's
and Streaming's climb furthest above theirs, so the data says those two
saturate at lower levels of media than their priors suggested. If you want
the math behind both transformations, [A first model](first_model) writes it
out.

### Response curves

Both transformations work on scaled media, and
{func}`~mmmjax.plot_response_curves` shows what they mean in dollars.

```{code-cell} ipython3
:tags: [skip-execution]

curves = mj.response_curves(model, results, quantity="mu")
```

```{code-cell} ipython3
:tags: [remove-cell]

curves = first_model_curves(model, results)
```

```{code-cell} ipython3
mj.plot_response_curves(curves, combine=True, channels=["Meta", "Streaming", "Generic search"])
```

The curves follow the incremental revenue as each channel's spending runs from
zero to twice its current level. A point marks the current spending, and a
dashed stretch extrapolates past it. With `combine=True` the channels you name
share one panel, and since overlaid bands blur together past a few channels,
the call names only three.

Meta sits furthest along its curve, where each extra dollar adds less, while
Streaming's curve still climbs steeply past its point. That's why Streaming's
marginal return above is the highest of these three.
[Media effects](media_effects) walks through each channel's curve on a panel of
its own.

For channels you buy by reach and frequency,
{func}`~mmmjax.plot_frequency_curves` draws the output of
{func}`~mmmjax.frequency_curves`.

## Budgets

The budget plots read the output of {func}`~mmmjax.optimize_budget`, and
[Budget optimization](budgets) explains how the optimizer finds its plan. The
plan here keeps each channel within 30 percent of its historical spending, and
with `include_metrics=True`, it records the incremental revenue each channel
brings under both splits.

```{code-cell} ipython3
:tags: [skip-execution]

plan = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_constraint_lower=0.3,
    spend_constraint_upper=0.3,
    include_metrics=True,
)
```

```{code-cell} ipython3
:tags: [remove-cell]

plan = first_model_limited_plan(model, results)
```

{func}`~mmmjax.plot_budget_response` shows the plan's worth by walking from the
historical revenue to the optimized one.

```{code-cell} ipython3
mj.plot_budget_response(plan)
```

The first bar is the incremental revenue of the historical split and the last
is that of the optimized one. Each channel's change sits between them, and the
cuts come first. The biggest cut, to Generic search, costs \$172,000, and the
biggest increase, to Streaming, adds \$204,000.

The subtitle puts the mean gain at \$134,000 and says the plan gains in 77
percent of the draws. Its 89 percent interval still runs from a loss of
\$170,000 to a gain of \$426,000, so the plan could lose revenue.

Behind those revenue changes are moves in spending, and
{func}`~mmmjax.plot_budget_spend` draws them.

```{code-cell} ipython3
mj.plot_budget_spend(plan)
```

This time each bar is a channel's optimized spending minus its historical
spending, and the colors match the plot above. Generic search gives up
\$126,000, while Streaming's gain of \$94,100 takes it to its upper limit.

Passing the plan to {func}`~mmmjax.plot_response_curves` marks both splits
on each channel's curve.

```{code-cell} ipython3
mj.plot_response_curves(curves, plan=plan)
```

Without `combine=True`, every channel gets a panel with its own axes. A circle
marks the historical spending and a triangle the optimized one, and the line
turns dashed outside the plan's limits. The optimizer moves spending from the
flat stretches of the curves to the steep ones until the next dollar earns
about the same everywhere or a channel reaches its limit. That's why
Streaming's triangle sits at the end of its solid stretch.

## Larger models

To show that the plots keep working as a model grows, the figures below draw
made-up results for a brand with 60 channels. Because those results aren't
real, the page leaves out their code.

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

Bar charts show every channel, so past about a dozen they grow wider.
In a notebook the wide chart appears at full size in a box that scrolls
sideways, as on this page. Names longer than 27 characters are
shortened so every label takes the same room.

```{code-cell} ipython3
:tags: [remove-input]

shapes = {
    "intercept": {},
    "growth": {},
    "curvature": {},
    "sigma": {},
    "annual_coefficients": {"annual_coefficients_dim_0": 4},
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

The other plots trim what they draw as a model grows.

- Response curves keep the ten channels with the most spending, and the
  adstock and saturation plots keep the first ten. Each says in its caption
  what it left out and takes `channels` to pick others.
- The rank, trace, and prior and posterior plots keep the elements that most
  need a look, as the brand's own plots above do.
- A model fitted by group gives each of its largest groups a panel, and
  `coords` and `n_groups` choose which groups and how many.
