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
no plot refits the model. Every plot below draws the ten-channel brand from
[A first model](first_model), and each section links the page that explains
what its plots measure.

The functions are there for convenience, and nothing stops you from plotting
another way. Results are ordinary xarray DataTree objects and every analysis
output is an ordinary xarray Dataset, so ArviZ, plotnine, matplotlib, or any
other plotting library can draw from them directly.
[Customizing plots](custom_plots) builds three figures of its own from the
same outputs.

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

{func}`~mmmjax.plot_rhat` draws every parameter's R-hat at once.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Parameters run down the side, and each point is one element of a parameter,
such as one channel's return. A value past ArviZ's limit of 1.01 would sit
right of the dotted line in orange, and the subtitle confirms that all 45
values of the brand's fit sit at or below it. [Sampling and
diagnostics](sampling) explains what R-hat measures, and `var_names` limits the
plot to the parameters you name.

{func}`~mmmjax.plot_rank` looks at the chains more closely.

```{code-cell} ipython3
mj.plot_rank(results)
plt.show()
```

Each panel follows one element, and each line one chain's ranks among all the
draws, so a line that bows away shows a chain that lingered in one part of the
distribution. The brand has more parameters than ArviZ draws in one figure, so
the plot keeps the 12 with the highest R-hat and says so in its title. Chain 3
bows away furthest, for Snapchat's retention rate.

{func}`~mmmjax.plot_trace_dist` makes the same choice with six rows.

```{code-cell} ipython3
mj.plot_trace_dist(results)
plt.show()
```

Each row pairs an element's density on the left with its draws in sampling
order on the right, with a line style for each chain. Even these six, the
worst by R-hat, overlap in their densities and run as flat, even bands.

## Fit

{func}`~mmmjax.plot_ppc_dist` compares the distribution of the observed
revenue with that of the posterior predictive draws.

```{code-cell} ipython3
mj.plot_ppc_dist(model, results)
plt.show()
```

Each blue curve is the distribution of one draw's predicted weeks, and the
black curve of the observed weeks stays inside their bundle down to the second
hump of busy weeks. With the output of {func}`~mmmjax.sample_prior` and
`group="prior"`, the same plot checks the priors before any fit, as
[Priors](priors) shows.

{func}`~mmmjax.plot_fit` keeps the weeks in order.

```{code-cell} ipython3
mj.plot_fit(model, results)
```

The black line is observed revenue, and the blue line and band are the mean
and 89 percent interval of the predictive draws, in dollars because the plot
undoes the outcome scaling. The subtitle gives an $R^2$ of 0.91, a weighted
mean absolute percentage error of 4.1 percent, and 93 percent of weeks inside
the band, and [Sampling and diagnostics](sampling) reads the weeks outside it.

:::{admonition} Intervals and point estimates
:class: tip

Every plot follows ArviZ's settings, which give 89 percent intervals around
the mean unless you change them. `ci_prob` changes the interval for one call,
and ArviZ's `rc_context`, as in
`az.rc_context({"stats.ci_prob": 0.9, "stats.point_estimate": "median"})`,
changes both for a block of code.
:::

`show_baseline=True` adds the baseline, the revenue the model expects with
the paid channels and Email removed and with price and promotion at their
baseline levels. {func}`~mmmjax.contributions` computes it from the same
draws, and `quantity` names the expected revenue the blocks return.

:::{admonition} `quantity` takes one of your names
:class: important

`mu` is the key the brand's `transformed_parameters` returns, so it's one of
your names rather than one mmmJAX supplies, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
explains. A model that returns its expected revenue under another key passes
that key here and in every analysis below.
:::

```{code-cell} ipython3
mj.plot_fit(model, results, show_baseline=True, quantity="mu")
```

The gap between the orange baseline and the blue predictions is what media
and the two treatments add each week. It opens widest during the flights and
promotions, while the baseline carries the trend, the seasons, and the two
controls.

{func}`~mmmjax.plot_residuals` shows what's left over after the predictions.

```{code-cell} ipython3
mj.plot_residuals(model, results)
```

Each week's residual is the observed revenue minus the prediction, and the
band is its 89 percent interval across draws. The line wanders around zero
without a trend or a season, though it sits above zero in all but one week of
February and March 2023.

{func}`~mmmjax.plot_ppc_tstat` turns checks like these into numbers.

```{code-cell} ipython3
mj.plot_ppc_tstat(model, results, quantity="mu")
plt.show()
```

Each panel computes one statistic on every predictive draw, the black dot
marks its value for the observed revenue, and the p in each title is the share
of draws that reach it. The standard deviation and the largest week sit near
the middle of their draws.

Residual autocorrelation works a little differently. Its residuals come from
each draw's expected revenue, which `quantity` names, so the observed values
draw as a black curve instead of a dot. Its p rounds to zero, and
[Sampling and diagnostics](sampling) reads the streaks in the residuals behind
it. `statistics` takes other names, quantiles such as `0.9`, or any function
from a series of weekly revenue to a number.

## Priors and posteriors

The comparisons in this section need draws from the priors alone.
{func}`~mmmjax.sample_prior` makes them from the `priors` mapping, which
states the model's priors again for these tools.

```{code-cell} ipython3
:tags: [skip-execution]

prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

prior_results = first_model_prior_results(model, priors)
```

{func}`~mmmjax.plot_prior_posterior` overlays each parameter's prior on its
posterior.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results)
plt.show()
```

Each panel draws a parameter's prior in blue and its posterior in orange, and
the plot keeps the 12 of 45 that the data narrowed least. Eleven belong to
paid channels and one is Email's half-saturation point. Most of their
posteriors sit on their priors, so the priors carry most of what the model
says about how these channels carry over and saturate. Snapchat's return and
Branded search's retention rate are the exceptions, with posteriors that lean
further right than their priors.
[Priors](priors) reads what the data changed.

{func}`~mmmjax.psense_summary` measures how much each posterior depends on the
priors and how much on the data.

```{code-cell} ipython3
:tags: [wide-table]

mj.psense_summary(results, priors=priors, var_names=["roi"])
```

Raising the priors or the likelihood to a power a little above or below one
reweights the draws without refitting, and each value measures how far that
moves a channel's return. The results hold no log prior of their own, so
`priors` provides it, the same terms the model's `log_density` writes. The
diagnosis points to a possible conflict between prior and data for eight
channels and to a strong prior with a weak likelihood for TikTok and Generic
search. The columns after it show that the ROI prior moves each return the
most.

{func}`~mmmjax.plot_psense` shows which way each posterior moves.

```{code-cell} ipython3
mj.plot_psense(results, priors=priors)
plt.show()
```

Each row draws an element's distribution twice, with the priors raised to the
powers in the legend on the left and the likelihood raised to them on the
right. The lines below give the point estimate and 89 percent interval at each
power. For Display's and Snapchat's returns a stronger prior pulls the
interval down while a stronger likelihood pushes it up, which is what you see
when a prior sits below what the data suggest.

## Contributions

{func}`~mmmjax.plot_contributions` sums the weeks in the output of
{func}`~mmmjax.contributions` and breaks revenue down into the baseline, each
channel, and each treatment.

```{code-cell} ipython3
effects = mj.contributions(model, results, quantity="mu", by="time")
mj.plot_contributions(effects)
```

Each bar starts where the one before it ends, so the bars walk from the
baseline through every input to all of the revenue. Each label gives a share
and its total in dollars, both posterior means. The baseline holds 79.8
percent and Meta, the largest paid channel, 3.5 percent, about \$1.93 million
over the three years. Promotion adds 3.0 percent, about \$1.61 million, while
price's orange bar takes away 2.5 percent, about \$1.36 million, because it
compares each week with the lowest price the brand charged.
[Media effects](media_effects) explains how the contributions are computed.

The waterfall treats three kinds of input differently, and
[A first model](first_model) sets up the three roles.

- The paid channels and Email are removed outright, so each bar is all that
  channel adds. Email has a bar here but no return below, because nothing was
  spent on it.
- Price and promotion are treatments, levers the brand sets, so their bars
  measure the change from a baseline level, the lowest price in the data and a
  week without a promotion. `treatment_baselines` sets other levels.
- Demand and holiday are controls, which the model adjusts for without
  reporting their effect, so they stay inside the baseline.

`by="time"` keeps the weeks instead.

```{code-cell} ipython3
mj.plot_contributions(effects, by="time")
```

The light gray area is the baseline, the ten largest inputs stack on top of
it, and the other three share the darker gray area, as the caption says. The
black line is total revenue. Linear TV and Streaming arrive in flights and
promotion swells in the weeks the brand runs one, while Generic search and
Display run every week. `include_baseline=False` drops the baseline and stacks
the inputs under a line that traces what removing them all at once would
cost.

:::{admonition} The axis starts above zero
:class: note

The vertical axis starts near the lowest weekly baseline rather than at zero,
so the inputs stay visible.
:::

## Returns

The return plots read the output of {func}`~mmmjax.media_metrics`, which
covers only the ten paid channels, since a return needs spending.
{func}`~mmmjax.plot_media_metrics` draws the return on investment by default.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

Each bar is a channel's mean return on a dollar with its 89 percent interval,
and the dashed line marks break-even. Channels run from the most spending to
the least, from Meta at \$3.49 to Snapchat at \$3.98. Snapchat's interval is
the widest, because the smallest budget leaves the model the least to learn
from. [Media effects](media_effects) defines the returns.

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

The average return says how well past spending paid off, and the marginal
return says what the next dollar would earn. Here the marginal return runs
from \$1.18 for Generic search to \$2.40 for Snapchat, and each channel's sits
below its average return, since its curve flattens as it spends.

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
return as the priors (blue) or the likelihood (orange) are raised to a power.
The dashed lines mark two Monte Carlo standard errors around the unscaled
value. Snapchat's mean falls as the priors strengthen and rises as the
likelihood does, so the priors pull it down, while Generic search stays the
lowest of the three at every power.

{func}`~mmmjax.plot_roi_bubbles` sets each channel's average return against
its marginal one.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

Each bubble puts a channel's average return on the horizontal axis and its
marginal return on the vertical one, with its area proportional to the
channel's spending. Dashed lines mark break-even on both axes. Every channel
sits above and to the right of break-even. Streaming and Snapchat sit highest,
while Generic search, the second largest budget, has the lowest marginal
return.

`metric="effectiveness"` puts the incremental revenue per impression on the
vertical axis instead.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns, metric="effectiveness")
```

Effectiveness asks how much each impression does and the return asks how much
each dollar does, so the two differ by what an impression costs. Branded
search and Streaming earn the most on each impression, and Display pairs a
middling return with the lowest effectiveness.

{func}`~mmmjax.plot_spend_vs_contribution` compares each channel's share of
the spending with its share of the incremental revenue.

```{code-cell} ipython3
mj.plot_spend_vs_contribution(returns)
```

Each channel gets a hatched bar for its share of the spending and a solid one
for its share of the incremental revenue, with its return above. A solid bar
taller than its hatched one means a return above the average of all ten.
Streaming's and YouTube's solid bars stand well above their hatched ones,
while Generic search's and Linear TV's fall short. The budget below still cuts
Generic search the most and Linear TV hardly at all, because it follows the
marginal returns above rather than these averages.

## Media transformations

{func}`~mmmjax.plot_adstock` passes one unit of exposure through the model's
own adstock function for every draw and gives each channel a panel. `prior`
adds the draws of {func}`~mmmjax.sample_prior` for comparison.

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
:class: note

The function receives each draw of a parameter through the argument of the
same name, the way the model's blocks receive their inputs. `parameters` maps
an argument to a parameter you named differently, as `alpha` to the brand's
`retention` here, or fixes it at a number, as the slope below.
:::

Each panel shows the share of an exposure's effect that lands in each later
week, with the posterior in blue and the prior in orange. For most channels
the posterior sits on the prior, so the data say little about how long their
effects last. YouTube's moves toward effects that fade faster and Branded
search's toward effects that last longer.
Without `prior`, `combine=True` draws several channels in one panel so you
can compare their decay directly. The same call with `organic_retention` in
place of `retention` draws Email's carryover.

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
level of carried media. `max_input` is in the units the function receives,
here carried media on the scale the model fits. Linear TV's and
Streaming's posterior curves climb furthest above their priors, while most of
the others stay close to theirs. [A first model](first_model) defines both
transformations.

{func}`~mmmjax.plot_response_curves` shows what the curves mean in dollars.

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

Each curve follows the incremental revenue as a channel's spending runs from
zero to twice its current level, with a point at the current spending and a
dashed stretch that extrapolates past it. `combine=True` draws the channels
you name in one panel, and without `channels` it takes the five with the most
spending. Overlaid bands blur together past a few channels, so the call names
three. Meta sits furthest along its curve, where each extra dollar adds
less, while Streaming's curve still climbs steeply past its point, which is
why its marginal return above is the highest of these three.
[Media effects](media_effects) reads the curves channel by channel.

## Budgets

The budget plots read the output of {func}`~mmmjax.optimize_budget`, here a
plan that keeps each channel within 30 percent of its historical spending.
[Budget optimization](budgets) explains how the optimizer finds it.
`include_metrics=True` records the incremental revenue each channel brings
under both splits, and {func}`~mmmjax.plot_budget_response` walks from
the historical revenue to the optimized one.

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

```{code-cell} ipython3
mj.plot_budget_response(plan)
```

The first bar is the incremental revenue of the historical split and the last
is that of the optimized one. Each channel's change sits between them, with
the cuts first. Cutting Generic search costs \$172,000 and Streaming's
increase adds \$204,000. The subtitle puts the mean gain at \$134,000 with an
89 percent interval from a loss of \$170,000 to a gain of \$426,000, and the
plan gains in 77 percent of the draws.

{func}`~mmmjax.plot_budget_spend` shows the moves themselves.

```{code-cell} ipython3
mj.plot_budget_spend(plan)
```

Each bar is a channel's optimized spending minus its historical spending, in
the same colors. Generic search gives up \$126,000 and Streaming gains
\$94,100, which takes Streaming to its upper limit.

Passing the plan to {func}`~mmmjax.plot_response_curves` marks both splits
on each channel's curve.

```{code-cell} ipython3
mj.plot_response_curves(curves, plan=plan)
```

Each channel gets a panel with its own axes, a circle at the historical
spending, and a triangle at the optimized one, and the line turns dashed
outside the plan's limits. The optimizer moves spending from the flat
stretches of the curves to the steep ones until the next dollar earns about
the same everywhere or a channel reaches its limit. That's why Streaming's
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

Channels bought by reach and frequency add
{func}`~mmmjax.plot_frequency_curves`, which draws the output of
{func}`~mmmjax.frequency_curves`.
