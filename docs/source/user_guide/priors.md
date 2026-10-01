---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Priors

Every prior in mmmJAX is yours to choose, because the library never picks one
for you. [A first model](first_model) writes each of its priors as a term of
`log_density` and states them again in a `priors` mapping for the tools that
draw from them. Each {class}`~mmmjax.Prior` in the mapping pairs a family from
[Distributions](distributions) with fixed settings. On this page you'll see
what those priors claim in revenue terms, check them before you fit, and
measure how far the data moves them.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_prior_results, first_model_results, stored

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

## What a prior claims

The model works on scaled data, but it states its belief about each channel
as a return on investment, the revenue a channel brings in per dollar spent
over the three years. A `Prior` can be sampled directly, so drawing from the
model's own `priors["roi"]` shows you that belief. Each row of `roi_draws`
holds one return per channel, and since every channel shares this prior, the
percentiles pool every channel's draws.

```{code-cell} ipython3
import jax
import numpy as np

roi_draws = priors["roi"].sample(jax.random.key(1), sample_shape=(10_000, len(data.channels)))
np.quantile(roi_draws, [0.05, 0.5, 0.95]).round(2)
```

Those three numbers are the 5th, 50th, and 95th percentiles, and they show
that the prior puts a channel's median return at \$2.72 of revenue per dollar
and gives a 90 percent chance that it lands between \$1.02 and \$7.29.

Returns only become revenue once you bring in what each channel spent.
Multiply each draw's returns by that spend, divide by total revenue, and you
get the share of revenue the prior expects media to explain.

```{code-cell} ipython3
spend = data.arrays["spend"].sum(axis=0)
total_revenue = data.arrays["outcome"].sum()
media_share = roi_draws @ spend / total_revenue
print(round(float(spend.sum()) / 1e6, 1), round(float(total_revenue) / 1e6, 1))
print(np.quantile(media_share, [0.05, 0.5, 0.95]).round(3))
```

The first line shows the ten channels spent \$2.9 million over the three years
against \$54.4 million of revenue. The prior's median share for media is 16.9
percent, and 90 percent of draws put it between 11.9 and 24.8 percent. If a
prior let media explain most of the revenue, you'd see it here before any fit.

Because all ten channels share this prior, it expects each channel's revenue to
grow with its spending, so before the fit the biggest budgets get the most
credit. For the channels that air together, [Checking the data](checking_data)
found that the data leaves the split of their lift to the prior.

Since the simulation records the true returns, you can see where they fall in
this prior.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

true_roi = pd.Series(brand.truth["roi"].values, index=data.channels, name="true_roi")
print(round(float(true_roi @ spend / total_revenue), 3))
true_roi.round(2)
```

Together the true returns explain 22.1 percent of revenue, inside the prior's
range but above its median. Going down the list, every paid channel's true
return sits above the prior's median of \$2.72, and YouTube's \$6.59 comes
close to the prior's 95th percentile of \$7.29. So for this brand the prior
expects less of the channels than they deliver, and the last section shows
where that decides the answer.

### Email's share

Email has no spend, so there's no return to hold a belief about. Instead the
model samples email's share of the three years' revenue, and multiplying draws
of `priors["organic_share"]` by total revenue turns that belief into dollars.
Since the simulation also records email's true contribution, you can check
this prior against the truth.

```{code-cell} ipython3
share_draws = priors["organic_share"].sample(jax.random.key(2), sample_shape=(10_000,))
share_quantiles = np.quantile(share_draws, [0.05, 0.5, 0.95])
true_share = brand.truth["contribution"].sel(channel="email").sum() / total_revenue
print(share_quantiles.round(3), (share_quantiles * total_revenue / 1e6).round(2))
print(round(float(true_share), 4))
```

The prior puts email's median share at 1.7 percent, or \$0.92 million, and
gives a 90 percent chance that it lands between 0.4 and 4.8 percent, \$0.2
million to \$2.6 million. Email's true share of 0.71 percent sits inside that
range but below the median, so this prior leans the other way and expects more
of email than it delivers.

### Price and promotion

Price and promotion are treatments, inputs the brand sets itself, and the
analyses report what each one adds to revenue over its lowest observed level.
The treatment prior is therefore a belief you'll read back in the results.

The coefficients act on standardized price and promotion, so to put a draw in
dollars of weekly revenue per dollar of price or per promotion week, divide it
by the treatment's standard deviation and multiply by the revenue scale.

```{code-cell} ipython3
import pandas as pd

treatment_draws = priors["treatment_coefficient"].sample(jax.random.key(3), sample_shape=(10_000, 2))
outcome_scale = scaling.transformations["outcome"].scale
treatment_scale = scaling.transformations["treatments"].scale
per_unit = treatment_draws * outcome_scale / treatment_scale
bounds = np.quantile(per_unit, [0.05, 0.95], axis=0)
pd.DataFrame(bounds, index=[0.05, 0.95], columns=data.columns["treatments"]).round(-2)
```

In 90 percent of draws a dollar on the price moves weekly revenue by
somewhere between a loss of \$15,600 and a gain of \$15,400, and a promotion
week moves it between a loss of \$51,600 and a gain of \$51,500. The prior
leaves the sign of both effects open and only bounds their size.

That bound matters because this brand cuts its price in promotion weeks, and
the data struggles to separate the two effects. You'll see in the last section
how much of the answer the prior decides.

### Carryover and saturation

Every paid channel and email get the same retention and half-saturation priors,
so drawing from each shows what the model expects of all of them.

```{code-cell} ipython3
carryover = priors["retention"].sample(jax.random.key(4), sample_shape=(10_000,))
saturation = priors["half_saturation"].sample(jax.random.key(5), sample_shape=(10_000,))
print(np.quantile(carryover, [0.05, 0.5, 0.95]).round(2))
print(np.quantile(saturation, [0.05, 0.5, 0.95]).round(2))
```

The first line puts the retention rate near one half, and its 90 percent range,
from 0.14 to 0.86, allows an effect mostly gone within a week or one that
lingers across the window. The second centers the half-saturation point on one,
a typical week on air once [Scaling](data.md#scaling) divides exposure by its
median, and keeps 90 percent of it between 0.44 and 2.29.

For most channels the data barely moves either prior, as
[Media transformations](plotting.md#media-transformations) on Plotting showed,
so these priors decide how long those effects last and where their curves bend.
If you know a channel's habits, give it a prior of its own, such as a lower
retention for search, since people act on a search ad in the week they see it.

## Simulating from the priors

Priors that look reasonable one at a time can still add up to an implausible
model. A prior predictive check catches that by simulating revenue from the
priors alone, before the model sees any data, and {func}`~mmmjax.sample_prior`
runs one with the `priors` mapping.

:::{admonition} Keys match your parameter names
:class: important

Because `sample_prior` pairs each key of `priors` with a parameter by name, the
keys have to be the names you declared in `parameters`, and a missing or extra
key raises an error. Those names are yours rather than supplied ones, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-blocks-get-their-inputs)
explains.
:::

```{code-cell} ipython3
:tags: [skip-execution]

prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

prior_results = first_model_prior_results(model, priors)
```

`sample_prior` draws each parameter from its prior and runs
`transformed_parameters` and `generated_quantities` on those draws. What
`generated_quantities` returns under the supplied name `"predictive"` lands in
the `prior_predictive` group as `"outcome"`, one simulated revenue series per
draw.

{func}`~mmmjax.plot_ppc_dist` puts those series back in dollars and draws
their distributions with [ArviZ](https://python.arviz.org/). On a prior check
it leaves the observed curve out unless `visuals` asks for it.

```{code-cell} ipython3
mj.plot_ppc_dist(model, prior_results, group="prior", visuals={"observed_dist": {}})
plt.show()
```

Each blue curve is the distribution of weekly revenue in one prior draw, and
the black curve is the observed distribution. The black curve sits inside the
bundle, a little left of its center, and that's what you want, because a
bundle that missed it would call for different priors.

The blue curves also spread much wider, and several reach below zero. A wider
bundle is expected, since that's what weakly informative priors are for, but
revenue can't fall below zero.

{func}`~mmmjax.plot_ppc_tstat` compares each draw's lowest and highest week
with the observed ones.

```{code-cell} ipython3
mj.plot_ppc_tstat(model, prior_results, group="prior", statistics=["min", "max"])
plt.show()
```

The black dots are the observed values, and each title's p is the share of
draws at or above its dot. Only 27 percent of draws keep their lowest week at
or above the observed lowest, and 96 percent push their highest week past the
observed highest, so the prior is wider than the data on both sides.

Part of the left curve lies below zero, and each draw there simulates at least
one week of negative revenue.

### Where the negative weeks come from

To count the negative weeks, `prior_revenue` puts the draws back in dollars
with the scaling's `inverse_transform`, and `negative` marks the draws with at
least one week below zero.

```{code-cell} ipython3
def prior_revenue(draws):
    prediction = draws["prior_predictive"]["outcome"].values
    revenue = np.asarray(scaling.transformations["outcome"].inverse_transform(prediction))
    return revenue


revenue = prior_revenue(prior_results)
negative = (revenue < 0).any(axis=-1).ravel()
print(round(float((revenue < 0).mean()), 4), round(float(negative.mean()), 3))
```

Only 1.5 percent of the simulated weeks are negative, but 26.6 percent of
draws contain at least one of them.

Grouping the draws by `negative` shows which priors let those weeks in. To
give each draw a single row, `sizes` averages the absolute value of each
parameter over its channels, controls, treatments, or seasonal terms.

```{code-cell} ipython3
:tags: [wide-table]

# The four seasonal coefficients have no axis from the data, so the draws number theirs.
axes = ["annual_coefficients_dim_0", "channel", "organic_channel", "control", "treatment"]
sizes = abs(prior_results["prior"].to_dataset()).mean(axes)
sizes.to_dataframe().groupby(negative).mean().rename_axis("negative_week").round(2)
```

The `True` row holds the draws with a negative week, and next to the `False`
row, their noise scale averages 0.99 against 0.75, their control coefficients
0.97 in size against 0.71, and their growth 0.91 in size against 0.73. The other
columns differ far less, so the negative weeks come from the priors on the
noise scale, the two control coefficients, and the trend's growth.

### Tighter priors

You can test that reading by halving the scale of those three priors, without
touching the fitted model. The dictionary union `priors | {...}` copies the
mapping with those three entries replaced, so the original `priors` stays as
it was.

```{code-cell} ipython3
tighter = priors | {
    "growth": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "sigma": mj.Prior(mj.half_normal, scale=0.5),
}
```

```{code-cell} ipython3
:tags: [skip-execution]

tighter_results = mj.sample_prior(model, tighter, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

tighter_results = stored(
    "tighter_prior",
    lambda: mj.sample_prior(model, tighter, draws=500, seed=0),
    groups=["prior_predictive"],
)
```

```{code-cell} ipython3
tighter_revenue = prior_revenue(tighter_results)
tighter_negative = (tighter_revenue < 0).any(axis=-1)
print(round(float((tighter_revenue < 0).mean()), 4), round(float(tighter_negative.mean()), 3))
```

With the three tighter priors, 0.18 percent of simulated weeks are negative
and 5 percent of draws contain one.

The prior check can't tell you whether to adopt the tighter priors, though.
They remove most of the impossible weeks, but they also claim more about how
far each control and the trend can move revenue. You only learn whether the
data agrees after a fit, from the prior sensitivity checks on
[Plotting](plotting.md#prior-sensitivity). The rest of the guide keeps the
original priors so that every page shares one fit.

## Checking the mapping against the density

The first model writes each prior twice, once as a term of `log_density` and
once in `priors`, so check that the two agree before you trust what
`sample_prior` shows.

:::{admonition} Keep the two in sync
:class: warning

`sample_prior` never calls `log_density`, so nothing forces the mapping you
pass it to match the density, and nothing raises an error when they differ.
The prior predictive check, {func}`~mmmjax.psense_summary`, and
{func}`~mmmjax.plot_psense` all read the mapping, so if you change a prior in
one place and not the other, each of them describes a model you never fit.
:::

{meth}`~mmmjax.Model.log_prob` evaluates `log_density` at a set of parameter
values, so at any draw it should equal the sum of the mapping's terms and the
likelihood. The cell below compares the two at five draws from `sample_prior`
and adds the same sum under `tighter` to show what a mismatch looks like.

Inside the cell, the likelihood scores the standardized revenue from the
`observed_data` group. {meth}`~mmmjax.Model.evaluate` hands back the keys
`transformed_parameters` returns, so `"mu"` is your name for expected revenue
rather than a supplied one.

```{code-cell} ipython3
outcome = prior_results["observed_data"]["outcome"].values


def mapping_log_prob(mapping, values):
    prior_terms = sum(prior(values[name]) for name, prior in mapping.items())
    likelihood = mj.normal(outcome, model.evaluate(values)["mu"], values["sigma"])
    total = prior_terms + likelihood
    return float(total)


rows = []
for draw in range(5):
    values = {name: prior_results["prior"][name].values[0, draw] for name in parameters}
    rows.append(
        {
            "log_prob": float(model.log_prob(values)),
            "priors": mapping_log_prob(priors, values),
            "tighter": mapping_log_prob(tighter, values),
        }
    )
pd.DataFrame(rows).round(2)
```

`log_prob` and `priors` match at every draw, so the mapping states the priors
the sampler fit. `tighter` misses at every draw, by 4.65 at the first, and
that gap is what a prior changed in only one place leaves behind. To adopt the
tighter priors, change them in both places and check again before you refit.

1. Replace the growth, control coefficient, and noise scale terms in
   `log_density`.
2. Replace the same three entries in `priors`.
3. Rerun the check above and confirm that `log_prob` and `priors` match.
4. Refit the model.

## Checking one parameter

{func}`~mmmjax.check_prior` reports how much prior mass falls outside limits
you consider plausible. A return below one means a channel doesn't earn back
what it costs, so break-even makes a natural lower limit for the ROI draws.

```{code-cell} ipython3
checked = mj.check_prior(prior_results["prior"]["roi"], lower=1.0)
checked["probability_below"].to_series()
```

The prior gives each channel between a 3.4 and a 7.2 percent chance of losing
money. Every channel shares one prior, so the differences between channels are
noise from 500 draws.

:::{admonition} Hierarchies and new families
:class: tip

When one prior depends on another, as in a hierarchy, `sample_prior` also
accepts a function that takes a random key and returns one draw of every
parameter. For a family the library lacks, [User-defined functions](functions)
adds one with {func}`~mmmjax.custom_distribution`.
:::

## What the data changed

Once the model is fitted, comparing each prior with its posterior shows how
much the data taught the model. `results` is the fit from [A first
model](first_model), and {func}`~mmmjax.plot_prior_posterior` plots its
posterior against the prior draws from `sample_prior`. `var_names` keeps the ten
returns, and `col_wrap` puts five panels in a row.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results, var_names=["roi"], col_wrap=5)
plt.show()
```

Each panel draws a channel's prior in blue and its posterior in orange. Where
an orange curve moves away from its blue one, or narrows, the data taught the
model something about that return. YouTube's and Streaming's orange curves
shift right of their blue ones and Snapchat's spreads a little further right,
while most of the other seven hardly move from their priors.

To put numbers on what the plot shows, {func}`~mmmjax.media_metrics` measures
each channel's return by removing the channel from the model, and
`quantity="mu"` points it at expected revenue. Because the model states each
return through {func}`~mmmjax.roi_coefficient`, these returns match the `roi`
draws above.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
prior_returns = mj.media_metrics(model, prior_results, quantity="mu", group="prior")
mj.plot_media_metrics({"Prior": prior_returns, "Posterior": returns})
```

Each bar marks a mean return with its 89 percent interval, the prior's in blue
and the posterior's in orange. The blue bars read between \$3.06 and \$3.36,
above the prior's median of \$2.72, because the prior's long right tail pulls
a mean up.

Only Linear TV's and Generic search's orange intervals are clearly shorter
than their blue ones, and both means drop, to \$2.79 and \$3.07. Snapchat's
mean moves up instead, to \$3.98 with an interval longer than its prior's, so
the data favors a higher return without pinning it down.

YouTube and Streaming move up as well, to \$4.06 and \$4.04. The other five
keep means between \$3.01 and \$3.49 and intervals about as long as their
priors', so for them the answer still comes mostly from the prior.

### Why the returns lean low

When [A first model](first_model.md#returns-against-the-truth) sets these
returns beside the true ones, eight of the ten posterior means sit below the
truth. That's what you'd expect from a prior whose median of \$2.72 sits below
every true return, as the first section showed, because where the data says
little, a posterior stays close to its prior.

The data says little here partly because several channels air together, and
the data sees their combined effect far better than each channel's part of it,
as [Checking the data](checking_data) shows. So the job of splitting that
combined effect among the channels falls mostly to the ROI prior. Streaming
airs in the same weeks as TikTok, Influencer, and Linear TV, and the split
leaves it above its truth and all three below theirs.

Linear TV shows that a narrower posterior can still be a wrong one. The data
narrows its interval more than any other channel's, but around a mean well
below the truth.

YouTube airs on its own calendar, and the data lifts its whole interval.
Even so, its true return sits near the prior's 95th percentile, so its mean
still falls well short of the truth.

### Email's share and the treatments

The first section also looked at email's share and the two treatment
coefficients, and {func}`~mmmjax.plot_prior_posterior` can show what the data
did to them.

```{code-cell} ipython3
mj.plot_prior_posterior(results, prior_results, var_names=["organic_share", "treatment_coefficient"])
plt.show()
```

Email's orange curve lies close to its blue one, with a slightly shorter right
tail, so its share is mostly the prior's. Because email goes out in the same
weeks as the flight channels, the data can't tell its revenue from theirs. So
the fit credits email with about what its prior expects, more than the true
0.71 percent from the first section.

Price's orange curve narrows and sits a little left of zero, but it still
spans zero, so the fit can't say whether a higher price loses revenue.
Promotion's curve moves right, and most of it lies above zero.

Behind both curves is the same problem the channels had, because the brand
cuts its price in every promotion week. As a result the data sees a promotion
week's price cut and flag together far better than either part, and the
treatment prior does the splitting.

:::{admonition} Priors that decide the answer
:class: warning

Where a posterior repeats its prior, as email's share largely does here, the
answer comes from the prior, so that prior needs the most thought.
:::

[Recovering the truth](recovery) checks the model's other answers against the
simulation, and [Plotting](plotting) finds the parameters the data narrowed
least across the whole model and measures how far each return depends on its
prior.
