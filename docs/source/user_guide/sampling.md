---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sampling and diagnostics

Sampling a model takes one call to {func}`~mmmjax.sample`, and deciding whether
to trust the draws takes a few more. You'll sample the ten-channel brand model
from [A first model](first_model), read its diagnostics, and compare its
predictions with the data.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model

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

## Divergences

Once {func}`~mmmjax.sample` has run the No-U-Turn sampler (NUTS) with window
adaptation, the first thing you should check is whether any transition
diverged.

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import first_model_results, stored

results = first_model_results(model)
```

```{code-cell} ipython3
int(results["sample_stats"]["diverging"].sum())
```

The count comes back zero, so none of the transitions in this fit diverged. A
divergent transition is a step where the sampler's simulated trajectory broke
down, usually in a part of the posterior that curves too sharply for the step
size. Because draws near those regions can't be trusted,
{func}`~mmmjax.sample` counts divergences in `sample_stats` and warns you when
any appear.

:::{admonition} When a fit diverges
:class: warning

Try a higher `target_accept` first, such as 0.9 or 0.95 in place of the
default of 0.8. Warmup then settles on a smaller step size, which follows a
sharply curved posterior more faithfully but needs more steps for every draw,
so the fit takes longer. If divergences persist even at a high target, the
model usually needs a change, often a reparameterization such as the
noncentered offsets on [Geo-level models](geo.md#priors).
:::

## What the results hold

```{code-cell} ipython3
results
```

What {func}`~mmmjax.sample` returns is an xarray
[DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
whose groups follow ArviZ's conventions, so ArviZ can read it as it is. You can
open each folder to see a
group's variables and labels, and the attributes at the root record how the
sampler was set up.

- `posterior` holds the draws.
- `sample_stats` holds the sampler's diagnostics.
- `observed_data` and `constant_data` hold the data the model saw.
- `posterior_predictive` and `log_likelihood` hold what `generated_quantities`
  returned under its `predictive` and `log_likelihood` keys.
- A `log_prior` key from `generated_quantities` gets a group of its own, and
  any other output lands in a `generated_quantities` group.
- `sampling_state` records where each chain stopped.

`unconstrained_posterior` holds the same draws on the scale the sampler moved
in. Keeping it costs no extra computation, since those are the positions NUTS
worked with all along.

```{code-cell} ipython3
import numpy as np

moved = results["unconstrained_posterior"]["sigma"]
print(bool(np.allclose(moved, np.log(results["posterior"]["sigma"]))))
```

The check prints `True`, so every saved noise scale is the exponential of the
position the sampler held, and retention works the same way through the logit.
The group stores a second copy of every parameter's draws, so if you don't
need it, `save_unconstrained=False` leaves it out.

:::{admonition} Where to look when chains struggle
:class: tip

Look at the unconstrained scale, since a funnel or a sharp ridge there is the
geometry NUTS faced. You can draw that group with ArviZ's plots by passing
them `group="unconstrained_posterior"`.
:::

## Convergence

[ArviZ](https://python.arviz.org/) reads `results` without any conversion,
because each of its functions on this page finds the groups it needs by name.
mmmJAX's diagnostic plots are ArviZ plots with settings that suit these models.

```{code-cell} ipython3
import arviz as az

print(az.summary(results))
```

The first four columns give each parameter's mean, standard deviation, and 89
percent interval, and the rest are diagnostics. `r_hat` compares the spread
between chains with the spread within them, and 1.00 means the four chains
agree, as they do on every row here.

`ess_bulk` and `ess_tail` estimate how many independent draws the 4,000
correlated ones are worth in the middle and the tails. A few hundred is
usually enough, and the lowest here, 1,590 for the intercept, is well past
that. The `mcse` columns give the Monte Carlo error in the mean and the
standard deviation.

{func}`~mmmjax.plot_rhat` draws every `r_hat` at once, and a glance at it
replaces a scan down the column.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Each parameter gets a box of its elements' values with a point for each, and
the dotted line marks ArviZ's limit of 1.01. The subtitle counts the values
past it, none of the 45 here. Because each parameter takes one box, a model
with hundreds of channels still fits in a single plot like this.

A trace plot shows the chains draw by draw, and with ten channels `coords`
keeps it readable.

```{code-cell} ipython3
pc = mj.plot_trace_dist(
    results,
    var_names=["roi"],
    coords={"channel": ["YouTube", "Linear TV", "Generic search"]},
    compact=False,
    aes={"color": ["channel"], "linestyle": ["chain"]},
    figure_kwargs={"figsize": (12, 9)},
)
pc.add_legend("channel")
plt.show()
```

`coords` keeps three of the ten channels, `compact=False` gives each its own
row, and `aes` gives each channel a color and each chain a line style. In all
three rows the chains' densities on the left overlap, and every trace on the
right is a flat band with no drift, so for these returns the chains agree.

A rank plot checks the chains more strictly than a trace plot does.

```{code-cell} ipython3
mj.plot_rank(
    results,
    var_names=["growth", "roi"],
    coords={"channel": ["TikTok", "Branded search", "Linear TV"]},
    col_wrap=2,
    figure_kwargs={"figsize": (12, 9)},
)
plt.show()
```

A draw's fractional rank is the share of all draws below it. Each line shows
one chain, and its height is how far the cumulative distribution of that
chain's fractional ranks strays from a uniform one. If the chains explore the
same distribution, their lines stay near zero. {func}`~mmmjax.plot_rank`
spaces the draws out first, so autocorrelation alone can't fail the test.

The p above each panel tests whether all four chains could come from one
distribution. A p below $\alpha$, the 1 percent level, fails the test, and
black dots then mark the stretches behind it.

Every panel passes here, though TikTok's return comes closest to failing. Its
chain 2 sags well below the others, and its p of 0.02 is the lowest of the
four. Among 45 elements a few small p-values turn up by chance, so check a
near miss like this one again with more draws before you change the model, as
[More draws](#more-draws) does.

## What the data pins down

Even when the chains agree, they tell you nothing about how much the data taught
the model. In the summary, Generic search's retention still spreads across most
of the range from 0 to 1, much as its Beta(2, 2) prior does, and even YouTube's,
the narrowest of the ten, keeps a standard deviation of 0.16. Beta(2, 2) has a
standard deviation of 0.22, and seven of the ten rates keep between 0.21
and 0.23, so for most channels the carryover behind every answer is still the
prior's.

[Priors](priors) compares each posterior with its prior to show what the data
did teach the model, and [Recovering the truth](recovery) checks the draws
against the simulation.

The draws also show whether the model can tell two channels apart, and a pair
plot sets them against each other.

```{code-cell} ipython3
az.plot_pair(
    results,
    var_names=["roi"],
    coords={"channel": ["Streaming", "Linear TV"]},
    visuals={"scatter": {"alpha": 0.2, "size": 4}},
    figure_kwargs={"figsize": (12, 7)},
)
plt.show()
```

Each point is one draw's pair of returns, and each return's density sits on
the diagonal. Streaming and Linear TV run their flights in the same weeks, as
[Checking the data](checking_data) shows, so the data sees little difference
between them. That's why the cloud tilts down, and a draw that credits one
channel more credits the other less. The data pins down what the two earn
together better than what either earns alone.

The fit leans toward Streaming in that trade, and
[A first model](first_model.md#returns-against-the-truth) shows Streaming's
return above the truth and Linear TV's well below it.

Give the same plot several parameters and it shows how the trend and the
treatments trade off against each other.

```{code-cell} ipython3
az.plot_pair(
    results,
    var_names=["intercept", "growth", "treatment_coefficient"],
    visuals={"scatter": {"alpha": 0.2, "size": 4}},
    figure_kwargs={"figsize": (12, 10)},
)
plt.show()
```

From the tilt of every cloud you can tell that all four parameters move as one
block. In the panel of growth against the intercept, the cloud slopes down,
since a lower start with faster growth fits the three years about as well as a
higher start with slower growth.

The narrowest cloud, promotion against price, is a ridge that climbs to the
right. Every promotion in the simulation cuts the price, as
[The example data](example_data) describes, so a larger price coefficient
takes revenue away from the promotion weeks and a larger promotion coefficient
gives it back.

Both coefficients also rise with the intercept and fall with growth, because
the list price climbs every year and can stand in for part of the trend.
Because chains cross a narrow ridge slowly, these four parameters have the
summary's lowest bulk effective sample sizes, from 1,590 for the intercept to
1,747 for promotion.

## Predictions against the data

A posterior predictive check asks whether data simulated from the fitted
model looks like the data it was fitted to. A simple version counts how often
the observed revenue falls inside the model's 90 percent predictive interval,
and {func}`~mmmjax.plot_fit` draws that interval week by week.

```{code-cell} ipython3
mj.plot_fit(model, results, ci_prob=0.9)
```

The subtitle counts 93 percent of the weeks inside the band, close to the 90
percent a well-calibrated model would give. Far fewer would tell you the model
misses patterns in revenue that its noise term can't absorb. Which weeks miss
matters as much as how many, and {func}`~mmmjax.plot_residuals` shows them.

```{code-cell} ipython3
mj.plot_residuals(model, results, ci_prob=0.9)
```

Each week's residual is the observed revenue minus the predictive draws. The
band is its 90 percent interval, so it excludes zero in exactly the weeks that
fell outside the interval of the fit plot.

When you look at the dates of those weeks, you'll find that three times they
come in neighboring pairs, above zero in early December 2022 and late March
2023 and below it in late November 2023. The line also stays below zero for
six weeks from early November to mid-December 2023.

That timing matters because the model's noise is independent from week to
week. Misses that arrive in pairs, and a residual that keeps its sign for
weeks, point to something the model leaves out that lasts longer than a week.

### Test statistics

A predictive check can also target a feature the model ought to reproduce,
such as the way revenue in one week resembles the week before. Because the
model's noise is independent from week to week, that persistence has to come
from the trend, the seasons, the controls and treatments, and the carryover of
the paid media and email. The lag-one autocorrelation measures it, and its
residual version measures what persists once each draw's expected revenue is
taken out.

```{code-cell} ipython3
mj.plot_ppc_tstat(
    model,
    results,
    statistics=["autocorrelation", "residual_autocorrelation"],
    quantity="mu",
)
plt.show()
```

{func}`~mmmjax.plot_ppc_tstat` computes each statistic for every predictive
draw and for the observed revenue. On the left, the blue curve shows the
draws' autocorrelations and the black dot the observed one. The title's p of
0.08 is the share of draws at or above it, so the model reproduces most of how
closely each week's revenue follows the last.

`quantity` makes the right panel take residuals from each draw's expected
revenue. Because expected revenue changes from draw to draw, so does the
observed value, and a black curve takes the place of the dot.

The replicated residuals center on zero, as independent noise should, while
the observed ones center near 0.2 and the p rounds to zero. That gap means the
residuals run in short streaks the model doesn't reproduce, the same
persistence you saw in the residual plot.

When neighboring weeks share part of their error, the 156 weeks carry less
independent evidence than the model assumes, so its intervals can come out
narrower than they should. A baseline that can drift, such as a Gaussian process
built with {func}`~mmmjax.prepare_hsgp`, or noise that carries part of each
week's error into the next would give those streaks a place in the model.

:::{admonition} Fit checks don't test causes
:class: warning

Every check so far compares the model with the data, so a model can pass them
all and still credit revenue to the wrong cause. To find out whether this one
does, [Recovering the truth](recovery) checks the causes against the
simulation.
:::

## More draws

{func}`~mmmjax.continue_sampling` picks up each chain where it stopped, with
the step size and mass matrix from warmup, so it adds draws without adapting
again.

```{code-cell} ipython3
:tags: [skip-execution]

more = mj.continue_sampling(model, results, draws=1000)
```

```{code-cell} ipython3
:tags: [remove-cell]

more = stored(
    "first_model_continued",
    lambda: mj.continue_sampling(model, results, draws=1000),
    groups=["posterior", "sample_stats"],
)
```

```{code-cell} ipython3
more["posterior"].sizes["draw"]
```

Each chain now holds 2,000 draws, and the second thousand continues the first
without a break. With twice the draws, you can give the near miss from the
rank plot a second look.

```{code-cell} ipython3
mj.plot_rank(
    more,
    var_names=["growth", "roi"],
    coords={"channel": ["TikTok", "Branded search", "Linear TV"]},
    col_wrap=2,
    figure_kwargs={"figsize": (12, 9)},
)
plt.show()
```

TikTok's return now passes with a p of 0.82, so its chain 2 had only wandered
for a while. The other three panels still pass, and Branded search's return,
whose p falls from 0.30 to 0.10, shows how far these values move from one run
to the next.

## New outputs from the same draws

You can also add outputs to a fit without sampling again. The analysis
functions convert their results back to revenue on their own, and a block of
your own can report dollars too. The new output here is each week's expected
revenue, the model's mean $\mu_t$ taken back through the outcome scaling,

$$
\operatorname{E}[R_t] = \bar{R} + s_R\, \mu_t.
$$

`transformed_parameters` already returns $\mu_t$ as `mu`, one of your names,
and $\bar{R}$ and $s_R$ sit inside `outcome_scaling`, a supplied name whose
`inverse_transform` applies them.

:::{admonition} Blocks work in scaled units
:class: important

A block receives the data after scaling, so `mu` holds standardized revenue
and whatever a block returns stays in those units. To report dollars, a block
asks for the supplied name `outcome_scaling` and converts, as this one does.
:::

```{code-cell} ipython3
def revenue_generated_quantities(key, outcome, outcome_scaling, mu, sigma):
    # Simulated revenue for every week, and each observed week's log likelihood.
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)

    # Each week's expected revenue in dollars, since mu holds standardized revenue.
    expected_revenue = outcome_scaling.inverse_transform(mu)

    # "predictive" and "log_likelihood" are supplied names, and expected_revenue is yours.
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
        "expected_revenue": expected_revenue,
    }


revenue_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=revenue_generated_quantities,
    generated_dims={"expected_revenue": "time"},
)
```

The new model passes the same blocks as the one from
[A first model](first_model) and swaps only `generated_quantities`. JAX
arithmetic doesn't carry axis names, so `generated_dims` labels the new output
with `time`, the week axis mmmJAX supplies.

```{code-cell} ipython3
:tags: [skip-execution]

revenue = mj.generate_quantities(revenue_model, results)
```

```{code-cell} ipython3
:tags: [remove-cell]

revenue = stored(
    "first_model_revenue",
    lambda: mj.generate_quantities(revenue_model, results),
    groups=["generated_quantities"],
)
```

```{code-cell} ipython3
expected = revenue["generated_quantities"]["expected_revenue"].sum("time").mean()
round(float(expected)), round(float(data.arrays["outcome"].sum()))
```

{func}`~mmmjax.generate_quantities` runs the new block on the draws already in
`results`, since the parameters haven't changed. Summed over the three years
and averaged over the draws, expected revenue comes back within about \$13,400
of the \$54.4 million observed.

## Sampler settings

The fit on this page keeps {func}`~mmmjax.sample`'s defaults of four chains,
1,000 warmup steps, and 1,000 draws. Chains that start from different random
points give $\hat R$ and the rank plot something to compare, and
[Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008) recommend running at
least four. The lowest bulk effective sample size, 1,590, is well past the 400
they ask for, so more draws would shrink the Monte Carlo error a little and
lengthen every analysis, because each one reruns your blocks for every draw.

Besides these, {func}`~mmmjax.sample` lets you set `target_accept`, how the
chains run, the maximum tree depth, and the mass matrix.

### Mass matrix

The default diagonal mass matrix scales each parameter on its own, while a
`"dense"` one also adapts to correlations between them, such as the ridge
between price and promotion, at the cost of more work per step.

```{code-cell} ipython3
:tags: [skip-execution]

dense = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7, mass_matrix="dense")
```

```{code-cell} ipython3
:tags: [remove-cell]

dense = stored(
    "first_model_dense",
    lambda: mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7, mass_matrix="dense"),
    groups=["posterior", "sample_stats"],
)
```

The first line counts divergences, this time for each chain, and the next two
give the leapfrog steps in an average draw's trajectory, for the dense run and
then for the diagonal one.

```{code-cell} ipython3
stats = dense["sample_stats"]
print(stats["diverging"].sum("draw").values)
print(stats["n_steps"].mean("draw").round().astype(int).values)
print(results["sample_stats"]["n_steps"].mean("draw").round().astype(int).values)
```

Chain 0 diverges three times and each of the other chains once, so the dense
run has 6 divergences where the diagonal one had none. Its trajectories are
less than half as long, 31 steps a draw against 65 to 79, because a mass
matrix that matches the correlations lets each step go further. But a longer
step is also more likely to break down where the posterior curves sharply, and
draws from a run with divergences can't be trusted as they are.

A rank plot of the dense run shows where its chains disagree.

```{code-cell} ipython3
mj.plot_rank(dense, var_names=["roi"])
plt.show()
```

Streaming's return fails the test with a p that rounds to 0.00, and black
dots mark the stretch behind the failure. Chain 1 runs above the others from
the early ranks on and peaks where the dots sit, because it holds too few of
Streaming's largest draws. The other nine returns pass, though Meta's only
narrowly, at 0.02.

```{code-cell} ipython3
print(az.summary(dense))
```

Next to the diagonal run, most estimates barely move, but the slow block from
the pair plots now mixes far better. The intercept's bulk effective sample
size rises from 1,590 to 4,581, growth's from 1,745 to 5,948, price's from
1,729 to 7,244, and promotion's from 1,747 to 7,318.

For this model, though, the gain buys little, because the diagonal run's
lowest, 1,590, was already well past the few hundred a summary needs. The
dense run also pays for its gain with divergences and a failed rank test, so
the diagonal default is the better choice here.

A dense matrix makes sense when a ridge holds the effective sample sizes near
a few hundred, and if its steps then diverge, a higher `target_accept` shortens
them.

### Running the chains

`chain_method` sets how {func}`~mmmjax.sample` runs the chains, and it takes
one of three values.

- `"sequential"`, the default, runs them one after another.
- `"vectorized"` batches every chain into one computation on a single device.
  It needs no setup, and it's the only way to run several chains at once on a
  single GPU. The batch holds every chain in memory, though, and each step
  waits for the chain with the longest trajectory, so it isn't always faster
  than running the chains in turn.
- `"parallel"` gives each chain its own device. Every GPU in a machine is
  already a device, while a CPU counts as one until the setting in
  [Installation](../getting_started/installation.md#cpu-devices-for-parallel-chains)
  asks for more before JAX runs anything. With four devices it runs four
  chains side by side, and asking for more chains than devices raises an error
  that points to the other two options.

```{code-cell} ipython3
:tags: [skip-execution]

vectorized = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7, chain_method="vectorized")
```

[Inference](inference) fits the same model without {func}`~mmmjax.sample` by
running other samplers on it and trying an approximation that skips MCMC.
