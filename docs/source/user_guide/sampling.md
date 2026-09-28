---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sampling and diagnostics

Sampling a model takes one call to {func}`~mmmjax.sample`, and deciding whether
to trust the draws takes a few more. This page samples the ten-channel brand
model from [A first model](first_model), reads its diagnostics, and compares
its predictions with the data.

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

{func}`~mmmjax.sample` runs the No-U-Turn sampler with window adaptation.
After a fit, check first whether any transition diverged.

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

None did. A divergent transition is a step where the sampler's simulated
trajectory broke down, usually in a part of the posterior that curves too
sharply for the step size. Draws near those regions can't be trusted, so
{func}`~mmmjax.sample` counts them in `sample_stats` and warns when any
appear.

:::{admonition} When a fit diverges
:class: warning

Try a higher `target_accept` first, since it shrinks the step size. If
divergences persist even at a high target, the model usually needs a change,
often a reparameterization that gives the sampler an easier shape to explore.
:::

## What the results hold

```{code-cell} ipython3
results
```

{func}`~mmmjax.sample` returns an xarray
[DataTree](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
whose groups follow ArviZ's conventions. Each folder opens to show a group's
variables and labels, and the attributes at the root record how the sampler
was set up.

- `posterior` holds the draws.
- `sample_stats` holds the sampler's diagnostics.
- `observed_data` and `constant_data` hold the data the model saw.
- `posterior_predictive` and `log_likelihood` hold what `generated_quantities`
  returned under its `predictive` and `log_likelihood` keys.
- A `log_prior` key from `generated_quantities` gets a group of its own, and
  any other output lands in a `generated_quantities` group.
- `sampling_state` records where each chain stopped.

`unconstrained_posterior` holds the same draws on the scale the sampler moved
in. It costs no extra computation, since those are the positions NUTS worked
with all along.

```{code-cell} ipython3
import numpy as np

moved = results["unconstrained_posterior"]["sigma"]
print(bool(np.allclose(moved, np.log(results["posterior"]["sigma"]))))
```

Every saved noise scale is the exponential of the position the sampler held,
and retention works the same way through the logit. `save_unconstrained=False`
leaves the group out.

:::{admonition} Where to look when chains struggle
:class: tip

Look at the unconstrained scale, since a funnel or a sharp ridge there is the
geometry NUTS faced. ArviZ's plots read that group with
`group="unconstrained_posterior"`.
:::

## Convergence

[ArviZ](https://python.arviz.org/) reads the results directly, and each of
its functions on this page takes `results` as it is and finds the groups it
needs by name. mmmJAX's diagnostic plots are ArviZ's, drawn with settings that
suit these models.

```{code-cell} ipython3
import arviz as az

print(az.summary(results))
```

Each row gives a parameter's mean, standard deviation, and 89 percent
interval before the diagnostics. `r_hat` compares the spread between chains
with the spread within them, and 1.00 means the four chains agree, as they do
on every row here. `ess_bulk` and `ess_tail` estimate how many independent
draws the 4,000 correlated ones are worth in the middle and the tails. A few
hundred is usually enough, and the lowest here, 1,590 for the intercept, is
well past that. The `mcse` columns give the Monte Carlo error in the mean and
the standard deviation. {func}`~mmmjax.plot_rhat` draws every `r_hat` at once.

```{code-cell} ipython3
mj.plot_rhat(results)
```

Each parameter gets a box of its elements' values with a point for each, and
the subtitle counts the values past ArviZ's limit of 1.01, none of the 45
here. A model with hundreds of channels still fits one such plot. A trace plot
shows the chains draw by draw, and with ten channels `coords` keeps it
readable.

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

Each row shows one channel's return, its density on the left and its draws on
the right. `coords` keeps three of the ten channels, `compact=False` gives each
its own row, and `aes` gives each channel a color and each chain a line style.
The chains' densities overlap, and every trace is a flat band with no drift.

Chains that agree say nothing about how much the data taught the model.
Generic search's retention still spreads across most of the range from 0 to
1, much as its Beta(2, 2) prior does, and even YouTube's, the narrowest of the
ten in the summary, keeps a standard deviation of 0.16. [Priors](priors)
compares each posterior with its prior, and [Recovering the truth](recovery)
checks the draws against the simulation. A rank plot checks the chains more
strictly.

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

Each line shows one chain, and its height is how far the cumulative
distribution of that chain's fractional ranks strays from a uniform one. A
draw's fractional rank is the share of all draws below it. Chains exploring
the same distribution stay near zero, and {func}`~mmmjax.plot_rank` spaces the
draws out first so that autocorrelation alone can't fail the test. The p
above each panel tests whether all four chains could come from one
distribution. A p below $\alpha$, the 1 percent level, fails the test, and
black dots then mark the stretches behind it.

None fails here. Chain 2 sags well below the others for TikTok's return, which
gives a p of 0.02, the closest of the four to failing. Among 45 elements a few
small p-values turn up by chance, so check a near miss like this one again
with more draws before you change the model, as [More draws](#more-draws)
does. The draws also show whether the model can tell two channels apart, and
a pair plot sets them against each other.

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
[Data and scaling](data.md) shows, so the data sees little difference between
them. The cloud tilts down, and a draw that credits one channel more credits
the other less. The data pins down what the two earn together better than
what either earns alone. The same plot takes several parameters at once.

```{code-cell} ipython3
az.plot_pair(
    results,
    var_names=["intercept", "growth", "treatment_coefficient"],
    visuals={"scatter": {"alpha": 0.2, "size": 4}},
    figure_kwargs={"figsize": (12, 10)},
)
plt.show()
```

The four parameters move as one block. The intercept falls as growth rises,
since a lower start with faster growth fits the three years about as well as
a higher start with slower growth.

The two treatments' coefficients lie along a narrow ridge that climbs to the
right. Every promotion in the simulation cuts the price, as
[The example data](example_data) describes, so a larger price coefficient
takes revenue away from the promotion weeks and a larger promotion coefficient
gives it back. The data settles their combined effect
on those weeks far better than either one alone. Both coefficients also rise
with the intercept and fall with growth, because the list price climbs every
year and can stand in for part of the trend. Chains cross a narrow ridge
slowly, and you can see it in the summary, where these four parameters have
the lowest bulk effective sample sizes, from 1,590 for the intercept to 1,747
for promotion.

## Predictions against the data

A posterior predictive check asks whether data simulated from the fitted
model looks like the data it was fitted to. A simple version counts how often
the observed revenue falls inside the model's 90 percent predictive interval,
and {func}`~mmmjax.plot_fit` draws that interval week by week.

```{code-cell} ipython3
mj.plot_fit(model, results, ci_prob=0.9)
```

The band shows each week's 90 percent predictive interval, the blue line its
mean, and the black line the observed revenue, all in dollars because the plot
undoes the outcome scaling first. The subtitle counts 93 percent of the weeks
inside, close to the 90 percent a well-calibrated model would give, next to an
$R^2$ of 0.91 and a weighted mean absolute percentage error of 4.1 percent.
Far fewer inside would mean the model misses patterns in revenue that its
noise term can't absorb. Which weeks miss matters as much as how many, and
{func}`~mmmjax.plot_residuals` shows them.

```{code-cell} ipython3
mj.plot_residuals(model, results, ci_prob=0.9)
```

Each week's residual is the observed revenue minus the predictive draws, and
the band is its 90 percent interval, so it excludes zero in exactly the weeks
that fell outside the interval of the fit plot. Three times those weeks come
in neighboring pairs, above zero in early December 2022 and late March 2023
and below it in late November 2023, and the line stays below zero for six
weeks from early November to mid-December 2023. The model's noise is
independent from week to week, so misses that arrive in pairs, and a residual
that keeps its sign for weeks, point to something the model leaves out that
lasts longer than a week.

A predictive check can also target a feature the model ought to reproduce.
Revenue in one week resembles the week before, and because the model's noise
is independent from week to week, that persistence has to come from the
trend, the seasons, the controls and treatments, and the carryover of the paid
media and email. The lag-one autocorrelation measures it, and its residual
version measures what persists once each draw's expected revenue is taken out.

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
draw and for the observed revenue, in dollars because it undoes the outcome
scaling first. On the left, the curve shows the draws' autocorrelations and
the black dot the observed one. The title's p of 0.08 is the share of draws at
or above it, about eight in a hundred, so the model reproduces most of how
closely each week's revenue follows the last.

The right panel takes residuals from each draw's expected revenue, which
`quantity` names, so the observed value changes from draw to draw and a black
curve takes the place of the dot. The replicated residuals center on zero, as
independent noise should, while the observed ones center near 0.2 and the p
rounds to zero. The residuals run in short streaks the model doesn't
reproduce, the same persistence the residual plot showed. `statistics` also
takes your own function from one series to a number, such as the largest week
or the share of weeks above a target.

The pointwise log likelihood supports a sharper check. Leave-one-out
cross-validation asks how well the model predicts each week when that week is
left out of the fit, and ArviZ estimates it from the draws already in hand
instead of refitting the model 156 times.

```{code-cell} ipython3
az.loo(results)
```

`elpd_loo` adds up the log predictive density of each week left out. The
number only means something next to another version's, which is how [Changing
the model](changing) uses it. `p_loo` estimates the effective number of
parameters. At about 15 it sits well under the 45 the model declares, since
channels that move together act like fewer parameters and the priors hold many
of the rest in place. A value well above the declared count would point to a
misspecified model. The Pareto $k$ values check the shortcut behind the
estimate, and all 156 weeks fall in the good range. The same shortcut gives a
calibration check.

```{code-cell} ipython3
az.plot_loo_pit(results, figure_kwargs={"figsize": (12, 5)})
plt.show()
```

LOO-PIT, the leave-one-out probability integral transform, is the probability
that a prediction made without a given week falls below that week's observed
revenue. In a calibrated model these values are uniform, and the line shows
how far their cumulative distribution strays from the uniform one. It stays
within 0.05 of zero, and the test in the corner gives p = 0.46, so the wiggles
are no larger than uniform values would show by chance. ArviZ took the log
likelihood, the predictive draws, and the observed data for this check from
the same `results`.

LOO-PIT pools the weeks, so a model that misses something lasting longer than
a week can still pass it, as this one does. The residual autocorrelation keeps
the weeks in order and catches what LOO-PIT lets through.

:::{admonition} Fit checks don't test causes
:class: important

Every check so far compares the model with the data. A model can pass them all
and still credit revenue to the wrong cause, and [Recovering the
truth](recovery) checks the causes against the simulation.
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

Each chain now holds 2,000 draws, the second thousand continuing the first
without a break. The near miss from the rank plot gets a second look.

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

A fit can also gain outputs without sampling again. The analysis functions
convert their results back to revenue on their own, and a block of your own
can report dollars too. The new output here is each week's expected revenue,
the model's mean $\mu_t$ taken back through the outcome scaling,

$$
\operatorname{E}[R_t] = \bar{R} + s_R\, \mu_t.
$$

$\mu_t$ is `mu`, one of your names, which `transformed_parameters` returns.
$\bar{R}$ and $s_R$ sit inside `outcome_scaling`, a supplied name whose
`inverse_transform` applies them.

:::{admonition} Blocks work in scaled units
:class: important

A block receives the data after scaling, so `mu` holds standardized revenue
and whatever a block returns stays in those units. To report dollars, a block
asks for the supplied name `outcome_scaling` and converts, as this one does.
:::

```{code-cell} ipython3
def revenue_generated_quantities(key, outcome, outcome_scaling, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
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

The new model passes the same blocks as the one from
[A first model](first_model) and swaps only `generated_quantities`.
{func}`~mmmjax.generate_quantities` runs the new block on the draws already in
`results`, which works because the parameters haven't changed. The expected
revenue over the three years comes back within about \$13,400 of the
\$54.4 million observed. JAX arithmetic doesn't carry axis names, so
`generated_dims` labels the new output with `time`, the week axis mmmJAX
supplies.

## Sampler settings

Besides `target_accept`, {func}`~mmmjax.sample` sets the number of chains and
how they run, the warmup length, the maximum tree depth, and the mass matrix.
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

The first line counts divergences, this time for each chain. The next two give
the leapfrog steps in an average draw's trajectory, for the dense run and then
for the diagonal one.

```{code-cell} ipython3
stats = dense["sample_stats"]
print(stats["diverging"].sum("draw").values)
print(stats["n_steps"].mean("draw").round().astype(int).values)
print(results["sample_stats"]["n_steps"].mean("draw").round().astype(int).values)
```

Chain 0 diverges three times and each of the other chains once, so the dense
run has 6 divergences where the diagonal one had none. Its trajectories are
less than half as long, 31 steps a draw against 65 to 79, because a mass
matrix that matches the correlations lets each step go further. A longer step
is also more likely to break down where the posterior curves sharply, and
draws from a run with divergences can't be trusted as they are. A rank plot
shows where the chains disagree.

```{code-cell} ipython3
mj.plot_rank(dense, var_names=["roi"])
plt.show()
```

Streaming's return fails the test with a p that rounds to 0.00, and black
dots mark the stretch behind the failure. Chain 1 runs above the others from
the early ranks on and peaks where the dots sit, since it holds too few of
Streaming's largest draws. The other nine returns pass, Meta's narrowly at
0.02.

```{code-cell} ipython3
print(az.summary(dense))
```

Most estimates barely move. The slow block from the pair plots mixes far
better, and the bulk effective sample sizes of the intercept, growth, price,
and promotion rise from 1,590, 1,745, 1,729, and 1,747 to 4,581, 5,948, 7,244,
and 7,318. Most other rows rise too, and the three that fall drop only a
little, Email's share the most, from 2,894 to 2,477. The diagonal run's
lowest, 1,590, was already well past the few hundred a summary needs, so the
gain buys little, and the dense run pays for it with divergences and a failed
rank test. For this model the diagonal default is the better choice. A dense
matrix makes sense when a ridge holds the effective sample sizes near a few
hundred, and if its steps then diverge, a higher `target_accept` shortens
them.

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

{func}`~mmmjax.sample` isn't the only way to fit the model, and
[Inference](inference) runs other samplers on it.
