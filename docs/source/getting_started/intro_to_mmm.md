---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Introduction to MMM

A marketing mix model (MMM) explains an outcome such as weekly sales through a
baseline, the effects of media channels, and other drivers like price. Fit to
a few years of weekly history, it tells you what each channel added, what a
dollar of spending returned, and how a budget might be split. This page walks
you through the math behind those answers and the Bayesian methods that
produce them. [What is mmmJAX](what_is_mmmjax) then shows how to write one as
an mmmJAX program.

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
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["figure.dpi"] = 100
```

## What an MMM estimates

When you ask an MMM what a channel did, you're asking a causal question. The
channel's effect compares the outcome with its spending against the outcome
without it, but each week shows you only one of the two. This is known as the
fundamental problem of causal inference ([Holland,
1986](https://doi.org/10.1080/01621459.1986.10478354)).

The model has to predict the outcome you don't see from observational data
rather than an experiment, so its estimates are causal only under the
assumptions that [What the data cannot settle](#what-the-data-cannot-settle)
lays out
([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
sections 1 and 2.2).

The potential outcomes framework ([Rubin,
1974](https://doi.org/10.1037/h0037350)) lets you state the missing outcome
precisely. Write $\mu_t(x)$ for the expected outcome in week $t$ when the
channels' exposures over all weeks, such as impressions, are $x$ and every
other input keeps its observed value. If you set channel $c$'s exposure to zero
from week $T_0$ to week $T_1$, you get $x^{(-c)}$, and the channel's
incremental outcome is

$$
\Delta_c = \sum_{t=T_0}^{T_1 + L} \big( \mu_t(x) - \mu_t(x^{(-c)}) \big).
$$

The sum runs $L$ weeks past the window, the longest carryover the model allows,
because exposure keeps working after the window ends, and
[Carryover](#carryover) below shows how the model captures that delayed effect.

### ROI and marginal ROI

Dividing $\Delta_c$ by the spend $S_c$ in the window gives the return on
investment, while the marginal return asks what one percent more spend in the
window would bring
([Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
eqs. 10 and 11),

$$
\text{ROI}_c = \frac{\Delta_c}{S_c},
\qquad
\text{mROI}_c = \frac{\sum_{t=T_0}^{T_1 + L} \big( \mu_t(x^{(1.01, c)}) - \mu_t(x) \big)}{0.01\, S_c},
$$

where $x^{(1.01, c)}$ scales channel $c$'s exposure in the window by 1.01. Jin
et al. and Chan and Perry call this return ROAS, and mmmJAX calls it ROI.
Neither subtracts the spend, so for a revenue outcome a return of one means the
channel brought in what it cost. When your outcome is units sold, the return
counts units per dollar, and its inverse is the cost per incremental unit.

{func}`~mmmjax.media_metrics` computes all three, but by default it counts only
the weeks you supply, so carryover past the last week goes uncounted.
[Media effects](../user_guide/media_effects) works through all three for the
User Guide's brand and shows what each one says about its channels.

:::{admonition} Compute returns draw by draw
:class: important

A Bayesian fit gives you these returns for every posterior draw. Jin et al.
warn against plugging the parameters' posterior means into the formulas,
because a return computed from average parameters generally isn't the
average return.
:::

### Response curves

If you scale the window's exposure by any factor $\omega$, the incremental
outcome traces channel $c$'s response curve against the spend $\omega S_c$,

$$
\Delta_c(\omega) = \sum_{t=T_0}^{T_1 + L} \big( \mu_t(x^{(\omega, c)}) - \mu_t(x^{(-c)}) \big).
$$

Because every week's exposure scales by the same factor and the sum counts
carryover, this isn't the one-week Hill curve of [Diminishing
returns](#diminishing-returns). The ROI is the slope of the line from the origin
to the curve at $\omega = 1$, and the mROI is the curve's slope there. Spend and
exposure scale together along the curve, so both the ROI and the mROI assume
each unit of exposure keeps its historical average cost.

{func}`~mmmjax.response_curves` draws these curves, and
{func}`~mmmjax.optimize_budget` moves spend toward the steeper ones. At the
optimum of its default objective every channel not held at a limit has the
same marginal return, since otherwise moving a dollar from a flatter curve to a
steeper one would raise the expected outcome. The optimizer keeps each
channel's pattern across weeks and regions
([Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
section 4.2).

## The model equation

An MMM is a regression of the outcome on a baseline, a term for each channel,
and a term for each control ([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
eq. 3; [Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
eq. 7). You can write its general form as

$$
y_t \sim p(y_t \mid \mu_t, \psi),
\qquad
g(\mu_t) = b_t + \sum_{c=1}^{C} f_c\big(x_{t-L,c}, \dots, x_{tc};\, \lambda_c\big) + \sum_{j=1}^{J} \gamma_j z_{tj}.
$$

Here $y_t$ is the outcome in week $t$, $b_t$ is the baseline described below,
$C$ and $J$ count the channels and controls, and
$\mu_t = \operatorname{E}[y_t]$ is the expected outcome $\mu_t(x)$ above. The
observation model $p$ describes the scatter around $\mu_t$ through nuisance
parameters $\psi$ such as a noise scale. The link $g$ is the identity for an
additive model, and a log link makes the terms multiply.

Channel $c$'s response $f_c$, with parameters $\lambda_c$ that may vary with
$t$, maps its exposure $x_{tc}$ in week $t$ and the $L$ weeks before to an
effect. That effect is usually nonnegative, and it's zero when all those
exposures are zero. Control $j$'s term is simpler, since it enters linearly
through $z_{tj}$ with coefficient $\gamma_j$.

The common choice for $f_c$ passes exposure through carryover and then
saturation,

$$
f_c = \beta_c h_{tc},
\qquad
h_{tc} = \operatorname{Hill}\big(\operatorname{Adstock}(x_{t-L,c}, \dots, x_{tc};\, \rho_c);\, \kappa_c, s_c\big),
$$

where $\rho_c$ is the retention, $\kappa_c$ the half-saturation point, $s_c$
the slope, and $\beta_c$ a coefficient that scales the result to the outcome,
so $\lambda_c = (\beta_c, \rho_c, \kappa_c, s_c)$. In the textbook MMM, with an
identity link and Normal noise, channel $c$ adds $\beta_c h_{tc}$ to week $t$.
Under any other link a channel's effect exists only as the difference in
$\mu_t$ defined above, and the channels' effects no longer add up to the
media's total.

Some channels report how many people saw their ads and how often, rather than
impressions. For those, $f_c$ can saturate in frequency and scale with reach,
as in the reach and frequency model of [Zhang et al.
(2023)](https://storage.googleapis.com/gweb-research2023-media/pubtools/7327.pdf),
and {func}`~mmmjax.reach_frequency_response` computes that response.

### The baseline

The baseline $b_t$ can be any function of time that absorbs trend,
seasonality, and holidays, but each form assumes something different about how
time moves the outcome.

- A Fourier series assumes a smooth yearly cycle.
- Indicators assume a fixed effect in the flagged weeks.
- A spline assumes a path that can turn at each knot.
- A Gaussian process allows any smooth path but reverts to its prior mean out
  of sample.

The choice matters because budgets often follow the seasons of demand, so time
is a confounder
([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
section 4.2.2). A more flexible $b_t$ trades bias in the media effects for
variance, and a national channel that moves with time can lose its effect to
the baseline.

mmmJAX doesn't build in any part of this equation, so you write $\mu_t$ in
`transformed_parameters` and the observation model in `log_density` yourself.
If you want a Fourier season or a Gaussian process,
{func}`~mmmjax.fourier_features` and {func}`~mmmjax.prepare_hsgp` build one.
[A first model](../user_guide/first_model) writes out the specification the
User Guide fits, and [Changing the model](../user_guide/changing) lets its
media effect drift over time through a Gaussian process.

:::{admonition} Where the analyses find the mean
:class: note

The analysis functions read $\mu_t$ itself, never $g(\mu_t)$, and difference it
draw by draw, so they work under any link. Because $\mu_t$ goes by
[one of your names](what_is_mmmjax.md#how-blocks-get-their-inputs), such as
`mu`, you pass that name to them as `quantity`. They read only what
`transformed_parameters` returns, so if you compute $\mu_t$ inside
`log_density`, it's out of their reach.
:::

## Carryover

Advertising keeps working after the week it runs, so a model that credits only
the current week misses that lag ([Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
sections 1 and 2.1). Adstock replaces each week's exposure with a weighted
average of it and the $L$ weeks before it,

$$
\operatorname{Adstock}(x_{t-L}, \dots, x_t;\, \rho)
= \frac{\sum_{\ell=0}^{L} w_\ell\, x_{t-\ell}}{\sum_{\ell=0}^{L} w_\ell},
\qquad
w_\ell = \rho^{\ell}
\quad \text{or} \quad
w_\ell = \rho^{(\ell - \theta)^2}.
$$

The first set of weights decays geometrically, so retention $\rho$ is the share
of the effect that survives from one week to the next. The second set instead
peaks at a delay of $\theta$ weeks. Dividing by their sum spreads each week's
exposure over time without changing its total. In mmmJAX, `max_lag` sets
that $L$, the number of earlier weeks.

A single week of exposure followed by silence shows the weights directly. The
figure below runs three retention rates through
{func}`~mmmjax.geometric_adstock` and one delayed curve through
{func}`~mmmjax.delayed_adstock`.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

import mmmjax as mj

max_lag = 12
pulse = np.zeros((max_lag + 1, 3))
pulse[0] = 1.0
retention = np.array([0.3, 0.6, 0.9])
geometric = mj.geometric_adstock(pulse, alpha=retention, max_lag=max_lag)
delayed = mj.delayed_adstock(pulse[:, 0], alpha=0.8, theta=3.0, max_lag=max_lag)
lags = np.arange(max_lag + 1)

fig, axis = plt.subplots(layout="constrained")
for column, value in enumerate(retention):
    axis.plot(lags, geometric[:, column], marker="o", label=f"Geometric, retention {value}")
axis.plot(lags, delayed, marker="o", linestyle="--", label="Delayed, retention 0.8, peak at lag 3")
axis.set_xlabel("Weeks after the exposure")
axis.set_ylabel("Share of the week's effect")
axis.set_title("Where one week of exposure lands")
axis.legend(frameon=False)
plt.show()
```

At retention 0.3 nearly all of the effect lands within a few weeks. At 0.9 the
curve still hasn't reached zero by lag 12, so a window that ends there cuts off
part of that carryover. The dashed delayed curve instead builds to its peak at
lag 3 and then fades.

When the lag is unclear, [Jin et al.
(2017)](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf)
suggest a long window, so the weights it leaves out are close to zero.
{func}`~mmmjax.weibull_pdf_adstock` and {func}`~mmmjax.weibull_cdf_adstock`
give two more shapes if neither set of weights fits your channel.

## Diminishing returns

At high spend, each extra dollar tends to add less than the one before. A
linear response adds the same amount for every dollar, so it can't capture
that saturation. mmmJAX writes the Hill curve on carried exposure $u$ as

$$
\operatorname{Hill}(u;\, \kappa, s) = \frac{u^{s}}{u^{s} + \kappa^{s}}.
$$

The curve starts at zero, reaches one half at the half-saturation point
$\kappa$, and approaches one as exposure grows. Because the curve stays below
one, $\beta_c$ is the channel's largest possible weekly effect under an
identity link. The slope $s$ sets the shape, concave from the first unit when
$s \le 1$ and S-shaped when $s > 1$.

Applying the Hill curve after adstock, as Jin et al. do, suits spend spread over
many weeks. For other saturation shapes, you can turn to
{func}`~mmmjax.logistic_saturation`, {func}`~mmmjax.root_saturation`, or
{func}`~mmmjax.log_saturation`.

```{code-cell} ipython3
:tags: [hide-input]

exposure = np.linspace(0.0, 4.0, 201)
half_saturation = np.array([0.5, 1.0, 2.0])
slope = np.array([0.5, 1.0, 2.0, 4.0])
by_half = mj.hill_saturation(exposure[:, None], half_saturation=half_saturation, slope=1.0)
by_slope = mj.hill_saturation(exposure[:, None], half_saturation=1.0, slope=slope)

fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(12, 7), layout="constrained")
for column, value in enumerate(half_saturation):
    left.plot(exposure, by_half[:, column], label=f"Half saturation {value}")
for column, value in enumerate(slope):
    right.plot(exposure, by_slope[:, column], label=f"Slope {value}")
left.set_title("Half saturation varies, slope 1")
right.set_title("Slope varies, half saturation 1")
for axis in (left, right):
    axis.axhline(0.5, color=".6", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Carried exposure")
    axis.legend(frameon=False, loc="lower right")
left.set_ylabel("Share of the maximum effect")
plt.show()
```

Every curve crosses the dotted line at its half-saturation point, so you can
read each $\kappa$ straight off the plot. In the left
panel a smaller $\kappa$ saturates sooner, and in the right panel the slopes
above one start flat before they climb.

The three parameters of $\beta \operatorname{Hill}$ can be nearly impossible
to tell apart. The output below compares two parameter sets from Jin et al.
over exposures between 0 and 1 and past that range.

```{code-cell} ipython3
:tags: [hide-input]

def response(exposure, coefficient, half_saturation, slope):
    curve = mj.hill_saturation(exposure, half_saturation=half_saturation, slope=slope)
    return coefficient * np.asarray(curve)


observed = np.linspace(0.0, 1.0, 101)
gap = response(observed, 0.3, 0.5, 1.0) - response(observed, 0.393, 0.95, 0.748)
beyond = np.array([5.0, 20.0])
print(round(float(np.abs(gap).max()), 3))
print(response(beyond, 0.3, 0.5, 1.0).round(2), response(beyond, 0.393, 0.95, 0.748).round(2))
```

Over that range the two curves never differ by more than 0.012, yet at an
exposure of 20 the first reaches 0.29 and the second 0.36. So when your data
stays inside that range, it pins down the curve but not its parameters.

:::{admonition} Fixing the slope
:class: tip

Jin et al. suggest fixing $s = 1$ for this reason, as the User Guide model
does. They also suggest a prior that keeps $\kappa$ inside the observed range,
since past it the parameters are equally hard to tell apart
([Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
sections 2.2 and 3).
:::

## The posterior

Bayesian inference treats the parameters as random and conditions on the data
with Bayes' rule, and the prior is where you bring in what earlier models or
experiments say about them
([Gelman et al., 2013](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf),
eqs. 1.1 and 1.2),

$$
p(\Phi \mid y) = \frac{p(\Phi)\, p(y \mid \Phi)}{p(y)} \propto p(\Phi)\, p(y \mid \Phi).
$$

Here $\Phi$ collects every parameter, $p(\Phi)$ is the prior, and
$p(y \mid \Phi)$ is the likelihood. In the model above the $T$ weeks are
independent given $\mu_t$, so taking logs turns the product into a sum,

$$
\log p(\Phi \mid y) = \log p(\Phi) + \sum_{t=1}^{T} \log p(y_t \mid \mu_t, \psi) + \text{const}.
$$

Your `log_density` block returns that sum, so you write one prior term for each
parameter and add the likelihood. Since samplers need the sum only up to a
constant, the evidence $p(y)$ is never computed. [What is
mmmJAX](what_is_mmmjax.md#how-parameters-get-their-priors) shows the block and
the Jacobian term mmmJAX adds when it moves a constrained parameter to an
unconstrained scale.

### Why priors matter

One MMM dataset carries little information for the number of parameters the
model has to estimate. Weekly national data gives you one point per week,
while each channel brings its own coefficient, carryover, and saturation
parameters ([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
section 4.1). With so little data the posterior can stay close to the prior,
and different priors give different returns ([Jin et al., 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf),
sections 3 and 7).

Because mmmJAX leaves every prior to you, you'll want to check what your priors
claim before the fit. The evidence
$p(y) = \int p(y \mid \Phi)\, p(\Phi)\, d\Phi$ is also the prior predictive
distribution, and {func}`~mmmjax.sample_prior` draws from it for that check
([Gabry et al., 2019](https://arxiv.org/abs/1709.01449), section 3).

With `group="prior"`, {func}`~mmmjax.media_metrics`,
{func}`~mmmjax.response_curves`, and {func}`~mmmjax.contributions` show what
those draws imply for other quantities. An ROI prior, for example, implies
priors on mROI and contributions, and one that credits media with more than
the whole outcome leaves a negative baseline. [Priors](../user_guide/priors)
simulates from the User Guide model's priors and reads the returns they imply.

## Drawing from the posterior

The posterior of a nonlinear model has no closed form, so
{func}`~mmmjax.sample` draws from it with BlackJAX's No-U-Turn sampler (NUTS).
NUTS moves the parameters like a particle over the log posterior, and JAX's
gradient lets it travel far along the posterior's shape in many small steps.
It extends each path until the path starts to turn back on itself
([Hoffman and Gelman, 2014](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)),
and its warmup phase tunes the step size before any draws are kept.

A divergent transition marks a region that curves too sharply for the tuned
step size, so even one divergence makes estimates from the fit suspect
([Betancourt, 2017](https://arxiv.org/abs/1701.02434), section 6.2).

You also check the draws by running several chains from different starting
points. Split $\hat R$ compares the spread between chains with the spread
within them and sits close to one when they agree ([Vehtari et al., 2021](https://arxiv.org/abs/1903.08008)).

A fit can converge and still be the wrong model, so posterior predictive
checks compare data simulated from each draw with the data you observed
([Gelman et al., 2013](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf),
section 6.3). [Sampling and diagnostics](../user_guide/sampling) reads these
checks on the User Guide's fit.

## Priors on ROI and calibration

It's hard to set a prior on a channel's coefficient, because what the
coefficient means depends on the channel's spend, carryover, and saturation.
[Zhang et al. (2023)](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/a09f404fdc3107fafb7a52cc5af6a80e4d0fda2b.pdf)
put the prior on the ROI $r_c$ instead and derive the coefficient from it.

Take an identity-link national model whose outcome you divide by $s_R$ before
the fit, as [Data and scaling](../user_guide/data.md#scaling) explains. If
channel $c$ spent $S_c$ over all weeks, it gets

$$
\beta_c = \frac{r_c S_c}{s_R \sum_t h_{tc}},
$$

so its contribution over the weeks, $s_R \beta_c \sum_t h_{tc}$, equals
$r_c S_c$. The cell asks {func}`~mmmjax.roi_coefficient` for returns of 2 and
0.5 on two channels and checks that the coefficients give them back.

```{code-cell} ipython3
import numpy as np

import mmmjax as mj

transformed = np.array([[0.5, 1.0], [0.6, 0.9], [0.4, 1.1]])
spend = np.array([[100.0, 50.0], [120.0, 40.0], [80.0, 60.0]])
coefficient = mj.roi_coefficient(np.array([2.0, 0.5]), transformed, spend, outcome_scale=40.0)
contribution = 40.0 * transformed.sum(axis=0) * coefficient
print(np.asarray(contribution / spend.sum(axis=0)).round(2))
```

The cell prints 2 and 0.5, so the coefficients reproduce the returns you asked
for. In your model you call {func}`~mmmjax.roi_coefficient` inside
`transformed_parameters`, and the coefficient gets no prior of its own, because
the prior sits on the ROI.

Both {func}`~mmmjax.roi_coefficient` and {func}`~mmmjax.contribution_coefficient`
assume each channel enters through the additive $\beta_c h_{tc}$ term.
[A first model](../user_guide/first_model.md#return-on-investment) uses this
parameterization, and [Priors](../user_guide/priors) shows what its ROI prior
claims in dollars.

### Calibration with experiments

An experiment such as a geo lift test measures one channel's return at one
spend level over a short window. That result can center your ROI prior, as
long as the experiment measures the same quantity, incremental outcome against
zero spend
([Zhang et al., 2023](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/a09f404fdc3107fafb7a52cc5af6a80e4d0fda2b.pdf),
section 3.4).

:::{admonition} Tests at unusual spend
:class: warning

A test at unusual spend sits at a different point on the saturation curve than
the average week, so its return can differ from an average week's. To build a
prior from it, pass {func}`~mmmjax.roi_coefficient` the test weeks' spend and
the response that spend causes.
:::

mmmJAX has no helper that turns a test result into a prior, so you write that
step yourself. A test can also enter `log_density` as an observation of the
lift the model implies for its weeks, and the test's standard error sets its
scale.

## Geo-level hierarchy

[Sun et al. (2017)](https://research.google.com/pubs/archive/46000.pdf) fit
regional data with a hierarchical model in which each region's coefficients
vary around shared values. Written with the same channel and control terms as
before, region $i$'s mean $\mu_{ti}$ in week $t$ follows

$$
g(\mu_{ti}) = b_{ti} + \sum_{c=1}^{C} \beta_{ic} h_{tic} + \sum_{j=1}^{J} \gamma_{ij} z_{tij},
\qquad
\beta_{ic} \sim \operatorname{Normal}(\beta_c, \eta_c).
$$

Each region $i$ has its own coefficient $\beta_{ic}$, drawn around the shared
$\beta_c$ with standard deviation $\eta_c$, and the baselines and control
coefficients vary the same way. Regional data gives one observation per region
per week and usually a wider range of spend, so when the regions respond alike,
pooling them tends to give tighter intervals than a national model.

Carryover and the Hill parameters stay the same in every region, because Sun
et al. expected that letting them vary would make the model hard to identify.

{func}`~mmmjax.prepare_data` with `groups` adds the region axis, as [Data and
scaling](../user_guide/data.md#groups) shows. {func}`~mmmjax.roi_coefficient`
accepts each region's deviations too, but its default makes $\beta_{ic}$
log-normal, so pass `effects="normal"` for the form above.

## What the data cannot settle

Even when your model predicts well, the split between correlated channels
stays uncertain, because the data shows little of one channel moving without
the others. Spend kept within a narrow band leaves the response curve
uncertain in the same way.

Like the two Hill curves above, curves that agree over the observed spend can
part ways outside it. That's how a model can give you a sensible marginal
return near current spend but a poor average return, because the average
return needs the curve all the way down to zero spend
([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
sections 4.1.2 and 4.1.3). {func}`~mmmjax.check_data` flags correlated channels
before you fit, and {func}`~mmmjax.plot_response_curves` dashes each curve past
the spend the data observed.

### Selection bias and controls

Selection bias may be the largest hurdle an MMM faces ([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
section 4.2). If you spend more in weeks when customers are already
interested, comparing weeks with the channel on and off mixes the channel's
effect with that demand. The problem shows up sharply in paid search, whose ads
appear only to people already searching, so its naive return comes out too
high ([Chen et al., 2018](https://arxiv.org/abs/1807.03292)).

A regression recovers $\mu_t(x)$ only if

$$
\operatorname{E}[y^{(x)} \mid z] = \operatorname{E}[y \mid x, z],
$$

where $y^{(x)}$ is the outcome under exposures $x$. That condition holds when
your controls $z$ include everything that drove both spending and the outcome
([Chen et al., 2018](https://arxiv.org/abs/1807.03292), section 3).

So which variables you put in depends on the causal role each one plays
([Cinelli et al., 2022](https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf)).

- Include confounders such as demand, since they drive both spending and the
  outcome.
- Leave out mediators such as site visits, since they lie between the ads and
  the outcome.
- Predictors that affect only the outcome are optional and only reduce
  variance.

Choosing the controls is still your causal call, because query volume, for
example, is both a confounder for search ads and a mediator for channels that
drive searches. [The example data](../user_guide/example_data) builds
selection bias into the User Guide's brand.

:::{admonition} Don't read controls causally
:class: danger

A control's coefficient holds the media fixed and may carry confounding of its
own, so it has no causal reading
([Westreich and Greenland, 2013](https://doi.org/10.1093/aje/kws412)). The
analyses therefore leave a control's effect in the baseline rather than report
it as a contribution.
:::

### Treatments and organic media

Price and promotions play a different role from controls, because you set them
the way you set a budget and you want to know what they did. As non-media
treatments they enter the model the way controls do, but their effect reads as
causal only when the controls cover whatever drove both them and the outcome.

Unlike a control, a treatment does get a contribution of its own.
{func}`~mmmjax.contributions` reports what each one added against a baseline
level rather than zero, since a price of zero means nothing.

Unpaid channels such as email go in as organic media, since their exposure
carries over and saturates like a paid channel's. {func}`~mmmjax.contributions`
removes it the same way, so they get a contribution too. With no spend to
divide by, though, they have no ROI.

{func}`~mmmjax.prepare_data` takes each of these roles, and [Data
and scaling](../user_guide/data.md#controls-treatments-and-organic-media)
assigns them for the User Guide's brand.

### Fit does not validate returns

Models with very different channel effects can also fit the past equally
well, since price, distribution, and seasonal proxies often predict sales
without any media. Those models still split the budget differently, and
ranking them by predictive accuracy won't weed out the ones whose returns are
wrong ([Chan and Perry, 2017](https://storage.googleapis.com/gweb-research2023-media/pubtools/3803.pdf),
section 4.3).

In real data the true return stays hidden, but {func}`~mmmjax.simulate_data`
returns the true contributions with every dataset. The User Guide checks its
model against them at the end of
[A first model](../user_guide/first_model.md#returns-against-the-truth) and
answer by answer on [Recovering the truth](../user_guide/recovery). None of
mmmJAX's analysis functions adds causal evidence beyond what your model assumes.
