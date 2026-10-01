---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A first model

This page builds the marketing mix model that the rest of the guide uses. It's
written for a brand with ten paid channels, an email newsletter, two controls,
a price and promotions, a trend, and a yearly season. You'll write the model
in math first and then turn it into blocks, fit it, and set its returns beside
the true ones.

## The data

The data is one national series from {func}`~mmmjax.simulate_data` with seed
7, and [The example data](example_data) describes how it's made. The cell
below prepares and scales it the same way [Data and scaling](data.md) does.
Its `channels` dictionary maps each channel's column stem to the name your
results will carry. {func}`~mmmjax.prepare_data` then gives each column the
model uses a role, and {func}`~mmmjax.fit_data_scaling` fits the scaling the
model works on.

```{code-cell} ipython3
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
    organic_media=["email_sends"],
    organic_channels=["Email"],
    controls=["demand", "holiday"],
    treatments=["price", "promotion"],
)
scaling = mj.fit_data_scaling(data, scale_outcome=True)
```

Demand and the holidays go in as `controls`, because both move revenue and the
brand spends more in the weeks they run high. If you left them out, the model
would credit their lift to whichever channels ran in those weeks. It adjusts
for each control, but since a control's coefficient has no causal reading, it
never reports a control's effect.

Price and promotions go in as `treatments` instead, because they're non-media
inputs you set yourself and want measured. A price of zero makes no sense, so
the analyses report a treatment against a baseline level, such as the lowest
price in the data.

The remaining input, email, goes in as `organic_media` under the name Email,
since its sends carry over and saturate the way a paid channel's impressions
do. Because the newsletter costs nothing to send, email gets a contribution in
the analyses but no return on investment. You can read more about each role,
and how mmmJAX scales it, in [Data and scaling](data.md).

## The model

The model treats each week's standardized revenue as a sum of parts, a baseline
plus the effects of the media, the controls, and the treatments. Whatever those
parts leave unexplained counts as normal noise around their total.

In the math below, week $t$ runs over the modeled weeks, $c$ over the ten paid
channels, $o$ over the organic channels (here only email), $j$ over the two
controls, and $i$ over the two treatments.

### Data transformations

Before the model sees the data, most inputs are rescaled, and the rule each one
follows depends on the kind of input it is. The table groups the inputs by
rule, and the highlighted symbol in each formula is the quantity the model
works on.

| Inputs | How they're scaled | In the math |
| --- | --- | --- |
| Paid impressions $z_{tc}$ and email's sends $n_{to}$ | Divided by the median over the weeks with any exposure | $\hl{x_{tc}} = \dfrac{z_{tc}}{m_c}$, $\hl{x_{to}} = \dfrac{n_{to}}{m_o}$ |
| Revenue $R_t$, controls $q_{tj}$, and treatments $w_{ti}$ | Standardized with the mean and standard deviation over the modeled weeks | $\hl{y_t} = \dfrac{R_t - \bar{R}}{s_R}$, $\hl{p_{tj}} = \dfrac{q_{tj} - \bar{q}_j}{s_j}$, $\hl{g_{ti}} = \dfrac{w_{ti} - \bar{w}_i}{s_i}$ |
| Days $e_t$ since the first training week | Converted to years for the trend | $\hl{\tau_t} = \dfrac{e_t}{365.25}$ |
| Day of the year $d_t$ and spending $v_{tc}$ | Left as they are, with spending in dollars | $\hl{S_c} = \sum_t v_{tc}$ |

Dividing each channel's exposure by its median over the weeks it ran keeps a
week off air at zero, which the analyses rely on when they take a channel away,
and it makes a typical week on air equal to about one whatever the channel's
volume. That shared unit is what lets one half-saturation prior mean the same
thing for every channel, email included.

Standardizing revenue, the controls, and the treatments lets their priors speak
in standard deviations, so the same priors make the same claims for a brand of
any size, and `outcome_scaling` keeps the mean and scale that turn results back
into dollars. Counting the trend in years serves the same end, since it keeps
$\tau_t$ below three over the data and the growth and curvature priors on scales
between 0.25 and one.

Spending stays in dollars because a return is revenue per dollar, and it
reaches the model only through each channel's total $S_c$ in the ROI
calibration below.

### Model equation

$$
\begin{aligned}
y_t &= \alpha + \delta_1 \tau_t + \delta_2 \tau_t^2
+ \sum_{k=1}^{2} \Big( a_k \sin\frac{2\pi k d_t}{365.25} + b_k \cos\frac{2\pi k d_t}{365.25} \Big) \\
&\quad + \sum_{c=1}^{10} \beta_c h_{tc} + \sum_{o} \lambda_o h_{to}
+ \sum_{j=1}^{2} \gamma_j p_{tj} + \sum_{i=1}^{2} \theta_i g_{ti} + \varepsilon_t, \\
\varepsilon_t &\sim \operatorname{Normal}(0, \sigma), \\
h_{tc} &= \operatorname{HillAdstock}\big(\{x_{t-\ell,c}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big), \\
h_{to} &= \operatorname{HillAdstock}\big(\{x_{t-\ell,o}\}_{\ell=0}^{8};\, \rho_o, \kappa_o\big)
\end{aligned}
$$

Everything but the noise $\varepsilon_t$ adds up to the mean $\mu_t$, so the
likelihood is $y_t \sim \operatorname{Normal}(\mu_t, \sigma)$ with noise scale
$\sigma$. In the baseline, $\alpha$ is the intercept, $\delta_1$ and $\delta_2$
are the trend's growth per year and its curvature, and $a_k$ and $b_k$ weigh the
season's harmonics. In the second line, $h_{tc}$ and $h_{to}$ are each paid
channel's and email's exposure after carryover and saturation, weighted by the
coefficients $\beta_c$ and $\lambda_o$, while $\gamma_j$ and $\theta_i$ weigh
the controls and the treatments.

Because the parts add, a channel brings the same revenue to a week whatever the
season, the trend, or the other channels are doing. That's what lets the
channels' contributions sum to the media's total, as
[Media effects](media_effects.md#all-together) shows, and what lets the ROI
calibration below turn a return into a coefficient.

If you believe a campaign earns more in a strong season, you could model log
revenue so the parts multiply, though the channels' effects would then stop
adding up, as
[Introduction to MMM](../getting_started/intro_to_mmm.md#the-model-equation)
explains. Each channel's coefficient also holds for all 156 weeks, and
[Changing the model](changing) shows what that misses when ads wear out.

The quadratic trend can rise and then fall with only two parameters, as the
simulated baseline does, but past the data $\tau_t^2$ keeps growing, and
[Scenarios](scenarios.md#a-forecast) finds the trend taking about \$736,000 off
next quarter against the same weeks a year earlier. A Gaussian process trend
from {func}`~mmmjax.prepare_hsgp` would instead settle back toward its mean past
the data.

Two Fourier harmonics give the season a smooth yearly shape that can peak twice
and whose rise and fall needn't mirror each other. More harmonics would allow
sharper shapes, but the fourth repeats every 13 weeks, about as often as the
brand's flights, so a season that flexible could claim part of their lift.

A treatment is reported against a baseline level $w_i^{0}$, and because its term
is a straight line, as in the simulation, moving it from $w_i^{0}$ to the levels
in the data adds

$$
s_R\, \theta_i \sum_t \frac{w_{ti} - w_i^{0}}{s_i}
$$

dollars of revenue over the training weeks.

Normal noise with one scale for every week is the simplest likelihood, and it's
safe here because revenue sits so many noise scales above zero that the
normal's lack of a floor never matters to the fit. The model also treats each week's noise as
independent of the last, and
[Sampling and diagnostics](sampling.md#test-statistics) finds residuals that run
in short streaks such noise wouldn't produce.

A Student-t likelihood from {func}`~mmmjax.student_t` would give unusual weeks
less pull, and a log-normal one would let the scatter grow with revenue once
[a scaling of your own](data.md#scaling-of-your-own) keeps every week positive.

### Media transformation

Adstock carries exposure forward by averaging the current week and the eight
before it with geometric weights, so the retention rate $\rho$ sets how much
past weeks count. The Hill saturation curve reaches half its maximum at
$\kappa$,

$$
\operatorname{Adstock}\big(\{x_{t-\ell}\}_{\ell=0}^{L};\, \rho\big)
= \frac{\sum_{\ell=0}^{L} \rho^{\ell}\, x_{t-\ell}}{\sum_{\ell=0}^{L} \rho^{\ell}},
\qquad
\operatorname{Hill}(u;\, \kappa, s) = \frac{u^{s}}{u^{s} + \kappa^{s}}.
$$

HillAdstock applies Adstock first and then Hill, the order that
[Introduction to MMM](../getting_started/intro_to_mmm.md#diminishing-returns)
says suits spending spread over many weeks. The model fixes the slope $s$ at
one, as
[Jin et al. (2017)](https://storage.googleapis.com/gweb-research2023-media/pubtools/3806.pdf)
suggest, because weekly data seldom tells it apart from $\kappa$ and the
coefficient. [Recovering the truth](recovery.md#linear-tvs-curve) shows what
that costs Linear TV, whose true curve starts slowly. To learn the slope
instead, declare it as a parameter with a prior of its own.

Since Adstock reaches eight weeks back, the first weeks draw on exposure from
before the data begins, and the model counts that exposure as zero.

### Return on investment

Rather than sampling the coefficient $\beta_c$ directly, the model samples
channel $c$'s return on investment $r_c$ and sets

$$
\beta_c = \frac{r_c S_c}{s_R \sum_{t} h_{tc}},
$$

where $h_{tc}$ is computed from the training exposure. The revenue scale $s_R$
turns a change in $y_t$ back into dollars, so channel $c$'s contribution over
the training weeks, $s_R \beta_c \sum_t h_{tc}$, comes to $r_c S_c$ dollars.
It's easier to hold a belief about the revenue a dollar of spending returns
than about a coefficient on scaled media, and that's why the prior goes on
$r_c$.

### Email's share of revenue

Email has no spending, so there's no return on investment to put a prior on.
Instead, the model samples $\phi_o$, the share of the training weeks' revenue
that email produced, and sets

$$
\lambda_o = \frac{\phi_o \sum_t R_t}{s_R \sum_{t} h_{to}},
$$

where $h_{to}$ is computed from the training sends. Email's contribution over
the training weeks, $s_R \lambda_o \sum_t h_{to}$, then comes to $\phi_o \sum_t
R_t$ dollars.

### Priors

Each parameter has its own fixed prior, so the model has no hyperpriors.

$$
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 1) \\
\delta_1 &\sim \operatorname{Normal}(0, 1) \\
\delta_2 &\sim \operatorname{Normal}(0, 0.25) \\
a_k, b_k &\sim \operatorname{Normal}(0, 0.5) \\
r_c &\sim \operatorname{LogNormal}(1, 0.6) \\
\rho_c &\sim \operatorname{Beta}(2, 2) \\
\kappa_c &\sim \operatorname{LogNormal}(0, 0.5) \\
\phi_o &\sim \operatorname{Beta}(2, 98) \\
\rho_o &\sim \operatorname{Beta}(2, 2) \\
\kappa_o &\sim \operatorname{LogNormal}(0, 0.5) \\
\gamma_j &\sim \operatorname{Normal}(0, 1) \\
\theta_i &\sim \operatorname{Normal}(0, 0.25) \\
\sigma &\sim \operatorname{HalfNormal}(1)
\end{aligned}
$$

:::{admonition} Distribution parameters
:class: note

These distributions take the same parameters as their mmmJAX functions, so
$\operatorname{LogNormal}(1, 0.6)$ is the distribution of a variable whose
logarithm has mean one and standard deviation 0.6, $\operatorname{Beta}(2, 2)$
has both shape parameters at two, and $\operatorname{HalfNormal}(1)$ has scale
one.
:::

Most of these priors are written with the scaled data in mind. Standardized
revenue has a standard deviation of one, so a scale of one on the intercept,
the growth, and the controls already spans it, and the Fourier coefficients get
half that. The curvature gets 0.25 because $\tau_t^2$ outgrows $\tau_t$ after
the first year. At a scale of one, the squared term alone would swing revenue
far past anything in the data by the last week.

The treatments get 0.25 too, because you believe that moving the price or
promotions by one standard deviation seldom moves weekly revenue by more than
half a standard deviation. A second reason for the tight scale is that the
price climbs with the trend and falls with the promotions, so the data alone
can't tell the price apart from them. A wider prior would then let the fit
credit the price with the brand's growth.

The ROI and share priors are the exceptions, since the return $r_c$ counts
dollars of revenue per dollar spent and the share $\phi_o$ is a fraction of
revenue. With a mean of 0.02, $\operatorname{Beta}(2, 98)$ says you expect
email to be a small channel that brings in about 2 percent of revenue.
[Priors](priors) covers what these choices claim about revenue and how to check
them.

## From math to code

To turn the math into code, you sort its symbols into the four groups of
[Write the math first](../getting_started/what_is_mmmjax.md#write-the-math-first).
Each prior and the likelihood, the statements written with $\sim$, becomes
one term of `target` in `log_density`.

:::{admonition} The code follows the math
:class: tip

Every argument in the block signatures below, apart from the random key,
carries a symbol above. If you can't place a name, you've usually skipped a
symbol, so find it in the math before you read on.
:::

The tables below have one tab for each place a name can come from, and every
row highlights its symbol in a formula that uses it.

::::{tab-set}

:::{tab-item} Supplied by mmmJAX

The data arrives under names mmmJAX supplies, and the media, controls,
treatments, and revenue come in already scaled.

| Symbol | In the math | Name in the code |
| --- | --- | --- |
| $y_t$ | $\hl{y_t} \sim \operatorname{Normal}(\mu_t, \sigma)$ | `outcome` |
| $x_{tc}$ | $h_{tc} = \operatorname{HillAdstock}\big(\{\hl{x_{t-\ell,c}}\}_{\ell=0}^{8};\, \rho_c, \kappa_c\big)$ | `media` |
| $x_{to}$ | $h_{to} = \operatorname{HillAdstock}\big(\{\hl{x_{t-\ell,o}}\}_{\ell=0}^{8};\, \rho_o, \kappa_o\big)$ | `organic_media` |
| $p_{tj}$ | $\sum_{j} \gamma_j \hl{p_{tj}}$ | `controls` |
| $g_{ti}$ | $\sum_{i} \theta_i \hl{g_{ti}}$ | `treatments` |
| $e_t$ | $\tau_t = \hl{e_t} / 365.25$ | `time` |
| $d_t$ | $a_k \sin(2\pi k \hl{d_t} / 365.25)$ | `day_of_year` |
| $x_{tc}$ and $x_{to}$ in the training weeks | $\beta_c = \dfrac{r_c S_c}{s_R \sum_t \operatorname{HillAdstock}(\{\hl{x_{t-\ell,c}}\};\, \rho_c, \kappa_c)}$ | `reference.media` and `reference.organic_media` |
| $v_{tc}$ | $S_c = \sum_t \hl{v_{tc}}$ | `reference.spend` |
| $R_t$ | $\lambda_o = \dfrac{\phi_o \sum_t \hl{R_t}}{s_R \sum_t h_{to}}$ | `outcome_scaling.inverse_transform(reference.outcome)` |
| $s_R$ | $\beta_c = \dfrac{r_c S_c}{\hl{s_R} \sum_t h_{tc}}$ | `outcome_scaling.scale` |

:::

:::{tab-item} Declared in parameters

You name each unknown yourself, and its subscript becomes the axis named in its
`dims`.

| Symbol | In the math | Name in the code | Axis in `dims` |
| --- | --- | --- | --- |
| $\alpha$ | $\mu_t = \hl{\alpha} + \delta_1 \tau_t + \cdots$ | `intercept` | none |
| $\delta_1$ | $\mu_t = \alpha + \hl{\delta_1} \tau_t + \cdots$ | `growth` | none |
| $\delta_2$ | $\mu_t = \alpha + \delta_1 \tau_t + \hl{\delta_2} \tau_t^2 + \cdots$ | `curvature` | none |
| $a_1$, $a_2$, $b_1$, $b_2$ | $\sum_{k=1}^{2} \big(\hl{a_k} \sin(2\pi k d_t / 365.25) + \hl{b_k} \cos(2\pi k d_t / 365.25)\big)$ | `annual_coefficients`, in that order | none, a plain shape of four |
| $r_c$ | $\beta_c = \dfrac{\hl{r_c} S_c}{s_R \sum_t h_{tc}}$ | `roi` | `"channel"` |
| $\rho_c$ | $h_{tc} = \operatorname{HillAdstock}(\ldots;\, \hl{\rho_c}, \kappa_c)$ | `retention` | `"channel"` |
| $\kappa_c$ | $h_{tc} = \operatorname{HillAdstock}(\ldots;\, \rho_c, \hl{\kappa_c})$ | `half_saturation` | `"channel"` |
| $\phi_o$ | $\lambda_o = \dfrac{\hl{\phi_o} \sum_t R_t}{s_R \sum_t h_{to}}$ | `organic_share` | `"organic_channel"` |
| $\rho_o$ | $h_{to} = \operatorname{HillAdstock}(\ldots;\, \hl{\rho_o}, \kappa_o)$ | `organic_retention` | `"organic_channel"` |
| $\kappa_o$ | $h_{to} = \operatorname{HillAdstock}(\ldots;\, \rho_o, \hl{\kappa_o})$ | `organic_half_saturation` | `"organic_channel"` |
| $\gamma_j$ | $\sum_{j} \hl{\gamma_j} p_{tj}$ | `control_coefficient` | `"control"` |
| $\theta_i$ | $\sum_{i} \hl{\theta_i} g_{ti}$ | `treatment_coefficient` | `"treatment"` |
| $\sigma$ | $y_t \sim \operatorname{Normal}(\mu_t, \hl{\sigma})$ | `sigma` | none |

:::

:::{tab-item} Computed in the blocks

The quantities built from the data and the unknowns take names of your own
too. `mu` is the only one `transformed_parameters` returns, since no other
block asks for the steps on the way to it.

| Symbol | In the math | Name in the code | Where it's computed |
| --- | --- | --- | --- |
| $\tau_t$ | $\mu_t = \alpha + \delta_1 \hl{\tau_t} + \delta_2 \hl{\tau_t}^2 + \cdots$ | `trend` | Returned by `transformed_data` |
| The sines and cosines of $d_t$ | $\sum_{k=1}^{2} \big(a_k \hl{\sin(2\pi k d_t / 365.25)} + b_k \hl{\cos(2\pi k d_t / 365.25)}\big)$ | `annual` | Returned by `transformed_data` |
| $\mu_t$ | $y_t \sim \operatorname{Normal}(\hl{\mu_t}, \sigma)$ | `mu` | Returned by `transformed_parameters` |
| $h_{tc}$ and $h_{to}$ in the training weeks | $\beta_c = \dfrac{r_c S_c}{s_R \sum_t \hl{h_{tc}}}$ | `trained` and `organic_trained` | Local to `transformed_parameters` |
| $h_{to}$ in the weeks being evaluated | $\mu_t = \cdots + \sum_{o} \lambda_o \hl{h_{to}} + \cdots$ | `organic_saturated` | Local to `transformed_parameters` |
| $\beta_c$ | $\mu_t = \cdots + \sum_{c} \hl{\beta_c} h_{tc} + \cdots$ | `coefficient` | Local to `transformed_parameters` |
| $\lambda_o$ | $\mu_t = \cdots + \sum_{o} \hl{\lambda_o} h_{to} + \cdots$ | `organic_coefficient` | Local to `transformed_parameters` |
| $\sum_t R_t$ | $\lambda_o = \dfrac{\phi_o \hl{\sum_t R_t}}{s_R \sum_t h_{to}}$ | `total_revenue` | Local to `transformed_parameters` |

:::

::::

A few symbols never get a name in the code, and the list below shows where each
one ends up instead.

- The raw columns $z_{tc}$, $n_{to}$, $q_{tj}$, and $w_{ti}$ never reach a
  block, since mmmJAX scales them first with the medians, means, and standard
  deviations in `scaling`.
- Revenue's mean $\bar{R}$ reaches the blocks only inside
  `outcome_scaling.inverse_transform`.
- $S_c$ is totaled inside {func}`~mmmjax.roi_coefficient` from
  `reference.spend`.
- The noise $\varepsilon_t$ lives in `mj.normal(outcome, mu, sigma)`, the
  likelihood line of `log_density`.
- The two harmonics, the eight lags, and the slope $s$ are the fixed numbers
  `order=2`, `max_lag=8`, and `slope=1.0`.
- The baseline level $w_i^{0}$ belongs to the analyses rather than the model,
  as [Media effects](media_effects) shows.

## Parameters

You declare each unknown in `parameters` with its support and its axes, so
mmmJAX knows which values it can take and how to label them.

```{code-cell} ipython3
parameters = {
    # The baseline's level, the trend's growth and curvature, and the season.
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    # No data axis runs over the four Fourier weights, and they need no labels, so a plain shape does.
    "annual_coefficients": mj.Real(4),
    # Each paid channel's return, carryover, and half-saturation point.
    "roi": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    # Email's share of revenue, carryover, and half-saturation point.
    "organic_share": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_retention": mj.Interval(0.0, 1.0, dims="organic_channel"),
    "organic_half_saturation": mj.Positive(dims="organic_channel"),
    # One coefficient for each control and each treatment, and the noise scale.
    "control_coefficient": mj.Real(dims="control"),
    "treatment_coefficient": mj.Real(dims="treatment"),
    "sigma": mj.Positive(),
}
```

{class}`~mmmjax.Positive` and {class}`~mmmjax.Interval` keep a parameter
inside its support, so every return stays positive and every retention rate
between zero and one. None of the declarations carries a prior, because the
priors go in `log_density` below.

`dims="channel"` gives a parameter one value per channel, labeled with the ten
names from the data, and `dims="organic_channel"` gives it one labeled Email.
`dims="control"` and `dims="treatment"` give it one per control or treatment
column.

## Trend and seasonality

The trend and the season's sines and cosines depend only on the data, so
`transformed_data` computes them.

```{code-cell} ipython3
def transformed_data(day_of_year, time):
    # Nothing here depends on a parameter, so this block runs once per dataset, not at every draw.
    annual = mj.fourier_features(day_of_year, period=365.25, order=2)

    # Years keep trend and trend**2 on the scale the growth and curvature priors assume,
    # and time keeps counting past the training weeks, so a forecast extends the trend.
    trend = time / 365.25

    return {"annual": annual, "trend": trend}
```

Each argument belongs to one of the groups in
[From math to code](#from-math-to-code), and mmmJAX matches each one by name.

:::{admonition} Supplied names and your names
:class: important

Supplied names such as `day_of_year` keep mmmJAX's spelling, and the
[Data and scaling](data.md#supplied-names) section lists every one of these. `annual` and
`trend`, like the keys of `parameters`, are your own names. You can rename one
as long as every block that uses it changes too and the new name isn't a
supplied one. If you rename a parameter, change its key in `priors` below as
well.
:::

`day_of_year` holds each week's calendar day, and
{func}`~mmmjax.fourier_features` turns it into two sine columns followed by two
cosine columns. The four entries of `annual_coefficients` weigh those columns
in the same order. In new data, `time` keeps counting from the first training
week, so a forecast carries the trend on past the training weeks, as
[Scenarios](scenarios) shows.

## Expected revenue

The model applies HillAdstock twice, to the training exposure and to the
exposure it's evaluating, so you write the transformation once as a helper. One
of the many awesome features of mmmJAX is that a
block can call any Python function written with JAX operations, and
[User-defined functions](functions) covers the rules.

```{code-cell} ipython3
# mmmJAX fills only a block's arguments, so media here is whatever the caller passes.
def hill_adstock(media, retention, half_saturation):
    # Eight weeks is the longest you believe exposure keeps working. It sets an array shape,
    # so it stays a fixed int rather than a parameter the sampler learns.
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)

    # The slope stays at one, so the curve rises fastest at the first impression.
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated
```

With the helper in place, `transformed_parameters` turns exposure into
expected revenue by building each term of the model equation and adding them
up.

```{code-cell} ipython3
def transformed_parameters(
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
):
    # Paid media's coefficients come from the returns. reference holds the training data in every
    # scenario, so the coefficients keep their fitted values when an analysis changes media. The
    # prior sits on ROI because revenue per dollar is easier to judge than a scaled coefficient.
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)

    # With no spend, email has no ROI, so its prior sits on its share of revenue. The share
    # is of dollars, since standardized revenue sums to zero over the training weeks.
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )

    # The baseline follows the trend and the season.
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients

    # Each effect reads the inputs the model is given, so a scenario that changes media,
    # sends, or prices changes it.
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient

    # The likelihood and every analysis read the expected revenue in each week.
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}
```

Inside the block, each parameter you ask for holds its value at the current
draw, so `roi` arrives as ten values, one ROI per channel.

:::{admonition} Coefficients come from the training data
:class: important

`reference` holds the training data in every scenario, while `media` holds
whatever the scenario sets. Analysis functions such as
{func}`~mmmjax.contributions` run the block again with one channel switched
off or a new plan in place of `media`. Because `trained` and the spending come
from `reference`, the coefficients stay those of the fitted model.
:::

The block returns the mean $\mu_t$ of the model equation as `mu`, a name of
your own. `log_density` asks for it below, and the analyses take it as
`quantity="mu"`.

:::{admonition} Media history
:class: note

This model treats the weeks before the data as having no exposure. When you
know what aired before your first week, pass it to
{func}`~mmmjax.prepare_data` as `media_history`, and carryover in the first
weeks starts from it instead of from zero. The history must hold every exposure
column, email's sends included, so `media` and `organic_media` both have more
rows than the outcome. The block then asks for the supplied `n_periods` and
keeps only the last `n_periods` rows of each once carryover has used the rest.

```python
media_effect = hill_adstock(media, retention, half_saturation)[-n_periods:] @ coefficient
organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
organic_effect = organic_saturated[-n_periods:] @ organic_coefficient
```

Carryover from before the window isn't a return on spending inside it, so
`trained` and `organic_trained` then need the response to in-window exposure
alone. The `response` parameters of {func}`~mmmjax.roi_coefficient` and
{func}`~mmmjax.contribution_coefficient` explain how to build it.
{func}`~mmmjax.media_response` runs the carryover, the saturation, and the
slice of the three lines above in one call.
:::

## The density

`log_density` adds up every prior and the likelihood, one term at a time, the
way the model block of a Stan program does.

```{code-cell} ipython3
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
    organic_share,
    organic_retention,
    organic_half_saturation,
    control_coefficient,
    treatment_coefficient,
    sigma,
):
    # The baseline's priors describe standardized revenue. The curvature's scale is smaller
    # because trend**2 outgrows trend after the first year.
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)

    # Paid media's returns, carryover, and saturation.
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)

    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)

    # The treatments get a tighter prior than the controls. The price climbs with the trend,
    # and a wider prior would let the fit credit price with growth.
    target += mj.normal(control_coefficient, 0.0, 1.0)
    target += mj.normal(treatment_coefficient, 0.0, 0.25)

    # The noise scale's prior and the likelihood, normal noise around the expected revenue.
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target
```

The first thirteen terms are the priors of the model above, in the same order,
and the last is its likelihood, $y_t \sim \operatorname{Normal}(\mu_t,
\sigma)$. Each function from [Distributions](distributions) returns the summed
log density of its first argument under the settings after it, so
`target += mj.lognormal(roi, 1.0, 0.6)` does what
`target += lognormal_lpdf(roi | 1, 0.6);` does in Stan. What goes into this
sum is up to you, since mmmJAX adds nothing to it except the Jacobian
adjustments its declarations make for their constraints.

## The priors as objects

`log_density` hands mmmJAX one number, the total of every term you added to
`target`. The sampler needs nothing more than that to fit the model, but a few
tools need to work with the priors themselves. {func}`~mmmjax.sample_prior`
draws each parameter from its prior so that [Priors](priors) can show you what
the model believes before it sees any data. `log_density` can score any values
you give it, but it has no way to produce new ones.
{func}`~mmmjax.psense_summary` and {func}`~mmmjax.plot_psense` measure how
much each estimate leans on its prior, so they need each prior's share of the
total rather than the total alone.

A dictionary of {class}`~mmmjax.Prior` objects gives those tools the priors in
a form they can draw from and score one at a time. Since each entry mirrors a
line of `log_density` above, you can carry its settings straight across.

```{code-cell} ipython3
# The prior tools read this mapping and log_density never does, so keep the two in step.
priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "growth": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "curvature": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "annual_coefficients": mj.Prior(mj.normal, location=0.0, scale=0.5),
    "roi": mj.Prior(mj.lognormal, location=1.0, scale=0.6),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "organic_share": mj.Prior(mj.beta, alpha=2.0, beta=98.0),
    "organic_retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "organic_half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "treatment_coefficient": mj.Prior(mj.normal, location=0.0, scale=0.25),
    "sigma": mj.Prior(mj.half_normal, scale=1.0),
}
```

Each {class}`~mmmjax.Prior` pairs a distribution function with fixed settings,
so `priors["roi"]` is the same lognormal with location 1.0 and scale 0.6 that
`log_density` uses for `roi`. Because the dictionary and `log_density` describe
the same model, it's worth keeping them matched whenever you change a prior.
[Priors](priors.md#checking-the-mapping-against-the-density) shows a check you
can run to confirm that the two agree.

## Generated quantities

`generated_quantities` runs once for every saved draw, and each time its first
argument gets a JAX random key for that draw while mmmJAX fills the rest by
name. The `_rng` functions draw with that key, as `mj.normal_rng` does in the
block below.

```{code-cell} ipython3
def generated_quantities(key, outcome, mu, sigma):
    # Simulated revenue for every week, and each observed week's log likelihood.
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)

    # "predictive" and "log_likelihood" are supplied names that tell mmmJAX where to store
    # the draws, and mmmJAX names both "outcome" so ArviZ pairs them with the observed data.
    return {"predictive": prediction, "log_likelihood": pointwise}
```

For each draw of the parameters, `prediction` redraws every week's
standardized revenue $\tilde{y}_t \sim \operatorname{Normal}(\mu_t, \sigma)$
from the posterior predictive distribution, and `pointwise` holds the log
likelihood $\log p(y_t \mid \mu_t, \sigma)$ of each observed week. Posterior
predictive checks compare the simulated revenue under `predictive` with the
data, and the sensitivity checks on [Plotting](plotting) read each week's log
likelihood under `log_likelihood`.

Each distribution's pointwise and `_rng` versions sit next to the summed one
that `log_density` uses, as [Distributions](distributions) describes.

## Fitting

{class}`~mmmjax.Model` binds the blocks to the data and checks their argument
names before anything runs. That way, most names that nothing supplies or
declares fail when you build the model rather than during sampling, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#common-mistakes) shows.
Once the model passes those checks, {func}`~mmmjax.sample` fits it with the
No-U-Turn sampler.

```{code-cell} ipython3
model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import first_model_results

results = first_model_results(model)
```

The draws, warmup, and chains in that call are {func}`~mmmjax.sample`'s
defaults, and [Sampler settings](sampling.md#sampler-settings) explains why they
suffice for this model. {func}`~mmmjax.sample` returns the draws in an xarray
DataTree, and its `posterior` group labels every parameter by chain and draw
and by any axis its declaration named.

```{code-cell} ipython3
results["posterior"]
```

At the top of the output, the dimensions show the four chains and 1,000 draws
alongside every axis a declaration named. Because `annual_coefficients` was
declared with a shape rather than `dims`, its axis gets the plain name
`annual_coefficients_dim_0` and runs from 0 to 3.
[Sampling and diagnostics](sampling) covers the other groups and what to look
at before you trust a fit.

## Returns against the truth

Real data never records the right answer, but this data is simulated, so
`brand.truth` holds each channel's true return. Because the chains pass the
checks on [Sampling and diagnostics](sampling), you can go straight to
comparing the fit's returns with the truth.

```{code-cell} ipython3
names = list(channels.values())
truth = brand.truth.assign_coords(channel=names + ["Email"], paid_channel=names)
returns = mj.media_metrics(model, results, quantity="mu")
true_returns = truth[["roi"]].rename(paid_channel="channel").expand_dims(chain=[0], draw=[0])
mj.plot_media_metrics({"Model": returns, "Truth": true_returns})
```

{func}`~mmmjax.media_metrics` computes each channel's return for every draw,
and [Media effects](media_effects) covers it in full. `assign_coords` gives the
simulation's channels the model's names, and `expand_dims` gives the true
returns a single draw, so {func}`~mmmjax.plot_media_metrics` draws them as a
result of their own.

In the plot, each blue bar is the model's mean return with its 89 percent
interval, the orange bar beside it is the true return, and the dashed line
marks break-even. The first thing to check is whether each interval holds the
truth, and all ten do, though YouTube's true return of \$6.59 sits near the top
of its interval. This is only one fit of one dataset, so treat it as a single
check rather than proof that the intervals hold the truth as often as they
claim.

You can see from the plot that the means tend to sit low. Eight of the ten
fall below the truth, by as much as 39 percent for TikTok and 38 percent for
YouTube, while Streaming's \$4.04 and Generic search's \$3.07 come in above
their true \$3.48 and \$2.99.
[Priors](priors.md#why-the-returns-lean-low) explains why the fit's returns
lean low and how the ROI prior pulls them there.

TikTok, Influencer, Linear TV, and Streaming air in the same weeks, so the data
sees their lift together and leaves the ROI prior to split it among them. In
this fit, that split puts Streaming above its truth and the other three below
theirs.

The next page, [Plotting](plotting), tours mmmJAX's built-in plots on this
fit, from convergence checks to budget plans. Later in the guide, three other
pages use this fit to answer questions about the brand.

- [Media effects](media_effects) measures how much revenue each channel,
  email, and treatment produced, and goes on to returns and response curves.
- [Recovering the truth](recovery) checks the fit's other answers against the
  simulation, from week-by-week contributions to response curves.
- [Budget optimization](budgets) turns the fit into a spending plan and checks
  its plans for the current budget against the simulation.
