---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A first model

This page builds the marketing mix model that the rest of the guide uses. The
brand has ten paid channels, an email newsletter, two controls, a price and
promotions, a trend, and a yearly season. You'll write the model out in
math, turn the math into blocks, and fit it. Most later pages start from this model and look at or change one
part of it, so run it once from top to bottom before you read on.

## The data

The data comes from {func}`~mmmjax.simulate_data` with seed 7 and one
national series. [The example data](example_data) describes it, and it's
prepared and scaled as in [Data and scaling](data.md). `channels` maps each
channel's column stem to the name your results will carry.
{func}`~mmmjax.prepare_data` gives each column the model uses a role, and
{func}`~mmmjax.fit_data_scaling` fits the scaling the model works on.

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

Demand and the holidays go in as `controls`. Both move revenue, and the brand
spends more in the weeks they run high, so leaving them out would credit
their lift to whichever channels ran then. The model adjusts for a control and
never reports its effect.

Price and promotions go in as `treatments`, the non-media inputs you set
yourself and want measured. Zero makes no sense as a price, so the analyses
report a treatment against a baseline level, such as the lowest price in the
data.

Email's sends go in as `organic_media` under the name Email. They carry over
and saturate like a paid channel's impressions, but the newsletter costs
nothing, so email gets a contribution and no return on investment.
[Data and scaling](data.md) says more about each role.

## The model

The model explains standardized revenue as a sum. A baseline follows a trend
and a yearly season, each paid channel and email add their part once their
exposure is carried forward and saturated, and the controls and the
treatments add their own effects. Normal noise sits around the total. Week $t$
runs over the modeled weeks, $c$ over the ten paid channels, $o$ over the
organic channels (here just email), $j$ over the two controls, and $i$ over
the two treatments. The subsections below spell the model out one piece at a
time, from the data transformations to the priors.

### Data transformations

The model works on scaled data. Each paid channel's impressions $z_{tc}$ are
divided by $m_c$, the channel's median over the weeks with any exposure, and
email's sends $n_{to}$ are divided by their own median $m_o$ in the same way.
Revenue $R_t$, each control $q_{tj}$, and each treatment $w_{ti}$ are
standardized with their mean and standard deviation over the modeled weeks.
The trend $\tau_t$ counts years, the days $e_t$ from the first training week
to week $t$ divided by the days in an average year,

$$
\begin{gathered}
x_{tc} = \frac{z_{tc}}{m_c}, \qquad
x_{to} = \frac{n_{to}}{m_o}, \qquad
y_t = \frac{R_t - \bar{R}}{s_R}, \\
p_{tj} = \frac{q_{tj} - \bar{q}_j}{s_j}, \qquad
g_{ti} = \frac{w_{ti} - \bar{w}_i}{s_i}, \qquad
\tau_t = \frac{e_t}{365.25}.
\end{gathered}
$$

The season runs on $d_t$, the day of the year of week $t$'s date.
Spending $v_{tc}$ stays in dollars, and $S_c = \sum_t v_{tc}$ is channel $c$'s
total over the training weeks. Email has no spending.

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

Everything except the noise $\varepsilon_t$ is the mean $\mu_t$, which the code
returns as `mu`, so the likelihood is $y_t \sim \operatorname{Normal}(\mu_t,
\sigma)$. The intercept is $\alpha$, $\delta_1$ is the trend's growth per
year and $\delta_2$ its curvature, and $a_k$ and $b_k$ weigh the sine and
cosine of the season's $k$th harmonic. Paid channel $c$'s exposure after
carryover and saturation is $h_{tc}$, with coefficient $\beta_c$, retention
rate $\rho_c$, and half-saturation point $\kappa_c$. Email's is $h_{to}$, with its own
coefficient $\lambda_o$, retention rate $\rho_o$, and half-saturation point
$\kappa_o$. Control $j$ has coefficient $\gamma_j$, treatment $i$ has
coefficient $\theta_i$, and $\sigma$ is the noise scale.

The controls and the treatments enter the equation the same way, but the
analyses read them differently. A control only adjusts the other estimates, so
no analysis reports $\gamma_j$. A treatment is reported against a baseline
level $w_i^{0}$, and because its term is linear, moving it from $w_i^{0}$ to
the levels in the data adds

$$
s_R\, \theta_i \sum_t \frac{w_{ti} - w_i^{0}}{s_i}
$$

dollars of revenue over the training weeks.

### Media transformation

Adstock averages the current week and the eight before it with geometric
weights, and the Hill curve reaches half its maximum at $\kappa$,

$$
\operatorname{Adstock}\big(\{x_{t-\ell}\}_{\ell=0}^{L};\, \rho\big)
= \frac{\sum_{\ell=0}^{L} \rho^{\ell}\, x_{t-\ell}}{\sum_{\ell=0}^{L} \rho^{\ell}},
\qquad
\operatorname{Hill}(u;\, \kappa, s) = \frac{u^{s}}{u^{s} + \kappa^{s}}.
$$

Weeks before the first one in the data count as zero exposure. HillAdstock
applies Adstock first and then Hill, with the slope $s$ fixed at one in this
model. Email's sends go through the same transformation as the paid
channels' impressions.

### Return on investment

The coefficient $\beta_c$ isn't a parameter of its own. The model samples
channel $c$'s return on investment $r_c$ instead and sets

$$
\beta_c = \frac{r_c S_c}{s_R \sum_{t} h_{tc}},
$$

with $h_{tc}$ computed from the training exposure. The revenue scale $s_R$
turns a change in $y_t$ back into dollars, so channel $c$'s contribution over
the training weeks, $s_R \beta_c \sum_t h_{tc}$, comes to $r_c S_c$ dollars.
The prior goes on $r_c$ because it's easier to hold a belief about the revenue
a dollar of spending returns than about a coefficient on scaled media.

### Email's share of revenue

Email has no spending, so there's no return on investment to put a prior on.
The model samples $\phi_o$ instead, the share of the training weeks' revenue
that email produced, and sets

$$
\lambda_o = \frac{\phi_o \sum_t R_t}{s_R \sum_{t} h_{to}},
$$

with $h_{to}$ computed from the training sends. Email's contribution over the
training weeks, $s_R \lambda_o \sum_t h_{to}$, then comes to $\phi_o \sum_t
R_t$ dollars.

### Priors

Each parameter has its own fixed prior, so the model has no hyperpriors. A
regional model would add them by drawing each region's coefficients from a
shared distribution with its own priors.

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

Each distribution takes the same parameters as its mmmJAX function, so
$\operatorname{LogNormal}(1, 0.6)$ is the distribution of a variable whose
logarithm has mean one and standard deviation 0.6, $\operatorname{Beta}(2, 2)$
has both shape parameters at two, and $\operatorname{HalfNormal}(1)$ has scale
one.

Most priors describe the scaled data. A scale of one on the intercept, the
growth, and the controls already spans standardized revenue, and the Fourier
coefficients get half that. The curvature gets 0.25 because $\tau_t^2$
outgrows $\tau_t$ after the first year, and at a scale of one the squared
term alone would swing revenue far past anything in the data by the last
week. Email's retention and half-saturation take the paid channels' priors,
since its sends are scaled by their median in the same way.

The treatments get 0.25 too, a belief that moving the price or promotions by
one standard deviation seldom moves weekly revenue by more than half a
standard deviation. The data alone can't tell the price apart from the trend
and the promotions, since the price climbs with one and falls with the other,
so a wider prior would let the fit credit the price with the brand's growth.

The ROI and share priors are the exceptions, on scales of their own, since
$r_c$ counts dollars of revenue per dollar spent and $\phi_o$ is a fraction of
revenue. $\operatorname{Beta}(2, 98)$ has mean 0.02, a belief that email is a
small channel that brings in about 2 percent of revenue. [Priors](priors)
covers what these choices claim about revenue and how to check them, and the
sections below write each part of this specification as code.

## From math to code

You write the code by sorting these symbols into the four groups of
[Write the math first](../getting_started/what_is_mmmjax.md#write-the-math-first),
and each prior and the likelihood, the statements written with $\sim$,
becomes one term of `target` in `log_density`.

:::{admonition} The code follows the math
:class: important

Every argument in the block signatures below, apart from the random key,
carries a symbol above. A name you can't place usually means a symbol you
skipped, so find it in the math before you read on.
:::

The tables below sort the symbols by where each name comes from, one tab for
each place.

::::{tab-set}

:::{tab-item} Supplied by mmmJAX

The data arrive under names mmmJAX supplies, with the media, controls,
treatments, and revenue already scaled.

| Symbol | Name in the code |
| --- | --- |
| $y_t$ | `outcome` |
| $x_{tc}$ | `media` |
| $x_{to}$ | `organic_media` |
| $p_{tj}$ | `controls` |
| $g_{ti}$ | `treatments` |
| $e_t$ | `time` |
| $d_t$ | `day_of_year` |
| $x_{tc}$ and $x_{to}$ in the training weeks | `reference.media` and `reference.organic_media` |
| $v_{tc}$ | `reference.spend` |
| $R_t$ | `outcome_scaling.inverse_transform(reference.outcome)` |
| $s_R$ | `outcome_scaling.scale` |

:::

:::{tab-item} Declared in parameters

Each unknown is one of your names, and its subscript becomes the axis named in
its `dims`.

| Symbol | Name in the code | Axis in `dims` |
| --- | --- | --- |
| $\alpha$ | `intercept` | none |
| $\delta_1$ | `growth` | none |
| $\delta_2$ | `curvature` | none |
| $a_1$, $a_2$, $b_1$, $b_2$ | `annual_coefficients`, in that order | none, a plain shape of four |
| $r_c$ | `roi` | `"channel"` |
| $\rho_c$ | `retention` | `"channel"` |
| $\kappa_c$ | `half_saturation` | `"channel"` |
| $\phi_o$ | `organic_share` | `"organic_channel"` |
| $\rho_o$ | `organic_retention` | `"organic_channel"` |
| $\kappa_o$ | `organic_half_saturation` | `"organic_channel"` |
| $\gamma_j$ | `control_coefficient` | `"control"` |
| $\theta_i$ | `treatment_coefficient` | `"treatment"` |
| $\sigma$ | `sigma` | none |

:::

:::{tab-item} Computed in the blocks

The quantities built from the data and the unknowns take names of your own
too. `mu` is the only one `transformed_parameters` returns, since no other
block asks for the steps on the way to it, and {func}`~mmmjax.roi_coefficient`
totals $S_c$ from `reference.spend` itself.

| Symbol | Name in the code | Where it's computed |
| --- | --- | --- |
| $\tau_t$ | `trend` | Returned by `transformed_data` |
| The sines and cosines of $d_t$ | `annual` | Returned by `transformed_data` |
| $\mu_t$ | `mu` | Returned by `transformed_parameters` |
| $h_{tc}$ and $h_{to}$ in the training weeks | `trained` and `organic_trained` | Local to `transformed_parameters` |
| $h_{to}$ in the weeks being evaluated | `organic_saturated` | Local to `transformed_parameters` |
| $\beta_c$ | `coefficient` | Local to `transformed_parameters` |
| $\lambda_o$ | `organic_coefficient` | Local to `transformed_parameters` |
| $\sum_t R_t$ | `total_revenue` | Local to `transformed_parameters` |

:::

::::

A few symbols never get a name in the code.

- The raw columns $z_{tc}$, $n_{to}$, $q_{tj}$, and $w_{ti}$ never reach a
  block, since mmmJAX scales them first with the medians, means, and standard
  deviations in `scaling`.
- Revenue's mean $\bar{R}$ reaches the blocks only inside
  `outcome_scaling.inverse_transform`, which applies it.
- The noise $\varepsilon_t$ lives in `mj.normal(outcome, mu, sigma)`, the
  likelihood line of `log_density`.
- The two harmonics, the eight lags, and the slope $s$ are the fixed numbers
  `order=2`, `max_lag=8`, and `slope=1.0`.
- The baseline level $w_i^{0}$ belongs to the analyses rather than the model,
  as [Media effects](media_effects) shows.

## Parameters

You declare each parameter with its support and its axes.

```{code-cell} ipython3
parameters = {
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "curvature": mj.Real(),
    # No data axis runs over the four Fourier weights, and they need no labels, so a plain shape does.
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
```

{class}`~mmmjax.Positive` and {class}`~mmmjax.Interval` keep a parameter
inside its support, and {class}`~mmmjax.LowerBound`,
{class}`~mmmjax.UpperBound`, {class}`~mmmjax.Simplex` for shares that sum to
one, and {class}`~mmmjax.CorrelationCholesky` for correlation matrices cover
other supports. None of the declarations carries a prior, since the priors go
in `log_density` below.

`dims="channel"` gives a parameter one value per channel, labeled with the ten
names from the data, and `dims="organic_channel"` gives it one labeled Email.
`dims="control"` and `dims="treatment"` give it one per control or treatment
column.

:::{admonition} Data axes and your own axes
:class: important

`"channel"` and the other three axes here come from the data, so they keep
mmmJAX's spelling and take the data's lengths and labels. An axis of your own
can have any name once you give its length or its labels, as
[Data and scaling](data.md#supplied-names) explains. The seasonal weights need
no named axis, so `mj.Real(4)` gives `annual_coefficients` a plain shape of
four.
:::

## Trend and seasonality

`transformed_data` computes the inputs that depend only on the data.

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

Supplied names such as `day_of_year` keep mmmJAX's spelling, and
[Data and scaling](data.md#supplied-names) lists every one. `annual` and
`trend` are your names, like the keys of `parameters`,
and you can rename one as long as every block that uses it changes too and the
new name isn't a supplied one. A renamed parameter also changes its key in
`priors` below.
:::

`transformed_data` runs when the model is built and again for new data, never
at every draw, and later blocks ask for its outputs by the keys it returns.
`day_of_year` holds each week's calendar day, and
{func}`~mmmjax.fourier_features` turns it into two sine columns and then two
cosine columns, the order of the four entries of `annual_coefficients`.

`time` counts the days since the first training week, and dividing it by
365.25 puts the trend $\tau_t$ in years. New weeks count from that same first
week, so a forecast carries the trend on past the training weeks, as
[Scenarios](scenarios) shows.

## Expected revenue

The model applies HillAdstock twice, once to the training exposure and once to
the exposure it's evaluating, so the transformation lives in a helper.

```{code-cell} ipython3
# mmmJAX fills only a block's arguments, so media here is whatever the caller passes.
def hill_adstock(media, retention, half_saturation):
    # Eight weeks is the longest you believe exposure keeps working. It sets an array shape,
    # so it stays a fixed int rather than a parameter the sampler learns.
    carried = mj.geometric_adstock(media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated
```

Each channel's exposure carries forward for up to eight weeks at its retention
rate and then passes through a saturation curve that reaches half its maximum
at `half_saturation`, so every extra impression is worth a little less than
the one before. A block can call any Python function written with JAX
operations, and [User-defined functions](functions) covers the rules.

`transformed_parameters` turns exposure into expected revenue.

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
    # reference holds the training data in every scenario, so the coefficients
    # keep their fitted values when an analysis changes media.
    trained = hill_adstock(reference.media, retention, half_saturation)
    # The prior sits on ROI because revenue per dollar is easier to judge than a scaled coefficient.
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    # With no spend, email has no ROI, so its prior sits on its share of revenue. The share
    # is of dollars, since standardized revenue sums to zero over the training weeks.
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = hill_adstock(media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}
```

Each parameter the block asks for holds that parameter's value at the
current draw, so `roi` arrives as ten values, one ROI per channel.

The first two statements compute $\beta_c$. `trained` is HillAdstock of the
training exposure in `reference.media`, and {func}`~mmmjax.roi_coefficient`
sets each coefficient so the channel's contribution over those weeks equals
its ROI times its total spending in `reference.spend`. `outcome_scaling.scale`
is the revenue scale $s_R$ that turns standardized revenue back into dollars.

:::{admonition} Coefficients come from the training data
:class: important

`reference` holds the training data in every scenario, while `media` holds
whatever the scenario sets. Analysis functions such as
{func}`~mmmjax.contributions` run the block again with one channel switched
off or a new plan in place of `media`, so `trained` and the spending come from
`reference` to keep the coefficients those of the fitted model.
:::

The next four statements compute email's $\lambda_o$ in the same way.
`organic_trained` is HillAdstock of the training sends in
`reference.organic_media`, `total_revenue` is the training weeks' revenue in
dollars, and {func}`~mmmjax.contribution_coefficient` sets the coefficient so
email's contribution over those weeks equals `organic_share` of that revenue.

The remaining statements build the trend and seasonal baseline, pass `media`
and `organic_media` through the same helper, and add the controls and the
treatments, each a product of its scaled columns and their coefficients.
Together they give the mean $\mu_t$ of the model equation. The block returns
it as `mu`, one of your names, which `log_density` asks for below and the
analyses take as `quantity="mu"`.

:::{admonition} Media history
:class: note

This model treats the weeks before the data as having no exposure. When you
know what aired before your first week, pass it to
{func}`~mmmjax.prepare_data` as `media_history`, and carryover in the first
weeks starts from it instead of from zero. The history must hold every exposure
column, email's sends included, so `media` and `organic_media` both have more
rows than the outcome, eight more for eight weeks of history. The block then
asks for the supplied `n_periods` and keeps only the last `n_periods` rows of
each once carryover has used the rest.

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
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.normal(curvature, 0.0, 0.25)
    target += mj.normal(annual_coefficients, 0.0, 0.5)
    target += mj.lognormal(roi, 1.0, 0.6)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    # You expect email to be small, and Beta(2, 98) puts its mean share of revenue at 2 percent.
    target += mj.beta(organic_share, 2.0, 98.0)
    target += mj.beta(organic_retention, 2.0, 2.0)
    target += mj.lognormal(organic_half_saturation, 0.0, 0.5)
    target += mj.normal(control_coefficient, 0.0, 1.0)
    # The price climbs with the trend. A wider prior would let the fit credit price with growth.
    target += mj.normal(treatment_coefficient, 0.0, 0.25)
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target
```

The first thirteen terms are the priors of the model above, in the same order,
and the last is its likelihood, $y_t \sim \operatorname{Normal}(\mu_t,
\sigma)$. Each function from [Distributions](distributions) returns the summed
log density of its first argument under the settings after it, so
`target += mj.lognormal(roi, 1.0, 0.6)` does what
`target += lognormal_lpdf(roi | 1, 0.6);` does in Stan. mmmJAX adds nothing to
this sum except the Jacobian adjustments its declarations make for their
constraints.

## The priors as objects

Some tools need the priors as objects they can draw from and score, and they
never run `log_density`. {func}`~mmmjax.sample_prior` draws from the priors
alone, which [Priors](priors) uses to check what the model believes before it
sees the data. {func}`~mmmjax.psense_summary` and {func}`~mmmjax.plot_psense`
score the posterior draws under them. A dictionary of {class}`~mmmjax.Prior`
objects states the same thirteen priors again for these tools.

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

A prior binds one of the distribution functions to fixed settings, so
`priors["roi"]` holds the lognormal with location 1.0 and scale 0.6 that
`log_density` writes for `roi`. Nothing ties the dictionary to the density,
so [Priors](priors) shows how to check that the two agree.

## Generated quantities

`generated_quantities` runs once for each saved draw. Its first argument is
the one mmmJAX doesn't fill by name. It gets a JAX random key for each draw,
and the `_rng` functions draw with it.

```{code-cell} ipython3
def generated_quantities(key, outcome, mu, sigma):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    # "predictive" and "log_likelihood" are supplied names that tell mmmJAX where to store
    # the draws, and "outcome" matches the observed data because ArviZ pairs them by name.
    return {
        "predictive": {"outcome": prediction},
        "log_likelihood": {"outcome": pointwise},
    }
```

For each draw of the parameters, `prediction` redraws every week's
standardized revenue $\tilde{y}_t \sim \operatorname{Normal}(\mu_t, \sigma)$
from the posterior predictive distribution, and `pointwise` is the log
likelihood $\log p(y_t \mid \mu_t, \sigma)$ of each observed week. Each
distribution's pointwise and `_rng` versions sit next to the summed one that
`log_density` uses, as [Distributions](distributions) describes.

The keys `predictive` and `log_likelihood` tell mmmJAX where to store the
draws. Posterior predictive checks compare the simulated revenue under
`predictive` with the data, and ArviZ compares models using each week's log
likelihood under `log_likelihood`. Both entries are named `outcome`, like the
observed revenue they describe, because ArviZ pairs groups by variable name.

## Fitting

{class}`~mmmjax.Model` binds the blocks to the data and checks their argument
names before anything runs, so a name that nothing supplies or declares fails
here rather than during sampling. The exception is a name only `log_density`
or `generated_quantities` asks for, which mmmJAX expects
`transformed_parameters` to return and checks the first time the block runs,
as [What is mmmJAX](../getting_started/what_is_mmmjax.md#common-mistakes)
shows. {func}`~mmmjax.sample` then fits the model with the No-U-Turn sampler.

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

Each of the four chains runs 1,000 warmup steps, which tune the sampler, and
then keeps 1,000 draws. {func}`~mmmjax.sample` returns them in an xarray
DataTree, whose `posterior` group labels every parameter by chain and draw and
by any axis its declaration named.

```{code-cell} ipython3
results["posterior"]
```

The channel parameters carry the ten channel names and the organic ones carry
Email. `control_coefficient` and `treatment_coefficient` carry their column
names, and `annual_coefficients` has a plain axis numbered 0 to 3.
[Sampling and diagnostics](sampling) covers the other groups and what to look
at before trusting a fit.

The next page, [Plotting](plotting), tours mmmJAX's built-in plots on this
fit, from convergence checks to budget plans. Three other pages ask the fit
questions.

- [Media effects](media_effects) measures how much revenue each channel,
  email, and treatment produced, and goes on to returns and response curves.
- [Recovering the truth](recovery) checks those answers against the
  simulation.
- [Budget optimization](budgets) turns them into a spending plan.
