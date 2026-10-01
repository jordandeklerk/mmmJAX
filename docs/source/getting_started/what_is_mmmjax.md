# What is mmmJAX

With mmmJAX you trade a packaged model for a program you write yourself. This
page shows how that program follows from your model's math and how mmmJAX runs
it. The code here is a sketch, with `...` standing in for your model's details,
so turn to the [Quickstart](quickstart) for a complete model you can run.

## Write the math first

As in Stan, you write the generative model out in math before you code it. The
model below is a small one with a trend, carried-over and saturated media, and
Normal noise.

$$
\begin{aligned}
\tau_t &= e_t / 365.25, \\
h_{tc} &= \operatorname{Hill}\big(\operatorname{Adstock}(x_{t-8,c}, \dots, x_{tc};\, \rho_c);\, \kappa_c, 1\big), \\
\mu_t &= \alpha + \delta \tau_t + \sum_{c} \beta_c h_{tc}, \\
y_t &\sim \operatorname{Normal}(\mu_t, \sigma).
\end{aligned}
$$

Here $y_t$ is week $t$'s outcome, $x_{tc}$ is channel $c$'s media, and $e_t$
counts the days since the first week, so $\tau_t$ measures time in years. The
adstock and Hill curves from [Introduction to MMM](intro_to_mmm.md#carryover)
carry channel $c$'s media into the next eight weeks at retention $\rho_c$ and
saturate it at half-saturation point $\kappa_c$ and slope one.

The mean $\mu_t$ adds an intercept $\alpha$, growth $\delta$ per year, and a
coefficient $\beta_c$ for each channel, and $\sigma$ is the noise scale. You
give every unknown a prior that says what you believe about it before the fit.

$$
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 1), & \delta &\sim \operatorname{Normal}(0, 1), \\
\beta_c &\sim \operatorname{HalfNormal}(1), & \rho_c &\sim \operatorname{Beta}(2, 2), \\
\kappa_c &\sim \operatorname{LogNormal}(0, 0.5), & \sigma &\sim \operatorname{HalfNormal}(1).
\end{aligned}
$$

You get from the math to the code by sorting its symbols into four groups.

- $y_t$, $x_{tc}$, and $e_t$ are data, so mmmJAX provides them as the
  supplied names `outcome`, `media`, and `time`.
- $\alpha$, $\delta$, $\beta_c$, $\rho_c$, $\kappa_c$, and $\sigma$ are
  unknowns, so each becomes one of your names declared in `parameters`, here
  `intercept`, `growth`, `coefficient`, `retention`, `half_saturation`, and
  `sigma`. The subscript $c$ becomes `dims="channel"`, an axis mmmJAX
  supplies.
- $\tau_t$ is computed from the data alone, so `transformed_data` returns it
  under your name `trend`.
- $\mu_t$ depends on the unknowns, so `transformed_parameters` returns it
  under your name `mu`. $h_{tc}$ is a step toward it that never leaves the
  block.

Each statement written with $\sim$, read as "is distributed as," adds one
term to `target` in `log_density`, so $y_t \sim \operatorname{Normal}(\mu_t,
\sigma)$ becomes `mj.normal(outcome, mu, sigma)`. The one for $y_t$ is the
likelihood and the other six are priors. The support a prior implies,
such as $\rho_c$ between zero and one, is what you declare in `parameters`.

```python
parameters = {
    "intercept": mj.Real(),
    "growth": mj.Real(),
    "coefficient": mj.Positive(dims="channel"),
    "retention": mj.Interval(0.0, 1.0, dims="channel"),
    "half_saturation": mj.Positive(dims="channel"),
    "sigma": mj.Positive(),
}


def transformed_data(time):
    trend = time / 365.25
    return {"trend": trend}


def transformed_parameters(
    media,
    trend,
    intercept,
    growth,
    coefficient,
    retention,
    half_saturation,
):
    saturated = ...
    mu = intercept + growth * trend + saturated @ coefficient
    return {"mu": mu}


def log_density(outcome, mu, intercept, growth, coefficient, retention, half_saturation, sigma):
    target = mj.normal(intercept, 0.0, 1.0)
    target += mj.normal(growth, 0.0, 1.0)
    target += mj.half_normal(coefficient, 1.0)
    target += mj.beta(retention, 2.0, 2.0)
    target += mj.lognormal(half_saturation, 0.0, 0.5)
    target += mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target
```

When you can't tell where a name in a block comes from, find its symbol in
the model's math. Its group tells you whether mmmJAX, `parameters`, or an
earlier block provides it, and each model in the User Guide lists its symbols
in a table with the name each one takes.

## A model is a program

A full model usually adds one more block, `generated_quantities`, to that
skeleton. That block runs once for each saved draw and uses the line for $y_t$
again, this time to simulate new outcomes.

You then pass the blocks to {class}`~mmmjax.Model` in the order they run. Since
{class}`~mmmjax.Model` knows each block by the keyword you pass it under, such
as `log_density=`, you can name the functions however you like.

```python
def transformed_data(...):
    # Features computed from the data alone, once for each dataset.
    return {...}


# A declaration of each parameter's shape and support.
parameters = {...}


def transformed_parameters(...):
    # Quantities built from parameters and data, every time the density is evaluated.
    return {...}


def log_density(...):
    # The priors and the likelihood, summed into one number.
    return target


def generated_quantities(...):
    # Predictions and the pointwise log likelihood, once for each saved draw.
    return {"predictive": ..., "log_likelihood": ...}


model = mj.Model(
    data=mj.Data(...),
    transformed_data=transformed_data,
    parameters=parameters,
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
```

`"predictive"` and `"log_likelihood"` are supplied names that decide where
mmmJAX stores the draws. With prepared data, mmmJAX names both `"outcome"` so
they line up with the observed data. `"log_prior"` is supplied too, for the
prior terms, and any other key you return is yours to name.

Only `parameters` and `log_density` are required, and you add the other blocks
as your model needs them. The block you put a calculation in decides how often
it runs, so work that depends only on the data belongs in `transformed_data`.

## How blocks get their inputs

You never call a block or pass it arguments yourself, because mmmJAX calls
each block, reads the argument names in its signature, and passes in whatever
each name refers to. So every name a block asks for is either a supplied name
that mmmJAX provides or one of your own names.

In a packaged MMM the supplied inputs stay hidden, since the model reads the
data and the calendar for you. In mmmJAX you ask for each one yourself, and a
block receives only the inputs its signature names. The sketch below shows this
with another model that builds a season from the calendar and reads price from
the treatments.

```python
parameters = {"roi": mj.Positive(dims="channel"), "sigma": mj.Positive()}


def transformed_data(day_of_year):
    season = ...
    return {"season": season}


def transformed_parameters(media, treatments, season, roi):
    mu = ...
    return {"mu": mu}


def log_density(outcome, mu, roi, sigma):
    target = ...
    return target
```

`day_of_year`, `media`, `treatments`, and `outcome` are supplied names in this
sketch. `roi` and `sigma` are yours as keys of `parameters`, `season`
as a key `transformed_data` returns, and `mu` as a key `transformed_parameters`
returns. Since `season` is yours, you could rename it in both blocks and
nothing would change. Names flow forward only, so a block can use what an
earlier block returned but never what a later one computes.

:::{admonition} Supplied names are fixed
:class: important

A supplied name has to be spelled exactly as mmmJAX spells it, so `media`
works and `impressions` fails. The data's axes in `dims` are supplied names
too, so `"channel"` has to match the data's name for that axis, while an axis
you add yourself can have any name. Your names can be anything that isn't
already a supplied name, but every block that uses one must spell it the same
way. A parameter called `spend` fails when the model is built, because
mmmJAX already supplies that name.
:::

mmmJAX ignores the order of the arguments, since it matches each argument by
name. The one exception is the random key, which `generated_quantities` takes
first by position, so its name is yours.

:::{admonition} Settings that aren't data
:class: tip

A value that sets an array's shape, such as a carryover length, has to be a
plain Python integer. Pass it as a constant, as in
`mj.Data(data, scaling=scaling, constants={"max_lag": 8})`, and ask for
`max_lag` by name in any block that needs it.
:::

### Data arrives by role

Your data arrives by role rather than by column, because roles are what mmmJAX
scales and rebuilds for new data. `media` is one array with a channel axis,
`treatments` holds price as one of its columns, and both come in already
scaled. [Data and scaling](../user_guide/data.md#supplied-names) lists the
supplied names the guide's data offers with what each holds and its axes, and
{attr}`~mmmjax.PreparedData.model_inputs` lists the ones your own data offers.

### Common mistakes

Each of these fails with an error that names the problem. The first asks for a
column instead of the supplied name that holds it.

```python
# Fails when the model is built, because price is a column, not a supplied name.
def transformed_parameters(media, price, season, roi):
    ...
```

```ipythontb
ValueError: transformed_parameters requests unknown input 'price'. Use a selected data role, declared
constant, or declared parameter. Available inputs are controls, day_of_year, media, media_day_of_year,
media_time, n_periods, organic_media, outcome, outcome_scaling, reference, roi, season, sigma, spend,
time, treatments.
```

The error lists every name that would have worked, and price is missing because
{func}`~mmmjax.prepare_data` stacks the columns you list as treatments, in the
order you listed them, into one `treatments` array. To use price on its own,
ask for `treatments` and take its column.

```python
# Works, because price is the first column of treatments.
def transformed_parameters(media, treatments, season, roi):
    price = treatments[:, 0]
    ...
```

The second mistake goes the other way and declares a parameter that no block
asks for.

```python
# Fails when the model is built, because no block requests intercept.
parameters = {
    "roi": mj.Positive(dims="channel"),
    "sigma": mj.Positive(),
    "intercept": mj.Real(),
}
```

```ipythontb
ValueError: log_density or transformed_parameters must request every declared parameter. Missing
parameters ['intercept']
```

The third mistake, a misspelled name in `log_density`, still lets the model
build.

```python
# Builds, then fails the first time log_density runs.
def log_density(outcome, muu, roi, sigma):
    ...
```

```ipythontb
ValueError: log_density requests unknown input 'muu', which transformed_parameters does not return.
Return it from transformed_parameters or use a selected data role, declared constant, or parameter.
Available inputs are controls, day_of_year, media, media_day_of_year, media_time, mu, n_periods,
organic_media, outcome, outcome_scaling, reference, roi, season, sigma, spend, time, treatments
```

mmmJAX assumes that any name only `log_density` or `generated_quantities`
asks for comes from `transformed_parameters`, so it can't catch this one until
the block runs. This time the list includes `mu`, because
`transformed_parameters` has returned it by then.

## How parameters get their priors

A declaration gives a parameter its shape and support, and `log_density` gives
it a prior.

```python
parameters = {"sigma": mj.Positive()}


def log_density(outcome, mu, sigma):
    target = mj.half_normal(sigma, 1.0)
    target += mj.normal(outcome, mu, sigma)
    return target
```

`mj.Positive()` keeps `sigma` above zero, and the half-normal term is its
prior. The sampler moves through an unconstrained space that holds `sigma` as
its logarithm, so mmmJAX adds the Jacobian adjustment for that change of
variables.

```python
# The density as you wrote it, at sigma = 0.5.
model.log_prob({"sigma": 0.5})
# The same plus log(0.5), with sigma held as its logarithm.
model.log_density({"sigma": jnp.log(0.5)}, model.data)
```

As the calls show, {meth}`~mmmjax.Model.log_prob` and
{meth}`~mmmjax.Model.log_density` differ by that adjustment and nothing else.

:::{admonition} The sampler won't stop you
:class: warning

A parameter with no term in `log_density` gets a flat prior over its support,
and mmmJAX will sample a model with more channels and curve shapes than the
data can separate. [Priors](../user_guide/priors) checks what each prior
implies, {func}`~mmmjax.psense_summary` measures how much the answer depends on
it, and {func}`~mmmjax.check_data` flags inputs that move together. Judging
what the data can identify is still up to you.
:::

## The pieces are plain functions

Some of mmmJAX's functions work on arrays and go inside your blocks, like the
media transformations, the seasonal and trend bases, the calibration helpers,
and the distributions. Others take the model and its results and run outside
it, like sampling, the analyses, the budget optimizer, and the plots.

```python
# Inside a block, on arrays.
carried = mj.geometric_adstock(media, alpha=retention, max_lag=max_lag)
saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
season = mj.fourier_features(day_of_year, period=365.25, order=2)
target += mj.normal(outcome, mu, sigma)

# Outside the model, on the model and its results.
results = mj.sample(model)
curves = mj.response_curves(model, results, quantity="mu")
plan = mj.optimize_budget(model, results, quantity="mu")
mj.plot_response_curves(curves, plan=plan)
```

Because a piece shapes the model only when your blocks call it, there's no
setting that switches it on, and any JAX function of your own can take its
place. The [API reference](../api/index) lists every piece, so look there
before you write your own.

## The model is a log density

A model doesn't know how it will be fit, and it doesn't have to, because a
sampler needs only three of its methods.

```python
position = model.initialize_random(key)
value, gradient = jax.value_and_grad(model.log_density)(position, model.data)
parameters = model.constrain(position)
```

The density is an ordinary JAX function, so it runs on a GPU when JAX finds
one, and any JAX sampler can fit it. {func}`~mmmjax.sample` runs NUTS on it,
and [Inference](../user_guide/inference) fits the same function with BlackJAX
and NumPyro directly.

### Blocks are traced

JAX traces your blocks with placeholder values before it compiles them, so a
block that breaks a tracing rule fails with an error from JAX rather than from
mmmJAX. The most common case is a Python `if` on a parameter.

```python
# Fails once the block is compiled, as mj.sample does, since threshold has no value then.
if threshold > 1.0:
    response = exposure - threshold
else:
    response = exposure
```

```ipythontb
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
...
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
```

`jnp.where` moves the same choice inside JAX, so the block traces cleanly.

```python
response = jnp.where(threshold > 1.0, exposure - threshold, exposure)
```

[What JAX needs from a
function](../user_guide/functions.md#what-jax-needs-from-a-function) covers
the other cases, such as calls to NumPy.

## How analyses get their answers

An analysis function answers its question by changing the data and running
your blocks again. The code below works out by hand what a dollar more on price
does at one posterior draw.

```python
# The parameter values at one posterior draw.
draw = ...

before = model.evaluate(draw)["mu"]
raised = model.prepare_data(frame.assign(price=frame["price"] + 1.0))
after = model.evaluate(draw, raised)["mu"]
effect = after - before
```

{meth}`~mmmjax.Model.evaluate` runs `transformed_parameters` at the values you
give it, and {meth}`~mmmjax.Model.prepare_data` scales the changed copy the way
the training data was scaled. The analysis functions do the same and compare
whichever output you name as `quantity`, but each one changes the data in its
own way.

- {func}`~mmmjax.contributions` sets each channel's exposure to zero and each
  treatment to its baseline level.
- {func}`~mmmjax.response_curves` scales a channel's spending up and down.
- {func}`~mmmjax.optimize_budget` tries new splits of the budget.

Because each one undoes the scaling before it reports, answers come back in
the outcome's own units. All of these answers are what your model implies
under its assumptions, so they add no causal evidence of their own.

Contributions and response curves are computed for every draw, so they carry
the posterior's uncertainty. The budget optimizer instead picks one plan by its
posterior mean response and reports that plan's response for every draw.

:::{admonition} Blocks run again on changed data
:class: important

Anything a prediction depends on has to arrive as a block argument. Anything
that fixes what a parameter means has to read the training inputs from the
`reference` argument, because that argument holds them unchanged in every
scenario.
:::

```python
# Moves with the scenario, so doubling every week's spend leaves media / peak unchanged.
peak = jnp.max(media, axis=0)

# Stays put, because reference.media is the training data in every scenario.
peak = jnp.max(reference.media, axis=0)
```

[Scenarios](../user_guide/scenarios) puts both rules to work on a forecast of
the next quarter.

## If you know Stan

The blocks line up with a Stan program's, so most of what you know carries
over.

- `transformed_data`, `parameters`, `transformed_parameters`, and
  `generated_quantities` match the Stan blocks of the same names.
- `log_density` plays the part of the `model` block.
- The prepared data stands in for the `data` block. Its variables are the
  supplied names, and sizes such as `N` and `C` come from the data instead of
  declarations.
- There's no `functions` block, since a block can call any Python function
  that JAX can trace.

In Stan every variable an earlier block declares is in scope, while in mmmJAX a
block names each input it needs as an argument. Only what a block returns
reaches the blocks after it, as [How blocks get their
inputs](#how-blocks-get-their-inputs) shows.

Transformed parameters aren't saved with the draws, so if you need one such as
`mu`, return it from `generated_quantities` or recompute it with
{meth}`~mmmjax.Model.evaluate`.

### Line by line

Since most lines translate one for one, each Stan line below sits as a comment
above its mmmJAX counterpart.

```python
# vector[4] seasonal;
"seasonal": mj.Real(4),
# vector<lower=0>[C] roi;
"roi": mj.Positive(dims="channel"),
# vector<lower=0, upper=1>[C] retention;
"retention": mj.Interval(0.0, 1.0, dims="channel"),

# target += normal_lpdf(seasonal | 0, 1);
target = mj.normal(seasonal, 0.0, 1.0)
# target += normal_lpdf(outcome | mu, sigma);
target += mj.normal(outcome, mu, sigma)

# outcome_rep = normal_rng(mu, sigma);
prediction = mj.normal_rng(key, mu, sigma)
# log_lik[n] = normal_lpdf(outcome[n] | mu[n], sigma);
pointwise = mj.normal_logpdf(outcome, mu, sigma)

# fit = model.sample(adapt_delta=0.95, max_treedepth=12)
results = mj.sample(model, target_accept=0.95, max_tree_depth=12)
```

There's no `~`, and each distribution function keeps every constant, as
`target +=` does. The elementwise `_logpdf` form replaces the loop Stan needs
for a pointwise log likelihood.

{class}`~mmmjax.LowerBound`, {class}`~mmmjax.UpperBound`,
{class}`~mmmjax.Simplex`, and {class}`~mmmjax.CorrelationCholesky` cover
Stan's other common constraints, and mmmJAX adds each declaration's Jacobian as
Stan does. You can write a constrained type mmmJAX lacks, such as `ordered`, as
a class that follows the {class}`~mmmjax.Parameterization` protocol.

:::{admonition} Split the random key
:class: warning

`generated_quantities` takes an explicit random key, while Stan's `_rng`
functions draw from a generator Stan manages. Split it with `jax.random.split`
when the block makes more than one draw.
:::

{func}`~mmmjax.sample` runs NUTS with Stan's windowed warmup, so tuning it will
feel familiar. Its `target_accept` plays the part of `adapt_delta`,
`max_tree_depth` of `max_treedepth`, and `mass_matrix` of `metric`, and all
three share Stan's defaults.
