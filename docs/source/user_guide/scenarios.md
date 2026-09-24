---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Scenarios

Every analysis function answers its question by running your blocks again on
data that differs from the training data, with the same draws. You can do the
same with any data you prepare. Doing it well takes the right inputs and
blocks that respond to them in the right way, and this page covers both
before it ends with renaming the inputs a block requests. The examples use the
model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)
```

## New data

`model.prepare_data` turns a frame with the training columns into inputs for
the model, reusing the fitted scaling and the training calendar, and
`model.evaluate` runs `transformed_parameters` on them at chosen parameter
values. Replaying weeks the model was trained on makes a good first test,
because the result should match the fit exactly.

```{code-cell} ipython3
point = results["posterior"].mean(("chain", "draw"))
fitted = model.evaluate(point)["mu"]

quarter = example.frame.iloc[143:]
replayed = model.evaluate(point, model.prepare_data(quarter))["mu"]
round(float(abs(replayed - fitted[143:]).max()), 3)
```

At the posterior mean the replay misses by 0.82 standard deviations of
revenue, about \$8,400, in its first week. The quarter starts at the end of
September, and on its own it has no weeks before it, so the carryover from
August and September is gone. Replaying the eight weeks before the quarter
along with it, and keeping only the quarter's own weeks, brings that carryover
back.

```{code-cell} ipython3
extended = example.frame.iloc[135:]
replayed = model.evaluate(point, model.prepare_data(extended))["mu"][8:]
round(float(abs(replayed - fitted[143:]).max()), 3)
```

With the eight weeks before it included, the quarter reproduces the fit. The
analysis functions keep the earlier weeks in place on their own, so this
matters for data you prepare yourself, such as a forecast or a plan that never
ran. To run the generated quantities on every draw for such data, pass the
frame itself to {func}`~mmmjax.generate_quantities`, as in
`new_data=extended`, and drop the first eight weeks of the result as above. It
prepares the frame the way `model.prepare_data` does and does not accept that
method's output. The draws land in ArviZ's `predictions` group, the new
inputs in `predictions_constant_data`, and the log likelihood of any new
outcomes in `predictions_log_likelihood`, so checks such as `az.loo` never
mistake a scenario for the fit.

## Predictions and definitions

Because the blocks run again for every scenario, each formula in them is
either a prediction or a definition. A prediction should follow the scenario,
the way `mu` falls when a channel's exposure is removed. A definition fixes
the meaning of a quantity from the training data, and it has to stay put when
the data changes.

:::{admonition} Scenario rule
:class: important

A prediction reads the inputs it is given, so it follows the scenario. A
definition reads the training arrays from `reference`, so it stays put when
the data changes.
:::

A trend scaled by the length of the training window is a definition, and the
cell below replays the last quarter of 2022 through two versions
of a small trend model,

$$
y_t = \alpha + g\, \tau_t + \varepsilon_t, \qquad
\varepsilon_t \sim \operatorname{Normal}(0, \sigma), \qquad
\alpha, g \sim \operatorname{Normal}(0, 1), \qquad
\sigma \sim \operatorname{HalfNormal}(1).
$$

The versions differ only in the trend $\tau_t$. The first divides week $t$ by
the last training week, $\tau_t = t / T_{\text{train}}$, and the second
divides it by the last week of whatever data it receives, $\tau_t = t /
T_{\text{data}}$.

```{code-cell} ipython3
def trend_from_training(time, reference):
    trend = time / reference.time.max()
    return {"trend": trend}


def trend_from_current(time):
    trend = time / time.max()
    return {"trend": trend}


def trend_mean(trend, intercept, growth):
    mu = intercept + growth * trend
    return {"mu": mu}


def trend_density(outcome, mu, intercept, growth, sigma):
    target = mj.normal(intercept, 0.0, 1.0) + mj.normal(growth, 0.0, 1.0)
    target += mj.half_normal(sigma, 1.0) + mj.normal(outcome, mu, sigma)
    return target


middle = example.frame.iloc[39:52]
trend_parameters = {"intercept": mj.Real(), "growth": mj.Real(), "sigma": mj.Positive()}
point = {"intercept": 0.0, "growth": 1.0, "sigma": 1.0}
for definition in (trend_from_training, trend_from_current):
    trend_model = mj.Model(
        parameters=trend_parameters,
        data=mj.Data(data, scaling=scaling),
        transformed_data=definition,
        transformed_parameters=trend_mean,
        log_density=trend_density,
    )
    fitted_trend = trend_model.evaluate(point)["mu"][39:52]
    replayed_trend = trend_model.evaluate(point, trend_model.prepare_data(middle))["mu"]
    print(definition.__name__, round(float(abs(replayed_trend - fitted_trend).max()), 3))
```

The first version reproduces the fit exactly, because `reference.time` keeps
the last training week in every evaluation. The second misses by as much as
0.67 on the model's standardized scale, about \$6,900 of weekly revenue,
because the quarter's own last week becomes the divisor and stretches its
trend up to the level at the end of the data. Nothing raises an error, and every scenario on that model would be
quietly wrong. Any formula that defines a quantity from the data, such as a
normalization, a centering, or a coefficient implied by a prior on returns,
reads its training arrays from `reference` for this reason. Fitted scaling is
already anchored this way, so only the definitions you write yourself need
it.

## Arrays captured from outside a block

The opposite mistake computes a prediction from an array held in an ordinary
Python variable instead of a block argument. The model fits exactly as
before, but the array never changes, so every scenario is ignored.

```{code-cell} ipython3
training_media = scaling.transform(data).arrays["media"]


def captured_transformed_parameters(
    controls,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(training_media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated @ coefficient + controls @ control_coefficient
    return {"mu": mu}


captured_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=captured_transformed_parameters,
    log_density=log_density,
)
captured = mj.contributions(captured_model, results, quantity="mu")
captured["contribution_share"].mean(("chain", "draw")).to_series()
```

As a generative model this is still [A first model](first_model), line for
line, and for the training weeks it computes exactly what the first model
computes. Yet removing a channel changes nothing, so both channels appear to
contribute nothing. Anything a prediction depends on should arrive as a block
argument.

## Your own names

The names a block requests are yours to choose. A `variables` mapping on
{class}`~mmmjax.Data` renames inputs, and `constants` adds settings that are
not arrays.

```{code-cell} ipython3
named_data = mj.Data(
    data,
    scaling=scaling,
    variables={
        "outcome": "outcome",
        "impressions": "media",
        "price": "controls",
    },
    constants={"max_lag": 8},
)


def named_transformed_parameters(
    impressions,
    price,
    max_lag,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(impressions, alpha=retention, max_lag=max_lag)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated @ coefficient + price @ control_coefficient
    return {"mu": mu}


named_model = mj.Model(
    parameters=parameters,
    data=named_data,
    transformed_parameters=named_transformed_parameters,
    log_density=log_density,
)
point = results["posterior"].mean(("chain", "draw"))
renamed = named_model.evaluate(point)["mu"]
original = model.evaluate(point)["mu"]
bool((renamed == original).all())
```

:::{note}
Once `variables` is given, the only data inputs blocks see are the ones it
declares, while constants keep their own names. Built-in inputs such as
`reference` and `outcome_scaling` need declaring too, and a
declared `reference` holds its arrays under the new names, so a mapping entry
`"training": "reference"` gives a block `training.impressions`.
:::

That is why the mapping keeps `outcome` under its own name, so the first
model's density still finds it. Constants
pass through unchanged, so `max_lag` stays a Python integer that JAX can use
to set an array's shape. The renamed model computes the same expected revenue
as the original, because renaming inputs leaves every line of the generative
model from [A first model](first_model) unchanged.
