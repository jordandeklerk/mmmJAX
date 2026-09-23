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
blocks that respond to them in the right way, and this page covers both. The
examples use the model from [A first model](first_model).

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
first_draw = results["posterior"].isel(chain=0, draw=0)
draw = {name: value.values for name, value in first_draw.items()}
fitted = model.evaluate(draw)["mu"]

quarter = example.frame.iloc[143:]
replayed = model.evaluate(draw, model.prepare_data(quarter))["mu"]
round(float(abs(replayed - fitted[143:]).max()), 3)
```

The replay misses by more than three quarters of a standard deviation of
revenue, about \$7,900, in its first week. The quarter starts at the end of
September, and without the weeks before it, the carryover from August and
September is gone. A scenario needs the same history the training data had.

```{code-cell} ipython3
quarter_data = mj.prepare_data(
    quarter,
    time="week",
    outcome="revenue",
    media=["linear_tv_impressions", "generic_search_impressions"],
    spend=["linear_tv_spend", "generic_search_spend"],
    channels=["TV", "Search"],
    controls=["price"],
    media_history=example.frame.iloc[135:143],
)
replayed = model.evaluate(draw, model.prepare_data(quarter_data))["mu"]
round(float(abs(replayed - fitted[143:]).max()), 3)
```

With the eight weeks before it as history, the quarter reproduces the fit.
The analysis functions keep the training history in place on their own, so
this matters for data you prepare yourself, such as a forecast or a plan
that never ran. {func}`~mmmjax.generate_quantities` takes the same prepared
data through `new_data` and runs the generated quantities on every draw.

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
cell below replays a quarter from the middle of the data through two versions
of a small trend model.

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

The first version divides by the last training date, which `reference.time`
keeps in every evaluation, so the quarter gets back the trend it had when the
model was fitted. The second divides by the last date of whatever data it
receives, which stretches the quarter's trend up to the level at the end of
the data. Nothing raises an error, and every scenario on that model would be
quietly wrong. Any formula that defines a quantity from the data, such as a
normalization, a centering, or a coefficient implied by a prior on returns,
reads its training arrays from `reference` for this reason. Fitted scaling is
already anchored this way, so only the definitions you write yourself need
it. Under a `variables` mapping, `reference` is declared like any other input
and its arrays follow the new names, as in `training.impressions`.

## Arrays captured from outside a block

The opposite mistake computes a prediction from an array held in an ordinary
Python variable instead of a block argument. The model fits exactly as
before, but the array never changes, so every scenario is ignored.

```{code-cell} ipython3
training_media = scaling.transform(data).arrays["media"]


def captured_transformed_parameters(
    controls,
    n_periods,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(training_media, alpha=retention, max_lag=8)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated[-n_periods:] @ coefficient + controls @ control_coefficient
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

For the training weeks this model computes exactly what the first model
computes, yet removing a channel changes nothing, so both channels appear to
contribute nothing. Anything a prediction depends on should arrive as a block
argument.
