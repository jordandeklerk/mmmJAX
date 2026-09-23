---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data and scaling

Every model starts from prepared, scaled data, and the choices made there
shape what the blocks receive. This page looks at what
{func}`~mmmjax.prepare_data` and {func}`~mmmjax.fit_data_scaling` produce for
the model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)
```

## Roles and inputs

Every column you select gets a role, such as the outcome, media, spend, or
controls. The roles, together with a few inputs computed from the dates, are
the names a block can request.

```{code-cell} ipython3
list(data.model_inputs)
```

`time` and `day_of_year` give each week's position and its place in the
calendar, and the `media_` versions do the same for the exposure window.
`n_periods` counts the modeled weeks. `outcome_scaling` and `reference`
matter once the model is evaluated on data other than its training data,
which [Scenarios](scenarios) covers.

## Arrays and media history

Prepared arrays put time first, then the group when the data has one, then
the channel or column.

```{code-cell} ipython3
data.arrays["media"].shape, data.arrays["outcome"].shape
```

The media array has eight more rows than the outcome. They hold the history
weeks, which sit on their own `media_time` axis starting before the first
modeled week. Carryover reads them, and a block that carries exposure forward
keeps the last `n_periods` rows of the result so that it lines up with the
outcome, as `saturated[-n_periods:]` does in the first model.
{func}`~mmmjax.media_response` combines carryover, saturation, and that slice
in one call.

## Checking the data

{func}`~mmmjax.check_data` looks for problems that no model can fix, which
makes it worth reading before you write any blocks.

```{code-cell} ipython3
checks = mj.check_data(data)
pairs = checks["pairs"].to_dataset().to_dataframe()
pairs.set_index(["feature_a", "feature_b"])["correlation"].round(2)
```

TV and search impressions barely move together, with a correlation of 0.11
across the 156 weeks, and neither follows price, so the data can tell their
effects apart. When two channels do move together, the data says little about
how to split the credit between them, and the priors end up doing more of that
work. The other groups in `checks` report the share of weeks without exposure,
which is just over two thirds for TV, along with spending that has no matching
exposure and variance inflation factors for the predictors.

## Scaling

{func}`~mmmjax.fit_data_scaling` fits one transform per role, and
{class}`~mmmjax.Data` applies them, so the blocks only ever see scaled
values.

```{code-cell} ipython3
media_transform = scaling.transformations["media"]
outcome_transform = scaling.transformations["outcome"]
media_transform.scale, outcome_transform.offset, outcome_transform.scale
```

Each channel's exposure is divided by its median $m_c$ over the weeks with any
exposure, so the model sees $x_{tc} = z_{tc} / m_c$ for raw impressions
$z_{tc}$. A typical week of TV becomes one, and a week without TV stays at
zero. Revenue becomes $y_t = (r_t - \bar{r}) / s_r$, centered on its mean
$\bar{r}$ and divided by its standard deviation $s_r$, and price is
standardized the same way. Spend is never scaled, because the analysis
functions report returns per dollar. The scales are fitted once and reused for
every scenario the model is asked about, so new data is measured against the
training data.

## Units inside a block

The analysis functions convert their results back to revenue on their own,
but anything a block returns stays in the units it was computed in. A block
that should report dollars asks for `outcome_scaling` and converts.

```{code-cell} ipython3
def revenue_generated_quantities(key, outcome, mu, sigma, outcome_scaling):
    prediction = mj.normal_rng(key, mu, sigma)
    pointwise = mj.normal_logpdf(outcome, mu, sigma)
    expected_revenue = outcome_scaling.inverse_transform(mu)
    return {
        "predictive": {"prediction": prediction},
        "log_likelihood": {"pointwise": pointwise},
        "expected_revenue": expected_revenue,
    }


revenue_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_parameters=transformed_parameters,
    log_density=log_density,
    generated_quantities=revenue_generated_quantities,
    generated_dims={"expected_revenue": "time"},
)
revenue = mj.generate_quantities(revenue_model, results)
expected = revenue["generated_quantities"]["expected_revenue"].sum("time").mean()
round(float(expected)), round(float(data.arrays["outcome"].sum()))
```

{func}`~mmmjax.generate_quantities` runs the new block on the draws already in
`results`, which works because the parameters have not changed. The expected
revenue over the three years comes back in dollars, within about \$1,100 of the
31.4 million observed. JAX arithmetic doesn't carry axis names, so
`generated_dims` labels the new output by week. The simulated `prediction`
stays on the scale of the likelihood, which is also the scale of the observed
revenue stored next to it, so predictive checks compare like with like.

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
        "weeks": "n_periods",
    },
    constants={"max_lag": 8},
)


def named_transformed_parameters(
    impressions,
    price,
    weeks,
    max_lag,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
):
    carried = mj.geometric_adstock(impressions, alpha=retention, max_lag=max_lag)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    mu = intercept + saturated[-weeks:] @ coefficient + price @ control_coefficient
    return {"mu": mu}


named_model = mj.Model(
    parameters=parameters,
    data=named_data,
    transformed_parameters=named_transformed_parameters,
    log_density=log_density,
)
first_draw = results["posterior"].isel(chain=0, draw=0)
draw = {name: value.values for name, value in first_draw.items()}
float(abs(named_model.evaluate(draw)["mu"] - model.evaluate(draw)["mu"]).max())
```

:::{note}
Once `variables` is given, blocks see only the names it declares. Built-in
inputs such as `n_periods`, `reference`, and `outcome_scaling` need declaring
too.
:::

That is why the mapping keeps `outcome` under its own name, so the first
model's density still finds it, and declares `n_periods` as `weeks`. Constants
pass through unchanged, so `max_lag` stays a Python integer that JAX can use
to set an array's shape. The renamed model computes the same expected revenue
as the original.

## Groups

Passing `groups` to {func}`~mmmjax.prepare_data` adds a group axis after time
to every observation array. {func}`~mmmjax.simulate_data` makes three regions
unless told otherwise.

```{code-cell} ipython3
regional = mj.simulate_data(seed=7, complexity="simple", noise_scale=0.02)
regional_data = mj.prepare_data(
    regional.frame,
    time="week",
    groups=["region"],
    outcome="revenue",
    media=["linear_tv_impressions", "generic_search_impressions"],
    spend=["linear_tv_spend", "generic_search_spend"],
    channels=["TV", "Search"],
    controls=["price"],
    media_history=regional.media_history,
)
regional_data.arrays["media"].shape, regional_data.arrays["outcome"].shape
```

Every region has to cover the same weeks. Declarations can then use
`dims=("group", "channel")` to give each region its own values, and the
region names label the draws in the results.
