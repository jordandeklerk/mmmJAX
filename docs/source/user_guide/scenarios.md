---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Scenarios

Every analysis function answers its question by running your blocks again,
with the same draws, on data that differs from the training data. You can do
the same with any data you prepare. To get it right you need the right inputs,
and blocks that respond to them correctly. This page covers both and ends by
renaming the supplied names a block asks for. The examples use the ten-channel
brand model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_results

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
plt.rcParams["date.converter"] = "concise"
```

## New data

`model.prepare_data` turns data with the training columns into model inputs,
and `model.evaluate` then runs `transformed_parameters` on those inputs at
whatever parameter values you choose.

:::{admonition} New data keeps the training calendar
:class: important

`model.prepare_data` scales new data with the fitted scaling, and its `time`
counts days from the first training week, not from the new data's own first
week. A week you prepare on its own gets the inputs it had in the fit.
:::

A good first test is to replay weeks the model was trained on, since the
result should match the fit exactly. The cell below replays the last quarter
of the data and prints how far each of its weeks lands from the fit.
`revenue_scale` converts the model's standardized revenue back to dollars.

```{code-cell} ipython3
point = results["posterior"].mean(("chain", "draw"))
fitted = model.evaluate(point)["mu"]
revenue_scale = scaling.transformations["outcome"].scale.item()

quarter = brand.frame.iloc[143:]
replayed = model.evaluate(point, model.prepare_data(quarter))["mu"]
miss = abs(replayed - fitted[143:]) * revenue_scale
miss.round(-2)
```

At the posterior mean the replay misses by about \$8,200 in its first week and
\$1,700 in its second, and the gap rounds to zero by the eighth. The quarter
starts at the end of September. On its own it has no weeks before it, so the
carryover of paid media and email from August and September is lost. The trend
gets the same `time` it had in the fit, and the media and email coefficients
read the training data from `reference`, as the next section explains, so the
lost carryover is the only gap. To bring it back, replay the eight weeks
before the quarter along with it and keep only the quarter's own weeks.

```{code-cell} ipython3
extended = brand.frame.iloc[135:]
replayed = model.evaluate(point, model.prepare_data(extended))["mu"][8:]
round(float(abs(replayed - fitted[143:]).max()), 3)
```

With the eight weeks before it included, the quarter reproduces the fit.

:::{admonition} Keep the weeks before your data
:class: warning

The analysis functions keep the earlier weeks in place for you, but data you
prepare yourself has to bring them along. Prepend the eight weeks before a
forecast, or before a plan that never ran, since the brand's adstock reaches
that far back. Then drop those weeks from the result.
:::

To run the generated quantities on every draw for data you prepare yourself,
pass the raw data to {func}`~mmmjax.generate_quantities`, as in
`new_data=extended`, and drop the first eight weeks of the result as above.
It prepares the data the way `model.prepare_data` does, so it won't take that
method's output. The draws land in ArviZ's `predictions` group, the new inputs
in `predictions_constant_data`, and the log likelihood of any new outcomes in
`predictions_log_likelihood`. That keeps checks such as `az.loo` from mistaking
a scenario for the fit.

## Predictions and definitions

The blocks run again for every scenario, so each formula in them is either a
prediction or a definition. A prediction, such as `mu`, falls when you remove
a channel's exposure. A definition uses the training data to fix what a
quantity means.

:::{admonition} Scenario rule
:class: important

A prediction reads the inputs it is given, so it follows the scenario. A
definition reads the training arrays from `reference`, which holds them
unchanged in every scenario, so it stays put when the data changes.
:::

The ROI coefficient in the brand's `transformed_parameters` is a definition.
In the math of [A first model](first_model) it sums over the training weeks
$\mathcal{T}$, whatever weeks $\mathcal{W}$ the model is evaluating. A second
version sums over $\mathcal{W}$ instead,

$$
\beta_c = \frac{r_c \sum_{t \in \mathcal{T}} v_{tc}}{s_R \sum_{t \in \mathcal{T}} h_{tc}}
\qquad \text{against} \qquad
\beta_c = \frac{r_c \sum_{t \in \mathcal{W}} v_{tc}}{s_R \sum_{t \in \mathcal{W}} h_{tc}},
$$

where $v_{tc}$ is channel $c$'s spending in week $t$ and $h_{tc}$ its exposure
after carryover and saturation. The two sets of weeks are the new symbols, and
each one maps to a pair of arrays that mmmJAX supplies.

- Over $\mathcal{T}$ the media and the spending are `reference.media` and
  `reference.spend`, which hold the training data in every scenario.
- Over $\mathcal{W}$ they're `media` and `spend`, which hold whatever data the
  model is given.

The brand's version reads the first pair, so `roi` is the return on the
training weeks' spend in every scenario. The cell below writes the second
version and keeps the rest of the brand's blocks.

On the training data both versions read the same arrays, so they fit the same
posterior and match `fitted` week for week. Each one then replays the last
quarter of 2022 at the posterior mean, with the eight weeks before it, and
prints its largest weekly miss in dollars.

```{code-cell} ipython3
def current_transformed_parameters(
    media,
    spend,
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
    trained = hill_adstock(media, retention, half_saturation)
    # Reads the scenario's own media and spend, so every scenario redefines roi.
    coefficient = mj.roi_coefficient(roi, trained, spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
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


current_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=current_transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)

last_quarter = brand.frame.iloc[31:52]
versions = {"reference.media and reference.spend": model, "media and spend": current_model}
replayed = {}
for arrays, candidate in versions.items():
    replayed[arrays] = candidate.evaluate(point, candidate.prepare_data(last_quarter))["mu"][8:]
    miss = abs(replayed[arrays] - fitted[39:52]) * revenue_scale
    print(arrays, round(float(miss.max()), -3))
```

Plotted in dollars, the two versions part ways across the whole quarter.

```{code-cell} ipython3
:tags: [hide-input]

revenue = scaling.transformations["outcome"]
weeks = last_quarter["week"].iloc[8:]

fig, axis = plt.subplots(layout="constrained")
axis.plot(weeks, revenue.inverse_transform(fitted[39:52]), color="black", label="Fit on the training data")
for arrays, style in zip(versions, ["--", "-"]):
    dollars = revenue.inverse_transform(replayed[arrays])
    axis.plot(weeks, dollars, style, label=f"Reading {arrays}")
axis.set_title("Expected revenue in the last quarter of 2022, replayed")
axis.legend(frameon=False)
plt.show()
```

The brand's own version, dashed, lies on the fit, because `reference` keeps
the training data in every evaluation. The second runs above the fit by as
much as \$39,000 in a week. Its coefficients return each channel's ROI on the
21 weeks it was given instead of the three training years, so the same `roi`
draws now imply a larger media effect. Nothing raises an error, yet every
scenario on that model would be wrong.

Email's coefficient is a definition too. It turns email's share of revenue
into a coefficient with `reference.organic_media` and `reference.outcome`, so
a scenario that doubles the sends or brings new revenue leaves that share
where the fit put it.

The brand's trend needs no `reference`, because the `time` in any data you
prepare counts days from the first training week. Dividing it by 365.25 gives
each week the same value in every scenario. A trend scaled by the data's own `time.max()`
would move with the scenario, and to stay a definition it would have to divide
by `reference.time.max()` instead.

Any other formula that defines a quantity from the data, such as a
normalization or a centering, should read its training arrays from `reference`
the same way. Fitted scaling is already anchored, so only the definitions you
write yourself need `reference`. A price scenario, for instance, changes a
treatment, and `model.prepare_data` standardizes the new prices with the
training mean and spread, so `treatment_coefficient` keeps the meaning it had
in the fit.

## Arrays captured from outside a block

The opposite mistake is computing a prediction from an array held in an
ordinary Python variable instead of a block argument. The model fits exactly
as before, but the array never changes, so no scenario can reach it. In the
math the media effect runs over the evaluated weeks $\mathcal{W}$, so its
exposure has to come from the supplied `media`. The cell below copies the
brand's `transformed_parameters` without `media` in its signature, so the
media effect reads `training_media` instead. It then
draws the new model's contributions with {func}`~mmmjax.plot_contributions`.

```{code-cell} ipython3
training_media = scaling.transform(data).arrays["media"]


def captured_transformed_parameters(
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
    trained = hill_adstock(reference.media, retention, half_saturation)
    coefficient = mj.roi_coefficient(roi, trained, reference.spend, outcome_scale=outcome_scaling.scale)
    organic_trained = hill_adstock(reference.organic_media, organic_retention, organic_half_saturation)
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = hill_adstock(training_media, retention, half_saturation) @ coefficient
    organic_saturated = hill_adstock(organic_media, organic_retention, organic_half_saturation)
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = controls @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}


captured_model = mj.Model(
    parameters=parameters,
    data=mj.Data(data, scaling=scaling),
    transformed_data=transformed_data,
    transformed_parameters=captured_transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
captured = mj.contributions(captured_model, results, quantity="mu")
mj.plot_contributions(captured)
```

The waterfall still credits email, price, and promotion, but the ten paid
channels get nothing and the baseline takes their revenue. That's the bug
showing, not a finding about the brand. Next to the brand's own model, the
paid channels' share of revenue drops to nothing.

```{code-cell} ipython3
paid = list(channels.values())
brand_shares = mj.contributions(model, results, quantity="mu")["contribution_share"]
brand_media = brand_shares.sel(channel=paid).sum("channel")
captured_media = captured["contribution_share"].sel(channel=paid).sum("channel")
round(float(brand_media.mean()), 3), round(float(captured_media.mean()), 3)
```

The brand's own model credits the paid channels with 18.1 percent of revenue,
and the captured version with 0.0. Yet as a generative model the captured
version is still the brand from [A first model](first_model), line for line,
and on the training weeks it computes exactly what the brand computes.
Removing a channel changes nothing only because the media effect never sees
the changed data. The coefficient in the same block still reads
`reference.media` and `reference.spend`, as a definition should. Only the
prediction is frozen, so pass anything a prediction depends on as a block
argument.

## Renaming supplied names

Supplied names are fixed until you rename them. A `variables` mapping on
{class}`~mmmjax.Data` maps names you choose to supplied ones, and `constants`
adds settings that aren't arrays.

:::{admonition} Declare every input
:class: warning

Once you pass `variables`, blocks see only the names it declares, `reference`
and `outcome_scaling` included. `reference` holds only declared arrays too, so
declare every array a block reads through it.
:::

The math stays that of [A first model](first_model), and only the names
change. The mapping below declares every input the brand's blocks request,
renames three of them, and passes the adstock's reach as a constant.

| Symbol | Name in the code | Where the name comes from |
| --- | --- | --- |
| $x_{tc}$, the paid channels' scaled impressions | `impressions` | The mapping's name for the supplied `media` |
| $v_{tc}$, their spending | `cost` | The mapping's name for the supplied `spend` |
| $p_{tj}$, the scaled controls | `confounders` | The mapping's name for the supplied `controls` |
| The adstock's longest lag $L = 8$ | `max_lag` | Passed as a constant |

`confounders` says why demand and holidays are in the model.

The brand's `transformed_data` requests none of the three renamed inputs, so
it runs unchanged.

```{code-cell} ipython3
named_data = mj.Data(
    data,
    scaling=scaling,
    variables={
        "outcome": "outcome",
        "impressions": "media",
        "cost": "spend",
        "organic_media": "organic_media",
        "confounders": "controls",
        "treatments": "treatments",
        "time": "time",
        "day_of_year": "day_of_year",
        "reference": "reference",
        "outcome_scaling": "outcome_scaling",
    },
    constants={"max_lag": 8},
)


def named_hill_adstock(impressions, retention, half_saturation, max_lag):
    carried = mj.geometric_adstock(impressions, alpha=retention, max_lag=max_lag)
    saturated = mj.hill_saturation(carried, half_saturation=half_saturation, slope=1.0)
    return saturated


def named_transformed_parameters(
    impressions,
    organic_media,
    confounders,
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
    max_lag,
):
    trained = named_hill_adstock(reference.impressions, retention, half_saturation, max_lag)
    coefficient = mj.roi_coefficient(roi, trained, reference.cost, outcome_scale=outcome_scaling.scale)
    organic_trained = named_hill_adstock(
        reference.organic_media, organic_retention, organic_half_saturation, max_lag
    )
    total_revenue = outcome_scaling.inverse_transform(reference.outcome).sum()
    organic_contribution = organic_share * total_revenue
    organic_coefficient = mj.contribution_coefficient(
        organic_contribution, organic_trained, outcome_scale=outcome_scaling.scale
    )
    baseline = intercept + growth * trend + curvature * trend**2 + annual @ annual_coefficients
    media_effect = named_hill_adstock(impressions, retention, half_saturation, max_lag) @ coefficient
    organic_saturated = named_hill_adstock(
        organic_media, organic_retention, organic_half_saturation, max_lag
    )
    organic_effect = organic_saturated @ organic_coefficient
    control_effect = confounders @ control_coefficient
    treatment_effect = treatments @ treatment_coefficient
    mu = baseline + media_effect + organic_effect + control_effect + treatment_effect
    return {"mu": mu}


named_model = mj.Model(
    parameters=parameters,
    data=named_data,
    transformed_data=transformed_data,
    transformed_parameters=named_transformed_parameters,
    log_density=log_density,
    generated_quantities=generated_quantities,
)
renamed = named_model.evaluate(point)["mu"]
bool((renamed == fitted).all())
```

The renamed model computes the same expected revenue as the original, because
renaming inputs leaves every line of the generative model from
[A first model](first_model) unchanged. mmmJAX still supplies `impressions`
and `confounders`, so they're supplied names, but you chose their names in
the mapping. A declared `reference` holds its arrays under their
new names, which is why `named_transformed_parameters` reads
`reference.impressions` and `reference.cost`. No block requests `cost` by
name, and the mapping declares it anyway so `reference` will hold it.

The mapping keeps `outcome`, `organic_media`, and `treatments` under their own
names. The brand's density and generated quantities find `outcome` that way,
and `reference.outcome` still holds the training revenue that email's
coefficient sums. Constants don't need declaring. They keep their own names
and pass through unchanged, so `max_lag` stays
a Python integer that JAX can use to set an array's shape.
