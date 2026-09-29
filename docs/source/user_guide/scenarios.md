---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Scenarios

A scenario asks what the fitted model expects from inputs it hasn't seen, such
as a plan for next quarter. The analysis functions take those inputs as
`new_data` and run your blocks on them with every posterior draw, so each
answer carries the fit's uncertainty. This page prepares next quarter's data
for the ten-channel brand from [A first model](first_model), forecasts its
revenue, and compares two ways to spend its budget.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_results

results = first_model_results(model)
```

## Next quarter's data

The data ends with the week of December 23, 2024. A plan for the thirteen
weeks after it needs every input the blocks read, and only the media and the
spending are yours to choose. The rest are assumptions, and the simplest is
that the quarter repeats the same weeks a year earlier. The cell below copies
the first quarter of 2024 a year forward, from the media and email sends to
demand, holidays, prices, and promotions. The new weeks have no revenue yet,
and none of the analyses below needs it.

```{code-cell} ipython3
import pandas as pd

weeks = pd.to_datetime(brand.frame["week"])
last_year = brand.frame[weeks.between("2024-01-01", "2024-03-25")]
next_quarter = last_year.drop(columns="revenue").assign(
    week=(pd.to_datetime(last_year["week"]) + pd.Timedelta(weeks=52)).dt.date
)
quarter = next_quarter["week"].tolist()
str(quarter[0]), str(quarter[-1])
```

Carryover reaches eight weeks back, so the weeks before the quarter still add
revenue inside it. The plan starts with the last eight weeks of the data, and
every analysis below counts only the quarter's own weeks.

```{code-cell} ipython3
plan = pd.concat([brand.frame.iloc[-8:].drop(columns="revenue"), next_quarter], ignore_index=True)
len(plan)
```

:::{admonition} Keep the weeks before your data
:class: warning

Data you prepare starts with no exposure before its first week. Prepend the
eight weeks before a forecast, or before a plan that never ran, since the
brand's adstock reaches that far back, and count only the weeks after them.
:::

## A forecast

{func}`~mmmjax.contributions` with `new_data=plan` runs the blocks on the plan
with every draw. Its `reference_response` is the expected revenue in each
week, `response_periods` counts only the quarter, and `periods` limits the
changes it makes for each channel's contribution to the quarter as well.

```{code-cell} ipython3
forecast = mj.contributions(
    model, results, quantity="mu", new_data=plan, periods=quarter, response_periods=quarter, by="time"
)
total = forecast["reference_response"].sum("time")
(total.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")) / 1e6).round(2).values
```

The model expects \$4.01 million of revenue in the quarter, with a 90 percent
interval from \$3.83 million to \$4.18 million.

```{code-cell} ipython3
:tags: [hide-input]

import plotnine as pn

band = forecast["reference_response"].quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
band = band.to_series().unstack("quantile").set_axis(["lower", "median", "upper"], axis=1).reset_index()
earlier = pd.DataFrame({"time": band["time"], "revenue": last_year["revenue"].to_numpy()})
(
    pn.ggplot(band, pn.aes("time"))
    + pn.geom_ribbon(pn.aes(ymin="lower", ymax="upper"), fill="#2a2eec", alpha=0.2)
    + pn.geom_line(pn.aes(y="median"), color="#2a2eec", size=1)
    + pn.geom_line(pn.aes(y="revenue"), data=earlier, linetype="dashed")
    + pn.scale_x_datetime(date_labels="%b %-d, %Y")
    + pn.scale_y_continuous(labels=lambda values: [f"${value / 1e3:,.0f}K" for value in values])
    + pn.labs(x="", y="Weekly revenue")
    + mj.theme_mmmjax()
)
```

The blue line and band are the forecast's median and 90 percent interval, and
the dashed line is the revenue the same weeks brought a year earlier. The
forecast follows last year's weekly shape, flights and all, but sits well
below it. The inputs are last year's, so the gap comes mostly from the one
input that moved on, the trend.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

posterior = results["posterior"]
revenue_scale = scaling.transformations["outcome"].scale.item()
years_2024 = (pd.to_datetime(last_year["week"]) - weeks.iloc[0]).dt.days.to_numpy() / 365.25
years_2025 = years_2024 + 364 / 365.25


def trend_revenue(years):
    growth = posterior["growth"].to_numpy()[..., None] * years
    curvature = posterior["curvature"].to_numpy()[..., None] * years**2
    return (growth + curvature).sum(-1) * revenue_scale


round(float(np.median(trend_revenue(years_2025) - trend_revenue(years_2024))), -3)
```

A year further along, the quadratic trend takes about \$736,000 off the
quarter, since its curvature bends it down past the last training week. A
forecast's level rests on how its trend extends past the data, which no week
in the data checks. Comparing plans within the same forecast avoids most of
that, since both plans share the trend.

## Two plans

The first plan repeats last year's split. The second moves a fifth of Generic
search's spending in the quarter into Streaming, the kind of move
[Budget optimization](budgets) found worth making. Impressions follow spending
at last year's cost per impression, so each channel's impressions scale with
its spending.

```{code-cell} ipython3
shifted = plan.copy()
inside = shifted["week"].isin(quarter)
moved = 0.2 * shifted.loc[inside, "generic_search_spend"].sum()
streaming = 1 + moved / shifted.loc[inside, "streaming_spend"].sum()
for name, factor in [("generic_search", 0.8), ("streaming", streaming)]:
    shifted.loc[inside, [f"{name}_spend", f"{name}_impressions"]] *= factor
round(moved)
```

The move shifts \$10,618 between the two channels and leaves the total
spending unchanged.

```{code-cell} ipython3
alternative = mj.contributions(
    model, results, quantity="mu", new_data=shifted, periods=quarter, response_periods=quarter, by="time"
)
gain = alternative["reference_response"].sum("time") - total
round(float(gain.mean())), round(float((gain > 0).mean()), 2)
```

Both plans run on the same draws, so the gain is measured draw by draw and the
trend's uncertainty cancels. The shift adds \$9,957 of expected revenue, about
94 cents for each dollar moved, and 79 percent of the draws favor it.

```{code-cell} ipython3
:tags: [hide-input]

draws = pd.DataFrame({"gain": gain.to_numpy().ravel()})
(
    pn.ggplot(draws, pn.aes("gain"))
    + pn.geom_histogram(bins=60, fill="#d2d3fb", color="#2a2eec", size=0.3)
    + pn.geom_vline(xintercept=0, linetype="dashed", color="#8c8c8c")
    + pn.scale_x_continuous(
        labels=lambda values: [f"{'−' if value < 0 else ''}${abs(value) / 1e3:,.0f}K" for value in values]
    )
    + pn.labs(x="Change in the quarter's expected revenue from the shift", y="Draws")
    + mj.theme_mmmjax()
)
```

Most draws sit to the right of zero, but the tail to the left shows the shift
losing money in about one draw in five, so the case for it is likely rather
than certain. {func}`~mmmjax.optimize_budget` takes the same `plan` as
`new_data`, with `spend_periods` and `response_periods` set to the quarter, to
search for the best split instead of testing one.

## Predictions and definitions

The plans changed the media and the spending and nothing else. Each draw's
coefficients stayed where the fit put them, because the brand's blocks compute
them from the training data in `reference`, and the trend kept counting days
from the first training week.

:::{admonition} Scenario rule
:class: important

A prediction reads the inputs it is given, so it follows the scenario. A
definition reads the training arrays from `reference`, which holds them
unchanged in every scenario, so it stays put when the data changes.
:::

A block that computed its coefficients from `media` and `spend` would
redefine every channel's return for each plan instead, and nothing would raise
an error. [A first model](first_model) marks which of its lines are
definitions, and any normalization or centering you write yourself needs the
same care.
