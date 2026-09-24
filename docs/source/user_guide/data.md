---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data and scaling

Every model starts from prepared, scaled data, and the choices made there
shape what the blocks receive. This page prepares, checks, and scales the
frame from [The example data](example_data) with
{func}`~mmmjax.prepare_data` and {func}`~mmmjax.fit_data_scaling`, the same
two calls [A first model](first_model) makes before it writes its blocks.

```{code-cell} ipython3
:tags: [remove-cell]

import mmmjax as mj

example = mj.simulate_data(seed=7, groups=None, complexity="simple", noise_scale=0.02)
```

## Roles and inputs

{func}`~mmmjax.prepare_data` selects the columns a model uses and gives each
one a role, such as the outcome, media, spend, or controls, with `channels`
naming the media columns. The roles, together with a few inputs computed from
the dates, are the names a block can request.

```{code-cell} ipython3
data = mj.prepare_data(
    example.frame,
    time="week",
    outcome="revenue",
    media=["linear_tv_impressions", "generic_search_impressions"],
    spend=["linear_tv_spend", "generic_search_spend"],
    channels=["TV", "Search"],
    controls=["price"],
)
list(data.model_inputs)
```

`time` and `day_of_year` give each week's position and its place in the
calendar, and `n_periods` counts the modeled weeks. The `media_` versions do
the same for the media's weeks, which match the modeled weeks unless
`media_history` adds earlier ones, as a note in [A first model](first_model)
explains. `outcome_scaling` lets a
block report its output in revenue, as [Sampling and diagnostics](sampling)
shows, and `reference` keeps the training arrays in reach when the model runs
on other data, which [Scenarios](scenarios) covers.

## Arrays

Prepared arrays put time first, then the group when the data has one, then
the channel or column.

```{code-cell} ipython3
data.arrays["media"].shape, data.arrays["outcome"].shape
```

The media array has one row per week and one column per channel, and the
outcome has one value per week, so a block can combine them directly.

## Any data frame

{func}`~mmmjax.prepare_data` reads its frame through
[narwhals](https://narwhals-dev.github.io/narwhals/), so your data can stay in
the library it already lives in. pandas, polars, and PyArrow all work
directly, as does any other eager frame narwhals supports, and nothing is
converted to pandas along the way.

```{code-cell} ipython3
import polars as pl
import pyarrow as pa

frames = {
    "polars": pl.from_pandas(example.frame),
    "pyarrow": pa.Table.from_pandas(example.frame),
}
for library, frame in frames.items():
    prepared = mj.prepare_data(
        frame,
        time="week",
        outcome="revenue",
        media=["linear_tv_impressions", "generic_search_impressions"],
        spend=["linear_tv_spend", "generic_search_spend"],
        channels=["TV", "Search"],
        controls=["price"],
    )
    matches = (prepared.arrays["media"] == data.arrays["media"]).all()
    print(library, bool(matches))
```

Both give the same media array as the pandas frame above. The same goes for
every place mmmJAX takes new data, such as the scenarios in
[Scenarios](scenarios).

## Checking the data

{func}`~mmmjax.check_data` looks for problems that no model can fix, which
makes it worth reading before you write any blocks.

```{code-cell} ipython3
checks = mj.check_data(data)
pairs = checks["pairs"].to_dataset().to_dataframe()
pairs[["feature_a", "feature_b", "correlation"]].round(2)
```

TV and search impressions barely move together, with a correlation of 0.11
across the 156 weeks, and neither follows price, so the data can tell their
effects apart. When two channels do move together, the data says little about
how to split the credit between them, and the priors end up doing more of that
work. The other groups in `checks` report the share of weeks without exposure,
which is just over two thirds for TV, along with spending that has no matching
exposure and variance inflation factors for the predictors.

## Scaling

{func}`~mmmjax.fit_data_scaling` fits a transform for each role it scales, and
{class}`~mmmjax.Data` applies them before any block runs.

```{code-cell} ipython3
scaling = mj.fit_data_scaling(data, scale_outcome=True)
media_transform = scaling.transformations["media"]
outcome_transform = scaling.transformations["outcome"]
media_transform.scale, outcome_transform.offset, outcome_transform.scale
```

Each channel's exposure is divided by its median $m_c$ over the weeks with any
exposure, so the model sees $x_{tc} = z_{tc} / m_c$ for raw impressions
$z_{tc}$. A typical week of TV becomes one, and a week without TV stays at
zero. Revenue becomes $y_t = (r_t - \bar{r}) / s_r$ with its mean $\bar{r}$ and
standard deviation $s_r$, and price is standardized the same way.

Revenue is standardized only because the call asks for `scale_outcome=True`.
By default the outcome keeps its own units, which is what a count likelihood
needs. Spend is never scaled, because the analysis functions report returns
per dollar.

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
)
regional_data.arrays["media"].shape, regional_data.arrays["outcome"].shape
```

Every region has to cover the same weeks. A model can then give each region
its own values by declaring a parameter with `dims=("group", "channel")`.
[A first model](first_model) declares its parameters the same way, with
`dims="channel"` for one value per channel.
