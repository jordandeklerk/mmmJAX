---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Budget optimization

A fitted model can say how revenue would change if the same money were spent
differently. {func}`~mmmjax.optimize_budget` searches for the split of a
budget that maximizes expected revenue. For every split it tries, it runs the
blocks again on the posterior draws with only the spending changed. The
examples use the ten-channel brand from [A first model](first_model). Only the
ten paid channels have spending to move, so Email's sends, the price, and the
promotions keep their observed values in every split the optimizer tries.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_limited_plan, first_model_plan, first_model_results, stored

results = first_model_results(model)
```

## The problem

Spending reaches the model through exposure. A new amount $s_c$ for paid
channel $c$ scales every week's impressions by the same factor, so the cost per
impression and the timing of the campaigns stay as they were. With $S_c$ the
channel's current spending, the impressions become

$$
z_{tc}(s) = z_{tc}\, \frac{s_c}{S_c}.
$$

Each posterior draw $k$ then gives total expected revenue $R_k(s)$ over the
period, and the optimizer looks for

$$
\max_{s} \; \frac{1}{K} \sum_{k=1}^{K} R_k(s)
\quad \text{subject to} \quad
\sum_c s_c = B, \qquad l_c \le s_c \le u_c,
$$

where $B$ is the budget and the limits $l_c$ and $u_c$ default to zero and the
whole budget.

:::{admonition} Each split arrives as new media and spend
:class: note

The optimizer hands every split to your blocks as new values of the supplied
`media` and `spend`, while `reference` keeps the training weeks. A coefficient
set from `spend` rather than `reference.spend` would redefine `roi` for every
split, which is why [Scenarios](scenarios) keeps definitions on `reference`.
:::

## Moving the current budget

Without a `budget`, the total stays at what was spent. `include_metrics=True`
also records each channel's returns under both splits for the plots below to
read. {func}`~mmmjax.plot_budget_spend` shows where the plan moves the money.

```{code-cell} ipython3
:tags: [skip-execution]

plan = mj.optimize_budget(model, results, quantity="mu", include_metrics=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

plan = first_model_plan(model, results)
```

```{code-cell} ipython3
mj.plot_budget_spend(plan)
```

Each bar is a channel's optimized spending minus its current spending.
Streaming gains the most, \$184,000, and YouTube next, \$65,800. Generic
search gives up the most, \$145,000, then Meta, \$84,800, and Display, Branded
search, and Linear TV give up smaller amounts.

{func}`~mmmjax.plot_media_metrics` with `metric="marginal_roi"` shows why. The
left panel holds the reference split, the spending as it was, and the right
panel the optimized one.

```{code-cell} ipython3
mj.plot_media_metrics(plan, metric="marginal_roi")
```

The returns are defined as in [Media effects](media_effects), and each bar is
a posterior mean, since the optimizer maximizes mean revenue. At the current
split, one more dollar on Snapchat brings \$2.40 and one more on Streaming
\$2.39, against \$1.18 on Generic search. Moving a dollar from Generic search
to Snapchat adds about \$1.22 of revenue. The optimizer keeps moving money
until every marginal return meets at about \$1.67, where no further move adds
anything.

The average return, the plot's default metric, can point the other way.

```{code-cell} ipython3
mj.plot_media_metrics(plan)
```

Meta has the fourth-highest ROI, \$3.49, yet the plan cuts it, because its next
dollar brings \$1.40 against TikTok's \$1.93. TikTok gains money even though
its ROI is only \$3.20. Meta's ROI then rises to \$3.85, because the dollars it
keeps are its most productive ones.

:::{admonition} Follow the marginal return
:class: important

Rank channels by what their next dollar brings, as `metric="marginal_roi"`
shows, not by their ROI.
:::

[Recovering the truth](recovery) checks these marginal returns against the
simulation. The truth backs the cut to Generic search but not the move into
Streaming.

`response_change` compares the two splits draw by draw, and
{func}`~mmmjax.plot_budget_response` sums it up above a waterfall from the
revenue the channels bring under the historical split to what they bring under
the optimized one.

```{code-cell} ipython3
mj.plot_budget_response(plan)
```

Each bar between the two totals adds one channel's change in revenue, so the
cut to Generic search costs \$204,000 and the increase in Streaming adds
\$368,000. The subtitle puts the mean gain at \$156,000 over the three years,
with an 89 percent interval from a loss of \$272,000 to a gain of \$612,000,
and 71 percent of the draws favor the new split.

## Limits on each channel

You rarely get to move money freely. `spend_constraint_lower` and
`spend_constraint_upper` keep each channel within a fraction of its current
spending, here 30 percent either way.

```{code-cell} ipython3
:tags: [skip-execution]

limited = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_constraint_lower=0.3,
    spend_constraint_upper=0.3,
    include_metrics=True,
)
```

```{code-cell} ipython3
:tags: [remove-cell]

limited = first_model_limited_plan(model, results)
```

```{code-cell} ipython3
mj.plot_budget_response(limited)
```

The limits shrink the loss at the low end of the interval from \$272,000 to
\$170,000 and raise the share of draws that gain from 71 to 77 percent. The
cost is a smaller mean gain, \$134,000 instead of \$156,000. Streaming now adds
\$204,000 rather than \$368,000. A smaller move keeps the plan close to the
spending the data has seen, where the model knows the most. This is the plan
that [Plotting](plotting) draws, and its response curves there show Streaming
stopping at its upper limit. `bounds` sets the limits in dollars instead.

## Counting risk

The optimizer maximizes the posterior mean by default. `utility_function`
takes any differentiable JAX function of the total revenue in each draw, so
your objective can count the spread as well.

```{code-cell} ipython3
import jax.numpy as jnp


def cautious(revenue):
    return jnp.mean(revenue) - jnp.std(revenue)
```

```{code-cell} ipython3
:tags: [skip-execution]

careful = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    utility_function=cautious,
    include_metrics=True,
)
```

```{code-cell} ipython3
:tags: [remove-cell]

careful = stored(
    "first_model_cautious_plan",
    lambda: mj.optimize_budget(model, results, quantity="mu", utility_function=cautious, include_metrics=True),
)
```

```{code-cell} ipython3
mj.plot_budget_response(careful)
```

Subtracting one standard deviation holds back on the channels the model knows
least. Snapchat, whose small budget leaves the model the least to learn from,
now adds \$57,700 of revenue instead of \$80,300, and Streaming adds \$175,000
instead of \$368,000. The mean gain falls only to \$123,000, while the loss at
the low end of the interval shrinks from \$272,000 to \$105,000 and 81 percent
of the draws gain. Each extra dollar on Snapchat pushes its biggest weeks
further past anything in the data, where the draws disagree more. The cautious
objective stops before the last dollars, the ones that add more spread than
revenue. Against the 30 percent limits it gives up \$11,000 of mean gain for a
much smaller loss at the low end, \$105,000 against \$170,000, because it holds
back where the draws disagree instead of on every channel alike.

## How big the budget should be

A `budget` sets a new total. At every size the best split equalizes the
marginal returns, and their common value is what one more dollar of budget
would bring.

```{code-cell} ipython3
:tags: [skip-execution]

sized = {
    budget: mj.optimize_budget(model, results, quantity="mu", budget=budget, include_metrics=True)
    for budget in [2_000_000, 3_000_000, 4_000_000, 5_000_000]
}
```

```{code-cell} ipython3
:tags: [remove-cell]

sized = {
    budget: stored(
        f"first_model_plan_{budget}",
        lambda: mj.optimize_budget(model, results, quantity="mu", budget=budget, include_metrics=True),
        variables=["marginal_roi"],
    )
    for budget in [2_000_000, 3_000_000, 4_000_000, 5_000_000]
}
```

```{code-cell} ipython3
import pandas as pd

marginal = {
    budget: plan["marginal_roi"].sel(allocation="optimized").mean(("chain", "draw")).to_series()
    for budget, plan in sized.items()
}
pd.DataFrame(marginal).round(2)
```

Each column holds the ten marginal returns at one budget, and they agree to
within a cent. The return on the next dollar falls from about \$2.27 at \$2
million to \$0.99 at \$5 million, just below break-even.

:::{admonition} Before you grow the budget
:class: warning

A larger budget pays for itself while the return on its next dollar,
multiplied by the profit margin on a dollar of revenue, stays above one.
Budgets far from what was spent also push spending beyond anything in the
data, where the response curves rest more on the model's shape and priors than
on evidence.
:::

## Planning a period

Budgets are usually set for a period. `spend_periods` picks the weeks whose
spending the plan changes, and `response_periods` picks the weeks whose
revenue counts.

:::{admonition} Let the revenue window run past the spending
:class: tip

Carryover spreads each week's exposure over the eight weeks after it, so the
revenue window here runs eight weeks past the end of 2023. Ending it with the
year would miss what the last weeks of 2023 carry into 2024.
:::

```{code-cell} ipython3
weeks = pd.to_datetime(brand.frame["week"])
spend_weeks = weeks[weeks.dt.year == 2023]
response_weeks = weeks[weeks.between("2023-01-01", "2024-02-19")]
```

```{code-cell} ipython3
:tags: [skip-execution]

yearly = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_periods=spend_weeks,
    response_periods=response_weeks,
)
```

```{code-cell} ipython3
:tags: [remove-cell]

yearly = stored(
    "first_model_yearly_plan",
    lambda: mj.optimize_budget(
        model,
        results,
        quantity="mu",
        spend_periods=spend_weeks,
        response_periods=response_weeks,
    ),
)
```

```{code-cell} ipython3
mj.plot_budget_spend(yearly)
```

The plan moves 2023's money mostly into Streaming, \$48,600, and YouTube,
\$18,700, and takes it mostly from Generic search and Branded search. To plan
weeks the data doesn't cover, you pass data for those weeks as `new_data`. The
next page, [Scenarios](scenarios), shows how to prepare it.

The optimizer moves money along the model's response curves, so if a curve
reads association as cause, the budget moves on that basis too.
