---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Budget optimization

A fitted model can say how revenue would change if the same money were spent
differently. {func}`~mmmjax.optimize_budget` searches for the split of a
budget that maximizes expected revenue, and every split it tries runs the
blocks again on the posterior draws with only the spending changed. The
examples use the model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)
```

## The problem

Spending reaches the model through exposure. A new amount $s_c$ for channel
$c$ scales every week's impressions by the same factor, which keeps the cost
per impression and the timing of the campaigns, so with $S_c$ the channel's
current spending the impressions become

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

## Moving the current budget

Without a `budget`, the total stays at what was spent.

```{code-cell} ipython3
plan = mj.optimize_budget(model, results, quantity="mu", include_metrics=True)
plan["spend"].to_pandas().round(-2)
```

The plan keeps the \$792,000 spent over the three years and moves about
\$84,000 from search to TV. `include_metrics=True` adds each channel's returns
at both splits, defined as in [Media effects](media_effects), and they show
why.

```{code-cell} ipython3
returns = plan[["roi", "marginal_roi"]].median(("chain", "draw"))
returns.to_dataframe()[["roi", "marginal_roi"]].round(2)
```

At the current split one more dollar on TV brings \$2.52 and one more on
search \$1.79, so moving a dollar from search to TV adds about 73 cents of
revenue. In the simulation the two are \$2.71 and \$2.20, as [Recovering the
truth](recovery) shows, so the move points the right way with a smaller gain
for each dollar. The optimizer keeps moving money until the two marginal returns meet
at about \$2.22, where no further move adds anything. The average return
points the other way. Search has the higher ROI, and it rises to \$4.52 after
the cut because the dollars search keeps are its most productive ones. A
budget should follow marginal returns, not average ones.

```{code-cell} ipython3
change = plan["response_change"]
print(change.quantile([0.05, 0.5, 0.95]).round(-3).values)
print(float((change > 0).mean()))
```

`response_change` compares the two splits draw by draw. The median gain is
about \$31,000 over the three years, the 90 percent interval runs from a loss
of \$14,000 to a gain of \$75,000, and 87 percent of the draws favor the new
split.

## Limits on each channel

A plan rarely gets to move money freely. `spend_constraint_lower` and
`spend_constraint_upper` keep each channel within a fraction of its current
spending, here 10 percent either way.

```{code-cell} ipython3
limited = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_constraint_lower=0.1,
    spend_constraint_upper=0.1,
)
limited["spend"].to_pandas().round(-2)
```

TV stops at its cap of \$403,200, and search keeps the rest.

```{code-cell} ipython3
print(limited["response_change"].quantile([0.05, 0.5, 0.95]).round(-3).values)
```

The median gain falls to about \$21,000, but even the 5 percent quantile is
now a gain. A smaller move keeps the plan close to the spending the data has
seen, where the model knows the most. `bounds` sets the limits in dollars
instead.

## Counting risk

The optimizer maximizes the posterior mean by default. `utility_function`
takes any differentiable JAX function of the total revenue in each draw, so
an objective can count the spread as well.

```{code-cell} ipython3
import jax.numpy as jnp


def cautious(revenue):
    return jnp.mean(revenue) - jnp.std(revenue)


careful = mj.optimize_budget(model, results, quantity="mu", utility_function=cautious)
careful["spend"].to_pandas().round(-2)
```

Subtracting one standard deviation moves about \$69,000 instead of \$84,000.

```{code-cell} ipython3
print(careful["response_change"].quantile([0.05, 0.5, 0.95]).round(-3).values)
```

The median gain barely changes, at about \$30,000, while the 5 percent
quantile improves from a loss of \$14,000 to a loss of \$6,000. Moving money
to TV takes its flights past the largest ones in the data, where the draws
disagree more, so the cautious objective stops before the last dollars that
add more spread than revenue.

## How big the budget should be

A `budget` sets a new total. At every size the best split equalizes the
marginal returns, and their common value is what one more dollar of budget
would bring.

```{code-cell} ipython3
import pandas as pd

marginal = {}
for budget in [600_000, 800_000, 1_000_000, 1_200_000]:
    sized = mj.optimize_budget(model, results, quantity="mu", budget=budget, include_metrics=True)
    best = sized["marginal_roi"].sel(allocation="optimized")
    marginal[budget] = best.median(("chain", "draw")).to_series()
pd.DataFrame(marginal).T.round(2)
```

The return on the next dollar falls from about \$2.70 at \$600,000 to \$1.60 at
\$1.2 million. A larger budget pays for itself while that return, multiplied
by the profit margin on a dollar of revenue, stays above one. Budgets far from
the current one also push spending beyond anything in the data, where the
response curves rest more on the model's shape and priors than on evidence.

## Planning a period

Budgets are usually set for a period. `spend_periods` picks the weeks whose
spending the plan changes, and `response_periods` the weeks whose revenue
counts. Carryover spreads each week's exposure over the eight weeks after it,
so the revenue window here runs eight weeks past the end of 2023.

```{code-cell} ipython3
weeks = pd.to_datetime(example.frame["week"])
spend_weeks = example.frame["week"][weeks.dt.year == 2023].tolist()
response_weeks = example.frame["week"][weeks.between("2023-01-01", "2024-02-19")].tolist()

yearly = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_periods=spend_weeks,
    response_periods=response_weeks,
)
yearly["spend"].to_pandas().round(-2)
```

The plan moves about \$18,000 of 2023's \$274,000 from search to TV. Ending the
revenue window with the year would miss what TV's last flights carry into
2024 and undervalue TV. A plan for weeks the data doesn't cover passes a
frame of those weeks as `new_data`, which the next page,
[Scenarios](scenarios), shows how to prepare.

Every plan on this page is what the model implies under its assumptions. The
optimizer moves money on the model's response curves, so a curve that reads
association as cause moves the budget on that basis too.
