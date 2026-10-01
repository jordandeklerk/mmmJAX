---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Budget optimization

A fitted model can tell you how revenue would change if the same money were
spent differently, and {func}`~mmmjax.optimize_budget` uses it to search for
the split of a budget that maximizes expected revenue. For every split it
tries, it runs the blocks again on the posterior draws with only the spending
changed.

The examples again use the ten-channel brand from
[A first model](first_model). Only the ten paid channels have spending to move,
so Email's sends, the price, and the promotions keep their observed values in
every split the optimizer tries.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_limited_plan, first_model_plan, first_model_results, stored

results = first_model_results(model)
```

## The problem

Spending reaches the model through exposure, so when paid channel $c$ gets a
new amount $s_c$, its impressions in every week scale by the same factor. That
way the cost per impression and the timing of the campaigns stay as they were.
With $S_c$ the channel's current spending, the impressions become

$$
z_{tc}(s) = z_{tc}\, \frac{s_c}{S_c}.
$$

Keeping the timing means a plan can only make each channel's existing weeks
bigger or smaller. A channel that airs in flights stays dark between them under
every plan, and trying a different calendar means preparing its weeks yourself,
as [Scenarios](scenarios) shows for next quarter.

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
`media` and `spend`, while `reference` keeps the training weeks. So a block
that sets a coefficient from `spend` rather than `reference.spend` would
redefine `roi` for every split.
[Scenarios](scenarios.md#predictions-and-definitions) shows why a block's
definitions read `reference` rather than the inputs each split changes.
:::

## Moving the current budget

If you leave out `budget`, the total stays at what was spent. Passing
`include_metrics=True` also records each channel's returns under both splits
for the plots below to read. Before you look at those returns,
{func}`~mmmjax.plot_budget_spend` shows where the plan moves the money.

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

Each bar is a channel's optimized spending minus its current spending, so the
green bars gain money and the red ones give it up. Streaming gains the most,
\$184,500, and YouTube comes next with \$65,800. Generic search gives up the
most, \$145,500, and Meta the next most, \$84,800.

### Why the money moves

To see why, plot the marginal returns with {func}`~mmmjax.plot_media_metrics`
and `metric="marginal_roi"`. The left panel holds the reference split, the
spending as it was, and the right panel the optimized one.

```{code-cell} ipython3
mj.plot_media_metrics(plan, metric="marginal_roi")
```

The returns are defined as in [Media effects](media_effects), and each bar is
a posterior mean. In the left panel, one more dollar on Snapchat brings \$2.40
and one more on Streaming \$2.39, against \$1.18 on Generic search, so moving a
dollar from Generic search to Snapchat is worth about \$1.22 of revenue.

The optimizer keeps moving money this way until every marginal return meets at
about \$1.67, as in the right panel. Once they meet, moving a dollar between
channels costs as much revenue in one as it brings in the other, so no further
move adds anything.

{func}`~mmmjax.optimize_budget` searches locally from today's proportions, and
here the start doesn't matter. With every Hill slope at one and the channels'
effects added together, each channel's revenue rises more slowly with every
dollar, so expected revenue has a single peak. S-shaped curves or a utility of
your own can give the search several places to stop, so compare a few starts
through `initial_spend` if you use either.

Without `metric`, the plot draws the average return, and that can point the
other way.

```{code-cell} ipython3
mj.plot_media_metrics(plan)
```

Meta has the fourth-highest ROI, \$3.49, yet the plan cuts it, because its next
dollar brings \$1.40 against TikTok's \$1.93. That's also why TikTok gains
money even though its ROI is only \$3.20. And once Meta is cut, its ROI rises
to \$3.85, because the dollars it keeps are its most productive ones.

:::{admonition} Follow the marginal return
:class: important

Rank channels by what their next dollar brings, as `metric="marginal_roi"`
shows, not by their ROI.
:::

On [Recovering the truth](recovery.md#marginal-returns) you saw these marginal
returns beside the true ones. The truth puts Generic search's next dollar last
too, so it backs that cut, but it ranks Streaming's seventh of the ten and
YouTube's first. Because the optimizer moves money along the model's response
curves, a curve that reads association as cause moves the budget on that basis
too. [Plans against the truth](#plans-against-the-truth) measures what that
costs by checking each plan against the simulation.

### What the plan gains

{func}`~mmmjax.plot_budget_response` draws a waterfall from the revenue the
channels bring under the historical split to what they bring under the
optimized one.

```{code-cell} ipython3
mj.plot_budget_response(plan)
```

Each bar between the two totals adds one channel's change in revenue. The cut
to Generic search costs \$204,000, for instance, and the increase in Streaming
adds \$368,000.

The subtitle puts the mean gain at \$156,000 over the three years, with an
89 percent interval from a loss of \$272,000 to a gain of \$612,000, and
71 percent of the draws favor the new split. So the plan likely helps, but its
interval leaves room for a loss.

## Limits on each channel

Most of the time you can't move money that freely, so `spend_constraint_lower`
and `spend_constraint_upper` keep each channel within a fraction of its current
spending, here 30 percent either way. If you'd rather set the limits in
dollars, pass `bounds` instead.

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

Compared with the free plan, the limits shrink the loss at the low end of the
interval from \$272,000 to \$170,000 and raise the share of draws that gain
from 71 to 77 percent. You pay for that with a smaller mean gain, \$134,000
instead of \$156,000.

The limits cap Streaming's increase at 30 percent, so it now adds \$204,000
rather than \$368,000. Because the plan moves less money, it stays closer to
the spending the data has seen, and that spending is what the model knows best.
This is the plan that [Plotting](plotting) draws, and its response curves
there show Streaming stopping at its upper limit.

## Counting risk

By default the optimizer maximizes the posterior mean, but `utility_function`
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

With one standard deviation subtracted, the optimizer holds back on the
channels the model knows least. Snapchat, whose small budget leaves the model
the least to learn from, now adds \$57,700 of revenue instead of \$80,300, and
Streaming adds \$175,000 instead of \$368,000. Yet the mean gain falls only to
\$123,000, while the loss at the low end of the interval shrinks from \$272,000
to \$105,000 and 81 percent of the draws gain.

The cautious objective stops before the last dollars, the ones that add more
spread than revenue. Compared with the 30 percent limits, it gives up \$11,000
of mean gain for a much smaller loss at the low end, \$105,000 against
\$170,000. The limits hold back every channel alike, while the cautious
objective holds back where the draws disagree.

## Plans against the truth

Because the brand is simulated, you can also find out what each plan would
really earn. {meth}`brand.contributions <mmmjax.SyntheticData.contributions>`
reruns its media effects with each channel's media scaled by the plan's ratio
of new to current spending, the same change the optimizer makes to the model's
media.

```{code-cell} ipython3
import pandas as pd

names = list(channels.values())


def true_change(candidate):
    spend = candidate["spend"].sel(channel=names)
    ratio = spend.sel(allocation="optimized") / spend.sel(allocation="reference")
    scaled = brand.contributions(dict(zip(channels, ratio.values.tolist())))
    change = (scaled - brand.contributions()).sum("time").sel(channel=list(channels))
    labeled = change.assign_coords(channel=names)
    return labeled


plans = {"free": plan, "limited": limited, "cautious": careful}
rows = {
    name: {
        "moved": candidate["spend"].diff("allocation").clip(min=0).sum().item(),
        "expected_gain": candidate["response_change"].mean().item(),
        "true_gain": true_change(candidate).sum().item(),
    }
    for name, candidate in plans.items()
}
gains = pd.DataFrame.from_dict(rows, orient="index")
gains["shortfall"] = gains["expected_gain"] - gains["true_gain"]
gains.round({"moved": -2, "expected_gain": -3, "true_gain": -3, "shortfall": -3})
```

The free plan moves \$342,600 and expects to gain \$156,000, but in the
simulation it would gain only about \$52,000, a third as much. The limited and
cautious plans move less, \$240,700 and \$185,400, and each lands about
\$8,000 short of what it expects.

The table below follows the free plan channel by channel to show why it falls
so far short. Each row gives the money moved, the revenue that move really
changes, and the change per dollar.

```{code-cell} ipython3
:tags: [hide-input]

spend = plan["spend"].sel(channel=names)
moved = (spend.sel(allocation="optimized") - spend.sel(allocation="reference")).to_series()
earned = true_change(plan).to_series()
free = pd.DataFrame({"moved": moved, "true_change": earned})
cut = free["moved"] < 0
totals = pd.DataFrame(
    {
        "moved": [free.loc[~cut, "moved"].sum(), free.loc[cut, "moved"].sum()],
        "true_change": [free.loc[~cut, "true_change"].sum(), free.loc[cut, "true_change"].sum()],
    },
    index=["all added", "all cut"],
)
table = pd.concat([free, totals])
table["per_dollar"] = table["true_change"] / table["moved"]
table.round({"moved": -2, "true_change": -2, "per_dollar": 2})
```

In the two totals at the bottom, each dollar the plan adds earns \$2.09 in
the simulation, and each dollar it cuts had earned \$1.94, so the \$342,600
it moves nets little. Streaming takes the most money yet earns the
least per dollar of the channels that gain, \$1.58, and the cut to Linear TV
costs the most per dollar, \$3.06.

On this one dataset, moving less kept the limited and cautious plans close to
what they expected. So the advice under
[Limits on each channel](#limits-on-each-channel), to stay near the spending
the data has seen, holds up against the truth.

## How big the budget should be

Passing a `budget` lets the optimizer split a new total rather than the current
one. At every size the best split equalizes the marginal returns, and their
common value is what one more dollar of budget would bring.

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
within a cent. Reading across the columns, you can watch the return on the
next dollar fall as the budget grows, from about \$2.27 at \$2 million to
\$0.99 at \$5 million, slightly below break-even.

:::{admonition} Before you grow the budget
:class: warning

A larger budget pays for itself while the return on its next dollar,
multiplied by the profit margin on a dollar of revenue, stays above one.
Budgets far from what was spent also push spending beyond anything in the
data, and out there the response curves rest more on the model's shape and
priors than on evidence.
:::

## Planning a period

`spend_periods` picks the weeks whose spending the plan changes, and
`response_periods` picks the weeks whose revenue counts.

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

For 2023, the plan moves money mostly into Streaming, \$48,600, and YouTube,
\$18,700, and takes it mostly from Generic search and Branded search.

To plan weeks the data doesn't cover, you pass data for those weeks as
`new_data`, and the next page, [Scenarios](scenarios), shows how to prepare
it.
