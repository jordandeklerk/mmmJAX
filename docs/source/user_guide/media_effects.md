---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Media effects

mmmJAX's analysis functions answer the usual marketing mix questions from a
fitted model, and they all work the same way. Each one takes the model, the
draws, and the name of the expected outcome. It runs the blocks again on a
changed copy of the data with the same draws and reports the difference in
revenue. The examples use the ten-channel brand from
[A first model](first_model), with its email newsletter as organic media and
its price and promotions as treatments.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_curves, first_model_results

results = first_model_results(model)
```

## Contributions

{func}`~mmmjax.contributions` takes one input away at a time and measures the
revenue that goes with it. `quantity="mu"` names the expected revenue that
`transformed_parameters` returns, and that's the quantity every analysis
function compares.

How an input goes away depends on its role.

- A paid channel or Email loses its exposure in every week.
- A treatment can't go away, since every week has some price, so it moves to a
  baseline level instead. By default that's the lowest level in the data.
- A control keeps its observed values in every comparison. Controls are in the
  model so what demand and the holidays did doesn't get credited to the inputs
  you asked about, and their effect stays in the baseline.

Each change reaches your blocks through a supplied input such as `media`,
while a coefficient set from `reference` stays put, as [Scenarios](scenarios)
explains.

If $m_t$ is expected revenue in dollars and $m_t^{(-c)}$ is the same week with
input $c$ taken away, each draw gives

$$
\Delta_c = \sum_t \big(m_t - m_t^{(-c)}\big),
\qquad
\text{share}_c = \frac{\Delta_c}{\sum_t m_t}.
$$

```{code-cell} ipython3
shares = mj.contributions(model, results, quantity="mu")
shares
```

Like the other analysis functions, it returns an xarray Dataset with a value
for every draw and the settings behind the result in its attributes. Its
variables hold the following.

- `incremental_response` holds each input's $\Delta_c$ in dollars.
- `contribution_share` holds each input's share of revenue.
- `baseline_response` holds the revenue left with every input taken away.
- `exposure` and `effectiveness` hold each channel's total exposure and the
  revenue per unit of it, such as Email's sends and its revenue per send. Both
  are missing for the treatments.
- `channel_type` holds each input's role.
- `treatment_baseline` holds the level each treatment moved to.

{func}`~mmmjax.plot_contributions` lays the baseline and the inputs end to
end.

```{code-cell} ipython3
mj.plot_contributions(shares)
```

Each label gives a share of the three years' revenue and its total in
dollars, both from posterior means. The baseline holds 79.8 percent, about
\$43.5 million, and that includes what demand and the holidays added. Meta
leads the rest at 3.5 percent, about \$1.93 million, ahead of the promotions at
3.0 percent, about \$1.61 million. Email adds 1.6 percent, about \$879,000,
with no spending behind it, and Snapchat comes last of the channels at 0.5
percent, about \$259,000. The price's bar runs backward, -2.5 percent or a
loss of about \$1.36 million, since it compares the prices the brand charged
with the lowest one and the higher prices cost revenue on average.

Those means hide how uncertain each input's revenue is. The 5th and 95th
percentiles of the draws bound a 90 percent interval.

```{code-cell} ipython3
incremental = shares["incremental_response"].quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
incremental.T.to_pandas().round(-3)
```

For every paid channel the 95th percentile is at least four times the 5th.
YouTube's interval runs from \$293,000 to \$1.25 million, and Snapchat's from
\$76,000 to \$588,000, a factor of nearly eight. Email's runs from \$175,000
to \$1.87 million. The price's interval runs from a loss of \$6.28 million
to a gain of \$3.39 million, so the model can't say whether the brand's
higher prices gained revenue or lost it. The promotions' interval crosses zero
too, from a loss of \$721,000 to a gain of \$3.86 million.

### Treatment baselines

A treatment's contribution depends on the level it is measured from, so the
baseline is part of the question you ask. `treatment_baselines` takes `"min"`,
`"max"`, or a level for each treatment, and a treatment it leaves out keeps
the lowest level.

```{code-cell} ipython3
highest = mj.contributions(
    model, results, quantity="mu", channels=["price"], treatment_baselines={"price": "max"}
)
price = highest["incremental_response"].sel(channel="price")
highest["treatment_baseline"].round(2).item(), price.quantile([0.05, 0.5, 0.95]).round(-3).values
```

Measured from the highest price, \$22.42, the price's contribution is what
the brand gained or lost by charging less. The median turns from a loss of
\$1.36 million to a gain of \$1.15 million, because in the median draw a lower
price brings in more revenue. Both intervals run across zero. A baseline
changes the question the contribution answers, and it can't make the data
say more about the price. [Recovering the truth](recovery) shows why the data
says so little.

### All together

`joint_incremental_response` takes every input away at once, the treatments
included. The model adds the channels' effects, so the paid channels' joint
effect is the sum of their increments within each draw. The simulation
recorded what they added in `brand.truth`.

```{code-cell} ipython3
paid = shares["incremental_response"].sel(channel=list(channels.values())).sum("channel")
true_paid = brand.truth["contribution"].sel(channel=list(channels)).sum()
paid.quantile([0.05, 0.95]).round(-3).values, round(float(true_paid), -4)
```

In 90 percent of the draws the ten channels together brought in between \$7.59
million and \$12.67 million, an interval that holds the true \$12.01 million.
Its upper end is only 1.7 times its lower one, against at least four times for
each channel alone.

:::{admonition} The total is firmer than the split
:class: important

Several channels run on one shared campaign calendar, as
[Data and scaling](data.md) shows, so the data pins down what paid media adds
as a whole far better than how that revenue splits among the channels.
:::

## Returns on spending

{func}`~mmmjax.media_metrics` divides each paid channel's incremental revenue
by the money spent on it, so $\text{ROI}_c = \Delta_c / S_c$ for total spend
$S_c$. Email and the treatments have no spending, so they have no return and
the function leaves them out. The marginal return asks what one more percent
of spending would bring,

$$
\text{mROI}_c = \frac{M_c(1.01\, S_c) - M_c(S_c)}{0.01\, S_c},
$$

where $M_c(S)$ is total expected revenue when the channel's spending is $S$
and every other input stays fixed. {func}`~mmmjax.plot_media_metrics` draws
the returns.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

Each bar is a channel's mean return on a dollar with its 89 percent interval,
and the dashed line marks break-even. Channels run from the most spending to
the least. Each dollar spent over the period returned from \$2.79 on Linear TV
to \$4.06 on YouTube. A return divides $\Delta_c$ by a fixed spend, so its
interval inherits the spread of the contributions above, and Snapchat's is
again the widest.

The marginal return can rank the channels differently.
{func}`~mmmjax.plot_roi_bubbles` sets it against the average return, with
each bubble's area proportional to the channel's spending.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

Meta returns \$3.49 on average against TikTok's \$3.20, yet its bubble sits
below TikTok's, so Meta earns less on the next dollar. The four channels that
spend in every week, Meta, Generic search, Branded search, and Display, sit
lowest of all. They're already further along their curves, where each dollar
adds less. That gap between the average and the marginal return
is what a budget decision turns on.

## Response curves

{func}`~mmmjax.response_curves` scales each paid channel's spending up and
down and records the revenue that comes with it. By default it goes from zero
to twice the current spending in 21 steps. You can pass
`multipliers` for other steps, and `curves["spend"]` holds the spending behind
each one. Spending converts to exposure at the observed cost per impression.

```{code-cell} ipython3
:tags: [skip-execution]

curves = mj.response_curves(model, results, quantity="mu")
```

```{code-cell} ipython3
:tags: [remove-cell]

curves = first_model_curves(model, results)
```

```{code-cell} ipython3
mj.plot_response_curves(curves)
```

{func}`~mmmjax.plot_response_curves` gives each channel a panel with its own
axes. The line is the mean revenue and the band its 89 percent interval, and
the point marks the current spending. Past the point the line turns dashed,
since the data never saw that much spending. Every curve bends. The dashed
stretch doubles a channel's spending but adds less revenue than the solid
stretch before it. That bend is why each marginal return above falls short of
its average return.

[Budget optimization](budgets) uses the same curves to split a budget between
the paid channels, and [Plotting](plotting) shows more ways to draw
contributions, returns, and curves. [Scenarios](scenarios) shows how the data
gets changed and what that asks of the blocks.

Every number on this page is what the model implies under its assumptions, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-analyses-get-their-answers)
explains. [Recovering the truth](recovery) checks each channel, Email, and the
treatments against the simulation.
