---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Media effects

mmmJAX's analysis functions answer the usual marketing mix questions from a
fitted model, such as what each channel brought in and what it returned.
Whichever one you call, it changes a copy of the data, runs the blocks again on
it with the same draws, and reports the difference in revenue.

The examples use the ten-channel brand from [A first model](first_model), whose
email newsletter enters the model as organic media and whose price and
promotions enter as treatments.

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_curves, first_model_prior_results, first_model_results

results = first_model_results(model)
prior_results = first_model_prior_results(model, priors)
```

## Contributions

{func}`~mmmjax.contributions` takes one input away at a time and measures how
much revenue goes with it. `quantity="mu"` points it at the expected revenue
that `transformed_parameters` returns, and that's the quantity every analysis
function compares.

What taking an input away means depends on its role, as the three cases below
show.

- A paid channel or Email loses its exposure in every week.
- A treatment can't go away, since every week has some price, so it moves to a
  baseline level instead. By default that's the lowest level in the data.
- A control keeps its observed values in every comparison and its effect stays
  in the baseline, since its coefficient has no causal reading, as
  [Data and scaling](data.md#controls) explains.

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
for every draw and the settings behind the result in its attributes. If you
look through its variables in the output above, you'll find the following.

- `incremental_response` holds each input's $\Delta_c$ in dollars.
- `contribution_share` holds each input's share of revenue.
- `baseline_response` holds the revenue left with every input taken away.
- `exposure` and `effectiveness` hold each channel's total exposure and the
  revenue per unit of it, such as Email's sends and its revenue per send.
- `channel_type` holds each input's role.
- `treatment_baseline` holds the level each treatment moved to.

{func}`~mmmjax.plot_contributions` lays the baseline and the inputs end to
end, so you can compare their shares of revenue.

```{code-cell} ipython3
mj.plot_contributions(shares)
```

Each label gives an input's share of the three years' revenue and its total in
dollars, both from posterior means. Most of the revenue, 79.8 percent or about
\$43.5 million, sits in the baseline, and that includes what demand and the
holidays added.

Among the inputs, Meta leads at 3.5 percent, about \$1.93 million, and the
promotions come next at 3.0 percent, about \$1.61 million. Email adds
1.6 percent, about \$879,000, even though nothing is spent on it, and Snapchat
comes last of the channels at 0.5 percent, about \$259,000.

The price's bar works differently, because it compares the prices the brand
charged with the lowest one. On average those higher prices cost revenue, so
the bar runs backward to -2.5 percent, a loss of about \$1.36 million.

### Uncertainty

The plot's means hide how uncertain each input's revenue is, but the draws
still carry that uncertainty, and their 5th and 95th percentiles bound a
90 percent interval.

```{code-cell} ipython3
incremental = shares["incremental_response"].quantile([0.05, 0.5, 0.95], dim=("chain", "draw"))
incremental.T.to_pandas().round(-3)
```

In every paid channel's row, the 95th percentile is at least four times the
5th. YouTube's interval runs from \$293,000 to \$1.25 million, and Snapchat's
from \$76,000 to \$588,000, a factor of nearly eight. By the same measure
Email's interval is wide too, from \$175,000 to \$1.87 million.

The price's interval runs from a loss of \$6.28 million to a gain of
\$3.39 million, so the model can't tell you whether the brand's higher prices
gained revenue or lost it. The promotions' interval crosses zero as well, from
a loss of \$721,000 to a gain of \$3.86 million.

### Treatment baselines

A treatment's contribution depends on the level you measure it from, so
picking the baseline is part of the question you ask. `treatment_baselines`
takes `"min"`, `"max"`, or a level for each treatment, and any treatment you
leave out keeps the lowest level.

```{code-cell} ipython3
highest = mj.contributions(
    model, results, quantity="mu", channels=["price"], treatment_baselines={"price": "max"}
)
price = highest["incremental_response"].sel(channel="price")
highest["treatment_baseline"].round(2).item(), price.quantile([0.05, 0.5, 0.95]).round(-3).values
```

Measured from the highest price, \$22.42, the price's contribution becomes
what the brand gained or lost by charging less. The median flips from a loss
of \$1.36 million to a gain of \$1.15 million, because in the median draw a
lower price brings in more revenue.

Moving the baseline can't make the data say more about the price, though, so
both intervals still run across zero. [Recovering the truth](recovery) shows
why the data says so little about the price.

### All together

`joint_incremental_response` takes every input away at once, the treatments
included, but because the model adds the channels' effects together, the paid
channels' joint effect is the sum of their increments within each draw. The
simulation recorded what they added in `brand.truth`, so you can set that sum
beside the true total.

```{code-cell} ipython3
paid = shares["incremental_response"].sel(channel=list(channels.values())).sum("channel")
true_paid = brand.truth["contribution"].sel(channel=list(channels)).sum()
paid.quantile([0.05, 0.95]).round(-3).values, round(float(true_paid), -4)
```

Together, the ten channels brought in between \$7.59 million and
\$12.67 million in 90 percent of the draws, and that interval holds the true
\$12.01 million near its top. Its upper end is only 1.7 times its lower one,
against at least four times for each channel alone.

Comparing it with the prior shows how much of that firmness came from the
data. `prior_results` holds the prior draws from [Priors](priors), and
`group="prior"` points {func}`~mmmjax.contributions` at them, so the same sum
gives the total before the model saw any data.

```{code-cell} ipython3
prior_shares = mj.contributions(model, prior_results, quantity="mu", group="prior")
prior_paid = prior_shares["incremental_response"].sel(channel=list(channels.values())).sum("channel")
prior_paid.quantile([0.05, 0.95]).round(-3).values
```

Under the prior, the total runs from \$6.23 million to \$14.21 million, so the
data narrowed it only modestly.

:::{admonition} Why the total is firmer
:class: note

The total is firmer than any one channel partly because it adds up ten of
them. Even under the prior, its upper end is only about 2.3 times its lower
one. [Recovering the truth](recovery.md#week-by-week) checks the total week by
week against the simulation.
:::

## Returns on spending

{func}`~mmmjax.media_metrics` divides each paid channel's incremental revenue
by the money spent on it, so $\text{ROI}_c = \Delta_c / S_c$ for total spend
$S_c$. Email and the treatments have no spending, so they have no return and
the function leaves them out. Where the ROI averages over all the spending,
the marginal return asks what one more percent of spending would bring,

$$
\text{mROI}_c = \frac{M_c(1.01\, S_c) - M_c(S_c)}{0.01\, S_c},
$$

where $M_c(S)$ is total expected revenue when the channel's spending is $S$
and every other input stays fixed. Unless you pass another `metric`,
{func}`~mmmjax.plot_media_metrics` draws the average returns, so the plot below
shows each channel's ROI.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
mj.plot_media_metrics(returns)
```

Each bar is a channel's mean return on a dollar with its 89 percent interval,
and the dashed line marks break-even, where a dollar spent brings back a
dollar. Over the three years, the average dollar returned between \$2.79 on
Linear TV and \$4.06 on YouTube. Because a return divides $\Delta_c$ by a fixed
spend, its interval inherits the spread of the contributions above, and
Snapchat's is again the widest.

The marginal return can order the channels differently, and
{func}`~mmmjax.plot_roi_bubbles` shows where by setting it against the average
return.

```{code-cell} ipython3
mj.plot_roi_bubbles(returns)
```

The horizontal axis holds each channel's average return and the vertical axis
its marginal return, and a bubble's area follows its spending. Meta returns
\$3.49 on average against TikTok's \$3.20, yet its bubble sits below TikTok's,
so Meta earns less on the next dollar.

The four channels that spend in every week, Meta, Generic search, Branded
search, and Display, sit lowest of all. They're already further along their
curves, so each extra dollar adds less. That gap between the average and the
marginal return is what a budget decision turns on.

## Response curves

{func}`~mmmjax.response_curves` scales each paid channel's spending up and
down and records the revenue that comes with it. By default it goes from zero
to twice the current spending in 21 steps, and you can pass `multipliers` for
other steps.

At each step the new spending buys exposure at the observed cost per impression,
so the curves assume an extra dollar buys impressions as cheaply as the week's
other dollars did. If your impressions cost more as you buy more of them,
{func}`~mmmjax.response_curves`, {func}`~mmmjax.media_metrics`, and
{func}`~mmmjax.optimize_budget` all take `spend_to_media`, a function of your
own from spending to impressions.

```{code-cell} ipython3
:tags: [skip-execution]

curves = mj.response_curves(model, results, quantity="mu")
```

```{code-cell} ipython3
:tags: [remove-cell]

curves = first_model_curves(model, results)
```

Once the curves are computed, {func}`~mmmjax.plot_response_curves` gives each
channel a panel with its own axes.

```{code-cell} ipython3
mj.plot_response_curves(curves)
```

In each panel, the line is the mean revenue, the band its 89 percent interval,
and the point the current spending. Past the point the line turns dashed,
since the data never saw that much spending.

Every curve bends, so the dashed stretch that doubles a channel's spending adds
less revenue than the solid stretch before it. That bend is also why each
marginal return above falls short of its average return, and it's what keeps a
budget plan from pouring every dollar into the channel with the best average
return.

[Budget optimization](budgets) uses these same curves to split a budget
between the paid channels, and [Scenarios](scenarios) shows how to change the
data yourself and what that asks of your blocks.

Every number on this page is what the model implies under its assumptions, as
[What is mmmJAX](../getting_started/what_is_mmmjax.md#how-analyses-get-their-answers)
explains. To see how well those numbers hold up,
[Recovering the truth](recovery) checks each channel, Email, and the
treatments against the simulation.
