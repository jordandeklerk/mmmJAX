---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Media effects and budgets

mmmJAX's analysis functions answer the usual marketing mix questions from a
fitted model, and they all work the same way. Each takes the model, the draws,
and the name of the expected outcome, runs the blocks again on a changed copy
of the data with the same draws, and reports the difference in revenue. The
examples use the model from [A first model](first_model).

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import first_model_results

results = first_model_results(model)
```

## Contributions

{func}`~mmmjax.contributions` removes one channel at a time and measures the
revenue that goes with it. `quantity="mu"` names the expected revenue that
`transformed_parameters` returns, which is what every analysis function
compares.

Writing $m_t$ for expected revenue in dollars and $m_t^{(-c)}$ for the same
week with channel $c$ removed, each draw gives

$$
\Delta_c = \sum_t \big(m_t - m_t^{(-c)}\big),
\qquad
\text{share}_c = \frac{\Delta_c}{\sum_t m_t}.
$$

```{code-cell} ipython3
shares = mj.contributions(model, results, quantity="mu")
incremental = shares["incremental_response"].quantile(
    [0.05, 0.5, 0.95], dim=("chain", "draw")
)
incremental.T.to_pandas().round(-3)
```

Over the three years, TV brought in about \$1.3 million of revenue and search
about \$1.7 million in the middle of the posterior, against true values of
\$1.36 million and \$1.93 million. The intervals matter more than the medians.
TV's effect is pinned down to within about \$150,000 either way, while search's
interval is more than four times as wide. Search runs every week, so the data
never shows revenue without it, and its coefficient trades off against the
intercept, with a correlation of -0.71 across draws. The same result holds
each channel's share of revenue in `contribution_share` and the revenue left
with every channel removed in `baseline_response`.

## Returns on spending

{func}`~mmmjax.media_metrics` divides each channel's incremental revenue by
the money spent on it, so $\text{ROI}_c = \Delta_c / S_c$ for total spend
$S_c$. The marginal return asks what one more percent of spending would bring,

$$
\text{mROI}_c = \frac{M_c(1.01\, S_c) - M_c(S_c)}{0.01\, S_c},
$$

where $M_c(S)$ is total expected revenue when the channel's spending is $S$
and every other input stays fixed.

```{code-cell} ipython3
returns = mj.media_metrics(model, results, quantity="mu")
returns[["roi", "marginal_roi"]].median(("chain", "draw")).to_pandas().round(2)
```

Each dollar spent on TV over the period returned about \$3.50 in revenue, and
each dollar on search about \$4.10, close to the true \$3.70 and \$4.50. The
marginal return, measured on one more percent of spending, tells a different
story. It is about \$2.50 for TV and \$1.80 for search, because search already
runs where its curve has flattened. That gap is what a budget decision turns
on.

## Response curves

{func}`~mmmjax.response_curves` scales each channel's spending up and down and
records the revenue that comes with it.

```{code-cell} ipython3
curves = mj.response_curves(
    model, results, quantity="mu", multipliers=[0.5, 1.0, 1.5, 2.0]
)
curves["incremental_response"].median(("chain", "draw")).to_pandas().round(-3)
```

Doubling TV spending from about \$366,500 to \$733,000 would add another
\$741,000 in revenue, about two dollars for each extra dollar, well below the
\$3.50 it earns on average. Spending converts to exposure at the observed cost
per impression, and `curves["spend"]` holds the spending behind each
multiplier.

## Budgets

{func}`~mmmjax.optimize_budget` looks for the split of a fixed budget that
maximizes expected revenue.

```{code-cell} ipython3
plan = mj.optimize_budget(model, results, quantity="mu")
plan["spend"].to_pandas().round(-2)
```

It keeps the total at the \$792,000 spent over the three years and moves about
\$89,000 from search to TV, where the marginal return is higher. The draws say
how much that move is worth.

```{code-cell} ipython3
change = plan["response_change"]
print(change.quantile([0.05, 0.5, 0.95]).round(-3).values)
print(float((change > 0).mean()))
```

The median gain is about \$35,000 over the three years, the interval runs from
a loss of \$13,000 to a gain of \$81,000, and 89 percent of the draws favor the
new split. The optimizer maximizes the posterior mean, and the draws show how
confident the model is in the move. Spending bounds, constraints on groups of
channels, and other objectives, such as one that penalizes risk, are options
on the same function.

Every number on this page is what the model implies under its assumptions.
The functions run the model's own equations on changed data, so their answers
are only as causal as the model is, and they add no evidence of their own.
[Scenarios](scenarios) explains how the data gets changed and what that asks
of the blocks.
