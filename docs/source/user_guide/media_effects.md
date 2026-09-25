---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Media effects

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
shares
```

Like the other analysis functions, it returns an xarray Dataset with a value
for every draw and the settings behind the result in its attributes.
`incremental_response` holds
each channel's $\Delta_c$ in dollars and `contribution_share` its share of
revenue, while `baseline_response` holds the revenue left with every channel
removed.

```{code-cell} ipython3
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
intercept, with a correlation of -0.69 across draws.

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

Each dollar spent on TV over the period returned about \$3.60 in revenue, and
each dollar on search about \$4.00, close to the true \$3.70 and \$4.50. The
marginal return, measured on one more percent of spending, tells a different
story. It is about \$2.50 for TV and \$1.80 for search, because in the model
search already runs where its curve has flattened. That gap is what a budget
decision turns on.

## Response curves

{func}`~mmmjax.response_curves` scales each channel's spending up and down and
records the revenue that comes with it.

```{code-cell} ipython3
curves = mj.response_curves(
    model, results, quantity="mu", multipliers=[0.5, 1.0, 1.5, 2.0]
)
curves["incremental_response"].median(("chain", "draw")).to_pandas().round(-3)
```

`curves["spend"]` holds the spending behind each multiplier.

```{code-cell} ipython3
curves["spend"].to_pandas().round(-2)
```

Doubling TV spending from about \$366,500 to \$733,000 would add another
\$733,000 in revenue, about two dollars for each extra dollar, well below the
\$3.60 it earns on average. Spending converts to exposure at the observed cost
per impression. [Budget optimization](budgets) uses the same curves to split a
budget between the channels, and [Plotting](plotting) shows how to draw
curves, contributions, and returns like these.

Every number on this page is what the model implies under its assumptions.
The functions run the model's own equations on changed data, so their answers
are only as causal as the model is, and they add no evidence of their own.
[Recovering the truth](recovery) checks these numbers against the simulation,
and [Scenarios](scenarios) explains how the data gets changed and what that
asks of the blocks.
