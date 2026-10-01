# User Guide

In this guide you'll build one marketing mix model and then work through each
part of it. The model is for a brand with ten paid channels, an email
newsletter, two controls, a price and promotions, a trend, and a yearly season.
Every page after [A first model](first_model) builds on the model you write
there, so the sections read best in order.

Because the brand is simulated, you know its true effects and can check the
model's answers against them. You'll do that for the returns at the end of
[A first model](first_model.md#returns-against-the-truth), for the other
answers on [Recovering the truth](recovery), and for the budget plans on
[Budget optimization](budgets.md#plans-against-the-truth).

| Section | Description |
| --- | --- |
| [Fundamentals](fundamentals) | The example data, how mmmJAX prepares and checks it, and the distributions models are written with |
| [Modeling and plotting](modeling_and_plotting) | The guide's first model, the plots that show its results, and how to customize them |
| [Priors and inference](priors_and_inference) | Checking priors, sampling, reading diagnostics, and other ways to fit |
| [Analysis](analysis) | Contributions, returns, budgets, and scenarios from the fitted model, and how close its answers come to the truth |
| [Extending the model](extending) | New model parts, regional models, and functions of your own |

```{toctree}
:hidden:
:maxdepth: 2

fundamentals
modeling_and_plotting
priors_and_inference
analysis
extending
```

<p class="mmmj-footer-logo">
  <img src="../_static/mmmjax-logo.svg" alt="mmmJAX logo">
</p>
