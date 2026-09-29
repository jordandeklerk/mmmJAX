# Getting Started

mmmJAX is an open-source Python library for Bayesian marketing mix modeling in
[JAX](https://docs.jax.dev/). You write the model as a
[Stan](https://mc-stan.org/)-style program of plain Python blocks, and
everything around it comes as library functions, from data preparation to
sampling, budget optimization, and plots.

```{toctree}
:hidden:
:maxdepth: 1

installation
intro_to_mmm
what_is_mmmjax
quickstart
```

## Who it's for

mmmJAX is for you if you've built marketing mix models before and you're
comfortable with Bayesian modeling. You know where a typical MMM's assumptions
break down for your data, and you'd rather change the model than work around
an API that fixes it. mmmJAX never picks a trend, a carryover, a prior, or a
likelihood for you, so every modeling decision is yours.

:::{admonition} New to MMM or Bayesian modeling
:class: warning

If you're new to marketing mix models or to Bayesian modeling in general,
expect mmmJAX to feel strange and hard at first. A packaged tool with sensible
defaults and a ready-made model is a more forgiving place to fit your first
models.
:::

## Where to start

The four pages build on each other, so read them in order.

| Page | Description |
| --- | --- |
| [Installation](installation) | Install mmmJAX with pip or uv, and set it up for a GPU, float64, and parallel chains |
| [Introduction to MMM](intro_to_mmm) | The math behind marketing mix models and the Bayesian methods that fit them |
| [What is mmmJAX](what_is_mmmjax) | How a model goes from its math to a program of blocks, and how mmmJAX runs it |
| [Quickstart](quickstart) | A complete model to copy and run on simulated data |

The [User Guide](../user_guide/index) then builds one model and works through
each part of it.

<p class="mmmj-footer-logo">
  <img src="../_static/mmmjax-logo.svg" alt="mmmJAX logo">
</p>
