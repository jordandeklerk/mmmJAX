# Extending the model

These pages change the model itself. They swap its response curve,
likelihood, and trend, fit it region by region, and write functions of your
own for its blocks.

| Guide | Description |
| --- | --- |
| [Changing the model](changing) | A new response curve, likelihood, and trend, compared with the first model |
| [Geo-level models](geo) | The brand fit by region, with returns that vary around a shared one |
| [User-defined functions](functions) | Your own response curve and distribution, and the rules JAX sets for them |

```{toctree}
:hidden:
:maxdepth: 1

changing
geo
functions
```
