---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Priors

mmmJAX never chooses a prior for you. Every prior is a line you write in
`log_density` with a function from [Distributions](distributions). This page
covers what the priors from [A first model](first_model) claim in revenue
terms and how to check them before fitting.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
```

## What a prior claims

The model works on scaled data, so its priors are statements about scaled
quantities, and converting one back to dollars is the quickest way to see what
it claims. The coefficient $\beta_c$ is the most a channel can add to weekly
revenue, reached only at full saturation, measured in standard deviations of
revenue, so it is worth $\beta_c s_r$ dollars a week for revenue's standard
deviation $s_r$.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

revenue_scale = scaling.transformations["outcome"].scale
draws = mj.half_normal_rng(jax.random.key(1), 2.0, sample_shape=(10_000,))
jnp.quantile(draws, jnp.array([0.5, 0.95])) * revenue_scale
```

Under its half-normal prior, a channel could add about \$14,000 a week at
saturation in the median case, with a 5 percent chance of more than \$40,000,
against average weekly revenue of about \$201,000. The simulation's true
values, about \$34,000 for TV and \$26,000 for search, sit comfortably inside
that range, and a single channel that doubles revenue is ruled out.

## Simulating from the priors

Priors that look reasonable one at a time can still add up to an implausible
model. A prior predictive check simulates revenue from the priors alone,
before the model sees any data.

```{code-cell} ipython3
priors = {
    "intercept": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "coefficient": mj.Prior(mj.half_normal, scale=2.0),
    "retention": mj.Prior(mj.beta, alpha=2.0, beta=2.0),
    "half_saturation": mj.Prior(mj.lognormal, location=0.0, scale=0.5),
    "control_coefficient": mj.Prior(mj.normal, location=0.0, scale=1.0),
    "sigma": mj.Prior(mj.half_normal, scale=1.0),
}
```

```{code-cell} ipython3
:tags: [skip-execution]

prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
```

```{code-cell} ipython3
:tags: [remove-cell]

from prerun import stored

prior_results = stored("first_model_prior", lambda: mj.sample_prior(model, priors, draws=500, seed=0))
```

```{code-cell} ipython3
import numpy as np

prediction = prior_results["prior_predictive"]["prediction"].values
prior_revenue = scaling.transformations["outcome"].inverse_transform(prediction)
print(round(float((prior_revenue < 0).mean()), 4))
print(np.quantile(prior_revenue, [0.05, 0.5, 0.95]).round(-3))
```

{func}`~mmmjax.sample_prior` draws each parameter from its
{class}`~mmmjax.Prior`, then runs `transformed_parameters` and
`generated_quantities` on those draws. None of the simulated weeks has
negative revenue, and the middle 90 percent falls between about \$183,000 and
\$247,000, around the observed weeks, which run from \$175,000 to \$224,000.

:::{admonition} Keep the two in sync
:class: warning

`sample_prior` never calls `log_density`, so nothing forces the mapping to
match the density. A prior changed in one place and not the other makes the
prior check describe a different model from the one you fit.
:::

Writing the density with the same objects keeps the two together, since
calling a `Prior` returns its summed log density.

```{code-cell} ipython3
def prior_log_density(
    outcome,
    mu,
    intercept,
    coefficient,
    retention,
    half_saturation,
    control_coefficient,
    sigma,
):
    target = priors["intercept"](intercept)
    target += priors["coefficient"](coefficient)
    target += priors["retention"](retention)
    target += priors["half_saturation"](half_saturation)
    target += priors["control_coefficient"](control_coefficient)
    target += priors["sigma"](sigma)
    target += mj.normal(outcome, mu, sigma)
    return target
```

## Checking one parameter

{func}`~mmmjax.check_prior` reports how much prior mass falls outside limits
you consider plausible.

```{code-cell} ipython3
checked = mj.check_prior(prior_results["prior"]["retention"], upper=0.8)
checked["probability_above"].to_series()
```

A retention of 0.8 would leave a third of an exposure's effect in place five
weeks later, and the beta prior gives that about a 10 percent chance for
either channel, which leaves room for TV's true retention of 0.7. When one
prior depends on another, as in a hierarchy, `sample_prior` also accepts a
function that takes a random key and returns one draw of every parameter. A
family the library lacks can be added with
{func}`~mmmjax.custom_distribution`, which [User-defined functions](functions)
covers.
