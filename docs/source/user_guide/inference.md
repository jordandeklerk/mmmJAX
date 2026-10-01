---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Inference

You don't have to fit a model with {func}`~mmmjax.sample`, because a model
exposes its log density and the transforms around it as plain JAX functions.
Any sampler that works with a JAX log density can fit it, and the analysis and
plotting functions accept the draws it returns. After a look at what the log
density guarantees, you'll run BlackJAX and NumPyro directly on the
ten-channel brand model from [A first model](first_model) and try an
approximation that skips MCMC.

The Pathfinder section needs 64-bit precision, so the whole page runs in it.
The cell below turns it on before the model is built, because a model keeps
the precision that was in effect when it was created, as
[Installation](../getting_started/installation.md#choosing-float32-or-float64)
explains.

```{code-cell} ipython3
import jax

jax.config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
:tags: [remove-cell]

%run -m prerun.first_model
from prerun import first_model_results, stored

results = first_model_results(model)
```

## The interface

A model works on an unconstrained position, a dictionary with one array for
each parameter under the name you declared in `parameters`. It provides
everything a gradient-based sampler needs to move around that space.

```{code-cell} ipython3
import jax.numpy as jnp

position = model.initialize_random(jax.random.key(0))
value, gradient = jax.value_and_grad(model.log_density)(position, model.data)
model.constrain(position)["retention"]
```

In the cell, {meth}`~mmmjax.Model.initialize_random` draws a starting point
and {meth}`~mmmjax.Model.log_density` evaluates the model at a position.
{meth}`~mmmjax.Model.constrain` maps a position back to the natural scale, and
{meth}`~mmmjax.Model.unconstrain` goes the other way. On the natural scale, each
of the ten retention rates in the output lies between zero and one.

## What the log density guarantees

A sampler needs a log density that returns one number, is finite at the
starting point, and has a gradient everywhere it goes. If $T$ maps an
unconstrained position $u$ to the parameters $\theta = T(u)$, the density the
sampler sees is

$$
\log p\big(T(u)\big) + \log \big\lvert \det J_T(u) \big\rvert ,
$$

and the declarations supply the second term, so a density written on the
natural scale is correct on the unconstrained one. The rest is up to the
blocks you write, so the cell below checks that the value and every part of
the gradient are finite at the starting point drawn above.

```{code-cell} ipython3
finite_gradient = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
bool(jnp.isfinite(value)), finite_gradient
```

Both come back `True`, so the brand model passes the check at that starting
point. When you write your own blocks, [User-defined functions](functions)
covers the rules JAX sets for code that runs inside one.

:::{admonition} Keep the density pure
:class: important

The density has to give the same value every time it sees the same position.
That rules out randomness and changing global state inside
`transformed_parameters` and `log_density`.
:::

## Labeling the draws

The samplers below return their draws on the unconstrained scale, so
`to_results` maps them to the natural scale and labels them the way mmmJAX
labels its own results.

```{code-cell} ipython3
import xarray as xr


def to_results(draws, diverging=None):
    constrained = jax.vmap(jax.vmap(model.constrain))(draws)
    dims = {name: parameter.dims for name, parameter in model.parameters.items()}
    # The seasonal coefficients have no dims, so their axis takes the name sample gives it.
    dims["annual_coefficients"] = ("annual_coefficients_dim_0",)
    posterior = xr.Dataset(
        {name: (("chain", "draw", *dims[name]), values) for name, values in constrained.items()},
        coords={
            "channel": list(data.channels),
            "organic_channel": list(data.organic_channels),
            "control": list(data.columns["controls"]),
            "treatment": list(data.columns["treatments"]),
            "annual_coefficients_dim_0": [0, 1, 2, 3],
        },
    )
    groups = {"posterior": posterior}
    if diverging is not None:
        groups["sample_stats"] = xr.Dataset({"diverging": (("chain", "draw"), diverging)})
    results = xr.DataTree.from_dict(groups)
    return results
```

:::{admonition} Label draws the way sample does
:class: note

{func}`~mmmjax.media_metrics`, {func}`~mmmjax.generate_quantities`, and the
other functions that evaluate the model on draws reject any other labels. A
data axis named in `dims`, one of the [supplied names](data.md#supplied-names),
needs the data's labels in the data's order. An axis of a parameter declared
without `dims` needs labels counting from zero and the parameter's name
followed by `_dim_0`, `_dim_1`, and so on.
:::

## BlackJAX

BlackJAX is the library {func}`~mmmjax.sample` uses for NUTS, and it can run on
the model directly. The function below tunes a NUTS kernel with window
adaptation, runs one chain, and keeps each draw's position along with whether
its transition diverged. Window adaptation's acceptance target of 0.8 and
diagonal mass matrix, and the kernel's limit of 10 on tree depth, are the
defaults {func}`~mmmjax.sample` uses too, so the two fits differ in their random
numbers rather than their sampler settings.

```{code-cell} ipython3
import blackjax


def logdensity(position):
    return model.log_density(position, model.data)


def run_chain(key, num_warmup=1000, num_draws=1000):
    init_key, warmup_key, sample_key = jax.random.split(key, 3)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, model.initialize_random(init_key), num_steps=num_warmup
    )
    kernel = blackjax.nuts(logdensity, **parameters)

    def one_step(state, step_key):
        state, info = kernel.step(step_key, state)
        return state, (state.position, info.is_divergent)

    _, (positions, diverging) = jax.lax.scan(
        one_step, state, jax.random.split(sample_key, num_draws)
    )
    return positions, diverging
```

`jax.vmap` runs four chains at once, one for each key, and stacks them into
one set of results.

```{code-cell} ipython3
:tags: [skip-execution]

keys = jax.random.split(jax.random.key(1), 4)
positions, diverging = jax.vmap(run_chain)(keys)
blackjax_results = to_results(positions, diverging)
```

```{code-cell} ipython3
:tags: [remove-cell]

def fit_blackjax():
    keys = jax.random.split(jax.random.key(1), 4)
    positions, diverging = jax.vmap(run_chain)(keys)
    return to_results(positions, diverging)


blackjax_results = stored("blackjax", fit_blackjax, groups=["posterior", "sample_stats"])
```

```{code-cell} ipython3
int(blackjax_results["sample_stats"]["diverging"].sum())
```

None of the 4,000 transitions diverged, so you can move on to
{func}`~mmmjax.plot_rhat` and check that the four chains agree.

```{code-cell} ipython3
mj.plot_rhat(blackjax_results)
```

The subtitle confirms that all 45 values sit at or below ArviZ's limit of 1.01,
and the other convergence checks in [Sampling and diagnostics](sampling), such
as the rank plot, read these results the same way. The checks on predictions
need predictive draws, and the last section adds them.

The next check is whether the two fits give the same answer.
{func}`~mmmjax.media_metrics` computes each channel's return on investment from
`blackjax_results` and from `results`, the fit that {func}`~mmmjax.sample`
produced in [A first model](first_model). {func}`~mmmjax.plot_media_metrics`
then draws the two side by side as one pair of bars for each channel.

```{code-cell} ipython3
sample_returns = mj.media_metrics(model, results, quantity="mu")
blackjax_returns = mj.media_metrics(model, blackjax_results, quantity="mu")
mj.plot_media_metrics({"sample": sample_returns, "BlackJAX": blackjax_returns})
```

Each pair of bars agrees in its mean and its interval, as two runs of the same
sampler on the same model should. Meta, for example, returns \$3.49 on a
dollar in the fit from `sample` and \$3.55 in the one from BlackJAX.

## NumPyro

NumPyro's NUTS accepts a potential function, the negative log density, in
place of a NumPyro model, along with starting positions for its chains.

```{code-cell} ipython3
from numpyro.infer import MCMC, NUTS


def potential(position):
    return -model.log_density(position, model.data)
```

```{code-cell} ipython3
:tags: [skip-execution]

mcmc = MCMC(
    NUTS(potential_fn=potential),
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    chain_method="sequential",
)
initial = jax.vmap(model.initialize_random)(jax.random.split(jax.random.key(3), 4))
mcmc.run(jax.random.key(2), init_params=initial, extra_fields=("diverging",))
diverging = mcmc.get_extra_fields(group_by_chain=True)["diverging"]
numpyro_results = to_results(mcmc.get_samples(group_by_chain=True), diverging)
```

```{code-cell} ipython3
:tags: [remove-cell]

def fit_numpyro():
    mcmc = MCMC(
        NUTS(potential_fn=potential),
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        chain_method="sequential",
        progress_bar=False,
    )
    initial = jax.vmap(model.initialize_random)(jax.random.split(jax.random.key(3), 4))
    mcmc.run(jax.random.key(2), init_params=initial, extra_fields=("diverging",))
    diverging = mcmc.get_extra_fields(group_by_chain=True)["diverging"]
    return to_results(mcmc.get_samples(group_by_chain=True), diverging)


numpyro_results = stored("numpyro", fit_numpyro, groups=["posterior", "sample_stats"])
```

The samples come back grouped by chain and keyed by parameter name, so the
same [`to_results`](#labeling-the-draws) labels them, and their returns can go
next to the ones from `sample`.

```{code-cell} ipython3
numpyro_returns = mj.media_metrics(model, numpyro_results, quantity="mu")
mj.plot_media_metrics({"sample": sample_returns, "NumPyro": numpyro_returns})
```

Even though NumPyro's NUTS is a separate implementation, it lands on the same
returns. The widest gap is YouTube's, \$4.15 from NumPyro against \$4.06 from
`sample`, and Branded search differs by almost as much, \$3.56 against \$3.48.
Each mean carries Monte Carlo error, 0.026 for YouTube's in the summary on
[Sampling and diagnostics](sampling.md#convergence), so two correct runs can
differ by several cents, and both gaps are small next to intervals that span
several dollars.

## Pathfinder

Pathfinder, also part of BlackJAX, fits a variational approximation along the
path of an optimizer instead of running a Markov chain. Of the approximations
along that path, it keeps the one with the highest ELBO, its estimate of how
close each one comes to the posterior, and draws from it.

```{code-cell} ipython3
:tags: [skip-execution]

approximate_key, sample_key = jax.random.split(jax.random.key(4))
state, _ = blackjax.vi.pathfinder.approximate(
    approximate_key, logdensity, model.initialize_random(jax.random.key(5)), maxiter=1000
)
draws, _ = blackjax.vi.pathfinder.sample(sample_key, state, 1000)
one_chain = {name: values[None] for name, values in draws.items()}
pathfinder_results = to_results(one_chain)
```

```{code-cell} ipython3
:tags: [remove-cell]

def fit_pathfinder():
    approximate_key, sample_key = jax.random.split(jax.random.key(4))
    state, _ = blackjax.vi.pathfinder.approximate(
        approximate_key, logdensity, model.initialize_random(jax.random.key(5)), maxiter=1000
    )
    draws, _ = blackjax.vi.pathfinder.sample(sample_key, state, 1000)
    one_chain = {name: values[None] for name, values in draws.items()}
    return to_results(one_chain)


pathfinder_results = stored("pathfinder", fit_pathfinder, groups=["posterior"])
```

On this model the default of 30 optimizer steps ends far from the posterior
mode, while the gradient is still large, so the call raises `maxiter`. Given
that room, the optimizer stops on its own well before the limit.

:::{admonition} Pathfinder in 32-bit
:class: warning

BlackJAX finds the log determinant of the approximation's covariance by taking
the log of a product of 45 variances, one per unconstrained parameter, and in
32-bit that product rounds to zero. Every ELBO then comes out as minus
infinity, and Pathfinder falls back to the approximation at its random starting
point, so its draws carry nothing the optimizer learned. To confirm a run
avoided this, check that `state.elbo` is finite.
:::

```{code-cell} ipython3
pathfinder_returns = mj.media_metrics(model, pathfinder_results, quantity="mu")
mj.plot_media_metrics({"sample": sample_returns, "Pathfinder": pathfinder_returns})
```

Pathfinder's orange intervals are far narrower than the blue ones from NUTS,
because an approximation chosen by its ELBO tends to understate how spread out
a posterior is. Its means move as well, and nine of the ten fall. TikTok's
drops the furthest, from \$3.20 to \$2.63, and Streaming's is the one that
rises, from \$4.04 to \$4.36.

:::{admonition} What Pathfinder is for
:class: warning

Use Pathfinder's draws for a quick look or as starting points for MCMC, not
as the final fit.
:::

## From draws to everything else

Once the draws are labeled, the rest of mmmJAX treats them like its own.
{func}`~mmmjax.generate_quantities` adds the predictions that ArviZ uses for
the checks in [Sampling and diagnostics](sampling), and
{func}`~mmmjax.plot_fit` draws the predictions against observed revenue.

```{code-cell} ipython3
:tags: [skip-execution]

generated = mj.generate_quantities(model, blackjax_results)
```

```{code-cell} ipython3
:tags: [remove-cell]

generated = stored(
    "blackjax_generated",
    lambda: mj.generate_quantities(model, blackjax_results),
    groups=["posterior_predictive", "observed_data"],
)
```

```{code-cell} ipython3
mj.plot_fit(model, generated)
```

The subtitle gives the BlackJAX draws an $R^2$ of 0.91, a weighted mean
absolute percentage error of 4.1 percent, and 93 percent of weeks inside the
band, much as [Plotting](plotting) finds for the draws from `sample`.

{func}`~mmmjax.continue_sampling` alone can't use these draws, because it
resumes from the sampler state that only {func}`~mmmjax.sample` records.
