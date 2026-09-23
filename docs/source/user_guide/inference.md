---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Inference

{func}`~mmmjax.sample` is one way to fit a model, not the only one. A model
exposes its log density and the transforms around it as plain JAX functions,
so any sampler that works with a JAX log density can fit it, and the analysis
functions accept the draws it returns. This page covers what the log density
guarantees, runs BlackJAX and NumPyro directly on the model from [A first
model](first_model), and tries an approximation that skips MCMC.

```{code-cell} ipython3
:tags: [remove-cell]

%run prerun/first_model.py
from prerun import stored
```

## The interface

A model works on an unconstrained position, a dictionary with one array per
parameter, and it provides everything a gradient-based sampler needs to move
around that space.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

position = model.initialize_random(jax.random.key(0))
value, gradient = jax.value_and_grad(model.log_density)(position, model.data)
model.constrain(position)["retention"]
```

{meth}`~mmmjax.Model.initialize_random` draws a starting point,
{meth}`~mmmjax.Model.log_density` evaluates the model at a position, and
{meth}`~mmmjax.Model.constrain` maps a position back to the natural scale,
where each retention rate lies between zero and one.
{meth}`~mmmjax.Model.unconstrain` goes the other way.

## What the log density guarantees

A sampler needs a log density that returns one number, is finite at the
starting point, and has a gradient everywhere it goes. If $T$ maps an
unconstrained position $u$ to the parameters $\theta = T(u)$, the density the
sampler sees is

$$
\log p\big(T(u)\big) + \log \big\lvert \det J_T(u) \big\rvert ,
$$

and the declarations supply the second term, so a density written on the
natural scale is correct on the unconstrained one. The rest depends on the
blocks you write.

```{code-cell} ipython3
finite_gradient = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
bool(jnp.isfinite(value)), finite_gradient
```

Both checks pass for the first model. The density also has to be pure, giving
the same value every time it sees the same position, which rules out
randomness and changing global state inside `transformed_parameters` and
`log_density`. It does not have to be normalized, since samplers only use
differences in the log density and its gradient, but mmmJAX's distributions
keep their constants, so pointwise log likelihoods stay comparable across
models. [User-defined functions](functions) covers the rules JAX sets for code
that runs inside a block.

## BlackJAX

BlackJAX is the library {func}`~mmmjax.sample` uses for NUTS, and it can run on
the model directly. The function below adapts a NUTS kernel with window
adaptation and then draws from one chain, keeping each draw's position and
whether its transition diverged.

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

The draws come back on the unconstrained scale. `to_results` maps them to the
natural scale and labels them the way mmmJAX labels its own results, with
chain and draw axes and the channel and control names from the data.

```{code-cell} ipython3
import xarray as xr


def to_results(draws, diverging=None):
    constrained = jax.vmap(jax.vmap(model.constrain))(draws)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", *model.parameters[name].dims), values)
            for name, values in constrained.items()
        },
        coords={"channel": list(data.channels), "control": list(data.columns["controls"])},
    )
    groups = {"posterior": posterior}
    if diverging is not None:
        groups["sample_stats"] = xr.Dataset({"diverging": (("chain", "draw"), diverging)})
    results = xr.DataTree.from_dict(groups)
    return results
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


blackjax_results = stored("blackjax", fit_blackjax)
```

```{code-cell} ipython3
int(blackjax_results["sample_stats"]["diverging"].sum())
```

None of the 4,000 transitions diverged, and the analysis functions take these
results as they are.

```{code-cell} ipython3
shares = mj.contributions(model, blackjax_results, quantity="mu")
shares["contribution_share"].mean(("chain", "draw")).to_series().round(3)
```

The shares match the ones {func}`~mmmjax.sample` produced in [A first
model](first_model), as two runs of the same sampler on the same model should.

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


numpyro_results = stored("numpyro", fit_numpyro)
```

```{code-cell} ipython3
shares = mj.contributions(model, numpyro_results, quantity="mu")
shares["contribution_share"].mean(("chain", "draw")).to_series().round(3)
```

NumPyro's samples come back grouped by chain and keyed by parameter name, so
the same `to_results` labels them.

## Pathfinder

Pathfinder, also part of BlackJAX, fits a variational approximation along the
path of an optimizer instead of running a Markov chain, and then draws from
that approximation.

```{code-cell} ipython3
:tags: [skip-execution]

approximate_key, sample_key = jax.random.split(jax.random.key(4))
state, _ = blackjax.vi.pathfinder.approximate(
    approximate_key, logdensity, model.initialize_random(jax.random.key(5))
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
        approximate_key, logdensity, model.initialize_random(jax.random.key(5))
    )
    draws, _ = blackjax.vi.pathfinder.sample(sample_key, state, 1000)
    one_chain = {name: values[None] for name, values in draws.items()}
    return to_results(one_chain)


pathfinder_results = stored("pathfinder", fit_pathfinder)
```

```{code-cell} ipython3
shares = mj.contributions(model, pathfinder_results, quantity="mu")
shares["contribution_share"].mean(("chain", "draw")).to_series().round(3)
```

The approximation puts search's share at about 5.1 percent, below the 5.6 and
5.7 percent the two NUTS runs found and the true 6.1 percent, so it serves for
a quick look or as starting points for MCMC rather than as the final fit.

## From draws to everything else

Once the draws are labeled, the rest of mmmJAX treats them like its own.
{func}`~mmmjax.generate_quantities` adds the predictions and pointwise log
likelihoods that ArviZ uses for the checks in [Fitting and checking](checking).

```{code-cell} ipython3
mj.generate_quantities(model, blackjax_results)
```

The one exception is {func}`~mmmjax.continue_sampling`, which resumes from the
sampler state that only {func}`~mmmjax.sample` records.
