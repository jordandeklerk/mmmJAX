---
name: writing-mmmjax-models
description: Writes, extends, and debugs mmmJAX marketing mix models built from Stan-style blocks (parameters, transformed_data, transformed_parameters, log_density, generated_quantities). Use when the user asks to write or build a model, add a channel, control, parameter, prior, likelihood, trend, seasonality, HSGP, hierarchy, calibration or custom function, prepare or scale data for a model, or fix a model that raises a binding or tracing error, has a non-finite log density, samples with divergences, or gives wrong scenario, contribution or ROI results.
---

# Writing mmmJAX models

A model is a dict of parameter declarations plus plain functions (blocks) that
mmmJAX fills by argument name. Start every new model from
`docs/source/user_guide/prerun/first_model.py`, the model the docs build runs.
It prepares `data` and `scaling`, declares `parameters`, defines
`transformed_parameters`, `log_density`, and `generated_quantities`, and builds
`model`. Signatures and parameter descriptions are in the numpydoc docstrings
in `mmmjax/`, and `mmmjax/__init__.py` lists the public API.

## Checks

Run these on every new or changed model.

```python
import jax
import jax.numpy as jnp

import mmmjax as mj

# Construction checks names only, and only jit exposes Python control flow on traced values.
position = model.initialize_random(jax.random.key(0))
value, gradient = jax.jit(jax.value_and_grad(model.log_density))(position, model.data)
finite = bool(jnp.isfinite(value)) and all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(gradient))
mu_shape = model.evaluate(model.constrain(position))["mu"].shape

results = mj.sample(model, draws=1000, warmup=1000, chains=4, seed=7)
divergences = int(results["sample_stats"]["diverging"].sum())
shares = mj.contributions(model, results, quantity="mu")
```

`finite` must be true and `mu_shape` must equal `data.arrays["outcome"].shape`
before sampling. The `contributions` call reruns the blocks on scenario data,
so shares of exactly zero point to rule 4. Check convergence as
`docs/source/user_guide/sampling.md` shows.

## How blocks bind

Every block argument is looked up by name among the data inputs, the
`Data(constants=...)`, the `transformed_data` outputs, the parameters, and the
`transformed_parameters` outputs. The data inputs are the roles passed to
`prepare_data` plus `time`, `media_time`, `day_of_year`, `media_day_of_year`,
`n_periods`, `outcome_scaling`, and `reference`, and `data.model_inputs` lists
them. Arguments must be plain named parameters, with no `*args`, `**kwargs`,
or positional-only ones.

- A parameter named after a data role or a built-in fails at construction, and
  so does a `transformed_data` output that reuses an input or parameter name.
  A `transformed_parameters` output that does so fails only at the first
  evaluation.
- `transformed_data` cannot request parameters. It returns arrays, or Python
  ints that stay static and can set shapes, with the same names for every
  dataset.
- `transformed_parameters` returns a dict. Its outputs are not saved unless
  `generated_quantities` returns them.
- `log_density` returns one real scalar, the sum of every prior and the
  likelihood.
- `generated_quantities` receives a random key as its first positional
  argument, under any name that is not an input. Split it with
  `jax.random.split` for more than one draw. The keys `predictive`,
  `log_likelihood`, and `log_prior` fill those result groups. Any other key
  becomes a generated quantity, which needs
  `Model(generated_dims={"name": "time"})` for axis labels.
- Parameter `dims` come from the data axes (`time`, `media_time`, `group`,
  `channel`, `control`, `organic_channel`, `rf_channel`, `organic_rf_channel`,
  `treatment`). Any other axis needs `Model(coords={"axis": labels})`.
- `Data(variables={"argument": "source"})` renames inputs. Once it is given,
  blocks see only the declared names plus constants and `transformed_data`
  outputs, so built-ins must be declared too, as in `"training": "reference"`.

## Rules

1. Every declared parameter needs its own prior line in `log_density`, per the
   prior rule in `.claude/CLAUDE.md`. A parameter requested only by
   `transformed_parameters` passes every check and silently gets a flat prior,
   so request each one in `log_density` as well. When a request leaves a prior
   open, or a fix needs a different prior or parameterization, propose it to
   the user instead of choosing silently.
2. Every model's `generated_quantities` returns both `predictive` and
   `log_likelihood`, not only docs fits. Draw with the likelihood's `_rng` form
   and score with its `_logpdf` or `_logpmf` form. Key both `"outcome"` even
   when `Data(variables=...)` renames the argument, because `observed_data`
   always names it `outcome`.
3. Priors describe the scaled data, which means media divided by its positive
   median, standardized controls and treatments, and the outcome only when
   `scale_outcome` is set. Multiply by
   `scaling.transformations["outcome"].scale` to read one in outcome units.
   When the user runs `sample_prior`, write `log_density` with the same
   `Prior` objects (`priors["sigma"](sigma)`), since `sample_prior` never calls
   `log_density`.
4. The analysis functions rerun the blocks on scenario data. A definition,
   such as a trend normalized by the training window, a centering, or an ROI
   or contribution calibration, reads training arrays from `reference`
   (`reference.time`, `reference.media`, `reference.spend`). A prediction reads
   the current inputs. An array captured from outside a block makes every
   scenario ignore it, and contributions come out as zero. See
   `docs/source/user_guide/scenarios.md`.
5. The analysis `quantity` must be a `transformed_parameters` output, floating,
   exactly the outcome's shape, and the expected outcome on the likelihood's
   scale. The functions undo `outcome_scaling` themselves. For a log link,
   return `jnp.exp(eta)` as its own output and pass that name.
6. Blocks are compiled under `jax.jit`, and the analysis functions also rerun
   `transformed_data` on traced scenario data. Use `jax.numpy` in every block,
   since NumPy in `transformed_data` builds and fits but fails in
   `contributions`, `media_metrics`, and `optimize_budget`. Keep randomness out
   of every block except `generated_quantities`. Shape-setting integers such
   as `max_lag`, a Fourier `order`, or a slice length must be Python ints,
   written inline, passed through `Data(constants=...)`, or returned by
   `transformed_data`.
7. With `prepare_data(..., media_history=earlier)`, exposure arrays gain
   leading rows and the outcome, spend, and controls do not. Request
   `n_periods` and slice `saturated[-n_periods:]`, or pass `n_periods` to
   `mj.media_response`.
8. For dated data `time` is days since the first period (0, 7, 14 for weekly
   data), so trend slopes, HSGP length scales, and Fourier periods on `time`
   are in days, and `day_of_year` runs from 1 to 366. Both stay 1-D in grouped
   data, so broadcast them with `[:, None]`.
9. Grouped data (`prepare_data(..., groups=["region"])`) adds a `group` axis
   after time, giving outcome `(time, group)` and media
   `(media_time, group, channel)`. A `(group, channel)` coefficient needs
   `(saturated * coefficient).sum(-1)` in place of `@`, while
   `controls @ control_coefficient` works in both layouts.
10. For float64, call `jax.config.update("jax_enable_x64", True)` before
    `fit_data_scaling`, `mj.Prior`, or `mj.Model`, which keep the precision in
    effect when they are built.
11. Count likelihoods need an unscaled outcome, which is the
    `fit_data_scaling` default. Build the predictor on the real line and use
    `negative_binomial_log` or `poisson_log`.

## Debugging

- `unknown input 'x'` means a misspelled name, a missing role, or, with
  `Data(variables=...)`, an undeclared input.
- `is ambiguous across ['data', 'parameter']` or `conflict with model-supplied
  inputs` means a parameter shares a data or built-in name. Rename the
  parameter.
- `must request every declared parameter` means no block asks for it, and
  rule 1 applies.
- `TracerBoolConversionError`, `TracerArrayConversionError`,
  `ConcretizationTypeError`, or `max_lag must be a static nonnegative integer`
  points to rule 6.
- A log density of `-inf` at the start comes from data outside the
  likelihood's support, such as a standardized outcome under `lognormal` or a
  fractional count under `poisson`. A non-finite gradient often comes from a
  Hill slope below one on a channel with zero-exposure weeks, which
  `mj.LowerBound(1.0)` prevents, as `changing.md` shows.
- `Transformed quantity 'x' is not available` or `The response quantity must be
  floating-point with the observation shape` points to rule 5.
- `generated_quantities requires input 'outcome'` means new data without an
  outcome column reached a block that scores the outcome.
- `New data must be an eager dataframe or PreparedData` most often means
  `new_data` received the output of `model.prepare_data`. Pass the frame
  itself.
- For divergences, raise `target_accept` toward 0.95, then try
  `mass_matrix="dense"`, then change the model, for example to a non-centered
  hierarchy, under rule 1.

## Deeper references

These User Guide pages in `docs/source/user_guide/` are executed examples.

- `data.md` for roles, arrays, groups, and scaling
- `distributions.md` for function forms, shapes, and support
- `priors.md` for `Prior`, `sample_prior`, and `check_prior`
- `changing.md` for variants, a free Hill slope, a Student-t likelihood, and
  Fourier seasonality with `coords`
- `functions.md` for custom functions, `custom_distribution`, and tracing
- `scenarios.md` for new data, `reference`, and `Data` variables and constants
- `sampling.md` for diagnostics, `continue_sampling`, `generate_quantities`,
  and `outcome_scaling` inside a block
- `inference.md` for external samplers
- `media_effects.md` and `budgets.md` for the analysis functions
- `../getting_started/installation.md` for precision and parallel chains

HSGP and ROI or contribution calibration have no User Guide page. The
docstring examples in `mmmjax/baselines/hsgp.py` and `mmmjax/media/calibration.py` cover them,
and `tests/baselines/test_hsgp.py` builds a full HSGP model.
