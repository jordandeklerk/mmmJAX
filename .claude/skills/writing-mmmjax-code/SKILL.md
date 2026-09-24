---
name: writing-mmmjax-code
description: Writes, changes, refactors, and reviews Python code and tests in mmmjax/ and tests/ to the project's standards for module layout, typing, signatures, input validation and error messages, frozen dataclasses and pytrees, public exports, numpydoc docstrings with ipython examples, comments, and test style. Use when the user asks to add or change a library function, class, module, distribution, or public export, write or fix tests, fix a lint or type failure, refactor or clean up code, review a diff, or asks about code style, conventions, standards, or best practices. JAX practice belongs to writing-jax-code, and a user's model blocks belong to writing-mmmjax-models.
---

# Writing mmmJAX code

The Code section of `.claude/CLAUDE.md` sets the rules, and this skill adds the specifics and reasons behind them, taken from the code. `pixi run lint` and `pixi run typecheck` already enforce formatting, imports, a sorted `__all__`, annotations, exception chaining, raw docstrings with backslashes, imperative summaries, and the `print` ban, so this skill leaves those out. `mmmjax/calibration.py` and `mmmjax/seasonality.py` each show most of what follows in one file.

## Before handing over

1. Run `pixi run lint` and `pixi run typecheck`.
2. Read the diff for what no tool checks, which is definition order, module-level constants, named returns, error message format, docstring types and Examples, and comment placement.
3. The avoid-ai-writing detector's hits on numpydoc `**name**` bullets, on the em dash in Returns field bullets, and on LaTeX blocks read as lists are false positives.

## Module layout

- RUF022 keeps `__all__` sorted isort-style, classes first and then functions alphabetically. The public API in `__all__` order therefore means a new public name goes at its sorted position in the file.
- Type aliases, StrEnums, and small private classes that signatures reference come right after `__all__`, because annotations evaluate at definition time and the package never adds `from __future__ import annotations`. Helpers never need to precede their callers, so they follow the public API.
- Private modules such as `_binding.py`, `_nuts.py`, `_results.py`, and `distributions/_normal.py` define no `__all__`. The one exception is `distributions/_distribution.py`, whose `__all__` lists the re-exported `custom_distribution`.
- Several older modules break this order. Do not copy their layout, and reorder one only as a pure move on a separate branch, never inside a feature diff.
- Module scope holds only `__all__`, `type` aliases, `__version__`, and the `_bind_distribution(...)` calls at the bottom of distribution modules. A missing value raises `LookupError` instead of returning a sentinel (`_lookup_input` in `_binding.py`), and a name used in several places stays a literal at each use, as `{"chain", "draw", "sample", "pred_id"}` does.
- The ban covers `mmmjax/` only. Tests keep small data constants such as `DISTRIBUTION_EXPORTS`.

## Types and signatures

- Take arrays as `ArrayLike` and dtypes as `DTypeLike` from `jax.typing`, return `jax.Array`, and type host arrays as `NDArray[np.float64]` or `NDArray[np.generic]`. jaxtyping is a declared dependency that no module imports, so do not start using it, and never write `jnp.ndarray`.
- Accept `Mapping` and `Sequence`, return `dict` and `tuple`, and return copies from accessors, as in `return dict(self._parameterizations)` (`Model.parameters`).
- Wrap untyped TFP and blackjax results in `cast(jax.Array, ...)` (`_normal_logpdf_kernel`). Keep `# type: ignore[code]` for untyped imports and calls, as on the `blackjax` import in `_nuts.py`.
- Public string options are `Literal[...]` in the signature and checked at runtime with `if effects not in ("lognormal", "normal")`. A reused option set gets a private alias, as in `type _Effects = Literal["lognormal", "normal"]` in `calibration.py`. Older modules that repeat an option literal are not precedent.
- Leading data or model arguments stay positional and every setting goes after `*`, as in `fourier_features(time, *, period, order)`. Distribution functions take their settings positionally with only `sample_shape` keyword-only. Learned parameters are positional in `adstock.py` and `saturation.py` and keyword-only in `hsgp_weights`, so follow the module when adding one.
- Analysis functions take `model` and `results` positionally and the rest after `*`. They reuse the shared keywords `quantity`, `group`, `spend_to_media`, `spend_to_rf`, `channels`, `new_data`, `spend_periods` or `periods`, `response_periods`, `by`, and `batch_size` in that relative order.
- Workflow functions take `seed: int`, and primitives take `key: jax.Array` as the first argument.
- Distribution arguments spell out the setting (`location`, `scale`, `probability`, `trials`, `concentration`) and map to the TFP names in the `_bind_distribution(...)` call.

## Validation and errors

Validate at the top of a public function, before any computation. Workflow functions check the model first with `if not isinstance(model, Model): raise TypeError("model must be a Model")`. Analysis functions get that check and their draws from `_response_inputs` or `_prepare_response` in `response.py` instead of repeating them.

- Raise `TypeError` for a wrong Python type or dtype and `ValueError` for a wrong value, range, or shape. `RuntimeError` is for a failure after valid input, such as nonfinite draws or a solver failure, and sampler diagnostics use `warnings.warn(..., RuntimeWarning, stacklevel=3)`. There are no custom exception classes.
- Some older checks fold a type error and a range error into one `ValueError`. Tests pin exception types with `pytest.raises`, so keep the existing type when editing an old check.
- `bool` is an `int`, so reject it with `isinstance(x, bool) or not isinstance(x, int)`. Workflow settings that may arrive as NumPy scalars use `numbers.Integral` or `numbers.Real` and exclude `(bool, np.bool_)` (`SpendConstraint.__post_init__`).
- Use `from None` only to replace an internal lookup failure with a user-facing message.
- `assert x is not None` narrows a type after an invariant holds and never validates input.

Messages have no trailing period. A message about an argument starts with its name exactly as the signature spells it, and any other message is in sentence case. A remedy follows as a second sentence after ". ", also without a final period. Report the bad input as `got {value!r}`, `got shape {x.shape}`, `got dtype {x.dtype}`, or `got {type(value).__name__}`, quote user-supplied names with `{name!r}`, and put option values in single quotes. Some older messages predate these rules and are not precedent.

```python
# media_response
raise ValueError(
    f"n_periods must be between 1 and the {media_shape[0]} supplied exposure periods, got {n_periods}. "
    "Use the number of modeling periods without counting earlier history"
)

# _as_scalar in model.py
raise ValueError(f"{name} must return a scalar, got shape {array.shape}")
```

## JAX

The `writing-jax-code` skill covers JAX practice, from where `jax.jit` goes to tracing, NaN-safe gradients, dtype promotion, and PRNG keys.

## Classes and pytrees

- Every dataclass is `frozen=True`, usually with `slots=True`, and adds `eq=False` when it holds arrays.
- A pytree is a frozen dataclass under `@jax.tree_util.register_dataclass`. Non-array fields carry `metadata={"static": True}` and must be hashable because they key the jit cache, which is why static mappings are `_FrozenMapping` and not `dict` (`_ModelData` in `_binding.py`).
- A public class that normalizes its inputs uses `init=False` with an `__init__` that validates, normalizes, and assigns through `object.__setattr__`, so equal specifications share jit cache keys (`Real.__init__` in `parameters.py`). A simpler class validates in `__post_init__`, as `SpendConstraint` does.
- Copy inputs at construction and return copies from accessors, so a caller's later edits never reach a model and results never alias their inputs.
- A closed set of internal names is a private StrEnum when code iterates over its members, as `_ResultGroup` in `model.py` and `_Family` in `contribution.py` do, and a private `Literal` alias otherwise, as `_Source` in `_binding.py` is. Dispatch on either with `match`. Small immutable records are NamedTuples, and the parameterization contract is a `runtime_checkable` Protocol.
- Library pieces are plain functions of arrays, not configurable component objects, and the user writes the model equation. A helper may size an approximation that encodes no belief, as `prepare_hsgp` sizes the HSGP domain.

## Naming

- Choose every name for how it reads to a user at the call site, from functions, classes, and parameters to result groups and variables. Use the plain word a modeler or marketer already knows, spell it out instead of abbreviating, and reuse the name the API already gives a concept (`quantity`, `channels`, `new_data`, `batch_size`) instead of a synonym.
- Where ArviZ, NumPyro, or Meridian already name the same thing, use their name, as `unconstrained_posterior` and `predictions` follow ArviZ's groups and `save_unconstrained` follows its `save_warmup`. A boolean names what it turns on, as `generate` does.
- When more than one public name reads well, put the options to the user instead of picking one silently. Private helpers and local variables name what they hold for the next reader, as `generation_keys` does.
- Underscore names are package-private, not module-private. Other mmmjax modules import them and read attributes such as `model._data`.
- Helpers are named by role, as `_validate_*` (returns None or raises), `_prepare_*`, `_resolve_*`, `_as_*` (converts and checks), and `_*_kernel` in distributions.
- Counts start with `n_` (`n_periods`, `n_basis`), never `num_`.

## Named values before return

The CLAUDE.md rule applies to code you write or edit. Most older returns still return expressions, so do not cite them as precedent. Pure delegation such as `return helper(self.values)` is fine, and anything with an operator or more than one call goes on its own line first.

```python
# Bad (media_response)
return jnp.asarray(response)[media_shape[0] - n_periods :]

# Good (contribution_coefficient)
total = jnp.sum(weighted, axis=0) if grouped else weighted
shared = target / total
return shared
```

## Adding a public name

1. Add it to the module's `__all__` and define it at its sorted position.
2. Import it in `mmmjax/__init__.py`, which imports without `as` aliases, and add it to that `__all__`.
3. For a distribution, also add `from mmmjax.distributions._x import name as name` and the `__all__` entry in `mmmjax/distributions/__init__.py`, and the sorted `DISTRIBUTION_EXPORTS` list in `tests/distributions/test_public_api.py`.
4. Add it to the autosummary list of the matching `docs/source/api/*.rst`, under `.. currentmodule:: mmmjax`.

A new distribution module follows `distributions/_normal.py`. It defines `<family>_logpdf` (or `_logpmf`) elementwise with `nan` for invalid settings, `<family>` as the `jnp.sum` of it, `_logcdf`, `_logsf`, and `_rng(key, ..., *, sample_shape=())`, then any alternate parameterization (`_logit`, `_log`) in the same order, the private kernels, and `_bind_distribution(...)` last. Inputs go through `_promote_inexact`, and `_random_shape` checks sample shapes.

## Docstrings

Every public docstring starts from a template in [docstrings.md](docstrings.md), which also holds the type spellings, default forms, and section rules. Read it before writing or editing a docstring.

## Comments

A comment sits on its own line above the code it explains, and the only trailing comment is `# type: ignore[code]`. There are no TODO or FIXME markers. The why is usually numerical stability, gradient behavior, ownership, or a JAX constraint, as in `# Convert counts directly to floating point to avoid narrowing them to int32` (`_broadcast_saturation_inputs` in `saturation.py`). Three lines is the most a real numerical subtlety gets. The secant comment in `optimize_budget` is an example.

## Tests

- `mmmjax/<module>.py` is tested in `tests/test_<module>.py` or a feature file such as `test_model_inputs.py`, and distributions in `tests/distributions/test_<name>.py`. That directory has no `__init__.py`, so a test file name must be unique across `tests/`. Fixtures live in the module that uses them and `conftest.py` holds none. Helpers stay private in the test module that uses them, as `_convolution_reference` does in `tests/test_adstock.py`, and `tests/helpers.py` holds only `importorskip`.
- A test module opens with `"""Tests for <subject>."""`. Tests are plain module-level functions without docstrings or classes, named as behavior sentences such as `test_normal_logpdf_rejects_invalid_parameters_without_repairing_them`. Validation tests say `rejects` or `requires` rather than `raises`, and export checks are `test_<name>_is_exported`.
- Import the API under test from the package root. Import a submodule only to assert re-export identity or to reach a private helper.
- Lay a test out as inputs and `expected`, a blank line, `result = f(...)`, a blank line, and the asserts (`test_normal_logpdf_matches_known_values`).
- Compute `expected` independently with SciPy, TFP, `jax.scipy.stats`, a private NumPy reference such as `_convolution_reference`, or exact literals.
- Compare with `np.testing.assert_allclose(result, expected, rtol=..., atol=...)` and explicit tolerances, commonly `rtol=2e-6` or `3e-6` in float32 with `atol=0`. Use `assert_array_equal` for exact values and `xr.testing.assert_identical` for labeled output, and assert `shape` and `dtype` as well. Distribution tests mostly write `assert jnp.allclose(...)`, so follow the file there.
- Check a numeric function eagerly and under `jax.jit` against the same `expected`. Where it applies, also check `jax.vmap` against a Python loop and gradients against a closed form or float64 finite differences.

  ```python
  # test_geometric_adstock_matches_numpy_convolution
  eager = function(media, retention)
  compiled = jax.jit(function)(media, retention)

  tolerance = 2e-6 if dtype == jnp.float32 else 2e-14
  for result in (eager, compiled):
      assert result.shape == media.shape
      assert result.dtype == media.dtype
      np.testing.assert_allclose(result, expected, rtol=tolerance, atol=1e-15)
  ```

- Cover both precisions. Parametrize `dtype` over `[jnp.float32, jnp.float64]` and call `pytest.skip("JAX 64-bit mode is disabled")` for float64 when x64 is off, or mark float64-only tests with `@pytest.mark.skipif(not jax.enable_x64.value, reason="JAX 64-bit mode is disabled")`. `with jax.enable_x64(...)` switches the mode inside one test.
- Test every validation error with `pytest.raises(ErrorType, match=...)` matched on a distinctive part of the message, often the argument name, and group several as an `"option, error, message"` table (`test_data_block_validates_declarations`).
- Stack parametrize decorators for products, use `pytest.param(..., id=...)` for values without a readable repr, and use a fixture with `params=` for backend matrices such as the dataframe libraries (the `frame_factory` fixture in `tests/test_data.py`).
- Keep real NUTS out of tests. The `nuts_calls` fixture in `tests/test_sampling.py` stubs `sampling._sample_nuts`, `_collect_results` builds result trees directly, and a stub that must never run raises `AssertionError`. An unavoidable real run stays at `draws=2, warmup=3, chains=1`.
- A test that needs several devices runs a subprocess with `JAX_NUM_CPU_DEVICES` and `JAX_PLATFORMS=cpu`, because JAX fixes the device count at startup (`test_parallel_nuts_on_two_cpu_devices_preserves_targets_generation_and_labels`). CI splits tests across pytest-xdist workers, so no test may depend on another.
- Build data inline as small polars frames or with `simulate_data(seed=...)`. There are no data files and no `tmp_path`.
- A new distribution's tests follow `tests/distributions/test_normal.py`. They cover known values, the scalar sum, broadcasting, `nan` for invalid settings, support endpoints, `logcdf` and `logsf` as complements, extreme tails, gradients, `vmap`, and an `rng` that matches transformed `jax.random` draws from the same key.
- A new scalar family also joins each family list in `tests/distributions/test_contracts.py`. Those lists run every scalar family through empty batches, the float32 floor, float64, `jax.jit`, `nan` draws for invalid settings, and key `vmap`. The family also gets a case in `benchmarks/cases.py` and a JAX reference in `benchmarks/references.py`, and `tests/distributions/test_references.py` checks both against the public API.
