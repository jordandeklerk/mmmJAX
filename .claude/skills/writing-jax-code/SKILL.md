---
name: writing-jax-code
description: Gives the JAX 0.10 practices for library and test code in mmmjax/ and tests/. It covers where jit goes, tracing and tracer or ConcretizationTypeError errors, static arguments, recompilation and closed-over constants, grad, vmap, scan, fori_loop and cond, NaN gradients, custom_jvp rules, dtype promotion and float64, PRNG keys, pytrees, devices and parallel chains, host transfer, debugging, performance, and deprecated or removed JAX APIs. Use when writing, reviewing, or debugging code that calls JAX, when recalling a JAX API from memory, when a function recompiles, runs slowly, or returns NaN gradients, or before changing the JAX pin. User model blocks belong to writing-mmmjax-models, and module layout, signatures, error messages, docstrings, and general test style belong to writing-mmmjax-code.
---

# Writing JAX code in mmmJAX

## Confirm the API in the installed JAX

`pyproject.toml` pins `jax>=0.10.0,<0.11`, and the default pixi env runs JAX 0.10.2 with blackjax 1.6.2 and a pinned tfp-nightly. An API added after 0.10.0 breaks users at the floor. `jax.ShapeDtypeStruct.like` arrived in 0.10.2, for example, so build a hollow argument with `jax.ShapeDtypeStruct(x.shape, x.dtype)` until the pin rises. Many JAX names changed or vanished between 0.4 and 0.10, so confirm any API you are unsure of in the installed source or by running it, never from memory.

```bash
PY=.pixi/envs/default/bin/python
JAX=.pixi/envs/default/lib/python3.14/site-packages/jax
$PY -c "import jax; print(jax.__version__)"
$PY -c "import inspect, jax; print(inspect.signature(jax.random.wrap_key_data))"
$PY -c "import jax.numpy as jnp; jnp.fix"               # AttributeError means it is gone
grep -rln "_deprecations = {" $JAX --include='*.py' | grep -v /_src/   # every public module with a deprecation table
$PY -c "import jax; print(sorted(jax.config.values))"   # every config flag
grep -rn "name=.jax_num_cpu_devices" $JAX/_src           # where one flag is defined, with its default
```

- A deprecation entry whose replacement is `None` raises `AttributeError` at once, even when its message says "will be removed". Run the name rather than trusting the message.
- docs.jax.dev/en/latest and the changelog run ahead of 0.10.2. The latest page on constants describes hoisting closed-over arrays, which 0.10.2 does only when `jax_use_simplified_jaxpr_constants` is on, and it is off by default.
- `pixi run typecheck` covers `mmmjax/` only, so a spelling that mypy and ty reject can hide in `tests/`. Check a doubtful one with `.pixi/envs/check/bin/mypy probe.py`.

## Old habits to replace

Each line gives the old form and its replacement, checked against 0.10.2. Removed names raise at once, and the rest still run, some with a deprecation warning and some silently. Write the new form in new code, and convert existing old forms only in a change of their own.

- `jax.random.PRNGKey(seed)` still runs silently but returns a raw `uint32[2]` key, so write `jax.random.key(seed)`. Annotate keys as `jax.Array`, since `jax.random.KeyArray` and `PRNGKeyArray` are gone. `np.asarray(key)` raises, so store with `jax.random.key_data` and restore with `jax.random.wrap_key_data`.
- `jax.tree_map`, `jax.tree_leaves`, `jax.tree_flatten` → `jax.tree.map`, `jax.tree.leaves`, `jax.tree.flatten`. `jax.tree_util.tree_*` still works as the legacy spelling, so write `jax.tree.*` in new code and keep `jax.tree_util` for `register_dataclass`.
- `jax.experimental.enable_x64` → `jax.enable_x64(True)`. Reads through `jax.config.x64_enabled`, `jax.config.jax_enable_x64`, or `jax.config.values["jax_enable_x64"]` → `jax.enable_x64.value`, which mypy and ty accept.
- `functools.partial(jax.jit, static_argnames=...)` → `@jax.jit(static_argnames=...)`. Options are keyword-only, so `jax.jit(f, (0,))` raises, and `jax.jit(..., backend=..., device=...)` is deprecated in favor of `jax.device_put` on the inputs.
- `jnp.clip(x, a_min=..., a_max=...)` → `jnp.clip(x, min=..., max=...)`.
- `jnp.array(x, dtype, False)` → `jnp.array(x, dtype=dtype, copy=False)`, since positional `copy`, `order`, and `ndmin` are deprecated.
- `dtype=array` and `x.astype(array)` → `dtype=array.dtype`.
- `jnp.round_`, `fix`, `product`, `cumproduct`, `sometrue`, `alltrue`, `in1d`, `trapz`, `row_stack`, `NINF`, `NaN`, `Inf` → `jnp.round`, `trunc`, `prod`, `cumprod`, `any`, `all`, `isin`, `trapezoid`, `vstack`, `-jnp.inf`, `jnp.nan`, `jnp.inf`. `jnp.reshape(newshape=)` → `shape=`, and `jnp.sort(kind=)` → `stable=True`.
- `jnp.shape`, `jnp.ndim`, or `jnp.sum` on a Python list warn or raise → `np.shape(value)` on the host, or `jnp.asarray(value)` first.
- A generator, `zip` object, or `dict.values()` as a pytree leaf warns since 0.10.1 → `tuple(...)` first. Primitives that read input dtypes through tree leaves see it too (`_prepare_adstock` in `media/adstock.py`). `jax.tree.map(f, None, x)` raises → pass `is_leaf=lambda x: x is None`.
- `jax.core.get_aval` → `jax.typeof`. `jax.core.ConcretizationTypeError` → `jax.errors.ConcretizationTypeError`. `jax.xla_computation` → `jax.jit(f).trace(*args)` or `.lower(*args)`. `jax.interpreters.xla.canonicalize_dtype` → `jax.dtypes.canonicalize_dtype`.
- `jax.experimental.shard_map.shard_map(check_rep=...)` → `jax.shard_map(check_vma=...)`. `pjit` → `jax.jit`, `with mesh:` → `with jax.set_mesh(mesh):`, and `jax.device_put_replicated` or `device_put_sharded` → `jax.device_put` with a `NamedSharding`. `jax.pmap` is in maintenance mode.
- `jax.experimental.host_callback` → `jax.debug.callback`, or `jax.pure_callback(..., vmap_method=...)`, which raises under `vmap` when `vmap_method` is missing.
- `x.device()` → `x.device` or `x.devices()`. CPU devices are now named `cpu:0`, so never match device name strings.
- Names under `jax.core`, `jax.interpreters`, and `jax.lib` are deprecated → the public modules. mmmjax imports none of them.

## Where jit goes and what it caches

- Never decorate a public function with `@jax.jit` or give it `static_argnums`. Users compose the primitives inside their own `jit`, `vmap`, and `grad`, and only drivers that own a whole computation call `jax.jit` (`inference/sampling.py`, `inference/_nuts.py`, and the three modules in `analysis/`).
- Pass data and posterior draws to a jitted function as arguments. JAX 0.10.2 inlines every closed-over array into the program as a constant, so a captured 4000 by 500 array adds 16 MB of HLO and several times the compile time. `Model.log_density(position, data)` takes its data as a pytree, and one jitted log density serves new array data of the same shape (`test_compiled_log_density_accepts_new_data` in `tests/model/test_model.py`).

  ```python
  # Bad, since it captures inputs as constants
  def logdensity(position):
      return model.log_density(position, inputs)

  values, gradients = jax.jit(jax.vmap(jax.value_and_grad(logdensity)))(positions)

  # Good
  evaluate = jax.jit(jax.vmap(jax.value_and_grad(model.log_density), in_axes=(0, None)))
  values, gradients = evaluate(positions, inputs)
  ```

  This is safe for a prepared model, whose `_ModelData` keeps ints and constants static. An unprepared `data` pytree may hold Python ints that a block uses as shapes or slice bounds, and each becomes a traced `int32` when passed as an argument, so keep those leaves static or `mj.sample(model, data={..., "n": 3})` stops working. blackjax wants a one-argument `logdensity`, so build that closure inside the traced function from a `data` argument.
- Within one driver call, build each jitted callable once and reuse it across chunks and batches, as `_sample_nuts` in `inference/_nuts.py` does. The cache keys on function identity, so `jax.jit(lambda ...)`, `jax.jit(jax.vmap(f))`, or `jax.jit(model.log_density)` built again traces again, since a bound method is a new object on every access. The Good example above fixes constant capture but still compiles on every driver call. Caching across calls needs a home that the module-scope rule allows and a key that includes `progress`, because blackjax's progress bar patches `scan` at trace time, so propose it before adding one.
- Keep argument shapes fixed across calls, because each new shape or dtype compiles again, and a short last chunk or batch is a new shape. In a stateless map such as `_evaluate_draws` in `inference/sampling.py`, pad the last batch to full size and keep its valid rows, or run `jax.lax.map(f, xs, batch_size=n)` inside one jit, which handles the remainder in the same program (`_sample_response` in `analysis/response.py`). Never pad a stateful scan such as the NUTS chunk in `inference/_nuts.py`, since padded steps would carry the sampler state past the last kept draw that `continue_sampling` resumes from. Choose a `chunk_size` that divides `draws` there, or mask the extra steps with `jnp.where(valid, new_state, state)`.
- Shape-setting arguments (`order`, `n_basis`, `max_lag`, `n_periods`, `sample_shape`) are Python values. Reject anything else with a `TypeError` that says the value must stay fixed under JIT compilation, and say in its Parameters entry to keep it static. Tests jit such functions with `static_argnames` (the static `order` test in `tests/baselines/test_seasonality.py`).
- `_ModelData.owner` in `model/_binding.py` is a static `object()` token that ties prepared data to its model (`_PreparedBlocks.validated_data` in `model/model.py`). Because it is static and compared by identity, each model's data also traces separately.
- Library code never calls `jax.config.update` and never sets the persistent compilation cache. Those belong to the application.

## Tracing

- Raise only on facts known at trace time, which are Python types, static ints, shapes, and dtypes. An invalid numeric value becomes `nan` through `jnp.where`, as the gradient section shows.
- A value check that should run whenever values are concrete goes through `_concrete_array` in `model/_binding.py`. It catches `jax.errors.ConcretizationTypeError`, the base of `TracerBoolConversionError`, and `jax.errors.TracerArrayConversionError`, and skips the check under tracing.
- Never detect tracing by type. A traced array passes `isinstance(value, jax.Array)` through the metaclass JAX 0.8.2 added, even though `jax.core.Tracer` no longer subclasses `jax.Array` at runtime. Use `isinstance(value, jax.Array)` only to separate arrays from host objects (`_transformed_array` in `model/_binding.py`).
- Host preparation (`prepare_data`, `prepare_hsgp`, `fit_*_scaling`, `Model` construction) runs outside JAX transformations with NumPy, checks values with `np.isfinite(...).all()`, and ends in one `jax.device_put` (`PreparedData._to_jax` in `data/prepare.py`). Anything that may be traced uses `jnp`.
- Convert user sequences once at the boundary, since `ArrayLike` excludes lists and a list under `jit` becomes one input per element.
- Outside tracing, `jax.lax.stop_gradient(1.0)` returns a JAX literal type that is not a `jax.Array`, so pass it arrays or wrap the result in `jnp.asarray`.

## Loops and control flow

- `jax.lax.while_loop` has no reverse mode. Use it only in a value path whose derivatives come from a custom JVP (`_beta_fraction` in `distributions/_beta_cdf.py`, `_binomial_tail_sum` in `distributions/_binomial.py`).
- Under `vmap` with a batched predicate, `lax.cond` becomes `select_n` and runs both branches. The fast-path conds in `_gamma.py`, `_beta.py`, `_binomial.py`, `_beta_cdf.py`, `_dirichlet.py`, and `_uniform.py` save nothing when chains are vectorized or draws are vmapped, so both branches must be safe on every input.

## Gradients and NaN safety

- Substitute a safe value before any `log`, `sqrt`, division, or power, then mask the output. The discarded branch of an outer `jnp.where` still sends `nan` into reverse-mode gradients.

  ```python
  # Good (hill_saturation, shortened)
  safe_media = jnp.where(valid, media_array, 1.0)
  safe_half = jnp.where(valid, half_array, 1.0)
  ...
  return jnp.where(valid, response, jnp.nan)

  # Bad, since a group that never sees a channel gets a nan gradient
  log_total = logsumexp(offsets + jnp.log(weighted), axis=0)
  ```

- `xlogy` fixes the value of `0 * log(0)` but not its gradient, so substitute safe values instead (`poisson_logpmf` in `distributions/_poisson.py`).
- Apply `jax.lax.stop_gradient` to a factor that cancels (`hill_saturation`). To take the value from a stable path and the derivative from the analytic one, write `raw + jax.lax.stop_gradient(stable - raw)` (`_stable_log_ratio` in `distributions/_utils.py`).
- In traced code, derive guards and tolerances from the working dtype, as in `32 * np.finfo(value.dtype).eps`, or pick them per dtype width as `_is_valid_simplex` in `distributions/_utils.py` does. Host-side SciPy code runs in float64 and may use fixed tolerances, as `analysis/optimization.py` does.

## Dtypes and precision

- The library assumes one precision per process, set at startup with `JAX_ENABLE_X64` or `jax.config.update`, and `with jax.enable_x64(...)` scopes it, which only tests should do. Precision errors tell the user to enable it (`PreparedData._to_jax` in `data/prepare.py`). Objects record the precision in effect when they are built (`Model.__init__`), and `continue_sampling` checks the stored one.
- `bool(jax.enable_x64)` raises `TypeError`. The flag is part of the jit cache key, so toggling it retraces.
- Promote numeric inputs to one floating dtype of at least float32 that follows the x64 setting, and convert integer counts straight to it so they never narrow to int32.

  ```python
  # fourier_features in baselines/seasonality.py
  dtype = jnp.result_type(*leaves)
  if not jnp.issubdtype(dtype, jnp.floating):
      dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
  dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
  ```

  Each module keeps its own copy of this helper, and the copies differ on bool inputs, so match the one in the module you edit.
- Python scalars are weakly typed and keep an array's dtype, while NumPy scalars and arrays are strong and turn float32 math into float64 under x64. Combine traced arrays with Python floats or with constants cast to `x.dtype` (`_log_ratio_deviance_series` in `distributions/_utils.py`). A Python float and a NumPy scalar in the same argument slot also compile separately.
- With x64 off, an explicit float64 request becomes float32 with only a `UserWarning`, a NumPy int64 array wraps silently (`np.array([2**40])` becomes 0), and an oversized Python int raises `OverflowError`. Check integer ranges on the host before converting (`PreparedData._to_jax` in `data/prepare.py`, `_transformed_array` in `model/_binding.py`).

## Pytrees

- `writing-mmmjax-code` covers registered dataclasses and their static fields, which the package writes as `field(..., metadata={"static": True})`. A static field never holds an array, and `Data.__init__` in `data/prepare.py` checks that its constants hash.
- Keep validation and array conversion out of a registered class's `__init__` and `__post_init__`, because transformations rebuild instances with placeholder leaves. Validate in a factory, as `fit_scaling` does for `Scaling`.

## Pin changes, custom rules, PRNG keys, devices, debugging, and testing

Read [reference.md](reference.md) before changing the JAX pin, writing or changing a custom JVP rule, creating, splitting, folding, or storing PRNG keys, touching the device mesh or host transfer in `inference/_nuts.py` or `inference/sampling.py`, chasing a recompile, a captured constant, or a `nan`, or testing JAX behavior such as jit agreement, gradients, or retracing.
