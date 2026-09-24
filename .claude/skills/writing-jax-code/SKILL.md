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
- A generator, `zip` object, or `dict.values()` as a pytree leaf warns since 0.10.1 → `tuple(...)` first. Primitives that read input dtypes through tree leaves see it too (`_prepare_adstock` in `adstock.py`). `jax.tree.map(f, None, x)` raises → pass `is_leaf=lambda x: x is None`.
- `jax.core.get_aval` → `jax.typeof`. `jax.core.ConcretizationTypeError` → `jax.errors.ConcretizationTypeError`. `jax.xla_computation` → `jax.jit(f).trace(*args)` or `.lower(*args)`. `jax.interpreters.xla.canonicalize_dtype` → `jax.dtypes.canonicalize_dtype`.
- `jax.experimental.shard_map.shard_map(check_rep=...)` → `jax.shard_map(check_vma=...)`. `pjit` → `jax.jit`, `with mesh:` → `with jax.set_mesh(mesh):`, and `jax.device_put_replicated` or `device_put_sharded` → `jax.device_put` with a `NamedSharding`. `jax.pmap` is in maintenance mode.
- `jax.experimental.host_callback` → `jax.debug.callback`, or `jax.pure_callback(..., vmap_method=...)`, which raises under `vmap` when `vmap_method` is missing.
- `x.device()` → `x.device` or `x.devices()`. CPU devices are now named `cpu:0`, so never match device name strings.
- Names under `jax.core`, `jax.interpreters`, and `jax.lib` are deprecated → the public modules. mmmjax imports none of them.

Write code that also holds on 0.11. There `jnp.empty` returns uninitialized memory, so write `jnp.zeros` where zeros matter. `jnp.broadcast_arrays` and `jnp.meshgrid` return tuples, and `jnp.take_along_axis` wraps negative indices by default in every mode, including `promise_in_bounds`. Moving to 0.11 also means raising the `numpy` and `scipy` floors in `pyproject.toml`, since 0.11.0 dropped NumPy 2.0 and SciPy 1.14.

## Where jit goes and what it caches

- Never decorate a public function with `@jax.jit` or give it `static_argnums`. Users compose the primitives inside their own `jit`, `vmap`, and `grad`, and only drivers that own a whole computation call `jax.jit` (`sampling.py`, `_nuts.py`, `response.py`, `optimization.py`, `contribution.py`).
- Pass data and posterior draws to a jitted function as arguments. JAX 0.10.2 inlines every closed-over array into the program as a constant, so a captured 4000 by 500 array adds 16 MB of HLO and several times the compile time. `Model.log_density(position, data)` takes its data as a pytree, and one jitted log density serves new array data of the same shape (`test_compiled_log_density_accepts_new_data` in `tests/test_model.py`).

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
- Within one driver call, build each jitted callable once and reuse it across chunks and batches, as `_sample_nuts` in `_nuts.py` does. The cache keys on function identity, so `jax.jit(lambda ...)`, `jax.jit(jax.vmap(f))`, or `jax.jit(model.log_density)` built again traces again, since a bound method is a new object on every access. The Good example above fixes constant capture but still compiles on every driver call. Caching across calls needs a home that the module-scope rule allows and a key that includes `progress`, because blackjax's progress bar patches `scan` at trace time, so propose it before adding one.
- Keep argument shapes fixed across calls, because each new shape or dtype compiles again, and a short last chunk or batch is a new shape. In a stateless map such as `_evaluate_draws` in `sampling.py`, pad the last batch to full size and keep its valid rows, or run `jax.lax.map(f, xs, batch_size=n)` inside one jit, which handles the remainder in the same program (`_sample_response` in `response.py`). Never pad a stateful scan such as the NUTS chunk in `_nuts.py`, since padded steps would carry the sampler state past the last kept draw that `continue_sampling` resumes from. Choose a `chunk_size` that divides `draws` there, or mask the extra steps with `jnp.where(valid, new_state, state)`.
- Shape-setting arguments (`order`, `n_basis`, `max_lag`, `n_periods`, `sample_shape`) are Python values. Reject anything else with a `TypeError` that says the value must stay fixed under JIT compilation, and say in its Parameters entry to keep it static. Tests jit such functions with `static_argnames` (the static `order` test in `tests/test_seasonality.py`).
- `_ModelData.owner` in `_binding.py` is a static `object()` token that ties prepared data to its model (`_PreparedBlocks.validated_data` in `model.py`). Because it is static and compared by identity, each model's data also traces separately.
- Library code never calls `jax.config.update` and never sets the persistent compilation cache. Those belong to the application.

## Tracing

- Raise only on facts known at trace time, which are Python types, static ints, shapes, and dtypes. An invalid numeric value becomes `nan` through `jnp.where`, as the gradient section shows.
- A value check that should run whenever values are concrete goes through `_concrete_array` in `_binding.py`. It catches `jax.errors.ConcretizationTypeError`, the base of `TracerBoolConversionError`, and `jax.errors.TracerArrayConversionError`, and skips the check under tracing.
- Never detect tracing by type. A traced array passes `isinstance(value, jax.Array)` through the metaclass JAX 0.8.2 added, even though `jax.core.Tracer` no longer subclasses `jax.Array` at runtime. Use `isinstance(value, jax.Array)` only to separate arrays from host objects (`_transformed_array` in `_binding.py`).
- Host preparation (`prepare_data`, `prepare_hsgp`, `fit_*_scaling`, `Model` construction) runs outside JAX transformations with NumPy, checks values with `np.isfinite(...).all()`, and ends in one `jax.device_put` (`PreparedData._to_jax` in `data.py`). Anything that may be traced uses `jnp`.
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
- In traced code, derive guards and tolerances from the working dtype, as in `32 * np.finfo(value.dtype).eps`, or pick them per dtype width as `_is_valid_simplex` in `distributions/_utils.py` does. Host-side SciPy code runs in float64 and may use fixed tolerances, as `optimization.py` does.

## Custom derivatives

- Use `jax.custom_jvp`, never `custom_vjp`. A JVP rule serves forward and reverse mode, while a `custom_vjp` function cannot be forward-differentiated. Give each rule a why-comment.
- Compute the primal inside the rule by calling the decorated function, so the rule still applies at every order of differentiation.

  ```python
  # _standardize in distributions/_normal.py, shortened
  @_standardize.defjvp
  def _standardize_jvp(primals, tangents):
      value, location, scale = primals
      value_tangent, location_tangent, scale_tangent = tangents
      standardized = _standardize(value, location, scale)
      ...
      return standardized, standardized_tangent
  ```

- Use `nondiff_argnums` or `nondiff_argnames` only for Python values such as a bool, which then come first in the rule's signature (`_log_betainc` in `distributions/_beta_cdf.py`). Integer arrays stay ordinary arguments.

## Dtypes and precision

- The library assumes one precision per process, set at startup with `JAX_ENABLE_X64` or `jax.config.update`, and `with jax.enable_x64(...)` scopes it, which only tests should do. Precision errors tell the user to enable it (`PreparedData._to_jax` in `data.py`). Objects record the precision in effect when they are built (`Model.__init__`), and `continue_sampling` checks the stored one.
- `bool(jax.enable_x64)` raises `TypeError`. The flag is part of the jit cache key, so toggling it retraces.
- Promote numeric inputs to one floating dtype of at least float32 that follows the x64 setting, and convert integer counts straight to it so they never narrow to int32.

  ```python
  # fourier_features in seasonality.py
  dtype = jnp.result_type(*leaves)
  if not jnp.issubdtype(dtype, jnp.floating):
      dtype = jnp.float64 if jax.dtypes.itemsize_bits(dtype) == 64 else jnp.float32
  dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(dtype, jnp.float32))
  ```

  Each module keeps its own copy of this helper, and the copies differ on bool inputs, so match the one in the module you edit.
- Python scalars are weakly typed and keep an array's dtype, while NumPy scalars and arrays are strong and turn float32 math into float64 under x64. Combine traced arrays with Python floats or with constants cast to `x.dtype` (`_log_ratio_deviance_series` in `distributions/_utils.py`). A Python float and a NumPy scalar in the same argument slot also compile separately.
- With x64 off, an explicit float64 request becomes float32 with only a `UserWarning`, a NumPy int64 array wraps silently (`np.array([2**40])` becomes 0), and an oversized Python int raises `OverflowError`. Check integer ranges on the host before converting (`PreparedData._to_jax` in `data.py`, `_transformed_array` in `_binding.py`).

## PRNG keys

- Workflow functions call `jax.random.key(int(seed))` and split once into one named key per consumer (`sample` in `sampling.py`).
- Where draws must match across chunking and `continue_sampling`, derive each key with `jax.random.fold_in` on the absolute draw index as `uint32` (`draw_keys` in `_nuts.py`, `_generation_keys` in `sampling.py`). `jax.random.split(key, (chains, draws))` makes every chain after the first depend on `draws`. A 1-D split keeps its prefix under the default `jax_threefry_partitionable`, but that rests on a config flag, so use `fold_in` wherever stability matters.
- Never both split a key and fold indices into that same key. Under the default `jax_threefry_partitionable`, `fold_in(key, i)` equals `split(key, n)[i]`, and key-reuse checking misses the overlap, so split once and fold indices only into the child keys.
- A typed `split` returns shape `(n,)` and accepts a shape tuple (`generate_quantities` in `sampling.py`).
- Accept seeds in `[0, 2**32)`. With x64 off, `jax.random.key` drops bits above 32, so `2**32` and `np.int64(2**40)` both give the stream of 0. `2**63` raises `OverflowError` in either mode.
- Store a key's implementation with its data, as `str(jax.random.key_impl(key))`, and pass it as `impl=` to `wrap_key_data`. Key data alone restores under whatever `JAX_DEFAULT_PRNG_IMPL` is set, which raises for an impl with a different key shape and silently switches streams under `philox4x32`, whose key data shares threefry's `(2,)` shape.
- Only the output distribution of `jax.random` is stable across JAX versions, so never hardcode draws, and treat a stored sampling state as reproducible within one JAX version.
- Host simulation uses `np.random.default_rng` with `SeedSequence.spawn` (`simulate_data`).

## Pytrees

- `writing-mmmjax-code` covers registered dataclasses and their static fields, which the package writes as `field(..., metadata={"static": True})`. A static field never holds an array, and `Data.__init__` in `data.py` checks that its constants hash.
- Keep validation and array conversion out of a registered class's `__init__` and `__post_init__`, because transformations rebuild instances with placeholder leaves. Validate in a factory, as `fit_scaling` does for `Scaling`.

## Devices and host transfer

- Parallel chains follow the mesh setup in `_sample_nuts` (`_nuts.py`). Build the mesh with `jax.make_mesh((chains,), ("chain",), axis_types=(AxisType.Auto,), devices=jax.local_devices()[:chains])`, because `make_mesh` has defaulted to `Explicit` axes since 0.9. Enter it with `jax.set_mesh`, place inputs with `NamedSharding` and `jax.device_put`, and wrap `jax.jit(jax.shard_map(...))`. `check_vma=False` there is safe only because every `out_specs` entry is sharded on `"chain"`, and a replicated spec would fail silently. `jax.P` is the short alias for `PartitionSpec`. mypy treats both constructors and `jax.effects_barrier()` as untyped calls, so they carry `# type: ignore[no-untyped-call]` as in `_nuts.py`.
- Check the device count before any work (`sample` in `sampling.py`). JAX fixes the CPU device count when its backend starts, so set `JAX_NUM_CPU_DEVICES` before importing JAX.
- Stream long results to the host in bounded chunks with `jax.device_get` into preallocated NumPy buffers, and `del` the device chunk before the next one (`transfer` in `_sample_nuts`, `_evaluate_draws` in `sampling.py`). `np.asarray(x)` and `jax.device_get(x)` both return read-only arrays, so copy with `np.array(x, copy=True)` before writing into one (`optimize_budget` in `optimization.py`).
- blackjax's progress bar runs `jax.debug.callback` inside `scan`, so `_progress` in `_nuts.py` calls `jax.effects_barrier()` in its `finally`.

## Debugging and dependencies

- Find recompiles with `with jax.log_compiles(True):` or `jax.explain_cache_misses(True)`. Find captured constants with `JAX_CAPTURED_CONSTANTS_WARN_BYTES=1000` and `JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=40`, since the `-1` that JAX's own warning suggests fails in 0.10.2.
- Inspect a staged program with `jax.jit(f).trace(*args).jaxpr` or `jax.jit(f).lower(*args).as_text()`. Lowered objects work only in the process that made them.
- `jax.debug_nans(True)` also stops on the `nan` that mmmjax returns on purpose for invalid parameters. Chase a non-finite log density with `jax.debug.print` or `jax.experimental.checkify` instead.
- The TFP JAX substrate reads the deprecated `jax.core.pytype_aval_mappings` on import, and the gamma, beta, student-t, inverse-gamma, Dirichlet, uniform, binomial, negative binomial, multinomial, and LKJ `_rng` functions, including their logit and log variants, warn that a TFP `jnp.shape(None)` call will become an error. pytest sets no `filterwarnings`, so these never fail a test and show up only in the warnings summary. Before widening the JAX pin, surface them against the pinned tfp-nightly.

  ```bash
  .pixi/envs/default/bin/python -W default::DeprecationWarning -c "import jax, mmmjax as mj; mj.gamma_rng(jax.random.key(0), 2.0, 1.0)"
  ```

## Testing JAX behavior

`writing-mmmjax-code` covers test layout, eager and `jax.jit` checks, both precisions, and multi-device subprocesses. These points add to it.

- Compare jitted and eager floating results with a tolerance, since XLA may fuse or reorder operations. Exact equality stays right for integer or boolean outputs such as discrete draws, and for one key through one code path.
- For a new or changed custom JVP, check the kernel against a closed form or against float64 finite differences, skipped when x64 is off, at interior, tail, and branch points. Check linearity in the tangents as `test_normal_log_probability_jvp_is_linear` does. `jax.test_util` needs its own import, and the float32 defaults are too loose.

  ```python
  from jax.test_util import check_grads

  check_grads(kernel, (alpha, beta, value), order=1, modes=("fwd", "rev"))
  ```

- Guard against retracing by warming a jitted function and calling it again under `with jax.no_tracing(True):`. Create every input before the context and check outputs after it, since any eager `jnp` call that has not yet run with that shape and dtype traces inside it, even `jnp.ones` or `x + 1`.
- Scope checks to one test with `jax.debug_key_reuse(True)`, `jax.numpy_dtype_promotion("strict")`, and `jax.check_tracer_leaks()` (`test_model_density_and_gradient_do_not_retain_tracers` in `tests/test_model.py`). Under key-reuse checking, `split` consumes its key and `fold_in` does not.
