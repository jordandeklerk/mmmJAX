# JAX reference for mmmJAX

These sections cover changing the JAX pin, custom derivatives, devices and host transfer, debugging, and testing JAX behavior.

## Changing the JAX pin

Write code that also holds on 0.11. There `jnp.empty` returns uninitialized memory, so write `jnp.zeros` where zeros matter. `jnp.broadcast_arrays` and `jnp.meshgrid` return tuples, and `jnp.take_along_axis` wraps negative indices by default in every mode, including `promise_in_bounds`. Moving to 0.11 also means raising the `numpy` and `scipy` floors in `pyproject.toml`, since 0.11.0 dropped NumPy 2.0 and SciPy 1.14.

- The pinned tfp-nightly raises JAX deprecation warnings on import and in several `_rng` functions, and pytest sets no `filterwarnings`, so they appear only in the warnings summary. Surface them before widening the JAX pin.

  ```bash
  .pixi/envs/default/bin/python -W default::DeprecationWarning -c "import jax, mmmjax as mj; mj.gamma_rng(jax.random.key(0), 2.0, 1.0)"
  ```

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

## Devices and host transfer

- Parallel chains follow the mesh setup in `_sample_nuts` (`_nuts.py`). Build the mesh with `jax.make_mesh((chains,), ("chain",), axis_types=(AxisType.Auto,), devices=jax.local_devices()[:chains])`, because `make_mesh` has defaulted to `Explicit` axes since 0.9. Enter it with `jax.set_mesh`, place inputs with `NamedSharding` and `jax.device_put`, and wrap `jax.jit(jax.shard_map(...))`. `check_vma=False` there is safe only because every `out_specs` entry is sharded on `"chain"`, and a replicated spec would fail silently. `jax.P` is the short alias for `PartitionSpec`. mypy treats both constructors and `jax.effects_barrier()` as untyped calls, so they carry `# type: ignore[no-untyped-call]` as in `_nuts.py`.
- Check the device count before any work (`sample` in `sampling.py`). JAX fixes the CPU device count when its backend starts, so set `JAX_NUM_CPU_DEVICES` before importing JAX.
- Stream long results to the host in bounded chunks with `jax.device_get` into preallocated NumPy buffers, and `del` the device chunk before the next one (`transfer` in `_sample_nuts`, `_evaluate_draws` in `sampling.py`). `np.asarray(x)` and `jax.device_get(x)` both return read-only arrays, so copy with `np.array(x, copy=True)` before writing into one (`optimize_budget` in `optimization.py`).
- blackjax's progress bar runs `jax.debug.callback` inside `scan`, so `_progress` in `_nuts.py` calls `jax.effects_barrier()` in its `finally`.

## Debugging and dependencies

- Find recompiles with `with jax.log_compiles(True):` or `jax.explain_cache_misses(True)`. Find captured constants with `JAX_CAPTURED_CONSTANTS_WARN_BYTES=1000` and `JAX_CAPTURED_CONSTANTS_REPORT_FRAMES=40`, since the `-1` that JAX's own warning suggests fails in 0.10.2.
- Inspect a staged program with `jax.jit(f).trace(*args).jaxpr` or `jax.jit(f).lower(*args).as_text()`. Lowered objects work only in the process that made them.
- `jax.debug_nans(True)` also stops on the `nan` that mmmjax returns on purpose for invalid parameters. Chase a non-finite log density with `jax.debug.print` or `jax.experimental.checkify` instead.

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
