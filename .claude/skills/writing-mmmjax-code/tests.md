# mmmJAX tests

These rules cover where a test goes, how it is laid out, and how it checks numerics, errors, and sampling.

- `mmmjax/<folder>/<module>.py` is tested in `tests/<folder>/test_<module>.py` or a feature file beside it such as `tests/model/test_model_inputs.py`, and distributions in `tests/distributions/test_<name>.py`. Package-wide checks such as `test_package.py` stay at the top of `tests/`. Test folders have no `__init__.py`, so a test file name must be unique across `tests/`. Fixtures live in the module that uses them and `conftest.py` holds none. Helpers stay private in the test module that uses them, as `_convolution_reference` does in `tests/media/test_adstock.py`, and `tests/helpers.py` holds only `importorskip`.
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
- Stack parametrize decorators for products, use `pytest.param(..., id=...)` for values without a readable repr, and use a fixture with `params=` for backend matrices such as the dataframe libraries (the `frame_factory` fixture in `tests/data/test_prepare.py`).
- Keep real NUTS out of tests. The `nuts_calls` fixture in `tests/inference/test_sampling.py` stubs `sampling._sample_nuts`, `_collect_results` builds result trees directly, and a stub that must never run raises `AssertionError`. An unavoidable real run stays at `draws=2, warmup=3, chains=1`.
- A test that needs several devices runs a subprocess with `JAX_NUM_CPU_DEVICES` and `JAX_PLATFORMS=cpu`, because JAX fixes the device count at startup (`test_parallel_nuts_on_two_cpu_devices_preserves_targets_generation_and_labels`). CI splits tests across pytest-xdist workers, so no test may depend on another.
- Build data inline as small polars frames or with `simulate_data(seed=...)`. There are no data files and no `tmp_path`.
- A new distribution's tests follow `tests/distributions/test_normal.py`. They cover known values, the scalar sum, broadcasting, `nan` for invalid settings, support endpoints, `logcdf` and `logsf` as complements, extreme tails, gradients, `vmap`, and an `rng` that matches transformed `jax.random` draws from the same key.
- A new scalar family also joins each family list in `tests/distributions/test_contracts.py`. Those lists run every scalar family through empty batches, the float32 floor, float64, `jax.jit`, `nan` draws for invalid settings, and key `vmap`. The family also gets a case in `benchmarks/cases.py` and a JAX reference in `benchmarks/references.py`, and `tests/distributions/test_references.py` checks both against the public API.
