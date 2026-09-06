# Running tests

Run the full suite on CPU with up to four workers:

```console
pixi run --frozen tests-parallel
```

This runs the same tests and assertions as the serial suite. The worker limit caps
the number of separate JAX processes and their compilation caches. The slowest
tests are listed at the end.

For targeted checks or debugging, run pytest directly:

```console
pixi run --frozen pytest tests/test_adstock.py
```

Test double precision in a separate process so JAX's dtype configuration stays
consistent throughout each run:

```console
JAX_ENABLE_X64=true pixi run --frozen tests-parallel
```

The parallel task uses CPU explicitly. Use `pixi run --frozen tests` for a serial
full run on the selected JAX backend. To adjust the CPU worker count, use
`JAX_PLATFORMS=cpu pixi run --frozen pytest -n 2 --dist=worksteal`.
