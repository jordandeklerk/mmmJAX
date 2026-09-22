# mmmJAX benchmarks

The benchmark suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to track the cost
of distribution evaluation, gradients, sampling, and tail probabilities across representative
model shapes, and a separate command compares mmmJAX with the equivalent public JAX operations.
The workloads cover known numerical edge cases, but they do not replace the correctness tests, and
the two implementations do not always do identical work because mmmJAX validates its parameters
more strictly.

## Running benchmarks

Install and enter the locked benchmark environment from the repository root, then run everything
below from that shell.

```console
pixi install -e benchmark
pixi shell -e benchmark
```

`spin bench` benchmarks the current checkout. Add `--quick` to run each benchmark once, `-t` to
select a suite or a benchmark class, and `--compare` to compare committed revisions. The full suite
and revision comparisons take several minutes, so use `--quick` when you only need to confirm that
the benchmarks execute.

```console
spin bench --quick
spin bench -t bench_density.ElementwiseLogProbability
spin compare
```

Revision comparisons install only committed project source, so commit package changes before
comparing them. The suite and its configuration always come from the current checkout and define
both runs. `asv.conf.json` pins JAX, jaxlib, and ASV Runner for the managed environment, and
`spin bench --help` and `spin compare --help` list every option.

`spin asv` exposes the lower-level ASV commands for keeping a result history.

```console
spin asv machine --yes
spin asv run --show-stderr HEAD
spin asv publish
spin asv preview
```

Generated `.asv` artifacts are ignored by Git, and machine metadata lives in `~/.asv-machine.json`.

## What is measured

- `bench_density` times elementwise log probabilities and summed log densities.
- `bench_event` times Dirichlet and Multinomial log probabilities, gradients, and sampling.
- `bench_grad` times log densities together with their parameter gradients.
- `bench_random` times random sampling for every distribution with a sampler.
- `bench_tail` times log-CDFs, log-survival functions, and their gradients.

Every suite uses float32 inputs in three profiles. `vector` pairs values of shape `(32,)` with
scalar parameters, `likelihood` pairs values of shape `(260, 8)` with parameters of shape `(8,)`,
and `channel_prior` pairs values of shape `(8, 465)` with parameters of shape `(465,)`. A fourth
`stress` profile with values of shape `(260, 8, 465)` is available only for paired comparisons.
Categorical parameters append their event axis to the parameter shape, and event distributions
use their own profiles with one 32-component event, 260 samples over eight batches of
four-component events, or eight samples from a shared 465-component vector.

ASV lists each parameter combination of a method as its own row. A timing is the sample median
followed by half the interquartile range, so a large second number means unstable samples that
deserve a rerun. Each sample averages 100 synchronized calls, five samples make a round, and every
timing excludes compilation while including the compiled call and host synchronization.

## Comparing with JAX

`spin compare` measures mmmJAX and the matching public JAX operations in one process. The default
workload uses the `channel_prior` profile with ordinary inputs and every density, gradient, and
sampling operation, and filters narrow it.

```console
spin compare --profiles vector --distributions normal
```

The report separates cache-cleared compilation from warm execution. Compilation timings are
descriptive, while warm results give the median, the median absolute deviation, throughput,
iteration counts, and the ratio of the two medians. References come from `jax.scipy.stats`,
`jax.random`, and plain JAX array operations, composed where no single public function matches an
mmmJAX parameterization. SciPy stays a correctness reference rather than a timing baseline because
it offers none of the JIT, differentiation, accelerator, or PRNG behavior being measured.

Tail inputs target log probabilities from about -4 to -35, discrete inputs cycle valid outcomes
across the sample and parameter axes, and gradients differentiate only the distribution parameters,
which matches how observed data is treated during inference. Some deep-tail references are omitted
because JAX's direct formulas lose accuracy there, and some gradients are mmmJAX-only because JAX
lacks the needed incomplete-Beta derivatives. `cases.py` and `references.py` hold the exact input
values and reference availability for every case. Two specialized comparisons cover the
concentrated Poisson inputs and the tail operations.

```console
spin compare --distributions poisson poisson_log --inputs concentrated \
  --operations logpmf log_density value_and_grad

spin compare --inputs ordinary tail \
  --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad
```

## Writing benchmarks

Follow ASV's [benchmark-writing guidance](https://asv.readthedocs.io/en/latest/writing_benchmarks.html)
for the fundamentals. Timing modules live in `benchmarks/benchmarks/` and are named `bench_*.py`,
workload definitions live in `cases.py`, and shared measurement utilities live in `common.py`.
Use ASV `time_` methods, build inputs and compiled functions in `setup`, and synchronize every
timed JAX result. Leave warm-up enabled and keep the shared `number` and `repeat` policy unless
measurements justify a change. Keep `params` and `param_names` stable across revisions, handle
cases that older revisions lack inside `setup`, and keep every module importable across supported
revisions, which also means no `time_` or `track_` prefixes in support modules. Increment the
suite's `version` whenever workload construction, setup, synchronization, or the measured
operation changes.

Before requesting review, check discovery and run the suite you changed.

```console
spin asv check -E existing
spin bench -t bench_tail --quick
```
