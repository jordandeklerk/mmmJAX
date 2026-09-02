# mmmJAX benchmarks

The mmmJAX benchmark suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to track
performance across the numerical workloads that matter for Bayesian marketing mix models. It covers
distribution evaluation, gradients, sampling, and tail probabilities across representative model
shapes, with separate comparisons against public JAX operations.

## Running benchmarks

Pixi provides the locked benchmark environment. Install and enter it from the repository root:

```console
pixi install -e benchmark
pixi shell -e benchmark
```

Run all remaining commands from this Pixi shell.

- Benchmark the current checkout: `spin bench`
- Check each benchmark once: `spin bench --quick`
- Run one suite: `spin bench -t bench_tail`
- Run one benchmark class: `spin bench -t bench_density.ElementwiseLogProbability`
- Compare committed `main` and `HEAD`: `spin bench --compare main`
- Compare one suite across revisions: `spin bench --compare main -t bench_tail`
- Compare mmmJAX with public JAX: `spin compare`

Use `spin bench --help` and `spin compare --help` for all available options.

Revision comparisons install the committed project source for each revision. The benchmark suite
and configuration from the current checkout define both runs, so use a clean working tree when those
files are changing. The managed environment pins JAX and jaxlib through `asv.conf.json`.

### Result history

`spin asv` runs lower-level ASV operations from the benchmark directory:

```console
spin asv machine --yes
spin asv run --show-stderr HEAD
spin asv publish
spin asv preview
```

Generated `.asv` artifacts are ignored by Git. Local machine metadata lives in
`~/.asv-machine.json`.

## What is measured

- `bench_density`: Elementwise log probabilities and summed log densities
- `bench_grad`: Log densities and their parameter gradients
- `bench_random`: Random sampling
- `bench_tail`: Log-CDFs, log-survival functions, and their parameter gradients

The ASV suites use float32 inputs and cover the `vector`, `likelihood`, and `channel_prior`
profiles. The `stress` profile remains opt-in for paired comparisons.

### Profiles

- `vector`: Values `(32,)` and scalar parameters for basic broadcasting
- `likelihood`: Values `(260, 8)` and parameters `(8,)` for grouped workloads
- `channel_prior`: Values `(8, 465)` and parameters `(465,)` for wide batches shared across groups
- `stress`: Values `(260, 8, 465)` and parameters `(8, 465)` for large nested structures

Categorical parameters append their event axis to the parameter batch shape. For example, the
`channel_prior` profile uses parameters shaped `(465, K)` and values shaped `(8, 465)`.

### Reading ASV results

ASV counts a parameterized method as one benchmark and expands its parameter combinations in the
results table. Timings show the sample median followed by half the interquartile range. A large value
after `±` indicates unstable samples and should be rerun. Density timings exclude compilation and
include the compiled JAX call and host synchronization.

## Comparing with JAX

`spin compare` measures mmmJAX and equivalent public JAX operations in the same process. Its default
workload uses the `channel_prior` profile, float32, ordinary inputs, both implementations, and all
standard density, gradient, and sampling operations. Filters can narrow the comparison:

```console
spin compare --profiles vector --distributions normal
```

The report separates cache-cleared JIT compilation from synchronized warm execution. Compilation
timings are descriptive. Warm results report the median, median absolute deviation, throughput,
iterations, and the relative mmmJAX and JAX medians.

References use public `jax.scipy.stats`, `jax.random`, and JAX array operations. Composed references
are used when no single public function matches an mmmJAX parameterization. SciPy remains an
independent correctness reference and is not a timing baseline because it does not provide the same
JIT, automatic differentiation, accelerator, or PRNG execution model.

## Special workloads

- Discrete inputs cycle valid outcomes across the sample and parameter dimensions. The Negative
  Binomial sampling reference composes public Gamma and Poisson random functions.
- Concentrated Poisson inputs span roughly two standard deviations around rates of `1e7` for
  float32 and `1e15` for float64. They remain exactly representable while exposing cancellation near
  the mode. Public JAX timings are omitted because its direct calculation is not numerically
  equivalent there, and sampling remains in the ordinary workload because the float64 rate exceeds
  the `int32` output range.
- Tail inputs target log probabilities from roughly -4 to -35. Gradient benchmarks sum elementwise
  values and differentiate only distribution parameters, matching the treatment of observed data
  during inference.

Ordinary tail-probability inputs cover standardized values from -2 to 2 for Laplace, Normal, and
LogNormal; 0.023 to 2.36 for Half Normal; `rate * value` from 0.1 to 3 for Exponential and 0.5 to 6
for Gamma; `scale / value` from 0.9 to 7.35 for Inverse Gamma; and probabilities from 0.25 to 0.75
for Uniform. Uniform anchors its evaluated endpoint at zero for representable float32 tails. Public
Laplace and Uniform references compose CDFs and reflected CDFs because JAX does not provide their
log-CDF or log-survival functions.

Run the specialized comparisons with:

```console
spin compare \
  --distributions poisson poisson_log \
  --inputs concentrated \
  --operations logpmf log_density value_and_grad

spin compare \
  --inputs ordinary tail \
  --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad
```

Exact workload values and reference availability are defined in `cases.py` and `references.py`.

## Writing benchmarks

See ASV's [benchmark-writing guidance](https://asv.readthedocs.io/en/latest/writing_benchmarks.html)
for the fundamentals. mmmJAX benchmarks should also follow these rules:

- Put timing modules in `benchmarks/benchmarks/` and name them `bench_*.py`
- Keep workload definitions in `cases.py` and shared measurement utilities in `common.py`
- Use ASV `time_` methods, prepare inputs and compiled functions in `setup`, and synchronize every
  timed JAX result
- Keep ASV warm-up enabled and keep benchmark runtimes practical
- Keep `params` and `param_names` stable across revisions, and handle unavailable historical cases
  in `setup`
- Keep every module importable across supported revisions and avoid benchmark prefixes such as
  `time_` or `track_` in support modules
- Increment the suite's explicit `version` when workload construction, setup, synchronization, or
  the measured operation changes

Before requesting review, check discovery and run the suite you changed:

```console
spin asv check -E existing
spin bench -t bench_tail --quick
```

## Scope

These workloads cover representative shapes and known numerical edge cases, but they do not replace
the distribution correctness tests. Implementations can also differ outside the measured inputs
because mmmJAX applies stricter parameter validation. Dirichlet is not yet registered because its
multivariate event axis requires a dedicated workload shape.
