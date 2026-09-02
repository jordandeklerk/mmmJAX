# mmmJAX benchmarks

The mmmJAX benchmark suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to track
performance across the numerical workloads that matter for Bayesian marketing mix models. It
measures distribution evaluation, gradients, sampling, and tail probabilities across representative
model shapes, with separate comparisons against public JAX operations.

## Usage

Pixi provides the locked benchmark environment. Install and enter it from the repository root:

```console
pixi install -e benchmark
pixi shell -e benchmark
```

Run the remaining commands from this Pixi shell. Spin is the main interface for the ASV suite.
Benchmark the current checkout without recording the results:

```console
spin bench
```

Run each benchmark once to check that the suite executes:

```console
spin bench --quick
```

Run a focused suite or benchmark class by passing an ASV expression:

```console
spin bench -t bench_tail
spin bench -t bench_density.ElementwiseLogProbability
```

Use `spin bench --help` to see all available options.

## Comparing revisions

Compare the current `HEAD` with committed `main` in an ASV-managed environment:

```console
spin bench --compare main
```

The same benchmark filters work in comparison mode:

```console
spin bench --compare main -t bench_tail
```

ASV installs the committed project source for each revision. The benchmark suite and configuration
from the current checkout define the workload for both runs, so use a clean working tree when those
files are changing. The managed environment pins JAX and jaxlib through `asv.conf.json` so dependency
changes do not appear as mmmJAX performance changes.

## Managing ASV results

The `spin asv` command runs lower-level ASV operations from the benchmark directory. Use it to
record results, inspect historical performance, or build the HTML report:

```console
spin asv machine --yes
spin asv run --show-stderr HEAD
spin asv publish
spin asv preview
```

The `.asv` artifacts are ignored by Git. ASV stores local machine metadata in
`~/.asv-machine.json`.

## Reading ASV results

ASV counts each parameterized method as one benchmark and expands its parameters in the results
table. For example, the elementwise density benchmark reports every distribution and profile even
though ASV identifies it as one benchmark.

Timing results show the sample median followed by half the interquartile range. A large value after
`±` indicates that the samples were spread out and the benchmark should be repeated before drawing
conclusions. Density timings exclude compilation and include the compiled JAX call and host
synchronization.

## Benchmark suites

The ASV suites separate density evaluation (`bench_density`), log-density gradients (`bench_grad`),
random sampling (`bench_random`), and tail probabilities (`bench_tail`). Density, gradient, and
sampling benchmarks use ordinary float32 inputs. Tail benchmarks measure log-CDF, log-survival, and
their parameter gradients on float32 tail inputs. Every suite covers the `vector`, `likelihood`, and
`channel_prior` profiles. ASV tracks mmmJAX across commits and does not use public JAX as a timing
baseline.

### Profiles

| Profile | Value shape | Parameter batch shape | Purpose |
| --- | --- | --- | --- |
| `vector` | `(32,)` | `()` | Flat workload with scalar parameter broadcasting |
| `likelihood` | `(260, 8)` | `(8,)` | Grouped workload with one parameter per group |
| `channel_prior` | `(8, 465)` | `(465,)` | Wide parameter batch broadcast across groups |
| `stress` | `(260, 8, 465)` | `(8, 465)` | Large nested parameter structure that remains opt-in |

Categorical parameters append their category event axis to the parameter batch shape. For example,
the `channel_prior` profile uses parameters shaped `(465, K)` while generating values shaped
`(8, 465)`.

## Paired JAX comparisons

The comparison command reports cache-cleared JIT compilation and synchronized warm execution for
mmmJAX and equivalent public JAX operations. Compilation timings are descriptive. Warm execution
comparisons report the percentage difference between the medians and should be read alongside the
reported median absolute deviations.

### Running comparisons

Run the default suite from the repository root:

```console
spin compare
```

The default uses the `channel_prior` profile, float32, ordinary inputs, all distributions, their
standard density, gradient, and RNG operations, and both implementations. Use filters to run a
smaller comparison:

```console
spin compare --profiles vector --distributions normal
```

All command-line options are available through:

```console
spin compare --help
```

### Reading the results

The report begins with the runtime, device, precision, and measurement settings. Results are then
grouped by profile, input set, dtype, and value count so those details are not repeated in every row.

The warm execution table reports the median, median absolute deviation (MAD), throughput, and timed
iterations for each implementation. When both implementations are present, the final column states
whether the mmmJAX median was shorter or longer than the JAX median in that run. The compilation
table is kept separate because cache-cleared compilation and warm execution measure different costs.

### Discrete distributions

Discrete workloads cycle valid integer outcomes across the sample and parameter batch dimensions:

```console
spin compare \
  --profiles vector \
  --distributions bernoulli bernoulli_logit binomial binomial_logit \
    categorical categorical_logit negative_binomial negative_binomial_log \
    poisson poisson_log \
  --operations logpmf log_density value_and_grad rng
```

JAX does not provide a Negative Binomial sampler, so that sampling baseline composes public Gamma and
Poisson random functions.

### Large-count Poisson inputs

Concentrated Poisson inputs exercise the stable deviance calculation near the mode:

```console
spin compare \
  --profiles vector \
  --distributions poisson poisson_log \
  --inputs concentrated \
  --operations logpmf log_density value_and_grad
```

Counts span roughly two standard deviations around rates of `1e7` for float32 and `1e15` for
float64. They remain exactly representable while exposing cancellation in the direct formula. Public
JAX is omitted because its calculation is not numerically equivalent at these values. Sampling stays
in the ordinary workload because the concentrated float64 rate exceeds the `int32` output range.

### CDF and survival functions

Log-CDF and log-survival benchmarks are opt-in:

```console
spin compare \
  --distributions exponential gamma half_normal inverse_gamma \
    laplace normal lognormal uniform \
  --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad \
  --inputs ordinary tail
```

Ordinary inputs cover the following ranges:

- Laplace, Normal, and LogNormal use standardized values from -2 to 2
- Half Normal uses standardized values from 0.023 to 2.36
- Exponential uses `rate * value` from 0.1 to 3
- Gamma uses `rate * value` from 0.5 to 6
- Inverse Gamma uses `scale / value` from 0.9 to 7.35
- Uniform covers probabilities from 0.25 to 0.75

Tail inputs produce log probabilities from roughly -4 to -35. Uniform anchors the evaluated endpoint
at zero so probabilities near `exp(-32)` remain representable in float32. Laplace directly targets
the same probability range. Gradient benchmarks sum elementwise results and differentiate only the
distribution parameters, matching how observed values are treated during inference.

JAX does not provide Laplace or Uniform log-CDF and log-survival functions. Their references compose
the corresponding public CDFs and evaluate reflected CDFs for survival probabilities to avoid
upper-tail cancellation.

### Reference implementations

Every registered workload has an equivalent implementation built from public JAX APIs. Some use a
single `jax.scipy.stats` or `jax.random` function. Others compose public JAX operations when no single
function matches the mmmJAX parameterization. This keeps compilation, automatic differentiation,
dtype, and device execution comparable.

SciPy is used as an independent numerical reference in the correctness tests. It is not included in
these timings because eager NumPy execution does not provide a comparable JIT, gradient, accelerator,
or PRNG baseline. A conventional CPU comparison with SciPy would need to be reported as a separate
benchmark.

Concentrated inputs report only mmmJAX timings when the public JAX calculation is not numerically
equivalent at the values being measured.

## Writing benchmarks

See ASV's [benchmark-writing guidance](https://asv.readthedocs.io/en/latest/writing_benchmarks.html)
for the fundamentals. mmmJAX benchmarks should also follow these conventions:

- Put timing modules in `benchmarks/benchmarks/` and name them `bench_*.py`
- Keep shared workload definitions in `cases.py` and shared measurement utilities in `common.py`
- Use ASV `time_` methods for regression timings instead of adding custom timers to the ASV suite
- Prepare arguments, compile functions, and complete the first JAX call in `setup`
- Synchronize every timed result because JAX execution can be asynchronous
- Keep `params` and `param_names` stable across the revisions being compared
- Keep benchmark modules importable with every supported revision and skip unavailable cases in
  `setup` rather than changing parameter definitions based on the installed mmmJAX version
- Keep runtimes reasonable so focused development runs remain useful

Each suite has an explicit ASV `version` because its behavior also depends on shared modules.
Increment the version when changing workload construction, setup, synchronization, or the operation
being measured. ASV imports every Python module under the configured benchmark directory, regardless
of its filename, so support modules must remain safe to import and must not expose benchmark prefixes
such as `time_` or `track_`.

Keep ASV's timing warm-up enabled because early synchronized calls can still be noisy after JAX
compilation. Before requesting review, return to the repository root, check discovery, and run the
suite you changed. For example:

```console
spin asv check -E existing
spin bench -t bench_tail --quick
```

## Scope

The benchmark inputs target representative model shapes and known numerical edge cases. They do not
replace the distribution correctness tests. mmmJAX also applies stricter parameter validation and
handles values outside these workloads, so implementations can differ beyond the ranges measured
here. Dirichlet is not yet registered because its multivariate event axis requires a dedicated
workload shape.
