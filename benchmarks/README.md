# Benchmarks

The distribution benchmarks compare mmmJAX with equivalent public JAX operations. They report
cache-cleared JIT compilation and synchronized warm execution for elementwise log probabilities,
summed log density, parameter gradients, and random sampling.

Compilation timings are descriptive. Warm execution comparisons report the ratio of the JAX median
to the mmmJAX median, so a value above one means mmmJAX had the lower median in that run. Read ratios
alongside the reported median absolute deviations.

## Running the benchmarks

Run the default suite from the repository root:

```console
pixi run benchmark-distributions
```

The default uses the `channel_prior` profile, float32, ordinary inputs, all distributions, their
standard density, gradient, and RNG operations, and both implementations. Use filters to run a
smaller comparison:

```console
pixi run benchmark-distributions --profiles vector --distributions normal
```

All command-line options are available through:

```console
pixi run benchmark-distributions --help
```

## Profiles

| Profile | Value shape | Parameter batch shape | Purpose |
| --- | --- | --- | --- |
| `vector` | `(32,)` | `()` | Flat workload with scalar parameter broadcasting |
| `likelihood` | `(260, 8)` | `(8,)` | Grouped workload with one parameter per group |
| `channel_prior` | `(8, 465)` | `(465,)` | Wide parameter batch broadcast across groups |
| `stress` | `(260, 8, 465)` | `(8, 465)` | Large nested parameter structure that remains opt-in |

Categorical parameters append their category event axis to the parameter batch shape. For example,
the `channel_prior` profile uses parameters shaped `(465, K)` while generating values shaped
`(8, 465)`.

## Discrete distributions

Discrete workloads cycle valid integer outcomes across the sample and parameter batch dimensions:

```console
pixi run benchmark-distributions \
  --profiles vector \
  --distributions bernoulli bernoulli_logit binomial binomial_logit \
    categorical categorical_logit negative_binomial negative_binomial_log \
    poisson poisson_log \
  --operations logpmf log_density value_and_grad rng
```

JAX does not provide a Negative Binomial sampler, so that sampling baseline composes public Gamma and
Poisson random functions.

## Large-count Poisson inputs

Concentrated Poisson inputs exercise the stable deviance calculation near the mode:

```console
pixi run benchmark-distributions \
  --profiles vector \
  --distributions poisson poisson_log \
  --inputs concentrated \
  --operations logpmf log_density value_and_grad
```

Counts span roughly two standard deviations around rates of `1e7` for float32 and `1e15` for
float64. They remain exactly representable while exposing cancellation in the direct formula. Public
JAX is omitted because its calculation is not numerically equivalent at these values. Sampling stays
in the ordinary workload because the concentrated float64 rate exceeds the `int32` output range.

## CDF and survival functions

Log-CDF and log-survival benchmarks are opt-in:

```console
pixi run benchmark-distributions \
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

## Reference implementations

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

## Scope

The benchmark inputs target representative model shapes and known numerical edge cases. They do not
replace the distribution correctness tests. mmmJAX also applies stricter parameter validation and
handles values outside these workloads, so implementations can differ beyond the ranges measured
here. Dirichlet is not yet registered because its multivariate event axis requires a dedicated
workload shape.
