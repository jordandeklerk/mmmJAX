# Benchmarks

Run the distribution benchmarks from the repository root:

```console
pixi run benchmark-distributions
```

The default suite measures cache-cleared JIT compilation and synchronized warm execution for mmmJAX and equivalent public JAX operations. It covers elementwise log densities or masses, summed log density, parameter gradients, and random sampling.

Filters can be passed directly, for example:

```console
pixi run benchmark-distributions --profiles vector --distributions normal
```

Bernoulli, Binomial, Negative Binomial, and Poisson parameterizations cycle valid integer outcomes across the sample dimensions:

```console
pixi run benchmark-distributions --profiles vector --distributions bernoulli bernoulli_logit binomial binomial_logit negative_binomial negative_binomial_log poisson poisson_log --operations logpmf log_density value_and_grad rng
```

JAX does not provide a Negative Binomial sampler, so its sampling baseline composes public Gamma and Poisson random functions.

Large-count Poisson inputs exercise the stable deviance calculation near the mode:

```console
pixi run benchmark-distributions --profiles vector --distributions poisson poisson_log --inputs concentrated --operations logpmf log_density value_and_grad
```

These inputs spread counts across roughly two standard deviations around rates of `1e7` for float32 and `1e15` for float64. The values remain exactly representable while still exposing cancellation in the direct formula. Public JAX is omitted because its result is not numerically equivalent at these values. Random sampling remains in the ordinary workload because the float64 concentrated rate exceeds the `int32` output range.

Exponential, Gamma, Half Normal, Inverse Gamma, Laplace, Normal, LogNormal, and Uniform log-CDF and log-survival benchmarks are opt-in:

```console
pixi run benchmark-distributions --distributions exponential gamma half_normal inverse_gamma laplace normal lognormal uniform --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad --inputs ordinary tail
```

Laplace, Normal, and LogNormal ordinary inputs use standardized values from -2 to 2, while Half Normal uses values from 0.023 to 2.36. Exponential inputs use `rate * value` from 0.1 to 3, Gamma uses `rate * value` from 0.5 to 6, and Inverse Gamma uses `scale / value` from 0.9 to 7.35. Uniform ordinary inputs cover probabilities from 0.25 to 0.75. Its tail intervals place the evaluated endpoint at zero so probabilities down to about `exp(-32)` remain representable in float32. Laplace tail inputs directly target the same probability range. Tail inputs produce log probabilities from roughly -4 to -32. Gradient benchmarks sum the elementwise results and differentiate only the distribution parameters, matching how observed values are treated during inference.

JAX does not provide Laplace or Uniform log-CDF and log-survival functions, so their comparisons compose the corresponding public CDFs. The survival references evaluate reflected CDFs to avoid upper-tail cancellation. mmmJAX also performs stricter parameter validation and handles extreme inputs beyond these benchmark workloads, so the implementations can differ outside the values measured here.
