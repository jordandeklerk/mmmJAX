# Benchmarks

Run the distribution benchmarks from the repository root:

```console
pixi run benchmark-distributions
```

The default suite measures cache-cleared JIT compilation and synchronized warm execution for mmmJAX and equivalent public JAX operations. It covers elementwise and summed log densities, parameter value-and-gradients, and random sampling.

Filters can be passed directly, for example:

```console
pixi run benchmark-distributions --profiles vector --distributions normal
```

Exponential, Gamma, Half Normal, Inverse Gamma, Normal, LogNormal, and Uniform log-CDF and log-survival benchmarks are opt-in:

```console
pixi run benchmark-distributions --distributions exponential gamma half_normal inverse_gamma normal lognormal uniform --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad --inputs ordinary tail
```

Normal and LogNormal ordinary inputs use standardized values from -2 to 2, while Half Normal uses values from 0.023 to 2.36. Exponential inputs use `rate * value` from 0.1 to 3, Gamma uses `rate * value` from 0.5 to 6, and Inverse Gamma uses `scale / value` from 0.9 to 7.35. Uniform ordinary inputs cover probabilities from 0.25 to 0.75. Its tail intervals place the evaluated endpoint at zero so probabilities down to about `exp(-32)` remain representable in float32. Tail inputs produce log probabilities from roughly -4 to -32. Gradient benchmarks sum the elementwise results and differentiate only the distribution parameters, matching how observed values are treated during inference.

JAX does not provide Uniform log-CDF or log-survival functions, so the Uniform comparison composes its public Uniform CDF. The survival reference evaluates the reflected CDF to avoid upper-tail cancellation. mmmJAX also validates finite ordered bounds and handles intervals whose ordinary width overflows, so the implementations have different behavior for invalid and extreme inputs outside these benchmark workloads.
