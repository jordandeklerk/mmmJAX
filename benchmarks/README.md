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

Normal and LogNormal log-CDF and log-survival benchmarks are opt-in:

```console
pixi run benchmark-distributions --distributions normal lognormal --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad --inputs ordinary tail
```

Ordinary inputs use standardized values from -2 to 2. Tail inputs use values from -8 to -2 for log-CDFs and 2 to 8 for log-survival functions. Gradient benchmarks sum the elementwise results and differentiate only the distribution parameters, matching how observed values are treated during inference.
