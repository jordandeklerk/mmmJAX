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

Exponential, Normal, and LogNormal log-CDF and log-survival benchmarks are opt-in:

```console
pixi run benchmark-distributions --distributions exponential normal lognormal --operations logcdf logcdf_value_and_grad logsf logsf_value_and_grad --inputs ordinary tail
```

Normal and LogNormal ordinary inputs use standardized values from -2 to 2, while Exponential inputs use rate-scaled values from 0.1 to 3. Tail inputs produce log probabilities from roughly -4 to -32. Gradient benchmarks sum the elementwise results and differentiate only the distribution parameters, matching how observed values are treated during inference.
