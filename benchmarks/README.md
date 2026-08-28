# Benchmarks

Run the distribution benchmarks from the repository root:

```console
pixi run benchmark-distributions
```

The suite measures cache-cleared JIT compilation and synchronized warm execution for mmmJAX and equivalent public JAX operations. It currently covers elementwise and summed log densities, parameter value-and-gradients, and random sampling.

Filters can be passed directly, for example:

```console
pixi run benchmark-distributions --profiles vector --distributions normal
```
