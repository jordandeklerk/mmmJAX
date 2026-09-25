# Installation

mmmJAX is in alpha and requires Python 3.12 or later. It is not on PyPI yet,
so install the development version from GitHub.

::::{tab-set}

:::{tab-item} Install with pip

```bash
pip install "git+https://github.com/jordandeklerk/mmmJAX.git"
```

:::

:::{tab-item} Install with uv

```bash
uv add "git+https://github.com/jordandeklerk/mmmJAX.git"
```

:::

::::

The install brings in JAX, BlackJAX for sampling, the JAX backend of
TensorFlow Probability for the distributions, and xarray for labeled results.
It also brings in [ArviZ](https://python.arviz.org/) for convergence
diagnostics and [plotnine](https://plotnine.org/) for the plots, both of which
read the results directly.

## Running on a GPU

JAX runs on the CPU unless an accelerator build is installed. On a machine
with an NVIDIA GPU and CUDA 12 drivers, the `gpu` extra installs the CUDA build
of JAX together with mmmJAX.

```bash
pip install "mmmjax[gpu] @ git+https://github.com/jordandeklerk/mmmJAX.git"
```

For other accelerators or CUDA versions, install the matching JAX wheel first
by following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install mmmJAX, which uses whichever device JAX finds. JAX also reserves
three quarters of the GPU's memory the first time it runs a computation, which
gets in the way when several notebooks or processes share one card. Setting
`XLA_PYTHON_CLIENT_PREALLOCATE=false` in the environment before Python starts
makes it allocate memory as needed instead.

## Checking the installation

```python
import jax
import mmmjax as mj

print(mj.__version__)
print(jax.devices())
```

The device list shows `CpuDevice` entries on a CPU and `CudaDevice` entries
when JAX can see a GPU. If you installed the CUDA build and still see only the
CPU, JAX could not load the CUDA libraries, and the JAX installation guide
lists the usual causes.

## Choosing float32 or float64

JAX computes in 32-bit floating point unless told otherwise, and so does
mmmJAX. That is fast and accurate enough for most marketing mix models,
especially once the data is scaled. Switch to 64-bit when a model involves
very long series, quantities that differ by many orders of magnitude, or a
comparison against a library that samples in 64-bit.

```python
import jax

jax.config.update("jax_enable_x64", True)
```

:::{admonition} Set precision first
:class: warning

Put this line at the top of the script or notebook, before you fit any
scaling, create priors, or build a model. Those objects keep the precision
that was in effect when they were created, so switching partway through leaves
some of them in 32-bit.
:::

Setting `JAX_ENABLE_X64=true` in the environment before Python starts has the
same effect. In 32-bit mode, integer inputs larger than about two billion
raise an error instead of silently wrapping around, and 64-bit mode lifts that
limit.

## Running chains at the same time

{func}`~mmmjax.sample` runs chains one after another by default. The other two
values of `chain_method` run them at the same time.

```python
results = mj.sample(model, chains=4, chain_method="vectorized")
```

Vectorized sampling batches every chain into one computation on a single
device. It needs no setup, and it is the only way to run several chains at
once on a single GPU. The batch holds every chain in memory at once, and each
step waits for the chain with the longest trajectory, so it is not always
faster than running the chains in turn.

Parallel sampling gives each chain its own device instead. Every GPU in a
machine is already a device, but a CPU counts as one until you ask for more
before JAX runs anything, just like the precision setting.

```python
import jax

jax.config.update("jax_num_cpu_devices", 4)
```

With four devices, `chain_method="parallel"` runs four chains side by side.
Asking for more chains than devices raises an error that points to the other
two options.

Once the installation works, the [Overview](overview) explains how mmmJAX
approaches a model, and the [User Guide](../user_guide/index) builds one.
