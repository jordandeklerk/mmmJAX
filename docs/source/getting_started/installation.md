# Installation

mmmJAX is in alpha and needs Python 3.12 or later. It isn't on PyPI yet, so
install the development version from GitHub.

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
It also brings in [ArviZ](https://python.arviz.org/) 1.3 or later for
convergence diagnostics and [plotnine](https://plotnine.org/) for the plots,
and both read the results directly.

## Running on a GPU

JAX runs on the CPU unless you install an accelerator build. On a machine with
an NVIDIA GPU and CUDA 12 drivers, the `gpu` extra installs the CUDA build of
JAX along with mmmJAX.

```bash
pip install "mmmjax[gpu] @ git+https://github.com/jordandeklerk/mmmJAX.git"
```

For other accelerators or CUDA versions, install the matching JAX wheel first
with the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install mmmJAX. It uses whichever device JAX finds.

:::{admonition} Sharing a GPU
:class: tip

JAX reserves three quarters of the GPU's memory the first time it runs a
computation, which gets in the way when several notebooks or processes share
one card. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the environment before
Python starts and JAX allocates memory as it needs it instead.
:::

## Checking the installation

```python
import jax
import mmmjax as mj

print(mj.__version__)
print(jax.devices())
```

The device list shows `CpuDevice` entries on a CPU and `CudaDevice` entries
when JAX can see a GPU. If you installed the CUDA build and still see only the
CPU, JAX couldn't load the CUDA libraries. The JAX installation guide lists the
usual causes.

## Choosing float32 or float64

JAX computes in 32-bit floating point unless you tell it otherwise, and so
does mmmJAX. That's fast and accurate enough for most marketing mix models,
especially once the data is scaled. Switch to 64-bit for very long series, for
quantities that differ by many orders of magnitude, or to compare against a
library that samples in 64-bit.

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

## CPU devices for parallel chains

A CPU counts as one JAX device, and running chains in parallel needs a device
for each chain. As with the precision, you have to set the number of CPU
devices before JAX runs anything.

```python
import jax

jax.config.update("jax_num_cpu_devices", 4)
```

[Sampler settings](../user_guide/sampling.md#sampler-settings) explains when
parallel chains pay off and how to ask for them. Once the installation works,
[Introduction to MMM](intro_to_mmm) covers the math and methods behind the
models mmmJAX fits.
