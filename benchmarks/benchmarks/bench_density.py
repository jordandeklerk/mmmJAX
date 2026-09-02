"""Benchmark steady-state distribution density operations."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import DISTRIBUTIONS, MMM_JAX_FUNCTIONS, PROFILES, make_arguments
from benchmarks.common import Arguments, BenchmarkFunction, compile_and_warm, synchronize


class _DensityBenchmark:
    params = (
        tuple(distribution.name for distribution in DISTRIBUTIONS),
        ("vector", "likelihood", "channel_prior"),
    )
    param_names = ("distribution", "profile")
    operation_name = ""

    arguments: Arguments
    compiled: Callable[..., object]

    def setup(self, distribution_name: str, profile_name: str) -> None:
        """Prepare and warm one compiled density operation."""
        distribution = next(distribution for distribution in DISTRIBUTIONS if distribution.name == distribution_name)
        functions = MMM_JAX_FUNCTIONS[distribution_name]
        arguments = make_arguments(
            distribution,
            PROFILES[profile_name],
            "ordinary",
            jnp.dtype(jnp.float32),
        )

        function: BenchmarkFunction
        if self.operation_name == "elementwise":
            function = functions.elementwise_log_probability
        elif self.operation_name == "log_density":
            function = functions.summed_log_probability
        elif self.operation_name == "value_and_grad":
            function = jax.value_and_grad(
                functions.summed_log_probability,
                argnums=distribution.gradient_argnums,
            )
        else:
            raise RuntimeError(f"unknown density benchmark operation {self.operation_name!r}")

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)


class ElementwiseLogProbability(_DensityBenchmark):
    """Measure elementwise log-PDF and log-PMF execution."""

    operation_name = "elementwise"

    def time_log_probability(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized elementwise density call."""
        synchronize(self.compiled(*self.arguments))


class SummedLogDensity(_DensityBenchmark):
    """Measure log-density reduction execution."""

    operation_name = "log_density"

    def time_log_density(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized summed density call."""
        synchronize(self.compiled(*self.arguments))


class ParameterGradient(_DensityBenchmark):
    """Measure parameter value-and-gradient execution."""

    operation_name = "value_and_grad"

    def time_value_and_grad(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized value-and-gradient call."""
        synchronize(self.compiled(*self.arguments))
