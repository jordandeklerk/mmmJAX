"""Benchmark steady-state distribution density operations."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import DISTRIBUTIONS, IMPLEMENTATIONS, PROFILES, make_benchmark_operation
from benchmarks.common import Arguments, synchronize


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
        operation_name = (
            distribution.log_probability_operation if self.operation_name == "elementwise" else self.operation_name
        )
        operation = make_benchmark_operation(
            IMPLEMENTATIONS["mmmjax"][distribution_name],
            distribution,
            PROFILES[profile_name],
            input_set="ordinary",
            operation=operation_name,
            dtype=jnp.dtype(jnp.float32),
            implementation="mmmjax",
        )
        if operation is None:
            raise RuntimeError(
                f"density benchmark operation {operation_name!r} is unavailable for {distribution_name!r}"
            )

        self.arguments = operation.arguments
        synchronize(self.arguments)
        self.compiled = jax.jit(operation.function).lower(*self.arguments).compile()
        synchronize(self.compiled(*self.arguments))


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
