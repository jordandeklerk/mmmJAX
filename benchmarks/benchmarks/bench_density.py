"""Benchmark steady-state distribution density operations."""

from collections.abc import Callable

import jax.numpy as jnp

from benchmarks.cases import DISTRIBUTIONS, DISTRIBUTIONS_BY_NAME, MMM_JAX_FUNCTIONS, PROFILES, make_arguments
from benchmarks.common import Arguments, BenchmarkFunction, compile_and_warm, synchronize


class _DensityBenchmark:
    version = "1"
    params = (
        tuple(distribution.name for distribution in DISTRIBUTIONS),
        ("vector", "likelihood", "channel_prior"),
    )
    param_names = ("distribution", "profile")
    operation_name = ""

    arguments: Arguments
    compiled: Callable[..., object]
    _case: tuple[str, str, str] | None = None

    def setup(self, distribution_name: str, profile_name: str) -> None:
        """Prepare and warm one compiled density operation."""
        case = (distribution_name, profile_name, self.operation_name)
        if self._case == case:
            return

        distribution = DISTRIBUTIONS_BY_NAME[distribution_name]
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
        else:
            raise RuntimeError(f"unknown density benchmark operation {self.operation_name!r}")

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)
        self._case = case


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
