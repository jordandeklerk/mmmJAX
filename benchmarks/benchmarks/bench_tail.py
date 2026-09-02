"""Benchmark steady-state tail probabilities and gradients."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import (
    DISTRIBUTIONS,
    DISTRIBUTIONS_BY_NAME,
    MMM_JAX_FUNCTIONS,
    PROFILES,
    TAIL_DISTRIBUTIONS,
    make_tail_arguments,
)
from benchmarks.common import Arguments, BenchmarkFunction, compile_and_warm, synchronize


class _TailBenchmark:
    version = "1"
    params = (
        tuple(distribution.name for distribution in DISTRIBUTIONS if distribution.name in TAIL_DISTRIBUTIONS),
        ("vector", "likelihood", "channel_prior"),
        ("logcdf", "logsf"),
    )
    param_names = ("distribution", "profile", "operation")
    with_gradient = False

    arguments: Arguments
    compiled: Callable[..., object]
    _case: tuple[str, str, str, bool] | None = None

    def setup(self, distribution_name: str, profile_name: str, operation: str) -> None:
        """Prepare and warm one compiled tail operation."""
        case = (distribution_name, profile_name, operation, self.with_gradient)
        if self._case == case:
            return

        distribution = DISTRIBUTIONS_BY_NAME[distribution_name]
        arguments = make_tail_arguments(
            distribution,
            PROFILES[profile_name],
            "tail",
            operation,
            jnp.dtype(jnp.float32),
        )
        functions = MMM_JAX_FUNCTIONS[distribution_name]
        probability = functions.logcdf if operation == "logcdf" else functions.logsf
        if probability is None:
            raise RuntimeError(f"{operation} benchmark is unavailable for {distribution_name!r}")

        function: BenchmarkFunction = probability
        if self.with_gradient:
            # Tail gradients reduce elementwise probabilities to the scalar used during inference
            def summed_probability(*function_arguments: jax.Array) -> jax.Array:
                return jnp.sum(probability(*function_arguments))

            function = jax.value_and_grad(
                summed_probability,
                argnums=distribution.gradient_argnums,
            )

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)
        self._case = case


class TailLogProbability(_TailBenchmark):
    """Measure elementwise log-CDF and log-survival execution."""

    def time_log_probability(self, distribution_name: str, profile_name: str, operation: str) -> None:
        """Measure one synchronized tail probability call."""
        synchronize(self.compiled(*self.arguments))


class TailLogProbabilityGradient(_TailBenchmark):
    """Measure parameter gradients of summed tail log probabilities."""

    with_gradient = True

    def time_value_and_grad(self, distribution_name: str, profile_name: str, operation: str) -> None:
        """Measure one synchronized tail value-and-gradient call."""
        synchronize(self.compiled(*self.arguments))
