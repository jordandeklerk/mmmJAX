"""Benchmark Dirichlet operations with an explicit event axis."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import DIRICHLET_FUNCTIONS, EVENT_PROFILES, make_dirichlet_arguments
from benchmarks.common import Arguments, BenchmarkFunction, compile_and_warm, synchronize


class _DirichletBenchmark:
    version = "2"
    # Batch short device calls so scheduler noise does not dominate each sample
    number = 100
    repeat = 5
    params = ("vector", "likelihood", "channel_prior")
    param_names = ("profile",)
    operation_name = ""

    arguments: Arguments
    compiled: Callable[..., object]
    _case: tuple[str, str] | None = None

    def setup(self, profile_name: str) -> None:
        """Prepare and warm one compiled Dirichlet operation."""
        case = (profile_name, self.operation_name)
        if self._case == case:
            return

        profile = EVENT_PROFILES[profile_name]
        value, concentration = make_dirichlet_arguments(profile, jnp.dtype(jnp.float32))

        function: BenchmarkFunction
        if self.operation_name == "elementwise":
            function = DIRICHLET_FUNCTIONS.elementwise_log_probability
            arguments = (value, concentration)
        elif self.operation_name == "log_density":
            function = DIRICHLET_FUNCTIONS.summed_log_probability
            arguments = (value, concentration)
        elif self.operation_name == "value_and_grad":
            function = jax.value_and_grad(
                DIRICHLET_FUNCTIONS.summed_log_probability,
                argnums=1,
            )
            arguments = (value, concentration)
        elif self.operation_name == "sampling":
            function = functools.partial(
                DIRICHLET_FUNCTIONS.rng,
                sample_shape=profile.sample_shape,
            )
            arguments = (jax.random.key(0), concentration)
        else:
            raise RuntimeError(f"unknown Dirichlet benchmark operation {self.operation_name!r}")

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)
        self._case = case


class DirichletLogProbability(_DirichletBenchmark):
    """Measure elementwise Dirichlet log-density execution."""

    operation_name = "elementwise"

    def time_log_probability(self, profile_name: str) -> None:
        """Measure one synchronized elementwise density call."""
        synchronize(self.compiled(*self.arguments))


class DirichletLogDensity(_DirichletBenchmark):
    """Measure summed Dirichlet log-density execution."""

    operation_name = "log_density"

    def time_log_density(self, profile_name: str) -> None:
        """Measure one synchronized summed density call."""
        synchronize(self.compiled(*self.arguments))


class DirichletLogDensityGradient(_DirichletBenchmark):
    """Measure Dirichlet concentration value-and-gradient execution."""

    operation_name = "value_and_grad"

    def time_value_and_grad(self, profile_name: str) -> None:
        """Measure one synchronized value-and-gradient call."""
        synchronize(self.compiled(*self.arguments))


class DirichletSampling(_DirichletBenchmark):
    """Measure synchronized Dirichlet sampling execution."""

    operation_name = "sampling"

    def time_sampling(self, profile_name: str) -> None:
        """Measure one synchronized sampling call."""
        synchronize(self.compiled(*self.arguments))
