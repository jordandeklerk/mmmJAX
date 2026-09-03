"""Benchmark distribution operations with an explicit event axis."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import (
    EVENT_DISTRIBUTIONS_BY_NAME,
    EVENT_MMM_JAX_FUNCTIONS,
    EVENT_PROFILES,
    make_event_arguments,
)
from benchmarks.common import Arguments, BenchmarkFunction, compile_and_warm, synchronize


class _EventBenchmark:
    # Batch short device calls so scheduler noise does not dominate each sample
    number = 100
    repeat = 5
    operation_name = ""

    arguments: Arguments
    compiled: Callable[..., object]
    _case: tuple[str, str, str] | None = None

    def _setup(self, distribution_name: str, profile_name: str) -> None:
        case = (distribution_name, profile_name, self.operation_name)
        if self._case == case:
            return

        distribution = EVENT_DISTRIBUTIONS_BY_NAME[distribution_name]
        profile = EVENT_PROFILES[profile_name]
        functions = EVENT_MMM_JAX_FUNCTIONS[distribution_name]
        event_arguments = make_event_arguments(
            distribution,
            profile,
            jnp.dtype(jnp.float32),
        )

        function: BenchmarkFunction
        if self.operation_name == "elementwise":
            function = functions.elementwise_log_probability
            arguments = event_arguments.log_probability
        elif self.operation_name == "log_density":
            function = functions.summed_log_probability
            arguments = event_arguments.log_probability
        elif self.operation_name == "value_and_grad":
            function = jax.value_and_grad(
                functions.summed_log_probability,
                argnums=distribution.gradient_argnums,
            )
            arguments = event_arguments.log_probability
        elif self.operation_name == "sampling":
            if functions.rng is None:
                raise RuntimeError(f"sampling benchmark is unavailable for {distribution_name!r}")
            function = functools.partial(
                functions.rng,
                sample_shape=profile.sample_shape,
            )
            arguments = (jax.random.key(0), *event_arguments.sampling_parameters)
        else:
            raise RuntimeError(f"unknown event benchmark operation {self.operation_name!r}")

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)
        self._case = case


class _DirichletBenchmark(_EventBenchmark):
    version = "2"
    params = ("vector", "likelihood", "channel_prior")
    param_names = ("profile",)

    def setup(self, profile_name: str) -> None:
        """Prepare and warm one compiled Dirichlet operation."""
        self._setup("dirichlet", profile_name)


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


class _MultinomialBenchmark(_EventBenchmark):
    version = "1"
    params = (
        ("multinomial", "multinomial_logit"),
        ("vector", "likelihood", "channel_prior"),
    )
    param_names = ("distribution", "profile")

    def setup(self, distribution_name: str, profile_name: str) -> None:
        """Prepare and warm one compiled Multinomial operation."""
        self._setup(distribution_name, profile_name)


class MultinomialLogProbability(_MultinomialBenchmark):
    """Measure elementwise Multinomial log-mass execution."""

    operation_name = "elementwise"

    def time_log_probability(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized elementwise log-mass call."""
        synchronize(self.compiled(*self.arguments))


class MultinomialLogDensity(_MultinomialBenchmark):
    """Measure summed Multinomial log-mass execution."""

    operation_name = "log_density"

    def time_log_density(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized summed log-mass call."""
        synchronize(self.compiled(*self.arguments))


class MultinomialLogDensityGradient(_MultinomialBenchmark):
    """Measure Multinomial parameter value-and-gradient execution."""

    operation_name = "value_and_grad"

    def time_value_and_grad(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized value-and-gradient call."""
        synchronize(self.compiled(*self.arguments))


class MultinomialSampling(_MultinomialBenchmark):
    """Measure synchronized Multinomial sampling execution."""

    operation_name = "sampling"

    def time_sampling(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized sampling call."""
        synchronize(self.compiled(*self.arguments))
