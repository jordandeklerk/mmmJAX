"""Benchmark steady-state distribution gradients."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import DISTRIBUTIONS, MMM_JAX_FUNCTIONS, PROFILES, make_arguments
from benchmarks.common import Arguments, compile_and_warm, synchronize


class LogDensityGradient:
    """Measure parameter value-and-gradient execution."""

    params = (
        tuple(distribution.name for distribution in DISTRIBUTIONS),
        ("vector", "likelihood", "channel_prior"),
    )
    param_names = ("distribution", "profile")

    arguments: Arguments
    compiled: Callable[..., object]

    def setup(self, distribution_name: str, profile_name: str) -> None:
        """Prepare and warm one compiled log-density gradient."""
        distribution = next(distribution for distribution in DISTRIBUTIONS if distribution.name == distribution_name)
        arguments = make_arguments(
            distribution,
            PROFILES[profile_name],
            "ordinary",
            jnp.dtype(jnp.float32),
        )
        function = jax.value_and_grad(
            MMM_JAX_FUNCTIONS[distribution_name].summed_log_probability,
            argnums=distribution.gradient_argnums,
        )

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)

    def time_value_and_grad(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized value-and-gradient call."""
        synchronize(self.compiled(*self.arguments))
