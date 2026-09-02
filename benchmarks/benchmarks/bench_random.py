"""Benchmark steady-state distribution sampling operations."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from benchmarks.cases import DISTRIBUTIONS, MMM_JAX_FUNCTIONS, PROFILES, make_parameters
from benchmarks.common import Arguments, compile_and_warm, synchronize


class Sampling:
    """Measure synchronized random sampling execution."""

    params = (
        tuple(distribution.name for distribution in DISTRIBUTIONS),
        ("vector", "likelihood", "channel_prior"),
    )
    param_names = ("distribution", "profile")

    arguments: Arguments
    compiled: Callable[..., object]

    def setup(self, distribution_name: str, profile_name: str) -> None:
        """Prepare and warm one compiled sampling operation."""
        distribution = next(distribution for distribution in DISTRIBUTIONS if distribution.name == distribution_name)
        profile = PROFILES[profile_name]
        parameters = make_parameters(
            distribution,
            profile,
            jnp.dtype(jnp.float32),
        )

        # The caller owns key advancement, so a fixed key isolates the sampler itself
        function = functools.partial(
            MMM_JAX_FUNCTIONS[distribution_name].rng,
            sample_shape=profile.sample_shape,
        )
        arguments = (jax.random.key(0), *parameters)

        self.arguments = arguments
        self.compiled = compile_and_warm(function, self.arguments)

    def time_sampling(self, distribution_name: str, profile_name: str) -> None:
        """Measure one synchronized sampling call."""
        synchronize(self.compiled(*self.arguments))
