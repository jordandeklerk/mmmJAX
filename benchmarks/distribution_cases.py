"""Workload definitions for distribution benchmarks."""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from benchmarks._timing import Arguments, BenchmarkFunction
from benchmarks.jax_references import JAX_REFERENCES
from mmmjax.distributions import (
    beta,
    beta_logpdf,
    beta_rng,
    exponential,
    exponential_logpdf,
    exponential_rng,
    gamma,
    gamma_logpdf,
    gamma_rng,
    half_normal,
    half_normal_logpdf,
    half_normal_rng,
    inverse_gamma,
    inverse_gamma_logpdf,
    inverse_gamma_rng,
    laplace,
    laplace_logpdf,
    laplace_rng,
    lognormal,
    lognormal_logpdf,
    lognormal_rng,
    normal,
    normal_logpdf,
    normal_rng,
    student_t,
    student_t_logpdf,
    student_t_rng,
    uniform,
    uniform_logpdf,
    uniform_rng,
)

Kernel = Callable[..., jax.Array]


@dataclass(frozen=True)
class BenchmarkProfile:
    """Describe one distribution workload."""

    value_shape: tuple[int, ...]
    parameter_shape: tuple[int, ...]
    sample_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate that density values and RNG draws use the same shape."""
        expected_value_shape = self.sample_shape + self.parameter_shape
        if self.value_shape != expected_value_shape:
            raise ValueError(
                "value_shape must equal sample_shape + parameter_shape, "
                f"got value_shape={self.value_shape}, sample_shape={self.sample_shape}, "
                f"and parameter_shape={self.parameter_shape}"
            )


@dataclass(frozen=True)
class DistributionFunctions:
    """Store one implementation of the public distribution operations."""

    logpdf: Kernel
    density: Kernel
    rng: Kernel


@dataclass(frozen=True)
class DistributionSpec:
    """Describe the ordinary inputs and supported regimes for a distribution."""

    name: str
    value_range: tuple[float, float]
    parameter_values: tuple[float, ...]
    supports_concentrated_regime: bool = False


@dataclass(frozen=True)
class BenchmarkOperation:
    """Describe one compiled operation and its arguments."""

    implementation: str
    name: str
    function: BenchmarkFunction
    arguments: Arguments


PROFILES: dict[str, BenchmarkProfile] = {
    # A quick run that still checks scalar parameter broadcasting
    "smoke": BenchmarkProfile(
        value_shape=(32,),
        parameter_shape=(),
        sample_shape=(32,),
    ),
    # Five years of weekly observations across eight geos
    "likelihood": BenchmarkProfile(
        value_shape=(260, 8),
        parameter_shape=(8,),
        sample_shape=(260,),
    ),
    # Geo-level parameters for 465 channels with channel-wise hyperparameters
    "channel_prior": BenchmarkProfile(
        value_shape=(8, 465),
        parameter_shape=(465,),
        sample_shape=(8,),
    ),
    # Weekly geo-channel values remain opt-in because this is nearly one million terms
    "stress": BenchmarkProfile(
        value_shape=(260, 8, 465),
        parameter_shape=(8, 465),
        sample_shape=(260,),
    ),
}

DISTRIBUTIONS = (
    DistributionSpec(
        name="beta",
        value_range=(0.05, 0.95),
        parameter_values=(2.5, 3.5),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="exponential",
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
    ),
    DistributionSpec(
        name="gamma",
        value_range=(0.1, 2.0),
        parameter_values=(2.5, 1.3),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="half_normal",
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
    ),
    DistributionSpec(
        name="inverse_gamma",
        value_range=(0.1, 2.0),
        parameter_values=(3.5, 1.2),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="laplace",
        value_range=(-1.0, 1.0),
        parameter_values=(0.2, 1.3),
    ),
    DistributionSpec(
        name="lognormal",
        value_range=(0.1, 2.0),
        parameter_values=(0.2, 0.8),
    ),
    DistributionSpec(
        name="normal",
        value_range=(-1.0, 1.0),
        parameter_values=(0.2, 1.3),
    ),
    DistributionSpec(
        name="student_t",
        value_range=(-1.0, 1.0),
        parameter_values=(5.0, 0.2, 1.3),
    ),
    DistributionSpec(
        name="uniform",
        value_range=(-0.5, 0.5),
        parameter_values=(-1.0, 1.0),
    ),
)

IMPLEMENTATIONS: dict[str, dict[str, DistributionFunctions]] = {
    "mmmjax": {
        "beta": DistributionFunctions(beta_logpdf, beta, beta_rng),
        "exponential": DistributionFunctions(exponential_logpdf, exponential, exponential_rng),
        "gamma": DistributionFunctions(gamma_logpdf, gamma, gamma_rng),
        "half_normal": DistributionFunctions(half_normal_logpdf, half_normal, half_normal_rng),
        "inverse_gamma": DistributionFunctions(inverse_gamma_logpdf, inverse_gamma, inverse_gamma_rng),
        "laplace": DistributionFunctions(laplace_logpdf, laplace, laplace_rng),
        "lognormal": DistributionFunctions(lognormal_logpdf, lognormal, lognormal_rng),
        "normal": DistributionFunctions(normal_logpdf, normal, normal_rng),
        "student_t": DistributionFunctions(student_t_logpdf, student_t, student_t_rng),
        "uniform": DistributionFunctions(uniform_logpdf, uniform, uniform_rng),
    },
    "jax": {
        name: DistributionFunctions(reference.logpdf, reference.density, reference.rng)
        for name, reference in JAX_REFERENCES.items()
    },
}

OPERATIONS = ("logpdf", "density", "value_and_grad", "rng")


def make_arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    regime: str,
    dtype: jnp.dtype,
) -> Arguments:
    """Build one profile's values and distribution parameters."""
    element_count = math.prod(profile.value_shape)
    if regime == "ordinary":
        lower, upper = distribution.value_range
        value = jnp.linspace(lower, upper, element_count, dtype=dtype).reshape(profile.value_shape)
        parameters = tuple(
            jnp.full(profile.parameter_shape, parameter, dtype=dtype) for parameter in distribution.parameter_values
        )
        return (value, *parameters)

    if not distribution.supports_concentrated_regime:
        raise ValueError(f"{distribution.name} does not define a concentrated benchmark regime")

    concentrated_shape = jnp.asarray(1e8 if dtype == jnp.dtype(jnp.float32) else 1e12, dtype=dtype)
    displacement = jnp.linspace(-1e-4, 1e-4, element_count, dtype=dtype).reshape(profile.value_shape)
    shape = jnp.full(profile.parameter_shape, concentrated_shape, dtype=dtype)
    if distribution.name == "beta":
        return jnp.asarray(0.5, dtype=dtype) + displacement, shape, shape

    value = jnp.asarray(1, dtype=dtype) + displacement
    return value, shape, shape


def make_operations(
    functions: DistributionFunctions,
    profile: BenchmarkProfile,
    arguments: Arguments,
    implementation: str,
) -> tuple[BenchmarkOperation, ...]:
    """Build elementwise, summed, gradient, and sampling operations for one implementation."""
    # Observed values are data, so model inference only differentiates parameters
    parameter_indices = tuple(range(1, len(arguments)))
    parameters = arguments[1:]
    return (
        BenchmarkOperation(implementation, "logpdf", functions.logpdf, arguments),
        BenchmarkOperation(implementation, "density", functions.density, arguments),
        BenchmarkOperation(
            implementation,
            "value_and_grad",
            jax.value_and_grad(functions.density, argnums=parameter_indices),
            arguments,
        ),
        BenchmarkOperation(
            implementation,
            "rng",
            functools.partial(functions.rng, sample_shape=profile.sample_shape),
            (jax.random.key(0), *parameters),
        ),
    )
