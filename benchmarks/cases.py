"""Define distribution benchmark cases."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mmmjax.distributions import (
    bernoulli,
    bernoulli_logit,
    bernoulli_logit_logpmf,
    bernoulli_logit_rng,
    bernoulli_logpmf,
    bernoulli_rng,
    beta,
    beta_logpdf,
    beta_rng,
    binomial,
    binomial_logit,
    binomial_logit_logpmf,
    binomial_logit_rng,
    binomial_logpmf,
    binomial_rng,
    categorical,
    categorical_logit,
    categorical_logit_logpmf,
    categorical_logit_rng,
    categorical_logpmf,
    categorical_rng,
    cauchy,
    cauchy_logpdf,
    cauchy_rng,
    exponential,
    exponential_logcdf,
    exponential_logpdf,
    exponential_logsf,
    exponential_rng,
    gamma,
    gamma_logcdf,
    gamma_logpdf,
    gamma_logsf,
    gamma_rng,
    half_normal,
    half_normal_logcdf,
    half_normal_logpdf,
    half_normal_logsf,
    half_normal_rng,
    inverse_gamma,
    inverse_gamma_logcdf,
    inverse_gamma_logpdf,
    inverse_gamma_logsf,
    inverse_gamma_rng,
    laplace,
    laplace_logcdf,
    laplace_logpdf,
    laplace_logsf,
    laplace_rng,
    lognormal,
    lognormal_logcdf,
    lognormal_logpdf,
    lognormal_logsf,
    lognormal_rng,
    negative_binomial,
    negative_binomial_log,
    negative_binomial_log_logpmf,
    negative_binomial_log_rng,
    negative_binomial_logpmf,
    negative_binomial_rng,
    normal,
    normal_logcdf,
    normal_logpdf,
    normal_logsf,
    normal_rng,
    poisson,
    poisson_log,
    poisson_log_logpmf,
    poisson_log_rng,
    poisson_logpmf,
    poisson_rng,
    student_t,
    student_t_logpdf,
    student_t_rng,
    uniform,
    uniform_logcdf,
    uniform_logpdf,
    uniform_logsf,
    uniform_rng,
)

from .common import Arguments

Kernel = Callable[..., jax.Array]


@dataclass(frozen=True)
class BenchmarkProfile:
    """Describe one distribution workload."""

    value_shape: tuple[int, ...]
    parameter_shape: tuple[int, ...]
    sample_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate that observed values and RNG draws use the same shape."""
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

    elementwise_log_probability: Kernel
    summed_log_probability: Kernel
    rng: Kernel
    logcdf: Kernel | None = None
    logsf: Kernel | None = None


@dataclass(frozen=True)
class DistributionSpec:
    """Describe benchmark inputs for a distribution."""

    name: str
    value_range: tuple[float, float] | None
    parameter_values: tuple[float | tuple[float, ...], ...]
    log_probability_operation: str = "logpdf"
    outcomes: tuple[int, ...] = ()
    supports_concentrated_inputs: bool = False
    supports_tail_inputs: bool = False
    gradient_parameter_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the operation name and value source."""
        if self.log_probability_operation not in {"logpdf", "logpmf"}:
            raise ValueError(
                f"log_probability_operation must be 'logpdf' or 'logpmf', got {self.log_probability_operation!r}"
            )
        has_value_range = self.value_range is not None
        has_outcomes = bool(self.outcomes)
        if has_value_range == has_outcomes:
            raise ValueError("exactly one of value_range or outcomes must define benchmark values")
        if self.gradient_parameter_indices is not None:
            if not self.gradient_parameter_indices:
                raise ValueError("gradient_parameter_indices must contain at least one parameter index")
            if len(set(self.gradient_parameter_indices)) != len(self.gradient_parameter_indices):
                raise ValueError("gradient_parameter_indices must not contain duplicate indices")
            invalid_indices = tuple(
                index for index in self.gradient_parameter_indices if index < 0 or index >= len(self.parameter_values)
            )
            if invalid_indices:
                raise ValueError(
                    "gradient_parameter_indices must refer to parameter_values, "
                    f"got invalid indices {invalid_indices} for {len(self.parameter_values)} parameters"
                )

    @property
    def gradient_argnums(self) -> tuple[int, ...]:
        """Return the callable argument positions used for parameter gradients."""
        # The first argument is observed data and remains fixed during model inference
        if self.gradient_parameter_indices is None:
            return tuple(range(1, len(self.parameter_values) + 1))
        return tuple(index + 1 for index in self.gradient_parameter_indices)


PROFILES: dict[str, BenchmarkProfile] = {
    # A small vector that checks scalar parameter broadcasting
    "vector": BenchmarkProfile(
        value_shape=(32,),
        parameter_shape=(),
        sample_shape=(32,),
    ),
    # A grouped workload with one parameter per group
    "likelihood": BenchmarkProfile(
        value_shape=(260, 8),
        parameter_shape=(8,),
        sample_shape=(260,),
    ),
    # A wide workload with parameters shared across leading batches
    "channel_prior": BenchmarkProfile(
        value_shape=(8, 465),
        parameter_shape=(465,),
        sample_shape=(8,),
    ),
    # A large nested workload that remains opt-in because it has nearly one million terms
    "stress": BenchmarkProfile(
        value_shape=(260, 8, 465),
        parameter_shape=(8, 465),
        sample_shape=(260,),
    ),
}

DISTRIBUTIONS = (
    DistributionSpec(
        name="bernoulli",
        value_range=None,
        parameter_values=(0.35,),
        log_probability_operation="logpmf",
        outcomes=(0, 1),
    ),
    DistributionSpec(
        name="bernoulli_logit",
        value_range=None,
        parameter_values=(-0.6,),
        log_probability_operation="logpmf",
        outcomes=(0, 1),
    ),
    DistributionSpec(
        name="beta",
        value_range=(0.05, 0.95),
        parameter_values=(2.5, 3.5),
        supports_concentrated_inputs=True,
    ),
    # Trial counts are observed data, so only probability parameters are differentiated
    DistributionSpec(
        name="binomial",
        value_range=None,
        parameter_values=(100.0, 0.35),
        log_probability_operation="logpmf",
        outcomes=(0, 20, 35, 50, 100),
        gradient_parameter_indices=(1,),
    ),
    DistributionSpec(
        name="binomial_logit",
        value_range=None,
        parameter_values=(100.0, -0.6),
        log_probability_operation="logpmf",
        outcomes=(0, 20, 35, 50, 100),
        gradient_parameter_indices=(1,),
    ),
    DistributionSpec(
        name="categorical",
        value_range=None,
        parameter_values=((0.1, 0.2, 0.3, 0.4),),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 2, 3),
    ),
    DistributionSpec(
        name="categorical_logit",
        value_range=None,
        parameter_values=((-1.0, 0.0, 0.5, 1.0),),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 2, 3),
    ),
    DistributionSpec(
        name="negative_binomial",
        value_range=None,
        parameter_values=(4.5, 2.5),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 3, 10, 25),
    ),
    DistributionSpec(
        name="negative_binomial_log",
        value_range=None,
        parameter_values=(1.5, 2.5),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 3, 10, 25),
    ),
    DistributionSpec(
        name="poisson",
        value_range=None,
        parameter_values=(4.5,),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 3, 10, 25),
        supports_concentrated_inputs=True,
    ),
    DistributionSpec(
        name="poisson_log",
        value_range=None,
        parameter_values=(1.5,),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 3, 10, 25),
        supports_concentrated_inputs=True,
    ),
    DistributionSpec(
        name="cauchy",
        value_range=(-2.0, 2.0),
        parameter_values=(0.2, 1.3),
    ),
    DistributionSpec(
        name="exponential",
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="gamma",
        value_range=(0.1, 2.0),
        parameter_values=(2.5, 1.3),
        supports_concentrated_inputs=True,
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="half_normal",
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="inverse_gamma",
        value_range=(0.1, 2.0),
        parameter_values=(3.5, 1.2),
        supports_concentrated_inputs=True,
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="laplace",
        value_range=(-1.0, 1.0),
        parameter_values=(0.2, 1.3),
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="lognormal",
        value_range=(0.1, 2.0),
        parameter_values=(0.2, 0.8),
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="normal",
        value_range=(-1.0, 1.0),
        parameter_values=(0.2, 1.3),
        supports_tail_inputs=True,
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
        supports_tail_inputs=True,
    ),
)

MMM_JAX_FUNCTIONS: dict[str, DistributionFunctions] = {
    "bernoulli": DistributionFunctions(bernoulli_logpmf, bernoulli, bernoulli_rng),
    "bernoulli_logit": DistributionFunctions(
        bernoulli_logit_logpmf,
        bernoulli_logit,
        bernoulli_logit_rng,
    ),
    "beta": DistributionFunctions(beta_logpdf, beta, beta_rng),
    "binomial": DistributionFunctions(binomial_logpmf, binomial, binomial_rng),
    "binomial_logit": DistributionFunctions(
        binomial_logit_logpmf,
        binomial_logit,
        binomial_logit_rng,
    ),
    "categorical": DistributionFunctions(categorical_logpmf, categorical, categorical_rng),
    "categorical_logit": DistributionFunctions(
        categorical_logit_logpmf,
        categorical_logit,
        categorical_logit_rng,
    ),
    "negative_binomial": DistributionFunctions(
        negative_binomial_logpmf,
        negative_binomial,
        negative_binomial_rng,
    ),
    "negative_binomial_log": DistributionFunctions(
        negative_binomial_log_logpmf,
        negative_binomial_log,
        negative_binomial_log_rng,
    ),
    "poisson": DistributionFunctions(poisson_logpmf, poisson, poisson_rng),
    "poisson_log": DistributionFunctions(
        poisson_log_logpmf,
        poisson_log,
        poisson_log_rng,
    ),
    "cauchy": DistributionFunctions(cauchy_logpdf, cauchy, cauchy_rng),
    "exponential": DistributionFunctions(
        exponential_logpdf,
        exponential,
        exponential_rng,
        logcdf=exponential_logcdf,
        logsf=exponential_logsf,
    ),
    "gamma": DistributionFunctions(
        gamma_logpdf,
        gamma,
        gamma_rng,
        logcdf=gamma_logcdf,
        logsf=gamma_logsf,
    ),
    "half_normal": DistributionFunctions(
        half_normal_logpdf,
        half_normal,
        half_normal_rng,
        logcdf=half_normal_logcdf,
        logsf=half_normal_logsf,
    ),
    "inverse_gamma": DistributionFunctions(
        inverse_gamma_logpdf,
        inverse_gamma,
        inverse_gamma_rng,
        logcdf=inverse_gamma_logcdf,
        logsf=inverse_gamma_logsf,
    ),
    "laplace": DistributionFunctions(
        laplace_logpdf,
        laplace,
        laplace_rng,
        logcdf=laplace_logcdf,
        logsf=laplace_logsf,
    ),
    "lognormal": DistributionFunctions(
        lognormal_logpdf,
        lognormal,
        lognormal_rng,
        logcdf=lognormal_logcdf,
        logsf=lognormal_logsf,
    ),
    "normal": DistributionFunctions(
        normal_logpdf,
        normal,
        normal_rng,
        logcdf=normal_logcdf,
        logsf=normal_logsf,
    ),
    "student_t": DistributionFunctions(student_t_logpdf, student_t, student_t_rng),
    "uniform": DistributionFunctions(
        uniform_logpdf,
        uniform,
        uniform_rng,
        logcdf=uniform_logcdf,
        logsf=uniform_logsf,
    ),
}

DISTRIBUTIONS_BY_NAME = {distribution.name: distribution for distribution in DISTRIBUTIONS}
TAIL_DISTRIBUTIONS = frozenset(distribution.name for distribution in DISTRIBUTIONS if distribution.supports_tail_inputs)


def _validate_distribution_cases() -> None:
    """Validate the relationships between benchmark specifications and functions."""
    distribution_names = tuple(distribution.name for distribution in DISTRIBUTIONS)
    if len(DISTRIBUTIONS_BY_NAME) != len(DISTRIBUTIONS):
        duplicate_names = tuple(sorted({name for name in distribution_names if distribution_names.count(name) > 1}))
        raise ValueError(f"distribution benchmark names must be unique, got duplicates {duplicate_names}")

    specification_names = set(DISTRIBUTIONS_BY_NAME)
    function_names = set(MMM_JAX_FUNCTIONS)
    if specification_names != function_names:
        missing_functions = tuple(sorted(specification_names - function_names))
        missing_specifications = tuple(sorted(function_names - specification_names))
        raise ValueError(
            "distribution specifications and functions must use the same names, "
            f"got specifications without functions {missing_functions} "
            f"and functions without specifications {missing_specifications}"
        )

    missing_tail_functions = tuple(
        name
        for name in sorted(TAIL_DISTRIBUTIONS)
        if MMM_JAX_FUNCTIONS[name].logcdf is None or MMM_JAX_FUNCTIONS[name].logsf is None
    )
    if missing_tail_functions:
        raise ValueError(
            "tail benchmark distributions must define both logcdf and logsf functions, "
            f"got incomplete functions for {missing_tail_functions}"
        )


_validate_distribution_cases()


def make_arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    input_set: str,
    dtype: jnp.dtype,
) -> Arguments:
    """Build one profile's values and distribution parameters."""
    if input_set not in {"ordinary", "concentrated"}:
        raise ValueError(f"density benchmarks do not support {input_set} inputs")

    element_count = math.prod(profile.value_shape)
    if input_set == "ordinary":
        if distribution.outcomes:
            outcomes = jnp.asarray(distribution.outcomes, dtype=jnp.int32)
            sample_count = math.prod(profile.sample_shape)
            parameter_count = math.prod(profile.parameter_shape)
            sample_indices = jnp.arange(sample_count, dtype=jnp.int32).reshape(
                profile.sample_shape + (1,) * len(profile.parameter_shape)
            )
            parameter_indices = jnp.arange(parameter_count, dtype=jnp.int32).reshape(
                (1,) * len(profile.sample_shape) + profile.parameter_shape
            )
            outcome_indices = (sample_indices + parameter_indices) % len(distribution.outcomes)
            value = outcomes[outcome_indices]
        else:
            if distribution.value_range is None:
                raise ValueError(f"{distribution.name} does not define ordinary benchmark values")
            lower, upper = distribution.value_range
            value = jnp.linspace(lower, upper, element_count, dtype=dtype).reshape(profile.value_shape)
        parameters = make_parameters(distribution, profile, dtype)
        return (value, *parameters)

    if not distribution.supports_concentrated_inputs:
        raise ValueError(f"{distribution.name} does not define concentrated benchmark inputs")

    concentration = 10_000_000 if dtype == jnp.dtype(jnp.float32) else 1_000_000_000_000_000
    if distribution.name in {"poisson", "poisson_log"}:
        count_dtype = jnp.int32 if dtype == jnp.dtype(jnp.float32) else jnp.int64
        standard_deviation = math.isqrt(concentration)
        value = jnp.rint(
            jnp.linspace(
                concentration - 2 * standard_deviation,
                concentration + 2 * standard_deviation,
                element_count,
                dtype=dtype,
            )
        ).astype(count_dtype)
        value = value.reshape(profile.value_shape)
        rate = jnp.full(profile.parameter_shape, concentration, dtype=dtype)
        parameter = jnp.log(rate) if distribution.name == "poisson_log" else rate
        return value, parameter

    concentrated_shape = jnp.asarray(concentration, dtype=dtype)
    displacement = jnp.linspace(-1e-4, 1e-4, element_count, dtype=dtype).reshape(profile.value_shape)
    shape = jnp.full(profile.parameter_shape, concentrated_shape, dtype=dtype)
    if distribution.name == "beta":
        return jnp.asarray(0.5, dtype=dtype) + displacement, shape, shape

    value = jnp.asarray(1, dtype=dtype) + displacement
    return value, shape, shape


def make_tail_arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    input_set: str,
    operation: str,
    dtype: jnp.dtype,
) -> Arguments:
    """Build ordinary or tail inputs for a log-CDF or log-survival operation."""
    if distribution.name not in TAIL_DISTRIBUTIONS:
        raise ValueError(f"{distribution.name} does not define log-CDF or log-survival benchmark inputs")
    if input_set not in {"ordinary", "tail"}:
        raise ValueError(f"log-CDF and log-survival benchmarks do not support {input_set} inputs")
    if operation not in {"logcdf", "logsf"}:
        raise ValueError(f"operation must be 'logcdf' or 'logsf', got {operation!r}")

    element_count = math.prod(profile.value_shape)
    if distribution.name == "exponential":
        # Rate-scaled inputs keep the probability range fixed if benchmark rates change
        (rate,) = make_parameters(distribution, profile, dtype)
        if input_set == "ordinary":
            scaled_value = jnp.linspace(0.1, 3.0, element_count, dtype=dtype)
        else:
            negative_log_probability = jnp.linspace(4.0, 32.0, element_count, dtype=dtype)
            if operation == "logcdf":
                scaled_value = -jnp.log1p(-jnp.exp(-negative_log_probability))
            else:
                scaled_value = negative_log_probability

        value = scaled_value.reshape(profile.value_shape) / rate
        return value, rate

    if distribution.name == "gamma":
        shape, rate = make_parameters(distribution, profile, dtype)
        if input_set == "ordinary":
            # Covers roughly the central 4% through 97% for the configured shape of 2.5
            scaled_value = jnp.linspace(0.5, 6.0, element_count, dtype=dtype)
        elif operation == "logcdf":
            # Geometric spacing spreads lower-tail log probabilities from about -32 to -4
            scaled_value = jnp.geomspace(5e-6, 0.35, element_count, dtype=dtype)
        else:
            # Linear spacing gives a similar spread in the upper tail
            scaled_value = jnp.linspace(6.8, 37.2, element_count, dtype=dtype)

        value = scaled_value.reshape(profile.value_shape) / rate
        return value, shape, rate

    if distribution.name == "half_normal":
        (scale,) = make_parameters(distribution, profile, dtype)
        if input_set == "ordinary":
            standardized = jnp.linspace(0.023, 2.36, element_count, dtype=dtype)
        elif operation == "logcdf":
            standardized = jnp.geomspace(1.6e-14, 0.023, element_count, dtype=dtype)
        else:
            standardized = jnp.linspace(2.36, 7.71, element_count, dtype=dtype)

        value = scale * standardized.reshape(profile.value_shape)
        return value, scale

    if distribution.name == "inverse_gamma":
        shape, scale = make_parameters(distribution, profile, dtype)
        if input_set == "ordinary":
            # Scaling by scale / value gives the corresponding unit-rate Gamma argument
            scaled_inverse_value = jnp.linspace(0.9, 7.35, element_count, dtype=dtype)
        elif operation == "logcdf":
            # The Inverse Gamma CDF maps to the Gamma upper tail
            scaled_inverse_value = jnp.linspace(8.43, 40.1, element_count, dtype=dtype)
        else:
            # The Inverse Gamma survival function maps to the Gamma lower tail
            scaled_inverse_value = jnp.geomspace(2e-4, 0.76, element_count, dtype=dtype)

        value = scale / scaled_inverse_value.reshape(profile.value_shape)
        return value, shape, scale

    if distribution.name == "laplace" and input_set == "tail":
        location, scale = make_parameters(distribution, profile, dtype)
        negative_log_probability = jnp.linspace(4.0, 32.0, element_count, dtype=dtype)
        log_two = jnp.asarray(math.log(2), dtype=dtype)

        if operation == "logcdf":
            standardized = log_two - negative_log_probability
        else:
            standardized = negative_log_probability - log_two

        value = location + scale * standardized.reshape(profile.value_shape)
        return value, location, scale

    if distribution.name == "uniform":
        if input_set == "ordinary":
            lower, upper = make_parameters(distribution, profile, dtype)
            if distribution.value_range is None:
                raise ValueError("uniform does not define ordinary benchmark values")
            value = jnp.linspace(
                *distribution.value_range,
                element_count,
                dtype=dtype,
            ).reshape(profile.value_shape)
        else:
            negative_log_probability = jnp.linspace(4.0, 32.0, element_count, dtype=dtype)
            tail_probability = jnp.exp(-negative_log_probability).reshape(profile.value_shape)

            # Anchoring the evaluated endpoint at zero keeps deep tail distances representable
            if operation == "logcdf":
                lower = jnp.zeros(profile.parameter_shape, dtype=dtype)
                upper = jnp.ones(profile.parameter_shape, dtype=dtype)
                value = tail_probability
            else:
                lower = -jnp.ones(profile.parameter_shape, dtype=dtype)
                upper = jnp.zeros(profile.parameter_shape, dtype=dtype)
                value = -tail_probability

        return value, lower, upper

    if input_set == "ordinary":
        standardized_lower, standardized_upper = -2.0, 2.0
    elif operation == "logcdf":
        standardized_lower, standardized_upper = -8.0, -2.0
    else:
        standardized_lower, standardized_upper = 2.0, 8.0

    standardized = jnp.linspace(
        standardized_lower,
        standardized_upper,
        element_count,
        dtype=dtype,
    ).reshape(profile.value_shape)
    location, scale = make_parameters(distribution, profile, dtype)
    transformed = location + scale * standardized
    value = jnp.exp(transformed) if distribution.name == "lognormal" else transformed
    return value, location, scale


def make_parameters(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    dtype: jnp.dtype,
) -> tuple[jax.Array, ...]:
    """Build broadcast parameters for one distribution workload."""
    parameters = []
    for parameter_value in distribution.parameter_values:
        parameter = jnp.asarray(parameter_value, dtype=dtype)
        # Vector parameters keep their event axes after the benchmark batch axes
        parameters.append(jnp.broadcast_to(parameter, profile.parameter_shape + parameter.shape))
    return tuple(parameters)
