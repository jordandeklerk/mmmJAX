"""Define distribution benchmark workloads."""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from benchmarks._timing import Arguments, BenchmarkFunction
from benchmarks.references import JAX_REFERENCES
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
    parameter_values: tuple[float, ...]
    log_probability_operation: str = "logpdf"
    outcomes: tuple[int, ...] = ()
    supports_concentrated_inputs: bool = False
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


@dataclass(frozen=True)
class BenchmarkOperation:
    """Describe one compiled operation and its arguments."""

    implementation: str
    name: str
    function: BenchmarkFunction
    arguments: Arguments


PROFILES: dict[str, BenchmarkProfile] = {
    # A small vector that checks scalar parameter broadcasting
    "vector": BenchmarkProfile(
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
    ),
    DistributionSpec(
        name="gamma",
        value_range=(0.1, 2.0),
        parameter_values=(2.5, 1.3),
        supports_concentrated_inputs=True,
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
        supports_concentrated_inputs=True,
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
    },
    "jax": {
        name: DistributionFunctions(
            reference.elementwise_log_probability,
            reference.summed_log_probability,
            reference.rng,
            logcdf=reference.logcdf,
            logsf=reference.logsf,
        )
        for name, reference in JAX_REFERENCES.items()
    },
}

DEFAULT_OPERATIONS = ("logpdf", "logpmf", "log_density", "value_and_grad", "rng")
LOG_PROBABILITY_OPERATIONS = (
    "logcdf",
    "logcdf_value_and_grad",
    "logsf",
    "logsf_value_and_grad",
)
OPERATIONS = DEFAULT_OPERATIONS + LOG_PROBABILITY_OPERATIONS
INPUT_SETS = ("ordinary", "concentrated", "tail")
LOG_PROBABILITY_DISTRIBUTIONS = frozenset(
    {"exponential", "gamma", "half_normal", "inverse_gamma", "laplace", "lognormal", "normal", "uniform"}
)


def make_arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    input_set: str,
    dtype: jnp.dtype,
) -> Arguments:
    """Build one profile's values and distribution parameters."""
    if input_set not in {"ordinary", "concentrated"}:
        raise ValueError(f"log-probability benchmarks do not support {input_set} inputs")

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
        parameters = _make_parameters(distribution, profile, dtype)
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


def make_log_probability_arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    input_set: str,
    operation: str,
    dtype: jnp.dtype,
) -> Arguments:
    """Build ordinary or tail inputs for a log-CDF or log-survival operation."""
    if distribution.name not in LOG_PROBABILITY_DISTRIBUTIONS:
        raise ValueError(f"{distribution.name} does not define log-CDF or log-survival benchmark inputs")
    if input_set not in {"ordinary", "tail"}:
        raise ValueError(f"log-CDF and log-survival benchmarks do not support {input_set} inputs")
    if operation not in {"logcdf", "logsf"}:
        raise ValueError(f"operation must be 'logcdf' or 'logsf', got {operation!r}")

    element_count = math.prod(profile.value_shape)
    if distribution.name == "exponential":
        # Rate-scaled inputs keep the probability range fixed if benchmark rates change
        (rate,) = _make_parameters(distribution, profile, dtype)
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
        shape, rate = _make_parameters(distribution, profile, dtype)
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
        (scale,) = _make_parameters(distribution, profile, dtype)
        if input_set == "ordinary":
            standardized = jnp.linspace(0.023, 2.36, element_count, dtype=dtype)
        elif operation == "logcdf":
            standardized = jnp.geomspace(1.6e-14, 0.023, element_count, dtype=dtype)
        else:
            standardized = jnp.linspace(2.36, 7.71, element_count, dtype=dtype)

        value = scale * standardized.reshape(profile.value_shape)
        return value, scale

    if distribution.name == "inverse_gamma":
        shape, scale = _make_parameters(distribution, profile, dtype)
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
        location, scale = _make_parameters(distribution, profile, dtype)
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
            lower, upper = _make_parameters(distribution, profile, dtype)
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
        lower, upper = -2.0, 2.0
    elif operation == "logcdf":
        lower, upper = -8.0, -2.0
    else:
        lower, upper = 2.0, 8.0

    standardized = jnp.linspace(lower, upper, element_count, dtype=dtype).reshape(profile.value_shape)
    location, scale = _make_parameters(distribution, profile, dtype)
    transformed = location + scale * standardized
    value = jnp.exp(transformed) if distribution.name == "lognormal" else transformed
    return value, location, scale


def make_operations(
    functions: DistributionFunctions,
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    arguments: Arguments,
    implementation: str,
) -> tuple[BenchmarkOperation, ...]:
    """Build elementwise, summed, gradient, and sampling operations for one implementation."""
    # Observed values are data, so model inference only differentiates parameters
    if distribution.gradient_parameter_indices is None:
        parameter_indices = tuple(range(1, len(arguments)))
    else:
        parameter_indices = tuple(index + 1 for index in distribution.gradient_parameter_indices)
    parameters = arguments[1:]
    return (
        BenchmarkOperation(
            implementation,
            distribution.log_probability_operation,
            functions.elementwise_log_probability,
            arguments,
        ),
        BenchmarkOperation(
            implementation,
            "log_density",
            functions.summed_log_probability,
            arguments,
        ),
        BenchmarkOperation(
            implementation,
            "value_and_grad",
            jax.value_and_grad(functions.summed_log_probability, argnums=parameter_indices),
            arguments,
        ),
        BenchmarkOperation(
            implementation,
            "rng",
            functools.partial(functions.rng, sample_shape=profile.sample_shape),
            (jax.random.key(0), *parameters),
        ),
    )


def make_log_probability_operations(
    functions: DistributionFunctions,
    arguments: Arguments,
    implementation: str,
    operation: str,
) -> tuple[BenchmarkOperation, ...]:
    """Build one log-probability operation and its parameter gradient."""
    if operation not in {"logcdf", "logsf"}:
        raise ValueError(f"operation must be 'logcdf' or 'logsf', got {operation!r}")

    function = functions.logcdf if operation == "logcdf" else functions.logsf
    if function is None:
        return ()

    parameter_indices = tuple(range(1, len(arguments)))
    summed_function = functools.partial(_sum_values, function)
    return (
        BenchmarkOperation(implementation, operation, function, arguments),
        BenchmarkOperation(
            implementation,
            f"{operation}_value_and_grad",
            jax.value_and_grad(summed_function, argnums=parameter_indices),
            arguments,
        ),
    )


def make_benchmark_operation(
    functions: DistributionFunctions,
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    *,
    input_set: str,
    operation: str,
    dtype: jnp.dtype,
    implementation: str,
) -> BenchmarkOperation | None:
    """Build one supported operation for a distribution workload."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown benchmark operation {operation!r}")
    if input_set not in INPUT_SETS:
        raise ValueError(f"unknown benchmark input set {input_set!r}")

    if operation in DEFAULT_OPERATIONS:
        if input_set == "tail" or (input_set == "concentrated" and not distribution.supports_concentrated_inputs):
            return None
        if input_set == "concentrated" and implementation == "jax":
            return None
        if input_set == "concentrated" and operation == "rng" and distribution.name in {"poisson", "poisson_log"}:
            return None

        arguments = make_arguments(distribution, profile, input_set, dtype)
        candidates = make_operations(functions, distribution, profile, arguments, implementation)
    else:
        if input_set == "concentrated" or distribution.name not in LOG_PROBABILITY_DISTRIBUTIONS:
            return None

        log_probability = "logcdf" if operation.startswith("logcdf") else "logsf"
        arguments = make_log_probability_arguments(distribution, profile, input_set, log_probability, dtype)
        candidates = make_log_probability_operations(functions, arguments, implementation, log_probability)

    return next((candidate for candidate in candidates if candidate.name == operation), None)


def _make_parameters(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    dtype: jnp.dtype,
) -> tuple[jax.Array, ...]:
    return tuple(
        jnp.full(profile.parameter_shape, parameter, dtype=dtype) for parameter in distribution.parameter_values
    )


def _sum_values(function: Kernel, *arguments: jax.Array) -> jax.Array:
    return jnp.sum(function(*arguments))
