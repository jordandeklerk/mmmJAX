"""Define distribution benchmark cases."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

import mmmjax.distributions as distribution_api

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
class EventProfile:
    """Describe a workload with a final event axis."""

    sample_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    event_size: int

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        """Return the batched vector parameter shape."""
        return (*self.batch_shape, self.event_size)

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the sampled event shape."""
        return (*self.sample_shape, *self.parameter_shape)


@dataclass(frozen=True)
class DistributionFunctions:
    """Store one implementation of the public distribution operations."""

    elementwise_log_probability: Kernel
    summed_log_probability: Kernel
    rng: Kernel | None
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
    supports_sampling: bool = True
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


@dataclass(frozen=True)
class EventDistributionSpec:
    """Describe benchmark metadata for an event distribution."""

    name: str
    log_probability_operation: str
    gradient_argnums: tuple[int, ...]


@dataclass(frozen=True)
class EventArguments:
    """Store density and sampling arguments for an event workload."""

    log_probability: Arguments
    sampling_parameters: Arguments


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

EVENT_PROFILES: dict[str, EventProfile] = {
    "vector": EventProfile(sample_shape=(), batch_shape=(), event_size=32),
    "likelihood": EventProfile(sample_shape=(260,), batch_shape=(8,), event_size=4),
    "channel_prior": EventProfile(sample_shape=(8,), batch_shape=(), event_size=465),
    "stress": EventProfile(sample_shape=(260,), batch_shape=(8,), event_size=465),
}

DISTRIBUTIONS = (
    DistributionSpec(
        name="bernoulli",
        value_range=None,
        parameter_values=(0.35,),
        log_probability_operation="logpmf",
        outcomes=(0, 1),
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="bernoulli_logit",
        value_range=None,
        parameter_values=(-0.6,),
        log_probability_operation="logpmf",
        outcomes=(0, 1),
        supports_tail_inputs=True,
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
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="poisson_log",
        value_range=None,
        parameter_values=(1.5,),
        log_probability_operation="logpmf",
        outcomes=(0, 1, 3, 10, 25),
        supports_concentrated_inputs=True,
        supports_tail_inputs=True,
    ),
    DistributionSpec(
        name="cauchy",
        value_range=(-2.0, 2.0),
        parameter_values=(0.2, 1.3),
        supports_tail_inputs=True,
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
        name="truncated_normal",
        value_range=(-0.5, 0.5),
        parameter_values=(0.2, 1.3, -1.0, 1.0),
    ),
    DistributionSpec(
        name="uniform",
        value_range=(-0.5, 0.5),
        parameter_values=(-1.0, 1.0),
        supports_tail_inputs=True,
    ),
)

EVENT_DISTRIBUTIONS = (
    EventDistributionSpec(
        name="dirichlet",
        log_probability_operation="logpdf",
        gradient_argnums=(1,),
    ),
    EventDistributionSpec(
        name="multinomial",
        log_probability_operation="logpmf",
        gradient_argnums=(1,),
    ),
    EventDistributionSpec(
        name="multinomial_logit",
        log_probability_operation="logpmf",
        gradient_argnums=(1,),
    ),
)


def _distribution_function(name: str) -> Kernel:
    """Resolve a public operation without breaking benchmarks for older revisions."""
    function = getattr(distribution_api, name, None)
    if function is not None:
        return cast(Kernel, function)

    def unavailable(*args: object, **kwargs: object) -> jax.Array:
        raise NotImplementedError(f"{name} is unavailable in this mmmJAX revision")

    return unavailable


MMM_JAX_FUNCTIONS: dict[str, DistributionFunctions] = {
    distribution.name: DistributionFunctions(
        _distribution_function(f"{distribution.name}_{distribution.log_probability_operation}"),
        _distribution_function(distribution.name),
        _distribution_function(f"{distribution.name}_rng") if distribution.supports_sampling else None,
        logcdf=(_distribution_function(f"{distribution.name}_logcdf") if distribution.supports_tail_inputs else None),
        logsf=_distribution_function(f"{distribution.name}_logsf") if distribution.supports_tail_inputs else None,
    )
    for distribution in DISTRIBUTIONS
}

EVENT_MMM_JAX_FUNCTIONS: dict[str, DistributionFunctions] = {
    distribution.name: DistributionFunctions(
        _distribution_function(f"{distribution.name}_{distribution.log_probability_operation}"),
        _distribution_function(distribution.name),
        _distribution_function(f"{distribution.name}_rng"),
    )
    for distribution in EVENT_DISTRIBUTIONS
}

DISTRIBUTIONS_BY_NAME = {distribution.name: distribution for distribution in DISTRIBUTIONS}
EVENT_DISTRIBUTIONS_BY_NAME = {distribution.name: distribution for distribution in EVENT_DISTRIBUTIONS}
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

    event_distribution_names = tuple(distribution.name for distribution in EVENT_DISTRIBUTIONS)
    if len(EVENT_DISTRIBUTIONS_BY_NAME) != len(EVENT_DISTRIBUTIONS):
        duplicate_names = tuple(
            sorted({name for name in event_distribution_names if event_distribution_names.count(name) > 1})
        )
        raise ValueError(f"event distribution benchmark names must be unique, got duplicates {duplicate_names}")

    overlapping_names = tuple(sorted(set(distribution_names).intersection(event_distribution_names)))
    if overlapping_names:
        raise ValueError(f"standard and event benchmark distribution names must be distinct, got {overlapping_names}")

    event_specification_names = set(event_distribution_names)
    event_function_names = set(EVENT_MMM_JAX_FUNCTIONS)
    if event_specification_names != event_function_names:
        missing_functions = tuple(sorted(event_specification_names - event_function_names))
        missing_specifications = tuple(sorted(event_function_names - event_specification_names))
        raise ValueError(
            "event distribution specifications and functions must use the same names, "
            f"got specifications without functions {missing_functions} "
            f"and functions without specifications {missing_specifications}"
        )

    profile_names = set(PROFILES)
    event_profile_names = set(EVENT_PROFILES)
    if profile_names != event_profile_names:
        missing_event_profiles = tuple(sorted(profile_names - event_profile_names))
        missing_standard_profiles = tuple(sorted(event_profile_names - profile_names))
        raise ValueError(
            "standard and event benchmark profiles must use the same names, "
            f"got standard profiles without event profiles {missing_event_profiles} "
            f"and event profiles without standard profiles {missing_standard_profiles}"
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
    if distribution.name in {"bernoulli", "bernoulli_logit"}:
        # Thresholds between the two outcomes exercise the parameter-dependent branch
        value = jnp.linspace(0.0, 0.9, element_count, dtype=dtype).reshape(profile.value_shape)
        if input_set == "ordinary":
            return (value, *make_parameters(distribution, profile, dtype))

        parameter_count = math.prod(profile.parameter_shape)
        if distribution.name == "bernoulli_logit":
            negative_log_probability = jnp.linspace(32.0, 4.0, parameter_count, dtype=dtype)
            logits = jnp.log(jnp.expm1(negative_log_probability))
            parameter = logits if operation == "logcdf" else -logits
        else:
            # Exponents 6..46 span log probabilities around -4..-32; cap the CDF
            # at the mantissa precision so subtracting the power of two stays exact
            maximum_exponent = min(46, jnp.finfo(dtype).nmant) if operation == "logcdf" else 46
            exponents = jnp.rint(jnp.linspace(maximum_exponent, 6, parameter_count, dtype=dtype))
            tail_probability = jnp.exp2(-exponents)
            parameter = 1 - tail_probability if operation == "logcdf" else tail_probability

        return value, parameter.reshape(profile.parameter_shape)

    if distribution.name in {"poisson", "poisson_log"}:
        if input_set == "ordinary":
            (parameter,) = make_parameters(distribution, profile, dtype)
            threshold_range = (0.0, 10.0)
        else:
            rate = jnp.full(profile.parameter_shape, 40.0, dtype=dtype)
            parameter = jnp.log(rate) if distribution.name == "poisson_log" else rate
            # These counts span log probabilities around -4..-34 without relying on the zero-count shortcut
            threshold_range = (2.0, 26.0) if operation == "logcdf" else (54.0, 99.0)

        value = jnp.floor(jnp.linspace(*threshold_range, element_count, dtype=dtype))
        return value.reshape(profile.value_shape), parameter

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

    if distribution.name == "cauchy" and input_set == "tail":
        location, scale = make_parameters(distribution, profile, dtype)
        negative_log_probability = jnp.linspace(4.0, 32.0, element_count, dtype=dtype)
        # Inverting the closed-form tail targets the same probability span as lighter-tailed cases
        tail_probability = jnp.exp(-negative_log_probability)
        pi = jnp.asarray(math.pi, dtype=dtype)
        standardized_distance = 1 / jnp.tan(pi * tail_probability)
        standardized = -standardized_distance if operation == "logcdf" else standardized_distance

        value = location + scale * standardized.reshape(profile.value_shape)
        return value, location, scale

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


def make_dirichlet_arguments(
    profile: EventProfile,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Build valid simplex values and concentrations for one event profile."""
    concentration = jnp.linspace(
        0.5,
        3.0,
        math.prod(profile.parameter_shape),
        dtype=dtype,
    ).reshape(profile.parameter_shape)
    positive_values = jnp.linspace(
        0.5,
        1.5,
        math.prod(profile.value_shape),
        dtype=dtype,
    ).reshape(profile.value_shape)
    value = positive_values / jnp.sum(positive_values, axis=-1, keepdims=True)
    return value, concentration


def make_multinomial_arguments(
    profile: EventProfile,
    dtype: jnp.dtype,
    *,
    logits: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build count events, category parameters, and trial totals."""
    positive_weights = jnp.linspace(
        0.5,
        1.5,
        math.prod(profile.parameter_shape),
        dtype=dtype,
    ).reshape(profile.parameter_shape)
    probabilities = positive_weights / jnp.sum(positive_weights, axis=-1, keepdims=True)
    parameters = jnp.log(probabilities) if logits else probabilities

    trials = jnp.full(profile.batch_shape, 100, dtype=jnp.int32)
    category = jnp.arange(profile.event_size, dtype=jnp.int32)
    count = 100 // profile.event_size + (category < 100 % profile.event_size)
    value = jnp.broadcast_to(count, profile.value_shape)
    return value, parameters, trials


def make_event_arguments(
    distribution: EventDistributionSpec,
    profile: EventProfile,
    dtype: jnp.dtype,
) -> EventArguments:
    """Build density and sampling inputs for one event distribution."""
    if distribution.name == "dirichlet":
        value, concentration = make_dirichlet_arguments(profile, dtype)
        return EventArguments(
            log_probability=(value, concentration),
            sampling_parameters=(concentration,),
        )

    if distribution.name in {"multinomial", "multinomial_logit"}:
        value, parameters, trials = make_multinomial_arguments(
            profile,
            dtype,
            logits=distribution.name == "multinomial_logit",
        )
        return EventArguments(
            log_probability=(value, parameters),
            sampling_parameters=(parameters, trials),
        )

    raise ValueError(f"unknown event benchmark distribution {distribution.name!r}")
