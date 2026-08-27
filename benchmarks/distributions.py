"""Benchmark public distribution primitives."""

import argparse
import functools
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

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
BenchmarkFunction = Callable[..., object]
Arguments = tuple[jax.Array, ...]


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
class DistributionSpec:
    """Describe the public functions and ordinary inputs for a distribution."""

    name: str
    logpdf: Kernel
    density: Kernel
    rng: Kernel
    value_range: tuple[float, float]
    parameter_values: tuple[float, ...]
    supports_concentrated_regime: bool = False


@dataclass(frozen=True)
class BenchmarkOperation:
    """Describe one compiled operation and its arguments."""

    name: str
    function: BenchmarkFunction
    arguments: Arguments


@dataclass(frozen=True)
class BenchmarkResult:
    """Store cold compilation and warm execution measurements."""

    distribution: str
    profile: str
    regime: str
    operation: str
    element_count: int
    dtype: str
    compile_ms: float
    execution_us: float


_PROFILES: dict[str, BenchmarkProfile] = {
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

_DISTRIBUTIONS = (
    DistributionSpec(
        name="beta",
        logpdf=beta_logpdf,
        density=beta,
        rng=beta_rng,
        value_range=(0.05, 0.95),
        parameter_values=(2.5, 3.5),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="exponential",
        logpdf=exponential_logpdf,
        density=exponential,
        rng=exponential_rng,
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
    ),
    DistributionSpec(
        name="gamma",
        logpdf=gamma_logpdf,
        density=gamma,
        rng=gamma_rng,
        value_range=(0.1, 2.0),
        parameter_values=(2.5, 1.3),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="half_normal",
        logpdf=half_normal_logpdf,
        density=half_normal,
        rng=half_normal_rng,
        value_range=(0.1, 2.0),
        parameter_values=(1.3,),
    ),
    DistributionSpec(
        name="inverse_gamma",
        logpdf=inverse_gamma_logpdf,
        density=inverse_gamma,
        rng=inverse_gamma_rng,
        value_range=(0.1, 2.0),
        parameter_values=(3.5, 1.2),
        supports_concentrated_regime=True,
    ),
    DistributionSpec(
        name="lognormal",
        logpdf=lognormal_logpdf,
        density=lognormal,
        rng=lognormal_rng,
        value_range=(0.1, 2.0),
        parameter_values=(0.2, 0.8),
    ),
    DistributionSpec(
        name="normal",
        logpdf=normal_logpdf,
        density=normal,
        rng=normal_rng,
        value_range=(-1.0, 1.0),
        parameter_values=(0.2, 1.3),
    ),
    DistributionSpec(
        name="student_t",
        logpdf=student_t_logpdf,
        density=student_t,
        rng=student_t_rng,
        value_range=(-1.0, 1.0),
        parameter_values=(5.0, 0.2, 1.3),
    ),
    DistributionSpec(
        name="uniform",
        logpdf=uniform_logpdf,
        density=uniform,
        rng=uniform_rng,
        value_range=(-0.5, 0.5),
        parameter_values=(-1.0, 1.0),
    ),
)

_OPERATIONS = ("logpdf", "density", "value_and_grad", "rng")


def _arguments(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    regime: str,
    dtype: jnp.dtype,
) -> Arguments:
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


def _operations(
    distribution: DistributionSpec,
    profile: BenchmarkProfile,
    arguments: Arguments,
) -> tuple[BenchmarkOperation, ...]:
    argument_indices = tuple(range(len(arguments)))
    parameters = arguments[1:]
    return (
        BenchmarkOperation("logpdf", distribution.logpdf, arguments),
        BenchmarkOperation("density", distribution.density, arguments),
        BenchmarkOperation(
            "value_and_grad",
            jax.value_and_grad(distribution.density, argnums=argument_indices),
            arguments,
        ),
        BenchmarkOperation(
            "rng",
            functools.partial(distribution.rng, sample_shape=profile.sample_shape),
            (jax.random.key(0), *parameters),
        ),
    )


def _compile(
    function: BenchmarkFunction,
    arguments: Arguments,
    *,
    repeats: int,
) -> tuple[Callable[..., object], float]:
    compile_timings = []
    for _ in range(repeats):
        jax.clear_caches()
        start = time.perf_counter()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_timings.append(time.perf_counter() - start)
    jax.block_until_ready(compiled(*arguments))
    return compiled, statistics.median(compile_timings) * 1_000


def _median_execution_us(
    function: Callable[..., object],
    arguments: Arguments,
    *,
    repeats: int,
    iterations: int,
) -> float:
    for _ in range(5):
        jax.block_until_ready(function(*arguments))

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            jax.block_until_ready(function(*arguments))
        timings.append((time.perf_counter() - start) / iterations)
    return statistics.median(timings) * 1_000_000


def _benchmark(
    distribution: DistributionSpec,
    profile_name: str,
    regime: str,
    operation: BenchmarkOperation,
    dtype: jnp.dtype,
    *,
    compile_repeats: int,
    repeats: int,
    iterations: int,
) -> BenchmarkResult:
    jax.block_until_ready(operation.arguments)
    compiled, compile_ms = _compile(
        operation.function,
        operation.arguments,
        repeats=compile_repeats,
    )
    execution_us = _median_execution_us(
        compiled,
        operation.arguments,
        repeats=repeats,
        iterations=iterations,
    )
    return BenchmarkResult(
        distribution=distribution.name,
        profile=profile_name,
        regime=regime,
        operation=operation.name,
        element_count=math.prod(_PROFILES[profile_name].value_shape),
        dtype=dtype.name,
        compile_ms=compile_ms,
        execution_us=execution_us,
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    headings = (
        "distribution",
        "profile",
        "regime",
        "operation",
        "elements",
        "dtype",
        "compile ms",
        "execution us",
    )
    rows = [
        (
            result.distribution,
            result.profile,
            result.regime,
            result.operation,
            f"{result.element_count:,}",
            result.dtype,
            f"{result.compile_ms:.2f}",
            f"{result.execution_us:.2f}",
        )
        for result in results
    ]
    widths = [max(len(row[index]) for row in [headings, *rows]) for index in range(len(headings))]
    print("  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _parse_args() -> argparse.Namespace:
    distribution_names = tuple(distribution.name for distribution in _DISTRIBUTIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", choices=tuple(_PROFILES), nargs="+", default=("channel_prior",))
    parser.add_argument("--distributions", choices=distribution_names, nargs="+", default=distribution_names)
    parser.add_argument("--operations", choices=_OPERATIONS, nargs="+", default=_OPERATIONS)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--regimes", choices=("ordinary", "concentrated"), nargs="+", default=("ordinary",))
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.compile_repeats <= 0:
        parser.error("--compile-repeats must be positive")
    if arguments.repeats <= 0:
        parser.error("--repeats must be positive")
    if arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    return arguments


def main() -> None:
    """Run the selected distribution benchmarks."""
    arguments = _parse_args()
    jax.config.update("jax_enable_compilation_cache", False)
    if arguments.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.dtype(arguments.dtype)

    selected_distributions = set(arguments.distributions)
    selected_operations = set(arguments.operations)
    print(f"backend={jax.default_backend()} device={jax.devices()[0]} x64={jax.config.x64_enabled}")

    results = []
    for profile_name in arguments.profiles:
        profile = _PROFILES[profile_name]
        for regime in arguments.regimes:
            for distribution in _DISTRIBUTIONS:
                if distribution.name not in selected_distributions:
                    continue
                if regime == "concentrated" and not distribution.supports_concentrated_regime:
                    continue

                distribution_arguments = _arguments(distribution, profile, regime, dtype)
                for operation in _operations(distribution, profile, distribution_arguments):
                    if operation.name not in selected_operations:
                        continue
                    results.append(
                        _benchmark(
                            distribution,
                            profile_name,
                            regime,
                            operation,
                            dtype,
                            compile_repeats=arguments.compile_repeats,
                            repeats=arguments.repeats,
                            iterations=arguments.iterations,
                        )
                    )

    if not results:
        raise SystemExit("No benchmark cases match the selected distributions and regimes")
    _print_results(results)


if __name__ == "__main__":
    main()
