"""Benchmark public distribution primitives."""

import argparse
import functools
import math
import platform
import statistics
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlib

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


@dataclass(frozen=True)
class TimingSummary:
    """Store a median timing and its median absolute deviation."""

    median_seconds: float
    mad_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Store cold compilation and warm execution measurements."""

    implementation: str
    distribution: str
    profile: str
    regime: str
    operation: str
    element_count: int
    dtype: str
    compile_timing: TimingSummary
    execution_timing: TimingSummary
    iterations: int


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

_IMPLEMENTATIONS: dict[str, dict[str, DistributionFunctions]] = {
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
    functions: DistributionFunctions,
    profile: BenchmarkProfile,
    arguments: Arguments,
    implementation: str,
) -> tuple[BenchmarkOperation, ...]:
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


def _compile(
    function: BenchmarkFunction,
    arguments: Arguments,
    *,
    repeats: int,
) -> tuple[Callable[..., object], TimingSummary]:
    compile_timings = []
    for _ in range(repeats):
        jax.clear_caches()
        start = time.perf_counter()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_timings.append(time.perf_counter() - start)
    jax.block_until_ready(compiled(*arguments))
    return compiled, _summarize_timings(compile_timings)


def _measure_execution(
    function: Callable[..., object],
    arguments: Arguments,
    *,
    repeats: int,
    iterations: int | None,
) -> tuple[TimingSummary, int]:
    timer = timeit.Timer(lambda: jax.block_until_ready(function(*arguments)))
    if iterations is None:
        iterations, _ = timer.autorange()

    execution_timings = [elapsed / iterations for elapsed in timer.repeat(repeat=repeats, number=iterations)]
    return _summarize_timings(execution_timings), iterations


def _summarize_timings(timings: list[float]) -> TimingSummary:
    median_seconds = statistics.median(timings)
    mad_seconds = statistics.median(abs(timing - median_seconds) for timing in timings)
    return TimingSummary(median_seconds=median_seconds, mad_seconds=mad_seconds)


def _benchmark(
    distribution: DistributionSpec,
    profile_name: str,
    regime: str,
    operation: BenchmarkOperation,
    dtype: jnp.dtype,
    *,
    compile_repeats: int,
    repeats: int,
    iterations: int | None,
) -> BenchmarkResult:
    jax.block_until_ready(operation.arguments)
    compiled, compile_timing = _compile(
        operation.function,
        operation.arguments,
        repeats=compile_repeats,
    )
    execution_timing, measured_iterations = _measure_execution(
        compiled,
        operation.arguments,
        repeats=repeats,
        iterations=iterations,
    )
    return BenchmarkResult(
        implementation=operation.implementation,
        distribution=distribution.name,
        profile=profile_name,
        regime=regime,
        operation=operation.name,
        element_count=math.prod(_PROFILES[profile_name].value_shape),
        dtype=dtype.name,
        compile_timing=compile_timing,
        execution_timing=execution_timing,
        iterations=measured_iterations,
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    headings = (
        "implementation",
        "distribution",
        "profile",
        "regime",
        "operation",
        "elements",
        "dtype",
        "compile ms (MAD)",
        "execution us (MAD)",
        "iterations",
    )
    rows = [
        (
            result.implementation,
            result.distribution,
            result.profile,
            result.regime,
            result.operation,
            f"{result.element_count:,}",
            result.dtype,
            f"{result.compile_timing.median_seconds * 1_000:.2f} ({result.compile_timing.mad_seconds * 1_000:.2f})",
            f"{result.execution_timing.median_seconds * 1_000_000:.2f} "
            f"({result.execution_timing.mad_seconds * 1_000_000:.2f})",
            f"{result.iterations:,}",
        )
        for result in results
    ]
    widths = [max(len(row[index]) for row in [headings, *rows]) for index in range(len(headings))]
    print("  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_environment(dtype: jnp.dtype, arguments: argparse.Namespace) -> None:
    device = jax.local_devices()[0]
    iteration_setting = "auto" if arguments.iterations is None else f"{arguments.iterations:,}"
    print(f"runtime python={platform.python_version()} jax={jax.__version__} jaxlib={jaxlib.__version__}")
    print(
        f"hardware system={platform.system()} machine={platform.machine()} backend={jax.default_backend()} "
        f"platform={device.platform} device={device.device_kind!r} global_devices={jax.device_count()} "
        f"local_devices={jax.local_device_count()} process={jax.process_index()}/{jax.process_count()}"
    )
    print(
        f"timing dtype={dtype.name} x64={jax.config.x64_enabled} compile_repeats={arguments.compile_repeats} "
        f"repeats={arguments.repeats} iterations={iteration_setting}"
    )


def _parse_args() -> argparse.Namespace:
    distribution_names = tuple(distribution.name for distribution in _DISTRIBUTIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", choices=tuple(_PROFILES), nargs="+", default=("channel_prior",))
    parser.add_argument("--distributions", choices=distribution_names, nargs="+", default=distribution_names)
    parser.add_argument(
        "--implementations",
        choices=tuple(_IMPLEMENTATIONS),
        nargs="+",
        default=tuple(_IMPLEMENTATIONS),
    )
    parser.add_argument("--operations", choices=_OPERATIONS, nargs="+", default=_OPERATIONS)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--regimes", choices=("ordinary", "concentrated"), nargs="+", default=("ordinary",))
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--iterations",
        type=int,
        help="fixed calls per repetition; omitted uses timeit autorange",
    )
    arguments = parser.parse_args()
    if arguments.compile_repeats <= 0:
        parser.error("--compile-repeats must be positive")
    if arguments.repeats <= 0:
        parser.error("--repeats must be positive")
    if arguments.iterations is not None and arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    if set(arguments.regimes) == {"concentrated"} and set(arguments.implementations) == {"jax"}:
        parser.error(
            "public JAX does not meet the concentrated-regime accuracy gate; "
            "use --regimes ordinary or --implementations mmmjax"
        )
    return arguments


def main() -> None:
    """Run the selected distribution benchmarks."""
    arguments = _parse_args()
    jax.config.update("jax_enable_compilation_cache", False)
    if arguments.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.dtype(arguments.dtype)

    selected_distributions = set(arguments.distributions)
    selected_implementations = set(arguments.implementations)
    selected_operations = set(arguments.operations)
    _print_environment(dtype, arguments)
    if "concentrated" in arguments.regimes and "jax" in selected_implementations:
        print("note=public JAX is omitted from concentrated regimes because it does not meet the accuracy gate")

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
                for implementation, functions_by_distribution in _IMPLEMENTATIONS.items():
                    if implementation not in selected_implementations:
                        continue
                    if regime == "concentrated" and implementation == "jax":
                        continue
                    functions = functions_by_distribution[distribution.name]
                    for operation in _operations(functions, profile, distribution_arguments, implementation):
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
