"""Benchmark public distribution primitives."""

import argparse
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlib

from benchmarks._timing import CompiledOperations, TimingSummary, compile_function, measure_executions
from benchmarks.distribution_cases import (
    DISTRIBUTIONS,
    IMPLEMENTATIONS,
    OPERATIONS,
    PROFILES,
    BenchmarkOperation,
    BenchmarkProfile,
    DistributionSpec,
    make_arguments,
    make_operations,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Store cache-cleared compilation and warm execution measurements."""

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


def _benchmark(
    distribution: DistributionSpec,
    profile_name: str,
    profile: BenchmarkProfile,
    regime: str,
    operations: tuple[BenchmarkOperation, ...],
    dtype: jnp.dtype,
    *,
    compile_repeats: int,
    repeats: int,
    iterations: int | None,
) -> tuple[BenchmarkResult, ...]:
    compiled_operations: CompiledOperations = {}
    compile_timings: dict[str, TimingSummary] = {}
    for operation in operations:
        jax.block_until_ready(operation.arguments)
        compiled, compile_timing = compile_function(
            operation.function,
            operation.arguments,
            repeats=compile_repeats,
        )
        compiled_operations[operation.implementation] = (compiled, operation.arguments)
        compile_timings[operation.implementation] = compile_timing

    execution_measurements = measure_executions(
        compiled_operations,
        repeats=repeats,
        iterations=iterations,
    )
    return tuple(
        BenchmarkResult(
            implementation=operation.implementation,
            distribution=distribution.name,
            profile=profile_name,
            regime=regime,
            operation=operation.name,
            element_count=math.prod(profile.value_shape),
            dtype=dtype.name,
            compile_timing=compile_timings[operation.implementation],
            execution_timing=execution_measurements[operation.implementation][0],
            iterations=execution_measurements[operation.implementation][1],
        )
        for operation in operations
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
        "M values/s",
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
            f"{result.element_count / result.execution_timing.median_seconds / 1_000_000:.3f}",
            f"{result.iterations:,}",
        )
        for result in results
    ]
    _print_table(headings, rows)


def _print_comparisons(results: list[BenchmarkResult]) -> None:
    reference_results = {
        (result.distribution, result.profile, result.regime, result.operation, result.dtype): result
        for result in results
        if result.implementation == "jax"
    }
    comparisons = []
    for result in results:
        if result.implementation != "mmmjax":
            continue

        key = (result.distribution, result.profile, result.regime, result.operation, result.dtype)
        reference = reference_results.get(key)
        if reference is None:
            continue

        comparisons.append(
            (
                result.distribution,
                result.profile,
                result.regime,
                result.operation,
                f"{result.element_count:,}",
                result.dtype,
                f"{result.execution_timing.median_seconds * 1_000_000:.2f}",
                f"{reference.execution_timing.median_seconds * 1_000_000:.2f}",
                f"{reference.execution_timing.median_seconds / result.execution_timing.median_seconds:.3f}x",
            )
        )

    if not comparisons:
        return

    print("\nWarm execution comparison")
    print("ratio = JAX median / mmmJAX median; values above 1 mean mmmJAX had the lower median in this run")
    print("ratios are descriptive; read them alongside the raw MADs above")
    _print_table(
        (
            "distribution",
            "profile",
            "regime",
            "operation",
            "elements",
            "dtype",
            "mmmJAX us",
            "JAX us",
            "JAX / mmmJAX",
        ),
        comparisons,
    )


def _print_table(headings: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(row[index]) for row in [headings, *rows]) for index in range(len(headings))]
    print("  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_environment(dtype: jnp.dtype, arguments: argparse.Namespace) -> None:
    device = jax.local_devices()[0]
    iteration_setting = "auto" if arguments.iterations is None else f"{arguments.iterations:,}"
    compares_implementations = len(set(arguments.implementations)) > 1 and "ordinary" in arguments.regimes
    execution_order = "counterbalanced" if compares_implementations else "single"
    print(f"runtime python={platform.python_version()} jax={jax.__version__} jaxlib={jaxlib.__version__}")
    print(
        f"hardware system={platform.system()} machine={platform.machine()} backend={jax.default_backend()} "
        f"platform={device.platform} device={device.device_kind!r} global_devices={jax.device_count()} "
        f"local_devices={jax.local_device_count()} process={jax.process_index()}/{jax.process_count()}"
    )
    print(
        f"timing dtype={dtype.name} x64={jax.config.x64_enabled} compile_repeats={arguments.compile_repeats} "
        f"repeats={arguments.repeats} iterations={iteration_setting} compile_order=fixed "
        f"timed_repeat_order={execution_order}"
    )
    print("note=compile timings are descriptive; paired execution comparisons use counterbalanced measurements")
    print("note=throughput counts profile values processed or generated per second")


def _parse_args() -> argparse.Namespace:
    distribution_names = tuple(distribution.name for distribution in DISTRIBUTIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", choices=tuple(PROFILES), nargs="+", default=("channel_prior",))
    parser.add_argument("--distributions", choices=distribution_names, nargs="+", default=distribution_names)
    parser.add_argument(
        "--implementations",
        choices=tuple(IMPLEMENTATIONS),
        nargs="+",
        default=tuple(IMPLEMENTATIONS),
    )
    parser.add_argument("--operations", choices=OPERATIONS, nargs="+", default=OPERATIONS)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--regimes", choices=("ordinary", "concentrated"), nargs="+", default=("ordinary",))
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--iterations",
        type=int,
        help="fixed calls per repetition; omitted uses timeit autorange",
    )
    arguments = parser.parse_args()
    for option in ("profiles", "distributions", "implementations", "operations", "regimes"):
        values = getattr(arguments, option)
        if len(values) != len(set(values)):
            parser.error(f"--{option} must not contain duplicate values")
    if arguments.compile_repeats <= 0:
        parser.error("--compile-repeats must be positive")
    if arguments.repeats <= 0:
        parser.error("--repeats must be positive")
    compares_implementations = len(set(arguments.implementations)) > 1 and "ordinary" in arguments.regimes
    if compares_implementations and arguments.repeats % 2 != 0:
        parser.error("--repeats must be even when comparing implementations so execution order stays balanced")
    if arguments.iterations is not None and arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    if set(arguments.regimes) == {"concentrated"} and set(arguments.implementations) == {"jax"}:
        parser.error(
            "public JAX is not numerically equivalent for the concentrated benchmark inputs; "
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
        print(
            "note=public JAX is omitted from concentrated regimes because it is not numerically equivalent "
            "for those benchmark inputs"
        )

    results: list[BenchmarkResult] = []
    for profile_name in arguments.profiles:
        profile = PROFILES[profile_name]
        for regime in arguments.regimes:
            for distribution in DISTRIBUTIONS:
                if distribution.name not in selected_distributions:
                    continue
                if regime == "concentrated" and not distribution.supports_concentrated_regime:
                    continue

                distribution_arguments = make_arguments(distribution, profile, regime, dtype)
                operations_by_implementation: dict[str, dict[str, BenchmarkOperation]] = {}
                for implementation, functions_by_distribution in IMPLEMENTATIONS.items():
                    if implementation not in selected_implementations:
                        continue
                    if regime == "concentrated" and implementation == "jax":
                        continue
                    functions = functions_by_distribution[distribution.name]
                    operations_by_implementation[implementation] = {
                        operation.name: operation
                        for operation in make_operations(
                            functions,
                            profile,
                            distribution_arguments,
                            implementation,
                        )
                    }

                for operation_name in OPERATIONS:
                    if operation_name not in selected_operations:
                        continue
                    results.extend(
                        _benchmark(
                            distribution,
                            profile_name,
                            profile,
                            regime,
                            tuple(operations[operation_name] for operations in operations_by_implementation.values()),
                            dtype,
                            compile_repeats=arguments.compile_repeats,
                            repeats=arguments.repeats,
                            iterations=arguments.iterations,
                        )
                    )

    if not results:
        raise SystemExit("No benchmark cases match the selected distributions and regimes")
    _print_results(results)
    _print_comparisons(results)


if __name__ == "__main__":
    main()
