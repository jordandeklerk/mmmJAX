"""Benchmark distribution primitives."""

import argparse
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlib

from benchmarks._timing import CompiledOperations, TimingSummary, compile_function, measure_executions
from benchmarks.workloads import (
    DEFAULT_OPERATIONS,
    DISTRIBUTIONS,
    IMPLEMENTATIONS,
    INPUT_SETS,
    LOG_PROBABILITY_OPERATIONS,
    OPERATIONS,
    PROFILES,
    BenchmarkOperation,
    BenchmarkProfile,
    DistributionSpec,
    make_benchmark_operation,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Store cache-cleared compilation and warm execution measurements."""

    implementation: str
    distribution: str
    profile: str
    input_set: str
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
    input_set: str,
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
            input_set=input_set,
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
        "input set",
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
            result.input_set,
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
        (result.distribution, result.profile, result.input_set, result.operation, result.dtype): result
        for result in results
        if result.implementation == "jax"
    }
    comparisons = []
    for result in results:
        if result.implementation != "mmmjax":
            continue

        key = (result.distribution, result.profile, result.input_set, result.operation, result.dtype)
        reference = reference_results.get(key)
        if reference is None:
            continue

        comparisons.append(
            (
                result.distribution,
                result.profile,
                result.input_set,
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
            "input set",
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
    compares_implementations = len(set(arguments.implementations)) > 1 and any(
        input_set != "concentrated" for input_set in arguments.inputs
    )
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
    parser.add_argument("--operations", choices=OPERATIONS, nargs="+", default=DEFAULT_OPERATIONS)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--inputs", choices=INPUT_SETS, nargs="+", default=("ordinary",))
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--iterations",
        type=int,
        help="fixed calls per repetition; omitted uses timeit autorange",
    )
    arguments = parser.parse_args()
    for option in ("profiles", "distributions", "implementations", "operations", "inputs"):
        values = getattr(arguments, option)
        if len(values) != len(set(values)):
            parser.error(f"--{option} must not contain duplicate values")
    if arguments.compile_repeats <= 0:
        parser.error("--compile-repeats must be positive")
    if arguments.repeats <= 0:
        parser.error("--repeats must be positive")
    selected_operations = set(arguments.operations)
    has_default_operation = bool(selected_operations.intersection(DEFAULT_OPERATIONS))
    has_log_probability_operation = bool(selected_operations.intersection(LOG_PROBABILITY_OPERATIONS))
    if "tail" in arguments.inputs and not has_log_probability_operation:
        parser.error(
            "--inputs tail requires a log-CDF or log-survival operation; "
            "choose logcdf, logcdf_value_and_grad, logsf, or logsf_value_and_grad"
        )
    if "concentrated" in arguments.inputs and not has_default_operation:
        parser.error(
            "--inputs concentrated requires logpdf, density, value_and_grad, or rng; "
            "log-CDF and log-survival operations support ordinary and tail inputs"
        )
    if has_log_probability_operation and not {"normal", "lognormal"}.intersection(arguments.distributions):
        parser.error("log-CDF and log-survival benchmarks are currently available only for normal and lognormal")

    compares_implementations = len(set(arguments.implementations)) > 1 and any(
        input_set != "concentrated" for input_set in arguments.inputs
    )
    if compares_implementations and arguments.repeats % 2 != 0:
        parser.error("--repeats must be even when comparing implementations so execution order stays balanced")
    if arguments.iterations is not None and arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    if set(arguments.inputs) == {"concentrated"} and set(arguments.implementations) == {"jax"}:
        parser.error(
            "public JAX is not numerically equivalent for the concentrated benchmark inputs; "
            "use --inputs ordinary or --implementations mmmjax"
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
    selected_log_probability_operations = selected_operations.intersection(LOG_PROBABILITY_OPERATIONS)
    _print_environment(dtype, arguments)
    if "concentrated" in arguments.inputs and "jax" in selected_implementations:
        print(
            "note=public JAX is omitted from concentrated inputs because it is not numerically equivalent "
            "for those benchmark inputs"
        )
    unsupported_log_probability_distributions = selected_distributions - {"normal", "lognormal"}
    if selected_log_probability_operations and unsupported_log_probability_distributions:
        omitted = ", ".join(sorted(unsupported_log_probability_distributions))
        print(f"note=log-CDF and log-survival operations are omitted for distributions without those APIs: {omitted}")

    results: list[BenchmarkResult] = []
    for profile_name in arguments.profiles:
        profile = PROFILES[profile_name]
        for input_set in arguments.inputs:
            for distribution in DISTRIBUTIONS:
                if distribution.name not in selected_distributions:
                    continue

                for operation_name in OPERATIONS:
                    if operation_name not in selected_operations:
                        continue

                    benchmark_operations: list[BenchmarkOperation] = []
                    for implementation, functions_by_distribution in IMPLEMENTATIONS.items():
                        if implementation not in selected_implementations:
                            continue
                        operation = make_benchmark_operation(
                            functions_by_distribution[distribution.name],
                            distribution,
                            profile,
                            input_set=input_set,
                            operation=operation_name,
                            dtype=dtype,
                            implementation=implementation,
                        )
                        if operation is not None:
                            benchmark_operations.append(operation)

                    if not benchmark_operations:
                        continue
                    results.extend(
                        _benchmark(
                            distribution,
                            profile_name,
                            profile,
                            input_set,
                            tuple(benchmark_operations),
                            dtype,
                            compile_repeats=arguments.compile_repeats,
                            repeats=arguments.repeats,
                            iterations=arguments.iterations,
                        )
                    )

    if not results:
        raise SystemExit("No benchmark cases match the selected distributions and inputs")
    _print_results(results)
    _print_comparisons(results)


if __name__ == "__main__":
    main()
