"""Benchmark distribution primitives."""

import argparse
import math
import platform
import textwrap
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
    LOG_PROBABILITY_DISTRIBUTIONS,
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
    grouped_results: dict[tuple[str, str, str, int], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.profile, result.input_set, result.dtype, result.element_count)
        grouped_results.setdefault(key, []).append(result)

    print("\nResults")
    for (profile, input_set, dtype, element_count), group in grouped_results.items():
        print(f"\n{profile} · {input_set} · {dtype} · {element_count:,} values")
        _print_execution_results(group)
        _print_compilation_results(group)


def _print_execution_results(results: list[BenchmarkResult]) -> None:
    cases: dict[tuple[str, str], dict[str, BenchmarkResult]] = {}
    for result in results:
        cases.setdefault((result.distribution, result.operation), {})[result.implementation] = result

    compares_implementations = any({"mmmjax", "jax"}.issubset(case) for case in cases.values())
    headings = ["Benchmark", "Implementation", "Median", "MAD", "Throughput", "Iterations"]
    if compares_implementations:
        headings.append("Median vs JAX")

    rows = []
    for (distribution, operation), case in cases.items():
        benchmark_name = f"{distribution} / {operation}"
        for implementation_index, (implementation, result) in enumerate(case.items()):
            median, mad = _format_timing(result.execution_timing)
            row = [
                benchmark_name if implementation_index == 0 else "",
                _format_implementation(implementation),
                median,
                mad,
                _format_throughput(result.element_count, result.execution_timing.median_seconds),
                f"{result.iterations:,}",
            ]
            if compares_implementations:
                row.append(_format_comparison(case) if implementation == "mmmjax" else "")
            rows.append(tuple(row))

    print("\nWarm execution")
    right_aligned = (2, 3, 4, 5)
    _print_table(tuple(headings), rows, right_aligned=right_aligned)


def _print_compilation_results(results: list[BenchmarkResult]) -> None:
    rows = []
    previous_case: tuple[str, str] | None = None
    for result in results:
        case = (result.distribution, result.operation)
        median, mad = _format_timing(result.compile_timing)
        rows.append(
            (
                f"{result.distribution} / {result.operation}" if case != previous_case else "",
                _format_implementation(result.implementation),
                median,
                mad,
            )
        )
        previous_case = case

    print("\nCompilation (descriptive)")
    _print_table(
        ("Benchmark", "Implementation", "Median", "MAD"),
        rows,
        right_aligned=(2, 3),
    )


def _print_table(
    headings: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right_aligned: Sequence[int] = (),
) -> None:
    widths = [max(len(row[index]) for row in [headings, *rows]) for index in range(len(headings))]
    right_aligned_indices = set(right_aligned)

    def align(value: str, index: int) -> str:
        if index in right_aligned_indices:
            return value.rjust(widths[index])
        return value.ljust(widths[index])

    print("  ".join(align(heading, index) for index, heading in enumerate(headings)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(align(value, index) for index, value in enumerate(row)).rstrip())


def _format_timing(timing: TimingSummary) -> tuple[str, str]:
    if timing.median_seconds < 1e-6:
        scale, unit = 1e9, "ns"
    elif timing.median_seconds < 1e-3:
        scale, unit = 1e6, "µs"
    elif timing.median_seconds < 1:
        scale, unit = 1e3, "ms"
    else:
        scale, unit = 1.0, "s"
    return (
        f"{timing.median_seconds * scale:.2f} {unit}",
        f"{timing.mad_seconds * scale:.2f} {unit}",
    )


def _format_throughput(element_count: int, median_seconds: float) -> str:
    values_per_second = element_count / median_seconds
    if values_per_second >= 1e9:
        return f"{values_per_second / 1e9:.2f} G values/s"
    if values_per_second >= 1e6:
        return f"{values_per_second / 1e6:.2f} M values/s"
    if values_per_second >= 1e3:
        return f"{values_per_second / 1e3:.2f} k values/s"
    return f"{values_per_second:.2f} values/s"


def _format_comparison(case: dict[str, BenchmarkResult]) -> str:
    if not {"mmmjax", "jax"}.issubset(case):
        return ""

    mmmjax_seconds = case["mmmjax"].execution_timing.median_seconds
    jax_seconds = case["jax"].execution_timing.median_seconds
    difference = (mmmjax_seconds - jax_seconds) / jax_seconds * 100
    if round(difference, 1) == 0:
        return "same when rounded"
    direction = "shorter" if difference < 0 else "longer"
    return f"{abs(difference):.1f}% {direction}"


def _format_implementation(implementation: str) -> str:
    if implementation == "mmmjax":
        return "mmmJAX"
    if implementation == "jax":
        return "JAX"
    return implementation


def _print_section(title: str, entries: Sequence[tuple[str, str]]) -> None:
    label_width = max(len(label) for label, _ in entries)
    print(f"\n{title}")
    for label, value in entries:
        print(f"  {label.ljust(label_width)}  {value}")


def _print_environment(dtype: jnp.dtype, arguments: argparse.Namespace) -> None:
    device = jax.local_devices()[0]
    compile_repeats = f"{arguments.compile_repeats} repeat{'s' if arguments.compile_repeats != 1 else ''}"
    execution_repeats = f"{arguments.repeats} repeat{'s' if arguments.repeats != 1 else ''}"
    if arguments.iterations is None:
        iteration_setting = "automatic iterations"
    else:
        iteration_setting = f"{arguments.iterations:,} iteration{'s' if arguments.iterations != 1 else ''}"
    compares_implementations = len(set(arguments.implementations)) > 1 and any(
        input_set != "concentrated" for input_set in arguments.inputs
    )
    execution_order = "counterbalanced order" if compares_implementations else "single implementation"
    print("mmmJAX distribution benchmarks")
    _print_section(
        "Environment",
        (
            ("Runtime", f"Python {platform.python_version()} · JAX {jax.__version__} · jaxlib {jaxlib.__version__}"),
            ("System", f"{platform.system()} {platform.machine()} · {jax.default_backend()} backend"),
            (
                "Device",
                f"{device.device_kind} · {jax.local_device_count()} local / {jax.device_count()} global · "
                f"process {jax.process_index() + 1} of {jax.process_count()}",
            ),
            ("Precision", f"{dtype.name} · x64 {'enabled' if jax.config.x64_enabled else 'disabled'}"),
        ),
    )
    measurement_entries = [
        (
            "Compilation",
            f"{compile_repeats} · cache cleared · fixed order · median ± MAD",
        ),
        (
            "Execution",
            f"{execution_repeats} · {iteration_setting} · {execution_order} · median ± MAD",
        ),
        ("Throughput", "profile values processed or generated per second"),
    ]
    if compares_implementations:
        measurement_entries.append(("Comparison", "mmmJAX relative to the JAX warm median · descriptive only"))
    _print_section(
        "Measurement",
        tuple(measurement_entries),
    )


def _print_notes(notes: Sequence[str]) -> None:
    if not notes:
        return
    print("\nNotes")
    for note in notes:
        print(f"  - {note}")


def _parse_args() -> argparse.Namespace:
    distribution_names = tuple(distribution.name for distribution in DISTRIBUTIONS)
    available_values = (
        ("profiles", tuple(PROFILES)),
        ("distributions", distribution_names),
        ("implementations", tuple(IMPLEMENTATIONS)),
        ("operations", OPERATIONS),
        ("dtypes", ("float32", "float64")),
        ("inputs", INPUT_SETS),
    )
    label_width = max(len(label) for label, _ in available_values)
    available_lines = ["Available values:"]
    for label, values in available_values:
        indentation = f"  {label.ljust(label_width)}  "
        available_lines.append(
            textwrap.fill(
                ", ".join(values),
                width=100,
                initial_indent=indentation,
                subsequent_indent=" " * len(indentation),
            )
        )

    parser = argparse.ArgumentParser(
        prog="benchmark-distributions",
        description="Benchmark mmmJAX distribution primitives against equivalent public JAX operations.",
        epilog="\n".join(available_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = parser.add_argument_group("selection")
    selection.add_argument(
        "--profiles",
        choices=tuple(PROFILES),
        nargs="+",
        default=("channel_prior",),
        metavar="PROFILE",
        help="workload profiles to run (default: channel_prior)",
    )
    selection.add_argument(
        "--distributions",
        choices=distribution_names,
        nargs="+",
        default=distribution_names,
        metavar="NAME",
        help="distributions to run (default: all)",
    )
    selection.add_argument(
        "--implementations",
        choices=tuple(IMPLEMENTATIONS),
        nargs="+",
        default=tuple(IMPLEMENTATIONS),
        metavar="NAME",
        help="implementations to measure (default: mmmjax jax)",
    )
    selection.add_argument(
        "--operations",
        choices=OPERATIONS,
        nargs="+",
        default=DEFAULT_OPERATIONS,
        metavar="OPERATION",
        help="operations to measure (default: standard density, gradient, and RNG operations)",
    )
    selection.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        metavar="DTYPE",
        help="floating-point precision (default: float32)",
    )
    selection.add_argument(
        "--inputs",
        choices=INPUT_SETS,
        nargs="+",
        default=("ordinary",),
        metavar="INPUT",
        help="input sets to run (default: ordinary)",
    )

    measurement = parser.add_argument_group("measurement")
    measurement.add_argument(
        "--compile-repeats",
        type=int,
        default=3,
        metavar="COUNT",
        help="cache-cleared compilation measurements (default: 3)",
    )
    measurement.add_argument(
        "--repeats",
        type=int,
        default=6,
        metavar="COUNT",
        help="warm execution measurements (default: 6)",
    )
    measurement.add_argument(
        "--iterations",
        type=int,
        metavar="COUNT",
        help="calls per execution measurement (default: automatic)",
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
            "--inputs concentrated requires an elementwise log-probability operation, "
            "log_density, value_and_grad, or rng; "
            "log-CDF and log-survival operations support ordinary and tail inputs"
        )
    if has_log_probability_operation and not LOG_PROBABILITY_DISTRIBUTIONS.intersection(arguments.distributions):
        supported = ", ".join(sorted(LOG_PROBABILITY_DISTRIBUTIONS))
        parser.error(f"log-CDF and log-survival benchmarks are currently available only for {supported}")

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
    selects_concentrated_poisson_rng = (
        "concentrated" in arguments.inputs
        and "rng" in selected_operations
        and bool({"poisson", "poisson_log"}.intersection(selected_distributions))
    )
    _print_environment(dtype, arguments)
    notes = []
    if "concentrated" in arguments.inputs and "jax" in selected_implementations:
        notes.append(
            "Public JAX is omitted from concentrated inputs because it is not numerically equivalent "
            "for those benchmark inputs"
        )
    if selects_concentrated_poisson_rng:
        notes.append("Concentrated Poisson RNG is omitted because its float64 rate exceeds the int32 output range")
    unsupported_log_probability_distributions = selected_distributions - LOG_PROBABILITY_DISTRIBUTIONS
    if selected_log_probability_operations and unsupported_log_probability_distributions:
        omitted = ", ".join(sorted(unsupported_log_probability_distributions))
        notes.append(f"Log-CDF and log-survival operations are omitted for distributions without those APIs: {omitted}")
    _print_notes(notes)

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
        if selects_concentrated_poisson_rng:
            raise SystemExit(
                "Concentrated Poisson RNG is unavailable because the float64 rate exceeds the int32 output range; "
                "use --inputs ordinary or choose logpmf, log_density, or value_and_grad"
            )
        raise SystemExit("No benchmark cases match the selected distributions and inputs")
    _print_results(results)


if __name__ == "__main__":
    main()
