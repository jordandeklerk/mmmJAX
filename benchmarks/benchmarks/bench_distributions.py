"""Run distribution primitive benchmarks."""

import argparse
import math
import textwrap
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from benchmarks.cases import (
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
from benchmarks.common import (
    BenchmarkResult,
    CompiledOperations,
    TimingSummary,
    compile_function,
    measure_executions,
    print_environment,
    print_notes,
    print_results,
    synchronize,
)


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
        synchronize(operation.arguments)
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
    update_jax_config = cast(Callable[[str, object], None], jax.config.update)
    update_jax_config("jax_enable_compilation_cache", False)
    if arguments.dtype == "float64":
        update_jax_config("jax_enable_x64", True)
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
    compares_implementations = len(set(arguments.implementations)) > 1 and any(
        input_set != "concentrated" for input_set in arguments.inputs
    )
    print_environment(
        dtype,
        compile_repeats=arguments.compile_repeats,
        repeats=arguments.repeats,
        iterations=arguments.iterations,
        compares_implementations=compares_implementations,
    )
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
    print_notes(notes)

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
    print_results(results)


if __name__ == "__main__":
    main()
