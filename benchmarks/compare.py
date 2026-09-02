"""Compare mmmJAX distribution primitives with public JAX operations."""

import argparse
import functools
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from benchmarks.cases import (
    DISTRIBUTIONS,
    EVENT_DISTRIBUTIONS,
    EVENT_MMM_JAX_FUNCTIONS,
    EVENT_PROFILES,
    MMM_JAX_FUNCTIONS,
    PROFILES,
    TAIL_DISTRIBUTIONS,
    BenchmarkProfile,
    DistributionFunctions,
    DistributionSpec,
    EventDistributionSpec,
    EventProfile,
    make_arguments,
    make_event_arguments,
    make_tail_arguments,
)
from benchmarks.common import (
    Arguments,
    BenchmarkFunction,
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
from benchmarks.references import JAX_REFERENCES

DEFAULT_OPERATIONS = ("logpdf", "logpmf", "log_density", "value_and_grad", "rng")
TAIL_OPERATIONS = (
    "logcdf",
    "logcdf_value_and_grad",
    "logsf",
    "logsf_value_and_grad",
)
OPERATIONS = DEFAULT_OPERATIONS + TAIL_OPERATIONS
INPUT_SETS = ("ordinary", "concentrated", "tail")
BenchmarkDistribution = DistributionSpec | EventDistributionSpec
BenchmarkProfileType = BenchmarkProfile | EventProfile


@dataclass(frozen=True)
class BenchmarkOperation:
    """Describe one compiled operation and its arguments."""

    implementation: str
    name: str
    function: BenchmarkFunction
    arguments: Arguments


IMPLEMENTATIONS: dict[str, dict[str, DistributionFunctions]] = {
    "mmmjax": {**MMM_JAX_FUNCTIONS, **EVENT_MMM_JAX_FUNCTIONS},
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


def make_operations(
    functions: DistributionFunctions,
    distribution: BenchmarkDistribution,
    profile: BenchmarkProfileType,
    arguments: Arguments,
    implementation: str,
    *,
    sampling_parameters: Arguments | None = None,
) -> tuple[BenchmarkOperation, ...]:
    """Build elementwise, summed, gradient, and sampling operations for one implementation."""
    if sampling_parameters is None:
        sampling_parameters = arguments[1:]
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
            jax.value_and_grad(functions.summed_log_probability, argnums=distribution.gradient_argnums),
            arguments,
        ),
        BenchmarkOperation(
            implementation,
            "rng",
            functools.partial(functions.rng, sample_shape=profile.sample_shape),
            (jax.random.key(0), *sampling_parameters),
        ),
    )


def make_tail_operations(
    functions: DistributionFunctions,
    distribution: DistributionSpec,
    arguments: Arguments,
    implementation: str,
    operation: str,
) -> tuple[BenchmarkOperation, ...]:
    """Build one tail probability operation and its parameter gradient."""
    if operation not in {"logcdf", "logsf"}:
        raise ValueError(f"operation must be 'logcdf' or 'logsf', got {operation!r}")

    function = functions.logcdf if operation == "logcdf" else functions.logsf
    if function is None:
        return ()

    summed_function = functools.partial(_sum_values, function)
    return (
        BenchmarkOperation(implementation, operation, function, arguments),
        BenchmarkOperation(
            implementation,
            f"{operation}_value_and_grad",
            jax.value_and_grad(summed_function, argnums=distribution.gradient_argnums),
            arguments,
        ),
    )


def make_benchmark_operation(
    functions: DistributionFunctions,
    distribution: BenchmarkDistribution,
    profile: BenchmarkProfileType,
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

    arguments: Arguments
    if isinstance(distribution, EventDistributionSpec):
        if not isinstance(profile, EventProfile):
            raise TypeError(f"{distribution.name} requires an event profile, got {type(profile).__name__}")
        if input_set != "ordinary" or operation not in DEFAULT_OPERATIONS:
            return None

        event_arguments = make_event_arguments(distribution, profile, dtype)
        candidates = make_operations(
            functions,
            distribution,
            profile,
            event_arguments.log_probability,
            implementation,
            sampling_parameters=event_arguments.sampling_parameters,
        )
        return next((candidate for candidate in candidates if candidate.name == operation), None)

    if not isinstance(profile, BenchmarkProfile):
        raise TypeError(f"{distribution.name} requires a scalar profile, got {type(profile).__name__}")

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
        if input_set == "concentrated" or distribution.name not in TAIL_DISTRIBUTIONS:
            return None

        tail_function = "logcdf" if operation.startswith("logcdf") else "logsf"
        arguments = make_tail_arguments(distribution, profile, input_set, tail_function, dtype)
        candidates = make_tail_operations(functions, distribution, arguments, implementation, tail_function)

    return next((candidate for candidate in candidates if candidate.name == operation), None)


def _sum_values(function: Callable[..., jax.Array], *arguments: jax.Array) -> jax.Array:
    return jnp.sum(function(*arguments))


def _benchmark(
    distribution: BenchmarkDistribution,
    profile_name: str,
    profile: BenchmarkProfileType,
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
    distributions = (*DISTRIBUTIONS, *EVENT_DISTRIBUTIONS)
    distribution_names = tuple(distribution.name for distribution in distributions)
    available_values = (
        ("profiles", tuple(PROFILES)),
        ("distributions", distribution_names),
        ("implementations", tuple(IMPLEMENTATIONS)),
        ("operations", OPERATIONS),
        ("dtype", ("float32", "float64")),
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
        prog="spin compare",
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
    has_tail_operation = bool(selected_operations.intersection(TAIL_OPERATIONS))
    if "tail" in arguments.inputs and not has_tail_operation:
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
    if has_tail_operation and not TAIL_DISTRIBUTIONS.intersection(arguments.distributions):
        supported = ", ".join(sorted(TAIL_DISTRIBUTIONS))
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
    selected_tail_operations = selected_operations.intersection(TAIL_OPERATIONS)
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
    unsupported_tail_distributions = selected_distributions - TAIL_DISTRIBUTIONS
    if selected_tail_operations and unsupported_tail_distributions:
        omitted = ", ".join(sorted(unsupported_tail_distributions))
        notes.append(f"Log-CDF and log-survival operations are omitted for distributions without those APIs: {omitted}")
    print_notes(notes)

    results: list[BenchmarkResult] = []
    distributions = (*DISTRIBUTIONS, *EVENT_DISTRIBUTIONS)
    for profile_name in arguments.profiles:
        for input_set in arguments.inputs:
            for distribution in distributions:
                if distribution.name not in selected_distributions:
                    continue

                if isinstance(distribution, EventDistributionSpec):
                    profile: BenchmarkProfileType = EVENT_PROFILES[profile_name]
                else:
                    profile = PROFILES[profile_name]

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
