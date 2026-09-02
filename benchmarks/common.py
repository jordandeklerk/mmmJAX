"""Provide shared utilities for mmmJAX benchmarks."""

import platform
import statistics
import time
import timeit
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from typing import cast

import jax
import jax.numpy as jnp

BenchmarkFunction = Callable[..., object]
Arguments = tuple[jax.Array, ...]
CompiledOperations = dict[str, tuple[Callable[..., object], Arguments]]

_clear_jax_caches = cast(Callable[[], None], jax.clear_caches)
_block_until_ready = cast(Callable[[object], object], jax.block_until_ready)


@dataclass(frozen=True)
class TimingSummary:
    """Store a median timing and its median absolute deviation."""

    median_seconds: float
    mad_seconds: float


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


def compile_function(
    function: BenchmarkFunction,
    arguments: Arguments,
    *,
    repeats: int,
) -> tuple[Callable[..., object], TimingSummary]:
    """Compile a function after clearing JAX's in-process caches."""
    compile_timings = []
    for _ in range(repeats):
        _clear_jax_caches()
        start = time.perf_counter()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_timings.append(time.perf_counter() - start)
    synchronize(compiled(*arguments))
    return compiled, _summarize_timings(compile_timings)


def compile_and_warm(
    function: BenchmarkFunction,
    arguments: Arguments,
) -> Callable[..., object]:
    """Compile a function and finish its warm-up call before timing."""
    synchronize(arguments)
    compiled = jax.jit(function).lower(*arguments).compile()
    synchronize(compiled(*arguments))
    return compiled


def measure_executions(
    compiled_operations: CompiledOperations,
    *,
    repeats: int,
    iterations: int | None,
) -> dict[str, tuple[TimingSummary, int]]:
    """Measure synchronized calls in counterbalanced implementation order."""
    timers = {
        implementation: timeit.Timer(partial(_execute_synchronized, function, arguments))
        for implementation, (function, arguments) in compiled_operations.items()
    }
    iteration_counts = {
        implementation: iterations if iterations is not None else timer.autorange()[0]
        for implementation, timer in timers.items()
    }
    execution_timings: dict[str, list[float]] = {implementation: [] for implementation in timers}
    implementation_order = tuple(timers)

    for repeat_index in range(repeats):
        order = implementation_order if repeat_index % 2 == 0 else tuple(reversed(implementation_order))
        for implementation in order:
            elapsed = timers[implementation].timeit(number=iteration_counts[implementation])
            execution_timings[implementation].append(elapsed / iteration_counts[implementation])

    return {
        implementation: (_summarize_timings(timings), iteration_counts[implementation])
        for implementation, timings in execution_timings.items()
    }


def synchronize(value: object) -> None:
    """Wait for JAX work to finish before recording elapsed time."""
    _block_until_ready(value)


def print_results(results: list[BenchmarkResult]) -> None:
    """Print warm execution and compilation results."""
    grouped_results: dict[tuple[str, str, str, int], list[BenchmarkResult]] = {}
    for result in results:
        key = (result.profile, result.input_set, result.dtype, result.element_count)
        grouped_results.setdefault(key, []).append(result)

    print("\nResults")
    for (profile, input_set, dtype, element_count), group in grouped_results.items():
        print(f"\n{profile} · {input_set} · {dtype} · {element_count:,} values")
        _print_execution_results(group)
        _print_compilation_results(group)


def print_environment(
    dtype: jnp.dtype,
    *,
    compile_repeats: int,
    repeats: int,
    iterations: int | None,
    compares_implementations: bool,
) -> None:
    """Print the runtime and measurement configuration."""
    device = jax.local_devices()[0]
    read_jax_config = cast(Callable[[str], object], jax.config.read)
    x64_enabled = bool(read_jax_config("jax_enable_x64"))
    compile_repeat_label = f"{compile_repeats} repeat{'s' if compile_repeats != 1 else ''}"
    execution_repeats = f"{repeats} repeat{'s' if repeats != 1 else ''}"
    if iterations is None:
        iteration_setting = "automatic iterations"
    else:
        iteration_setting = f"{iterations:,} iteration{'s' if iterations != 1 else ''}"
    execution_order = "counterbalanced order" if compares_implementations else "single implementation"
    print("mmmJAX distribution benchmarks")
    _print_section(
        "Environment",
        (
            ("Runtime", f"Python {platform.python_version()} · JAX {jax.__version__} · jaxlib {version('jaxlib')}"),
            ("System", f"{platform.system()} {platform.machine()} · {jax.default_backend()} backend"),
            (
                "Device",
                f"{device.device_kind} · {jax.local_device_count()} local / {jax.device_count()} global · "
                f"process {jax.process_index() + 1} of {jax.process_count()}",
            ),
            ("Precision", f"{dtype.name} · x64 {'enabled' if x64_enabled else 'disabled'}"),
        ),
    )
    measurement_entries = [
        (
            "Compilation",
            f"{compile_repeat_label} · cache cleared · fixed order · median ± MAD",
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


def print_notes(notes: Sequence[str]) -> None:
    """Print benchmark omissions and qualifications."""
    if not notes:
        return
    print("\nNotes")
    for note in notes:
        print(f"  - {note}")


def _execute_synchronized(function: BenchmarkFunction, arguments: Arguments) -> None:
    synchronize(function(*arguments))


def _summarize_timings(timings: list[float]) -> TimingSummary:
    median_seconds = statistics.median(timings)
    mad_seconds = statistics.median(abs(timing - median_seconds) for timing in timings)
    return TimingSummary(median_seconds=median_seconds, mad_seconds=mad_seconds)


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
    _print_table(tuple(headings), rows, right_aligned=(2, 3, 4, 5))


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
