"""Timing primitives for mmmJAX benchmarks."""

import statistics
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import jax

BenchmarkFunction = Callable[..., object]
Arguments = tuple[jax.Array, ...]
CompiledOperations = dict[str, tuple[Callable[..., object], Arguments]]


@dataclass(frozen=True)
class TimingSummary:
    """Store a median timing and its median absolute deviation."""

    median_seconds: float
    mad_seconds: float


def compile_function(
    function: BenchmarkFunction,
    arguments: Arguments,
    *,
    repeats: int,
) -> tuple[Callable[..., object], TimingSummary]:
    """Compile a function after clearing JAX's in-process caches."""
    compile_timings = []
    for _ in range(repeats):
        jax.clear_caches()
        start = time.perf_counter()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_timings.append(time.perf_counter() - start)
    jax.block_until_ready(compiled(*arguments))
    return compiled, _summarize_timings(compile_timings)


def measure_executions(
    compiled_operations: CompiledOperations,
    *,
    repeats: int,
    iterations: int | None,
) -> dict[str, tuple[TimingSummary, int]]:
    """Measure synchronized calls in counterbalanced implementation order."""
    timers = {
        implementation: timeit.Timer(
            lambda function=function, arguments=arguments: jax.block_until_ready(function(*arguments))
        )
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


def _summarize_timings(timings: list[float]) -> TimingSummary:
    median_seconds = statistics.median(timings)
    mad_seconds = statistics.median(abs(timing - median_seconds) for timing in timings)
    return TimingSummary(median_seconds=median_seconds, mad_seconds=mad_seconds)
