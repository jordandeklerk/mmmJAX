#!/usr/bin/env bash
set -euo pipefail

# CI runs this on Linux; local pytest commands do not need the resource monitor
lscpu
free --mebi
python -c 'import jax, platform; print(f"Python {platform.python_version()}, JAX {jax.__version__}, x64={jax.config.x64_enabled}")'

args=(
  -n "${TEST_WORKERS:-2}" --dist=worksteal
  --splits "${TEST_SPLITS:-1}" --group "${TEST_GROUP:-1}"
  --splitting-algorithm least_duration
  --durations=20 --junitxml=test-results.xml
)
if [[ "${TEST_COVERAGE:-false}" == "true" ]]; then
  args+=(--cov --cov-report=xml --cov-report=term)
fi

# Keep memory, swap and CPU activity visible even if the runner stops before pytest finishes
vmstat --wide --timestamp 15 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

/usr/bin/time --verbose python -m pytest "${args[@]}" "$@"
