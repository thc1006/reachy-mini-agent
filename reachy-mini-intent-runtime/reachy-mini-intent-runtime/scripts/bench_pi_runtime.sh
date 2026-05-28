#!/usr/bin/env bash
# bench_pi_runtime.sh — measure stop-latency under a synthetic dance load.
#
# Default mode: --mock  (runs the simulator, prints latency stats, exits 0)
# Opt-in:       --real-hardware  (requires Reachy Mini + systemd + sudo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="mock"
DURATION_MS=30000
for arg in "$@"; do
  case "$arg" in
    --mock) MODE="mock" ;;
    --real-hardware) MODE="real" ;;
    --duration-ms=*) DURATION_MS="${arg#*=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

PY="${PYTHON:-python}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
  PY="${REPO_ROOT}/.venv/Scripts/python.exe"
fi

echo "[bench_pi_runtime] MODE=${MODE} duration_ms=${DURATION_MS}"

if [[ "$MODE" == "real" ]]; then
  echo "[bench_pi_runtime] real-hardware mode is opt-in and not implemented in this stub." >&2
  echo "[bench_pi_runtime] would: start dance, sleep, issue stop, measure latency via systemd journal." >&2
  exit 2
fi

# Mock bench: in-process simulation using the scheduler + mock adapter.
"$PY" - <<'PYBENCH'
import statistics
import sys
import time

from reachy_intent_runtime.models import ActionPriority, MotionCommand
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.scheduler import ActionScheduler


def one_trial(load_ticks: int) -> float:
    adapter = MockMotionAdapter()
    sched = ActionScheduler(adapter=adapter)
    sched.submit(
        MotionCommand(
            name="dance",
            tool="dance",
            priority=ActionPriority.BACKGROUND,
            interruptible=True,
            duration_ms=30_000,
            chunk_ms=250,
        )
    )
    for _ in range(load_ticks):
        sched.tick(25)
    t0 = time.perf_counter()
    sched.submit(
        MotionCommand(
            name="stop_dance",
            tool="stop_dance",
            priority=ActionPriority.CRITICAL,
            interruptible=False,
            duration_ms=0,
            chunk_ms=100,
        )
    )
    t1 = time.perf_counter()
    assert "stop_current" in adapter.log
    return (t1 - t0) * 1000.0


latencies = [one_trial(load_ticks=load) for load in (10, 25, 50, 100, 200, 400)]
print(f"[bench_pi_runtime] trials={len(latencies)}")
print(f"[bench_pi_runtime] latency_ms_min={min(latencies):.3f}")
print(f"[bench_pi_runtime] latency_ms_max={max(latencies):.3f}")
print(f"[bench_pi_runtime] latency_ms_mean={statistics.mean(latencies):.3f}")
print(f"[bench_pi_runtime] latency_ms_p95={sorted(latencies)[int(0.95 * (len(latencies)-1))]:.3f}")
print("[bench_pi_runtime] mock bench OK")
sys.exit(0)
PYBENCH
