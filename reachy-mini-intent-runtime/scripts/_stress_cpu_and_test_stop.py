"""Helper invoked by scripts/stress_cpu_and_test_stop.sh.

Kept as a real .py file so multiprocessing.spawn (Windows) can re-import the
worker function. Heredoc-stdin invocation breaks spawn on Windows.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

from reachy_intent_runtime.models import ActionPriority, MotionCommand
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.scheduler import ActionScheduler


def burn(stop_flag, deadline: float) -> None:
    x = 0
    while not stop_flag.is_set() and time.perf_counter() < deadline:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()

    stop_flag = mp.Event()
    deadline = time.perf_counter() + args.duration

    procs: list[mp.Process] = []
    for _ in range(args.workers):
        p = mp.Process(target=burn, args=(stop_flag, deadline))
        p.start()
        procs.append(p)

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

    time.sleep(min(0.2, args.duration / 4))
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
    latency_ms = (t1 - t0) * 1000.0

    stop_flag.set()
    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()

    assert "stop_current" in adapter.log, "stop_current missing from adapter log"
    print(f"[stress_cpu_and_test_stop] stop_dispatch_latency_ms={latency_ms:.3f}")
    print(f"[stress_cpu_and_test_stop] workers_burned={args.workers} duration_s={args.duration}")
    print("[stress_cpu_and_test_stop] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
