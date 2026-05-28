"""Mock-by-default motion worker stub.

Production worker would drive the Reachy Mini SDK in chunks. Here we ship a
hardware-free simulator that demonstrates the cooperative-yield contract from
SDD-04: the worker yields control at least every YIELD_BUDGET_MS so the
scheduler can interrupt long actions.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Generator
from dataclasses import dataclass, field

from .cancellation import CancellationToken

YIELD_BUDGET_MS = 500


@dataclass
class MotionWorkerSimulator:
    """Pure-Python simulator of the motion worker yield loop.

    `simulate_action` returns the list of simulated timestamps (in ms) at which
    the worker would yield to the scheduler. The contract guarantees the
    spacing between consecutive yields is <= YIELD_BUDGET_MS.

    `run_iter` is a generator variant that also honours an optional
    `cancel_token`. Workers call `cancel_token.is_cancelled` at every yield
    boundary and exit early when the token fires.
    """

    yield_budget_ms: int = YIELD_BUDGET_MS
    # Optional CancellationToken checked at every yield boundary in run_iter().
    cancel_token: CancellationToken | None = None
    # Set to True by run_iter() when it exits early due to cancellation.
    cancelled: bool = field(default=False, init=False)
    # Optional total_duration_ms used by run_iter() (ignored by simulate_action).
    total_duration_ms: int = 0

    def simulate_action(
        self,
        total_ms: int,
        step_ms: int = 50,
        stop_at_ms: int | None = None,
    ) -> list[int]:
        """Simulate a long action and return yield timestamps in ms."""
        if step_ms <= 0:
            raise ValueError("step_ms must be > 0")
        yields: list[int] = []
        elapsed = 0
        since_last_yield = 0
        while elapsed < total_ms:
            elapsed += step_ms
            since_last_yield += step_ms
            if since_last_yield >= self.yield_budget_ms:
                yields.append(elapsed)
                since_last_yield = 0
            if stop_at_ms is not None and elapsed >= stop_at_ms:
                # Insert a final yield to acknowledge the stop within budget.
                if not yields or yields[-1] != elapsed:
                    yields.append(elapsed)
                break
        # Ensure we always have at least one yield for sanity
        if not yields:
            yields.append(elapsed)
        return yields

    def run_iter(
        self,
        step_ms: int = 50,
    ) -> Generator[int, None, None]:
        """Iterate over simulated yield points, honouring cancel_token.

        Yields the simulated elapsed time (ms) at each cooperative checkpoint.
        Exits early when `self.cancel_token` is set and cancelled. Sets
        `self.cancelled = True` if the early-exit path was taken.

        Args:
            step_ms: Simulated milliseconds per iteration step (default 50).
        """
        if step_ms <= 0:
            raise ValueError("step_ms must be > 0")
        total_ms = max(0, self.total_duration_ms)
        elapsed = 0
        since_last_yield = 0
        self.cancelled = False

        while elapsed < total_ms or total_ms == 0:
            # Check cancellation token at the top of each step (yield boundary).
            if self.cancel_token is not None and self.cancel_token.is_cancelled:
                self.cancelled = True
                return

            elapsed += step_ms
            since_last_yield += step_ms

            if since_last_yield >= self.yield_budget_ms:
                since_last_yield = 0
                yield elapsed
                # Check cancellation again immediately after yielding control.
                if self.cancel_token is not None and self.cancel_token.is_cancelled:
                    self.cancelled = True
                    return

            if total_ms > 0 and elapsed >= total_ms:
                break

        # Final yield if we have not yet emitted one at/after total_ms
        if total_ms > 0:
            yield elapsed


@dataclass
class _MockSignalState:
    stop: bool = False
    received: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m reachy_intent_runtime.motion_worker`.

    Mock by default. Prints structured log lines + responds to SIGTERM/SIGINT.
    """
    parser = argparse.ArgumentParser(description="Reachy Mini motion worker (mock by default)")
    parser.add_argument("--mock", action="store_true", default=True, help="mock mode (default)")
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="opt-in real hardware mode (gated; not implemented in stub)",
    )
    parser.add_argument("--duration-ms", type=int, default=3000)
    args = parser.parse_args(argv)

    if args.real_hardware:
        print("[motion_worker] real-hardware mode is opt-in and not implemented in stub")
        return 2

    state = _MockSignalState()

    def _handler(signum: int, _frame: object) -> None:  # pragma: no cover - signal path
        state.stop = True
        state.received.append(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    sim = MotionWorkerSimulator()
    yields = sim.simulate_action(total_ms=args.duration_ms)
    for ts in yields:
        print(f"[motion_worker] mock yield at {ts}ms")
        if state.stop:
            break
    print(f"[motion_worker] done yields={len(yields)} stop={state.stop}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
