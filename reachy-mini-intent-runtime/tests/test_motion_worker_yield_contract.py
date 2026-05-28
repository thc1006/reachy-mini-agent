"""SDD-04 acceptance: motion_worker yields control at least every YIELD_BUDGET_MS.

The motion worker is a long-running loop. To remain interruptible, it MUST
yield (await sleep / cooperative checkpoint) at least every 500ms of simulated
time across a long action.
"""

from __future__ import annotations

from reachy_intent_runtime.motion_worker import (
    YIELD_BUDGET_MS,
    MotionWorkerSimulator,
)


def test_motion_worker_yields_at_least_every_500ms_over_long_action() -> None:
    sim = MotionWorkerSimulator()
    yields = sim.simulate_action(total_ms=30_000, step_ms=50)
    # Each consecutive yield must be <= YIELD_BUDGET_MS apart.
    deltas = [b - a for a, b in zip(yields, yields[1:], strict=False)]
    assert deltas, "expected at least one yield delta"
    assert max(deltas) <= YIELD_BUDGET_MS, (
        f"max yield gap {max(deltas)}ms exceeded {YIELD_BUDGET_MS}ms budget"
    )


def test_motion_worker_yields_first_within_budget() -> None:
    sim = MotionWorkerSimulator()
    yields = sim.simulate_action(total_ms=10_000, step_ms=50)
    assert yields[0] <= YIELD_BUDGET_MS, (
        f"first yield at {yields[0]}ms exceeded budget {YIELD_BUDGET_MS}ms"
    )


def test_motion_worker_yield_count_proportional_to_duration() -> None:
    sim = MotionWorkerSimulator()
    yields_short = sim.simulate_action(total_ms=2_000, step_ms=50)
    yields_long = sim.simulate_action(total_ms=20_000, step_ms=50)
    assert len(yields_long) > len(yields_short)


def test_motion_worker_stop_signal_breaks_loop_within_budget() -> None:
    sim = MotionWorkerSimulator()
    yields = sim.simulate_action(total_ms=30_000, step_ms=50, stop_at_ms=750)
    # Once stop is signalled, the worker must terminate within YIELD_BUDGET_MS.
    assert yields[-1] <= 750 + YIELD_BUDGET_MS, (
        f"worker did not stop within budget; last yield at {yields[-1]}ms"
    )


# ---------------------------------------------------------------------------
# Phase-10: CancellationToken integration tests
# ---------------------------------------------------------------------------


def test_motion_worker_simulator_stops_at_yield_when_token_cancelled() -> None:
    """Simulator run_iter() stops within one extra step after token is cancelled."""
    from reachy_intent_runtime.cancellation import CancellationToken

    token = CancellationToken()
    sim = MotionWorkerSimulator(total_duration_ms=10000, yield_budget_ms=500)
    sim.cancel_token = token

    yield_count = 0
    for _ in sim.run_iter():
        yield_count += 1
        if yield_count == 3:
            token.cancel(reason="external_stop")

    # After cancel the loop must exit — at most one extra iteration to detect it
    assert yield_count <= 4, f"expected <=4 yields after cancel at 3, got {yield_count}"


def test_motion_worker_simulator_completes_without_token() -> None:
    """run_iter() without a cancel_token runs to completion."""
    sim = MotionWorkerSimulator(total_duration_ms=2000, yield_budget_ms=500)
    yields = list(sim.run_iter())
    # 2000ms / 500ms budget = 4 yields
    assert len(yields) >= 1
    assert yields[-1] >= 2000  # elapsed covers full duration


def test_motion_worker_simulator_cancelled_property_set_after_cancel() -> None:
    """After token cancels, sim.cancelled is True."""
    from reachy_intent_runtime.cancellation import CancellationToken

    token = CancellationToken()
    sim = MotionWorkerSimulator(total_duration_ms=5000, yield_budget_ms=500)
    sim.cancel_token = token
    token.cancel(reason="immediate")
    yields = list(sim.run_iter())
    assert sim.cancelled is True
    # With immediate cancel, iterator should produce at most one yield
    assert len(yields) <= 1


def test_motion_worker_simulator_no_cancel_cancelled_is_false() -> None:
    """When token is never cancelled, sim.cancelled is False after completion."""
    sim = MotionWorkerSimulator(total_duration_ms=1000, yield_budget_ms=500)
    list(sim.run_iter())
    assert sim.cancelled is False
