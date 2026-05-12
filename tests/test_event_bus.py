"""Pytest coverage for orchestrator.event_bus.EventBus.

Promoted from _b2_bus_test.py (the local smoke scratch) plus one extra
case covering the publish-during-stopping short-circuit landed alongside
PR #1 review fixes.

Covers:

  1. subscribe + publish + receive (FIFO)
  2. wildcard subscriber sees every topic
  3. DROP_OLDEST evicts front of full queue, latest events still arrive
  4. DROP_NEWEST drops the incoming event, oldest events still arrive
  5. BLOCK policy blocks the publisher when queue full
  6. stop() drains and joins
  7. multiple subscribers for the same topic each receive their own copy
  8. subscribe after start() is rejected
  9. publish before start() is rejected
 10. publish during stop() is a silent no-op (does not enqueue behind sentinel)

Why a separate test file (and not pytest parameterisation): each case
exercises a distinct invariant of the bus contract; mixing them into a
parameterised matrix makes failures harder to diagnose.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import ClassVar

import pytest

from orchestrator.event_bus import DropPolicy, EventBus
from orchestrator.events import Event


# ---------------------------------------------------------------------------
# Local event types — keep tests independent of production event taxonomy.
# ---------------------------------------------------------------------------

@dataclass
class _T(Event):
    topic: ClassVar[str] = "test.t"
    n: int = 0


@dataclass
class _U(Event):
    topic: ClassVar[str] = "test.u"
    n: int = 0


def _collect(target: list, slow_ms: float = 0.0):
    def _h(ev):
        if slow_ms:
            time.sleep(slow_ms / 1000.0)
        target.append(ev)
    return _h


def _wait_until(predicate, timeout: float = 1.5) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEventBusBasics:
    def test_basic_pubsub_preserves_order(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(_T.topic, _collect(out), name="basic")
        bus.start()
        try:
            for i in range(5):
                bus.publish(_T(n=i))
            assert _wait_until(lambda: len(out) == 5)
            assert [e.n for e in out] == [0, 1, 2, 3, 4]
        finally:
            bus.stop()

    def test_wildcard_subscriber_sees_all_topics(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(EventBus.WILDCARD, _collect(out), name="all")
        bus.start()
        try:
            bus.publish(_T(n=1))
            bus.publish(_U(n=2))
            bus.publish(_T(n=3))
            assert _wait_until(lambda: len(out) == 3)
        finally:
            bus.stop()

    def test_multi_subscriber_each_receive_own_copy(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe(_T.topic, _collect(a), name="sub_a")
        bus.subscribe(_T.topic, _collect(b), name="sub_b")
        bus.start()
        try:
            for i in range(5):
                bus.publish(_T(n=i))
            assert _wait_until(lambda: len(a) == 5 and len(b) == 5)
            assert [e.n for e in a] == [0, 1, 2, 3, 4]
            assert [e.n for e in b] == [0, 1, 2, 3, 4]
        finally:
            bus.stop()


class TestEventBusDropPolicies:
    def test_drop_oldest_evicts_front_keeps_latest(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(
            _T.topic, _collect(out, slow_ms=80),
            name="slow", queue_size=2, drop=DropPolicy.DROP_OLDEST,
        )
        bus.start()
        try:
            for i in range(10):
                bus.publish(_T(n=i))
            # Give subscriber enough time to drain its bounded queue.
            time.sleep(1.5)
            stats = bus.stats()[0]
            assert stats["dropped"] > 0
            # The most recent events should be in the output.
            assert max(e.n for e in out) == 9
        finally:
            bus.stop()

    def test_drop_newest_keeps_oldest(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(
            _T.topic, _collect(out, slow_ms=80),
            name="slow", queue_size=2, drop=DropPolicy.DROP_NEWEST,
        )
        bus.start()
        try:
            for i in range(10):
                bus.publish(_T(n=i))
            time.sleep(1.5)
            stats = bus.stats()[0]
            assert stats["dropped"] > 0
            # The earliest events should be in the output.
            assert min(e.n for e in out) == 0
        finally:
            bus.stop()

    def test_block_policy_blocks_publisher_and_drops_nothing(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(
            _T.topic, _collect(out, slow_ms=100),
            name="slow", queue_size=2, drop=DropPolicy.BLOCK,
        )
        bus.start()
        try:
            t0 = time.perf_counter()
            # 6 events, queue=2, handler=100ms → publisher should wait
            # ~ (6 - 2) × 100ms = 400ms once the queue fills.
            for i in range(6):
                bus.publish(_T(n=i))
            wall_ms = (time.perf_counter() - t0) * 1000
            # Drain.
            assert _wait_until(lambda: len(out) == 6)
            stats = bus.stats()[0]
            assert stats["dropped"] == 0
            assert stats["blocked_publishes"] >= 1
            assert wall_ms > 200
            assert [e.n for e in out] == list(range(6))
        finally:
            bus.stop()


class TestEventBusLifecycle:
    def test_stop_drains_pending_events_before_join(self):
        bus = EventBus()
        out: list = []
        bus.subscribe(_T.topic, _collect(out), name="drain")
        bus.start()
        for i in range(20):
            bus.publish(_T(n=i))
        t0 = time.perf_counter()
        bus.stop(timeout=2.0)
        wall = time.perf_counter() - t0
        assert wall < 1.5, f"stop too slow: {wall}s"
        assert len(out) == 20

    def test_subscribe_after_start_rejected(self):
        bus = EventBus()
        bus.subscribe(_T.topic, lambda e: None, name="early")
        bus.start()
        try:
            with pytest.raises(RuntimeError):
                bus.subscribe(_T.topic, lambda e: None, name="late")
        finally:
            bus.stop()

    def test_publish_before_start_rejected(self):
        bus = EventBus()
        bus.subscribe(_T.topic, lambda e: None, name="x")
        with pytest.raises(RuntimeError):
            bus.publish(_T(n=1))

    def test_publish_during_stop_is_silent_noop(self):
        """PR #1 review #2: events published after stop() started must not
        land behind the shutdown sentinel where they'd never be processed.
        publish() short-circuits silently when _stopping is True."""
        bus = EventBus()
        out: list = []
        # Slow handler — so stop() spends time draining and a racer can
        # try to publish during that window.
        bus.subscribe(
            _T.topic, _collect(out, slow_ms=50),
            name="slow_drain", queue_size=64, drop=DropPolicy.BLOCK,
        )
        bus.start()
        # Prime the queue with 5 events the subscriber will need ~250ms
        # to process; stop() will wait through that.
        for i in range(5):
            bus.publish(_T(n=i))

        racer_publishes_succeeded = []

        def _racer():
            # Spin briefly so we start publishing after stop() is called.
            time.sleep(0.02)
            for j in range(100):
                try:
                    bus.publish(_T(n=1000 + j))
                    racer_publishes_succeeded.append(j)
                except Exception:
                    return
                time.sleep(0.001)

        racer = threading.Thread(target=_racer, daemon=True)
        racer.start()
        bus.stop(timeout=3.0)
        racer.join(timeout=2.0)

        # The pre-stop 5 events must all have been processed.
        n_pre_stop = sum(1 for e in out if e.n < 1000)
        assert n_pre_stop == 5

        # Racer publishes during stop are *silently dropped*, NOT inserted
        # behind the sentinel. So out should contain ZERO events whose
        # n >= 1000 — those were short-circuited.
        n_post_stop = sum(1 for e in out if e.n >= 1000)
        assert n_post_stop == 0, (
            f"events published during stop() leaked into the queue: "
            f"{n_post_stop} extra"
        )
