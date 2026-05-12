"""Threaded topic-routed pub/sub bus.

Design choices:

* **Per-subscriber bounded queue.** Slow consumers cannot back-pressure
  fast ones because each subscriber gets its own queue. The publisher's
  cost per subscriber is a single ``Queue.put_nowait``/``put`` call.
* **Topic strings, not classes.** Each ``Event`` subclass has a class-
  level ``topic`` string. Subscriptions match by exact topic, or by
  the wildcard ``"*"`` for tracing/test sinks.
* **Drop policy is per-subscription, not per-topic.** A face-tracking
  motion consumer might want to drop old ``face.seen`` events to stay
  current; a tool dispatcher must not drop ``dialog.tool``.

Anti-design:

* No async/await: the rest of the stack (urllib, pyaudio, reachy SDK)
  is blocking-IO, asyncio would force a much larger rewrite. See
  ``docs/multitask-arch.md`` §3.2.
* No priority queue: ordering within a topic is FIFO. If you need
  priority across topics, run two buses.
"""
from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .events import Event

log = logging.getLogger(__name__)


class DropPolicy(enum.Enum):
    BLOCK = "block"           # publisher waits if the subscriber queue is full
    DROP_OLDEST = "drop_oldest"  # silently evict the front of the queue
    DROP_NEWEST = "drop_newest"  # silently drop the event being published


@dataclass
class _Subscription:
    topics: frozenset
    handler: Callable[[Event], None]
    queue_size: int
    drop: DropPolicy
    name: str
    _q: "queue.Queue[Event]"
    _thread: Optional[threading.Thread] = None
    _stop: Optional[threading.Event] = None
    # stats
    n_received: int = 0
    n_dropped: int = 0
    n_blocked_publishes: int = 0


class EventBus:
    """Threaded pub/sub. Construct, ``subscribe(...)``, ``start()``,
    then ``publish(event)``. Call ``stop()`` to drain and shut down."""

    WILDCARD = "*"
    _SENTINEL: object = object()

    def __init__(self) -> None:
        self._subs: list[_Subscription] = []
        self._lock = threading.Lock()
        self._started = False
        self._stopping = False

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def subscribe(
        self,
        topics: Iterable[str] | str,
        handler: Callable[[Event], None],
        *,
        name: str = "",
        queue_size: int = 64,
        drop: DropPolicy = DropPolicy.BLOCK,
    ) -> None:
        """Register ``handler`` for ``topics``.

        Args:
            topics: topic string, sequence of topic strings, or
                ``EventBus.WILDCARD`` to receive everything.
            handler: callable invoked from the subscriber's own thread.
                Must be reasonably fast — slow handlers are why
                ``queue_size`` + ``drop`` exist.
            name: human-readable label for logging.
            queue_size: bounded queue capacity. Smaller = lower latency
                but more dropping/blocking under load.
            drop: how to handle the case where the queue is full.
        """
        if self._started:
            raise RuntimeError("subscribe() after start() not supported")
        if isinstance(topics, str):
            topics = (topics,)
        sub = _Subscription(
            topics=frozenset(topics),
            handler=handler,
            queue_size=queue_size,
            drop=drop,
            name=name or handler.__name__,
            _q=queue.Queue(maxsize=queue_size),
        )
        with self._lock:
            self._subs.append(sub)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        for sub in self._subs:
            sub._stop = threading.Event()
            sub._thread = threading.Thread(
                target=self._sub_loop,
                args=(sub,),
                name=f"bus-{sub.name}",
                daemon=True,
            )
            sub._thread.start()
        self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        """Signal subscribers to drain and exit. Returns once joined
        (or ``timeout`` per subscriber elapses)."""
        if not self._started:
            return
        self._stopping = True
        for sub in self._subs:
            try:
                sub._q.put_nowait(self._SENTINEL)  # type: ignore[arg-type]
            except queue.Full:
                # drop one and try again so subscriber definitely exits
                try:
                    sub._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    sub._q.put_nowait(self._SENTINEL)  # type: ignore[arg-type]
                except queue.Full:
                    pass
            if sub._stop is not None:
                sub._stop.set()
        for sub in self._subs:
            if sub._thread is not None:
                sub._thread.join(timeout=timeout)
        self._started = False

    # ------------------------------------------------------------------
    # publish
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        if not self._started:
            raise RuntimeError("publish() before start()")
        if event.ts == 0.0:
            event.ts = time.time()
        topic = type(event).topic
        for sub in self._subs:
            if self.WILDCARD not in sub.topics and topic not in sub.topics:
                continue
            self._enqueue(sub, event)

    def _enqueue(self, sub: _Subscription, event: Event) -> None:
        try:
            sub._q.put_nowait(event)
            return
        except queue.Full:
            pass
        if sub.drop == DropPolicy.BLOCK:
            sub.n_blocked_publishes += 1
            sub._q.put(event)
        elif sub.drop == DropPolicy.DROP_OLDEST:
            try:
                sub._q.get_nowait()
                sub.n_dropped += 1
            except queue.Empty:
                pass
            try:
                sub._q.put_nowait(event)
            except queue.Full:
                sub.n_dropped += 1
        elif sub.drop == DropPolicy.DROP_NEWEST:
            sub.n_dropped += 1

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _sub_loop(self, sub: _Subscription) -> None:
        while True:
            try:
                item = sub._q.get(timeout=0.5)
            except queue.Empty:
                if sub._stop is not None and sub._stop.is_set():
                    return
                continue
            if item is self._SENTINEL:
                return
            sub.n_received += 1
            try:
                sub.handler(item)  # type: ignore[arg-type]
            except Exception:
                log.exception("subscriber %s raised", sub.name)

    # ------------------------------------------------------------------
    # introspection (for tests + bench)
    # ------------------------------------------------------------------

    def stats(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "topics": sorted(s.topics),
                "received": s.n_received,
                "dropped": s.n_dropped,
                "blocked_publishes": s.n_blocked_publishes,
                "drop_policy": s.drop.value,
            }
            for s in self._subs
        ]
