"""Motion actor.

Owns (in production): ``mini.set_target`` (high-rate tracking
setpoints), ``mini.goto_target`` (one-shot trajectories used by tools
and recenter), the ``_motion_lock`` arbitrating between them.

This POC simulates the motor backend with bounded latency and acts as
the single writer of motion: tracking events and tool calls both
publish, motion serialises.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .event_bus import DropPolicy, EventBus
from .events import (
    DialogTool,
    FaceLost,
    FaceSeen,
    MotionDone,
)


@dataclass
class MotionConfig:
    tracking_setpoint_ms: float = 5.0   # 200Hz upper bound — fast no-op
    tool_goto_ms: float = 200.0          # one-shot trajectory
    recenter_ms: float = 600.0


@dataclass
class MotionStats:
    n_tracking_setpoints: int = 0
    n_tool_calls: int = 0
    n_recenters: int = 0
    last_tool_latency_ms: float = 0.0


class Motion:
    def __init__(
        self,
        bus: EventBus,
        cfg: Optional[MotionConfig] = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg or MotionConfig()
        self.stats = MotionStats()
        self._serial = threading.Lock()

        bus.subscribe(
            FaceSeen.topic,
            self._on_face_seen,
            name="motion.face",
            queue_size=8,
            drop=DropPolicy.DROP_OLDEST,
        )
        bus.subscribe(
            FaceLost.topic,
            self._on_face_lost,
            name="motion.recenter",
            queue_size=2,
        )
        bus.subscribe(
            DialogTool.topic,
            self._on_tool,
            name="motion.tool",
            queue_size=4,
        )

    def _on_face_seen(self, ev: FaceSeen) -> None:
        with self._serial:
            time.sleep(self.cfg.tracking_setpoint_ms / 1000.0)
            self.stats.n_tracking_setpoints += 1

    def _on_face_lost(self, ev: FaceLost) -> None:
        with self._serial:
            time.sleep(self.cfg.recenter_ms / 1000.0)
            self.stats.n_recenters += 1
            self.bus.publish(MotionDone(action="recenter",
                                        duration_ms=self.cfg.recenter_ms))

    def _on_tool(self, ev: DialogTool) -> None:
        t0 = time.perf_counter()
        with self._serial:
            time.sleep(self.cfg.tool_goto_ms / 1000.0)
            self.stats.n_tool_calls += 1
        self.stats.last_tool_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.bus.publish(MotionDone(action=ev.name,
                                    duration_ms=self.stats.last_tool_latency_ms))
