from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count

from .cancellation import CancellationToken
from .models import ActionPriority, MotionCommand, SchedulerEvent
from .motion_adapter import MotionAdapter


@dataclass
class _Running:
    command: MotionCommand
    remaining_ms: int
    until_boundary_ms: int
    token: CancellationToken = field(default_factory=CancellationToken)


@dataclass
class ActionScheduler:
    """Deterministic priority scheduler for interruptible Reachy Mini actions."""

    adapter: MotionAdapter
    events: list[SchedulerEvent] = field(default_factory=list)
    _queue: list[tuple[int, int, MotionCommand]] = field(default_factory=list)
    _seq: count = field(default_factory=count)
    _running: _Running | None = None
    # Maps object-id of MotionCommand -> CancellationToken for currently-running
    # or recently-started commands. Keyed by id() to avoid holding a strong ref
    # that would prevent GC of finished commands.
    _tokens: dict[int, CancellationToken] = field(default_factory=dict)

    @property
    def running(self) -> MotionCommand | None:
        return self._running.command if self._running else None

    def token_for(self, command: MotionCommand) -> CancellationToken | None:
        """Return the CancellationToken associated with *command*, or None.

        Returns None if the command was never submitted or never started running.
        The token is created when the command starts running and remains
        accessible until the scheduler is discarded.
        """
        return self._tokens.get(id(command))

    def submit(self, command: MotionCommand) -> None:
        if self._should_preempt(command):
            assert self._running is not None
            running_cmd = self._running.command
            running_id = id(running_cmd)
            # Cancel the running command's token before stopping it.
            reason = f"preempt_by_{command.name}_{command.priority.name.lower()}"
            self._running.token.cancel(reason=reason)
            self.events.append(
                SchedulerEvent(
                    "preempt",
                    running_cmd.name,
                    {
                        "by": command.name,
                        "from_priority": running_cmd.priority.name.lower(),
                        "to_priority": command.priority.name.lower(),
                    },
                )
            )
            self.adapter.stop_current()
            self._tokens.pop(running_id, None)
            self._running = None
        elif (
            self._running is not None
            and command.priority == ActionPriority.CRITICAL
            and not self._running.command.interruptible
        ):
            # Cannot preempt — running command is non-interruptible; log audit event.
            self.events.append(
                SchedulerEvent(
                    "preempt_blocked",
                    self._running.command.name,
                    {
                        "reason": "non_interruptible",
                        "requested": command.name,
                    },
                )
            )
        heappush(self._queue, (int(command.priority), next(self._seq), command))
        self.events.append(
            SchedulerEvent("queued", command.name, {"priority": command.priority.name.lower()})
        )
        self._start_next_if_idle()

    def tick(self, elapsed_ms: int) -> None:
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        if self._running is None:
            self._start_next_if_idle()
            return

        running = self._running
        running.remaining_ms = max(0, running.remaining_ms - elapsed_ms)
        running.until_boundary_ms = max(0, running.until_boundary_ms - elapsed_ms)

        if running.remaining_ms == 0:
            self.events.append(SchedulerEvent("finished", running.command.name))
            self._tokens.pop(id(running.command), None)
            self._running = None
            self._start_next_if_idle()
            return

        if running.until_boundary_ms == 0:
            running.until_boundary_ms = max(1, running.command.chunk_ms)
            if self._queue and self._queue[0][0] < int(running.command.priority):
                _, _, next_command = heappop(self._queue)
                if running.command.interruptible:
                    # Cancel the running command's token at the boundary preempt.
                    reason = f"preempt_by_{next_command.name}_{next_command.priority.name.lower()}"
                    running.token.cancel(reason=reason)
                    running_id = id(running.command)
                    self.events.append(
                        SchedulerEvent(
                            "boundary_preempt", running.command.name, {"by": next_command.name}
                        )
                    )
                    self.adapter.stop_current()
                    self._tokens.pop(running_id, None)
                    self._running = None
                    self.submit(next_command)
                else:
                    self.events.append(
                        SchedulerEvent(
                            "boundary_preempt_blocked",
                            running.command.name,
                            {
                                "reason": "non_interruptible",
                                "requested": next_command.name,
                            },
                        )
                    )
                    heappush(
                        self._queue, (int(next_command.priority), next(self._seq), next_command)
                    )

    def _should_preempt(self, command: MotionCommand) -> bool:
        return (
            self._running is not None
            and command.priority < self._running.command.priority
            and self._running.command.interruptible
            and command.priority == ActionPriority.CRITICAL
        )

    def _start_next_if_idle(self) -> None:
        if self._running is not None or not self._queue:
            return
        _, _, command = heappop(self._queue)
        duration = max(0, command.duration_ms)
        chunk = max(1, command.chunk_ms)
        token = CancellationToken()
        self._tokens[id(command)] = token
        self._running = _Running(
            command=command, remaining_ms=duration, until_boundary_ms=chunk, token=token
        )
        self.adapter.start(command)
        self.events.append(SchedulerEvent("started", command.name))
        if duration == 0:
            self.events.append(SchedulerEvent("finished", command.name))
            self._tokens.pop(id(command), None)
            self._running = None
            self._start_next_if_idle()
