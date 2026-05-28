"""Cooperative cancellation token for motion workers.

Workers periodically call .check() at yield boundaries. The scheduler (or any
preempt source) calls .cancel(reason=...) to signal that the worker should stop
at its next yield point. First .cancel() reason wins (idempotent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns


class CancellationError(Exception):
    """Raised by CancellationToken.check() when the token has been cancelled."""


@dataclass
class CancellationToken:
    """Cooperative cancellation signal for motion workers.

    A worker periodically calls .check() at yield boundaries. The scheduler (or
    any preempt source) calls .cancel(reason=...) to signal that the worker
    should stop at its next yield point. First .cancel() reason wins (idempotent).
    """

    _cancelled: bool = field(default=False, init=False, repr=False)
    _reason: str | None = field(default=None, init=False, repr=False)
    _cancelled_at_ns: int | None = field(default=None, init=False, repr=False)

    @property
    def is_cancelled(self) -> bool:
        """True once cancel() has been called."""
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """Cancellation reason string, or None if not yet cancelled."""
        return self._reason

    @property
    def cancelled_at_ms(self) -> int | None:
        """Wall-clock milliseconds (perf_counter_ns // 1_000_000) at cancellation, or None."""
        return None if self._cancelled_at_ns is None else self._cancelled_at_ns // 1_000_000

    def cancel(self, reason: str | None = None) -> None:
        """Signal cancellation. Only the first call takes effect (idempotent)."""
        if not self._cancelled:
            self._cancelled = True
            self._reason = reason if reason is not None else "unspecified"
            self._cancelled_at_ns = perf_counter_ns()

    def check(self) -> None:
        """Raise CancellationError if this token has been cancelled.

        Workers call this at every cooperative yield boundary.
        """
        if self._cancelled:
            raise CancellationError(self._reason)
