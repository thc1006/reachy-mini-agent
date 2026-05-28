"""Tests for CancellationToken — RED phase first, GREEN after implementation."""

from __future__ import annotations

import pytest

from reachy_intent_runtime.cancellation import CancellationError, CancellationToken


def test_token_starts_uncancelled() -> None:
    t = CancellationToken()
    assert not t.is_cancelled
    assert t.reason is None


def test_token_cancel_sets_state_and_reason() -> None:
    t = CancellationToken()
    t.cancel(reason="preempt_by_critical")
    assert t.is_cancelled
    assert t.reason == "preempt_by_critical"


def test_token_cancel_is_idempotent_first_reason_wins() -> None:
    t = CancellationToken()
    t.cancel(reason="first")
    t.cancel(reason="second")
    assert t.reason == "first"


def test_token_check_raises_when_cancelled() -> None:
    t = CancellationToken()
    t.check()  # no raise
    t.cancel(reason="x")
    with pytest.raises(CancellationError):
        t.check()


def test_token_cancelled_at_timestamp_recorded() -> None:
    t = CancellationToken()
    assert t.cancelled_at_ms is None
    t.cancel(reason="x")
    assert t.cancelled_at_ms is not None  # int milliseconds


def test_token_default_reason_when_not_provided() -> None:
    t = CancellationToken()
    t.cancel()
    assert t.reason == "unspecified"


def test_token_check_does_not_raise_when_not_cancelled() -> None:
    t = CancellationToken()
    # Must not raise — callable multiple times
    t.check()
    t.check()


def test_token_cancelled_at_ms_is_integer() -> None:
    t = CancellationToken()
    t.cancel(reason="x")
    assert isinstance(t.cancelled_at_ms, int)


def test_token_cancel_idempotent_cancelled_at_does_not_change() -> None:
    t = CancellationToken()
    t.cancel(reason="first")
    ts_first = t.cancelled_at_ms
    t.cancel(reason="second")
    assert t.cancelled_at_ms == ts_first
