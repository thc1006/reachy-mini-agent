"""Brain observability — Top 5 bundle (D2-OBS-BRAIN).

Single import surface for robot_brain.py to enable:
  1. structlog JSON → stderr → journald  (with stdlib fallback)
  2. faulthandler — all-thread traceback dump every 60 s if hung
  3. Per-thread heartbeat + gated systemd watchdog (gupnp-spin defence)
  4. prometheus_client metrics + HTTP exporter on BRAIN_PROMETHEUS_PORT (default 9101)
  5. pybreaker + tenacity wrappers for vLLM / Whisper / Kokoro HTTP backends
     with canned-phrase fallback ("嗯…我想想") via injected speak callback

Every dependency import is defensively wrapped so the module degrades to
no-op if structlog / prometheus_client / pybreaker / tenacity are absent —
brain stays bootable on a stripped venv.

The watchdog gate is the load-bearing piece: when any registered thread
heartbeat is older than WATCHDOG_THRESHOLD seconds we STOP sending
WATCHDOG=1 to systemd, so systemd kills + restarts the unit. This catches
GStreamer / gupnp-spin starvation that the old idle-loop watchdog couldn't
see (the idle loop was still alive, but worker threads were frozen).
"""
from __future__ import annotations

import faulthandler
import os
import socket
import sys
import threading
import time
from typing import Callable, Optional


# ── 1. structlog JSON logger (with stdlib fallback) ─────────────────────────
try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    _HAS_STRUCTLOG = False


class _PrintLogger:
    """Stdlib fallback when structlog is unavailable. JSON-ish key=value
    output to stderr so journald still picks it up."""

    def _emit(self, level: str, event: str, **kv) -> None:
        parts = [f"level={level}", f"event={event!r}"]
        for k, v in kv.items():
            parts.append(f"{k}={v!r}")
        print(" ".join(parts), file=sys.stderr, flush=True)

    def info(self, event: str, **kv) -> None:    self._emit("info", event, **kv)
    def warning(self, event: str, **kv) -> None: self._emit("warning", event, **kv)
    def error(self, event: str, **kv) -> None:   self._emit("error", event, **kv)
    def debug(self, event: str, **kv) -> None:   self._emit("debug", event, **kv)
    def bind(self, **kv):  # structlog API parity
        return self


def configure_structlog() -> object:
    """Configure structlog with JSON renderer + timestamps; return bound logger.
    Returns _PrintLogger fallback if structlog import failed."""
    if not _HAS_STRUCTLOG:
        return _PrintLogger()
    try:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(ensure_ascii=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger("brain")
    except Exception as e:
        print(f"[obs] structlog configure failed: {e}", file=sys.stderr, flush=True)
        return _PrintLogger()


# ── 2. faulthandler ─────────────────────────────────────────────────────────
_FAULTHANDLER_INTERVAL_S = 60


def enable_faulthandler(interval_s: int = _FAULTHANDLER_INTERVAL_S) -> None:
    """Enable faulthandler + dump all-thread traceback every interval_s seconds
    while ANY thread is hung longer than that. The dump goes to stderr which
    journald captures; we get a free thread-state snapshot whenever the brain
    locks up."""
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
        # repeat=True so it fires every interval while a hang persists; cancels
        # itself if the interpreter is healthy enough to process signals.
        faulthandler.dump_traceback_later(interval_s, repeat=True, file=sys.stderr)
    except Exception as e:
        print(f"[obs] faulthandler enable failed: {e}", file=sys.stderr, flush=True)


# ── 3. Per-thread heartbeat + watchdog ──────────────────────────────────────
# Default 90 s (raised from 60 s) — vision_worker's slow VL HTTP call to
# vllm0528 can legitimately take 30-60 s; a 60 s ceiling would mark a healthy
# worker stale mid-call. The watchdog still tightens systemd's 120 s
# WatchdogSec, just with more headroom for the slowest legitimate work.
WATCHDOG_THRESHOLD_S = float(os.getenv("BRAIN_WATCHDOG_THRESHOLD_S", "90"))
WATCHDOG_INTERVAL_S = float(os.getenv("BRAIN_WATCHDOG_INTERVAL_S", "10"))
# After the seeding loop completes, main() flips `_watchdog_armed_at`. Before
# that point the watchdog tolerates "no heartbeats yet" without complaint.
# Once armed, if 30 s elapse with still no heartbeats, the watchdog treats
# that as a stale state (suspect every worker thread died during startup)
# and STOPS sending WATCHDOG=1 so systemd restarts the unit.
_watchdog_armed_at: Optional[float] = None
_WATCHDOG_POST_ARM_GRACE_S = 30.0

_thread_heartbeat: dict[str, float] = {}
_heartbeat_lock = threading.Lock()


def pulse(name: str) -> None:
    """Record a heartbeat for the named thread. Cheap (~1µs); safe to call
    in tight loops."""
    with _heartbeat_lock:
        _thread_heartbeat[name] = time.monotonic()


def clear_pulse(name: str) -> None:
    """Remove a heartbeat registration — use for ephemeral threads (e.g.
    dialog_loop) when they exit, so the watchdog stops treating them as
    'always-on' workers that must keep pulsing."""
    with _heartbeat_lock:
        _thread_heartbeat.pop(name, None)


def heartbeat_snapshot() -> dict[str, float]:
    """Return a copy of the per-thread heartbeat dict (monotonic timestamps)."""
    with _heartbeat_lock:
        return dict(_thread_heartbeat)


def heartbeat_min_age_s() -> Optional[float]:
    """Return seconds since the OLDEST registered heartbeat, or None if no
    threads have pulsed yet (still in startup)."""
    snap = heartbeat_snapshot()
    if not snap:
        return None
    now = time.monotonic()
    return now - min(snap.values())


def arm_watchdog() -> None:
    """Mark the moment after which the watchdog should expect at least one
    heartbeat within _WATCHDOG_POST_ARM_GRACE_S. Call from main() AFTER
    workers have been seeded with their first pulse() — see robot_brain.main()."""
    global _watchdog_armed_at
    _watchdog_armed_at = time.monotonic()


def start_watchdog_thread(
    sd_notify_fn: Callable[[str], None],
    stop_event: threading.Event,
    logger: object,
    threshold_s: float = WATCHDOG_THRESHOLD_S,
    interval_s: float = WATCHDOG_INTERVAL_S,
) -> Optional[threading.Thread]:
    """Start the gated watchdog. Every interval_s it:
      - If NOTIFY_SOCKET unset → no-op (dev / non-systemd runs); return None
      - If no heartbeats registered yet AND <30 s since arm → send WATCHDOG=1
        (startup grace window for workers to register first pulse)
      - If no heartbeats registered AND >=30 s since arm → STALE, do NOT notify
      - If oldest heartbeat age < threshold_s → send WATCHDOG=1
      - Else: SKIP sending, let systemd WatchdogSec time out and restart.

    This replaces the old unconditional 30 s idle-loop pulse, which would
    cheerfully keep notifying systemd while every worker thread was deadlocked
    inside GStreamer / gupnp.
    """
    if "NOTIFY_SOCKET" not in os.environ:
        # Not under systemd notify supervision — the watchdog has nothing to
        # protect (no WatchdogSec timeout to head off, no journald audience).
        # Log once at INFO so dev runs make this visible, then bail.
        try:
            logger.info("watchdog_disabled_no_systemd")
        except Exception:
            pass
        return None

    def _run():
        try:
            logger.info("watchdog_started",
                        threshold_s=threshold_s, interval_s=interval_s)
        except Exception:
            pass
        while not stop_event.is_set():
            try:
                age = heartbeat_min_age_s()
                armed_at = _watchdog_armed_at
                if age is None:
                    # No heartbeats yet — tolerate during startup grace, but
                    # once we've been armed for >grace, treat as stale.
                    if armed_at is not None and (time.monotonic() - armed_at) > _WATCHDOG_POST_ARM_GRACE_S:
                        try:
                            logger.error("watchdog_stale_post_grace",
                                         grace_s=_WATCHDOG_POST_ARM_GRACE_S,
                                         armed_age_s=round(time.monotonic() - armed_at, 1))
                        except Exception:
                            pass
                        # Skip notify — let systemd WatchdogSec trip
                    else:
                        sd_notify_fn("WATCHDOG=1")
                elif age < threshold_s:
                    sd_notify_fn("WATCHDOG=1")
                else:
                    # Stale — let systemd kill us. Log so the journald entry
                    # explains the restart.
                    try:
                        snap = heartbeat_snapshot()
                        logger.error("watchdog_stale_no_pulse",
                                     min_age_s=round(age, 1),
                                     snapshot={k: round(time.monotonic() - v, 1)
                                               for k, v in snap.items()})
                    except Exception:
                        pass
            except Exception as e:
                try:
                    logger.error("watchdog_error", err=str(e))
                except Exception:
                    pass
            stop_event.wait(interval_s)

    t = threading.Thread(target=_run, name="brain-watchdog", daemon=True)
    t.start()
    return t


# ── 4. prometheus_client metrics ────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, start_http_server,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    Counter = Gauge = Histogram = None
    start_http_server = None
    _HAS_PROMETHEUS = False


class _NoOpMetric:
    """No-op stand-in when prometheus_client is unavailable. Mimics enough of
    Counter/Gauge/Histogram surface that instrumented call sites don't crash."""

    def labels(self, **_): return self
    def inc(self, *_a, **_k): pass
    def set(self, *_a, **_k): pass
    def observe(self, *_a, **_k): pass


# Module-level metric handles (instantiated lazily to avoid duplicate
# registration when robot_brain.py is reloaded in dev shells).
state_transitions: object = _NoOpMetric()
stt_latency: object = _NoOpMetric()
llm_latency: object = _NoOpMetric()
tts_latency: object = _NoOpMetric()
emergency_phrase: object = _NoOpMetric()
dialog_outcome: object = _NoOpMetric()
pipeline_e2e: object = _NoOpMetric()
backend_circuit_open: object = _NoOpMetric()
# Fallback / empty-result counters — incremented INSTEAD of recording near-0
# histogram observations on fallback branches (canned phrase, empty STT,
# silent TTS), so the latency histograms reflect real backend work only.
llm_fallback_total: object = _NoOpMetric()
stt_fallback_total: object = _NoOpMetric()
tts_fallback_total: object = _NoOpMetric()
# Wave4-P4: ElderFallGuard MediaPipe Pose fall detector. `fall_events_total`
# counts state-machine transitions and confirmed falls; `fall_state` is a
# numeric gauge (0=NORMAL, 1=SUSPECT, 2=FALL_CONFIRMED) for live dashboarding.
fall_events_total: object = _NoOpMetric()
fall_state: object = _NoOpMetric()
# H-F2: webhook delivery outcomes for FallGuard alerts.
# result=success (1st attempt) | retry (≥2nd attempt) | failed (DLQ written)
fall_webhook_outcome_total: object = _NoOpMetric()
# Wave4-P3 (#73): MotionQueue layered motion + barge-in interrupt. Counts every
# enqueue/start/complete/abort outcome so we can verify barge-in latency
# percentiles and see how often dialog actually interrupts in-flight motion.
motion_actions_total: object = _NoOpMetric()
motion_queue_depth: object = _NoOpMetric()
motion_run_seconds: object = _NoOpMetric()
# Wave4-P3 (#73) Option B: barge-in. Increments when motion_abort.request_abort
# fires (reason=user_voice_detected | external) and when do_action observes
# the abort mid-step (reason=in_step). Both labels coexist per abort event
# so you can sanity-check the dispatch→handle path matches.
brain_motion_abort_total: object = _NoOpMetric()

# Wave6-P4 (2026-05-29): on-demand VLM tool_call observability. Counts each
# LLM-emitted `query_vision` invocation broken down by outcome so we can spot
# VLM timeouts / empty responses vs the implicit success path (periodic
# vision_worker on a separate counter elsewhere).
vision_tool_call_total: object = _NoOpMetric()

_metrics_initialized = False
_metrics_init_lock = threading.Lock()


def init_metrics() -> bool:
    """Define + register all Prometheus metrics. Returns True on success."""
    global state_transitions, stt_latency, llm_latency, tts_latency
    global emergency_phrase, dialog_outcome, pipeline_e2e, backend_circuit_open
    global llm_fallback_total, stt_fallback_total, tts_fallback_total
    global fall_events_total, fall_state, fall_webhook_outcome_total
    global motion_actions_total, motion_queue_depth, motion_run_seconds
    global brain_motion_abort_total
    global vision_tool_call_total
    global _metrics_initialized
    if not _HAS_PROMETHEUS:
        return False
    with _metrics_init_lock:
        if _metrics_initialized:
            return True
        try:
            stt_buckets = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
            llm_buckets = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
            state_transitions = Counter(
                "brain_state_transitions_total",
                "State machine transitions",
                ["from_state", "to_state"],
            )
            stt_latency = Histogram(
                "brain_stt_latency_seconds",
                "STT (Whisper) latency, end-to-end per call",
                buckets=stt_buckets,
            )
            llm_latency = Histogram(
                "brain_llm_latency_seconds",
                "LLM (vLLM) latency, end-to-end per call",
                buckets=llm_buckets,
            )
            tts_latency = Histogram(
                "brain_tts_latency_seconds",
                "TTS (Edge/Kokoro/HaGen) latency, end-to-end per utterance",
                buckets=stt_buckets,
            )
            emergency_phrase = Counter(
                "brain_emergency_phrase_total",
                "Elder-care emergency phrase matches",
                ["phrase"],
            )
            dialog_outcome = Counter(
                "brain_dialog_outcome_total",
                "Dialog turn outcomes",
                ["result"],  # completed | noise_fallback | watchdog_restart | error
            )
            pipeline_e2e = Histogram(
                "brain_pipeline_e2e_seconds",
                "End-to-end pipeline latency: mic-stop → speaker-start",
                buckets=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
            )
            backend_circuit_open = Gauge(
                "brain_backend_circuit_open",
                "1 if pybreaker circuit is open (backend degraded)",
                ["backend"],
            )
            llm_fallback_total = Counter(
                "brain_llm_fallback_total",
                "LLM call fell back to canned phrase / empty result",
                ["reason"],
            )
            stt_fallback_total = Counter(
                "brain_stt_fallback_total",
                "STT call returned empty / fell back to silence",
                ["reason"],
            )
            tts_fallback_total = Counter(
                "brain_tts_fallback_total",
                "TTS call skipped / errored without playing",
                ["reason"],
            )
            fall_events_total = Counter(
                "brain_fall_events_total",
                "ElderFallGuard fall-detection state-machine events",
                ["state"],  # normal | suspect | confirmed | recovered | error
            )
            fall_state = Gauge(
                "brain_fall_state",
                "ElderFallGuard current state (0=NORMAL,1=SUSPECT,2=FALL_CONFIRMED)",
            )
            fall_webhook_outcome_total = Counter(
                "brain_fall_webhook_outcome_total",
                "FallGuard webhook delivery outcome (with retry envelope)",
                ["result"],  # success | retry | failed
            )
            motion_actions_total = Counter(
                "brain_motion_actions_total",
                "MotionQueue action outcomes",
                ["action", "outcome"],   # outcome: enqueued | started | completed | aborted | dropped | error
            )
            motion_queue_depth = Gauge(
                "brain_motion_queue_depth",
                "Pending items in MotionQueue (post-priority filtering)",
            )
            motion_run_seconds = Histogram(
                "brain_motion_run_seconds",
                "Wall time from MotionQueue start→completed/aborted per action",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            brain_motion_abort_total = Counter(
                "brain_motion_abort_total",
                "Cooperative motion abort events (Wave4-P3 #73 Option B barge-in)",
                ["reason"],  # user_voice_detected | in_step | external
            )
            vision_tool_call_total = Counter(
                "brain_vision_tool_call_total",
                "LLM-emitted query_vision tool_call outcomes",
                ["result"],  # success | error | timeout | no_frame
            )
            _metrics_initialized = True
            return True
        except Exception as e:
            print(f"[obs] init_metrics failed: {e}", file=sys.stderr, flush=True)
            return False


def start_metrics_exporter(logger: object) -> bool:
    """Start the Prometheus scrape endpoint on BRAIN_PROMETHEUS_PORT.
    Returns True on success; logs + swallows OSError (port in use) so brain
    keeps booting even if a stale exporter is already listening."""
    if not _HAS_PROMETHEUS:
        try:
            logger.warning("prometheus_unavailable",
                           reason="prometheus_client not installed")
        except Exception:
            pass
        return False
    if not init_metrics():
        return False
    port = int(os.getenv("BRAIN_PROMETHEUS_PORT", "9101"))
    addr = os.getenv("BRAIN_PROMETHEUS_ADDR", "0.0.0.0")
    try:
        start_http_server(port, addr=addr)
        try:
            logger.info("prometheus_exporter_started",
                        port=port, addr=addr, host=socket.gethostname())
        except Exception:
            pass
        return True
    except (OSError, PermissionError) as e:
        # PermissionError is a subclass of OSError on POSIX but we list it
        # explicitly so the intent (port < 1024 / SELinux bind denial) is
        # documented; either variant is logged with the exception class.
        try:
            logger.error("prometheus_exporter_port_busy",
                         port=port, addr=addr,
                         err=str(e), err_class=type(e).__name__)
        except Exception:
            pass
        return False
    except Exception as e:
        try:
            logger.error("prometheus_exporter_error", err=str(e))
        except Exception:
            pass
        return False


# ── 5. pybreaker + tenacity wrappers ───────────────────────────────────────
try:
    import pybreaker
    _HAS_PYBREAKER = True
except ImportError:
    pybreaker = None
    _HAS_PYBREAKER = False

try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, retry_if_not_exception_type,
    )
    _HAS_TENACITY = True
except ImportError:
    retry = None
    retry_if_not_exception_type = None
    _HAS_TENACITY = False


_breakers: dict[str, object] = {}
_breaker_lock = threading.Lock()
# Hot-path cache: the production backends are read every dialog turn —
# fetching them through the lock on every call_with_breaker is wasted
# contention. Populated lazily on first get_breaker() for these names; reads
# of the cache are lock-free (dict.get is atomic for str keys in CPython).
# 2026-06-01 (CosyVoice review M4): added "cosyvoice" so the new TTS backend
# also takes the lock-free hot path on every speak() turn.
_KNOWN_BACKENDS = frozenset({"vllm", "whisper", "kokoro", "cosyvoice"})
_breaker_cache: dict[str, object] = {}

# Per-backend default fail_max override. Most backends use the global
# fail_max=5 (configured at call site). CosyVoice is self-hosted on vllm0528
# with no warm capacity for a cold restart, so we open its circuit faster
# (fail_max=2) and let edge-tts take over rather than burn ~8 s × 5 = 40 s
# of degraded UX before the breaker trips. Read by callers that pass
# `fail_max=_BACKEND_FAIL_MAX.get(name, 5)` to get_breaker().
_BACKEND_FAIL_MAX: dict[str, int] = {"cosyvoice": 2}


class _NoOpBreaker:
    """When pybreaker is missing, calls just go straight through."""
    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)
    @property
    def current_state(self): return "closed"


def get_breaker(name: str, fail_max: int = 5, reset_timeout_s: int = 30) -> object:
    """Get-or-create a circuit breaker for the named backend. On open, the
    Prometheus brain_backend_circuit_open{backend=name} gauge flips to 1.

    Hot-path: for the three known backends (vllm/whisper/kokoro) the cached
    breaker is returned lock-free once populated. Unknown names still take the
    lock for safety.
    """
    if name in _KNOWN_BACKENDS:
        cached = _breaker_cache.get(name)
        if cached is not None:
            return cached
    with _breaker_lock:
        if name in _breakers:
            return _breakers[name]
        if not _HAS_PYBREAKER:
            _breakers[name] = _NoOpBreaker()
            if name in _KNOWN_BACKENDS:
                _breaker_cache[name] = _breakers[name]
            return _breakers[name]
        try:
            class _Listener(pybreaker.CircuitBreakerListener):
                def state_change(self, cb, old_state, new_state):
                    is_open = 1 if (new_state and getattr(new_state, "name", "") == "open") else 0
                    try:
                        backend_circuit_open.labels(backend=name).set(is_open)
                    except Exception:
                        pass

            cb = pybreaker.CircuitBreaker(
                fail_max=fail_max,
                reset_timeout=reset_timeout_s,
                listeners=[_Listener()],
                name=name,
            )
            _breakers[name] = cb
            if name in _KNOWN_BACKENDS:
                _breaker_cache[name] = cb
            try:
                backend_circuit_open.labels(backend=name).set(0)
            except Exception:
                pass
            return cb
        except Exception as e:
            print(f"[obs] get_breaker({name}) failed: {e}", file=sys.stderr, flush=True)
            _breakers[name] = _NoOpBreaker()
            if name in _KNOWN_BACKENDS:
                _breaker_cache[name] = _breakers[name]
            return _breakers[name]


def call_with_breaker(name: str, fn: Callable, *args, **kwargs):
    """Run fn(*args, **kwargs) through the named breaker + tenacity retry.

    Retry policy: exponential 0.5 → 2 s, max 3 attempts.
    Breaker policy: 5 consecutive failures opens the circuit for 30 s.

    Raises whatever fn raises (or pybreaker.CircuitBreakerError) — caller
    decides fallback behaviour (canned phrase, partial response, etc.).

    Per-backend fail_max overrides come from _BACKEND_FAIL_MAX; backends
    without an entry use the global default (5). This is applied on the
    first get_breaker() call for a name and cached for subsequent calls.
    """
    breaker = get_breaker(name, fail_max=_BACKEND_FAIL_MAX.get(name, 5))

    if not _HAS_TENACITY:
        return breaker.call(fn, *args, **kwargs)

    # H-O3 fix: tenacity's default `retry_if_exception_type(Exception)` would
    # also retry on pybreaker.CircuitBreakerError, which defeats the breaker
    # (we'd wait + try again instead of failing fast). Compose with
    # retry_if_not_exception_type so CircuitBreakerError raises immediately
    # and the caller can return the canned-phrase fallback.
    if _HAS_PYBREAKER:
        retry_pred = (
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(pybreaker.CircuitBreakerError)
        )
    else:
        retry_pred = retry_if_exception_type(Exception)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_pred,
        reraise=True,
    )
    def _attempt():
        return breaker.call(fn, *args, **kwargs)

    return _attempt()


def is_circuit_open(name: str) -> bool:
    """True if the named circuit is currently open (backend degraded)."""
    with _breaker_lock:
        cb = _breakers.get(name)
    if cb is None:
        return False
    try:
        st = getattr(cb, "current_state", "closed")
        # pybreaker exposes .current_state as a string like "open" / "closed" / "half-open".
        return str(st).lower() == "open"
    except Exception:
        return False


# Spoken when the vLLM call fails / circuit is open. Overridable per
# deployment via BRAIN_LLM_FALLBACK_PHRASE — useful for English-first deploys
# ("Let me think…") or shorter strings.
CANNED_FALLBACK_PHRASE = os.getenv("BRAIN_LLM_FALLBACK_PHRASE", "嗯…我想想。")
