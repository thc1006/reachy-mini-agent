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
WATCHDOG_THRESHOLD_S = float(os.getenv("BRAIN_WATCHDOG_THRESHOLD_S", "60"))
WATCHDOG_INTERVAL_S = float(os.getenv("BRAIN_WATCHDOG_INTERVAL_S", "10"))

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


def start_watchdog_thread(
    sd_notify_fn: Callable[[str], None],
    stop_event: threading.Event,
    logger: object,
    threshold_s: float = WATCHDOG_THRESHOLD_S,
    interval_s: float = WATCHDOG_INTERVAL_S,
) -> threading.Thread:
    """Start the gated watchdog. Every interval_s it:
      - If no heartbeats registered yet → send WATCHDOG=1 (startup grace)
      - If oldest heartbeat age < threshold_s → send WATCHDOG=1
      - Else: SKIP sending, let systemd WatchdogSec time out and restart.

    This replaces the old unconditional 30 s idle-loop pulse, which would
    cheerfully keep notifying systemd while every worker thread was deadlocked
    inside GStreamer / gupnp.
    """

    def _run():
        try:
            logger.info("watchdog_started",
                        threshold_s=threshold_s, interval_s=interval_s)
        except Exception:
            pass
        while not stop_event.is_set():
            try:
                age = heartbeat_min_age_s()
                if age is None or age < threshold_s:
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

_metrics_initialized = False
_metrics_init_lock = threading.Lock()


def init_metrics() -> bool:
    """Define + register all Prometheus metrics. Returns True on success."""
    global state_transitions, stt_latency, llm_latency, tts_latency
    global emergency_phrase, dialog_outcome, pipeline_e2e, backend_circuit_open
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
    except OSError as e:
        try:
            logger.error("prometheus_exporter_port_busy",
                         port=port, addr=addr, err=str(e))
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
        retry_if_exception_type,
    )
    _HAS_TENACITY = True
except ImportError:
    retry = None
    _HAS_TENACITY = False


_breakers: dict[str, object] = {}
_breaker_lock = threading.Lock()


class _NoOpBreaker:
    """When pybreaker is missing, calls just go straight through."""
    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)
    @property
    def current_state(self): return "closed"


def get_breaker(name: str, fail_max: int = 5, reset_timeout_s: int = 30) -> object:
    """Get-or-create a circuit breaker for the named backend. On open, the
    Prometheus brain_backend_circuit_open{backend=name} gauge flips to 1."""
    with _breaker_lock:
        if name in _breakers:
            return _breakers[name]
        if not _HAS_PYBREAKER:
            _breakers[name] = _NoOpBreaker()
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
            try:
                backend_circuit_open.labels(backend=name).set(0)
            except Exception:
                pass
            return cb
        except Exception as e:
            print(f"[obs] get_breaker({name}) failed: {e}", file=sys.stderr, flush=True)
            _breakers[name] = _NoOpBreaker()
            return _breakers[name]


def call_with_breaker(name: str, fn: Callable, *args, **kwargs):
    """Run fn(*args, **kwargs) through the named breaker + tenacity retry.

    Retry policy: exponential 0.5 → 2 s, max 3 attempts.
    Breaker policy: 5 consecutive failures opens the circuit for 30 s.

    Raises whatever fn raises (or pybreaker.CircuitBreakerError) — caller
    decides fallback behaviour (canned phrase, partial response, etc.).
    """
    breaker = get_breaker(name)

    if not _HAS_TENACITY:
        return breaker.call(fn, *args, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(Exception),
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


CANNED_FALLBACK_PHRASE = "嗯…我想想。"
