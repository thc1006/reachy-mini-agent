"""Wave4-P11 #85 Option (c) — gupnp-igd-thread CPU cap sidecar.

BACKGROUND
----------
Reachy Mini SDK 1.7.3 uses GStreamer webrtcbin → libnice 0.1.22 →
libgupnp-igd-1.6 for ICE UPnP candidate gathering. On a Tailscale
userspace network there is no IGD reachable, but libgupnp-igd's
GMainLoop worker thread (`gupnp-igd-thread`, comm truncated by kernel to
15 chars → `gupnp-igd-threa`) busy-spins regardless, accumulating 33-43%
of one core within ~5 min of brain startup. State stays `R (running)`
with `wchan=0` — pure userspace spin, no syscall to interrupt.

Option (a) — env var disable: does not exist upstream. Dead.
Option (b1) — monkey-patch SDK to skip UPnP setup: requires deeper
              SDK reading + risks WebRTC regression. Deferred.
Option (c) — symptom mitigation, ZERO SDK risk: lower the offending
              thread to SCHED_IDLE and pin it to a single (already-allowed)
              CPU core. The thread keeps spinning but only when nothing
              else wants the core; other threads on the same core are
              never starved (SCHED_IDLE is strictly lower priority than
              SCHED_OTHER). CPU% measured by `top` may stay high under
              an idle system but real contention drops to ~0.

DESIGN NOTES
------------
- Brain runs under systemd user unit with AllowedCPUs=2,3 (Pi 4). We
  pin gupnp to core 3 (lowest of the allowed set) so cores 2 still
  has full SCHED_OTHER bandwidth for the dialog / vision workers.
- The gupnp thread re-spawns on every WebRTC reconnect cycle with a
  fresh TID. We re-detect every `interval_s` (default 30 s) so a
  reconnect after a Tailscale flap is recovered automatically.
- We call sched_setscheduler from INSIDE the brain process. The brain
  owns those threads so no CAP_SYS_NICE is needed (kernel allows a
  process to lower priority of its own threads).
- The cap loop pulses obs.pulse("gupnp_cap") each iteration so the
  watchdog sees us as alive; on shutdown the caller MUST clear_pulse
  + set the stop_event so the watchdog doesn't mark us stale.

ENV VARS
--------
  BRAIN_GUPNP_CAP_ENABLED   1/0 — default 1 (symptom relief is default-ON).
  BRAIN_GUPNP_CAP_CORE      CPU id to pin to (default 3).
  BRAIN_GUPNP_CAP_INTERVAL  Re-scan period seconds (default 30).

LIMITATIONS
-----------
- Linux-only. On non-Linux (CI macOS / Windows dev box) every operation
  no-ops gracefully — find_gupnp_tids() returns [] when /proc/self/task
  is absent.
- The cap is per-process: if the brain itself restarts, the new brain
  re-spawns the sidecar from main() so the new gupnp thread is recaptured
  automatically — no external babysitter needed.
- We do NOT kill / unload libgupnp-igd; the WebRTC ICE pipeline still
  works (just slower UPnP path probing, which is irrelevant on Tailscale
  anyway since no IGD answers).
"""
from __future__ import annotations

import errno
import os
import threading
import time
from typing import Callable, Optional

from _env import env_bool


# Truncated comm name as the kernel reports it (15-char limit in
# /proc/$pid/task/$tid/comm).
GUPNP_COMM_PREFIX = "gupnp-igd-threa"

CAP_ENABLED = env_bool("BRAIN_GUPNP_CAP_ENABLED", default=True)
CAP_CORE = int(os.getenv("BRAIN_GUPNP_CAP_CORE", "3"))
CAP_INTERVAL_S = int(os.getenv("BRAIN_GUPNP_CAP_INTERVAL", "30"))

# Track TIDs we've already capped this brain lifetime so we don't waste
# syscalls re-applying or spam the log on every re-scan.
_capped_tids: set[int] = set()
_capped_lock = threading.Lock()


def find_gupnp_tids(task_dir: str = "/proc/self/task") -> list[int]:
    """Return TIDs of threads whose `comm` starts with GUPNP_COMM_PREFIX.

    Walks /proc/self/task/*/comm. Returns [] on any I/O error (e.g.
    non-Linux host, /proc not mounted) so callers can be platform-agnostic.
    """
    out: list[int] = []
    try:
        entries = os.listdir(task_dir)
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return out
    for entry in entries:
        # Each entry is a TID (numeric); skip anything else defensively.
        try:
            tid = int(entry)
        except ValueError:
            continue
        comm_path = os.path.join(task_dir, entry, "comm")
        try:
            with open(comm_path, "r") as f:
                comm = f.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            # Thread may have exited between listdir and open — that's fine.
            continue
        if comm.startswith(GUPNP_COMM_PREFIX):
            out.append(tid)
    return out


def apply_cap(tid: int, core: int = CAP_CORE) -> tuple[bool, str]:
    """Apply SCHED_IDLE + single-core affinity to the given TID.

    Returns (success, reason). reason is "" on success, otherwise a short
    diagnostic suitable for structured log fields.

    Idempotent: if the kernel rejects sched_setscheduler with EPERM (CAP_*
    missing) or EINVAL (unsupported policy on this kernel) we report the
    failure but do NOT raise — the brain stays up.
    """
    # SCHED_IDLE may not exist on every os module build (older Pythons),
    # but on CPython Linux it's exposed since 3.3. Defensive getattr keeps
    # the import side of this module portable to test mocks.
    sched_idle = getattr(os, "SCHED_IDLE", None)
    sched_param = getattr(os, "sched_param", None)
    sched_setscheduler = getattr(os, "sched_setscheduler", None)
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if sched_idle is None or sched_param is None or sched_setscheduler is None:
        return False, "sched_setscheduler_unavailable"
    try:
        sched_setscheduler(tid, sched_idle, sched_param(0))
    except OSError as e:
        return False, f"sched_setscheduler errno={e.errno} ({errno.errorcode.get(e.errno, '?')})"
    except Exception as e:  # noqa: BLE001 — surface any unexpected type to log
        return False, f"sched_setscheduler unexpected {type(e).__name__}: {e}"
    if sched_setaffinity is not None:
        try:
            sched_setaffinity(tid, {core})
        except OSError as e:
            # Affinity failure is non-fatal — the SCHED_IDLE drop already
            # prevents starvation of higher-priority work.
            return True, f"affinity_failed errno={e.errno} ({errno.errorcode.get(e.errno, '?')})"
        except Exception as e:  # noqa: BLE001
            return True, f"affinity_failed {type(e).__name__}: {e}"
    return True, ""


def cap_loop(
    stop_event: threading.Event,
    interval_s: int = CAP_INTERVAL_S,
    pulse_fn: Optional[Callable[[], None]] = None,
    logger: object = None,
    core: int = CAP_CORE,
) -> None:
    """Run forever (until stop_event is set) re-detecting gupnp threads.

    Each iteration:
      1. pulse_fn() — heartbeat for the watchdog.
      2. find_gupnp_tids() — discover current set of gupnp TIDs.
      3. For each TID we haven't already capped: apply_cap + log.
      4. stop_event.wait(interval_s) — interruptible sleep.

    pulse_fn and logger are optional so this loop is unit-testable
    without dragging in brain_observability.
    """
    if pulse_fn is not None:
        try:
            pulse_fn()
        except Exception:
            pass
    while not stop_event.is_set():
        try:
            tids = find_gupnp_tids()
            for tid in tids:
                with _capped_lock:
                    already = tid in _capped_tids
                if already:
                    continue
                ok, reason = apply_cap(tid, core=core)
                if ok:
                    with _capped_lock:
                        _capped_tids.add(tid)
                    if logger is not None:
                        try:
                            logger.info(
                                "gupnp_cap_applied",
                                tid=tid,
                                comm=GUPNP_COMM_PREFIX,
                                core=core,
                                affinity_note=reason or "ok",
                            )
                        except Exception:
                            pass
                else:
                    if logger is not None:
                        try:
                            logger.error(
                                "gupnp_cap_failed",
                                tid=tid,
                                comm=GUPNP_COMM_PREFIX,
                                reason=reason,
                            )
                        except Exception:
                            pass
        except Exception as e:  # noqa: BLE001 — never let the loop die silently
            if logger is not None:
                try:
                    logger.error("gupnp_cap_loop_error", err=str(e))
                except Exception:
                    pass
        if pulse_fn is not None:
            try:
                pulse_fn()
            except Exception:
                pass
        # Interruptible sleep so stop_event.set() unblocks immediately.
        stop_event.wait(interval_s)


def _reset_capped_state_for_tests() -> None:
    """Test-only: clear the module-level set of already-capped TIDs.

    Production code never calls this. Tests use it to isolate cases.
    """
    with _capped_lock:
        _capped_tids.clear()
