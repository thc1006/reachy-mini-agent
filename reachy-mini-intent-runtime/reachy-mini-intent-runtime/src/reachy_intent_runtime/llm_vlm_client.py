"""Mock-by-default off-board LLM/VLM client stub.

Production client would maintain an HTTP / WebSocket session to an off-board
LLM / VLM endpoint (no local inference on the CM4 — see ADR-0004). Here we
ship a hardware-free heartbeat loop that:

- is importable without any LLM SDK or network connection
- emits a structured heartbeat log every `--heartbeat-interval` seconds
- terminates cleanly on SIGTERM / SIGINT (so systemd `stop` works)
- refuses `--real` mode (exit 2) until a real client is wired in

Mirrors the audio_listener stub pattern so systemd ExecStart never points at
a one-shot fixture (which would exit 0 and make systemctl report
`active (exited)` while looking healthy).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time
from types import FrameType

LOGGER = logging.getLogger("reachy_intent_runtime.llm_vlm_client")

_DEFAULT_HEARTBEAT_INTERVAL_S = 60.0


class _StopRequested(Exception):
    """Raised when a SIGTERM / SIGINT is received so the main loop can exit cleanly."""


def _install_signal_handlers() -> None:
    """Translate SIGTERM / SIGINT into a clean loop exit."""

    def _handler(signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("[llm_vlm_client] signal=%s -> shutting down", signum)
        raise _StopRequested()

    # SIGTERM is what systemd sends on `systemctl stop`. SIGINT covers Ctrl-C.
    # signal.signal must run in the main thread on POSIX, and SIGTERM is not
    # available on Windows. We accept best-effort installation.
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):  # pragma: no cover
            signal.signal(sig, _handler)


def run_heartbeat_loop(
    interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
    max_iterations: int | None = None,
) -> int:
    """Run the mock heartbeat loop until SIGTERM or `max_iterations` reached.

    `max_iterations` exists so unit tests can drive the loop without sleeping
    the test runner. In production (`main()`) it is `None` -> loop forever.
    Returns 0 on graceful shutdown.
    """

    _install_signal_handlers()
    LOGGER.info(
        "[llm_vlm_client] mock heartbeat starting interval_s=%.1f max_iterations=%s",
        interval_s,
        max_iterations,
    )
    iteration = 0
    try:
        while max_iterations is None or iteration < max_iterations:
            LOGGER.info("[llm_vlm_client] mock heartbeat iteration=%d", iteration)
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            time.sleep(interval_s)
    except _StopRequested:
        LOGGER.info("[llm_vlm_client] graceful shutdown after iteration=%d", iteration)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m reachy_intent_runtime.llm_vlm_client`."""
    parser = argparse.ArgumentParser(
        description="Reachy Mini off-board LLM/VLM client (mock heartbeat by default)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="run the mock heartbeat loop (default)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="opt-in real off-board client (not implemented in stub)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=_DEFAULT_HEARTBEAT_INTERVAL_S,
        help="seconds between heartbeat log lines (default: 60)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="stop after N heartbeats (testing aid; default: run forever)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.real:
        print(
            "[llm_vlm_client] --real mode is opt-in and not implemented in stub; "
            "wire your off-board LLM/VLM HTTP/WebSocket client here.",
            file=sys.stderr,
        )
        return 2

    return run_heartbeat_loop(
        interval_s=args.heartbeat_interval,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
