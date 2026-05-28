"""Mock-by-default audio listener stub.

Production listener would consume microphone PCM, run VAD + wake-word + stop-
phrase detection, and emit critical-stop events. Here we ship a hardware-free
mock that calls `orchestrator.route_stop()` synchronously, suitable for tests
and `python -m` smoke runs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .orchestrator_worker import OrchestratorWorker


@dataclass
class MockAudioListener:
    """Synchronous mock listener used in tests + smoke runs."""

    orchestrator: OrchestratorWorker

    def emit_stop(self) -> None:
        """Pretend we detected '停止' / 'stop' in the audio stream."""
        self.orchestrator.route_stop(source="audio_mock")


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m reachy_intent_runtime.audio_listener`."""
    parser = argparse.ArgumentParser(description="Reachy Mini audio listener (mock by default)")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="opt-in real hardware mode (gated; not implemented in stub)",
    )
    args = parser.parse_args(argv)

    if args.real_hardware:
        print("[audio_listener] real-hardware mode is opt-in and not implemented in stub")
        return 2

    from .motion_adapter import MockMotionAdapter
    from .scheduler import ActionScheduler

    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    orchestrator = OrchestratorWorker(scheduler=scheduler)
    listener = MockAudioListener(orchestrator=orchestrator)

    print(f"[audio_listener] starting mock={args.mock}")
    listener.emit_stop()
    print(f"[audio_listener] adapter_log={adapter.log}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
