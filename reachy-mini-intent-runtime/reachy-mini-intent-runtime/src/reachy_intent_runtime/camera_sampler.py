"""Mock-by-default camera sampler stub.

Production sampler would grab frames from the Reachy Mini camera and forward
them downstream (off-board VLM, optional local face/pose). The mock just emits
synthetic ticks so `python -m` and contract tests can exercise the loop.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class MockCameraSampler:
    """Synthetic frame source. Yields N frames + exits."""

    fps: int = 5

    def sample_n(self, n: int) -> list[dict]:
        return [{"frame_id": i, "fps": self.fps, "mock": True} for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m reachy_intent_runtime.camera_sampler`."""
    parser = argparse.ArgumentParser(description="Reachy Mini camera sampler (mock by default)")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="opt-in real hardware mode (gated; not implemented in stub)",
    )
    args = parser.parse_args(argv)

    if args.real_hardware:
        print("[camera_sampler] real-hardware mode is opt-in and not implemented in stub")
        return 2

    sampler = MockCameraSampler()
    frames = sampler.sample_n(args.frames)
    for frame in frames:
        print(f"[camera_sampler] frame={frame}")
    print(f"[camera_sampler] done total={len(frames)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
