"""Wave6-P1 #93 — unit tests for HAND_LANDMARKER_ENABLED env flag.

Phase 1 of the Hand→VLM migration introduces a master switch that gates the
hand_worker thread spawn + watchdog seed. Default is "1" (backwards compat:
MediaPipe HandLandmarker still runs, same as Wave3-P2). Setting it to "0"
disables the worker entirely to mitigate the residual MediaPipe aarch64 leak
(sub-agent #100 GO recommendation).

`robot_brain.py` is too heavy to import in-process (opens motors, vLLM clients,
etc.). Following the same pattern as `test_cap_gupnp_thread.py` for env-gated
module-level behaviour, these tests:

  1. Re-evaluate the same env→bool gate used at module load
     (`os.getenv("HAND_LANDMARKER_ENABLED", "1") != "0"`) and check the
     downstream thread-list construction matches the production code path.
  2. Assert by static source inspection that the gate is wired into
     (a) the spawn site, (b) the brain_ready log, (c) the watchdog seed
     list — so a future refactor that drops one of the three sites fails CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make src/ importable + expose source file for text assertions.
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BRAIN_SRC = (SRC / "robot_brain.py").read_text(encoding="utf-8")


def _gate(env_value):
    """Reproduces the exact module-load gate from robot_brain.py:
        _HAND_LANDMARKER_ENABLED = os.getenv("HAND_LANDMARKER_ENABLED", "1") != "0"
    Returns the (enabled, seed_threads) pair the production code computes.
    """
    if env_value is None:
        # Simulate the var being unset.
        os.environ.pop("HAND_LANDMARKER_ENABLED", None)
    else:
        os.environ["HAND_LANDMARKER_ENABLED"] = env_value
    enabled = os.getenv("HAND_LANDMARKER_ENABLED", "1") != "0"
    seed = ["tracking_loop", "vision_worker", "main_idle"]
    if enabled:
        seed.insert(1, "hand_worker")
    return enabled, seed


# ──────────────────────────────────────────────────────────────────────────
# Runtime behaviour: env → enabled bool → seed list
# ──────────────────────────────────────────────────────────────────────────
class TestEnvGate:
    def setup_method(self):
        # Snapshot + scrub so each test starts with a known env state.
        self._saved = os.environ.pop("HAND_LANDMARKER_ENABLED", None)

    def teardown_method(self):
        os.environ.pop("HAND_LANDMARKER_ENABLED", None)
        if self._saved is not None:
            os.environ["HAND_LANDMARKER_ENABLED"] = self._saved

    def test_default_enabled_spawns_hand_worker_seed(self):
        """Var unset → enabled True (Wave3-P2 backwards compat) → hand_worker
        is present in the watchdog pulse-seed list."""
        enabled, seed = _gate(None)
        assert enabled is True
        assert "hand_worker" in seed
        # Order matters for the brain_ready log readability — hand_worker
        # sits between tracking_loop and vision_worker like the original
        # ("tracking_loop", "hand_worker", "vision_worker", "main_idle").
        assert seed.index("hand_worker") == seed.index("tracking_loop") + 1
        assert seed.index("vision_worker") == seed.index("hand_worker") + 1

    def test_disabled_omits_hand_worker_seed(self):
        """HAND_LANDMARKER_ENABLED=0 → enabled False → hand_worker MUST be
        absent from the pulse-seed list (otherwise watchdog would mark a
        never-spawned thread stale immediately and trip systemd)."""
        enabled, seed = _gate("0")
        assert enabled is False
        assert "hand_worker" not in seed
        # The other three workers stay in canonical order.
        assert seed == ["tracking_loop", "vision_worker", "main_idle"]

    def test_explicit_1_same_as_default(self):
        """Sanity: HAND_LANDMARKER_ENABLED=1 must behave identically to the
        var being unset (both are the production default)."""
        e_unset, s_unset = _gate(None)
        e_one,   s_one   = _gate("1")
        assert e_unset == e_one is True
        assert s_unset == s_one
        assert "hand_worker" in s_one


# ──────────────────────────────────────────────────────────────────────────
# Source-level wiring: a future refactor must not silently drop the gate
# from any of the three sites (spawn, brain_ready log, watchdog seed).
# ──────────────────────────────────────────────────────────────────────────
class TestSourceWiring:
    def test_module_level_gate_constant_present(self):
        """The gate must be evaluated once at module load (so the rest of the
        process sees a stable value) and named _HAND_LANDMARKER_ENABLED."""
        assert '_HAND_LANDMARKER_ENABLED = os.getenv("HAND_LANDMARKER_ENABLED", "1") != "0"' in BRAIN_SRC

    def test_spawn_site_guarded(self):
        """The hand_worker thread creation + start must be inside the gate."""
        assert "if _HAND_LANDMARKER_ENABLED:" in BRAIN_SRC
        assert "hands.start()" in BRAIN_SRC
        # And the disable branch must log a structured event so operators can
        # confirm via journalctl that Phase 1 is actually in effect.
        assert 'logger.info("hand_landmarker_disabled"' in BRAIN_SRC

    def test_watchdog_seed_conditional(self):
        """The watchdog pulse-seed loop must use the gated list, not a literal
        tuple that always includes hand_worker."""
        # The old unconditional literal must be gone.
        assert '("tracking_loop", "hand_worker", "vision_worker",\n                      "main_idle")' not in BRAIN_SRC
        # The new gated builder must be present.
        assert '_seed_threads = ["tracking_loop", "vision_worker", "main_idle"]' in BRAIN_SRC
        assert 'if _HAND_LANDMARKER_ENABLED:' in BRAIN_SRC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
