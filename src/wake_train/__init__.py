"""Wake-word custom training pipeline for Reachy Mini.

Two scopes:
  POC  — 2 TTS voices + heavy augmentation + ~2k synthetic positives + 0 real
         recordings. Targets ~3-5 GB disk and ~2 h on a single 5090.
  PROD — many TTS voices + 50 real recordings + full openWakeWord negative
         corpus. Targets ~30 GB and ~half a day.

Inference target is the Reachy Mini CM4 (Pi 4 class). Phase B1 baseline
measured 29.6 ms per 80 ms frame for 5 bundled heads (2.7x realtime); shared
mel preprocessing dominates and each extra classifier head costs only ~1 ms,
so adding our custom head fits the existing budget.

CLI entrypoint (run from repo root with src/ on PYTHONPATH):
    PYTHONPATH=src python -m wake_train.cli all --scope poc
"""

__all__ = [
    "config",
    "phrases",
    "augment",
    "negatives",
    "manifest",
    "synth",
    "train",
    "export",
    "eval",
]
