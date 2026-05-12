"""Export trained classifier head to ONNX matching the bundled openWakeWord
v0.1 head contract.

The bundled heads (alexa_v0.1.onnx etc.) expose:
    input   onnx::Flatten_0  shape=(1, 16, 96)
    output  13               shape=(1, 1)

We mirror that exactly so the Phase B1 baseline harness on the Pi 4 can pick
the new model up by path with no other change. The sigmoid is folded into the
exported graph so callers see a probability, matching the bundled heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import WakeConfig
from .train import FRAME_STACK, _build_classifier

log = logging.getLogger(__name__)

EMBEDDING_DIM = 96
BUNDLED_INPUT_NAME = "onnx::Flatten_0"


@dataclass
class ExportResult:
    onnx_path: Path
    input_shape: tuple[int, ...]
    bytes: int
    smoke_score: float


def export(cfg: WakeConfig) -> ExportResult:
    import torch
    import torch.nn as nn

    weights_path = cfg.artifacts_dir / f"{cfg.name}.head.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing trained head: {weights_path}")
    blob = torch.load(weights_path, map_location="cpu", weights_only=True)

    head = _build_classifier(blob["hidden_dim"], blob["n_layers"], dropout=0.0)
    head.load_state_dict(blob["state_dict"])

    class HeadWithSigmoid(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.sigmoid(self.inner(x))

    wrapped = HeadWithSigmoid(head).eval()
    # Match bundled head: fixed batch 1, shape (1, 16, 96)
    dummy = torch.zeros(1, FRAME_STACK, EMBEDDING_DIM)

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        cfg.onnx_path,
        input_names=[BUNDLED_INPUT_NAME],
        output_names=["score"],
        opset_version=15,
    )
    size = cfg.onnx_path.stat().st_size

    import onnxruntime as ort

    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    if tuple(in_meta.shape) != (1, FRAME_STACK, EMBEDDING_DIM):
        raise RuntimeError(
            f"ONNX input shape {in_meta.shape} != bundled (1, {FRAME_STACK}, {EMBEDDING_DIM})"
        )
    out = sess.run(None, {in_meta.name: dummy.numpy()})
    score = float(out[0][0, 0])
    if not np.isfinite(score):
        raise RuntimeError(f"smoke run produced non-finite score: {score}")
    log.info("exported %s (%d bytes, smoke score=%.4f)", cfg.onnx_path, size, score)

    return ExportResult(
        onnx_path=cfg.onnx_path,
        input_shape=tuple(in_meta.shape),
        bytes=size,
        smoke_score=score,
    )
