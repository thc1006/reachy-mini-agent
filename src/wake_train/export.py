"""Export trained classifier head to ONNX.

The bundled openWakeWord heads (alexa.onnx etc.) take a 4D tensor
(batch, 1, 16, embedding_dim) representing 16 stacked embedding frames and
emit a single logit. We mirror that contract so the new model is a drop-in:
the Phase B1 baseline script can load it via wakeword_model_paths=[onnx_path]
without any other change.

Smoke test: run the exported model on a zero tensor of the right shape and
confirm a finite scalar comes out.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import WakeConfig
from .train import _build_head

log = logging.getLogger(__name__)

OWW_FRAME_STACK = 16  # number of embedding frames per inference window


@dataclass
class ExportResult:
    onnx_path: Path
    input_shape: tuple[int, ...]
    bytes: int


def export(cfg: WakeConfig) -> ExportResult:
    import torch
    import torch.nn as nn

    weights_path = cfg.artifacts_dir / f"{cfg.name}.head.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing trained head: {weights_path}")
    blob = torch.load(weights_path, map_location="cpu", weights_only=True)
    head = _build_head(blob["embedding_dim"], blob["hidden_dim"],
                       blob["n_layers"], dropout=0.0)
    head.load_state_dict(blob["state_dict"])
    head.eval()

    embedding_dim = blob["embedding_dim"]

    class WindowHead(nn.Module):
        """Wrap the per-frame head into an OWW-compatible windowed model.

        Input  : (batch, 1, OWW_FRAME_STACK, embedding_dim) — averaged across the
                 16-frame window before scoring, matching the bundled v0.1 heads.
        Output : (batch, 1) — wake probability after sigmoid.
        """

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # average pool over the 16-frame stack to a single embedding
            squeezed = x.squeeze(1).mean(dim=1)
            return torch.sigmoid(self.inner(squeezed))

    wrapped = WindowHead(head).eval()
    dummy = torch.zeros(1, 1, OWW_FRAME_STACK, embedding_dim)

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        cfg.onnx_path,
        input_names=["onnx::Cast_0"],
        output_names=["score"],
        opset_version=15,
        dynamic_axes={"onnx::Cast_0": {0: "batch"}, "score": {0: "batch"}},
    )
    size = cfg.onnx_path.stat().st_size

    # Smoke run
    import onnxruntime as ort

    sess = ort.InferenceSession(str(cfg.onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
    score = float(out[0][0, 0])
    if not np.isfinite(score):
        raise RuntimeError(f"smoke run produced non-finite score: {score}")
    log.info("exported %s (%d bytes, smoke score=%.4f)", cfg.onnx_path, size, score)

    return ExportResult(
        onnx_path=cfg.onnx_path,
        input_shape=tuple(dummy.shape),
        bytes=size,
    )
