#!/usr/bin/env bash
# Run the bundled-models baseline harness with our custom ONNX appended.
# Confirms the new head loads cleanly and per-frame latency is still under the
# Pi 4 budget (~30 ms baseline + ~1 ms per extra head).
#
# Run on the Reachy CM4:
#     bash scripts/wake_train/verify_pi.sh
set -euo pipefail

VENV="${WAKE_VENV:-$HOME/wake_venv}"
PY="$VENV/bin/python"
ONNX="${WAKE_ONNX:-$HOME/wake_artifacts/hei_ruiqi_poc_v0.onnx}"

if [ ! -f "$ONNX" ]; then
  echo "ONNX not found: $ONNX (scp it from the training host first)" >&2
  exit 2
fi

"$PY" - <<PY
import json, time, statistics, resource
import numpy as np
from openwakeword.model import Model
import openwakeword

bundled = openwakeword.get_pretrained_model_paths()
custom = "$ONNX"
mdl = Model(wakeword_model_paths=bundled + [custom])
print("[load] models:", list(mdl.models.keys()))

FRAME, SR, N = 1280, 16000, 1000
for _ in range(20):
    mdl.predict(np.random.randint(-2000, 2000, size=FRAME, dtype=np.int16))

lats = []
for _ in range(N):
    chunk = np.random.randint(-2000, 2000, size=FRAME, dtype=np.int16)
    t = time.perf_counter()
    mdl.predict(chunk)
    lats.append((time.perf_counter() - t) * 1000)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
out = dict(
    n_models=len(mdl.models),
    mean_ms=round(statistics.mean(lats), 3),
    p95_ms=round(sorted(lats)[int(N*0.95)], 3),
    p99_ms=round(sorted(lats)[int(N*0.99)], 3),
    rss_MB=round(rss, 1),
)
print(json.dumps(out, indent=2))
PY
