# wake_train — custom wake-word training pipeline

Trains a custom openWakeWord-compatible head (`嘿瑞奇` / `Hey Reachy`) and
exports an ONNX artifact that the Phase B1 baseline harness on the Reachy
CM4 (Pi 4 class) can load by path with no other code change.

Branch isolation: developed on `feature/wake-command-training`; runtime
integration into `src/robot_brain.py` is intentionally **not** in this
branch — keep this module a pure training line.

## Two scopes

| | POC | PROD |
|---|---|---|
| Voices (Piper) | 2 | 8+ |
| Phrases | 2 (hei_ruiqi, hey_reachy) | 5 |
| Real recordings | 0 | 50 |
| Synth positives | ~800 raw + augmentation | ~5000 raw + augmentation |
| Negative shards (HF dscripka/openwakeword_features) | 2 (~3 GB) | 16 (~24 GB) |
| Training steps | 50k | 200k |
| Wall time on 5090 | ~2 h | ~half a day |
| Disk | ~3–5 GB | ~30 GB |

Promote POC → PROD only after a POC eval clears the gates in
[#Gates](#gates).

## Pipeline

```
phrases -> Piper synth (jittered sampling) -> heavy aug
                                                |
                                                v
                                          features (openWakeWord embedding)
                                                |
                                                v
HF precomputed negatives (.npy) ---------> classifier head (PyTorch)
                                                |
                                                v
                                            ONNX export
                                                |
                                                v
                       latency (x86 CPU + Pi 4) + held-out recall + FAR
```

## Quickstart (POC on 5090, ~2 h)

```bash
# On the 5090 host (Tailscale 100.124.198.14)
git clone https://github.com/thc1006/reachy-mini-agent ~/reachy-mini
cd ~/reachy-mini
git checkout feature/wake-command-training

source ~/wake_train/bin/activate          # venv prepared during Phase B1 prep
pip install -q piper-tts openwakeword onnxruntime numpy soundfile \
              torch huggingface_hub

bash scripts/wake_train/run_poc.sh
```

Outputs:

* `data/wake_train/poc/synth/<voice>/<slug>/<seed>.wav` — Piper synth
* `data/wake_train/poc/aug/<voice>/<slug>/<seed>_augNN.wav` — augmented
* `data/wake_train/poc/neg/negative_features_*.npy` — HF cache
* `data/wake_train/poc/manifest.json` — training manifest
* `artifacts/wake/poc/hei_ruiqi_poc_v0.head.pt` — torch weights
* `artifacts/wake/poc/hei_ruiqi_poc_v0.onnx` — exported model
* `artifacts/wake/poc/hei_ruiqi_poc_v0.report.json` — latency + accuracy

## Verify on the robot

```bash
# scp the ONNX to the Pi
scp artifacts/wake/poc/hei_ruiqi_poc_v0.onnx \
    reachy@reachy-mini:~/wake_artifacts/

# then on the Pi
WAKE_ONNX=~/wake_artifacts/hei_ruiqi_poc_v0.onnx \
    bash scripts/wake_train/verify_pi.sh
```

The baseline (5 bundled heads) measured 29.6 ms / 2.7× realtime — adding the
custom head should land near 30–31 ms / ≥ 2.6× (~1 ms per added head; the
shared mel preprocessing dominates).

## Gates (POC → PROD)

Promote only when **all** of these hold on POC artifacts:

1. ONNX export succeeds; smoke run returns a finite score.
2. x86 CPU latency mean ≤ 5 ms / p99 ≤ 8 ms (proxy for ≤ 35 ms on Pi 4).
3. Held-out recall ≥ 0.85 at threshold 0.5.
4. Negative FAR ≤ 1.0/hr at threshold 0.5 against the HF neg corpus.
5. Pi 4 verification mean ≤ 32 ms / p99 ≤ 34 ms.

If a gate fails, iterate on the POC config — do **not** burn the half-day
PROD budget chasing a recipe that hasn't proven itself on the POC scale.

## Production additions

Once gates pass:

```bash
SCOPE=prod bash scripts/wake_train/run_poc.sh
```

Before that:

* Drop 50 real human recordings of the wake phrase(s) into `_wake_probe_real/`
  (any sane sub-layout — manifest walks recursively as long as
  `_wake_probe_real/<voice>/<slug>/<...>.wav`).
* Confirm 24 GB free on the training disk.
* Re-run the gate check after PROD eval — the bar is raised
  (recall ≥ 0.95, FAR ≤ 0.5/hr).

## Module layout

```
src/wake_train/
  config.py        POC / PROD profiles (dataclasses, no heavy deps)
  phrases.py       canonical wake-phrase catalog
  synth/piper.py   Piper TTS positive synthesis
  augment.py       numpy-only aug primitives + pipeline
  negatives.py     HuggingFace precomputed-feature download
  manifest.py      build training manifest JSON
  train.py         PyTorch classifier head training
  export.py        ONNX export + smoke run
  eval/            latency + accuracy probes
  cli.py           argparse CLI (synth/augment/negatives/manifest/train/export/eval/all)
```

## Notes / pitfalls

* `openwakeword.utils.AudioFeatures` exposes `._raw_embeddings` — that
  attribute is documented to be stable but unprefixed; if a future release
  breaks it, `train.py` / `eval/accuracy.py` need the matching update.
* Piper voices live under `$WAKE_TRAIN_VOICES` (default
  `~/wake_train/voices`). Each voice needs both the `.onnx` and the
  `.onnx.json` config — Phase B1 prep already downloaded
  `zh_CN-huayan-medium`.
* The ONNX I/O contract mirrors the bundled v0.1 heads: input
  `(batch, 1, 16, embedding_dim)` (16-frame stack), output `(batch, 1)`
  sigmoid. Keep it that way so the Pi-side baseline harness stays a
  zero-edit drop-in.
