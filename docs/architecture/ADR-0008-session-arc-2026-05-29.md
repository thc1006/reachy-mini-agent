# ADR-0008: 2026-05-29 session arc — daemon attack victory + VLM live + Hand kill + Pollen issue #1165

- **Status**: Accepted
- **Date**: 2026-05-29
- **Author**: hctsai1006
- **Supersedes**: none
- **Related**: ADR-0006 (mem0-bridge, deferred), ADR-0007 (Pi 4 memory victory — the headline win this session shipped), `docs/architecture/brain-on-pi-2026-05-29.md`, `project_session_2026_05_29`, `project_brain_on_pi_2026_05_29`, `project_daemon_attack_2026_05_29`, `feedback_daemon_hf_central_relay_storm`, `feedback_glibc_arena_bloat_aarch64`, `feedback_vllm_cudagraph_oom_at_low_util`, `feedback_mediapipe_aarch64_leak`, `reference_pi4_memory_budget_2026_05_29`, `reference_moondream2_card1`, `reference_vllm0528`

---

## 1. Context

Starting state at the top of 2026-05-29 (post Wave 2/3 from previous session):

- Pi brain ran on Pi 4 4 GB under `MemoryMax=1800M` (cgroup actually enforced after DTB fix in Wave 2).
- `reachy-brain.service` was being SIGKILLed every **~15–18 minutes**. Assumed culprit: MediaPipe `HandLandmarker` C++ packet-graph leak (documented `feedback_mediapipe_aarch64_leak`).
- VLM path was still text-only `vllm0528:8000` placeholder; no real vision answers in production.
- ADR-0005 selected vLLM + Moondream2 on card 1 as Option D but had not landed.

Three intertwined problem domains needed to converge in one day: (a) confirm the leak source, (b) ship a real VLM endpoint, (c) find the actual memory hog that was driving the watchdog cycle.

## 2. Decisions (chronological, with commit SHAs)

### D1 — Wave 5/6 ship: VLM Option D live on `vllm0528:8002` card 1

Commit **`44695e9`**. Moondream2 brought up on the second V100 (card 1) of vllm0528, sharing the box with the existing `:8000` text model. Reference card: `reference_moondream2_card1`. This unblocked all downstream vision work for the rest of the session.

### D2 — Discovery: "Tier 1+ baseline" was an auto-trim artifact

While verifying Moondream2 throughput, found that the prior "Tier 1+" run had silently collapsed `kv_cache_auto_trim` to an effective `max_num_seqs=1`. The headline numbers were a single-stream measurement, not a true Tier 1+ baseline. Lesson captured in `feedback_vllm_cudagraph_oom_at_low_util`.

### D3 — vLLM relaunch with explicit knobs

Relaunched with `gpu_memory_utilization=0.72`, `max_num_seqs=4`, `--disable-custom-all-reduce`. Last flag identifies a 1Cat-vLLM fork bug interacting badly with V100 SXM2 NVLink topology. No commit (vllm0528 config), captured in `reference_vllm0528`.

### D4 — MediaPipe Hand kill: feature flag then full delete

Two-stage to keep the rollback window short:
- **Wave6-P1 (commit `3f197f6`)** — feature-flag the Hand path off.
- **Wave6-P2 (commit `421452d`)** — full delete, **-302 LOC**. Permanently removes the documented leak source from the brain process.

### D5 — Forensics that flipped the model: daemon dwarfs brain

One `ps -eo pid,rss,comm --sort=-rss | head -10` pass on the Pi:

| Process | RSS |
|---|---|
| `reachy-mini-daemon` (Pollen SDK) | **1791–2007 MB** |
| `robot_brain` (ours) | 689 MB |

Daemon was **~4× brain footprint**. The watchdog cycle was not really a brain leak — it was system pressure with daemon eating the headroom. Full forensics in `project_daemon_attack_2026_05_29`.

### D6 — Three parallel research streams (D1 / D2 / D3 sub-agents)

Same session, independent agents: **D1** smaps breakdown (~2.4 GB anon mappings dominated by glibc arenas, not Python heap), **D2** Pollen community signal (no prior public report), **D3** daemon WebRTC source walk (found `reachy_mini.media.central_signaling_relay` reconnecting with no exponential backoff).

### D7 — D3 game-changer: HF central-relay reconnect storm

`journalctl` showed **17,000+ reconnects since boot**, ~720 cycles/hr at `RECONNECT_INTERVAL=5s`. Each leaks Python / GLib / Rust (pyo3 tokio) state and churns glibc arenas. Captured `feedback_daemon_hf_central_relay_storm`.

### D8 — D1 × D3 cross-validation

glibc per-thread arena bloat × central-relay churn = compounding bloat engine. Removing either alone leaves the other doing damage; both is strictly better than the sum. Captured `feedback_glibc_arena_bloat_aarch64`.

### D9 — Daemon attack ship (the headline win — see ADR-0007 for the full decision)

Two-knob deploy:
1. `mv ~/.cache/huggingface/token ~/.cache/huggingface/token.bak-2026-05-29` — kills the relay subsystem (no token = relay never starts).
2. Systemd drop-in `Environment=MALLOC_ARENA_MAX=2` on `reachy-mini-daemon.service` (and matching drop-in on `reachy-brain.service`).

Result: daemon RSS **1791 MB → 178 MB (−1829 MB, −91%)**; Pi available memory **437 → 2182 MB (5×)**. Documented end-to-end in ADR-0007.

### D10 — Option A H1 RED fix: vision_worker was POSTing to text port

Commit **`05c40f5`**. `vision_worker` had been hitting `vllm0528:8000` (text endpoint) with images, crashing the model. Routed through `_ask_vision()` to the correct VLM port `:8002`. Production verification — real scene description came back:

> *"A person wearing headphones and a blue shirt is sitting in a room with a window and a desk"*

First real VLM answer in production this project has ever logged.

### D11 — Pollen GitHub issue #1165 filed

https://github.com/pollen-robotics/reachy_mini/issues/1165 — filed under `thc1006`, with full cross-validation evidence (D1 + D3 + the ADR-0007 fix), 7 edits during the day to add detail. Offered to submit a PR for exponential-backoff on the relay; awaiting maintainer response.

### D12 — ALSA mixer fix

`PCM,0` was at **5% (-57 dB)** — TTS_GAIN alone could never compensate. Set to **80% (-12 dB)**, `alsactl store` persisted. No commit (host config).

### D13 — Vision architecture realization

`vision_worker` uses a generic `"describe"` prompt and the dialog loop skips polling vision unless explicitly invoked. The LLM therefore can't answer specific questions like "what am I holding?" even though VLM is live. **Wave6-P4 in flight** to ship a richer prompt + a `query_vision` tool the LLM can call on demand.

### D14 — Session-arc documentation (this ADR)

ADR-0008 captures the arc so future agents see the full story (daemon attack as the headline, but VLM ship + Hand kill + Pollen contribution + audio fix as the supporting wins).

## 3. Consequences

### Positive (verified this session)

- **Pi available RAM 5×** (437 → 2182 MB) — see ADR-0007 validation block.
- **VLM live in production** with real scene description proof.
- **MediaPipe Hand leak permanently removed** (-302 LOC, no flag to flip).
- **Public Pollen GitHub issue #1165** — we are the first to publish this cross-validation.
- **Audio path fixed** end-to-end (ALSA + TTS_GAIN together).
- **Brain watchdog cycle ~15 min → hours**; 8 h soak in flight to confirm "indefinite".

### Negative / accepted trade-offs

- **HF App-Store remote-view path lost.** We don't use it; reversible by restoring the token.
- **Vision worker still uses generic `"describe"` prompt** — Wave6-P4 in flight to add `query_vision` tool.
- **Did not patch upstream** Pollen relay backoff or glibc arena defaults — `mv token` + `MALLOC_ARENA_MAX=2` are workarounds, not fixes. Issue #1165 is the long-term track.

## 4. Discoveries that changed our model

1. **"Tier 1+ baseline" was an auto-trim artifact** — `kv_cache_auto_trim` collapsed effective `max_num_seqs` to 1, so prior numbers were single-stream.
2. **Daemon dwarfs brain on Pi 4 4GB (~4× footprint).** Watchdog firing wasn't a brain leak — it was system pressure with daemon eating the headroom.
3. **HF central-relay storm is the real allocation engine** — 17,000+ reconnects/boot churns Python+Rust state continuously.
4. **glibc per-thread arena bloat is amplified by churn** — Pi 4's default `MALLOC_ARENA_MAX = 8 × ncpu = 32` arenas, each growing to ~64 MB. The relay storm fed all 32 and glibc never released.
5. **`vision_worker` had a bypass bug** — POSTing images to text port `:8000` crashed silently; needed routing through `_ask_vision()` to `:8002`.
6. **ALSA hardware mixer at 5% was the audio bottleneck**, not application gain — TTS_GAIN alone could not compensate −57 dB of hardware attenuation.

## 5. Lessons captured (cross-link to memory)

- `feedback_daemon_hf_central_relay_storm`, `feedback_glibc_arena_bloat_aarch64`, `feedback_vllm_cudagraph_oom_at_low_util`
- `reference_pi4_memory_budget_2026_05_29`, `reference_moondream2_card1`
- `project_daemon_attack_2026_05_29`, `project_brain_on_pi_2026_05_29`, `project_session_2026_05_29`
- ADR-0006 (mem0-bridge, deferred — Pi headroom now sufficient), ADR-0007 (the daemon attack decision; this session's biggest single win)

## 6. Validation

- Pi free RAM: **437 MB → 2182 MB (5×)**.
- daemon RSS: **1791 MB → 178 MB (−91%)**.
- VLM endpoint: **HTTP 200**, real scene description verified in production.
- Pollen GitHub issue **#1165 filed publicly** under thc1006.
- Production VLM proof string logged: *"A person wearing headphones and a blue shirt is sitting in a room with a window and a desk"*.
- ADR-0007 §5 validation block (5 commands, all pass).

## 7. Open follow-ups

1. **Wave6-P4 vision fixes** — richer prompt + `query_vision` tool (in flight).
2. **Pollen issue #1165** — monitor maintainer response; submit backoff PR if welcomed.
3. **ADR-0006 mem0-bridge** — deferred; Pi headroom now sufficient. Revisit only when elder-care flag flips on in production or a second brain instance lands on the same Pi.
4. **#73 motion queue `RESTART_FRESH`** — deferred (large effort, no current pain).
5. **supervisord migration on vllm0528** — deferred to a lower-risk window.
6. **8 h soak** to confirm the watchdog cycle truly stretched to "indefinite" rather than "merely longer."

## 8. Alternatives considered (and rejected)

| Alternative | Verdict | Why rejected |
|---|---|---|
| **Pi 5 upgrade (8 GB)** | Rejected | BOM cost (~$80/unit + case + PSU), no field plan, doesn't fix the *defect*. Pi 5 with same defaults still hits relay storm + arena bloat. |
| **`reachy-mini-daemon --no-media`** | Rejected | Too aggressive — brain needs `unixfdsink` from daemon's GStreamer pipeline for camera taps. Would kill local WebRTC. |
| **jemalloc / mimalloc via `LD_PRELOAD`** | Rejected | Daemon includes a Rust pyo3 module that statically links `jemalloc-sys` → double-init segfault. mimalloc aarch64 wheels unstable. `MALLOC_ARENA_MAX=2` is zero-dependency. |
| **STT swap to multilingual Whisper** | Rejected | User kept Breeze-ASR-25 — it supports English fine in this deployment. No need to add migration risk. |
| **Ship ADR-0006 mem0-bridge first** | Deferred | Saves ~380 MB *on the brain*, but daemon was the real problem. Would have masked the daemon issue and OOM cycle would return later. |

## 9. Note

This ADR documents the session arc; the per-decision deep-dives live in the linked ADRs and memory files. ADR-0007 is the headline mechanism; this ADR is the narrative wrapper that explains how we got there in one day.
