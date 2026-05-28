# Brain-on-Pi topology (Plan B) — 2026-05-29

Status: deployed. Supersedes the s1-hosted brain that died on 2026-05-28 (see
[`project_s1_brick_2026_05_28`](../../) memory).

## TL;DR

`robot_brain.py` now runs as a systemd **user** unit on the Reachy Mini Pi (CM4
wireless), co-located with the daemon. STT / LLM / TTS run remotely on the
TWCC NGC container `vllm0528` over Tailscale HTTP. WebRTC stays on loopback
so wifi / Tailscale jitter cannot kill the media plane.

## Old vs new topology

```
OLD (pre-2026-05-28, s1 host) — DEAD
┌──────────────────┐  WebRTC/WS (Tailscale UDP)  ┌────────────────────┐
│ Reachy Mini (Pi) │ ◄────────────────────────► │ s1 (2× RTX 3090)    │
│  daemon :8000    │      jitter = stall         │  robot_brain         │
│                  │                              │  vLLM / Whisper / Kokoro │
└──────────────────┘                              └────────────────────┘

NEW (2026-05-29, brain on Pi)
┌────────────────────────────────────────┐   HTTP (Tailscale TCP)   ┌────────────────────┐
│ Reachy Mini (Pi CM4, aarch64)          │ ────────────────────►   │ vllm0528 (V100×2)  │
│  ┌──────────────┐ loopback             │   STT  :9000             │  vLLM :8000        │
│  │ daemon :8000 │ ◄──── WebRTC ────────┤   LLM  :8000             │  Whisper :9000     │
│  └──────────────┘                       │   TTS  :8880             │  Kokoro  :8880     │
│  ┌──────────────────────────────────┐   │                          └────────────────────┘
│  │ robot-brain.service (user unit)   │   │
│  │  Type=notify  WatchdogSec=120     │   │
│  │  MemoryMax=1800M (enforced)       │   │
│  │  AllowedCPUs=2,3  Nice=-5         │   │
│  └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

## Why Plan A (brain on vllm0528) failed

Original plan was to host the brain on `vllm0528` for zero LLM RTT. **Blocked
by Tailscale userspace networking**: the TWCC NGC container has no kernel
module access, so `tailscaled` runs `--tun=userspace-networking` and exposes
only a SOCKS5 proxy on `localhost:1055`. SOCKS5 is **TCP-only** — WebRTC
peer connection needs UDP for ICE / media transport. Brain-on-vllm0528 could
talk LLM/STT/TTS locally but could not establish the WebRTC peer to the Pi
daemon. Hard block, no workaround short of getting kernel networking
(requires TWCC platform change, out of scope).

## Why Plan B works

- WebRTC stays on **loopback** (`127.0.0.1`) — daemon and brain both on the Pi,
  zero network in the media plane. No DERP, no Tailscale, no wifi.
- LLM / STT / TTS go out **only as HTTP**, which works fine over Tailscale TCP
  even when userspace-mode is in play on the other end.
- Brain process is small (~700 MB resident with elder-care off) — Pi 4 has
  4 GB and we can cap at 1.8 GB to leave room for daemon (~600 MB) and OS.

## Resource flow (per dialog turn)

| hop | path | typical RTT |
|---|---|---|
| mic capture | XMOS → daemon → loopback WebRTC → brain | <5 ms |
| STT | brain → `vllm0528:9000` (faster-whisper, large-v3-turbo int8_float16) | 250-400 ms |
| LLM | brain → `vllm0528:8000` (Qwen3.6-35B-A3B-AWQ, 78 tok/s) | 400-1500 ms |
| TTS synth | brain → `vllm0528:8880` (Kokoro CPU mode) | ~3 s per 1 s audio (CPU-bound, see Known gaps) |
| speaker out | brain → loopback WebRTC → daemon → speaker | <5 ms |

Local fallback: if `WHISPER_URL` is unset, brain lazy-imports `faster-whisper`
and runs STT on the Pi CPU. Skipped at startup when `WHISPER_URL` is set
(commit `1207877`) to avoid loading a 1+ GB model we never call.

## Pi systemd hardening summary

The unit is a **user service** under `user@1000` (no need for root), launched
via `systemctl --user enable --now reachy-brain.service`.

Key hardening (Wave 1–3, applied 2026-05-29):

- `Type=notify` + `WatchdogSec=120` — brain pings sd_notify on every dialog
  tick. 120 s silence ⇒ kernel SIGKILL + systemd restart.
- `MemoryMax=1800M` (**now enforced**, see cgroup story below).
- `AllowedCPUs=2,3` — pin to CM4 big cores via cpuset delegation, leave 0/1
  for daemon and kernel softirqs. Requires `Delegate=memory cpu cpuset` on
  `user@1000.service` drop-in.
- `Nice=-5` and `OOMScoreAdjust=-500` — protect against being preempted by
  background app installs, last-priority OOM target.
- `Restart=on-failure`, `RestartSec=3`, `StartLimitBurst=10`.
- `ExecStartPre=/usr/bin/pkill -f reachy_brain_runner || true` to clean any
  lingering process after manual `systemctl --user stop`.

### cgroup memory enforcement — the DTB story

`MemoryMax=` was silently ignored for every systemd unit on the Pi until
2026-05-29. Root cause: the Reachy Mini OS image ships
`bcm2711-rpi-cm4.dtb` with `chosen.bootargs="... cgroup_disable=memory"`
baked in. The kernel boots with the memory cgroup controller **off**, so
systemd cannot impose memory limits regardless of unit syntax.

Fix: patch the DTB chosen.bootargs (or override via `/boot/cmdline.txt` if
the bootloader honours it on this image), reboot, verify with
`cat /proc/cgroups | grep memory` (column 4 must be `1`). Add zram swap
(`zramswap` package, 50 % of RAM) so the OOM killer has cushion before it
fires.

After fix: `systemctl --user show reachy-brain.service -p MemoryMax` returns
`1887436800` (1.8 GB) and a deliberate leak test trips the limit at ~1.8 GB
as expected. See `feedback_dtb_cgroup_disable_reachy.md`.

## vllm0528 — Tier 1+ winning config snippet

Currently deployed on `vllm0528:8000` (V100-SXM2-32GB ×2, NVLink NV2). Full
detail in [`reference_vllm0528`](../../) memory:

```bash
NCCL_P2P_LEVEL=NVL NCCL_P2P_DISABLE=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
nohup ~/venvs/v100-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/hctsai1006/models/Qwen3.6-35B-A3B-AWQ \
  --served-model-name qwen36-awq \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  > ~/vllm-logs/serve_tier1plus.log 2>&1 & disown
```

Versus baseline (`--enforce-eager`, util 0.45, len 8192): completion
13.04 → 78+ tok/s, ~6× speedup. PIECEWISE CUDA Graphs is the dominant
factor; prefix-caching helps multi-turn warm path. Util raised from 0.45 to
0.85 after the TWCC zombie-PID lesson (`feedback_no_ab_restart_on_twcc.md`)
— single-launch, no A/B-by-restart.

Whisper rebound to `0.0.0.0:9000` (was loopback-only) so the brain on the Pi
can reach it; `cpu_threads=4` tuned for the container's CPU share. Kokoro
still CPU-mode pending GPU re-attempt B4.

## Elder care opt-in env

Wave 3 P6/P7/P8 shipped behind `ELDER_CARE_MODE` flag (currently **off** in
production):

| env | default | meaning |
|---|---|---|
| `ELDER_CARE_MODE` | `0` | master switch; everything below ignored when off |
| `ELDER_CARE_WEBHOOK_URL` | unset | async POST target for incident events |
| `ELDER_CARE_LOG_PATH` | `/var/log/reachy/elder_care.jsonl` | per-event JSONL |
| `ELDER_CARE_MOTION_LOCK_SEC` | `30` | minimum spacing between cue actions |

Features (when enabled): regex-precise keyword detection for distress/help
phrases, async webhook dispatch (no blocking the dialog loop), antenna +
LED cue with motion lock, numpy hoisted to module scope (was per-call).
See commit `a739298` for the A4 review fix-loop.

## Known gaps / Wave 4–5 candidates

1. **Kokoro on GPU** — CPU mode is ~3 s per 1 s audio. GPU re-attempt
   blocked on `onnxruntime-gpu` ABI mismatch with driver 535. Re-attempt
   tracked as B4. Real-time conversation works but TTS dominates latency.
2. **No API key on vllm0528:8000 / :8880** — single-user Tailnet, MEDIUM
   priority. Add `--api-key` to vLLM launcher when convenient.
3. **MediaPipe HandLandmarker leak** — mitigated by periodic recreate +
   state-gate (commit `b5f2a4f`). Real fix is upstream
   (mediapipe#5217 / #4785). Re-evaluate on mediapipe ≥0.10.21.
4. **ADR-0005 phases A → D** — intent-runtime wiring into prod brain is
   PROPOSED, not implemented. Next sprint.
5. **vllm0528 supervisord wrap** — services are still nohup. Should write a
   `supervisord.conf` snippet so a TWCC container restart auto-relaunches
   vLLM + Whisper + Kokoro instead of needing manual rehydrate.
6. **Brain ↔ vllm0528 over Tailscale direct** — currently goes via DERP some
   percentage of the time; check `tailscale netcheck` and tune for direct
   UDP if latency variance becomes a complaint.

## Cross-references

- `docs/architecture/webrtc-reconnect-strategy.md` — earlier study (still
  relevant for the loopback case, watchdog patterns ported as-is)
- Memory: `reference_vllm0528`, `project_brain_on_pi_2026_05_29`,
  `feedback_dtb_cgroup_disable_reachy`, `feedback_no_ab_restart_on_twcc`,
  `feedback_mediapipe_aarch64_leak`
- Commits: `7c83c05` (VLLM_HOST/PORT), `02f4c11` (REACHY_MEDIA_REMOTE
  opt-in), `1207877` (lazy whisper), `38589e6` (elder care), `b5f2a4f`
  (HandLandmarker leak mitigation)
