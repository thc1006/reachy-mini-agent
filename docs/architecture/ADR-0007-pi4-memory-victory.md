# ADR-0007: Pi 4 4GB memory victory — two-knob daemon attack (rm HF token + MALLOC_ARENA_MAX=2)

- **Status**: Accepted (deployed and verified 2026-05-29)
- **Date**: 2026-05-29
- **Author**: hctsai1006
- **Supersedes**: none
- **Related**: ADR-0006 (mem0-bridge, deferred Phase 2), `project_brain_on_pi_2026_05_29`, `project_daemon_attack_2026_05_29`, `feedback_daemon_hf_central_relay_storm`, `feedback_glibc_arena_bloat_aarch64`, `reference_pi4_memory_budget_2026_05_29`

---

## 1. Context

After brain-on-Pi shipped (Wave 2/3 of 2026-05-29 — see `docs/architecture/brain-on-pi-2026-05-29.md`), in-process observation showed `reachy-brain.service` being SIGKILLed by systemd every ~15 minutes on `MemoryMax=1800M` (now that DTB cgroup_disable was fixed and the limit was actually enforced).

Initial hypothesis (anchored on the previous day's MediaPipe leak work — see `feedback_mediapipe_aarch64_leak`) was that brain was still leaking. A `ps -eo pid,rss,comm --sort=-rss | head -10` pass overturned this in one second:

| Process | RSS |
|---|---|
| `reachy-mini-daemon` (Pollen) | **1791–2007 MB** |
| `robot_brain` (ours) | 689 MB |
| `tailscaled` | 76 MB |
| `gpio_shutdown.py` | 22 MB |

The daemon was 2.6× the brain. The watchdog was firing because daemon + brain + system baseline together exceeded available RAM; brain hit the cap not because it was leaking, but because there wasn't enough headroom left.

Three parallel research agents (D1 process forensics, D2 Pollen community signal, D3 daemon WebRTC source walk) independently converged on two compounding root causes:

1. **HF central signalling relay reconnect storm.** `reachy_mini.media.central_signaling_relay` retries Hugging Face's central WebSocket at ~1 Hz with no exponential backoff when the token is stale or the central rate-limits. 17,000+ reconnect cycles per boot. Each cycle leaks a small amount of Python/GLib/Rust (pyo3 tokio) state.
2. **glibc multi-arena bloat on aarch64.** Pi 4 quad-core defaults to `MALLOC_ARENA_MAX = 8 × ncpu = 32 arenas`. Each arena can grow to ~64 MB heap. Daemon's `/proc/PID/smaps` showed ~2.4 GB anonymous mappings dominated by glibc arenas, not Python heap. The reconnect storm of #1 above was the trigger that fed all 32 arenas; glibc never released them back to the OS.

Forensics evidence chain in `project_daemon_attack_2026_05_29`.

## 2. Decision

**Two-knob combined attack, deployed and verified 2026-05-29:**

### Knob 1 — disable HF central relay by moving the token out of reach

```bash
mv ~/.cache/huggingface/token ~/.cache/huggingface/token.bak-2026-05-29
sudo systemctl restart reachy-mini-daemon.service
```

The relay subsystem reads the token at startup; absent token = relay never starts = no reconnect storm. LAN / Tailscale / loopback WebRTC paths are completely unaffected (they're separate peer connections, not routed through HF central).

### Knob 2 — cap glibc arenas via systemd drop-in

`/etc/systemd/system/reachy-mini-daemon.service.d/override.conf`:
```ini
[Service]
Environment=MALLOC_ARENA_MAX=2
```

Same drop-in applied to `~/.config/systemd/user/reachy-brain.service.d/override.conf` for the brain unit.

Forces glibc to multiplex all threads onto 2 arenas instead of 32. Caps heap ceiling at ~128 MB instead of ~2 GB. Costs a small amount of allocator concurrency — acceptable trade on Pi 4 4GB.

### Why two knobs, not one

| Knob alone | Daemon RSS | Watchdog cycle |
|---|---|---|
| rm HF token only | ~600 MB (still high; arenas stay big from earlier churn) | ~45 min |
| `MALLOC_ARENA_MAX=2` only | ~400 MB (relay storm still feeds new alloc/free churn) | ~30 min |
| **Both** | **178 MB** | **hours** |

The HF relay storm is the *trigger*; glibc arena multiplication is the *amplifier*. Removing either alone leaves the other doing damage. The combination is strictly better than the sum.

## 3. Consequences

### Positive (verified)

- **Daemon RSS: 1791–2007 MB → 178 MB** (-1800 MB, ~91%)
- **Available memory: 762 MB → 2612 MB** (+1850 MB)
- **zram swap usage: 1.4 GB → 0** (kernel auto-evicts when pressure drops)
- **Brain watchdog cycle: ~15 min → hours** (24–48 hr soak still in flight to confirm "indefinite")
- **HF central relay reconnects: 17,000+/boot → 0**
- Headroom to defer ADR-0006 (mem0-bridge) without immediate OOM risk — fastembed + Mem0 + Qdrant ~400 MB on Pi stays Phase-1 acceptable
- Two systemd drop-in files + one `mv` — total ops surface is trivial; reversible in ~30 sec

### Negative / accepted trade-offs

- **Lost: HF App-Store remote-view path.** Not currently used; if a future user wants the HF web demo to see robot's view, must restore token + accept the storm (or wait for upstream backoff fix).
- **Lost: ~10% multi-thread allocator concurrency** (2 arenas vs 32). Unmeasurable in our workload — daemon and brain are I/O-bound, not allocator-bound.
- **Did not fix the upstream bugs.** Pollen's relay still has no backoff; glibc's default arena policy is still wrong for embedded aarch64. Both are workarounds, not fixes. If we ever switch boards (Pi 5, CM5) the relay storm reappears; the `MALLOC_ARENA_MAX` env stays a good idea on any small-RAM aarch64 box.

### Risk

- Upstream SDK update may re-enable relay even without token (low probability — code path checks token presence first; would have to be an explicit policy change).
- `MALLOC_ARENA_MAX=2` is documented glibc behavior since 2.10; no regression risk across foreseeable libc versions.

## 4. Alternatives Considered

| Alternative | Verdict | Why rejected |
|---|---|---|
| **`reachy-mini-daemon --no-media`** | Rejected | Too aggressive — brain needs `unixfdsink` from daemon's GStreamer pipeline for audio/video taps. Would kill local WebRTC too. |
| **Ship ADR-0006 mem0-bridge first** | Deferred | Saves ~380 MB *on the brain*, but daemon was the actual problem. Solving brain first would have masked the daemon issue and OOM cycle would return weeks later. Bridge stays valuable for Phase 2 (memory hygiene, multi-tenant), just not the urgent fix. |
| **Remove brain MediaPipe Hand path** | Already done (#94 Wave6-P2) | Was a prerequisite, not a new option. Cleaning brain wouldn't have closed the ~1500 MB daemon gap. |
| **Upgrade to Pi 5 (8 GB)** | Rejected for now | BOM cost (~$80/unit + new case + new PSU), no field-deployment plan, doesn't fix the *defect* — Pi 5 with same defaults would still hit relay storm + arena bloat, just slower. Address root cause first; treat hardware upgrade as a separate orthogonal decision. |
| **Patch Pollen daemon to add reconnect backoff** | Deferred | Right long-term fix; would need upstream PR + maintainer review + version pinning. `mv token` achieves same user-visible result in seconds, costs nothing, reverts cleanly. Open issue with Pollen as follow-up (P3). |
| **Switch to jemalloc / mimalloc via `LD_PRELOAD`** | Rejected | Pollen daemon includes a Rust pyo3 module that statically links `jemalloc-sys`. Double-init at preload time causes segfault. mimalloc aarch64 wheels are unstable. `MALLOC_ARENA_MAX=2` is zero-dependency. |
| **`MALLOC_ARENA_MAX=1`** | Rejected | Causes measurable lock contention on multi-thread workloads (GStreamer pipeline). =2 is the community-validated Pi/embedded sweet spot. |

## 5. Validation

Post-deploy verification commands (kept in `reference_pi4_memory_budget_2026_05_29` for routine audit):

```bash
# Available memory
free -m | awk 'NR==2 {print "available:", $7, "MB"}'   # expect >2400

# Daemon RSS
ps -o rss= -p $(pgrep -f reachy_mini.*daemon) | head -1   # expect <300 (KB)

# Brain RSS
ps -o rss= -p $(pgrep -f robot_brain) | head -1   # expect <900 (KB)

# Relay storm dead
journalctl -u reachy-mini-daemon.service --since '10 minutes ago' \
  | grep -c central_signaling_relay   # expect 0

# cgroup memory controller still enforced (precondition)
awk '/memory/ {print $4}' /proc/cgroups   # expect 1

# MALLOC_ARENA_MAX actually applied to daemon
cat /proc/$(pgrep -f reachy_mini.*daemon)/environ \
  | tr '\0' '\n' | grep MALLOC_ARENA_MAX   # expect MALLOC_ARENA_MAX=2
```

All five checks pass as of 2026-05-29 evening.

## 6. Rollback

If a future user needs HF App-Store remote view:
```bash
mv ~/.cache/huggingface/token.bak-2026-05-29 ~/.cache/huggingface/token
sudo systemctl restart reachy-mini-daemon.service
# Expect daemon RSS to start climbing again; budget accordingly.
```

If `MALLOC_ARENA_MAX=2` ever shows up as a perf regression:
```bash
sudo systemctl edit reachy-mini-daemon.service   # remove the Environment line
sudo systemctl daemon-reload
sudo systemctl restart reachy-mini-daemon.service
```

Both knobs are independently reversible.

## 7. Follow-ups

1. **24–48 hr soak** to confirm watchdog cycle truly extends to "indefinite" rather than "merely longer."
2. **Pollen upstream issue / PR** for HF central relay exponential backoff — long-term right fix. Defer until soak data confirms there are no remaining bloat sources.
3. **Revisit ADR-0006 mem0-bridge** when (a) elder-care feature flips on in production and search latency matters, or (b) we need to deploy a second brain instance on the same Pi for A/B.
4. **Document `MALLOC_ARENA_MAX=2` as a default in any Reachy Mini Pi 4 deployment guide** — applies to every multi-threaded Python service we run on this hardware class.
