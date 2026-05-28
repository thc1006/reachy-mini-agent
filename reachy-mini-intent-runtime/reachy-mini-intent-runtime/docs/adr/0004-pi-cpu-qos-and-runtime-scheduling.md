# ADR-0004: Pi / CM4 CPU QoS and Runtime Scheduling

- Status: Accepted
- Date: 2026-05-28
- Extends: ADR-0003 (Pi resource budget and worker split)

## Context

ADR-0003 settled the principle that the Pi-side runtime must stay responsive to
audio / stop / orchestrator events while motion executes. It deliberately left
the concrete OS-level mechanism unspecified ("nice, taskset, or systemd slices
after measuring Pi behavior").

Phase 6 fixes that gap. Reachy Mini Wireless ships on a Raspberry Pi CM4
(CM4104016: 4 cores, 4 GB RAM, 16 GB flash, Wi-Fi). The official conversation
app runs realtime audio, vision sampling, layered motion, and async tool
dispatch concurrently. Without an explicit QoS policy, a long dance loop can
starve the audio capture thread and the stop-phrase detector, which means the
robot becomes deaf the moment the user wants to interrupt it.

We also have user-facing constraints:

- the daemon and SDK already consume part of the 4 cores;
- no kernel patching, no out-of-tree schedulers, no custom sched_ext;
- everything must be testable on a developer laptop without systemd / cgroups;
- realtime scheduling is a privilege escalation surface — opt-in only.

## Decision

Adopt a **3-tier QoS strategy** using only stock systemd + cgroup v2 + POSIX
nice. Every Reachy Mini Pi-side process runs inside a single slice
(`reachy-runtime.slice`) and declares its tier via `CPUWeight=`, `CPUQuota=`,
`MemoryMax=`, and `Nice=`.

### Runtime component split

| Component | Tier | CPUWeight | CPUQuota | MemoryMax | Nice | Notes |
|---|---|---:|---:|---:|---:|---|
| `reachy-audio-listener` | 1 — sense | 900 | 80% | 64M | -5 | Top weight; deafness = safety risk. Realtime opt-in. |
| `reachy-orchestrator` | 2 — decide | 600 | 80% | 128M | -3 | Wraps classifier + scheduler. |
| `reachy-motion-worker` | 2 — act | 300 | 120% | 384M | 0 | Chunked dance / emotion executor. |
| `reachy-camera-sampler` | 3 — observe | 150 | 60% | 128M | 5 | May drop frames under pressure. |
| `reachy-llm-vlm-client` | 3 — assist | 100 | 40% | 64M | 10 | Thin off-board client; no local inference. |

The `reachy-runtime.slice` enables `CPUAccounting=`, `MemoryAccounting=`, and
`IOAccounting=` so operators can see real numbers (`systemd-cgtop`,
`systemctl status`) on the CM4.

### Realtime opt-in

`CPUSchedulingPolicy=fifo` / `rr` (POSIX `SCHED_FIFO` / `SCHED_RR`) are
**commented out** in every unit. They become opt-in only when
`bench_pi_runtime.sh --real-hardware` shows audio underruns or stop-latency
P95 above the SDD-04 budget. Enabling realtime requires:

1. an unmitigated benchmark showing the regression,
2. a code-review sign-off,
3. a follow-up ADR amendment recording the policy and CPU runtime limit.

### Off-board placement

No local VLM, no local LLM on the CM4 by default. The
`reachy-llm-vlm-client.service` is a thin HTTP / WebSocket client to an
off-board endpoint. This matches the official conversation app's guidance.

## Consequences

Trade-offs we accept:

- **No kernel maintenance**: stock systemd + cgroup v2 means upstream Raspberry
  Pi OS updates do not break us. We give up the finer-grained control a custom
  scheduler (sched_ext, BPF) would provide.
- **CPU oversubscription is allowed**: total `CPUQuota` across services is 380%
  on a 4-core box. Under contention, `CPUWeight` arbitrates.
- **Realtime is not free**: even SCHED_RR with a sane priority can starve the
  kernel work that audio capture depends on. We pay the engineering cost of
  benchmarking before flipping the switch.
- **Operators must use the bundled installer**: hand-editing
  `/etc/systemd/system` units bypasses the contract. `install_systemd_units.sh`
  is idempotent and ships with the repo.

## Verification

Per-tier checklist (run after `install_systemd_units.sh --install`):

- Tier 1 (audio listener)
  - `systemctl status reachy-audio-listener.service` → active, MemoryCurrent < 64M.
  - Inject 60 s synthetic dance load → `journalctl -u reachy-audio-listener.service`
    shows no `audio_underrun` warnings.
- Tier 2 (orchestrator + motion worker)
  - `systemd-cgtop reachy-runtime.slice` shows CPU% under combined 200% during
    a single dance.
  - Stop-phrase issued during dance → `bench_pi_runtime.sh --real-hardware`
    reports P95 stop latency under 500 ms (SDD-01 NFR target).
- Tier 3 (camera sampler + LLM/VLM client)
  - Under stress, camera FPS may drop but the sampler must not OOM-kill.
  - LLM client failures degrade gracefully — the orchestrator must keep
    serving stop / hush from the rule classifier alone.

All Phase 6 unit tests (`test_cpu_budget_policy.py`,
`test_interrupt_dispatch_steps.py` [renamed P7 MED-B2 from
`test_interrupt_latency_under_load.py`], `test_motion_worker_yield_contract.py`,
`test_audio_stop_priority_contract.py`, `test_systemd_units_parse.py`) must
remain green and run without root / systemd / cgroups.

## Future work

- Evaluate sched_ext / BPF schedulers on Pi 5-class hardware. Out of scope on
  Pi CM4 because the upstream kernel does not ship sched_ext by default. Per
  user constraint 2026-05-28, sched_ext / BPF schedulers are out-of-scope
  until a real CM4 benchmark proves stock systemd cannot meet SDD-01 NFR
  (P95 critical stop < 500ms).
- Re-visit realtime SCHED_FIFO once we have a 2-week field bench from a real
  hospital pilot.
- Investigate `IOWeight=` once Phase 7 introduces local model caching.
