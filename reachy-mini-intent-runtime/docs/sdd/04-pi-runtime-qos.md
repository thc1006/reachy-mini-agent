# SDD-04: Pi / CM4 Runtime QoS

Counterpart to ADR-0004. Defines the testable contract for the 3-tier QoS
strategy that protects audio / stop / orchestrator on the Reachy Mini Wireless
CM4.

## Scope

In scope:

- per-service CPU weight + quota + memory cap + nice level
- systemd slice that aggregates the runtime processes
- worker stubs (audio, motion, camera, orchestrator) callable as `python -m`
- mock-mode bench / stress scripts

Out of scope:

- custom Linux schedulers, sched_ext / BPF
- realtime SCHED_FIFO / SCHED_RR (opt-in only, gated by benchmark + ADR amend)
- local VLM / LLM inference on CM4

## Acceptance criteria

The phase is **done** when all six criteria below hold:

1. `deploy/systemd/reachy-runtime.slice` exists with `CPUAccounting=yes`,
   `MemoryAccounting=yes`, `IOAccounting=yes`.
2. Each of the five `reachy-*.service` units references
   `Slice=reachy-runtime.slice`, declares `CPUWeight=`, `CPUQuota=`,
   `MemoryMax=`, `Nice=`, and uses `/opt/reachy-mini-intent-runtime` as
   `WorkingDirectory=`.
3. The audio listener has the highest `CPUWeight` and the LLM / VLM client has
   the lowest, per ADR-0004 Tier 1 / Tier 3.
4. Critical-stop dispatch from `audio_listener.emit_stop()` to
   `adapter.stop_current()` completes in <=1 scheduler step in the synchronous
   mock pipeline (step count, not wall-clock latency — P7 reword for honesty;
   real-hardware wall-clock P95 < 500ms remains the SDD-01 NFR target, gated by
   `scripts/bench_pi_runtime.sh --real-hardware`).
5. The motion worker simulator yields cooperative control at least every
   `YIELD_BUDGET_MS` (500 ms) over a 30 s synthetic action.
6. All Phase 6 unit tests pass without root, systemd, cgroup access, or any
   real hardware.

## Component diagram

```
+--------------------------+
|  audio_listener (Tier 1) |  CPUWeight=900 CPUQuota=80% Nice=-5  64M
+-----------+--------------+
            |  emit_stop()
            v
+--------------------------+
|  orchestrator   (Tier 2) |  CPUWeight=600 CPUQuota=80% Nice=-3 128M
|  - classifier            |
|  - ActionScheduler       |
+-----------+--------------+
            |  submit(MotionCommand)
            v
+--------------------------+
|  motion_worker  (Tier 2) |  CPUWeight=300 CPUQuota=120% Nice=0 384M
|  - chunked exec          |
+--------------------------+

+--------------------------+        +-----------------------------+
| camera_sampler  (Tier 3) |        |  llm_vlm_client    (Tier 3) |
| CPUWeight=150 60% Nice=5 |        |  CPUWeight=100 40% Nice=10  |
| 128M                     |        |  64M (thin off-board client)|
+--------------------------+        +-----------------------------+

All five services run inside reachy-runtime.slice
  CPUAccounting=yes  MemoryAccounting=yes  IOAccounting=yes
  CPUQuota=380%       MemoryMax=2G
```

## `CpuBudgetConfig` dataclass shape

Implemented in `src/reachy_intent_runtime/cpu_qos.py`:

```python
@dataclass(frozen=True)
class CpuBudgetConfig:
    cpu_weight: int             # systemd CPUWeight, 1..10000 (validated)
    cpu_quota: str              # systemd CPUQuota, e.g. "80%" (validated)
    memory_max: str             # systemd MemoryMax, e.g. "384M" or "1G"
    slice_name: str = "reachy-runtime.slice"
    nice: int = 0               # POSIX nice, -20..19 (validated)
```

Constructor raises `ValueError` on out-of-range `cpu_weight` / `nice` or
malformed `cpu_quota` / `memory_max`.

Helpers:

- `parse_cpu_quota(raw: str) -> float` — `"80%" -> 0.80`, raises on bad input.
- `parse_memory_max(raw: str) -> int` — `"384M" -> 402653184`, accepts `M` / `G`.
- `validate_budget_total(configs, cpu_cores=4) -> list[str]` — warns if total
  `CPUQuota` exceeds `cpu_cores * 1.5`.
- `validate_against_slice_cap(configs, slice_cap_pct=380.0) -> list[str]` —
  warns if total per-service `CPUQuota` exceeds the slice's own `CPUQuota`
  cap (default 380% matches the shipped `reachy-runtime.slice`). Added P7
  MED-B3 so operators get a heads-up before cgroup v2 silently throttles.

## Test plan

- `tests/test_cpu_budget_policy.py` — config / parser / validator contract.
- `tests/test_interrupt_dispatch_steps.py` — stop dispatch step-count contract
  under simulated scheduler pressure (renamed P7 MED-B2 from
  `test_interrupt_latency_under_load.py` — the old name implied wall-clock
  latency which the tests do not measure).
- `tests/test_motion_worker_yield_contract.py` — cooperative yield invariant.
- `tests/test_audio_stop_priority_contract.py` — sensor → orchestrator → scheduler
  hop budget.
- `tests/test_systemd_units_parse.py` — each unit file parses, declares the
  required keys, has no realtime fields active, has no typos.

## Bench / stress scripts

- `scripts/bench_pi_runtime.sh` — defaults to `--mock`. In-process simulation
  measures stop-dispatch latency under increasing synthetic load. The
  `--real-hardware` mode is documented but intentionally not implemented in
  this phase (Phase 7 will wire it to systemd journal scraping).
- `scripts/stress_cpu_and_test_stop.sh` — defaults to `--mock`. Spawns N
  CPU-bound `multiprocessing` workers, then measures the time between
  `scheduler.submit(stop)` and `adapter.stop_current` appearing in the log.

Both scripts exit 0 in mock mode and exit 2 if invoked with
`--real-hardware`, to make the gating explicit.
