# ADR-0003: Raspberry Pi Resource Budget and Worker Split

- Status: Accepted
- Date: 2026-05-28

## Context

Reachy Mini Wireless runs on Raspberry Pi 4. The meeting requirement is that motion execution must not starve audio, vision sampling, LLM/VLM routing, or stop handling. Official conversation app documentation warns that local vision is not supported directly on Reachy Mini Wireless / Raspberry Pi; keep local vision on laptop/workstation.

## Decision

Use a split runtime:

- Pi-side: motion scheduler, SDK adapter, microphone/VAD heartbeat, low-cost interrupt detector, lightweight telemetry.
- Off-board: LLM, VLM, optional local vision model, long context, dashboard.

Use OS-level resource controls where possible:

- run heavy off-board services away from Pi;
- run Pi critical interrupt loop with higher priority;
- avoid CPU-bound work in the scheduler event loop;
- optionally use `nice`, `taskset`, or systemd slices after measuring Pi behavior.

## Consequences

- The robot remains responsive even during expressive motions.
- The system can degrade gracefully when network LLM/VLM is slow.
- Integration requires a clear JSON contract between off-board agent and Pi scheduler.

## Verification

- Stress test with continuous dance + camera sampling + repeated stop commands.
- Record CPU utilization and stop latency.
- Fail the run if critical loop heartbeat misses two consecutive deadlines.
