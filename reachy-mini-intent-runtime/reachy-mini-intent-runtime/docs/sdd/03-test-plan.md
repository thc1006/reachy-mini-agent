# SDD 03: Test Plan

## Unit tests

- `test_classifier.py`
  - dance command triggers `dance`;
  - “我喜歡跳舞” is chat;
  - “停止跳舞” triggers critical `stop_dance`;
  - “噓” triggers critical quiet/stop.
- `test_scheduler.py`
  - critical stop preempts running dance;
  - background actions queue FIFO;
  - chunk boundary allows interactive command;
  - non-interruptible critical section is not force-resumed incorrectly.
- `test_resource_policy.py`
  - policy emits warning when CPU exceeds threshold;
  - critical loop heartbeat deadline miss is detected.

## Integration tests without hardware

- Run `python -m reachy_intent_runtime.demo --script demo/hospital_interrupt_scenario.json`.
- Confirm event log shows dance start, stop command, preemption, nurse response.

## Hardware tests

1. Connect Reachy Mini.
2. Start official conversation app with external profile.
3. Say “Reachy，跳 30 秒的舞”.
4. At T+5s say “停止跳舞” and/or make hush gesture if gesture path is implemented.
5. Record visible stop latency.
6. Repeat under CPU load with camera sampling enabled.

## Acceptance threshold

- No test failure in `./scripts/verify.sh`.
- Critical stop can interrupt a long dance in mock tests.
- Hardware stop-latency measurements documented in `docs/experiments/` before public demo claims.
