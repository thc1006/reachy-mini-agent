# CLAUDE.md

## Project purpose

This repo implements an interruptible intent/action runtime for Reachy Mini. The target behavior is: users can naturally ask Reachy Mini to dance or express emotions, but Reachy must still hear/see high-priority commands such as stop, hush, pause, call nurse, or emergency stop while a motion is running.

## Required commands

- Install: `pip install -e '.[dev]'`
- Verify all: `./scripts/verify.sh`
- Unit tests: `pytest -q`
- Lint: `ruff check .`
- Format: `ruff format .`
- Demo: `python -m reachy_intent_runtime.demo --script demo/hospital_interrupt_scenario.json`

## Workflow rules

- Use SDD + TDD: update specification/tests before implementation.
- Keep changes small and reviewable.
- Prefer mockable adapters over direct SDK calls in core logic.
- Never claim hardware validation unless the real Reachy Mini command was run.
- When changing scheduling semantics, update `docs/adr/0001-cooperative-priority-scheduler.md`.
- When changing command-vs-chat behavior, update `docs/adr/0002-command-vs-chat-router.md`.

## Intent routing architecture (post-P8)

- `RuleIntentClassifier` returns three `IntentKind` values: `command`, `chat`, `ambiguous`.
- `OrchestratorWorker` (`src/reachy_intent_runtime/orchestrator_worker.py`) is the
  single bridge: it calls the classifier, dispatches `command` to the scheduler,
  forwards `chat` to a `chat_handler` callback (or logs `chat_skipped`), and
  routes `ambiguous` to an `ambiguous_handler` callback (or logs
  `ambiguous_pending_llm_handoff`). **Ambiguous results must never be silently
  dropped** — see ADR-0002 §HIGH-A2.
- New utterance categories should be added to the classifier with a failing test
  first, then mapped through `_ACTION_TO_COMMAND` if they need a `MotionCommand`.
- `reachy_intent_runtime.llm_vlm_client` is a heartbeat-only placeholder daemon.
  Wire a real off-board LLM/VLM client before production deployment by editing
  `deploy/systemd/reachy-llm-vlm-client.service` `ExecStart=`.

## Safety and privacy rules

- Do not read or print `.env`, `.env.*`, or `secrets/**`.
- Do not add API keys, tokens, or patient data to fixtures.
- For hospital demos, use synthetic patient scenarios only.

## CM4 runtime constraints

These rules cover Reachy Mini Wireless (Raspberry Pi CM4, 4 cores, 4 GB RAM)
and are enforced by ADR-0004 / SDD-04. Violations require an ADR amendment.

- Do **not** run heavy local VLM / LLM on the CM4 by default. The
  `reachy-llm-vlm-client.service` is a thin off-board API client. Any local
  inference proposal needs a benchmark and an ADR amendment.
- **Protect audio, stop, and orchestrator first.** Tier 1 (audio listener) and
  Tier 2 (orchestrator + motion worker) keep their CPU budget under contention;
  Tier 3 (camera, LLM/VLM client) is allowed to degrade.
- **Use systemd + cgroups before realtime.** Tune `CPUWeight=`, `CPUQuota=`,
  `MemoryMax=`, `Nice=` first. Do not edit `/etc/systemd/system` units by hand;
  use `scripts/install_systemd_units.sh` so the contract stays auditable.
- **Realtime is opt-in.** `CPUSchedulingPolicy=fifo`/`rr` is commented out in
  every unit. Turn it on only after `scripts/bench_pi_runtime.sh --real-hardware`
  shows the regression and an ADR amendment records the policy + priority.

## Quality gate

Before final response or commit, run `./scripts/verify.sh`. If it fails, report the failing command and fix it before continuing.
