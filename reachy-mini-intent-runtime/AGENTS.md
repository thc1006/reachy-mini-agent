# AGENTS.md

This file is the repository-level operating guide for AI coding agents.

## Mission

Build a robust Reachy Mini intent runtime that lets an LLM/VLM conversation app trigger official Reachy Mini SDK dances and emotions through natural language while preserving interruptibility, responsiveness, CPU budget, and testability.

## Setup commands

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'
./scripts/verify.sh
```

## Run commands

```bash
python -m reachy_intent_runtime.demo --script demo/hospital_interrupt_scenario.json
python -m pytest -q
ruff check .
ruff format .
```

## Required workflow

1. Read `docs/sdd/01-requirements.md` and the relevant ADR before changing behavior.
2. Write or update tests first for every behavior change.
3. Confirm the new or changed test fails for the intended reason.
4. Implement the smallest change that passes tests.
5. Run `./scripts/verify.sh` before declaring completion.
6. Do not hide failing tests. If hardware is unavailable, use `MockMotionAdapter` and document the unverified hardware assumption.

## Sub-agent guide

The following Claude Code sub-agents live under `.claude/agents/` and are
loaded automatically by Claude Code. Use the right one for the right task:

| Sub-agent | When to use |
|---|---|
| `architecture-reviewer` | Reviewing ADR/SDD changes, decoupling decisions, runtime contracts |
| `reachy-sdk-researcher` | Investigating official Pollen Robotics SDK / conversation app behavior |
| `runtime-scheduler-reviewer` | Reviewing scheduler.py / cancellation.py / motion_worker.py changes -- concurrency, preempt, yield, cancellation token contract |
| `tdd-reviewer` | Reviewing TDD discipline (red-green-refactor); ensuring tests fail before implementation |
| `test-engineer` | Converting requirements / ADR / SDD into pytest test plans (test-first specification) |

## Architecture constraints

- The robot must never become deaf/blind merely because a dance/emotion is running.
- `stop_dance`, `stop_emotion`, emergency stop, and user discomfort signals are high-priority actions.
- Long motions must be represented as interruptible chunks unless the official SDK proves native pause/resume support.
- LLM/VLM calls must not block the motion scheduler event loop.
- Vision sampling, speech/VAD, intent routing, and motion execution must be separable workers or processes.
- Keep the Pi-side runtime light. Heavy VLM/LLM should run off-board unless benchmark data proves it is safe.
- Intent routing is a 3-lane contract: `command` -> scheduler; `chat` -> chat handler; `ambiguous` -> ambiguous handler (LLM-bound). The `OrchestratorWorker` (src/reachy_intent_runtime/orchestrator_worker.py) is the single bridge. **Ambiguous results must never be silently dropped.**
- The CM4 runtime is governed by ADR-0004 (Pi/CM4 CPU QoS) and SDD-04. Use systemd/cgroups (Tier 2) before considering realtime scheduling (Tier 3 -- opt-in only).
- `reachy_intent_runtime.llm_vlm_client` is a placeholder daemon. Wire a real off-board client by editing `deploy/systemd/reachy-llm-vlm-client.service` `ExecStart=` before production.
- Use `data/reachy_official_actions.yaml` (catalog) and `data/command_router_examples.zh-en.yaml` (router dataset) as the **source of truth** for action metadata and utterance->intent mapping. Edit the YAML before changing classifier or scheduler logic.
- Motion workers must honor `CancellationToken` from `src/reachy_intent_runtime/cancellation.py`. Calling `token.check()` at every yield boundary is the cooperative-cancellation contract per ADR-0001 (2026-05-28 update).

## Coding style

- Python 3.12+.
- Prefer dataclasses and typed protocols for hardware abstractions.
- Keep hardware adapters thin; keep logic in pure, testable modules.
- No direct network downloads in tests.
- No credentials in repository files.
- New YAML datasets (`data/*.yaml`) must use 2-space indent, comments explaining provenance, and a header noting verification status.
- New sub-agent .md files must have YAML frontmatter with `name` + `description` minimum; optionally `tools`, `model`, `effort`.

## Definition of done

A task is done only when:

- The behavior is specified in docs or tests.
- Unit tests pass.
- The design impact is reflected in ADR/SDD if architectural.
- `./scripts/verify.sh` passes locally.
- Any hardware-only behavior is marked with a clear verification checklist.
- Any change touching utterance classification must update `data/command_router_examples.zh-en.yaml` AND ensure `tests/test_command_vs_chat_router.py` passes.
- Any change touching action metadata must update `data/reachy_official_actions.yaml` AND ensure `tests/test_action_catalog.py` passes.
