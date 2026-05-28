# ADR-0005: Wire intent-runtime into production robot_brain.py (phased plan)

- Status: Proposed
- Date: 2026-05-29

## Context

The 2026-05-28 ship landed the runtime as a self-contained subdir with 300
unit + integration tests, 4 ADRs (0001–0004), 4 SDDs, and 6 hardened
systemd units. Coverage is good and the public API is stable:

| Module | Public surface |
|---|---|
| `reachy_intent_runtime.classifier` | `RuleIntentClassifier` → `IntentKind.{command,chat,ambiguous}` |
| `reachy_intent_runtime.scheduler` | `ActionScheduler` (3-lane: critical / interactive / background) |
| `reachy_intent_runtime.cancellation` | `CancellationToken`, `CancellationError` |
| `reachy_intent_runtime.orchestrator_worker` | `OrchestratorWorker` (classifier → scheduler bridge) |

But the runtime is **dead code in production**. `grep -rn "intent_runtime\|reachy_intent" src/` over the parent repo returns zero matches. Every LLM tool dispatch on the live robot still goes through the legacy synchronous path:

```
src/robot_brain.py:1597  from robot_tools import get_tool_specs, parse_tool_calls, execute_tool
src/robot_brain.py:1758  calls = parse_tool_calls(msg) if LLM_TOOLS else []
src/robot_brain.py:1761  # Append assistant turn ... + tool results   ← blocking
```

`execute_tool(...)` runs each tool call synchronously on the dialog thread, with no priority lane, no preemption, and no cancellation. A long `dance` therefore blocks the chat loop end-to-end, defeating the runtime's whole reason to exist (see [ADR-0001](0001-cooperative-priority-scheduler.md) §Context).

This ADR scopes the wiring work. It is not an implementation: each phase below is its own PR with its own ADR amendment + tests. The point of recording the plan now is to keep the runtime from rotting while the next sprint is being scheduled.

## Decision

Wire the runtime in **four small phases** behind a single feature flag, `LLM_TOOLS_RUNTIME` (values: `legacy` (default) | `intent_runtime`). Each phase ships independently, preserves all 91 parent tests + 300 runtime tests, and adds its own smoke. Rollback is a single env flip.

### Phase A — Classifier as pre-filter (smallest wedge)

Wrap the existing `parse_tool_calls` result in `RuleIntentClassifier.classify()`. When the classifier returns `command` or `ambiguous`, defer to the runtime path for that turn; otherwise stay on `legacy`. New flag: `LLM_TOOLS_RUNTIME=intent_runtime` enables the wrap.

- **Touch points**: `src/robot_brain.py` around line 1758. Add `from reachy_intent_runtime.classifier import RuleIntentClassifier` (guarded import; fall back to `legacy` if the subdir isn't installed).
- **Acceptance**: with the flag off, all 91 parent tests pass unchanged. With the flag on, an existing dialog smoke (e.g. `tests/test_smoke.py`) plus one new test asserting the classifier was invoked.
- **Risk**: low. Wrap is read-only against legacy dispatch.

### Phase B — Scheduler owns long tool calls

Route every `execute_tool(...)` call whose handler is annotated as chunkable through `ActionScheduler.submit_background(...)`. Short, latency-sensitive tools (`note_head_command`, `tool_move_head`) stay on the legacy synchronous path. Long, queueable tools (`dance`, `play_emotion`) go to the background lane.

- **Touch points**: `src/robot_tools.py` (annotate handlers), `src/robot_brain.py:1758–1762` (dispatch branch), new `src/intent_runtime_adapter.py` (parent-repo glue — does NOT live inside the subdir, keeps it import-clean).
- **Acceptance**: a dance can be issued while a chat is in flight, and the chat does not block. Re-validates [ADR-0001](0001-cooperative-priority-scheduler.md) §Verification on real hardware.
- **Risk**: medium. Must preserve the existing `LLM_TOOL_MAX_ITERS=3` round-trip semantics for chat-driven tool chains.

### Phase C — Cancellation via audio stop word

Wire the existing wake/STT pipeline (`src/robot_brain.py` audio listener, currently routes to whisper-server) to emit a `CancellationToken.cancel()` against the running scheduler lane when an audio match for the stop phrase is detected. Per [user memory](../../../memory/wake_word_design_choices_2026_05_13.md): re-use the simple `re.search(r'嘿瑞奇|hey reachy|停|stop|安靜|hush')` post-Whisper rather than re-introducing an ML wake-word model.

- **Touch points**: audio listener (TBD line range), one new test `tests/test_audio_stop_word_cancels.py` asserting end-to-end: STT chunk with stop word → cancel token fired → `motion_worker` exits within the contract latency from [ADR-0001](0001-cooperative-priority-scheduler.md).
- **Acceptance**: hardware smoke — start a dance, say "嘿瑞奇 停下來", confirm motion halts within the latency budget.
- **Risk**: medium. Latency budget interacts with STT VAD timing — re-bench against `project_synth_e2e_bench_2026_05_13` numbers (avg STT 239 ms).

### Phase D — Full replacement

Remove the `LLM_TOOLS_RUNTIME=legacy` branch and drop the direct `execute_tool` import path. All tool dispatch goes through `OrchestratorWorker`. Delete `src/robot_tools.py` dispatcher (the tool handlers themselves stay as adapters).

- **Touch points**: `src/robot_brain.py:1597, 1758–1762`; `src/robot_tools.py` (shrink to handler module).
- **Acceptance**: feature flag removed; all combined tests pass; production smoke equivalent to the v5.0 audit table (see `memory/project_v5_publication_2026_05_17.md`).
- **Risk**: high if shipped before Phases A–C have soaked in production for ≥ 1 week each.

## Consequences

Positive:

- Recovers the value of today's 300-test runtime — it stops being a museum piece.
- Each phase has a single rollback (env flip), so the cutover risk is bounded.
- The chat loop stops blocking on long motions — meets the original hospital-interrupt acceptance from [SDD-01](../sdd/01-requirements.md).

Negative:

- Four PRs spanning the 2700-line `robot_brain.py`. Every PR is a merge-conflict risk against ongoing work.
- The runtime's adapter layer (Phase B `intent_runtime_adapter.py`) is new surface that has to be reviewed for thread-safety against the existing `_motion_lock`.
- Phase C lifts the audio listener out of its current single-purpose role; if that listener is also doing wake-word duty, we now have two consumers and need to be explicit about ordering.

## Out of scope today

- Any code change to `src/robot_brain.py` or to handlers under `src/robot_tools.py`.
- A real benchmark of cancellation latency on the Pi 4 / CM4 hardware budget from [ADR-0004](0004-pi-cpu-qos-and-runtime-scheduling.md).
- The s1 → vllm0528 backend migration — a separate, in-flight track (see `memory/project_s1_brick_2026_05_28.md`).

## Verification

When each phase ships, this ADR is amended with the PR + commit SHA + smoke result, not deleted. The runtime's existing 300 tests must stay GREEN at every phase; new tests live alongside the touched code, not in the runtime subdir.

The "done" condition for this ADR is Phase D merged + a clean grep:

```
grep -rn "robot_tools.execute_tool\|legacy.*tool" src/ → 0 matches
```

Until then this ADR stays `Proposed` and the runtime stays a tracked subdir with no production dependents.
