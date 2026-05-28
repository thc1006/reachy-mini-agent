# ADR-0001: Cooperative Priority Scheduler for Reachy Mini Motions

- Status: Accepted
- Date: 2026-05-28

## Context

The meeting requirement is that Reachy Mini may execute official SDK dances/emotions from natural language, but must not become unable to hear, see, or respond while a long motion is running. The official conversation app exposes queued motion tools such as `dance`, `stop_dance`, `play_emotion`, and `stop_emotion`, but the implementation still needs application-level policy for prioritization, interruption, and smoothness.

## Decision

Use a deterministic cooperative priority scheduler in front of SDK motion calls.

The scheduler has three lanes:

1. `critical`: emergency stop, stop dance, stop emotion, user distress, explicit “stop”.
2. `interactive`: call nurse, answer patient, look at user, acknowledge request.
3. `background`: dance, idle emotion, breathing, playful gestures.

Long background motions must be chunked into short segments unless the SDK provides verified native pause/resume. At every chunk boundary, the scheduler checks high-priority queues. If a critical event is pending, the current motion is stopped or not resumed.

## Consequences

Positive:

- Users can interrupt long dances.
- Motion smoothness is tunable through chunk size.
- Tests can simulate behavior without hardware.
- LLM/VLM latency no longer directly controls safety-critical interruption.

Negative:

- Chunking may create visible gaps if the check loop is too slow.
- Official motion macros may need preprocessing into smaller segments.
- True pause/resume may be impossible without SDK support.

## Verification

- Unit test: high-priority stop preempts running dance.
- Unit test: regular background actions queue behind current action.
- Hardware test: issue 30-second dance, then say/gesture stop at T+5s; measure time to visible stop.
- Hardware metric: target P95 stop latency < 500 ms for Pi-local critical events, < 2 s for LLM/VLM-mediated events.

## 2026-05-28 update — audit events

Two new informational `SchedulerEvent` types were added to the public `scheduler.events` list for observability and audit:

- `preempt_blocked`: emitted inside `submit()` when a new CRITICAL command arrives but the running command has `interruptible=False`. Fields: `command` = running command name, `details.reason = "non_interruptible"`, `details.requested` = incoming command name. No stop is issued; the new command is queued normally.

- `boundary_preempt_blocked`: emitted inside `tick()` at a chunk boundary when a higher-priority queued command cannot preempt the running command because `running.command.interruptible == False`. Fields: same shape as `preempt_blocked`. The candidate command is re-pushed to the queue.

Additionally, the existing `preempt` event's `details` dict now includes `from_priority` and `to_priority` (lower-case `ActionPriority.name` strings, e.g. `"background"` and `"critical"`) for richer audit trails.

## 2026-05-28 update — CancellationToken contract (Phase 10 Track C)

A new `CancellationToken` module (`reachy_intent_runtime.cancellation`) and explicit token wiring were added to the scheduler and motion worker simulator.

### New scheduler API

```python
scheduler.token_for(command: MotionCommand) -> CancellationToken | None
```

- Returns the `CancellationToken` created when `command` began running, or `None` if the command was never submitted or never started running.
- The token is stored in `scheduler._tokens` keyed by `id(command)` and persists until the scheduler is discarded.

### Cancellation reason format

When the scheduler cancels a running command's token the reason string is:

```
preempt_by_<command_name>_<priority>
```

Examples: `"preempt_by_stop_dance_critical"`, `"preempt_by_call_nurse_interactive"`.

### Motion worker cooperative contract

Production motion workers MUST check their `CancellationToken` at every cooperative yield boundary. The `MotionWorkerSimulator.run_iter()` generator demonstrates the pattern:

1. Check `token.is_cancelled` at the top of each step before doing work.
2. Yield the current elapsed time at each budget boundary.
3. Check `token.is_cancelled` again immediately after the yield point.
4. Set `self.cancelled = True` and `return` (stop iteration) when cancelled.

### Token lifecycle rules

- One `CancellationToken` is created per command start (never reused across submissions).
- Natural completion leaves the token uncancelled; only a preempt path calls `.cancel()`.
- `MockMotionAdapter` does not inspect tokens; correctness is asserted via `scheduler.token_for(cmd).is_cancelled` directly in tests.
- Real motion workers MUST honour the token to satisfy the SDD-04 ≤500 ms stop-latency contract.

## Token lifecycle invariants (Phase 12 clarification)

1. Tokens exist only for currently-running commands. Queued commands have no token until they start executing. Callers must therefore expect `scheduler.token_for(cmd)` to return `None` for commands in `_queue` or `_priority_queue`.
2. Tokens are removed from `_tokens` immediately on natural completion OR on preempt completion. Long-running orchestrators may rely on bounded memory.
3. `preempt_blocked` and `boundary_preempt_blocked` events do NOT signal any token — they describe situations where preempt was REQUESTED but the running command was non-interruptible, so the running token is NOT cancelled.
4. `CancellationToken` is NOT thread-safe. The ActionScheduler is the sole writer; if a future change introduces a multi-threaded model, add a `threading.Lock` to the token's internal state.
