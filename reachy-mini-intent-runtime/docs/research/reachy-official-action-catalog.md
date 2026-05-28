# Reachy Mini Official Action Catalog — Research Note

- Phase: 10
- Track: A (foundational data layer)
- Status: **DRAFT** — not hardware-verified.
- Last updated: 2026-05-28
- Owner: intent-runtime team
- Code: `data/reachy_official_actions.yaml`, `src/reachy_intent_runtime/action_catalog.py`
- Tests: `tests/test_action_catalog.py`

## Why this exists

The Reachy Mini conversation app exposes a small set of LLM tools (`move_head`,
`camera`, `head_tracking`, `dance`, `stop_dance`, `play_emotion`,
`stop_emotion`, `idle_do_nothing`) but does not publish a machine-readable
inventory of the *named actions* those tools can invoke (e.g. which dances live
inside `pollen-robotics/reachy-mini-dances-library`, expected durations,
interruption behavior). This is the foundational P0 gap called out by the user
in the Phase 10 plan.

Without a single source of truth the orchestrator is forced to inline magic
constants — see the original `_ACTION_TO_COMMAND` dict in
`src/reachy_intent_runtime/orchestrator_worker.py`. That works for four
hard-coded entries but fails the moment we want to add dance/emotion variants,
encode CPU risk, or drive segmentation decisions from data.

The catalog moves that knowledge out of code and into a versioned, validated
YAML file that the orchestrator (Track B) will eventually consume.

## Provenance

Inputs that fed this draft:

1. **Reachy Mini conversation app README** — canonical list of LLM tools and
   their intended semantics. Source URL captured in
   `docs/research/2026-05-28-official-stack.md`.
2. **`pollen-robotics/reachy-mini-dances-library`** Hugging Face dataset —
   canonical naming convention for dances (`happy`, `sad`, `energetic`,
   `slow`, `bow`, `wave` are typical entries; verify upstream before shipping
   to production).
3. **`pollen-robotics/reachy-mini-emotions-library`** Hugging Face dataset —
   per-emotion JSON trajectory + paired WAV; this is why we mark every emotion
   `requires_segmentation: true` (audio + trajectory both need clean stop
   semantics).
4. **ADR-0001 (cooperative priority scheduler)** — defines the segmentation
   rule we encode here: any non-interruptible long action must be chunked.

## Methodology

For each entry we recorded:

- `name`: a stable identifier the orchestrator will key off of. We intentionally
  use a `<type>_<flavour>` convention (e.g. `dance_happy`) instead of bare
  upstream IDs so we can re-target without touching call sites.
- `official_tool`: the upstream LLM tool name (`dance`, `play_emotion`,
  `move_head`, …). Multiple catalog entries can map to the same tool — that is
  expected for dances/emotions that share a tool but differ by `dance_name`
  parameter.
- `estimated_duration_ms`: **upper-bound estimate** derived from public tool
  descriptions and typical library sample lengths. These numbers are
  intentionally pessimistic so the scheduler over-budgets rather than
  under-budgets.
- `interruptible`: tri-state `true | false | unknown`. We use `unknown`
  liberally because the official SDK does not document per-tool preemption
  semantics. Per CLAUDE.md "never claim hardware validation unless the real
  Reachy Mini command was run", we refuse to guess.
- `has_stop_tool` / `stop_tool`: encodes which actions have a discrete stop
  primitive (`stop_dance`, `stop_emotion`). The orchestrator's CRITICAL lane
  uses this to know what to dispatch for an emergency stop.
- `requires_segmentation`: derived from ADR-0001. Any action whose
  `interruptible` is not provably `true` AND whose duration is long must be
  chunked so the scheduler can preempt at boundary points.
- `cpu_risk`: `low | medium | high`. This feeds Tier-3 throttling decisions on
  the CM4 (see CLAUDE.md "CM4 runtime constraints").
- `source`: short provenance string for audit.
- `notes`: free text — keep short, link out for detail.

## How to update

1. Add or edit entries in `data/reachy_official_actions.yaml`. Mind the
   schema — `tests/test_action_catalog.py` will flag invalid `type`,
   `cpu_risk`, missing fields, or inconsistent `stop_dance` / `stop_emotion`
   mappings.
2. If you need a new top-level field, extend `ActionEntry` and
   `_build_entry` together, and add a regression test. Keep the dataclass
   `frozen=True` for safe worker-pool sharing.
3. When an entry is verified on real hardware, add a `verified_at_<YYYY-MM-DD>`
   field (the loader will tolerate it as long as we add it to `_build_entry`).
   Until then leave `interruptible: unknown` for anything we have not measured.
4. After editing, run `./scripts/verify.sh`.

## Open questions for hardware verification

The following can only be answered by running the actual Reachy Mini SDK
against real hardware. They are intentionally captured here so the next
hardware session has a checklist:

1. **Actual dance durations.** Our 4 s – 15 s numbers are public-doc estimates;
   measure each dance in `pollen-robotics/reachy-mini-dances-library` end-to-end.
2. **Actual emotion durations.** Emotions library entries pair trajectory +
   WAV; the audible portion may extend past the visible motion.
3. **Hard-stop latency.** How long between `stop_dance` being issued and the
   antenna/head actually freezing? ADR-0001 targets P95 < 500 ms for Pi-local
   critical events.
4. **Queue vs overlap semantics.** When two `play_emotion` calls arrive
   back-to-back, does the SDK queue, blend, or overwrite? This determines
   whether our scheduler must enforce ordering or can rely on the SDK.
5. **Native pause/resume.** Is there any action that the SDK can pause and
   resume cleanly? If yes, we can drop `requires_segmentation: true` for it
   and avoid chunking overhead.
6. **`head_tracking` lifecycle.** Confirm it is a toggleable background loop
   (our assumption) vs a fire-and-forget; this affects the
   `head_tracking_on` / `head_tracking_off` entries.
7. **`camera` capture cost on CM4.** Measure CPU and wall-time of one
   `camera` call under Tier-2 contention. If high we may demote to Tier-3.
8. **Stop-while-stop edge case.** What happens if `stop_dance` arrives while
   nothing is dancing? Catalog assumes no-op; verify on hardware.

## Non-goals

- The catalog does NOT decide priority — that lives in
  `OrchestratorWorker._ACTION_TO_COMMAND` (today) or its data-driven
  replacement (Track B). The catalog only describes the action, not its
  scheduling weight.
- The catalog does NOT contain Chinese/English utterance examples — those
  live in `data/command_router_examples.zh-en.yaml` (Track B).
- The catalog does NOT speak to LLM tool-call formatting — that is owned by
  the conversation app side.
