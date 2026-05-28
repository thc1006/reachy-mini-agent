# Command-vs-Chat Router — Acceptance Criteria

## Purpose

This document defines the **canonical set of utterance → intent mappings**
that the local rule-based guardrail (`RuleIntentClassifier`) must satisfy.
The rules live in code (`src/reachy_intent_runtime/classifier.py`), but the
**behavioural contract** lives in a YAML dataset that anyone — including
non-engineers — can read, edit, and extend.

This file is the human-readable companion to:

- `data/command_router_examples.zh-en.yaml` — the dataset (source of truth)
- `tests/test_command_vs_chat_router.py` — the gate that runs every example
  on every commit
- `docs/adr/0002-command-vs-chat-router.md` — the ADR explaining *why* the
  classifier is structured the way it is

## Source of truth

```
data/command_router_examples.zh-en.yaml
```

Every example in this file is loaded by
`tests/test_command_vs_chat_router.py` and asserted against the live
classifier. If the YAML and the classifier disagree, **the YAML wins**: the
classifier patterns must be strengthened until they match the YAML again
(see *Maintenance* below).

## How to add a new utterance (for non-engineers)

1. Open `data/command_router_examples.zh-en.yaml`.
2. Find the group whose `expect:` block matches what the robot should do,
   then add your line to its `examples:` list. (If no group fits, copy an
   existing group as a template and rename it.)
3. From the repo root, run:
   ```
   pytest -q tests/test_command_vs_chat_router.py
   ```
   If every example passes, you are done — commit the YAML. If your new
   example fails, the classifier does not yet handle that wording; file a
   ticket so an engineer can extend the patterns (do **not** delete your
   example — failure means the dataset is teaching us something).

## How to interpret each `kind`

| `kind`      | What the runtime does                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `command`   | Classifier emits a `MotionCommand` and the scheduler dispatches it. Required: `action` and `priority` (`critical` / `background`).     |
| `chat`      | Forwarded to the chat handler (LLM, log, or no-op). No motion is started.                                                              |
| `ambiguous` | Forwarded to the LLM for context-aware disambiguation. No motion is started until the LLM responds. **Must never be silently dropped** (see ADR-0002 §HIGH-A2). |

## Acceptance gate

- `test_router_example_classifies_as_expected`: **100 %** of YAML examples
  must classify with the declared `kind` (and `action` / `priority` when
  declared). Any failure blocks the build.
- `test_dataset_has_minimum_coverage`: dataset must contain **≥ 50**
  utterances.
- `test_dataset_covers_all_intent_kinds`: dataset must cover all three
  `IntentKind` values (`command`, `chat`, `ambiguous`).

All three gates run under the standard `./scripts/verify.sh` quality gate.

## Out of scope

The following kinds of utterance are **not** the rule-based guardrail's
responsibility and therefore do **not** belong in this dataset. They are
the LLM stage's job:

- **Memory-bound references** — e.g. *"do that again"*, *"再來一次"*. These
  require dialogue history the local classifier does not see.
- **Persona / preference negotiation** — e.g. *"從現在開始你要更活潑一點"*.
- **Multi-turn corrections** — e.g. *"不對啦，我說的是另一支舞"*.
- **Open-ended chit-chat with no robot-action intent** — handled by the
  chat handler downstream; the classifier just routes it as `kind: chat`.

## Maintenance

The dataset is the contract. If a real-world utterance is mis-routed in
production:

1. **Add the utterance to the YAML first** under the group that reflects
   the *correct* expected behaviour. The test will fail — that failure is
   the bug ticket.
2. **Then** strengthen the patterns in
   `src/reachy_intent_runtime/classifier.py` until the test goes green
   again, taking care not to regress any other group.
3. If the fix changes ordering or semantics in `classify()`, update
   `docs/adr/0002-command-vs-chat-router.md` (per CLAUDE.md workflow
   rules).

Never delete a failing example to "make the test pass" — the dataset's
value is precisely that it records what the runtime *should* do, including
cases the classifier does not yet handle correctly.
