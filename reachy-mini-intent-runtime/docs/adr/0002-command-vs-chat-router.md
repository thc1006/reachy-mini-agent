# ADR-0002: Separate Command Intent from General Conversation

- Status: Accepted
- Date: 2026-05-28

## Context

The robot must decide whether a natural-language utterance is an executable instruction or ordinary chat. False positives are dangerous: “我喜歡跳舞” should not trigger dance. False negatives are also bad: “Reachy，先不要跳了” must stop a running dance.

## Decision

Use a two-stage router:

1. Deterministic local guardrail for high-priority commands and obvious non-commands.
2. LLM decision prompt only for ambiguous cases, returning strict JSON.

The local guardrail must handle stop/pause/hush/emergency patterns without waiting for cloud LLM. The LLM prompt must explicitly classify into `chat`, `command`, or `ambiguous`, include confidence, target action, priority, and reason.

## Consequences

- Critical stops remain fast.
- LLM receives fewer trivial routing decisions.
- Prompt examples can be concise and canonical.
- The routing surface can be tested with fixture utterances.

## Verification

- Unit tests for Chinese and English command/non-command pairs.
- Regression tests for meeting examples: “跳支舞”, “噓”, “停止跳舞”, “我喜歡跳舞”.
- Real microphone test: interrupt while audio and motion are both active.

## 2026-05-28 fix appendix (Phase 4 adversarial review — HIGH items)

Classifier hardened against negation false-positives and descriptive over-fire.
OrchestratorWorker now wires RuleIntentClassifier and handles all three intent
lanes explicitly. Test count: 62 classifier + 15 orchestrator = 77 total in
those two files (full suite: 155 tests across all modules).

### New `ambiguous` lane

`IntentKind="ambiguous"` (defined in `models.py`) is now returned by the classifier
for utterances that the rule engine cannot safely resolve without LLM context:
- Negated-stop utterances (e.g. "不要停止跳舞", "別停", "don't stop dancing")
- Capability-question plus dance-imperative combos (e.g. "你會跳舞嗎，跳給我看")
- Semantic emotion modifiers without a named emotion (e.g. "難過一點", "笑一下", "做個鬼臉")

### Negation guard rule

Any utterance matching `_negation_stop_patterns` is intercepted BEFORE
`_stop_patterns` fires and routed to `ambiguous` (confidence 0.72).
This is safety-critical: "不要停止跳舞" ("don't stop dancing") must never produce
`stop_dance CRITICAL`.  Patterns caught:
- `不要.*停` / `別.*停` / `請繼續` / `繼續(跳|動)`
- `don'?t\s+stop` / `do\s+not\s+stop` / `please\s+continue`

### Imperative anchoring for hush descriptors

`安靜` pattern changed from a bare substring match to an anchored form:
`(^|[，。！？\s！])安靜([！。\s]|$)`.  This prevents descriptive uses
("圖書館很安靜", "我安靜地坐著", "他很安靜地離開") from firing `stop_emotion CRITICAL`.
Bare `噓` and `小聲` are unchanged (only used imperatively).

### Dance refusal guard

`_chat_not_command_patterns` extended with `我.*(不|才不).*(想|要|喜歡).*跳舞`
so refusals ("我不想跳舞", "我才不要跳舞") route to `chat`, not `dance BACKGROUND`.

### New classify() check ordering (11 steps)

1. empty / whitespace → chat
2. capability_plus_imperative → ambiguous  (before chat_not_command)
3. chat_not_command → chat
4. negation_stop_patterns → ambiguous  (before stop_patterns)
5. stop_patterns → critical command
6. hush_patterns → critical command  (imperative-anchored)
7. dance_command_patterns → background command
8. emotion_command_patterns → background command
9. ambiguous_emotion_patterns → ambiguous
10. question markers → chat
11. fallback → chat

---

## 2026-05-28 update

Classifier strengthened per user specification (34 tests, up from 4):

Pattern changes:
- `_dance_command_patterns`: alternation extended from `(一|支|個|段)?` to
  `(一|支|個|段|一段|一支|一個)?` so two-character measure phrases such as
  “跳一段舞” match correctly. The old alternation only consumed one character.
- `_dance_command_patterns`: `幫我` prefix made optional via `(幫我)?` so
  “幫我跳個舞” is classified as a dance command.
- `_chat_not_command_patterns`: added `你.*跳舞.*[嗎？?]` and
  `會.*跳舞.*[嗎？?]` so capability questions (“你會跳舞嗎？”) resolve to
  chat rather than firing a dance command.  This is required because
  `_chat_not_command_patterns` is checked before `_dance_command_patterns`.

New utterance pairs validated:
- “不要跳了” → stop_dance (CRITICAL)
- “停下” → stop_dance (CRITICAL)
- “stop dancing” / “Stop Dancing” / “STOP” → stop_dance (CRITICAL)
- “噓” → stop_emotion (CRITICAL)
- “安靜” → stop_emotion (CRITICAL)
- “be quiet please” → stop_emotion (CRITICAL)
- “跳支舞” (bare, no Reachy prefix) → dance (BACKGROUND)
- “跳一段舞” → dance (BACKGROUND)
- “幫我跳個舞” → dance (BACKGROUND)
- “dance for me” / “dance now” → dance (BACKGROUND)
- “我喜歡跳舞” → chat
- “i like dancing” → chat
- “dance is fun” → chat
- “你會跳舞嗎？” → chat (ordering contract: chat_not_command precedes dance)
- “Reachy 噓” (bilingual) → stop_emotion (CRITICAL)
- “我聽到噓聲” → chat (不是命令; 噓 anchored fix, HIGH-A1)
- “觀眾在噓他” → chat (descriptive 噓, not imperative)

### HIGH-A1 fix: 噓 imperative anchoring

`r”噓”` bare pattern over-fired on nominal uses (“我聽到噓聲”, “觀眾在噓他”,
“他被噓下台”).  Changed to `r”(^|[，。！？\s])噓([，！。\s]|$)”` so 噓 must
stand alone or be bounded by whitespace / CJK punctuation.  Bare imperative
“噓”, “噓！”, “噓，小聲一點”, and the bilingual “Reachy 噓” all still fire
`stop_emotion CRITICAL`.

### HIGH-A2: OrchestratorWorker ambiguous handoff

`OrchestratorWorker` is the canonical home for the ambiguous intent handoff.
Three explicit dispatch lanes replace the previous silent drop:

- `command` → `_ACTION_TO_COMMAND` lookup → `scheduler.submit(MotionCommand)`
- `chat`    → `chat_handler(IntentResult)` callback, or log “chat_skipped”
- `ambiguous` → `ambiguous_handler(IntentResult)` callback (if wired), then
  always log “ambiguous_pending_llm_handoff” to `worker.events`.  Silent drop
  is forbidden: the event entry is written unconditionally.

Default behavior for ambiguous: log-and-observe (no LLM bridge yet).  Future:
wire an `ambiguous_handler` callback to an LLM client that returns a refined
`IntentResult`.  The `OrchestratorWorker` constructor accepts the callback as
`ambiguous_handler=<callable>` so the bridge can be injected without changing
this module.
