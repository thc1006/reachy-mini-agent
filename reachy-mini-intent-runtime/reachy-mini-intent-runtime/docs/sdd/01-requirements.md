# SDD 01: Requirements

## Primary goal

Reachy Mini should naturally map user language to official SDK dances/emotions/actions while preserving responsiveness, interruption, and resource isolation.

## Functional requirements

### FR-001 Command-vs-chat routing

The system shall classify utterances into:

- `chat`: no robot action;
- `command`: executable robot action;
- `ambiguous`: ask clarifying question or choose safe no-op.

### FR-002 Natural language emoji/action mapping

The system shall map utterances such as:

- “跳支舞” → `dance`
- “開心一點” → positive emotion / `play_emotion`
- “噓，小聲一點” → stop/quiet behavior
- “停止跳舞” → `stop_dance`

### FR-003 Interruptibility

The system shall allow high-priority commands to interrupt long-running dances/emotions.

### FR-004 Motion smoothness

Chunked motions shall not create human-visible stop-start jitter beyond the configured smoothness threshold.

### FR-005 Resource protection

The Pi-side scheduler, VAD/interrupt loop, and camera sampling must keep running while background motions execute.

### FR-006 Hospital assistant demo

The system shall support a synthetic hospital assistant use case:

- patient orientation;
- “call nurse” response;
- IV drip anxiety comfort message;
- expressive empathy;
- stop/quiet command during dance.

## Non-functional requirements

- P95 Pi-local critical stop latency target: < 500 ms.
- P95 LLM-mediated stop latency target: < 2 s.
- Core scheduler unit tests must run without hardware.
- No real patient data in tests or demo scripts.
