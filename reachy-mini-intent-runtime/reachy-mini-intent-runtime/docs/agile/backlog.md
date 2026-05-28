# Agile Backlog

## Epic A: Intent routing

### A1 Local command guardrail

As a user, I want “停止跳舞” to stop motion immediately, so that Reachy remains safe and responsive.

Acceptance:

- Chinese and English stop/hush phrases map to critical actions.
- Non-imperative mentions do not trigger actions.

### A2 LLM JSON router prompt

As a developer, I want ambiguous utterances routed through a strict JSON prompt, so that Claude/LLM output is predictable.

Acceptance:

- Prompt returns only JSON.
- Examples cover command, chat, ambiguous.

## Epic B: Motion scheduling

### B1 Priority queue scheduler

Acceptance:

- Critical > interactive > background.
- FIFO within same priority.

### B2 Cooperative chunking

Acceptance:

- Long dance can be represented as chunks.
- Critical event is checked at chunk boundary.

### B3 Native SDK interrupt experiment

Acceptance:

- Run CLI test: start dance, issue stop while dance runs.
- Document whether SDK supports immediate stop/pause/resume.

## Epic C: Resource isolation

### C1 Pi-side resource monitor

Acceptance:

- CPU threshold warning in logs.
- Heartbeat deadline miss detection.

### C2 Process split script

Acceptance:

- Document Pi-side vs workstation commands.
- Do not run heavy VLM on Pi unless measured.

## Epic D: Hospital assistant demo

### D1 Synthetic dialogue script

Acceptance:

- Orientation, call nurse, IV drip comfort, stop while dancing.

### D2 Demo recorder checklist

Acceptance:

- Time-lapse/progress capture checklist.
- Synthetic data only.
