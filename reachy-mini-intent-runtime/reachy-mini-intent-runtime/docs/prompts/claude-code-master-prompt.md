# Claude Code Master Prompt

Copy this into Claude Code at the project root.

```text
You are working in the reachy-mini-intent-runtime repository.

Goal:
Build a production-oriented prototype that lets Reachy Mini use natural language to trigger official SDK dances/emotions/actions while preserving interruption, responsiveness, and CPU budget. The teacher's key requirement is: Reachy may dance, but it must still hear/see the user and stop when asked before the dance finishes.

Required context:
1. Read AGENTS.md, CLAUDE.md, README.md.
2. Read docs/adr/0001-cooperative-priority-scheduler.md, docs/adr/0002-command-vs-chat-router.md, docs/adr/0003-pi-resource-budget-and-worker-split.md.
3. Read docs/sdd/01-requirements.md, docs/sdd/02-design.md, docs/sdd/03-test-plan.md.
4. Read docs/research/2026-05-28-official-stack.md and refresh upstream facts if internet access is available.

Development method:
Use SDD + TDD. For each task:
- clarify behavior in docs or tests;
- write/update failing tests first;
- implement minimal code;
- run ./scripts/verify.sh;
- update ADR/SDD only when architectural behavior changes.

First implementation target:
1. Strengthen the local command guardrail in src/reachy_intent_runtime/classifier.py.
2. Ensure “跳支舞” triggers background dance.
3. Ensure “我喜歡跳舞” is chat, not command.
4. Ensure “停止跳舞”, “不要跳了”, “stop dancing”, and “噓” become critical stop/quiet actions.
5. Strengthen ActionScheduler so critical actions preempt current interruptible background motion.
6. Keep all core logic hardware-free and covered by tests.

Second target:
1. Implement external_content/external_tools/interruptible_action.py so it can be loaded by the official Reachy Mini conversation app as an external tool.
2. Keep it thin: validate input, call this package, return high-signal JSON.
3. Do not hard-code secrets or patient data.

Research target:
Create/update docs/research/2026-05-28-installable-stack.md with exact current install commands and version caveats for:
- reachy_mini SDK;
- reachy_mini_conversation_app;
- uv;
- mediapipe head tracking;
- optional YOLO head tracking;
- off-board VLM/LLM path;
- Claude Code project settings/hooks/subagents.
Only include commands you have source-verified. If a command is uncertain, mark it as TODO with the source URL and verification question.

Quality gate:
Before final response, run ./scripts/verify.sh. If it fails, fix the failure or report exactly why it cannot be fixed.
```
