# Research Refresh Prompt for Claude Code

```text
Please refresh docs/research/2026-05-28-official-stack.md into docs/research/latest-installable-stack.md.

Requirements:
- Use official sources first: Pollen Robotics, Hugging Face Reachy Mini docs, official GitHub repos, Anthropic/Claude Code docs.
- Verify exact installation commands for Reachy Mini SDK and reachy_mini_conversation_app.
- Verify whether the conversation app still exposes dance, stop_dance, play_emotion, stop_emotion.
- Verify the current method for external profiles and external tools.
- Verify Claude Code current settings/hook/subagent file schema.
- If a source has changed since 2026-05-28, say exactly what changed and update the docs.
- Do not install anything until commands are source-verified.
- Produce a Markdown file with: source URL, command, version/date, confidence, and caveats.
```
