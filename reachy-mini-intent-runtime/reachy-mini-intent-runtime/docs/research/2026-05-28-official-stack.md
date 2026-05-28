# 2026-05-28 Research Snapshot: Reachy Mini + Claude Code Stack

This note is a source-grounded starting point for Claude Code. Re-check upstream docs before installing in a production or lab demo environment.

## Reachy Mini official / upstream facts

### Reachy Mini SDK and hardware

- Reachy Mini documentation describes two variants: Reachy Mini Wireless runs on-board with Raspberry Pi 4 and IMU; Reachy Mini Lite runs on a personal computer and wall power.
- The Python SDK reference says `ReachyMini()` now auto-detects USB/localhost/network connection mode, with optional explicit `connection_mode="localhost_only" | "network"`.
- SDK movement includes `goto_target` for smooth interpolation of head, antennas, and body yaw, and `set_target` for high-frequency instant control.

Source URLs:

- https://huggingface.co/docs/reachy_mini/en/index
- https://huggingface.co/docs/reachy_mini/en/SDK/python-sdk

### Recorded moves and official motion content

- Hugging Face docs describe recorded moves and pre-built libraries for dances and emotions.
- Dataset identifiers used by official examples:
  - `pollen-robotics/reachy-mini-dances-library`
  - `pollen-robotics/reachy-mini-emotions-library`
- The emotions dataset describes each move as a JSON trajectory with head pose, antennas, body yaw sampled over time, paired with a WAV audio track.

Source URLs:

- https://huggingface.co/docs/reachy_mini/examples/recorded_moves
- https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library

### Conversation app

The official conversation app README describes:

- realtime audio conversation loop;
- optional VLM / local vision path;
- layered motion system that queues primary moves while blending speech-reactive wobble and head-tracking;
- async tool dispatch;
- LLM tools including `move_head`, `camera`, `head_tracking`, `dance`, `stop_dance`, `play_emotion`, `stop_emotion`, `idle_do_nothing`;
- external profiles/tools through `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`, `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`, `REACHY_MINI_CUSTOM_PROFILE`;
- warning that `--local-vision` is not supported when running directly on Reachy Mini Wireless / Raspberry Pi, so local vision should stay on laptop/workstation.

Source URL:

- https://github.com/pollen-robotics/reachy_mini_conversation_app

## Claude Code / Anthropic facts used in this repo

- Claude Code is an agentic coding tool that reads the codebase, edits files, runs commands, and integrates with development tools.
- Official docs recommend `CLAUDE.md` for persistent project context, keeping it concise and human-readable.
- Project settings live in `.claude/settings.json`; local/private settings should live in `.claude/settings.local.json`.
- Hooks can run at lifecycle events such as `Stop`, `PreToolUse`, `PostToolUse`; they can format, block risky commands, inject context, or run checks.
- Subagents are Markdown files with YAML frontmatter under `.claude/agents/`; only `name` and `description` are required, with optional tools/model/effort/isolation.
- Anthropic tool-use docs emphasize detailed tool descriptions, meaningful namespacing, fewer more capable tools, and high-signal tool responses.
- Anthropic context-engineering guidance recommends keeping context informative but tight, and using diverse canonical examples instead of stuffing every edge case into the prompt.

Source URLs:

- https://code.claude.com/docs/en/overview
- https://code.claude.com/docs/en/best-practices
- https://code.claude.com/docs/en/settings
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/sub-agents
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

## Engineering implication

Do not make the LLM directly responsible for timing-critical interruption. The LLM should classify or request actions, but a deterministic local scheduler must enforce priority, interruption, and CPU budget.

Recommended runtime split:

1. **Pi-side always-on loop**: speech/VAD heartbeat, lightweight intent/event receiver, motion scheduler, emergency interrupt, SDK adapter.
2. **Off-board heavy loop**: LLM/VLM, local vision model, large ASR/TTS if needed, long context.
3. **Shared contract**: JSON action messages with priority, duration, chunk size, interrupt policy, and deadline.
