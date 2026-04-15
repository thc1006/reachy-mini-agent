# Changelog

All notable changes to this project will be documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-04-16

First public release.

### Added
- End-to-end conversational pipeline: YuNet face tracking, MediaPipe hand-gesture reactions, faster-whisper STT, Ollama (Qwen3) chat, Kokoro + edge-tts speech, WebRTC bidirectional audio/video to the Reachy Mini daemon.
- Vision-language scene description (`qwen2.5vl:7b`) injected into the LLM system prompt every 10 s, sandboxed inside an untrusted-view fence to resist prompt injection from text in the camera frame.
- SHA-256 content-addressed disk cache for edge-tts with LRU eviction, peak normalization and configurable gain stage.
- `prewarm_tts_cache.py` that batch-generates WAV cache entries for every hardcoded phrase plus the most repeated past robot utterances from `conversation_log.jsonl`.
- Persistent conversation memory (last 20 turns replayed into every LLM call).
- Three-service deployment model (brain + Kokoro + Whisper) with user-level systemd unit templates.
- Full documentation: architecture, hardware requirements, end-to-end setup guide, troubleshooting table.

[Unreleased]: https://github.com/thc1006/reachy-mini-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thc1006/reachy-mini-agent/releases/tag/v0.1.0
