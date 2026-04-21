# Changelog

All notable changes to this project will be documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Unified vision-language model**: retired the separate `qwen2.5vl:7b`
  worker; `qwen3.6:35b-a3b` (MoE, 3 B active, MMMU 81.7, RefCOCO 92) now
  serves both dialog and vision, saving ~6 GB VRAM on GPU 0 and cutting
  vision latency from 7 s cold to ~1.2 s warm.
- **Streaming LLM → sentence chunker → TTS queue** pipeline
  (`src/streaming_tts.py`). The first sentence starts speaking while the
  LLM is still generating the rest, halving perceived first-audio latency.
  Sentence chunker handles abbreviations, decimals, and CJK terminators
  (`。！？`).
- **Tool calling** (`src/robot_tools.py`) with an OpenAI-schema registry
  and an Ollama multi-turn loop. Eight tools shipped: `get_current_time`,
  `stop_motion`, `move_head`, `play_emotion`, `recall_memory`, plus three
  vision tools `see_what`, `find_in_view`, `count_items` backed by the
  unified Qwen3.6 VL.
- **Long-term memory** (`src/robot_memory.py`) backed by Mem0 + embedded
  Qdrant + local `bge-m3` embedder. Per-turn fact extraction runs async;
  an incremental rolling summary regenerates every N turns by folding the
  previous summary together with new turns (O(1) in total history).
- **Persona post-processor** that strips stray emotion prefixes (`Nod!`,
  `Think!`) the model emits despite prompt instructions, and rewrites the
  Japanese reading "Hideyoshi" back to "Hsiu-Chi".
- **Cross-worker lock** serialising the dialog LLM with the background
  vision worker on the shared Ollama endpoint (prevents vision-worker
  timeouts during active dialog).
- New optional-dependency groups: `memory` (mem0ai, qdrant-client,
  ollama) and `dev` (huggingface-hub[cli], pytest, pandas, matplotlib).
- New env knobs in `scripts/run_robot.sh`: `VISION_MODEL`,
  `OLLAMA_THINK`, `LLM_STREAMING`, `LLM_TOOLS`.

### Changed
- TTS playback pad increased from 0.3 s to 1.2 s so the WebRTC +
  GStreamer + USB audio buffer fully drains before `stop_playing`.
- Conversation window widened from 20 to 60 messages (30 Q&A turns)
  now that the LLM runs with 16 k context.

### Fixed
- Streaming path used to return 0 chars when qwen3.6 leaked
  `<think>...</think>` into `content` under long Mem0 system prompts —
  added a cross-chunk stripper and a whitespace-tolerant marker regex
  (`"\s*speech\s*"\s*:\s*"`).
- `bge-m3` returning NaN embeddings on pure-whitespace / ZWJ inputs now
  silently drops the turn (HTTP 500 handler) instead of polluting Qdrant.
- `flush_summary(timeout)` previously ignored the timeout argument; now
  uses Future-tracking + `concurrent.futures.wait` for a real bound.
- Executor swap in `flush_summary` is now serialised under
  `_summary_lock` so concurrent `_schedule_summary_maybe` never submits
  to a shut-down pool.
- `VISION_MODEL` / `OLLAMA_HOST` are now resolved per-call so runtime
  env changes take effect without restart.
- `_parse_count_response` picks the integer nearest the query term in
  the VL reply instead of blindly the first integer, avoiding the
  "0 cups but 3 plates" failure mode when asking about cups.
- `_sanitize_description` strips `{}[]<>` and backtick characters from
  user-controlled descriptions before they are interpolated into VL
  prompts (prompt-injection defence).

### Tests
- Suite grows from a smoke test to **107 passed / 3 skipped / 3 xfailed**
  across 11 test files covering the sentence chunker, speech extractor,
  TTS queue, tools, vision tools, memory, rolling summary, three
  integration/review-fix test files, and documented-failure tests for
  Ollama's (non-)support of speculative decoding.

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
