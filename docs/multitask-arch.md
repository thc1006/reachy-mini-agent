# B2 — Multitask Architecture (perception / dialog / motion)

**Branch**: `feature/multitask-arch`
**Worktree**: `C:/Users/thc1006/Desktop/reachy-mini-b2`
**Status**: design + POC + bench (this PR)
**Baseline commit**: 4e5f5c4 (master)

## 1. Why this exists

The current `src/robot_brain.py` already spawns four long-lived threads
(`tracking_loop`, `hand_worker`, `vision_worker`, `greet_and_talk`) plus
several transient threads, so the system is *not* purely serial. The real
serial bottleneck is the **conversation turn** inside `do_conversation()`:

```
record_utterance()    blocks until SILENCE_DURATION (1.4s) past last speech
        |             — mic gated by _speaking_event during prior TTS
        v
transcribe()          ~100-300ms (remote Whisper s1) / 500-1500ms (local fallback)
        v
ask_llm() / streaming TTFB    150-2000ms first chunk (vLLM TP=2 k=2/k=3)
        v
TTS playback          500-1500ms (Edge) / 200-500ms (Kokoro) / 300-2000ms (HaGen)
                      — _speaking_event keeps mic muted for whole playback + 0.4s drain
```

Cumulative user-end-of-speech → first-robot-audio (TTFB) is empirically
~850-1650ms (memory `project_reachy_capability_expansion_2026_05_11`).
Two specific consequences:

1. **No barge-in.** While the robot speaks, the mic is hard-gated by
   `_speaking_event` (commit 5ad699a). User cannot interrupt; long replies
   feel stuck.
2. **No prefill overlap.** STT only runs after the user stops talking,
   so LLM prefill begins ~SILENCE_DURATION + transcription_ms after the
   user's *last* word. The 1.4s silence wait alone is bigger than vLLM's
   own TTFB.

The B2 goal is to split this monolithic turn loop into three actors
communicating over an event bus, so:

- **Perception** can keep recording + emitting *partial* transcripts even
  while **Dialog** is mid-LLM-call,
- **Motion** is the single writer of `mini.set_target` (today it is co-owned
  by tracking_loop and tool dispatch with `_motion_lock`), so we get a
  clean place to fuse face-track + tool-driven head moves,
- the mic-gating shifts from "block during TTS playback" to "ignore STT
  results whose audio overlaps known TTS playback window" (or, future:
  AEC). Even without AEC, full duplex becomes architecturally possible.

## 2. Current architecture (as of 4e5f5c4)

### 2.1 Threads

| Thread | Frequency | Owns | Calls |
| --- | --- | --- | --- |
| `tracking_loop` | ~20-50 Hz | YuNet face detect, motion writes via `_motion_lock` (non-blocking) | `mini.media.get_frame()`, `mini.set_target()`, state-machine transitions, spawns `greet_and_talk` |
| `hand_worker` | 10 Hz | MediaPipe hand landmarker under `_hand_lock` | reads `_latest_frame` global, spawns `speak()` Thread on stable gesture |
| `vision_worker` | 1 / `VISION_INTERVAL`s (30s default) | scene caption | HTTP to vLLM/Ollama `/v1/chat/completions`, writes `_scene_desc` under `_scene_lock` |
| `greet_and_talk` (transient) | per CONVERSATION entry | `do_conversation()` body | record→STT→LLM→TTS, spawns `do_action` Threads for actions returned by LLM |
| `do_action` (transient, N per turn) | per action | one short motor move + a `speak()` line | `mini.goto_target()` |
| `speak()` (transient, N per chunk) | per TTS request | `_RobotSpeaker.play_audio` | `mini.media.play_sound()` |
| `TTSQueue` worker pool | 2 workers | concurrent TTS synth | edge/kokoro/hagen HTTP |
| `TTSQueue` player | 1 thread | in-order playback callback | calls `mini.media.play_sound()` via speaker |
| `main` | 0.5s sleep loop | lifecycle / SIGINT handler | `stop_event.set()` on shutdown |

### 2.2 Shared state and locks

| Object | Type | Writers | Readers |
| --- | --- | --- | --- |
| `_state` | `State` enum | `set_state()` | most threads |
| `_state_lock` | `Lock` | guarded by `get_state`/`set_state` | — |
| `_motion_lock` | `Lock` | non-blocking acquire in `tracking_loop`; held in `do_action` | only motion writers |
| `_hand_lock` | `Lock` | hand_worker (mediapipe not thread-safe) | — |
| `_latest_frame` / `_latest_frame_t` | globals | tracking_loop writes (implicitly via `mini.media.get_frame`) | hand_worker, vision_worker |
| `_scene_desc` / `_scene_desc_t` | globals | vision_worker | `_current_scene()` → LLM system prompt |
| `_scene_lock` | `Lock` | vision_worker | `_current_scene()` |
| `_llm_inflight_lock` | `Lock` | dialog (foreground LLM) | vision_worker (best-effort acquire) |
| `_whisper_lock` | `Lock` | `_transcribe_local` | — |
| `_conv_lock` | `Lock` | `_log_turn`, `_load_conv_memory` | — |
| `_MEM_SEARCH_LOCK` | `Lock` | `_cached_mem_search` | — |
| `_speaking_event` | `Event` | `_stream_tts` sets, drains, clears | `_record_via_robot_mic`, `_record_via_pc_mic` wait via `_wait_not_speaking` |
| `TTSQueue._lock` | `Lock` | futures list mutex | — |

### 2.3 What is already concurrent

- Vision caption (every 30s) overlaps with dialog LLM call (vLLM continuous
  batching, measured ~4% degradation in `project_vllm_option_f_2026_04_25`).
- TTS synthesis is double-buffered via `TTSQueue(max_concurrent=2)` while
  the player thread emits audio in submission order.
- Face tracking + hand gestures + scene caption all run independently of
  the conversation thread.

### 2.4 What is still serial

- **Inside one conversation turn**: STT, LLM TTFB, and first TTS audio are
  strictly sequential.
- **Mic ↔ TTS**: `_speaking_event` mutex makes them mutually exclusive at
  the API level, not just on the audio device.
- **Motion writer split**: tracking + tool dispatch both call into the
  same `mini.set_target`/`goto_target` API with `_motion_lock` arbitrating,
  but there is no single owner of the motion timeline — collisions are
  resolved by *time of arrival*, not priority.
- **Vision freshness**: scene caption fires on a wall-clock interval,
  not on demand from the dialog turn. With `VISION_INTERVAL=30` the scene
  TTL can drift past 30s during a multi-turn conversation.

## 3. Target architecture

Three actors, one bus, typed events. Each actor owns one external resource
exclusively; everything else flows through the bus.

```
                  +---------------------+
                  |     Event Bus       |
                  |  (topic pub/sub,    |
                  |  bounded queues,    |
                  |  backpressure)      |
                  +----+-----+-----+----+
                       |     |     |
              +--------+     |     +--------+
              v              v              v
       +-------------+ +-----------+ +-------------+
       | Perception  | |  Dialog   | |   Motion    |
       +-------------+ +-----------+ +-------------+
        owns: camera   owns: LLM     owns: mini.set_target
              mic            memory        goto_target
              face det       prompts       _motion_lock
              hand det
              VAD/STT
              vision-cap
```

### 3.1 Event taxonomy

Naming convention: `subject.verb` (past tense for state events,
imperative for commands).

| Topic | Payload | Producer | Consumers |
| --- | --- | --- | --- |
| `face.seen` | `{bbox, dx, dy, conf, frame_ts}` | perception | motion (tracking), dialog (greeting trigger) |
| `face.lost` | `{last_seen_ts}` | perception | motion (recenter), dialog |
| `hand.gesture` | `{n_fingers, stable_ms}` | perception | dialog (canned response) |
| `user.speech.started` | `{ts}` | perception (VAD) | dialog (cancel pending speak) |
| `user.speech.partial` | `{text, ts}` | perception (streaming STT) | dialog (warm prefill) |
| `user.speech.final` | `{text, audio_window, ts}` | perception | dialog |
| `scene.described` | `{text, ts}` | perception (vision worker) | dialog |
| `dialog.thinking` | `{ts}` | dialog (LLM started) | motion (subtle head idle) |
| `dialog.speech.chunk` | `{text, idx, is_first}` | dialog (LLM stream) | tts/audio |
| `dialog.speech.final` | `{text, actions, ts}` | dialog | logging/memory |
| `dialog.tool` | `{name, args}` | dialog | motion (move_head/play_emotion) |
| `audio.speak.started` | `{ts}` | audio | perception (mic-suppress window opens) |
| `audio.speak.ended` | `{ts}` | audio | perception (mic-suppress window closes after 0.4s drain) |
| `motion.done` | `{action, duration_ms}` | motion | dialog |

### 3.2 Why threading and not asyncio (this iteration)

| Aspect | threading (chosen) | asyncio |
| --- | --- | --- |
| Existing code style | already threads + Lock | full rewrite of `urlopen`, `pyaudio`, `mini.media.*` |
| CPU-bound bits (cv2 / np / mediapipe) | release the GIL natively | no benefit; need executor |
| HTTP to vLLM | `urllib` blocking is fine | needs `aiohttp` |
| `reachy_mini` SDK | blocking calls | no async surface |
| Migration delta from baseline | small (event bus + three workers) | large (refactor ~all I/O) |

We pick **threading + bounded `queue.Queue` per subscriber** for B2. An
asyncio variant is a *future* iteration that becomes attractive only if/when
we add WhisperLiveKit-style streaming STT (which is asyncio-native). Until
then, threading keeps the cognitive load close to baseline.

### 3.3 Failure modes and how the design handles them

| Risk | Mitigation |
| --- | --- |
| Slow subscriber backs up bus | bounded queue per subscriber + per-topic drop-policy (`drop_oldest` for high-rate face/mic, `block` for `dialog.tool`) |
| Two motion sources race | motion actor is the *only* writer of `set_target`; tracking and tool dispatch publish; actor serializes |
| Echo loop (TTS → mic) | perception subscribes to `audio.speak.{started,ended}` and ignores audio frames inside [started, ended+0.4s]; partial-STT results within that window are discarded |
| Vision and dialog hit the same vLLM endpoint | dialog publishes `dialog.busy`; perception's vision-cap loop subscribes and skips during the busy window — same effect as today's `_llm_inflight_lock` but via events |
| Dialog crashes mid-turn | actor wraps the per-turn function in `try/except` and re-emits `dialog.error`; bus keeps running |
| Bus thread leaks on shutdown | bus exposes a `stop()` that closes all subscriber queues; actors check a shared `stop_event` between event reads |

## 4. POC scope and non-scope

### 4.1 In scope (this PR)

- `src/orchestrator/event_bus.py` — threaded, topic-routed pub/sub
- `src/orchestrator/events.py` — typed event dataclasses
- `src/orchestrator/perception.py` — driver that simulates STT/vision/face events
- `src/orchestrator/dialog.py` — driver that simulates LLM streaming + tool calls
- `src/orchestrator/motion.py` — driver that simulates motor latency
- `src/orchestrator/runner.py` — wires actors + bus, runs one or many turns
- `bench_multitask.py` — head-to-head: serial vs concurrent on a fixed
  synthetic workload, N=20 turns, prints + JSON-dumps the latencies

### 4.2 Out of scope

- Hooking into the real `robot_brain.py`. The POC runs entirely on
  simulated workloads so the benchmark is reproducible on any machine
  (Pi 4, dev laptop, CI). Hooking real STT/LLM/TTS is the *next* PR
  once the architecture is signed off.
- Streaming Whisper. Today's Whisper path is *batch*; the POC simulates
  the partial-transcript feature it would unlock, so we can prove the
  ceiling for prefill-overlap savings.
- AEC (acoustic echo cancellation). The POC keeps the same mic-suppress
  window as production, just routed through events.

## 5. Benchmark plan

For each scenario, the same synthetic workload (deterministic delays) is
run N=20 turns. Latencies measured per turn:

| Metric | Definition |
| --- | --- |
| `ttfb_audio_ms` | from end-of-user-utterance to first robot audio sample played |
| `turn_total_ms` | from end-of-user-utterance to robot finishes speaking |
| `mic_blocked_ms` | wall-time the mic was muted (cannot capture new user audio) |
| `scene_age_ms_at_llm` | age of the most recent scene description when the LLM call started |

Hypotheses, with the numbers from baseline memory:

1. **TTFB drops** from ~`silence_wait + stt + ttfb_llm + ttfb_tts` to
   ~`max(stt, ttfb_llm) + ttfb_tts` once STT can run in parallel with
   user-still-talking (partial-transcript path).
2. **Mic-blocked time drops to 0** under the event-driven mic-suppress
   window, because the perception actor never *stops* capturing — it just
   discards frames known to overlap TTS.
3. **Scene freshness improves** because dialog publishes `dialog.thinking`
   and perception triggers an on-demand caption, instead of waiting for
   the 30s interval.

Concrete pass criteria (synthetic workload — see `bench_multitask.py` for
the exact constants):

| Metric | Serial baseline | Target | Pass if |
| --- | --- | --- | --- |
| `ttfb_audio_ms` p50 | ~2100ms | ≤ 1500ms | ≥ 30% reduction |
| `turn_total_ms` p50 | ~3500ms | ≤ 3000ms | ≥ 15% reduction |
| `mic_blocked_ms` p50 | ~1500ms | 0ms | == 0 |
| `scene_age_ms_at_llm` p95 | ~30000ms | ≤ 5000ms | ≥ 80% reduction |

### 5.1 Measured results (N=20, bench_multitask.py @ commit on this branch)

Two sweeps were run: one with optimistic streaming-STT endpointing
(`--stt-endpointing-ms 150`, the WhisperLiveKit best case) and one with
realistic endpointing (`--stt-endpointing-ms 350`). Both PASS the
criteria; the realistic config is the more honest projection of what
production should expect.

**Optimistic (150ms endpointing):**

| Metric | Serial p50 | Concurrent p50 | Delta | p95 delta |
| --- | --- | --- | --- | --- |
| `ttfb_audio_ms` | 2252ms | 802ms | -64.4% | -64.4% |
| `turn_total_ms` | 3654ms | 2204ms | -39.7% | -39.5% |
| `mic_blocked_ms` | 1802ms | 0ms | -100% | -100% |
| `scene_age_finite_ms` | 15650ms | 4006ms | -74.4% | -84.9% |

**Realistic (350ms endpointing):**

| Metric | Serial p50 | Concurrent p50 | Delta | p95 delta |
| --- | --- | --- | --- | --- |
| `ttfb_audio_ms` | 2252ms | 1002ms | -55.5% | -54.7% |
| `turn_total_ms` | 3654ms | 2405ms | -34.2% | -33.8% |
| `mic_blocked_ms` | 1802ms | 0ms | -100% | -100% |
| `scene_age_finite_ms` | 16816ms | 4207ms | -75.0% | -85.0% |

### 5.2 Caveats and limitations of the synthetic bench

1. **Latencies are sleeps.** Real STT/LLM/TTS have jitter, GPU thermal
   throttling, network hiccups (memory: vLLM 200s stalls happen rarely),
   and contention with vision calls (4% degradation measured).
2. **WhisperLiveKit endpointing** has not been re-measured for the
   Reachy stack since memory `project_community_state_2026_05_07`
   logged "mature, go". 150ms is the best-case in their benchmarks;
   350ms is the conservative production projection.
3. **Prefix-cache OFF** is the production winner (k=2/k=3 + cache OFF).
   Streaming partial-transcripts cannot warm the LLM KV cache while
   user is still talking. Once vLLM #38182 (MTP×prefix-cache conflict)
   is fixed, partial-prefill warming gains an additional ~200-400ms.
4. **scene_age p50 in the serial baseline** is a function of how the
   30s periodic worker's tick aligns with each turn. With a 30s tick
   and ~3.65s turn pace, the worker fires every ~8 turns; p50 lands
   near 15-17s, p95 near 30s. The concurrent path's on-demand vision
   triggers every turn, so the next turn always sees a ~4s-old scene.
5. **Motion writes** are single-owner in the POC (motion actor). The
   real ``mini.set_target`` vs ``goto_target`` collision risk is
   architecturally eliminated by this design but the migration PR has
   to wrap the existing tracking_loop carefully (see §6 step 3).
6. **Echo loop is not modelled.** In real hardware, even with mic-not-
   blocked, a 0.4s drain + RMS gate is still needed unless AEC is added.
   The POC treats the suppression window as a *statistic* (mic_blocked
   counter) without rejecting any specific audio samples.

If the POC hits all four targets (it does, in both sweeps), the
architecture is justified and the *next* PR is the migration of
`do_conversation()` to use the bus.

## 6. Migration plan (after POC sign-off — not in this PR)

1. Land the POC + bench (this PR).
2. Wrap existing `vision_worker` so it publishes `scene.described` instead
   of writing `_scene_desc` directly. Backwards-compat: `_current_scene()`
   subscribes and caches.
3. Wrap `tracking_loop` motion writes — emit `face.seen`/`face.lost` and
   route through motion actor.
4. Wrap `do_conversation()` — split into perception (record + STT) and
   dialog (LLM + emit chunks) actors. TTSQueue stays internal to dialog
   for this step; audio actor split deferred.
5. Add streaming STT (WhisperLiveKit on s1) producing
   `user.speech.partial` — measure real TTFB improvement.
6. Add AEC, drop the mic-suppress window — measure barge-in viability.

Each step is an independent, revertable commit on `feature/multitask-arch`.

## 7. References

- `src/robot_brain.py` baseline at 4e5f5c4 — `do_conversation` line 1871,
  `tracking_loop` line 1922, `vision_worker` line 288, `hand_worker`
  line 236, `_speaking_event` line 820, `main` line 2157.
- `src/streaming_tts.py` — `SentenceChunker`, `TTSQueue` (already a
  good example of producer/in-order-player decoupling).
- `project_v916_session_complete_2026_04_30.md` — origin of the
  `_speaking_event` design, vLLM+vision concurrency observation.
- `project_reachy_capability_expansion_2026_05_11.md` — TTFB numbers
  for OpenAI Realtime / Gemini Live / our stack.
- `project_v917_future_work_2026_04_30.md` — prior list of follow-ups;
  this PR addresses items #1 (speaking flag → event-driven) and lays
  groundwork for #2 (vision-in-CONVERSATION redesign) and #3 (STT
  streaming).
