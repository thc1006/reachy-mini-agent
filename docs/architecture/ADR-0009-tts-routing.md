# ADR-0009: TTS_BACKEND is the canonical TTS routing knob

- **Status**: Accepted
- **Date**: 2026-06-01
- **Author**: hctsai1006
- **Supersedes**: none (deprecates the legacy `TTS_ENGINE` env)
- **Related**: CosyVoice review YELLOW (sub-agent `acf961132ab646d92`),
  `project_cosyvoice_deploy_2026_06_01.md` (memory placeholder if not yet
  written — same-day deploy of CosyVoice 2 0.5B FastAPI wrapper on vllm0528
  card 0, port 8881), `src/robot_brain.py::_stream_tts`,
  `scripts/cosyvoice_server.py`, `src/brain_observability.py::_KNOWN_BACKENDS`.

## 1. Context

Two env knobs currently select the TTS engine:

- `TTS_BACKEND` (introduced with CosyVoice rollout). Recognized: `edge`,
  `kokoro`, `cosyvoice`, `edge_then_cosyvoice`. Read first in `_stream_tts`.
- `TTS_ENGINE` (legacy, ~2026-05-11 HaGen rollout). Recognized: `edge`,
  `kokoro`, `hagen`. Read only when `TTS_BACKEND` is unset/empty.

Pi `/home/pollen/brain/.env` and the repo `.env.example` set BOTH —
`TTS_BACKEND=edge` and `TTS_ENGINE=edge`. The `_stream_tts` ladder branches
on `TTS_BACKEND` first and returns the audio before the legacy switch is
reached. Operators trying to canary HaGen by flipping `TTS_ENGINE=hagen`
without also touching `TTS_BACKEND` get silent override: their canary never
fires and the bug is invisible in dashboards.

## 2. Decision

**`TTS_BACKEND` is the canonical TTS routing selector.** `TTS_ENGINE` is
deprecated; brain logs a one-shot `tts_engine_deprecated` warning at
startup whenever `TTS_ENGINE` is set, regardless of value, to make the
override visible in journald. The legacy `TTS_ENGINE` switch remains in
`_stream_tts` to preserve HaGen muscle memory during the deprecation
window but is documented as a fallback path only.

`.env.example` ships only `TTS_BACKEND=edge`; the `TTS_ENGINE` line is
commented out with a "DEPRECATED" pointer to this ADR. The Pi `.env` has
`TTS_ENGINE=edge` commented to `#TTS_ENGINE_DEPRECATED=edge` so the audit
trail is preserved without affecting brain dispatch.

## 3. Consequences

**Positive:**

- One knob to reason about. Canary rollouts (HaGen, future engines) extend
  `TTS_BACKEND` rather than reviving the legacy switch.
- Startup warning surfaces silent overrides in journald; the failure mode
  goes from "canary appears broken" to "warning explains the override."
- `_KNOWN_BACKENDS` in `brain_observability.py` now includes `cosyvoice`,
  so the breaker lookup on every TTS turn is lock-free (hot-path parity
  with `vllm` / `whisper` / `kokoro`).
- Per-backend `fail_max` override (`cosyvoice: 2` vs default `5`) caps
  worst-case degraded UX at ~16 s before edge fallback kicks in.

**Negative:**

- One more file to keep in sync at rollout time (the Pi `.env` `sed`
  command, the repo `.env.example`, this ADR). The startup warning is the
  forcing function.

**Migration:** Operators reading this ADR for the first time should
`grep TTS_ENGINE /home/pollen/brain/.env` and comment the line out. Brain
restart will then stop logging the deprecation warning.
