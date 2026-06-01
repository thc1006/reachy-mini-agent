# ADR-0006: Move Mem0 + fastembed + Qdrant off Pi brain to `mem0-bridge` on vllm0528

- **Status**: Proposed (pre-design, blocks on ADR-0005 / Option A vLLM+Moondream2 ship)
- **Date**: 2026-05-29
- **Author**: hctsai1006
- **Supersedes**: none
- **Related**: `src/robot_memory.py` post-#98 (commit f0b33fe), `reference_vllm0528.md`, `project_brain_on_pi_2026-05-29`

---

## 1. Context

**Current state (Pi brain, post-#98 f0b33fe):**
- `src/robot_memory.py` runs in-process: provider-aware Mem0 (`MEM0_LLM_PROVIDER=openai` → vllm0528:8000, `MEM0_EMBED_PROVIDER=fastembed` → BAAI/bge-small-zh-v1.5 512-dim CPU).
- Qdrant **embedded** mode at `/home/pollen/brain/.qdrant_memory` (single-writer LMDB lock).
- LLM-based extraction + rolling-summary call already remote to vllm0528:8000/v1; only embeddings + vector store are local.
- Init: lazy on first turn via `_get_robot_memory()`; M-M4 spawns eager-warm thread after `brain_ready` so first dialog doesn't pay 5–8 s cold-load.
- RSS cost on Pi: fastembed ONNX runtime + bge-small-zh weights + Qdrant segments + Mem0 wrappers ≈ **~400 MB** of the 4 GB total.

**Target state:**
- Pi `robot_memory.py` is a ~50 LOC HTTP client. No fastembed, onnxruntime, qdrant_client, or mem0 imported.
- `mem0-bridge` runs on vllm0528 under supervisord, owns fastembed + Qdrant + Mem0 lifecycle.
- Brain → bridge over Tailscale TCP (HTTP/JSON). Same public API (`add_turn`, `search`, `get_rolling_summary`, `flush_summary`, `close`).
- Pi save: **~350–400 MB RSS**, plus shrunken `requirements-brain.txt` (no `fastembed`, `onnxruntime`, `qdrant-client`, `mem0ai`).

---

## 2. Decision

**Run a single `mem0-bridge.service` FastAPI app on vllm0528:8003 that wraps Mem0 (with fastembed CPU + embedded Qdrant in-process), exposing a small REST API consumed by a thin HTTP client on the Pi.** Keep embedding model **BAAI/bge-small-zh-v1.5 (512-dim)** to avoid re-embedding migration. Reuse the existing nginx Bearer auth pattern from #71. Single service (not split qdrant-server + bridge) — simpler, no extra port, the bridge process is the single writer (preserves M-M6 invariant).

Key choices:
| # | Decision | Why |
|---|---|---|
| 1 | Single FastAPI process (mem0 + fastembed + qdrant embedded) | One writer (M-M6); 350 MB on vllm0528 is negligible vs 32 GB host RAM; fewer moving parts on no-systemd container |
| 2 | Port **8003** | 8000 vLLM / 8001 Whisper-not-yet / 9000 Whisper actual / 8880 Kokoro; 8003 is free |
| 3 | Keep **bge-small-zh-v1.5 (512-dim)** | Existing Qdrant collection is 512-dim; upgrading to bge-m3 1024 = re-embed every stored elder fact. Defer (see §12) |
| 4 | Qdrant **embedded** in-process | Already proven on Pi; running separate qdrant-server adds ops surface for no benefit at our scale (~few-thousand facts) |
| 5 | **REST + JSON**, Bearer auth, same key as vLLM nginx | Lowest cognitive load; matches existing tooling |
| 6 | Brain client async via `ThreadPoolExecutor` (preserve current behavior) | `add_turn` stays fire-and-forget; main loop never blocks on bridge RTT |
| 7 | Rolling summary stays in the bridge | Bridge already has the full log; brain shouldn't ship 20-turn windows over the wire each cycle |

---

## 3. HTTP API spec

Base URL: `http://vllm0528:8003`. All requests `Authorization: Bearer $MEM0_BRIDGE_TOKEN`.

### `POST /v1/memory/add`
```json
// Request
{
  "user_id": "default",
  "text": "User: 我吃血壓藥\nAssistant: 好的，已記下",
  "metadata": {"namespace": "elder_facts", "source": "voice_turn"}
}
// Response 202 (fire-and-forget; bridge schedules its own thread)
{"queued": true, "id": "uuid4"}
```

### `POST /v1/memory/search`
```json
// Request
{
  "user_id": "default",
  "query": "user medications",
  "limit": 3,
  "filters": {"metadata.namespace": "elder_facts"}
}
// Response 200
{
  "results": [
    {"memory": "User takes blood pressure medication daily", "score": 0.82},
    ...
  ]
}
```

### `GET /v1/memory/summary?user_id=default`
```json
// 200
{"summary": "The user is a 70-year-old...", "covered_turns": 240, "updated_ts": "2026-05-29T11:32:14Z"}
// 404 if no summary yet
```

### `POST /v1/memory/summary/flush?user_id=default&timeout=240`
Blocks until any in-flight summary regen finishes. Returns `{"ok": true}` or `{"ok": false, "still_running": 1}` on timeout (matches current `flush_summary()` semantic).

### `GET /health`
`{"status": "ok"}` — liveness, no dependency check.

### `GET /ready`
`{"ready": true, "qdrant": "ok", "embed_dim": 512, "collection": "reachy_memory", "facts": 1247}` — readiness; Pi uses this in degradation check.

### Error semantics
- `5xx` from bridge → brain logs + treats `add` as drop, `search` as empty list, `summary` as `""`. **Never propagate to dialog loop.**
- `401` → brain logs `auth_failed` once per hour (would mean token rotation needed); same degradation as 5xx.
- Brain timeout: **2 s** for `add` (queued), **4 s** for `search` (matches current `timeout=4.0` arg), **30 s** for `summary/flush`.

---

## 4. Brain-side stub (`src/robot_memory.py` new shape, ~50 LOC sketch)

```python
"""HTTP client wrapper for mem0-bridge on vllm0528:8003.
Preserves the public API of the legacy in-process RobotMemory so
robot_brain.py needs zero changes beyond import-source."""
from __future__ import annotations
import json, os, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import List

class RobotMemory:
    def __init__(self, user_id: str = "default", **_ignored):
        self.user_id = user_id
        self.base = os.getenv("MEM0_BRIDGE_URL", "http://vllm0528:8003")
        self.token = os.getenv("MEM0_BRIDGE_TOKEN", "")
        self.namespace = os.getenv("MEM0_NAMESPACE", "elder_facts")
        self.enabled = os.getenv("ROBOT_MEMORY", "1") == "1"
        self._exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem0-http")
        self._lock = threading.Lock()
        # Probe /ready once; if down, disable to skip per-turn HTTP cost.
        if self.enabled:
            try:
                self._req("GET", "/ready", None, timeout=2.0)
                print(f"  [robot_memory] bridge ok @ {self.base}")
            except Exception as e:
                print(f"  [robot_memory] bridge unreachable, disabled: {e}")
                self.enabled = False

    def _req(self, method, path, body, timeout):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def add_turn(self, user_text: str, bot_text: str) -> None:
        if not self.enabled: return
        text = f"User: {(user_text or '').strip()}\nAssistant: {(bot_text or '').strip()}"
        body = {"user_id": self.user_id, "text": text[:4000],
                "metadata": {"namespace": self.namespace, "source": "voice_turn"}}
        self._exec.submit(self._add_safe, body)

    def _add_safe(self, body):
        try: self._req("POST", "/v1/memory/add", body, timeout=2.0)
        except Exception as e: print(f"  [robot_memory] add err: {e}")

    def search(self, query: str, limit: int = 3, timeout: float = 4.0,
               namespace: str | None = None) -> List[str]:
        if not self.enabled or not query: return []
        ns = namespace if namespace is not None else self.namespace
        body = {"user_id": self.user_id, "query": query, "limit": limit,
                "filters": {"metadata.namespace": ns} if ns else {}}
        try:
            data = self._req("POST", "/v1/memory/search", body, timeout=timeout)
            return [r["memory"] for r in data.get("results", []) if r.get("memory")]
        except Exception as e:
            print(f"  [robot_memory] search err: {e}"); return []

    def get_rolling_summary(self) -> str:
        if not self.enabled: return ""
        try:
            return self._req("GET", f"/v1/memory/summary?user_id={self.user_id}",
                             None, timeout=3.0).get("summary", "")
        except Exception: return ""

    def flush_summary(self, timeout: float = 240.0) -> bool:
        if not self.enabled: return True
        try:
            return self._req("POST",
                f"/v1/memory/summary/flush?user_id={self.user_id}&timeout={int(timeout)}",
                None, timeout=timeout + 5).get("ok", False)
        except Exception: return False

    def flush(self, timeout: float = 60.0) -> None:
        # local executor flush — bridge has its own queue
        self._exec.shutdown(wait=True); self._exec = ThreadPoolExecutor(max_workers=2)

    def close(self, timeout: float = 5.0) -> None:
        if self._exec: self._exec.shutdown(wait=True); self._exec = None
```

Backward compat (§9): if `MEM0_BRIDGE_URL` is unset *and* `MEM0_LOCAL=1`, dynamically `from .robot_memory_local import RobotMemory as _Local` and delegate (keep old file renamed for dev).

---

## 5. Bridge-side service (`mem0-bridge/`)

Layout on vllm0528 at `/home/hctsai1006/mem0-bridge/`:
```
mem0-bridge/
  app.py             # FastAPI app
  config.py          # env-driven config
  requirements.txt   # fastapi uvicorn mem0ai fastembed qdrant-client
  run.sh             # nohup wrapper for supervisord
  .qdrant_memory/    # persistent volume (NFS-backed via /home)
```

`app.py` sketch (~120 LOC):
```python
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from mem0 import Memory
import os, threading, time

TOKEN = os.environ["MEM0_BRIDGE_TOKEN"]
QPATH = os.getenv("QDRANT_PATH", "/home/hctsai1006/mem0-bridge/.qdrant_memory")

CFG = {
  "llm": {"provider": "openai", "config": {
      "model": "qwen36-awq",
      "openai_base_url": "http://127.0.0.1:8000/v1",  # local vLLM
      "api_key": "dummy", "temperature": 0.1}},
  "embedder": {"provider": "fastembed",
      "config": {"model": "BAAI/bge-small-zh-v1.5"}},
  "vector_store": {"provider": "qdrant", "config": {
      "path": QPATH, "collection_name": "reachy_memory",
      "embedding_model_dims": 512}},
}
mem = Memory.from_config(CFG)
write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0-w")
summary_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0-s")
write_lock = threading.Lock()
app = FastAPI()

def _auth(authz: str):
    if not authz or not authz.startswith("Bearer ") or authz[7:] != TOKEN:
        raise HTTPException(401, "bad token")

class AddReq(BaseModel):
    user_id: str; text: str; metadata: dict = {}

@app.post("/v1/memory/add")
def add(req: AddReq, authorization: str = Header(None)):
    _auth(authorization)
    def _do():
        with write_lock:
            try: mem.add(req.text, user_id=req.user_id, metadata=req.metadata)
            except Exception as e: print(f"add err: {e}", flush=True)
    write_pool.submit(_do)
    return {"queued": True}

class SearchReq(BaseModel):
    user_id: str; query: str; limit: int = 3; filters: dict = {}

@app.post("/v1/memory/search")
def search(req: SearchReq, authorization: str = Header(None)):
    _auth(authorization)
    f = {"user_id": req.user_id, **req.filters}
    r = mem.search(query=req.query, filters=f, limit=req.limit)
    items = r.get("results", []) if isinstance(r, dict) else r
    return {"results": items}

# /v1/memory/summary GET + flush POST + /health + /ready follow same pattern;
# summary uses summary_pool, port the existing _regenerate_summary() body
# from the legacy robot_memory.py (incremental watermark + atomic write).
```

Launch:
```bash
MEM0_BRIDGE_TOKEN=$(cat ~/secrets/mem0_bridge_token) \
  ~/venvs/mem0-bridge/bin/uvicorn app:app --host 0.0.0.0 --port 8003 \
  --workers 1  # single worker — Qdrant single-writer + Mem0 stateful
```

---

## 6. Supervisord stanza (vllm0528)

Append to `/home/hctsai1006/supervisord/conf.d/mem0-bridge.conf` (per existing pattern — see vllm0528 reference note, all services currently nohup pending Wave 4-5 supervisord migration):

```ini
[program:mem0-bridge]
command=/home/hctsai1006/mem0-bridge/run.sh
directory=/home/hctsai1006/mem0-bridge
autostart=true
autorestart=true
startretries=5
startsecs=10
stopsignal=TERM
stopwaitsecs=30                  ; allow Qdrant graceful flush on SIGTERM
stdout_logfile=/home/hctsai1006/vllm-logs/mem0-bridge.log
stderr_logfile=/home/hctsai1006/vllm-logs/mem0-bridge.err
environment=MEM0_BRIDGE_TOKEN="%(ENV_MEM0_BRIDGE_TOKEN)s",QDRANT_PATH="/home/hctsai1006/mem0-bridge/.qdrant_memory"
user=hctsai1006
```

**Boot dependency**: deliberately *no* `depends_on` for vLLM. Bridge starts independently; if vLLM is down only the LLM-based extraction + summary path fails, but `search` (pure embedding+vector lookup) still works. This preserves elder-care lookup during vLLM cold starts.

`run.sh`:
```bash
#!/bin/bash
exec /home/hctsai1006/venvs/mem0-bridge/bin/uvicorn app:app \
     --host 0.0.0.0 --port 8003 --workers 1 --log-level info
```

---

## 7. Migration plan (one-shot)

Existing data lives at `/home/pollen/brain/.qdrant_memory` (Pi). One-shot script `tools/migrate_qdrant_to_bridge.sh`:

```bash
# 1. Brain side: ensure no writer
ssh reachy 'sudo systemctl stop reachy-brain'
# 2. Snapshot (embedded Qdrant flushes on close, but verify no .lock files)
ssh reachy 'tar -C /home/pollen/brain -czf /tmp/qdrant_memory.tgz .qdrant_memory'
# 3. Move
scp reachy:/tmp/qdrant_memory.tgz vllm0528:/tmp/
ssh vllm0528 'mkdir -p ~/mem0-bridge && cd ~/mem0-bridge && \
              tar -xzf /tmp/qdrant_memory.tgz && \
              ls .qdrant_memory'
# 4. Start bridge (supervisord)
ssh vllm0528 'supervisorctl start mem0-bridge'
# 5. Verify count
curl -H "Authorization: Bearer $TOK" http://vllm0528:8003/ready
# expect facts > 0 matching pre-migration count
# 6. Brain side: deploy new robot_memory.py (HTTP client), set env, start
ssh reachy 'sudo systemctl start reachy-brain'
# 7. Smoke: ask robot "what medications do I take?" → expect prior facts surfaced
```

Validation gate: pre-migration `mem.search("medication")` result list MUST equal post-migration result list (sorted by score, top-3). Script halts and rolls back if mismatch.

---

## 8. Failure modes + degradation

| Failure | Detected by | Brain behavior |
|---|---|---|
| Bridge unreachable (Tailscale flap, vllm0528 container restart) | HTTP timeout on `_req` | `add_turn` drops silently; `search` returns `[]`; `get_rolling_summary` returns `""`. Dialog continues with no memory context — same UX as elder_care flag-off mode |
| Bridge 5xx | HTTP error | Same as above; logged once per minute (rate-limited) |
| 401 (token rotation) | HTTP 401 | Log + degrade; ops alert; brain still serves dialog |
| vLLM down on vllm0528 (bridge up) | Bridge add returns 202 still, summary regen fails silently inside bridge | `search` still works (pure embedding); `summary` becomes stale until vLLM returns. Acceptable. |
| Qdrant corruption on bridge | Bridge `/ready` returns 503 | Brain disables on init; ops must restore from Pi snapshot or accept fresh DB |
| Container restart on TWCC | supervisord restarts bridge; Qdrant on NFS persists | <30 s downtime; brain auto-retries on next turn |
| Brain SIGKILL mid-add | n/a from bridge POV (just lost request) | Worst case: 1–2 dropped facts. No corruption (remote Qdrant, brain has no DB to corrupt) — **net improvement vs current embedded mode**. |

---

## 9. Rollback plan

If bridge is broken or unacceptable latency:

1. Set `ROBOT_MEMORY=0` on Pi → brain runs with zero memory; smoke OK, ship dialog.
2. To restore local: `pip install fastembed onnxruntime qdrant-client mem0ai` on Pi venv (back to ~400 MB RSS), `MEM0_LOCAL=1` env, `rsync` Qdrant DB back from vllm0528 to `/home/pollen/brain/.qdrant_memory`, restart brain. The legacy `robot_memory_local.py` (renamed from current file) remains in tree exactly for this.
3. Time to rollback: ~10 min for env flip alone; ~30 min including DB rsync.

---

## 10. Memory accounting

| Component | Current Pi RSS | After bridge | Delta |
|---|---|---|---|
| `mem0ai` package + Pydantic models | ~30 MB | 0 | -30 |
| `fastembed` + `onnxruntime` | ~150 MB | 0 | -150 |
| BAAI/bge-small-zh-v1.5 ONNX weights (mmap + warm cache) | ~120 MB | 0 | -120 |
| `qdrant-client` + embedded segments + LMDB | ~80 MB | 0 | -80 |
| HTTP client (urllib stdlib) | 0 | ~2 MB | +2 |
| **Total** | **~380 MB** | **~2 MB** | **~-378 MB** |

Estimate vs the 400 MB target: within 5%. Headroom for the remaining 4 GB Pi budget improves from ~tight to ~comfortable for vLLM-client + Moondream2 client + face tracker concurrent. **Validation gate post-implementation**: `pmap $(pgrep -f robot_brain)` before/after, expect ≥350 MB drop.

---

## 11. Latency budget

| Hop | Current (in-process) | Bridge (HTTP) | Delta |
|---|---|---|---|
| `add_turn` queueing | ~50 µs (executor.submit) | ~50 µs (executor.submit) + bridge POST happens off main thread | 0 perceived |
| `search` (per-turn pre-LLM lookup) | ~80–200 ms (embed + Qdrant lookup CPU on Pi) | ~50–200 ms Tailscale RTT + ~30–80 ms bridge embed (V100 host CPU, faster than Pi) + ~5 ms Qdrant lookup | **~roughly neutral or +50 ms p50, +100 ms p95** |
| `summary` regen | ~3–8 s (LLM call dominates, embedding small) | Same — entirely runs on bridge | 0 |

**Target**: `search` p95 ≤ 350 ms. **Budget guardrail**: if measured p95 > 400 ms post-deploy, fall back to keeping `search` local and only moving `add` / `summary` to bridge (hybrid mode, ~150 MB saved). Decision deferred until measured.

Tailscale RTT measured baseline (Pi ↔ vllm0528): typical 50–150 ms over CHT mobile, 80 ms median p50. Document this in deploy notes.

---

## 12. Open questions / deferred decisions

1. **Embedding upgrade bge-small-zh → bge-m3 (512 → 1024 dim)**: defer. Better recall, but re-embed all elder facts (one-shot script via `Memory.get_all` → re-add). Revisit if elder-care recall complaints surface.
2. **Hybrid mode** (keep `search` local on Pi, only move `add` + `summary`): hold as fallback if measured p95 search latency unacceptable. Saves ~150 MB instead of 380, but worth it for UX.
3. **Bearer token rotation cadence**: TBD ops policy. Current vLLM nginx token rotates manually; bridge inherits same pattern.
4. **Prometheus scrape** (§19 in pre-design questions): bridge exposes `/metrics` (FastAPI + `prometheus-fastapi-instrumentator`). Deferred to a follow-up since current vllm0528 has no scraper running yet.
5. **Multi-user (`user_id` ≠ default)**: schema supports it but production is single-user. Don't build admin UI until needed.
6. **Qdrant-server split**: defer indefinitely; bridge-in-process Qdrant is sufficient at < 10k facts.
7. **TWCC container migration**: if vllm0528 container is moved cross-host by platform, Qdrant on NFS persists. Verify post-migration on first such event.

---

## Implementation checklist (ranked)

1. [ ] **Snapshot current Pi Qdrant** read-only first (`tar -czf` to `/tmp`) before any changes — recoverable baseline.
2. [ ] On vllm0528, create `/home/hctsai1006/venvs/mem0-bridge` venv, install `fastapi uvicorn[standard] mem0ai fastembed qdrant-client`. Verify fastembed downloads bge-small-zh on first import; warm it explicitly with a 1-shot script.
3. [ ] Write `mem0-bridge/app.py` per §5 sketch, with `/health` + `/ready` + `/v1/memory/add|search|summary|summary/flush`. Port `_regenerate_summary` body from legacy `robot_memory.py` verbatim (it's already tested).
4. [ ] Smoke bridge in foreground on vllm0528 (`uvicorn ... --host 127.0.0.1 --port 8003`) with `curl` against all endpoints. Confirm Bearer auth rejects bad tokens.
5. [ ] One-shot migration script `tools/migrate_qdrant_to_bridge.sh` per §7 — dry-run first (skip step 6 brain start), validate `/ready` reports fact count ≥ pre-migration.
6. [ ] Side-by-side smoke: keep Pi brain on legacy `robot_memory.py`, run bridge in parallel pointed at a *copy* of the Qdrant DB. Issue same `search` queries against both, confirm result lists agree top-3.
7. [ ] Rename current `src/robot_memory.py` → `src/robot_memory_local.py` (for rollback path §9). Add deprecation header.
8. [ ] Write new `src/robot_memory.py` per §4 sketch (~50 LOC HTTP client). Preserve public API surface.
9. [ ] Add tests `tests/test_robot_memory_bridge.py`: mock bridge via `responses` or local fastapi TestClient; cover degradation (5xx → empty list), auth failure, timeout, namespace filter passthrough.
10. [ ] Add supervisord stanza per §6 to vllm0528. Verify `supervisorctl status mem0-bridge` = RUNNING after reload.
11. [ ] Update `requirements-brain.txt` — remove `mem0ai`, `fastembed`, `onnxruntime`, `qdrant-client`. Add nothing (urllib is stdlib).
12. [ ] Execute migration during low-usage window (script §7 steps 1–7). Hold Pi brain stopped for ~5 min during scp + bridge start.
13. [ ] Post-deploy: 24h soak watching `vllm-logs/mem0-bridge.log` for `add err` / Qdrant warnings; `pmap $(pgrep robot_brain)` to confirm ≥350 MB RSS reduction.
14. [ ] Measure `search` p50/p95 over Tailscale with 100 representative queries. If p95 > 400 ms, escalate to hybrid mode (defer §12.2).
15. [ ] Update ADR-0006 status `Proposed → Accepted` and add `Outcomes` section with measured numbers + any deviations.
