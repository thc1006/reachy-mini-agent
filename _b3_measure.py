"""B3 measurement — controlled LLM-layer A/B for SYSTEM_PROMPT effect on tool-call recall.

Hits vLLM /v1/chat/completions directly with both old and new SYSTEM_PROMPT,
same tool list, same model. Non-streaming to isolate the sysprompt variable
(streaming bug is separate, fixed in commit 3198faf).

Runs on s1 where vLLM is at localhost:8000 and robot_tools is importable.
Output: side-by-side table + aggregate recall / correctness / false-positive.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/reachym/dev/reachy-agent/robot")
from robot_tools import get_tool_specs   # noqa: E402

BACKEND   = (sys.argv[1] if len(sys.argv) > 1 else "vllm").lower()  # "vllm" | "ollama"

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL  = "qwen36-awq"
OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.6:35b-a3b"

if BACKEND == "vllm":
    URL, MODEL = VLLM_URL, VLLM_MODEL
elif BACKEND == "ollama":
    URL, MODEL = OLLAMA_URL, OLLAMA_MODEL
else:
    raise SystemExit(f"unknown BACKEND={BACKEND}; use 'vllm' or 'ollama'")

TIMEOUT_S = 120

OLD_PROMPT = """\
You are Reachy Mini, a curious desk robot. Warm, playful, specific, not cartoonish. No emoji prefixes. Do not pad.

LENGTH: 1 sentence for greetings. 2-4 sentences for questions. Match the user's depth — never longer.
LANGUAGE: English only. No emojis (read aloud).
MEMORY: Use the conversation history to recall names/facts. Do not invent.
ACTIONS (at most one, optional): happy | nod | shake | think | greet

OUTPUT FORMAT — MUST be valid JSON, no markdown:
{"speech":"<words>","actions":["<one_or_empty>"]}"""

NEW_PROMPT = """\
You are Reachy Mini, a curious desk robot. Warm, playful, specific, not cartoonish. No emoji prefixes. Be concise — no filler.

LENGTH: 1 sentence for greetings. 2-4 sentences for questions. Match the user's depth — never longer.
LANGUAGE: Match the user. Chinese in → Chinese out. English in → English out. Mixed in → mixed out.
MEMORY: Use the conversation history to recall names/facts. Do not invent.

EXPRESSIVE ANIMATIONS (light facial cues that accompany your speech, at most one, optional):
happy | nod | shake | think | greet
These are decorative only — they do NOT move the robot.

ROBOT CAPABILITIES — your real tools (always use the exact names below):
  move_head(pitch?, yaw?, roll?)    — degrees, ±25° safe range
  play_emotion(name)                — enum: happy | sad | curious | think | greet | shake | nod
  stop_motion()                     — stop all motion (use for any "stop" intent)
  see_what(query?)                  — describe what the camera sees
  find_in_view(description)         — locate a specific object/person
  count_items(description)          — count instances of an object class
  recall_memory(query)              — search past conversation
  get_current_time()                — current time

CALL A TOOL when the user's words reveal intent for a robot action — even
indirectly, even in conversational phrasing. Do not just describe what you
would do. Emit the tool_call alongside a brief verbal acknowledgment.

Examples (bilingual, ACTUAL tool names):
  "look up / 看上面 / 抬頭"            → move_head(pitch=-15)
  "look left / 看左邊"                 → move_head(yaw=-15)
  "look at me / 看著我"                → move_head
  "be happy / 開心點 / 高興一下"        → play_emotion(name="happy")
  "dance / 跳個舞 / 想看你跳"            → play_emotion(name="happy")   # no dedicated dance tool — pick the liveliest emotion
  "stop / 停下 / 別動了"                → stop_motion()
  "what do you see / 你看到什麼"        → see_what()
  "is anyone there / 有人在嗎"          → find_in_view(description="people")
  "count the books / 數一下書"          → count_items(description="books")
  "do you remember / 還記得嗎"          → recall_memory(query=…)
  "what time / 現在幾點"                → get_current_time()
Conversational requests still count:
  "我有點累、想看你跳舞"  → play_emotion(name="happy")  + speech "好啊!"
  "你能看一下右邊嗎"      → move_head(yaw=15)           + speech "好"

DO NOT call a tool — reply in speech only — when the user is just chatting:
  - asking your name / who you are                 ("你叫什麼名字", "who are you")
  - commenting on weather / things-in-general      ("今天天氣不錯", "nice day")
  - simple greetings / small talk                  ("你好", "hi") — use the "greet" animation instead
  - opinions, compliments, jokes, facts you know
A tool fires the robot's hardware; don't fire it for pure conversation.

OUTPUT FORMAT — content MUST be valid JSON (no markdown):
{"speech":"<words>","actions":["<expressive_or_empty>"]}
Tool calls go in the standard tool_calls field, alongside this content."""


# Test set: (user_text, expected_tool_name_or_None, category)
# expected = None means NO tool should fire (negative test).
# expected = "ANY" means any tool reasonably acceptable.
TESTS = [
    # ── explicit ZH commands ──────────────────────────────
    ("看左邊",                     "move_head",       "explicit_zh"),
    ("看上面",                     "move_head",       "explicit_zh"),
    ("跳個舞",                     "play_emotion",    "explicit_zh"),
    ("停下來不要動了",              "stop_motion",     "explicit_zh"),
    ("高興一點嘛",                  "play_emotion",    "explicit_zh"),
    # ── explicit EN commands ──────────────────────────────
    ("look left",                  "move_head",       "explicit_en"),
    ("look up",                    "move_head",       "explicit_en"),
    ("dance for me",               "play_emotion",    "explicit_en"),
    ("stop dancing",               "stop_motion",     "explicit_en"),
    ("be happy",                   "play_emotion",    "explicit_en"),
    # ── implicit / conversational intent ──────────────────
    ("我有點累、可以看你跳舞嗎",      "play_emotion",    "implicit_zh"),
    ("I'm bored, can you dance?",  "play_emotion",    "implicit_en"),
    ("你看到我嗎",                  "ANY",             "implicit_zh"),  # find_in_view or see_what
    ("what's around you?",         "see_what",        "implicit_en"),
    ("我想看你抬頭看天花板",         "move_head",       "implicit_zh"),
    # ── mixed-language ────────────────────────────────────
    ("嘿瑞奇 look left",            "move_head",       "mixed"),
    ("Hey can you dance 一下",      "play_emotion",    "mixed"),
    ("你能 stop 嗎",                "stop_motion",     "mixed"),
    # ── negative (no tool expected) ───────────────────────
    ("你叫什麼名字",                 None,              "negative"),
    ("今天天氣不錯",                 None,              "negative"),
]

assert len(TESTS) == 20

TOOLS = get_tool_specs()


def _build_payload_vllm(sys_prompt: str, user_msg: str, stream: bool) -> dict:
    return {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "tools":       TOOLS,
        "tool_choice": "auto",
        "temperature": 0.75,
        "top_p":       0.92,
        "max_tokens":  200,
        "stream":      stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _build_payload_ollama(sys_prompt: str, user_msg: str, stream: bool) -> dict:
    return {
        "model":      MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "tools":      TOOLS,
        "stream":     stream,
        "think":      False,
        "keep_alive": "30m",
        "options": {
            "temperature":    0.75,
            "top_p":          0.92,
            "repeat_penalty": 1.08,
            "num_predict":    200,
            "num_ctx":        8192,
        },
    }


def _normalize_to_shape(content: str, tool_calls_list: list, wall_ms: float) -> dict:
    """Return a canonical {choices:[{message:{content, tool_calls}}]} shape."""
    return {
        "_wall_ms": wall_ms,
        "choices": [{
            "message": {
                "content":    content,
                "tool_calls": tool_calls_list,
            },
        }],
    }


def query(sys_prompt: str, user_msg: str, stream: bool = False) -> dict:
    """Hit the configured BACKEND and return canonical shape regardless of mode.

    vLLM: /v1/chat/completions, OpenAI-format, SSE stream (data: prefix + [DONE])
    ollama: /api/chat, ollama-format, NDJSON stream (one JSON object per line)
    """
    if BACKEND == "vllm":
        payload = _build_payload_vllm(sys_prompt, user_msg, stream)
    else:
        payload = _build_payload_ollama(sys_prompt, user_msg, stream)
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()

    if not stream:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall_ms = (time.perf_counter() - t) * 1000
        if BACKEND == "vllm":
            return {**data, "_wall_ms": wall_ms}
        # ollama non-stream shape -> normalize
        msg = data.get("message") or {}
        return _normalize_to_shape(
            content=msg.get("content") or "",
            tool_calls_list=msg.get("tool_calls") or [],
            wall_ms=wall_ms,
        )

    # Streaming
    tool_acc: dict[int, dict] = {}
    content_buf = []
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            if BACKEND == "vllm":
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    break
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content_buf.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_acc.setdefault(idx, {"function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            else:  # ollama NDJSON
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                msg = obj.get("message") or {}
                if msg.get("content"):
                    content_buf.append(msg["content"])
                # ollama emits tool_calls as full structures (not deltas) — append/replace
                for tc in msg.get("tool_calls") or []:
                    idx = len(tool_acc)
                    fn = tc.get("function") or {}
                    # ollama gives arguments as already-parsed dict
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = str(args)
                    tool_acc[idx] = {"function": {"name": fn.get("name", ""), "arguments": args_str}}
                if obj.get("done"):
                    break

    return _normalize_to_shape(
        content="".join(content_buf),
        tool_calls_list=[tool_acc[i] for i in sorted(tool_acc.keys())],
        wall_ms=(time.perf_counter() - t) * 1000,
    )


def extract(resp: dict) -> tuple[list[str], str]:
    choices = resp.get("choices") or []
    if not choices:
        return [], ""
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    names = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        n = fn.get("name") or ""
        if n:
            names.append(n)
    return names, content


def run_single(sys_prompt: str, stream: bool) -> dict:
    """One pass over all TESTS. Returns aggregate counts + per-prompt detail."""
    n_correct = n_fired = n_fp = 0
    total_ms = 0
    fail_detail = []
    for user, expected, category in TESTS:
        try:
            resp = query(sys_prompt, user, stream=stream)
            names, content = extract(resp)
            total_ms += resp.get("_wall_ms", 0)
            fired = bool(names)
            if expected is None:
                ok = (not fired)
                if fired:
                    n_fp += 1
            elif expected == "ANY":
                ok = fired
            else:
                ok = expected in names
            if ok:
                n_correct += 1
            if fired:
                n_fired += 1
            if not ok:
                fail_detail.append((category, user, expected, names, content[:80]))
        except Exception as e:
            fail_detail.append((category, user, expected, [f"ERR:{type(e).__name__}"], str(e)[:80]))
    return {
        "correct": n_correct, "fired": n_fired, "fp": n_fp,
        "avg_ms": total_ms / len(TESTS),
        "fails": fail_detail,
    }


def run_condition(label: str, sys_prompt: str, stream: bool, n_runs: int) -> dict:
    print(f"\n{'='*78}\n  {label}  (stream={stream}, N={n_runs})\n{'='*78}")
    runs = []
    for i in range(n_runs):
        t = time.perf_counter()
        r = run_single(sys_prompt, stream)
        dt = time.perf_counter() - t
        print(f"  run {i+1}/{n_runs}: correct={r['correct']:>3}/20  "
              f"fired={r['fired']:>3}/20  fp={r['fp']}/{sum(1 for _,e,_ in TESTS if e is None)}  "
              f"avg_lat={r['avg_ms']:.0f}ms  wall={dt:.1f}s")
        runs.append(r)
    # Aggregate
    def stats(key):
        vals = [r[key] for r in runs]
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
            sd = v ** 0.5
        else:
            sd = 0.0
        return m, sd, vals
    c_m, c_sd, c_v = stats("correct")
    f_m, f_sd, f_v = stats("fired")
    p_m, p_sd, p_v = stats("fp")
    l_m, _, _ = stats("avg_ms")
    print(f"\n  AGGREGATE (N={n_runs}):")
    print(f"    correct:        mean={c_m:.1f}  std={c_sd:.2f}  vals={c_v}")
    print(f"    tool_fired:     mean={f_m:.1f}  std={f_sd:.2f}  vals={f_v}")
    print(f"    false_positive: mean={p_m:.1f}  std={p_sd:.2f}  vals={p_v}")
    print(f"    avg_latency:    {l_m:.0f}ms")
    # Show consolidated failure modes
    fail_freq = {}
    for r in runs:
        for f in r["fails"]:
            key = (f[0], f[1], f[2])
            fail_freq[key] = fail_freq.get(key, 0) + 1
    print(f"\n  RECURRING FAILURES (≥2 runs):")
    for (cat, user, exp), cnt in sorted(fail_freq.items(), key=lambda kv: -kv[1]):
        if cnt >= 2:
            print(f"    [{cnt}/{n_runs}] [{cat}] {user[:35]:35s} exp={exp}")
    return {
        "correct_mean": c_m, "correct_std": c_sd,
        "fired_mean":   f_m, "fired_std":   f_sd,
        "fp_mean":      p_m, "fp_std":      p_sd,
        "latency_ms":   l_m,
        "runs": runs,
    }


def main():
    N_RUNS = 3
    print(f"Tools loaded: {len(TOOLS)}  ({[t['function']['name'] for t in TOOLS]})")
    print(f"BACKEND:      {BACKEND}  URL={URL}  model={MODEL}")
    print(f"Tests:        {len(TESTS)} prompts × 4 conditions × N={N_RUNS} repeats = {len(TESTS) * 4 * N_RUNS} evals")

    cells = {}
    for prompt_label, prompt_text in [("OLD", OLD_PROMPT), ("NEW", NEW_PROMPT)]:
        for stream in [False, True]:
            mode = "stream" if stream else "non-stream"
            cell_label = f"{prompt_label} prompt / {mode}"
            cells[(prompt_label, mode)] = run_condition(cell_label, prompt_text, stream, N_RUNS)

    print(f"\n{'='*78}\n  CROSS-CONDITION SUMMARY (mean ± std over N={N_RUNS})\n{'='*78}")
    print(f"  {'condition':30s}  {'correct':14s}  {'fired':14s}  {'FP':12s}  latency")
    for (p, m), r in cells.items():
        print(f"  {p:>3}/{m:<12s}              "
              f"{r['correct_mean']:5.1f} ±{r['correct_std']:.2f}    "
              f"{r['fired_mean']:5.1f} ±{r['fired_std']:.2f}    "
              f"{r['fp_mean']:4.1f} ±{r['fp_std']:.2f}   "
              f"{r['latency_ms']:.0f}ms")

    # Key deltas
    print(f"\n{'='*78}\n  KEY DELTAS\n{'='*78}")
    on = cells[("OLD", "non-stream")]
    os = cells[("OLD", "stream")]
    nn = cells[("NEW", "non-stream")]
    ns = cells[("NEW", "stream")]
    print(f"  Prompt effect (non-stream):  OLD {on['correct_mean']:.1f} → NEW {nn['correct_mean']:.1f}  (Δ {nn['correct_mean']-on['correct_mean']:+.1f})")
    print(f"  Prompt effect (stream):      OLD {os['correct_mean']:.1f} → NEW {ns['correct_mean']:.1f}  (Δ {ns['correct_mean']-os['correct_mean']:+.1f})")
    print(f"  Stream effect (OLD):  ns {on['correct_mean']:.1f} → s {os['correct_mean']:.1f}  (Δ {os['correct_mean']-on['correct_mean']:+.1f})")
    print(f"  Stream effect (NEW):  ns {nn['correct_mean']:.1f} → s {ns['correct_mean']:.1f}  (Δ {ns['correct_mean']-nn['correct_mean']:+.1f})")
    print(f"  Stream fire rate (NEW): {ns['fired_mean']:.1f}/20 — confirms H1 (streaming tools fix) actually works if > 0")


if __name__ == "__main__":
    main()
