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

VLLM_URL  = "http://localhost:8000/v1/chat/completions"
MODEL     = "qwen36-awq"
TIMEOUT_S = 90

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


def query(sys_prompt: str, user_msg: str) -> dict:
    payload = {
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
        "stream":      False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["_wall_ms"] = (time.perf_counter() - t) * 1000
    return data


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


def run_condition(label: str, sys_prompt: str) -> dict:
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    rows = []
    n_correct = 0
    n_fired = 0
    n_fp = 0
    n_pos = sum(1 for _, e, _ in TESTS if e is not None)
    n_neg = sum(1 for _, e, _ in TESTS if e is None)
    total_ms = 0
    for user, expected, category in TESTS:
        try:
            resp = query(sys_prompt, user)
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
            rows.append((category, user, expected, names, content[:60], ok))
        except Exception as e:
            rows.append((category, user, expected, [f"ERR:{type(e).__name__}"], str(e)[:60], False))

    # Print table
    for cat, user, expected, got, content, ok in rows:
        mark = "✓" if ok else "✗"
        exp = expected or "(none)"
        got_s = ",".join(got) if got else "(no tool)"
        print(f"  {mark} [{cat:11s}] {user[:30]:30s} exp={exp:15s} got={got_s:25s}  | {content!r}")

    print(f"\n  CORRECT:        {n_correct}/20")
    print(f"  TOOL_FIRED:     {n_fired}/20")
    print(f"  FALSE_POSITIVE: {n_fp}/{n_neg}")
    print(f"  AVG LATENCY:    {total_ms/20:.0f}ms")
    return {"correct": n_correct, "fired": n_fired, "fp": n_fp, "rows": rows}


def main():
    print(f"Tools loaded: {len(TOOLS)}  ({[t['function']['name'] for t in TOOLS]})")
    print(f"vLLM:         {VLLM_URL}  model={MODEL}")
    old = run_condition("OLD prompt (master)", OLD_PROMPT)
    new = run_condition("NEW prompt (B3)",     NEW_PROMPT)
    print(f"\n{'='*78}\n  DELTA")
    print(f"{'='*78}")
    print(f"  correct:        {old['correct']:>3} → {new['correct']:>3}   (Δ {new['correct']-old['correct']:+d})")
    print(f"  tool_fired:     {old['fired']:>3} → {new['fired']:>3}   (Δ {new['fired']-old['fired']:+d})")
    print(f"  false_positive: {old['fp']:>3} → {new['fp']:>3}   (Δ {new['fp']-old['fp']:+d})")


if __name__ == "__main__":
    main()
