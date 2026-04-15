"""Pre-warm edge-tts cache:
- all static phrases from robot_brain.py
- top-N most frequent past robot speeches from conversation_log.jsonl
"""
import asyncio, io, json, hashlib
from collections import Counter
from pathlib import Path
import soundfile as sf
from edge_tts import Communicate

VOICE = "en-US-AnaNeural"
HERE = Path(__file__).parent
CACHE_DIR = HERE / "tts_cache"
CONV_LOG = HERE / "conversation_log.jsonl"
TOP_N_CONVO = 40   # top-N 最常出現的過去 robot 話語
MIN_REPEAT = 2     # 至少重複 N 次才值得 cache
CACHE_DIR.mkdir(exist_ok=True)

STATIC_PHRASES = [
    "System online! I'll say hi to anyone who comes by!",
    "No worries! Come chat with me anytime!",
    "Byeee! Catch you later!",
    "Whoa! You scared me getting so close!",
    "Hey hey hey, personal space please, hehe!",
    "Eep! Too close! But uh... hi there!",
    "Hiii! I've been waiting for someone to chat with, yay!",
    "Ooh a human! Hi hi hi! I'm Reachy Mini, nice to meet you!",
    "Hey hey! I totally noticed you first, hehe!",
    "Hey! Yeah you over there! Come chat with me!",
    "Hiiii! You're so far, come closer pleeease!",
    "Psst! Anyone there? Come say hi!",
    "Nobody's here... sooo boring.",
    "Hmmm, I wonder what's happening today.",
    "Hellooo? Anyone wanna play with me?",
    "Careful now, don't step on anything cute!",
    "Ooh! Number one! You're my number one!",
    "One finger! Yep, I see it!",
    "Peace and love! Hehe!",
    "Two! Like a V for victory!",
    "Three! Wow, three fingers!",
    "Three little piggies, yay!",
    "Four! Almost a high five!",
    "Four fingers! Nice hand you got there!",
    "High five! Yeaah!",
    "Woohoo! Five! Gimme five!",
    "Six! You sneaky with two hands!",
    "Six fingers, ooh fancy!",
    "Seven! Lucky number!",
    "Seven fingers, that's magical!",
    "Eight! Octopus vibes!",
    "Nine! Just one more!",
    "Nine fingers, so close to ten!",
    "Ten! You got all of them out, amazing!",
]

def collect_past_robot_speeches():
    if not CONV_LOG.exists():
        return []
    counter = Counter()
    for line in CONV_LOG.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            s = (r.get("robot") or "").strip()
            if not s:
                continue
            # 去掉 emoji
            import re
            EMO = re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
                             "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF"
                             "\U00002700-\U000027BF" "\U0001F900-\U0001F9FF"
                             "\U0001FA00-\U0001FAFF" "\U00002600-\U000026FF"
                             "\u200d\ufe0f" "]+")
            s = EMO.sub("", s).strip()
            if 3 <= len(s) <= 200:
                counter[s] += 1
        except Exception:
            continue
    top = [(t, n) for t, n in counter.most_common(TOP_N_CONVO) if n >= MIN_REPEAT]
    return top

def cache_path(text, voice=VOICE):
    h = hashlib.sha256(f"{voice}::{text}".encode()).hexdigest()[:12]
    slug = "".join(c if c.isalnum() else "_" for c in text[:40]).strip("_")
    return CACHE_DIR / f"{voice}_{h}_{slug}.wav"

async def warm_one(text):
    cp = cache_path(text)
    if cp.exists():
        return "SKIP", cp.stat().st_size
    # 跳過純標點 / 太短 / 無字母句
    if not any(c.isalnum() for c in text):
        return "SKIP-junk", 0
    try:
        buf = io.BytesIO()
        async for c in Communicate(text, voice=VOICE).stream():
            if c["type"] == "audio":
                buf.write(c["data"])
        if buf.tell() == 0:
            return "FAIL-empty", 0
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        sf.write(str(cp), data, sr, format="WAV")
        return "NEW", cp.stat().st_size
    except Exception as e:
        return f"FAIL:{type(e).__name__}", 0

async def main():
    convo = collect_past_robot_speeches()
    all_items = [(p, 0) for p in STATIC_PHRASES] + convo
    seen = set()
    uniq = []
    for t, n in all_items:
        if t in seen:
            continue
        seen.add(t)
        uniq.append((t, n))
    print(f"Will pre-warm {len(uniq)} phrases ({len(STATIC_PHRASES)} static + {len(convo)} from convo log)")
    new_n = skip_n = bytes_n = 0
    fail_n = 0
    for i, (text, freq) in enumerate(uniq, 1):
        status, size = await warm_one(text)
        bytes_n += size
        if status == "NEW":
            new_n += 1
        elif status.startswith("SKIP"):
            skip_n += 1
        else:
            fail_n += 1
        tag = f"×{freq}" if freq else "static"
        print(f"[{i}/{len(uniq)}] {status:12} {tag:>6} {size:>7}B  {text[:60]}")
    print(f"\n== Done. NEW={new_n} SKIP={skip_n} FAIL={fail_n} total_size={bytes_n/1024:.0f}KB dir={CACHE_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
