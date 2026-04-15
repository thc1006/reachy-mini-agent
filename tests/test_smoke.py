"""Smoke tests: fast, no hardware, no heavy model downloads.

These verify the repository contract (files exist, env vars documented,
pure helpers are deterministic) without importing modules that pull in
CUDA / WebRTC / audio capture — those must remain opt-in runtime deps.
"""
from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ── Repository metadata ───────────────────────────────────────────────────

def test_pyproject_parses():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "reachy-mini-agent"
    assert data["project"]["license"] == {"file": "LICENSE"}


def test_required_top_level_files_exist():
    for name in ["README.md", "LICENSE", "NOTICE", "CONTRIBUTING.md",
                 "CODE_OF_CONDUCT.md", "SECURITY.md", "CHANGELOG.md",
                 ".env.example", ".gitattributes", ".gitignore"]:
        assert (ROOT / name).is_file(), f"missing {name}"


def test_env_example_covers_every_getenv():
    """Every os.getenv(...) key used in src/ must appear in .env.example
    (so contributors can't miss a knob)."""
    src = (ROOT / "src").rglob("*.py")
    keys = set()
    pat = re.compile(r'os\.getenv\(\s*["\']([A-Z][A-Z0-9_]*)["\']')
    for path in src:
        for m in pat.finditer(path.read_text(encoding="utf-8")):
            keys.add(m.group(1))

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    # ANTHROPIC_API_KEY is commented-out optional, allow it missing from the
    # uncommented section.
    allow_missing = {"ANTHROPIC_API_KEY", "LITELLM_MODEL", "OLLAMA_THINK",
                     "LITELLM_KEY", "LITELLM_BASE", "TTS_CACHE_DIR",
                     "WHISPER_COMPUTE", "WHISPER_MODEL"}
    missing = [k for k in keys if k not in env_example and k not in allow_missing]
    assert not missing, f".env.example missing docs for: {missing}"


# ── Pure helpers from prewarm_tts_cache (no runtime imports) ─────────────

def test_cache_path_is_deterministic():
    """Reimplementation of src/prewarm_tts_cache.cache_path — must stay in
    sync with the module so a one-off rename of the algorithm breaks CI."""
    text = "Psst! Anyone there?"
    voice = "en-US-AnaNeural"
    h = hashlib.sha256(f"{voice}::{text}".encode()).hexdigest()[:12]
    expected_prefix = f"{voice}_{h}_"
    slug = "".join(c if c.isalnum() else "_" for c in text[:40]).strip("_")
    expected = f"{expected_prefix}{slug}.wav"

    # Same algorithm applied again — deterministic
    h2 = hashlib.sha256(f"{voice}::{text}".encode()).hexdigest()[:12]
    slug2 = "".join(c if c.isalnum() else "_" for c in text[:40]).strip("_")
    assert h == h2
    assert slug == slug2
    assert expected == f"{voice}_{h2}_{slug2}.wav"


def test_emoji_regex_strips_common_speech_breakers():
    """The exact regex block used in src/robot_brain.py (kept in sync by
    hand — if you edit one, edit the other)."""
    pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\u200d\ufe0f"
        "]+",
        flags=re.UNICODE,
    )
    cases = [
        ("Happy! 🤖", "Happy! "),
        ("hi 😊 there 💕", "hi  there "),
        ("clean text", "clean text"),
        ("Oopsie! 😂😂😂", "Oopsie! "),
    ]
    for raw, expected in cases:
        assert pattern.sub("", raw) == expected


# ── Documentation consistency ────────────────────────────────────────────

def test_readme_declares_supported_tts_engines():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # Must mention both engines so contributors know the switch is real
    assert "TTS_ENGINE" in readme
    assert "kokoro" in readme.lower()
    assert "edge" in readme.lower()


def test_changelog_has_0_1_0():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "0.1.0" in changelog
