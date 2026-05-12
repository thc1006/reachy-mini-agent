"""Canonical wake-phrase catalog.

Each entry maps a slug (used in filesystem paths and ONNX head names — must be
ASCII, kebab-safe) to the text passed to the TTS engine and the language tag
the TTS engine should select on.

POC ships with hei_ruiqi (primary CN) and hey_reachy (primary EN). Production
adds short forms and a polite punctuated variant. Multi-head cost on Pi 4 is
~1 ms per added head (Phase B1), so adding heads is cheap; the binding cost is
training data quality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phrase:
    slug: str
    text: str
    lang: str  # ISO 639-1 or Piper voice-prefix matching language
    notes: str = ""


CATALOG: dict[str, Phrase] = {
    "hei_ruiqi": Phrase(
        slug="hei_ruiqi",
        text="嘿瑞奇",
        lang="zh",
        notes="primary CN wake phrase",
    ),
    "hei_ruiqi_polite": Phrase(
        slug="hei_ruiqi_polite",
        text="嘿，瑞奇。",
        lang="zh",
        notes="punctuation variant — prosody differs",
    ),
    "ruiqi": Phrase(
        slug="ruiqi",
        text="瑞奇",
        lang="zh",
        notes="short CN form",
    ),
    "hey_reachy": Phrase(
        slug="hey_reachy",
        text="Hey Reachy",
        lang="en",
        notes="primary EN wake phrase",
    ),
    "reachy": Phrase(
        slug="reachy",
        text="Reachy",
        lang="en",
        notes="short EN form",
    ),
}


def get(slug: str) -> Phrase:
    try:
        return CATALOG[slug]
    except KeyError as e:
        raise KeyError(
            f"unknown phrase slug: {slug!r}; known: {sorted(CATALOG)}"
        ) from e


def resolve(slugs: tuple[str, ...]) -> list[Phrase]:
    return [get(s) for s in slugs]
