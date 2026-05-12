"""Phrase catalog invariants."""

from __future__ import annotations

import pytest

from wake_train import phrases as ph


def test_primary_slugs_present():
    assert "hei_ruiqi" in ph.CATALOG
    assert "hey_reachy" in ph.CATALOG


def test_slugs_are_ascii_safe():
    """Slugs end up on the filesystem and as ONNX head names — must be ASCII."""
    for slug in ph.CATALOG:
        slug.encode("ascii")  # raises if any non-ASCII char


def test_text_is_human_readable():
    assert ph.CATALOG["hei_ruiqi"].text == "嘿瑞奇"
    assert ph.CATALOG["hey_reachy"].text == "Hey Reachy"


def test_resolve_preserves_order():
    out = ph.resolve(("hey_reachy", "hei_ruiqi"))
    assert [p.slug for p in out] == ["hey_reachy", "hei_ruiqi"]


def test_resolve_raises_on_unknown():
    with pytest.raises(KeyError, match="unknown phrase slug"):
        ph.resolve(("nope",))
