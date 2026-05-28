"""Motion catalog — single source of truth for emotion + dance clip names.

Wave4-P6 (#76). Replaces ad-hoc hardcoded lists in robot_brain.py / robot_tools.py
with a dynamic catalog that:

  1. Loads from a disk cache (~/.cache/reachy_motion_catalog.json) if fresh
     (< CATALOG_TTL_S, default 24h).
  2. Otherwise queries the HuggingFace datasets API ONCE per startup window
     (single HTTP call per dataset, two datasets total — emotions + dances).
  3. Falls back to data/catalog_fallback.json (bundled in repo) if both the
     cache is stale AND HF is unreachable. Network failure is non-fatal —
     the bundled fallback mirrors the published clip list as of 2026-05-29.

Coordinates with #73 (motion queue) by exposing ONLY the catalog/registry
half. The exported ``MOTION_CATALOG`` dict maps every triggerable name
(direct clip names + LLM-friendly aliases + generic dance synonyms) to a
``MotionSpec`` describing how to play it. The queue/dispatch side (#73)
reads MOTION_CATALOG[name] and decides scheduling.

The actual clip JSON / MP4 frame data is fetched LAZILY by the daemon
itself when a play request hits /api/move/play/recorded-move-dataset/...
— we never download MP4s on the brain side, so Pi SD-card writes are
limited to the small cache JSON (~4 KB).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
import urllib.parse as _urlparse
import urllib.request as _urlreq
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Public dataset names (canonical, used in daemon play URLs).
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"

# Cache: refresh from HF if older than this many seconds. 24h default.
CATALOG_TTL_S = int(os.environ.get("REACHY_MOTION_CATALOG_TTL_S", str(24 * 3600)))

# Disk cache location — small JSON, written at most once per CATALOG_TTL_S.
CACHE_PATH = Path(
    os.environ.get(
        "REACHY_MOTION_CATALOG_CACHE",
        str(Path.home() / ".cache" / "reachy_motion_catalog.json"),
    )
)

# Bundled fallback shipped in repo; used when both cache + HF fail.
_REPO_ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = Path(
    os.environ.get(
        "REACHY_MOTION_CATALOG_FALLBACK",
        str(_REPO_ROOT / "data" / "catalog_fallback.json"),
    )
)

HF_API_TIMEOUT_S = 5.0
HF_API_TREE_URL = "https://huggingface.co/api/datasets/{dataset}/tree/main"

# H-M1: backoff window after an HF failure so we don't hammer the API
# during an outage. Starts at HF_FAILURE_BACKOFF_MIN_S, doubles on each
# consecutive failure up to HF_FAILURE_BACKOFF_MAX_S, resets on success.
HF_FAILURE_BACKOFF_MIN_S = 60.0
HF_FAILURE_BACKOFF_MAX_S = 30 * 60.0  # 30 min ceiling


@dataclasses.dataclass(frozen=True)
class MotionSpec:
    """One triggerable motion entry.

    Attributes
    ----------
    name:
        The trigger name as the LLM / brain code emits it (post-lowercase).
    clip:
        The actual clip name on the daemon (== name for direct hits,
        resolved alias for English-friendly aliases).
    dataset:
        Full HF dataset name; combined with clip into the daemon play URL.
    kind:
        "emotion" | "dance" | "alias" | "generic_dance".
    tags:
        Free-form labels (e.g. {"happy", "alias"}). #73 motion queue can use
        these to pick priority / coalesce neighbors. Empty set for now.
    """

    name: str
    clip: str
    dataset: str
    kind: str
    tags: frozenset[str] = dataclasses.field(default_factory=frozenset)

    @property
    def play_path(self) -> str:
        """Daemon REST path fragment, e.g. /api/move/play/recorded-move-dataset/<enc>/<clip>."""
        enc = _urlparse.quote(self.dataset, safe="")
        return f"/api/move/play/recorded-move-dataset/{enc}/{self.clip}"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        log.warning("motion_catalog: failed reading %s: %s", path, e)
        return None


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("motion_catalog: cache write failed (%s): %s", path, e)


def _hf_list_clips(dataset: str, timeout: float = HF_API_TIMEOUT_S) -> list[str]:
    """List clip names (basename without .json) from a HF dataset.

    Returns [] on any failure — caller falls back. Single HTTP GET per call.
    """
    url = HF_API_TREE_URL.format(dataset=dataset)
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "reachy-motion-catalog/1.0"})
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.info("motion_catalog: HF list failed for %s: %s", dataset, e)
        return []
    out: list[str] = []
    if not isinstance(entries, list):
        return []
    for e in entries:
        if not isinstance(e, dict):
            continue
        path = e.get("path") or ""
        if path.endswith(".json"):
            out.append(path[:-5])
    return sorted(out)


def _load_fallback() -> dict:
    data = _read_json(FALLBACK_PATH)
    if data is None:
        # Even the bundled fallback is missing — return a minimal skeleton
        # rather than crash the brain on startup.
        log.error("motion_catalog: fallback missing at %s — using empty skeleton", FALLBACK_PATH)
        return {
            "emotions": [], "dances": [], "aliases": {},
            "generic_dance_synonyms": ["dance"], "default_dance": "yeah_nod",
        }
    return data


def _fresh(cache: dict, now: float) -> bool:
    ts = cache.get("_fetched_at")
    if not isinstance(ts, (int, float)):
        return False
    return (now - float(ts)) < CATALOG_TTL_S


def _fetch_from_hf(fallback: dict) -> tuple[dict, bool]:
    """Try HF; fall back to bundled lists per-dataset on failure.

    Returns (raw_dict, hf_ok). hf_ok is True iff at least one dataset
    returned a non-empty list — the caller uses this to decide whether
    to stamp _fetched_at (H-M1: don't poison the 24h cache with a
    failure result)."""
    hf_emotions = _hf_list_clips(EMOTIONS_DATASET)
    hf_dances = _hf_list_clips(DANCES_DATASET)
    emotions = hf_emotions or list(fallback.get("emotions", []))
    dances = hf_dances or list(fallback.get("dances", []))
    hf_ok = bool(hf_emotions) or bool(hf_dances)
    raw = {
        "emotions": emotions,
        "dances": dances,
        "aliases": dict(fallback.get("aliases", {})),
        "generic_dance_synonyms": list(fallback.get("generic_dance_synonyms", ["dance"])),
        "default_dance": fallback.get("default_dance", "yeah_nod"),
        "_source": "hf+fallback_aliases" if hf_ok else "fallback_only",
    }
    if hf_ok:
        raw["_fetched_at"] = time.time()
    # If HF failed completely we deliberately omit _fetched_at so the
    # next load_catalog() call retries (after the backoff window).
    return raw, hf_ok


def _build_catalog(raw: dict) -> dict[str, MotionSpec]:
    """Materialize the {name: MotionSpec} map from the resolved raw dict.

    Precedence for duplicate names (later overwrites earlier):
      1. emotions clips
      2. dances clips (so 'dance3' from emotions stays an emotion; dance names
         only collide if HF publishes overlapping names, which they don't today)
      3. aliases (LLM-friendly English words → emotion clips)
      4. generic dance synonyms ("dance", "boogie" ...) → default_dance
    """
    out: dict[str, MotionSpec] = {}
    for clip in raw.get("emotions", []):
        if not isinstance(clip, str) or not clip:
            continue
        n = clip.lower()
        out[n] = MotionSpec(name=n, clip=clip, dataset=EMOTIONS_DATASET, kind="emotion")
    for clip in raw.get("dances", []):
        if not isinstance(clip, str) or not clip:
            continue
        n = clip.lower()
        out[n] = MotionSpec(name=n, clip=clip, dataset=DANCES_DATASET, kind="dance")
    aliases = raw.get("aliases", {}) or {}
    emotions_set = {c.lower() for c in raw.get("emotions", []) if isinstance(c, str)}
    for alias, clip in aliases.items():
        if not isinstance(alias, str) or not isinstance(clip, str):
            continue
        if not alias or not clip:
            continue
        if clip.lower() not in emotions_set:
            # Skip aliases whose target clip is no longer published.
            log.debug("motion_catalog: alias %r → %r dropped (clip missing)", alias, clip)
            continue
        out[alias.lower()] = MotionSpec(
            name=alias.lower(), clip=clip, dataset=EMOTIONS_DATASET,
            kind="alias", tags=frozenset({"alias"}),
        )
    default_dance = raw.get("default_dance", "yeah_nod")
    dances_set = {c.lower() for c in raw.get("dances", []) if isinstance(c, str)}
    if default_dance.lower() not in dances_set and dances_set:
        default_dance = sorted(dances_set)[0]
    for syn in raw.get("generic_dance_synonyms", []):
        if not isinstance(syn, str) or not syn:
            continue
        out[syn.lower()] = MotionSpec(
            name=syn.lower(), clip=default_dance, dataset=DANCES_DATASET,
            kind="generic_dance", tags=frozenset({"generic"}),
        )
    return out


_lock = threading.Lock()
_loaded_raw: Optional[dict] = None
# H-M1: backoff state. _hf_failed_at is wall-time of the most recent HF
# failure; _hf_consecutive_failures drives exponential backoff. Both
# reset to 0 / None on the next successful HF call.
_hf_failed_at: float = 0.0
_hf_consecutive_failures: int = 0


def _hf_backoff_active(now: float) -> bool:
    """True iff we should skip the HF call because we recently failed.
    Backoff doubles per consecutive failure, capped at HF_FAILURE_BACKOFF_MAX_S.
    """
    if _hf_failed_at <= 0 or _hf_consecutive_failures <= 0:
        return False
    window = min(
        HF_FAILURE_BACKOFF_MAX_S,
        HF_FAILURE_BACKOFF_MIN_S * (2 ** (_hf_consecutive_failures - 1)),
    )
    return (now - _hf_failed_at) < window


def _load_raw(force_refresh: bool = False) -> dict:
    """Resolve the raw catalog dict: cache → HF (if not backing off) → fallback.

    H-M1: never stamps _fetched_at on HF failure. If HF returns nothing
    for both datasets we keep the prior disk cache's timestamp (so next
    call retries promptly once the backoff window passes) and we DON'T
    rewrite the cache file with a falsified fresh timestamp.
    """
    global _loaded_raw, _hf_failed_at, _hf_consecutive_failures
    with _lock:
        if _loaded_raw is not None and not force_refresh:
            return _loaded_raw
        fallback = _load_fallback()
        now = time.time()
        cache = _read_json(CACHE_PATH)
        if cache is not None and _fresh(cache, now) and not force_refresh:
            _loaded_raw = cache
            return cache

        # If we're inside the backoff window after a recent HF failure,
        # serve from the stale cache (preferred — keeps any prior HF
        # data) or the bundled fallback (last resort). Don't touch HF.
        if _hf_backoff_active(now) and not force_refresh:
            log.info("motion_catalog: HF backoff active "
                     "(consecutive_failures=%d, age=%.0fs); using stale cache/fallback",
                     _hf_consecutive_failures, now - _hf_failed_at)
            if cache is not None:
                _loaded_raw = cache
                return cache
            raw = dict(fallback)
            raw["_source"] = "fallback_only_backoff"
            _loaded_raw = raw
            return raw

        fetched, hf_ok = _fetch_from_hf(fallback)
        if hf_ok:
            # Real fresh data — write through to disk, reset failure counter.
            _write_json(CACHE_PATH, fetched)
            _hf_failed_at = 0.0
            _hf_consecutive_failures = 0
        else:
            # HF failure: record backoff, do NOT stamp the cache as fresh.
            # If a prior cache exists on disk, prefer its timestamp so we
            # don't accidentally re-fetch every call — the backoff guard
            # above already prevents hammering HF, and the in-memory copy
            # below carries us until the next force_refresh or process
            # restart. Crucially we leave CACHE_PATH untouched on failure.
            _hf_failed_at = now
            _hf_consecutive_failures += 1
            log.warning("motion_catalog: HF fetch failed for both datasets "
                        "(consecutive_failures=%d); using fallback, NOT stamping cache",
                        _hf_consecutive_failures)
        _loaded_raw = fetched
        return fetched


def reset_hf_backoff_for_tests() -> None:
    """Drop in-process HF failure state so tests can exercise both
    backoff-active and post-backoff paths deterministically."""
    global _hf_failed_at, _hf_consecutive_failures
    with _lock:
        _hf_failed_at = 0.0
        _hf_consecutive_failures = 0


def load_catalog(force_refresh: bool = False) -> dict[str, MotionSpec]:
    """Return the materialized {name: MotionSpec} map. Idempotent / cached."""
    raw = _load_raw(force_refresh=force_refresh)
    return _build_catalog(raw)


def reset_for_tests() -> None:
    """Drop in-process cache + HF backoff state so tests can re-load
    with fresh mocks."""
    global _loaded_raw, _hf_failed_at, _hf_consecutive_failures
    with _lock:
        _loaded_raw = None
        _hf_failed_at = 0.0
        _hf_consecutive_failures = 0


# Module-level export — robot_brain.py / robot_tools.py import this directly.
# Built eagerly at first import so brain startup gets a single observability
# event (motion_catalog_loaded entries=N) instead of a lazy first-use surprise.
try:
    MOTION_CATALOG: dict[str, MotionSpec] = load_catalog()
    log.info("motion_catalog_loaded entries=%d", len(MOTION_CATALOG))
except Exception as _e:  # never block import
    log.exception("motion_catalog: eager load failed, exporting empty map: %s", _e)
    MOTION_CATALOG = {}
