"""Positive-example synthesis backends.

POC uses Piper TTS (synth.piper). Production may swap in F5-TTS for
cross-lingual code-switching speaker diversity once the POC pipeline is
proven (see project memory: F5-TTS is the silver bullet for zh-en
code-switch). Keep backends pluggable via a small `synthesize` ABI.
"""

from .piper import synthesize_phrase_set

__all__ = ["synthesize_phrase_set"]
