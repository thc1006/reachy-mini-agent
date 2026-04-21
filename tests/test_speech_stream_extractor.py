"""S2 TDD - Part 3: Extract "speech" field value from streaming JSON.

LLM outputs {"speech":"Hello there!","actions":["nod"]} in token-by-token
stream. We need to detect the speech value span and yield chars as they arrive,
BEFORE the full JSON is closed. Actions metadata is extracted after the stream
ends via normal JSON parse of the full buffer.
"""
import pytest
from streaming_tts import SpeechStreamExtractor


def _feed_all(deltas):
    ex = SpeechStreamExtractor()
    out = []
    for d in deltas:
        chunk = ex.feed(d)
        if chunk:
            out.append(chunk)
    return "".join(out), ex


class TestBasic:
    def test_full_response_at_once(self):
        out, _ = _feed_all(['{"speech":"Hello there!","actions":["nod"]}'])
        assert out == "Hello there!"

    def test_token_by_token(self):
        # simulate token streaming (1-2 chars each)
        full = '{"speech":"Hi pal.","actions":[]}'
        out, _ = _feed_all([full[i:i+2] for i in range(0, len(full), 2)])
        assert out == "Hi pal."

    def test_actions_field_first(self):
        # LLM sometimes flips order
        out, _ = _feed_all(['{"actions":["happy"],"speech":"Yay!"}'])
        assert out == "Yay!"

    def test_with_leading_whitespace(self):
        out, _ = _feed_all(['  \n {"speech":"Hi.","actions":[]}  '])
        assert out == "Hi."


class TestEscapes:
    def test_escaped_quote_inside_speech(self):
        # LLM might write: "speech":"She said \"hi\" to me"
        raw = r'{"speech":"She said \"hi\" to me.","actions":[]}'
        out, _ = _feed_all([raw])
        assert out == 'She said "hi" to me.'

    def test_escaped_newline(self):
        raw = r'{"speech":"Line one.\nLine two.","actions":[]}'
        out, _ = _feed_all([raw])
        assert out == "Line one.\nLine two."

    def test_escaped_backslash(self):
        raw = r'{"speech":"C:\\path","actions":[]}'
        out, _ = _feed_all([raw])
        assert out == r"C:\path"


class TestMarkdownFence:
    def test_with_json_code_fence(self):
        # model ignores "no markdown" instruction
        raw = '```json\n{"speech":"Hmm OK.","actions":[]}\n```'
        out, _ = _feed_all([raw])
        assert out == "Hmm OK."


class TestEdgeCases:
    def test_no_speech_field(self):
        out, _ = _feed_all(['{"actions":["shake"]}'])
        assert out == ""

    def test_empty_speech(self):
        out, _ = _feed_all(['{"speech":"","actions":[]}'])
        assert out == ""

    def test_partial_stream_no_end(self):
        # stream cuts off mid-speech (e.g. num_predict exhausted)
        out, _ = _feed_all(['{"speech":"Hello wor'])
        assert out == "Hello wor"   # emit what we have

    def test_only_opening_brace(self):
        out, _ = _feed_all(['{'])
        assert out == ""


class TestBoundarySplit:
    """The 'speech":"' marker is split across deltas — must still find it."""
    def test_marker_split_across_deltas(self):
        out, _ = _feed_all(['{"spee', 'ch":"', 'Hello."}'])
        assert out == "Hello."

    def test_closing_quote_in_own_delta(self):
        out, _ = _feed_all(['{"speech":"Hello', '"', ',"actions":[]}'])
        assert out == "Hello"
