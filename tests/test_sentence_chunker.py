"""S2 TDD - Part 1: Sentence chunker for streaming LLM output.

The chunker consumes a stream of token deltas and yields complete sentences
as soon as a terminator (. ! ?) is seen. The LAST partial sentence on stream
close is flushed via finalize().

Design:
    stream = SentenceChunker()
    for delta in token_deltas:
        for sentence in stream.feed(delta):
            yield sentence
    for sentence in stream.finalize():
        yield sentence
"""
import pytest

# Expected import — will fail at first run (RED)
from streaming_tts import SentenceChunker


def _chunks(text_deltas, finalize=True):
    """Helper: run chunker over deltas, return list of completed sentences."""
    ch = SentenceChunker()
    out = []
    for d in text_deltas:
        out.extend(ch.feed(d))
    if finalize:
        out.extend(ch.finalize())
    return out


class TestBasicSentences:
    def test_single_complete_sentence(self):
        assert _chunks(["Hello world."]) == ["Hello world."]

    def test_two_sentences_together(self):
        assert _chunks(["Hi there. How are you?"]) == ["Hi there.", "How are you?"]

    def test_sentence_split_across_deltas(self):
        # streaming typical: each delta is a token or two
        assert _chunks(["Hel", "lo ", "wor", "ld.", " Bye!"]) == ["Hello world.", "Bye!"]

    def test_partial_at_end_flushed_on_finalize(self):
        # LLM sometimes forgets final punctuation
        assert _chunks(["Hi there", " pal"]) == ["Hi there pal"]

    def test_partial_at_end_not_flushed_without_finalize(self):
        ch = SentenceChunker()
        out = list(ch.feed("Hi there"))
        assert out == []   # nothing ready


class TestPunctuationVariants:
    def test_exclamation(self):
        assert _chunks(["Wow! Amazing."]) == ["Wow!", "Amazing."]

    def test_question(self):
        assert _chunks(["Really? Yes."]) == ["Really?", "Yes."]

    def test_mixed(self):
        assert _chunks(["A. B! C? D."]) == ["A.", "B!", "C?", "D."]

    def test_ellipsis_not_split_midway(self):
        # "..." should not yield 3 tiny sentences — treat as one terminator
        got = _chunks(["Hmm... I think so."])
        assert got == ["Hmm...", "I think so."]

    def test_multiple_exclamations(self):
        # "Yay!!" keep together, single sentence
        assert _chunks(["Yay!! Alright."]) == ["Yay!!", "Alright."]


class TestEdgeCases:
    def test_empty(self):
        assert _chunks([]) == []

    def test_only_whitespace(self):
        assert _chunks(["   \n  "]) == []

    def test_abbreviation_Mr_does_NOT_split(self):
        # "Mr. Smith left." should be one sentence, not "Mr." + "Smith left."
        got = _chunks(["Mr. Smith left."])
        assert got == ["Mr. Smith left."], f"got {got}"

    def test_decimal_number_does_NOT_split(self):
        # "It costs 3.14 dollars." is one sentence
        got = _chunks(["It costs 3.14 dollars."])
        assert got == ["It costs 3.14 dollars."], f"got {got}"

    def test_sentence_must_have_letters(self):
        # pure punctuation shouldn't yield
        assert _chunks(["...   ..."]) == []

    def test_leading_whitespace_stripped(self):
        assert _chunks(["   Hello."]) == ["Hello."]

    def test_consecutive_spaces_collapsed_ok(self):
        # two spaces between sentences shouldn't create empty output
        assert _chunks(["One.  Two."]) == ["One.", "Two."]


class TestJSONPrefixHandling:
    """robot_brain's LLM returns JSON like {"speech":"...","actions":[]}.
    The chunker will be fed ONLY the speech string (extracted upstream),
    but if somehow JSON-ish prefix leaks through, chunker should handle gracefully."""

    def test_plain_text_no_json(self):
        # Primary use case
        got = _chunks(["I really think that is fine."])
        assert got == ["I really think that is fine."]

    def test_quotes_preserved_in_sentence(self):
        got = _chunks(['She said "hi" to me.'])
        assert got == ['She said "hi" to me.']
