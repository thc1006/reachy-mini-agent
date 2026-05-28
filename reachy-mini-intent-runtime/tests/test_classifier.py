"""Classifier tests.

Ordering contract (checked by test_classify_ordering_contract):
  1. empty / whitespace  → chat
  2. chat_not_command    → chat   (precedes stop so "我喜歡跳舞" never triggers command)
  3. stop_patterns       → critical stop_dance / stop_emotion
  4. hush_patterns       → critical stop_emotion
  5. dance_command       → background dance
  6. emotion_command     → background play_emotion
  7. question marker     → chat
  8. fallback            → chat
"""

import pytest

from reachy_intent_runtime.classifier import RuleIntentClassifier
from reachy_intent_runtime.models import ActionPriority

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_clf = RuleIntentClassifier()


def classify(text: str):
    return _clf.classify(text)


# ---------------------------------------------------------------------------
# Original baseline tests (must stay green)
# ---------------------------------------------------------------------------


def test_dance_command_triggers_background_dance() -> None:
    result = classify("Reachy，跳支舞")
    assert result.kind == "command"
    assert result.action == "dance"
    assert result.priority == ActionPriority.BACKGROUND


def test_preference_about_dance_is_chat_not_command() -> None:
    result = classify("我其實很喜歡跳舞")
    assert result.kind == "chat"
    assert result.action is None


def test_stop_dance_is_critical() -> None:
    result = classify("停止跳舞")
    assert result.kind == "command"
    assert result.action == "stop_dance"
    assert result.priority == ActionPriority.CRITICAL


def test_hush_is_critical_quiet_action() -> None:
    result = classify("噓，小聲一點")
    assert result.kind == "command"
    assert result.action == "stop_emotion"
    assert result.priority == ActionPriority.CRITICAL


# ---------------------------------------------------------------------------
# CRITICAL stop commands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "停止跳舞",  # existing — regression guard
        "不要跳了",  # NEW — 不要.*(跳|動|說) with 了 suffix
        "停下",  # variant — 停下 already in patterns
        "stop dancing",  # English lowercase
        "Stop Dancing",  # mixed case (re.IGNORECASE)
        "STOP",  # uppercase bare stop
    ],
)
def test_critical_stop_dance(utterance: str) -> None:
    r = classify(utterance)
    assert r.kind == "command", f"{utterance!r} → kind={r.kind}"
    assert r.action == "stop_dance", f"{utterance!r} → action={r.action}"
    assert r.priority == ActionPriority.CRITICAL, f"{utterance!r} → priority={r.priority}"


@pytest.mark.parametrize(
    "utterance",
    [
        "噓",  # NEW standalone 噓
        "噓，小聲一點",  # existing — regression guard
        "安靜",  # variant — already in hush patterns
        "be quiet please",  # English variant
    ],
)
def test_critical_stop_emotion(utterance: str) -> None:
    r = classify(utterance)
    assert r.kind == "command", f"{utterance!r} → kind={r.kind}"
    assert r.action == "stop_emotion", f"{utterance!r} → action={r.action}"
    assert r.priority == ActionPriority.CRITICAL, f"{utterance!r} → priority={r.priority}"


# ---------------------------------------------------------------------------
# BACKGROUND dance commands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "跳支舞",  # NEW bare form (no Reachy prefix)
        "Reachy，跳支舞",  # existing — regression guard
        "跳一段舞",  # variant — 段 must be in alternation
        "幫我跳個舞",  # variant — 幫我 prefix
        "dance for me",  # English variant
        "dance now",  # English variant
        "表演一段舞",  # variant — 表演.*舞
    ],
)
def test_background_dance_command(utterance: str) -> None:
    r = classify(utterance)
    assert r.kind == "command", f"{utterance!r} → kind={r.kind}"
    assert r.action == "dance", f"{utterance!r} → action={r.action}"
    assert r.priority == ActionPriority.BACKGROUND, f"{utterance!r} → priority={r.priority}"


# ---------------------------------------------------------------------------
# CHAT — not a command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "我喜歡跳舞",  # NEW bare form (我.*喜歡.*跳舞 with zero chars between)
        "我其實很喜歡跳舞",  # existing — regression guard
        "跳舞很好玩",  # variant — 跳舞.*很好
        "我想學跳舞",  # variant — 我.*想.*學跳舞
        "i like dancing",  # English — i like .*danc
        "dance is fun",  # English — dance.*is fun
        "你會跳舞嗎？",  # question with 嗎 AND contains dance word — chat wins
    ],
)
def test_chat_not_command(utterance: str) -> None:
    r = classify(utterance)
    assert r.kind == "chat", f"{utterance!r} → kind={r.kind}"
    assert r.action is None, f"{utterance!r} → action={r.action}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string_is_chat_high_confidence() -> None:
    r = classify("")
    assert r.kind == "chat"
    assert r.action is None
    assert r.confidence >= 0.99


def test_whitespace_only_is_chat() -> None:
    r = classify("   ")
    assert r.kind == "chat"
    assert r.action is None


def test_bare_question_mark_is_chat() -> None:
    r = classify("?")
    assert r.kind == "chat"
    assert r.action is None


def test_filler_is_chat() -> None:
    r = classify("嗯")
    assert r.kind == "chat"
    assert r.confidence > 0


def test_multilingual_reachy_hush_is_critical_stop_emotion() -> None:
    """'Reachy 噓' mixes English name with Chinese hush — must still be critical."""
    r = classify("Reachy 噓")
    assert r.kind == "command"
    assert r.action == "stop_emotion"
    assert r.priority == ActionPriority.CRITICAL


# ---------------------------------------------------------------------------
# Ordering contract — chat_not_command precedes dance command patterns
# ---------------------------------------------------------------------------


def test_classify_ordering_contract_chat_before_dance() -> None:
    """'你會跳舞嗎？' contains 跳舞 but must resolve to chat, not dance command."""
    r = classify("你會跳舞嗎？")
    assert r.kind == "chat"
    assert r.action is None


# ---------------------------------------------------------------------------
# HIGH-1: Negated-stop must NOT trigger stop_dance (safety-critical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "不要停止跳舞",
        "請別停下",
        "請繼續跳舞，不要停",
        "別停",
        "don't stop dancing",
        "do not stop the dance",
    ],
)
def test_negated_stop_does_not_trigger_stop_dance(utterance: str) -> None:
    """Negated-stop utterances must NEVER produce action=stop_dance."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.action != "stop_dance", f"{utterance!r} wrongly triggered stop_dance"


@pytest.mark.parametrize(
    "utterance",
    [
        "不要停止跳舞",
        "請別停下",
        "請繼續跳舞，不要停",
        "別停",
        "don't stop dancing",
        "do not stop the dance",
    ],
)
def test_negated_stop_routes_to_ambiguous(utterance: str) -> None:
    """Negated-stop utterances must route to ambiguous (LLM-owned per ADR-0002)."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.kind == "ambiguous", f"{utterance!r} → kind={result.kind!r}, expected 'ambiguous'"


# ---------------------------------------------------------------------------
# HIGH-2: Dance refusal must NOT trigger dance command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "我不想跳舞",
        "我才不要跳舞",
        "我不喜歡跳舞",
    ],
)
def test_dance_refusal_is_not_command(utterance: str) -> None:
    """Negated dance desire/preference must never produce action=dance."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.action != "dance", f"{utterance!r} wrongly triggered dance"


# ---------------------------------------------------------------------------
# HIGH-3: Descriptive quiet context must NOT fire command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "我安靜地坐著",
        "圖書館很安靜",
        "他很安靜地離開",
    ],
)
def test_descriptive_quiet_is_not_command(utterance: str) -> None:
    """Descriptive uses of 安靜 must NOT produce kind=command."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.kind != "command", f"{utterance!r} wrongly fired command"


# ---------------------------------------------------------------------------
# MED-2: ambiguous IntentKind — semantic emotion and capability+imperative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "難過一點",
        "笑一下",
        "做個鬼臉",
    ],
)
def test_ambiguous_emotion_routes_to_ambiguous(utterance: str) -> None:
    """Semantic emotion utterances must route to ambiguous (LLM-owned)."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.kind == "ambiguous", f"{utterance!r} → kind={result.kind!r}, expected 'ambiguous'"


def test_capability_plus_imperative_is_ambiguous() -> None:
    """Capability question combined with imperative must route to ambiguous."""
    result = RuleIntentClassifier().classify("你會跳舞嗎，跳給我看")
    assert result.kind == "ambiguous"


# ---------------------------------------------------------------------------
# HIGH-A1: Bare 噓 must NOT fire on descriptive / nominal uses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "我聽到噓聲",
        "觀眾在噓他",
        "他被噓下台",
    ],
)
def test_hush_does_not_fire_on_descriptive_xu(utterance: str) -> None:
    """Descriptive / nominal uses of 噓 must NOT produce kind=command."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.kind != "command", f"{utterance!r} wrongly fired command"


@pytest.mark.parametrize(
    "utterance",
    [
        "噓",
        "噓！",
        "噓，小聲一點",
    ],
)
def test_bare_hush_still_fires_as_imperative(utterance: str) -> None:
    """Bare imperative 噓 must still route to stop_emotion CRITICAL."""
    result = RuleIntentClassifier().classify(utterance)
    assert result.action == "stop_emotion", f"{utterance!r} → action={result.action!r}"
    assert result.priority == ActionPriority.CRITICAL
