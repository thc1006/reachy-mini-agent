from __future__ import annotations

import re

from .models import ActionPriority, IntentResult


class RuleIntentClassifier:
    """Small deterministic guardrail for commands that must not wait for an LLM.

    Check order inside classify():
      1. empty / whitespace               → chat  (fast exit, confidence 0.99)
      2. capability_plus_imperative       → ambiguous  (e.g. "你會跳舞嗎，跳給我看";
                                                  must precede chat_not_command so the
                                                  combined form is not swallowed as chat)
      3. chat_not_command                 → chat  (precedes stop/dance so topic mentions
                                                  like "我不想跳舞" never fire a command)
      4. negation_stop_patterns           → ambiguous  (negated-stop, e.g. "不要停止跳舞";
                                                  must run BEFORE stop_patterns so the
                                                  negated form is intercepted first)
      5. stop_patterns                    → critical command  stop_dance / stop_emotion
      6. hush_patterns                    → critical command  stop_emotion  (imperative-
                                                  anchored; descriptive 安靜 and
                                                  nominal/descriptive 噓聲 excluded)
      7. dance_command_patterns           → background command  dance
      8. emotion_command_patterns         → background command  play_emotion
      9. ambiguous_emotion_patterns       → ambiguous  (semantic emotion; LLM owns mapping)
     10. question markers                 → chat
     11. fallback                         → chat  (confidence 0.55)
    """

    # ------------------------------------------------------------------
    # Pattern sets — class-level constants so they are compiled once and
    # shared across instances.
    # ------------------------------------------------------------------

    # Step 3: negated-stop guard — must precede _stop_patterns.
    # Any utterance matching here is ambiguous (defer to LLM context).
    _negation_stop_patterns = [
        # Chinese negation before stop
        r"不要.*停",
        r"別.*停",
        r"請繼續",
        r"繼續(跳|動)",
        # English negation before stop
        r"don'?t\s+stop",
        r"do\s+not\s+stop",
        r"please\s+continue",
    ]

    _stop_patterns = [
        # Chinese stop/cancel imperatives
        r"停止",
        r"停下",
        r"不要.*(跳|動|說)",
        r"先不要",
        r"別.*(跳|動|說)",
        # English stop/cancel imperatives  (IGNORECASE applied in _matches_any)
        r"stop( dancing| dance| moving)?",
        r"pause",
        r"cancel",
        r"abort",
    ]

    # Imperative-anchored hush patterns.
    # Bare single-interjection 噓 and clear imperative phrases are kept.
    # Descriptive uses like "圖書館很安靜" / "我安靜地坐著" / "我聽到噓聲" must NOT match.
    _hush_patterns = [
        # 噓 anchored: must stand alone or be surrounded by whitespace / punctuation.
        # "噓聲"、"觀眾在噓他"、"被噓下台" must NOT fire.
        # Whole-string bare "噓" → (^|…)噓(…|$) with ^ at start and $ at end.
        # "噓，小聲一點" → 噓 is followed by ，which is in the right-side class.
        r"(^|[，。！？\s])噓([，！。\s]|$)",
        r"小聲",
        # 安靜 anchored: must be at start / after punctuation / or followed by
        # punctuation / end-of-string.  "很安靜" and "安靜地" must NOT match.
        r"(^|[，。！？\s！])安靜([！。\s]|$)",
        # English equivalents — require imperative context
        r"(^|\s)hush(\s|$)",
        r"be\s+quiet",
        r"(^|\s)quiet(\s|[！。,]|$)",
    ]

    _dance_command_patterns = [
        # Chinese dance requests — bare measure words (支/個/段) and compound
        # measure phrases (一段/一支/一個) are all accepted; the alternation must
        # include two-character compounds so "跳一段舞" matches correctly.
        r"(幫我)?跳(一|支|個|段|一段|一支|一個)?舞",
        r"跳.*dance",
        # English dance requests
        r"dance( for me| now| please)?",
        # Performance framing
        r"表演.*舞",
    ]

    _emotion_command_patterns = [
        r"開心一點",
        r"高興一點",
        r"裝可愛",
        r"賣萌",
        r"express .*emotion",
        r"show .*emotion",
    ]

    # Step 3b (before chat_not_command): capability question combined with imperative.
    # "你會跳舞嗎，跳給我看" contains both a 嗎-question AND a dance imperative —
    # the combined form is ambiguous and must be caught before chat_not_command
    # would swallow the whole utterance as chat.
    _capability_plus_imperative_patterns = [
        r"[嗎？?].*[，,].*跳給",
        r"你會.*嗎.*跳給",
    ]

    # Step 8: ambiguous semantic emotion — LLM-owned.
    _ambiguous_emotion_patterns = [
        # Semantic emotion modifiers without explicit emotion name
        r"難過一點",
        r"笑一下",
        r"做個鬼臉",
    ]

    _chat_not_command_patterns = [
        # Statements of personal preference or opinion about dancing
        r"我.*喜歡.*跳舞",
        r"我.*想.*學跳舞",
        r"跳舞.*很好",
        # Negated dance desire/preference — refusal, NOT a command to dance
        r"我.*(不|才不).*(想|要|喜歡).*跳舞",
        # Questions about dancing / the robot's capability — must stay chat
        r"你.*跳舞.*[嗎？?]",
        r"會.*跳舞.*[嗎？?]",
        # English counterparts
        r"dance.*is fun",
        r"i like .*danc",
    ]

    # ------------------------------------------------------------------

    def classify(self, utterance: str) -> IntentResult:
        """Classify *utterance* into an IntentResult using deterministic rules."""
        text = utterance.strip().lower()

        # 1. Empty / whitespace — no action.
        if not text:
            return IntentResult("chat", None, None, 0.99, "空白輸入，不執行動作")

        # 2. Capability-question + imperative combo — ambiguous, must precede
        #    chat_not_command so "你會跳舞嗎，跳給我看" is not swallowed as chat.
        if self._matches_any(text, self._capability_plus_imperative_patterns):
            return IntentResult(
                "ambiguous", None, None, 0.68, "能力詢問加指令複合語境，交由LLM判斷"
            )

        # 3. Chat-not-command guard — topic mentions that look like commands.
        if self._matches_any(text, self._chat_not_command_patterns):
            return IntentResult("chat", None, None, 0.86, "描述偏好或話題，並非要求機器人執行")

        # 4. Negated-stop guard — intercept before stop_patterns fire.
        #    "不要停止跳舞" must not become CRITICAL stop_dance.
        if self._matches_any(text, self._negation_stop_patterns):
            return IntentResult("ambiguous", None, None, 0.72, "否定停止語境，交由LLM判斷")

        # 5. Stop / cancel imperatives (CRITICAL).
        if self._matches_any(text, self._stop_patterns):
            action = "stop_emotion" if "emotion" in text or "表情" in text else "stop_dance"
            return IntentResult(
                "command", action, ActionPriority.CRITICAL, 0.98, "偵測到停止/取消指令"
            )

        # 6. Hush / quiet imperatives (CRITICAL) — imperative-anchored only.
        if self._matches_any(text, self._hush_patterns):
            return IntentResult(
                "command", "stop_emotion", ActionPriority.CRITICAL, 0.93, "偵測到安靜/降低干擾指令"
            )

        # 7. Dance requests (BACKGROUND).
        if self._matches_any(text, self._dance_command_patterns):
            return IntentResult(
                "command", "dance", ActionPriority.BACKGROUND, 0.90, "偵測到跳舞請求"
            )

        # 8. Emotion-expression requests (BACKGROUND).
        if self._matches_any(text, self._emotion_command_patterns):
            return IntentResult(
                "command", "play_emotion", ActionPriority.BACKGROUND, 0.82, "偵測到情緒表達請求"
            )

        # 9. Ambiguous semantic emotion — LLM-owned.
        if self._matches_any(text, self._ambiguous_emotion_patterns):
            return IntentResult("ambiguous", None, None, 0.65, "語意情緒，交由LLM判斷")

        # 10. Question markers → conversational chat.
        if any(
            marker in text for marker in ["?", "？", "嗎", "呢", "為什麼", "how", "what", "why"]
        ):
            return IntentResult("chat", None, None, 0.78, "疑問句或聊天內容")

        # 11. Fallback.
        return IntentResult("chat", None, None, 0.55, "未偵測到明確動作指令")

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        """Return True if *text* matches any pattern (case-insensitive)."""
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
