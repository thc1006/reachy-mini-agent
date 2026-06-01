"""明確語音指令 → 意圖規格（純資料、無副作用）。

業界正解（Home Assistant 兩層架構）：已知短指令用 pattern-match 確定性執行、
其餘長句/對話才丟 LLM。本模組只做「文字 → 意圖規格 dict」的純比對，不執行任何
動作（執行在 robot_brain：do_look / play_emotion / do_action / daemon wake-sleep），
所以可獨立單元測試、不需 import 重量級的 robot_brain。

spec dict 的 kind：
  look    — 看方向：{body_yaw(rad), yaw(deg), pitch(deg)}（基座大幅度 + 頭）
  emotion — 官方情緒 clip：{clip}（play_emotion 動態播 95 個 emotions-library）
  dance   — 跳舞：{name}（generic "dance" 或具體舞名）
  action  — 本地腳本動作：{name} = nod/shake/look_around（do_action 分支）
  wake    — daemon /api/move/play/wake_up
  sleep   — daemon /api/move/play/goto_sleep
  stop    — stop_motion

env：INTENT_SHORTCUT（預設開）、INTENT_SHORTCUT_MAX_LEN（預設 18 字）。
"""
import os
import re

_FALSY = {"0", "false", "no", "off", "", "none"}


def _truthy(v: str) -> bool:
    return (v or "").strip().lower() not in _FALSY


# 看方向的幅度（2026-06-01 實測：body_yaw daemon clamp ~±1.05rad≈60°，取 1.0 安全大幅度）
_LOOK_BODY = 1.0     # rad，基座旋轉（大幅度「轉去看」）
_LOOK_HEAD = 17.0    # deg，頭部 yaw 疊加（更自然）
_LOOK_PITCH = 18.0   # deg，抬頭/低頭幅度

# 規則 = (regex_pattern, spec)。**順序 = 優先序**（具體/明確指令在前，避免被通用詞搶）。
# ⚠ emotion clip 名稱**必須**對得上 daemon 真實清單（19 dances + 81 emotions，
# 2026-06-02 verified）。錯名會 404 靜默失敗（之前 yawn1/sneeze1/happy1/thinking1
# 等 8 個全錯）。可用 clip 見本檔末 _VALID_CLIPS。
_RULES = [
    # ── 停止（最高優先：別動/停下不能被別的搶）──────────────────────────────
    (r"停下|停止|別動|不要動|別跳了|停一下|不要跳|cancel|stop", {"kind": "stop", "ack": "好。"}),
    # ── 喚醒 / 睡覺（明確詞，跟 tired/sleep1 情緒區分）──────────────────────
    (r"起床|醒醒|醒一醒|醒來|起來吧|wake\s*up", {"kind": "wake", "ack": "早安！"}),
    (r"去睡覺|去睡|睡吧|該睡了?|睡一下|go\s*to\s*sleep", {"kind": "sleep", "ack": "晚安。"}),
    # ── 看向特定方位（SDK look_at_world；複合方位要排在單純看左右之前，否則
    #    「看右下」會被「看右」搶走）。x=前/y=左右(左正)/z=上下(上正)，米。──────
    (r"看左上|看向左上|左上方", {"kind": "look_at", "x": 0.5, "y": 0.3, "z": 0.3, "ack": "好。"}),
    (r"看右上|看向右上|右上方", {"kind": "look_at", "x": 0.5, "y": -0.3, "z": 0.3, "ack": "好。"}),
    (r"看左下|看向左下|左下方", {"kind": "look_at", "x": 0.5, "y": 0.3, "z": -0.3, "ack": "好。"}),
    (r"看右下|看向右下|右下方", {"kind": "look_at", "x": 0.5, "y": -0.3, "z": -0.3, "ack": "好。"}),
    (r"看天花板|看上面天|看最上", {"kind": "look_at", "x": 0.3, "y": 0.0, "z": 0.5, "ack": "好。"}),
    (r"看地板|看地上|看最下", {"kind": "look_at", "x": 0.3, "y": 0.0, "z": -0.5, "ack": "好。"}),
    # ── 看方向（body_yaw 大幅度 + head）────────────────────────────────────
    (r"看.{0,2}右|往右|向右|右邊|轉右|look\s*right|turn\s*right",
     {"kind": "look", "body_yaw": _LOOK_BODY, "yaw": _LOOK_HEAD, "pitch": 0.0, "ack": "好。"}),
    (r"看.{0,2}左|往左|向左|左邊|轉左|look\s*left|turn\s*left",
     {"kind": "look", "body_yaw": -_LOOK_BODY, "yaw": -_LOOK_HEAD, "pitch": 0.0, "ack": "好。"}),
    (r"看.{0,2}上|往上|向上|抬頭|抬起頭|look\s*up",
     {"kind": "look", "body_yaw": 0.0, "yaw": 0.0, "pitch": -_LOOK_PITCH, "ack": "好。"}),
    (r"看.{0,2}下|往下|向下|低頭|低下頭|look\s*down",
     {"kind": "look", "body_yaw": 0.0, "yaw": 0.0, "pitch": _LOOK_PITCH, "ack": "好。"}),
    (r"看.{0,2}前|向前|看著我|看我這|正面|回正|回中|看中間|look\s*at\s*me|look\s*forward",
     {"kind": "look", "body_yaw": 0.0, "yaw": 0.0, "pitch": 0.0, "ack": "好。"}),
    # 注意：「轉圈舞」是跳舞、不是 look_around，所以這裡的轉圈不可吃到「舞」。
    (r"轉一圈(?!舞)|轉圈(?!舞)|四處看|看一看|張望|環顧|look\s*around",
     {"kind": "action", "name": "look_around", "ack": None}),
    # ── 天線動作（招牌情緒表達；wiggle=來回擺動）──────────────────────────
    (r"動天線|擺天線|搖天線|天線動|wiggle.*antenna|antenna",
     {"kind": "antenna", "right": 70.0, "left": 70.0, "wiggle": True, "ack": None}),
    (r"豎起天線|天線豎|耳朵豎|perk", {"kind": "antenna", "right": 75.0, "left": 75.0, "wiggle": False, "ack": None}),
    (r"垂下天線|天線垂|耳朵垂|沮喪天線|droop", {"kind": "antenna", "right": -60.0, "left": -60.0, "wiggle": False, "ack": None}),
    # ── 招呼 / 道別（welcoming1/come1/go_away1 真實存在）─────────────────────
    (r"打招呼|歡迎|招呼|welcome|hello", {"kind": "emotion", "clip": "welcoming1", "ack": "嗨！"}),
    (r"過來|過來這|come\s*here", {"kind": "emotion", "clip": "come1", "ack": "來吧。"}),
    (r"走開|走開啦|別煩|go\s*away", {"kind": "emotion", "clip": "go_away1", "ack": None}),
    # ── 正面情緒 ──────────────────────────────────────────────────────────
    (r"開心|高興|快樂|愉快|happy|cheerful", {"kind": "emotion", "clip": "cheerful1", "ack": None}),
    (r"興奮|超興奮|好嗨|excited|enthusiastic", {"kind": "emotion", "clip": "enthusiastic1", "ack": None}),
    (r"大笑|笑一個|哈哈|好好笑|laugh", {"kind": "emotion", "clip": "laughing1", "ack": None}),
    (r"驕傲|得意|自豪|proud", {"kind": "emotion", "clip": "proud1", "ack": None}),
    (r"愛你|喜歡你|我愛你|愛心|love\s*you|loving", {"kind": "emotion", "clip": "loving1", "ack": None}),
    (r"感謝|謝謝你|感激|grateful|thankful", {"kind": "emotion", "clip": "grateful1", "ack": None}),
    (r"放鬆|平靜|冷靜|安心|calm|serenity|relax", {"kind": "emotion", "clip": "serenity1", "ack": None}),
    (r"鬆一口氣|鬆口氣|還好|relief", {"kind": "emotion", "clip": "relief1", "ack": None}),
    (r"成功|做到了?|太棒了?|耶|success", {"kind": "emotion", "clip": "success1", "ack": None}),
    (r"驚奇|哇喔|哇|好神奇|amazed", {"kind": "emotion", "clip": "amazed1", "ack": None}),
    # ── 負面情緒 ──────────────────────────────────────────────────────────
    (r"難過|傷心|不開心|哭|sad", {"kind": "emotion", "clip": "sad1", "ack": None}),
    (r"生氣|憤怒|好氣|火大|angry|furious", {"kind": "emotion", "clip": "furious1", "ack": None}),
    (r"暴怒|氣炸|抓狂|rage", {"kind": "emotion", "clip": "rage1", "ack": None}),
    (r"煩躁|不耐煩|急死|irritated|annoyed", {"kind": "emotion", "clip": "irritated1", "ack": None}),
    (r"沒耐心|快一點啦|impatient", {"kind": "emotion", "clip": "impatient1", "ack": None}),
    (r"沮喪|挫折|無力|frustrat", {"kind": "emotion", "clip": "frustrated1", "ack": None}),
    (r"失望|不滿|displeased|disappoint", {"kind": "emotion", "clip": "displeased1", "ack": None}),
    (r"害怕|好可怕|恐怖|嚇死|scared|afraid|fear", {"kind": "emotion", "clip": "scared1", "ack": None}),
    (r"焦慮|緊張|不安|anxiety|anxious", {"kind": "emotion", "clip": "anxiety1", "ack": None}),
    (r"孤單|寂寞|好孤獨|lonely", {"kind": "emotion", "clip": "lonely1", "ack": None}),
    (r"無聊|好無聊|boring|bored", {"kind": "emotion", "clip": "boredom1", "ack": None}),
    (r"噁心|好噁|disgust", {"kind": "emotion", "clip": "disgusted1", "ack": None}),
    (r"輕蔑|不屑|瞧不起|contempt", {"kind": "emotion", "clip": "contempt1", "ack": None}),
    (r"委屈|認了|算了吧|resigned", {"kind": "emotion", "clip": "resigned1", "ack": None}),
    # ── 中性 / 認知情緒 ───────────────────────────────────────────────────
    (r"驚訝|好驚訝|surprised", {"kind": "emotion", "clip": "surprised1", "ack": None}),
    (r"好奇|curious", {"kind": "emotion", "clip": "curious1", "ack": None}),
    (r"想一下|讓我想|思考一下|思考|沉思|think|thoughtful", {"kind": "emotion", "clip": "thoughtful1", "ack": None}),
    (r"困惑|疑惑|搞不懂|confused", {"kind": "emotion", "clip": "confused1", "ack": None}),
    (r"不確定|不一定|也許吧|uncertain", {"kind": "emotion", "clip": "uncertain1", "ack": None}),
    (r"疑問|想問|請問你|inquir", {"kind": "emotion", "clip": "inquiring1", "ack": None}),
    (r"專心|注意聽|attentive", {"kind": "emotion", "clip": "attentive1", "ack": None}),
    (r"害羞|不好意思|shy", {"kind": "emotion", "clip": "shy1", "ack": None}),
    (r"幫忙|我來幫|helpful|help", {"kind": "emotion", "clip": "helpful1", "ack": None}),
    (r"了解|懂了?|我明白|understand", {"kind": "emotion", "clip": "understanding1", "ack": None}),
    (r"無所謂|不在乎|隨便|indifferent", {"kind": "emotion", "clip": "indifferent1", "ack": None}),
    # ── 疲倦 / 睡眠情緒（跟 sleep 指令區分：這是「表演累」不是真去睡）──────
    (r"好累|累了?$|疲倦|tired|exhausted", {"kind": "emotion", "clip": "tired1", "ack": None}),
    (r"累垮|精疲力盡|累爆", {"kind": "emotion", "clip": "exhausted1", "ack": None}),
    (r"想睡|好睏|打瞌睡|sleepy|drowsy", {"kind": "emotion", "clip": "sleep1", "ack": None}),
    # ── 趣味 / 失誤 ───────────────────────────────────────────────────────
    (r"糟糕|哎呀|oops|糟了", {"kind": "emotion", "clip": "oops1", "ack": None}),
    (r"裝死|演死掉|dying|裝暈", {"kind": "emotion", "clip": "dying1", "ack": None}),
    (r"觸電|電到|electric", {"kind": "emotion", "clip": "electric1", "ack": None}),
    (r"說好|同意|對啊|是的|yes", {"kind": "emotion", "clip": "yes1", "ack": None}),
    (r"說不|不要|不行|拒絕|say\s*no", {"kind": "emotion", "clip": "no1", "ack": None}),
    # ── 點頭 / 搖頭（本地腳本動作，簡單可靠）──────────────────────────────
    (r"點.?頭|點個頭|nod", {"kind": "action", "name": "nod", "ack": None}),
    (r"搖.?頭|搖個頭|shake.*head", {"kind": "action", "name": "shake", "ack": None}),
    # ── 指定舞蹈（19 支官方舞、講出舞名直接跳）────────────────────────────
    (r"搖擺舞|左右搖|side.?to.?side", {"kind": "dance", "name": "side_to_side_sway", "ack": "好！"}),
    (r"轉圈舞|轉圈圈|暈頭|dizzy", {"kind": "dance", "name": "dizzy_spin", "ack": "好！"}),
    (r"小雞舞|啄食|chicken", {"kind": "dance", "name": "chicken_peck", "ack": "好！"}),
    (r"點頭舞|yeah", {"kind": "dance", "name": "yeah_nod", "ack": "好！"}),
    (r"搖滾|groovy|律動", {"kind": "dance", "name": "groovy_sway_and_roll", "ack": "好！"}),
    # ── 通用跳舞（需「舞」字或 dance，放最後當 fallback）──────────────────
    (r"跳.{0,3}舞|跳一支|跳個舞|跳一下舞|跳支舞|跳舞|dance", {"kind": "dance", "name": "dance", "ack": "好，看我跳舞！"}),
]
_COMPILED = [(re.compile(p), s) for p, s in _RULES]

# daemon 真實 clip 清單（2026-06-02 從 /api/move/recorded-move-datasets/list 抓）。
# 用來 self-check 規則表沒用到不存在的名字（之前 yawn1/happy1/thinking1 等 8 個錯名
# 會 404 靜默失敗）。play_emotion 對 emotion clip 直接播；generic "dance" 由
# robot_tools 解析成預設舞，所以 dance kind 的 name="dance" 也算合法。
_VALID_DANCES = {
    "chicken_peck", "chin_lead", "dizzy_spin", "grid_snap", "groovy_sway_and_roll",
    "head_tilt_roll", "interwoven_spirals", "jackson_square", "neck_recoil",
    "pendulum_swing", "polyrhythm_combo", "sharp_side_tilt", "side_glance_flick",
    "side_peekaboo", "side_to_side_sway", "simple_nod", "stumble_and_recover",
    "uh_huh_tilt", "yeah_nod",
}
_VALID_EMOTIONS = {
    "amazed1", "anxiety1", "attentive1", "attentive2", "boredom1", "boredom2",
    "calming1", "cheerful1", "come1", "confused1", "contempt1", "curious1",
    "dance1", "dance2", "dance3", "disgusted1", "displeased1", "displeased2",
    "downcast1", "dying1", "electric1", "enthusiastic1", "enthusiastic2",
    "exhausted1", "fear1", "frustrated1", "furious1", "go_away1", "grateful1",
    "helpful1", "helpful2", "impatient1", "impatient2", "incomprehensible2",
    "indifferent1", "inquiring1", "inquiring2", "inquiring3", "irritated1",
    "irritated2", "laughing1", "laughing2", "lonely1", "lost1", "loving1",
    "no1", "no_excited1", "no_sad1", "oops1", "oops2", "proud1", "proud2",
    "proud3", "rage1", "relief1", "relief2", "reprimand1", "reprimand2",
    "reprimand3", "resigned1", "sad1", "sad2", "scared1", "serenity1", "shy1",
    "sleep1", "success1", "success2", "surprised1", "surprised2", "thoughtful1",
    "thoughtful2", "tired1", "uncertain1", "uncomfortable1", "understanding1",
    "understanding2", "welcoming1", "welcoming2", "yes1", "yes_sad1",
}


def _self_check_clips() -> list:
    """回傳規則表裡 daemon 上不存在的 clip 名（應為空）。emotion → 必須在
    _VALID_EMOTIONS；dance 的具體舞名 → _VALID_DANCES（generic "dance" 例外）。"""
    bad = []
    for _, spec in _RULES:
        k = spec.get("kind")
        if k == "emotion":
            c = spec.get("clip", "")
            if c not in _VALID_EMOTIONS:
                bad.append(("emotion", c))
        elif k == "dance":
            n = spec.get("name", "")
            if n != "dance" and n not in _VALID_DANCES:
                bad.append(("dance", n))
    return bad


# 疑問句標記：emotion/dance 類遇到疑問句不觸發（「你會跳舞嗎」是問題、不該跳）。
# look/stop/wake/sleep 是請求式、即使帶問號也執行（「可以看右邊嗎」要看）。
_QUESTION_RE = re.compile(r"[?？]|嗎|呢|什麼|怎麼|為何|為什麼|可不可以|能不能|會不會|是不是|嘛")


def _max_len() -> int:
    try:
        return int(os.getenv("INTENT_SHORTCUT_MAX_LEN", "18"))
    except (TypeError, ValueError):
        return 18


# emotion/dance 額外用更短的字數上限（去空白），擋掉「我今天去公園散步很開心」這種
# 閒聊夾帶情緒詞的長句被誤觸發；祈使類（look/stop/wake/sleep）不受此限。
_EMO_DANCE_MAX_LEN = int(os.getenv("INTENT_EMO_DANCE_MAX_LEN", "9"))


def match_intent(text: str):
    """文字 → 意圖規格 dict（copy），或 None。保守匹配：只在輸入夠短時攔截，
    避免誤抓長對話；emotion/dance 遇疑問句讓給 LLM。永不 raise。"""
    if not _truthy(os.getenv("INTENT_SHORTCUT", "1")):
        return None
    try:
        t = (text or "").strip()
        if not t or len(t.replace(" ", "")) > _max_len():
            return None
        clen = len(t.replace(" ", ""))
        is_q = bool(_QUESTION_RE.search(t))
        for rx, spec in _COMPILED:
            if rx.search(t):
                kind = spec.get("kind")
                # emotion/dance 詞常夾在閒聊長句裡（「我今天去公園很開心」），
                # 用更嚴的長度上限 + 疑問句一律讓給 LLM；look/stop/wake/sleep/action
                # 是祈使句、放寬到 _max_len。
                if kind in ("emotion", "dance") and (is_q or clen > _EMO_DANCE_MAX_LEN):
                    continue
                return dict(spec)
    except Exception:
        return None
    return None
