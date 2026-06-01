# CLAUDE.md — Reachy Mini 長者照護機器人

> Claude Code 每次 session 開頭會自動讀這份。目的：clone 下來後不跑錯檔、不重踩已知雷。

## 這是什麼專案
Reachy Mini Wireless（Raspberry Pi 4）長者照護對話機器人。會講話、看人、做表情動作。
管線：STT(Breeze-ASR-25) → LLM(qwen3.6 / vLLM) → TTS(edge/kokoro) + 視覺(VLM) + 馬達動作。
LLM/STT/TTS/VLM 都跑在遠端 GPU box（vllm0528，透過 Tailscale）；brain 跑在機器人本機 Pi。

## ⚠️ canonical code 在哪（別跑錯檔）
- **正式程式碼 = `src/`**。機器人上跑的是 `/home/pollen/brain/src/`（用 scp 從本機 `src/` 部署）。
- `src/robot_brain.py` — 主程式（狀態機、STT、LLM、TTS、面孔追蹤、對話迴圈）
- `src/robot_tools.py` — LLM 工具（move_head / play_emotion / 視覺工具）
- `src/robot_intents.py` — 語音意圖直通規則表（看/情緒/跳舞/天線/系統指令）
- **`docs/drafts/` = 拋棄式測試暫存，不是程式碼**。已 gitignore。別在這裡找邏輯、別 import。
- 頂層一堆 `_*.py`、`bench*/`、`*.pptx`、`*.pdf`、`chatterbox_eval_out/` 等都是 **gitignore 掉的 local 實驗檔**，clone 不會有，別當真。
- `reachy-mini-intent-runtime/` 是 tracked 子專案（獨立 ADR/tests），跟主 brain 不同層。

## 部署 + 驗證流程（改 brain code 後）
```bash
# 1. 本機 AST 檢查（py 語法）
python -c "import ast; ast.parse(open('src/robot_brain.py',encoding='utf-8').read()); print('AST_OK')"
# 2. scp 到 Pi
scp src/robot_brain.py reachy-mini:/home/pollen/brain/src/
# 3. Pi 上 py_compile + 重啟 + 驗證 active/nrest/無 traceback
ssh reachy-mini 'cd /home/pollen/brain && /home/pollen/brain-venv/bin/python -m py_compile src/robot_brain.py && echo OK'
ssh reachy-mini 'systemctl --user restart reachy-brain.service; sleep 12; systemctl --user is-active reachy-brain.service'
```
- brain 是 **`systemctl --user reachy-brain.service`**（不是 sudo）。
- daemon 是 `systemctl reachy-mini-daemon.service`（馬達控制，帶 `--wake-up-on-start` drop-in）。

## 🔴 踩過的雷（務必遵守，否則重蹈）
1. **Edit 回「String not found」= 沒改到檔**（中文 docstring 常對不上）→ 當場停、重讀、用唯一字串。改完**必跑行為實測**（import + 呼叫看回值），不能只 AST/py_compile（runtime error 過得了 = 假綠燈）。
2. **改完不能只信 AST**：NameError/UnboundLocal 是 runtime。要 `python -c` 實際 import 或 Pi 上跑一次。
3. **馬達單位 = 弧度**：daemon `/api/move/goto` 的 head_pose roll/pitch/yaw 吃 **radians**。move_head 要 `math.radians()` 轉。看左右大幅度要動 `body_yaw`（基座）不只 head yaw。
4. **emotion/dance clip 名必須對得上 daemon 真實清單**（19 dances + 81 emotions）。憑印象寫錯名會 404 靜默失敗。robot_intents.py 有 `_self_check_clips()` 自驗，改規則表後必跑。
5. **vLLM tool-call-parser = qwen3_xml**（不是 hermes，hermes 接不住 Qwen-XML）。
6. **LLM speech 偶爾回空**（vLLM+qwen3+thinking off 的已知行為）→ SYSTEM_PROMPT 有「speech 永不空」指令壓制（實測 85%→100%）。
7. **STT 收音是硬體極限**：單 USB mic，rms<0.05 收不到。講話要靠近 30cm、正常音量。不是 brain bug，軟體 AGC 救不回弱訊號。
8. **commit 規範**：author email `84045975+thc1006@users.noreply.github.com`，**絕不帶任何 AI 署名**（Co-Authored-By / Generated with 等一律禁止）。
9. **回應用台灣繁體中文**。
10. **TWCC/vllm0528 服務只能 cooperative restart**（supervisorctl / SIGTERM），不要 SIGKILL。不要動 5090。

## 環境設定（Pi `/home/pollen/brain/.env`，不在 git）
關鍵 env：`TTS_GAIN`(音量倍率)、`LANG=zh_TW.UTF-8`、`SILENCE_THRESHOLD`、`INTENT_SHORTCUT`、`VOICE_BARGE_IN`(預設關)、`MAX_UTTER_S`。硬體喇叭音量在 ALSA `amixer -c 0 sset PCM <n>%`（重開機會掉回預設）。

## 測試
`pytest`（tests/ 在 git 裡）。改 brain 邏輯後相關 test 要過。
