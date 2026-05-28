# Reachy Mini Intent Runtime

可中斷、可排程、可測試的 Reachy Mini 官方 dances/emotions 自然語言調用骨架。

本專案不是要重寫 Pollen Robotics / Hugging Face 的 Reachy Mini SDK，而是提供一個可以交給 Claude Code 繼續開發的工程骨架：

1. 將「一般聊天」與「動作指令」分流。
2. 將「跳舞 / emotion / head move」轉成可排程 action。
3. 用 priority queue + cooperative chunking 讓 `stop_dance`、`stop_emotion`、噓/停止手勢等高優先級事件能打斷長動作。
4. 保留 CPU 給語音、camera sampling、VLM/LLM tool routing，不讓 Reachy Mini 在跳舞時完全聽不到使用者。
5. 用 ADR + SDD + TDD + hooks 讓 Claude Code 開發時遵守軟體工程流程。

## 最新外部事實基線（2026-05-28）

- 官方 Reachy Mini conversation app 的 README 描述了 realtime voice、vision pipeline、layered motion system、async tool dispatch。
- 官方 app 已暴露 `dance`、`stop_dance`、`play_emotion`、`stop_emotion` 等 LLM tools。
- 官方 app 支援 external profiles/tools，可透過 `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` 與 `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` 載入外部工具。
- `--local-vision` 不建議直接跑在 Reachy Mini Wireless / Raspberry Pi 上；local vision 應放在 laptop/workstation，robot daemon 留在 robot 上。
- 官方 Python SDK 可用 `ReachyMini()` 自動偵測 localhost/network 連線，支援 `goto_target`、`set_target`、`play_move` 等動作控制。

更多來源摘要見 `docs/research/2026-05-28-official-stack.md`。

## 快速開始

```bash
cd reachy-mini-intent-runtime
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'
./scripts/verify.sh
python -m reachy_intent_runtime.demo --script demo/hospital_interrupt_scenario.json
```

沒有實體 Reachy Mini 時，預設使用 `MockMotionAdapter`，可以先驗證 intent router、scheduler、interrupt behavior 與測試。

## 與官方 Reachy Mini conversation app 整合

先安裝官方 app，建議使用官方 README 的 `uv` 流程：

```bash
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app.git
cd reachy_mini_conversation_app
uv venv --python python3.12 .venv
source .venv/bin/activate
uv sync --group dev
uv sync --extra mediapipe_vision --group dev
```

把本專案的 external content 指向官方 app：

```bash
export REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=/absolute/path/to/reachy-mini-intent-runtime/external_content/external_profiles
export REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=/absolute/path/to/reachy-mini-intent-runtime/external_content/external_tools
export REACHY_MINI_CUSTOM_PROFILE=hospital_assistant
reachy-mini-conversation-app --head-tracker mediapipe
```

若要跑本機 VLM，請依官方提醒不要直接在 Reachy Mini Wireless / Raspberry Pi 上開 `--local-vision`；把 local vision 移到 laptop/workstation，robot 上保留 daemon 與輕量 action runtime。

## 建議開發節奏

採敏捷 Sprint + SDD + TDD：

- Sprint 0：跑通 mock scheduler、classifier、verify hook。
- Sprint 1：接官方 conversation app external profile/tool。
- Sprint 2：盤點官方 dances/emotions library，建立 action manifest。
- Sprint 3：實測 `dance` / `stop_dance` 中斷延遲與 gap smoothness。
- Sprint 4：CPU resource budget、head tracking、camera sampling 與 speech loop coexistence。
- Sprint 5：病房小幫手 use case demo：入院 orientation、點滴快結束安撫、請護士、跳舞中停止。

## Pi / CM4 runtime QoS (3-tier strategy)

Reachy Mini Wireless 跑在 Raspberry Pi CM4（4 核 / 4GB / 16GB flash）。Phase 6
為了避免「跳舞時聽不到 stop」、「VLM 把 CPU 吃光」這類情境，定義了一套只用
**systemd + cgroup v2 + POSIX nice** 的 3-tier QoS 策略（不改 kernel、不上
sched_ext、realtime SCHED_FIFO/RR 預設關閉、需 benchmark 才開）。

完整論述見 [`docs/adr/0004-pi-cpu-qos-and-runtime-scheduling.md`](docs/adr/0004-pi-cpu-qos-and-runtime-scheduling.md)
與測試契約 [`docs/sdd/04-pi-runtime-qos.md`](docs/sdd/04-pi-runtime-qos.md)。

### Runtime 元件切分

| 元件 | Tier | CPUWeight | CPUQuota | MemoryMax | Nice | 備註 |
|---|---|---:|---:|---:|---:|---|
| `reachy-audio-listener` | 1 sense | 900 | 80% | 64M | -5 | 最高權重，聾掉=安全風險 |
| `reachy-orchestrator` | 2 decide | 600 | 80% | 128M | -3 | classifier + scheduler |
| `reachy-motion-worker` | 2 act | 300 | 120% | 384M | 0 | 跳舞 / emotion 分段執行 |
| `reachy-camera-sampler` | 3 observe | 150 | 60% | 128M | 5 | 壓力下可掉 frame |
| `reachy-llm-vlm-client` | 3 assist | 100 | 40% | 64M | 10 | 對外 API client、不跑本機推論 |

所有服務都掛在 `reachy-runtime.slice`（CPUAccounting / MemoryAccounting /
IOAccounting 全開、總 CPUQuota=380%、MemoryMax=2G）。

> ⚠️ The shipped `reachy-llm-vlm-client.service` is a placeholder that emits
> heartbeat logs only (no real off-board API client). Wire your off-board
> LLM/VLM client by editing its `ExecStart=` before production deployment.

### 安裝 / 移除 systemd units

```bash
# dry-run（預設）— 印出會做什麼、不動檔案
bash scripts/install_systemd_units.sh --dry-run

# 真正安裝到 /etc/systemd/system（需 sudo）
sudo bash scripts/install_systemd_units.sh --install

# 移除
sudo bash scripts/install_systemd_units.sh --uninstall
```

### Bench / stress（mock-by-default、無硬體可跑）

```bash
bash scripts/bench_pi_runtime.sh                # mock 模擬 stop-dispatch latency
bash scripts/stress_cpu_and_test_stop.sh         # 4 個 CPU-bound worker + stop latency
```

`--real-hardware` 旗標保留給未來 Phase 7 接 systemd journal scraping，現階段
顯式回傳 exit 2 作為 gate。

## 專案目錄

```text
.
├── AGENTS.md
├── CLAUDE.md
├── .claude/
│   ├── settings.json
│   ├── hooks/task-complete-quality-gate.sh
│   └── agents/
├── docs/
│   ├── adr/
│   ├── agile/
│   ├── prompts/
│   ├── research/
│   └── sdd/
├── external_content/
│   ├── external_profiles/hospital_assistant/
│   └── external_tools/
├── src/reachy_intent_runtime/
├── tests/
└── scripts/
```

## 品質門檻

每次交給 Claude Code 實作後，至少執行：

```bash
./scripts/verify.sh
```

該腳本會跑：

- `ruff check .`
- `ruff format --check .`
- `pytest -q`

若本機未安裝 ruff/pytest，先執行 `pip install -e '.[dev]'`。
