# WebRTC "Lost connection" reconnect strategy for Reachy Mini Wireless
# Reachy Mini Wireless 失聯重連策略 — 2026-05-16

> **TL;DR / 結論在最前面**
>
> - **1 小時**：在 `robot_brain.py` 的 `tracking_loop` 加入 `MAX_LOST_FRAMES` 上限 + `sys.exit(1)` → 讓 systemd `Restart=on-failure` 接手。同時把 `Restart=on-failure`、`RestartSec=3`、`StartLimitBurst=10`、`WatchdogSec=30` 寫進 unit。
> - **1 天**：在 robot-brain 包一層 `ReachySupervisor`：抓 `ConnectionError("Lost connection with the server.")` → `mini.client.disconnect()` → 指數 backoff 重連 → 重建 `MediaManager(WEBRTC)`。在 watchdog 上做 daemon `/api/state` 健康偵測，2 連敗才退出。
> - **1 週**：把 control plane 從 WebRTC/WS 拆出來改走 daemon REST（已存在的 `/api/move/goto`、`/api/state/present_head_pose`）；只用 WebRTC 取 frame。在 robot 上加 USB→Ethernet 轉接 (避開 CM4 wifi)，並把 robot 本身的 systemd daemon 設 `Restart=always`。
>
> **Top 3 prioritized actions（給呼叫端）**：
> 1. `robot_brain.tracking_loop` 加 exception-counter + `sys.exit(1)` 觸發 systemd 重啟（30 分鐘可上線）
> 2. 在 systemd unit 補 `Restart=on-failure` + `WatchdogSec=30` 主動 liveness（10 分鐘）
> 3. 寫 `ReachySupervisor` wrapper：偵測到 `Lost connection` 就 disconnect + 指數 backoff reconnect + media re-init（半天可寫完）

---

## 0. 為什麼會持續失聯 / Root cause（why it happens 3×/day）

SDK `WSClient._check_alive` 是個**被動心跳**：每秒清掉 `_heartbeat` event，再等 daemon 在 1 秒內推任何訊息回來。一旦 Tailscale 抖動 / DERP 重連 / Pi 4 wifi 短暫掉 RSSI，1 秒內沒收到任何 server frame，`_is_alive` 就被設成 `False`、下一次 `send_command` 直接丟 `ConnectionError("Lost connection with the server.")`。
（見本機 SDK `reachy_mini/io/ws_client.py:109-119, 142`）

**SDK 完全沒有 reconnect 路徑** — 即使在 GitHub `main` 分支（已超過本機 1.6.3）也沒有：`_check_alive` 只做 status report、不自救（fetched via `main` source 2026-05-16）。社群 issue 也確認這點：[`reachy_mini_conversation_app#167`](https://github.com/pollen-robotics/reachy_mini_conversation_app/issues/167)（fastrtc 在 ~2 分鐘自己 emit `endSession`，Backlog、無 maintainer 回覆）。

具體在 robot_brain 的後果（`src/robot_brain.py:2238-2370`）：`tracking_loop` 用 `mini.set_target(...)` → SDK 內部 `client.send_command` → `if not self._is_alive: raise ConnectionError`。`tracking_loop` 的 `except Exception as e: print(...)` 把錯誤吃掉但**不退出 loop 也不嘗試重連**，於是 stderr 每幀印一行直到使用者手動 kill。已部署的 120 s log-silent watchdog 因為 log 有在寫所以不會觸發。

額外加重因素：
1. **WebRTC ICE 30 s stall** — 已知 wireless bug：[`reachy_mini#888` "WebRTC ICE misconfiguration in GstWebRTCClient may cause intermittent 30s connection delays"](https://github.com/pollen-robotics/reachy_mini/issues/888) 指出 `webrtcbin` 沒設 STUN、`_configure_webrtcbin` 在 `pad-added` 之後才跑，ICE gathering 用 default config 結果 IPv6 link-local / TCP passive candidate 各等 30 s。
2. **USB hub 12 s 後 disconnect** — [`reachy_mini#904`](https://github.com/pollen-robotics/reachy_mini/issues/904) wireless USB hub + ReSpeaker 開機 ~12 s 後斷，root cause unknown、被標為 hardware-level。
3. **Tailscale DERP relay 性能限制** — DERP 是 TCP-over-TLS、無 QoS，丟包率 / 延遲都比 direct UDP 高很多（[Tailscale 官方文件](https://tailscale.com/docs/reference/connection-types)；[peer relay beta blog](https://tailscale.com/blog/peer-relays-beta) 顯示 DERP 在跨國場景常退化）。本機在 HKG relay 走 DERP 機率高。

---

## 1. 具體 code patches（依優先級）/ Concrete patches

### Patch #1 — `tracking_loop` exit-on-repeat（effort 10 min, risk LOW）

讓 systemd 接手。**目前 `tracking_loop:2369` 的 `except Exception as e: print(f"  [追蹤錯誤] {e}", flush=True)` 是元兇**，加 counter：

```python
# 在 tracking_loop 開頭
_LOST_CONN_LIMIT = 3
_lost_conn_count = 0

# 在 except 區塊
except Exception as e:
    print(f"  [追蹤錯誤] {e}", flush=True)
    if "Lost connection" in str(e):
        _lost_conn_count += 1
        if _lost_conn_count >= _LOST_CONN_LIMIT:
            print(f"  [追蹤] {_LOST_CONN_LIMIT} 次失聯、退出讓 systemd 重啟", flush=True)
            stop_event.set()
            os._exit(2)  # bypass cleanup that itself uses client
    else:
        _lost_conn_count = 0  # 不同錯誤、reset
```

`os._exit(2)` 而非 `sys.exit(1)` 因為 `__del__` 會呼叫 `client.disconnect()`、在已死的 socket 上會卡住。

### Patch #2 — systemd unit hardening（effort 10 min, risk LOW）

在 robot-brain 的 service file（s1 上）：

```ini
[Service]
Restart=on-failure
RestartSec=3
StartLimitBurst=10
StartLimitIntervalSec=120
TimeoutStopSec=10
# 主動 liveness：app 必須每 ≤15s call sd_notify(WATCHDOG=1)
WatchdogSec=30
Type=notify
```

對應 Python 端（最小 implementation）：

```python
# top of robot_brain.py
try:
    from systemd.daemon import notify as _sd_notify
except ImportError:
    _sd_notify = lambda _: None
_sd_notify("READY=1")

def _watchdog_ping_loop():
    while True:
        # 只在 mini.client._is_alive == True 時 ping
        if getattr(mini, "client", None) and mini.client._is_alive:
            _sd_notify("WATCHDOG=1")
        time.sleep(10)

threading.Thread(target=_watchdog_ping_loop, daemon=True).start()
```

連狀態斷時**故意不 ping** → 30 s 後 systemd 自動 SIGTERM + 重啟。這比 120 s 純 log-silent 偵測更靈敏，因為 log 還在寫的 stuck 場景也會被抓到。設計依據：[Ubuntu systemd watchdog guide](https://oneuptime.com/blog/post/2026-03-02-configure-systemd-restartsec-watchdogsec-ubuntu/view) 建議 ping 間隔 = `WatchdogSec / 2`。

### Patch #3 — `ReachySupervisor` 重連 wrapper（effort 4 hr, risk MED）

在 `robot_brain.py` 頂層用 supervisor 包裝 `ReachyMini`，把所有 SDK call 走 retry：

```python
class ReachySupervisor:
    """Wraps ReachyMini with automatic reconnect on ConnectionError."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._backoffs = [0.5, 1, 2, 5, 10, 30]  # cap at 30s per https://websocket.org/guides/reconnection/
        self._mini = self._connect()
        self._lock = threading.RLock()

    def _connect(self):
        for attempt, delay in enumerate(self._backoffs):
            try:
                m = ReachyMini(**self._kwargs)
                print(f"[supervisor] connected (attempt {attempt})", flush=True)
                return m
            except (ConnectionError, TimeoutError, OSError) as e:
                print(f"[supervisor] connect attempt {attempt} failed: {e}; "
                      f"retry in {delay}s", flush=True)
                # jitter to avoid thundering herd
                time.sleep(delay * (0.75 + 0.5 * random.random()))
        raise ConnectionError("Supervisor: exhausted retries")

    def _reconnect(self):
        with self._lock:
            try:
                self._mini.media_manager.close()
            except Exception:
                pass
            try:
                self._mini.client.disconnect()
            except Exception:
                pass
            self._mini = self._connect()

    def __getattr__(self, name):
        attr = getattr(self._mini, name)
        if not callable(attr):
            return attr
        def wrapped(*a, **kw):
            try:
                return attr(*a, **kw)
            except ConnectionError as e:
                if "Lost connection" not in str(e):
                    raise
                print(f"[supervisor] {name} hit Lost connection — reconnecting", flush=True)
                self._reconnect()
                return getattr(self._mini, name)(*a, **kw)  # one retry after reconnect
        return wrapped
```

重要：`mini.media` 是個 `MediaManager` property、不是 method，所以 `__getattr__` 對它走 `not callable` 直接 return，**新 connect 後它自動指向新的 media manager**（因為 `self._mini` 已被換掉）。但 caller 若把 `mini.media.get_frame` 綁進 local var 就會抓到舊 ref，故 supervisor 鼓勵 caller 每次重新 `mini.media.get_frame()`。

`tracking_loop` 已是這個模式（line 2295 `frame = mini.media.get_frame()`），所以 drop-in 相容。

### Patch #4 — 把 `set_target` 高頻路徑改走 daemon REST（effort 6 hr, risk MED-HIGH）

WebSocket 是 control plane 的 SPOF。`mini.set_target` 走 `WSClient.send_command` → 一斷就整片癱瘓。但 daemon 同時開 HTTP API（`http://reachy-mini:8000/api/...`、見 `daemon/app/routers/*.py`），裡頭已有 `move.py`、`state.py`、`motors.py`。改造方案：

```python
# Replace mini.set_target with idempotent REST POST
def set_target_rest(pitch_deg, yaw_deg, body_yaw_rad):
    try:
        requests.post(f"http://{HOST}:8000/api/move/set_target",
                      json={"head_pitch": pitch_deg, "head_yaw": yaw_deg,
                            "body_yaw": body_yaw_rad},
                      timeout=0.2)  # short timeout, fire-and-forget
    except requests.RequestException:
        pass  # next frame will retry; WSClient stays untouched
```

**Caveat**：實際 daemon 是否暴露 `/api/move/set_target` 要先 `curl` 確認；如果只有 `goto`、就用 `mini.set_target_head_pose` 退而求其次（仍走 WS、但至少 set_target 是 fire-and-forget 不需 ack）。我**沒有實證**這條 endpoint 存在 — see "Negative findings" §5。

### Patch #5 — STUN bypass for local Tailscale（effort 30 min, risk LOW）

照 [reachy_mini#888](https://github.com/pollen-robotics/reachy_mini/issues/888) 的建議，在 supervisor 啟動時 monkey-patch `_on_deep_element_added`：

```python
def _patch_webrtc_stun(client):
    """Disable STUN to avoid 30s ICE stall on Tailscale (no NAT to traverse)."""
    orig = client._on_deep_element_added
    def patched(bin_, sub_bin, element):
        orig(bin_, sub_bin, element)
        f = element.get_factory()
        if f and f.get_name() == "webrtcbin":
            element.set_property("stun-server", None)
            element.set_property("bundle-policy", 3)  # max-bundle
    client._on_deep_element_added = patched
```

Tailscale 兩端都在同個 magic IP 段、不需要 STUN/TURN，移掉它能省 30 s ICE gathering 時間（也就是現在 reconnect 後第一張 frame 慢的可能元兇）。

---

## 2. Best-practice patterns referenced

| 模式 | 出處 | 我們現況 vs 應有 |
|---|---|---|
| **Capped exponential backoff with jitter** | [websocket.org reconnection guide](https://websocket.org/guides/reconnection/)：cap 30s、25% jitter | 現況：無重連；應有：0.5→1→2→5→10→30s with ±25% jitter |
| **`websockets.connect()` async iterator auto-retry** | [websockets 16.0 docs](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html)：default retry 在 EOFError/OSError/TimeoutError/5xx | SDK 用 `websockets.sync.client` 沒有這個語法糖、必須手刻 |
| **systemd `WatchdogSec` + `sd_notify(WATCHDOG=1)`** | [Ubuntu systemd watchdog guide](https://oneuptime.com/blog/post/2026-03-02-configure-systemd-restartsec-watchdogsec-ubuntu/view)：ping 間隔 = WatchdogSec/2 | 現況：120s log-silent；應有：30s active liveness + 還在 ping ≠ 還在通訊（gate on `_is_alive`） |
| **Sequence-number message replay after reconnect** | [websocket.org reconnection guide](https://websocket.org/guides/reconnection/)：server replay messages with higher seq | 不適用 — 控制流是 state-snapshot type（set_target absolute pose），重連後重發最新一次即可 |
| **Pet watchdog only when healthy** | gist.github.com/Spindel/1d07533ef94a4589d348 — "stop notifying systemd if subsystem is dead" | Patch #2 即此模式 |
| **Tailscale prefer direct over DERP** | [Tailscale connection types](https://tailscale.com/docs/reference/connection-types)：DERP throughput / QoS 差 | 看 `tailscale netcheck` / `tailscale ping` 是否走 direct；若 HKG relay 則考慮 peer relay 或 USB-Ethernet |

---

## 3. Architectural alternatives / 架構級替代方案

| 選項 | 成本 | 效益 | 風險 | 建議 |
|---|---|---|---|---|
| **A. robot_brain 整個搬上 robot CM4** | 高（Pi 4 ARM 跑 vLLM client + STT pipeline 很拼，但 robot_brain 本身只是 orchestrator、不跑 LLM/STT，可行） | 消滅網路抖動；mini 走 `localhost` | LLM/STT/vision call 仍走 Tailscale 出去到 s1、ECONNRESET 換另一個地方爆 | **不建議** — 把問題從「mini ↔ brain」搬到「brain ↔ s1」、攔截面差不多 |
| **B. USB→Ethernet adapter 給 robot 走有線** | $20 + 一條 cable | 拔掉 wifi 干擾源；Pi 4 wifi 已知不穩（[RPi 論壇 #339699](https://forums.raspberrypi.com/viewtopic.php?t=339699)） | CM4 wireless 變體有沒有 USB Type-A 可用待確認；走 ethernet 時 wifi 還是 default route 要 metric 調整 | **強烈建議**做為長期方案 |
| **C. Tailscale peer relay / direct mode tuning** | 配置數小時 | 跳 DERP 減 latency；用 [peer relay beta](https://tailscale.com/blog/peer-relays-beta) 拿 27-35 Mbps vs DERP 約 2-3 Mbps | 兩端都要 v1.84+；本機 IPN 防火牆規則要開 41641/UDP | **建議**配 B 一起做 |
| **D. control plane 換 ZeroMQ / NATS / gRPC**，video 仍走 WebRTC | 高（要寫 daemon-side 對應 endpoint） | 控制與媒體解耦；單一通道斷不全死 | 改 daemon、不再是 vanilla SDK | **不建議**：daemon 已暴露 REST、用 Patch #4 等效 |
| **E. 把 face_tracker 移到 daemon backend (跑在 mini 本機)** | 中（要 fork SDK 加 backend 插件） | 完全消除 control plane 抖動的影響；視覺 + 控制都 local | 失去 robot_brain 統一 orchestration、tool_call 跟 face_track 衝突要重新 coordinate | **未來考量**：SDK 1.8+ 有 plugin 化趨勢 ([reachy_mini#1080](https://github.com/pollen-robotics/reachy_mini/issues/1080) 可作 audio board config 從 remote 套用) |

---

## 4. What Pollen officially recommends / DOESN'T

**官方 troubleshooting doc** ([reachy_mini/docs/source/troubleshooting.md](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md)) **完全沒提**：
- WebRTC reconnect / Lost connection
- Keepalive 或 ping_interval 設定
- Long-running client robustness
- Network jitter / DERP / Tailscale 配置

只談 mDNS / conference network client isolation / 中國地區存取，**屬於初次連線問題、不是長時間穩定問題**。

**Pollen 自家的 conversation app** (`reachy_mini_conversation_app/main.py`) 用 `try ... except ConnectionError: sys.exit(1)` 模式 — **連線失敗就死、不重連**（quoted from `main.py` 透過 WebFetch）。 Daemon 連線 timeout 也一樣 `sys.exit(1)`。換言之：**Pollen 自己也預期你用 systemd 重啟、SDK 層不打算做 reconnect**。

Issue `#167` (`endSession` 在 2 分鐘) 跟 issue `#904` (USB hub 12s 後 disconnect)：**兩個都標 Backlog、沒有 maintainer 回覆、沒有 PR 對應**。

Issue `#888` (WebRTC ICE 30s stall)：open、有完整 RCA + patch 建議、無 maintainer 動作 — 我們可以**自己套這個 patch**（見 Patch #5），不需等官方。

---

## 5. Negative findings / 明確未找到的東西

- **SDK `WSClient` 沒有 `auto_reconnect`、`ping_interval`、`ping_timeout` 參數**（檢查 1.6.3 source + main branch via WebFetch；只有 `connect()` 用 default `websockets.sync.client.connect`，無 keepalive override）。
- **`GstWebRTCClient` 沒有 reconnect API、`disconnected` callback、`on_connection_state_changed` hook**（檢查 `media/webrtc_client_gstreamer.py` 全 file；`_on_bus_message` 只處理 EOS + 過濾掉 `not-negotiated` warning、收到真的 ERROR 是 `return False` 讓 bus removed、pipeline 不會被通知 caller）。
- **未找到** Pollen 官方推薦的 reconnect pattern、無 KB article、無 forum sticky。
- **未找到** `mini.media.is_connected()` 之類的 health-check API。Caller 只能用 `mini.client._is_alive`（private attribute、官方不保證穩定）。
- **未確認** daemon 是否暴露 `/api/move/set_target`（vs 已存在的 `/api/move/goto`） — Patch #4 落地前要 `curl http://reachy-mini:8000/openapi.json | jq '.paths'` 列一下 endpoint 才知道。
- **未找到** brevdev/reachy-personal-assistant repo（CES 2026 NVIDIA × Pollen demo）的 reconnect 寫法 — 搜尋無結果、推測該 demo 是 controlled-environment、不處理斷線。
- **未找到** Pi 4 / CM4 wireless 變體跑 wifi + WebRTC 的官方 best practice；通用 RPi 4 forum 普遍指 wifi power management、driver 不穩、power draw 是元兇 ([forum #339699](https://forums.raspberrypi.com/viewtopic.php?t=339699))。

---

## 6. 1 小時 / 1 天 / 1 週 / If you only have ...

- **1 小時**：Patch #1 (`tracking_loop` exit-on-repeat) + Patch #2 (systemd `WatchdogSec=30`)。完。系統會在 3 次 Lost connection 內自動冷重啟、最壞 30s 一定恢復。
- **1 天**：上面 + Patch #3 (`ReachySupervisor`) + Patch #5 (STUN bypass)。多數情況**無需** systemd 重啟、in-process 重連即可。
- **1 週**：上面全部 + Patch #4 (daemon REST control plane) + USB→Ethernet adapter (Option B) + Tailscale peer relay tuning (Option C)。把 wifi 跟 DERP 兩個元兇從鏈路上拔掉。

---

## Sources

- Local SDK 1.6.3：`C:\Users\thc1006\Desktop\reachy-mini\.venv\Lib\site-packages\reachy_mini\io\ws_client.py:74-167`、`media/media_manager.py`、`media/webrtc_client_gstreamer.py`
- Local app：`C:\Users\thc1006\Desktop\reachy-mini\src\robot_brain.py:2238-2370`
- [pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md)
- [pollen-robotics/reachy_mini_conversation_app/issues/167](https://github.com/pollen-robotics/reachy_mini_conversation_app/issues/167) — endSession in ~2 min
- [pollen-robotics/reachy_mini/issues/888](https://github.com/pollen-robotics/reachy_mini/issues/888) — WebRTC ICE 30s stall
- [pollen-robotics/reachy_mini/issues/904](https://github.com/pollen-robotics/reachy_mini/issues/904) — USB hub disconnect 12s after boot
- [pollen-robotics/reachy_mini/issues/905](https://github.com/pollen-robotics/reachy_mini/issues/905) — Motor + daemon connection lost
- [pollen-robotics/reachy_mini/issues/687](https://github.com/pollen-robotics/reachy_mini/issues/687) — Wireless daemon graceful shutdown race
- [pollen-robotics/reachy_mini/issues/1062](https://github.com/pollen-robotics/reachy_mini/issues/1062) — Jetson webrtcbin compat
- [pollen-robotics/reachy_mini/issues/1080](https://github.com/pollen-robotics/reachy_mini/issues/1080) — Remote webRTC audio config
- [pollen-robotics/reachy_mini_conversation_app/blob/main/src/reachy_mini_conversation_app/main.py](https://github.com/pollen-robotics/reachy_mini_conversation_app/blob/main/src/reachy_mini_conversation_app/main.py) — `sys.exit(1)` on connect fail
- [websockets 16.0 docs — connect() auto-retry iterator](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html)
- [websocket.org — WebSocket Reconnection: State Sync and Recovery Guide](https://websocket.org/guides/reconnection/)
- [Configure systemd RestartSec and WatchdogSec on Ubuntu](https://oneuptime.com/blog/post/2026-03-02-configure-systemd-restartsec-watchdogsec-ubuntu/view)
- [Tailscale Connection Types](https://tailscale.com/docs/reference/connection-types)
- [Tailscale Peer Relays Beta](https://tailscale.com/blog/peer-relays-beta)
- [Pi 4 forum #339699 — Pi 4 keeps dropping WiFi](https://forums.raspberrypi.com/viewtopic.php?t=339699)
- gist.github.com/Spindel/1d07533ef94a4589d348 — Python systemd watchdog example
