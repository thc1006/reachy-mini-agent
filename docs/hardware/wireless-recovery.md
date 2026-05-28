# Reachy Mini Wireless 斷線復原指南 / Recovery Guide

**Date / 日期**: 2026-05-16
**Robot / 機器**: Reachy Mini Wireless (CM4), daemon SDK 1.7.3, prior IP `<REDACTED_TAILSCALE_IP>` / `<REDACTED_LAN_IP>`
**Symptom / 症狀**: 斷電重開後 Tailscale offline 8h+、LAN ping 不通、mDNS `reachy-mini.local` 不解析

---

## TL;DR — Canonical recovery path / 標準復原路徑

機器的 `Reachy Mini AP Launcher Service` 確實是 fallback AP。官方一級復原路徑是 **Bluetooth GATT**（不是 SSH、不是按鍵組合）。三條路按優先順序：

1. **Bluetooth via Reachy Mini Control desktop app** (recommended)
2. **Web Bluetooth dashboard** (Chrome/Edge/Opera)
3. **nRF Connect** (Android/iOS, manual GATT writes)

---

## Step 1 — 確認 AP 是否真的有起來 / Verify AP came up

在筆電 wifi 掃描清單裡找 SSID `reachy-mini-ap`（hyphen，全小寫）。官方文件確認這就是 fallback / first-boot AP 名稱。[(HF docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started)

If AP **不出現**：
- 確認頭部主板上的 switch 在 **debug** 位置，不是 download 位置。[(troubleshooting)](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md)
- 若 switch 正確仍無 AP → 走 Step 3 Bluetooth；都不行則需要 reflash ISO（最後手段）。

---

## Step 2 — 若 AP 可見：直接 web 設定 / If AP visible: web setup

1. 加入 `reachy-mini-ap`（open network、無密碼）。
2. 瀏覽器開 **`http://reachy-mini.local:8000/settings`**（同一個 daemon、AP NIC 上）。[(HF docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started)
3. 填 home wifi SSID + password → Connect。
4. 連上後 AP 自動消失；失敗時 daemon 會自動把 AP 重啟讓你重試。[(Wiki)](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_get_started/)

或更省事：用 desktop app **Reachy Mini Control** → 啟動畫面 footer 點 **"First time WiFi setup"**（1.7.x 有些版本字串叫 "First time connecting to your WiFi…"，同一個 wizard）。[(HF reset docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset)

---

## Step 3 — AP 不見、wifi 已忘：Bluetooth 強制復原 / Force AP via Bluetooth

機器永遠開著 BLE GATT service，名稱 **`ReachyMini`**。三種 client 任一皆可。

### 3a. Reachy Mini Control desktop app（首選）

App 內 First Time WiFi Setup wizard → footer **"Try the Bluetooth Console"** → 直接點 reset hotspot。[(HF reset docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset)

### 3b. Web Bluetooth（Chrome/Edge/Opera）

開 [https://pollen-robotics.github.io/reachy_mini/](https://pollen-robotics.github.io/reachy_mini/) → 配對 `ReachyMini` → 點 Reset Hotspot。需要實體靠近（BLE ~10m）。[(HF reset docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset)

### 3c. nRF Connect (advanced) — manual GATT writes

手機裝 nRF Connect → Scan → 選 **ReachyMini** → 進 **Unknown Service** → 找到 **WRITE** characteristic。

**永遠先送 PIN**：PIN = 序號最後 5 碼。例：序號 `xxxxxxxx4918400018` → 送 `PIN_00018`。

| Command | Plain text | Hex |
|---|---|---|
| Auth | `PIN_xxxxx` | `50494E5F` + ASCII hex of 5 digits |
| Query | `STATUS` | `535441545553` |
| **Force AP** | `CMD_HOTSPOT` | `434D445F484F5453504F54` |
| Restart daemon | `CMD_RESTART_DAEMON` | `434D445F524553544152545F4441454D4F4E` |
| Full reset | `CMD_SOFTWARE_RESET` | `434D445F534F4654574152455F5245534554` |

⚠ `CMD_SOFTWARE_RESET` 之後機器需 **~5 分鐘** 才回得來。[(HF reset docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset)

---

## Step 4 — 都失敗：SSH local console / Last resort

若 AP 起得來但 web setting page 失靈（Issue #599 描述的 venv corruption），SSH 進去：

- Username `pollen`, password `root`. [(HF docs)](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started)
- 連到 `reachy-mini.local` 或 router DHCP table 抓到的 IP。
- 跑 `reachyminios_check` 驗證系統。
- 若 daemon restart-loop（zenoh AttributeError 138+ retries），手動 `mv venvs venvs.bad && cp -r /restore/venvs ./venvs && systemctl restart`（這個 auto-recovery 仍是 enhancement request、未 merge，2026-05-16 為止）。[(Issue #599)](https://github.com/pollen-robotics/reachy_mini/issues/599)

---

## 1.7.x version-specific notes

- **1.7.3 (2026-05-13)** 的 "bluetooth import regression fix" 是把 BT GATT service import path 修回來、**不是 BT 行為變更**。1.7.3 BT recovery 路徑等同 1.7.0+。
- AP launcher 從首發就在；no 1.7.x 改過 SSID 名或 port。
- 沒有官方 **button-press combo** 進 setup mode；hardware path 只有 OFF → 5s → ON（單純 reboot，不是 reset）。[(troubleshooting)](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md)

---

## Negative findings / 沒查到的

- **沒有** 官方 button-combo factory reset。
- **沒有** documented AP password（open network、靠 BLE PIN 把關 config endpoint）。
- **沒有** 機器顯示自己當前 IP 的 LED pattern 或 audio cue（社群 issue 提過、未實作）。

---

## Recommended order for your case / 你這台建議順序

1. 先在筆電 wifi list 找 `reachy-mini-ap`（30 秒）→ 若有，Step 2。
2. 若沒有 → 開 Chrome [Web BT dashboard](https://pollen-robotics.github.io/reachy_mini/) 找 ReachyMini、送 PIN + `CMD_HOTSPOT`。
3. 仍不行 → nRF Connect 重試一次（手機 BLE stack 有時比 laptop 穩）。
4. 仍不行 → 物理重開 OFF/5s/ON，重來 1–3。
5. 最後 → 拆殼 SD/eMMC、reflash ISO（不要先做、太重）。

---

## Sources / 出處

- [HF docs · Reset via Bluetooth](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reset)
- [HF docs · Wireless setup guide](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started)
- [GitHub · troubleshooting.md](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/troubleshooting.md)
- [GitHub Issue #599 · power-loss resiliency](https://github.com/pollen-robotics/reachy_mini/issues/599)
- [Seeed Studio Wiki · Wireless Setup](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_get_started/)
- [Seeed Studio Wiki · BT Reset](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_reset/)
- [Web Bluetooth dashboard](https://pollen-robotics.github.io/reachy_mini/)
- [Reachy Mini Control download](https://hf.co/reachy-mini/#/download)
