# Reachy Mini Wireless — Recovery & Reflash 指南 / Recovery & Reflash Guide

**Date / 日期:** 2026-05-16
**Scope / 範圍:** Reachy Mini **Wireless** (CM4 + WiFi + battery), 卡死於 power-cycle 後，已用 USB 線接到 Windows laptop

---

## 0. Key Finding — USB-C 救不了你 / USB-C Won't Save You

Pollen Robotics 官方 FAQ 明文寫：
> "Wireless units do not expose the robot over USB the way the Lite version does, so plugging a USB-C cable into your laptop will not give you a working connection." ([HF Troubleshooting](https://huggingface.co/docs/reachy_mini/troubleshooting))

機殼上那個 USB-C 是 **CM4 的 USB host output**（用來插 USB key、不是 device gadget），規格表明確列為 "USB-C output (i.e. one can plug a device such as a usb key)" ([Hardware datasheet](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/hardware))。所以 laptop 不會看到 RNDIS Ethernet gadget、也不會自動拿到 169.254 / 10.x.x.x IP、Device Manager 不會冒出任何新裝置。**第一個結論：把那條 USB-C 線拔掉，它幫不到你。**

---

## 1. 先試「輕量恢復」/ Lighter Recovery Paths to Try FIRST

按 ROI 由高到低：

### 1.1 Power off, wait 5s, power on（官方第一招）
Pollen FAQ 首條："press OFF, wait 5 seconds, then press ON. This simple procedure fixes several common and well-known issues." ([HF Troubleshooting](https://huggingface.co/docs/reachy_mini/troubleshooting))

### 1.2 確認頭部 SW1 switch 在 **DEBUG** 而不是 DOWNLOAD
這是 wireless 卡死 #1 真因。Pollen FAQ "Wireless Access point doesn't show up - RPI doesn't boot" 條：
> "There is a switch on the board in the head that needs to be in a given position... It's possible that this switch was moved during assembly or maybe even a factory mistake. Please check that the switch is on the 'debug' and not on 'download' position."

需要打開頭部後殼確認；若被撞到切到 DOWNLOAD，CM4 會卡在 USB recovery 不開機、AP 不出。

### 1.3 找 `reachy-mini-ap` Wi-Fi
正常開機 ~30-60s 後 robot 會發 AP：
- SSID: `reachy-mini-ap` / password: `reachy-mini`
- Robot IP: `10.42.0.1` ([Seeed Wiki](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_get_started/))
- 連上後 SSH： `ssh pollen@reachy-mini.local` password `root` ([HF Get Started](https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/get_started))
- 或 browser `http://reachy-mini.local:8000/`

### 1.4 確認既有 Wi-Fi 上有沒有 robot
之前如果已 onboard 進家裡 Wi-Fi，AP 不會自動出。先在 router 的 DHCP client list 找 `reachy-mini`，或用 Reachy Mini Control 桌面 app 掃。([HF Troubleshooting](https://huggingface.co/docs/reachy_mini/troubleshooting) "reachy-mini.local doesn't resolve")

### 1.5 USB-C-to-Ethernet adapter（替代有線）
官方推薦的「tethered link」是 **USB-C-to-Ethernet 轉接頭 + 網路線**，不是 USB 對 USB：
> "For a tethered link, use a USB-C-to-Ethernet adapter plus an Ethernet cable—this simply replaces Wi-Fi with wired Ethernet." ([HF Troubleshooting](https://huggingface.co/docs/reachy_mini/troubleshooting))

如果手邊有 USB-C 網卡 + 短 RJ45，可試。

### 1.6 Reachy Mini Control 的 "Reset apps environment" / "Full Environment Reset"
若能連上 robot 但 app 層卡住，桌面 app 的 Settings → Environment 有兩個按鈕可清 `apps_venv` 或全部 venv，不需重刷 OS。([HF Troubleshooting](https://huggingface.co/docs/reachy_mini/troubleshooting))

---

## 2. 官方 Reflash 程序（最後手段）/ Official Reflash Procedure

來源：[`pollen-robotics/reachy_mini/docs/.../reflash_the_rpi_ISO.md`](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/platforms/reachy_mini/reflash_the_rpi_ISO.md) · [Seeed Wiki mirror](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_reflash_the_rpi_iso/) · [HF docs](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reflash_the_rpi_ISO)

**儲存裝置：CM4 內建 eMMC**（不是 SD card；wireless 用 CM4104016 = 16GB eMMC）([HW datasheet](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/hardware))

### Windows 步驟
1. **下載**最新 ISO `.zip` from [`reachy-mini-os/releases`](https://github.com/pollen-robotics/reachy-mini-os/releases) （latest: v0.2.3, 2026-01-14；Windows 不需 `.bmap`）
2. **裝 rpiboot GUI**: <https://github.com/raspberrypi/usbboot/raw/master/win32/rpiboot_setup.exe>
3. **拆頭部後殼**找到 PCB 上的 **SW1** switch → 撥到 **DOWNLOAD**
4. **拔掉 USB-C output 那條**（不是它），改插 **USB2 micro-USB port**（PCB 上標 USB2 的那顆，照圖 [`wireless_switch.png`](https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/wireless_switch.png)）
5. Robot **power on** → laptop Device Manager 應該出現 **"BCM2711 Boot"** 或類似 → 啟動 RPiBoot.exe → 等幾秒
6. RPiBoot 跑完 → Windows 多一顆 **"RPi-MSD-..."** USB Mass Storage（就是 eMMC）
7. **Raspberry Pi Imager** → "Use custom" → 選下載的 `.zip` → target 那顆 RPi-MSD → Write
8. 完成後：**power off** → SW1 撥回 **DEBUG** → 拔 USB → 重新開機
9. **驗證**：連上 AP，`ssh pollen@reachy-mini.local`，跑 `reachyminios_check`

### Linux/macOS 替代
```bash
# Linux
sudo bmaptool copy <image>.zip --bmap <image>.bmap /dev/sda
# macOS — 用 /dev/rdiskX 不是 /dev/diskX
```

---

## 3. Reflash 前必備份 / Back Up Before Reflash

Reflash = factory reset，全部清掉。先 SSH 進去（若還連得上）撈：

| 類別 | 路徑/方式 | 說明 |
|---|---|---|
| **Tailscale 認證** | `sudo cp -r /var/lib/tailscale ~/tailscale-backup` | 不備份 = 重 onboard 進 tailnet，s1/reachy-mini node ACL 重設 |
| **Wi-Fi credentials** | `/etc/NetworkManager/system-connections/` | 不備就要重打家裡 Wi-Fi |
| **Daemon configs** | `/etc/reachy-mini-daemon/` + `~/.config/reachy_mini/` | 馬達 calibration 通常在 daemon assets 內，多半不會壞 |
| **Installed apps** | `/venvs/apps_venv` + app 原碼 | conversation app 等要重裝；自寫 app 撈 src |
| **robot-brain 自寫程式碼** | 通常你的 robot_brain 跑在 s1 不在 robot — 自確認 | 本機 working tree 在 `C:\Users\thc1006\Desktop\reachy-mini\`，安全 |
| **錄音/錄影/log** | `~/recordings`、`/var/log/reachy*` | 不重要可棄 |
| **`/etc/hosts`、systemd unit overrides** | 自寫的 service 檔 | 例如 `systemd/whisper-server.service` |

備份指令範例：
```bash
ssh pollen@reachy-mini.local "sudo tar czf /tmp/reachy-backup.tgz \
  /var/lib/tailscale /etc/NetworkManager/system-connections \
  /etc/reachy-mini-daemon ~/.config/reachy_mini /etc/systemd/system/*.service"
scp pollen@reachy-mini.local:/tmp/reachy-backup.tgz .
```

---

## 4. 風險與時間估計 / Risks & Time Estimate

| 項目 | 估計 |
|---|---|
| 下載 ISO (v0.2.3 ~1-2 GB) | 5-15 min |
| Tear-down 頭殼 + 撥 SW1 | 5-10 min（需要小起子，會看到內部 FPC，小心鏡頭排線） |
| rpiboot + Imager 寫入 | 8-15 min（eMMC 16 GB） |
| 重新組裝 + 開機 + AP | 5 min |
| 重連 Wi-Fi、Tailscale、重裝 conversation app、systemd 服務 | 30-60 min |
| **總計** | **約 60-110 min**（不含重新校正馬達 / 重訓 wake-word 等 personal artifacts） |

**主要風險**：
- 拆頭殼時碰到 mic FPC 排線（先前已有過 mic 全 0 的歷史教訓）
- SW1 撥回 DEBUG 忘了 → 重新開機沒 AP，又得拆一次
- Tailscale 沒備 → s1 production 的 SSH alias 全失效，要重 onboard 並更新 `~/.ssh/config`
- 重裝完 SDK 版本可能跳到 1.7.x，跟本機 worktree 的 1.6.3 假設有差

---

## Sources
- [HF — Reachy Mini Wireless Setup](https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/get_started)
- [HF — Reflash Procedure](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/reflash_the_rpi_ISO)
- [HF — Hardware Datasheet](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/hardware)
- [HF — Troubleshooting & FAQ](https://huggingface.co/docs/reachy_mini/troubleshooting)
- [GitHub — pollen-robotics/reachy_mini reflash doc](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/platforms/reachy_mini/reflash_the_rpi_ISO.md)
- [GitHub — reachy-mini-os releases](https://github.com/pollen-robotics/reachy-mini-os/releases)
- [Seeed Wiki — Reflash mirror](https://wiki.seeedstudio.com/reachymini_platforms_reachy_mini_reflash_the_rpi_iso/)
- [GitHub — raspberrypi/usbboot](https://github.com/raspberrypi/usbboot)
- [Jeff Geerling — CM4 eMMC usbboot tutorial](https://www.jeffgeerling.com/blog/2020/how-flash-raspberry-pi-os-compute-module-4-emmc-usbboot/)
