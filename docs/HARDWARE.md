# Hardware requirements

## Robot

**[Reachy Mini](https://www.pollen-robotics.com/reachy-mini/)** (Pollen Robotics) running firmware ≥ 1.6.3.
The robot contains a Raspberry Pi CM4 that runs the Pollen daemon and exposes WebRTC on port 8443 (signaling) + WebSocket on 8000 (SDK).

## Brain host (tested configurations)

| Role | Minimum | Recommended | Our test rig |
|---|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 | Ubuntu 24.04 |
| GPU | 1× 4 GB CUDA (tiny whisper + 3B LLM) | 1× 16 GB (large whisper + 7–8B LLM + VLM) | 2× RTX 3090 24 GB |
| RAM | 16 GB | 32 GB | 64 GB |
| Disk | 20 GB | 50 GB | 2.8 TB |
| Python | 3.10 | 3.12 | 3.12 |

**You do not need a powerful machine at home.** Because all media flows over WebRTC, the brain host can be anywhere the robot can reach over IP — LAN, campus network, or a VPN like [Tailscale](https://tailscale.com/).

## Network layouts we use

### Same LAN

Robot and brain on the same Wi-Fi → `REACHY_HOST=reachy-mini.local` works via mDNS.

### Brain offsite (via Tailscale)

1. Install Tailscale on both robot and brain host:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sudo sh
   sudo tailscale up --ssh
   ```
2. Grab the robot's Tailscale IP from `tailscale status`.
3. Set `REACHY_HOST=100.x.x.x` in your `.env`.

WebRTC P2P usually punches straight through NAT via ICE. Tailscale's DERP relay is a fallback.

## GStreamer Rust plugins

The daemon's WebRTC stack requires `gst-plugins-rs` (specifically `webrtcsrc` / `webrtcsink`). Ubuntu's `gstreamer1.0-plugins-rs` package is too old on 22.04 and missing on 24.04. Build from source once:

```bash
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev libglib2.0-dev libssl-dev libnice-dev \
    gstreamer1.0-{plugins-base,plugins-good,plugins-bad,plugins-ugly,libav} \
    cargo

# rustup for MSRV ≥ 1.83 (system cargo is too old on 24.04)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install cargo-c

git clone --depth 1 --branch 0.14 https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs
cd gst-plugins-rs
cargo cinstall -p gst-plugin-webrtc -p gst-plugin-webrtchttp \
    --prefix=/opt/gst-plugins-rs --libdir=/opt/gst-plugins-rs/lib/x86_64-linux-gnu
```

Then either set `GST_PLUGIN_PATH=/opt/gst-plugins-rs/lib/x86_64-linux-gnu` in `.env` or symlink into `/usr/lib/x86_64-linux-gnu/gstreamer-1.0/`.

Verify: `gst-inspect-1.0 webrtcsrc` should print a "Factory Details" block.

## CUDA / cuDNN

`faster-whisper` and `kokoro-onnx` both need cuDNN 9. The Python wheels ship their own cuDNN but the loader can't find it by default. `scripts/run_robot.sh` prepends the wheel's lib dirs to `LD_LIBRARY_PATH`:

```bash
SP="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
```

## Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
```

To keep both loaded simultaneously (needed for vision + chat without swap):

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment=OLLAMA_MAX_LOADED_MODELS=2
Environment=OLLAMA_KEEP_ALIVE=30m
Environment=OLLAMA_NUM_PARALLEL=1
```
