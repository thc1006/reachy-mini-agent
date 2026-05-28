# SmolVLA × Reachy Mini — public fine-tune verification

**Date:** 2026-05-13
**Scope:** 2025-09 → 2026-05-13
**Question:** Has anyone publicly fine-tuned SmolVLA (or any VLA — Pi0, GR00T, etc.) for the Reachy Mini robot specifically?

---

## Bottom line

**No.** Across HF Hub, GitHub, arXiv, and social/blog channels, there is **zero published evidence** of a SmolVLA fine-tune for Reachy Mini (or Reachy Mini Wireless). The same is true for Pi0 / Pi0-FAST / GR00T fine-tunes targeting Reachy Mini head trajectory.

Foundational prerequisites are also missing:
- Reachy Mini is **not yet officially integrated into LeRobot** (PR #2726 is still open since 2025-12-27, last touched 2026-04-01).
- **No teleop / head-trajectory dataset for Reachy Mini exists on HF Hub.** Pollen's `reachy-mini-*` datasets cover only emotions, dances, app-store metadata, and benchmarks — nothing time-series action.
- Pollen Robotics has published only 3 models on HF — none are VLAs, and the only policies are `act_reachy2_*` (Reachy 2, not Mini).

**Confidence: HIGH** that no public Reachy-Mini-specific VLA fine-tune exists.

---

## Search 1 — HF Hub: `smolvla*` models

Browsed `huggingface.co/models?search=smolvla`. Found ~19 distinct community fine-tunes:

| Owner | Model | Target robot |
|---|---|---|
| lerobot | smolvla_base | (base) |
| Sa74ll, rancheng222, renyiyu, un1c0rnio, JiabinQ | smolvla_so101_* | SO-101 |
| aiwhisperer, alexis779, DanqingZ, jccj, pranavsaroha | smolvla_so100_* | SO-100 |
| Haribot099, ns69956, semi01, jiuyal2, Jamesbass, jtz18, masato-ka, fulloa10 | various | unspecified / SO-arm |

**All target SO-100/SO-101 manipulation arms or simulation.** Zero mention of Reachy in names or visible cards.

## Search 2 — HF Hub: pollen-robotics & reachy-mini orgs

`huggingface.co/pollen-robotics`:
- **Models (3):** `anyskin-slip-detection`, `act_reachy2_mobile_household_apple`, `act_reachy2_static_cup`. Both ACT policies are for **Reachy 2**, not Mini. No VLA.
- **Datasets (16+):** all Reachy 2 manipulation / pick-and-place / mobile, plus Reachy Mini `emotions-library`, `dances-library`, `official-app-store`. No trajectory/teleop dataset for Reachy Mini.

`huggingface.co/reachy-mini` org page failed to render in fetch (returned "Refreshing"), but cross-referenced via dataset search — confirms emotions/dances/app-store only.

## Search 3 — HF Hub: datasets matching "reachy"

~20 datasets. Reachy Mini-specific ones:
- `pollen-robotics/reachy-mini-emotions-library` — emotion clip library
- `pollen-robotics/reachy-mini-dances-library` — dance clip library
- `pollen-robotics/reachy-mini-official-app-store` — app catalog (8 apps, see Search 4)
- `yourbench/reachy_mini_info_benchmark`, `alekgomez/reachy_mini_info_benchmark` — text Q&A benchmarks, **not trajectory**

All Reachy-teleop / pick-place / multimodal datasets are for **Reachy 2** (haixuantao, cadene, glannuzel, simheo, CompeteSAI).

## Search 4 — Reachy Mini App Store (apps as of 2026-05-13)

`pollen-robotics/reachy-mini-official-app-store` lists 8 apps:
1. cdeplanne/wake_me_up
2. pollen-robotics/reachy_mini_conversation_app  ← uses **SmolVLM2** (vision-language, not VLA) via `--local-vision` flag
3. pollen-robotics/reachy_mini_radio
4. pollen-robotics/red_light_green_light
5. dlouapre/coding_lab
6. RemiFabre/marionette
7. Boopster/reachy_mini_metronome
8. pollen-robotics/reachy_mini_testbench

**None depend on smolvla / lerobot.smolvla / pi0 / vla.** Important distinction: `reachy_mini_conversation_app` uses **SmolVLM2** (vision-language captioning) — this is the LLM agent stack we already know, not a VLA policy.

Note: prior memory mentioned "200+ apps as of 2026-05-06 launch" — the official curated catalog dataset only enumerates 8. Either remaining apps are unofficial/community, or the 200+ figure was marketing. Either way no VLA in the official 8.

## Search 5 — GitHub

- **huggingface/lerobot PR #2726** "feat(robots): Integrate Reachy Mini" by `ravediamond`, opened 2025-12-27, last update 2026-04-01, **still open**. Adds hardware/teleop + keyboard control. **No SmolVLA / no policy / no dataset.** This means Reachy Mini cannot be the target of `lerobot-train --policy=smolvla` upstream yet.
- **huggingface/lerobot issues #1316, #1370, #1791, #2259, #2915** — all SmolVLA fine-tuning threads; targets are SO-100/SO-101/Franka/LIBERO. No Reachy Mini mentions.
- `site:github.com "reachy_mini" "smolvla"` — only hit is the NVIDIA blog source (uses GR00T as building-block reference, not fine-tune).
- `huggingface/VLAb` (SmolVLA pretrain repo) — no Reachy Mini configs.

## Search 6 — arXiv

- SmolVLA paper `arXiv:2506.01844` (Jun 2025). Abstract says "range of simulated and real-world benchmarks" but does not enumerate Reachy. Companion blog/community work shows SO-100/101 + Franka + LIBERO targets only. No Reachy Mini.
- No paper found cross-referencing SmolVLA + social robot / head gaze / Reachy.

## Search 7 — Pi0 / Pi0-FAST / GR00T for Reachy Mini

- LeRobot Pi0.5 docs exist; no Reachy Mini config.
- NVIDIA × Reachy Mini blog (huggingface/blog/nvidia-reachy-mini.md) — uses **pre-trained** Nemotron Nano 3 + Nemotron Nano 2 VL + mentions GR00T N1.6 as a "building block." **No fine-tuning. No training.** It's an agent-orchestration demo on DGX Spark.
- No Pi0 Reachy Mini fine-tunes on HF.
- No GR00T Reachy Mini fine-tunes on HF.

## Search 8 — Twitter/X & Reddit

- `"reachy mini" "smolvla"` returns the Pollen launch tweet + general SmolVLA discussion. No community report of a Reachy-Mini SmolVLA fine-tune.
- No threads from Cadene / Wolf / Pollen team announcing a Reachy-Mini-specific VLA.

---

## Tangential findings (worth surfacing)

1. **SmolVLM2 (not SmolVLA) is the only "Smol*" model touching Reachy Mini today.** It's used inside `reachy_mini_conversation_app` for local vision-tool calls. Two different models — easy to confuse.

2. **Pollen has ACT policies for Reachy 2** (`act_reachy2_mobile_household_apple`, `act_reachy2_static_cup`) — so the org has the in-house skill to publish learned policies. The absence of any Reachy-Mini policy is a choice, not a capability gap. Probable reason: Reachy Mini has no arms/gripper → no manipulation task → no obvious "imitation learning" target.

3. **PR #2726 is the bottleneck.** Until Reachy Mini lands in LeRobot main, `lerobot-train` can't even ingest a Reachy Mini dataset cleanly. Any third party doing a fine-tune would have to fork — which leaves a visible fork trace; we found none.

4. **Architecture mismatch makes SmolVLA-for-head-gaze unconventional.** SmolVLA is trained on community manipulation data (SO-arms, dual-arm, mobile). A 460M flow-matching policy targeting 4-DOF head pose (pitch, yaw, roll, body_yaw) is way over-parameterized for that action space. Anyone trying it would have novelty value — meaning if it existed, it would likely be visible.

---

## Final assessment

- **Has anyone publicly fine-tuned SmolVLA for Reachy Mini? No.** Confidence: HIGH.
- **Has anyone publicly fine-tuned any VLA (Pi0/GR00T/etc.) for Reachy Mini? No.** Confidence: HIGH.
- **Does the upstream infrastructure exist? Not yet** — LeRobot PR #2726 still open, no Reachy Mini teleop dataset on Hub.
- **Closest adjacent work:** Pollen's `act_reachy2_*` (wrong robot), the conversation app's SmolVLM2 (wrong model class), NVIDIA's GR00T-as-building-block demo (no training).

If the user is considering being first to publish a SmolVLA-on-Reachy-Mini fine-tune for head trajectory: the space is genuinely empty, but the architecture/embodiment match is unusual. Would be novel datapoint regardless.
