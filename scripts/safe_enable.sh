#!/usr/bin/env bash
# safe_enable.sh — soft motor-enable for Reachy Mini
#
# Default REST endpoint /api/motors/set_mode/enabled snaps the head at full
# servo speed to whatever the daemon's stored target was (which can be very
# different from the head's gravity-settled physical pose, especially if
# face_tracker had been writing set_target while motors were disabled in a
# previous session). This script avoids the snap by:
#
#   1. Verify motors are disabled (skip work if already enabled)
#   2. Wait briefly for the head to settle under gravity (1.5 s)
#   3. Query the current physical pose
#   4. Overwrite daemon's stored target with that pose (set_target)
#   5. Enable motors (now the snap distance is ~0)
#   6. Optionally smooth-goto to neutral over 3 s
#
# Empirical: with this sequence the enable snap rate dropped from ~42°/s to
# ~10°/s on 2026-05-15. Residual is from disable→enable network latency drift.
#
# Usage:
#   ./safe_enable.sh                 # soft enable, leave head where it settles
#   ./safe_enable.sh --goto-neutral  # soft enable + smooth move to neutral
#
# Requires curl + python3 on the calling host, and ssh access to the robot
# implied via REACHY_HOST (default: reachy-mini.local).

set -euo pipefail

HOST="${REACHY_HOST:-reachy-mini.local}"
BASE="http://${HOST}:8000"

# 1. Skip if already enabled
mode=$(curl -sf "$BASE/api/motors/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["mode"])')
if [ "$mode" = "enabled" ]; then
    echo "[safe_enable] motors already enabled — nothing to do"
    exit 0
fi
echo "[safe_enable] motors disabled — proceeding"

# 2. Settle wait
echo "[safe_enable] waiting 1.5 s for gravity settle…"
sleep 1.5

# 3. Read settled pose
POSE=$(curl -sf "$BASE/api/state/present_head_pose")
PITCH=$(echo "$POSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pitch"])')
YAW=$(echo "$POSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["yaw"])')
ROLL=$(echo "$POSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["roll"])')
echo "[safe_enable] settled pose: pitch=$(python3 -c "print(f'{$PITCH*57.3:+.1f}')")° yaw=$(python3 -c "print(f'{$YAW*57.3:+.1f}')")° roll=$(python3 -c "print(f'{$ROLL*57.3:+.1f}')")°"

# 4. Pre-set daemon target = settled pose
curl -sf -X POST "$BASE/api/move/set_target" \
    -H "Content-Type: application/json" \
    -d "{\"target_head_pose\":{\"x\":0,\"y\":0,\"z\":0,\"roll\":$ROLL,\"pitch\":$PITCH,\"yaw\":$YAW},\"target_body_yaw\":0,\"target_antennas\":[0,0]}" \
    > /dev/null
echo "[safe_enable] daemon target = current pose"

# 5. Enable
curl -sf -X POST "$BASE/api/motors/set_mode/enabled" \
    -H "Content-Type: application/json" -d '{}' > /dev/null
echo "[safe_enable] motors enabled (no snap)"

# 6. Optional smooth goto to neutral
if [ "${1:-}" = "--goto-neutral" ]; then
    echo "[safe_enable] smooth goto neutral (3 s)…"
    curl -sf -X POST "$BASE/api/move/goto" \
        -H "Content-Type: application/json" \
        -d '{"head_pose":{"x":0,"y":0,"z":0,"roll":0,"pitch":0,"yaw":0},"duration":3.0,"interpolation":"minjerk"}' \
        > /dev/null
    sleep 3.2
    echo "[safe_enable] done"
fi
