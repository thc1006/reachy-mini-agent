#!/usr/bin/env bash
# robot_brain_watchdog.sh — restart robot-brain.service if its log goes silent
#
# We've observed (2026-05-13, 2026-05-15) that robot-brain occasionally
# enters a zombie state where:
#   - systemd shows the service "active (running)"
#   - the Python process is alive, often consuming CPU
#   - but no log lines are written for many minutes
#   - root cause is usually a WebRTC pipeline disconnect (gstreamer
#     End-of-stream) or an LLM stall that blocks the main loop, leaving
#     the rest of the threads spinning on Lost-connection retries
#
# This watchdog runs every minute (via robot-brain-watchdog.timer) and
# triggers a `systemctl --user restart robot-brain.service` if the log
# has been silent for too long AND the daemon endpoint is still reachable
# (i.e. the network/daemon side is fine, the brain is the one stuck).
#
# Threshold: 120 s. Even an idle robot writes a `[視覺]` line every 30 s
# from the vision worker, so 120 s without writes strongly indicates a
# stall. The threshold can be tuned via WATCHDOG_THRESHOLD env.

set -euo pipefail

THRESHOLD="${WATCHDOG_THRESHOLD:-120}"
LOG_FILE="${ROBOT_BRAIN_LOG:-/home/reachym/dev/reachy-agent/robot/logs/robot_brain.log}"
DAEMON_HOST="${REACHY_HOST:-100.85.191.3}"
DAEMON_URL="http://${DAEMON_HOST}:8000"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] watchdog: $*"; }

# 1. Daemon reachable? If not, the brain restart wouldn't help anyway.
if ! curl -sf -m 3 "${DAEMON_URL}/api/state/present_head_pose" > /dev/null 2>&1; then
    log "daemon unreachable, skipping check"
    exit 0
fi

# 2. Log file exists?
if [ ! -f "$LOG_FILE" ]; then
    log "log file not found: $LOG_FILE"
    exit 0
fi

# 3. Compute log mtime age.
mtime=$(stat -c %Y "$LOG_FILE")
now=$(date +%s)
age=$((now - mtime))

if [ "$age" -le "$THRESHOLD" ]; then
    # All good. Silent if healthy to avoid log spam.
    exit 0
fi

# 4. Stall detected. Restart.
log "log silent for ${age}s (>${THRESHOLD}s threshold), restarting robot-brain.service"
systemctl --user restart robot-brain.service
log "restart issued. waiting 5s to confirm new PID..."
sleep 5
new_pid=$(systemctl --user show robot-brain.service -p MainPID --value 2>/dev/null || echo "?")
log "new MainPID=${new_pid}"
