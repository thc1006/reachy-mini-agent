#!/usr/bin/env bash
# install_systemd_units.sh — copy / enable / uninstall Reachy Mini systemd units.
#
# Modes:
#   --dry-run     (default) print what would happen, do not touch the filesystem
#   --install     copy units to /etc/systemd/system, run daemon-reload + enable
#   --uninstall   stop, disable, remove the units
#
# Idempotent: re-running --install only updates files that changed; --uninstall
# is a no-op if the units are already removed.
#
# --install pre-flight:
#   * verifies /opt/reachy-mini-intent-runtime/.venv/bin/python exists
#   * creates the `reachy` system user if missing
#   * creates /var/log/reachy owned by reachy:reachy
# These steps are surfaced (but skipped) by --dry-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deploy/systemd"
TARGET_DIR="/etc/systemd/system"

# Production layout invariants. Match WorkingDirectory= in deploy/systemd/*.service.
PRODUCTION_PREFIX="/opt/reachy-mini-intent-runtime"
PRODUCTION_VENV_PYTHON="${PRODUCTION_PREFIX}/.venv/bin/python"
RUNTIME_USER="reachy"
RUNTIME_GROUP="reachy"
RUNTIME_LOG_DIR="/var/log/reachy"

UNITS=(
  "reachy-runtime.slice"
  "reachy-orchestrator.service"
  "reachy-audio-listener.service"
  "reachy-motion-worker.service"
  "reachy-camera-sampler.service"
  "reachy-llm-vlm-client.service"
)

SERVICES=(
  "reachy-orchestrator.service"
  "reachy-audio-listener.service"
  "reachy-motion-worker.service"
  "reachy-camera-sampler.service"
  "reachy-llm-vlm-client.service"
)

MODE="dry-run"
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --install) MODE="install" ;;
    --uninstall) MODE="uninstall" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

log() { printf '[install_systemd_units] %s\n' "$*"; }

require_sudo() {
  if [[ $EUID -ne 0 ]]; then
    echo "error: --$1 requires root (use sudo)" >&2
    exit 3
  fi
}

verify_units_exist() {
  for u in "${UNITS[@]}"; do
    if [[ ! -f "${DEPLOY_DIR}/${u}" ]]; then
      echo "error: missing unit file: ${DEPLOY_DIR}/${u}" >&2
      exit 4
    fi
  done
}

preflight_venv() {
  # Verify the production venv interpreter exists before we install units that
  # reference it. Without this check, systemctl start would fail with a cryptic
  # `203/EXEC` error after the install supposedly succeeded.
  if [[ ! -x "${PRODUCTION_VENV_PYTHON}" ]]; then
    cat >&2 <<EOF
ERROR: ${PRODUCTION_VENV_PYTHON} not found or not executable.
Repair:
  sudo install -d -o ${RUNTIME_USER} ${PRODUCTION_PREFIX} && \\
  sudo -u ${RUNTIME_USER} git clone <repo-url> ${PRODUCTION_PREFIX} && \\
  sudo -u ${RUNTIME_USER} python3 -m venv ${PRODUCTION_PREFIX}/.venv && \\
  sudo -u ${RUNTIME_USER} ${PRODUCTION_VENV_PYTHON} -m pip install -e ${PRODUCTION_PREFIX}
Then re-run: sudo bash scripts/install_systemd_units.sh --install
EOF
    exit 5
  fi
  log "preflight: ${PRODUCTION_VENV_PYTHON} OK"
}

ensure_runtime_user() {
  if id "${RUNTIME_USER}" >/dev/null 2>&1; then
    log "user: ${RUNTIME_USER} already exists"
  else
    log "user: creating system user ${RUNTIME_USER}"
    useradd --system --no-create-home --shell /usr/sbin/nologin "${RUNTIME_USER}"
  fi
}

ensure_log_dir() {
  if [[ -d "${RUNTIME_LOG_DIR}" ]]; then
    log "log dir: ${RUNTIME_LOG_DIR} already exists"
  else
    log "log dir: creating ${RUNTIME_LOG_DIR}"
    mkdir -p "${RUNTIME_LOG_DIR}"
  fi
  chown "${RUNTIME_USER}:${RUNTIME_GROUP}" "${RUNTIME_LOG_DIR}"
}

do_dry_run() {
  log "MODE=dry-run (no changes will be made)"
  log "source dir : ${DEPLOY_DIR}"
  log "target dir : ${TARGET_DIR}"
  log "would preflight: verify ${PRODUCTION_VENV_PYTHON} exists"
  log "would: create system user ${RUNTIME_USER} (if missing): useradd --system --no-create-home --shell /usr/sbin/nologin ${RUNTIME_USER}"
  log "would: mkdir -p ${RUNTIME_LOG_DIR} && chown ${RUNTIME_USER}:${RUNTIME_GROUP} ${RUNTIME_LOG_DIR}"
  for u in "${UNITS[@]}"; do
    log "would install: ${u}"
  done
  log "would: systemctl daemon-reload"
  for s in "${SERVICES[@]}"; do
    log "would: systemctl enable ${s}"
  done
  log "done (dry-run)."
}

do_install() {
  require_sudo install
  log "MODE=install"
  preflight_venv
  ensure_runtime_user
  ensure_log_dir
  for u in "${UNITS[@]}"; do
    install -m 0644 "${DEPLOY_DIR}/${u}" "${TARGET_DIR}/${u}"
    log "installed: ${TARGET_DIR}/${u}"
  done
  systemctl daemon-reload
  for s in "${SERVICES[@]}"; do
    systemctl enable "$s"
  done
  log "done (install). Start with: systemctl start reachy-audio-listener.service (the reachy-runtime.slice auto-activates via Slice= directive)."
}

do_uninstall() {
  require_sudo uninstall
  log "MODE=uninstall"
  for s in "${SERVICES[@]}"; do
    systemctl stop "$s" 2>/dev/null || true
    systemctl disable "$s" 2>/dev/null || true
  done
  for u in "${UNITS[@]}"; do
    rm -f "${TARGET_DIR}/${u}"
    log "removed: ${TARGET_DIR}/${u}"
  done
  systemctl daemon-reload
  log "done (uninstall). Note: ${RUNTIME_USER} user and ${RUNTIME_LOG_DIR} are left in place; remove manually if desired."
}

verify_units_exist
case "$MODE" in
  dry-run)  do_dry_run ;;
  install)  do_install ;;
  uninstall) do_uninstall ;;
esac
