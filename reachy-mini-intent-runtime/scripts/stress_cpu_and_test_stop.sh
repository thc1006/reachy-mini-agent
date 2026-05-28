#!/usr/bin/env bash
# stress_cpu_and_test_stop.sh — spawn CPU-bound workers + measure stop dispatch
#
# Default: --mock  (in-process multiprocessing; portable; no systemd / root needed)
# Opt-in:  --real-hardware  (would target the running Reachy Mini stack)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="mock"
WORKERS=4
DURATION_S=2
for arg in "$@"; do
  case "$arg" in
    --mock) MODE="mock" ;;
    --real-hardware) MODE="real" ;;
    --workers=*) WORKERS="${arg#*=}" ;;
    --duration=*) DURATION_S="${arg#*=}" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-python}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
  PY="${REPO_ROOT}/.venv/Scripts/python.exe"
fi

echo "[stress_cpu_and_test_stop] MODE=${MODE} workers=${WORKERS} duration=${DURATION_S}s"

if [[ "$MODE" == "real" ]]; then
  echo "[stress_cpu_and_test_stop] real-hardware mode is opt-in and not implemented in this stub." >&2
  exit 2
fi

# Invoke a real .py file so multiprocessing.spawn (Windows) can re-import.
"$PY" "${SCRIPT_DIR}/_stress_cpu_and_test_stop.py" --workers="$WORKERS" --duration="$DURATION_S"
