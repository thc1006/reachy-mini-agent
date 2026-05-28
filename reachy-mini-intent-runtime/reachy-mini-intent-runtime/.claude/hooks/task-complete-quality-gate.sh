#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ./scripts/verify.sh ]]; then
  echo "quality-gate: ./scripts/verify.sh missing or not executable" >&2
  exit 2
fi

./scripts/verify.sh
