#!/usr/bin/env bash
set -euo pipefail
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[dev]'
./scripts/verify.sh
