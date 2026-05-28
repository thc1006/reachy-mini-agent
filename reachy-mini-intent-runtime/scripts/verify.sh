#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
ruff check .
ruff format --check .
