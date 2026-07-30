#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

exec uv run python -m benchmarks.smoke --output "$1"
