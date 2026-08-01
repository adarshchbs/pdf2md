#!/usr/bin/env bash
set -euo pipefail

pnpm --dir frontend install --frozen-lockfile

uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8010 &
backend_pid=$!
pnpm --dir frontend exec vite --host 127.0.0.1 &
frontend_pid=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$frontend_pid" "$backend_pid" 2>/dev/null || true
  wait "$frontend_pid" "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Browse the Vite URL. It hot-reloads UI code and proxies /comparison to FastAPI.
wait "$frontend_pid"
