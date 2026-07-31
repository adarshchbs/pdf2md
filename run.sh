pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build
uv run uvicorn app.main:app --reload --port 8010
