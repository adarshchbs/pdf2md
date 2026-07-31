# Quality gates for pdf2md. `make check` applies safe formatting and import fixes,
# then verifies lint and strict type checking.

PY_PATHS := app/pdf2md app/upload_endpoint.py app/comparison_router.py app/main.py tests

.PHONY: check fmt lint type test all sync lock

sync:
	uv sync --all-groups

lock:
	uv lock --check

check: fmt lint type

fmt:
	uv run ruff format $(PY_PATHS)
	uv run ruff check --fix $(PY_PATHS)

lint:
	uv run ruff format --check $(PY_PATHS)
	uv run ruff check $(PY_PATHS)

type:
	uv run basedpyright

test:
	uv run pytest

all: check test
