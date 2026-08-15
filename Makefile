.PHONY: test integration_test lint format build check

test:
	uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/

integration_test:
	uv run --group test --group test_integration pytest -n auto tests/integration_tests/

lint:
	uv run --group lint ruff check .
	uv run --group lint mypy langchain_typesense
	uv run --group lint ty check

format:
	uv run --group lint ruff check --fix .
	uv run --group lint ruff format .

build:
	uv build

check: lint test build
