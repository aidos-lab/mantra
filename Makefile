.PHONY: tests coverage check lint format-check format hooks release notebooks convert

check: lint format-check tests

lint:
	uv run ruff check .

format-check:
	uv run black --check .

format:
	uv run black .

hooks:
	uv run pre-commit install

tests:
	uv run pytest

coverage:
	uv run coverage run -m pytest
	uv run coverage report

release:
	uv build --wheel

notebooks:
	rm -r docs/notebooks/
	uv run jupyter nbconvert --execute --output-dir=docs/notebooks --to markdown examples/*.ipynb

convert:
	rm -r docs/notebooks/
	uv run jupyter nbconvert --output-dir=docs/notebooks --to markdown examples/*.ipynb
