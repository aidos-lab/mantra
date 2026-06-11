.PHONY: tests coverage notebooks

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

