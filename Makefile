.PHONY: install test lint clean

install:
	pip install -e '.[dev]'

test:
	python -m pytest -vv

lint:
	python -m compileall -q src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
