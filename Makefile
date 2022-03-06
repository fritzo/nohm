all: test

install: FORCE
	pip install -e .[test]

lint: FORCE
	python -m flake8
	python -m black --check .
	python -m isort --check .
	python -m mypy --install-types --non-interactive *.py nohm test

format: FORCE
	python -m black .
	python -m isort .

test: lint FORCE
	python -m pytest -vx .

FORCE:
