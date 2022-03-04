all: test

install: FORCE
	pip install -e .[test]

lint: FORCE
	flake8
	black --check .
	isort --check .
	mypy *.py nohm test

format: FORCE
	black .
	isort .

test: lint FORCE
	pytest -vx .

FORCE:
