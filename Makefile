.PHONY: all clean pytest coverage flake8 black mypy isort

CMD:=poetry run
PYMODULE:=src
TESTS:=tests

install:
	poetry install
	poetry run pre-commit install

# Run the unit tests using `pytest`
pytest:
	$(CMD) pytest $(PYMODULE) $(TESTS)

lint:
	$(CMD) ruff $(PYMODULE) --config .ruff.toml
	$(CMD) ruff $(TESTS) --config .ruff.toml

# Generate a unit test coverage report using `pytest-cov`
coverage:
	$(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html

# Perform static type checking using `mypy`
mypy:
	$(CMD) mypy $(PYMODULE) $(TESTS)

# Generate a setup.py file from pyproject.toml
setup.py: pyproject.toml
	$(CMD) dephell deps convert

# Check all the files against pre-commit hooks
check:
	$(CMD) pre-commit run