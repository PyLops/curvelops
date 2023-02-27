PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
PYTEST := $(shell command -v pytest 2> /dev/null)

.PHONY: install dev-install tests doc watchdoc servedoc lint typeannot coverage

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

pytestcheck:
ifndef PYTEST
	$(error "Ensure pytest is in your PATH")
endif
	@echo Using pytest: $(PYTEST)

install:
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt && $(PIP) install -e .

tests:
	make pytestcheck
	$(PYTEST) tests

doc:
	cd docs && rm -rf build && sphinx-apidoc -f -M -o source/ ../curvelops && make html && cd -
	# Add -P to sphinx-apidoc to include private files

watchdoc:
	while inotifywait -q -r curvelops/ -e create,delete,modify; do { make doc; }; done

servedoc:
	$(PYTHON) -m http.server --directory docs/build/html/

lint:
	flake8 docs/ curvelops/ tests/

typeannot:
	mypy curvelops/

coverage:
	coverage run -m pytest && coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/
