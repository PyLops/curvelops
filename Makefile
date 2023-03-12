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

lint:
	flake8 docs/ curvelops/ tests/

typeannot:
	mypy curvelops/

coverage:
	coverage run -m pytest && coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/

watchdoc:
	while inotifywait -q -r curvelops/ docssrc/source/ -e create,delete,modify; do { make docupdate; }; done

servedoc:
	$(PYTHON) -m http.server --directory docssrc/build/html/

doc:
    # Add after rm: sphinx-apidoc -f -M -o source/ ../curvelops
    # Use -O to include private files
	cd docssrc  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && make html && cd ..

docupdate:
	cd docssrc && make html && cd ..

docgithub:
	cd docssrc && make github && cd ..

docpush:
	git checkout gh-pages && git merge gh-pages-triage && cd docssrc && make github &&\
	cd ../docs && git add . && git commit -m "Updated documentation" &&\
	git push origin gh-pages && git checkout gh-pages-triage
