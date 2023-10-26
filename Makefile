
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = jfarhan-
PROJECT_NAME = Elastic-Data-Factory.git
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

SHELL := bash

.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.DEFAULT: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

build:  ## build python distributable artifacts
	python setup.py sdist bdist_wheel bdist_egg egg_info
.PHONY: build

clean-build: ## Remove build artifacts.
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
.PHONY: clean-build

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
.PHONY: venv

venv: ## Create a virtual environment
	python3.9 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade setuptools
	venv/bin/pip install --upgrade wheel
.PHONY: venv

dependencies: ## Installs packages in requirements.txt into the virtual environment
	pip install -r requirements.txt --index-url https://artifactory.foc.zone/artifactory/api/pypi/rdf-pypi-virtual/simple --extra-index-url https://artifactory.foc.zone/artifactory/api/pypi/pypi-remote/simple
.PHONY: dependencies

update-dependencies:  ## Updates all of the dependency files to the latest versions
	pip-compile requirements.in > requirements.txt
.PHONY: update-dependencies

clean-venv: ## Uninstall all packages in virtual environment.
	pip freeze | xargs pip uninstall -y
.PHONY: clean-venv

build-docs: ## Build the html documentation.
	mkdocs build
.PHONY: build-docs

view-docs: ## Start a web browser pointed at the html documentation.
	open ./site/index.html
.PHONY: view-docs

clean-docs: ## Delete all files in the /docs/build directory.
	rm -rf site
.PHONY: clean-docs

test: clean-pyc  ## Run all tests found in the /tests directory.
	py.test --verbose --color=yes ./tests
.PHONY: test

check-annotations:  ## Check type annotations of functions and methods
	flake8 elasticdatafactory --max-line-length=120 --ignore=ANN101,ANN102
.PHONY: check-annotations

check-codestyle:  ##  Check the style of the code
	pycodestyle elasticdatafactory --max-line-length=120
.PHONY: check-codestyle

check-docstyle:  ##  Check the style of the docstrings
	pydocstyle elasticdatafactory --convention=google
.PHONY: check-docstyle

check-security:  ## checks for common security vulnerabilities
	bandit -r elasticdatafactory
.PHONY: check-security

security-report:  ## checks for common security vulnerabilities and outputs a report
	mkdir -p sonar_reports
	bandit -r elasticdatafactory --format json > sonar_reports/bandit_report.json
.PHONY: security-report

convert-examples:  ## convert the example notebooks into Markdown files in docs folder
	jupyter nbconvert --to markdown examples/*.ipynb --output-dir='./docs/examples'
.PHONY: convert-examples
