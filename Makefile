# Makefile for ANTsPyMM
# Multi-channel/time-series medical image processing with ANTsPy

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= ruff

.PHONY: help install install-dev develop test test-strict test-cov lint lint-fix compile audit build clean clean-all report

# Default target
help:
	@echo "ANTsPyMM Development & Maintenance Tasks"
	@echo "========================================="
	@echo "  make install       Install antspymm into the current environment"
	@echo "  make install-dev   Install antspymm in editable mode (-e .)"
	@echo "  make test          Run unit test suite via pytest"
	@echo "  make test-strict   Run pytest treating deprecation warnings as fatal errors"
	@echo "  make test-cov      Run pytest with statement coverage report"
	@echo "  make lint          Run code quality and style checks using ruff"
	@echo "  make lint-fix      Run ruff and automatically apply safe fixes"
	@echo "  make compile       Verify bytecode compilation without SyntaxWarnings"
	@echo "  make audit         Run full verification gate (compile + lint + test)"
	@echo "  make build         Build source distribution and binary wheel"
	@echo "  make clean         Remove build artifacts, caches, and compiled bytecode"
	@echo "  make clean-all     Remove all caches and agent scratch directories"
	@echo "  make report        Open visual HTML audit reports in browser (macOS)"

install:
	$(PIP) install .

install-dev: develop

develop:
	$(PIP) install -e .

test:
	$(PYTEST) -v tests/

test-strict:
	$(PYTEST) -v -W error::DeprecationWarning -W error::FutureWarning tests/

test-cov:
	$(PYTEST) -v --cov=antspymm --cov-report=term-missing tests/

lint:
	$(RUFF) check antspymm tests

lint-fix:
	$(RUFF) check --fix antspymm tests

compile:
	$(PYTHON) -W error::SyntaxWarning -m compileall antspymm tests docs

audit: compile
	$(RUFF) check antspymm tests
	$(PYTEST) -v tests/

build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*$$py.class" -delete

clean-all: clean
	rm -rf .agents/

report:
	@if [ -f "AUDIT_REPORT.html" ]; then \
		echo "Opening AUDIT_REPORT.html..."; \
		open AUDIT_REPORT.html; \
	elif [ -f "reports/code_audit_report.html" ]; then \
		echo "Opening reports/code_audit_report.html..."; \
		open reports/code_audit_report.html; \
	fi
