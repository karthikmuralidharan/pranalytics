# Makefile for GitHub PR Velocity Analytics Tool

.PHONY: help install install-dev test test-coverage lint format clean build dist demo standalone verify-install dev-setup

# Default target
help:
	@echo "GitHub PR Velocity Analytics Tool - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install        Install package dependencies with UV"
	@echo "  install-dev    Install development dependencies with UV"
	@echo "  test          Run unit tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run code quality checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  dist          Create distribution ready package"
	@echo "  demo          Run demo analysis on sample repository"
	@echo "  standalone    Make script executable"
	@echo "  verify-install Verify installation"
	@echo "  dev-setup     Complete development environment setup"

# Installation targets using UV
install:
	uv sync

install-dev:
	uv sync --dev

# Testing targets
test:
	uv run pytest test_pr_velocity_analytics.py -v

test-coverage:
	uv run pytest test_pr_velocity_analytics.py --cov=pr_velocity_analytics --cov-report=html --cov-report=term

# Code quality targets
lint:
	uv run flake8 pr_velocity_analytics.py test_pr_velocity_analytics.py
	uv run mypy pr_velocity_analytics.py

format:
	uv run black pr_velocity_analytics.py test_pr_velocity_analytics.py
	uv run isort pr_velocity_analytics.py test_pr_velocity_analytics.py

# Build and distribution targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .uv/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

build: clean
	uv build

dist: build
	@echo "Distribution packages created in dist/"
	@ls -la dist/

# Demo target
demo:
	@echo "Running demo analysis on microsoft/vscode repository..."
	@echo "Note: Requires GITHUB_TOKEN environment variable"
	uv run python pr_velocity_analytics.py microsoft/vscode --tags bug feature --since 30d --output demo_report.csv

# Standalone script creation
standalone:
	@echo "Creating standalone executable script..."
	chmod +x pr_velocity_analytics.py
	@echo "Script is now executable. Run with: ./pr_velocity_analytics.py"

# Installation verification
verify-install:
	@echo "Verifying installation..."
	uv run python -c "import pr_velocity_analytics; print('✓ Module imports successfully')"
	uv run python pr_velocity_analytics.py --help
	@echo "✓ Installation verified successfully"

# Development setup
dev-setup: install-dev
	@echo "Setting up pre-commit hooks..."
	uv run pre-commit install
	@echo "✓ Development environment ready"

# Additional UV-specific targets
uv-lock:
	uv lock

uv-update:
	uv lock --upgrade

uv-tree:
	uv tree