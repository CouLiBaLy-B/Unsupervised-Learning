.PHONY: install install-dev test lint format clean help run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linters (ruff, black)"
	@echo "  make format        - Auto-format code (ruff --fix, black)"
	@echo "  make clean         - Remove cache files"
	@echo "  make run           - Run the Streamlit app"
	@echo "  make all           - Install deps + format + lint + test"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install -r requirements-dev.txt

# Run tests with coverage
test:
	python -m pytest tests/ -v --cov=src/simulations --cov=src/models --cov=src/ml

# Run linters
lint:
	ruff check src/ pages/ tests/
	black --check src/ pages/ tests/

# Auto-format code
format:
	ruff check src/ pages/ tests/ --fix
	black src/ pages/ tests/

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run the Streamlit app
run:
	streamlit run pages/home.py

# Full pipeline
all: install-dev format lint test
