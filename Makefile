# 🚨 Streamlit Fraud Detection System - Makefile

.PHONY: help install clean start dev docker test lint format

# Default target
help:
	@echo "🚨 Streamlit Fraud Detection System - Available Commands"
	@echo "========================================================"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install     - Install all dependencies"
	@echo "  make clean       - Clean up temporary files"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make start       - Start Streamlit dashboard"
	@echo "  make dev         - Start in development mode"
	@echo "  make quick       - Start quick CSV analyzer"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker      - Start with Docker Compose"
	@echo "  make docker-down - Stop Docker containers"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo ""

# Installation
install:
	@echo "📦 Installing Streamlit Fraud Detection System..."
	python3 -m venv streamlit-env
	./streamlit-env/bin/pip install --upgrade pip
	./streamlit-env/bin/pip install -r requirements.txt
	@echo "✅ Installation complete!"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf .DS_Store
	rm -rf logs/*.log
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

# Start main dashboard
start:
	@echo "🚀 Starting Streamlit Dashboard..."
	./scripts/start.sh

# Development mode
dev:
	@echo "🔧 Starting in development mode..."
	source streamlit-env/bin/activate && \
	streamlit run app/main.py --server.port 8502 --server.runOnSave true

# Quick CSV analyzer
quick:
	@echo "⚡ Starting Quick CSV Analyzer..."
	source streamlit-env/bin/activate && \
	streamlit run app/quick_analyzer.py --server.port 8503

# Setup helper
setup:
	@echo "🔧 Starting Setup Helper..."
	source streamlit-env/bin/activate && \
	streamlit run app/setup_helper.py --server.port 8504

# Docker commands
docker:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f docker/docker-compose.yml down

docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose -f docker/docker-compose.yml build

# Testing
test:
	@echo "🧪 Running tests..."
	source streamlit-env/bin/activate && \
	python -m pytest tests/ -v

# Linting
lint:
	@echo "🔍 Running linting..."
	source streamlit-env/bin/activate && \
	flake8 *.py --max-line-length=100 --ignore=E501,W503

# Code formatting
format:
	@echo "✨ Formatting code..."
	source streamlit-env/bin/activate && \
	black *.py --line-length=100

# Create sample data
sample-data:
	@echo "📊 Creating sample data..."
	mkdir -p data
	source streamlit-env/bin/activate && \
	python scripts/generate_data.py

# Check system
check:
	@echo "🔍 Checking system requirements..."
	@python3 --version
	@pip3 --version
	@echo "Checking required files..."
	@ls -la app/main.py app/fraud_processor.py requirements.txt

# Install pre-commit hooks
hooks:
	@echo "🪝 Installing pre-commit hooks..."
	source streamlit-env/bin/activate && \
	pip install pre-commit && \
	pre-commit install

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	source streamlit-env/bin/activate && \
	pip install --upgrade -r requirements.txt

# Create deployment package
package:
	@echo "📦 Creating deployment package..."
	mkdir -p dist
	tar -czf dist/streamlit-fraud-detection-$(shell date +%Y%m%d).tar.gz \
		app/ scripts/ docker/ requirements.txt Makefile \
		data/ --exclude=data/massive/ --exclude=streamlit-env/

# Show logs
logs:
	@echo "📋 Showing recent logs..."
	tail -f logs/*.log 2>/dev/null || echo "No logs found"

# System info
info:
	@echo "💻 System Information:"
	@echo "====================="
	@echo "OS: $(shell uname -s)"
	@echo "Python: $(shell python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Streamlit: $(shell source streamlit-env/bin/activate && streamlit version 2>/dev/null || echo 'Not installed')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not found')"
	@echo "Current directory: $(PWD)"
	@echo "Virtual env: $(shell [ -d streamlit-env ] && echo '✅ Exists' || echo '❌ Not found')"
