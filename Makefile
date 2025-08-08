# Fraud Detection System - Makefile
# Development and deployment automation

.PHONY: help install install-dev install-test clean lint format test test-unit test-integration test-frontend test-security test-performance build run run-dev run-prod docker-build docker-run docker-compose-up docker-compose-down deploy health-check logs backup restore

# Default target
help:
	@echo "Fraud Detection System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-test     Install testing dependencies"
	@echo "  clean           Clean up temporary files and caches"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black and isort"
	@echo "  type-check      Run type checking with mypy"
	@echo "  security-check  Run security analysis"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-frontend   Run frontend tests only"
	@echo "  test-security   Run security tests"
	@echo "  test-performance Run performance tests"
	@echo "  test-coverage   Generate test coverage report"
	@echo ""
	@echo "Development:"
	@echo "  run             Run the application (development mode)"
	@echo "  run-api         Run API server only"
	@echo "  run-frontend    Run Streamlit frontend only"
	@echo "  run-redis       Start Redis server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-run      Run application in Docker"
	@echo "  docker-compose-up   Start all services with docker-compose"
	@echo "  docker-compose-down Stop all services"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy          Deploy to production"
	@echo "  health-check    Check application health"
	@echo "  logs            View application logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  backup          Backup application data"
	@echo "  restore         Restore application data"
	@echo "  update-deps     Update dependencies"

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
SAFETY := safety
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Directories
SRC_DIR := .
BACKEND_DIR := backend
TESTS_DIR := tests
DOCS_DIR := docs

# Docker settings
DOCKER_IMAGE_API := fraud-detection-api
DOCKER_IMAGE_FRONTEND := fraud-detection-frontend
DOCKER_TAG := latest

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Development environment
dev:
	@echo "🚀 Starting development environment..."
	@echo "Starting PostgreSQL and Redis..."
	docker-compose up -d postgres redis neo4j timescaledb elasticsearch
	@echo "Starting API backend..."
	cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8080 &
	@echo "Starting Next.js frontend..."
	cd frontend && npm run dev &
	@echo "✅ Development environment ready:"
	@echo "   Frontend: http://localhost:8501"
	@echo "   API: http://localhost:8080"

# Start API only
api:
	@echo "🔌 Starting FastAPI backend..."
	cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8080

# Start frontend only
frontend:
	@echo "📊 Starting Next.js frontend..."
	cd frontend && npm run dev

# Docker deployment
docker:
	@echo "🐳 Starting Docker deployment..."
	docker-compose up -d
	@echo "✅ Services started:"
	@echo "   Frontend: http://localhost:8501"
	@echo "   API: http://localhost:8080"
	@echo "   PostgreSQL: localhost:5432"
	@echo "   Neo4j: localhost:7474"
	@echo "   Redis: localhost:6379"
	@echo "   TimescaleDB: localhost:5433"
	@echo "   Elasticsearch: localhost:9200"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3000"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build
	@echo "✅ Docker images built"

# Run comprehensive tests
test:
	@echo "🧪 Running comprehensive test suite..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests completed. Coverage report: htmlcov/index.html"

# Code linting
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ --max-line-length=100 --exclude=__pycache__
	@echo "✅ Linting completed"

# Code formatting
format:
	@echo "🎨 Formatting code with Black..."
	black src/ --line-length 100
	@echo "✅ Code formatted"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation available in docs/ directory"

# Production deployment
deploy:
	@echo "🚀 Deploying to production..."
	@echo "Building production images..."
	docker-compose build
	@echo "✅ Production deployment ready"

# Database partition management
partition-status:
	@echo "📊 Checking partition status..."
	docker-compose exec postgres psql -U fraud_user -d fraud_detection -c "SELECT * FROM v_partition_status ORDER BY partition_date DESC LIMIT 10;"
	@echo "✅ Partition status check completed"

# Create future partitions manually
partition-create:
	@echo "🔧 Creating future partitions..."
	docker-compose exec postgres psql -U fraud_user -d fraud_detection -c "SELECT create_monthly_partitions('transactions', CURRENT_DATE, 6);"
	@echo "✅ Future partitions created"

# Partition health check
partition-health:
	@echo "🏥 Running partition health check..."
	docker-compose exec postgres psql -U fraud_user -d fraud_detection -c "SELECT * FROM partition_health_check();"
	@echo "✅ Partition health check completed"

# Database migration
migrate:
	@echo "🔄 Running database migration..."
	python scripts/migrate_database.py
	@echo "✅ Database migration completed"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed"

# System health check
health:
	@echo "🏥 System health check..."
	@echo "Checking services..."
	docker-compose ps

# View logs
logs:
	@echo "📋 Application logs:"
	docker-compose logs -f --tail=100

# Stop all services
stop:
	@echo "🛑 Stopping all services..."
	docker-compose down
	@echo "✅ All services stopped"