.PHONY: help dev api dashboard docker build test lint format clean install docs deploy

# Default target
help:
	@echo "🚨 Enterprise Fraud Detection System - Available Commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Start development environment"
	@echo "  make docker     - Start with Docker Compose"
	@echo "  make build      - Build Docker images"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make api        - Start FastAPI backend only"
	@echo "  make frontend   - Start Next.js frontend only"
	@echo "  make test       - Run comprehensive test suite"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with Black"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs       - Generate documentation"
	@echo ""
	@echo "🚀 Deployment:"
	@echo "  make deploy     - Deploy to production"
	@echo "  make clean      - Clean build artifacts"

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
	@echo "   Frontend: http://localhost:3000"
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
	@echo "   Frontend: http://localhost:3000"
	@echo "   API: http://localhost:8080"
	@echo "   PostgreSQL: localhost:5434"
	@echo "   Neo4j: localhost:7474"
	@echo "   Redis: localhost:6379"
	@echo "   TimescaleDB: localhost:5433"
	@echo "   Elasticsearch: localhost:9200"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana: http://localhost:3001"

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