#!/bin/bash

# 🚨 Enterprise Fraud Detection System - Complete System Startup
# This script starts the entire fraud detection system

set -e

echo "🚨 Starting Enterprise Fraud Detection System..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs models data/samples data/results

# Create external volume for PostgreSQL if it doesn't exist
if ! docker volume ls | grep -q "fraud-detection-system_postgres_data"; then
    print_status "Creating PostgreSQL external volume..."
    docker volume create fraud-detection-system_postgres_data
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose down --remove-orphans

# Pull latest images
print_status "Pulling latest Docker images..."
docker-compose pull

# Build custom images
print_status "Building custom Docker images..."
docker-compose build --no-cache

# Start core databases first
print_status "Starting core databases..."
docker-compose up -d postgres neo4j timescaledb redis elasticsearch

# Wait for databases to be ready
print_status "Waiting for databases to initialize..."
sleep 30

# Check database health
print_status "Checking database health..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U fraud_admin -d fraud_detection > /dev/null 2>&1; then
        print_success "PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "PostgreSQL failed to start"
        exit 1
    fi
    sleep 2
done

# Start monitoring services
print_status "Starting monitoring services..."
docker-compose up -d prometheus grafana

# Start processing services
print_status "Starting processing services..."
docker-compose up -d kafka zookeeper flink-jobmanager flink-taskmanager

# Start application services
print_status "Starting application services..."
docker-compose up -d fraud-api

# Wait for API to be ready
print_status "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8080/api/health > /dev/null 2>&1; then
        print_success "API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "API failed to start"
        exit 1
    fi
    sleep 2
done

# Start frontend
print_status "Starting Streamlit frontend..."
docker-compose up -d fraud-frontend

# Start additional services
print_status "Starting additional services..."
docker-compose up -d feature-engine ml-trainer

# Final health check
print_status "Performing final health check..."
sleep 10

# Check all services
services=("postgres" "neo4j" "timescaledb" "redis" "elasticsearch" "fraud-api" "fraud-frontend")
all_healthy=true

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "${service}.*Up"; then
        print_success "${service} is running"
    else
        print_error "${service} is not running"
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    print_success "All services are running successfully!"
    echo ""
    echo "🎉 Enterprise Fraud Detection System is now running!"
    echo "=================================================="
    echo ""
    echo "📊 Access Points:"
    echo "  • Frontend Dashboard: http://localhost:3000"
    echo "  • API Documentation: http://localhost:8080/docs"
    echo "  • API Health Check:  http://localhost:8080/api/health"
    echo ""
    echo "🗄️  Databases:"
    echo "  • PostgreSQL:        localhost:5434"
    echo "  • Neo4j Browser:     http://localhost:7474"
    echo "  • TimescaleDB:       localhost:5433"
    echo "  • Redis:             localhost:6379"
    echo "  • Elasticsearch:     http://localhost:9200"
    echo ""
    echo "📈 Monitoring:"
    echo "  • Prometheus:        http://localhost:9090"
    echo "  • Grafana:           http://localhost:3001"
    echo "  • Redis Insight:     http://localhost:8001"
    echo ""
    echo "🔧 Management:"
    echo "  • View logs:         docker-compose logs -f"
    echo "  • Stop system:       docker-compose down"
    echo "  • Restart service:   docker-compose restart <service>"
    echo ""
    echo "🚀 System is ready for fraud detection!"
else
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi