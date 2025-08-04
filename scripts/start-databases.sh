#!/bin/bash
# Database Setup Script for Local Development
# Starts all database services using Docker Compose

echo "🚀 Starting Fraud Detection Database Infrastructure..."
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/samples
mkdir -p data/results
mkdir -p logs
mkdir -p models
mkdir -p config/grafana/dashboards
mkdir -p config/grafana/datasources

# Start core databases first
echo "🗄️  Starting core databases..."
docker-compose -f docker-compose-future.yml up -d postgres timescaledb redis elasticsearch

# Wait for databases to be healthy
echo "⏳ Waiting for databases to be ready..."
echo "   - PostgreSQL..."
until docker-compose -f docker-compose-future.yml exec -T postgres pg_isready -U fraud_admin -d fraud_detection; do
    sleep 2
done

echo "   - TimescaleDB..."
until docker-compose -f docker-compose-future.yml exec -T timescaledb pg_isready -U timescale_admin -d fraud_metrics; do
    sleep 2
done

echo "   - Redis..."
until docker-compose -f docker-compose-future.yml exec -T redis redis-cli --no-auth-warning -a RedisStack2024! ping; do
    sleep 2
done

echo "   - Elasticsearch..."
until curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; do
    sleep 5
done

# Start Neo4j
echo "🔗 Starting Neo4j..."
docker-compose -f docker-compose-future.yml up -d neo4j

echo "   - Neo4j..."
sleep 10  # Neo4j takes a bit longer to start
until docker-compose -f docker-compose-future.yml exec -T neo4j cypher-shell -u neo4j -p FraudGraph2024! "RETURN 1" > /dev/null 2>&1; do
    sleep 5
done

# Start streaming services
echo "📡 Starting streaming services..."
docker-compose -f docker-compose-future.yml up -d zookeeper kafka
sleep 10

echo "🔄 Starting stream processing..."
docker-compose -f docker-compose-future.yml up -d flink-jobmanager flink-taskmanager

# Start monitoring
echo "📊 Starting monitoring services..."
docker-compose -f docker-compose-future.yml up -d prometheus grafana

echo ""
echo "✅ All services started successfully!"
echo "=================================================="
echo "🌐 Available Services:"
echo "   PostgreSQL:      localhost:5432 (fraud_admin/FraudDetection2024!)"
echo "   Neo4j Browser:   http://localhost:7474 (neo4j/FraudGraph2024!)"
echo "   Neo4j Bolt:      bolt://localhost:7687"
echo "   TimescaleDB:     localhost:5433 (timescale_admin/TimeScale2024!)"
echo "   Redis:           localhost:6379 (password: RedisStack2024!)"
echo "   RedisInsight:    http://localhost:8001"
echo "   Elasticsearch:   http://localhost:9200"
echo "   Kafka:           localhost:29092"
echo "   Flink Dashboard: http://localhost:8081"
echo "   Prometheus:      http://localhost:9090"
echo "   Grafana:         http://localhost:3000 (admin/GrafanaAdmin2024!)"
echo ""
echo "📚 Next steps:"
echo "   1. Run database initialization scripts"
echo "   2. Load sample data"
echo "   3. Start application services"
echo ""
echo "🛑 To stop all services: docker-compose -f docker-compose-future.yml down"
