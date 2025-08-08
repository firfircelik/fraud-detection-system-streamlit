#!/bin/bash
# Simple Database Setup - Core databases only
echo "🚀 Starting Core Databases..."

# Check Docker
if ! docker --version > /dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data logs models config/grafana/{dashboards,datasources}

# Start only PostgreSQL first
echo "🗄️ Starting PostgreSQL..."
docker-compose up -d postgres

# Wait for PostgreSQL
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U fraud_admin -d fraud_detection 2>/dev/null; do
    echo "   PostgreSQL starting..."
    sleep 3
done
echo "✅ PostgreSQL is ready!"

# Start Neo4j
echo "🔗 Starting Neo4j..."
docker-compose up -d neo4j

# Wait for Neo4j
echo "⏳ Waiting for Neo4j to be ready..."
sleep 15
until docker-compose exec -T neo4j cypher-shell -u neo4j -p FraudGraph2024! "RETURN 1" 2>/dev/null; do
    echo "   Neo4j starting..."
    sleep 5
done
echo "✅ Neo4j is ready!"

# Start Redis
echo "🔴 Starting Redis..."
docker-compose up -d redis

# Wait for Redis
echo "⏳ Waiting for Redis to be ready..."
until docker-compose exec -T redis redis-cli --no-auth-warning -a RedisStack2024! ping 2>/dev/null; do
    echo "   Redis starting..."
    sleep 2
done
echo "✅ Redis is ready!"

echo ""
echo "🎉 Core databases are running!"
echo "================================"
echo "PostgreSQL: localhost:5432 (fraud_admin/FraudDetection2024!)"
echo "Neo4j Browser: http://localhost:7474 (neo4j/FraudGraph2024!)"
echo "Redis: localhost:6379 (RedisStack2024!)"
echo ""
echo "✨ You can now test database connections!"
