#!/bin/bash

# 🚀 Fraud Detection System - Docker Startup Script

echo "🎯 Starting Fraud Detection System with Docker..."
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/massive
mkdir -p logs
mkdir -p config
mkdir -p prometheus
mkdir -p grafana/dashboards
mkdir -p grafana/datasources
mkdir -p nginx
mkdir -p sql

# Generate some sample data first
echo "📊 Generating sample transaction data..."
if [ ! -f "data/sample_transactions.csv" ]; then
    cat > data/sample_transactions.csv << 'EOF'
transaction_id,user_id,amount,currency,merchant_id,category,timestamp,lat,lon,device_id,ip_address,user_age,user_income,transaction_hour,transaction_day,is_fraud
550e8400-e29b-41d4-a716-446655440000,USER_000000001,99.99,USD,MERCHANT_00000001,electronics,2024-01-15T10:30:00Z,40.7128,-74.0060,DEVICE_0000000001,192.168.1.100,35,medium,10,1,0
550e8400-e29b-41d4-a716-446655440001,USER_000000002,1500.00,USD,MERCHANT_00000002,electronics,2024-01-15T02:15:30Z,51.5074,-0.1278,DEVICE_0000000002,41.102.197.50,28,high,2,1,1
550e8400-e29b-41d4-a716-446655440002,USER_000000003,25.50,USD,MERCHANT_00000003,grocery,2024-01-15T14:45:15Z,35.6762,139.6503,DEVICE_0000000003,192.168.1.101,42,medium,14,1,0
EOF
    echo "✅ Sample data created: data/sample_transactions.csv"
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U fraud_user -d fraud_detection > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "⚠️  PostgreSQL is not ready yet"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "⚠️  Redis is not ready yet"
fi

# Check Kafka
if docker-compose exec -T kafka kafka-topics --list --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    echo "✅ Kafka is ready"
else
    echo "⚠️  Kafka is not ready yet"
fi

# Wait a bit more for fraud services
echo "⏳ Waiting for fraud detection services..."
sleep 60

# Test the API
echo "🧪 Testing fraud detection API..."
if curl -s -f http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "✅ Fraud Detection Service 1 is healthy"
else
    echo "⚠️  Fraud Detection Service 1 is not ready yet"
fi

if curl -s -f http://localhost:8081/api/health > /dev/null 2>&1; then
    echo "✅ Fraud Detection Service 2 is healthy"
else
    echo "⚠️  Fraud Detection Service 2 is not ready yet"
fi

# Test load balancer
if curl -s -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Load Balancer is working"
else
    echo "⚠️  Load Balancer is not ready yet"
fi

echo ""
echo "🎉 Fraud Detection System is starting up!"
echo "=================================================="
echo "📊 Services:"
echo "   • Fraud Detection API: http://localhost:8080"
echo "   • Fraud Detection API 2: http://localhost:8081"
echo "   • Load Balanced API: http://localhost"
echo "   • PostgreSQL: localhost:5432"
echo "   • Redis: localhost:6379"
echo "   • Kafka: localhost:9092"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Jaeger: http://localhost:16686"
echo ""
echo "🧪 Test Commands:"
echo "   curl http://localhost/health"
echo "   curl -X POST http://localhost/api/transactions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"accountId\":\"acc_test123\",\"merchantId\":\"mer_amazon\",\"amount\":99.99,\"currency\":\"USD\"}'"
echo ""
echo "📊 View Logs:"
echo "   docker-compose logs -f fraud-service-1"
echo "   docker-compose logs -f fraud-service-2"
echo ""
echo "🛑 Stop Services:"
echo "   docker-compose down"
echo ""
echo "Happy fraud detecting! 🕵️‍♂️"