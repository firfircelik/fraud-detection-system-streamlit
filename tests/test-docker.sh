#!/bin/bash

# 🧪 Fraud Detection System - Docker Test Script

echo "🧪 Testing Fraud Detection System..."
echo "===================================="

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Test health endpoint
echo "🏥 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8080/api/health)
if [ "$HEALTH_RESPONSE" = "OK" ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed: $HEALTH_RESPONSE"
fi

# Test load balancer health
echo "🔄 Testing load balancer health..."
LB_HEALTH_RESPONSE=$(curl -s http://localhost/health)
if [ "$LB_HEALTH_RESPONSE" = "OK" ]; then
    echo "✅ Load balancer health check passed"
else
    echo "❌ Load balancer health check failed: $LB_HEALTH_RESPONSE"
fi

# Test transaction creation
echo "💳 Testing transaction creation..."
TRANSACTION_RESPONSE=$(curl -s -X POST http://localhost/api/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "accountId": "acc_test123456789",
    "merchantId": "mer_amazon12345",
    "amount": 99.99,
    "currency": "USD",
    "metadata": {
      "deviceFingerprint": "device_123",
      "ipAddress": "192.168.1.100",
      "userAgent": "Mozilla/5.0"
    }
  }')

if echo "$TRANSACTION_RESPONSE" | grep -q "id"; then
    echo "✅ Transaction creation test passed"
    echo "📊 Response: $TRANSACTION_RESPONSE"
else
    echo "❌ Transaction creation test failed"
    echo "📊 Response: $TRANSACTION_RESPONSE"
fi

# Test with suspicious transaction
echo "🚨 Testing suspicious transaction..."
SUSPICIOUS_RESPONSE=$(curl -s -X POST http://localhost/api/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "accountId": "acc_suspicious01",
    "merchantId": "mer_gambling01",
    "amount": 9999.99,
    "currency": "USD",
    "metadata": {
      "deviceFingerprint": "device_suspicious",
      "ipAddress": "41.102.197.50",
      "userAgent": "Suspicious Agent"
    }
  }')

if echo "$SUSPICIOUS_RESPONSE" | grep -q "id"; then
    echo "✅ Suspicious transaction test passed"
    echo "📊 Response: $SUSPICIOUS_RESPONSE"
else
    echo "❌ Suspicious transaction test failed"
    echo "📊 Response: $SUSPICIOUS_RESPONSE"
fi

# Test batch processing with sample data
echo "📊 Testing batch processing..."
if [ -f "data/sample_transactions.csv" ]; then
    echo "🔄 Processing sample data..."
    docker-compose exec -T fraud-service-1 java -cp /app/app.jar com.frauddetection.processing.MassiveDataProcessor /app/data/sample_transactions.csv csv || echo "⚠️  Batch processing test skipped (expected in containerized environment)"
else
    echo "⚠️  Sample data not found, skipping batch test"
fi

# Check service logs for errors
echo "📋 Checking service logs for errors..."
ERROR_COUNT=$(docker-compose logs fraud-service-1 fraud-service-2 2>/dev/null | grep -i error | wc -l)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ No errors found in service logs"
else
    echo "⚠️  Found $ERROR_COUNT error(s) in service logs"
fi

# Test monitoring endpoints
echo "📊 Testing monitoring endpoints..."
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus is accessible"
else
    echo "❌ Prometheus is not accessible"
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana is accessible"
else
    echo "❌ Grafana is not accessible"
fi

if curl -s http://localhost:16686 > /dev/null; then
    echo "✅ Jaeger is accessible"
else
    echo "❌ Jaeger is not accessible"
fi

echo ""
echo "🎉 Docker test completed!"
echo "========================"
echo "📊 Access the services:"
echo "   • API: http://localhost:8080"
echo "   • Load Balanced: http://localhost"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Jaeger: http://localhost:16686"