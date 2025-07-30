#!/bin/bash

# 🚀 Quick Test Script for Fraud Detection System
# This script will download a sample dataset and test your system

echo "🎯 Starting Fraud Detection System Test..."
echo "======================================"

# Create data directory if it doesn't exist
mkdir -p data

# Download a small test dataset
echo "📥 Downloading sample dataset..."
curl -s -L https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv -o data/creditcard_sample.csv

if [ $? -eq 0 ]; then
    echo "✅ Sample dataset downloaded successfully!"
    echo "📊 Dataset size: $(wc -l < data/creditcard_sample.csv) transactions"
else
    echo "⚠️  Could not download dataset, using local sample..."
    cp data/transactions_sample.csv data/test_data.csv
fi

# Check if Docker is running
echo "🔍 Checking Docker status..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker is running"
    
    # Start the system
    echo "🚀 Starting fraud detection system..."
    docker-compose up -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Test health endpoint
    echo "🏥 Testing health endpoint..."
    curl -s http://localhost:8080/api/v1/health | jq .
    
    # Test with sample data
    echo "🧪 Testing with sample transactions..."
    curl -s -X POST http://localhost:8080/api/v1/transactions/batch \
        -H "Content-Type: application/json" \
        -d @data/sample_batch.json | jq .
    
    echo ""
    echo "🎉 System test complete!"
    echo "📊 Check logs: docker-compose logs -f"
    echo "🌐 API docs: http://localhost:8080/docs"
    echo "📈 Metrics: http://localhost:3000 (Grafana)"
    
else
    echo "❌ Docker is not running. Please start Docker first."
    echo "📋 Manual test commands:"
    echo "   sbt run  # Start the application"
    echo "   curl http://localhost:8080/api/v1/health"
fi

echo ""
echo "📚 Next steps:"
echo "   1. Check DATASETS.md for more datasets"
echo "   2. Use the API to load larger datasets"
echo "   3. Monitor with Grafana at http://localhost:3000"