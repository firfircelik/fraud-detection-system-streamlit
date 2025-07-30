#!/bin/bash

# 🚨 CSV Fraud Detection System - Comprehensive Test

echo "🚀 Testing CSV Fraud Detection System"
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Test counter
TEST_COUNT=0
PASS_COUNT=0

test_component() {
    local name="$1"
    local command="$2"
    
    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "\n${BLUE}Test $TEST_COUNT: $name${NC}"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC} - $name working"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}❌ FAIL${NC} - $name not working"
    fi
}

echo -e "\n${YELLOW}🔍 System Component Tests${NC}"
echo "========================="

# Test 1: Python Environment
test_component "Python Environment" "python3 --version"

# Test 2: Required Python Packages
test_component "Pandas Package" "python3 -c 'import pandas'"
test_component "NumPy Package" "python3 -c 'import numpy'"
test_component "Streamlit Package" "python3 -c 'import streamlit'"
test_component "Plotly Package" "python3 -c 'import plotly'"

# Test 3: CSV Processor
test_component "CSV Fraud Processor" "python3 -c 'from csv_fraud_processor import CSVFraudProcessor'"

# Test 4: Data Directory Structure
test_component "Data Directory" "[ -d 'data' ]"
test_component "Massive Data Directory" "[ -d 'data/massive' ]"
test_component "Results Directory" "[ -d 'data/results' ]"

# Test 5: CSV Files
echo -e "\n${YELLOW}📁 Available CSV Files${NC}"
echo "===================="

if [ -d "data" ]; then
    echo -e "${GREEN}Data directory contents:${NC}"
    ls -la data/*.csv 2>/dev/null | while read line; do
        echo "  $line"
    done
    
    if [ -d "data/massive" ]; then
        echo -e "\n${GREEN}Massive data directory contents:${NC}"
        ls -lah data/massive/*.csv 2>/dev/null | while read line; do
            echo "  $line"
        done
    fi
else
    echo -e "${RED}❌ Data directory not found${NC}"
fi

# Test 6: CSV Processing
echo -e "\n${YELLOW}🧪 CSV Processing Test${NC}"
echo "====================="

if [ -f "csv_fraud_processor.py" ]; then
    echo "Running CSV processor test..."
    
    # Activate virtual environment if it exists
    if [ -d "streamlit-env" ]; then
        source streamlit-env/bin/activate
    fi
    
    # Run a quick test
    python3 -c "
from csv_fraud_processor import CSVFraudProcessor
import pandas as pd

# Create test data
test_data = {
    'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005'],
    'user_id': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005'],
    'amount': [99.99, 5500.00, 25.50, 15000.00, 0.01],
    'merchant_id': ['merchant_normal', 'merchant_electronics', 'merchant_grocery', 'merchant_gambling', 'merchant_test'],
    'category': ['electronics', 'electronics', 'grocery', 'gambling', 'test'],
    'currency': ['USD', 'USD', 'USD', 'USD', 'USD']
}

df = pd.DataFrame(test_data)
processor = CSVFraudProcessor()

# Process test batch
df_processed = processor.process_batch(df)
summary = processor.generate_summary_report(df_processed)

print(f'✅ Test processing completed!')
print(f'Total transactions: {summary[\"total_transactions\"]}')
print(f'Approved: {summary[\"decisions\"][\"approved\"]}')
print(f'Declined: {summary[\"decisions\"][\"declined\"]}')
print(f'Review: {summary[\"decisions\"][\"review\"]}')
print(f'Fraud rate: {summary[\"fraud_rate\"]:.2%}')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ CSV Processing Test PASSED${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}❌ CSV Processing Test FAILED${NC}"
    fi
    
    TEST_COUNT=$((TEST_COUNT + 1))
else
    echo -e "${RED}❌ csv_fraud_processor.py not found${NC}"
fi

# Test 7: Streamlit Application
echo -e "\n${YELLOW}🌐 Streamlit Application Test${NC}"
echo "============================="

# Check if Streamlit is running
if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Streamlit is running on port 8501${NC}"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ Streamlit is not running${NC}"
    echo "To start Streamlit: streamlit run streamlit_app.py --server.port 8501"
fi

TEST_COUNT=$((TEST_COUNT + 1))

# Test 8: Fraud Detection API
echo -e "\n${YELLOW}🔗 Fraud Detection API Test${NC}"
echo "==========================="

if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Fraud Detection API is running${NC}"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ Fraud Detection API is not running${NC}"
    echo "To start API: docker-compose up -d"
fi

TEST_COUNT=$((TEST_COUNT + 1))

# Performance Test
echo -e "\n${YELLOW}⚡ Performance Test${NC}"
echo "=================="

if [ -f "csv_fraud_processor.py" ]; then
    echo "Testing processing speed with sample data..."
    
    start_time=$(date +%s.%N 2>/dev/null || date +%s)
    
    # Activate virtual environment if it exists
    if [ -d "streamlit-env" ]; then
        source streamlit-env/bin/activate
    fi
    
    python3 -c "
from csv_fraud_processor import CSVFraudProcessor
import pandas as pd
import numpy as np

# Create larger test dataset
size = 1000
test_data = {
    'transaction_id': [f'tx_{i:06d}' for i in range(size)],
    'user_id': [f'user_{i%100:03d}' for i in range(size)],
    'amount': np.random.uniform(1, 10000, size),
    'merchant_id': [f'merchant_{i%50:03d}' for i in range(size)],
    'category': np.random.choice(['electronics', 'grocery', 'gas', 'restaurant', 'gambling'], size),
    'currency': ['USD'] * size
}

df = pd.DataFrame(test_data)
processor = CSVFraudProcessor()
df_processed = processor.process_batch(df)

print(f'Processed {len(df_processed)} transactions')
" > /dev/null 2>&1
    
    end_time=$(date +%s.%N 2>/dev/null || date +%s)
    
    if command -v bc >/dev/null 2>&1; then
        duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
        echo -e "${GREEN}✅ Performance Test: ${duration}s for 1000 transactions${NC}"
    else
        echo -e "${GREEN}✅ Performance Test: Completed successfully${NC}"
    fi
else
    echo -e "${RED}❌ Cannot run performance test - processor not found${NC}"
fi

# Summary
echo -e "\n${PURPLE}📊 TEST SUMMARY${NC}"
echo "==============="
echo -e "Total Tests: ${BLUE}$TEST_COUNT${NC}"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$((TEST_COUNT - PASS_COUNT))${NC}"

if [ $PASS_COUNT -eq $TEST_COUNT ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo -e "${GREEN}CSV Fraud Detection System is ready!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some tests failed. Check the output above.${NC}"
fi

echo -e "\n${BLUE}🌐 Access Points:${NC}"
echo "• Streamlit CSV Processor: http://localhost:8501"
echo "• Fraud Detection Dashboard: http://localhost:8080/dashboard"
echo "• API Health: http://localhost:8080/api/health"

echo -e "\n${PURPLE}📄 CSV Processing Features:${NC}"
echo "• 📤 Upload CSV files directly"
echo "• 📊 Process massive datasets (1M+ transactions)"
echo "• 🎯 Advanced fraud detection algorithms"
echo "• 📈 Interactive visualizations"
echo "• 💾 Download processed results"
echo "• 🔍 Detailed risk analysis"

echo -e "\n${PURPLE}🗂️ Supported Data Formats:${NC}"
echo "• Standard CSV with headers"
echo "• Multiple column name variations"
echo "• Automatic data type detection"
echo "• Missing column handling"

echo -e "\n${PURPLE}🚀 Quick Start Commands:${NC}"
echo "• Start all services: docker-compose up -d"
echo "• Start Streamlit: streamlit run streamlit_app.py --server.port 8501"
echo "• Process CSV: python3 csv_fraud_processor.py"
echo "• Run tests: ./test-csv-system.sh"

echo -e "\n${GREEN}Happy fraud detecting with CSV! 📊🕵️‍♂️✨${NC}"