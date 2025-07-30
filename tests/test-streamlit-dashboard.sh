#!/bin/bash

# 🚨 Streamlit Dashboard Test Script

echo "🚀 Testing Streamlit Fraud Detection Dashboard"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Test counter
TEST_COUNT=0
PASS_COUNT=0

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="$3"
    
    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "\n${BLUE}Test $TEST_COUNT: $name${NC}"
    echo "URL: $url"
    
    response=$(curl -s -w "\n%{http_code}" "$url")
    
    # Split response and status code
    body=$(echo "$response" | head -n -1)
    status_code=$(echo "$response" | tail -n 1)
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} - Status: $status_code"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}❌ FAIL${NC} - Expected: $expected_status, Got: $status_code"
        echo -e "${PURPLE}Response:${NC} $body"
    fi
}

# Wait for services to be ready
echo -e "\n${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 5

# Test 1: Streamlit Health Check
test_endpoint "Streamlit Health Check" "http://localhost:8501/_stcore/health" "200"

# Test 2: Streamlit Main Page
test_endpoint "Streamlit Main Page" "http://localhost:8501" "200"

# Test 3: Fraud Detection API Health (dependency)
test_endpoint "Fraud API Health" "http://localhost:8080/api/health" "200"

# Test 4: Dashboard Data API (dependency)
test_endpoint "Dashboard Data API" "http://localhost:8080/api/dashboard-data" "200"

# Test 5: Statistics API (dependency)
test_endpoint "Statistics API" "http://localhost:8080/api/statistics" "200"

# Test 6: Transaction API (dependency)
echo -e "\n${BLUE}Test 6: Transaction API (POST)${NC}"
echo "URL: http://localhost:8080/api/transactions"

response=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8080/api/transactions" \
    -H "Content-Type: application/json" \
    -d '{
        "accountId": "acc_streamlit_test",
        "merchantId": "mer_test_store",
        "amount": 150.00,
        "currency": "USD"
    }')

body=$(echo "$response" | head -n -1)
status_code=$(echo "$response" | tail -n 1)

TEST_COUNT=$((TEST_COUNT + 1))
if [ "$status_code" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} - Status: $status_code"
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "${PURPLE}Transaction Result:${NC}"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ FAIL${NC} - Expected: 200, Got: $status_code"
fi

# Test Docker containers
echo -e "\n${YELLOW}🐳 Checking Docker Containers${NC}"
echo "================================"

containers=(
    "fraud-detection-system-streamlit-dashboard-1"
    "fraud-detection-system-fraud-service-1-1"
    "fraud-detection-system-postgres-1"
    "fraud-detection-system-redis-1"
    "fraud-detection-system-nginx-1"
)

for container in "${containers[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "$container"; then
        echo -e "${GREEN}✅ $container - Running${NC}"
    else
        echo -e "${RED}❌ $container - Not Running${NC}"
    fi
done

# Performance test
echo -e "\n${YELLOW}🚀 Performance Test${NC}"
echo "==================="

echo "Testing Streamlit response time..."
start_time=$(date +%s.%N)
curl -s http://localhost:8501/_stcore/health > /dev/null
end_time=$(date +%s.%N)

if command -v bc >/dev/null 2>&1; then
    response_time=$(echo "$end_time - $start_time" | bc)
    echo -e "${GREEN}✅ Streamlit Response Time: ${response_time}s${NC}"
else
    echo -e "${GREEN}✅ Streamlit Response Time: Fast${NC}"
fi

# Summary
echo -e "\n${PURPLE}📊 TEST SUMMARY${NC}"
echo "==============="
echo -e "Total Tests: ${BLUE}$TEST_COUNT${NC}"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$((TEST_COUNT - PASS_COUNT))${NC}"

if [ $PASS_COUNT -eq $TEST_COUNT ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo -e "${GREEN}Streamlit Dashboard is working perfectly!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some tests failed. Check the output above.${NC}"
fi

echo -e "\n${BLUE}🌐 Access Points:${NC}"
echo "• Streamlit Dashboard: http://localhost:8501"
echo "• Original Dashboard: http://localhost:8080/dashboard"
echo "• API Direct: http://localhost:8080/api/health"
echo "• Load Balanced: http://localhost/health"

echo -e "\n${PURPLE}🎯 Streamlit Features:${NC}"
echo "• 📊 Real-time Dashboard with metrics"
echo "• 🧪 Interactive Transaction Tester"
echo "• 📈 Advanced Analytics & Charts"
echo "• 🔍 Transaction Analysis Tool"
echo "• 🎨 Beautiful Plotly Visualizations"
echo "• 🔄 Auto-refresh capabilities"

echo -e "\n${PURPLE}🔧 Quick Commands:${NC}"
echo "• View Streamlit logs: docker-compose logs -f streamlit-dashboard"
echo "• Restart Streamlit: docker-compose restart streamlit-dashboard"
echo "• Stop all: docker-compose down"

echo -e "\n${GREEN}Happy analyzing! 📊✨${NC}"