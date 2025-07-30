#!/bin/bash

# 🚀 Streamlit Fraud Detection Dashboard Launcher

echo "🚨 Starting Advanced Fraud Detection Dashboard"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if fraud detection system is running
echo -e "\n${BLUE}🔍 Checking fraud detection system...${NC}"
if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Fraud detection system is running!${NC}"
else
    echo -e "${RED}❌ Fraud detection system is not running!${NC}"
    echo -e "${YELLOW}Please start it first with: docker-compose up -d${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 is not installed!${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "streamlit-env" ]; then
    echo -e "${YELLOW}⚠️ Creating virtual environment...${NC}"
    python3 -m venv streamlit-env
fi

# Activate virtual environment
echo -e "\n${BLUE}🔧 Activating virtual environment...${NC}"
source streamlit-env/bin/activate

# Install requirements if needed
echo -e "\n${BLUE}📦 Installing Python dependencies...${NC}"
pip install -r requirements.txt --quiet

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Streamlit installation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies installed!${NC}"

# Start Streamlit
echo -e "\n${BLUE}🚀 Starting Streamlit dashboard...${NC}"
echo -e "${YELLOW}Dashboard will open at: http://localhost:8502${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Launch Streamlit with increased file upload limit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
streamlit run streamlit_app.py --server.port 8502 --server.address 0.0.0.0 --server.maxUploadSize=500