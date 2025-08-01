#!/bin/bash

# 🚨 Streamlit Fraud Detection System - Optimized Launcher
# Streamlit için optimize edilmiş başlatma scripti

echo "🚨 Starting Streamlit Fraud Detection System"
echo "============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Header
echo -e "${PURPLE}"
echo "  ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗██╗     ██╗████████╗"
echo "  ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██║     ██║╚══██╔══╝"
echo "  ███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║██║     ██║   ██║   "
echo "  ╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██║     ██║   ██║   "
echo "  ███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗██║   ██║   "
echo "  ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝   ╚═╝   "
echo -e "${NC}"
echo "             🚨 Advanced Fraud Detection Dashboard 🚨"
echo ""

# System check
echo -e "${BLUE}🔍 System Requirements Check...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
    echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 is not installed!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ pip3 found${NC}"
fi

# Virtual Environment Setup
echo -e "\n${BLUE}🔧 Virtual Environment Setup...${NC}"

if [ ! -d "streamlit-env" ]; then
    echo -e "${YELLOW}⚠️ Creating new virtual environment...${NC}"
    python3 -m venv streamlit-env
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created!${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source streamlit-env/bin/activate

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Virtual environment activated!${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment!${NC}"
    exit 1
fi

# Install/Update dependencies
echo -e "\n${BLUE}📦 Installing dependencies...${NC}"
echo -e "${YELLOW}This may take a few minutes for first-time setup...${NC}"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies!${NC}"
    echo "Try manually: pip install -r requirements.txt"
    exit 1
fi

# Verify Streamlit installation
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Streamlit installation failed!${NC}"
    exit 1
else
    STREAMLIT_VERSION=$(streamlit version | head -1 | cut -d " " -f 2)
    echo -e "${GREEN}✅ Streamlit ${STREAMLIT_VERSION} ready!${NC}"
fi

# Check required files
echo -e "\n${BLUE}📁 Checking project files...${NC}"

REQUIRED_FILES=(
    "streamlit_app.py"
    "csv_fraud_processor.py"
    "requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file found${NC}"
    else
        echo -e "${RED}❌ $file missing!${NC}"
        exit 1
    fi
done

# Create data directory if not exists
if [ ! -d "data" ]; then
    echo -e "${YELLOW}📁 Creating data directory...${NC}"
    mkdir -p data/results
    echo -e "${GREEN}✅ Data directory created${NC}"
fi

# Set environment variables for optimal performance
echo -e "\n${BLUE}⚙️ Configuring Streamlit...${NC}"
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Port selection
PORT=8502
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}⚠️ Port $PORT is busy, trying 8503...${NC}"
    PORT=8503
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}⚠️ Port $PORT is also busy, trying 8504...${NC}"
        PORT=8504
    fi
fi

# Final setup
echo -e "\n${GREEN}🚀 Starting Streamlit Fraud Detection Dashboard...${NC}"
echo -e "${YELLOW}=============================================${NC}"
echo -e "${BLUE}📊 Dashboard URL: ${NC}${GREEN}http://localhost:$PORT${NC}"
echo -e "${BLUE}🔧 CSV Processor: ${NC}${GREEN}Upload files up to 500MB${NC}"
echo -e "${BLUE}📈 Real-time Analytics: ${NC}${GREEN}Interactive fraud analysis${NC}"
echo -e "${BLUE}🧪 Transaction Tester: ${NC}${GREEN}Test individual transactions${NC}"
echo -e "${YELLOW}=============================================${NC}"
echo -e "${RED}Press Ctrl+C to stop the dashboard${NC}"
echo ""

# Launch Streamlit with optimal settings
streamlit run streamlit_app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 500 \
    --server.maxMessageSize 500 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    --global.developmentMode false

# Cleanup on exit
trap 'echo -e "\n${YELLOW}👋 Shutting down Streamlit dashboard...${NC}"; deactivate; exit' INT TERM
