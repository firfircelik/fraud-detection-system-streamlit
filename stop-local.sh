#!/bin/bash

# Stop Fraud Detection System

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log "Stopping Fraud Detection System..."

# Stop backend
if [ -f "backend.pid" ]; then
    kill $(cat backend.pid) 2>/dev/null || true
    rm backend.pid
    log "Backend stopped"
fi

# Stop frontend
if [ -f "frontend.pid" ]; then
    kill $(cat frontend.pid) 2>/dev/null || true
    rm frontend.pid
    log "Frontend stopped"
fi

# Kill any remaining processes
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "streamlit run streamlit_app.py" 2>/dev/null || true

success "All services stopped"
