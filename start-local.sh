#!/bin/bash

# Start Fraud Detection System Locally

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Check if services are running
check_services() {
    log "Checking required services..."
    
    # Check PostgreSQL
    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        warning "PostgreSQL is not running. Starting..."
        brew services start postgresql@15
        sleep 3
    fi
    
    # Check Redis
    if ! redis-cli ping &> /dev/null; then
        warning "Redis is not running. Starting..."
        brew services start redis
        sleep 2
    fi
    
    success "All services are running"
}

# Start backend
start_backend() {
    log "Starting backend API..."
    cd backend
    source fraud-env/bin/activate
    
    # Set environment variables
    export POSTGRES_URL="postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection"
    export REDIS_URL="redis://localhost:6379"
    
    # Start FastAPI server
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    
    cd ..
    success "Backend started on http://localhost:8000"
}

# Start frontend
start_frontend() {
    log "Starting Streamlit frontend..."
    source streamlit-env/bin/activate
    
    # Start Streamlit
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    
    success "Frontend started on http://localhost:8501"
}

# Cleanup function
cleanup() {
    log "Stopping services..."
    
    if [ -f "backend.pid" ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm backend.pid
    fi
    
    if [ -f "frontend.pid" ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm frontend.pid
    fi
    
    success "Services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Main execution
log "Starting Fraud Detection System locally..."

check_services
start_backend
sleep 5  # Wait for backend to start
start_frontend

echo ""
success "🚀 Fraud Detection System is running!"
echo -e "${GREEN}Frontend: http://localhost:8501${NC}"
echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
echo ""
warning "Press Ctrl+C to stop all services"

# Wait for interrupt
wait
