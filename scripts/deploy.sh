#!/bin/bash

# 🚨 Fraud Detection System - Deployment Script
# Usage: ./scripts/deploy.sh [local|docker|cloud]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_MODE=${1:-local}
PROJECT_NAME="fraud-detection-system"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Header
echo -e "${PURPLE}"
echo "  ███████╗██████╗  █████╗ ██╗   ██╗██████╗ "
echo "  ██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗"
echo "  █████╗  ██████╔╝███████║██║   ██║██║  ██║"
echo "  ██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║"
echo "  ██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝"
echo "  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ "
echo -e "${NC}"
echo "🚨 Fraud Detection System Deployment Script"
echo "==========================================="

# System check
check_requirements() {
    log "Checking system requirements..."
    
    case $DEPLOY_MODE in
        docker)
            if ! command -v docker &> /dev/null; then
                error "Docker is not installed. Please install Docker first."
            fi
            if ! command -v docker-compose &> /dev/null; then
                error "Docker Compose is not installed. Please install Docker Compose first."
            fi
            success "Docker requirements met"
            ;;
        local)
            if ! command -v python3 &> /dev/null; then
                error "Python 3 is not installed."
            fi
            success "Local requirements met"
            ;;
        cloud)
            if ! command -v gcloud &> /dev/null; then
                warning "Google Cloud CLI not found. Install with: curl https://sdk.cloud.google.com | bash"
            fi
            success "Cloud requirements checked"
            ;;
    esac
}

# Local deployment
deploy_local() {
    log "Starting local deployment..."
    
    # Backend setup
    log "Setting up backend..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "fraud-env" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv fraud-env
    fi
    
    # Activate virtual environment
    source fraud-env/bin/activate
    
    # Install backend dependencies
    log "Installing backend dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Start backend
    log "Starting backend server..."
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    
    cd ..
    
    # Start Streamlit frontend
    log "Starting Streamlit frontend..."
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    FRONTEND_PID=$!
    
    success "Local deployment successful!"
    echo -e "${GREEN}Streamlit Frontend: http://localhost:8501${NC}"
    echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    
    # Wait for interrupt
    trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
    wait
}

# Docker deployment
deploy_docker() {
    log "Starting Docker deployment..."
    
    # Build images
    log "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services
    log "Starting Docker services..."
    docker-compose up -d
    
    # Wait for services
    log "Waiting for services to start..."
    sleep 10
    
    # Check status
    if docker-compose ps | grep -q "Up"; then
        success "Docker deployment successful!"
        echo -e "${GREEN}Frontend: http://localhost:3000${NC}"
        echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
        echo -e "${BLUE}View logs: docker-compose logs -f${NC}"
    else
        error "Docker deployment failed. Check logs for details."
    fi
}

# Production deployment
deploy_production() {
    log "Starting production deployment..."
    
    # Create production configuration
    cat > docker-compose.prod.yml << EOF
version: '3.8'
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:password@postgres:5432/fraud_db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - API_BASE_URL=http://backend:8000
    restart: unless-stopped
    depends_on:
      - backend
EOF
    
    # Deploy
    docker-compose -f docker-compose.prod.yml up -d
    
    success "Production deployment complete!"
    echo -e "${GREEN}Production Streamlit frontend: http://localhost${NC}"
    echo -e "${GREEN}Production API: http://localhost:8000${NC}"
}

# Main deployment function
main() {
    log "Deployment mode: $DEPLOY_MODE"
    
    case $DEPLOY_MODE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        prod|production)
            deploy_production
            ;;
        *)
            echo "Usage: $0 [local|docker|prod]"
            echo ""
            echo "Examples:"
            echo "  $0 local     - Deploy locally with virtual environment"
            echo "  $0 docker    - Deploy with Docker Compose"
            echo "  $0 prod      - Deploy production version"
            exit 1
            ;;
    esac
}

# Run deployment
check_requirements
main