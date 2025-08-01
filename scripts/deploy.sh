#!/bin/bash

# 🚨 Streamlit Fraud Detection System - Deployment Script
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
PROJECT_NAME="fraud-detection-dashboard"

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
echo "  ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗██╗     ██╗████████╗"
echo "  ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██║     ██║╚══██╔══╝"
echo "  ███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║██║     ██║   ██║   "
echo "  ╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██║     ██║   ██║   "
echo "  ███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗██║   ██║   "
echo "  ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝   ╚═╝   "
echo -e "${NC}"
echo "🚨 Streamlit Fraud Detection Deployment Script"
echo "=============================================="

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
    
    # Check if virtual environment exists
    if [ ! -d "streamlit-env" ]; then
        log "Creating virtual environment..."
        python3 -m venv streamlit-env
    fi
    
    # Activate virtual environment
    source streamlit-env/bin/activate
    
    # Install dependencies
    log "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create data directory
    mkdir -p data/results
    
    # Start Streamlit
    success "Starting Streamlit locally..."
    echo -e "${YELLOW}Dashboard will be available at: http://localhost:8502${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    
    streamlit run app/main.py \
        --server.port 8502 \
        --server.address 0.0.0.0 \
        --server.maxUploadSize 500 \
        --server.enableCORS false \
        --browser.gatherUsageStats false
}

# Docker deployment
deploy_docker() {
    log "Starting Docker deployment..."
    
    # Build images
    log "Building Docker images..."
    docker-compose -f docker-compose.streamlit.yml build --no-cache
    
    # Start services
    log "Starting Docker services..."
    docker-compose -f docker-compose.streamlit.yml up -d
    
    # Wait for services
    log "Waiting for services to start..."
    sleep 10
    
    # Check status
    if docker-compose -f docker-compose.streamlit.yml ps | grep -q "Up"; then
        success "Docker deployment successful!"
        echo -e "${GREEN}Dashboard: http://localhost:8501${NC}"
        echo -e "${GREEN}API: http://localhost:8080${NC}"
        echo -e "${BLUE}View logs: docker-compose -f docker-compose.streamlit.yml logs -f${NC}"
    else
        error "Docker deployment failed. Check logs for details."
    fi
}

# Production deployment
deploy_production() {
    log "Starting production deployment..."
    
    # Create production configuration
    cat > docker-compose.prod.yml << EOF
services:
  streamlit-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "80:8501"
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
      - STREAMLIT_SERVER_ENABLE_CORS=false
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
    
    # Deploy
    docker-compose -f docker-compose.prod.yml up -d
    
    success "Production deployment complete!"
    echo -e "${GREEN}Production dashboard: http://localhost${NC}"
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